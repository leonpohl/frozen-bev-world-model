# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
seg_decoder_torch.py — BEVFusion seg decoder as pure PyTorch
================================================================================

PURPOSE
-------
Decodes a BEV seg latent [256,128,128] into a segmentation mask
[6,200,200] — WITHOUT mmdet3d / mmcv. This allows the real mIoU to be
measured directly in the training environment (PyTorch 2.1.2) instead of
in the BEVFusion docker container.

The decoder is a 1:1 replica of what inference_seg.py assembles in the
docker via the mmdet3d builders:

    Latent [256,128,128]
       -> SECOND backbone     -> ([128,128,128], [256,64,64])
       -> SECONDFPN neck       -> [512,128,128]
       -> BEVGridTransform     -> [512,200,200]
       -> Classifier           -> [6,200,200] logits
       -> sigmoid + threshold  -> bool mask [6,200,200]

The mmdet3d-dependent parts (SECOND, SECONDFPN) are rebuilt here as plain
nn.Module classes so that the weight keys EXACTLY match those in the
checkpoint and can be loaded with strict=True. BEVGridTransform and
Classifier are already pure PyTorch in inference_seg.py and are
adopted as-is.

MODULARITY
------------------------------
render_mask() and save_comparison() are included here, but are NOT called
by the training validation. This lets the visualization simply import and
plug in the PNG output (pred/real/comparison) without code duplication.

WEIGHT KEYS (mapping checkpoint -> submodule)
----------------------------------------------
    decoder.backbone.*          -> SECONDBackbone   (self.blocks.*)
    decoder.neck.*              -> SECONDFPNNeck     (self.deblocks.*)
    heads.map.classifier.*      -> classifier        (nn.Sequential, 0..6)
    (BEVGridTransform has no learnable parameters — pure geometry)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Order & threshold EXACTLY as in inference_seg.py. Do not change,
# otherwise the mIoU numbers will no longer match the baseline.
MAP_CLASSES = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]
MAP_SCORE = 0.5

MAP_PALETTE = {
    'drivable_area': (166, 206, 227),
    'ped_crossing':  (251, 154, 153),
    'walkway':       (227,  26,  28),
    'stop_line':     (253, 191, 111),
    'carpark_area':  (255, 127,   0),
    'divider':       (106,  61, 154),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECOND backbone (decoder backbone) — pure PyTorch replica
# ─────────────────────────────────────────────────────────────────────────────
# Equivalent to mmdet3d SECOND with:
#   in_channels=256, out_channels=[128,256], layer_nums=[5,5],
#   layer_strides=[1,2], norm=BN(eps=1e-3, momentum=0.01), conv bias=False
#
# Structure per block i:
#   Conv(in->out, 3, stride=layer_strides[i], pad=1, bias=False), BN, ReLU
#   then layer_nums[i]x:  Conv(out->out,3,pad=1,bias=False), BN, ReLU
# Keys: blocks.{i}.{0..(3+3*layer_num-1)}  -> matches decoder.backbone.*
class SECONDBackbone(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=(128, 256),
                 layer_nums=(5, 5),
                 layer_strides=(1, 2),
                 bn_eps=1e-3,
                 bn_momentum=0.01):
        super().__init__()
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            layers = [
                nn.Conv2d(in_filters[i], out_channels[i], 3,
                          stride=layer_strides[i], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[i], eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                layers.append(nn.Conv2d(out_channels[i], out_channels[i], 3,
                                        padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels[i], eps=bn_eps,
                                             momentum=bn_momentum))
                layers.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)


# ─────────────────────────────────────────────────────────────────────────────
# SECONDFPN neck — pure PyTorch replica
# ─────────────────────────────────────────────────────────────────────────────
# Equivalent to mmdet3d SECONDFPN with:
#   in_channels=[128,256], out_channels=[256,256], upsample_strides=[1,2],
#   upsample_cfg=deconv(bias=False), use_conv_for_no_stride=True,
#   conv_cfg=Conv2d(bias=False), norm=BN(eps=1e-3, momentum=0.01)
#
# stride==1 + use_conv_for_no_stride=True -> Conv2d(k=1,s=1) instead of deconv
# stride==2                               -> ConvTranspose2d(k=2,s=2)
# Keys: deblocks.{i}.{0,1}  -> matches decoder.neck.*
class SECONDFPNNeck(nn.Module):
    def __init__(self,
                 in_channels=(128, 256),
                 out_channels=(256, 256),
                 upsample_strides=(1, 2),
                 bn_eps=1e-3,
                 bn_momentum=0.01):
        super().__init__()
        self.in_channels = in_channels
        deblocks = []
        for i, out_c in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                up = nn.ConvTranspose2d(in_channels[i], out_c,
                                        kernel_size=stride, stride=stride,
                                        bias=False)
            else:
                # use_conv_for_no_stride=True -> 1x1 conv instead of upsampling
                k = int(round(1.0 / stride))
                up = nn.Conv2d(in_channels[i], out_c, kernel_size=k, stride=k,
                               bias=False)
            deblocks.append(nn.Sequential(
                up,
                nn.BatchNorm2d(out_c, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        assert len(x) == len(self.in_channels), \
            f"Neck expected {len(self.in_channels)} feature maps, got {len(x)}"
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        out = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return [out]


# ─────────────────────────────────────────────────────────────────────────────
# BEVGridTransform — 1:1 from inference_seg.py (pure geometry, no weights)
# ─────────────────────────────────────────────────────────────────────────────
class BEVGridTransform(nn.Module):
    def __init__(self, *, input_scope, output_scope, prescale_factor=1):
        super().__init__()
        self.input_scope     = input_scope
        self.output_scope    = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x):
        if self.prescale_factor != 1:
            x = F.interpolate(x, scale_factor=self.prescale_factor,
                              mode='bilinear', align_corners=False)
        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
                self.input_scope, self.output_scope):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))
        u, v = torch.meshgrid(coords, indexing='ij')
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Encapsulated decoder: latent -> (logits/probs/mask)
# ─────────────────────────────────────────────────────────────────────────────
class SegDecoder(nn.Module):
    """
    Fully frozen seg decoder. After build_seg_decoder() ready for
    pure forward inference. Carries no gradients and is never updated
    during training.
    """
    def __init__(self, map_classes=MAP_CLASSES):
        super().__init__()
        self.map_classes = list(map_classes)
        n_cls = len(self.map_classes)

        self.backbone = SECONDBackbone()
        self.neck     = SECONDFPNNeck()
        self.transform = BEVGridTransform(
            input_scope =[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope=[[-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]],
        )
        # Classifier EXACTLY as in inference_seg.py / BEVSegmentationHead
        # (BN default eps here, NOT 1e-3 — deliberately as in the original).
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, n_cls, 1),
        )

    def _forward_logits_impl(self, latent_bchw):
        """Shared core (WITHOUT no_grad decorator, so that the task loss
        can let gradients flow THROUGH the frozen decoder to the latent;
        the decoder parameters themselves stay frozen/eval)."""
        x = self.backbone(latent_bchw)     # tuple
        x = self.neck(x)                    # [ [B,512,128,128] ]
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.transform(x)               # [B,512,200,200]
        x = self.classifier(x)              # [B,n_cls,200,200]
        return x

    @torch.no_grad()
    def forward_logits(self, latent_bchw):
        """latent_bchw: [B,256,128,128] float32 -> logits [B,n_cls,200,200]"""
        return self._forward_logits_impl(latent_bchw)

    def forward_logits_grad(self, latent_bchw):
        """Like forward_logits, but WITH gradient flow (decoder task loss).
        The caller must ensure float32 (autocast off) — like latent_to_mask."""
        return self._forward_logits_impl(latent_bchw)

    @torch.no_grad()
    def latent_to_mask(self, latent, device=None, threshold=MAP_SCORE):
        """
        latent: np.ndarray OR torch.Tensor, [256,128,128] or [B,256,128,128]
        returns: (mask_bool, probs)  each [B,n_cls,200,200], or without B if single
        """
        single = False
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent.astype(np.float32))
        latent = latent.float()
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
            single = True
        if device is not None:
            latent = latent.to(device)

        # IMPORTANT: autocast explicitly OFF here -> float32, as in the docker reference.
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.forward_logits(latent)
            probs  = torch.sigmoid(logits)

        mask = (probs >= threshold)
        probs = probs.detach().cpu().numpy()
        mask  = mask.detach().cpu().numpy()
        if single:
            return mask[0], probs[0]
        return mask, probs


# ─────────────────────────────────────────────────────────────────────────────
# Loading weights (strict=True is the actual proof of correctness)
# ─────────────────────────────────────────────────────────────────────────────
def build_seg_decoder(checkpoint_path, device, map_classes=MAP_CLASSES,
                      verbose=True):
    """
    Builds the SegDecoder and loads the weights from bevfusion-seg.pth.
    strict=True on every submodule: if even one key name or shape
    is wrong, an error is thrown here -> strongest architecture check.
    """
    dec = SegDecoder(map_classes=map_classes)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    def _sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    bb_sd  = _sub('decoder.backbone.')
    neck_sd = _sub('decoder.neck.')
    cls_sd = _sub('heads.map.classifier.')

    dec.backbone.load_state_dict(bb_sd, strict=True)
    dec.neck.load_state_dict(neck_sd, strict=True)
    dec.classifier.load_state_dict(cls_sd, strict=True)

    if verbose:
        print(f"[seg_decoder] backbone   keys: {len(bb_sd)}")
        print(f"[seg_decoder] neck       keys: {len(neck_sd)}")
        print(f"[seg_decoder] classifier keys: {len(cls_sd)}")
        # Sanity: the last conv must have n_cls outputs
        last_w = cls_sd.get('6.weight')
        if last_w is not None:
            assert last_w.shape[0] == len(map_classes), \
                f"Class mismatch: ckpt {last_w.shape[0]} vs {len(map_classes)}"
            print(f"[seg_decoder] class check OK: {len(map_classes)} -> {map_classes}")
        print(f"[seg_decoder] strict-load successful -> architecture correct.")

    dec = dec.to(device).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


# ─────────────────────────────────────────────────────────────────────────────
# IoU — 1:1 from inference_seg.py
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou(pred_mask, real_mask, classes=MAP_CLASSES):
    """pred_mask/real_mask: bool [n_cls,200,200] -> dict{class: iou|None, mIoU}"""
    results = {}
    valid = []
    for c, name in enumerate(classes):
        inter = np.logical_and(pred_mask[c], real_mask[c]).sum()
        union = np.logical_or(pred_mask[c], real_mask[c]).sum()
        if union > 0:
            iou = float(inter) / float(union)
            results[name] = round(iou, 4)
            valid.append(iou)
        else:
            results[name] = None
    results['mIoU'] = round(float(np.mean(valid)), 4) if valid else 0.0
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers — visualization only (NOT called by the validation)
# ─────────────────────────────────────────────────────────────────────────────
def render_mask(mask, classes=MAP_CLASSES, background=(240, 240, 240)):
    """mask: bool [C,H,W] -> RGB uint8 [H,W,3]"""
    canvas = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    canvas[:] = background
    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[mask[k]] = MAP_PALETTE[name]
    return canvas


def save_comparison(pred_mask, real_mask, fpath, classes=MAP_CLASSES):
    """pred | real | diff side by side as PNG (requires cv2)."""
    import cv2
    pred_img = render_mask(pred_mask, classes)
    real_img = render_mask(real_mask, classes)
    H, W = pred_mask.shape[1], pred_mask.shape[2]
    diff = np.full((H, W, 3), 240, dtype=np.uint8)
    both_active = np.any(real_mask, axis=0)
    agree       = np.all(pred_mask == real_mask, axis=0)
    diff[agree & both_active] = (100, 200, 100)
    diff[~agree]              = (220,  50,  50)
    sep = np.full((H, 4, 3), 100, dtype=np.uint8)
    combined = np.concatenate([pred_img, sep, real_img, sep, diff], axis=1)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    cv2.imwrite(fpath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (without real weights): architecture wiring + key names
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Self-test seg_decoder_torch (random weights) ===")
    dec = SegDecoder()
    dec.eval()

    # 1) Shapes through the whole chain
    x = torch.randn(2, 256, 128, 128)
    bb = dec.backbone(x)
    print("backbone out shapes:", [tuple(t.shape) for t in bb])
    assert tuple(bb[0].shape) == (2, 128, 128, 128)
    assert tuple(bb[1].shape) == (2, 256, 64, 64)
    nk = dec.neck(bb)
    print("neck out shape:     ", tuple(nk[0].shape))
    assert tuple(nk[0].shape) == (2, 512, 128, 128)
    tr = dec.transform(nk[0])
    print("transform out shape:", tuple(tr.shape))
    assert tuple(tr.shape) == (2, 512, 200, 200)
    lo = dec.classifier(tr)
    print("classifier out:     ", tuple(lo.shape))
    assert tuple(lo.shape) == (2, 6, 200, 200)

    # 2) latent_to_mask end-to-end
    m, p = dec.latent_to_mask(np.random.randn(256, 128, 128).astype(np.float32),
                              device='cpu')
    print("mask shape:", m.shape, "dtype:", m.dtype, "| probs:", p.shape)
    assert m.shape == (6, 200, 200) and m.dtype == bool

    # 3) compute_iou plausible (same mask -> mIoU 1.0)
    iou = compute_iou(m, m)
    print("self-IoU (must be 1.0 where union>0):", iou['mIoU'])

    # 4) Print state-dict keys (must fit the checkpoint mapping)
    print("\n--- backbone keys (erste 6) ---")
    for k in list(dec.backbone.state_dict().keys())[:6]:
        print("  ", k)
    print("--- neck keys ---")
    for k in dec.neck.state_dict().keys():
        print("  ", k)
    print("--- classifier keys ---")
    for k in dec.classifier.state_dict().keys():
        print("  ", k)

    print("\n[OK] Architecture wiring & key names plausible.")
