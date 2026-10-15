# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

CFG_PATH = "configs/nuscenes/seg/fusion-bev256d2-lss.yaml"
CKPT = "pretrained/bevfusion-seg.pth"
TRAIN_LATENTS = Path("/output/dump_train_minimal")
VAL_LATENTS = Path("/output/dump_seg_minimal")
GT_TRAIN = "/output/gt_masks_seg/gt_masks_pipe_train.npz"
GT_VAL = "/output/gt_masks_seg/gt_masks_pipe_val.npz"
OUT = Path("/output/adapted")
CLASSES = ["drivable_area", "ped_crossing", "walkway", "stop_line",
           "carpark_area", "divider"]


class LatentGTDataset(Dataset):
    """(Prediction latent [256,128,128] fp32, GT mask [6,200,200] float)."""

    def __init__(self, latent_dir, gt_npz_path, tokens):
        self.latent_dir = Path(latent_dir)
        self.gt_path = gt_npz_path
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        if not hasattr(self, "_gt"):                    # lazy per worker
            self._gt = np.load(self.gt_path)
        x = np.load(self.latent_dir / f"bev_latent_{tok}.npy").astype(np.float32)
        if x.ndim == 4:
            x = x[0]
        gt = np.unpackbits(self._gt[tok]).reshape(6, 200, 200).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gt)


@torch.no_grad()
def val_iou(model, loader, device, max_batches=None):
    """Fast proxy IoU (threshold 0.5) on val PREDICTIONS vs GT."""
    model.eval()
    inter = torch.zeros(6, device=device)
    union = torch.zeros(6, device=device)
    for bi, (x, gt) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, gt = x.to(device), gt.to(device).bool()
        feat = model.decoder["backbone"](x)
        feat = model.decoder["neck"](feat)
        probs = model.heads["map"](feat)   # eval branch ALREADY applies sigmoid!
        pred = probs > 0.5
        inter += (pred & gt).flatten(2).sum(dim=(0, 2)).float()
        union += (pred | gt).flatten(2).sum(dim=(0, 2)).float()
    model.train()
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return float(iou.mean()), {c: float(v) for c, v in zip(CLASSES, iou)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "head"], default="full",
                    help="full = decoder.backbone+neck+heads.map | head = heads.map only")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed of the head training (init+shuffle)")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    configs.load(CFG_PATH, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CFG_PATH)
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu")
    model = model.to(device)

    # --- Freezing: only the selected post-fuser part learns ------------------
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.heads["map"].parameters())
    if args.mode == "full":
        trainable += list(model.decoder["backbone"].parameters())
        trainable += list(model.decoder["neck"].parameters())
    for p in trainable:
        p.requires_grad = True
    model.train()
    if args.mode == "head":                      # pin BN in the frozen decoder
        model.decoder["backbone"].eval()
        model.decoder["neck"].eval()
    n_tr = sum(p.numel() for p in trainable)
    print(f"[adapt] mode={args.mode}, trainable {n_tr/1e6:.2f}M parameters")

    gt_train = np.load(GT_TRAIN)
    tokens = sorted(f.stem.replace("bev_latent_", "")
                    for f in TRAIN_LATENTS.glob("bev_latent_*.npy"))
    tokens = [t for t in tokens if t in gt_train.files]
    gt_val = np.load(GT_VAL)
    vtokens = sorted(f.stem.replace("bev_latent_", "")
                     for f in VAL_LATENTS.glob("bev_latent_*.npy"))
    vtokens = [t for t in vtokens if t in gt_val.files][:1000]
    print(f"[adapt] {len(tokens)} train / {len(vtokens)} val pairs")

    dl = DataLoader(LatentGTDataset(TRAIN_LATENTS, GT_TRAIN, tokens),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vdl = DataLoader(LatentGTDataset(VAL_LATENTS, GT_VAL, vtokens),
                     batch_size=args.batch, shuffle=False, num_workers=4)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    OUT.mkdir(exist_ok=True)

    m0, per0 = val_iou(model, vdl, device, max_batches=(1 if args.smoke else None))
    print(f"[adapt] BEFORE adaptation: proxy IoU {m0:.4f} | " +
          " ".join(f"{c.split('_')[0]}={v:.3f}" for c, v in per0.items()), flush=True)

    best = m0
    for ep in range(1, args.epochs + 1):
        t0, run = time.time(), 0.0
        for bi, (x, gt) in enumerate(dl):
            x, gt = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feat = model.decoder["backbone"](x)
                feat = model.decoder["neck"](feat)
                losses = model.heads["map"](feat, gt)   # focal loss dict
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            if (bi + 1) % 100 == 0 or args.smoke:
                print(f"  ep{ep} [{bi+1:>5}/{len(dl)}] loss={run/(bi+1):.4f} "
                      f"{(bi+1)/(time.time()-t0):4.1f} it/s", flush=True)
            if args.smoke and bi >= 4:
                break
        miou, per = val_iou(model, vdl, device,
                            max_batches=(1 if args.smoke else None))
        print(f"[adapt] ep{ep}: proxy IoU {miou:.4f} (before: {m0:.4f}) | " +
              " ".join(f"{c.split('_')[0]}={v:.3f}" for c, v in per.items()), flush=True)
        torch.save({"state_dict": model.state_dict()},
                   OUT / (f"seg_adapted_{args.mode}_last.pth" if args.seed == 42
                   else f"seg_adapted_{args.mode}_s{args.seed}_last.pth"))
        if miou > best:
            best = miou
            torch.save({"state_dict": model.state_dict()},
                       OUT / (f"seg_adapted_{args.mode}_best.pth" if args.seed == 42
                          else f"seg_adapted_{args.mode}_s{args.seed}_best.pth"))
            print(f"[adapt] new best ({best:.4f}) saved", flush=True)
        if args.smoke:
            break
    print(f"[adapt] DONE mode={args.mode} best={best:.4f} (start {m0:.4f})")


if __name__ == "__main__":
    main()
