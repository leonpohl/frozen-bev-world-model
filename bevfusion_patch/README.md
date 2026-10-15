# BEVFusion patch: latent extraction and latent injection

This directory contains the tool used to extract the BEV latents from
BEVFusion and to inject them back in for evaluation.
It consists of two parts:

1. **`latent_saver.py`** — a ready-made module, copied into the root
   directory of the BEVFusion checkout (next to `tools/`).
2. **A ~30-line hook** in
   `mmdet3d/models/fusion_models/bevfusion.py`, directly **after the
   ConvFuser and before the decoder** (see the code below).

Both originally come from an earlier project assignment and are supplied
here for reproducibility. BEVFusion itself
(mit-han-lab/bevfusion, Apache 2.0) is not part of this repository.

## The hook (in `BEVFusion.forward_single`, after the fuser)

```python
        # =========================================================
        # Latent saver hook (controlled via environment variables)
        # =========================================================
        import os
        if not self.training and os.environ.get("SAVE_BEV_LATENTS", "0") == "1":
            from latent_saver import save_latent
            save_latent(x, metas)

        # =========================================================
        # Latent injection hook: replaces the fuser output with a
        # loaded latent (raw scale!) — turns the downstream chain
        # (decoder -> head -> eval) into a measurement machine for
        # world-model predictions.
        # Env: LOAD_BEV_LATENTS=1, LATENT_LOAD_DIR=<dir with bev_latent_<tok>.npy>
        # Default off -> behavior unchanged. Do not combine with SAVE.
        # =========================================================
        if not self.training and os.environ.get("LOAD_BEV_LATENTS", "0") == "1":
            import numpy as np
            _dir = os.environ.get("LATENT_LOAD_DIR", "/output/latents_in")
            _tok = None
            if isinstance(metas, (list, tuple)) and len(metas) > 0:
                _tok = metas[0].get("sample_idx", None) or metas[0].get("token", None)
            _p = os.path.join(_dir, "bev_latent_{}.npy".format(_tok))
            _arr = np.load(_p).astype(np.float32)
            _t = torch.from_numpy(_arr).to(x.device)
            if _t.ndim == 3:
                _t = _t.unsqueeze(0)
            assert _t.shape == x.shape, \
                "latent_injection: shape {} != fuser {}".format(tuple(_t.shape), tuple(x.shape))
            x = _t
```

Insertion point: in `forward_single`, immediately after

```python
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
```

and before `x = self.decoder["backbone"](x)`.

## Usage

**Extraction** (one directory of individual `.npy` files per split;
seg: 256×128×128, det: 256×180×180, fp16):

```bash
SAVE_BEV_LATENTS=1 FLATTEN_BEV_LATENTS=0 LATENT_DTYPE=float32 \
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map
# Train split: additionally
#   --cfg-options data.test.ann_file=data/nuscenes/nuscenes_infos_train.pkl
```

**Injection** (annotation-anchored evaluation of predictions —
convert the latents back to raw scale first, det: factor 1/0.3641):

```bash
LOAD_BEV_LATENTS=1 LATENT_LOAD_DIR=/output/latents_in \
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map
```

Further switches of `latent_saver.py` (all via env): `LATENT_DTYPE`
(`float16`/`float32`), `FLATTEN_BEV_LATENTS` (spatial average instead of
the full map), `LATENT_POOL`/`LATENT_POOL_FACTOR`/`LATENT_POOL_MODE`
(optional spatial pooling). The target directory is fixed to
`/output/latents` (Docker mount of the container); adjust line 46 of
`latent_saver.py` if needed.
