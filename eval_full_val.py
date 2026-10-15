# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- importing train_linux: provides all model/config/decoder symbols; ---
# --- main() stays inactive (under the __main__ guard). -------------------
import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-Val mIoU of a checkpoint (streaming, memory-safe)."
    )
    p.add_argument("--config",     required=True, help="YAML as in training")
    p.add_argument("--checkpoint", required=True, help="Path to best_miou.pt")
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--output",     default=None,
                   help="Target JSON (default: next to the checkpoint as "
                        "miou_fullval.json)")
    p.add_argument("--n_subset",   type=int, default=None,
                   help="Only for sanity checks: instead of full Val, draw a "
                        "subset of size N (seed-42 selection as in training). "
                        "Default None = all Val samples.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Seed of the subset selection (only relevant when "
                        "--n_subset is set; ignored for full Val).")
    p.add_argument("--z_mode",     default="mean", choices=["mean", "sample"],
                   help="18/B1: z from the prior — mean (deterministic, "
                        "default) or sample (stochastic). Ignored without CVAE.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, seg_decoder, dataset, device, indices, z_mode="mean"):
    """
    Streams over `indices` of the Val dataset and averages the IoU.
    Fusion of validate_seg() (decode + aggregation) and the direct
    index access from build_seg_val_cache() -- without pre-stacking.
    """
    model.eval()
    all_ious = []
    n_total  = len(indices)
    t0       = time.time()
    use_amp  = (device.type == "cuda")

    for k, i in enumerate(indices):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device).float()   # [1,3,256,128,128]  (1d: GPU-Upcast)
        tgt = sample["target"].float()                           # [256,128,128] (CPU); fp32 for decoder
        ego = sample["ego_delta"].unsqueeze(0).to(device).float() if "ego_delta" in sample else None  # Ego delta

        # 1) World-model forward (AMP as in training)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_latent = model(inp, ego, z_mode=z_mode)  # [1,256,128,128]

        # 2) Decode -- the decoder deliberately runs in float32 (like validate_seg)
        pred_mask, _ = seg_decoder.latent_to_mask(pred_latent[0], device=device)
        real_mask, _ = seg_decoder.latent_to_mask(tgt,            device=device)

        # 3) IoU
        iou = T.compute_iou(pred_mask, real_mask, T.MAP_CLASSES)
        all_ious.append(iou)

        # tee-stable progress every 50 samples
        if (k + 1) % 50 == 0 or (k + 1) == n_total:
            el   = time.time() - t0
            it_s = (k + 1) / el if el > 0 else 0.0
            eta  = (n_total - k - 1) / it_s if it_s > 0 else 0.0
            run  = float(np.mean([d["mIoU"] for d in all_ious]))
            sys.stdout.write(
                f"\r  Decode {100.0*(k+1)/n_total:5.1f}%  "
                f"[{k+1:>5}/{n_total}]  mIoU={run:.4f}  "
                f"{it_s:4.1f} it/s  ETA {eta/60:5.1f}min   "
            )
            sys.stdout.flush()
    sys.stdout.write("\r" + " " * 90 + "\r")
    sys.stdout.flush()

    mean_miou = float(np.mean([d["mIoU"] for d in all_ious]))
    per_class = {}
    for cls in T.MAP_CLASSES:
        vals = [d[cls] for d in all_ious if d.get(cls) is not None]
        per_class[cls] = round(float(np.mean(vals)), 4) if vals else None
    return mean_miou, per_class, len(all_ious)


def main():
    args = parse_args()

    # --- Config + reproducibility (same helpers as train_linux) --------------
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit(
            "[eval_full_val] seg_decoder_torch not importable -- without "
            "a decoder there is no mIoU. (The same issue would disable "
            "validate_seg in training.)"
        )

    # --- Build model + load checkpoint (weights only, no optimizer) ----------
    model_cfg    = cfg["model"]
    model_config = T.ModelConfig(**model_cfg)
    model        = T.BEVWorldModel(model_config).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))

    # --- Frozen seg decoder from the same config source ----------------------
    decode_cfg = cfg.get("decode_validation", {})
    seg_ckpt   = decode_cfg.get("seg_checkpoint")
    if not (seg_ckpt and Path(seg_ckpt).exists()):
        raise SystemExit(
            f"[eval_full_val] seg_checkpoint not found: {seg_ckpt}\n"
            f"                (check decode_validation.seg_checkpoint in the YAML.)"
        )
    seg_decoder = T.build_seg_decoder(seg_ckpt, device, verbose=True)

    # --- Val dataset (build_dataloaders returns train/val/test; only val used)
    print("[eval_full_val] Building DataLoader...")
    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset
    n_avail = len(dataset)

    # --- Sample selection: identical logic to build_seg_val_cache() ----------
    #     n_subset=None -> all (permutation of all indices -> sorted -> [0..N-1]).
    if args.n_subset is None:
        indices = list(range(n_avail))
        mode = f"FULL-VAL ({n_avail} Samples)"
    else:
        rng     = np.random.default_rng(args.seed)
        n_take  = min(args.n_subset, n_avail)
        indices = sorted(rng.choice(n_avail, size=n_take, replace=False).tolist())
        mode = f"SUBSET (n={n_take}, seed={args.seed}) -- sanity check"

    print(f"\n{'='*64}")
    print(f"  Full-Val-mIoU  |  Phase: {args.phase}  |  {mode}")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"{'='*64}\n")

    t0 = time.time()
    mean_miou, per_class, n_done = evaluate(
        model, seg_decoder, dataset, device, indices, z_mode=args.z_mode)
    dt = time.time() - t0

    # --- Result --------------------------------------------------------------
    result = {
        "checkpoint":  str(args.checkpoint),
        "config":      str(args.config),
        "phase":       args.phase,
        "n_samples":   n_done,
        "full_val":    args.n_subset is None,
        "z_mode":      args.z_mode,
        "mIoU":        round(mean_miou, 4),
        "per_class":   per_class,
        "eval_time_sec": round(dt, 1),
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(args.output) if args.output else \
        Path(args.checkpoint).parent / "miou_fullval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"{'='*64}")
    print(f"  DONE    |  {n_done} samples in {dt/60:.1f} min")
    print(f"  mIoU (Full-Val): {mean_miou:.4f}")
    print(f"  per-class:       " +
          "  ".join(f"{c}={per_class[c]}" for c in T.MAP_CLASSES))
    print(f"  written:         {out}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
