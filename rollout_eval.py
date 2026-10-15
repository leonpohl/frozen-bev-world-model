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

import train_linux as T


def parse_args():
    p = argparse.ArgumentParser(description="autoregressive Rollout-Eval k=1..K.")
    p.add_argument("--config",     required=True, help="YAML as in training (paths/decoder).")
    p.add_argument("--checkpoint", required=True, help="best_miou.pt of the config to roll out.")
    p.add_argument("--phase",      default="cell", choices=["frame", "cell"])
    p.add_argument("--k",          type=int, default=4, help="max rollout horizon (default 4).")
    p.add_argument("--output",     default=None, help="Target JSON (default next to the checkpoint).")
    p.add_argument("--max_windows", type=int, default=None,
                   help="Only N (seed-42) windows instead of all -- smoke/proxy. Default None = all.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--mode",       default="det", choices=["det", "sample"],
                   help="det = deterministic forward; "
                        "sample = ONE flow sample per step (requires flow_head=flow).")
    p.add_argument("--flow_steps", type=int, default=1,
                   help="18/B2: Euler steps per flow sample (s1~s10 finding).")
    return p.parse_args()


def _chan_std_mean(latent: torch.Tensor) -> float:
    """Mean over channels of the per-channel std (H,W) -- like loss_std / std-ratio."""
    return float(latent.float().std(dim=[-2, -1]).mean().item())


def build_windows(dataset, win_len, max_windows, seed):
    """All scene-internal, consecutive windows of length win_len (frame dicts)."""
    windows = []
    for scene in dataset._all_scenes:                 # val scenes (scene_indices=None)
        for i in range(len(scene) - win_len + 1):
            windows.append(scene[i:i + win_len])
    if max_windows is not None and max_windows < len(windows):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(windows), size=max_windows, replace=False).tolist())
        windows = [windows[j] for j in idx]
    return windows


@torch.no_grad()
def rollout_eval(model, seg_decoder, dataset, device, K, n_input, windows,
                 mode="det", flow_steps=1):
    model.eval()
    use_amp = (device.type == "cuda")
    miou = {k: [] for k in range(1, K + 1)}
    stdr = {k: [] for k in range(1, K + 1)}
    pers = {k: [] for k in range(1, K + 1)}
    n_total = len(windows)
    t0 = time.time()

    for wi, win in enumerate(windows):
        # Latents [256,128,128] of all n_input+K frames onto the GPU
        lat = [torch.from_numpy(dataset._load_latent(f)).float().to(device) for f in win]

        # Real reference masks: decode(F3_real) for persistence + decode(F_{3+k}_real)
        last_real_mask, _ = seg_decoder.latent_to_mask(lat[n_input - 1], device=device)
        real_masks = {}
        for k in range(1, K + 1):
            rm, _ = seg_decoder.latent_to_mask(lat[n_input - 1 + k], device=device)
            real_masks[k] = rm

        # Autoregressive rollout starting from the real context [F1..F_n_input]
        cur = torch.stack(lat[:n_input], dim=0).unsqueeze(0)   # [1, n_input, 256,128,128]
        for k in range(1, K + 1):
            with torch.cuda.amp.autocast(enabled=use_amp):
                if mode == "sample":
                    # 18/B2: ONE flow sample per step; generator fixed per
                    # (window, step) -> reproducible trajectory.
                    _g = torch.Generator(device=device.type)
                    _g.manual_seed(100000 + wi * 100 + k)
                    pred = model.flow_sample(cur, steps=flow_steps, generator=_g)
                else:
                    pred = model(cur)                          # [1,256,128,128]
            pred_l = pred[0].float()

            pm, _ = seg_decoder.latent_to_mask(pred_l, device=device)
            miou[k].append(T.compute_iou(pm, real_masks[k], T.MAP_CLASSES)["mIoU"])
            stdr[k].append(_chan_std_mean(pred_l) / max(_chan_std_mean(lat[n_input - 1 + k]), 1e-8))
            pers[k].append(T.compute_iou(last_real_mask, real_masks[k], T.MAP_CLASSES)["mIoU"])

            # Shift the prediction back into the window (inference.py pattern)
            cur = torch.cat([cur[:, 1:], pred_l.unsqueeze(0).unsqueeze(0)], dim=1)

        if (wi + 1) % 25 == 0 or (wi + 1) == n_total:
            el = time.time() - t0
            it_s = (wi + 1) / el if el > 0 else 0.0
            eta = (n_total - wi - 1) / it_s if it_s > 0 else 0.0
            m1 = float(np.mean(miou[1])); mK = float(np.mean(miou[K]))
            sys.stdout.write(f"\r  Window {wi+1:>5}/{n_total}  mIoU(1)={m1:.4f} "
                             f"mIoU({K})={mK:.4f}  {it_s:4.1f} w/s  ETA {eta/60:5.1f}min   ")
            sys.stdout.flush()
    sys.stdout.write("\r" + " " * 90 + "\r"); sys.stdout.flush()

    curve = {}
    for k in range(1, K + 1):
        curve[k] = {
            "mIoU":      round(float(np.mean(miou[k])), 4),
            "std_ratio": round(float(np.mean(stdr[k])), 4),
            "mIoU_pers": round(float(np.mean(pers[k])), 4),
            "n_windows": len(miou[k]),
        }
    return curve


def main():
    args = parse_args()
    cfg = T.load_config(args.config, args.phase)
    T.set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not T.SEG_DECODER_AVAILABLE:
        raise SystemExit("[rollout_eval] seg_decoder_torch not importable -- no mIoU possible.")

    n_input = cfg["model"].get("n_frames", 3)
    win_len = n_input + args.k

    model = T.BEVWorldModel(T.ModelConfig(**cfg["model"])).to(device)
    T.load_checkpoint(args.checkpoint, model, device=str(device))

    seg_ckpt = cfg.get("decode_validation", {}).get("seg_checkpoint")
    if not (seg_ckpt and Path(seg_ckpt).exists()):
        raise SystemExit(f"[rollout_eval] seg_checkpoint missing: {seg_ckpt}")
    seg_decoder = T.build_seg_decoder(seg_ckpt, device, verbose=True)

    print("[rollout_eval] Building val DataLoader (only for scenes/latents)...")
    _, val_loader, _ = T.build_dataloaders(cfg)
    dataset = val_loader.dataset

    windows = build_windows(dataset, win_len, args.max_windows, args.seed)
    if not windows:
        raise SystemExit(f"[rollout_eval] no windows of length {win_len} found.")

    print(f"\n{'='*64}")
    print(f"  Rollout-Eval  |  K={args.k}  |  window length={win_len} "
          f"(n_input={n_input})  |  {len(windows)} windows"
          f"{' (max_windows)' if args.max_windows else ' (ALL)'}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"{'='*64}\n")

    t0 = time.time()
    curve = rollout_eval(model, seg_decoder, dataset, device, args.k, n_input, windows,
                         mode=args.mode, flow_steps=args.flow_steps)
    dt = time.time() - t0

    result = {
        "checkpoint": str(args.checkpoint),
        "config":     str(args.config),
        "mode":       args.mode,
        "flow_steps": args.flow_steps if args.mode == "sample" else None,
        "K":          args.k,
        "n_windows":  len(windows),
        "n_input":    n_input,
        "curve":      {str(k): curve[k] for k in curve},
        "eval_time_sec": round(dt, 1),
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.output) if args.output else \
        Path(args.checkpoint).parent / "rollout_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"{'='*64}")
    print(f"  DONE    |  {len(windows)} windows in {dt/60:.1f} min")
    print(f"  {'k':>2}  {'mIoU':>8}  {'std-Ratio':>10}  {'Persistence':>11}")
    for k in range(1, args.k + 1):
        c = curve[k]
        print(f"  {k:>2}  {c['mIoU']:>8.4f}  {c['std_ratio']:>10.4f}  {c['mIoU_pers']:>11.4f}")
    print(f"  -> {out}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
