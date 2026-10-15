# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""inference.py stub for syntax check"""
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--phase", choices=["frame","cell"], default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--rollout", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--no-save", dest="no_save", action="store_true",
                   help="Do not write pred_/real_ npy dumps (saves "
                        "~10GB per value). Metrics + inference_log.json stay untouched.")
    p.add_argument("--wandb_log", action="store_true")
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def resolve_inference_params(args, cfg):
    inf = cfg.get("inference", {})
    phase = args.phase or cfg.get("model", {}).get("phase", "frame")
    pnum  = 1 if phase == "frame" else 2
    checkpoint = args.checkpoint or inf.get(f"checkpoint_phase{pnum}") or f"checkpoints/phase{pnum}/best_val_loss.pt"
    output_dir = args.output or inf.get(f"output_dir_phase{pnum}") or f"predictions/phase{pnum}"
    batch_size = args.batch_size or inf.get("batch_size", 1)
    device_str = args.device or inf.get("device", "auto")
    rollout    = args.rollout if args.rollout is not None else inf.get("rollout_steps", 0)
    max_samples = args.max_samples or inf.get("max_samples", None)
    no_save     = args.no_save
    print(f"\n[Inference-Parameter]")
    print(f"  Phase:      {phase}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output:     {output_dir}")
    print(f"  Batch-Size: {batch_size}")
    print(f"  Device:     {device_str}")
    print(f"  Rollout:    {rollout}")
    print(f"  MaxSamples: {max_samples if max_samples else 'all'}")
    print(f"  No-Save:    {no_save}")
    return {"phase": phase, "checkpoint": checkpoint, "output_dir": output_dir,
            "batch_size": batch_size, "device_str": device_str, "rollout": rollout,
            "max_samples": max_samples, "no_save": no_save}

def compute_metrics(pred, target):
    with torch.no_grad():
        mse     = F.mse_loss(pred, target).item()
        cos_sim = F.cosine_similarity(pred.flatten().unsqueeze(0), target.flatten().unsqueeze(0)).item()
        pf = pred.view(256, -1);  rf = target.view(256, -1)
        pm = pf.mean(1); ps = pf.std(1); rm = rf.mean(1); rs = rf.std(1)
        dist_mean = F.mse_loss(pm, rm).item()
        dist_std  = F.mse_loss(ps, rs).item()
    return {"mse": mse, "cosine_sim": cos_sim,
            "pred_mean": pm.mean().item(), "pred_std": ps.mean().item(),
            "real_mean": rm.mean().item(), "real_std": rs.mean().item(),
            "dist_mean": dist_mean, "dist_std": dist_std}

def process_and_save(pred, target, tokens, output_dir, no_save=False):
    records = []
    for i, token in enumerate(tokens):
        pi = pred[i]; ti = target[i]
        metrics   = compute_metrics(pi, ti)
        if no_save:
            # Sweep mode: no npy dumps. pred_std/real_std (for the std ratio) come
            # from compute_metrics, NOT from the npys -> harvest stays intact.
            records.append({"token": token, "pred_path": None,
                            "real_path": None, **metrics})
        else:
            pred_path = output_dir / f"pred_{token}.npy"
            real_path = output_dir / f"real_{token}.npy"
            np.save(str(pred_path), pi.cpu().numpy())
            np.save(str(real_path), ti.cpu().numpy())
            records.append({"token": token,
                            "pred_path": str(pred_path.resolve()),
                            "real_path": str(real_path.resolve()),
                            **metrics})
    return records

def autoregressive_rollout(model, first_input, n_steps, token, output_dir, device):
    records = []
    cur = first_input.clone()
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            pred = model(cur)
            pp = output_dir / f"pred_rollout_{token}_step{step:02d}.npy"
            np.save(str(pp), pred[0].cpu().numpy())
            records.append({"token": token, "rollout_step": step,
                            "pred_path": str(pp.resolve()),
                            "pred_mean": pred[0].mean().item(),
                            "pred_std":  pred[0].std().item()})
            cur = torch.cat([cur[:, 1:], pred.unsqueeze(1)], dim=1)
    return records

def build_test_loader(cfg, batch_size):
    """
    Builds the DataLoader directly from the inference latents -- NO split.

    Inference needs no train/val/test split: we simply load ALL
    sequences from the given latent directory and predict over them.
    BEVLatentDataset with scene_indices=None loads all scenes
    without any splitting.

    config.yaml inference block:
        inference:
          pkl_path:   "...nuscenes_infos_val.pkl"
          latent_dir: "...Latents/Val"
    """
    from bev_dataset import BEVDatasetConfig, BEVLatentDataset
    from torch.utils.data import DataLoader

    inf_cfg = cfg.get("inference", {})

    # Read paths from the inference block
    pkl_path   = inf_cfg.get("pkl_path")
    latent_dir = inf_cfg.get("latent_dir")

    if not pkl_path or not latent_dir:
        raise ValueError(
            "[Error] inference.pkl_path and inference.latent_dir "
            "must be set in config.yaml."
        )

    print(f"[DataLoader] PKL:       {pkl_path}")
    print(f"[DataLoader] Latents:   {latent_dir}")

    # Dataset: scene_indices=None → ALL scenes, no split
    dataset_cfg = BEVDatasetConfig(
        sources        = [(pkl_path, latent_dir)],
        n_input_frames = cfg["model"].get("n_frames", 3),
    )
    dataset = BEVLatentDataset(dataset_cfg, scene_indices=None)

    if len(dataset) == 0:
        raise ValueError(
            "[Error] Dataset is empty. Too few frames for a sliding window?\n"
            f"  PKL:       {pkl_path}\n"
            f"  Latents:   {latent_dir}\n"
            f"  n_frames:  {cfg['model'].get('n_frames', 3)} (needs at least 4 frames per scene)"
        )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,   # Inference: always deterministic
        num_workers = 0,       # no multiprocessing needed
        pin_memory  = False,
    )

    print(f"[DataLoader] {len(dataset)} sequences → {len(loader)} batches (batch_size={batch_size})")
    return loader

def setup_model(cfg, checkpoint_path, phase, device):
    from config import ModelConfig
    from bev_world_model import BEVWorldModel
    from checkpointing import load_checkpoint
    model_cfg = ModelConfig(
        phase=phase,
        d_model=cfg["model"].get("d_model",256),
        n_heads=cfg["model"].get("n_heads",8),
        n_layers=cfg["model"].get("n_layers",4),
        dropout=cfg["model"].get("dropout",0.1),
    )
    model = BEVWorldModel(model_cfg).to(device)
    load_checkpoint(checkpoint_path=checkpoint_path, model=model,
                    optimizer=None, scheduler=None, device=str(device))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n:,} parameters | Phase: {phase} | Device: {device}")
    return model

def run_inference(model, test_loader, output_dir, device, phase, checkpoint_path,
                  rollout_steps=0, max_samples=None, no_save=False):
    all_records = []
    n_processed = 0
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Inference — Phase {phase}  |  {len(test_loader)} Batches")
    if rollout_steps > 0: print(f"  Rollout: {rollout_steps} steps")
    if max_samples:       print(f"  Smoke test: max {max_samples} samples")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            if max_samples is not None and n_processed >= max_samples:
                print(f"  Smoke test finished ({max_samples} samples).")
                break
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            tokens = batch["target_token"]
            if isinstance(tokens, str): tokens = [tokens]
            elif isinstance(tokens, torch.Tensor): tokens = [str(t.item()) for t in tokens]
            else: tokens = [str(t) for t in tokens]

            pred = model(inputs)
            records = process_and_save(pred, target, tokens, output_dir, no_save=no_save)
            all_records.extend(records)
            n_processed += len(tokens)

            if rollout_steps > 0:
                autoregressive_rollout(model, inputs[:1], rollout_steps,
                                       tokens[0], output_dir/"rollout", device)

            if (bidx+1) % 10 == 0 or bidx == 0:
                elapsed = time.time() - t_start
                rem = (elapsed/(bidx+1))*(len(test_loader)-bidx-1) if bidx > 0 else 0
                mm = np.mean([r["mse"] for r in all_records])
                mc = np.mean([r["cosine_sim"] for r in all_records])
                md = np.mean([r["dist_mean"] for r in all_records])
                print(f"  [{bidx+1:>4}/{len(test_loader)}] n={n_processed} | "
                      f"MSE={mm:.5f} | CosSim={mc:.4f} | dist={md:.5f}"
                      + (f" | ~{rem/60:.1f}min" if rem > 0 else ""))

    total = time.time() - t_start
    summary = {
        "n_samples": n_processed,
        "mean_mse":         float(np.mean([r["mse"]        for r in all_records])),
        "std_mse":          float(np.std( [r["mse"]        for r in all_records])),
        "mean_cosine_sim":  float(np.mean([r["cosine_sim"] for r in all_records])),
        "mean_dist_mean":   float(np.mean([r["dist_mean"]  for r in all_records])),
        "mean_dist_std":    float(np.mean([r["dist_std"]   for r in all_records])),
        "mean_dist":        float(np.mean([r["dist_mean"]+r["dist_std"] for r in all_records])),
        "phase": phase, "checkpoint": checkpoint_path,
        "total_time_sec": round(total, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n{'='*60}")
    print(f"  DONE! {n_processed} samples in {total:.1f}s")
    print(f"  MSE:   {summary['mean_mse']:.6f} ± {summary['std_mse']:.6f}")
    print(f"  CosSim:{summary['mean_cosine_sim']:.4f}")
    print(f"  dist:  {summary['mean_dist']:.6f}  ← decoder indicator (→ 0 = good)")
    print(f"{'='*60}\n")
    return summary, all_records

def save_results(all_records, summary, output_dir):
    log_path = output_dir / "inference_log.json"
    with open(log_path, "w") as f:
        json.dump({"summary": summary, "records": all_records}, f, indent=2)
    print(f"[Saved] {log_path}  ({len(all_records)} records)")

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    params = resolve_inference_params(args, cfg)

    ds = params["device_str"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if ds == "auto" else torch.device(ds)
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    cfg["model"]["phase"] = params["phase"]
    model       = setup_model(cfg, params["checkpoint"], params["phase"], device)
    test_loader = build_test_loader(cfg, params["batch_size"])

    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if params["rollout"] > 0:
        (output_dir / "rollout").mkdir(parents=True, exist_ok=True)

    summary, all_records = run_inference(
        model=model, test_loader=test_loader, output_dir=output_dir,
        device=device, phase=params["phase"], checkpoint_path=params["checkpoint"],
        rollout_steps=params["rollout"], max_samples=params["max_samples"],
        no_save=params["no_save"])

    save_results(all_records, summary, output_dir)
    print(f"✓ Output: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
