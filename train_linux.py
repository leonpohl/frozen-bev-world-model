# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
train.py — BEV World Model: main training script
=========================================================

This script is the "conductor" — it imports all sub-scripts from code/
and orchestrates the entire training process.

DIRECTORY STRUCTURE:
    bev_worldmodel/
    ├── train.py          ← this script
    ├── config.yaml       ← all hyperparameters
    └── code/
        ├── config.py
        ├── bev_world_model.py
        ├── bev_dataloader.py
        ├── checkpointing.py
        └── ...

USAGE:
    # Phase 1 (frame level, 3 tokens, fast)
    python train.py --config config.yaml --phase frame

    # Phase 2 (cell level, 3072 tokens, scientific core)
    python train.py --config config.yaml --phase cell

    # Resume training from a checkpoint
    python train.py --config config.yaml --phase frame \\
        --resume checkpoints/phase1/epoch_020.pt

All modules (config.py, bev_world_model.py, ...) live in the repository
root, next to this script — plain flat imports, no sys.path tricks.
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
import os
import time
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# SSIM loss — optional, falls back to MSE if not installed
# Installation: pip install pytorch-msssim
try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("[Info] pytorch-msssim not installed. SSIM loss disabled.")
    print("       Install with: pip install pytorch-msssim")

# --- add code/ to the Python search path ---
# __file__ = .../bev_worldmodel/train.py
# Path(__file__).parent = .../bev_worldmodel/
# / "code" = .../bev_worldmodel/code/
# Now we can import all modules directly (flat layout)
from config          import ModelConfig
from bev_world_model import BEVWorldModel
# bev_dataloader is imported lazily inside build_dataloaders() (smoke_test-compatible)
from checkpointing   import save_checkpoint, load_checkpoint, find_best_checkpoint

# Seg decoder (pure PyTorch, no mmdet3d)
# seg_decoder_torch.py lives in the repository root next to this script.
# If the import fails (e.g. bevfusion-seg.pth unreachable), training
# continues anyway — only validate_seg() is skipped.
try:
    from seg_decoder_torch import build_seg_decoder, compute_iou, MAP_CLASSES, MAP_SCORE
    SEG_DECODER_AVAILABLE = True
except ImportError as _seg_err:
    SEG_DECODER_AVAILABLE = False
    print(f"[Warning] seg_decoder_torch not importable: {_seg_err}")
    print("          validate_seg() is skipped (Val loss only).")

# wandb is optional — if not installed, a dummy is used
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Logging is skipped.")

# matplotlib for visualizations (optional)
try:
    import matplotlib
    matplotlib.use("Agg")   # no display needed (headless server)
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[Warning] matplotlib not installed. Visualizations are skipped.")


# =============================================================================
# BLOCK 1 — Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Seeds ALL random generators with the same seed.

    Why all of them?
        - random:           Python built-in (e.g. DataLoader shuffling)
        - numpy:            NumPy operations
        - torch:            CPU tensors
        - torch.cuda:       GPU tensors (all GPUs)
        - deterministic:    cuDNN always picks the same algorithm

    Without this, two identical runs with the same hyperparameters
    produce different results — which makes ablation studies
    meaningless. seed=42 is constant throughout the whole master's thesis.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN autotune the fastest algorithm per conv
    # shape. Our shapes are constant -> a one-time search in the first
    # step, pure gain afterwards. Requires deterministic=False; bit-level
    # reproducibility OF THE SAME seed is thereby dropped. The noise-floor
    # thresholds are based on seed-TO-seed variance and remain unaffected.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# =============================================================================
# BLOCK 2 — Config loading
# =============================================================================

def load_config(config_path: str, phase_override: Optional[str] = None) -> dict:
    """
    Reads config.yaml and returns a dict.

    The optional phase_override comes from the --phase argument.
    It overrides model.phase from the YAML file — so the same
    config.yaml can be used for both phases:

        python train.py --config config.yaml --phase frame
        python train.py --config config.yaml --phase cell

    without editing the file every time.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)     # YAML → Python dict

    if phase_override is not None:
        cfg["model"]["phase"] = phase_override
        print(f"[Config] Phase overridden: '{phase_override}'")

    return cfg


# =============================================================================
# BLOCK 3 — Loss function
# =============================================================================

# --- Optional extra loss terms (default 0 = bit-identical to all ------------
# historical runs). Module-level block, set in main() from train_cfg:
#   lambda_sliced: sliced distribution guard — M random projections of the C
#     channels, mean+spread of the projections matched between pred and the
#     REAL target (generalization of per-channel std to arbitrary
#     directions; deliberately AGAINST the real distribution, not N(0,I)).
#   energy_alpha: energy-weighted reconstruction — per-cell weight
#     w = 1 + alpha * E_real/mean(E_real) on TERM 1 (cell-weight path).
#   lambda_peak: peak preservation — MSE(maxpool3x3(E_pred), maxpool3x3(E_real))
#     on the cell energy (per-channel L2), tolerant to 1-cell jitter.
EXTRA_LOSS = {"lambda_sliced": 0.0, "sliced_M": 64,
              "energy_alpha": 0.0, "lambda_peak": 0.0}


def compute_loss(
    pred:         torch.Tensor,   # [B, 256, 128, 128] — model output
    target:       torch.Tensor,   # [B, 256, 128, 128] — real next latent
    # --- all weights explicit, fully YAML-controllable ---
    lambda_mse:   float = 1.0,   # Multi-Scale MSE (war implizit 1.0)
    lambda_cos:   float = 0.1,   # Cosine-Similarity (Null-Kollaps-Schutz)
    lambda_mean:  float = 0.1,   # Channel-mean term  (NEW: split from std)
    lambda_std:   float = 0.1,   # Channel-Std-Term         (NEU: eigener Regler)
    lambda_grad:  float = 0.0,   # Gradient term / anti-blur (NEW, default OFF)
    lambda_ssim:  float = 0.1,   # SSIM term (switchable off via 0.0)
    recon_loss:   str   = "mse", # TERM-1-Distanz — "mse"|"l1"|"smooth_l1"
    cell_weight:  torch.Tensor = None,  # [B,128,128] cell weights for
                                        # TERM 1 (None = old path, bit-identical)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss — fully configurable via YAML.

    Each term has its own lambda; lambda=0 switches the term off with
    zero compute cost (no gradient, no allocation).

    ┌─────────────────────────────────────────────────────────────────┐
    │ TERM 1: Multi-Scale MSE  (lambda_mse, Default 1.0)              │
    │   (MSE_128 + MSE_64 + MSE_32) / 3 — coarse + medium + fine      │
    │   structure optimized simultaneously.                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 2: Cosine Similarity  (lambda_cos, Default 0.1)            │
    │   1 - CosSim(pred.flatten, target.flatten)                      │
    │   Prevents null collapse on sparse input (mean≈0.14).           │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 3a: Channel mean  (lambda_mean, default 0.1)               │
    │   MSE(pred_mean_per_channel, target_mean_per_channel)           │
    │   Ensures statistical decoder compatibility.                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 3b: Channel-Std  (lambda_std, Default 0.1)                 │
    │   MSE(pred_std_per_channel, target_std_per_channel)             │
    │   Primary lever against regression-to-mean / std gap.           │
    │   CORE of the sweep study.                              │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 4: Spatial gradient  (lambda_grad, default 0.0 = OFF)      │
    │   MSE(∂pred/∂x, ∂target/∂x) + MSE(∂pred/∂y, ∂target/∂y)       │
    │   Anti-blur: penalizes missing high frequencies / spatial edges.│
    │   Finite differences in x and y — no extra cost when 0.         │
    ├─────────────────────────────────────────────────────────────────┤
    │ TERM 5: SSIM  (lambda_ssim, default 0.1, off via 0.0)           │
    │   1 - SSIM(channel-averaged)                                    │
    │   Structural similarity: edges, contrasts, local patterns.      │
    └─────────────────────────────────────────────────────────────────┘

    Returns:
        (loss_total, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim)
        — all 7 terms separately, for wandb logging and correctness checks.

    Baseline equivalence:
        lambda_mse=1.0, lambda_cos=0.1, lambda_mean=0.1, lambda_std=0.1,
        lambda_grad=0.0, lambda_ssim=0.1
        → loss_total ≈ old loss (< 1e-6 deviation from
          Float32 reassociation; no semantic difference).
    """
    # ------------------------------------------------------------------
    # TERM 1: multi-scale reconstruction (distance selectable)
    #   recon_loss controls ONLY TERM 1: "mse" (default, backward-compatible),
    #   "l1" (absolute) or "smooth_l1" (Huber, close to DINO-Foresight). The
    #   return name stays loss_mse (logging/tuple compatibility).
    # ------------------------------------------------------------------
    if recon_loss == "mse":
        _recon = F.mse_loss
    elif recon_loss == "l1":
        _recon = F.l1_loss
    elif recon_loss == "smooth_l1":
        _recon = F.smooth_l1_loss
    else:
        raise ValueError(
            f"recon_loss must be 'mse'|'l1'|'smooth_l1', was '{recon_loss}'.")
    if EXTRA_LOSS["energy_alpha"] > 0 and cell_weight is None:
        # Energy weighting: cell weight from the REAL cell energy (on the fly,
        # no GT files needed); mean-1 normalization per sample.
        _E = target.norm(dim=1)                                   # [B,H,W]
        _E = _E / _E.mean(dim=[-2, -1], keepdim=True).clamp_min(1e-8)
        cell_weight = 1.0 + EXTRA_LOSS["energy_alpha"] * _E
    if cell_weight is None:
        # Old path — UNCHANGED (bit-identical to all historical runs).
        loss_128 = _recon(pred, target)
        loss_64  = _recon(
            F.avg_pool2d(pred,   kernel_size=2),
            F.avg_pool2d(target, kernel_size=2),
        )
        loss_32  = _recon(
            F.avg_pool2d(pred,   kernel_size=4),
            F.avg_pool2d(target, kernel_size=4),
        )
    else:
        # cell-weighted TERM 1. Per scale: average the element loss over
        # channels -> [B,H,W], average with the (pooled) weights and divide
        # by the mean weight — for w==1 exactly the unweighted value.
        def _wrecon(p, t, w):
            el = _recon(p, t, reduction="none").mean(dim=1)   # [B,H,W]
            return (el * w).mean() / w.mean()
        w128 = cell_weight
        w64  = F.avg_pool2d(w128.unsqueeze(1), kernel_size=2).squeeze(1)
        w32  = F.avg_pool2d(w128.unsqueeze(1), kernel_size=4).squeeze(1)
        loss_128 = _wrecon(pred, target, w128)
        loss_64  = _wrecon(F.avg_pool2d(pred, 2), F.avg_pool2d(target, 2), w64)
        loss_32  = _wrecon(F.avg_pool2d(pred, 4), F.avg_pool2d(target, 4), w32)
    loss_mse = (loss_128 + loss_64 + loss_32) / 3.0

    # ------------------------------------------------------------------
    # TERM 2: Cosine Similarity
    # ------------------------------------------------------------------
    if lambda_cos > 0:
        loss_cos = 1.0 - F.cosine_similarity(
            pred.flatten(1),
            target.flatten(1),
            dim=1,
        ).mean()
    else:
        loss_cos = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # TERM 3a+b: channel statistics (split for the sweep)
    # ------------------------------------------------------------------
    # mean/std over H and W per channel: [B, 256]
    pred_ch_mean   = pred.mean(dim=[-2, -1])
    target_ch_mean = target.mean(dim=[-2, -1])
    pred_ch_std    = pred.std(dim=[-2, -1])
    target_ch_std  = target.std(dim=[-2, -1])

    # Separate terms → separate levers (std, optional mean)
    loss_mean = F.mse_loss(pred_ch_mean, target_ch_mean)
    loss_std  = F.mse_loss(pred_ch_std,  target_ch_std)

    # ------------------------------------------------------------------
    # TERM 4: spatial gradient (anti-blur, default OFF)
    # ------------------------------------------------------------------
    # Finite differences: pred_dx [B,256,H,W-1], pred_dy [B,256,H-1,W]
    # Compute only when lambda_grad > 0 → no overhead in the default case
    if lambda_grad > 0:
        pred_dx   = pred[..., :, 1:]   - pred[..., :, :-1]    # ∂/∂x
        pred_dy   = pred[..., 1:, :]   - pred[..., :-1, :]    # ∂/∂y
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        loss_grad = (F.mse_loss(pred_dx, target_dx) +
                     F.mse_loss(pred_dy, target_dy))
    else:
        loss_grad = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # TERM 5: SSIM (switchable off via lambda_ssim=0)
    # ------------------------------------------------------------------
    if SSIM_AVAILABLE and lambda_ssim > 0:
        pred_2d    = pred.mean(dim=1, keepdim=True).float()
        target_2d  = target.mean(dim=1, keepdim=True).float()
        data_range = float(target_2d.max() - target_2d.min() + 1e-6)
        loss_ssim  = (1.0 - ssim_fn(pred_2d, target_2d,
                                     data_range=data_range,
                                     size_average=True))
    else:
        loss_ssim = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # Optional terms: sliced distribution guard + peak term (default 0 = OFF)
    # ------------------------------------------------------------------
    _l_sl = EXTRA_LOSS["lambda_sliced"]
    if _l_sl > 0:
        _M = int(EXTRA_LOSS["sliced_M"])
        _u = torch.randn(_M, pred.shape[1], device=pred.device, dtype=pred.dtype)
        _u = _u / _u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        _pp = torch.einsum("mc,bchw->bmhw", _u, pred).flatten(2)    # [B,M,HW]
        _tt = torch.einsum("mc,bchw->bmhw", _u, target).flatten(2)
        loss_sliced = (F.mse_loss(_pp.mean(-1), _tt.mean(-1))
                       + F.mse_loss(_pp.std(-1), _tt.std(-1)))
    else:
        loss_sliced = torch.tensor(0.0, device=pred.device)
    _l_pk = EXTRA_LOSS["lambda_peak"]
    if _l_pk > 0:
        _Ep = F.max_pool2d(pred.norm(dim=1, keepdim=True), 3, stride=1, padding=1)
        _Et = F.max_pool2d(target.norm(dim=1, keepdim=True), 3, stride=1, padding=1)
        loss_peak = F.mse_loss(_Ep, _Et)
    else:
        loss_peak = torch.tensor(0.0, device=pred.device)

    # ------------------------------------------------------------------
    # Combination — fully YAML-driven
    # ------------------------------------------------------------------
    loss_total = (lambda_mse  * loss_mse
                + lambda_cos  * loss_cos
                + lambda_mean * loss_mean
                + lambda_std  * loss_std
                + lambda_grad * loss_grad
                + lambda_ssim * loss_ssim
                + _l_sl       * loss_sliced
                + _l_pk       * loss_peak)

    return loss_total, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim


# =============================================================================
# BLOCK 4 — One training epoch
# =============================================================================

def train_one_epoch(
    model:       torch.nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler:      torch.cuda.amp.GradScaler,
    device:      torch.device,
    lambda_mse:   float = 1.0,
    lambda_cos:   float = 0.1,
    lambda_mean:  float = 0.1,
    lambda_std:   float = 0.1,
    lambda_grad:  float = 0.0,
    lambda_ssim:  float = 0.1,
    recon_loss:   str   = "mse",   # passed through to compute_loss
    ema_state:    Optional[dict] = None,   # shadow weights for EMA
    ema_decay:    float = 0.9999,          # None -> EMA branch fully inactive
    seg_decoder=None,                      # frozen decoder for the task loss
    lambda_task:  float = 0.0,             # 16f: weight of BCE(decode(pred), decode(real))
    task_every:   int   = 1,               # task loss only on every n-th batch
    vae_active:   bool  = False,           # 18/B1: CVAE path (posterior sampling) on
    vae_beta:     float = 0.0,             # 18/B1: annealed KL weight of this epoch
    kl_free_bits: float = 0.05,            # 18/B1: minimum KL/dim without penalty
    flow_active:  bool  = False,           # 18/B2: FM residual training (frozen backbone)
) -> Tuple[float, float, float, float, float, float, float, dict]:
    """
    Runs exactly one training epoch.

    Iterates over all batches in the train_loader, computes loss,
    backpropagates gradients and updates the weights.

    MIXED PRECISION (autocast + GradScaler):
        Normalerweise: alles in float32 (4 Byte/Zahl)
        With mixed precision: forward pass in float16 (2 bytes/number)
            → 2× faster, ~half the VRAM
            → especially important for Phase 2 (3072 tokens)

        GradScaler problem:
            float16 has less precision — small gradients become 0.
            GradScaler internally multiplies the loss by a large
            scale factor before backward() and divides afterwards.
            This keeps even small gradients representable in float16.

    GRADIENT CLIPPING:
        clip_grad_norm_(max_norm=1.0):
            If the total norm of all gradients > 1.0, all gradients
            are scaled proportionally until the norm = 1.0.
            Prevents exploding gradients, especially in the first
            epochs while the weights are still random.

    TIMING-INSTRUMENTIERUNG:
        Per batch, data_time (waiting for the next batch from the loader,
        i.e. the time between the end of the last compute step and
        receiving the next batch) and compute_time (forward+backward+
        optimizer step) are measured with time.perf_counter() and summed
        over the epoch.
        data_time_ratio = data_time / (data_time + compute_time):
            high   -> I/O is still the bottleneck (loader is lagging behind)
            low    -> GPU is the bottleneck, the data pipeline suffices
        Makes the decision gate MEASURABLE instead of estimating it
        via nvidia-smi.

    Returns:
        (avg_total_loss, avg_mse_loss, avg_cos_loss, avg_mean_loss,
         avg_std_loss, avg_grad_loss, avg_ssim_loss, timing) — averages
        over the epoch; timing is a dict with data_time/compute_time/
        data_time_ratio (seconds or fraction 0..1).
    """
    model.train()   # dropout ON, BatchNorm in training mode

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_mean = 0.0
    total_std  = 0.0
    total_grad = 0.0
    total_ssim = 0.0
    total_task = 0.0     # 16f: decoder task loss (averaged over task batches only)
    n_task     = 0
    total_kl   = 0.0     # 18/B1: raw KL (before free bits/beta), collapse diagnostic
    n_kl       = 0
    n_batches  = len(loader)

    # tee-stable progress: writes directly to stdout with \r + flush.
    # Unlike tqdm (which buffers through a pipe), this stays visible live
    # in the tee log file / tmux scrollback.
    _t_epoch = time.time()

    # Timing instrumentation: data_time = time waiting for the loader (time
    # between the end of the previous compute step and receiving this
    # batch), compute_time = everything after that up to and including
    # the optimizer step.
    total_data_time    = 0.0
    total_compute_time = 0.0
    _t_mark = time.perf_counter()   # start: before the first batch wait

    for _bi, batch in enumerate(loader):
        _t_got_batch = time.perf_counter()
        total_data_time += _t_got_batch - _t_mark

        # Data to GPU (or CPU if no GPU)
        # non_blocking=True: the transfer runs asynchronously, CPU keeps working
        # 1d: the loader delivers fp16 (half the host-to-device volume); upcast
        # to fp32 happens HERE on the GPU (.float() after .to), before the
        # model/AvgPool.
        inputs = batch["input"].to(device, non_blocking=True).float()   # [B, 3, 256, 128, 128]
        target = batch["target"].to(device, non_blocking=True).float()  # [B, 256, 128, 128]

        # Clear gradients from the previous step
        # set_to_none=True is faster than zero_grad() since no assignment to 0
        optimizer.zero_grad(set_to_none=True)

        _ego = batch["ego_delta"].to(device).float() if "ego_delta" in batch else None
        # Cell weights (only present when data.gt_masks_train_path is set)
        _cw = batch["cell_weight"].to(device, non_blocking=True).float() \
            if "cell_weight" in batch else None

        # --- 18/B2: flow-matching branch (replaces the regression loss) -----
        # The frozen backbone provides x_det; only v_theta is learned on the
        # residual r = target - x_det (see thesis, chapter on generative heads).
        if flow_active:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    x_det = model(inputs, _ego)
                r    = target - x_det
                eps  = torch.randn_like(r)
                tau  = torch.rand(r.shape[0], device=device)
                r_tau = (1.0 - tau[:, None, None, None]) * eps + tau[:, None, None, None] * r
                v    = model.flow(r_tau, tau, x_det, inputs[:, -1])
                loss = F.mse_loss(v.float(), (r - eps).float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _t_mark = time.perf_counter()
            total_compute_time += _t_mark - _t_got_batch
            if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
                elapsed = time.time() - _t_epoch
                it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
                sys.stdout.write(
                    f"\r  Train {100.0*(_bi+1)/n_batches:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                    f"fm_loss={total_loss/(_bi+1):.4f}  {it_s:4.1f} it/s  ETA {eta/60:4.1f}min   ")
                sys.stdout.flush()
            continue

        # --- Forward Pass in float16 ---
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs, _ego,
                         target=target if vae_active else None)  # [B, 256, 128, 128]
            loss, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim = compute_loss(
                pred, target,
                lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
                recon_loss=recon_loss,
                cell_weight=_cw,   # None = old path
            )
            # --- KL term (18/B1) -----------------------------------------
            # KL(q||p) of two diagonal Gaussians, per dim; clamp with free
            # bits (no gradient below the threshold -> anti-collapse), then
            # weight with the annealed beta. Raw KL logged separately.
            if vae_active:
                mu_q, lv_q, mu_p, lv_p = [t.float() for t in model.last_vae_stats]
                kl_dim = 0.5 * (lv_p - lv_q
                                + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p)
                                - 1.0)                                  # [B, Z]
                kl_raw = kl_dim.sum(dim=1).mean()
                kl_pen = kl_dim.clamp(min=kl_free_bits).sum(dim=1).mean()
                loss = loss + vae_beta * kl_pen
                total_kl += kl_raw.item()
                n_kl     += 1

        # --- Decoder task loss -------------------------------------
        # Aligns training with the metric (LAW/DriveFuture pattern):
        # BCE between decode(pred) and decode(real) — the decoder is a
        # multi-label sigmoid, hence BCE instead of CE. The decoder stays
        # frozen/eval; gradients flow ONLY through it to the pred latent.
        # Deliberately outside autocast (the decoder is float32-calibrated,
        # like validate_seg).
        _task_active = (seg_decoder is not None and lambda_task > 0.0
                        and (_bi % task_every == 0))
        if _task_active:
            with torch.no_grad():
                _real_logits = seg_decoder.forward_logits(target)
                _real_mask   = (torch.sigmoid(_real_logits) >= MAP_SCORE).float()
            _pred_logits = seg_decoder.forward_logits_grad(pred.float())
            loss_task = F.binary_cross_entropy_with_logits(_pred_logits, _real_mask)
            loss = loss + lambda_task * loss_task
            total_task += loss_task.item()
            n_task     += 1

        # --- Backward Pass (skaliert) ---
        scaler.scale(loss).backward()

        # De-scale gradients before clipping (otherwise wrong comparison)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights (GradScaler checks whether gradients had inf/nan)
        scaler.step(optimizer)

        # Adjust the scale factor for the next step
        # (increased when all OK, decreased on inf/nan)
        scaler.update()

        # --- EMA update -----------------------------------------
        # After the optimizer step: shadow weights follow the just-updated
        # model (ema = decay*ema + (1-decay)*param). Only active when
        # ema_state was passed -> otherwise bit-identical to the old path.
        if ema_state is not None:
            with torch.no_grad():
                for _n, _p in model.named_parameters():
                    ema_state[_n].mul_(ema_decay).add_(_p.detach(), alpha=1.0 - ema_decay)

        # Accumulate losses (.item() detaches the tensor from the graph)
        total_loss += loss.item()
        total_mse  += loss_mse.item()
        total_cos  += loss_cos.item()
        total_mean += loss_mean.item()
        total_std  += loss_std.item()
        total_grad += loss_grad.item()
        total_ssim += loss_ssim.item()

        _t_mark = time.perf_counter()
        total_compute_time += _t_mark - _t_got_batch

        # --- Progress every 20 batches (tee-stable) ---
        if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
            elapsed = time.time() - _t_epoch
            it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
            eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
            pct     = 100.0 * (_bi + 1) / n_batches
            sys.stdout.write(
                f"\r  Train {pct:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                f"loss={total_loss/(_bi+1):.4f}  "
                f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
            )
            sys.stdout.flush()

    # Clear the progress line -> only the epoch overview remains at the end
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    _denom = total_data_time + total_compute_time
    timing = {
        "data_time":       total_data_time,
        "compute_time":    total_compute_time,
        "data_time_ratio": (total_data_time / _denom) if _denom > 0 else 0.0,
        # mean task loss (kept in the timing dict to keep the return
        # arity stable); None when the task loss is inactive.
        "task_loss":       (total_task / n_task) if n_task > 0 else None,
        # 18/B1: raw KL (collapse diagnostic); None when CVAE is inactive.
        "kl_loss":         (total_kl / n_kl) if n_kl > 0 else None,
    }

    return (total_loss / n_batches, total_mse  / n_batches,
            total_cos  / n_batches, total_mean / n_batches,
            total_std  / n_batches, total_grad / n_batches,
            total_ssim / n_batches, timing)


# =============================================================================
# BLOCK 5 — Validation
# =============================================================================

def validate(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    lambda_mse:   float = 1.0,
    lambda_cos:   float = 0.1,
    lambda_mean:  float = 0.1,
    lambda_std:   float = 0.1,
    lambda_grad:  float = 0.0,
    lambda_ssim:  float = 0.1,
    recon_loss:   str   = "mse",   # passed through to compute_loss
    flow_active:  bool  = False,   # 18/B2: FM val loss instead of regression loss
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Evaluates the model on the validation set.

    NO gradient update — forward pass + loss only.

    model.eval() vs model.train():
        eval():   dropout OFF (deterministic outputs)
                  BatchNorm uses stored running_mean/var
        train():  dropout ON (random neurons deactivated)
                  BatchNorm computes statistics fresh

    torch.no_grad():
        Tells PyTorch: "No autograd tracking needed."
        PyTorch normally stores all intermediate results for
        backward() — that costs a lot of memory. With no_grad() this is
        unnecessary. Attention: only for evaluation/inference, never in training!

    Returns:
        (avg_total_loss, avg_mse_loss, avg_cos_loss, avg_dist_loss, avg_ssim_loss)
    """
    model.eval()

    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_mean = 0.0
    total_std  = 0.0
    total_grad = 0.0
    total_ssim = 0.0
    n_batches  = len(loader)

    _t_val = time.time()

    with torch.no_grad():
        for _bi, batch in enumerate(loader):
            inputs = batch["input"].to(device, non_blocking=True).float()   # 1d: GPU-Upcast
            target = batch["target"].to(device, non_blocking=True).float()

            _ego = batch["ego_delta"].to(device).float() if "ego_delta" in batch else None
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if flow_active:
                    # 18/B2: deterministic FM val loss — tau/eps from a fixed
                    # SEEDED generator so that the val-loss patience runs on a
                    # low-noise, comparable quantity.
                    _g = torch.Generator(device=device.type); _g.manual_seed(1234 + _bi)
                    x_det = model(inputs, _ego)
                    r    = target - x_det
                    eps  = torch.empty_like(r).normal_(generator=_g)
                    tau  = torch.empty(r.shape[0], device=device).uniform_(generator=_g)
                    r_tau = (1.0 - tau[:, None, None, None]) * eps + tau[:, None, None, None] * r
                    v    = model.flow(r_tau, tau, x_det, inputs[:, -1])
                    loss = F.mse_loss(v.float(), (r - eps).float())
                    z = torch.zeros((), device=device)
                    loss_mse = loss_cos = loss_mean = loss_std = loss_grad = loss_ssim = z
                else:
                    pred = model(inputs, _ego)
                    loss, loss_mse, loss_cos, loss_mean, loss_std, loss_grad, loss_ssim = compute_loss(
                        pred, target,
                        lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
                        recon_loss=recon_loss,
                    )

            total_loss += loss.item()
            total_mse  += loss_mse.item()
            total_cos  += loss_cos.item()
            total_mean += loss_mean.item()
            total_std  += loss_std.item()
            total_grad += loss_grad.item()
            total_ssim += loss_ssim.item()

            # --- Progress every 20 batches (tee-stable, cleared at the end) ---
            if (_bi + 1) % 20 == 0 or (_bi + 1) == n_batches:
                elapsed = time.time() - _t_val
                it_s    = (_bi + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_batches - _bi - 1) / it_s if it_s > 0 else 0.0
                pct     = 100.0 * (_bi + 1) / n_batches
                sys.stdout.write(
                    f"\r  Val   {pct:5.1f}%  [{_bi+1:>4}/{n_batches}]  "
                    f"loss={total_loss/(_bi+1):.4f}  "
                    f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
                )
                sys.stdout.flush()

    # Clear the progress line
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    return (total_loss / n_batches, total_mse / n_batches,
            total_cos  / n_batches, total_mean / n_batches,
            total_std  / n_batches, total_grad / n_batches,
            total_ssim / n_batches)


# =============================================================================
# BLOCK 5b — Decode validation
# =============================================================================

def validate_seg(
    model:        torch.nn.Module,
    seg_decoder,                        # SegDecoder instance, frozen
    subset_inputs:  torch.Tensor,       # [N, 3, 256, 128, 128] val-Subset
    subset_targets: torch.Tensor,       # [N, 256, 128, 128]    corresponding reals
    real_masks_cache: list,             # precomputed real masks (list of [6,200,200])
    device:       torch.device,
    flow_steps:   int = 0,              # 18/B2: >0 -> pred = flow_sample(steps),
                                        # fixed seed (selection on sampling quality)
) -> Tuple[float, dict]:
    """
    Decodes val-subset predictions through the frozen BEVFusion seg
    decoder and returns mean mIoU + per-class values.

    PROCEDURE PER SAMPLE:
      1. model(input) -> pred-Latent [256,128,128]              (World-Model)
      2. seg_decoder.latent_to_mask(pred) -> pred_mask [6,200,200] (BEVFusion)
      3. compute_iou(pred_mask, real_mask_cached)
      -> mIoU averaged over all subset samples

    OPTIMIZATION: The real masks depend only on the fixed target latent and
    the frozen decoder — they never change across epochs. They are
    precomputed once in build_seg_val_cache() and only read here.
    This way validate_seg() needs only N forward passes through the world
    model + N decoder passes for the preds (instead of 2N) per epoch.

    MODULARITY:
    For PNG output (pred/real/comparison), the visualization could call
    render_mask() + save_comparison() here after compute_iou() —
    the functions are available in seg_decoder_torch.py but deliberately
    NOT called here.

    Returns:
        (mean_miou, per_class_dict)
    """
    if not SEG_DECODER_AVAILABLE or seg_decoder is None:
        return 0.0, {}

    model.eval()
    all_ious = []
    n_total  = len(subset_inputs)
    _t_dec   = time.time()

    with torch.no_grad():
        # Process samples one by one: the decoder expects [1,256,128,128];
        # batch decoding would be possible but unnecessary (N is small, ~300).
        for i in range(n_total):
            inp = subset_inputs[i:i+1].to(device)   # [1,3,256,128,128]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if flow_steps > 0:                   # 18/B2: ein FM-Sample
                    _g = torch.Generator(device=device.type); _g.manual_seed(9000 + i)
                    pred_latent = model.flow_sample(inp, steps=flow_steps, generator=_g)
                else:
                    pred_latent = model(inp)         # [1,256,128,128]

            # The decoder deliberately runs in float32 (autocast=False in latent_to_mask)
            pred_mask, _ = seg_decoder.latent_to_mask(
                pred_latent[0], device=device)       # [6,200,200] bool

            real_mask = real_masks_cache[i]          # [6,200,200] bool, vorberechnet

            iou = compute_iou(pred_mask, real_mask, MAP_CLASSES)
            all_ious.append(iou)

            # --- Progress every 20 samples (tee-stable, cleared at the end) ---
            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                elapsed = time.time() - _t_dec
                it_s    = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_total - i - 1) / it_s if it_s > 0 else 0.0
                pct     = 100.0 * (i + 1) / n_total
                run_miou = float(np.mean([d['mIoU'] for d in all_ious]))
                sys.stdout.write(
                    f"\r  Decode {pct:5.1f}%  [{i+1:>3}/{n_total}]  "
                    f"mIoU={run_miou:.4f}  "
                    f"{it_s:4.1f} it/s  ETA {eta/60:4.1f}min   "
                )
                sys.stdout.flush()

    # Clear the progress line
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Mean over the subset
    mean_miou = float(np.mean([d['mIoU'] for d in all_ious]))
    per_class = {}
    for cls in MAP_CLASSES:
        vals = [d[cls] for d in all_ious if d.get(cls) is not None]
        per_class[cls] = round(float(np.mean(vals)), 4) if vals else None

    return mean_miou, per_class


def build_seg_val_cache(
    seg_decoder,
    val_loader:     torch.utils.data.DataLoader,
    device:         torch.device,
    n_subset:       int   = 300,
    seed:           int   = 42,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Selects a fixed, reproducible subset of the Val dataset and
    precomputes the real masks once.

    IMPORTANT: accesses val_loader.dataset directly by index (no loader
    iteration) -- avoids worker deadlocks when breaking out of a running
    DataLoader with num_workers > 0.

    Returns:
        subset_inputs   [N, 3, 256, 128, 128]  CPU-Tensor
        subset_targets  [N, 256, 128, 128]      CPU-Tensor
        real_masks      list of N bool [6,200,200]
    """
    if not SEG_DECODER_AVAILABLE or seg_decoder is None:
        return None, None, []

    print(f"[validate_seg] Building Val subset (n={n_subset}, seed={seed})...")
    rng     = np.random.default_rng(seed)
    dataset = val_loader.dataset
    n_avail = len(dataset)
    n_take  = min(n_subset, n_avail)

    # Reproducible index selection directly on the dataset
    idx = sorted(rng.choice(n_avail, size=n_take, replace=False).tolist())

    # Direct __getitem__ accesses -- no DataLoader, no workers
    inputs_list  = []
    targets_list = []
    print(f"[validate_seg] Loading {n_take} samples directly from the dataset...")
    for i in idx:
        sample = dataset[i]
        inputs_list.append(sample["input"])
        targets_list.append(sample["target"])

    # 1d: the loader/dataset now delivers fp16. The seg decoder (real masks
    # below, decode_validation) deliberately runs in float32 and produces the
    # mIoU reference -> upcast to fp32 ONCE here so the decode/metric path
    # stays bit-identical to before (no fp16 in the mIoU).
    subset_inputs  = torch.stack(inputs_list,  dim=0).cpu().float()
    subset_targets = torch.stack(targets_list, dim=0).cpu().float()

    # Decode the real masks once (never changes)
    print(f"[validate_seg] Decoding {n_take} real masks (one-off)...")
    real_masks = []
    with torch.no_grad():
        for i in range(n_take):
            mask, _ = seg_decoder.latent_to_mask(
                subset_targets[i], device=device)
            real_masks.append(mask)

    print(f"[validate_seg] Cache ready: {n_take} samples, "
          f"{len(real_masks)} real masks precomputed.")
    return subset_inputs, subset_targets, real_masks


# =============================================================================
# BLOCK 6 — Visualization for wandb
# =============================================================================

def visualize_predictions(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    epoch:     int,
    n_samples: int = 4,
) -> None:
    """
    Creates channel-averaged heatmaps (predicted vs. real) and uploads
    them as wandb.Image.

    WHY CHANNEL AVERAGING?
        The latent tensor has 256 channels — not directly visualizable.
        .mean(dim=1): [B, 256, 128, 128] → [B, 128, 128]
        Each pixel shows the average activation across all channels.
        This gives an intuitive picture: bright areas = lots of BEV activity
        (e.g. detected objects, roads), dark = empty.

    Called only when matplotlib AND wandb are available.
    """
    if not MPL_AVAILABLE or not WANDB_AVAILABLE:
        return

    model.eval()

    # Fetch a single batch for visualization
    # iter() + next() = first batch without looping the whole DataLoader
    batch = next(iter(loader))
    inputs = batch["input"][:n_samples].to(device).float()   # 1d: GPU-Upcast
    target = batch["target"][:n_samples].to(device).float()
    _ego = batch["ego_delta"][:n_samples].to(device).float() if "ego_delta" in batch else None

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(inputs, _ego)

    # To CPU, float32, numpy for matplotlib
    pred_vis   = pred.float().mean(dim=1).cpu().numpy()    # [n_samples, 128, 128]
    target_vis = target.float().mean(dim=1).cpu().numpy()  # [n_samples, 128, 128]

    # Grid: n_samples rows × 3 columns (predicted | real | difference)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]  # consistent indexing when n_samples=1

    for i in range(n_samples):
        diff = np.abs(pred_vis[i] - target_vis[i])

        im0 = axes[i, 0].imshow(pred_vis[i],   cmap="viridis", interpolation="nearest")
        axes[i, 0].set_title(f"Sample {i}: Predicted t+1", fontsize=9)
        axes[i, 0].axis("off")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(target_vis[i], cmap="viridis", interpolation="nearest")
        axes[i, 1].set_title(f"Sample {i}: Real t+1", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(diff,           cmap="hot",     interpolation="nearest")
        axes[i, 2].set_title(f"Sample {i}: |Pred - Real|", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Epoch {epoch:03d}", fontsize=11)
    plt.tight_layout()

    wandb.log({"predictions": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


# =============================================================================
# BLOCK 7 — Building the DataLoader (from config.yaml)
# =============================================================================

def build_dataloaders(cfg: dict) -> Tuple:
    """
    Reads the data section from config.yaml and builds the DataLoaders.

    Two modes (from bev_dataloader.py):
        "split":    one PKL, split internally into train/val/test
                    → for nuScenes mini / development
        "explicit": separate PKLs per split
                    → for full nuScenes (700/150/150 scenes)

    The YAML configuration looks like this:

        data:
          mode: "split"
          sources:
            - pkl_path:   "/data/nuscenes/infos_val.pkl"
              latent_dir: "/data/nuscenes/latents/val"
          train_ratio: 0.70
          val_ratio:   0.15

    For "explicit" with HDF5 cache (full dataset):

        data:
          mode: "explicit"
          train_sources:
            - pkl_path:   "/data/nuscenes/nuscenes_infos_train.pkl"
              latent_dir: "/data/nuscenes/latents/seg/train"
          val_sources:
            - pkl_path:   "/data/nuscenes/nuscenes_infos_val.pkl"
              latent_dir: "/data/nuscenes/latents/seg/val"
          train_h5_path: "/data/Cache/bev_latents_train.h5"
          val_h5_path:   "/data/Cache/bev_latents_val.h5"

    This function translates this into the appropriate function call.
    """
    # Lazy import: import here so that smoke_test.py can import train.py
    # without bev_dataloader being executed immediately
    from bev_dataloader import make_dataloaders_split, make_dataloaders_explicit

    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    mode       = data_cfg.get("mode", "split")

    # --- Memmap packing -- integrated, switchable part ---
    # data.use_packed controls EVERYTHING via one switch:
    #   true       -> create missing packs at startup (idempotent, skip if
    #                 present) AND read from the memmap at runtime
    #   false/absent -> plain .npy as before; packed_dir is ignored
    # No separate script needed -- packing is part of the train cycle.
    use_packed   = data_cfg.get("use_packed", False)
    packed_dtype = data_cfg.get("packed_dtype", "float32")
    if use_packed:
        # Check the packed_dir requirement -> no silent partially-packed state
        if mode == "explicit":
            missing = [k for k in ("train_packed_dir", "val_packed_dir")
                       if not data_cfg.get(k)]
        else:
            missing = [] if data_cfg.get("packed_dir") else ["packed_dir"]
        if missing:
            raise ValueError(
                f"data.use_packed=true, but {missing} is missing from the config. "
                f"Either set packed_dir or set use_packed to false."
            )
        import pack_latents
        print(f"[build_dataloaders] use_packed=true (dtype={packed_dtype}) -> "
              f"ensuring packs (idempotent, skip if present)...", flush=True)
        pack_latents.ensure_packed(cfg, dtype_str=packed_dtype)

    # --- Node-local /dev/shm staging -- optional switch ---
    # data.stage_to_shm: true -> the PACKED files are copied to /dev/shm once
    # and the *_packed_dir keys of the config then point there (RAM speed
    # instead of the BeeGFS network). Must run AFTER ensure_packed (stages
    # the finished packed.npy). Cleanup at job end via cleanup_shm_staging()
    # (see run_training) or the sbatch trap.
    if data_cfg.get("stage_to_shm", False):
        import shm_staging
        shm_staging.stage_and_rewrite(cfg)   # redirects data_cfg[*_packed_dir] in place

    if mode == "split":
        # sources from YAML: list of {pkl_path, latent_dir} dicts
        # → convert to a list of tuples (as bev_dataloader expects)
        sources = [
            (s["pkl_path"], s["latent_dir"])
            for s in data_cfg["sources"]
        ]

        # HDF5 mode: optional h5_path in config.yaml
        # If set → passed as BEVDatasetConfig.h5_path
        # (latent_dir stays for scene detection, h5_path for data loading)
        h5_path = data_cfg.get("h5_path", None)

        # Memmap pack mode: use only when use_packed=true.
        packed_dir = data_cfg.get("packed_dir", None) if use_packed else None

        loaders = make_dataloaders_split(
            sources       = sources,
            train_ratio   = data_cfg.get("train_ratio", 0.70),
            val_ratio     = data_cfg.get("val_ratio",   0.15),
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
            h5_path       = h5_path,
            packed_dir    = packed_dir,
            prefetch_factor = train_cfg.get("prefetch_factor", 4),
            latent_scale  = data_cfg.get("latent_scale", 1.0),   # det->seg scale
        )

    elif mode == "explicit":
        def _parse(key):
            return [(s["pkl_path"], s["latent_dir"]) for s in data_cfg.get(key, [])]

        # HDF5 mode: optionally a separate h5_path per split (unlike
        # "split" there is no SINGLE common source here but already separate
        # train/val/test PKLs → correspondingly separate caches.
        loaders = make_dataloaders_explicit(
            train_sources = _parse("train_sources"),
            val_sources   = _parse("val_sources"),
            test_sources  = _parse("test_sources") or None,
            batch_size    = train_cfg["batch_size"],
            n_input_frames= model_cfg["n_frames"],
            num_workers   = train_cfg.get("num_workers", 4),
            seed          = train_cfg["seed"],
            train_h5_path = data_cfg.get("train_h5_path", None),
            val_h5_path   = data_cfg.get("val_h5_path",   None),
            test_h5_path  = data_cfg.get("test_h5_path",  None),
            train_packed_dir = data_cfg.get("train_packed_dir", None) if use_packed else None,
            val_packed_dir   = data_cfg.get("val_packed_dir",   None) if use_packed else None,
            latent_scale     = data_cfg.get("latent_scale", 1.0),   # Det->Seg-Skala
            test_packed_dir  = data_cfg.get("test_packed_dir",  None) if use_packed else None,
            prefetch_factor = train_cfg.get("prefetch_factor", 4),
            # Sampler/weighting options (all defaults OFF = bit-identical)
            gt_masks_train_path  = data_cfg.get("gt_masks_train_path", None),
            rare_classes         = tuple(data_cfg.get("rare_classes",
                                         ("stop_line", "ped_crossing", "divider"))),
            rare_weight          = float(data_cfg.get("rare_weight", 1.0)),
            sampler_mode         = data_cfg.get("sampler_mode", "off"),
            sampler_alpha        = float(data_cfg.get("sampler_alpha", 0.5)),
            dynamics_scores_path = data_cfg.get("dynamics_scores_path", None),
        )

    else:
        raise ValueError(f"data.mode must be 'split' or 'explicit', not '{mode}'.")

    return loaders["train"], loaders["val"], loaders.get("test")


# =============================================================================
# BLOCK 8 — Main loop
# =============================================================================

def main() -> None:
    """
    Orchestrates the entire training process.

    Procedure:
        1. Parse arguments, load config
        2. Set seed (reproducibility)
        3. Determine device (GPU/CPU)
        4. Build model, optimizer, scheduler, GradScaler
        5. Build the DataLoader
        6. Initialize wandb
        7. Optional: load checkpoint (--resume)
        8. Training loop:
           for epoch in range(epochs):
               train_one_epoch()
               validate()
               scheduler.step()
               wandb.log()
               visualize() (every log_every epochs)
               save_checkpoint() (when val loss improves)
               save_checkpoint() (periodic snapshot)
               Early stopping check
        9. wandb.finish()
    """

    # -------------------------------------------------------------------------
    # 1. Arguments & config
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="BEV World Model Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--phase",  type=str, default=None,
                        choices=["frame", "cell"],
                        help="Overrides model.phase from config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (.pt) to resume from")
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Checkpoint from which ONLY the model weights "
                             "are loaded (finetune; optimizer/epoch start fresh)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.phase)
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    ckpt_cfg   = cfg.get("checkpoints", {})
    wandb_cfg  = cfg.get("wandb", {})

    # -------------------------------------------------------------------------
    # 2. Reproducibility
    # -------------------------------------------------------------------------
    set_seed(train_cfg["seed"])

    # -------------------------------------------------------------------------
    # 3. Device
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BEV World Model Training")
    print(f"  Phase:  {model_cfg['phase']}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # 4. Model
    # -------------------------------------------------------------------------
    # ModelConfig is the Python dataclass from code/config.py
    # **model_cfg unpacks the dict from the YAML into keyword arguments
    model_config = ModelConfig(**model_cfg)
    model        = BEVWorldModel(model_config).to(device)

    # --- 18/B2: flow mode — FREEZE the backbone, only the head learns -----
    # Must happen BEFORE building the optimizer (it filters on requires_grad).
    flow_active = getattr(model_config, "flow_head", "off") == "flow"
    if flow_active:
        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith("flow."):
                p.requires_grad = False
                n_frozen += 1
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[18/B2] Flow mode: backbone frozen ({n_frozen} tensors), "
              f"trainable only flow.* ({n_train:,} params)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  (phase: {model_config.phase})")
    print(f"Tokens: {model_config.n_tokens} per forward pass\n")

    # -------------------------------------------------------------------------
    # 4b. Optimizer
    # AdamW = Adam + weight decay correction (Loshchilov & Hutter, 2019)
    # Weight decay acts as L2 regularization: pushes weights toward 0
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        # 18/B2: only trainable params (in flow mode the backbone is
        # frozen; otherwise identical to model.parameters()).
        [p for p in model.parameters() if p.requires_grad],
        lr           = train_cfg["lr"],
        weight_decay = train_cfg["weight_decay"],
        # fused=True bundles the per-parameter updates into a single CUDA
        # kernel (stage-0 profile: ~1000 aten::add_ calls/step). Only available
        # on GPU -> coupled to device; compatible with GradScaler.
        fused        = (device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # 4c. Learning Rate Scheduler
    # CosineAnnealingLR: the LR follows a cosine curve from lr down to eta_min
    #
    #   lr(t) = eta_min + 0.5*(lr - eta_min) * (1 + cos(π * t / T_max))
    #
    # t=0: LR = lr (maximum)
    # t=T_max/2: LR ≈ eta_min + (lr-eta_min)/2 (halfway)
    # t=T_max: LR = eta_min (minimum)
    #
    # Advantage over StepLR: no abrupt jump, smooth cooling
    # -------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = train_cfg["epochs"],
        eta_min = 1e-6,
    )

    # -------------------------------------------------------------------------
    # 4d. GradScaler for mixed precision
    # enabled=False on CPU (AMP only makes sense on CUDA)
    # -------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------------------------------------------------------------------------
    # 5. DataLoader
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"  Train: {len(train_loader)} Batches")
    print(f"  Val:   {len(val_loader)} Batches")
    if test_loader:
        print(f"  Test:  {len(test_loader)} Batches")
    print()

    # -------------------------------------------------------------------------
    # 5b. Seg-Decoder + Val-Subset-Cache
    # -------------------------------------------------------------------------
    # decode_cfg controls the entire decode validation from the YAML.
    # If the block is missing or enabled: false, training runs as before
    # (val loss only, no mIoU). Backward compatible.
    decode_cfg     = cfg.get("decode_validation", {})
    decode_enabled = decode_cfg.get("enabled", False) and SEG_DECODER_AVAILABLE
    seg_decoder    = None
    seg_subset_inputs  = None
    seg_subset_targets = None
    seg_real_masks     = []
    decode_every = decode_cfg.get("decode_every", 5)
    n_subset     = decode_cfg.get("n_subset", 300)

    if decode_enabled:
        seg_ckpt = decode_cfg.get("seg_checkpoint")
        if seg_ckpt and os.path.exists(seg_ckpt):
            print(f"[seg_decoder] Loading seg decoder: {seg_ckpt}")
            try:
                seg_decoder = build_seg_decoder(seg_ckpt, device, verbose=True)
                seg_subset_inputs, seg_subset_targets, seg_real_masks = \
                    build_seg_val_cache(seg_decoder, val_loader, device,
                                        n_subset=n_subset,
                                        seed=train_cfg.get("seed", 42))
                print(f"[seg_decoder] Decode validation active: "
                      f"every {decode_every} epochs, {len(seg_real_masks)} samples\n")
            except Exception as e:
                print(f"[seg_decoder] Loading the decoder failed: {e}")
                print("          Falling back to val-loss only.\n")
                decode_enabled = False
                seg_decoder    = None
        else:
            print(f"[seg_decoder] seg_checkpoint not found: {seg_ckpt}")
            print("          Falling back to val-loss only.\n")
            decode_enabled = False

    # -------------------------------------------------------------------------
    # 6. Initialize wandb
    # -------------------------------------------------------------------------
    run_name = f"phase_{model_config.phase}_{time.strftime('%m%d_%H%M')}"

    if WANDB_AVAILABLE:
        wandb.init(
            project = wandb_cfg.get("project", "bev_worldmodel"),
            entity  = wandb_cfg.get("entity"),   # None = personal account
            name    = run_name,
            config  = cfg,    # entire config as wandb hyperparameters
        )
        wandb.watch(model, log="gradients", log_freq=100)
    print(f"Run: {run_name}\n")

    # -------------------------------------------------------------------------
    # 7. Optional: load checkpoint (--resume)
    # -------------------------------------------------------------------------
    start_epoch   = 0
    best_val_loss = float("inf")
    patience_cnt  = 0
    resume_ckpt   = None    # keeps the checkpoint for the
                             # mIoU-restore block further down (the target
                             # variables only exist AFTER this block).

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = load_checkpoint(
            checkpoint_path = args.resume,
            model           = model,
            optimizer       = optimizer,
            scheduler       = scheduler,
            device          = str(device),
        )
        start_epoch   = ckpt["epoch"] + 1      # next epoch
        best_val_loss = ckpt["best_val_loss"]
        resume_ckpt   = ckpt                   # mIoU-Plateau-Resume
        print(f"  → Starte ab Epoch {start_epoch}, "
              f"bester Val-Loss bisher: {best_val_loss:.6f}\n")
    elif args.init_weights:
        # init_weights: load ONLY weights (finetune) -- optimizer/scheduler/
        # epoch stay fresh (no resume). Counters/best values start at 0/-1 so
        # the finetune run writes its own best_*.pt.
        print(f"Init weights (finetune, no resume): {args.init_weights}")
        load_checkpoint(
            checkpoint_path = args.init_weights,
            model           = model,
            device          = str(device),
            # 18/B2: new extra modules (flow.*) are missing from the old
            # checkpoint and keep their fresh initialization.
            allow_missing   = flow_active,
        )

    # -------------------------------------------------------------------------
    # 8. TRAINING LOOP
    # -------------------------------------------------------------------------
    checkpoint_dir = ckpt_cfg.get("dir", "checkpoints")
    save_every     = ckpt_cfg.get("save_every", 10)
    log_every      = train_cfg.get("log_every", 5)
    use_cosine     = train_cfg.get("use_cosine", True)   # legacy key, only used for print/logging
    lambda_mse     = train_cfg.get("lambda_mse",  1.0)
    lambda_cos     = train_cfg.get("lambda_cos",  0.1)
    lambda_mean    = train_cfg.get("lambda_mean", 0.1)
    lambda_std     = train_cfg.get("lambda_std",  0.1)
    lambda_grad    = train_cfg.get("lambda_grad", 0.0)
    lambda_ssim    = train_cfg.get("lambda_ssim", 0.1)
    recon_loss     = train_cfg.get("recon_loss", "mse")   # TERM 1 distance
    # Extra terms into the module-level block (default 0 = bit-identical)
    EXTRA_LOSS["lambda_sliced"] = float(train_cfg.get("lambda_sliced", 0.0))
    EXTRA_LOSS["sliced_M"]      = int(train_cfg.get("sliced_M", 64))
    EXTRA_LOSS["energy_alpha"]  = float(train_cfg.get("energy_alpha", 0.0))
    EXTRA_LOSS["lambda_peak"]   = float(train_cfg.get("lambda_peak", 0.0))
    if any(v > 0 for k, v in EXTRA_LOSS.items() if k != "sliced_M"):
        print(f"[extra-loss] Extra loss active: {EXTRA_LOSS}")
    lambda_task    = float(train_cfg.get("lambda_task", 0.0))   # optional decoder task loss
    task_every     = int(train_cfg.get("task_loss_every", 1))   # every n-th batch
    if lambda_task > 0.0 and seg_decoder is None:
        raise SystemExit("[task_loss] lambda_task > 0 requires decode validation "
                         "(decode_validation.enabled + loadable seg_checkpoint) -- "
                         "no task loss without a decoder.")
    if lambda_task > 0.0:
        print(f"[task_loss] Decoder task loss active: lambda_task={lambda_task}, "
              f"task_loss_every={task_every}")

    # --- 18/B1: CVAE configuration + consistency guards ----------------------
    lambda_kl       = float(train_cfg.get("lambda_kl", 0.0))
    kl_anneal_ep    = int(train_cfg.get("kl_anneal_epochs", 5))
    kl_free_bits    = float(train_cfg.get("kl_free_bits", 0.05))
    vae_active      = getattr(model_config, "vae_mode", "off") == "cvae"
    if vae_active and lambda_kl <= 0.0:
        raise SystemExit("[18/B1] vae_mode=cvae requires lambda_kl > 0 "
                         "(otherwise unregularized posterior).")
    if not vae_active and lambda_kl > 0.0:
        raise SystemExit("[18/B1] lambda_kl > 0 requires model.vae_mode=cvae.")
    if vae_active:
        print(f"[18/B1] CVAE active: z_dim={getattr(model_config, 'vae_z_dim', 32)}, "
              f"lambda_kl={lambda_kl}, anneal={kl_anneal_ep} epochs, "
              f"free_bits={kl_free_bits}/dim")

    # --- 18/B2: flow guards ----------------------------------------------------
    flow_val_steps = int(train_cfg.get("flow_val_steps", 10))
    if flow_active and vae_active:
        raise SystemExit("[18/B2] flow_head=flow and vae_mode=cvae together "
                         "is not intended (one generative mode per run).")
    if flow_active and not (args.init_weights or args.resume):
        raise SystemExit("[18/B2] flow_head=flow requires --init_weights "
                         "(frozen backbone from the headline checkpoint) or --resume.")
    n_epochs       = train_cfg["epochs"]
    patience       = train_cfg["patience"]

    # mIoU tracking in parallel to the val loss
    best_miou     = -1.0
    miou_patience_cnt = 0
    best_epoch_miou = -1

    # --- EMA: weight averaging. Default OFF -> ema_state stays None and the
    # training/validation paths are bit-identical to the old behavior. Not
    # resume-safe (shadow weights are not persisted in the checkpoint) --
    # acceptable because the headline runs are single-shot. -------------------
    ema_enabled   = bool(train_cfg.get("ema", False))
    ema_decay     = float(train_cfg.get("ema_decay", 0.9999))
    ema_state     = None
    best_miou_ema = -1.0
    if ema_enabled:
        ema_state = {name: p.detach().clone()
                     for name, p in model.named_parameters()}
        print(f"[EMA] active: decay={ema_decay}, {len(ema_state)} param tensors")
    stop_reason     = "max_epochs"     # overwritten on break
    # Early stopping continues to run on val loss (unchanged).
    # best_miou.pt is saved additionally (checkpoint for mIoU selection).
    # Both checkpoints run in parallel.

    # --- mIoU plateau early stopping ---. Idea: the last ~25 epochs up to
    # the mIoU optimum often bring only marginal gains -> cut off as soon as
    # no SIGNIFICANT mIoU progress happens any more. "Significant" = larger
    # than the seed noise floor (miou_min_delta) so noise dips do not kill it
    # spuriously. Compared against a SEPARATE reference (miou_stop_ref), NOT
    # against best_miou -- otherwise a tiny 0.001 improvement would reset the
    # counter and it would never stop. Only active when decode is enabled
    # (otherwise there is no mIoU -> clean fallback to pure val-loss stopping).
    miou_early_stop = decode_enabled and train_cfg.get("miou_early_stop", False)
    miou_patience   = train_cfg.get("miou_patience", 3)      # "three times in a row"
                                                             # (2 proved too strict in testing
                                                             # with a tight min_delta)
    miou_min_delta  = train_cfg.get("miou_min_delta", 0.014) # noise floor (seed variability)
    miou_stop_ref   = -1.0   # last mIoU value that was significantly better

    # --- Restore the mIoU plateau state from the resume checkpoint ---
    # Without this, best_miou would stay at -1.0 after a resume -> the next
    # decode epoch would ALWAYS count as a "new best" and best_miou.pt might
    # be overwritten with a WORSE value. Old checkpoints (saved before this
    # patch) deliver exactly the defaults above via .get() -> resuming from
    # an old checkpoint behaves unchanged as before.
    if resume_ckpt is not None:
        best_miou         = resume_ckpt.get("best_miou", best_miou)
        best_epoch_miou   = resume_ckpt.get("best_epoch_miou", best_epoch_miou)
        miou_patience_cnt = resume_ckpt.get("miou_patience_cnt", miou_patience_cnt)
        miou_stop_ref      = resume_ckpt.get("miou_stop_ref", miou_stop_ref)
        patience_cnt        = resume_ckpt.get("patience_cnt", patience_cnt)
        print(f"  → mIoU state restored: best_miou={best_miou:.4f} "
              f"(epoch {best_epoch_miou}), plateau={miou_patience_cnt}/{miou_patience}\n")

    print(f"Starting training: {n_epochs} epochs, patience={patience}")
    print(f"  Loss: {lambda_mse}*{recon_loss.upper()} + {lambda_cos}*CosSim + {lambda_mean}*Mean "
          f"+ {lambda_std}*Std + {lambda_grad}*Grad + {lambda_ssim}*SSIM")
    if decode_enabled:
        print(f"  Decode validation: every {decode_every} epochs, "
              f"{len(seg_real_masks)} val-subset samples")
    if miou_early_stop:
        print(f"  mIoU plateau stopping: active (patience={miou_patience} measurements, "
              f"min_delta={miou_min_delta}, in addition to val-loss patience={patience})")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'mIoU':>8}  {'LR':>10}  {'Time':>8}  {'data%':>7}")
    print("-" * 78)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # 18/B1: beta annealing — linear 0 -> lambda_kl over kl_anneal_epochs
        vae_beta = lambda_kl * min(1.0, (epoch + 1) / max(1, kl_anneal_ep)) \
                   if vae_active else 0.0

        # --- Training epoch ---
        train_loss, train_mse, train_cos, train_mean, train_std, train_grad, train_ssim, train_timing = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
            recon_loss=recon_loss,
            ema_state=ema_state, ema_decay=ema_decay,
            seg_decoder=seg_decoder if lambda_task > 0.0 else None,
            lambda_task=lambda_task, task_every=task_every,
            vae_active=vae_active, vae_beta=vae_beta, kl_free_bits=kl_free_bits,
            flow_active=flow_active,
        )

        # --- Validation (latent space) ---
        val_loss, val_mse, val_cos, val_mean, val_std, val_grad, val_ssim = validate(
            model, val_loader, device,
            lambda_mse, lambda_cos, lambda_mean, lambda_std, lambda_grad, lambda_ssim,
            recon_loss=recon_loss,
            flow_active=flow_active,
        )

        # --- Decode validation (every decode_every epochs) ---
        # Always decode the first epoch for an early reference point.
        miou_this_epoch  = None
        per_class_this   = {}
        do_decode = (decode_enabled and seg_decoder is not None and
                     (epoch == start_epoch or (epoch + 1) % decode_every == 0))
        if do_decode:
            miou_this_epoch, per_class_this = validate_seg(
                model, seg_decoder,
                seg_subset_inputs, seg_subset_targets, seg_real_masks,
                device,
                flow_steps=flow_val_steps if flow_active else 0,
            )

        # --- Scheduler ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # --- Console output ---
        miou_str = f"{miou_this_epoch:>8.4f}" if miou_this_epoch is not None else "        "
        _kl_str = (f"  KL={train_timing['kl_loss']:.3f} (b={vae_beta:.4f})"
                   if train_timing.get("kl_loss") is not None else "")
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
              f"{miou_str}  {current_lr:>10.2e}  {elapsed:>6.1f}s  "
              f"data%={train_timing['data_time_ratio']*100:5.1f}{_kl_str}", flush=True)

        # --- wandb logging ---
        log_dict = {
            "epoch":        epoch,
            "train/loss":   train_loss,
            "train/mse":    train_mse,
            "train/cos":    train_cos,
            "train/mean":   train_mean,
            "train/std":    train_std,
            "train/grad":   train_grad,
            "train/ssim":   train_ssim,
            "val/loss":     val_loss,
            "val/mse":      val_mse,
            "val/cos":      val_cos,
            "val/mean":     val_mean,
            "val/std":      val_std,
            "val/grad":     val_grad,
            "val/ssim":     val_ssim,
            "lr":           current_lr,
            # Timing instrumentation -- makes the
            # I/O-vs-compute decision gate measurable.
            "train/data_time_s":    train_timing["data_time"],
            "train/compute_time_s": train_timing["compute_time"],
            "train/data_time_ratio": train_timing["data_time_ratio"],
        }
        if train_timing.get("task_loss") is not None:              # Task-Loss
            log_dict["train/task"] = train_timing["task_loss"]
        if train_timing.get("kl_loss") is not None:                # 18/B1
            log_dict["train/kl"]      = train_timing["kl_loss"]
            log_dict["train/kl_beta"] = vae_beta
        if miou_this_epoch is not None:
            log_dict["val/miou"] = miou_this_epoch
            for cls, v in per_class_this.items():
                if v is not None:
                    log_dict[f"val/miou_{cls}"] = v
        if WANDB_AVAILABLE:
            wandb.log(log_dict)

        # --- Visualizations every log_every epochs ---
        if epoch % log_every == 0:
            visualize_predictions(model, val_loader, device, epoch)

        # --- Checkpoint: best model by val loss (as before) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename="best_val_loss.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU plateau state
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch_loss"] = epoch
        else:
            patience_cnt += 1

        # --- Checkpoint: best model by mIoU ---
        # Saved only in epochs in which decode_seg ran.
        # In parallel to best_val_loss.pt — neither replaces the other.
        if miou_this_epoch is not None and miou_this_epoch > best_miou:
            best_miou = miou_this_epoch
            best_epoch_miou = epoch
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename="best_miou.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU-Plateau-Zustand
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )
            if WANDB_AVAILABLE:
                wandb.run.summary["best_miou"]       = best_miou
                wandb.run.summary["best_epoch_miou"] = epoch
            print(f"         -> new best_miou.pt: {best_miou:.4f} (epoch {epoch})", flush=True)

        # --- EMA decode validation + best_miou_ema.pt ------------
        # Only in decode epochs and only when EMA is active. The live weights
        # are briefly swapped for the shadow weights, validated, and swapped
        # back -> the ongoing training remains completely untouched.
        # The mIoU plateau early stop continues to run on the RAW mIoU
        # (below); EMA does not interfere with the stop logic.
        if ema_enabled and miou_this_epoch is not None:
            _backup = {name: p.detach().clone()
                       for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                p.data.copy_(ema_state[name])
            miou_ema, _per_class_ema = validate_seg(
                model, seg_decoder,
                seg_subset_inputs, seg_subset_targets, seg_real_masks, device,
            )
            for name, p in model.named_parameters():
                p.data.copy_(_backup[name])
            print(f"         [EMA] mIoU={miou_ema:.4f} (raw={miou_this_epoch:.4f})", flush=True)
            if WANDB_AVAILABLE:
                wandb.log({"val/miou_ema": miou_ema, "epoch": epoch})
            if miou_ema > best_miou_ema:
                best_miou_ema = miou_ema
                for name, p in model.named_parameters():
                    p.data.copy_(ema_state[name])
                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                    config=model_config, checkpoint_dir=checkpoint_dir,
                    filename="best_miou_ema.pt",
                    best_miou=best_miou_ema, best_epoch_miou=epoch,
                    miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                    patience_cnt=patience_cnt,
                )
                for name, p in model.named_parameters():
                    p.data.copy_(_backup[name])
                if WANDB_AVAILABLE:
                    wandb.run.summary["best_miou_ema"] = best_miou_ema
                print(f"         -> new best_miou_ema.pt: {best_miou_ema:.4f} (epoch {epoch})", flush=True)

        # --- Update the mIoU plateau counter (only in decode epochs) ---
        # Significant progress (> noise floor above the reference) resets the
        # counter and raises the reference. Otherwise count up. The
        # best_miou.pt above is unaffected -> the saved checkpoint is ALWAYS
        # the best one, never the one from the stopping moment.
        if miou_early_stop and miou_this_epoch is not None:
            if miou_this_epoch > miou_stop_ref + miou_min_delta:
                miou_stop_ref = miou_this_epoch
                miou_patience_cnt = 0
            else:
                miou_patience_cnt += 1
                print(f"         mIoU plateau: {miou_patience_cnt}/{miou_patience} "
                      f"measurements without significant progress "
                      f"(>{miou_min_delta:.3f} above {miou_stop_ref:.4f})", flush=True)

        # --- Checkpoint: periodic snapshot ---
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, val_loss=val_loss, best_val_loss=best_val_loss,
                config=model_config, checkpoint_dir=checkpoint_dir,
                filename=f"epoch_{epoch+1:03d}.pt",
                best_miou=best_miou, best_epoch_miou=best_epoch_miou,           # mIoU-Plateau-Zustand
                miou_patience_cnt=miou_patience_cnt, miou_stop_ref=miou_stop_ref,
                patience_cnt=patience_cnt,
            )

        # --- Early stopping (val loss OR mIoU plateau) ---
        # Val-loss patience as before (continuity). ADDITIONALLY the mIoU
        # plateau, if active -- whichever triggers first stops. The best
        # checkpoint (best_miou.pt / best_val_loss.pt) is already saved at
        # this point, so we only throw away marginal final epochs.
        if patience_cnt >= patience:
            stop_reason = "val_loss_patience"
            print(f"\nEarly stopping: {patience_cnt} epochs without val-loss improvement.")
            break
        if miou_early_stop and miou_patience_cnt >= miou_patience:
            stop_reason = "miou_plateau"
            print(f"\nEarly stopping: {miou_patience_cnt} mIoU measurements without "
                  f"significant progress (>{miou_min_delta:.3f}). "
                  f"Best mIoU {best_miou:.4f} already saved as best_miou.pt.")
            break

    # -------------------------------------------------------------------------
    # 9. Wrap-up
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training finished.")
    print(f"  Best val loss: {best_val_loss:.6f}")
    if decode_enabled and best_miou >= 0:
        print(f"  Best mIoU:      {best_miou:.4f}  -> best_miou.pt")
    best_ckpt = find_best_checkpoint(checkpoint_dir, model_config.phase)
    if best_ckpt:
        print(f"  Best checkpoint (loss): {best_ckpt}")
    print(f"{'='*60}\n")

    # --- machine-readable run summary for sweep_runner.py ---
    import json as _json
    _phase_num = 2 if model_config.phase == "cell" else 1
    _summary_path = Path(checkpoint_dir) / f"phase{_phase_num}" / "run_summary.json"
    _summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_summary_path, "w") as _f:
        _json.dump({
            "best_miou":       best_miou,
            "best_epoch_miou": best_epoch_miou,
            "best_val_loss":   best_val_loss,
            "stop_reason":     stop_reason,
            "epochs_run":      epoch + 1,
            "lambdas": {"mse": lambda_mse, "cos": lambda_cos, "mean": lambda_mean,
                        "std": lambda_std, "grad": lambda_grad, "ssim": lambda_ssim},
            "recon_loss": recon_loss,   # TERM 1 distance (mse|l1|smooth_l1)
            "seed": train_cfg.get("seed", 42),
        }, _f, indent=2)
    print(f"[summary] run_summary.json -> {_summary_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

    # --- Clean up /dev/shm staging ---
    # At NORMAL job end. On a hard kill (scancel/timeout/OOM) this line no
    # longer runs -> the trap in the sbatch script is the safeguard there
    # (deliberately redundant, because leftover /dev/shm garbage blocks the
    # shared node for other users).
    if cfg["data"].get("stage_to_shm", False):
        import shm_staging
        shm_staging.cleanup(cfg)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
