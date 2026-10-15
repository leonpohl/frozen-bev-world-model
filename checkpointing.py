# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
checkpointing.py — BEV World Model: checkpoint management
==========================================================

Saves and loads model weights separated by phase.

WHY AN OWN MODULE?
    Saving weights sounds trivial — but it is error-prone.
    Common mistakes:
        - Saving only model.state_dict(), forgetting the optimizer
          → training cannot be resumed (momentum lost)
        - Not saving the config
          → when loading you no longer know which architecture is behind it
        - Loading a Phase 1 checkpoint into a Phase 2 model
          → shape mismatch error, hard to debug

    This module does it right — a checkpoint contains EVERYTHING
    needed to resume training or reproduce results.

FILE STRUCTURE:
    checkpoints/
    ├── phase1/
    │   ├── best_val_loss.pt      ← best model (by val loss)
    │   ├── epoch_010.pt          ← periodic snapshots
    │   └── epoch_020.pt
    └── phase2/
        ├── best_val_loss.pt
        └── epoch_010.pt

WHAT GOES INTO A CHECKPOINT:
    {
        "model_state_dict":     OrderedDict of all parameters + buffers
        "optimizer_state_dict": AdamW momentum/variance for all parameters
        "scheduler_state_dict": CosineAnnealingLR current step
        "epoch":                int — which epoch was just completed
        "val_loss":             float — val loss of this epoch
        "best_val_loss":        float — best val loss so far
        "config":               dict — all ModelConfig parameters
        "phase":                str — "frame" or "cell"
        "timestamp":            str — when saved (for logs)
    }
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from config import ModelConfig


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    config: ModelConfig,
    checkpoint_dir: str,
    filename: str,
    best_miou: float = -1.0,            # mIoU-plateau resume
    best_epoch_miou: int = -1,
    miou_patience_cnt: int = 0,
    miou_stop_ref: float = -1.0,
    patience_cnt: int = 0,
) -> Path:
    """
    Saves a complete training checkpoint.

    A checkpoint is more than just the weights — it is a complete
    snapshot of the training state. With it one can:
        1. Resume training exactly at this point (epoch, optimizer)
        2. Load the model for inference (model_state_dict, config)
        3. Trace when and how good the model was (val_loss, timestamp)

    Args:
        model:          the BEVWorldModel (or any nn.Module)
        optimizer:      AdamW instance — contains momentum + variance
        scheduler:      CosineAnnealingLR — contains the current LR step
        epoch:          current epoch (0-based)
        val_loss:       val loss of this epoch
        best_val_loss:  best val loss over all epochs so far
        config:         ModelConfig — stored as a dict
        checkpoint_dir: base directory (e.g. "checkpoints")
        filename:       file name (e.g. "best_val_loss.pt" or "epoch_010.pt")

    Returns:
        Path to the saved file
    """
    # Create directory: checkpoints/phase1/ or checkpoints/phase2/
    save_dir = Path(checkpoint_dir) / f"phase{1 if config.phase == 'frame' else 2}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    checkpoint = {
        # --- Model ---
        # state_dict() returns an OrderedDict: {"layer.weight": Tensor, ...}
        # It contains parameters (trainable) AND buffers (non-trainable,
        # e.g. BatchNorm running_mean). Both are required for correct loading.
        "model_state_dict": model.state_dict(),

        # --- Optimizer ---
        # AdamW stores for each parameter:
        #   - exp_avg:   weighted mean of the gradients (momentum)
        #   - exp_avg_sq: weighted mean of the squared gradients
        #   - step:      how often this parameter was already updated
        # Without it: training can be resumed, but the first
        # epochs after loading are "cold" — as if starting from scratch.
        "optimizer_state_dict": optimizer.state_dict(),

        # --- Scheduler ---
        # CosineAnnealingLR knows where in the cosine cycle it currently is.
        # Without it: the LR jumps back to the beginning when loading.
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,

        # --- Training progress ---
        "epoch":          epoch,
        "val_loss":       val_loss,
        "best_val_loss":  best_val_loss,

        # --- mIoU-plateau state ---
        # Missing in checkpoints from BEFORE this patch -> caller uses
        # .get() defaults when loading (= old fresh-init behavior). No break
        # of existing checkpoints (e.g. best_miou.pt).
        "best_miou":         best_miou,
        "best_epoch_miou":   best_epoch_miou,
        "miou_patience_cnt": miou_patience_cnt,
        "miou_stop_ref":     miou_stop_ref,
        "patience_cnt":      patience_cnt,

        # --- Architecture info ---
        # Important: store config as a plain dict, not as a dataclass object.
        # dataclass objects could cause problems when loading if the class
        # has changed. A dict is always readable.
        "config":  asdict(config),
        "phase":   config.phase,

        # --- Reproducibility ---
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved: {save_path}  "
          f"(epoch={epoch}, val_loss={val_loss:.6f})")
    return save_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
    allow_missing: bool = False,   # 18/B2: strict=False loading for models with
                                   # NEW additional modules (e.g. flow.*) that may
                                   # be missing in the old checkpoint. Unexpected
                                   # keys remain an error nonetheless.
) -> dict:
    """
    Loads a checkpoint and restores the training state.

    SAFETY CHECK:
        Before loading weights, the function checks whether the phase
        of the checkpoint matches the phase of the model. Loading a
        Phase 1 checkpoint into a Phase 2 model would produce a cryptic
        shape mismatch error — this check gives a clear error message.

    Args:
        checkpoint_path: path to the .pt file
        model:           instantiated BEVWorldModel (must match the phase!)
        optimizer:       optional — if None, only the model is loaded
                         (useful for inference or evaluation)
        scheduler:       optional — analogous to optimizer
        device:          "cpu" or "cuda" — where the tensors are loaded to

    Returns:
        The complete checkpoint dict (for logging, epoch continuation, etc.)

    Raises:
        ValueError:  when the checkpoint phase does not match the model phase
        FileNotFoundError: when checkpoint_path does not exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # map_location: load the checkpoint onto the desired device.
    # Important when the checkpoint was saved on GPU but loaded on CPU
    # (e.g. for evaluation on a machine without GPU).
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Phase check ---
    # This is the most important safety check.
    # If we do not check here, a shape mismatch error occurs later
    # that is hard to debug ("expected shape [32, 128] but got [3, 256]")
    saved_phase = checkpoint.get("phase", "unknown")
    model_config = getattr(model, "config", None)

    if model_config is not None and saved_phase != model_config.phase:
        raise ValueError(
            f"Phase mismatch!\n"
            f"  Checkpoint: phase='{saved_phase}'\n"
            f"  Model:      phase='{model_config.phase}'\n"
            f"  A Phase 1 checkpoint cannot be loaded into a Phase 2 model.\n"
            f"  Make sure you use the right checkpoint directory."
        )

    # --- Load model weights ---
    # strict=True (default): ALL parameters must match.
    # If a parameter is missing or an extra parameter is present → error.
    # That is the desired behavior — silent mismatches are dangerous.
    # EXCEPTION (18/B2, allow_missing=True): models with new additional
    # modules (flow.*) load old checkpoints with strict=False; missing keys keep
    # their fresh initialization. Unexpected keys are STILL an error.
    if allow_missing:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(
                f"[Checkpoint] Unexpected keys during allow_missing loading: "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if missing:
            print(f"[Checkpoint] allow_missing: {len(missing)} keys freshly "
                  f"initialized (e.g. {missing[0]})")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # --- Restore optimizer + scheduler (only for training continuation) ---
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch     = checkpoint.get("epoch", 0)
    val_loss  = checkpoint.get("val_loss", float("inf"))
    timestamp = checkpoint.get("timestamp", "unknown")

    print(f"[Checkpoint] Loaded: {checkpoint_path}")
    print(f"  Phase:     {saved_phase}")
    print(f"  Epoch:     {epoch}")
    print(f"  Val loss:  {val_loss:.6f}")
    print(f"  Saved:     {timestamp}")

    return checkpoint


# ---------------------------------------------------------------------------
# Helper: find the best checkpoint
# ---------------------------------------------------------------------------

def find_best_checkpoint(checkpoint_dir: str, phase: str) -> Optional[Path]:
    """
    Returns the path to the best val-loss checkpoint, if present.

    Useful when starting evaluation or ablation runs:
    You do not need to know the exact file name.

    Args:
        checkpoint_dir: base directory (e.g. "checkpoints")
        phase:          "frame" or "cell"

    Returns:
        Path to best_val_loss.pt, or None if not present
    """
    phase_num  = 1 if phase == "frame" else 2
    best_path  = Path(checkpoint_dir) / f"phase{phase_num}" / "best_val_loss.pt"
    return best_path if best_path.exists() else None
