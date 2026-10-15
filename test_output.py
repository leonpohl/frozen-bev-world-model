# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
test_output.py — BEV World Model: Sanity Check for Output Heads
======================================================

Tests FrameLevelOutputHead, CellLevelOutputHead and UpsamplingHead
for correct shapes, gradient flow and NaN-freedom.

USAGE:
    # From the project folder (where config.py lives):
    python test_output.py

EXPECTED OUTPUT:
    All 6 tests green (✓), ending with "All 6 tests passed".

PREREQUISITES:
    The following files must be in the same folder:
        config.py
        output_head.py
        upsampling_head.py
"""

import torch
from config          import ModelConfig
from output_head     import FrameLevelOutputHead, CellLevelOutputHead
from upsampling_head import UpsamplingHead

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg: str):
    print(f"  ✓ {msg}")

def check_shape(tensor, expected, name="Output"):
    assert tensor.shape == torch.Size(expected), \
        f"{name}: wrong shape! Expected {expected}, got {list(tensor.shape)}"
    ok(f"Shape correct: {list(tensor.shape)}")

def check_no_nan(tensor, name="Output"):
    assert not torch.isnan(tensor).any(), f"{name}: NaN in tensor!"
    assert not torch.isinf(tensor).any(), f"{name}: Inf in tensor!"
    ok("No NaN / Inf")

def check_gradients(module, name):
    n = sum(1 for _ in module.parameters())
    for pname, p in module.named_parameters():
        assert p.grad is not None, \
            f"No gradient for {name}.{pname} — gradient flow broken!"
    ok(f"Gradient flow through all {n} parameter tensors")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

B = 4   # Batch size for all tests

cfg_frame = ModelConfig(phase="frame")   # Phase 1: 3 Tokens
cfg_cell  = ModelConfig(phase="cell")    # Phase 2: 3072 Tokens

# ===========================================================================
# TEST 1 — FrameLevelOutputHead: Shape
# ===========================================================================
section("TEST 1 — FrameLevelOutputHead: Shape")

head1 = FrameLevelOutputHead(cfg_frame)

# Simulated Transformer output Phase 1: [B, n_frames, d_model]
x1 = torch.randn(B, cfg_frame.n_frames, cfg_frame.d_model)
print(f"  Input:  {list(x1.shape)}  [B, n_frames={cfg_frame.n_frames}, d_model={cfg_frame.d_model}]")

out1 = head1(x1)
print(f"  Output: {list(out1.shape)}  [B, d_model, grid_size, grid_size]")

check_shape(out1, [B, cfg_frame.d_model, cfg_frame.grid_size, cfg_frame.grid_size])
check_no_nan(out1)

# ===========================================================================
# TEST 2 — FrameLevelOutputHead: Gradient-Flow
# ===========================================================================
section("TEST 2 — FrameLevelOutputHead: Gradient-Flow")

x1 = torch.randn(B, cfg_frame.n_frames, cfg_frame.d_model)
out1 = head1(x1)
out1.mean().backward()
check_gradients(head1, "FrameLevelOutputHead")

n_params = sum(p.numel() for p in head1.parameters())
ok(f"Total parameters: {n_params:,}  (Linear {cfg_frame.d_model}×{cfg_frame.d_model} + bias)")

# ===========================================================================
# TEST 3 — CellLevelOutputHead: Shape
# ===========================================================================
section("TEST 3 — CellLevelOutputHead: Shape")

head2 = CellLevelOutputHead(cfg_cell)

# Simulated Transformer output Phase 2: [B, n_frames*grid², d_model]
n_tokens = cfg_cell.n_tokens   # 3 × 32 × 32 = 3072
x2 = torch.randn(B, n_tokens, cfg_cell.d_model)
print(f"  Input:  {list(x2.shape)}  [B, n_tokens={n_tokens}, d_model={cfg_cell.d_model}]")
print(f"          (= {cfg_cell.n_frames} frames × {cfg_cell.grid_size}² cells)")

out2 = head2(x2)
print(f"  Output: {list(out2.shape)}  [B, d_model, grid_size, grid_size]")

check_shape(out2, [B, cfg_cell.d_model, cfg_cell.grid_size, cfg_cell.grid_size])
check_no_nan(out2)

# ===========================================================================
# TEST 4 — CellLevelOutputHead: Gradient-Flow + Ablation n_frames=2
# ===========================================================================
section("TEST 4 — CellLevelOutputHead: Gradient-Flow + Ablation n_frames=2")

x2 = torch.randn(B, n_tokens, cfg_cell.d_model)
out2 = head2(x2)
out2.mean().backward()
check_gradients(head2, "CellLevelOutputHead")

# Ablation: n_frames=2 → only 2048 tokens
# The CellLevelOutputHead always takes the last 1024 tokens (frame t)
# → must also work with 2048 tokens
cfg_abl  = ModelConfig(phase="cell", n_frames=2)
head_abl = CellLevelOutputHead(cfg_abl)
x_abl    = torch.randn(B, cfg_abl.n_tokens, cfg_abl.d_model)   # [B, 2048, 256]
out_abl  = head_abl(x_abl)
check_shape(out_abl, [B, cfg_abl.d_model, cfg_abl.grid_size, cfg_abl.grid_size],
            name="Ablation n_frames=2")
ok(f"Ablation n_frames=2: {list(x_abl.shape)} → {list(out_abl.shape)}")

# ===========================================================================
# TEST 5 — UpsamplingHead: Shape (both phases)
# ===========================================================================
section("TEST 5 — UpsamplingHead: Shape (Phase 1 and Phase 2)")

# UpsamplingHead is phase-agnostic — same code, same parameters
upsample = UpsamplingHead(cfg_frame)

# Phase 1
x_up1 = torch.randn(B, cfg_frame.d_model, cfg_frame.grid_size, cfg_frame.grid_size)
print(f"  Phase 1 Input:  {list(x_up1.shape)}")
out_up1 = upsample(x_up1)
print(f"  Phase 1 Output: {list(out_up1.shape)}")
check_shape(out_up1, [B, cfg_frame.d_model, 128, 128], name="Phase 1 Upsample")
check_no_nan(out_up1)

# Phase 2 — same module, same shapes in and out
x_up2 = torch.randn(B, cfg_cell.d_model, cfg_cell.grid_size, cfg_cell.grid_size)
print(f"\n  Phase 2 Input:  {list(x_up2.shape)}")
out_up2 = upsample(x_up2)
print(f"  Phase 2 Output: {list(out_up2.shape)}")
check_shape(out_up2, [B, cfg_cell.d_model, 128, 128], name="Phase 2 Upsample")
check_no_nan(out_up2)

n_params_up = sum(p.numel() for p in upsample.parameters())
ok(f"Total parameters: {n_params_up:,}")

# ===========================================================================
# TEST 6 — End-to-End: Output Head → Upsampling Head (both phases)
# ===========================================================================
section("TEST 6 — End-to-End: Output Head → Upsampling Head")

# Phase 1: complete output path
print("  Phase 1:")
head1_e2e   = FrameLevelOutputHead(cfg_frame)
upsample1   = UpsamplingHead(cfg_frame)

x = torch.randn(B, cfg_frame.n_frames, cfg_frame.d_model)
out = upsample1(head1_e2e(x))
check_shape(out, [B, cfg_frame.d_model, 128, 128], name="Phase 1 End-to-End")

out.mean().backward()
check_gradients(head1_e2e, "FrameLevelOutputHead")
check_gradients(upsample1, "UpsamplingHead")

# Phase 2: complete output path
print("\n  Phase 2:")
head2_e2e   = CellLevelOutputHead(cfg_cell)
upsample2   = UpsamplingHead(cfg_cell)

x = torch.randn(B, cfg_cell.n_tokens, cfg_cell.d_model)
out = upsample2(head2_e2e(x))
check_shape(out, [B, cfg_cell.d_model, 128, 128], name="Phase 2 End-to-End")

out.mean().backward()
check_gradients(head2_e2e, "CellLevelOutputHead")
check_gradients(upsample2, "UpsamplingHead")

# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print("  ✅  All 6 tests passed.")
print(f"{'='*60}\n")
