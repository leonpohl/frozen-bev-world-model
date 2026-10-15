# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
output_head.py — BEV World Model: Output Heads
===================================================================

Converts the Transformer output back into a spatial feature grid.

CONTEXT IN THE PIPELINE:
    ...
    Transformer      [B,    3, 256]  (Phase 1)
                  or [B, 3072, 256]  (Phase 2)
         ↓
    Output Head      [B, 256, 32, 32]   ← THIS FILE
         ↓
    Upsampling Head  [B, 256, 128, 128] ← upsampling_head.py

TWO CLASSES — ONE PER PHASE:
    FrameLevelOutputHead   Phase 1 — 3 frame tokens in, last token out
    CellLevelOutputHead    Phase 2 — 3072 cell tokens in, frame-t grid out

    Both produce the same output: [B, 256, 32, 32]
    → the UpsamplingHead afterwards is identical for both phases.

IMPORTS FOR OTHER MODULES:
    from output_head import FrameLevelOutputHead, CellLevelOutputHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ===========================================================================
# FrameLevelOutputHead — Phase 1
# ===========================================================================

class FrameLevelOutputHead(nn.Module):
    """
    Converts [B, 3, 256] → [B, 256, 32, 32].

    STRATEGY:
        Phase 1 has only 3 frame tokens — one per frame (t-2, t-1, t).
        We take only the last token (frame t) because, through
        self-attention, it has already absorbed the information of all 3 frames.
        Then we project it with Linear+GELU and inflate it
        with bilinear upsampling from [1,1] to [32,32].

    WHY ONLY THE LAST TOKEN?
        The Transformer is not a causal model (no masking) —
        every token has seen all others. Token t has therefore already
        "processed" t-2 and t-1. It is the most compact summary
        of the current state → ideal basis for the t+1 prediction.

    WHY BILINEAR UPSAMPLE INSTEAD OF ConvTranspose?
        We go from [1,1] to [32,32] — a factor of 32 in one step.
        Bilinear is parameter-free and stable. The actual learnable
        reconstruction of the spatial structure is done by the UpsamplingHead
        afterwards with ConvTranspose layers.

    DATA FLOW:
        [B,  3, 256]  Transformer output
            ↓  [:, -1, :]       last token = frame t
        [B, 256]
            ↓  Linear(256→256) + GELU
        [B, 256]
            ↓  unsqueeze(-1).unsqueeze(-1)
        [B, 256, 1, 1]
            ↓  bilinear upsample → (32, 32)
        [B, 256, 32, 32]

    Input:  [B, n_frames, d_model]
    Output: [B, d_model, grid_size, grid_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.proj   = nn.Linear(config.d_model, config.d_model)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_frames, d_model]  — e.g. [B, 3, 256]

        Returns:
            [B, d_model, grid_size, grid_size]  — e.g. [B, 256, 32, 32]
        """
        G = self.config.grid_size   # 32

        # Step 1: take the last frame token
        # x[:, -1, :] → takes index n_frames-1 = frame t
        x = x[:, -1, :]                             # [B, d_model]

        # Step 2: linear projection + activation
        x = self.act(self.proj(x))                  # [B, d_model]

        # Step 3: append two spatial dimensions (H=1, W=1)
        # F.interpolate needs 4D: [B, C, H, W]
        x = x.unsqueeze(-1).unsqueeze(-1)           # [B, d_model, 1, 1]

        # Step 4: bilinear upsample to grid_size × grid_size
        x = F.interpolate(
            x,
            size=(G, G),
            mode="bilinear",
            align_corners=False,
        )                                            # [B, d_model, 32, 32]

        return x


# ===========================================================================
# CellLevelOutputHead — Phase 2
# ===========================================================================

class CellLevelOutputHead(nn.Module):
    """
    Converts [B, 3072, 256] → [B, 256, 32, 32].

    STRATEGY:
        Phase 2 has 3072 tokens = 3 frames × 32×32 cells.
        The token order from the embedding is:
            Tokens    0..1023  → frame t-2
            Tokens 1024..2047  → frame t-1
            Tokens 2048..3071  → frame t   ← the one we want

        We take the last 1024 tokens (= frame t), project
        each individually with Linear+GELU, and then fold them directly back
        into a [B, 256, 32, 32] grid — no upsample needed because the
        spatial structure was already encoded in the embedding.

    WHY NO BILINEAR UPSAMPLE LIKE IN PHASE 1?
        In Phase 1 we had only 1 vector without spatial info —
        we had to "invent" the structure.
        In Phase 2 each of the 1024 tokens already has an exact
        spatial position (learned via PE_xy in the embedding).
        The reshape from [B,1024,256] → [B,256,32,32] is lossless:
        we only rearrange, we invent nothing.
        That is the scientific core of Phase 2.

    WHY [:, -n_spatial:, :] INSTEAD OF [:, 2048:3072, :]?
        Robustness for ablation studies: with n_frames=2 there are only
        2048 tokens, with n_frames=4 there are 4096.
        "Last 1024 tokens" = always frame t, no matter how many frames.

    DATA FLOW:
        [B, 3072, 256]  Transformer output
            ↓  [:, -1024:, :]   last 1024 tokens = frame t
        [B, 1024, 256]
            ↓  Linear(256→256) + GELU
        [B, 1024, 256]
            ↓  permute(0, 2, 1)
        [B, 256, 1024]
            ↓  reshape(B, 256, 32, 32)
        [B, 256, 32, 32]

    Input:  [B, n_frames * grid_size², d_model]  — e.g. [B, 3072, 256]
    Output: [B, d_model, grid_size, grid_size]   — e.g. [B,  256,  32,  32]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.n_spatial = config.n_spatial_tokens   # 32×32 = 1024
        self.proj      = nn.Linear(config.d_model, config.d_model)
        self.act       = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_frames * grid_size², d_model]  — e.g. [B, 3072, 256]

        Returns:
            [B, d_model, grid_size, grid_size]  — e.g. [B, 256, 32, 32]
        """
        B = x.shape[0]
        G = self.config.grid_size    # 32
        n = self.n_spatial           # 1024

        # Step 1: cut out the last n_spatial tokens = frame t
        # Works for any n_frames (ablation-safe)
        x = x[:, -n:, :]            # [B, 1024, d_model]

        # Step 2: linear projection + activation
        # nn.Linear works on the last dim → each token independently
        x = self.act(self.proj(x))  # [B, 1024, d_model]

        # Step 3: bring the channel dim to the front, then fold spatially
        # permute: [B, 1024, 256] → [B, 256, 1024]
        # reshape: [B, 256, 1024] → [B, 256, 32, 32]
        # The order matches CellLevelEmbedding:
        # there it was flattened row by row → here it is unfolded row by row
        x = x.permute(0, 2, 1)                      # [B, d_model, n_spatial]
        x = x.reshape(B, self.config.d_model, G, G) # [B, d_model, 32, 32]

        return x
