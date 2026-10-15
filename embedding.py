# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
embedding.py — BEV World Model: Token Embedding Module
========================================================================

Converts the downsampled BEV latents [B, T, C, 32, 32] into a
sequence of d_model-dimensional tokens that the Transformer
can process.

TWO VARIANTS — ONLY ONE LINE DIFFERS IN THE MAIN MODEL:

  Phase 1 (frame level):   3 tokens   — one token per frame
  Phase 2 (cell level):  3072 tokens  — one token per spatial cell

  # In BEVWorldModel:
  self.embedding = FrameLevelEmbedding(config)  # Phase 1
  self.embedding = CellLevelEmbedding(config)   # Phase 2 — just swap this line

WHY DO WE EVEN NEED AN EMBEDDING?
    A Transformer processes sequences of vectors — concretely:
    [sequence length, d_model]. Our BEV latents, however, are 3D feature maps
    [C, H, W]. The embedding resolves this mismatch: it projects the
    spatial feature maps into a flat sequence of d_model vectors.

    In addition, the positional encoding adds information to the Transformer
    about WHERE (spatially) and WHEN (temporally) a token comes from —
    without PE the token order would be meaningless.

WHY POSITIONAL ENCODING?
    Transformer attention is permutation-invariant: if you permute all
    tokens, the output does not change. For our world model, however,
    it is crucial that t-2 comes before t-1 before t (temporally), and in Phase 2
    also where in the BEV grid each cell lies (spatially). PE makes this
    structure explicit through added learned vectors.
"""

import torch
import torch.nn as nn

from config import ModelConfig


# ---------------------------------------------------------------------------
# Phase 1: Frame-level embedding (3 tokens)
# ---------------------------------------------------------------------------

class FrameLevelEmbedding(nn.Module):
    """
    Compresses each BEV frame into a single d_model vector (token).

    Strategy:
        1. GlobalAvgPool: [C, 32, 32] → [C]
           Summarizes the spatial information of each channel into a
           scalar: "how active is channel k in this frame?"
        2. Linear(C, d_model): projects into the Transformer space
        3. + PE_t: adds a learned vector per frame position
           so the Transformer knows whether it sees frame t-2, t-1 or t

    Result: [B, T, d_model] — T tokens, one per frame.

    WHY GlobalAvgPool INSTEAD OF Flatten?
        Flatten would give [C, 32, 32] → [C×32×32] = [262,144]-dim vector.
        Linear(262144, 256) would have 67 M parameters — just for the embedding.
        GlobalAvgPool + Linear(256, 256) has only ~66k parameters.
        In return we lose the spatial information within a frame —
        that is the trade-off of Phase 1. Phase 2 (CellLevelEmbedding)
        keeps the spatial structure through cell tokens.

    Args:
        config: ModelConfig with d_model and n_frames

    Input:  [B, T, C, H, W]  (after DownsamplingModule: H=W=32, C=256)
    Output: [B, T, d_model]  — T=3 tokens
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # GlobalAvgPool: averages over H and W → [B*T, C]
        # AdaptiveAvgPool2d(1) means: output size = 1×1, no matter how large
        # the input is. This way the module also works when H/W
        # change due to other downsampling configs.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Linear projection: C → d_model
        # Even if C == d_model (both 256), this projection is important:
        # it gives the model the opportunity to learn a different basis
        # than the one BEVFusion uses internally. Without it it would be just an
        # identity function — with it, it is a learned re-projection.
        self.proj = nn.Linear(config.d_model, config.d_model)

        # Temporal positional encoding: a learned lookup table
        # with n_frames entries, each d_model-dimensional.
        # PE_t[0] = "I am frame t-2"
        # PE_t[1] = "I am frame t-1"
        # PE_t[2] = "I am frame t"
        # These vectors are learned during training.
        self.pe_t = nn.Embedding(config.n_frames, config.d_model)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  — downsampled BEV latents

        Returns:
            tokens: [B, T, d_model]  — T frame tokens
        """
        B, T, C, H, W = x.shape

        # --- Step 1: fold B and T together ---
        # AdaptiveAvgPool2d expects [N, C, H, W].
        x = x.view(B * T, C, H, W)                      # [B*T, C, 32, 32]

        # --- Step 2: GlobalAvgPool over H and W ---
        # squeeze(-1).squeeze(-1): removes the two size-1 dimensions
        # that AdaptiveAvgPool2d(1) appends → [B*T, C, 1, 1] → [B*T, C]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B*T, C]

        # --- Step 3: linear projection ---
        x = self.proj(x)                                  # [B*T, d_model]

        # --- Step 4: unfold back to [B, T, d_model] ---
        x = x.view(B, T, self.config.d_model)             # [B, T, 256]

        # --- Step 5: add the temporal positional encoding ---
        # torch.arange(T) = [0, 1, 2] for T=3
        # pe_t(t_idx) = lookup → [T, d_model] = [3, 256]
        # Broadcasting: [B, T, 256] + [T, 256] → [B, T, 256]
        # Every batch entry gets the same PE vectors added.
        t_idx = torch.arange(T, device=x.device)         # [T]
        x = x + self.pe_t(t_idx)                          # [B, T, 256]

        return x  # [B, 3, 256] — 3 tokens, ready for the Transformer


# ---------------------------------------------------------------------------
# Phase 2: cell-level embedding (3072 tokens)
# ---------------------------------------------------------------------------

class CellLevelEmbedding(nn.Module):
    """
    Treats each spatial cell of each frame as its own token.

    Strategy:
        1. Flatten: [C, 32, 32] → [32×32, C] = [1024, C] per frame
           Each of the 1024 cells becomes a candidate token.
        2. Linear(C, d_model): projects into the Transformer space
        3. + PE_xy: adds spatial PE (concat of PE_x and PE_y)
           so the Transformer knows where in the BEV grid a cell lies
        4. + PE_t: adds temporal PE

    Result: [B, T×H×W, d_model] = [B, 3072, 256] — 3072 tokens.

    WHY SEPARATE PE_x AND PE_y INSTEAD OF A 2D EMBEDDING?
        A 2D embedding table would have 32×32 = 1024 entries of 256 dim
        = 262k parameters. By factorizing into PE_x (32 × 128) and
        PE_y (32 × 128) with concatenation we have only 2 × 32 × 128
        = 8192 parameters — 32× more economical, yet complete.
        Additionally the model can learn x- and y-structure separately.

    Args:
        config: ModelConfig with d_model, n_frames, grid_size

    Input:  [B, T, C, H, W]  (after DownsamplingModule: H=W=32, C=256)
    Output: [B, T×H×W, d_model]  — 3072 tokens
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        G = config.grid_size  # 32

        # Projection: C → d_model (identical to Phase 1)
        self.proj = nn.Linear(config.d_model, config.d_model)

        # Spatial PE — factorized into x and y:
        # PE_x: embedding for column position (0..31) → d_model//2 = 128 dim
        # PE_y: embedding for row position   (0..31) → d_model//2 = 128 dim
        # Concatenation → 256 dim = d_model
        # d_model must be even (checked by __post_init__ in ModelConfig)
        self.pe_x = nn.Embedding(G, config.d_model // 2)
        self.pe_y = nn.Embedding(G, config.d_model // 2)

        # Temporal PE — identical to Phase 1
        self.pe_t = nn.Embedding(config.n_frames, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]  — downsampled BEV latents

        Returns:
            tokens: [B, T*H*W, d_model]  — 3072 tokens
        """
        B, T, C, H, W = x.shape
        G = self.config.grid_size

        # --- Step 1: spatial flatten ---
        # Goal: every (frame, row, column) combination becomes a token.
        # permute: [B, T, C, H, W] → [B, T, H, W, C]
        #   (channel to the last position for easy reshaping)
        # reshape: → [B, T*H*W, C]
        #   all cells of all frames into one long sequence
        x = x.permute(0, 1, 3, 4, 2)               # [B, T, H, W, C]
        x = x.reshape(B, T * H * W, C)              # [B, 3072, 256]

        # --- Step 2: linear projection ---
        x = self.proj(x)                             # [B, 3072, d_model]

        # --- Step 3: build the spatial PE ---
        # col: [0, 1, ..., 31] — column indices
        # row: [0, 1, ..., 31] — row indices
        col = torch.arange(W, device=x.device)       # [32]
        row = torch.arange(H, device=x.device)       # [32]

        # PE_x lookup: [32, 128]
        # PE_y lookup: [32, 128]
        pe_x = self.pe_x(col)                        # [W, d_model//2]
        pe_y = self.pe_y(row)                        # [H, d_model//2]

        # Span the grid: every cell gets its (x,y) PE vector.
        # pe_x.unsqueeze(0).expand(H, -1, -1): [H, W, d_model//2]
        #   → the same column PE for all rows
        # pe_y.unsqueeze(1).expand(-1, W, -1): [H, W, d_model//2]
        #   → the same row PE for all columns
        # torch.cat(..., dim=-1): [H, W, d_model]
        # reshape: [H*W, d_model] = [1024, 256]
        pe_xy = torch.cat([
            pe_x.unsqueeze(0).expand(H, -1, -1),    # [H, W, d_model//2]
            pe_y.unsqueeze(1).expand(-1, W, -1),    # [H, W, d_model//2]
        ], dim=-1).reshape(H * W, self.config.d_model)  # [1024, 256]

        # Repeat PE_xy for all T frames:
        # [1024, 256] → repeat(T, 1) → [3072, 256]
        pe_xy = pe_xy.unsqueeze(0).expand(T, -1, -1).reshape(
            T * H * W, self.config.d_model
        )                                            # [3072, 256]

        # --- Step 4: temporal PE ---
        # Each cell belongs to exactly one frame.
        # t_idx: for the first H*W tokens t=0, next H*W t=1, etc.
        # repeat_interleave: [0,0,...0, 1,1,...1, 2,2,...2] (H*W times each)
        t_idx = torch.arange(T, device=x.device).repeat_interleave(H * W)
        pe_t  = self.pe_t(t_idx)                    # [3072, 256]

        # --- Step 5: add the PE ---
        # Broadcasting: [B, 3072, 256] + [3072, 256] → [B, 3072, 256]
        x = x + pe_xy + pe_t                        # [B, 3072, 256]

        return x  # [B, 3072, 256] — ready for the Transformer

    def extra_repr(self) -> str:
        G = self.config.grid_size
        T = self.config.n_frames
        return (f"grid={G}×{G}, n_frames={T}, "
                f"n_tokens={T*G*G}, d_model={self.config.d_model}")
