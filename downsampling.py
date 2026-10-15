# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
downsampling.py — BEV World Model: downsampling module
================================================================

Reduces the spatial resolution of the BEV latents from 128×128 to 32×32
before they flow into the embedding module.

WHY DOWNSAMPLING?
    BEVFusion provides latents of shape [256, 128, 128].
    In Phase 2 (cell level) each spatial cell becomes a token.
    Without downsampling: 3 × 128 × 128 = 49,152 tokens
    With downsampling:    3 ×  32 ×  32 =  3,072 tokens

    The attention matrix scales quadratically with the token count:
        49,152² = 2.4 billion entries  → not trainable
         3,072² =     9.4 million entries → manageable on an A100

WHY AvgPool INSTEAD OF MaxPool OR CONV?
    - AvgPool: averages all 4×4 pixels → preserves global activation strength.
               Good for continuous BEV feature maps.
    - MaxPool: takes only the strongest pixel → loses weak activations.
    - Strided conv: would have learnable parameters → would transform
                    BEVFusion latents instead of just compressing them.
    AvgPool is the "most honest" downsampler: no weights, no
    transformation, just compression via averaging.

WHY kernel=4, stride=4?
    128 / 4 = 32 — fits exactly without padding.
    stride = kernel → no overlap; each 4×4 region flows exactly
    once into exactly one output pixel.

SHAPE OVERVIEW:
    Input:  [B, T, C, 128, 128]
    Output: [B, T, C,  32,  32]
    Parameter: 0
"""

import torch
import torch.nn as nn


class DownsamplingModule(nn.Module):
    """
    Spatial downsampling of BEV latent sequences via average pooling.

    Stateless (0 parameters) — identical for Phase 1 and Phase 2.

    Args:
        kernel_size: size of the pooling window (default: 4)
        stride:      step size (default: 4 — no overlap)

    Example:
        >>> ds = DownsamplingModule()
        >>> x = torch.randn(4, 3, 256, 128, 128)
        >>> ds(x).shape
        torch.Size([4, 3, 256, 32, 32])
    """

    def __init__(self, kernel_size: int = 4, stride: int = 4):
        super().__init__()
        # nn.AvgPool2d expects [N, C, H, W].
        # kernel=stride=4: 128 → (128-4)/4 + 1 = 32. Exact, no padding needed.
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
               B=batch, T=frames(3), C=channels(256), H=W=128

        Returns:
            [B, T, C, H//stride, W//stride]  →  [B, 3, 256, 32, 32]

        Strategy — temporarily fold B and T together:
            AvgPool2d does not understand 5D tensors, only [N, C, H, W].
            We treat B*T as one large batch dimension,
            perform the pooling, and then fold back afterwards.
            view() shares memory with x — no copy.
        """
        B, T, C, H, W = x.shape

        # Step 1: [B, T, C, H, W] → [B*T, C, H, W]
        # All T frames of all B samples are treated as one batch.
        x = x.view(B * T, C, H, W)

        # Step 2: pooling on H and W — N and C remain untouched.
        x = self.pool(x)                      # [B*T, C, 32, 32]

        # Step 3: [B*T, C, H_out, W_out] → [B, T, C, H_out, W_out]
        # Read H_out and W_out from the result (not hardcoded)
        # so that the module also works with other kernel/stride configs.
        _, _, H_out, W_out = x.shape
        x = x.view(B, T, C, H_out, W_out)

        return x

    def extra_repr(self) -> str:
        """Shown by print(model)."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, params=0"
