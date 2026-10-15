# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
upsampling_head.py — BEV World Model: Upsampling Head
==========================================================================

Scales the 32×32 feature grid back to the original BEV resolution
128×128 — identical for Phase 1 and Phase 2.

CONTEXT IN THE PIPELINE:
    ...
    Output Head      [B, 256, 32, 32]   ← output_head.py
         ↓
    Upsampling Head  [B, 256, 128, 128] ← THIS FILE
         ↓
    Predicted Latent [B, 256, 128, 128] ← MSE against the real t+1

WHY THIS STEP?
    The downsampling at the start of the pipeline turned [128,128] into a
    [32,32] grid (AvgPool, factor 4). The output head brings us back
    to [32,32]. Now we have to restore the original resolution:
    [32,32] → [128,128], i.e. a factor of 4 again.

    Why not simply bilinear interpolate?
    Bilinear upsampling only copies/interpolates — it cannot learn new
    feature structures. ConvTranspose2d, on the other hand, has learnable
    weights: it can reconstruct real spatial details, not just
    smooth them. This is crucial because the MSE loss is measured directly on
    the 128×128 latents.

IMPORTS FOR OTHER MODULES:
    from upsampling_head import UpsamplingHead
"""

import torch
import torch.nn as nn

from config import ModelConfig


# ===========================================================================
# UpsamplingHead — Phase 1 and Phase 2 identical
# ===========================================================================

class UpsamplingHead(nn.Module):
    """
    Scales [B, 256, 32, 32] → [B, 256, 128, 128] with learnable layers.

    ARCHITECTURE — 3 layers:

        ConvTranspose2d(256→256, kernel=4, stride=2, padding=1)
        + BatchNorm2d(256) + GELU
            → [B, 256, 64, 64]    (×2 in each spatial dim)

        ConvTranspose2d(256→256, kernel=4, stride=2, padding=1)
        + BatchNorm2d(256) + GELU
            → [B, 256, 128, 128]  (another ×2)

        Conv2d(256→256, kernel=3, padding=1)
            → [B, 256, 128, 128]  (shape unchanged — fine-tuning)

    WHY TWO ConvTranspose INSTEAD OF ONE?
        We need a factor of 4 in total (32 → 128).
        Two steps of factor 2 each are more stable than one step of factor 4:
        - Factor 4 in a single ConvTranspose needs kernel=8 — very large,
          hard to optimize, prone to checkerboard artifacts.
        - Two steps of factor 2 each (kernel=4, stride=2) are the
          standard in modern decoder architectures (e.g. U-Net, DCGAN).

    WHY ConvTranspose2d WITH kernel=4, stride=2, padding=1?
        This combination doubles the spatial dimension exactly:
            Output_size = (Input_size - 1) * stride - 2*padding + kernel
                        = (32 - 1) * 2 - 2*1 + 4
                        = 62 - 2 + 4 = 64  ✓  (first layer)
                        = (64 - 1) * 2 - 2*1 + 4 = 128  ✓  (second layer)
        kernel=4, stride=2, padding=1 is the "magic combination"
        for exact 2× upsampling without edge effects.

    WHY BATCHNORM AFTER CONVTRANSPOSE?
        ConvTranspose can produce very different activation
        scales internally. BatchNorm normalizes the channels to
        mean≈0 and std≈1 — this stabilizes training
        and allows higher learning rates.

    WHY A FINAL Conv2d?
        The ConvTranspose layers handle the upsampling.
        The final Conv2d(3×3) is a "refinement step": it can
        smooth or sharpen local structures without changing the resolution.
        It gives the model the opportunity to refine the
        upsampled features once more before
        the MSE loss is computed.

    WHY NO Sigmoid/Tanh AT THE END?
        BEVFusion latents are raw feature values without a fixed
        value range — they can be arbitrarily positive or negative.
        A sigmoid would bound the output to [0,1],
        tanh to [-1,1] — both would be wrong and would distort the MSE
        loss. We leave the activations open (linear).

    DATA FLOW:
        [B, 256,  32,  32]  ← output head
            ↓  ConvTranspose(stride=2) + BN + GELU
        [B, 256,  64,  64]
            ↓  ConvTranspose(stride=2) + BN + GELU
        [B, 256, 128, 128]
            ↓  Conv2d(3×3) — no stride, no size change
        [B, 256, 128, 128]  ← predicted BEV latent

    Input:  [B, d_model, grid_size, grid_size]      — e.g. [B, 256,  32,  32]
    Output: [B, d_model, grid_size*4, grid_size*4]  — e.g. [B, 256, 128, 128]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        C = config.d_model  # 256 — channel count stays constant through all layers

        # ------------------------------------------------------------------
        # Layer 1: 32×32 → 64×64
        # ------------------------------------------------------------------
        # ConvTranspose2d: "transposed convolution" / "deconvolution"
        #   in_channels=C, out_channels=C: channel count stays 256
        #   kernel_size=4: 4×4 filter kernel
        #   stride=2: output pixels cover 2× more area than input pixels
        #   padding=1: cuts off 1 pixel at the edge → exact 2× magnification
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=C),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Layer 2: 64×64 → 128×128
        # ------------------------------------------------------------------
        # Identical configuration to layer 1 — another ×2 upsampling.
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=C),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Layer 3: 128×128 → 128×128 (refinement, no size change)
        # ------------------------------------------------------------------
        # Conv2d with kernel=3, padding=1: output has the same size as input.
        # No BatchNorm, no GELU — direct linear output.
        # This is the last layer before the MSE loss; a linear output
        # gives the loss the full value range.
        self.refine = nn.Conv2d(C, C, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, grid_size, grid_size]  — e.g. [B, 256, 32, 32]

        Returns:
            [B, d_model, grid_size*4, grid_size*4]  — e.g. [B, 256, 128, 128]
        """
        x = self.up1(x)      # [B, 256,  32,  32] → [B, 256,  64,  64]
        x = self.up2(x)      # [B, 256,  64,  64] → [B, 256, 128, 128]
        x = self.refine(x)   # [B, 256, 128, 128] → [B, 256, 128, 128]
        return x
