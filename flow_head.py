# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
flow_head.py — Flow-matching residual head.

Learns the DISTRIBUTION of the residual r = x_real - x_det via flow matching
(Lipman 2023, rectified-flow paths; anchored on VFMF/FlowWM 2025/26):

    r_tau = (1 - tau) * eps + tau * r,   eps ~ N(0,I), tau ~ U[0,1]
    L_FM  = || v_theta(r_tau, tau, cond) - (r - eps) ||^2

Sampling: Euler ODE from tau=0 (noise) to tau=1 (residual), N steps.
Conditioning cond = [x_det, last input frame] (concatenated as channels);
tau as sinusoidal embedding, applied via FiLM to every block. Size scaled
via flow_dim/flow_layers config; default ~6.5M params = parity with the
6M backbone ("same capacity, different objective"). Output conv is
zero-init (v starts at 0 -> stable start of training).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(tau: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """tau [B] in [0,1] -> sine/cosine embedding [B, dim] (DDPM convention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
    )
    ang = tau.float()[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowHead(nn.Module):
    """Velocity field v_theta(r_tau, tau, cond) -> [B, C, H, W]."""

    def __init__(self, config):
        super().__init__()
        C = config.d_model                                  # Latent channels (256)
        W = getattr(config, "flow_dim", 224)
        L = getattr(config, "flow_layers", 8)
        self.n_layers = L
        self.width = W

        # Input: r_tau + x_det + skip (last input frame) as channels
        self.inp = nn.Conv2d(3 * C, W, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(W, W, kernel_size=3, padding=1) for _ in range(L)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(8, W) for _ in range(L))
        # tau embedding -> per-block FiLM (scale, shift)
        self.tau_mlp = nn.Sequential(
            nn.Linear(128, W), nn.SiLU(), nn.Linear(W, 2 * W * L),
        )
        self.out = nn.Conv2d(W, C, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, r_tau, tau, x_det, skip):
        """
        r_tau: [B,C,H,W] point on the noise->residual path
        tau:   [B]       path position in [0,1]
        x_det: [B,C,H,W] deterministic backbone prediction (conditioning)
        skip:  [B,C,H,W] last input frame (conditioning)
        """
        B = r_tau.shape[0]
        film = self.tau_mlp(sinusoidal_embedding(tau)).view(B, self.n_layers, 2, self.width)
        x = self.inp(torch.cat([r_tau, x_det, skip], dim=1))
        for i, (blk, nrm) in enumerate(zip(self.blocks, self.norms)):
            h = nrm(x)
            s, b = film[:, i, 0], film[:, i, 1]
            h = h * (1.0 + s[:, :, None, None]) + b[:, :, None, None]
            x = x + blk(F.silu(h))
        return self.out(F.silu(x))

    @torch.no_grad()
    def sample(self, x_det, skip, steps: int = 10, generator=None):
        """
        Euler integration dx/dtau = v_theta from tau=0 (eps~N(0,I)) to tau=1.
        Returns: residual sample r_hat [B,C,H,W]. generator (torch.Generator
        on the same device) makes the sample reproducible.
        """
        if generator is not None:
            r = torch.empty_like(x_det).normal_(generator=generator)
        else:
            r = torch.randn_like(x_det)
        dt = 1.0 / steps
        for k in range(steps):
            tau = torch.full((x_det.shape[0],), k * dt, device=x_det.device)
            r = r + dt * self.forward(r, tau, x_det, skip)
        return r
