# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
bev_world_model.py — BEV World Model: complete model
============================================================================

Assembles all individual modules into a single nn.Module.

PIPELINE COMPLETE:

    Input:  [B, 3, 256, 128, 128]   ← 3 consecutive BEV latents
         |
         v
    DownsamplingModule               128×128 → 32×32 (AvgPool, 0 parameters)
         |
         v
    FrameLevelEmbedding              [B, 3, 256, 32, 32] → [B,    3, 256]
    or CellLevelEmbedding            [B, 3, 256, 32, 32] → [B, 3072, 256]
         |
         v
    TransformerEncoder               [B, seq, 256] → [B, seq, 256]
    (N=4 blocks, phase-agnostic)
         |
         v
    FrameLevelOutputHead             [B,    3, 256] → [B, 256, 32, 32]
    or CellLevelOutputHead           [B, 3072, 256] → [B, 256, 32, 32]
         |
         v
    UpsamplingHead                   [B, 256, 32, 32] → [B, 256, 128, 128]
         |
         v
    Output: [B, 256, 128, 128]      ← predicted next BEV latent t+1

SWITCH PHASE:
    Only change config.phase — "frame" or "cell".
    All other modules remain identical.

    cfg = ModelConfig(phase="frame")   # Phase 1: fast, 3 tokens
    cfg = ModelConfig(phase="cell")    # Phase 2: complete, 3072 tokens

IMPORTS FOR TRAIN.PY:
    from bev_world_model import BEVWorldModel
    from config          import ModelConfig

    cfg   = ModelConfig(phase="frame")
    model = BEVWorldModel(cfg).to(device)
"""

import torch
import torch.nn as nn

from config          import ModelConfig
from downsampling    import DownsamplingModule
from embedding       import FrameLevelEmbedding, CellLevelEmbedding
from transformer     import TransformerEncoder
from output_head     import FrameLevelOutputHead, CellLevelOutputHead
from upsampling_head import UpsamplingHead


class BEVWorldModel(nn.Module):
    """
    Complete BEV world model — orchestrates all submodules.

    Takes 3 consecutive BEV latents and predicts the next one.

    WHICH MODULES ARE LOADED?
        Depending on config.phase, __init__ selects the right classes:

        Phase 1 ("frame"):
            embedding   = FrameLevelEmbedding  → 3 tokens
            output_head = FrameLevelOutputHead  → last token → grid

        Phase 2 ("cell"):
            embedding   = CellLevelEmbedding   → 3072 tokens
            output_head = CellLevelOutputHead  → last frame → grid

        All other modules (downsample, transformer, upsample) are
        phase-agnostic — identical code for both phases.

    PARAMETER OVERVIEW (Phase 1, N=4):
        DownsamplingModule       0 parameters   (AvgPool — no weights)
        FrameLevelEmbedding     ~132k parameters (Linear + PE_t)
        TransformerEncoder      ~3.2M parameters (4 blocks)
        FrameLevelOutputHead     ~66k parameters (Linear)
        UpsamplingHead          ~2.7M parameters (ConvTranspose × 2 + conv)
        ─────────────────────────────────────────
        Total Phase 1:          ~6.1M parameters

    PARAMETER OVERVIEW (Phase 2, N=4):
        DownsamplingModule       0 parameters
        CellLevelEmbedding      ~74k parameters  (Linear + PE_x + PE_y + PE_t)
        TransformerEncoder      ~3.2M parameters
        CellLevelOutputHead     ~66k parameters
        UpsamplingHead          ~2.7M parameters
        ─────────────────────────────────────────
        Total Phase 2:          ~6.0M parameters

    Args:
        config: ModelConfig with all hyperparameters and phase control

    Input:  [B, n_frames, C, H, W]      — e.g. [B, 3, 256, 128, 128]
    Output: [B, C, H, W]                — e.g. [B, 256, 128, 128]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Downsampling — identical for both phases
        # ------------------------------------------------------------------
        # 128×128 → 32×32, 0 parameters, no training needed
        self.downsample = DownsamplingModule()

        # ------------------------------------------------------------------
        # 2. Embedding — phase-dependent
        # ------------------------------------------------------------------
        # config.phase controls which class is instantiated.
        # To switch to Phase 2: just set config.phase = "cell".
        if config.phase == "frame":
            self.embedding = FrameLevelEmbedding(config)   # → [B, 3, 256]
        else:
            self.embedding = CellLevelEmbedding(config)    # → [B, 3072, 256]

        # ------------------------------------------------------------------
        # 3. Transformer — identical for both phases
        # ------------------------------------------------------------------
        # Processes [B, seq_len, 256] — seq_len is 3 or 3072,
        # the Transformer does not notice the difference.
        self.transformer = TransformerEncoder(config)

        # ------------------------------------------------------------------
        # 4. Output head — phase-dependent
        # ------------------------------------------------------------------
        # Both heads deliver [B, 256, 32, 32] — UpsamplingHead is identical.
        if config.phase == "frame":
            self.output_head = FrameLevelOutputHead(config)  # last token
        else:
            self.output_head = CellLevelOutputHead(config)   # last frame

        # ------------------------------------------------------------------
        # 5. Upsampling — identical for both phases
        # ------------------------------------------------------------------
        # [B, 256, 32, 32] → [B, 256, 128, 128]
        self.upsample = UpsamplingHead(config)

        # ------------------------------------------------------------------
        # 6. Gated skip connection
        # ------------------------------------------------------------------
        # Learnable gate α ∈ [0,1] per channel and spatial position.
        # Controls how strongly the Transformer output vs. frame t is used.
        #
        # WHY A GATE instead of simple addition?
        #   Simple addition:    pred = transformer_out + skip
        #   → the model must learn exactly Δ(t→t+1)
        #   → with large movement (vehicles) skip fights against transformer_out
        #
        #   Gated:  pred = α * transformer_out + (1-α) * skip
        #   → α=0: only skip (static areas — road, buildings)
        #   → α=1: only transformer (dynamic areas — vehicles)
        #   → the model learns itself where to predict and where to copy
        #
        # IMPLEMENTATION: 1×1 conv → sigmoid
        #   Input:  transformer_out [B, 256, 128, 128]
        #   Output: gate α          [B, 256, 128, 128] ∈ [0,1]
        #   Parameters: 256×256+256 = 65,792 (small but learnable)
        C = config.d_model  # 256
        self.gate = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=True),  # 1x1 conv, channel-wise
            nn.Sigmoid(),                                # → [0, 1]
        )
        # Initialize gate bias to 0 → sigmoid(0) = 0.5
        # At the beginning: 50/50 between transformer and skip
        nn.init.zeros_(self.gate[0].bias)

        # --- Ego-motion conditioning (FiLM on the embedded tokens) ---
        # ego_delta per sample: [n_frames*4] = [dx,dy,cos,sin] per step (incl.
        # F_last->target). The mode selects the slice: action=last 4, state=first
        # (n_frames-1)*4, both=all. FiLM starts as IDENTITY (gamma=1,beta=0)
        # -> ego_cond_mode="off" is bit-identical to the model without conditioning.
        # Modes: off | action/state/both (FiLM on tokens) | gate (bias on gate
        # logits) | token (ego token prepended to the sequence). action/gate/token use the
        # F3->F4 action (4-dim). All additional heads ZERO-INIT -> start = old model.
        self.ego_cond_mode = getattr(config, "ego_cond_mode", "off")
        self._n_frames = config.n_frames
        if self.ego_cond_mode != "off":
            _ed = {"action": 4, "gate": 4, "token": 4,
                   "state": (config.n_frames - 1) * 4,
                   "both": config.n_frames * 4}[self.ego_cond_mode]
            if self.ego_cond_mode in ("action", "state", "both"):
                self.film = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, 2 * C))
                nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            elif self.ego_cond_mode == "gate":
                # per-channel additive bias on the gate logits (zero-init = no-op)
                self.film_gate = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, C))
                nn.init.zeros_(self.film_gate[-1].weight); nn.init.zeros_(self.film_gate[-1].bias)
            elif self.ego_cond_mode == "token":
                # ego token prepended to the token sequence
                self.ego_token_mlp = nn.Sequential(nn.Linear(_ed, C), nn.GELU(), nn.Linear(C, C))
                nn.init.zeros_(self.ego_token_mlp[-1].weight); nn.init.zeros_(self.ego_token_mlp[-1].bias)

        # --- CVAE head -------------------------------------------
        # The latent variable z encodes "which of the possible futures". Prior
        # p(z|h) sees only the context (token mean h), posterior q(z|h,target)
        # additionally the real target frame (via the existing downsampling, no
        # new encoder). z is injected per FiLM onto the tokens before the output head;
        # FiLM ZERO-INIT -> at init, z has null effect = off-identical
        # main path. vae_mode="off" (default) builds NONE of these modules.
        self.vae_mode = getattr(config, "vae_mode", "off")
        if self.vae_mode == "cvae":
            Z = getattr(config, "vae_z_dim", 32)
            self.vae_prior = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 2 * Z))
            self.vae_post  = nn.Sequential(nn.Linear(2 * C, C), nn.GELU(), nn.Linear(C, 2 * Z))
            self.vae_film  = nn.Sequential(nn.Linear(Z, C), nn.GELU(), nn.Linear(C, 2 * C))
            nn.init.zeros_(self.vae_film[-1].weight); nn.init.zeros_(self.vae_film[-1].bias)
        # KL statistics of the last forward: (mu_q, logvar_q, mu_p, logvar_p)
        # or (None, None, mu_p, logvar_p) at inference. Always present.
        self.last_vae_stats = None

        # --- Flow-matching residual head -------------------------
        # IMPORTANT: does NOT change forward() — the head is used exclusively
        # explicitly (training: model.flow(...); sampling:
        # model.flow_sample(...)). flow_head="off" builds nothing.
        if getattr(config, "flow_head", "off") == "flow":
            from flow_head import FlowHead
            self.flow = FlowHead(config)

    def _ego_sig(self, ego_delta):
        """Signal slice per mode: state=first (n-1)*4, both=all, otherwise F3->F4 (last 4)."""
        if self.ego_cond_mode == "state":
            return ego_delta[:, :(self._n_frames - 1) * 4]
        if self.ego_cond_mode == "both":
            return ego_delta
        return ego_delta[:, -4:]

    def forward(self, x: torch.Tensor, ego_delta: torch.Tensor = None,
                target: torch.Tensor = None, z_mode: str = "mean") -> torch.Tensor:
        """
        Complete forward pass through all 5 stages + skip connection.

        Args:
            x: [B, n_frames, C, H, W]  — e.g. [B, 3, 256, 128, 128]
               Batch of consecutive BEV latents (t-2, t-1, t)

        Returns:
            [B, C, H, W]  — e.g. [B, 256, 128, 128]
            Predicted BEV latent for time step t+1

        SKIP CONNECTION — WHY AND WHERE:
            skip = x[:, -1, ...]  stores frame t (the most recent input)
            BEFORE the downsampling destroys the spatial details.

            After the upsampling, skip is added:
                output = upsample(transformer(embedding(downsample(x)))) + skip

            Semantically that means:
                pred(t+1) = Δ + t
                          = "what changes" + "what is currently here"

            The Transformer therefore only learns the DIFFERENCE (Δ) between
            t and t+1 — not t+1 from scratch. That is much easier
            because at 2Hz (0.5s between frames) most of the BEV scene
            stays identical (road, static objects, buildings).

            Additional effect: the fine channel structures from BEVFusion
            (high-frequency details at 128×128) are preserved — directly
            important for later decoder compatibility.
        """

        # --- Skip connection: remember frame t BEFORE downsampling ---
        # x[:, -1, :, :, :] = last frame in the time dimension = frame t
        # Shape: [B, 256, 128, 128] — full resolution, all details
        skip = x[:, -1, :, :, :]                    # [B, C, H, W]

        # Stage 1: spatial downsampling
        # [B, 3, 256, 128, 128] → [B, 3, 256, 32, 32]
        x = self.downsample(x)

        # Stage 2: token embedding + positional encoding
        # Phase 1: [B, 3, 256, 32, 32] → [B,    3, 256]
        # Phase 2: [B, 3, 256, 32, 32] → [B, 3072, 256]
        x = self.embedding(x)

        # Ego conditioning. FiLM (action/state/both) modulates the tokens;
        # token prepends an ego token; gate acts later on the gate logits.
        _ego_tok = False
        if self.ego_cond_mode != "off" and ego_delta is not None:
            e = self._ego_sig(ego_delta)
            if self.ego_cond_mode in ("action", "state", "both"):
                gamma, beta = self.film(e).chunk(2, dim=-1)      # each [B, C]
                x = (1.0 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)
            elif self.ego_cond_mode == "token":
                tok = self.ego_token_mlp(e).unsqueeze(1)         # [B,1,C]
                x = torch.cat([tok, x], dim=1)                   # [B, seq+1, C]
                _ego_tok = True

        # Stage 3: Transformer (self-attention + FFN × N)
        # [B, seq, 256] → [B, seq, 256]  — shape invariant
        x = self.transformer(x)
        if _ego_tok:
            x = x[:, 1:]                                         # cut off the ego token again

        # --- CVAE — sample z and inject via FiLM -------------
        # Training (target given): z ~ q(z|h,target) via reparameterization.
        # Inference (target=None): z from p(z|h) — z_mode "mean" (deterministic,
        # default: all existing eval paths stay reproducible) or
        # "sample" (stochastic, for diversity/sharpness eval).
        # logvar clamp [-8, 8] against exp overflow in early epochs.
        if self.vae_mode == "cvae":
            h = x.mean(dim=1)                                    # [B, C] context
            mu_p, lv_p = self.vae_prior(h).chunk(2, dim=-1)
            lv_p = lv_p.clamp(-8.0, 8.0)
            if target is not None:
                t = self.downsample(target.unsqueeze(1)).squeeze(1)   # [B,C,32,32]
                t = t.flatten(2).mean(dim=2)                          # [B, C]
                mu_q, lv_q = self.vae_post(torch.cat([h, t], dim=-1)).chunk(2, dim=-1)
                lv_q = lv_q.clamp(-8.0, 8.0)
                z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * lv_q)
                self.last_vae_stats = (mu_q, lv_q, mu_p, lv_p)
            else:
                if z_mode == "sample":
                    z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * lv_p)
                else:
                    z = mu_p
                self.last_vae_stats = (None, None, mu_p, lv_p)
            gz, bz = self.vae_film(z).chunk(2, dim=-1)           # each [B, C]
            x = (1.0 + gz).unsqueeze(1) * x + bz.unsqueeze(1)

        # Stage 4: output head — tokens back into a grid
        # Phase 1: [B,    3, 256] → [B, 256, 32, 32]
        # Phase 2: [B, 3072, 256] → [B, 256, 32, 32]
        x = self.output_head(x)

        # Stage 5: upsampling to the original resolution
        # [B, 256, 32, 32] → [B, 256, 128, 128]
        x = self.upsample(x)

        # --- Gated skip connection ---
        # Gate α = sigmoid(conv1x1(transformer_out)) ∈ [0,1]
        # pred = α * transformer_out + (1-α) * skip
        #
        # α close to 0: skip dominates → static areas are copied
        # α close to 1: transformer dominates → movement is predicted
        #
        # Advantage over simple addition (x + skip):
        #   - no fight between Δ and skip with large movement
        #   - the model can differentiate spatially (per pixel, per channel)
        #   - visualizable: gate.mean(dim=1) shows where the model predicts vs copies
        # gate mode: per-channel ego bias on the gate logits (before sigmoid),
        # otherwise the normal gate. Zero-init -> start identical to the gate without ego.
        # residual mode (ResWorld/DeltaWorld pattern): hard residual
        # instead of gate blend -- the head predicts the DELTA to the last frame.
        if getattr(self.config, "output_mode", "gated") == "residual":
            return skip + x                         # [B, 256, 128, 128]

        if self.ego_cond_mode == "gate" and ego_delta is not None:
            b = self.film_gate(self._ego_sig(ego_delta))          # [B, C]
            alpha = torch.sigmoid(self.gate[0](x) + b[:, :, None, None])
        else:
            alpha = self.gate(x)                    # [B, 256, 128, 128] ∈ [0,1]
        x     = alpha * x + (1.0 - alpha) * skip    # [B, 256, 128, 128]

        return x

    @torch.no_grad()
    def flow_sample(self, x_input, ego_delta=None, steps: int = 10, generator=None,
                    scale: float = 1.0):
        """
        18/B2: ONE stochastic sample x = x_det + scale * r_hat.
        x_input: [B, n_frames, C, H, W] (like forward). Requires flow_head=flow.
        scale (18/B2.1): post-hoc residual scaling (temperature analog)
        for variance calibration; 1.0 = unchanged.
        """
        x_det = self.forward(x_input, ego_delta)
        skip = x_input[:, -1, :, :, :]
        r_hat = self.flow.sample(x_det, skip, steps=steps, generator=generator)
        return x_det + scale * r_hat

    def count_parameters(self) -> dict:
        """
        Returns the parameter count per module.
        Useful for VRAM estimation and the thesis table.

        Returns:
            dict with module names → parameter counts
        """
        modules = {
            "downsample":   self.downsample,
            "embedding":    self.embedding,
            "transformer":  self.transformer,
            "output_head":  self.output_head,
            "upsample":     self.upsample,
            "gate":         self.gate,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in modules.items()
        }
        counts["total"] = sum(counts.values())
        return counts


# ===========================================================================
# Self-test when the file is executed directly
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BEVWorldModel — self-test")
    print("=" * 60)

    B = 2   # small batch for a fast test

    for phase in ["frame", "cell"]:
        print(f"\n--- Phase: '{phase}' ---")

        cfg   = ModelConfig(phase=phase)
        model = BEVWorldModel(cfg)
        model.eval()

        # Simulated input: 3 BEV latents [B, 3, 256, 128, 128]
        x = torch.randn(B, cfg.n_frames, 256, 128, 128)
        print(f"  Input:  {list(x.shape)}")

        with torch.no_grad():
            out = model(x)
        print(f"  Output: {list(out.shape)}")

        assert out.shape == (B, 256, 128, 128), \
            f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output!"
        print(f"  ✓ Shape correct | ✓ No NaN")

        # Parameter overview
        counts = model.count_parameters()
        print(f"\n  Parameters per module:")
        for name, n in counts.items():
            if name != "total":
                print(f"    {name:<15} {n:>10,}")
        print(f"    {'─'*26}")
        print(f"    {'total':<15} {counts['total']:>10,}")

    print(f"\n{'='*60}")
    print("  ✅  Self-test passed.")
    print(f"{'='*60}\n")
