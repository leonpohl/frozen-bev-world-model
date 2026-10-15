# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
config.py — BEV World Model: central model configuration
=====================================================================

Defines ModelConfig as a Python dataclass.

What is a dataclass?
    A dataclass is a Python class that mainly serves to
    hold data — similar to a struct in C or a NamedTuple,
    but with type annotations, default values, and an automatically
    generated __repr__. You write less boilerplate than with
    a normal class.

Why a central config class?
    Without a config, all hyperparameters (d_model=256, n_heads=8, ...)
    would be passed through all modules as individual arguments.
    That means: one change must be made in 5 places.
    With ModelConfig there is exactly one place — all modules read
    their parameters from there. This is especially important when we
    switch between Phase 1 (frame level) and Phase 2 (cell level),
    because then only `phase` has to be changed.

Usage:
    from model.config import ModelConfig

    # Default — Phase 1 (frame level, 3 tokens)
    cfg = ModelConfig()

    # Phase 2 (cell level, 3072 tokens)
    cfg = ModelConfig(phase="cell", n_layers=4)

    # Ablation: fewer layers
    cfg = ModelConfig(phase="cell", n_layers=2)

    # Access to parameters
    print(cfg.d_model)    # 256
    print(cfg.n_heads)    # 8
    print(cfg)            # pretty __repr__ from the dataclass decorator
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Central configuration for the BEV world model.

    EMBEDDING PARAMETER:
        d_model:     dimension of all token vectors in the Transformer.
                     256 is the natural value because BEVFusion latents
                     have 256 channels → no dimensional break.

        n_frames:    number of consecutive BEV frames as input.
                     Default 3 = (t-2, t-1, t) → predict (t+1).
                     In the ablation study we also test 2 and 4.

        grid_size:   spatial resolution after downsampling.
                     128×128 → AvgPool(stride=4) → 32×32.
                     32 is the standard. Only needed in Phase 2
                     (CellLevelEmbedding) to dimension PE_x and PE_y.

    TRANSFORMER PARAMETERS:
        n_heads:     number of attention heads in multi-head self-attention.
                     8 heads at d_model=256 → d_head = 256/8 = 32
                     per head. More heads = more "viewpoints" on
                     the sequence, but also more parameters.

        d_ff:        dimension of the inner layer in the feed-forward network
                     (FFN) of each Transformer block.
                     1024 = 4 × d_model — that is the standard factor
                     from the original Transformer paper (Vaswani 2017).

        n_layers:    number of stacked Transformer blocks (depth).
                     Default 4. In the ablation study we test 2 and 6.
                     More layers = more sequential processing steps,
                     higher capacity, but also more VRAM.

        dropout:     dropout rate for regularization.
                     0.1 = 10% of activations are randomly set to 0
                     during training. Prevents overfitting.
                     Applied to attention output and FFN output.

    PHASE CONTROL:
        phase:       "frame" = Phase 1 (3 tokens, fast, for prototyping)
                     "cell"  = Phase 2 (3072 tokens, full spatial
                                        attention — the scientific core)

                     This single string value controls which embedding module
                     and which output head is instantiated in BEVWorldModel.

    DERIVED VALUES (properties):
        n_tokens:    total number of tokens after embedding.
                     Phase 1: 3 (one token per frame)
                     Phase 2: n_frames × grid_size² = 3 × 32² = 3072
                     Important for VRAM estimation (attention is O(n²)).

        d_head:      dimension per attention head = d_model / n_heads.
                     Must be an integer — checked in __post_init__.
    """

    # --- Embedding ---
    d_model:    int   = 256     # token dimension (= BEVFusion channel count)
    n_frames:   int   = 3       # number of input frames
    grid_size:  int   = 32      # spatial resolution after downsampling (32×32)

    # --- Transformer ---
    n_heads:    int   = 8       # attention heads
    d_ff:       int   = 1024    # FFN inner dimension (= 4 × d_model)
    n_layers:   int   = 4       # number of Transformer blocks
    dropout:    float = 0.1     # dropout rate

    # --- Phase control ---
    phase:      str   = "frame" # "frame" (Phase 1) or "cell" (Phase 2)

    # --- Ego-motion conditioning (FiLM) ---
    ego_cond_mode: str = "off"  # off | state | action | both (default off = compatible)
    output_mode: str = "gated"  # gated (default, as before) | residual
                                # (pred = last frame + head output, no gate)
    vae_mode:   str = "off"     # 18/B1: off (default, no modules) | cvae
                                # (prior/posterior/z-FiLM, zero-init = off start)
    vae_z_dim:  int = 32        # 18/B1: dimension of the latent variable z
    flow_head:  str = "off"     # 18/B2: off (default, no module) | flow
                                # (FM residual head on the frozen backbone; forward
                                # stays unchanged, sampling via flow.sample)
    flow_dim:   int = 224       # 18/B2: channel width v_theta (~6.5M = backbone parity)
    flow_layers: int = 8        # 18/B2: number of residual conv blocks

    # ------------------------------------------------------------------
    # Validation after __init__
    # ------------------------------------------------------------------

    def __post_init__(self):
        """
        Called automatically after the generated __init__.

        Here we validate constraints that the type annotator cannot
        express — e.g. that d_model is divisible by n_heads.
        Early error messages are better than mysterious CUDA errors
        during training.
        """
        # d_model must be divisible by n_heads (otherwise d_head is not an int)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
                f"d_head = {self.d_model}/{self.n_heads} "
                f"= {self.d_model / self.n_heads} is not an integer."
            )

        # phase must be "frame" or "cell"
        if self.phase not in ("frame", "cell"):
            raise ValueError(
                f"phase must be 'frame' or 'cell', not '{self.phase}'."
            )

        # n_frames at least 2 (1 would not be "temporal reasoning")
        if self.n_frames < 2:
            raise ValueError(
                f"n_frames must be at least 2, is {self.n_frames}."
            )

        # Dropout between 0 and 1
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), is {self.dropout}."
            )

    # ------------------------------------------------------------------
    # Derived properties (read-only, not part of dataclass state)
    # ------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        """
        Dimension per attention head.

        In multi-head attention, d_model is split across n_heads:
        each head works in a d_head-dimensional subspace.
        Guaranteed integer by __post_init__.
        """
        return self.d_model // self.n_heads

    @property
    def n_tokens(self) -> int:
        """
        Total number of tokens after embedding — phase-dependent.

        Phase 1 (frame): 3 tokens — one per frame.
            → attention matrix: 3×3 (trivial, hardly any VRAM)

        Phase 2 (cell): n_frames × grid_size² tokens.
            → at default: 3 × 32 × 32 = 3072 tokens
            → attention matrix: 3072×3072 (~9M values per batch element!)
            → hence batch size 4 instead of 16 in Phase 2.

        This property is used in unit tests and VRAM estimates.
        """
        if self.phase == "frame":
            return self.n_frames
        else:  # "cell"
            return self.n_frames * self.grid_size * self.grid_size

    @property
    def n_spatial_tokens(self) -> int:
        """
        Number of spatial tokens per frame (only relevant for Phase 2).
        grid_size² = 32² = 1024 — needed in CellLevelOutputHead
        to cut out the tokens of the last frame.
        """
        return self.grid_size * self.grid_size


# ---------------------------------------------------------------------------
# Short self-test when the file is executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ModelConfig self-test ===\n")

    # Default config (Phase 1)
    cfg1 = ModelConfig()
    print(f"Phase 1 Config:\n{cfg1}")
    print(f"  d_head   = {cfg1.d_head}")
    print(f"  n_tokens = {cfg1.n_tokens}")
    print()

    # Phase 2 Config
    cfg2 = ModelConfig(phase="cell")
    print(f"Phase 2 Config:\n{cfg2}")
    print(f"  d_head   = {cfg2.d_head}")
    print(f"  n_tokens = {cfg2.n_tokens}  (= {cfg2.n_frames} × {cfg2.grid_size}² )")
    print()

    # Ablation: fewer layers, more frames
    cfg3 = ModelConfig(phase="cell", n_layers=2, n_frames=4)
    print(f"Ablation Config (n_layers=2, n_frames=4):\n{cfg3}")
    print(f"  n_tokens = {cfg3.n_tokens}  (= {cfg3.n_frames} × {cfg3.grid_size}² )")
    print()

    # Error test: d_model not divisible by n_heads
    print("Error test (d_model=100, n_heads=8 → should ValueError):")
    try:
        bad = ModelConfig(d_model=100, n_heads=8)
    except ValueError as e:
        print(f"  ✓ ValueError correctly caught: {e}")
    print()

    # Error test: invalid phase
    print("Error test (phase='invalid' → should ValueError):")
    try:
        bad = ModelConfig(phase="invalid")
    except ValueError as e:
        print(f"  ✓ ValueError correctly caught: {e}")
