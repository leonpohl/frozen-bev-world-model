# SPDX-FileCopyrightText: 2026 Vincent Mann
#
# SPDX-License-Identifier: MIT
"""
transformer.py — BEV World Model: Transformer core
============================================================

Implements the Transformer stack that is the heart of the world model.

ARCHITECTURE OVERVIEW:
    Input:  [B, seq_len, d_model]
    Output: [B, seq_len, d_model]

    seq_len is PHASE-AGNOSTIC:
        Phase 1 (frame level):  seq_len =    3  (one token per frame)
        Phase 2 (cell level):   seq_len = 3072  (one token per BEV cell)

    The Transformer does not notice the difference — it simply processes
    a sequence of vectors, no matter how long it is.

WHY PRE-LAYERNORM?
    There are two variants:
        Post-LN (original Transformer paper):
            x = LayerNorm(x + Attention(x))   ← less stable in practice
        Pre-LN (modern standard, e.g. GPT-2):
            x = x + Attention(LayerNorm(x))   ← more stable, converges better

    Pre-LN normalizes BEFORE computing. That means the attention
    always works with normalized values — no exploding gradients,
    no dead neurons. For our training setup (6 weeks,
    limited VRAM) stability matters more than the last percent of performance.

WHY RESIDUAL CONNECTIONS (x + ...)?
    The network learns only what it must CHANGE, not everything from scratch.
    Mathematically: gradients flow directly back through the addition
    without having to pass through attention or FFN → no vanishing gradient.
    Practically: even if attention is "wrong", the result is
    at least as good as the direct pass-through of x.

WHY FFN AFTER THE ATTENTION?
    Attention mixes information between tokens ("what do the others say?").
    The FFN transforms each token individually, nonlinearly ("what does that mean for me?").
    Both steps are necessary — attention alone cannot learn complex
    functions per token, FFN alone cannot model interaction between
    tokens.

FILE STRUCTURE:
    model/
    ├── config.py — ModelConfig dataclass
    ├── downsampling.py — DownsamplingModule
    ├── embedding.py — FrameLevelEmbedding, CellLevelEmbedding
    └── transformer.py — TransformerBlock, TransformerEncoder ← this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# TransformerBlock — a single block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer block in the Pre-LayerNorm variant.

    COMPLETE FORWARD EQUATION:
        # Attention branch:
        x = x + dropout(Attention(LayerNorm(x)))
        # FFN branch:
        x = x + dropout(FFN(LayerNorm(x)))

    STRUCTURE IN DETAIL:

    1. SELF-ATTENTION (nn.MultiheadAttention):
       - "Self" means: query, key and value all come from x itself.
         Each token asks itself: "what is relevant to me in the other tokens?"
       - Query Q = x @ W_Q   [B, seq_len, d_model]
         Key   K = x @ W_K   [B, seq_len, d_model]
         Value V = x @ W_V   [B, seq_len, d_model]
       - Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_head)) @ V
         The @ K.T gives the "similarity" between each token pair.
         Division by sqrt(d_head) prevents the values from becoming too large.
       - NO MASKING: all tokens may see all others.
         That differs from language models (there t may not see t+1).
         For us that is fine — we predict the FUTURE frame t+1,
         but all input frames (t-2, t-1, t) are equal.

       - n_heads=8, d_head=32: instead of one large attention there are 8 parallel
         "heads", each working in d_head=32 dimensions. Different heads
         can learn different relation types (e.g. spatially near cells
         vs. temporally similar frames).

    2. FEED-FORWARD NETWORK (FFN):
       - Linear(256 → 1024): expansion — more capacity for nonlinear transformation
       - GELU activation: "Gaussian Error Linear Unit", softer than ReLU,
         the modern standard since GPT-2/BERT. No hard cutoff at 0.
       - Dropout(0.1): regularization — 10% of the neurons are randomly
         switched off during training. Prevents overfitting.
       - Linear(1024 → 256): back to the d_model dimension

    3. LAYERNORM (Pre-LN):
       - Normalizes the activations to mean≈0, std≈1
       - elementwise_affine=True (default): learnable γ and β parameters
         that the network uses to "undo" the normalization when necessary
       - Operates over the last dimension (d_model=256), not over the batch

    Args:
        config: ModelConfig with d_model, n_heads, d_ff, dropout

    Input:  [B, seq_len, d_model]  (seq_len = 3 or 3072)
    Output: [B, seq_len, d_model]  (same shape — the Transformer is shape-preserving)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # --- Pre-LayerNorm before attention ---
        # normalized_shape=d_model: normalizes only over the feature dimension,
        # not over batch or sequence. Each token is normalized independently.
        self.norm1 = nn.LayerNorm(config.d_model)

        # --- Multi-head self-attention ---
        # embed_dim=d_model: input and output dimension
        # num_heads=n_heads: number of parallel attention heads
        # dropout=dropout: dropout on the attention weights (not output!)
        # batch_first=True: expects [B, seq_len, d_model] instead of [seq_len, B, d_model]
        #                   That is important — the PyTorch default used to be batch last!
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --- Dropout on attention output ---
        # Separate dropout after the attention output (in addition to the attention-internal one).
        # This follows the architecture from the original "Attention is All You Need" paper.
        self.dropout1 = nn.Dropout(config.dropout)

        # --- Pre-LayerNorm before FFN ---
        self.norm2 = nn.LayerNorm(config.d_model)

        # --- Feed-forward network ---
        # Sequential combines several layers into one module.
        # The order: Linear → GELU → Dropout → Linear
        # Why GELU instead of ReLU?
        #   ReLU:  max(0, x)  — hard cutoff, gradient=0 for x<0 (dead neurons)
        #   GELU:  x * Φ(x)  — soft transition, no hard 0 clipping
        #   In practice: GELU converges more reliably for Transformer architectures
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),    # 256 → 1024 (expansion)
            nn.GELU(),                                  # activation
            nn.Dropout(config.dropout),                 # regularization
            nn.Linear(config.d_ff, config.d_model),    # 1024 → 256 (back-transformation)
        )

        # --- Dropout on FFN output ---
        self.dropout2 = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-LayerNorm Transformer block forward pass.

        Args:
            x: [B, seq_len, d_model]  — token sequence

        Returns:
            x: [B, seq_len, d_model]  — transformed token sequence (same shape!)

        STEP BY STEP:

        Step 1 — attention branch:
            normed = LayerNorm(x)           # normalize
            attn_out, _ = Attention(normed, normed, normed)  # self-attention
            x = x + Dropout(attn_out)       # residual + dropout

        Step 2 — FFN branch:
            normed = LayerNorm(x)           # normalize (now with attention info)
            ffn_out = FFN(normed)           # feed-forward
            x = x + Dropout(ffn_out)        # residual + dropout

        The same x passes through both branches and is each time added with the
        result of the branch — these are the residual connections.
        """

        # ---------------------------------------------------------------
        # ATTENTION BRANCH
        # ---------------------------------------------------------------

        # Step 1a: LayerNorm before attention (pre-LN)
        normed = self.norm1(x)              # [B, seq_len, d_model]

        # Step 1b: multi-head self-attention
        # query=normed, key=normed, value=normed — "self" because all three are the same
        # need_weights=False: we do NOT need the attention weights for the
        #   training forward pass — that saves memory and is somewhat faster.
        #   For visualization in the unit tests we fetch them separately.
        attn_out, _ = self.attn(
            query=normed,
            key=normed,
            value=normed,
            need_weights=False,
        )                                   # [B, seq_len, d_model]

        # Step 1c: dropout + residual addition
        x = x + self.dropout1(attn_out)    # [B, seq_len, d_model]

        # ---------------------------------------------------------------
        # FFN BRANCH
        # ---------------------------------------------------------------

        # Step 2a: LayerNorm before FFN (pre-LN)
        normed = self.norm2(x)              # [B, seq_len, d_model]

        # Step 2b: feed-forward network
        ffn_out = self.ffn(normed)          # [B, seq_len, d_model]

        # Step 2c: dropout + residual addition
        x = x + self.dropout2(ffn_out)     # [B, seq_len, d_model]

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function for visualization and unit tests.
        Returns the attention weights without influencing the normal
        forward pass.

        Called ONLY for debugging/visualization — not in training.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            weights: [B, n_heads, seq_len, seq_len]
                     weights[b, h, i, j] = how much token i attends to token j
                     (head h, batch element b)
        """
        self.eval()  # disable dropout for consistent attention weights
        with torch.no_grad():
            normed = self.norm1(x)
            _, weights = self.attn(
                query=normed,
                key=normed,
                value=normed,
                need_weights=True,
                average_attn_weights=False,  # weights per head, not averaged
            )
        return weights  # [B, n_heads, seq_len, seq_len]


# ---------------------------------------------------------------------------
# TransformerEncoder — stack of N blocks
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack of N TransformerBlock instances.

    This is the actual "depth" of the model. N=4 blocks means:
    the sequence is passed 4 times through the attention+FFN cycle.

    WHY MULTIPLE BLOCKS?
        Each block can only model "one layer" of relations.
        Block 1 might learn: "cell (x,y) at t-1 resembles (x,y) at t"
        Block 2 learns: "if (x,y) moved, (x+1,y) moves too"
        Block 3 learns: "these movement patterns indicate a moving car"
        Block 4 learns: "cars keep driving straight when there is no curve"
        → hierarchical, abstract reasoning across multiple levels

    WHY nn.ModuleList INSTEAD OF A PYTHON LIST?
        Python list: PyTorch knows nothing about the modules inside it.
            → parameters are not registered → not trained!
        nn.ModuleList: PyTorch registers all contained modules.
            → parameters appear in model.parameters() → they are trained.

    Args:
        config: ModelConfig with n_layers and all block parameters

    Input:  [B, seq_len, d_model]
    Output: [B, seq_len, d_model]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Create N identically configured blocks.
        # Each block has its OWN parameters — they share the
        # architecture, but not the weights. This is important!
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes x sequentially through all N Transformer blocks.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            x: [B, seq_len, d_model]  — after N attention+FFN cycles

        The shape does NOT change through the stack — that is the beauty
        of the Transformer architecture. Whether 3 or 3072 tokens: the code
        is identical, only runtime and VRAM demand change
        (quadratically with seq_len due to the attention matrix).
        """
        for block in self.blocks:
            x = block(x)       # [B, seq_len, d_model] → [B, seq_len, d_model]
        return x

    def get_all_attention_weights(self, x: torch.Tensor):
        """
        Returns the attention weights of all blocks.
        For ablation studies and thesis visualizations.

        Returns:
            List of [B, n_heads, seq_len, seq_len] — one per block
        """
        all_weights = []
        for block in self.blocks:
            weights = block.get_attention_weights(x)
            all_weights.append(weights)
            # pass x through the block (without storing the weights)
            with torch.no_grad():
                x = block(x)
        return all_weights
