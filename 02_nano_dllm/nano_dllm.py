"""
nano_dllm.py — A BPE-level diffusion language model trained on FineWeb-Edu.

Phase 2 of Open-dLLM: upgrades from char-level (Phase 1) to BPE tokenization,
real web data (FineWeb-Edu streaming), cosine noise schedule, and SwiGLU MLP.

The model is a depth-parametrized transformer where a single `depth` dial
controls all dimensions: n_layer=depth, n_embd=depth*64, n_head=depth.

Architecture Overview
=====================

    Input: "The [MASK] brown [MASK]"     (partially masked BPE token sequence)
           |
           v
    +-------------------+
    | Token Embedding    |               vocab_size=32768 (BPE)           [NEW]
    | (32768 -> D)       |               D = depth * 64
    +-------------------+                depth=6 -> D=384, depth=12 -> D=768
           |
           v
    +-------------------+
    | RMSNorm            |               Functional, no learnable params
    +-------------------+
           |
           v
    +-------------------+
    | Transformer Block  |  x depth      n_layer = depth
    | +----- Attention -+|               n_head = depth, head_dim = 64
    | | Bidirectional!   ||              [DIFF 2] is_causal=False
    | | + RoPE           ||              Rotary positional embeddings
    | | + QK Norm        ||
    | +-----------------+|
    | +----- MLP -------+|
    | | SwiGLU           ||             [NEW] replaces ReLU^2
    | | Linear(D, 8D/3)  ||             gate + up projection
    | | SiLU * gate      ||
    | | Linear(8D/3, D)  ||
    | +-----------------+|
    +-------------------+
           |
           v
    +-------------------+
    | RMSNorm            |
    +-------------------+
           |
           v
    +-------------------+
    | LM Head            |               Linear(D -> 32768)
    | (D -> vocab)       |
    +-------------------+
           |
           v
    Output: logits (B, T, 32768)         Predict original token at EVERY position

    Depth parametrization                                                [NEW]
    =====================
    depth=6  -> 384-dim,  6 heads,  6 layers  (~26M params)
    depth=12 -> 768-dim, 12 heads, 12 layers  (~150M params)

    5 [DIFF] markers (same as Phase 1):
      [DIFF 1] Mask token in vocabulary
      [DIFF 2] Bidirectional attention (is_causal=False)
      [DIFF 3] Random masking of input tokens
      [DIFF 4] Loss only on masked positions
      [DIFF 5] Parallel confidence-based decoding

    6 [NEW] markers for Phase 2:
      [NEW 1] BPE tokenizer (vocab_size=32768)
      [NEW 2] Depth parametrization (single dial controls model size)
      [NEW 3] SwiGLU MLP (replaces ReLU^2)
      [NEW 4] Cosine noise schedule (replaces uniform)
      [NEW 5] FineWeb-Edu streaming data pipeline
      [NEW 6] Cosine LR decay with warmup

References:
    - MDLM (Sahoo et al., NeurIPS 2024): arxiv.org/abs/2406.07524
    - LLaDA (Nie et al., 2025): arxiv.org/abs/2502.09992
    - nanochat: github.com/karpathy/nanochat
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from datasets import load_dataset

# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="nano_dllm — BPE diffusion language model")
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--depth", type=int, default=6, help="model depth (controls all dims)")
    parser.add_argument("--prompt", type=str, default=None, help="text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="max tokens to generate")
    return parser.parse_args()

args = parse_args()

# ============================================================================
# Hyperparameters
# ============================================================================
# A single `depth` dial controls the entire model size.                  [NEW 2]
# This is the key simplification over Phase 1: instead of separately
# tuning n_layer, n_embd, n_head, we derive everything from one number.

depth = args.depth
n_layer = depth
n_embd = depth * 64     # depth=6 -> 384, depth=12 -> 768
n_head = depth
head_dim = 64            # fixed across all depths

# Training hyperparameters
batch_size = 32
block_size = 512         # context length in BPE tokens
max_iters = 20000
eval_interval = 500
eval_iters = 50
learning_rate = 1e-3
min_lr = 1e-4            # floor for cosine LR decay                    [NEW 6]
warmup_iters = 1000      # linear warmup before cosine decay            [NEW 6]
weight_decay = 0.1
grad_clip = 1.0

# ============================================================================
# Device Detection
# ============================================================================
# Same as Phase 1: prefer CUDA, then MPS (Apple Silicon), then CPU.

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)

# ============================================================================
# Tokenizer: BPE with [MASK] at id=0
# ============================================================================
# Phase 1 used a char-level tokenizer (vocab=66). Phase 2 upgrades to a
# trained BPE tokenizer (vocab=32768) for real-world text.              [NEW 1]
#
# The [MASK] token was registered as special_tokens[0] during training,
# so it always gets id=0 — same convention as Phase 1.

tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()   # 32768
mask_token_id = tokenizer.token_to_id("[MASK]")  # 0                    [DIFF 1]

def encode(text):
    """Encode text to BPE token IDs."""
    return tokenizer.encode(text).ids

def decode(ids):
    """Decode BPE token IDs back to text."""
    return tokenizer.decode(ids)

# ============================================================================
# Data Pipeline: FineWeb-Edu Streaming
# ============================================================================
#
# Phase 1 loaded Tiny Shakespeare into RAM (~1MB). Phase 2 streams from
# FineWeb-Edu (~1.5T tokens) — we never download the whole dataset.     [NEW 5]
#
# Data Flow
# =========
#
#     FineWeb-Edu (streaming)
#            |
#            v
#     Tokenize (BPE, truncate to 512)
#            |
#            v
#     Pad to block_size with [MASK]
#            |
#            v
#     Sample t ~ U[0,1] per sequence         [NEW 4] cosine schedule
#     mask_prob = 1 - cos²(t · π/2)
#            |
#            v
#     Apply random mask                       [DIFF 3]
#     (x[noise] = [MASK])
#            |
#            v
#     Return (x, targets, mask, t)
#
#
#                     Cosine Noise Schedule
#                     =====================
#     alpha_t
#     1.0 |****
#         |    ****
#         |        ***
#     0.5 |           ***
#         |              ***
#         |                 ****
#     0.0 |                     ****
#         +-----|------|------|------|
#         0    0.25   0.5   0.75   1.0    t
#
#     mask_prob = 1 - alpha_t = 1 - cos²(t · π/2)
#     (probability of masking each token)
#
# At t=0: mask_prob=0 (fully clean). At t=1: mask_prob=1 (fully masked).
# The cosine shape provides more training signal at intermediate noise
# levels compared to a linear schedule (MDLM, Section 3.2).

# Create streaming iterator for training data                           [NEW 5]
# buffer_size controls how many docs are buffered for shuffling.
# Larger = better randomness but more memory.
_train_iter = None
_val_iter = None

def _make_train_iter():
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    return iter(ds.shuffle(seed=42, buffer_size=10_000))

def _make_val_iter():
    """Lazy-init validation iterator on first val call.
    Uses .skip(100_000) to avoid overlap with training data."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    return iter(ds.skip(100_000))


def get_batch(split="train"):
    """
    Build a training batch from streamed FineWeb-Edu documents.

    Returns:
        x:       (batch_size, block_size) — masked input token IDs
        targets: (batch_size, block_size) — original unmasked token IDs
        mask:    (batch_size, block_size) — True where noise-masked AND not padding
        t:       (batch_size,) — sampled timestep per sequence (for ELBO weighting)
    """
    global _train_iter, _val_iter

    # Pick the right iterator (lazy-init val on first call)
    if split == "train":
        if _train_iter is None:
            _train_iter = _make_train_iter()
        it = _train_iter
    else:
        if _val_iter is None:
            _val_iter = _make_val_iter()
        it = _val_iter

    # Pull batch_size documents, tokenize, truncate, pad
    token_seqs = []
    padding_masks = []  # True where token is real (not padding)
    for _ in range(batch_size):
        try:
            doc = next(it)
        except StopIteration:
            # Reinitialize iterator (extremely unlikely with FineWeb-Edu's size)
            if split == "train":
                _train_iter = _make_train_iter()
                it = _train_iter
            else:
                _val_iter = _make_val_iter()
                it = _val_iter
            doc = next(it)

        ids = encode(doc["text"])

        # Truncate to block_size
        ids = ids[:block_size]
        seq_len = len(ids)

        # Pad shorter sequences with mask_token_id (right-padding)
        pad_len = block_size - seq_len
        if pad_len > 0:
            ids = ids + [mask_token_id] * pad_len

        token_seqs.append(ids)
        # True for real tokens, False for padding
        pad_mask = [True] * seq_len + [False] * pad_len
        padding_masks.append(pad_mask)

    # Stack into tensors
    targets = torch.tensor(token_seqs, dtype=torch.long)           # (B, T)
    padding = torch.tensor(padding_masks, dtype=torch.bool)        # (B, T)

    # --- Cosine noise schedule ---                                      [NEW 4]
    # Sample a timestep t ~ U[0,1] for each sequence in the batch.
    # The cosine schedule determines what fraction of tokens to mask:
    #   alpha_t = cos²(t · π/2)       — fraction of tokens kept clean
    #   mask_prob = 1 - alpha_t        — fraction of tokens masked
    t = torch.rand(batch_size)                                         # (B,)
    mask_prob = 1.0 - torch.cos(t * math.pi / 2).square()             # (B,)

    # Per-token binary noise mask: each token independently masked with prob mask_prob
    noise = torch.rand(batch_size, block_size) < mask_prob.unsqueeze(1)  # (B, T)

    # Don't noise-mask padding positions — they're already [MASK] tokens.
    # The loss mask should only include positions that were (a) noise-masked
    # AND (b) are real tokens (not padding).
    mask = noise & padding

    # [DIFF 3]: Replace noise-masked positions with [MASK] token
    x = targets.clone()
    x[mask] = mask_token_id

    # Move everything to device
    x = x.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    t = t.to(device)

    return x, targets, mask, t


# ============================================================================
# RMSNorm (Functional)
# ============================================================================
# No learnable parameters — purely functional normalization.
# This is simpler than LayerNorm and works just as well for transformers.

def norm(x):
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================
# RoPE encodes position by rotating pairs of dimensions in the query/key
# vectors. Unlike learned position embeddings, RoPE naturally handles
# relative positions and generalizes to unseen sequence lengths.

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to queries or keys."""
    assert x.ndim == 4  # (B, T, H, D) from multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


# ============================================================================
# Multi-Head Attention
# ============================================================================
#
# [DIFF 2] Bidirectional attention (is_causal=False)
#
# Causal vs Bidirectional Attention Masks
# ========================================
#
#   Causal (GPT):              Bidirectional (dLLM):
#   +--------+                 +--------+
#   |X . . . |                 |X X X X |
#   |X X . . |                 |X X X X |
#   |X X X . |                 |X X X X |
#   |X X X X |                 |X X X X |
#   +--------+                 +--------+
#     "See past only"            "See everything"
#
# Why bidirectional? A dLLM must see ALL tokens (including other mask tokens)
# to figure out what goes in each masked position. If token 5 is masked, the
# model needs context from BOTH sides to predict it — unlike GPT which only
# needs the left context.

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to queries, keys, values then reshape for multi-head
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply RoPE for relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # QK-norm: stabilizes training at scale
        q, k = norm(q), norm(k)

        # Transpose to (B, H, T, D) for attention computation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [DIFF 2]: Bidirectional attention — every token sees every other token
        # GPT would use is_causal=True here
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# ============================================================================
# SwiGLU (Gated Feed-Forward Network)                                 [NEW 3]
# ============================================================================
#
# Phase 1 (ReLU²):                    Phase 2 (SwiGLU):
#   x ──→ Linear(d, 4d) ──→ ReLU²     x ──→ w1: Linear(d, h) ──→ SiLU ──┐
#        ──→ Linear(4d, d) ──→ out          w2: Linear(d, h) ──────────→ * ──→ w3: Linear(h, d) ──→ out
#                                       where h = round_up(8/3 * d, 256)
#
# SwiGLU uses a gating mechanism: one projection (w1) passes through SiLU
# activation, then element-wise multiplies with another projection (w2).
# This gated structure gives the model more expressiveness per parameter
# than a plain ReLU/GELU MLP. Used in LLaMA, Gemma, and most modern LLMs.
#
# The hidden_dim is rounded up to a multiple of 256 for GPU memory alignment.

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = ((int(8 / 3 * n_embd) + 255) // 256) * 256
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)  # up
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)  # down

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============================================================================
# Transformer Block
# ============================================================================
# Pre-norm architecture: normalize BEFORE attention and MLP, not after.
# Residual connections around both sub-layers.

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = SwiGLU()  # [NEW 3] SwiGLU replaces ReLU² MLP

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)  # Attention with pre-norm
        x = x + self.mlp(norm(x))            # SwiGLU with pre-norm
        return x


# ============================================================================
# Model
# ============================================================================
#
# The full diffusion language model. Architecturally, this is a standard
# transformer (like GPT) with two key differences:
#   1. Bidirectional attention [DIFF 2]
#   2. Loss computed only on masked positions [DIFF 4]
#
# Phase 2 additionally upgrades the loss to ELBO-weighted [NEW 3]:
#   L = mean_i( (1/t_i) * CE_masked_i )
#
# This comes from the continuous-time ELBO in MDLM (NeurIPS 2024):
#   L = integral_0^1 (1/t) * E[CE on masked positions] dt
#
# ELBO Loss Weighting
# ===================
#
#   Sample 1: t=0.1 (10% masked, hard) → weight = 1/0.1 = 10.0
#   Sample 2: t=0.5 (50% masked, medium) → weight = 1/0.5 = 2.0
#   Sample 3: t=0.9 (90% masked, easy) → weight = 1/0.9 = 1.1
#
#   Low t = fewer masks = harder predictions = MORE weight
#   High t = many masks = easier predictions = LESS weight

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings: each of the 32768 BPE tokens gets an n_embd-dim vector
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Precompute rotary embeddings for up to 2x block_size positions
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output projection: predict which of the vocab_size tokens belongs at each position
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Normal(0, 0.02) initialization for Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        """Precompute cos/sin tables for RoPE."""
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Add batch and head dims: (1, T, 1, D//2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, mask=None, t=None):
        """
        Forward pass.

        Args:
            idx: input token IDs, shape (B, T)
            targets: ground-truth token IDs, shape (B, T). None during generation.
            mask: boolean mask, shape (B, T). True where tokens were masked.
            t: timestep per sequence, shape (B,). Used for ELBO weighting.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.size()

        # Embed tokens and normalize
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Slice rotary embeddings to current sequence length
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # [DIFF 4]: Loss on masked positions only
            # [NEW 3]: ELBO weighting by 1/t
            #
            # Phase 1: loss = CE_masked.sum() / mask.sum()  (unweighted)
            # Phase 2: loss = mean_i( (1/t_i) * CE_masked_i )  (ELBO weighted)
            #
            # Why 1/t? The continuous-time ELBO from MDLM (NeurIPS 2024):
            #   L = integral_0^1 (1/t) * E[CE on masked positions] dt
            # Low t means less masking → harder predictions → higher weight.

            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1), reduction="none"
            )
            per_token_loss = per_token_loss.view(B, T)

            if mask is not None and t is not None:
                # Per-sample masked loss
                masked_loss = (per_token_loss * mask.float()).sum(dim=1)  # (B,)
                mask_count = mask.float().sum(dim=1).clamp(min=1)         # (B,)
                per_sample_loss = masked_loss / mask_count                # (B,)

                # ELBO weight: 1/t, clamped for stability
                elbo_weight = 1.0 / t.clamp(min=1e-4)  # (B,)

                loss = (per_sample_loss * elbo_weight).mean()
            elif mask is not None:
                # Fallback: masked loss without ELBO weighting
                mask_flat = mask.view(-1).float()
                loss = (per_token_loss.view(-1) * mask_flat).sum() / mask_flat.sum()
            else:
                loss = per_token_loss.mean()

        return logits, loss


# ============================================================================
# Generation: Confidence-Based Parallel Decoding                       [DIFF 5]
# ============================================================================
#
# Same algorithm as Phase 1, adapted for BPE tokenization.
# Instead of characters, we now decode BPE subwords — the fundamental
# difference is that each decoded position represents a subword piece,
# not a single character. A single "word" may span multiple positions.
#
# Generation Process (Iterative Unmasking)
# =========================================
#
#   Start:  [prompt tokens] [_] [_] [_] [_] [_] [_] [_] [_]
#           (each [_] is a masked BPE subword position)
#
#   Step 1: Forward pass -> logits at all positions
#           Softmax/temp -> probabilities
#           Zero out mask_token_id prob (model should never "generate" [MASK])
#           Top-k probs -> confidence = sum(top_k_probs)
#
#           Position:  prompt  p+1  p+2  p+3  p+4  p+5  p+6  p+7  p+8
#           Confidence:  --   0.92 0.97 0.45 0.98 0.88 0.71 0.96 0.55
#           Threshold:                        0.95
#           Decode?:     --    no   YES   no   YES  no   no   YES  no
#
#           After:  [prompt] [_] [▁the] [_] [▁of] [_] [_] [▁and] [_]
#                                  ^          ^                ^
#                             3 BPE tokens decoded in one step!
#
#   Step 2: Re-run model with updated context...
#           Now more positions become confident (they see neighbors)
#
#           After:  [prompt] [▁mean] [▁the] [_] [▁of] [_] [▁is] [▁and] [▁it]
#
#   Step 3: ...continue until all masks are filled...
#
#   Result: [prompt] [▁mean] [▁the] [▁world] [▁of] [▁life] [▁is] [▁and] [▁it]
#
# If NO position exceeds the threshold, we force-decode the SINGLE most
# confident position to guarantee progress.

@torch.no_grad()
def generate(model, max_new_tokens=512, prompt=None, temp=0.8,
             confidence_threshold=0.95, top_k=5):
    """
    Generate text using confidence-based parallel decoding with BPE.

    Args:
        model: the diffusion language model
        max_new_tokens: how many new BPE tokens to generate
        prompt: text string to use as context (None = start from all masks)
        temp: sampling temperature (lower = more deterministic)
        confidence_threshold: minimum confidence to unmask a position
        top_k: number of top candidates to consider for sampling
    """
    was_training = model.training
    model.eval()  # noqa: B018

    # Encode prompt to BPE token IDs
    if prompt is not None:
        prompt_ids = encode(prompt)
    else:
        prompt_ids = []

    prompt_len = len(prompt_ids)
    all_tokens = list(prompt_ids)
    total_steps = 0
    t_start = time.time()

    # Generate one block at a time (for sequences longer than block_size)
    while len(all_tokens) - prompt_len < max_new_tokens:
        # How many tokens to generate in this block
        block_len = min(
            block_size - prompt_len,
            prompt_len + max_new_tokens - len(all_tokens),
        )

        # Initialize: prompt + all masks within block_size window
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        if prompt_len > 0:
            x[0, :prompt_len] = torch.tensor(
                all_tokens[-prompt_len:], dtype=torch.long, device=device
            )

        # Track which positions still need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        # Iterative unmasking loop
        while masked.any():
            total_steps += 1

            # 1. Forward pass to get predictions
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)

            # Zero out mask_token_id probability — the model should never
            # "generate" a [MASK] token during decoding
            probs[:, :, mask_token_id] = 0.0

            # 2. Get top-k candidates and compute confidence
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)  # Confidence = P(top_k)

            # 3. Decide which masked positions to decode
            decode_mask = (confidences >= confidence_threshold) & masked

            # 4. If nothing exceeds threshold, force-decode the best one
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            # 5. Sample from normalized top-k distribution
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(
                top_k_probs_norm.view(-1, top_k), 1
            ).view(1, block_size)
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            # 6. Update: place sampled tokens, clear mask at decoded positions
            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        # Extract generated tokens from this block
        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    # Decode full sequence with BPE tokenizer
    tokens_generated = len(all_tokens) - prompt_len
    elapsed = time.time() - t_start
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float("inf")
    avg_per_step = tokens_generated / total_steps if total_steps > 0 else 0

    print(f"Generation stats: {total_steps} steps, {tokens_generated} tokens, "
          f"{avg_per_step:.1f} avg decoded/step, {tok_per_sec:.1f} tok/s")

    if was_training:
        model.train()
    return decode(all_tokens)


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def estimate_loss(model):
    """Estimate train and val loss by averaging over eval_iters batches."""
    out = {}
    was_training = model.training
    model.eval()  # noqa: B018
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, targets, mask, t = get_batch(split)
            _, loss = model(x, targets, mask, t)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


# ============================================================================
# Learning Rate Schedule: Cosine Decay with Warmup                     [NEW 6]
# ============================================================================
#
#     Learning Rate Schedule
#     ======================
#     lr
#     1e-3 |     ********
#          |    *         *****
#          |   *                ****
#          |  *                     ***
#     1e-4 | *                         **
#          +--|---------|---------|------->
#          0  warmup    mid       max_iters
#
# Linear warmup from 0 -> learning_rate over warmup_iters steps,
# then cosine decay from learning_rate -> min_lr over the remaining steps.

def get_lr(step):
    """Return the learning rate for a given training step."""
    # Linear warmup
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    # Cosine decay
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================================
# Main: Training and Generation
# ============================================================================
#
# Training Loop
# =============
#
#   FineWeb-Edu (streaming)
#        |
#        v
#   get_batch() -> (x, targets, mask, t)
#        |
#        v
#   model(x, targets, mask, t) -> loss
#        |              [DIFF 4] masked positions
#        v              [NEW 3] ELBO weighted by 1/t
#   loss.backward()
#        |
#        v
#   clip_grad_norm + optimizer.step()
#        |
#        v
#   repeat max_iters times
#
# The training loop follows a standard recipe:
#   1. Sample a batch of tokenized documents from FineWeb-Edu
#   2. Apply cosine noise schedule: sample t, mask tokens with probability 1-cos^2(t*pi/2)
#   3. Forward pass through the model -> ELBO-weighted cross-entropy loss
#   4. Backward pass + gradient clipping + optimizer step
#   5. Periodically evaluate on held-out data and generate a sample
#
# Weight decay is applied only to 2D+ parameters (weight matrices), not to
# biases or normalization parameters (1D). This is standard practice: decaying
# biases and norms hurts training stability with no regularization benefit.

if __name__ == "__main__":

    # --- 1. Print configuration ---
    print("=" * 60)
    print("nano_dllm — BPE diffusion language model")
    print("=" * 60)
    print(f"  depth      = {depth}")
    print(f"  n_layer    = {n_layer}")
    print(f"  n_embd     = {n_embd}")
    print(f"  n_head     = {n_head}")
    print(f"  block_size = {block_size}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  device     = {device}")
    print("=" * 60)

    # --- 2. Set up weights path ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", f"nano_dllm_d{depth}.pt")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # --- 3. Instantiate model ---
    model = Model().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{param_count / 1e6:.2f}M parameters")

    # --- 4. Load or train ---
    if os.path.exists(weights_path) and not args.train:
        # Load pre-trained weights
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {weights_path}")

    elif args.train:
        # Train from scratch

        # Separate parameters into decay (2D+ weight matrices) and no-decay
        # (biases, norm params). Weight decay on biases/norms hurts stability.
        decay_params = [p for p in model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

        # Training loop
        t0 = time.time()
        for step in range(max_iters):
            # Update learning rate per step (warmup + cosine decay)
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate and generate a sample periodically
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                      f"val loss {losses['val']:.4f} | lr {lr:.6f}")
                if step > 0:
                    sample = generate(model, max_new_tokens=100, temp=0.8, top_k=5)
                    print(f"--- sample ---\n{sample[:300]}\n--- end sample ---")

            # Forward pass: get batch -> model -> loss
            x, targets, mask, t = get_batch("train")
            logits, loss = model(x, targets, mask, t)

            # Backward pass
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step + zero gradients (set_to_none=True is faster than zeroing)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Progress indicator every 100 steps
            if step % 100 == 0 and step > 0:
                dt = time.time() - t0
                tokens_per_sec = (step * batch_size * block_size) / dt
                print(f"  step {step:5d} | loss {loss.item():.4f} | {tokens_per_sec:.0f} tok/s")

        # Save trained weights
        torch.save(model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")

    else:
        print(f"No weights found at {weights_path}")
        print("Run with --train to train from scratch.")
        sys.exit(1)

    # --- 5. Generate text ---
    print("\n" + "=" * 60)
    print("Generating text...")
    print("=" * 60)
    prompt_text = args.prompt if args.prompt else None
    t_gen = time.time()
    text = generate(model, max_new_tokens=args.max_tokens, prompt=prompt_text, temp=0.8, top_k=5)
    gen_time = time.time() - t_gen
    print(text)
    print(f"\nGeneration: {len(encode(text))} tokens in {gen_time:.2f}s")
