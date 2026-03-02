"""
block_dllm.py — A block diffusion language model trained on FineWeb-Edu.

Phase 3 of Open-dLLM: upgrades from full-sequence diffusion (Phase 2) to
block diffusion with a staircase attention mask and KV caching. Uses Qwen 3
tokenizer (~152K vocab) and tied embeddings for a ~200M param budget.

The key change from Phase 2: replace is_causal=False (fully bidirectional)
with an explicit staircase attention mask — bidirectional within each block,
causal across blocks. Everything else stays identical: RMSNorm, SwiGLU,
RoPE, cosine noise schedule, ELBO-weighted loss, confidence-based decoding.

Architecture Overview
=====================

    Input: "The [MASK] brown [MASK]"     (partially masked token sequence)
           |
           v
    +-------------------+
    | Token Embedding    |               vocab_size=151670 (Qwen 3 + [MASK])  [DIFF]
    | (151670 -> D)      |               D = depth * 64
    +-------------------+                depth=10 -> D=640
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
    | | Staircase Mask   ||              [DIFF] replaces is_causal=False
    | | + RoPE           ||              Rotary positional embeddings
    | | + QK Norm        ||
    | +-----------------+|
    | +----- MLP -------+|
    | | SwiGLU           ||
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
    | LM Head (tied)     |               Shares weights with embedding     [NEW]
    | (D -> vocab)       |
    +-------------------+
           |
           v
    Output: logits (B, T, 151670)        Predict original token at EVERY position

    Block Diffusion                                                         [NEW]
    ===============
    Sequence is divided into blocks of size --block-size (e.g. 4 tokens).
    During training, each block gets an independent noise level.
    The staircase mask ensures blocks attend bidirectionally within
    themselves but causally across blocks — bridging AR and diffusion.

    --block-size controls the AR <-> diffusion spectrum:
      block_size=1   -> fully autoregressive (1 token per diffusion step)
      block_size=4   -> 4 tokens denoised in parallel per block (default)
      block_size=512 -> full-sequence diffusion (equivalent to Phase 2)

    Depth parametrization (same as Phase 2)
    =======================================
    depth=6  -> 384-dim,  6 heads,  6 layers
    depth=10 -> 640-dim, 10 heads, 10 layers
    depth=12 -> 768-dim, 12 heads, 12 layers

    [DIFF] markers (changes from Phase 2):
      [DIFF 1] Qwen 3 tokenizer (vocab_size=151670, replaces custom BPE)
      [DIFF 2] Staircase attention mask (replaces is_causal=False)
      [DIFF 3] Per-block noise levels (replaces per-sequence)
      [DIFF 4] Tied embeddings (LM head shares embedding weights)

    [NEW] markers for Phase 3:
      [NEW 1] Block diffusion with configurable block size
      [NEW 2] KV caching across blocks for inference
      [NEW 3] Block-by-block generation with EOS termination

References:
    - BD3-LMs (Arriola et al., ICLR 2025 Oral): arxiv.org/abs/2503.09573
    - Mercury (Inception Labs, 2025): arxiv.org/abs/2506.17298
    - MDLM (Sahoo et al., NeurIPS 2024): arxiv.org/abs/2406.07524
    - LLaDA (Nie et al., 2025): arxiv.org/abs/2502.09992
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="block_dllm — block diffusion language model")
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--depth", type=int, default=10, help="model depth (controls all dims)")
    parser.add_argument("--prompt", type=str, default=None, help="text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="max tokens to generate")
    parser.add_argument("--block-size", type=int, default=4, choices=[0, 1, 2, 4, 8, 16],
                        help="block size for diffusion (0=full sequence)")          # [NEW 1]
    parser.add_argument("--denoise-steps", type=int, default=10,
                        help="denoising steps per block during generation")         # [NEW 1]
    return parser.parse_args()

args = parse_args()

# ============================================================================
# Hyperparameters
# ============================================================================
# Depth parametrization — same as Phase 2.

depth = args.depth
n_layer = depth
n_embd = depth * 64     # depth=10 -> 640
n_head = depth
head_dim = 64            # fixed across all depths

# Block size                                                                 [NEW 1]
block_size_seq = 512     # context length in tokens (same as Phase 2)
block_size_blk = block_size_seq if args.block_size == 0 else args.block_size

# Training hyperparameters
batch_size = 32
max_iters = 20000
eval_interval = 500
eval_iters = 50
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 1000
weight_decay = 0.1
grad_clip = 1.0

# ============================================================================
# Device Detection
# ============================================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)

# ============================================================================
# Tokenizer: Qwen 3 with [MASK]
# ============================================================================
# Phase 2 used a custom BPE tokenizer (vocab=32768). Phase 3 upgrades to
# Qwen 3 (~152K vocab) with tied embeddings to stay within param budget. [DIFF 1]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
vocab_size = len(tokenizer)  # 151670 (151669 + 1 for [MASK])
mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
eos_token_id = tokenizer.eos_token_id

def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)

def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


# ============================================================================
# Data Pipeline: FineWeb-Edu Streaming                                  [DIFF]
# ============================================================================
#
# Ported from Phase 2 with per-block timestep sampling.
#
# Data Flow
# =========
#
#     FineWeb-Edu (streaming)
#            |
#            v
#     Tokenize (Qwen 3, truncate to block_size_seq=512)                 [DIFF]
#            |
#            v
#     Pad to block_size_seq with [MASK]
#            |
#            v
#     Sample t ~ U[0,1] PER BLOCK (not per sequence)                   [DIFF 3]
#     Repeat to per-token shape: (B, num_blocks) -> (B, block_size_seq)
#     mask_prob = 1 - cos²(t · π/2)
#            |
#            v
#     Apply random mask per token
#     (x[noise] = [MASK])
#            |
#            v
#     Concatenate: x_input = [x_t || x_0]                              [NEW]
#            |
#            v
#     Return (x_input, targets, mask, t, attn_mask)
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
#
# [DIFF 3] Per-block timesteps: each block gets an independent noise level.
# For block_size_blk=4 and block_size_seq=512, there are 128 blocks.
# Each block's tokens share the same t, giving a staircase noise pattern.
# When block_size_blk == block_size_seq, this reduces to Phase 2's
# per-sequence timestep.

# Create streaming iterator for training data
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
    """Uses .skip(100_000) to avoid overlap with training data."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    return iter(ds.skip(100_000))

# Cached staircase mask — same for all batches, computed once on first call.
_cached_staircase_mask = None

num_blocks = block_size_seq // block_size_blk


def get_batch(split="train"):
    """
    Build a training batch from streamed FineWeb-Edu documents.

    Returns:
        x_input:   (B, 2*block_size_seq) — [x_t || x_0] concatenated       [NEW]
        targets:   (B, block_size_seq)   — original unmasked token IDs
        mask:      (B, block_size_seq)   — True where noise-masked AND real token
        t:         (B, block_size_seq)   — per-token timestep               [DIFF 3]
        attn_mask: (2*block_size_seq, 2*block_size_seq) — staircase mask (cached)
    """
    global _train_iter, _val_iter, _cached_staircase_mask

    # Build staircase mask once, then reuse
    if _cached_staircase_mask is None:
        _cached_staircase_mask = build_staircase_mask(
            block_size_seq, block_size_blk
        ).to(device)

    # Pick the right iterator (lazy-init on first call)
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

        # Truncate to block_size_seq
        ids = ids[:block_size_seq]
        seq_len = len(ids)

        # Pad shorter sequences with mask_token_id (right-padding)
        pad_len = block_size_seq - seq_len
        if pad_len > 0:
            ids = ids + [mask_token_id] * pad_len

        token_seqs.append(ids)
        # True for real tokens, False for padding
        pad_mask = [True] * seq_len + [False] * pad_len
        padding_masks.append(pad_mask)

    # Stack into tensors
    targets = torch.tensor(token_seqs, dtype=torch.long)           # (B, L)
    padding = torch.tensor(padding_masks, dtype=torch.bool)        # (B, L)

    # --- Cosine noise schedule with per-block timesteps ---         [DIFF 3]
    # Sample one t ~ U[0,1] per block, then repeat to per-token shape.
    # For block_size_blk == block_size_seq, num_blocks=1 -> same as Phase 2.
    t_blocks = torch.rand(batch_size, num_blocks)                    # (B, num_blocks)
    t = t_blocks.repeat_interleave(block_size_blk, dim=1)           # (B, L)
    mask_prob = 1.0 - torch.cos(t * math.pi / 2).square()           # (B, L)

    # Per-token binary noise mask: each token independently masked
    noise = torch.rand(batch_size, block_size_seq) < mask_prob       # (B, L)

    # Don't noise-mask padding positions — they're already [MASK] tokens.
    # The loss mask should only include positions that were (a) noise-masked
    # AND (b) are real tokens (not padding).
    mask = noise & padding                                           # (B, L)

    # Build x_t: replace noise-masked positions with [MASK]
    x_noisy = targets.clone()
    x_noisy[mask] = mask_token_id

    # [NEW] Concatenate: x_input = [x_t || x_0]
    x_input = torch.cat([x_noisy, targets], dim=1)                   # (B, 2L)

    # Move everything to device
    x_input = x_input.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    t = t.to(device)

    return x_input, targets, mask, t, _cached_staircase_mask


# ============================================================================
# RMSNorm (Functional)
# ============================================================================
# No learnable parameters — purely functional normalization.

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================
# RoPE encodes position by rotating pairs of dimensions in the query/key
# vectors. Naturally handles relative positions and generalizes to unseen
# sequence lengths.

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


# ============================================================================
# Staircase Attention Mask                                               [NEW]
# ============================================================================
# The core innovation of BD3-LMs. For a doubled training sequence [x_t || x_0]
# of length 2n, the mask controls information flow:
#
#   M_BD  (block-diagonal):      same block, same half -> bidirectional
#   M_OBC (offset block-causal): x_t attends to x_0 from EARLIER blocks
#   M_BC  (block-causal):        x_0 attends to x_0 causally (current + earlier)
#
# Returns float tensor (2n, 2n): 0.0 where allowed, -inf where blocked.

def build_staircase_mask(seq_len, block_size_blk):
    n = seq_len
    total = 2 * n

    pos = torch.arange(total)
    q = pos.unsqueeze(1)   # (2n, 1)
    kv = pos.unsqueeze(0)  # (1, 2n)

    x0_flag_q = (q >= n)
    x0_flag_kv = (kv >= n)

    block_q = (q % n) // block_size_blk
    block_kv = (kv % n) // block_size_blk

    # M_BD: same block, same half — full bidirectional within a block
    m_bd = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # M_OBC: x_t queries attend to x_0 keys from current or earlier blocks
    m_obc = (block_q >= block_kv) & x0_flag_kv & ~x0_flag_q

    # M_BC: x_0 queries attend to x_0 keys from current or earlier blocks
    m_bc = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    allow = m_bd | m_obc | m_bc
    mask = torch.where(allow, 0.0, float('-inf'))
    return mask


def _visualize_mask(mask, seq_len):
    """Debug helper: print (2n, 2n) mask as a character grid."""
    n = seq_len
    total = 2 * n
    assert mask.shape == (total, total)

    num_blocks = (n + block_size_blk - 1) // block_size_blk

    # Column header
    halves = ["x_t", "x_0"]
    header_half = "         "
    header_blk = "         "
    for half in halves:
        span = n
        pad = span // 2 - len(half) // 2
        header_half += " " * pad + half + " " * (span - pad - len(half))
    print(header_half)

    for half in halves:
        for b in range(num_blocks):
            label = f"b{b}"
            blen = min(block_size_blk, n - b * block_size_blk)
            pad = blen // 2 - len(label) // 2
            header_blk += " " * pad + label + " " * (blen - pad - len(label))
    print(header_blk)

    # Rows
    for r in range(total):
        half = "x_t" if r < n else "x_0"
        b_idx = (r % n) // block_size_blk
        pos_in_blk = (r % n) % block_size_blk
        if pos_in_blk == 0:
            label = f"{half} b{b_idx}: "
        else:
            label = "         "
        row_chars = ""
        for c in range(total):
            row_chars += "." if mask[r, c] == 0.0 else "X"
        print(label + row_chars)


# ============================================================================
# Multi-Head Attention                                                  [DIFF 2]
# ============================================================================
#
# Phase 2: F.scaled_dot_product_attention(q, k, v, is_causal=False)
# Phase 3: F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
#
# The attn_mask is a float tensor where -inf = "don't attend", 0.0 = "attend".
# This enables the staircase mask during training (bidirectional within blocks,
# causal across blocks). During inference with KV cache, attn_mask is None
# and we fall back to is_causal=False (all cached tokens are finalized).
#
# KV Cache (for block-by-block generation)                              [NEW 2]
# ========================================
#
#   Block 0 (finalized):  K0, V0 cached
#   Block 1 (finalized):  K1, V1 cached
#   Block 2 (denoising):  K2, V2 computed fresh each step
#
#   Attention: Q from block 2, K/V = [K0|K1|K2], [V0|V1|V2]
#   No mask needed — cached blocks are finalized, current block is bidirectional.
#
# RoPE with shared positions during training                            [DIFF]
# ==============================================
#   Input: [x_t || x_0] of length 2L
#   x_t at positions 0..L-1 gets RoPE for 0..L-1
#   x_0 at positions L..2L-1 ALSO gets RoPE for 0..L-1
#   This tells the model position i in x_t corresponds to position i in x_0.

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.kv_cache = None  # [NEW 2] (cached_k, cached_v) or None

    def reset_cache(self):
        self.kv_cache = None

    def update_cache(self, k, v):
        """Append finalized block's K,V to the cache."""
        if self.kv_cache is None:
            self.kv_cache = (k, v)
        else:
            old_k, old_v = self.kv_cache
            self.kv_cache = (torch.cat([old_k, k], dim=2),
                             torch.cat([old_v, v], dim=2))

    def forward(self, x, cos_sin, attn_mask=None):
        B, T, C = x.size()

        # Project to queries, keys, values then reshape for multi-head
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply RoPE — shared positions for [x_t || x_0] during training
        cos, sin = cos_sin
        if T == cos.size(1):
            # Sequence length matches cos/sin — apply directly
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        else:
            # Training with [x_t || x_0]: T = 2L, cos/sin covers L positions
            # Both halves share the same position IDs (0..L-1)
            L = cos.size(1)
            assert T == 2 * L, f"Expected T={T} == 2*L={2*L}"
            q_t, q_0 = q[:, :L], q[:, L:]
            k_t, k_0 = k[:, :L], k[:, L:]
            q_t = apply_rotary_emb(q_t, cos, sin)
            q_0 = apply_rotary_emb(q_0, cos, sin)
            k_t = apply_rotary_emb(k_t, cos, sin)
            k_0 = apply_rotary_emb(k_0, cos, sin)
            q = torch.cat([q_t, q_0], dim=1)
            k = torch.cat([k_t, k_0], dim=1)

        # QK-norm: stabilizes training at scale
        q, k = norm(q), norm(k)

        # Transpose to (B, H, T, D) for attention computation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [NEW 2] Prepend cached K,V from finalized blocks during generation
        if self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # [DIFF 2] Explicit attn_mask replaces is_causal=False
        # During training: staircase mask (float tensor, -inf/0.0)
        # During inference with KV cache: None -> is_causal=False (all visible)
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# ============================================================================
# SwiGLU (Gated Feed-Forward Network)
# ============================================================================
# Identical to Phase 2. Gate + up projection through SiLU, then down.
# hidden_dim rounded to multiple of 256 for GPU memory alignment.

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
# [DIFF] Threads attn_mask through to attention.

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = SwiGLU()

    def forward(self, x, cos_sin, attn_mask=None):
        x = x + self.attn(norm(x), cos_sin, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x


# ============================================================================
# Model                                                                 [DIFF]
# ============================================================================
#
# Changes from Phase 2:
#   [DIFF 1] vocab_size = 151670 (Qwen 3 + [MASK]), up from 32768
#   [DIFF 4] Tied embeddings: lm_head.weight = token_emb.weight
#   [DIFF 2] Explicit attn_mask threaded through all blocks
#   [DIFF 3] Per-token ELBO weighting (t is B,L not B,)
#
# RoPE precomputed for block_size_seq * 2 = 1024 positions to handle
# [x_t || x_0] concatenation during training.
#
# Training forward pass:
#   Input: [x_t || x_0] shape (B, 2L)
#   Logits extracted from first half only: [:, :L]
#   Loss computed on masked positions in x_t with per-token ELBO weight 1/t
#
# ELBO Loss with Per-Token Timesteps                                    [DIFF 3]
# ==========================================
#   Phase 2: t is (B,) — one timestep per sequence
#   Phase 3: t is (B, L) — one timestep per token (each block has own noise)
#
#   Weight per token: 1/t[b, i], applied to per-token CE loss
#   Loss = mean over all masked tokens of (1/t * CE)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings: Qwen 3 vocab + [MASK]                       [DIFF 1]
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Precompute rotary embeddings for 2x context length
        # (handles [x_t || x_0] during training)
        self.rotary_seq_len = block_size_seq * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output projection — tied with embedding                       [DIFF 4]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights (tied weight initialized once via token_emb)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def reset_kv_cache(self):
        """Clear KV caches in all attention layers."""
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(self, idx, targets=None, mask=None, t=None, attn_mask=None):
        """
        Forward pass.

        Args:
            idx: input token IDs, shape (B, T) where T = 2L during training
            targets: ground-truth token IDs, shape (B, L). None during generation.
            mask: boolean mask, shape (B, L). True where tokens were masked.
            t: timestep per token, shape (B, L). Used for per-token ELBO weighting.
            attn_mask: float attention mask (B*n_head or 1, T, T) or None.
                       -inf = don't attend, 0.0 = attend.

        Returns:
            logits: (B, L, vocab_size) during training, (B, T, vocab_size) during inference
            loss: scalar or None
        """
        B, T = idx.size()

        # Embed tokens and normalize
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Slice rotary embeddings to current sequence length
        # During training: T = 2L, but cos/sin sliced to L (shared-position RoPE)
        # During inference: T = block_size_blk, cos/sin sliced to T
        if targets is not None:
            L = targets.size(1)
            cos_sin = (self.cos[:, :L], self.sin[:, :L])
        else:
            cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks, threading attn_mask
        for block in self.blocks:
            x = block(x, cos_sin, attn_mask=attn_mask)
        x = norm(x)

        # During training: extract logits from first half (x_t predictions)
        if targets is not None:
            L = targets.size(1)
            logits = self.lm_head(x[:, :L])  # (B, L, vocab_size)
        else:
            logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # [DIFF 3] Per-token ELBO weighting: t is (B, L) not (B,)
            # Weight: 1/t per token, applied to per-token CE loss on masked positions
            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1), reduction="none"
            )
            per_token_loss = per_token_loss.view(B, L)

            if mask is not None and t is not None:
                # Per-token ELBO weight: 1/t, clamped for numerical stability
                elbo_weight = 1.0 / t.clamp(min=1e-4)  # (B, L)

                # Mask out unmasked positions, apply ELBO weight
                weighted_loss = per_token_loss * mask.float() * elbo_weight  # (B, L)
                mask_count = mask.float().sum().clamp(min=1)
                loss = weighted_loss.sum() / mask_count
            elif mask is not None:
                mask_flat = mask.view(-1).float()
                loss = (per_token_loss.view(-1) * mask_flat).sum() / mask_flat.sum()
            else:
                loss = per_token_loss.mean()

        return logits, loss


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def estimate_loss(model):
    """Estimate train and val loss by averaging over eval_iters batches."""
    out = {}
    was_training = model.training
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_input, targets, mask, t, attn_mask = get_batch(split)
            _, loss = model(x_input, targets, mask, t, attn_mask)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


# ============================================================================
# Learning Rate Schedule: Cosine Decay with Warmup
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
#   get_batch() -> (x_input, targets, mask, t, attn_mask)
#        |
#        v
#   model(x_input, targets, mask, t, attn_mask) -> loss
#        |              [DIFF] masked positions with staircase mask
#        v              [DIFF 3] ELBO weighted by per-token 1/t
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
#   2. Apply cosine noise schedule per block: sample t, mask tokens
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
    print("block_dllm — block diffusion language model")
    print("=" * 60)
    print(f"  depth          = {depth}")
    print(f"  n_layer        = {n_layer}")
    print(f"  n_embd         = {n_embd}")
    print(f"  n_head         = {n_head}")
    print(f"  block_size_seq = {block_size_seq}")
    print(f"  block_size_blk = {block_size_blk}")
    print(f"  num_blocks     = {num_blocks}")
    print(f"  vocab_size     = {vocab_size}")
    print(f"  device         = {device}")
    print("=" * 60)

    # --- 2. Set up weights path ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(
        script_dir, "weights", f"block_dllm_d{depth}_b{block_size_blk}.pt"
    )
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # --- 3. Instantiate model ---
    model = Model().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    # Tied embeddings: lm_head shares token_emb weights, so subtract one copy
    tied_params = model.token_emb.weight.numel()
    print(f"{param_count / 1e6:.2f}M parameters ({(param_count - tied_params) / 1e6:.2f}M unique, "
          f"{tied_params / 1e6:.2f}M tied)")

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

            # Evaluate periodically
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                      f"val loss {losses['val']:.4f} | lr {lr:.6f}")
                # sample generation will be added with Task 7
                if device == "mps":
                    torch.mps.empty_cache()

            # Forward pass: get batch -> model -> loss
            x_input, targets, mask, t, attn_mask = get_batch("train")
            logits, loss = model(x_input, targets, mask, t, attn_mask)

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
                tokens_per_sec = (step * batch_size * block_size_seq) / dt
                print(f"  step {step:5d} | loss {loss.item():.4f} | {tokens_per_sec:.0f} tok/s")

        # Save trained weights
        torch.save(model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")

    elif args.prompt is not None:
        # Generation requested but no weights found
        print(f"No weights found at {weights_path}")
        print("Run with --train to train from scratch.")
        sys.exit(1)

    # --- 5. Generate text ---
    if args.prompt is not None:
        # sample generation will be added with Task 7
        print("\n" + "=" * 60)
        print("Generation (--prompt) will be added with Task 7.")
        print("=" * 60)

    elif not args.train:
        # --- Debug: Mask verification and batch shape printing ---
        print("=== Staircase Mask Verification ===")
        print(f"seq_len=8, block_size_blk=4 (2 blocks)\n")
        test_mask = build_staircase_mask(seq_len=8, block_size_blk=4)
        assert test_mask.shape == (16, 16), f"Expected (16, 16), got {test_mask.shape}"
        # Temporarily override global for visualization label formatting
        _saved_blk = block_size_blk
        block_size_blk = 4
        _visualize_mask(test_mask, seq_len=8)
        block_size_blk = _saved_blk
        print(f"\nMask shape: {test_mask.shape}")
        print(f"Allowed entries: {(test_mask == 0.0).sum().item()}")
        print(f"Blocked entries: {(test_mask == float('-inf')).sum().item()}")

        # --- Batch Shape Verification ---
        print(f"\n=== Batch Shape Verification ===")
        print(f"block_size_seq={block_size_seq}, block_size_blk={_saved_blk}, "
              f"num_blocks={block_size_seq // _saved_blk}, batch_size={batch_size}")
        L = block_size_seq
        print(f"  x_input:   (B, 2*L)  = ({batch_size}, {2 * L})")
        print(f"  targets:   (B, L)    = ({batch_size}, {L})")
        print(f"  mask:      (B, L)    = ({batch_size}, {L})")
        print(f"  t:         (B, L)    = ({batch_size}, {L})")
        print(f"  attn_mask: (2L, 2L)  = ({2 * L}, {2 * L})")
