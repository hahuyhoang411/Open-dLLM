"""
block_dllm.py — A block diffusion language model trained on FineWeb-Edu.

Phase 3 of Open-dLLM: upgrades from full-sequence diffusion (Phase 2) to
block diffusion with a staircase attention mask and KV caching. Same BPE
tokenizer (vocab=32768) and depth parametrization as Phase 2.

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
    | Token Embedding    |               vocab_size=32768 (BPE)
    | (32768 -> D)       |               D = depth * 64
    +-------------------+                depth=6 -> D=384
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
    | LM Head            |               Linear(D -> vocab)
    | (D -> vocab)       |
    +-------------------+
           |
           v
    Output: logits (B, T, 32768)         Predict original token at EVERY position

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
    depth=6  -> 384-dim,  6 heads,  6 layers  (~26M params)
    depth=10 -> 640-dim, 10 heads, 10 layers
    depth=12 -> 768-dim, 12 heads, 12 layers

    [DIFF] markers (changes from Phase 2):
      [DIFF 1] Staircase attention mask (replaces is_causal=False)
      [DIFF 2] Per-block noise levels (replaces per-sequence)

    [NEW] markers for Phase 3:
      [NEW 1] Block diffusion with configurable block size
      [NEW 2] KV caching across blocks for inference
      [NEW 3] Block-by-block generation

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
from tokenizers import Tokenizer

# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="block_dllm — block diffusion language model")
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--depth", type=int, default=6, help="model depth (controls all dims)")
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
n_embd = depth * 64     # depth=6 -> 384, depth=12 -> 768
n_head = depth
head_dim = 64            # fixed across all depths

# Block size                                                                 [NEW 1]
block_size_seq = 512     # context length in tokens (same as Phase 2)
block_size_blk = block_size_seq if args.block_size == 0 else args.block_size
assert block_size_seq % block_size_blk == 0, \
    f"block_size_seq={block_size_seq} must be divisible by block_size_blk={block_size_blk}"

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
# Tokenizer: BPE with special tokens [MASK]=0, <|endoftext|>=1, <|padding|>=2
# ============================================================================
# Same BPE tokenizer as Phase 2 (vocab=32768). Special token IDs are stable:
#   [MASK]=0       — diffusion noise token
#   <|endoftext|>=1 — document boundary / EOS (enables generation stopping)
#   <|padding|>=2   — right-padding for short sequences

tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()   # 32768
mask_token_id = tokenizer.token_to_id("[MASK]")       # 0
eos_token_id = tokenizer.token_to_id("<|endoftext|>")  # 1
pad_token_id = tokenizer.token_to_id("<|padding|>")    # 2

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)


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
#     Tokenize (BPE) + append <|endoftext|>, truncate to block_size_seq=512
#            |
#            v
#     Pad to block_size_seq with <|padding|>
#            |
#            v
#     Sample t ~ U[0,1] PER BLOCK (not per sequence)                   [DIFF 2]
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
# [DIFF 2] Per-block timesteps: each block gets an independent noise level.
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
        t:         (B, block_size_seq)   — per-token timestep               [DIFF 2]
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

        # Append EOS to mark document boundary
        ids = ids + [eos_token_id]

        # Truncate to block_size_seq (EOS may be clipped for long docs — that's fine)
        ids = ids[:block_size_seq]
        seq_len = len(ids)

        # Pad shorter sequences with pad_token_id (right-padding)
        pad_len = block_size_seq - seq_len
        if pad_len > 0:
            ids = ids + [pad_token_id] * pad_len

        token_seqs.append(ids)
        # True for real tokens, False for padding
        pad_mask = [True] * seq_len + [False] * pad_len
        padding_masks.append(pad_mask)

    # Stack into tensors
    targets = torch.tensor(token_seqs, dtype=torch.long)           # (B, L)
    padding = torch.tensor(padding_masks, dtype=torch.bool)        # (B, L)

    # --- Cosine noise schedule with per-block timesteps ---         [DIFF 2]
    # Sample one t ~ U[0,1] per block, then repeat to per-token shape.
    # For block_size_blk == block_size_seq, num_blocks=1 -> same as Phase 2.
    t_blocks = torch.rand(batch_size, num_blocks)                    # (B, num_blocks)
    t = t_blocks.repeat_interleave(block_size_blk, dim=1)           # (B, L)
    mask_prob = 1.0 - torch.cos(t * math.pi / 2).square()           # (B, L)

    # Per-token binary noise mask: each token independently masked
    noise = torch.rand(batch_size, block_size_seq) < mask_prob       # (B, L)

    # Don't noise-mask padding positions — they're <|padding|> tokens.
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

    # M_OBC: x_t queries attend to x_0 keys from STRICTLY EARLIER blocks only
    # Using > (not >=) to prevent x_t from peeking at its own block's clean x_0,
    # which would leak the training labels to masked positions.
    m_obc = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q

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
        self.kv_cache = None   # [NEW 2] (cached_k, cached_v) or None
        self.cache_mode = False # [NEW 2] when True, auto-append K,V to cache

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

        # [NEW 2] Save current block's K,V BEFORE prepending cache
        # (we only want to cache the current block, not the full sequence)
        k_current, v_current = k, v

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

        # [NEW 2] Auto-cache current block's K,V after attention
        if self.cache_mode:
            self.update_cache(k_current, v_current)

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
#   [DIFF 1] Explicit attn_mask threaded through all blocks
#   [DIFF 2] Per-token ELBO weighting (t is B,L not B,)
#
# RoPE precomputed for block_size_seq * 2 = 1024 positions to handle
# [x_t || x_0] concatenation during training.
#
# Training forward pass:
#   Input: [x_t || x_0] shape (B, 2L)
#   Logits extracted from first half only: [:, :L]
#   Loss computed on masked positions in x_t with per-token ELBO weight 1/t
#
# ELBO Loss with Per-Token Timesteps                                    [DIFF 2]
# ==========================================
#   Phase 2: t is (B,) — one timestep per sequence
#   Phase 3: t is (B, L) — one timestep per token (each block has own noise)
#
#   Weight per token: 1/t[b, i], applied to per-token CE loss
#   Loss = mean over all masked tokens of (1/t * CE)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings: BPE vocab (32768)
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Precompute rotary embeddings: 2x context length for training
        # ([x_t || x_0]), or 4096 for long generation with KV cache
        self.rotary_seq_len = max(block_size_seq * 2, 4096)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output projection
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

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

    def set_cache_mode(self, enabled):
        """Toggle auto-caching of K,V in all attention layers."""
        for block in self.blocks:
            block.attn.cache_mode = enabled

    def forward(self, idx, targets=None, mask=None, t=None, attn_mask=None,
                pos_offset=0):
        """
        Forward pass.

        Args:
            idx: input token IDs, shape (B, T) where T = 2L during training
            targets: ground-truth token IDs, shape (B, L). None during generation.
            mask: boolean mask, shape (B, L). True where tokens were masked.
            t: timestep per token, shape (B, L). Used for per-token ELBO weighting.
            attn_mask: float attention mask (B*n_head or 1, T, T) or None.
                       -inf = don't attend, 0.0 = attend.
            pos_offset: RoPE position offset for KV-cached generation.       [NEW 2]
                        Block at position i gets RoPE positions
                        [pos_offset, pos_offset + block_size_blk).

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
        #                   with pos_offset for block position             [NEW 2]
        if targets is not None:
            L = targets.size(1)
            cos_sin = (self.cos[:, :L], self.sin[:, :L])
        else:
            cos_sin = (self.cos[:, pos_offset:pos_offset + T],
                       self.sin[:, pos_offset:pos_offset + T])

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
            # [DIFF 2] Per-token ELBO weighting: t is (B, L) not (B,)
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
# Generation: Block-by-Block with KV Cache                              [NEW 3]
# ============================================================================
#
# Unlike Phase 2 (full-sequence diffusion), Phase 3 generates one block at a
# time with KV caching — combining AR's sequential block generation with
# diffusion's parallel in-block decoding.
#
# Block-by-Block Generation with KV Cache
# ========================================
#
#   Prompt: "The quick brown fox"   block_size_blk=4   denoise_steps=10
#
#   Step 0: Warm up KV cache with prompt blocks
#   +-----------+
#   | The quick |  -> forward pass, cache K0,V0
#   +-----------+
#   | brown fox |  -> forward pass (attends to K0,V0), cache K1,V1
#   +-----------+
#
#   Step 1: Generate block 2 (first new block)
#   KV cache: [K0,V0 | K1,V1]
#   Block input: [_ _ _ _]  (all [MASK])
#
#     Denoise iteration 1:
#       Input: [_ _ _ _]   pos_offset=8
#       Q from block 2, K/V = [K0|K1|K_current], [V0|V1|V_current]
#       Confidence:  0.45  0.98  0.72  0.91
#       Decode?:      no   YES    no    no
#       Result: [_ jumped _ _]
#
#     Denoise iteration 2:
#       Input: [_ jumped _ _]
#       Confidence:  0.97  --   0.88  0.96
#       Decode?:     YES   --    no   YES
#       Result: [▁ jumped ▁over _]
#
#     Denoise iteration 3:
#       Input: [▁ jumped ▁over _]
#       Confidence:  --    --   --   0.99
#       Result: [▁ jumped ▁over ▁the]  <- block fully denoised!
#
#     Final pass: cache K2,V2 from denoised block
#
#   Step 2: Generate block 3
#   KV cache: [K0,V0 | K1,V1 | K2,V2]
#   Block input: [_ _ _ _]
#   ...continue until max_new_tokens or EOS...
#
# Key differences from Phase 2:
#   1. Input is ONE BLOCK (block_size_blk tokens), not full sequence
#   2. Attention uses KV cache: Q_current @ [K_cached | K_current]
#   3. RoPE offset: block i gets positions [i*block_size_blk, (i+1)*block_size_blk)
#   4. denoise_steps caps iterations per block (Phase 2 runs until all unmasked)
#   5. After each block: one final forward pass with cache_mode=True to save K,V

@torch.no_grad()
def generate(model, max_new_tokens=512, prompt=None, denoise_steps=10,
             temp=0.8, top_k=5, confidence_threshold=0.95):
    was_training = model.training
    model.eval()

    # Encode prompt
    prompt_ids = encode(prompt) if prompt else []
    prompt_len = len(prompt_ids)

    # Reset KV cache for fresh generation
    model.reset_kv_cache()
    model.set_cache_mode(False)

    all_tokens = list(prompt_ids)
    total_steps = 0
    t_start = time.time()

    # --- Warm up KV cache with full prompt blocks ---
    # Split prompt into full blocks, cache each one
    n_full_prompt_blocks = prompt_len // block_size_blk
    prompt_remainder = prompt_len % block_size_blk
    pos_offset = 0

    model.set_cache_mode(True)
    for i in range(n_full_prompt_blocks):
        start = i * block_size_blk
        end = start + block_size_blk
        block_ids = torch.tensor(
            [prompt_ids[start:end]], dtype=torch.long, device=device
        )
        model(block_ids, pos_offset=pos_offset)
        pos_offset += block_size_blk
    model.set_cache_mode(False)

    # --- Block-by-block generation loop ---
    tokens_generated = 0
    done = False

    while tokens_generated < max_new_tokens and not done:
        # How many positions in this block come from the prompt remainder?
        fill_from_prompt = min(prompt_remainder, block_size_blk)
        gen_positions = block_size_blk - fill_from_prompt

        # Initialize block: prompt remainder (if any) + [MASK] tokens
        block = torch.full(
            (1, block_size_blk), mask_token_id, dtype=torch.long, device=device
        )
        if fill_from_prompt > 0:
            remainder_start = n_full_prompt_blocks * block_size_blk
            block[0, :fill_from_prompt] = torch.tensor(
                prompt_ids[remainder_start:remainder_start + fill_from_prompt],
                dtype=torch.long, device=device,
            )

        # Track which positions need decoding (only the [MASK] ones)
        masked = torch.zeros(1, block_size_blk, dtype=torch.bool, device=device)
        masked[0, fill_from_prompt:] = True

        # --- Denoise loop: iteratively unmask within this block ---
        for step in range(denoise_steps):
            if not masked.any():
                break
            total_steps += 1

            # Forward pass: just the current block, KV cache handles context
            logits, _ = model(block, pos_offset=pos_offset)
            probs = F.softmax(logits / temp, dim=-1)

            # Never generate [MASK] or <|padding|> tokens
            probs[:, :, mask_token_id] = 0.0
            probs[:, :, pad_token_id] = 0.0

            # Top-k confidence-based unmasking
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)  # (1, block_size_blk)

            # Unmask positions above confidence threshold
            decode_mask = (confidences >= confidence_threshold) & masked

            # If nothing qualifies, force-unmask the most confident masked position
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences,
                    torch.tensor(-float("inf"), device=device),
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            # Sample from normalized top-k distribution
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(
                top_k_probs_norm.view(-1, top_k), 1
            ).view(1, block_size_blk)
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            # Place sampled tokens at decoded positions
            block = torch.where(decode_mask, sampled_tokens, block)
            masked = masked & ~decode_mask

        # --- Cache the finalized block ---
        model.set_cache_mode(True)
        model(block, pos_offset=pos_offset)
        model.set_cache_mode(False)
        pos_offset += block_size_blk

        # Consume prompt remainder (only affects first generated block)
        prompt_remainder = 0

        # Extract generated tokens from this block (skip prompt remainder)
        new_tokens = block[0, fill_from_prompt:].tolist()

        # Check for EOS — truncate at first occurrence
        if eos_token_id is not None:
            for i, tok in enumerate(new_tokens):
                if tok == eos_token_id:
                    new_tokens = new_tokens[:i]
                    done = True
                    break

        all_tokens.extend(new_tokens)
        tokens_generated += len(new_tokens)

    # --- Stats ---
    elapsed = time.time() - t_start
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float("inf")
    print(f"Generation stats: {total_steps} denoise steps, {tokens_generated} tokens, "
          f"{tok_per_sec:.1f} tok/s")

    # Reset KV cache so stale generation state doesn't leak into training
    model.reset_kv_cache()

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
#        v              [DIFF 2] ELBO weighted by per-token 1/t
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

            # Evaluate periodically
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                      f"val loss {losses['val']:.4f} | lr {lr:.6f}")
                if step > 0:
                    sample = generate(model, max_new_tokens=64, temp=0.8, top_k=5)
                    print(f"--- sample ---\n{sample[:300]}\n--- end sample ---")
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
        sample = generate(model, max_new_tokens=args.max_tokens, prompt=args.prompt,
                          denoise_steps=args.denoise_steps, temp=0.8, top_k=5)
        print(sample)

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
