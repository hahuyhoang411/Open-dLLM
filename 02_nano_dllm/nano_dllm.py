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
