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
