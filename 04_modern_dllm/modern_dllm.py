"""
modern_dllm.py — A modern block diffusion language model with fused kernels.

Phase 4 of Open-dLLM: upgrades from Phase 3's block diffusion with modern
training infrastructure: AMP fp16, gradient checkpointing, Liger fused
kernels (RMSNorm, SwiGLU, FusedLinearCrossEntropy), mask ratio bandwidth,
and complementary masking. Targets Kaggle T4 (CC 7.5, 16GB).

Phase 4 changes from Phase 3 [P4]:
    [P4-1] AMP fp16 + GradScaler for mixed precision training
    [P4-2] Gradient checkpointing per transformer block
    [P4-3] Liger fused kernels (SwiGLU, FusedLinearCrossEntropy)
    [P4-4] Mask ratio bandwidth: clamp t to [t_min, t_max]
    [P4-5] Complementary masking: train on both mask m and complement 1-m
    [P4-6] FlexAttention: compiled block-sparse staircase mask kernel
    [P4-7] GQA: grouped query attention (4:1 ratio)
    [P4-8] CART noise rescheduling: context-adaptive ELBO weights (Dream 7B)
    [P4-9] Liger fused learnable RMSNorm (7x faster, replaces parameterless norm)
    [P4-11] Muon optimizer for 2D weights + AdamW for embeddings/1D params
    [P4-12] Gradient accumulation for effective batch scaling
    [P4-13] torch.compile full model (Triton/inductor)
    [P4-14] DDP multi-GPU via torchrun
    [P4-15] CUDA tuning (cudnn.benchmark, no_grad, empty_cache)
    [P4-16] Qwen3-0.6B architecture (16L/1024d/16h/4kv/2816MLP, tied embeddings, MLP dropout)
    [P4-17] WSD scheduler (warmup-stable-decay, replaces cosine)
    [P4-18] Multi-source 100B dataset (FinePDFs+DCLM+FineWeb-Edu)
    [P4-19] Fix training loss: real-token normalization, CART off by default

Requires: PyTorch 2.5+ (FlexAttention, GQA enable_gqa), CUDA (Triton kernels)

Retained from Phase 3:
    - Block diffusion with staircase attention mask
    - Per-block noise levels with cosine schedule
    - KV caching for block-by-block generation
    - Confidence-based decoding with top-k sampling
    - BPE tokenizer (vocab=32768): <|mask|>=0, <|endoftext|>=1, <|padding|>=2

References:
    - BD3-LMs (Arriola et al., ICLR 2025 Oral): arxiv.org/abs/2503.09573
    - Fast-dLLM v2 (NVIDIA, 2025): arxiv.org/abs/2509.26328
    - Liger Kernel (LinkedIn, 2024): arxiv.org/abs/2410.10989
    - MDLM (Sahoo et al., NeurIPS 2024): arxiv.org/abs/2406.07524
    - LLaDA (Nie et al., 2025): arxiv.org/abs/2502.09992
"""

import os
import sys
import math
import time
import argparse
import contextlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint       # [P4-2]
import torch.distributed as dist                                       # [P4-14]
from torch.nn.parallel import DistributedDataParallel as DDP           # [P4-14]
from datasets import load_dataset
from tokenizers import Tokenizer

# [P4-3] Liger fused kernels — drop-in replacements for PyTorch primitives.
# FusedLinearCrossEntropy: never materializes the full (B*L, vocab) logit tensor.
# LigerSiLUMulFunction: fused SiLU + elementwise multiply (1.6x memory savings).
# [P4-9] LigerRMSNorm: learnable RMSNorm with fused backward, 7x faster.
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )
    from liger_kernel.transformers import LigerRMSNorm
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

# [P4-11] Muon optimizer: Newton-Schulz orthogonalization for 2D hidden weights.
# ~50% less optimizer memory than AdamW, ~1.35x faster convergence.
# pip install git+https://github.com/KellerJordan/Muon (NOT PyPI 'muon' which is bioinformatics)
try:
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

# [P4-6] FlexAttention: compiled custom attention patterns at ~90% FlashAttention2
# speed. Replaces the materialized float staircase mask with a block-sparse kernel
# that skips entirely-masked 128x128 tiles. Requires PyTorch 2.5+ and CUDA.
try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask, BlockMask,
    )
    _compiled_flex = torch.compile(flex_attention)
    FLEX_AVAILABLE = True
except (ImportError, AttributeError):
    FLEX_AVAILABLE = False
    BlockMask = type(None)  # dummy for isinstance check fallback

# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="modern_dllm — modern block diffusion LM")
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--prompt", type=str, default=None, help="text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="max tokens to generate")
    # [P4-16] Architecture dims — explicit constants, configurable for exploration
    parser.add_argument("--n-layer", type=int, default=16, help="number of transformer layers")
    parser.add_argument("--n-embd", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--n-head", type=int, default=16, help="number of query heads")
    parser.add_argument("--n-kv-head", type=int, default=4,                         # [P4-7]
                        help="number of KV heads for GQA (default: 4 for 4:1 ratio)")
    parser.add_argument("--mlp-hidden", type=int, default=2816, help="MLP hidden dimension")
    parser.add_argument("--seq-len", type=int, default=1024,                        # [P4-6]
                        help="sequence length in tokens (default: 1024)")
    parser.add_argument("--block-size", type=int, default=32, choices=[0, 1, 2, 4, 8, 16, 32],
                        help="block size for diffusion (0=full sequence)")
    parser.add_argument("--batch-size", type=int, default=16,                       # [P4-6]
                        help="batch size per step (doubled by complementary masking)")
    parser.add_argument("--denoise-steps", type=int, default=10,
                        help="denoising steps per block during generation")
    parser.add_argument("--no-amp", action="store_true",                            # [P4-1]
                        help="disable AMP fp16 (for debugging or CPU)")
    parser.add_argument("--no-liger", action="store_true",                          # [P4-3]
                        help="disable Liger fused kernels (use PyTorch fallbacks)")
    parser.add_argument("--no-flex", action="store_true",                           # [P4-6]
                        help="disable FlexAttention (use float staircase mask)")
    parser.add_argument("--cart", action="store_true",                               # [P4-8]
                        help="enable CART noise rescheduling (default: off, use uniform 1/t)")
    parser.add_argument("--cart-p", type=float, default=0.1,                        # [P4-8]
                        help="CART geometric distribution parameter (default: 0.1)")
    parser.add_argument("--no-grad-ckpt", action="store_true",                      # [P4-2]
                        help="disable gradient checkpointing (faster but more VRAM)")
    parser.add_argument("--no-compile", action="store_true",                        # [P4-13]
                        help="disable torch.compile (skip Triton/inductor compilation)")
    parser.add_argument("--no-muon", action="store_true",                           # [P4-11]
                        help="disable Muon optimizer (use AdamW for all params)")
    parser.add_argument("--grad-accum-steps", type=int, default=4,                  # [P4-12]
                        help="gradient accumulation steps (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.1,                       # [P4-16]
                        help="MLP dropout rate (default: 0.1, 0 to disable)")
    return parser.parse_args()

args = parse_args()

# ============================================================================
# Device Detection + DDP                                              [P4-14]
# ============================================================================
# DDP auto-detected via RANK env var (set by torchrun). Single-GPU: ddp=False.
# Launch: torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

torch.manual_seed(1337 + ddp_rank)  # different seed per rank for data diversity
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # [P4-15] auto-tune cuDNN for fixed seq_len
    torch.set_float32_matmul_precision("high")  # [P4-16] TF32 on Ampere+ (no-op on T4)

# ============================================================================
# Hyperparameters
# ============================================================================
# [P4-16] Qwen3-0.6B-inspired architecture, scaled to ~213M params for T4 16GB.
# All dims configurable via CLI for exploration (defaults: 16L/1024d/16h/4kv/2816MLP).

n_layer = args.n_layer
n_embd = args.n_embd
n_head = args.n_head
head_dim = n_embd // n_head
n_kv_head = args.n_kv_head   # [P4-7] GQA (default 4:1 ratio)
mlp_hidden = args.mlp_hidden  # [P4-16] Qwen3-style MLP width

assert n_embd % n_head == 0, f"n_embd={n_embd} must be divisible by n_head={n_head}"
assert n_head % n_kv_head == 0, f"n_head={n_head} must be divisible by n_kv_head={n_kv_head}"

# Block size
block_size_seq = args.seq_len
block_size_blk = block_size_seq if args.block_size == 0 else args.block_size
assert block_size_seq % block_size_blk == 0, \
    f"block_size_seq={block_size_seq} must be divisible by block_size_blk={block_size_blk}"

# Training hyperparameters
batch_size = args.batch_size
max_iters = 50_000
eval_interval = 1000
eval_iters = 50
learning_rate = 6e-4 * ddp_world_size  # [P4-14] linear scaling rule for DDP
warmup_iters = 2000
decay_start = int(0.8 * max_iters)     # [P4-17] WSD: stable -> decay transition
weight_decay = 0.1
grad_clip = 1.0
dropout = args.dropout                  # [P4-16] MLP dropout (Gao et al.)

# [P4-4] Mask ratio bandwidth: clamp t to avoid trivially easy/impossible noise
# t near 0 = almost clean (trivial reconstruction, low training signal)
# t near 1 = almost fully masked (near-random guessing, high variance gradients)
t_min = 0.05
t_max = 0.95

# Feature flags — runtime-resolved from CLI args + availability
use_amp = not args.no_amp and torch.cuda.is_available()                    # [P4-1]
use_liger = not args.no_liger and LIGER_AVAILABLE                          # [P4-3]
use_flex = not args.no_flex and FLEX_AVAILABLE and torch.cuda.is_available()  # [P4-6]
use_cart = args.cart                                                       # [P4-8]
cart_p = args.cart_p
use_grad_ckpt = not args.no_grad_ckpt                                     # [P4-2]
use_compile = not args.no_compile and torch.cuda.is_available()            # [P4-13]
use_muon = not args.no_muon and MUON_AVAILABLE                             # [P4-11]
grad_accum_steps = args.grad_accum_steps                                   # [P4-12]

# ============================================================================
# Tokenizer: BPE with 14 special tokens (SFT-ready)
# ============================================================================
# BPE tokenizer (vocab=32768). Special token IDs 0-13 are reserved:
#   <|mask|>=0     — diffusion noise token
#   <|endoftext|>=1 — document boundary / EOS (enables generation stopping)
#   <|padding|>=2   — right-padding for short sequences

tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()   # 32768
mask_token_id = tokenizer.token_to_id("<|mask|>")      # 0
eos_token_id = tokenizer.token_to_id("<|endoftext|>")  # 1
pad_token_id = tokenizer.token_to_id("<|padding|>")    # 2

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)


# ============================================================================
# Data Pipeline: Multi-Source 100B Streaming                      [DIFF][P4-18]
# ============================================================================
#
# Ported from Phase 2 with per-block timestep sampling.
#
# Data Flow
# =========
#
#     Multi-source 100B dataset (streaming)
#            |
#            v
#     Tokenize (BPE) + append <|endoftext|>, truncate to block_size_seq
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
#     (x[noise] = <|mask|>)
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
        "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",  # [P4-18]
        split="train",
        streaming=True,
    )
    return iter(ds.shuffle(seed=42 + ddp_rank, buffer_size=10_000))  # [P4-14]

def _make_val_iter():
    """Uses .skip(100_000) to avoid overlap with training data."""
    ds = load_dataset(
        "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",  # [P4-18]
        split="train",
        streaming=True,
    )
    return iter(ds.skip(100_000))

# [P4-8] CART: Context-Adaptive Rescheduling of Timesteps (Dream 7B, arXiv:2508.15487)
# Replaces uniform 1/t ELBO weighting with context-adaptive weights.
# For each masked position, sums Geo(cart_p, |distance|) over all unmasked real tokens.
# Tokens near context get higher scores -> lower loss weight (they're easier).
# Tokens far from context get lower scores -> higher loss weight (harder, more signal).
# Implemented as conv1d with symmetric geometric kernel for O(B*L) efficiency.

_cart_kernel_cache = {}

def _compute_cart_weights(mask, padding, p=0.1):
    """Compute CART context-adaptive ELBO weights.

    Args:
        mask: (B, L) bool, True where tokens are noise-masked
        padding: (B, L) bool, True where tokens are real (not padding)
        p: geometric distribution parameter (default 0.1 from Dream 7B config)

    Returns:
        weights: (B, L) float, per-token ELBO weights
    """
    B, L = mask.shape
    context = padding & ~mask  # unmasked real tokens = context

    # Cache the symmetric geometric kernel — it never changes during a run.
    key = (L, p)
    if key not in _cart_kernel_cache:
        max_dist = min(L, 100)
        d = torch.arange(1, max_dist + 1, dtype=torch.float32)
        geo = 0.5 * p * ((1 - p) ** (d - 1))
        kernel = torch.cat([geo.flip(0), torch.zeros(1), geo]).view(1, 1, -1)
        _cart_kernel_cache[key] = (kernel, max_dist)
    kernel, max_dist = _cart_kernel_cache[key]

    # Convolve context indicator with geometric kernel
    ctx = F.pad(context.float().unsqueeze(1), (max_dist, max_dist))
    cart_scores = F.conv1d(ctx, kernel).squeeze(1)  # (B, L)

    # Inverse: less context -> higher weight (concentrate signal on hard tokens)
    # Cap max weight to 1/t_min (=20), matching the uniform 1/t range.
    # Ref: Dream 7B (arXiv:2412.06264). Disabled by default; enable with --cart.
    return (1.0 / cart_scores.clamp(min=1e-4)).clamp(max=1.0 / t_min)


# Cached staircase mask — same for all batches, computed once on first call.
_cached_staircase_mask = None

num_blocks = block_size_seq // block_size_blk


def get_batch(split="train"):
    """
    Build a training batch from streamed documents.

    Returns:
        x_input:   (2B, 2*L) — [x_t || x_0] with complementary masking    [P4-5]
        targets:   (2B, L)   — original unmasked token IDs
        mask:      (2B, L)   — True where noise-masked AND real token
        elbo_w:    (2B, L)   — pre-computed ELBO weights (CART or 1/t)     [P4-8]
        attn_mask: BlockMask or (2L, 2L) float — staircase mask (cached)
    """
    global _train_iter, _val_iter, _cached_staircase_mask

    # Build staircase mask once, then reuse.
    # [P4-6] FlexAttention: BlockMask (compiled kernel, block-sparse)
    # Fallback: float tensor (2L, 2L) with 0.0 / -inf
    if _cached_staircase_mask is None:
        if use_flex:
            _cached_staircase_mask = build_staircase_block_mask(
                block_size_seq, block_size_blk
            )
        else:
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
            # Reinitialize iterator (extremely unlikely with 100B tokens)
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

    # --- Cosine noise schedule with per-block timesteps ---
    # Sample one t ~ U[t_min, t_max] per block, then repeat to per-token shape.
    # [P4-4] Mask ratio bandwidth: clamp to [t_min, t_max] to avoid extreme noise.
    t_blocks = torch.rand(batch_size, num_blocks)                    # (B, num_blocks)
    t_blocks = t_min + (t_max - t_min) * t_blocks                   # [P4-4] scale to [t_min, t_max]
    t = t_blocks.repeat_interleave(block_size_blk, dim=1)           # (B, L)
    mask_prob = 1.0 - torch.cos(t * math.pi / 2).square()           # (B, L)

    # Per-token binary noise mask: each token independently masked
    noise_rand = torch.rand(batch_size, block_size_seq)              # (B, L)
    noise = noise_rand < mask_prob                                   # (B, L)

    # Don't noise-mask padding positions — they're <|padding|> tokens.
    mask = noise & padding                                           # (B, L)

    # Guarantee at least 1 masked token per sequence (Ref: JinjieNi/MegaDLMs)
    zero_masked = mask.sum(dim=1) == 0
    if zero_masked.any():
        for seq_idx in zero_masked.nonzero(as_tuple=True)[0]:
            real_pos = padding[seq_idx].nonzero(as_tuple=True)[0]
            if len(real_pos) > 0:
                pick = real_pos[torch.randint(len(real_pos), (1,))]
                mask[seq_idx, pick] = True

    # Build x_t: replace noise-masked positions with <|mask|>
    x_noisy = targets.clone()
    x_noisy[mask] = mask_token_id

    # [P4-5] Complementary masking: create second sample with complement mask.
    # If position i is masked in sample 1, it's unmasked in sample 2 (and vice versa).
    # This ensures every real position is learned at least once per batch.
    # (Fast-dLLM v2, arXiv:2509.26328)
    comp_noise = ~noise                                              # complement
    comp_mask = comp_noise & padding

    # Same min-1-masked guarantee for complement
    zero_masked_comp = comp_mask.sum(dim=1) == 0
    if zero_masked_comp.any():
        for seq_idx in zero_masked_comp.nonzero(as_tuple=True)[0]:
            real_pos = padding[seq_idx].nonzero(as_tuple=True)[0]
            if len(real_pos) > 0:
                pick = real_pos[torch.randint(len(real_pos), (1,))]
                comp_mask[seq_idx, pick] = True

    x_noisy_comp = targets.clone()
    x_noisy_comp[comp_mask] = mask_token_id
    # Complementary sample uses 1-t (symmetric noise level)
    t_comp = 1.0 - t

    # [P4-8] Pre-compute ELBO weights: either CART (context-adaptive) or uniform 1/t
    if use_cart:
        elbo_w = _compute_cart_weights(mask, padding, cart_p)
        elbo_w_comp = _compute_cart_weights(comp_mask, padding, cart_p)
    else:
        elbo_w = 1.0 / t.clamp(min=1e-4)
        elbo_w_comp = 1.0 / t_comp.clamp(min=1e-4)

    # Concatenate original + complement along batch dimension
    # Effective batch size doubles: 2*B samples per step
    targets_2x = torch.cat([targets, targets], dim=0)               # (2B, L)
    mask_2x = torch.cat([mask, comp_mask], dim=0)                   # (2B, L)
    elbo_w_2x = torch.cat([elbo_w, elbo_w_comp], dim=0)            # (2B, L)

    # Build [x_t || x_0] for both original and complement
    x_input = torch.cat([
        torch.cat([x_noisy, targets], dim=1),                       # (B, 2L) original
        torch.cat([x_noisy_comp, targets], dim=1),                  # (B, 2L) complement
    ], dim=0)                                                        # (2B, 2L)

    # Move everything to device
    x_input = x_input.to(device)
    targets_2x = targets_2x.to(device)
    mask_2x = mask_2x.to(device)
    elbo_w_2x = elbo_w_2x.to(device)

    return x_input, targets_2x, mask_2x, elbo_w_2x, _cached_staircase_mask


# ============================================================================
# RMSNorm                                                              [P4-9]
# ============================================================================
# Learnable RMSNorm: normalize by RMS, then scale by a learned gain parameter.
# LigerRMSNorm fuses the backward pass into a single Triton kernel (7x faster).
# Fallback: PyTorch 2.4+ nn.RMSNorm or manual implementation.

def _make_rms_norm(dim):
    if use_liger:
        return LigerRMSNorm(dim)
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    # Manual fallback for older PyTorch
    class _RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
        def forward(self, x):
            return F.rms_norm(x, (x.size(-1),)) * self.weight
    return _RMSNorm(dim)


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================
# RoPE encodes position by rotating pairs of dimensions in the query/key
# vectors. Naturally handles relative positions and generalizes to unseen
# sequence lengths.

def _apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


def apply_rotary_emb(q, k, cos, sin):
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


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
# Two implementations:
#   build_staircase_mask() — returns float tensor (2n, 2n), works everywhere
#   build_staircase_block_mask() — returns FlexAttention BlockMask, compiled kernel

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


# [P4-6] FlexAttention staircase mask: compiled block-sparse kernel.
# Instead of materializing a (2L, 2L) float tensor, defines a pointwise mask_mod
# function that FlexAttention compiles into a fused Triton kernel. Entirely-masked
# 128x128 tiles are skipped — never loaded or computed. For our staircase pattern
# with ~50% sparsity, this is ~2x faster than SDPA with explicit mask.

def build_staircase_block_mask(seq_len, blk_size):
    """Build a FlexAttention BlockMask for the staircase pattern.

    Args:
        seq_len: number of real tokens (L). Full attention over 2L positions.
        blk_size: diffusion block size (e.g. 32).

    Returns:
        BlockMask object, cacheable across forward passes.
    """
    n = seq_len

    def staircase_mask_mod(b, h, q_idx, kv_idx):
        x0_q = (q_idx >= n)
        x0_kv = (kv_idx >= n)
        blk_q = (q_idx % n) // blk_size
        blk_kv = (kv_idx % n) // blk_size

        # M_BD: same block, same half -> bidirectional
        m_bd = (blk_q == blk_kv) & (x0_q == x0_kv)
        # M_OBC: x_t queries attend to x_0 keys from STRICTLY earlier blocks
        m_obc = (blk_q > blk_kv) & x0_kv & ~x0_q
        # M_BC: x_0 queries attend to x_0 keys from current or earlier blocks
        m_bc = (blk_q >= blk_kv) & x0_kv & x0_q

        return m_bd | m_obc | m_bc

    return create_block_mask(
        staircase_mask_mod,
        B=None, H=None,          # same mask for all batches and heads
        Q_LEN=2 * n, KV_LEN=2 * n,
        device="cuda",
    )


# ============================================================================
# Multi-Head Attention with GQA                                        [P4-7]
# ============================================================================
#
# Phase 3: Standard MHA — n_head query heads, n_head KV heads
# Phase 4: GQA — n_head query heads, n_kv_head KV heads (4:1 or 2:1 ratio)
#
# GQA reduces KV projection parameters and cache size. Multiple query heads
# share the same key/value heads, grouped evenly.
#
# Attention dispatch (via attn_mask type):
#   BlockMask    -> FlexAttention (compiled, block-sparse) [P4-6]
#   float tensor -> SDPA with explicit mask (fallback)
#   None         -> SDPA is_causal=False (inference with KV cache)
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
        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)  # [P4-7] GQA
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)  # [P4-7] GQA
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.q_norm = _make_rms_norm(head_dim)                           # [P4-9] QK-norm
        self.k_norm = _make_rms_norm(head_dim)                           # [P4-9] QK-norm
        self.kv_cache = None
        self.cache_mode = False

    def reset_cache(self):
        self.kv_cache = None

    def update_cache(self, k, v):
        if self.kv_cache is None:
            self.kv_cache = (k, v)
        else:
            old_k, old_v = self.kv_cache
            self.kv_cache = (torch.cat([old_k, k], dim=2),
                             torch.cat([old_v, v], dim=2))

    def forward(self, x, cos_sin, attn_mask=None):
        B, T, C = x.size()

        # [P4-7] GQA: q has n_head heads, k/v have n_kv_head heads
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, n_kv_head, head_dim)

        # Apply RoPE — shared positions for [x_t || x_0] during training
        cos, sin = cos_sin
        if T == cos.size(1):
            q, k = apply_rotary_emb(q, k, cos, sin)
        else:
            L = cos.size(1)
            assert T == 2 * L, f"Expected T={T} == 2*L={2*L}"
            q_t, q_0 = q[:, :L], q[:, L:]
            k_t, k_0 = k[:, :L], k[:, L:]
            q_t, k_t = apply_rotary_emb(q_t, k_t, cos, sin)
            q_0, k_0 = apply_rotary_emb(q_0, k_0, cos, sin)
            q = torch.cat([q_t, q_0], dim=1)
            k = torch.cat([k_t, k_0], dim=1)

        # [P4-9] QK-norm: learnable per-head RMSNorm prevents attention logit growth
        q, k = self.q_norm(q), self.k_norm(k)

        # Transpose to (B, H, T, D) for attention computation
        q = q.transpose(1, 2)                    # (B, n_head, T, D)
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # (B, n_kv_head, T, D)

        # Save current block's K,V BEFORE prepending cache
        k_current, v_current = k, v

        # Prepend cached K,V from finalized blocks during generation
        if self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # [P4-6] Attention dispatch based on mask type:
        #   BlockMask -> FlexAttention (compiled block-sparse kernel)
        #   float tensor -> SDPA with explicit mask
        #   None -> SDPA is_causal=False (inference)
        _gqa = n_kv_head != n_head
        if FLEX_AVAILABLE and isinstance(attn_mask, BlockMask):
            y = _compiled_flex(q, k, v, block_mask=attn_mask, enable_gqa=_gqa)
        elif attn_mask is not None:
            if _gqa:
                # Repeat KV heads to match query heads for SDPA
                repeats = n_head // n_kv_head
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            if _gqa:
                repeats = n_head // n_kv_head
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Auto-cache current block's K,V after attention
        if self.cache_mode:
            self.update_cache(k_current, v_current)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# ============================================================================
# SwiGLU (Gated Feed-Forward Network)
# ============================================================================
# Gate + up projection through SiLU, then down.
# [P4-16] mlp_hidden=2816 (Qwen3-0.6B-style), MLP dropout after down proj.

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(n_embd, mlp_hidden, bias=False)  # gate
        self.w2 = nn.Linear(n_embd, mlp_hidden, bias=False)  # up
        self.w3 = nn.Linear(mlp_hidden, n_embd, bias=False)  # down
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()  # [P4-16]

    def forward(self, x):
        # [P4-3] Liger fuses SiLU + elementwise multiply into one Triton kernel,
        # avoiding materializing a full (B, T, mlp_hidden) intermediate tensor.
        gate, up = self.w1(x), self.w2(x)
        if use_liger:
            return self.drop(self.w3(LigerSiLUMulFunction.apply(gate, up)))
        return self.drop(self.w3(F.silu(gate) * up))


# ============================================================================
# Transformer Block
# ============================================================================
# Pre-norm architecture: normalize BEFORE attention and MLP, not after.
# [DIFF] Threads attn_mask through to attention.

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = _make_rms_norm(n_embd)   # [P4-9] pre-attention norm
        self.attn = MultiHeadAttention()
        self.ln2 = _make_rms_norm(n_embd)   # [P4-9] pre-MLP norm
        self.mlp = SwiGLU()

    def _forward(self, x, cos_sin, attn_mask=None):
        x = x + self.attn(self.ln1(x), cos_sin, attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x, cos_sin, attn_mask=None):
        # [P4-2] Gradient checkpointing: recompute activations during backward
        # instead of storing them. Saves ~60% peak activation memory at the cost
        # of ~25% extra compute. Only active during training with use_grad_ckpt.
        if self.training and use_grad_ckpt:
            return grad_checkpoint(self._forward, x, cos_sin, attn_mask,
                                   use_reentrant=False)
        return self._forward(x, cos_sin, attn_mask=attn_mask)


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
        self.emb_norm = _make_rms_norm(n_embd)      # [P4-9] post-embedding norm

        # Precompute rotary embeddings: 2x context length for training
        # ([x_t || x_0]), or 4096 for long generation with KV cache
        self.rotary_seq_len = max(block_size_seq * 2, 4096)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Final norm + output projection
        self.final_norm = _make_rms_norm(n_embd)     # [P4-9] pre-lm_head norm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        # [P4-16] Tied embeddings: share input embeddings with output lm_head.
        # Saves V*D = 32768*1024 = ~33.6M params. Must come AFTER _init_weights
        # so both get initialized, then the pointer is shared.
        self.lm_head.weight = self.token_emb.weight

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
        x = self.emb_norm(x)    # [P4-9]

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
        x = self.final_norm(x)  # [P4-9]

        # During training: extract hidden states from first half (x_t predictions)
        if targets is not None:
            L = targets.size(1)
            x_pred = x[:, :L]  # (B, L, n_embd) — hidden states for x_t half
        else:
            x_pred = x

        if targets is None:
            logits = self.lm_head(x_pred)  # (B, T, vocab_size)
            loss = None
        else:
            # [P4-3] Compute per-token cross-entropy loss.
            # When Liger is available, use fused linear+CE to avoid materializing
            # the full (B*L, vocab_size) logit tensor. This saves ~1GB for
            # vocab=32768 at batch=32, seq=512.
            if use_liger:
                # Returns vary by version: (loss, z_loss, ...) — unpack first element
                liger_out = LigerFusedLinearCrossEntropyFunction.apply(
                    x_pred.contiguous().view(-1, n_embd),  # _input (B*L, D)
                    self.lm_head.weight,                    # weight (vocab, D)
                    targets.contiguous().view(-1),          # target (B*L,)
                    None,                                   # bias
                    None,                                   # ce_weight
                    -100,                                   # ignore_index
                    0.0,                                    # lse_square_scale (disabled)
                    0.0,                                    # label_smoothing
                    "none",                                 # reduction
                )
                per_token_loss = (liger_out[0] if isinstance(liger_out, tuple) else liger_out).view(B, L)
            else:
                logits = self.lm_head(x_pred)  # (B, L, vocab_size)
                per_token_loss = F.cross_entropy(
                    logits.view(-1, vocab_size), targets.view(-1), reduction="none"
                )
                per_token_loss = per_token_loss.view(B, L)

            if mask is not None and t is not None:
                # [P4-8] t contains pre-computed ELBO weights from get_batch():
                # either uniform 1/t or CART context-adaptive weights.
                elbo_weight = t  # (B, L) — pre-computed

                # [P4-19] Normalize by ALL real tokens, not just masked ones.
                # The 1/t importance weight already accounts for the masked fraction,
                # so dividing by N_masked double-counts (inflates loss ~3.3x).
                # Ref: ZHZisZZ/dllm (maskable_mask.sum()), JinjieNi/MegaDLMs (loss_mask.sum())
                weighted_loss = per_token_loss * mask.float() * elbo_weight  # (B, L)
                real_count = (targets != pad_token_id).float().sum().clamp(min=1)
                loss = weighted_loss.sum() / real_count
            elif mask is not None:
                # [P4-19] Same real-token normalization for non-ELBO path
                mask_flat = mask.view(-1).float()
                real_count = (targets != pad_token_id).float().sum().clamp(min=1)
                loss = (per_token_loss.view(-1) * mask_flat).sum() / real_count
            else:
                loss = per_token_loss.mean()

            logits = None  # don't return logits during training to save memory

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
#   Block input: [_ _ _ _]  (all <|mask|>)
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

        # Initialize block: prompt remainder (if any) + <|mask|> tokens
        block = torch.full(
            (1, block_size_blk), mask_token_id, dtype=torch.long, device=device
        )
        if fill_from_prompt > 0:
            remainder_start = n_full_prompt_blocks * block_size_blk
            block[0, :fill_from_prompt] = torch.tensor(
                prompt_ids[remainder_start:remainder_start + fill_from_prompt],
                dtype=torch.long, device=device,
            )

        # Track which positions need decoding (only the <|mask|> ones)
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

            # Never generate <|mask|> or <|padding|> tokens
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
    out = {}
    was_training = model.training
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_input, targets, mask, t, attn_mask = get_batch(split)
            # [P4-1] autocast during loss estimation to match training dtype
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x_input, targets, mask, t, attn_mask)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


# ============================================================================
# Learning Rate Schedule: WSD (Warmup-Stable-Decay)                   [P4-17]
# ============================================================================
#
#     WSD Learning Rate Schedule
#     ==========================
#     factor
#     1.0 |     ****************************
#         |    *                             ****
#         |   *                                  ***
#         |  *                                      **
#     0.0 | *                                         *
#         +--|----------|---------------------|------->
#         0  warmup     stable plateau        decay  max_iters
#
# Returns a [0, 1] multiplier. Each param group stores its own initial_lr,
# so Muon (0.02) and AdamW (6e-4) share the same schedule shape.

def get_lr_factor(step):
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    if step < decay_start:
        return 1.0
    # Linear decay from 1.0 -> 0.0 over [decay_start, max_iters)
    return max(0.0, 1.0 - (step - decay_start) / (max_iters - decay_start))


# ============================================================================
# Main: Training and Generation
# ============================================================================
#
# Training Loop
# =============
#
#   Multi-source 100B dataset (streaming)
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
#   1. Sample a batch of tokenized documents from the multi-source dataset
#   2. Apply cosine noise schedule per block: sample t, mask tokens
#   3. Forward pass through the model -> ELBO-weighted cross-entropy loss
#   4. Backward pass + gradient clipping + optimizer step
#   5. Periodically evaluate on held-out data and generate a sample
#
# Weight decay is applied only to 2D+ parameters (weight matrices), not to
# biases or normalization parameters (1D). This is standard practice: decaying
# biases and norms hurts training stability with no regularization benefit.

if __name__ == "__main__":

    # --- 1. Print configuration (master only) ---
    if master_process:
        print("=" * 60)
        print("modern_dllm — modern block diffusion language model (Phase 4)")
        print("=" * 60)
        print(f"  n_layer        = {n_layer}")
        print(f"  n_embd         = {n_embd}")
        print(f"  n_head         = {n_head}")
        print(f"  n_kv_head      = {n_kv_head} (GQA {n_head//n_kv_head}:1)")
        print(f"  mlp_hidden     = {mlp_hidden}")
        print(f"  block_size_seq = {block_size_seq}")
        print(f"  block_size_blk = {block_size_blk}")
        print(f"  num_blocks     = {num_blocks}")
        eff = 2 * batch_size * grad_accum_steps * ddp_world_size
        print(f"  batch_size     = {batch_size} (effective: {eff} = "
              f"2x{batch_size} x {grad_accum_steps} accum x {ddp_world_size} GPU)")
        print(f"  vocab_size     = {vocab_size}")
        print(f"  max_iters      = {max_iters}")
        print(f"  warmup         = {warmup_iters}, decay_start = {decay_start}")  # [P4-17]
        print(f"  lr             = {learning_rate} (Muon: 0.02, AdamW: 6e-4)")
        print(f"  dropout        = {dropout}")                             # [P4-16]
        print(f"  device         = {device}")
        print(f"  ddp            = {ddp} (world_size={ddp_world_size})")   # [P4-14]
        print(f"  use_amp        = {use_amp}")                             # [P4-1]
        print(f"  use_liger      = {use_liger}")                           # [P4-3]
        print(f"  use_flex       = {use_flex}")                            # [P4-6]
        print(f"  use_cart       = {use_cart} (p={cart_p})")               # [P4-8]
        print(f"  use_compile    = {use_compile}")                         # [P4-13]
        print(f"  use_muon       = {use_muon}")                            # [P4-11]
        print(f"  t_range        = [{t_min}, {t_max}]")                   # [P4-4]
        print("=" * 60)

    # --- 2. Set up weights path ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(
        script_dir, "weights", f"modern_dllm_b{block_size_blk}.pt"
    )
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # --- 3. Instantiate model ---
    model = Model().to(device)
    if master_process:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{param_count / 1e6:.2f}M parameters")
        if torch.cuda.is_available():                                      # [P4-15]
            print(f"Peak VRAM after model init: "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # [P4-13] torch.compile: fuse operations via Triton/inductor for 20-40% speedup.
    # reduce-overhead mode uses CUDA graphs — ideal for fixed-shape training tensors.
    # Applied before DDP wrapping; compile is transparent to DDP.
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
        if master_process:
            print("torch.compile enabled (reduce-overhead mode)")

    # [P4-14] DDP wrapper: synchronizes gradients across GPUs after each backward pass.
    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[-1])])

    # Unwrap model for weight loading and inference (DDP/compile may wrap it)
    raw_model = model.module if ddp else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model

    # --- 4. Load or train ---
    if os.path.exists(weights_path) and not args.train:
        # Load pre-trained weights (always into unwrapped model)
        raw_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        if master_process:
            print(f"Loaded weights from {weights_path}")

    elif args.train:
        # Train from scratch

        # [P4-11] Muon optimizer: split params into Muon (2D hidden weights) and
        # AdamW (embeddings, lm_head, norms, biases). Muon uses Newton-Schulz
        # orthogonalization for ~50% less optimizer memory and faster convergence.
        # MuonWithAuxAdam requires DDP; SingleDeviceMuonWithAuxAdam for single GPU.
        if use_muon:
            embed_names = {"token_emb.weight", "lm_head.weight"}
            muon_params, adam_params = [], []
            seen = set()  # deduplicate tied weights
            for name, p in raw_model.named_parameters():
                if not p.requires_grad or p.data_ptr() in seen:
                    continue
                seen.add(p.data_ptr())
                if p.ndim == 2 and name not in embed_names:
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            param_groups = [
                dict(params=muon_params, lr=0.02, momentum=0.95,
                     weight_decay=weight_decay, use_muon=True),
                dict(params=adam_params, lr=6e-4, betas=(0.9, 0.95),
                     eps=1e-10, weight_decay=weight_decay, use_muon=False),
            ]
            MuonCls = MuonWithAuxAdam if ddp else SingleDeviceMuonWithAuxAdam
            optimizer = MuonCls(param_groups)
            if master_process:
                print(f"Muon optimizer ({MuonCls.__name__}): "
                      f"{len(muon_params)} Muon params, "
                      f"{len(adam_params)} AdamW params")
        else:
            decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
            no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

        # [P4-17] Store initial_lr per param group for WSD schedule
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        # [P4-1] AMP: GradScaler prevents fp16 gradient underflow by dynamically
        # scaling the loss before backward. It checks for inf/NaN grads each step
        # and adjusts the scale factor — no manual tuning needed.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        effective_batch = 2 * batch_size * grad_accum_steps * ddp_world_size

        # [P4-14] Preload datasets + cache staircase mask on ALL ranks before training.
        # HF streaming dataset setup takes minutes — doing this inside the training loop
        # causes DDP timeout (rank 1 waits at barrier while rank 0 loads datasets in eval).
        if master_process:
            print("Preloading datasets...")
        get_batch("train")  # triggers _make_train_iter() + _cached_staircase_mask init
        get_batch("val")    # triggers _make_val_iter()
        if ddp:
            dist.barrier()  # all ranks finish data loading before training
        if master_process:
            print("Datasets ready.")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"VRAM before training: {alloc:.2f} GB allocated, "
                      f"{resv:.2f} GB reserved, {total:.1f} GB total")

        # Training loop
        t0 = time.time()
        for step in range(max_iters):
            # [P4-17] WSD schedule: factor * initial_lr per param group
            lr_factor = get_lr_factor(step)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_factor

            # [P4-14] DDP eval: ALL ranks run estimate_loss (uses raw_model, no DDP ops)
            # so no rank waits while another loads data or runs forward passes.
            # Only master prints results and generates samples.
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(raw_model)
                if master_process:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                          f"val loss {losses['val']:.4f} | lr {lr:.6f}")
                    if step > 0:
                        sample = generate(raw_model, max_new_tokens=64, temp=0.8, top_k=5)
                        print(f"--- sample ---\n{sample[:300]}\n--- end sample ---")
                    if "mps" in str(device):
                        torch.mps.empty_cache()
                    elif "cuda" in str(device):
                        torch.cuda.empty_cache()                           # [P4-15]

            # [P4-12] Gradient accumulation: accumulate gradients over micro-steps,
            # scale loss by 1/grad_accum_steps so total gradient magnitude is correct.
            # [P4-14] DDP no_sync: skip all_reduce on intermediate micro-steps. Only
            # the last micro-step triggers NCCL all_reduce to average gradients.
            for micro_step in range(grad_accum_steps):
                no_sync = ddp and micro_step < grad_accum_steps - 1
                ctx = model.no_sync() if no_sync else contextlib.nullcontext()
                with ctx:
                    x_input, targets, mask, elbo_w, attn_mask = get_batch("train")
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits, loss = model(x_input, targets, mask, elbo_w, attn_mask)
                        loss = loss / grad_accum_steps
                    scaler.scale(loss).backward()

            # Gradient clipping: unscale first so clip threshold is in true gradient scale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step + zero gradients (set_to_none=True is faster than zeroing)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Progress indicator every 10 steps (master only)
            if master_process and step % 10 == 0 and step > 0:
                dt = time.time() - t0
                tokens_per_sec = (step * effective_batch * block_size_seq) / dt
                vram_info = ""
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    alloc = torch.cuda.memory_allocated() / 1e9
                    vram_info = f" | VRAM {alloc:.1f}/{peak:.1f} GB"
                print(f"  step {step:5d} | loss {loss.item() * grad_accum_steps:.4f} "
                      f"| {tokens_per_sec:.0f} tok/s{vram_info}")
            # Log peak VRAM after first complete step
            if step == 0 and master_process and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"VRAM after step 0: {alloc:.2f} GB allocated, "
                      f"{peak:.2f} GB peak, {resv:.2f} GB reserved, "
                      f"{total:.1f} GB total ({peak/total*100:.0f}% used)")

        # Save trained weights (master only, unwrapped model)
        if master_process:
            torch.save(raw_model.state_dict(), weights_path)
            print(f"Saved weights to {weights_path}")

    elif args.prompt is not None:
        # Generation requested but no weights found
        print(f"No weights found at {weights_path}")
        print("Run with --train to train from scratch.")
        sys.exit(1)

    # --- 5. Generate text (master only) ---
    if master_process and args.prompt is not None:
        sample = generate(raw_model, max_new_tokens=args.max_tokens, prompt=args.prompt,
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
        # [P4-5] Complementary masking doubles batch: 2*B samples per step
        print(f"\n=== Batch Shape Verification ===")
        print(f"block_size_seq={block_size_seq}, block_size_blk={_saved_blk}, "
              f"num_blocks={block_size_seq // _saved_blk}, batch_size={batch_size}")
        B2 = 2 * batch_size
        L = block_size_seq
        print(f"  x_input:   (2B, 2*L) = ({B2}, {2 * L})  [comp. masking]")
        print(f"  targets:   (2B, L)   = ({B2}, {L})")
        print(f"  mask:      (2B, L)   = ({B2}, {L})")
        print(f"  t:         (2B, L)   = ({B2}, {L})")
        mask_type = "BlockMask (FlexAttention)" if use_flex else f"float ({2*L}, {2*L})"
        print(f"  attn_mask: {mask_type}")
        print(f"\n=== GQA Configuration ===")
        print(f"  n_head={n_head}, n_kv_head={n_kv_head}, ratio={n_head//n_kv_head}:1")
        print(f"  Q proj: ({n_embd}, {n_head * head_dim})")
        print(f"  K proj: ({n_embd}, {n_kv_head * head_dim})")
        print(f"  V proj: ({n_embd}, {n_kv_head * head_dim})")

    # [P4-14] Clean up DDP process group
    if ddp:
        dist.destroy_process_group()
