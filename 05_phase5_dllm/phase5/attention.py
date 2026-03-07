"""
Attention module: RoPE, staircase masks, Multi-Head Attention with GQA + Gated Query.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import (
    n_embd, n_head, n_kv_head, head_dim, seq_len, block_size,
    use_flex, use_liger,
    _compiled_flex, _FLEX_AVAILABLE, BlockMask, create_block_mask,
)

try:
    from .optim import _MaxLogitsTracker
except ImportError:
    _MaxLogitsTracker = None


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================

def _apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


def apply_rotary_emb(q, k, cos, sin):
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


# ============================================================================
# Staircase Attention Mask
# ============================================================================
# BD3-LMs staircase for [x_t || x_0] of length 2n:
#   M_BD  (block-diagonal): same block, same half -> bidirectional
#   M_OBC (offset block-causal): x_t attends to x_0 from STRICTLY EARLIER blocks
#   M_BC  (block-causal): x_0 attends to x_0 causally (current + earlier)

def build_staircase_mask(seq_len, blk_size):
    n = seq_len
    total = 2 * n

    pos = torch.arange(total)
    q = pos.unsqueeze(1)
    kv = pos.unsqueeze(0)

    x0_flag_q = (q >= n)
    x0_flag_kv = (kv >= n)

    block_q = (q % n) // blk_size
    block_kv = (kv % n) // blk_size

    # M_BD: same block, same half
    m_bd = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    # M_OBC: x_t queries attend to x_0 keys from STRICTLY earlier blocks (> not >=)
    m_obc = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
    # M_BC: x_0 queries attend to x_0 keys from current or earlier blocks
    m_bc = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    allow = m_bd | m_obc | m_bc
    return torch.where(allow, 0.0, float('-inf'))


def build_staircase_block_mask(seq_len, blk_size, doc_ids=None):
    """FlexAttention BlockMask for staircase pattern.

    Args:
        seq_len: real token count (L). Attention over 2L positions.
        blk_size: diffusion block size.
        doc_ids: (B, L) int tensor of document IDs for packing. None = single doc.
    """
    n = seq_len

    if doc_ids is not None:
        # doc_ids lives on CUDA for the mask_mod closure
        _doc_ids = doc_ids.to("cuda")

        def staircase_mask_mod(b, h, q_idx, kv_idx):
            x0_q = (q_idx >= n)
            x0_kv = (kv_idx >= n)
            blk_q = (q_idx % n) // blk_size
            blk_kv = (kv_idx % n) // blk_size

            m_bd = (blk_q == blk_kv) & (x0_q == x0_kv)
            m_obc = (blk_q > blk_kv) & x0_kv & ~x0_q
            m_bc = (blk_q >= blk_kv) & x0_kv & x0_q
            staircase = m_bd | m_obc | m_bc

            same_doc = _doc_ids[b, q_idx % n] == _doc_ids[b, kv_idx % n]
            return staircase & same_doc

        B = doc_ids.shape[0]
        return create_block_mask(
            staircase_mask_mod,
            B=B, H=None,
            Q_LEN=2 * n, KV_LEN=2 * n,
            device="cuda",
        )
    else:
        def staircase_mask_mod(b, h, q_idx, kv_idx):
            x0_q = (q_idx >= n)
            x0_kv = (kv_idx >= n)
            blk_q = (q_idx % n) // blk_size
            blk_kv = (kv_idx % n) // blk_size

            m_bd = (blk_q == blk_kv) & (x0_q == x0_kv)
            m_obc = (blk_q > blk_kv) & x0_kv & ~x0_q
            m_bc = (blk_q >= blk_kv) & x0_kv & x0_q
            return m_bd | m_obc | m_bc

        return create_block_mask(
            staircase_mask_mod,
            B=None, H=None,
            Q_LEN=2 * n, KV_LEN=2 * n,
            device="cuda",
        )


def _visualize_mask(mask, seq_len, blk_size):
    n = seq_len
    total = 2 * n
    assert mask.shape == (total, total)

    num_blocks = (n + blk_size - 1) // blk_size

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
            blen = min(blk_size, n - b * blk_size)
            pad = blen // 2 - len(label) // 2
            header_blk += " " * pad + label + " " * (blen - pad - len(label))
    print(header_blk)

    for r in range(total):
        half = "x_t" if r < n else "x_0"
        b_idx = (r % n) // blk_size
        pos_in_blk = (r % n) % blk_size
        if pos_in_blk == 0:
            label = f"{half} b{b_idx}: "
        else:
            label = "         "
        row_chars = ""
        for c in range(total):
            row_chars += "." if mask[r, c] == 0.0 else "X"
        print(label + row_chars)


# ============================================================================
# Multi-Head Attention with GQA + Gated Query Attention
# ============================================================================
# GQA: n_head=9 query heads, n_kv_head=3 KV heads (3:1 ratio)
# Gated Query Attention (arXiv:2505.06708): sigmoid gate on SDPA output
#   before output projection, eliminates attention sinks.
#
# Attention dispatch:
#   BlockMask -> FlexAttention (compiled block-sparse)
#   float tensor -> SDPA with explicit mask
#   None -> SDPA is_causal=False (inference with KV cache)
#
# RoPE with positions argument for document packing:
#   positions=None: slice cos/sin[0..L-1] (Phase 4 behavior)
#   positions=(B,L): gather per-token cos/sin (reset at doc boundaries)
#   Training [x_t || x_0]: cat positions twice for both halves

class MultiHeadAttention(nn.Module):
    def __init__(self, make_norm_fn):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)

        # QK-norm: learnable per-head RMSNorm
        self.q_norm = make_norm_fn(head_dim)
        self.k_norm = make_norm_fn(head_dim)

        # Gated Query Attention: zero-init gate -> identity at init
        self.w_gate = nn.Linear(n_embd, n_head * head_dim, bias=False)

        # KV cache for generation
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

    def forward(self, x, cos, sin, attn_mask=None, positions=None):
        B, T, C = x.size()
        x_pre = x  # save for gate

        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, n_kv_head, head_dim)

        # Apply RoPE
        if positions is not None:
            # Per-token positions from document packing
            # cos, sin are (1, max_pos, 1, D/2) — gather per position
            pos = positions  # (B, T)
            cos_pos = cos[0, pos, 0, :]  # (B, T, D/2)
            sin_pos = sin[0, pos, 0, :]
            cos_pos = cos_pos[:, :, None, :]  # (B, T, 1, D/2)
            sin_pos = sin_pos[:, :, None, :]
            q = _apply_rotary_emb(q, cos_pos, sin_pos)
            k = _apply_rotary_emb(k, cos_pos[:, :, :1, :].expand(-1, -1, n_kv_head, -1),
                                  sin_pos[:, :, :1, :].expand(-1, -1, n_kv_head, -1))
        elif T == cos.size(1):
            # Inference or single-half: positions 0..T-1
            q, k = apply_rotary_emb(q, k, cos, sin)
        else:
            # Training [x_t || x_0]: both halves share positions 0..L-1
            L = cos.size(1)
            assert T == 2 * L, f"Expected T={T} == 2*L={2*L}"
            q_t, q_0 = q[:, :L], q[:, L:]
            k_t, k_0 = k[:, :L], k[:, L:]
            q_t, k_t = apply_rotary_emb(q_t, k_t, cos, sin)
            q_0, k_0 = apply_rotary_emb(q_0, k_0, cos, sin)
            q = torch.cat([q_t, q_0], dim=1)
            k = torch.cat([k_t, k_0], dim=1)

        # QK-norm
        q, k = self.q_norm(q), self.k_norm(k)

        # Track attention logit upper bound for QK-Clip
        # (works with any attention backend, including FlexAttention)
        if self.training and _MaxLogitsTracker is not None:
            _MaxLogitsTracker._update(
                q.detach().float().norm(p=2, dim=-1).max().item()
                * k.detach().float().norm(p=2, dim=-1).max().item()
                / math.sqrt(head_dim)
            )

        # Transpose to (B, H, T, D)
        q = q.transpose(1, 2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        # Save current K,V before prepending cache
        k_current, v_current = k, v

        if self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Attention dispatch
        _gqa = n_kv_head != n_head
        if _FLEX_AVAILABLE and use_flex and isinstance(attn_mask, BlockMask):
            y = _compiled_flex(q, k, v, block_mask=attn_mask, enable_gqa=_gqa)
        elif attn_mask is not None:
            if _gqa:
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

        # Gated Query Attention: sigmoid gate before output projection
        # w_gate is zero-init -> sigmoid(0) = 0.5 at init (halves SDPA output)
        gate = self.w_gate(x_pre)  # (B, T, n_head * head_dim)
        gate = gate.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, H, T, D)
        y = y * torch.sigmoid(gate)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
