"""Attention module: RoPE, staircase masks, Multi-Head Attention with GQA + optional Gated Query."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, _check_flex

# Pre-compile flex_attention once (lazy singleton, not per-forward)
_compiled_flex = None


def _get_compiled_flex():
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================

def _apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
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

def build_staircase_mask(seq_len, blk_size, doc_ids=None):
    """Dense staircase mask. If doc_ids provided, enforces doc boundaries per batch row.

    Args:
        seq_len: L (real token count). Mask is (2L, 2L) or (B, 2L, 2L).
        blk_size: diffusion block size.
        doc_ids: (B, L) int tensor. If given, returns (B, 1, 2L, 2L) for broadcast over heads.
    """
    n = seq_len
    total = 2 * n

    _device = doc_ids.device if doc_ids is not None else None
    pos = torch.arange(total, device=_device)
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

    allow = m_bd | m_obc | m_bc  # (2L, 2L)

    if doc_ids is not None:
        # Enforce doc boundaries: both halves use same doc_ids
        # doc_ids: (B, L) -> expand to (B, 2L) positions via % n indexing
        d_q = doc_ids[:, pos.unsqueeze(1) % n]   # (B, 2L, 1) via broadcast
        d_kv = doc_ids[:, pos.unsqueeze(0) % n]  # (B, 1, 2L)
        same_doc = (d_q == d_kv)                  # (B, 2L, 2L)
        allow = allow.unsqueeze(0) & same_doc     # (B, 2L, 2L)
        return torch.where(allow, 0.0, float('-inf')).unsqueeze(1)  # (B, 1, 2L, 2L)

    return torch.where(allow, 0.0, float('-inf'))


def build_staircase_block_mask(seq_len, blk_size, doc_ids=None):
    """FlexAttention BlockMask for staircase pattern.

    Args:
        seq_len: real token count (L). Attention over 2L positions.
        blk_size: diffusion block size.
        doc_ids: (B, L) int tensor of document IDs for packing. None = single doc.
    """
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    n = seq_len

    if doc_ids is not None:
        _doc_ids = doc_ids.to('cuda')

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
            device='cuda',
        )

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
        device='cuda',
    )


# ============================================================================
# Multi-Head Attention with GQA + optional Gated Query Attention
# ============================================================================
# GQA: n_head query heads, n_kv_head KV heads
# Gated Query Attention (arXiv:2505.06708): sigmoid gate on SDPA output
#   before output projection, eliminates attention sinks.
#
# Attention dispatch:
#   BlockMask -> FlexAttention (compiled block-sparse)
#   float tensor -> SDPA with explicit mask
#   None -> SDPA is_causal=False (inference with KV cache)
#
# RoPE with positions argument for document packing:
#   positions=None: slice cos/sin[0..L-1] (simple case)
#   positions=(B,L): gather per-token cos/sin (reset at doc boundaries)
#   Training [x_t || x_0]: cat positions twice for both halves

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_embd = cfg.n_embd
        n_head = cfg.n_head
        n_kv_head = cfg.n_kv_head
        head_dim = cfg.head_dim

        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)

        # QK-norm: learnable per-head RMSNorm (conditional)
        if cfg.use_qk_norm:
            self.q_norm = _make_rms_norm(head_dim, cfg)
            self.k_norm = _make_rms_norm(head_dim, cfg)
        else:
            self.q_norm = None
            self.k_norm = None

        # Gated Query Attention: zero-init gate -> identity at init (conditional)
        if cfg.use_gated_query:
            self.w_gate = nn.Linear(n_embd, n_head * head_dim, bias=False)
        else:
            self.w_gate = None

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
        cfg = self.cfg
        n_head = cfg.n_head
        n_kv_head = cfg.n_kv_head
        head_dim = cfg.head_dim

        B, T, C = x.size()
        x_pre = x  # save for gate

        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, n_kv_head, head_dim)

        # QK-norm BEFORE RoPE (Qwen3 order: proj -> norm -> RoPE)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE
        if positions is not None:
            pos = positions  # (B, T)
            cos_pos = cos[0, pos, 0, :]  # (B, T, D/2)
            sin_pos = sin[0, pos, 0, :]
            cos_pos = cos_pos[:, :, None, :]  # (B, T, 1, D/2)
            sin_pos = sin_pos[:, :, None, :]
            q = _apply_rotary_emb(q, cos_pos, sin_pos)
            k = _apply_rotary_emb(k, cos_pos[:, :, :1, :].expand(-1, -1, n_kv_head, -1),
                                  sin_pos[:, :, :1, :].expand(-1, -1, n_kv_head, -1))
        elif cos.size(1) == T:
            q, k = apply_rotary_emb(q, k, cos, sin)
        else:
            # Training [x_t || x_0]: both halves share positions 0..L-1
            L = cos.size(1)
            assert T == 2 * L, f'Expected T={T} == 2*L={2 * L}'
            q_t, q_0 = q[:, :L], q[:, L:]
            k_t, k_0 = k[:, :L], k[:, L:]
            q_t, k_t = apply_rotary_emb(q_t, k_t, cos, sin)
            q_0, k_0 = apply_rotary_emb(q_0, k_0, cos, sin)
            q = torch.cat([q_t, q_0], dim=1)
            k = torch.cat([k_t, k_0], dim=1)

        # Track attention logit upper bound for QK-Clip
        # Keep as 0-d GPU tensor — no .item() here (graph breaks + recompilation
        # inside torch.compile + grad_checkpoint → CUDA illegal memory access).
        # .item() deferred to _MaxLogitsTracker.consume() in optimizer.step().
        if self.training:
            try:
                from .optim import _MaxLogitsTracker
                if _MaxLogitsTracker is not None:
                    with torch.no_grad():
                        logit_bound = (
                            q.float().norm(p=2, dim=-1).max()
                            * k.float().norm(p=2, dim=-1).max()
                            / math.sqrt(cfg.head_dim)
                        )
                    _MaxLogitsTracker._update(logit_bound)
            except ImportError:
                pass

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

        # Check FlexAttention availability
        _flex_available = _check_flex()
        if _flex_available:
            from torch.nn.attention.flex_attention import BlockMask
            _is_block_mask = isinstance(attn_mask, BlockMask)
        else:
            _is_block_mask = False

        if _flex_available and cfg.use_flex and _is_block_mask:
            y = _get_compiled_flex()(q, k, v, block_mask=attn_mask, enable_gqa=_gqa)
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

        # Gated Query Attention (conditional)
        if self.w_gate is not None:
            gate = self.w_gate(x_pre)  # (B, T, n_head * head_dim)
            gate = gate.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, H, T, D)
            y = y * torch.sigmoid(gate)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# ============================================================================
# RMSNorm factory (shared with model.py)
# ============================================================================

def _make_rms_norm(dim, cfg: Config):
    eps = cfg.rms_eps
    if cfg.use_liger:
        try:
            from liger_kernel.transformers import LigerRMSNorm
            return LigerRMSNorm(dim, eps=eps)
        except ImportError:
            pass
    if hasattr(nn, 'RMSNorm'):
        return nn.RMSNorm(dim, eps=eps)

    class _RMSNorm(nn.Module):
        def __init__(self, d, e):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self._eps = e

        def forward(self, x):
            return F.rms_norm(x, (x.size(-1),), self.weight, self._eps)
    return _RMSNorm(dim, eps)
