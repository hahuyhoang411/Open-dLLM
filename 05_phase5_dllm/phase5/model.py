"""
Model: RMSNorm, SwiGLU MLP, Transformer Block, full Model with tied embeddings.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import (
    n_layer, n_embd, n_head, head_dim, n_kv_head, mlp_hidden,
    vocab_size, seq_len, block_size, pad_token_id,
    dropout, use_liger, use_grad_ckpt,
)
from .attention import MultiHeadAttention

try:
    from .loss import compute_loss
except ImportError:
    compute_loss = None


# ============================================================================
# RMSNorm
# ============================================================================

def _make_rms_norm(dim):
    if use_liger:
        from liger_kernel.transformers import LigerRMSNorm
        return LigerRMSNorm(dim)
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    class _RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
        def forward(self, x):
            return F.rms_norm(x, (x.size(-1),)) * self.weight
    return _RMSNorm(dim)


# ============================================================================
# SwiGLU (Gated Feed-Forward Network)
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(n_embd, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if use_liger:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
            return self.drop(self.down_proj(LigerSiLUMulFunction.apply(gate, up)))
        return self.drop(self.down_proj(F.silu(gate) * up))


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = _make_rms_norm(n_embd)
        self.attn = MultiHeadAttention(make_norm_fn=_make_rms_norm)
        self.mlp_norm = _make_rms_norm(n_embd)
        self.mlp = SwiGLU()

    def _forward(self, x, cos, sin, attn_mask=None, positions=None):
        x = x + self.attn(self.attn_norm(x), cos, sin,
                          attn_mask=attn_mask, positions=positions)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward(self, x, cos, sin, attn_mask=None, positions=None):
        if self.training and use_grad_ckpt:
            return grad_checkpoint(self._forward, x, cos, sin, attn_mask, positions,
                                   use_reentrant=False)
        return self._forward(x, cos, sin, attn_mask=attn_mask, positions=positions)


# ============================================================================
# Model
# ============================================================================

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.emb_norm = _make_rms_norm(n_embd)

        # Precompute RoPE for max(2*seq_len, 8192) positions
        rotary_len = max(seq_len * 2, 8192)
        cos, sin = self._precompute_rotary(rotary_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.final_norm = _make_rms_norm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Zero-init Gated Query Attention gates: sigmoid(0)=0.5 dampens attention at start
        for block in self.blocks:
            nn.init.zeros_(block.attn.w_gate.weight)
        # Tied embeddings AFTER init so both get initialized, then pointer shared
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary(self, length, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # (1, length, 1, head_dim//2)
        return cos[None, :, None, :], sin[None, :, None, :]

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.cache_mode = True

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def set_cache_mode(self, enabled):
        """Toggle auto-caching of K,V without destroying existing cache."""
        for block in self.blocks:
            block.attn.cache_mode = enabled

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.cache_mode = False
            block.attn.kv_cache = None

    def count_params(self):
        # Deduplicate tied weights by data_ptr
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def forward(self, idx, targets=None, mask=None, elbo_weight=None,
                attn_mask=None, positions=None, pos_offset=0):
        """
        Args:
            idx: (B, T) token IDs. T=2L during training ([x_t || x_0]).
            targets: (B, L) ground-truth IDs. None during generation.
            mask: (B, L) bool — True where tokens were masked.
            elbo_weight: (B, L) pre-computed ELBO importance weights.
            attn_mask: BlockMask, float tensor, or None.
            positions: (B, L) per-token RoPE positions from doc packing. None = sequential.
            pos_offset: RoPE offset for KV-cached generation.
        """
        B, T = idx.size()

        x = self.token_emb(idx)
        x = self.emb_norm(x)

        # RoPE slicing
        if positions is not None:
            # Document packing: per-token positions, doubled for [x_t || x_0]
            if targets is not None:
                L = targets.size(1)
                pos_doubled = torch.cat([positions, positions], dim=1)  # (B, 2L)
            else:
                pos_doubled = positions
            cos, sin = self.cos, self.sin
            rope_positions = pos_doubled
        else:
            rope_positions = None
            if targets is not None:
                L = targets.size(1)
                cos = self.cos[:, :L]
                sin = self.sin[:, :L]
            else:
                cos = self.cos[:, pos_offset:pos_offset + T]
                sin = self.sin[:, pos_offset:pos_offset + T]

        for block in self.blocks:
            x = block(x, cos, sin, attn_mask=attn_mask, positions=rope_positions)
        x = self.final_norm(x)

        # Training: extract x_t half predictions
        if targets is not None:
            L = targets.size(1)
            x_pred = x[:, :L]
        else:
            x_pred = x

        if targets is None:
            logits = self.lm_head(x_pred)
            return logits, None

        # Loss computation
        if compute_loss is not None:
            loss = compute_loss(x_pred, targets, mask, elbo_weight,
                                self.lm_head.weight, use_liger=use_liger)
        else:
            # Inline fallback CE
            logits = self.lm_head(x_pred)
            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1), reduction="none"
            ).view(B, L)

            if mask is not None and elbo_weight is not None:
                weighted = per_token_loss * mask.float() * elbo_weight
                real_count = (targets != pad_token_id).float().sum().clamp(min=1)
                loss = weighted.sum() / real_count
            elif mask is not None:
                real_count = (targets != pad_token_id).float().sum().clamp(min=1)
                loss = (per_token_loss * mask.float()).sum() / real_count
            else:
                loss = per_token_loss.mean()

        return None, loss
