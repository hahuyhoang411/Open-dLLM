"""Model: RMSNorm, SwiGLU MLP, Transformer Block, full Model with tied embeddings."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import (
    checkpoint as grad_checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from .attention import MultiHeadAttention
from .config import (
    dropout,
    head_dim,
    mlp_hidden,
    n_embd,
    n_layer,
    seq_len,
    use_grad_ckpt,
    use_liger,
    vocab_size,
)


# ============================================================================
# Selective Activation Checkpointing (SAC)
# ============================================================================
# Save outputs of expensive ops (matmuls, attention); recompute cheap ops
# (norms, activations, residual adds). Cuts grad_ckpt overhead from ~33%
# to ~2-15% while preserving most memory savings.
# Ref: torchtitan/distributed/activation_checkpoint.py

_SAC_SAVE_OPS = {
    torch.ops.aten.mm.default,                                           # nn.Linear (bias=False)
    torch.ops.aten.addmm.default,                                       # nn.Linear (bias=True, defensive)
    torch.ops.aten._scaled_mm.default,                                   # FP8 matmul (if not opaque)
    torch.ops.aten._scaled_dot_product_flash_attention.default,          # SDPA flash
    torch.ops.aten._scaled_dot_product_efficient_attention.default,      # SDPA efficient
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,          # SDPA cuDNN
    torch.ops.aten._scaled_dot_product_attention_math.default,           # SDPA math fallback
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten.max.default,                                          # abs max (scaling factors, etc.)
}

# FlexAttention is a Higher-Order Op — add if available (PyTorch 2.10+, fixed in PR #150080)
try:
    _SAC_SAVE_OPS.add(torch._higher_order_ops.flex_attention)
except AttributeError:
    pass


def _sac_policy(ctx, op, *args, **kwargs):
    if op in _SAC_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _sac_context_fn():
    return create_selective_checkpoint_contexts(_sac_policy)


# ============================================================================
# RMSNorm
# ============================================================================

_RMS_EPS = 1e-5  # SmolLM2 spec


def _make_rms_norm(dim):
    if use_liger:
        from liger_kernel.transformers import LigerRMSNorm
        return LigerRMSNorm(dim, eps=_RMS_EPS)
    if hasattr(nn, 'RMSNorm'):
        return nn.RMSNorm(dim, eps=_RMS_EPS)

    class _RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return F.rms_norm(x, (x.size(-1),), self.weight, _RMS_EPS)
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
            return grad_checkpoint(
                self._forward, x, cos, sin, attn_mask, positions,
                use_reentrant=False,
                context_fn=_sac_context_fn,
            )
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
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.final_norm = _make_rms_norm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Scaled init for residual projections: std / sqrt(2 * n_layers)
        # Ref: SmolLM2 vgpt2, LLaDA modeling_llada.py:129,155
        residual_std = (1.0 / math.sqrt(n_embd)) / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
        # Zero-init Gated Query Attention gates: sigmoid(0)=0.5 dampens attention at start
        for block in self.blocks:
            nn.init.zeros_(block.attn.w_gate.weight)
        # Tied embeddings AFTER init so both get initialized, then pointer shared
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        # SmolLM2-135M: std = 1/sqrt(hidden_size) = 1/sqrt(576) = 0.0417
        std = 1.0 / math.sqrt(n_embd)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

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

    def forward(self, idx, targets=None, attn_mask=None, positions=None, pos_offset=0):
        """Forward pass for training ([x_t || x_0]) and generation (single block).

        Training: returns (hidden_states, None) — loss computed externally.
        Generation: returns (logits, None) — lm_head applied internally.
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

        # Training: return hidden states — loss computed outside model
        # (keeps Liger FLCE .item() graph breaks outside compiled blocks)
        return x_pred, None
