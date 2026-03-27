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

from .attention import MultiHeadAttention, _make_rms_norm
from .config import Config


# ============================================================================
# Selective Activation Checkpointing (SAC)
# ============================================================================
# Save outputs of expensive ops (matmuls, attention); recompute cheap ops
# (norms, activations, residual adds). Cuts grad_ckpt overhead from ~33%
# to ~2-15% while preserving most memory savings.
# Ref: torchtitan/distributed/activation_checkpoint.py

_SAC_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten.max.default,
}

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
# SwiGLU (Gated Feed-Forward Network)
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.cfg.use_liger:
            try:
                from liger_kernel.ops.swiglu import LigerSiLUMulFunction
                return self.drop(self.down_proj(LigerSiLUMulFunction.apply(gate, up)))
            except ImportError:
                pass
        return self.drop(self.down_proj(F.silu(gate) * up))


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = _make_rms_norm(cfg.n_embd, cfg)
        self.attn = MultiHeadAttention(cfg)
        self.mlp_norm = _make_rms_norm(cfg.n_embd, cfg)
        self.mlp = SwiGLU(cfg)

    def _forward(self, x, cos, sin, attn_mask=None, positions=None):
        x = x + self.attn(self.attn_norm(x), cos, sin,
                          attn_mask=attn_mask, positions=positions)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward(self, x, cos, sin, attn_mask=None, positions=None):
        if self.training and self.cfg.use_grad_ckpt:
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Embedding norm (Qwen3 doesn't use it; SmolLM2 does)
        if cfg.use_emb_norm:
            self.emb_norm = _make_rms_norm(cfg.n_embd, cfg)
        else:
            self.emb_norm = None

        # Precompute RoPE for max(2*seq_len, 8192) positions
        rotary_len = max(cfg.seq_len * 2, 8192)
        cos, sin = self._precompute_rotary(rotary_len, cfg.rope_base, cfg.head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = _make_rms_norm(cfg.n_embd, cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(lambda m: self._init_weights(m, cfg))

        # Scaled init for residual projections: std / sqrt(2 * n_layers)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

        # Zero-init Gated Query Attention gates if present
        if cfg.use_gated_query:
            for block in self.blocks:
                nn.init.zeros_(block.attn.w_gate.weight)

        # Tied embeddings AFTER init so both get initialized, then pointer shared
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module, cfg):
        std = 0.02  # Qwen3 default
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    @staticmethod
    def _precompute_rotary(length, base, head_dim):
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
        for block in self.blocks:
            block.attn.cache_mode = enabled

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.cache_mode = False
            block.attn.kv_cache = None

    def count_params(self):
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def forward(self, idx, targets=None, attn_mask=None, positions=None, pos_offset=0):
        """Forward pass for training ([x_t || x_0]) and generation (single block).

        Training: returns (hidden_states, None) -- loss computed externally.
        Generation: returns (logits, None) -- lm_head applied internally.
        """
        cfg = self.cfg
        B, T = idx.size()

        x = self.token_emb(idx)
        if self.emb_norm is not None:
            x = self.emb_norm(x)

        # RoPE slicing
        if positions is not None:
            if targets is not None:
                L = targets.size(1)
                pos_doubled = torch.cat([positions, positions], dim=1)
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

        # Training: return hidden states -- loss computed outside model
        return x_pred, None
