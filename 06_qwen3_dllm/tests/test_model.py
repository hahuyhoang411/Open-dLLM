"""Tests for Phase 6 model and attention modules."""

import math
import pytest
import torch

from phase6.config import Config


# Small config for fast tests
def small_cfg(**overrides):
    defaults = dict(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000.0, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        mask_token_id=0, pad_token_id=2,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ============================================================================
# 1. Model instantiation
# ============================================================================

def test_model_instantiation():
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)
    assert model is not None


def test_model_instantiation_qwen3_config():
    """Default Qwen3-0.6B config doesn't crash."""
    from phase6.model import Model
    cfg = Config()
    model = Model(cfg)
    assert model is not None


# ============================================================================
# 2. Forward pass shapes -- training
# ============================================================================

def test_forward_training_shape():
    """Training input (B=2, 2L) -> hidden_states (B=2, L, n_embd)."""
    from phase6.model import Model
    from phase6.attention import build_staircase_mask
    cfg = small_cfg()
    model = Model(cfg)
    model.eval()

    B, L = 2, cfg.seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
    targets = torch.randint(0, cfg.vocab_size, (B, L))
    mask = build_staircase_mask(L, cfg.block_size)

    with torch.no_grad():
        out, _ = model(idx, targets=targets, attn_mask=mask)

    assert out.shape == (B, L, cfg.n_embd), f"Expected (2, {L}, {cfg.n_embd}), got {out.shape}"


# ============================================================================
# 3. Forward pass shapes -- generation
# ============================================================================

def test_forward_generation_shape():
    """Generation input (B=1, T=8) -> logits (B=1, 8, vocab_size)."""
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)
    model.eval()

    B, T = 1, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(idx)

    assert logits.shape == (B, T, cfg.vocab_size), \
        f"Expected (1, 8, {cfg.vocab_size}), got {logits.shape}"


# ============================================================================
# 4. Param count
# ============================================================================

def test_param_count():
    """Verify param count deduplication works with tied embeddings."""
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)

    counted = model.count_params()
    assert counted > 0
    # Tied embeddings: deduplicated count <= total named params
    all_params = sum(p.numel() for p in model.parameters())
    assert counted <= all_params


def test_tied_embeddings():
    """lm_head.weight should be the same tensor as token_emb.weight."""
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


# ============================================================================
# 5. Staircase mask: M_OBC uses strict > (not >=)
# ============================================================================

def test_staircase_mask_obc_strict_greater():
    """M_OBC: x_t queries attend to x_0 keys from STRICTLY earlier blocks.
    Block 0 of x_t should NOT see block 0 of x_0 (that would be >= leakage).
    """
    from phase6.attention import build_staircase_mask
    L, blk = 16, 4  # 4 blocks of 4 tokens
    mask = build_staircase_mask(L, blk)

    # x_t block 0 = positions [0..3], x_0 block 0 = positions [16..19]
    # M_OBC should NOT allow x_t block 0 to attend x_0 block 0
    for qt in range(0, 4):
        for kv0 in range(L, L + 4):
            assert mask[qt, kv0] == float('-inf'), \
                f"x_t[{qt}] should NOT attend x_0[{kv0}] (same block, OBC leakage)"

    # x_t block 1 = positions [4..7], x_0 block 0 = positions [16..19]
    # M_OBC SHOULD allow x_t block 1 to attend x_0 block 0
    for qt in range(4, 8):
        for kv0 in range(L, L + 4):
            assert mask[qt, kv0] == 0.0, \
                f"x_t[{qt}] SHOULD attend x_0[{kv0}] (earlier block)"


# ============================================================================
# 6. RoPE output shape
# ============================================================================

def test_rope_output_shape():
    """RoPE should preserve input shape with head_dim=128."""
    from phase6.attention import _apply_rotary_emb
    B, T, H, D = 2, 16, 4, 128
    x = torch.randn(B, T, H, D)
    cos = torch.randn(1, T, 1, D // 2)
    sin = torch.randn(1, T, 1, D // 2)
    y = _apply_rotary_emb(x, cos, sin)
    assert y.shape == (B, T, H, D)


def test_rope_head_dim_32():
    """RoPE works with small head_dim=32 too."""
    from phase6.attention import _apply_rotary_emb
    B, T, H, D = 2, 8, 4, 32
    x = torch.randn(B, T, H, D)
    cos = torch.randn(1, T, 1, D // 2)
    sin = torch.randn(1, T, 1, D // 2)
    y = _apply_rotary_emb(x, cos, sin)
    assert y.shape == (B, T, H, D)


# ============================================================================
# 7. Gated Query Attention toggle
# ============================================================================

def test_gated_query_enabled():
    """Model with use_gated_query=True has w_gate parameters."""
    from phase6.model import Model
    cfg = small_cfg(use_gated_query=True)
    model = Model(cfg)
    assert hasattr(model.blocks[0].attn, 'w_gate')
    assert model.blocks[0].attn.w_gate is not None


def test_gated_query_disabled():
    """Model with use_gated_query=False has no w_gate."""
    from phase6.model import Model
    cfg = small_cfg(use_gated_query=False)
    model = Model(cfg)
    assert model.blocks[0].attn.w_gate is None


def test_gated_query_forward():
    """Forward pass works with gated query enabled."""
    from phase6.model import Model
    cfg = small_cfg(use_gated_query=True)
    model = Model(cfg)
    model.eval()

    B, T = 1, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)


# ============================================================================
# 8. Embedding norm toggle
# ============================================================================

def test_emb_norm_enabled():
    """Model with use_emb_norm=True applies embedding norm."""
    from phase6.model import Model
    cfg = small_cfg(use_emb_norm=True)
    model = Model(cfg)
    assert model.emb_norm is not None


def test_emb_norm_disabled():
    """Model with use_emb_norm=False skips embedding norm."""
    from phase6.model import Model
    cfg = small_cfg(use_emb_norm=False)
    model = Model(cfg)
    assert model.emb_norm is None


def test_emb_norm_forward():
    """Forward pass works with emb_norm enabled."""
    from phase6.model import Model
    cfg = small_cfg(use_emb_norm=True)
    model = Model(cfg)
    model.eval()

    B, T = 1, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)


# ============================================================================
# 9. Q projection output dim = n_head * head_dim (not n_embd)
# ============================================================================

def test_q_proj_output_dim():
    """c_q output = n_head * head_dim, which differs from n_embd for Qwen3."""
    from phase6.attention import MultiHeadAttention
    cfg = small_cfg()
    # n_embd=64, n_head=4, head_dim=32 -> c_q out = 128, not 64
    attn = MultiHeadAttention(cfg)
    assert attn.c_q.out_features == cfg.n_head * cfg.head_dim
    assert attn.c_q.out_features == 128  # 4 * 32, not 64
    assert attn.c_proj.in_features == cfg.n_head * cfg.head_dim
    assert attn.c_proj.out_features == cfg.n_embd


def test_q_proj_qwen3_dims():
    """Qwen3-0.6B dimensions: Q=2048 out, K/V=1024 out, proj 2048->1024."""
    from phase6.attention import MultiHeadAttention
    cfg = Config()
    attn = MultiHeadAttention(cfg)
    assert attn.c_q.in_features == 1024
    assert attn.c_q.out_features == 2048   # 16 * 128
    assert attn.c_k.in_features == 1024
    assert attn.c_k.out_features == 1024   # 8 * 128
    assert attn.c_v.in_features == 1024
    assert attn.c_v.out_features == 1024   # 8 * 128
    assert attn.c_proj.in_features == 2048  # 16 * 128
    assert attn.c_proj.out_features == 1024


# ============================================================================
# QK-norm toggle
# ============================================================================

def test_qk_norm_enabled():
    from phase6.attention import MultiHeadAttention
    cfg = small_cfg(use_qk_norm=True)
    attn = MultiHeadAttention(cfg)
    assert attn.q_norm is not None
    assert attn.k_norm is not None


def test_qk_norm_disabled():
    from phase6.attention import MultiHeadAttention
    cfg = small_cfg(use_qk_norm=False)
    attn = MultiHeadAttention(cfg)
    assert attn.q_norm is None
    assert attn.k_norm is None


# ============================================================================
# RoPE uses config
# ============================================================================

def test_rotary_uses_config():
    """RoPE buffer shape should use head_dim from config."""
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)
    # cos shape: (1, rotary_len, 1, head_dim//2)
    assert model.cos.shape[-1] == cfg.head_dim // 2


def test_rotary_base_affects_frequencies():
    """Different rope_base should produce different cos/sin."""
    from phase6.model import Model
    cfg1 = small_cfg(rope_base=10000.0)
    cfg2 = small_cfg(rope_base=1_000_000.0)
    m1 = Model(cfg1)
    m2 = Model(cfg2)
    assert not torch.allclose(m1.cos, m2.cos)


# ============================================================================
# Staircase mask with doc_ids
# ============================================================================

def test_staircase_mask_doc_ids():
    """Staircase mask with doc_ids enforces document boundaries."""
    from phase6.attention import build_staircase_mask
    L, blk = 16, 4
    B = 2
    doc_ids = torch.zeros(B, L, dtype=torch.long)
    doc_ids[:, 8:] = 1

    mask = build_staircase_mask(L, blk, doc_ids=doc_ids)
    assert mask.shape == (B, 1, 2 * L, 2 * L)
    assert torch.isfinite(mask).any()
    assert (mask == float('-inf')).any()


# ============================================================================
# KV cache
# ============================================================================

def test_kv_cache_enable_disable():
    from phase6.model import Model
    cfg = small_cfg()
    model = Model(cfg)

    model.enable_kv_cache()
    for block in model.blocks:
        assert block.attn.cache_mode is True

    model.disable_kv_cache()
    for block in model.blocks:
        assert block.attn.cache_mode is False
        assert block.attn.kv_cache is None
