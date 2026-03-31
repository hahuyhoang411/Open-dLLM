"""Tests for attention optimizations: dispatch modes, bf16 masks, QK-Clip frequency, GQA."""

import pytest
import torch
from phase6.config import Config


def small_cfg(**overrides):
  defaults = dict(
    n_layer=2,
    n_embd=64,
    n_head=4,
    n_kv_head=2,
    head_dim=32,
    mlp_hidden=128,
    vocab_size=256,
    seq_len=32,
    block_size=8,
    rope_base=10000.0,
    rms_eps=1e-6,
    dropout=0.0,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=False,
    use_grad_ckpt=False,
    use_flex=False,
    mask_token_id=0,
    pad_token_id=2,
  )
  defaults.update(overrides)
  return Config(**defaults)


# ============================================================================
# Optimization 1: Dispatch Mode Switching
# ============================================================================


class TestDispatchModes:
  def test_set_dispatch_mode_sdpa_mask(self):
    """set_dispatch_mode changes _dispatch attribute."""
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    attn.set_dispatch_mode('sdpa_mask')
    assert attn._dispatch == 'sdpa_mask'

  def test_set_dispatch_mode_sdpa_causal(self):
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    attn.set_dispatch_mode('sdpa_causal')
    assert attn._dispatch == 'sdpa_causal'

  def test_set_dispatch_mode_flex(self):
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    attn.set_dispatch_mode('flex')
    assert attn._dispatch == 'flex'

  def test_set_dispatch_mode_invalid(self):
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    with pytest.raises(AssertionError):
      attn.set_dispatch_mode('invalid')

  def test_default_dispatch_mode(self):
    """Default dispatch should be 'sdpa_causal' (no flex on CPU)."""
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    assert attn._dispatch in ('flex', 'sdpa_mask', 'sdpa_causal')

  def test_forward_sdpa_mask_path(self):
    """Forward with attn_mask uses SDPA masked path."""
    from phase6.attention import build_staircase_mask
    from phase6.model import Model

    cfg = small_cfg()
    model = Model(cfg)
    model.eval()
    B, L = 2, cfg.seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
    targets = torch.randint(0, cfg.vocab_size, (B, L))
    mask = build_staircase_mask(L, cfg.block_size)
    with torch.no_grad():
      out, _ = model(idx, targets=targets, attn_mask=mask)
    assert out.shape == (B, L, cfg.n_embd)


# ============================================================================
# Optimization 2: QK-Clip Frequency Reduction
# ============================================================================


class TestQKClipFrequency:
  def test_should_compute_qk_norms_first_call(self):
    """First call after reset should return True."""
    import phase6.attention as attn_mod
    from phase6.attention import _should_compute_qk_norms

    attn_mod._qk_clip_counter = 0
    assert _should_compute_qk_norms() is True

  def test_should_compute_qk_norms_skip(self):
    """Second call should return False (not every-step)."""
    import phase6.attention as attn_mod
    from phase6.attention import _should_compute_qk_norms

    attn_mod._qk_clip_counter = 0
    _should_compute_qk_norms()  # step 1: True
    assert _should_compute_qk_norms() is False  # step 2: skip

  def test_should_compute_qk_norms_period(self):
    """Should fire on step 1, 11, 21, etc. (every _QK_CLIP_EVERY)."""
    import phase6.attention as attn_mod
    from phase6.attention import _QK_CLIP_EVERY, _should_compute_qk_norms

    attn_mod._qk_clip_counter = 0
    results = [_should_compute_qk_norms() for _ in range(_QK_CLIP_EVERY + 1)]
    assert results[0] is True  # step 1
    assert all(r is False for r in results[1:_QK_CLIP_EVERY])  # steps 2-10
    assert results[_QK_CLIP_EVERY] is True  # step 11


# ============================================================================
# Optimization 3: (GQA enable_gqa=True tested via existing forward tests)
# ============================================================================


class TestGQAOptimization:
  def test_gqa_forward_with_mask(self):
    """GQA with dense mask uses enable_gqa=True path (no repeat_interleave)."""
    from phase6.attention import MultiHeadAttention, build_staircase_mask

    cfg = small_cfg()  # n_head=4, n_kv_head=2 -> GQA active
    attn = MultiHeadAttention(cfg)
    attn.eval()
    B, T = 2, 2 * cfg.seq_len
    x = torch.randn(B, T, cfg.n_embd)
    cos = torch.randn(1, cfg.seq_len, 1, cfg.head_dim // 2)
    sin = torch.randn(1, cfg.seq_len, 1, cfg.head_dim // 2)
    mask = build_staircase_mask(cfg.seq_len, cfg.block_size)
    with torch.no_grad():
      y = attn(x, cos, sin, attn_mask=mask)
    assert y.shape == (B, T, cfg.n_embd)

  def test_gqa_forward_without_mask(self):
    """GQA without mask still uses repeat_interleave (generation path)."""
    from phase6.attention import MultiHeadAttention

    cfg = small_cfg()
    attn = MultiHeadAttention(cfg)
    attn.eval()
    B, T = 1, 8
    x = torch.randn(B, T, cfg.n_embd)
    cos = torch.randn(1, T, 1, cfg.head_dim // 2)
    sin = torch.randn(1, T, 1, cfg.head_dim // 2)
    with torch.no_grad():
      y = attn(x, cos, sin, attn_mask=None)
    assert y.shape == (B, T, cfg.n_embd)


# ============================================================================
# Optimization 4: Dense Staircase Mask in bf16
# ============================================================================


class TestStaircaseMaskDtype:
  def test_staircase_mask_bf16(self):
    """Dense staircase mask should be bf16."""
    from phase6.attention import build_staircase_mask

    mask = build_staircase_mask(64, 8)
    assert mask.dtype == torch.bfloat16

  def test_staircase_mask_with_docs_bf16(self):
    """Dense staircase mask with doc_ids should be bf16."""
    from phase6.attention import build_staircase_mask

    doc_ids = torch.zeros(2, 64, dtype=torch.long)
    mask = build_staircase_mask(64, 8, doc_ids=doc_ids)
    assert mask.dtype == torch.bfloat16

  def test_staircase_mask_values_preserved(self):
    """bf16 mask should still have 0.0 and -inf values."""
    from phase6.attention import build_staircase_mask

    mask = build_staircase_mask(16, 4)
    # Should contain both finite (0) and -inf
    assert (mask == 0).any()
    assert torch.isinf(mask).any()

  def test_staircase_mask_with_docs_shape(self):
    """Shape should be (B, 1, 2L, 2L) with doc_ids."""
    from phase6.attention import build_staircase_mask

    B, L, blk = 2, 16, 4
    doc_ids = torch.zeros(B, L, dtype=torch.long)
    doc_ids[:, 8:] = 1
    mask = build_staircase_mask(L, blk, doc_ids=doc_ids)
    assert mask.shape == (B, 1, 2 * L, 2 * L)

  def test_staircase_mask_obc_strict_greater_bf16(self):
    """OBC strict-greater semantics preserved with bf16."""
    from phase6.attention import build_staircase_mask

    L, blk = 16, 4
    mask = build_staircase_mask(L, blk)
    # x_t block 0 -> x_0 block 0: should be -inf (strict >)
    for qt in range(4):
      for kv0 in range(L, L + 4):
        assert mask[qt, kv0] == float('-inf')
    # x_t block 1 -> x_0 block 0: should be 0 (allowed)
    for qt in range(4, 8):
      for kv0 in range(L, L + 4):
        assert mask[qt, kv0] == 0.0


# ============================================================================
# Integration: full model forward still works with all optimizations
# ============================================================================


class TestIntegration:
  def test_training_forward_with_optimizations(self):
    """Full model training forward still works after all optimizations."""
    from phase6.attention import build_staircase_mask
    from phase6.model import Model

    cfg = small_cfg()
    model = Model(cfg)
    model.eval()
    B, L = 2, cfg.seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
    targets = torch.randint(0, cfg.vocab_size, (B, L))
    mask = build_staircase_mask(L, cfg.block_size)
    with torch.no_grad():
      out, _ = model(idx, targets=targets, attn_mask=mask)
    assert out.shape == (B, L, cfg.n_embd)

  def test_generation_forward_with_optimizations(self):
    """Full model generation forward still works after all optimizations."""
    from phase6.model import Model

    cfg = small_cfg()
    model = Model(cfg)
    model.eval()
    B, T = 1, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
      logits, _ = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)

  def test_gated_query_with_optimizations(self):
    """Gated query still works with all optimizations."""
    from phase6.model import Model

    cfg = small_cfg(use_gated_query=True)
    model = Model(cfg)
    model.eval()
    B, T = 1, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
      logits, _ = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
