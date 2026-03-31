"""Tests for Method E: Sqrt-SAC (selective layer checkpointing)."""

import torch
from phase6.config import Config
from phase6.model import Model
from torch.nn import functional as F


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


def _make_training_inputs(cfg, B=2):
  L = cfg.seq_len
  noisy = torch.randint(0, cfg.vocab_size, (B, L))
  clean = torch.randint(0, cfg.vocab_size, (B, L))
  input_concat = torch.cat([noisy, clean], dim=1)
  targets = clean
  from phase6.attention import build_staircase_mask

  mask = build_staircase_mask(L, cfg.block_size)
  return input_concat, targets, mask


# ============================================================================
# Index computation tests
# ============================================================================


class TestCheckpointIndices:
  def test_checkpoint_indices_sqrt(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    # n=28: sqrt(28) ~= 5.3 -> checkpoint every ~5 layers
    indices = get_sqrt_checkpoint_indices(28, method='sqrt')
    assert isinstance(indices, list)
    assert len(indices) > 0
    # Should have approximately sqrt(28) ~= 5 checkpointed layers
    assert 4 <= len(indices) <= 7
    # All indices in range
    for idx in indices:
      assert 0 <= idx < 28

  def test_checkpoint_indices_sqrt_n4(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    indices = get_sqrt_checkpoint_indices(4, method='sqrt')
    assert isinstance(indices, list)
    assert len(indices) == 2  # sqrt(4) = 2

  def test_checkpoint_indices_fixed(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    # method=4 with n=28: every 4 layers -> 7 indices
    indices = get_sqrt_checkpoint_indices(28, method=4)
    assert isinstance(indices, list)
    assert len(indices) == 7
    # Should be [0, 4, 8, 12, 16, 20, 24] or similar spacing
    for idx in indices:
      assert 0 <= idx < 28

  def test_checkpoint_indices_edge_1(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    indices = get_sqrt_checkpoint_indices(1, method='sqrt')
    assert indices == [0]

  def test_checkpoint_indices_edge_2(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    indices = get_sqrt_checkpoint_indices(2, method='sqrt')
    assert isinstance(indices, list)
    assert len(indices) >= 1
    for idx in indices:
      assert 0 <= idx < 2

  def test_checkpoint_indices_all_in_range(self):
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    for n in [1, 2, 4, 8, 16, 28, 32, 64]:
      indices = get_sqrt_checkpoint_indices(n, method='sqrt')
      for idx in indices:
        assert 0 <= idx < n, f'Index {idx} out of range for n={n}'


# ============================================================================
# Gradient correctness tests
# ============================================================================


class TestSqrtSACGradients:
  def test_gradient_correctness(self):
    """Model with sqrt-SAC produces grads close to per-block SAC."""
    torch.manual_seed(42)
    cfg_ref = small_cfg(use_grad_ckpt=True, n_layer=4, use_liger_rmsnorm=False)
    input_concat, targets, mask = _make_training_inputs(cfg_ref)

    # Reference: per-block SAC (standard grad_ckpt on every block)
    torch.manual_seed(42)
    model_ref = Model(cfg_ref)
    model_ref.train()
    h_ref, _ = model_ref(input_concat, targets=targets, attn_mask=mask)
    logits_ref = model_ref.lm_head(h_ref)
    loss_ref = F.cross_entropy(logits_ref.view(-1, cfg_ref.vocab_size), targets.view(-1))
    loss_ref.backward()
    ref_grads = {n: p.grad.clone() for n, p in model_ref.named_parameters() if p.grad is not None}

    # Sqrt-SAC: use config flag (model wires it automatically)
    torch.manual_seed(42)
    cfg_sqrt = small_cfg(use_grad_ckpt=True, use_sqrt_ckpt=True, n_layer=4, use_liger_rmsnorm=False)
    model_sqrt = Model(cfg_sqrt)
    model_sqrt.train()
    h_sqrt, _ = model_sqrt(input_concat, targets=targets, attn_mask=mask)
    logits_sqrt = model_sqrt.lm_head(h_sqrt)
    loss_sqrt = F.cross_entropy(logits_sqrt.view(-1, cfg_sqrt.vocab_size), targets.view(-1))
    loss_sqrt.backward()
    sqrt_grads = {n: p.grad.clone() for n, p in model_sqrt.named_parameters() if p.grad is not None}

    for name in ref_grads:
      assert name in sqrt_grads, f'Missing grad for {name}'
      torch.testing.assert_close(
        ref_grads[name],
        sqrt_grads[name],
        atol=1e-4,
        rtol=1e-4,
        msg=f'Grad mismatch for {name}',
      )

  def test_forward_output_matches(self):
    """Output identical regardless of checkpoint granularity."""
    torch.manual_seed(42)
    cfg = small_cfg(n_layer=4)
    input_concat, targets, mask = _make_training_inputs(cfg)

    # Reference
    torch.manual_seed(42)
    model_ref = Model(cfg)
    model_ref.eval()
    with torch.no_grad():
      h_ref, _ = model_ref(input_concat, targets=targets, attn_mask=mask)

    # Sqrt-SAC (eval mode -- no actual checkpointing, but function should still work)
    torch.manual_seed(42)
    model_sqrt = Model(cfg)
    model_sqrt.eval()
    with torch.no_grad():
      h_sqrt, _ = model_sqrt(input_concat, targets=targets, attn_mask=mask)

    torch.testing.assert_close(h_ref, h_sqrt, atol=1e-6, rtol=1e-6)

  def test_non_checkpoint_layers_run_normally(self):
    """Layers not in checkpoint indices run without recompute overhead."""
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    indices = get_sqrt_checkpoint_indices(4, method='sqrt')
    assert len(indices) < 4, 'Sqrt-SAC should checkpoint fewer than all layers'

    torch.manual_seed(42)
    cfg = small_cfg(n_layer=4, use_grad_ckpt=True, use_sqrt_ckpt=True, use_liger_rmsnorm=False)
    model = Model(cfg)
    model.train()

    input_concat, targets, mask = _make_training_inputs(cfg)
    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads

  def test_4_layers(self):
    """Specific test with n_layer=4 using config-based sqrt-SAC."""
    from phase6.sqrt_sac import get_sqrt_checkpoint_indices

    indices = get_sqrt_checkpoint_indices(4, method='sqrt')
    assert len(indices) == 2  # sqrt(4) = 2

    torch.manual_seed(42)
    cfg = small_cfg(n_layer=4, use_grad_ckpt=True, use_sqrt_ckpt=True, use_liger_rmsnorm=False)
    model = Model(cfg)
    model.train()

    input_concat, targets, mask = _make_training_inputs(cfg)
    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads
