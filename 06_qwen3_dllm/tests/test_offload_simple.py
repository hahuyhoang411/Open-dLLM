"""Tests for Method B: Simple CPU Offload."""

import copy

import pytest
import torch
from torch.nn import functional as F

from phase6.config import Config
from phase6.model import Model, _sac_context_fn


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


def _make_training_inputs(cfg, B=2):
  L = cfg.seq_len
  noisy = torch.randint(0, cfg.vocab_size, (B, L))
  clean = torch.randint(0, cfg.vocab_size, (B, L))
  input_concat = torch.cat([noisy, clean], dim=1)
  targets = clean
  from phase6.attention import build_staircase_mask
  mask = build_staircase_mask(L, cfg.block_size)
  return input_concat, targets, mask


def _get_reference_grads(cfg, input_concat, targets, mask):
  """Standard grad_ckpt grads -- the reference."""
  torch.manual_seed(42)
  cfg_ref = copy.deepcopy(cfg)
  cfg_ref.use_grad_ckpt = True
  model = Model(cfg_ref)
  model.train()
  h, _ = model(input_concat, targets=targets, attn_mask=mask)
  logits = model.lm_head(h)
  loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
  loss.backward()
  return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _get_offload_simple_grads(cfg, input_concat, targets, mask):
  """Grads using simple_offload_checkpoint."""
  from phase6.offload_simple import simple_offload_checkpoint
  torch.manual_seed(42)
  cfg_off = copy.deepcopy(cfg)
  cfg_off.use_grad_ckpt = True
  model = Model(cfg_off)

  # Monkey-patch block.forward to use simple_offload_checkpoint
  for block in model.blocks:
    orig_forward = block._forward

    def _patched_forward(fn=orig_forward):
      def wrapper(x, cos, sin, attn_mask=None, positions=None):
        return simple_offload_checkpoint(
          fn, x, cos, sin, attn_mask, positions,
          use_reentrant=False,
          context_fn=_sac_context_fn,
        )
      return wrapper

    block.forward = _patched_forward()

  model.train()
  h, _ = model(input_concat, targets=targets, attn_mask=mask)
  logits = model.lm_head(h)
  loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
  loss.backward()
  return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


class TestSimpleOffload:
  def test_gradient_correctness(self):
    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    ref_grads = _get_reference_grads(cfg, input_concat, targets, mask)
    off_grads = _get_offload_simple_grads(cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in off_grads, f"Missing grad for {name}"
      torch.testing.assert_close(
        ref_grads[name], off_grads[name], atol=1e-4, rtol=1e-4,
        msg=f"Grad mismatch for {name}",
      )

  def test_forward_output_matches(self):
    from phase6.offload_simple import simple_offload_checkpoint
    torch.manual_seed(42)
    cfg = small_cfg()
    input_concat, targets, mask = _make_training_inputs(cfg)

    # Reference: standard forward
    torch.manual_seed(42)
    model_ref = Model(cfg)
    model_ref.eval()
    with torch.no_grad():
      h_ref, _ = model_ref(input_concat, targets=targets, attn_mask=mask)

    # Test: offload checkpoint on each block
    torch.manual_seed(42)
    model_test = Model(cfg)
    model_test.eval()
    for block in model_test.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    with torch.no_grad():
      h_test, _ = model_test(input_concat, targets=targets, attn_mask=mask)

    torch.testing.assert_close(h_ref, h_test, atol=1e-6, rtol=1e-6)

  def test_works_on_cpu(self):
    from phase6.offload_simple import simple_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg()
    model = Model(cfg)
    model.train()

    def dummy_fn(x):
      return x * 2.0

    x = torch.randn(2, 8, 64)
    out = simple_offload_checkpoint(dummy_fn, x, use_reentrant=False)
    assert out.shape == x.shape

  def test_with_sac_context(self):
    from phase6.offload_simple import simple_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg()
    model = Model(cfg)
    model.train()
    input_concat, targets, mask = _make_training_inputs(cfg)

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(
            fn, x, cos, sin, attn_mask, positions,
            use_reentrant=False,
            context_fn=_sac_context_fn,
          )
        return wrapper

      block.forward = _patch()

    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    # Verify grads exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads

  def test_multiple_steps(self):
    from phase6.offload_simple import simple_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    model = Model(cfg)

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    model.train()
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.use_grad_ckpt = True

    for step in range(3):
      model.zero_grad()
      input_concat, targets, mask = _make_training_inputs(cfg)
      h, _ = model(input_concat, targets=targets, attn_mask=mask)
      logits = model.lm_head(h)
      loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
      loss.backward()
      has_grads = any(p.grad is not None for p in model.parameters())
      assert has_grads, f"No grads at step {step}"

  def test_no_grad_mode(self):
    from phase6.offload_simple import simple_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg()
    model = Model(cfg)
    model.eval()

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    input_concat, targets, mask = _make_training_inputs(cfg)
    with torch.no_grad():
      h, _ = model(input_concat, targets=targets, attn_mask=mask)
    assert h.shape == (2, cfg.seq_len, cfg.n_embd)
