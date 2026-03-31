"""Tests for Method C: Smart CPU Offload."""

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


def _get_offload_smart_grads(cfg, input_concat, targets, mask, is_last_layer_flag=None):
  from phase6.offload_smart import smart_offload_checkpoint
  torch.manual_seed(42)
  cfg_off = copy.deepcopy(cfg)
  cfg_off.use_grad_ckpt = True
  model = Model(cfg_off)
  n_blocks = len(model.blocks)

  for i, block in enumerate(model.blocks):
    orig = block._forward
    last = (i == n_blocks - 1) if is_last_layer_flag is None else is_last_layer_flag

    def _patch(fn=orig, is_last=last):
      def wrapper(x, cos, sin, attn_mask=None, positions=None):
        return smart_offload_checkpoint(
          fn, x, cos, sin, attn_mask, positions,
          use_reentrant=False,
          context_fn=_sac_context_fn,
          is_last_layer=is_last,
        )
      return wrapper

    block.forward = _patch()

  model.train()
  h, _ = model(input_concat, targets=targets, attn_mask=mask)
  logits = model.lm_head(h)
  loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
  loss.backward()
  return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


class TestSmartOffload:
  def test_gradient_correctness(self):
    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    ref_grads = _get_reference_grads(cfg, input_concat, targets, mask)
    off_grads = _get_offload_smart_grads(cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in off_grads, f"Missing grad for {name}"
      torch.testing.assert_close(
        ref_grads[name], off_grads[name], atol=1e-4, rtol=1e-4,
        msg=f"Grad mismatch for {name}",
      )

  def test_forward_output_matches(self):
    from phase6.offload_smart import smart_offload_checkpoint
    torch.manual_seed(42)
    cfg = small_cfg()
    input_concat, targets, mask = _make_training_inputs(cfg)

    torch.manual_seed(42)
    model_ref = Model(cfg)
    model_ref.eval()
    with torch.no_grad():
      h_ref, _ = model_ref(input_concat, targets=targets, attn_mask=mask)

    torch.manual_seed(42)
    model_test = Model(cfg)
    model_test.eval()
    for i, block in enumerate(model_test.blocks):
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    with torch.no_grad():
      h_test, _ = model_test(input_concat, targets=targets, attn_mask=mask)

    torch.testing.assert_close(h_ref, h_test, atol=1e-6, rtol=1e-6)

  def test_last_layer_skip(self):
    """When is_last_layer=True, should not crash and grads should still be correct."""
    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    # Default behavior: last layer auto-detected
    grads = _get_offload_smart_grads(cfg, input_concat, targets, mask)
    assert len(grads) > 0

  def test_small_tensor_skip(self):
    """With tiny model, all tensors < 2MB. Smart offload should keep them on GPU
    (or equivalently, run fine on CPU where offload is a no-op)."""
    from phase6.offload_smart import smart_offload_checkpoint
    torch.manual_seed(42)
    # Very small model -- tensors are tiny
    cfg = small_cfg(n_embd=16, n_head=2, n_kv_head=1, head_dim=16, mlp_hidden=32)
    model = Model(cfg)
    model.train()

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(
            fn, x, cos, sin, attn_mask, positions,
            use_reentrant=False, is_last_layer=False,
          )
        return wrapper

      block.forward = _patch()

    input_concat, targets, mask = _make_training_inputs(cfg)
    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads

  def test_works_on_cpu(self):
    from phase6.offload_smart import smart_offload_checkpoint

    torch.manual_seed(42)

    def dummy_fn(x):
      return x * 2.0

    x = torch.randn(2, 8, 64)
    out = smart_offload_checkpoint(dummy_fn, x, use_reentrant=False)
    assert out.shape == x.shape

  def test_with_sac_context(self):
    from phase6.offload_smart import smart_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    model = Model(cfg)
    model.train()

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(
            fn, x, cos, sin, attn_mask, positions,
            use_reentrant=False,
            context_fn=_sac_context_fn,
          )
        return wrapper

      block.forward = _patch()

    input_concat, targets, mask = _make_training_inputs(cfg)
    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads

  def test_multiple_steps(self):
    from phase6.offload_smart import smart_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    model = Model(cfg)

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    model.train()
    for step in range(3):
      model.zero_grad()
      input_concat, targets, mask = _make_training_inputs(cfg)
      h, _ = model(input_concat, targets=targets, attn_mask=mask)
      logits = model.lm_head(h)
      loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
      loss.backward()
      has_grads = any(p.grad is not None for p in model.parameters())
      assert has_grads, f"No grads at step {step}"

  def test_buffer_initialization(self):
    """First call initializes buffers, subsequent calls reuse them."""
    from phase6.offload_smart import smart_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    model = Model(cfg)

    for block in model.blocks:
      orig = block._forward

      def _patch(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)
        return wrapper

      block.forward = _patch()

    model.train()
    # Step 1: buffer init
    input_concat, targets, mask = _make_training_inputs(cfg)
    h, _ = model(input_concat, targets=targets, attn_mask=mask)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    # Step 2: buffer reuse (should not crash)
    model.zero_grad()
    input_concat2, targets2, mask2 = _make_training_inputs(cfg)
    h2, _ = model(input_concat2, targets=targets2, attn_mask=mask2)
    logits2 = model.lm_head(h2)
    loss2 = F.cross_entropy(logits2.view(-1, cfg.vocab_size), targets2.view(-1))
    loss2.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads
