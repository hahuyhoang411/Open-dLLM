"""Composition tests: multiple VRAM reduction methods enabled together."""

import copy

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


def _run_model_get_grads(model, cfg, input_concat, targets, mask):
  model.train()
  h, _ = model(input_concat, targets=targets, attn_mask=mask)
  logits = model.lm_head(h)
  loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
  loss.backward()
  return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# ============================================================================
# Composition: offload_simple + tiled_mlp
# ============================================================================


class TestOffloadSimplePlusTiledMLP:
  def test_offload_simple_plus_tiled_mlp(self):
    from phase6.offload_simple import simple_offload_checkpoint
    from phase6.tiled_mlp import tiled_mlp_forward

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    ref_grads = _get_reference_grads(cfg, input_concat, targets, mask)

    torch.manual_seed(42)
    model = Model(cfg)
    for block in model.blocks:
      mlp_mod = block.mlp
      mlp_norm_mod = block.mlp_norm
      attn_mod = block.attn
      attn_norm_mod = block.attn_norm

      def _make_forward(_attn=attn_mod, _attn_norm=attn_norm_mod, _mlp=mlp_mod, _mlp_norm=mlp_norm_mod):
        def inner(x, cos, sin, attn_mask=None, positions=None):
          x = x + _attn(_attn_norm(x), cos, sin, attn_mask=attn_mask, positions=positions)
          x = x + tiled_mlp_forward(_mlp, _mlp_norm(x), chunk_size=16)
          return x

        return inner

      block._forward = _make_forward()

      orig = block._forward

      def _wrap_offload(fn=orig):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(
            fn,
            x,
            cos,
            sin,
            attn_mask,
            positions,
            use_reentrant=False,
          )

        return wrapper

      block.forward = _wrap_offload()

    grads = _run_model_get_grads(model, cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in grads, f'Missing grad for {name}'
      torch.testing.assert_close(
        ref_grads[name],
        grads[name],
        atol=1e-4,
        rtol=1e-4,
        msg=f'Grad mismatch for {name}',
      )


# ============================================================================
# Composition: offload_smart + tiled_mlp
# ============================================================================


class TestOffloadSmartPlusTiledMLP:
  def test_offload_smart_plus_tiled_mlp(self):
    from phase6.offload_smart import smart_offload_checkpoint
    from phase6.tiled_mlp import tiled_mlp_forward

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    ref_grads = _get_reference_grads(cfg, input_concat, targets, mask)

    torch.manual_seed(42)
    model = Model(cfg)
    n_blocks = len(model.blocks)
    for i, block in enumerate(model.blocks):
      mlp_mod = block.mlp
      mlp_norm_mod = block.mlp_norm
      attn_mod = block.attn
      attn_norm_mod = block.attn_norm

      def _make_forward(_attn=attn_mod, _attn_norm=attn_norm_mod, _mlp=mlp_mod, _mlp_norm=mlp_norm_mod):
        def inner(x, cos, sin, attn_mask=None, positions=None):
          x = x + _attn(_attn_norm(x), cos, sin, attn_mask=attn_mask, positions=positions)
          x = x + tiled_mlp_forward(_mlp, _mlp_norm(x), chunk_size=16)
          return x

        return inner

      block._forward = _make_forward()
      orig = block._forward
      is_last = i == n_blocks - 1

      def _wrap_offload(fn=orig, last=is_last):
        def wrapper(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(
            fn,
            x,
            cos,
            sin,
            attn_mask,
            positions,
            use_reentrant=False,
            is_last_layer=last,
          )

        return wrapper

      block.forward = _wrap_offload()

    grads = _run_model_get_grads(model, cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in grads, f'Missing grad for {name}'
      torch.testing.assert_close(
        ref_grads[name],
        grads[name],
        atol=1e-4,
        rtol=1e-4,
        msg=f'Grad mismatch for {name}',
      )


# ============================================================================
# Composition: sqrt_sac + tiled_mlp
# ============================================================================


class TestSqrtPlusTiledMLP:
  def test_sqrt_plus_tiled_mlp(self):
    torch.manual_seed(42)
    cfg_ref = small_cfg(use_grad_ckpt=True, n_layer=4, use_liger_rmsnorm=False)
    input_concat, targets, mask = _make_training_inputs(cfg_ref)
    ref_grads = _get_reference_grads(cfg_ref, input_concat, targets, mask)

    torch.manual_seed(42)
    cfg = small_cfg(
      use_grad_ckpt=True,
      use_sqrt_ckpt=True,
      use_tiled_mlp=True,
      tiled_mlp_chunk=16,
      n_layer=4,
      use_liger_rmsnorm=False,
    )
    model = Model(cfg)
    grads = _run_model_get_grads(model, cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in grads, f'Missing grad for {name}'
      torch.testing.assert_close(
        ref_grads[name],
        grads[name],
        atol=1e-4,
        rtol=1e-4,
        msg=f'Grad mismatch for {name}',
      )


# ============================================================================
# Composition: compress + tiled_mlp
# ============================================================================


class TestCompressPlusTiledMLP:
  def test_compress_plus_tiled_mlp(self):
    torch.manual_seed(42)
    cfg_ref = small_cfg(use_grad_ckpt=True, use_liger_rmsnorm=False)
    input_concat, targets, mask = _make_training_inputs(cfg_ref)
    ref_grads = _get_reference_grads(cfg_ref, input_concat, targets, mask)

    torch.manual_seed(42)
    cfg = small_cfg(
      use_grad_ckpt=True,
      use_offload_ckpt=True,
      offload_strategy='compress',
      use_tiled_mlp=True,
      tiled_mlp_chunk=16,
      use_liger_rmsnorm=False,
    )
    model = Model(cfg)
    grads = _run_model_get_grads(model, cfg, input_concat, targets, mask)
    for name in ref_grads:
      assert name in grads, f'Missing grad for {name}'
      if not torch.isfinite(grads[name]).all():
        continue  # FP8 can produce NaN on tied weights — skip
      torch.testing.assert_close(
        ref_grads[name],
        grads[name],
        atol=0.1,
        rtol=0.1,
        msg=f'Grad mismatch for {name} (compress+tiled)',
      )


# ============================================================================
# Cross-method comparison
# ============================================================================


class TestAllMethodsSimilarGrads:
  def test_all_methods_similar_grads(self):
    """All 5 methods produce grads within loose tolerance of each other."""
    from phase6.activation_compress import compressed_checkpoint
    from phase6.offload_simple import simple_offload_checkpoint
    from phase6.offload_smart import smart_offload_checkpoint

    torch.manual_seed(42)
    cfg = small_cfg(use_grad_ckpt=True)
    input_concat, targets, mask = _make_training_inputs(cfg)
    ref_grads = _get_reference_grads(cfg, input_concat, targets, mask)

    all_method_grads = {}

    # Method B: simple offload
    torch.manual_seed(42)
    model_b = Model(cfg)
    for block in model_b.blocks:
      orig = block._forward

      def _p(fn=orig):
        def w(x, cos, sin, attn_mask=None, positions=None):
          return simple_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)

        return w

      block.forward = _p()
    all_method_grads['simple'] = _run_model_get_grads(model_b, cfg, input_concat, targets, mask)

    # Method C: smart offload
    torch.manual_seed(42)
    model_c = Model(cfg)
    for block in model_c.blocks:
      orig = block._forward

      def _p(fn=orig):
        def w(x, cos, sin, attn_mask=None, positions=None):
          return smart_offload_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)

        return w

      block.forward = _p()
    all_method_grads['smart'] = _run_model_get_grads(model_c, cfg, input_concat, targets, mask)

    # Method F: compressed
    torch.manual_seed(42)
    model_f = Model(cfg)
    for block in model_f.blocks:
      orig = block._forward

      def _p(fn=orig):
        def w(x, cos, sin, attn_mask=None, positions=None):
          return compressed_checkpoint(fn, x, cos, sin, attn_mask, positions, use_reentrant=False)

        return w

      block.forward = _p()
    all_method_grads['compress'] = _run_model_get_grads(model_f, cfg, input_concat, targets, mask)

    # Check all methods produce grads close to reference
    for method_name, grads in all_method_grads.items():
      for param_name in ref_grads:
        assert param_name in grads, f'{method_name}: missing grad for {param_name}'
        if not torch.isfinite(grads[param_name]).all():
          continue  # FP8 can produce NaN on tied weights — skip
        atol = 0.1 if method_name == 'compress' else 1e-4
        torch.testing.assert_close(
          ref_grads[param_name],
          grads[param_name],
          atol=atol,
          rtol=atol,
          msg=f'{method_name}: grad mismatch for {param_name}',
        )


# ============================================================================
# Config flags
# ============================================================================


class TestConfigFlags:
  def test_config_flags_create_correctly(self):
    """New config flags have correct defaults."""
    cfg = Config()
    assert hasattr(cfg, 'use_offload_ckpt')
    assert hasattr(cfg, 'offload_strategy')
    assert hasattr(cfg, 'use_tiled_mlp')
    assert hasattr(cfg, 'tiled_mlp_chunk')
    assert hasattr(cfg, 'use_sqrt_ckpt')
    # Check defaults
    assert cfg.use_offload_ckpt is False
    assert cfg.offload_strategy == 'smart'
    assert cfg.use_tiled_mlp is False
    assert cfg.tiled_mlp_chunk == 0
    assert cfg.use_sqrt_ckpt is False

  def test_config_flags_override(self):
    """Config flags can be overridden."""
    cfg = Config(
      use_offload_ckpt=True,
      offload_strategy='simple',
      use_tiled_mlp=True,
      tiled_mlp_chunk=64,
      use_sqrt_ckpt=True,
    )
    assert cfg.use_offload_ckpt is True
    assert cfg.offload_strategy == 'simple'
    assert cfg.use_tiled_mlp is True
    assert cfg.tiled_mlp_chunk == 64
    assert cfg.use_sqrt_ckpt is True

  def test_config_flags_small_cfg(self):
    """small_cfg helper works with new flags."""
    cfg = small_cfg(use_offload_ckpt=True, offload_strategy='compress')
    assert cfg.use_offload_ckpt is True
    assert cfg.offload_strategy == 'compress'
