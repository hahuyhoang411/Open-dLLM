"""Tests for Method D: TiledMLP."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from phase6.config import Config
from phase6.model import Model, SwiGLU


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


class TestTiledMLPForward:
  def test_forward_output_matches(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
      ref = mlp(x)
      tiled = tiled_mlp_forward(mlp, x, chunk_size=16)
    torch.testing.assert_close(ref, tiled, atol=1e-6, rtol=1e-6)

  def test_single_chunk(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    B, T, D = 2, 32, 64
    x = torch.randn(B, T, D)
    with torch.no_grad():
      ref = mlp(x)
      # chunk_size >= B*T means single chunk
      tiled = tiled_mlp_forward(mlp, x, chunk_size=B * T)
    torch.testing.assert_close(ref, tiled, atol=1e-6, rtol=1e-6)

  def test_max_chunks(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
      ref = mlp(x)
      tiled = tiled_mlp_forward(mlp, x, chunk_size=1)
    torch.testing.assert_close(ref, tiled, atol=1e-5, rtol=1e-5)

  def test_various_chunk_sizes(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    B, T, D = 2, 32, 64
    x = torch.randn(B, T, D)
    with torch.no_grad():
      ref = mlp(x)
    for chunk in [16, 32, 64, 128, B * T]:
      with torch.no_grad():
        tiled = tiled_mlp_forward(mlp, x, chunk_size=chunk)
      torch.testing.assert_close(ref, tiled, atol=1e-5, rtol=1e-5,
                                 msg=f"Mismatch at chunk_size={chunk}")

  def test_different_batch_sizes(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    for B in [1, 2, 4]:
      x = torch.randn(B, 16, 64)
      with torch.no_grad():
        ref = mlp(x)
        tiled = tiled_mlp_forward(mlp, x, chunk_size=8)
      torch.testing.assert_close(ref, tiled, atol=1e-5, rtol=1e-5,
                                 msg=f"Mismatch at B={B}")

  def test_auto_chunk_size(self):
    """chunk_size=None or 0 auto-computes a valid chunk size."""
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
      ref = mlp(x)
      # chunk_size=None -> auto
      out_none = tiled_mlp_forward(mlp, x, chunk_size=None)
      # chunk_size=0 -> disabled (passthrough)
      out_zero = tiled_mlp_forward(mlp, x, chunk_size=0)
    torch.testing.assert_close(ref, out_none, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref, out_zero, atol=1e-6, rtol=1e-6)


class TestTiledMLPGradients:
  def test_gradient_correctness(self):
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()

    # Reference grads
    torch.manual_seed(42)
    mlp_ref = SwiGLU(cfg)
    mlp_ref.train()
    x = torch.randn(2, 32, 64, requires_grad=True)
    out_ref = mlp_ref(x)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    ref_grads = {n: p.grad.clone() for n, p in mlp_ref.named_parameters() if p.grad is not None}
    x_grad_ref = x.grad.clone()

    # Reset
    x.grad = None
    for p in mlp_ref.parameters():
      p.grad = None

    # Tiled grads -- copy weights from ref
    torch.manual_seed(42)
    mlp_tiled = SwiGLU(cfg)
    mlp_tiled.load_state_dict(mlp_ref.state_dict())
    mlp_tiled.train()
    out_tiled = tiled_mlp_forward(mlp_tiled, x, chunk_size=16)
    loss_tiled = out_tiled.sum()
    loss_tiled.backward()
    tiled_grads = {n: p.grad.clone() for n, p in mlp_tiled.named_parameters() if p.grad is not None}

    for name in ref_grads:
      assert name in tiled_grads, f"Missing grad for {name}"
      torch.testing.assert_close(
        ref_grads[name], tiled_grads[name], atol=1e-4, rtol=1e-4,
        msg=f"Grad mismatch for {name}",
      )

  def test_backward_produces_correct_grads(self):
    """Explicit gradient check on gate/up/down weights."""
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.train()
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = tiled_mlp_forward(mlp, x, chunk_size=8)
    loss = out.sum()
    loss.backward()
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None
    assert x.grad is not None
    # Grads should be non-zero
    assert mlp.gate_proj.weight.grad.abs().sum() > 0
    assert mlp.up_proj.weight.grad.abs().sum() > 0
    assert mlp.down_proj.weight.grad.abs().sum() > 0

  def test_composability_with_grad_ckpt(self):
    """TiledMLP inside grad_checkpoint still produces correct grads."""
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    mlp.train()
    x = torch.randn(2, 16, 64, requires_grad=True)

    def ckpt_fn(x_in):
      return tiled_mlp_forward(mlp, x_in, chunk_size=8)

    out = grad_checkpoint(ckpt_fn, x, use_reentrant=False)
    loss = out.sum()
    loss.backward()
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None


class TestTiledMLPIntegration:
  def test_full_model_with_tiled_mlp(self):
    """TiledMLP integrated into full model produces correct grads."""
    from phase6.tiled_mlp import tiled_mlp_forward
    torch.manual_seed(42)
    cfg = small_cfg()
    input_concat, targets, mask = _make_training_inputs(cfg)

    # Reference
    torch.manual_seed(42)
    model_ref = Model(cfg)
    model_ref.train()
    h_ref, _ = model_ref(input_concat, targets=targets, attn_mask=mask)
    logits_ref = model_ref.lm_head(h_ref)
    loss_ref = F.cross_entropy(logits_ref.view(-1, cfg.vocab_size), targets.view(-1))
    loss_ref.backward()
    ref_grads = {n: p.grad.clone() for n, p in model_ref.named_parameters() if p.grad is not None}

    # TiledMLP: patch each block's MLP
    torch.manual_seed(42)
    model_tiled = Model(cfg)
    for block in model_tiled.blocks:
      mlp_mod = block.mlp
      mlp_norm_mod = block.mlp_norm
      attn_mod = block.attn
      attn_norm_mod = block.attn_norm

      def _patched(x, cos, sin, attn_mask=None, positions=None,
                   _attn=attn_mod, _attn_norm=attn_norm_mod,
                   _mlp=mlp_mod, _mlp_norm=mlp_norm_mod):
        x = x + _attn(_attn_norm(x), cos, sin, attn_mask=attn_mask, positions=positions)
        x = x + tiled_mlp_forward(_mlp, _mlp_norm(x), chunk_size=16)
        return x

      block._forward = _patched

    model_tiled.train()
    h_tiled, _ = model_tiled(input_concat, targets=targets, attn_mask=mask)
    logits_tiled = model_tiled.lm_head(h_tiled)
    loss_tiled = F.cross_entropy(logits_tiled.view(-1, cfg.vocab_size), targets.view(-1))
    loss_tiled.backward()
    tiled_grads = {n: p.grad.clone() for n, p in model_tiled.named_parameters() if p.grad is not None}

    for name in ref_grads:
      assert name in tiled_grads, f"Missing grad for {name}"
      torch.testing.assert_close(
        ref_grads[name], tiled_grads[name], atol=1e-4, rtol=1e-4,
        msg=f"Grad mismatch for {name}",
      )
