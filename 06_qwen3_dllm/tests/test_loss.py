"""Tests for Phase 6 loss module — chunked CE with auto-scaling chunk size."""

import math
import torch
import pytest
from dataclasses import dataclass


@dataclass
class TinyCfg:
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Chunk size auto-scaling
# ---------------------------------------------------------------------------

def test_chunk_size_shrinks_for_larger_vocab():
    from phase6.loss import _compute_chunk_size
    small = _compute_chunk_size(256)
    large = _compute_chunk_size(151_936)
    assert large < small, f"larger vocab should yield smaller chunks: {large} >= {small}"


def test_chunk_size_floor():
    from phase6.loss import _compute_chunk_size
    huge = _compute_chunk_size(10_000_000)  # absurdly large vocab
    assert huge >= 1024, f"chunk_size should never go below 1024: {huge}"


def test_chunk_size_phase5_compat():
    """vocab=49152 should yield ~16384 (Phase 5 default)."""
    from phase6.loss import _compute_chunk_size
    cs = _compute_chunk_size(49_152)
    assert 8192 <= cs <= 16384, f"expected ~15258 for vocab=49152, got {cs}"


def test_chunk_size_phase6():
    """vocab=151936 should yield ~4096-5120 range."""
    from phase6.loss import _compute_chunk_size
    cs = _compute_chunk_size(151_936)
    assert 1024 <= cs <= 8192, f"expected ~4936 for vocab=151936, got {cs}"


# ---------------------------------------------------------------------------
# Loss shape and value
# ---------------------------------------------------------------------------

def _make_inputs(batch=2, seq=16, dim=32, vocab=256):
    h = torch.randn(batch, seq, dim, requires_grad=True)
    w = torch.randn(vocab, dim, requires_grad=True)
    targets = torch.randint(1, vocab, (batch, seq))  # avoid pad_token_id=0
    mask = torch.ones(batch, seq, dtype=torch.bool)
    elbo_weight = torch.ones(batch, seq)
    return h, w, targets, mask, elbo_weight


def test_loss_is_scalar():
    from phase6.loss import compute_loss
    h, w, t, m, ew = _make_inputs()
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"


def test_loss_at_init_approx_ln_vocab():
    """Small-scale weights → CE ≈ ln(vocab_size) for near-uniform logits."""
    from phase6.loss import compute_loss
    torch.manual_seed(42)
    vocab = 256
    dim = 64
    batch, seq = 8, 64
    # Scale weights small so logits ≈ 0 → softmax ≈ uniform → CE ≈ ln(vocab)
    h = torch.randn(batch, seq, dim, requires_grad=True) * 0.01
    w = torch.randn(vocab, dim, requires_grad=True) * 0.01
    t = torch.randint(1, vocab, (batch, seq))
    m = torch.ones(batch, seq, dtype=torch.bool)
    ew = torch.ones(batch, seq)
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    expected = math.log(vocab)
    assert abs(loss.item() - expected) < 0.5, \
        f"init loss {loss.item():.2f} far from ln({vocab})={expected:.2f}"


# ---------------------------------------------------------------------------
# Gradient flow — CRITICAL (Liger FLCE bug was zero grads)
# ---------------------------------------------------------------------------

def test_gradients_flow_hidden():
    from phase6.loss import compute_loss
    h, w, t, m, ew = _make_inputs()
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    loss.backward()
    assert h.grad is not None, "hidden_states should have gradients"
    assert h.grad.abs().sum() > 0, "hidden_states gradients should be non-zero"


def test_gradients_flow_lm_head():
    from phase6.loss import compute_loss
    h, w, t, m, ew = _make_inputs()
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    loss.backward()
    assert w.grad is not None, "lm_head_weight should have gradients"
    assert w.grad.abs().sum() > 0, "lm_head_weight gradients should be non-zero"


# ---------------------------------------------------------------------------
# Mask weighting
# ---------------------------------------------------------------------------

def test_no_mask_zero_loss():
    """All mask=False → no positions contribute → loss should be 0."""
    from phase6.loss import compute_loss
    h, w, t, _, ew = _make_inputs()
    mask = torch.zeros_like(t, dtype=torch.bool)
    loss = compute_loss(h, t, mask, ew, w, TinyCfg())
    assert loss.item() == pytest.approx(0.0, abs=1e-6), \
        f"all-false mask should give 0 loss, got {loss.item()}"


def test_elbo_weight_scales_loss():
    """Higher ELBO weight → proportionally larger loss."""
    from phase6.loss import compute_loss
    torch.manual_seed(7)
    h, w, t, m, _ = _make_inputs(batch=4, seq=32, dim=32, vocab=256)
    ew1 = torch.ones(4, 32)
    ew2 = torch.full((4, 32), 3.0)
    loss1 = compute_loss(h.detach().requires_grad_(True), t, m, ew1, w, TinyCfg())
    loss2 = compute_loss(h.detach().requires_grad_(True), t, m, ew2, w, TinyCfg())
    ratio = loss2.item() / max(loss1.item(), 1e-8)
    assert 2.5 < ratio < 3.5, f"3x weight should give ~3x loss, got ratio {ratio:.2f}"


# ---------------------------------------------------------------------------
# Padding exclusion
# ---------------------------------------------------------------------------

def test_padding_excluded_from_normalization():
    """Padding tokens (target==pad_token_id) don't count in denominator."""
    from phase6.loss import compute_loss
    torch.manual_seed(99)
    h, w, t, m, ew = _make_inputs(batch=2, seq=16, dim=32, vocab=256)
    # Make half the sequence padding
    t[:, 8:] = 0  # pad_token_id=0
    m[:, 8:] = False  # don't mask padding
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    assert loss.item() > 0, "loss should still be positive from non-pad tokens"
    assert not torch.isnan(loss), "loss should not be NaN"


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_no_nan_inf():
    from phase6.loss import compute_loss
    torch.manual_seed(0)
    h, w, t, m, ew = _make_inputs(batch=4, seq=64, dim=64, vocab=512)
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    assert not torch.isnan(loss), "loss is NaN"
    assert not torch.isinf(loss), "loss is Inf"


# ---------------------------------------------------------------------------
# Small vs large vocab
# ---------------------------------------------------------------------------

def test_small_vocab():
    from phase6.loss import compute_loss
    h, w, t, m, ew = _make_inputs(vocab=256)
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    assert loss.item() > 0


def test_larger_vocab():
    from phase6.loss import compute_loss
    h, w, t, m, ew = _make_inputs(vocab=4096)
    loss = compute_loss(h, t, m, ew, w, TinyCfg())
    assert loss.item() > 0
