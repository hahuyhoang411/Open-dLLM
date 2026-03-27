"""Tests for Phase 6 schedule module — noise, ELBO, LR, CART."""

import torch
import pytest


def test_import_no_crash():
    """schedule.py must be importable without triggering config globals."""
    from phase6.schedule import sample_timesteps, apply_noise, compute_elbo_weight, get_lr_factor


def test_sample_timesteps_shape():
    from phase6.schedule import sample_timesteps
    t_blocks, t = sample_timesteps(batch_size=4, num_blocks=8, block_size=4, t_min=0.1)
    assert t_blocks.shape == (4, 8)
    assert t.shape == (4, 32)


def test_sample_timesteps_range():
    from phase6.schedule import sample_timesteps
    _, t = sample_timesteps(batch_size=16, num_blocks=4, block_size=8, t_min=0.1)
    assert t.min() >= 0.1
    assert t.max() < 1.0


def test_sample_timesteps_per_block_constant():
    """All tokens in the same block should have the same t."""
    from phase6.schedule import sample_timesteps
    _, t = sample_timesteps(batch_size=2, num_blocks=4, block_size=8, t_min=0.1)
    for b in range(2):
        for blk in range(4):
            block_vals = t[b, blk * 8:(blk + 1) * 8]
            assert torch.allclose(block_vals, block_vals[0].expand_as(block_vals))


def test_apply_noise_masks_tokens():
    from phase6.schedule import apply_noise
    targets = torch.randint(10, 100, (2, 32))
    t = torch.full((2, 32), 0.5)
    x_noisy, mask = apply_noise(targets, t, mask_token_id=0, block_size=8)
    assert mask.any(), "some tokens should be masked"
    assert (x_noisy[mask] == 0).all(), "masked positions should have mask_token_id"
    assert (x_noisy[~mask] == targets[~mask]).all(), "unmasked positions unchanged"


def test_apply_noise_min_one_per_block():
    """Per-block min-1-masked guarantee even at low t."""
    from phase6.schedule import apply_noise
    torch.manual_seed(42)
    targets = torch.randint(10, 100, (4, 32))
    t = torch.full((4, 32), 0.01)  # very low noise → most blocks empty without guarantee
    _, mask = apply_noise(targets, t, mask_token_id=0, block_size=8)
    mask_blocks = mask.view(4, 4, 8)
    per_block_count = mask_blocks.sum(dim=2)
    assert (per_block_count >= 1).all(), f"every block must have >=1 mask: {per_block_count}"


def test_apply_noise_respects_padding():
    from phase6.schedule import apply_noise
    targets = torch.randint(10, 100, (2, 16))
    targets[:, 12:] = 0  # pad_token_id=0
    t = torch.full((2, 16), 0.9)
    _, mask = apply_noise(targets, t, mask_token_id=1, pad_token_id=0, block_size=4)
    assert not mask[:, 12:].any(), "padding positions should never be masked"


def test_compute_elbo_weight():
    from phase6.schedule import compute_elbo_weight
    t = torch.tensor([0.1, 0.5, 1.0])
    w = compute_elbo_weight(t, t_min=0.1)
    assert torch.allclose(w, torch.tensor([10.0, 2.0, 1.0]))


def test_compute_elbo_weight_clamp():
    from phase6.schedule import compute_elbo_weight
    t = torch.tensor([0.01, 0.05])
    w = compute_elbo_weight(t, t_min=0.1)
    assert torch.allclose(w, torch.tensor([10.0, 10.0])), "values below t_min should be clamped"


def test_get_lr_factor_warmup():
    from phase6.schedule import get_lr_factor
    assert get_lr_factor(0, warmup_iters=100, decay_start=800, max_iters=1000) == pytest.approx(0.01)
    assert get_lr_factor(99, warmup_iters=100, decay_start=800, max_iters=1000) == pytest.approx(1.0)


def test_get_lr_factor_stable():
    from phase6.schedule import get_lr_factor
    assert get_lr_factor(500, warmup_iters=100, decay_start=800, max_iters=1000) == 1.0


def test_get_lr_factor_decay():
    from phase6.schedule import get_lr_factor
    factor = get_lr_factor(900, warmup_iters=100, decay_start=800, max_iters=1000)
    assert 0.0 < factor < 1.0
    assert get_lr_factor(1000, warmup_iters=100, decay_start=800, max_iters=1000) == 0.0


def test_compute_cart_weights_shape():
    from phase6.schedule import compute_cart_weights
    mask = torch.tensor([[True, False, True, False, True, False, True, False]])
    padding = torch.ones_like(mask)
    w = compute_cart_weights(mask, padding, p=0.1)
    assert w.shape == (1, 8)
    assert (w[~mask] == 0).all(), "unmasked positions should have weight 0"
    assert (w[mask] > 0).any(), "some masked positions should have positive weight"
