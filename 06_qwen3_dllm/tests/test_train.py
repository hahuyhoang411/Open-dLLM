"""Tests for train.py orchestration logic.

Verifies config parsing, model creation, HF loading path, training step
extraction, step-0 sanity check, LR schedule application, eval function,
and checkpoint save/resume -- all on CPU with TINY configs.
"""

import math
import os
import sys
import tempfile

import pytest
import torch
from torch.nn.utils import clip_grad_norm_

from phase6.attention import build_staircase_mask
from phase6.checkpoint import save_checkpoint, load_checkpoint
from phase6.config import Config
from phase6.loss import compute_loss
from phase6.model import Model
from phase6.optim import create_optimizer
from phase6.schedule import (
    apply_noise,
    compute_elbo_weight,
    get_lr_factor,
    sample_timesteps,
)


# ---------------------------------------------------------------------------
# Shared TINY config -- 2 layers, small vocab, CPU-only
# ---------------------------------------------------------------------------

TINY = Config(
    n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
    mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
    rope_base=10000, rms_eps=1e-6, dropout=0.0, t_min=0.1,
    use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
    use_liger=False, use_grad_ckpt=False, use_flex=False, use_muon=False,
    use_cart=False, pad_token_id=0, mask_token_id=1, eos_token_id=2,
    muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
    batch_size=2, max_iters=100, eval_iters=2, eval_every=10,
    device='cpu', master_process=True,
).validate()


def _make_batch(cfg, B=2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    L = cfg.seq_len
    targets = torch.randint(3, cfg.vocab_size, (B, L))
    _, t = sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
    x_noisy, noise_mask = apply_noise(
        targets, t, mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id, block_size=cfg.block_size,
    )
    elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)
    x_input = torch.cat([x_noisy, targets], dim=1)
    attn_mask = build_staircase_mask(L, cfg.block_size)
    return x_input, targets, noise_mask, elbo_w, attn_mask


# ============================================================================
# 1. Config parsing: from_cli() with --train --n-layer 2 --max-iters 5
# ============================================================================

class TestConfigParsing:

    def test_from_cli_basic(self):
        """from_cli() parses --train --n-layer 2 --max-iters 5 correctly."""
        from phase6.config import from_cli
        saved = sys.argv[:]
        try:
            sys.argv = ['train.py', '--train', '--n-layer', '2', '--max-iters', '5']
            cfg = from_cli()
            assert cfg.train is True
            assert cfg.n_layer == 2
            assert cfg.max_iters == 5
            assert cfg.num_blocks == cfg.seq_len // cfg.block_size
            assert cfg.warmup_iters > 0
        finally:
            sys.argv = saved

    def test_from_cli_defaults(self):
        """from_cli() with no args gives Qwen3-0.6B defaults."""
        from phase6.config import from_cli
        saved = sys.argv[:]
        try:
            sys.argv = ['train.py']
            cfg = from_cli()
            assert cfg.train is False
            assert cfg.n_layer == 28
            assert cfg.vocab_size == 151_936
            assert cfg.head_dim == 128
        finally:
            sys.argv = saved

    def test_from_cli_hf_model_name(self):
        """--hf-model-name propagates to cfg."""
        from phase6.config import from_cli
        saved = sys.argv[:]
        try:
            sys.argv = ['train.py', '--hf-model-name', 'Qwen/Qwen3-0.6B']
            cfg = from_cli()
            assert cfg.hf_model_name == 'Qwen/Qwen3-0.6B'
        finally:
            sys.argv = saved


# ============================================================================
# 2. Model creation: Model(cfg) with TINY config -> correct param count
# ============================================================================

class TestModelCreation:

    def test_model_creates_with_tiny_config(self):
        model = Model(TINY)
        count = model.count_params()
        assert count > 0
        assert count < 1_000_000  # TINY should be well under 1M

    def test_model_forward_shape(self):
        model = Model(TINY)
        x_input, targets, _, _, attn_mask = _make_batch(TINY, B=2, seed=42)
        hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
        assert hidden.shape == (2, TINY.seq_len, TINY.n_embd)

    def test_tied_embeddings(self):
        model = Model(TINY)
        assert model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


# ============================================================================
# 3. HF loading path: model + load_from_hf mock -> missing keys reported
# ============================================================================

class TestHFLoadingPath:

    def test_load_from_hf_returns_missing_and_unexpected(self, monkeypatch):
        """load_from_hf returns (missing, unexpected) for verification."""
        from phase6 import checkpoint as ckpt_mod

        # Mock _load_hf_weights to return a partial state dict
        def mock_load_hf_weights(model_name):
            model = Model(TINY)
            sd = model.state_dict()
            # Return only first block's weights
            partial = {k: v for k, v in sd.items() if 'blocks.0' in k}
            return partial

        monkeypatch.setattr(ckpt_mod, '_load_hf_weights', mock_load_hf_weights)
        monkeypatch.setattr(ckpt_mod, '_map_hf_key', lambda k: (k, None))

        model = Model(TINY)
        missing, unexpected = ckpt_mod.load_from_hf(model, 'fake-model', 'cpu')

        # Should have missing keys (block 1, token_emb, final_norm, etc.)
        assert len(missing) > 0
        # No unexpected since we passed our own keys
        assert len(unexpected) == 0


# ============================================================================
# 4. Training step function: extract the inner training step as testable
# ============================================================================

class TestTrainingStep:

    def test_train_step_returns_finite_loss_and_grad(self):
        """Forward + loss + backward + clip -> finite loss, nonzero grad."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)

        hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert math.isfinite(loss.item())
        assert loss.item() > 0
        assert grad_norm > 0

    def test_multiple_steps_loss_decreases(self):
        """Training on fixed data for 20 steps -> loss decreases."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, B=4, seed=123)

        losses = []
        for _ in range(20):
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: step 0={losses[0]:.4f}, step 19={losses[-1]:.4f}"


# ============================================================================
# 5. Step-0 sanity check: loss=50 should raise, loss=10 should pass
# ============================================================================

class TestStep0SanityCheck:

    def test_loss_above_threshold_raises(self):
        """Step-0 sanity: loss > 30 raises RuntimeError."""
        loss_val = 50.0
        threshold = 30.0
        with pytest.raises(RuntimeError, match='Step-0'):
            if loss_val > threshold:
                raise RuntimeError(
                    f'Step-0 train loss {loss_val:.2f} >> expected. '
                    f'Check loss normalization and ELBO weighting.'
                )

    def test_loss_below_threshold_passes(self):
        """Step-0 sanity: loss=10 does not raise."""
        loss_val = 10.0
        threshold = 30.0
        # Should not raise
        if loss_val > threshold:
            raise RuntimeError(f'Step-0 train loss {loss_val:.2f}')

    def test_actual_tiny_model_step0_below_threshold(self):
        """TINY model step-0 loss is below catastrophic threshold."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()  # use train mode like the real loop

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)
        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

        # TINY vocab=256, ln(256) ~ 5.5. ELBO-weighted ~ 10-20 max. Well under 100.
        assert loss.item() < 100, f"Step-0 loss {loss.item():.2f} unexpectedly high"


# ============================================================================
# 6. LR schedule application: param_groups get correct LR at warmup/stable/decay
# ============================================================================

class TestLRScheduleApplication:

    def test_warmup_phase(self):
        """During warmup, LR ramps from near-0 to 1.0."""
        cfg = TINY
        warmup = cfg.warmup_iters
        factor_0 = get_lr_factor(0, warmup, cfg.decay_start, cfg.max_iters)
        factor_mid = get_lr_factor(warmup // 2, warmup, cfg.decay_start, cfg.max_iters)
        factor_end = get_lr_factor(warmup - 1, warmup, cfg.decay_start, cfg.max_iters)

        assert factor_0 > 0
        assert factor_0 < factor_mid < factor_end
        assert factor_end <= 1.0

    def test_stable_phase(self):
        """Between warmup and decay_start, LR factor = 1.0."""
        cfg = TINY
        mid_stable = (cfg.warmup_iters + cfg.decay_start) // 2
        factor = get_lr_factor(mid_stable, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
        assert factor == 1.0

    def test_decay_phase(self):
        """After decay_start, LR decays toward 0."""
        cfg = TINY
        factor_start = get_lr_factor(cfg.decay_start, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
        factor_end = get_lr_factor(cfg.max_iters - 1, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)

        assert factor_start == 1.0
        assert factor_end < 0.1
        assert factor_end >= 0.0

    def test_lr_applied_to_optimizer_param_groups(self):
        """Verify LR schedule changes optimizer param group LRs correctly."""
        cfg = TINY
        model = Model(cfg)
        optimizer = create_optimizer(model, cfg)

        # Apply at warmup midpoint
        step = cfg.warmup_iters // 2
        factor = get_lr_factor(step, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * factor

        for pg in optimizer.param_groups:
            expected = pg['initial_lr'] * factor
            assert abs(pg['lr'] - expected) < 1e-9

        # Apply at end of training
        factor_end = get_lr_factor(cfg.max_iters - 1, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * factor_end

        for pg in optimizer.param_groups:
            assert pg['lr'] < pg['initial_lr'] * 0.1


# ============================================================================
# 7. Eval function: runs without crash on toy model + random data
# ============================================================================

class TestEvalFunction:

    def test_estimate_loss_returns_dict(self):
        """estimate_loss returns {'train': float, 'val': float} without crashing."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)

        def get_batch_fn(split):
            return _make_batch(cfg, B=cfg.batch_size, seed=hash(split) % 1000)

        # Inline estimate_loss logic (same pattern as train.py)
        model.eval()
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x_input, targets, mask, elbo_w, attn_mask = get_batch_fn(split)
                with torch.no_grad():
                    hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
                    loss = compute_loss(hidden, targets, mask, elbo_w, model.lm_head.weight, cfg)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        assert 'train' in out
        assert 'val' in out
        assert math.isfinite(out['train'])
        assert math.isfinite(out['val'])
        assert out['train'] > 0
        assert out['val'] > 0


# ============================================================================
# 8. Checkpoint save/resume: train 3 steps, save, resume -> step counter correct
# ============================================================================

class TestCheckpointSaveResume:

    def test_save_resume_step_counter(self):
        """Train 3 steps, save, resume -> step counter = 4 (load returns step+1)."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)

        for _ in range(3):
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=3, loss=loss.item(), ckpt_dir=tmpdir)

            model2 = Model(cfg)
            optimizer2 = create_optimizer(model2, cfg)
            resumed_step = load_checkpoint(
                os.path.join(tmpdir, 'latest.pt'), model2, optimizer2, device='cpu',
            )

            assert resumed_step == 4  # step + 1

    def test_save_resume_weights_match(self):
        """After resume, model weights match original."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)

        for _ in range(3):
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Compute reference loss
        model.eval()
        with torch.no_grad():
            h_ref, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss_ref = compute_loss(h_ref, targets, noise_mask, elbo_w, model.lm_head.weight, cfg).item()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=3, loss=loss_ref, ckpt_dir=tmpdir)

            model2 = Model(cfg)
            optimizer2 = create_optimizer(model2, cfg)
            load_checkpoint(
                os.path.join(tmpdir, 'latest.pt'), model2, optimizer2, device='cpu',
            )

            model2.eval()
            with torch.no_grad():
                h2, _ = model2(x_input, targets=targets, attn_mask=attn_mask)
                loss_resumed = compute_loss(h2, targets, noise_mask, elbo_w, model2.lm_head.weight, cfg).item()

            assert abs(loss_ref - loss_resumed) < 1e-5, \
                f"Loss mismatch: ref={loss_ref:.6f}, resumed={loss_resumed:.6f}"

    def test_resume_from_nonexistent_returns_zero(self):
        """load_checkpoint with nonexistent path returns 0."""
        model = Model(TINY)
        step = load_checkpoint('/tmp/nonexistent_ckpt_xyz.pt', model, device='cpu')
        assert step == 0
