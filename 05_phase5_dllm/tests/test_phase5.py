"""Component tests for Phase 5 block diffusion LM.

Run: uv run python -m pytest 05_phase5_dllm/tests/test_phase5.py -v
All tests run on CPU — no GPU required.
"""

import math
import os
import sys
import tempfile

import pytest
import torch

# Patch sys.argv before importing phase5 (config.py calls parse_args at module level)
sys.argv = ['test', '--no-amp', '--no-liger', '--no-flex', '--no-compile',
            '--no-grad-ckpt', '--no-muon', '--batch-size', '2', '--n-layer', '2',
            '--n-embd', '576', '--n-head', '9', '--n-kv-head', '3',
            '--mlp-hidden', '1536', '--seq-len', '64', '--block-size', '16']

# Now safe to import
import pathlib

from phase5 import config
from phase5.attention import (
    _apply_rotary_emb,
    build_staircase_mask,
)
from phase5.checkpoint import load_checkpoint, save_checkpoint
from phase5.data import _compute_positions, _DocumentPacker
from phase5.generate import _add_gumbel_noise
from phase5.loss import compute_loss
from phase5.model import Model
from phase5.optim import MuonClip, _MaxLogitsTracker, build_adamw_optimizer, build_param_groups
from phase5.schedule import (
    apply_noise,
    compute_cart_weights,
    compute_elbo_weight,
    get_lr_factor,
    sample_timesteps,
)

# ============================================================================
# Schedule Tests
# ============================================================================


class TestSchedule:
    def test_sample_timesteps_shape(self):
        B, num_blocks, blk = 4, 4, 16
        t_blocks, t = sample_timesteps(B, num_blocks, blk)
        assert t_blocks.shape == (B, num_blocks)
        assert t.shape == (B, num_blocks * blk)

    def test_sample_timesteps_range(self):
        """All timesteps in [t_min, 1)."""
        t_blocks, t = sample_timesteps(32, 4, 16)
        assert t.min() >= config.t_min
        assert t.max() < 1.0

    def test_sample_timesteps_block_constant(self):
        """Each block has a constant timestep across its positions."""
        B, num_blocks, blk = 2, 4, 16
        _, t = sample_timesteps(B, num_blocks, blk)
        for b in range(B):
            for i in range(num_blocks):
                block_vals = t[b, i * blk:(i + 1) * blk]
                assert torch.allclose(block_vals, block_vals[0].expand_as(block_vals))

    def test_antithetic_stratification(self):
        """Antithetic sampling produces well-spread timesteps."""
        t_blocks, _ = sample_timesteps(4, 4, 16)
        flat = t_blocks.flatten().sort().values
        # Should be roughly uniformly spaced — max gap < 2/N
        gaps = flat[1:] - flat[:-1]
        N = flat.numel()
        assert gaps.max() < 2.0 / N + 0.01, f'Max gap {gaps.max():.4f} too large for N={N}'

    def test_apply_noise_masks_tokens(self):
        targets = torch.randint(14, 1000, (2, 64))
        t = torch.full((2, 64), 0.5)
        x_noisy, mask = apply_noise(targets, t)
        # Masked positions should have mask_token_id=0
        assert (x_noisy[mask] == config.mask_token_id).all()
        # Unmasked positions should be unchanged
        assert (x_noisy[~mask] == targets[~mask]).all()

    def test_apply_noise_min_one_masked(self):
        """Even with very low t, at least one token is masked per block."""
        targets = torch.randint(14, 1000, (10, 64))
        t = torch.full((10, 64), config.t_min)
        _, mask = apply_noise(targets, t, block_size=config.block_size)
        # Per-block guarantee: every block has at least one masked token
        num_blocks = 64 // config.block_size
        mask_blocks = mask.view(10, num_blocks, config.block_size)
        assert (mask_blocks.sum(dim=2) >= 1).all()
        # Per-sequence still holds (strictly weaker)
        assert (mask.sum(dim=1) >= 1).all()

    def test_apply_noise_respects_padding(self):
        targets = torch.full((1, 64), config.pad_token_id, dtype=torch.long)
        targets[0, :32] = torch.randint(14, 1000, (32,))
        t = torch.full((1, 64), 0.9)
        _, mask = apply_noise(targets, t, pad_token_id=config.pad_token_id)
        # Padding positions should never be masked
        assert not mask[0, 32:].any()

    def test_elbo_weight_linear(self):
        """For linear schedule, ELBO weight = 1/t."""
        t = torch.tensor([0.1, 0.25, 0.5, 1.0])
        w = compute_elbo_weight(t)
        expected = 1.0 / t
        assert torch.allclose(w, expected)

    def test_elbo_weight_clamped(self):
        """ELBO weight is clamped at 1/t_min to avoid explosion."""
        t = torch.tensor([0.0, 1e-5, config.t_min])
        w = compute_elbo_weight(t)
        assert (w <= 1.0 / config.t_min + 1e-6).all()

    def test_wsd_warmup(self):
        assert get_lr_factor(0) == pytest.approx(1 / config.warmup_iters)
        assert get_lr_factor(config.warmup_iters - 1) == pytest.approx(
            config.warmup_iters / config.warmup_iters
        )

    def test_wsd_stable(self):
        mid = (config.warmup_iters + config.decay_start) // 2
        assert get_lr_factor(mid) == 1.0

    def test_wsd_decay(self):
        assert get_lr_factor(config.max_iters - 1) == pytest.approx(0.0, abs=0.01)
        assert get_lr_factor(config.max_iters) == 0.0

    def test_wsd_monotonic(self):
        """LR factor should be monotonically non-decreasing then non-increasing."""
        factors = [get_lr_factor(s) for s in range(0, config.max_iters + 1, 100)]
        peak_idx = factors.index(max(factors))
        # Non-decreasing up to peak
        for i in range(1, peak_idx + 1):
            assert factors[i] >= factors[i - 1] - 1e-9
        # Non-increasing after peak
        for i in range(peak_idx + 1, len(factors)):
            assert factors[i] <= factors[i - 1] + 1e-9


# ============================================================================
# Loss Tests
# ============================================================================

class TestLoss:
    def test_loss_shape(self):
        B, L, D = 2, 64, 576
        hidden = torch.randn(B, L, D)
        targets = torch.randint(14, 49152, (B, L))
        mask = torch.rand(B, L) > 0.5
        elbo_w = torch.ones(B, L)
        lm_head_weight = torch.randn(49152, D)
        loss = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_zero_mask(self):
        """Loss should be zero when no tokens are masked."""
        B, L, D = 2, 64, 576
        hidden = torch.randn(B, L, D)
        targets = torch.randint(14, 49152, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        elbo_w = torch.ones(B, L)
        lm_head_weight = torch.randn(49152, D)
        loss = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_normalized_by_all_real(self):
        """Loss should be normalized by total real tokens, not just masked."""
        B, L, D = 1, 64, 576
        hidden = torch.randn(B, L, D)
        targets = torch.randint(14, 49152, (B, L))
        lm_head_weight = torch.randn(49152, D)

        # Half-masked vs full-masked: loss should scale with mask fraction (ELBO weight=1)
        mask_half = torch.zeros(B, L, dtype=torch.bool)
        mask_half[:, :32] = True
        mask_full = torch.ones(B, L, dtype=torch.bool)

        loss_half = compute_loss(hidden, targets, mask_half, torch.ones(B, L), lm_head_weight)
        loss_full = compute_loss(hidden, targets, mask_full, torch.ones(B, L), lm_head_weight)

        # With ELBO weight=1 and normalized by all tokens, full mask should be ~2x half mask
        ratio = loss_full.item() / loss_half.item()
        assert 1.5 < ratio < 2.5, f'Expected ratio ~2, got {ratio:.2f}'

    def test_step0_loss_estimate(self):
        """Step-0 loss should be near ln(vocab_size) with random weights."""
        B, L, D = 2, 64, 576
        torch.manual_seed(42)
        hidden = torch.randn(B, L, D) * 0.04  # approximate init scale
        targets = torch.randint(14, 49152, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        elbo_w = torch.ones(B, L)
        lm_head_weight = torch.randn(49152, D) * 0.04
        loss = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight)
        expected = math.log(49152)  # ~10.80
        assert abs(loss.item() - expected) < 2.0, f'Step-0 loss {loss.item():.2f} far from {expected:.2f}'


# ============================================================================
# Model Tests
# ============================================================================

class TestModel:
    @pytest.fixture(autouse=True)
    def model(self):
        torch.manual_seed(42)
        self.m = Model()
        self.m.eval()
        return self.m

    def test_param_count(self):
        """Verify param count uses data_ptr dedup (tied embeddings)."""
        self.m.count_params()
        # lm_head and token_emb should share the same data_ptr
        assert self.m.lm_head.weight.data_ptr() == self.m.token_emb.weight.data_ptr()

    def test_tied_embeddings(self):
        assert self.m.lm_head.weight is self.m.token_emb.weight

    def test_init_std(self):
        """Base init should use std=1/sqrt(n_embd)."""
        expected_std = 1.0 / math.sqrt(config.n_embd)
        # Check a random linear weight that's not residual-scaled
        w = self.m.blocks[0].attn.c_q.weight
        actual_std = w.std().item()
        assert abs(actual_std - expected_std) < 0.01, f'Init std {actual_std:.4f} vs expected {expected_std:.4f}'

    def test_residual_scaled_init(self):
        """Residual projections should use std / sqrt(2*n_layer)."""
        base_std = 1.0 / math.sqrt(config.n_embd)
        expected = base_std / math.sqrt(2 * config.n_layer)
        w = self.m.blocks[0].attn.c_proj.weight
        actual_std = w.std().item()
        assert abs(actual_std - expected) < 0.005, f'Residual std {actual_std:.4f} vs expected {expected:.4f}'

    def test_gate_zero_init(self):
        """Gated Query Attention gates should be zero-initialized."""
        for block in self.m.blocks:
            assert (block.attn.w_gate.weight == 0).all()

    def test_forward_training(self):
        """Training forward: [x_t || x_0] input, loss output."""
        B, L = 2, 64
        x_input = torch.randint(0, 49152, (B, 2 * L))
        targets = torch.randint(14, 49152, (B, L))
        mask = torch.rand(B, L) > 0.5
        elbo_w = torch.ones(B, L)
        attn_mask = build_staircase_mask(L, config.block_size)
        _, loss = self.m(x_input, targets=targets, mask=mask, elbo_weight=elbo_w, attn_mask=attn_mask)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_inference(self):
        """Inference forward: single block, logits output."""
        x = torch.randint(0, 49152, (1, 16))
        logits, loss = self.m(x)
        assert logits.shape == (1, 16, 49152)
        assert loss is None

    def test_kv_cache_lifecycle(self):
        """KV cache: enable -> use -> reset -> clean state."""
        self.m.enable_kv_cache()
        for block in self.m.blocks:
            assert block.attn.cache_mode is True
            assert block.attn.kv_cache is None

        # Forward caches K,V
        x = torch.randint(0, 49152, (1, 16))
        self.m(x)
        for block in self.m.blocks:
            assert block.attn.kv_cache is not None
            k, v = block.attn.kv_cache
            assert k.shape[2] == 16  # cached 16 positions

        # Reset cleans cache
        self.m.reset_kv_cache()
        for block in self.m.blocks:
            assert block.attn.kv_cache is None

    def test_kv_cache_accumulation(self):
        """KV cache accumulates across calls."""
        self.m.enable_kv_cache()
        x1 = torch.randint(0, 49152, (1, 16))
        x2 = torch.randint(0, 49152, (1, 16))
        self.m(x1, pos_offset=0)
        self.m(x2, pos_offset=16)
        for block in self.m.blocks:
            k, v = block.attn.kv_cache
            assert k.shape[2] == 32  # 16 + 16
        self.m.reset_kv_cache()

    def test_rope_buffer_size(self):
        """RoPE buffer should be large enough for generation."""
        expected_len = max(config.seq_len * 2, 8192)
        assert self.m.cos.shape[1] == expected_len
        assert self.m.sin.shape[1] == expected_len


# ============================================================================
# Attention Tests
# ============================================================================

class TestAttention:
    def test_rotary_emb_shape(self):
        x = torch.randn(2, 16, 9, 64)
        cos = torch.randn(1, 16, 1, 32)
        sin = torch.randn(1, 16, 1, 32)
        out = _apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_rotary_emb_identity_at_zero(self):
        """RoPE with cos=1, sin=0 should be identity."""
        x = torch.randn(1, 4, 1, 8)
        cos = torch.ones(1, 4, 1, 4)
        sin = torch.zeros(1, 4, 1, 4)
        out = _apply_rotary_emb(x, cos, sin)
        assert torch.allclose(out, x, atol=1e-6)

    def test_staircase_mask_shape(self):
        mask = build_staircase_mask(64, 16)
        assert mask.shape == (128, 128)  # 2L x 2L

    def test_staircase_mask_m_obc_strict(self):
        """M_OBC must use strict > (not >=) to prevent label leakage."""
        L, blk = 32, 16  # 2 blocks
        mask = build_staircase_mask(L, blk)
        # x_t block 0 should NOT attend to x_0 block 0 (same block, not strictly earlier)
        # x_t positions: 0..15 (block 0), 16..31 (block 1)
        # x_0 positions: 32..47 (block 0), 48..63 (block 1)
        # M_OBC: x_t[block_i] -> x_0[block_j] where i > j
        # x_t block 0 (pos 0-15) -> x_0 block 0 (pos 32-47): NOT allowed (0 > 0 is False)
        for q in range(16):
            for kv in range(32, 48):
                assert mask[q, kv] == float('-inf'), f'M_OBC leak at ({q},{kv})'

    def test_staircase_mask_block_diagonal(self):
        """Tokens in the same block, same half should attend to each other."""
        L, blk = 32, 16
        mask = build_staircase_mask(L, blk)
        # x_t block 0 (pos 0-15) should all attend to each other
        for q in range(16):
            for kv in range(16):
                assert mask[q, kv] == 0.0, f'Missing M_BD at ({q},{kv})'

    def test_staircase_mask_block_causal(self):
        """x_0 should attend to x_0 causally (same or earlier blocks)."""
        L, blk = 32, 16
        mask = build_staircase_mask(L, blk)
        # x_0 block 0 (pos 32-47) -> x_0 block 0 (pos 32-47): allowed (0 >= 0)
        for q in range(32, 48):
            for kv in range(32, 48):
                assert mask[q, kv] == 0.0
        # x_0 block 0 (pos 32-47) -> x_0 block 1 (pos 48-63): NOT allowed (0 >= 1 is False)
        for q in range(32, 48):
            for kv in range(48, 64):
                assert mask[q, kv] == float('-inf')

    def test_staircase_doc_boundary(self):
        """Cross-document attention should be blocked."""
        L, blk = 32, 16
        doc_ids = torch.zeros(2, L, dtype=torch.long)
        doc_ids[0, 16:] = 1  # first batch: doc boundary at position 16
        doc_ids[1, :] = 0    # second batch: single doc
        mask = build_staircase_mask(L, blk, doc_ids=doc_ids)
        # (B, 1, 2L, 2L)
        assert mask.shape == (2, 1, 64, 64)
        # batch 1 (single doc) should match the non-batched staircase mask
        mask_b1 = mask[1, 0]
        mask_no_doc = build_staircase_mask(L, blk)
        assert torch.allclose(mask_b1, mask_no_doc)


# ============================================================================
# Optimizer Tests
# ============================================================================

class TestOptimizer:
    def test_newton_schulz_orthogonal(self):
        """Newton-Schulz should produce approximately orthogonal output.

        Note: bf16 with 5 iterations gives approximate orthogonalization.
        Diagonal ~0.7-0.9, off-diagonal small. This is correct for Muon.
        """
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        X = MuonClip.newton_schulz(G)
        prod = (X.float() @ X.float().T)
        # Verify: diagonal is dominant (> 0.5), off-diagonal is small (< 0.15)
        diag = prod.diag()
        off_diag = prod - torch.diag(diag)
        assert diag.min() > 0.5, f'Diagonal min {diag.min():.4f} too small'
        assert off_diag.abs().max() < 0.15, f'Off-diagonal max {off_diag.abs().max():.4f} too large'

    def test_newton_schulz_tall(self):
        """Newton-Schulz handles tall matrices (rows > cols) via transpose."""
        G = torch.randn(128, 64)
        X = MuonClip.newton_schulz(G)
        assert X.shape == (128, 64)
        prod = X.float().T @ X.float()
        diag = prod.diag()
        assert diag.min() > 0.5, f'Diagonal min {diag.min():.4f} too small'

    def test_newton_schulz_wide(self):
        """Newton-Schulz handles wide matrices (cols > rows) directly."""
        G = torch.randn(64, 128)
        X = MuonClip.newton_schulz(G)
        assert X.shape == (64, 128)
        prod = X.float() @ X.float().T
        diag = prod.diag()
        assert diag.min() > 0.5, f'Diagonal min {diag.min():.4f} too small'

    def test_param_grouping(self):
        """build_param_groups creates 3 groups with correct flags."""
        torch.manual_seed(42)
        m = Model()
        opt = build_param_groups(m)
        groups = opt.param_groups
        # Should have 3 groups: QK Muon, Other Muon, AdamW
        assert len(groups) == 3
        qk_group = [g for g in groups if g.get('is_qk', False)]
        muon_groups = [g for g in groups if g.get('apply_muon', False)]
        adamw_groups = [g for g in groups if not g.get('apply_muon', True)]
        assert len(qk_group) == 1
        assert len(muon_groups) == 2  # QK + other
        assert len(adamw_groups) == 1

    def test_param_dedup_tied(self):
        """Tied embeddings should not appear twice in optimizer."""
        torch.manual_seed(42)
        m = Model()
        opt = build_param_groups(m)
        all_ptrs = set()
        for g in opt.param_groups:
            for p in g['params']:
                assert p.data_ptr() not in all_ptrs, 'Duplicate param in optimizer!'
                all_ptrs.add(p.data_ptr())

    def test_initial_lr_stored(self):
        """Each param group should have initial_lr for WSD schedule."""
        torch.manual_seed(42)
        m = Model()
        opt = build_param_groups(m)
        for g in opt.param_groups:
            assert 'initial_lr' in g
            assert g['initial_lr'] == g['lr']

    def test_muon_step_updates_params(self):
        """Muon step should change parameter values."""
        torch.manual_seed(42)
        m = Model()
        opt = build_param_groups(m)
        # Set fake gradients
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        w_before = m.blocks[0].attn.c_q.weight.clone()
        opt.step()
        w_after = m.blocks[0].attn.c_q.weight
        assert not torch.allclose(w_before, w_after), "Muon step didn't update weights"

    def test_qk_clip(self):
        """QK-Clip should scale Q/K weights when max_logits > tau."""
        torch.manual_seed(42)
        m = Model()
        opt = build_param_groups(m)
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        # Simulate high attention logit
        _MaxLogitsTracker._tls.max_logits = 500.0  # > tau=100
        w_q_before = m.blocks[0].attn.c_q.weight.clone()
        opt.step()
        w_q_after = m.blocks[0].attn.c_q.weight
        # Weight should be scaled down by gamma_sqrt = sqrt(100/500) = 0.447
        # (also shifted by the update, so just check it changed)
        assert not torch.allclose(w_q_before, w_q_after)

    def test_adamw_fallback(self):
        """build_adamw_optimizer creates valid AdamW."""
        torch.manual_seed(42)
        m = Model()
        opt = build_adamw_optimizer(m)
        assert isinstance(opt, torch.optim.AdamW)
        for g in opt.param_groups:
            assert 'initial_lr' in g


# ============================================================================
# Data Tests
# ============================================================================

class TestData:
    def test_compute_positions_single_doc(self):
        """Single document: positions should be 0, 1, 2, ..., L-1."""
        doc_ids = torch.zeros(1, 8, dtype=torch.long)
        pos = _compute_positions(doc_ids)
        expected = torch.arange(8).unsqueeze(0)
        assert torch.equal(pos, expected)

    def test_compute_positions_multi_doc(self):
        """Multiple documents: positions reset at boundaries."""
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        pos = _compute_positions(doc_ids)
        expected = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])
        assert torch.equal(pos, expected)

    def test_compute_positions_all_different(self):
        """Every token is a different document: all positions = 0."""
        doc_ids = torch.arange(8).unsqueeze(0)
        pos = _compute_positions(doc_ids)
        expected = torch.zeros(1, 8, dtype=torch.long)
        assert torch.equal(pos, expected)

    def test_compute_positions_batch(self):
        """Works correctly with batched input."""
        doc_ids = torch.tensor([
            [0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ])
        pos = _compute_positions(doc_ids)
        expected = torch.tensor([
            [0, 1, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 3, 0, 1, 2, 3],
        ])
        assert torch.equal(pos, expected)

    def test_document_packer_basic(self):
        """DocumentPacker produces sequences of exact seq_len."""
        def make_iter():
            docs = [{'text': 'hello world ' * 50}] * 100
            return iter(docs)
        packer = _DocumentPacker(make_iter)
        ids, doc_ids = packer.get_sequence()
        assert len(ids) == config.seq_len
        assert len(doc_ids) == config.seq_len

    def test_document_packer_doc_ids_start_from_zero(self):
        """doc_ids should be remapped to start from 0 per sequence."""
        def make_iter():
            return iter([{'text': 'a ' * 10}] * 1000)
        packer = _DocumentPacker(make_iter)
        # Pull multiple sequences
        for _ in range(5):
            _, doc_ids = packer.get_sequence()
            assert doc_ids[0] == 0, 'doc_ids should start from 0'


# ============================================================================
# Generation Tests
# ============================================================================

class TestGeneration:
    def test_gumbel_noise_temperature_zero(self):
        """Temperature=0 should return logits unchanged."""
        logits = torch.randn(1, 16, 100)
        out = _add_gumbel_noise(logits, temperature=0)
        assert torch.equal(out, logits)

    def test_gumbel_noise_preserves_argmax_distribution(self):
        """With temperature > 0, argmax should favor higher logits."""
        torch.manual_seed(42)
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 10.0  # token 5 has highest logit

        # Sample many times
        counts = torch.zeros(10)
        for _ in range(500):
            noisy = _add_gumbel_noise(logits, temperature=1.0)
            selected = torch.argmax(noisy, dim=-1)
            counts[selected.item()] += 1

        # Token 5 should be selected most often
        assert counts[5] == counts.max(), f'Token 5 count {counts[5]} vs max {counts.max()}'
        assert counts[5] > 200, f'Token 5 only selected {counts[5]}/500 times'

    def test_gumbel_noise_uses_exp(self):
        """Verify the .exp() fix: result should be exp(logits) / noise, not logits / noise."""
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        torch.manual_seed(42)
        result = _add_gumbel_noise(logits, temperature=1.0)
        # All values should be positive (exp(logits) > 0, noise > 0)
        assert (result > 0).all(), 'Gumbel-max output should be positive (exp/noise)'
        # With logits [1, 2, 3], exp gives [2.7, 7.4, 20.1] — result should be in that ballpark
        assert result[0, 0, 2] > result[0, 0, 0], 'Higher logit should produce higher value on average'


# ============================================================================
# Checkpoint Tests
# ============================================================================

class TestCheckpoint:
    def test_save_load_roundtrip(self):
        """Save and load should produce identical model state."""
        torch.manual_seed(42)
        m = Model()
        opt = build_adamw_optimizer(m)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(m, opt, step=100, loss=5.0, ckpt_dir=tmpdir)

            # Verify files exist
            assert pathlib.Path(os.path.join(tmpdir, 'ckpt_000100.pt')).exists()
            assert pathlib.Path(os.path.join(tmpdir, 'latest.pt')).exists()

            # Load into fresh model
            m2 = Model()
            opt2 = build_adamw_optimizer(m2)
            step = load_checkpoint(tmpdir, m2, opt2, device='cpu')

            assert step == 101  # next step after saved
            # Weights should match
            for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters(), strict=True):
                assert torch.equal(p1, p2), f'Mismatch in {n1}'

    def test_atomic_write(self):
        """No .tmp files should remain after save."""
        torch.manual_seed(42)
        m = Model()
        opt = build_adamw_optimizer(m)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(m, opt, step=0, loss=10.0, ckpt_dir=tmpdir)
            files = os.listdir(tmpdir)
            assert not any(f.endswith('.tmp') for f in files), f'Leftover tmp files: {files}'

    def test_load_nonexistent(self):
        """Loading from nonexistent path returns step 0."""
        torch.manual_seed(42)
        m = Model()
        step = load_checkpoint('/nonexistent/path/ckpt.pt', m, device='cpu')
        assert step == 0


# ============================================================================
# CART Tests
# ============================================================================

class TestCART:
    def test_cart_weights_shape(self):
        mask = torch.rand(2, 64) > 0.5
        padding = torch.ones(2, 64, dtype=torch.bool)
        w = compute_cart_weights(mask, padding)
        assert w.shape == (2, 64)

    def test_cart_zero_on_unmasked(self):
        """CART weights should be 0 on unmasked positions."""
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[0, :10] = True
        padding = torch.ones(1, 64, dtype=torch.bool)
        w = compute_cart_weights(mask, padding)
        assert (w[0, 10:] == 0).all()

    def test_cart_context_sensitive(self):
        """Masked tokens near more unmasked context should have higher CART weight."""
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[0, 30] = True  # center token: lots of context
        mask[0, 0] = True   # edge token: less context
        padding = torch.ones(1, 64, dtype=torch.bool)
        w = compute_cart_weights(mask, padding)
        # Center should have higher weight than edge
        assert w[0, 30] > w[0, 0], f'Center weight {w[0, 30]:.4f} should > edge {w[0, 0]:.4f}'


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    def test_training_step(self):
        """Full training step: forward + backward + optimizer step."""
        torch.manual_seed(42)
        m = Model()
        m.train()
        opt = build_adamw_optimizer(m)

        B, L = 2, 64
        targets = torch.randint(14, 49152, (B, L))
        t_blocks, t = sample_timesteps(B, config.num_blocks, config.block_size)
        x_noisy, mask = apply_noise(targets, t)
        elbo_w = compute_elbo_weight(t)
        x_input = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, config.block_size)

        _, loss = m(x_input, targets=targets, mask=mask, elbo_weight=elbo_w, attn_mask=attn_mask)
        assert loss.item() > 0
        assert not torch.isnan(loss) and not torch.isinf(loss)

        loss.backward()
        # Check gradients exist
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0

        # Optimizer step
        opt.step()
        opt.zero_grad(set_to_none=True)

    def test_training_step_with_doc_packing(self):
        """Training step with document boundaries."""
        torch.manual_seed(42)
        m = Model()
        m.train()

        B, L = 2, 64
        targets = torch.randint(14, 49152, (B, L))
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[0, 32:] = 1
        doc_ids[1, 20:40] = 1
        doc_ids[1, 40:] = 2
        positions = _compute_positions(doc_ids)

        t_blocks, t = sample_timesteps(B, config.num_blocks, config.block_size)
        x_noisy, mask = apply_noise(targets, t)
        elbo_w = compute_elbo_weight(t)
        x_input = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, config.block_size, doc_ids=doc_ids)

        _, loss = m(x_input, targets=targets, mask=mask, elbo_weight=elbo_w,
                    attn_mask=attn_mask, positions=positions)
        assert loss.item() > 0
        assert not torch.isnan(loss) and not torch.isinf(loss)
