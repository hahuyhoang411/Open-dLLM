"""Component tests for Phase 5 block diffusion LM.

Run: uv run python -m pytest 05_optimized_dllm/tests/test_phase5.py -v
All tests run on CPU — no GPU required.
"""

import math
import os
import sys
import tempfile

import pytest
import torch

# Patch sys.argv before importing phase5 (config.py calls parse_args at module level)
sys.argv = [
  'test',
  '--no-amp',
  '--no-liger',
  '--no-flex',
  '--no-compile',
  '--no-grad-ckpt',
  '--no-muon',
  '--batch-size',
  '2',
  '--n-layer',
  '2',
  '--n-embd',
  '576',
  '--n-head',
  '9',
  '--n-kv-head',
  '3',
  '--mlp-hidden',
  '1536',
  '--seq-len',
  '64',
  '--block-size',
  '16',
]

# Now safe to import
import pathlib

from phase5 import config
from phase5.attention import (
  MultiHeadAttention,
  _apply_rotary_emb,
  build_staircase_mask,
)
from phase5.checkpoint import load_checkpoint, save_checkpoint
from phase5.data import _compute_positions, _DocumentPacker, _PreTokenizedPacker
from phase5.generate import _add_gumbel_noise, generate
from phase5.loss import compute_loss
from phase5.model import Model

from torch.nn import functional as F
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
        block_vals = t[b, i * blk : (i + 1) * blk]
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
    assert get_lr_factor(config.warmup_iters - 1) == pytest.approx(config.warmup_iters / config.warmup_iters)

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
    """Training forward: [x_t || x_0] input, hidden states + external loss."""
    B, L = 2, 64
    x_input = torch.randint(0, 49152, (B, 2 * L))
    targets = torch.randint(14, 49152, (B, L))
    mask = torch.rand(B, L) > 0.5
    elbo_w = torch.ones(B, L)
    attn_mask = build_staircase_mask(L, config.block_size)
    hidden, out = self.m(x_input, targets=targets, attn_mask=attn_mask)
    assert hidden.shape == (B, L, config.n_embd)
    assert out is None  # loss computed externally in Phase 5
    loss = compute_loss(hidden, targets, mask, elbo_w, self.m.lm_head.weight)
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
    doc_ids[1, :] = 0  # second batch: single doc
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
    prod = X.float() @ X.float().T
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
    """QK-Clip scales Q/K weights by sqrt(tau/max_logits) when triggered."""
    import math

    torch.manual_seed(42)
    m = Model()
    opt = build_param_groups(m)
    # Zero all grads so only QK-clip scaling acts (no Muon/AdamW update)
    for p in m.parameters():
      if p.requires_grad:
        p.grad = torch.zeros_like(p)
    # Also zero weight decay for the QK group so p *= 1.0 before clip
    for g in opt.param_groups:
      if g.get('is_qk', False):
        g['weight_decay'] = 0.0
    _MaxLogitsTracker._tls.max_logits = 400.0  # > tau=100
    w_q_before = m.blocks[0].attn.c_q.weight.clone()
    opt.step()
    w_q_after = m.blocks[0].attn.c_q.weight
    # With zero grad + zero WD, only QK-clip acts: p *= sqrt(100/400) = 0.5
    expected_scale = math.sqrt(100.0 / 400.0)
    assert torch.allclose(w_q_after, w_q_before * expected_scale, atol=1e-5), (
      f'Expected scale {expected_scale}, got ratio {(w_q_after / w_q_before).mean():.4f}'
    )

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

  @staticmethod
  def _make_local_packer(docs, tmp_dir):
    """Create a _PreTokenizedPacker backed by a local Arrow dataset."""
    from datasets import Dataset, Features, Sequence, Value
    import numpy as np

    features = Features({'input_ids': Sequence(Value('uint16'))})
    ds = Dataset.from_dict({'input_ids': docs}, features=features)
    ds_path = os.path.join(tmp_dir, 'test_ds')
    ds.save_to_disk(ds_path)

    class LocalPacker(_PreTokenizedPacker):
      def __init__(self, path, **kwargs):
        from datasets import load_from_disk

        ds_local = load_from_disk(path)
        self._ds = ds_local
        self._n = len(ds_local)
        self._rng = np.random.RandomState(42)
        self._eos_id = config.eos_token_id
        self._buf = []
        self._order = self._rng.permutation(self._n)
        self._cursor = 0

    return LocalPacker(ds_path)

  def test_pretokenized_packer_basic(self):
    """_PreTokenizedPacker produces sequences of exact seq_len."""
    import numpy as np
    import shutil

    rng = np.random.RandomState(42)
    docs = []
    for _ in range(200):
      length = rng.randint(50, 500)
      ids = rng.randint(3, config.vocab_size, size=length).tolist() + [config.eos_token_id]
      docs.append(ids)

    tmp_dir = tempfile.mkdtemp()
    packer = self._make_local_packer(docs, tmp_dir)

    for _ in range(10):
      ids, doc_ids = packer.get_sequence()
      assert len(ids) == config.seq_len
      assert len(doc_ids) == config.seq_len

    shutil.rmtree(tmp_dir)

  def test_pretokenized_packer_doc_ids(self):
    """doc_ids start from 0 and increment at EOS boundaries."""
    import shutil

    docs = [[10, 20, 30, config.eos_token_id]] * 500
    tmp_dir = tempfile.mkdtemp()
    packer = self._make_local_packer(docs, tmp_dir)
    ids, doc_ids = packer.get_sequence()

    assert doc_ids[0] == 0
    assert max(doc_ids) >= 1
    for i in range(1, len(doc_ids)):
      assert doc_ids[i] >= doc_ids[i - 1]
    # EOS token gets same doc_id as preceding tokens, then increments
    for i, tid in enumerate(ids):
      if tid == config.eos_token_id and i + 1 < len(ids):
        assert doc_ids[i + 1] == doc_ids[i] + 1

    shutil.rmtree(tmp_dir)

  def test_pretokenized_packer_epoch_boundary(self):
    """Packer reshuffles when all docs consumed (epoch boundary)."""
    import shutil

    docs = [[i + 3, config.eos_token_id] for i in range(10)]
    tmp_dir = tempfile.mkdtemp()
    packer = self._make_local_packer(docs, tmp_dir)
    # 10 docs × 2 tokens = 20 tokens per epoch
    # seq_len=64 requires 4+ epochs to fill one sequence
    ids, doc_ids = packer.get_sequence()
    assert len(ids) == config.seq_len
    unique_tokens = set(t for t in ids if t != config.eos_token_id)
    assert len(unique_tokens) <= 10

    shutil.rmtree(tmp_dir)

  def test_pretokenized_packer_matches_document_packer(self):
    """Both packers produce consistent doc_id semantics."""
    import shutil

    docs = [[100, 200, 300, config.eos_token_id]] * 500
    tmp_dir = tempfile.mkdtemp()
    packer = self._make_local_packer(docs, tmp_dir)
    ids, doc_ids = packer.get_sequence()

    assert doc_ids[0] == 0
    for i in range(1, len(doc_ids)):
      assert doc_ids[i] >= doc_ids[i - 1]

    # Verify positions computed from doc_ids reset at boundaries
    doc_ids_t = torch.tensor([doc_ids], dtype=torch.long)
    pos = _compute_positions(doc_ids_t)
    assert pos[0, 0].item() == 0
    for i in range(1, len(doc_ids)):
      if doc_ids[i] != doc_ids[i - 1]:
        assert pos[0, i].item() == 0

    shutil.rmtree(tmp_dir)

  def test_val_loader_persists_between_batches(self, monkeypatch):
    import phase5.data as data_mod

    class DummyPacker:
      created = 0

      def __init__(self):
        DummyPacker.created += 1

      def get_sequence(self):
        return [config.eos_token_id] * config.seq_len, [0] * config.seq_len

    monkeypatch.setattr(data_mod, '_DocumentPacker', lambda make_iter: DummyPacker())
    monkeypatch.setattr(data_mod, '_make_val_iter', lambda: iter([{'text': 'x'}]))
    data_mod._val_packer = None

    data_mod.get_batch('val')
    data_mod.get_batch('val')

    assert DummyPacker.created == 1

  def test_reset_val_loader_recreates_packer(self, monkeypatch):
    import phase5.data as data_mod

    class DummyPacker:
      def get_sequence(self):
        return [config.eos_token_id] * config.seq_len, [0] * config.seq_len

    monkeypatch.setattr(data_mod, '_DocumentPacker', lambda make_iter: DummyPacker())
    monkeypatch.setattr(data_mod, '_make_val_iter', lambda: iter([{'text': 'x'}]))
    data_mod._val_packer = None

    data_mod.get_batch('val')
    first = data_mod._val_packer
    data_mod.reset_val_loader()
    second = data_mod._val_packer

    assert first is not second


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
    mask[0, 0] = True  # edge token: less context
    padding = torch.ones(1, 64, dtype=torch.bool)
    w = compute_cart_weights(mask, padding)
    # Center should have higher weight than edge
    assert w[0, 30] > w[0, 0], f'Center weight {w[0, 30]:.4f} should > edge {w[0, 0]:.4f}'


# ============================================================================
# FP8 Tests
# ============================================================================


class TestFP8:
  def test_float8_linear_from_float(self):
    """Float8Linear.from_float shares weight with original module."""
    from phase5.fp8 import Float8Linear

    linear = torch.nn.Linear(576, 1536, bias=False)
    fp8 = Float8Linear.from_float(linear)
    assert fp8.weight is linear.weight
    assert fp8.weight.data_ptr() == linear.weight.data_ptr()

  def test_convert_counts(self):
    """convert_to_float8_training converts the right number of layers."""
    from phase5.fp8 import Float8Linear, convert_to_float8_training

    torch.manual_seed(42)
    m = Model()
    convert_to_float8_training(m, module_filter_fn=lambda mod, fqn: fqn != 'lm_head')
    num_fp8 = sum(1 for mod in m.modules() if isinstance(mod, Float8Linear))
    # 8 linear layers per block x n_layer blocks
    # Per block: c_q, c_k, c_v, c_proj, w_gate + gate_proj, up_proj, down_proj = 8
    assert num_fp8 == 8 * config.n_layer, f'Expected {8 * config.n_layer} FP8, got {num_fp8}'
    assert isinstance(m.lm_head, torch.nn.Linear)
    assert not isinstance(m.lm_head, Float8Linear)

  def test_disable_fp8_restores(self):
    """disable_fp8 context manager swaps modules and restores them."""
    from phase5.fp8 import Float8Linear, convert_to_float8_training, disable_fp8

    torch.manual_seed(42)
    m = Model()
    convert_to_float8_training(m)
    assert isinstance(m.blocks[0].attn.c_q, Float8Linear)
    with disable_fp8(m):
      assert not isinstance(m.blocks[0].attn.c_q, Float8Linear)
      assert isinstance(m.blocks[0].attn.c_q, torch.nn.Linear)
    assert isinstance(m.blocks[0].attn.c_q, Float8Linear)

  def test_fp8_module_structure(self):
    """Float8Linear has correct in/out features and shared weight."""
    from phase5.fp8 import Float8Linear

    linear = torch.nn.Linear(576, 576, bias=False)
    fp8 = Float8Linear.from_float(linear)
    assert fp8.in_features == 576
    assert fp8.out_features == 576
    assert fp8.weight is linear.weight

  def test_optimizer_after_fp8_conversion(self):
    """Optimizer params still reference correct tensors after FP8 conversion."""
    from phase5.fp8 import convert_to_float8_training

    torch.manual_seed(42)
    m = Model()
    opt = build_adamw_optimizer(m)
    q_weight_ptr = m.blocks[0].attn.c_q.weight.data_ptr()
    convert_to_float8_training(m, module_filter_fn=lambda mod, fqn: fqn != 'lm_head')
    assert m.blocks[0].attn.c_q.weight.data_ptr() == q_weight_ptr
    opt_ptrs = {p.data_ptr() for g in opt.param_groups for p in g['params']}
    assert q_weight_ptr in opt_ptrs


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

    hidden, _ = m(x_input, targets=targets, attn_mask=attn_mask)
    loss = compute_loss(hidden, targets, mask, elbo_w, m.lm_head.weight)
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

    hidden, _ = m(x_input, targets=targets, attn_mask=attn_mask, positions=positions)
    loss = compute_loss(hidden, targets, mask, elbo_w, m.lm_head.weight)
    assert loss.item() > 0
    assert not torch.isnan(loss) and not torch.isinf(loss)


# ============================================================================
# Tokenizer Tests
# ============================================================================


class TestTokenizer:
  def test_vocab_size(self):
    from phase5.tokenizer import load_tokenizer

    tok = load_tokenizer()
    assert tok.get_vocab_size() == 49_152

  def test_encode_decode_roundtrip(self):
    from phase5.tokenizer import encode, decode

    text = 'Hello, world! This is a test.'
    ids = encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0
    reconstructed = decode(ids)
    assert text in reconstructed or reconstructed.strip() == text.strip()

  def test_encode_produces_valid_ids(self):
    from phase5.tokenizer import encode

    ids = encode('The quick brown fox jumps over the lazy dog.')
    # All IDs should be >= 14 (specials are 0-13, real tokens start at 14)
    assert all(i >= 14 for i in ids), f'Got special token ID in normal text: {[i for i in ids if i < 14]}'
    # All IDs should be < vocab_size
    assert all(i < 49_152 for i in ids)

  def test_special_tokens_not_in_normal_text(self):
    from phase5.tokenizer import encode

    ids = encode('A normal English sentence without special tokens.')
    assert config.mask_token_id not in ids
    assert config.eos_token_id not in ids
    assert config.pad_token_id not in ids

  def test_encode_empty_string(self):
    from phase5.tokenizer import encode

    ids = encode('')
    assert isinstance(ids, list)
    assert len(ids) == 0


# ============================================================================
# Generate Tests (end-to-end)
# ============================================================================


class TestGenerateEndToEnd:
  @pytest.fixture(autouse=True)
  def model(self):
    torch.manual_seed(42)
    self.m = Model()
    self.m.eval()
    return self.m

  def test_generate_produces_output(self):
    """generate() returns non-empty decoded text."""
    from phase5.tokenizer import encode, decode

    result = generate(self.m, encode, decode, prompt='Hello', max_new_tokens=32, denoise_steps=3, temperature=0.8)
    assert isinstance(result, str)
    assert len(result) > 0

  def test_generate_without_prompt(self):
    """generate() works with empty prompt."""
    from phase5.tokenizer import encode, decode

    result = generate(self.m, encode, decode, prompt='', max_new_tokens=16, denoise_steps=2, temperature=1.0)
    assert isinstance(result, str)

  def test_generate_respects_max_tokens(self):
    """Output tokens should not vastly exceed max_new_tokens."""
    from phase5.tokenizer import encode, decode

    result = generate(self.m, encode, decode, prompt='The', max_new_tokens=16, denoise_steps=2, temperature=0.5)
    # Tokens can slightly exceed max_new_tokens due to block granularity
    # (block_size=16), but should not be absurdly long
    result_ids = encode(result) if result else []
    prompt_ids = encode('The')
    generated_count = len(result_ids) - len(prompt_ids)
    # Allow up to 2x block_size overshoot due to block granularity
    assert generated_count <= 16 + 2 * config.block_size

  def test_generate_cleans_up_cache(self):
    """generate() should reset KV cache on exit (no stale state)."""
    from phase5.tokenizer import encode, decode

    generate(self.m, encode, decode, prompt='Test', max_new_tokens=16, denoise_steps=2)
    # After generate, cache should be clean
    for block in self.m.blocks:
      assert block.attn.kv_cache is None
      assert block.attn.cache_mode is False

  def test_generate_restores_training_mode(self):
    """generate() should restore model.training state."""
    from phase5.tokenizer import encode, decode

    self.m.train()
    assert self.m.training is True
    generate(self.m, encode, decode, prompt='Hi', max_new_tokens=16, denoise_steps=2)
    assert self.m.training is True

  def test_generate_temperature_zero(self):
    """Temperature=0 should be deterministic (greedy)."""
    from phase5.tokenizer import encode, decode

    torch.manual_seed(42)
    r1 = generate(self.m, encode, decode, prompt='A', max_new_tokens=16, denoise_steps=3, temperature=0)
    torch.manual_seed(42)
    r2 = generate(self.m, encode, decode, prompt='A', max_new_tokens=16, denoise_steps=3, temperature=0)
    assert r1 == r2


# ============================================================================
# Multi-Head Attention Tests (isolated)
# ============================================================================


class TestMultiHeadAttention:
  @pytest.fixture(autouse=True)
  def attn(self):
    from phase5.model import _make_rms_norm

    torch.manual_seed(42)
    self.att = MultiHeadAttention(make_norm_fn=_make_rms_norm)
    self.att.eval()
    return self.att

  def _make_rope(self, length):
    """Make RoPE cos/sin for testing."""
    d = config.head_dim
    channel_range = torch.arange(0, d, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (channel_range / d))
    t = torch.arange(length, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin

  def test_output_shape(self):
    """Attention output matches input shape."""
    B, T = 2, 32
    x = torch.randn(B, T, config.n_embd)
    cos, sin = self._make_rope(T)
    out = self.att(x, cos, sin)
    assert out.shape == (B, T, config.n_embd)

  def test_gqa_ratio(self):
    """GQA: 9 query heads, 3 KV heads (3:1 ratio)."""
    assert self.att.c_q.out_features == config.n_head * config.head_dim  # 9*64=576
    assert self.att.c_k.out_features == config.n_kv_head * config.head_dim  # 3*64=192
    assert self.att.c_v.out_features == config.n_kv_head * config.head_dim  # 3*64=192

  def test_gate_init_halves_output(self):
    """Zero-init gate produces sigmoid(0)=0.5, halving SDPA output."""
    # Gate is only zero-init'ed in Model.__init__(), so do it manually here
    torch.nn.init.zeros_(self.att.w_gate.weight)
    x = torch.randn(1, 4, config.n_embd)
    gate = self.att.w_gate(x)
    gate_sigmoid = torch.sigmoid(gate)
    assert torch.allclose(gate_sigmoid, torch.full_like(gate_sigmoid, 0.5), atol=1e-6)

  def test_with_dense_mask(self):
    """Attention works with dense float mask (SDPA path)."""
    B, T = 1, 32
    x = torch.randn(B, T, config.n_embd)
    cos, sin = self._make_rope(T)
    mask = torch.zeros(T, T)
    out = self.att(x, cos, sin, attn_mask=mask)
    assert out.shape == (B, T, config.n_embd)
    assert not torch.isnan(out).any()

  def test_without_mask(self):
    """Attention works without mask (is_causal=False path)."""
    B, T = 1, 16
    x = torch.randn(B, T, config.n_embd)
    cos, sin = self._make_rope(T)
    out = self.att(x, cos, sin, attn_mask=None)
    assert out.shape == (B, T, config.n_embd)

  def test_with_positions(self):
    """Attention works with explicit per-token positions (doc packing path)."""
    B, T = 2, 16
    x = torch.randn(B, T, config.n_embd)
    cos, sin = self._make_rope(64)  # precomputed for max positions
    positions = torch.arange(T).unsqueeze(0).expand(B, -1)
    out = self.att(x, cos, sin, positions=positions)
    assert out.shape == (B, T, config.n_embd)

  def test_training_rope_split(self):
    """Training [x_t || x_0]: cos has L positions but T=2L tokens."""
    B, L = 1, 32
    T = 2 * L
    x = torch.randn(B, T, config.n_embd)
    cos, sin = self._make_rope(L)  # only L positions
    out = self.att(x, cos, sin)
    assert out.shape == (B, T, config.n_embd)

  def test_kv_cache_grows(self):
    """KV cache accumulates across calls."""
    self.att.cache_mode = True
    x1 = torch.randn(1, 16, config.n_embd)
    x2 = torch.randn(1, 16, config.n_embd)
    cos, sin = self._make_rope(32)

    self.att(x1, cos[:, :16], sin[:, :16])
    assert self.att.kv_cache is not None
    k1, v1 = self.att.kv_cache
    assert k1.shape[2] == 16

    self.att(x2, cos[:, :16], sin[:, :16])
    k2, v2 = self.att.kv_cache
    assert k2.shape[2] == 32  # accumulated
    self.att.reset_cache()

  def test_gradients_flow_through_gate(self):
    """Gradients flow through the gated query attention path."""
    self.att.train()
    x = torch.randn(1, 16, config.n_embd, requires_grad=True)
    cos, sin = self._make_rope(16)
    out = self.att(x, cos, sin)
    out.sum().backward()
    assert x.grad is not None
    assert self.att.w_gate.weight.grad is not None


# ============================================================================
# Model Cache Mode Tests
# ============================================================================


class TestModelCacheModes:
  @pytest.fixture(autouse=True)
  def model(self):
    torch.manual_seed(42)
    self.m = Model()
    self.m.eval()
    return self.m

  def test_set_cache_mode(self):
    """set_cache_mode toggles caching without clearing existing cache."""
    self.m.enable_kv_cache()
    x = torch.randint(0, 49152, (1, 16))
    self.m(x)  # fills cache
    # Toggle off: cache should persist
    self.m.set_cache_mode(False)
    for block in self.m.blocks:
      assert block.attn.cache_mode is False
      assert block.attn.kv_cache is not None  # not cleared
    self.m.reset_kv_cache()

  def test_disable_kv_cache(self):
    """disable_kv_cache clears both mode and cached data."""
    self.m.enable_kv_cache()
    x = torch.randint(0, 49152, (1, 16))
    self.m(x)
    self.m.disable_kv_cache()
    for block in self.m.blocks:
      assert block.attn.cache_mode is False
      assert block.attn.kv_cache is None  # cleared


# ============================================================================
# Rotary Embedding Wrapper Tests
# ============================================================================


class TestApplyRotaryEmbPair:
  def test_apply_rotary_emb_pair(self):
    """apply_rotary_emb applies RoPE to both Q and K."""
    from phase5.attention import apply_rotary_emb

    q = torch.randn(1, 4, 9, 64)
    k = torch.randn(1, 4, 3, 64)
    cos = torch.ones(1, 4, 1, 32)
    sin = torch.zeros(1, 4, 1, 32)
    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    # With cos=1, sin=0: identity
    assert torch.allclose(q_out, q, atol=1e-6)
    assert torch.allclose(k_out, k, atol=1e-6)

  def test_apply_rotary_emb_equivariance(self):
    """RoPE preserves inner product when Q and K share the same rotation angle."""
    from phase5.attention import apply_rotary_emb

    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    # Use proper rotation angles (cos^2 + sin^2 = 1)
    theta = torch.tensor([0.5, 1.0, 1.5, 2.0]).view(1, 1, 1, 4)
    cos = theta.cos()
    sin = theta.sin()
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    # RoPE is a rotation: Q^T K is preserved when Q and K rotate by the same angle
    # Compute proper inner product: sum over head_dim
    dot_orig = (q.view(-1) * k.view(-1)).sum()
    dot_rot = (q_rot.view(-1) * k_rot.view(-1)).sum()
    assert torch.allclose(dot_orig, dot_rot, atol=1e-5), (
      f'RoPE broke inner product: {dot_orig.item():.4f} vs {dot_rot.item():.4f}'
    )


# ============================================================================
# MaxLogitsTracker Tests
# ============================================================================


class TestMaxLogitsTracker:
  def test_update_and_consume(self):
    _MaxLogitsTracker._tls.max_logits = None  # reset
    _MaxLogitsTracker._update(5.0)
    _MaxLogitsTracker._update(10.0)
    _MaxLogitsTracker._update(3.0)
    val = _MaxLogitsTracker.consume()
    assert val == 10.0

  def test_consume_resets(self):
    _MaxLogitsTracker._tls.max_logits = None
    _MaxLogitsTracker._update(7.0)
    _MaxLogitsTracker.consume()
    assert _MaxLogitsTracker.consume() is None

  def test_update_with_tensors(self):
    _MaxLogitsTracker._tls.max_logits = None
    _MaxLogitsTracker._update(torch.tensor(5.0))
    _MaxLogitsTracker._update(torch.tensor(12.0))
    val = _MaxLogitsTracker.consume()
    assert val == pytest.approx(12.0)

  def test_mixed_tensor_and_float(self):
    _MaxLogitsTracker._tls.max_logits = None
    _MaxLogitsTracker._update(torch.tensor(5.0))
    _MaxLogitsTracker._update(20.0)
    val = _MaxLogitsTracker.consume()
    assert val == pytest.approx(20.0)


# ============================================================================
# MuonClip AdamW Fallback Tests
# ============================================================================


class TestMuonClipAdamW:
  def test_adamw_step_updates_1d_params(self):
    """AdamW path handles 1D params (norms, biases)."""
    torch.manual_seed(42)
    m = Model()
    opt = build_param_groups(m)
    # Only norm weights are 1D in this model
    norm_weight = m.blocks[0].attn_norm.weight.clone()
    for p in m.parameters():
      if p.requires_grad:
        p.grad = torch.randn_like(p) * 0.01
    opt.step()
    # Norm weight should have changed (handled by AdamW group)
    assert not torch.allclose(norm_weight, m.blocks[0].attn_norm.weight)

  def test_adamw_bias_correction(self):
    """AdamW step uses bias-corrected moments."""
    p = torch.nn.Parameter(torch.randn(10))
    p.grad = torch.ones(10)
    opt = MuonClip([dict(params=[p], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, apply_muon=False)])
    for pg in opt.param_groups:
      pg['initial_lr'] = pg['lr']

    p_before = p.data.clone()
    opt.step()
    p_after = p.data.clone()
    # After one step with grad=1, p should decrease
    assert (p_after < p_before).all()
    # Check state was initialized
    assert opt.state[p]['step'] == 1
    assert 'exp_avg' in opt.state[p]
    assert 'exp_avg_sq' in opt.state[p]

  def test_weight_decay_applied(self):
    """Both Muon and AdamW paths apply decoupled weight decay."""
    p = torch.nn.Parameter(torch.ones(16, 16) * 2.0)
    p.grad = torch.zeros(16, 16)  # zero grad: only WD acts
    opt = MuonClip([dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.5, apply_muon=True)])
    for pg in opt.param_groups:
      pg['initial_lr'] = pg['lr']
    opt.step()
    # With zero grad and WD=0.5, p should shrink: p *= (1 - 0.01*0.5) = 0.995
    expected = 2.0 * (1 - 0.01 * 0.5)
    assert torch.allclose(p.data, torch.full_like(p.data, expected), atol=1e-4)


# ============================================================================
# FP8 Forward Numerics Tests
# ============================================================================


class TestFP8Numerics:
  def test_to_fp8_scale_and_clamp(self):
    """_to_fp8 scales tensor to fit FP8 range."""
    from phase5.fp8 import _to_fp8

    x = torch.randn(64, 64)
    x_fp8, inv_scale = _to_fp8(x, torch.float8_e4m3fn)
    # FP8 values should be within range (cast to float for .abs().max())
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    assert x_fp8.float().abs().max() <= fp8_max
    # Inverse scale should approximately recover original magnitude
    # E4M3 has ~3 bits mantissa → relative error up to ~12% for small values
    recovered = x_fp8.float() * inv_scale
    rel_error = (recovered - x).abs() / x.abs().clamp(min=0.01)
    assert rel_error.median() < 0.1, f'Median relative error {rel_error.median():.3f} too high'

  def test_to_fp8_preserves_sign(self):
    from phase5.fp8 import _to_fp8

    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, -3.0])
    x_fp8, inv_scale = _to_fp8(x, torch.float8_e4m3fn)
    recovered = x_fp8.float() * inv_scale
    # Signs should match
    for i in range(len(x)):
      if x[i] != 0:
        assert recovered[i].sign() == x[i].sign()

  @pytest.mark.skipif(not torch.cuda.is_available(), reason='FP8 matmul requires CUDA')
  def test_float8_linear_forward_numerics(self):
    """Float8Linear.forward produces results close to nn.Linear."""
    from phase5.fp8 import Float8Linear

    torch.manual_seed(42)
    linear = torch.nn.Linear(64, 128, bias=False).cuda().bfloat16()
    fp8 = Float8Linear.from_float(linear).cuda()
    x = torch.randn(2, 16, 64, device='cuda', dtype=torch.bfloat16)

    ref = F.linear(x, linear.weight)
    fp8_out = fp8(x)

    assert fp8_out.shape == ref.shape
    # FP8 should be close but not exact
    rel_error = (fp8_out - ref).abs().max() / ref.abs().max()
    assert rel_error < 0.05, f'FP8 relative error {rel_error:.4f} too high'

  @pytest.mark.skipif(not torch.cuda.is_available(), reason='FP8 backward requires CUDA')
  def test_float8_linear_backward(self):
    """Float8Linear backward produces non-zero gradients."""
    from phase5.fp8 import Float8Linear

    torch.manual_seed(42)
    linear = torch.nn.Linear(64, 128, bias=False).cuda().bfloat16()
    fp8 = Float8Linear.from_float(linear).cuda()
    x = torch.randn(2, 16, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    out = fp8(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0
    assert fp8.weight.grad is not None
    assert fp8.weight.grad.abs().max() > 0


# ============================================================================
# PreTokenizedPacker Tests (with mock dataset)
# ============================================================================


class TestPreTokenizedPacker:
  def test_epoch_style_no_replacement(self):
    """_PreTokenizedPacker iterates without replacement within an epoch."""
    import numpy as np

    n_docs = 10
    accessed_indices = []

    class MockDataset:
      def __len__(self):
        return n_docs

      def __getitem__(self, indices):
        accessed_indices.extend(indices)
        return {'input_ids': [list(range(14, 114)) for _ in indices]}

    from phase5.data import _PreTokenizedPacker

    packer = _PreTokenizedPacker.__new__(_PreTokenizedPacker)
    packer._ds = MockDataset()
    packer._n = n_docs
    packer._rng = np.random.RandomState(42)
    packer._eos_id = config.eos_token_id
    packer._buf = []
    packer._order = packer._rng.permutation(n_docs)
    packer._cursor = 0

    # Pull enough to consume all 10 docs
    for _ in range(20):
      packer._refill()

    # First 10 indices should be a permutation of 0..9 (no replacement)
    first_epoch = accessed_indices[:n_docs]
    assert sorted(first_epoch) == list(range(n_docs)), f'First epoch not a permutation: {first_epoch}'

  def test_epoch_boundary_reshuffles(self):
    """Cursor resets and order reshuffles at epoch boundary."""
    import numpy as np
    from phase5.data import _PreTokenizedPacker

    # Use larger n to make permutation collision astronomically unlikely
    n_docs = 50
    packer = _PreTokenizedPacker.__new__(_PreTokenizedPacker)
    packer._n = n_docs
    packer._rng = np.random.RandomState(42)
    packer._order = packer._rng.permutation(n_docs)
    packer._cursor = n_docs  # at epoch boundary

    order_before = packer._order.copy()
    packer._ds = type('', (), {'__getitem__': lambda self, idx: {'input_ids': [[14, 15, 16] for _ in idx]}})()
    packer._eos_id = 1
    packer._buf = []
    packer._refill()

    # Cursor should have reset and consumed a batch
    assert packer._cursor > 0
    assert packer._cursor <= n_docs
    # Order should have been reshuffled (different permutation)
    assert not np.array_equal(packer._order, order_before), 'Order should change after epoch boundary reshuffle'


# ============================================================================
# Model Forward Path Tests
# ============================================================================


class TestModelForwardPaths:
  @pytest.fixture(autouse=True)
  def model(self):
    torch.manual_seed(42)
    self.m = Model()
    self.m.eval()
    return self.m

  def test_inference_with_pos_offset(self):
    """Inference with pos_offset shifts RoPE positions."""
    x = torch.randint(0, 49152, (1, 16))
    logits_0, _ = self.m(x, pos_offset=0)
    logits_32, _ = self.m(x, pos_offset=32)
    # Different pos_offset should give different logits
    assert not torch.allclose(logits_0, logits_32)

  def test_training_extracts_x_t_half(self):
    """Training forward returns only x_t half (first L of 2L)."""
    B, L = 2, 64
    x_input = torch.randint(0, 49152, (B, 2 * L))
    targets = torch.randint(14, 49152, (B, L))
    attn_mask = build_staircase_mask(L, config.block_size)
    hidden, _ = self.m(x_input, targets=targets, attn_mask=attn_mask)
    assert hidden.shape == (B, L, config.n_embd)

  def test_training_with_positions(self):
    """Training with explicit positions uses per-token RoPE."""
    B, L = 2, 64
    x_input = torch.randint(0, 49152, (B, 2 * L))
    targets = torch.randint(14, 49152, (B, L))
    positions = torch.arange(L).unsqueeze(0).expand(B, -1)
    attn_mask = build_staircase_mask(L, config.block_size)
    hidden, _ = self.m(x_input, targets=targets, attn_mask=attn_mask, positions=positions)
    assert hidden.shape == (B, L, config.n_embd)

  def test_rope_positions_affect_output(self):
    """Different document positions produce different outputs."""
    B, L = 1, 64
    x_input = torch.randint(0, 49152, (B, 2 * L))
    targets = torch.randint(14, 49152, (B, L))
    attn_mask = build_staircase_mask(L, config.block_size)

    pos_sequential = torch.arange(L).unsqueeze(0)
    pos_reset = torch.cat([torch.arange(32), torch.arange(32)]).unsqueeze(0)

    h1, _ = self.m(x_input, targets=targets, attn_mask=attn_mask, positions=pos_sequential)
    h2, _ = self.m(x_input, targets=targets, attn_mask=attn_mask, positions=pos_reset)
    # Different positions produce different hidden states
    assert not torch.allclose(h1, h2)


# ============================================================================
# Chunked Loss Tests
# ============================================================================


class TestChunkedLoss:
  def test_chunked_matches_naive(self):
    """Chunked CE with grad_checkpoint matches naive full-logits CE."""
    torch.manual_seed(42)
    B, L, D = 1, 32, 576
    hidden = torch.randn(B, L, D)
    targets = torch.randint(14, 49152, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    lm_head_weight = torch.randn(49152, D)

    # Chunked loss (our implementation)
    loss_chunked = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight)

    # Naive loss (full logits materialized)
    logits = hidden.view(-1, D) @ lm_head_weight.T
    ce = F.cross_entropy(logits, targets.view(-1), reduction='none')
    w = (mask.float() * elbo_w).view(-1)
    real_count = (targets != config.pad_token_id).float().sum().clamp(min=1)
    loss_naive = (ce * w).sum() / real_count

    assert torch.allclose(loss_chunked, loss_naive, atol=1e-4), (
      f'Chunked {loss_chunked.item():.6f} vs Naive {loss_naive.item():.6f}'
    )

  def test_loss_gradient_nonzero(self):
    """Loss backward through chunked CE produces non-zero gradients."""
    B, L, D = 1, 32, 576
    hidden = torch.randn(B, L, D, requires_grad=True)
    targets = torch.randint(14, 49152, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    lm_head_weight = torch.randn(49152, D, requires_grad=True)

    loss = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight)
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().max() > 0
    assert lm_head_weight.grad is not None
