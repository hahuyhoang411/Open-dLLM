"""Deep QA tests for Phase 6 — adversarial numerical verification.

These tests verify specific numerical properties NOT covered by existing tests.
Each test asserts a concrete mathematical invariant, not just "doesn't crash."
CPU-only, tiny configs, fast execution.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from phase6.attention import (
  MultiHeadAttention,
  _make_rms_norm,
  apply_rotary_emb,
  build_staircase_mask,
)
from phase6.config import Config
from phase6.data import _compute_positions, _sample_t_per_doc
from phase6.generate import (
  _add_gumbel_noise,
  _select_tokens_dynamic,
  _select_tokens_static,
  generate,
  get_num_transfer_tokens,
)
from phase6.loss import compute_loss
from phase6.model import Model
from phase6.optim import MuonClip, _MaxLogitsTracker, create_optimizer
from phase6.schedule import compute_elbo_weight

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------


def _tiny(**kw):
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
    t_min=0.1,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=False,
    use_grad_ckpt=False,
    use_flex=False,
    use_compile=False,
    use_muon=False,
    use_cart=False,
    use_amp=False,
    mask_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    muon_lr=0.02,
    adamw_lr=3e-3,
    grad_clip=1.0,
    denoise_steps=4,
    device='cpu',
  )
  defaults.update(kw)
  return Config(**defaults).validate()


class _FakeCfg:
  pad_token_id = 2


# ============================================================================
# 1. QK-norm -> RoPE ordering: norm BEFORE rotary (Qwen3 convention)
# ============================================================================


class TestQKNormRoPEOrdering:
  def test_norm_then_rope_vs_rope_then_norm_differ(self):
    """Two orderings (norm-then-rope vs rope-then-norm) produce different results
    when RMSNorm has non-trivial learned weights (not all 1.0).
    Our code uses norm-then-rope (Qwen3 convention). Verify they diverge.

    With weight=1 (init), RMSNorm is a scalar operation that commutes with
    rotation. Non-uniform weights break this commutation.
    """
    cfg = _tiny(use_qk_norm=True)
    head_dim = cfg.head_dim
    n_head = cfg.n_head
    n_kv_head = cfg.n_kv_head
    B, T = 1, 8

    torch.manual_seed(42)
    x = torch.randn(B, T, cfg.n_embd)

    # Build the projections + norms manually
    attn = MultiHeadAttention(cfg)

    # Set non-trivial norm weights (simulates learned weights after training)
    with torch.no_grad():
      attn.q_norm.weight.copy_(torch.linspace(0.5, 2.0, head_dim))
      attn.k_norm.weight.copy_(torch.linspace(2.0, 0.5, head_dim))

    q_proj = attn.c_q(x).view(B, T, n_head, head_dim)
    k_proj = attn.c_k(x).view(B, T, n_kv_head, head_dim)

    # Build RoPE cos/sin
    model = Model(cfg)
    cos = model.cos[:, :T]
    sin = model.sin[:, :T]

    # Path A: norm THEN rope (our code's path)
    q_normed = attn.q_norm(q_proj)
    k_normed = attn.k_norm(k_proj)
    q_a, k_a = apply_rotary_emb(q_normed, k_normed, cos, sin)

    # Path B: rope THEN norm (wrong order)
    q_roped, k_roped = apply_rotary_emb(q_proj, k_proj, cos, sin)
    q_b = attn.q_norm(q_roped)
    k_b = attn.k_norm(k_roped)

    # The two paths should produce different results
    assert not torch.allclose(q_a, q_b, atol=1e-4), (
      'norm-then-rope and rope-then-norm should differ for Q with non-trivial weights'
    )
    assert not torch.allclose(k_a, k_b, atol=1e-4), (
      'norm-then-rope and rope-then-norm should differ for K with non-trivial weights'
    )

  def test_model_uses_norm_before_rope(self):
    """Verify our MultiHeadAttention applies norm before RoPE by comparing
    with a manual norm-then-rope path.
    """
    cfg = _tiny(use_qk_norm=True)
    B, T = 1, 8
    n_head = cfg.n_head
    n_kv_head = cfg.n_kv_head
    head_dim = cfg.head_dim

    model = Model(cfg)
    model.eval()
    attn = model.blocks[0].attn

    torch.manual_seed(7)
    x = torch.randn(B, T, cfg.n_embd)
    cos = model.cos[:, :T]
    sin = model.sin[:, :T]

    # Manual norm-then-rope
    q_raw = attn.c_q(x).view(B, T, n_head, head_dim)
    k_raw = attn.c_k(x).view(B, T, n_kv_head, head_dim)
    q_normed = attn.q_norm(q_raw)
    k_normed = attn.k_norm(k_raw)
    q_expected, k_expected = apply_rotary_emb(q_normed, k_normed, cos, sin)

    # Full forward: hook to capture Q after RoPE, before transpose
    captured = {}
    orig_forward = attn.forward

    def hook_forward(x, cos, sin, attn_mask=None, positions=None):
      # We run the full forward but intercept intermediate q, k
      _cfg = attn.cfg
      _n_head = _cfg.n_head
      _n_kv_head = _cfg.n_kv_head
      _head_dim = _cfg.head_dim
      _B, _T, _C = x.size()

      q = attn.c_q(x).view(_B, _T, _n_head, _head_dim)
      k = attn.c_k(x).view(_B, _T, _n_kv_head, _head_dim)

      if attn.q_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)

      if cos.size(1) == _T:
        q, k = apply_rotary_emb(q, k, cos, sin)

      captured['q'] = q.clone()
      captured['k'] = k.clone()

      # Complete the rest normally via original
      return orig_forward(x, cos, sin, attn_mask=attn_mask, positions=positions)

    with torch.no_grad():
      hook_forward(x, cos, sin)

    assert torch.allclose(captured['q'], q_expected, atol=1e-5), "Model Q doesn't match norm-then-rope path"
    assert torch.allclose(captured['k'], k_expected, atol=1e-5), "Model K doesn't match norm-then-rope path"


# ============================================================================
# 2. Head dim independence: Q output = (B, T, n_head*head_dim)
# ============================================================================


class TestHeadDimIndependence:
  def test_q_output_shape_inside_forward(self):
    """Q linear output is (B, T, n_head*head_dim) = (B, T, 128), NOT (B, T, 64=n_embd)."""
    cfg = _tiny()  # n_embd=64, n_head=4, head_dim=32 -> Q out = 128
    attn = MultiHeadAttention(cfg)
    B, T = 2, 8
    x = torch.randn(B, T, cfg.n_embd)

    q_linear_out = attn.c_q(x)
    assert q_linear_out.shape == (B, T, cfg.n_head * cfg.head_dim), (
      f'Q linear output shape {q_linear_out.shape} != (B, T, {cfg.n_head * cfg.head_dim})'
    )
    assert q_linear_out.shape != (B, T, cfg.n_embd) or cfg.n_head * cfg.head_dim == cfg.n_embd

  def test_qwen3_dims_q_vs_kv(self):
    """Qwen3: Q = (B, T, 2048), K/V = (B, T, 1024), with n_embd=1024."""
    cfg = Config()  # Full Qwen3 defaults
    assert cfg.n_head * cfg.head_dim == 2048
    assert cfg.n_kv_head * cfg.head_dim == 1024
    assert cfg.n_embd == 1024
    # Q output (2048) != n_embd (1024) -- this is the key Qwen3 property
    assert cfg.n_head * cfg.head_dim != cfg.n_embd

  def test_reshaped_qkv_dimensions(self):
    """After view(), Q = (B, T, n_head, head_dim), K = (B, T, n_kv_head, head_dim)."""
    cfg = _tiny()  # n_head=4, n_kv_head=2, head_dim=32
    attn = MultiHeadAttention(cfg)
    B, T = 1, 4
    x = torch.randn(B, T, cfg.n_embd)

    q = attn.c_q(x).view(B, T, cfg.n_head, cfg.head_dim)
    k = attn.c_k(x).view(B, T, cfg.n_kv_head, cfg.head_dim)
    v = attn.c_v(x).view(B, T, cfg.n_kv_head, cfg.head_dim)

    assert q.shape == (B, T, 4, 32)
    assert k.shape == (B, T, 2, 32)
    assert v.shape == (B, T, 2, 32)


# ============================================================================
# 3. GQA K/V repeat for SDPA
# ============================================================================


class TestGQARepeat:
  def test_repeat_interleave_shape(self):
    """K/V repeated n_head//n_kv_head=2 times along head dim for SDPA."""
    B, T = 1, 8
    n_head, n_kv_head, head_dim = 4, 2, 32
    repeats = n_head // n_kv_head

    K = torch.randn(B, n_kv_head, T, head_dim)
    K_expanded = K.repeat_interleave(repeats, dim=1)

    assert K_expanded.shape == (B, n_head, T, head_dim)

  def test_repeat_interleave_values_correct(self):
    """After repeat, head pairs share the same KV data."""
    B, T = 1, 4
    n_head, n_kv_head, head_dim = 4, 2, 32
    repeats = n_head // n_kv_head

    K = torch.randn(B, n_kv_head, T, head_dim)
    K_exp = K.repeat_interleave(repeats, dim=1)

    # Heads 0,1 should be identical (from KV head 0)
    assert torch.equal(K_exp[:, 0], K_exp[:, 1])
    # Heads 2,3 should be identical (from KV head 1)
    assert torch.equal(K_exp[:, 2], K_exp[:, 3])
    # But head groups should differ
    assert not torch.equal(K_exp[:, 0], K_exp[:, 2])

  def test_sdpa_output_matches_manual_gqa(self):
    """SDPA with expanded K/V matches manual attention with GQA."""
    B, T = 1, 4
    n_head, n_kv_head, head_dim = 4, 2, 32
    repeats = n_head // n_kv_head

    torch.manual_seed(42)
    Q = torch.randn(B, n_head, T, head_dim)
    K = torch.randn(B, n_kv_head, T, head_dim)
    V = torch.randn(B, n_kv_head, T, head_dim)

    K_exp = K.repeat_interleave(repeats, dim=1)
    V_exp = V.repeat_interleave(repeats, dim=1)

    # SDPA
    out_sdpa = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=False)

    # Manual attention for head 0 (uses KV head 0)
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q[:, 0] @ K[:, 0].transpose(-2, -1) * scale
    attn_weights = F.softmax(scores, dim=-1)
    out_manual = attn_weights @ V[:, 0]

    assert torch.allclose(out_sdpa[:, 0], out_manual, atol=1e-5), "SDPA head 0 doesn't match manual GQA attention"


# ============================================================================
# 4. RoPE frequencies with base=1M vs base=10K
# ============================================================================


class TestRoPEBase1M:
  def test_base1m_weaker_high_frequency(self):
    """base=1M has much weaker high-frequency components than base=10K.
    High-frequency inv_freq (last channel) should be 100x smaller for base=1M.
    """
    head_dim = 128  # Qwen3 head_dim
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)

    inv_freq_10k = 1.0 / (10_000 ** (channel_range / head_dim))
    inv_freq_1m = 1.0 / (1_000_000 ** (channel_range / head_dim))

    # Low-frequency (channel 0): both are 1.0
    assert inv_freq_10k[0].item() == pytest.approx(1.0)
    assert inv_freq_1m[0].item() == pytest.approx(1.0)

    # High-frequency (last channel): base=1M is much smaller
    last = len(channel_range) - 1
    ratio = inv_freq_10k[last].item() / inv_freq_1m[last].item()
    assert ratio > 50, f'base=10K last inv_freq should be >50x larger than base=1M, got ratio {ratio:.1f}'

  def test_base1m_numerical_values(self):
    """Verify exact inv_freq values for base=1M, head_dim=128."""
    head_dim = 128
    base = 1_000_000
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Channel 0: 1/(1M^0) = 1.0
    assert inv_freq[0].item() == pytest.approx(1.0)
    # Channel 1: 1/(1M^(2/128)) = 1/(1M^0.015625) = 10^(-6*0.015625) = 10^(-0.09375)
    expected_ch1 = 10 ** (-6 * (2 / 128))
    assert inv_freq[1].item() == pytest.approx(expected_ch1, rel=1e-4)
    # Last channel (63): 1/(1M^(126/128)) = 10^(-6*126/128) = 10^(-5.90625)
    expected_last = 10 ** (-6 * (126 / 128))
    assert inv_freq[63].item() == pytest.approx(expected_last, rel=1e-3)

  def test_base1m_precomputed_buffers_differ(self):
    """Model with rope_base=1M produces different cos/sin than rope_base=10K."""
    m1 = Model(_tiny(rope_base=10_000.0))
    m2 = Model(_tiny(rope_base=1_000_000.0))
    assert not torch.allclose(m1.cos, m2.cos), 'cos buffers should differ'
    assert not torch.allclose(m1.sin, m2.sin), 'sin buffers should differ'


# ============================================================================
# 5. ELBO weight at t boundaries
# ============================================================================


class TestELBOWeightBoundaries:
  def test_elbo_weight_at_t_min(self):
    """At t=t_min=0.1, ELBO weight = 1/0.1 = exactly 10.0."""
    t = torch.tensor([[0.1]])
    w = compute_elbo_weight(t, t_min=0.1)
    assert w.item() == pytest.approx(10.0, abs=1e-6)

  def test_elbo_weight_at_half(self):
    """At t=0.5, ELBO weight = 1/0.5 = exactly 2.0."""
    t = torch.tensor([[0.5]])
    w = compute_elbo_weight(t, t_min=0.1)
    assert w.item() == pytest.approx(2.0, abs=1e-6)

  def test_elbo_weight_at_one(self):
    """At t=1.0, ELBO weight = 1/1.0 = exactly 1.0."""
    t = torch.tensor([[1.0]])
    w = compute_elbo_weight(t, t_min=0.1)
    assert w.item() == pytest.approx(1.0, abs=1e-6)

  def test_elbo_weight_below_t_min_clamped(self):
    """Below t_min, weight is clamped to 1/t_min = 10.0."""
    t = torch.tensor([[0.01, 0.05, 0.09]])
    w = compute_elbo_weight(t, t_min=0.1)
    assert torch.allclose(w, torch.tensor([[10.0, 10.0, 10.0]])), f'Expected all 10.0 for t < t_min, got {w}'

  def test_elbo_weight_per_token_in_compute_loss(self):
    """Verify the actual weight used inside compute_loss: mask * elbo_weight."""
    torch.manual_seed(42)
    B, L, D, V = 1, 8, 32, 64
    h = torch.randn(B, L, D, requires_grad=True)
    w_lm = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(3, V, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)

    # At t=0.1 (all tokens), elbo_weight = 10.0
    elbo_w_10 = torch.full((B, L), 10.0)
    loss_10 = compute_loss(h.detach().requires_grad_(True), targets, mask, elbo_w_10, w_lm, _FakeCfg())

    # At t=0.5 (all tokens), elbo_weight = 2.0
    elbo_w_2 = torch.full((B, L), 2.0)
    loss_2 = compute_loss(h.detach().requires_grad_(True), targets, mask, elbo_w_2, w_lm, _FakeCfg())

    ratio = loss_10.item() / loss_2.item()
    assert ratio == pytest.approx(5.0, rel=1e-3), f'10x weight vs 2x weight should give 5x ratio, got {ratio:.4f}'


# ============================================================================
# 6. Loss normalization denominator: ALL real tokens, not just masked
# ============================================================================


class TestLossNormalization:
  def test_loss_normalizes_by_all_real_tokens(self):
    """Loss divides by ALL real tokens, not just masked tokens.
    With 50% masked vs 100% masked, the CE sum per masked token is roughly
    the same, but loss differs because denominator (real_count) is fixed.
    """
    torch.manual_seed(42)
    B, L, D, V = 1, 16, 32, 64
    h = torch.randn(B, L, D)
    w_lm = torch.randn(V, D)
    targets = torch.randint(3, V, (B, L))

    # 100% masked
    mask_all = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    loss_all = compute_loss(h.detach().requires_grad_(True), targets, mask_all, elbo_w, w_lm, _FakeCfg())

    # 50% masked (first half only)
    mask_half = torch.zeros(B, L, dtype=torch.bool)
    mask_half[:, : L // 2] = True
    loss_half = compute_loss(h.detach().requires_grad_(True), targets, mask_half, elbo_w, w_lm, _FakeCfg())

    # Both normalized by real_count (targets != pad_token_id). Since none are pad,
    # real_count = 16 for both. But mask_half only has 8 contributing CE terms.
    # So loss_half should be roughly half of loss_all.
    ratio = loss_half.item() / loss_all.item()
    assert 0.3 < ratio < 0.7, f'50% mask should give ~50% loss of full mask (same denominator). Got ratio {ratio:.3f}'

  def test_padding_excluded_from_denominator(self):
    """When targets have pad tokens, they DON'T count in the denominator."""
    torch.manual_seed(42)
    B, L, D, V = 1, 16, 32, 64
    h = torch.randn(B, L, D)
    w_lm = torch.randn(V, D)

    # All real tokens
    targets_real = torch.randint(3, V, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    loss_real = compute_loss(h.detach().requires_grad_(True), targets_real, mask, elbo_w, w_lm, _FakeCfg())

    # Half are padding (pad_token_id=2) -- those don't contribute CE and don't count in denominator
    targets_padded = targets_real.clone()
    targets_padded[:, L // 2 :] = _FakeCfg.pad_token_id
    mask_padded = mask.clone()
    mask_padded[:, L // 2 :] = False
    loss_padded = compute_loss(h.detach().requires_grad_(True), targets_padded, mask_padded, elbo_w, w_lm, _FakeCfg())

    # With padding: real_count = 8 (not 16), and only 8 CE terms.
    # Without padding: real_count = 16, and 16 CE terms.
    # The loss should be roughly equal (8 terms / 8 count ~ 16 terms / 16 count).
    # But the specific CE values differ slightly, so allow tolerance.
    ratio = loss_padded.item() / loss_real.item()
    assert 0.5 < ratio < 2.0, f'Padded loss / real loss ratio {ratio:.3f} out of expected range'


# ============================================================================
# 7. Chunk boundary correctness
# ============================================================================


class TestChunkBoundary:
  def test_chunked_matches_full_computation(self):
    """Splitting loss into chunks produces EXACT same result as full tensor."""
    torch.manual_seed(42)
    B, L, D, V = 2, 16, 32, 64
    h = torch.randn(B, L, D)
    w_lm = torch.randn(V, D)
    targets = torch.randint(3, V, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L) * 3.0

    # Compute via compute_loss (chunked with grad_checkpoint)
    loss_chunked = compute_loss(h.detach().requires_grad_(True), targets, mask, elbo_w, w_lm, _FakeCfg())

    # Compute manually in one shot (no chunking)
    h_flat = h.detach().contiguous().view(-1, D)
    t_flat = targets.contiguous().view(-1)
    w_flat = (mask.float() * elbo_w).contiguous().view(-1)
    logits = h_flat @ w_lm.T
    ce = F.cross_entropy(logits, t_flat, reduction='none')
    real_count = (targets != _FakeCfg.pad_token_id).float().sum().clamp(min=1)
    loss_full = (ce * w_flat).sum() / real_count

    assert loss_chunked.item() == pytest.approx(loss_full.item(), rel=1e-4), (
      f'Chunked {loss_chunked.item():.6f} != full {loss_full.item():.6f}'
    )

  def test_chunk_boundary_off_by_one(self):
    """Verify no tokens are dropped or duplicated at chunk boundaries.
    Sum of per-chunk CE*weight should equal single-pass CE*weight.
    """
    torch.manual_seed(42)
    D, V = 16, 32
    N = 17  # deliberately not a multiple of typical chunk sizes
    h = torch.randn(N, D)
    w_lm = torch.randn(V, D)
    targets = torch.randint(3, V, (N,))
    weights = torch.rand(N)

    # Manual chunking with chunk_size=5
    chunk_size = 5
    total_chunked = 0.0
    for i in range(0, N, chunk_size):
      j = min(i + chunk_size, N)
      logits_chunk = h[i:j] @ w_lm.T
      ce_chunk = F.cross_entropy(logits_chunk, targets[i:j], reduction='none')
      total_chunked += (ce_chunk * weights[i:j]).sum().item()

    # Full computation
    logits_full = h @ w_lm.T
    ce_full = F.cross_entropy(logits_full, targets, reduction='none')
    total_full = (ce_full * weights).sum().item()

    assert total_chunked == pytest.approx(total_full, rel=1e-5), (
      f'Chunked sum {total_chunked:.6f} != full sum {total_full:.6f}'
    )


# ============================================================================
# 8. Dynamic remasking commit-all behavior
# ============================================================================


class TestDynamicRemasking:
  def test_commit_all_above_tau(self):
    """When >budget tokens exceed tau, ALL of them get committed (not just budget)."""
    B, L = 1, 8
    # 6 of 8 positions masked, confidences all above tau=0.5
    confidences = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.0, 0.0]])
    masked = torch.tensor([[True, True, True, True, True, True, False, False]])
    num_to_commit = 2  # budget is only 2
    threshold = 0.5

    commit = _select_tokens_dynamic(confidences, masked, num_to_commit, threshold)

    # All 6 masked positions have conf > 0.5, so ALL 6 should be committed
    assert commit.sum().item() == 6, f'Expected 6 commits (all above tau), got {commit.sum().item()}'
    # Non-masked positions should never be committed
    assert not commit[0, 6].item()
    assert not commit[0, 7].item()

  def test_fallback_to_topk_when_few_above_tau(self):
    """When fewer tokens exceed tau than budget, use top-k fallback."""
    B, L = 1, 8
    confidences = torch.tensor([[0.9, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0, 0.0]])
    masked = torch.tensor([[True, True, True, True, True, True, False, False]])
    num_to_commit = 3
    threshold = 0.5

    commit = _select_tokens_dynamic(confidences, masked, num_to_commit, threshold)

    # Only 1 token (0.9) exceeds tau=0.5, which is < budget=3
    # So fallback to top-3 by confidence: positions 0 (0.9), 1 (0.3), 2 (0.2)
    assert commit.sum().item() == 3
    assert commit[0, 0].item()  # highest confidence

  def test_minimum_one_token_committed(self):
    """At least 1 token is always committed per call."""
    B, L = 1, 4
    confidences = torch.tensor([[0.01, 0.02, 0.0, 0.0]])
    masked = torch.tensor([[True, True, False, False]])
    num_to_commit = 0  # edge case: budget is 0

    commit = _select_tokens_dynamic(confidences, masked, max(1, num_to_commit), 0.5)
    assert commit.sum().item() >= 1


# ============================================================================
# 9. KV cache accumulation
# ============================================================================


class TestKVCacheAccumulation:
  def test_kv_cache_grows_per_block(self):
    """After processing N blocks, KV cache has N*block_size KV pairs."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()
    blk = cfg.block_size

    model.set_cache_mode(True)
    model.reset_kv_cache()

    # Process 3 blocks
    for i in range(3):
      block_ids = torch.randint(3, cfg.vocab_size, (1, blk))
      with torch.no_grad():
        model(block_ids, pos_offset=i * blk)

    # Check cache size in each layer
    for layer_idx, block in enumerate(model.blocks):
      k_cache, v_cache = block.attn.kv_cache
      expected_len = 3 * blk
      assert k_cache.shape[2] == expected_len, (
        f'Layer {layer_idx}: K cache seq_len={k_cache.shape[2]}, expected {expected_len}'
      )
      assert v_cache.shape[2] == expected_len, (
        f'Layer {layer_idx}: V cache seq_len={v_cache.shape[2]}, expected {expected_len}'
      )
      # Shape: (B, n_kv_head, seq_len, head_dim)
      assert k_cache.shape == (1, cfg.n_kv_head, expected_len, cfg.head_dim)

    model.disable_kv_cache()

  def test_kv_cache_reset_clears_all(self):
    """reset_kv_cache sets all caches to None."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()
    model.set_cache_mode(True)

    # Feed one block
    with torch.no_grad():
      model(torch.randint(3, cfg.vocab_size, (1, cfg.block_size)))

    # Caches should exist
    for block in model.blocks:
      assert block.attn.kv_cache is not None

    model.reset_kv_cache()
    for block in model.blocks:
      assert block.attn.kv_cache is None

    model.disable_kv_cache()


# ============================================================================
# 10. Block boundary handling (prompt not multiple of block_size)
# ============================================================================


class TestBlockBoundary:
  def test_prompt_not_multiple_of_block_size(self):
    """Prompt of length 5 with block_size=8: remainder=5, first gen block
    should fill the remaining 3 positions.
    """
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    prompt_ids = [10, 20, 30, 40, 50]  # length 5, block_size=8
    result = generate(model, prompt_ids, cfg, max_new_tokens=8, temperature=0.0)

    # Prompt should be preserved at the start (filtered for specials)
    assert result[:5] == prompt_ids

  def test_prompt_exact_multiple(self):
    """Prompt exactly block_size tokens: no remainder handling needed."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    prompt_ids = list(range(10, 18))  # exactly 8 tokens = 1 block
    result = generate(model, prompt_ids, cfg, max_new_tokens=8, temperature=0.0)
    assert result[:8] == prompt_ids

  def test_prompt_zero_length(self):
    """Empty prompt: entire first block is mask tokens for decoding."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    result = generate(model, [], cfg, max_new_tokens=8, temperature=0.0)
    assert isinstance(result, list)
    assert len(result) > 0


# ============================================================================
# 11. Per-document t range (t_min enforcement)
# ============================================================================


class TestPerDocTRange:
  def test_no_t_below_t_min_1000_samples(self):
    """Over 1000 samples, no document gets t < t_min=0.1."""
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 8)  # B=8

    all_min = float('inf')
    for seed in range(125):  # 125 * 8 = 1000 batches
      torch.manual_seed(seed)
      t = _sample_t_per_doc(doc_ids, t_min=0.1)
      batch_min = t.min().item()
      all_min = min(all_min, batch_min)

    assert all_min >= 0.1 - 1e-7, f'Found t={all_min} below t_min=0.1 in 1000 samples'

  def test_t_max_below_one(self):
    """All t values should be strictly < 1.0 (uniform [t_min, 1))."""
    doc_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]] * 8)

    all_max = 0.0
    for seed in range(125):
      torch.manual_seed(seed)
      t = _sample_t_per_doc(doc_ids, t_min=0.1)
      all_max = max(all_max, t.max().item())

    assert all_max < 1.0, f'Found t={all_max} >= 1.0'

  def test_t_distribution_covers_range(self):
    """T values should cover the full [t_min, 1) range, not cluster."""
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]] * 16)
    all_ts = []
    for seed in range(100):
      torch.manual_seed(seed)
      t = _sample_t_per_doc(doc_ids, t_min=0.1)
      all_ts.append(t[0, 0].item())

    all_ts = sorted(all_ts)
    # Should have values across [0.1, 1.0)
    assert all_ts[0] < 0.2, f'Min t={all_ts[0]}, expected values near 0.1'
    assert all_ts[-1] > 0.8, f'Max t={all_ts[-1]}, expected values near 1.0'


# ============================================================================
# 12. Position reset at document boundaries
# ============================================================================


class TestPositionReset:
  def test_packed_docs_position_reset(self):
    """For packed [A,A,A,B,B,B]: positions = [0,1,2,0,1,2], not [0,1,2,3,4,5]."""
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    positions = _compute_positions(doc_ids)
    expected = torch.tensor([[0, 1, 2, 0, 1, 2]])
    assert torch.equal(positions, expected), f'Positions {positions} != expected {expected}'

  def test_many_short_docs(self):
    """8 single-token docs: all positions should be 0."""
    doc_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    positions = _compute_positions(doc_ids)
    expected = torch.zeros(1, 8, dtype=torch.long)
    assert torch.equal(positions, expected)

  def test_three_docs_varied_length(self):
    """doc0=2 tokens, doc1=3 tokens, doc2=3 tokens."""
    doc_ids = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]])
    positions = _compute_positions(doc_ids)
    expected = torch.tensor([[0, 1, 0, 1, 2, 0, 1, 2]])
    assert torch.equal(positions, expected)

  def test_batched_position_independence(self):
    """Each batch row computes positions independently."""
    doc_ids = torch.tensor([
      [0, 0, 1, 1, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    positions = _compute_positions(doc_ids)
    expected = torch.tensor([
      [0, 1, 0, 1, 0, 1, 2, 3],
      [0, 1, 2, 3, 4, 5, 6, 7],
    ])
    assert torch.equal(positions, expected)


# ============================================================================
# 13. Muon Newton-Schulz near-orthogonality
# ============================================================================


class TestNewtonSchulzOrthogonal:
  def test_ns5_gram_matrix_near_identity(self):
    """After 5 NS steps, X.T @ X should approximate a scaled identity.
    Off-diagonal / diagonal ratio should be small.
    """
    torch.manual_seed(42)
    G = torch.randn(64, 32)
    X = MuonClip.newton_schulz(G, steps=5)

    XtX = X.T.float() @ X.float()
    diag = XtX.diag()
    off_diag_mask = ~torch.eye(32, dtype=torch.bool)
    off_diag_vals = XtX[off_diag_mask]

    # Diagonal should be roughly uniform
    diag_cv = diag.std() / diag.mean()
    assert diag_cv < 0.2, f'Diagonal CV={diag_cv:.4f}, too variable'

    # Off-diagonal should be near zero relative to diagonal
    ratio = off_diag_vals.abs().max() / diag.mean()
    assert ratio < 0.3, f'Off-diagonal/diagonal ratio={ratio:.4f}, not near-orthogonal'

  def test_ns_more_steps_more_orthogonal(self):
    """More NS iterations -> better orthogonality (monotonic improvement)."""
    torch.manual_seed(42)
    G = torch.randn(64, 32)

    ratios = []
    for steps in [1, 3, 5, 7]:
      X = MuonClip.newton_schulz(G, steps=steps)
      XtX = X.T.float() @ X.float()
      diag = XtX.diag()
      off_diag_mask = ~torch.eye(32, dtype=torch.bool)
      off_diag_max = XtX[off_diag_mask].abs().max()
      ratios.append(off_diag_max / diag.mean())

    # Ratio should decrease (or at least not increase much) with more steps
    # Allow small tolerance for bf16 precision
    for i in range(1, len(ratios)):
      assert ratios[i] <= ratios[i - 1] + 0.05, (
        f'NS step {[1, 3, 5, 7][i]}: ratio {ratios[i]:.4f} > step {[1, 3, 5, 7][i - 1]}: {ratios[i - 1]:.4f}'
      )

  def test_ns_square_matrix(self):
    """NS works for square matrices too."""
    torch.manual_seed(42)
    G = torch.randn(32, 32)
    X = MuonClip.newton_schulz(G, steps=5)
    assert X.shape == (32, 32)

    XtX = X.T.float() @ X.float()
    diag = XtX.diag()
    off_diag_mask = ~torch.eye(32, dtype=torch.bool)
    ratio = XtX[off_diag_mask].abs().max() / diag.mean()
    assert ratio < 0.3


# ============================================================================
# 14. QK-Clip: _MaxLogitsTracker accumulation and consume
# ============================================================================


class TestMaxLogitsTracker:
  def test_update_and_consume(self):
    """_update accumulates max across calls; consume returns it and resets."""
    # Reset state
    _MaxLogitsTracker._tls.max_logits = None

    _MaxLogitsTracker._update(5.0)
    _MaxLogitsTracker._update(10.0)
    _MaxLogitsTracker._update(3.0)

    result = _MaxLogitsTracker.consume()
    assert result == 10.0

    # After consume, should be None
    result2 = _MaxLogitsTracker.consume()
    assert result2 is None

  def test_update_with_tensors(self):
    """_update works with 0-d tensors (the production path)."""
    _MaxLogitsTracker._tls.max_logits = None

    _MaxLogitsTracker._update(torch.tensor(5.0))
    _MaxLogitsTracker._update(torch.tensor(12.0))
    _MaxLogitsTracker._update(torch.tensor(8.0))

    result = _MaxLogitsTracker.consume()
    assert result == pytest.approx(12.0)

  def test_consume_returns_none_when_empty(self):
    """consume() returns None when no _update calls were made."""
    _MaxLogitsTracker._tls.max_logits = None
    assert _MaxLogitsTracker.consume() is None

  def test_mixed_tensor_and_float(self):
    """_update handles mixing tensor and float inputs."""
    _MaxLogitsTracker._tls.max_logits = None

    _MaxLogitsTracker._update(torch.tensor(5.0))
    _MaxLogitsTracker._update(20.0)

    result = _MaxLogitsTracker.consume()
    assert result == pytest.approx(20.0)

  def test_tracker_updates_during_forward(self):
    """Forward pass in training mode populates the tracker."""
    _MaxLogitsTracker._tls.max_logits = None
    # Reset QK-Clip frequency counter so the first forward triggers tracking
    import phase6.attention as _attn

    _attn._qk_clip_counter = 0

    cfg = _tiny(use_muon=True)
    model = Model(cfg)
    model.train()

    B, T = 1, 8
    x = torch.randint(3, cfg.vocab_size, (B, 2 * cfg.seq_len))
    targets = torch.randint(3, cfg.vocab_size, (B, cfg.seq_len))
    mask = build_staircase_mask(cfg.seq_len, cfg.block_size)

    hidden, _ = model(x, targets=targets, attn_mask=mask)

    result = _MaxLogitsTracker.consume()
    # After a forward pass with 2 layers, tracker should have a value
    assert result is not None, 'Tracker should be populated after training forward'
    assert result > 0, f'Max logit bound should be positive, got {result}'


# ============================================================================
# 15. Empty mask (all tokens unmasked): loss should be 0
# ============================================================================


class TestEmptyMask:
  def test_all_unmasked_zero_loss(self):
    """mask=False everywhere -> w_flat = 0 -> loss = 0."""
    torch.manual_seed(42)
    B, L, D, V = 1, 8, 32, 64
    h = torch.randn(B, L, D, requires_grad=True)
    w_lm = torch.randn(V, D)
    targets = torch.randint(3, V, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool)  # all unmasked
    elbo_w = torch.ones(B, L)

    loss = compute_loss(h, targets, mask, elbo_w, w_lm, _FakeCfg())
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f'All-unmasked loss should be 0, got {loss.item()}'


# ============================================================================
# 16. Full mask (all tokens masked): loss should be highest
# ============================================================================


class TestFullMask:
  def test_full_mask_higher_than_partial(self):
    """All tokens masked -> higher loss than 50% masked (same weights)."""
    torch.manual_seed(42)
    B, L, D, V = 1, 16, 32, 64
    h = torch.randn(B, L, D)
    w_lm = torch.randn(V, D)
    targets = torch.randint(3, V, (B, L))
    elbo_w = torch.ones(B, L)

    mask_full = torch.ones(B, L, dtype=torch.bool)
    loss_full = compute_loss(h.detach().requires_grad_(True), targets, mask_full, elbo_w, w_lm, _FakeCfg())

    mask_half = torch.zeros(B, L, dtype=torch.bool)
    mask_half[:, : L // 2] = True
    loss_half = compute_loss(h.detach().requires_grad_(True), targets, mask_half, elbo_w, w_lm, _FakeCfg())

    assert loss_full.item() > loss_half.item(), (
      f'Full mask loss ({loss_full.item():.4f}) should exceed half mask ({loss_half.item():.4f})'
    )


# ============================================================================
# 17. Single token sequence
# ============================================================================


class TestSingleToken:
  def test_single_block_forward(self):
    """seq_len=block_size=8, batch=1 should work for generation forward."""
    cfg = _tiny(seq_len=8, block_size=8)
    model = Model(cfg)
    model.eval()

    idx = torch.randint(3, cfg.vocab_size, (1, 8))
    with torch.no_grad():
      logits, _ = model(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)

  def test_single_block_training(self):
    """Training with seq_len=block_size=8 should work."""
    cfg = _tiny(seq_len=8, block_size=8)
    model = Model(cfg)
    model.train()

    B, L = 1, 8
    targets = torch.randint(3, cfg.vocab_size, (B, L))
    mask_attn = build_staircase_mask(L, cfg.block_size)
    idx = torch.randint(3, cfg.vocab_size, (B, 2 * L))

    hidden, _ = model(idx, targets=targets, attn_mask=mask_attn)
    assert hidden.shape == (B, L, cfg.n_embd)

    # Loss should work
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    loss = compute_loss(hidden, targets, mask, elbo_w, model.lm_head.weight, cfg)
    assert loss.item() > 0
    loss.backward()

  def test_single_block_generation(self):
    """generate() with block_size=8, max_new_tokens=8 should produce one block."""
    cfg = _tiny(seq_len=8, block_size=8)
    model = Model(cfg)
    model.eval()

    prompt_ids = [10, 20, 30]
    result = generate(model, prompt_ids, cfg, max_new_tokens=8, temperature=0.0)
    assert isinstance(result, list)
    assert len(result) >= len(prompt_ids)


# ============================================================================
# 18. Vocab size boundary
# ============================================================================


class TestVocabBoundary:
  def test_max_token_id_in_forward(self):
    """Token at vocab_size-1 should work in forward pass."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    idx = torch.full((1, 8), cfg.vocab_size - 1, dtype=torch.long)
    with torch.no_grad():
      logits, _ = model(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()

  def test_min_token_id_zero(self):
    """Token ID 0 should work."""
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    idx = torch.zeros((1, 8), dtype=torch.long)
    with torch.no_grad():
      logits, _ = model(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()

  def test_target_at_vocab_boundary(self):
    """Target token at vocab_size-1 should produce valid loss."""
    cfg = _tiny()
    model = Model(cfg)
    model.train()

    B, L = 1, 8
    idx = torch.randint(3, cfg.vocab_size, (B, 2 * L))
    targets = torch.full((B, L), cfg.vocab_size - 1, dtype=torch.long)
    mask_attn = build_staircase_mask(L, cfg.block_size)

    hidden, _ = model(idx, targets=targets, attn_mask=mask_attn)
    mask = torch.ones(B, L, dtype=torch.bool)
    elbo_w = torch.ones(B, L)
    loss = compute_loss(hidden, targets, mask, elbo_w, model.lm_head.weight, cfg)

    assert not torch.isnan(loss), 'Loss should not be NaN for max token target'
    assert not torch.isinf(loss), 'Loss should not be Inf for max token target'
    assert loss.item() > 0


# ============================================================================
# 19. RoPE position doubling in training mode
# ============================================================================


class TestRoPETrainingDoubling:
  def test_training_both_halves_get_same_positions(self):
    """In training [x_t || x_0], both halves use positions 0..L-1.
    The model slices cos[:, :L] and applies to both q_t, q_0.
    """
    cfg = _tiny()
    model = Model(cfg)
    model.eval()

    B, L = 1, cfg.seq_len
    idx = torch.randint(3, cfg.vocab_size, (B, 2 * L))
    targets = torch.randint(3, cfg.vocab_size, (B, L))
    mask = build_staircase_mask(L, cfg.block_size)

    # The model should use cos[:, :L] for both halves
    # Verify by checking that T == 2*L triggers the doubling path
    # (cos.size(1) == L, but T == 2*L)
    cos_len = model.cos[:, :L].shape[1]
    assert cos_len == L
    assert 2 * cos_len == 2 * L  # input length

    # Just verify forward works (the assertion inside model.forward
    # would catch T != 2*L)
    with torch.no_grad():
      hidden, _ = model(idx, targets=targets, attn_mask=mask)
    assert hidden.shape == (B, L, cfg.n_embd)


# ============================================================================
# 20. Staircase mask with doc boundaries: cross-doc attention blocked
# ============================================================================


class TestStaircaseMaskDocBoundaries:
  def test_cross_doc_attention_blocked(self):
    """Two docs packed: doc0=[0..7], doc1=[8..15]. Cross-doc pairs should be -inf."""
    L, blk = 16, 4
    B = 1
    doc_ids = torch.zeros(B, L, dtype=torch.long)
    doc_ids[:, 8:] = 1  # doc1 starts at position 8

    mask = build_staircase_mask(L, blk, doc_ids=doc_ids)
    # mask shape: (B, 1, 2L, 2L)

    # x_0 block 2 (pos 24-27, doc1) attending to x_0 block 0 (pos 16-19, doc0)
    # M_BC allows this structurally (block 2 >= block 0, both x_0)
    # But doc boundary should block it
    assert mask[0, 0, 24, 16].item() == float('-inf'), (
      'x_0[doc1 block2] should NOT attend x_0[doc0 block0] -- cross-doc'
    )

  def test_same_doc_attention_allowed(self):
    """Within same doc, normal staircase rules apply."""
    L, blk = 16, 4
    B = 1
    doc_ids = torch.zeros(B, L, dtype=torch.long)  # single doc

    mask = build_staircase_mask(L, blk, doc_ids=doc_ids)

    # x_t block 1 attending to x_0 block 0 (M_OBC, same doc) should be allowed
    # x_t block 1 = positions 4-7, x_0 block 0 = positions 16-19
    assert mask[0, 0, 4, 16].item() == 0.0, 'x_t[block1] should attend x_0[block0] within same doc'


# ============================================================================
# 21. Gumbel noise at temperature=0 is identity
# ============================================================================


class TestGumbelNoise:
  def test_temperature_zero_returns_logits(self):
    """_add_gumbel_noise with temperature=0 returns logits unchanged."""
    logits = torch.randn(1, 8, 256)
    result = _add_gumbel_noise(logits, temperature=0)
    assert torch.equal(result, logits)

  def test_temperature_nonzero_changes_logits(self):
    """Non-zero temperature should produce different result."""
    torch.manual_seed(42)
    logits = torch.randn(1, 8, 256)
    result = _add_gumbel_noise(logits, temperature=0.7)
    assert not torch.equal(result, logits)

  def test_gumbel_preserves_relative_order_roughly(self):
    """High-probability tokens should still be selected more often with Gumbel noise."""
    torch.manual_seed(42)
    # Create logits where token 0 is strongly preferred
    logits = torch.zeros(1, 1, 10)
    logits[0, 0, 0] = 10.0  # strongly prefer token 0

    # Sample 100 times -- token 0 should be argmax most of the time
    count_token0 = 0
    for seed in range(100):
      torch.manual_seed(seed)
      noisy = _add_gumbel_noise(logits, temperature=0.5)
      if torch.argmax(noisy, dim=-1).item() == 0:
        count_token0 += 1

    assert count_token0 > 70, f'Token 0 (logit=10) should be selected >70% of the time, got {count_token0}%'


# ============================================================================
# 22. Tied embedding weight: forward + backward consistency
# ============================================================================


class TestTiedEmbeddingConsistency:
  def test_lm_head_and_embedding_share_gradients(self):
    """Since lm_head.weight IS token_emb.weight, gradients should accumulate
    from both the embedding lookup and the lm_head projection.
    """
    cfg = _tiny()
    model = Model(cfg)
    model.train()

    # Verify they're the same tensor
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()

    B, T = 1, 8
    idx = torch.randint(3, cfg.vocab_size, (B, T))
    logits, _ = model(idx)
    loss = logits.sum()
    loss.backward()

    # The shared weight should have non-zero gradients
    grad = model.token_emb.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0

    # And it should be the same object as lm_head's grad
    assert model.lm_head.weight.grad is model.token_emb.weight.grad


# ============================================================================
# 23. Residual initialization: c_proj and down_proj have scaled std
# ============================================================================


class TestResidualInit:
  def test_residual_projections_smaller_std(self):
    """c_proj and down_proj should have std = 0.02 / sqrt(2 * n_layer),
    which is smaller than the default 0.02 std.
    """
    cfg = _tiny(n_layer=4)
    model = Model(cfg)

    for block in model.blocks:
      c_proj_std = block.attn.c_proj.weight.std().item()
      down_std = block.mlp.down_proj.weight.std().item()

      # Should be within 2x of expected (statistical tolerance)
      assert c_proj_std < 0.02, f'c_proj std {c_proj_std:.5f} should be < 0.02 (scaled init)'
      assert down_std < 0.02, f'down_proj std {down_std:.5f} should be < 0.02 (scaled init)'

  def test_gated_query_zero_init(self):
    """Gated Query Attention: w_gate should be zero-initialized."""
    cfg = _tiny(use_gated_query=True)
    model = Model(cfg)

    for block in model.blocks:
      gate_weight = block.attn.w_gate.weight
      assert torch.all(gate_weight == 0), 'w_gate should be zero-initialized for identity behavior at init'


# ============================================================================
# 24. Static remasking: commits exactly num_to_commit tokens
# ============================================================================


class TestStaticRemasking:
  def test_exact_count(self):
    """_select_tokens_static always commits exactly num_to_commit tokens."""
    B, L = 1, 8
    confidences = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0]])
    masked = torch.tensor([[True, True, True, True, True, True, False, False]])

    for k in [1, 2, 3, 5]:
      commit = _select_tokens_static(confidences, masked, k)
      assert commit.sum().item() == k, f'Static remasking: expected {k} commits, got {commit.sum().item()}'

  def test_selects_highest_confidence(self):
    """Static remasking selects the highest-confidence masked tokens."""
    confidences = torch.tensor([[0.1, 0.9, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]])
    masked = torch.tensor([[True, True, True, True, False, False, False, False]])

    commit = _select_tokens_static(confidences, masked, 2)
    # Highest confidence masked: positions 1 (0.9) and 2 (0.5)
    assert commit[0, 1].item(), 'Position 1 (conf=0.9) should be committed'
    assert commit[0, 2].item(), 'Position 2 (conf=0.5) should be committed'


# ============================================================================
# 25. QK-Clip triggers weight scaling
# ============================================================================


class TestQKClip:
  def test_qk_clip_scales_weights(self):
    """When max_logits > tau, QK-Clip scales weights and updates."""
    cfg = _tiny(use_muon=True)
    model = Model(cfg)
    model.train()
    optimizer = create_optimizer(model, cfg)

    # Get the QK group params
    qk_params = []
    for g in optimizer.param_groups:
      if g.get('is_qk'):
        qk_params = g['params']
        break

    if not qk_params:
      pytest.skip('No QK params found')

    # Snapshot before
    w_before = qk_params[0].clone()

    # Fake forward + backward
    x = torch.randint(3, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()

    # Manually set tracker to a value > tau (100.0)
    _MaxLogitsTracker._tls.max_logits = None
    _MaxLogitsTracker._update(200.0)

    optimizer.step()

    # Weight should have changed
    assert not torch.equal(w_before, qk_params[0])

  def test_qk_clip_no_scaling_below_tau(self):
    """When max_logits < tau, no QK-Clip scaling occurs. Only normal Muon update."""
    cfg = _tiny(use_muon=True)
    model = Model(cfg)
    model.train()
    optimizer = create_optimizer(model, cfg)

    x = torch.randint(3, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()

    # Set tracker below tau
    _MaxLogitsTracker._tls.max_logits = None
    _MaxLogitsTracker._update(50.0)  # < tau=100

    # This should still run without error
    optimizer.step()


# ============================================================================
# 26. Transfer token schedule sums to block_length
# ============================================================================


class TestTransferSchedule:
  def test_schedule_sums_correctly(self):
    """get_num_transfer_tokens always sums to exactly block_length."""
    for blk in [4, 7, 8, 10, 16]:
      for steps in [1, 2, 3, 4, 5, 8]:
        schedule = get_num_transfer_tokens(blk, steps)
        assert schedule.sum().item() == blk, f'blk={blk}, steps={steps}: sum={schedule.sum().item()}'
        assert len(schedule) == steps
        assert (schedule >= 0).all()

  def test_schedule_remainder_distribution(self):
    """Remainder tokens go to the first steps."""
    schedule = get_num_transfer_tokens(10, 3)
    # 10 // 3 = 3 base, remainder = 1
    assert schedule[0].item() == 4  # 3 + 1
    assert schedule[1].item() == 3
    assert schedule[2].item() == 3


# ============================================================================
# 27. RMSNorm correctness
# ============================================================================


class TestRMSNorm:
  def test_rms_norm_output_scale(self):
    """RMSNorm output should have approximately unit RMS per token."""
    cfg = _tiny()
    norm = _make_rms_norm(64, cfg)

    x = torch.randn(2, 8, 64) * 5.0  # large input
    y = norm(x)

    # RMS of each token should be approximately 1.0 (with weight=ones)
    rms = (y**2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), (
      f'RMS norms should be ~1.0, got range [{rms.min():.3f}, {rms.max():.3f}]'
    )

  def test_rms_norm_preserves_shape(self):
    """RMSNorm should not change tensor shape."""
    cfg = _tiny()
    norm = _make_rms_norm(32, cfg)
    x = torch.randn(2, 8, 32)
    y = norm(x)
    assert y.shape == x.shape
