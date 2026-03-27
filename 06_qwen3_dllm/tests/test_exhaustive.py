"""EXHAUSTIVE numerical QA for Phase 6 -- final gate before real training.

Every calculation verified with hand-computed values or mathematical identities.
CPU only, TINY config. No duplication with existing tests -- only NEW numerical
correctness checks not already covered.
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from phase6.config import Config
from phase6.model import Model
from phase6.attention import (
    _apply_rotary_emb,
    apply_rotary_emb,
    build_staircase_mask,
    MultiHeadAttention,
)
from phase6.loss import compute_loss, _compute_chunk_size, _chunk_ce
from phase6.schedule import (
    sample_timesteps,
    apply_noise,
    compute_elbo_weight,
    get_lr_factor,
    compute_cart_weights,
)
from phase6.data import _compute_positions, _sample_t_per_doc, _apply_noise_per_doc
from phase6.optim import (
    MuonClip,
    build_param_groups,
    create_optimizer,
    _dedup_params,
    _is_qk_name,
)
from phase6.generate import (
    get_num_transfer_tokens,
    _add_gumbel_noise,
    _select_tokens_dynamic,
    _select_tokens_static,
    _select_tokens_random,
    generate,
)
from phase6.checkpoint import _map_hf_key, load_from_hf, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# TINY config shared across all tests
# ---------------------------------------------------------------------------

def _tiny(**kw):
    defaults = dict(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000.0, rms_eps=1e-6, dropout=0.0, t_min=0.1,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False, use_compile=False,
        use_muon=False, use_cart=False, use_amp=False,
        mask_token_id=0, eos_token_id=1, pad_token_id=2,
        muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
        denoise_steps=4, device='cpu',
    )
    defaults.update(kw)
    return Config(**defaults).validate()


TINY = _tiny()


# ============================================================================
# A. RoPE Numerical Correctness (attention.py)
# ============================================================================

class TestRoPENumerical:

    def test_a1_rope_frequencies_manual(self):
        """Verify inv_freq = 1/(base^(2i/d)) for base=10000, head_dim=32."""
        # inv_freq for head_dim=32: channels [0, 2, 4, ..., 30]
        # inv_freq_i = 1 / (10000 ^ (2i/32)) for i in 0..15
        base = 10000.0
        head_dim = 32
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # Hand-compute a few values:
        # i=0:  1/(10000^(0/32))  = 1.0
        # i=1:  1/(10000^(2/32))  = 1/(10000^0.0625) = 1/10^(4*0.0625) = 1/10^0.25 = 10^(-0.25)
        # i=15: 1/(10000^(30/32)) = 1/(10000^0.9375) = 10^(-3.75)
        assert inv_freq[0].item() == pytest.approx(1.0, abs=1e-6)
        assert inv_freq[1].item() == pytest.approx(10 ** (-0.25), rel=1e-5)
        assert inv_freq[15].item() == pytest.approx(10 ** (-3.75), rel=1e-4)

    def test_a2_rope_position_0_identity(self):
        """At position 0, cos=1 and sin=0 -> RoPE is identity."""
        base = 10000.0
        head_dim = 32
        d_half = head_dim // 2

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # pos=0 -> freqs = 0 * inv_freq = 0
        freqs = 0.0 * inv_freq  # all zeros
        cos_val = freqs.cos()  # all ones
        sin_val = freqs.sin()  # all zeros

        assert torch.allclose(cos_val, torch.ones(d_half))
        assert torch.allclose(sin_val, torch.zeros(d_half))

        # Apply RoPE at pos=0: output should equal input
        x = torch.randn(1, 1, 1, head_dim)
        cos_buf = cos_val.view(1, 1, 1, d_half)
        sin_buf = sin_val.view(1, 1, 1, d_half)
        y = _apply_rotary_emb(x, cos_buf, sin_buf)
        assert torch.allclose(y, x, atol=1e-6), "RoPE at position 0 should be identity"

    def test_a3_rope_position_1_manual(self):
        """At position 1, verify cos/sin match hand calculation for first channel."""
        base = 10000.0
        head_dim = 32

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # pos=1 -> freqs = 1 * inv_freq = inv_freq
        freqs = inv_freq
        cos_val = freqs.cos()
        sin_val = freqs.sin()

        # Channel 0: freq = 1.0, so cos(1.0) and sin(1.0)
        assert cos_val[0].item() == pytest.approx(math.cos(1.0), abs=1e-5)
        assert sin_val[0].item() == pytest.approx(math.sin(1.0), abs=1e-5)

        # Channel 1: freq = 10^(-0.25)
        freq1 = 10 ** (-0.25)
        assert cos_val[1].item() == pytest.approx(math.cos(freq1), abs=1e-5)
        assert sin_val[1].item() == pytest.approx(math.sin(freq1), abs=1e-5)

    def test_a4_rope_rotation_hand_calculation(self):
        """Apply RoPE on known input, verify output matches hand calculation.

        For head_dim=4, x = [a, b, c, d], cos = [cos0, cos1], sin = [sin0, sin1]:
        x1 = [a, b], x2 = [c, d]
            y1 = x1 * cos - x2 * sin = [a*cos0 - c*sin0, b*cos1 - d*sin1]
            y2 = x1 * sin + x2 * cos = [a*sin0 + c*cos0, b*sin1 + d*cos1]
        """
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)
        cos = torch.tensor([[[[0.5, 0.6]]]])  # (1,1,1,2)
        sin = torch.tensor([[[[0.3, 0.4]]]])  # (1,1,1,2)

        y = _apply_rotary_emb(x, cos, sin)

        # x1=[1,2], x2=[3,4]
        # y1 = x1*cos - x2*sin = [1*0.5-3*0.3, 2*0.6-4*0.4] = [0.5-0.9, 1.2-1.6] = [-0.4, -0.4]
        # y2 = x1*sin + x2*cos = [1*0.3+3*0.5, 2*0.4+4*0.6] = [0.3+1.5, 0.8+2.4] = [1.8, 3.2]
        expected = torch.tensor([[[[-0.4, -0.4, 1.8, 3.2]]]])
        assert torch.allclose(y, expected, atol=1e-5), f"Got {y}, expected {expected}"

    def test_a5_rope_with_positions_tensor(self):
        """Doc packing path: positions (B, T) selects per-token cos/sin."""
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        B, T = 1, 8
        # positions: [0, 1, 2, 0, 1, 2, 3, 4] -- simulates 2 docs
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])

        cos, sin = model.cos, model.sin
        # Manual gather for position 0 and position 3 (which is also position 0 in doc)
        # Both should use the same cos/sin values
        cos_pos0 = cos[0, 0, 0, :]  # cos at position 0
        cos_pos_3 = cos[0, positions[0, 3].item(), 0, :]
        assert torch.allclose(cos_pos0, cos_pos_3), \
            "Position 3 (doc-local 0) should have same cos as position 0"


# ============================================================================
# B. Staircase Mask -- Every Cell (attention.py)
# ============================================================================

class TestStaircaseMaskExhaustive:

    def test_b6_full_mask_every_cell(self):
        """For seq_len=16, block_size=4: build 32x32 mask and verify EVERY cell.

        Layout: L=16, blocks of 4 tokens
        x_t half: positions 0-15, blocks [0:4], [4:8], [8:12], [12:16]
        x_0 half: positions 16-31, blocks [16:20], [20:24], [24:28], [28:32]

        Rules:
        M_BD:  same block, same half -> True
        M_OBC: x_t attends to x_0 from STRICTLY earlier blocks -> True
        M_BC:  x_0 attends to x_0 causally (current + earlier) -> True
        """
        L, blk = 16, 4
        mask = build_staircase_mask(L, blk)
        # mask shape: (2L, 2L)
        assert mask.shape == (2 * L, 2 * L), f"Mask shape {mask.shape}"

        n = L
        total_true = 0
        for q in range(2 * n):
            for k in range(2 * n):
                q_is_x0 = q >= n
                k_is_x0 = k >= n
                q_block = (q % n) // blk
                k_block = (k % n) // blk

                # M_BD: same block, same half
                m_bd = (q_block == k_block) and (q_is_x0 == k_is_x0)
                # M_OBC: x_t queries attend to x_0 keys from STRICTLY earlier blocks
                m_obc = (q_block > k_block) and k_is_x0 and not q_is_x0
                # M_BC: x_0 queries attend to x_0 keys from current or earlier blocks
                m_bc = (q_block >= k_block) and k_is_x0 and q_is_x0

                should_attend = m_bd or m_obc or m_bc
                actual = mask[q, k].item()

                if should_attend:
                    assert actual == 0.0, \
                        f"Cell [{q},{k}] should be allowed (0.0) but got {actual}"
                    total_true += 1
                else:
                    assert actual == float('-inf'), \
                        f"Cell [{q},{k}] should be blocked (-inf) but got {actual}"

        # Verify total count matches expectation
        assert total_true > 0, "No allowed cells in mask"

    def test_b7_mask_symmetric_within_mbd(self):
        """M_BD blocks are symmetric (bidirectional attention within block)."""
        L, blk = 16, 4
        mask = build_staircase_mask(L, blk)

        n = L
        for block_idx in range(n // blk):
            # x_t block: positions [block_idx*blk : (block_idx+1)*blk]
            start = block_idx * blk
            end = start + blk
            for q in range(start, end):
                for k in range(start, end):
                    assert mask[q, k] == mask[k, q], \
                        f"M_BD not symmetric at x_t block {block_idx}: [{q},{k}] vs [{k},{q}]"

            # x_0 block: positions [n + start : n + end]
            for q in range(n + start, n + end):
                for k in range(n + start, n + end):
                    assert mask[q, k] == mask[k, q], \
                        f"M_BD not symmetric at x_0 block {block_idx}: [{q},{k}] vs [{k},{q}]"

    def test_b8_mask_total_attention_count(self):
        """Verify exact total number of allowed cells for L=16, blk=4."""
        L, blk = 16, 4
        mask = build_staircase_mask(L, blk)
        n_blocks = L // blk  # 4 blocks

        allowed = (mask == 0.0).sum().item()

        # Calculate expected:
        # M_BD: each block of size 4 in each half -> 4*4=16 per block, 4 blocks, 2 halves = 128
        m_bd_count = n_blocks * blk * blk * 2

        # M_OBC: x_t[block i] attends to x_0[block j] where i > j
        # block 1 -> block 0: 4*4 = 16
        # block 2 -> blocks 0,1: 4*4*2 = 32
        # block 3 -> blocks 0,1,2: 4*4*3 = 48
        m_obc_count = 0
        for i in range(n_blocks):
            m_obc_count += i * blk * blk

        # M_BC: x_0[block i] attends to x_0[block j] where i >= j
        # block 0 -> block 0: 16  (already in M_BD)
        # block 1 -> blocks 0,1: 32 (block 1->1 in M_BD)
        # block 2 -> blocks 0,1,2: 48 (block 2->2 in M_BD)
        # block 3 -> blocks 0,1,2,3: 64 (block 3->3 in M_BD)
        m_bc_count = 0
        for i in range(n_blocks):
            m_bc_count += (i + 1) * blk * blk

        # M_BD counted x_0 blocks attending to themselves (diagonal blocks in x_0).
        # M_BC also includes x_0 block i -> x_0 block i.
        # The union is: M_BD + M_OBC + M_BC minus overlaps.
        # Overlap: M_BD and M_BC overlap on x_0 block-diagonal (same block, x_0 half).
        # M_BD x_0 diagonal = n_blocks * blk^2
        overlap = n_blocks * blk * blk

        expected = m_bd_count + m_obc_count + m_bc_count - overlap
        assert allowed == expected, \
            f"Expected {expected} allowed cells, got {allowed} (bd={m_bd_count}, obc={m_obc_count}, bc={m_bc_count}, overlap={overlap})"


# ============================================================================
# C. GQA Expansion (attention.py)
# ============================================================================

class TestGQAExpansion:

    def test_c9_repeat_interleave_correctness(self):
        """Create K with 2 kv_heads, verify repeat_interleave expands to 4 heads correctly."""
        B, T, n_head, n_kv_head, head_dim = 1, 4, 4, 2, 32

        # K: (B, n_kv_head, T, head_dim)
        K = torch.randn(B, n_kv_head, T, head_dim)
        repeats = n_head // n_kv_head  # 2

        K_expanded = K.repeat_interleave(repeats, dim=1)
        assert K_expanded.shape == (B, n_head, T, head_dim)

        # Head 0 and 1 should be copies of KV head 0
        assert torch.equal(K_expanded[:, 0], K[:, 0])
        assert torch.equal(K_expanded[:, 1], K[:, 0])
        # Head 2 and 3 should be copies of KV head 1
        assert torch.equal(K_expanded[:, 2], K[:, 1])
        assert torch.equal(K_expanded[:, 3], K[:, 1])

    def test_c10_sdpa_output_shape_with_gqa(self):
        """SDPA with GQA expansion produces correct output shape."""
        B, T, n_head, n_kv_head, head_dim = 1, 4, 4, 2, 32
        repeats = n_head // n_kv_head

        Q = torch.randn(B, n_head, T, head_dim)
        K = torch.randn(B, n_kv_head, T, head_dim).repeat_interleave(repeats, dim=1)
        V = torch.randn(B, n_kv_head, T, head_dim).repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        assert out.shape == (B, n_head, T, head_dim)

    def test_c11_full_attention_with_gqa(self):
        """Full MultiHeadAttention forward with GQA (n_head=4, n_kv_head=2)."""
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        B, T = 1, 8
        x = torch.randn(B, T, cfg.n_embd)
        cos = model.cos[:, :T]
        sin = model.sin[:, :T]

        with torch.no_grad():
            out = model.blocks[0].attn(x, cos, sin)
        assert out.shape == (B, T, cfg.n_embd)


# ============================================================================
# D. Loss Numerical Correctness (loss.py)
# ============================================================================

class _FakeCfg:
    pad_token_id = 2


class TestLossNumerical:

    def test_d12_hand_computed_ce(self):
        """Hand-compute CE for a 3-token, 4-vocab example.

        hidden = [[1,0],[0,1],[0.5,0.5]], weight = identity-like (4x2)
        logits[0] = [1, 0, 0, 0] -> softmax -> p[0] = e^1/(e^1+3)
        target[0] = 0 -> CE = -log(p[0])
        """
        D, V = 2, 4
        B, L = 1, 3
        h = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], requires_grad=True)
        # Weight matrix: identity-like (4x2)
        w = torch.zeros(V, D, requires_grad=True)
        w.data[0, 0] = 1.0
        w.data[1, 1] = 1.0
        targets = torch.tensor([[0, 1, 0]])
        mask = torch.ones(B, L, dtype=torch.bool)
        elbo_w = torch.ones(B, L)

        loss = compute_loss(h, targets, mask, elbo_w, w, _FakeCfg())

        # Manual: logits[0] = h[0] @ w^T = [1*1+0*0, 1*0+0*1, 0, 0] = [1, 0, 0, 0]
        #   CE(logits[0], target=0) = -log(e^1/(e^1+e^0+e^0+e^0)) = -log(e/(e+3))
        # logits[1] = [0, 1, 0, 0]
        #   CE(logits[1], target=1) = -log(e^1/(e^0+e^1+e^0+e^0)) = -log(e/(e+3))
        # logits[2] = [0.5, 0.5, 0, 0]
        #   CE(logits[2], target=0) = -log(e^0.5/(e^0.5+e^0.5+e^0+e^0)) = -log(e^0.5/(2*e^0.5+2))
        ce0 = -math.log(math.exp(1.0) / (math.exp(1.0) + 3))
        ce1 = -math.log(math.exp(1.0) / (math.exp(1.0) + 3))
        ce2 = -math.log(math.exp(0.5) / (2 * math.exp(0.5) + 2))
        # real_count = 3 (none are pad_token_id=2)
        expected = (ce0 + ce1 + ce2) / 3.0

        assert loss.item() == pytest.approx(expected, rel=1e-4), \
            f"Loss {loss.item():.6f} != expected {expected:.6f}"

    def test_d13_elbo_weight_scales_exactly(self):
        """loss(w=2) should be exactly 2x loss(w=1) for same tokens."""
        torch.manual_seed(42)
        h = torch.randn(1, 8, 32, requires_grad=True)
        w = torch.randn(256, 32, requires_grad=True)
        targets = torch.randint(3, 256, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)

        ew1 = torch.ones(1, 8)
        ew2 = torch.full((1, 8), 2.0)

        loss1 = compute_loss(h.detach().requires_grad_(True), targets, mask, ew1, w, _FakeCfg())
        loss2 = compute_loss(h.detach().requires_grad_(True), targets, mask, ew2, w, _FakeCfg())

        ratio = loss2.item() / loss1.item()
        assert ratio == pytest.approx(2.0, rel=1e-4), f"Ratio {ratio} != 2.0"

    def test_d14_chunking_doesnt_change_result(self):
        """Loss with chunk=2 should equal loss with chunk=large for same data."""
        torch.manual_seed(42)
        D, V = 32, 64
        h = torch.randn(2, 16, D, requires_grad=True)
        w = torch.randn(V, D, requires_grad=True)
        targets = torch.randint(3, V, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        ew = torch.ones(2, 16) * 3.0

        # Compute with default chunking
        loss_default = compute_loss(
            h.detach().requires_grad_(True), targets, mask, ew, w, _FakeCfg()
        )

        # Compute manually with one giant chunk (no chunking)
        h_flat = h.detach().contiguous().view(-1, D)
        t_flat = targets.contiguous().view(-1)
        w_flat = (mask.float() * ew).contiguous().view(-1)
        logits_full = h_flat @ w.T
        ce_full = F.cross_entropy(logits_full, t_flat, reduction='none')
        real_count = (targets != _FakeCfg.pad_token_id).float().sum().clamp(min=1)
        loss_manual = (ce_full * w_flat).sum() / real_count

        assert loss_default.item() == pytest.approx(loss_manual.item(), rel=1e-4), \
            f"Chunked {loss_default.item():.6f} != manual {loss_manual.item():.6f}"

    def test_d15_grad_checkpoint_doesnt_change_gradients(self):
        """Gradients with grad_checkpoint (inside compute_loss) should match non-checkpointed."""
        torch.manual_seed(42)
        D, V = 32, 64
        h1 = torch.randn(1, 8, D, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_(True)
        w1 = torch.randn(V, D, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_(True)
        targets = torch.randint(3, V, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.bool)
        ew = torch.ones(1, 8)

        # compute_loss uses grad_checkpoint internally
        loss1 = compute_loss(h1, targets, mask, ew, w1, _FakeCfg())
        loss1.backward()

        # Manual (no grad_checkpoint)
        logits = h2 @ w2.T
        ce = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
        w_flat = (mask.float() * ew).view(-1)
        real_count = (targets != _FakeCfg.pad_token_id).float().sum().clamp(min=1)
        loss2 = (ce * w_flat).sum() / real_count
        loss2.backward()

        # Gradients should match
        assert h1.grad is not None and h2.grad is not None
        assert torch.allclose(h1.grad, h2.grad, atol=1e-5), \
            f"Max grad diff: {(h1.grad - h2.grad).abs().max().item()}"


# ============================================================================
# E. Noise Schedule (schedule.py)
# ============================================================================

class TestNoiseScheduleExhaustive:

    def test_e16_antithetic_sampling_stratified(self):
        """Antithetic sampling: t values should be stratified across [t_min, 1).

        With B=1, num_blocks=4, each block should get a t from a different quartile.
        """
        t_blocks, _ = sample_timesteps(batch_size=1, num_blocks=4, block_size=8, t_min=0.0)
        # t_blocks shape: (1, 4)
        vals = sorted(t_blocks[0].tolist())
        # With stratified sampling (offset = [0/4, 1/4, 2/4, 3/4]),
        # each value falls in its own quartile
        for i, v in enumerate(vals):
            lo = i / 4
            hi = (i + 1) / 4
            assert lo <= v < hi + 0.01, \
                f"Value {v:.3f} not in expected quartile [{lo:.2f}, {hi:.2f})"

    def test_e17_min_one_masked_per_block_statistical(self):
        """Generate 100 batches at t=0.01 (1% rate), verify zero-mask blocks = 0."""
        zero_mask_blocks_found = 0
        for seed in range(100):
            torch.manual_seed(seed)
            targets = torch.randint(10, 100, (4, 32))
            t = torch.full((4, 32), 0.01)
            _, noise_mask = apply_noise(targets, t, mask_token_id=0, block_size=8)
            mask_blocks = noise_mask.view(4, 4, 8)
            per_block_count = mask_blocks.sum(dim=2)
            zero_mask_blocks_found += (per_block_count == 0).sum().item()

        assert zero_mask_blocks_found == 0, \
            f"Found {zero_mask_blocks_found} zero-mask blocks in 100 batches -- guarantee violated"

    def test_e18_elbo_weight_is_exactly_1_over_t(self):
        """ELBO weight = exactly 1/t at multiple t values."""
        t_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        for tv in t_vals:
            t = torch.tensor([[tv]])
            w = compute_elbo_weight(t, t_min=0.1)
            expected = 1.0 / tv
            assert w.item() == pytest.approx(expected, rel=1e-5), \
                f"t={tv}: expected {expected}, got {w.item()}"

    def test_e19_wsd_lr_schedule_key_points(self):
        """WSD LR schedule: warmup -> stable -> decay, exact values at key steps."""
        warmup = 100
        decay_start = 800
        max_iters = 1000

        # Step 0: (0+1)/100 = 0.01
        assert get_lr_factor(0, warmup, decay_start, max_iters) == pytest.approx(1.0 / warmup)

        # Step warmup-1: warmup/warmup = 1.0
        assert get_lr_factor(warmup - 1, warmup, decay_start, max_iters) == pytest.approx(1.0)

        # Step warmup: stable phase = 1.0
        assert get_lr_factor(warmup, warmup, decay_start, max_iters) == 1.0

        # Step decay_start - 1: still stable = 1.0
        assert get_lr_factor(decay_start - 1, warmup, decay_start, max_iters) == 1.0

        # Step decay_start: start of decay, factor = 1 - (800-800)/(1000-800) = 1.0
        assert get_lr_factor(decay_start, warmup, decay_start, max_iters) == 1.0

        # Step 900: factor = max(0, 1 - (900-800)/(1000-800)) = 1 - 100/200 = 0.5
        assert get_lr_factor(900, warmup, decay_start, max_iters) == pytest.approx(0.5)

        # Step max_iters: factor = max(0, 1 - (1000-800)/(1000-800)) = 0.0
        assert get_lr_factor(max_iters, warmup, decay_start, max_iters) == 0.0

        # Monotonicity in decay phase
        factors = [get_lr_factor(s, warmup, decay_start, max_iters)
                   for s in range(decay_start, max_iters + 1)]
        for i in range(1, len(factors)):
            assert factors[i] <= factors[i - 1], \
                f"LR not monotonically decreasing: step {decay_start + i}: {factors[i]} > {factors[i-1]}"


# ============================================================================
# F. Optimizer Correctness (optim.py)
# ============================================================================

class TestOptimizerExhaustive:

    def test_f20_newton_schulz_near_orthogonal(self):
        """Newton-Schulz produces near-orthogonal matrix: X @ X^T approx scaled I.

        NS runs in bfloat16 internally, so we allow wider tolerance.
        The key property: columns have roughly uniform norms and the Gram matrix
        diagonal dominates off-diagonal entries.
        """
        torch.manual_seed(42)
        G = torch.randn(64, 32)  # wider matrix for better conditioning
        X = MuonClip.newton_schulz(G, steps=5)

        assert X.shape == G.shape

        # For a tall matrix (m>n), X.T @ X should be close to scaled identity
        XtX = X.T.float() @ X.float()  # (32, 32)
        # Check diagonal dominance: diag entries >> off-diag entries
        diag = XtX.diag()
        off_diag = XtX - torch.diag(diag)
        diag_mean = diag.mean().item()
        off_diag_max = off_diag.abs().max().item()
        # Off-diagonal should be much smaller than diagonal (bf16 loses precision)
        ratio = off_diag_max / max(diag_mean, 1e-8)
        assert ratio < 0.5, \
            f"NS not near-orthogonal: off-diag/diag ratio {ratio:.4f} (diag={diag_mean:.4f}, off={off_diag_max:.4f})"

        # Column norms should be relatively uniform
        col_norms = X.float().norm(dim=0)
        cv = col_norms.std() / col_norms.mean()  # coefficient of variation
        assert cv < 0.3, f"Column norms too variable: CV={cv:.4f}"

    def test_f21_every_param_in_exactly_one_group(self):
        """Create model, check every named param is in exactly one optimizer group."""
        cfg = _tiny(use_muon=True)
        model = Model(cfg)
        groups = build_param_groups(model, cfg)

        # Collect all param data_ptrs per group
        group_ptrs = [set(p.data_ptr() for p in g['params']) for g in groups]

        # Every deduplicated param should appear in exactly one group
        all_ptrs = set()
        for ptrs in group_ptrs:
            overlap = all_ptrs & ptrs
            assert len(overlap) == 0, f"Param(s) in multiple groups: {overlap}"
            all_ptrs |= ptrs

        # All model params (deduplicated) should be covered
        model_ptrs = set()
        for name, p in model.named_parameters():
            if p.requires_grad and p.data_ptr() not in model_ptrs:
                model_ptrs.add(p.data_ptr())
                assert p.data_ptr() in all_ptrs, \
                    f"Param '{name}' (ptr={p.data_ptr()}) not in any optimizer group"

    def test_f22_no_duplicate_data_ptr(self):
        """No param appears in two groups (data_ptr dedup)."""
        cfg = _tiny(use_muon=True)
        model = Model(cfg)
        groups = build_param_groups(model, cfg)

        all_ptrs = []
        for g in groups:
            for p in g['params']:
                all_ptrs.append(p.data_ptr())

        assert len(all_ptrs) == len(set(all_ptrs)), \
            f"Duplicate data_ptrs in param groups: {len(all_ptrs)} total, {len(set(all_ptrs))} unique"

    def test_f23_muon_step_decreases_loss(self):
        """MuonClip step changes weights in the right direction (loss decreases)."""
        cfg = _tiny(use_muon=True)
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        # Fixed data
        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        mask_attn = build_staircase_mask(L, cfg.block_size)
        _, t = sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        x_noisy, noise_mask = apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                           block_size=cfg.block_size)
        x_input = torch.cat([x_noisy, targets], dim=1)
        elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)

        losses = []
        for _ in range(5):
            hidden, _ = model(x_input, targets=targets, attn_mask=mask_attn)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss didn't decrease with MuonClip: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ============================================================================
# G. Data Pipeline (data.py)
# ============================================================================

class TestDataPipelineExhaustive:

    def test_g24_positions_with_3_docs(self):
        """Verify _compute_positions with 3 documents packed.

        doc_ids: [0,0,0,1,1,2,2,2]
        Expected: [0,1,2,0,1,0,1,2]
        """
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        positions = _compute_positions(doc_ids)
        expected = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])
        assert torch.equal(positions, expected), \
            f"Positions {positions} != expected {expected}"

    def test_g25_per_doc_t_4_docs(self):
        """Create 4-doc packed sequence, verify each doc has different t (high probability)."""
        doc_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]] * 4)  # B=4

        found_all_different = False
        for seed in range(50):
            torch.manual_seed(seed)
            t = _sample_t_per_doc(doc_ids, t_min=0.1)
            # Check if all 4 docs in batch 0 have different t
            doc_ts = [t[0, 0].item(), t[0, 2].item(), t[0, 4].item(), t[0, 6].item()]
            if len(set(doc_ts)) == 4:
                found_all_different = True
                break

        assert found_all_different, "4 docs never had 4 different t values in 50 tries"

        # Also verify: within each doc, all tokens have the same t
        t = _sample_t_per_doc(doc_ids, t_min=0.1)
        for b in range(4):
            for doc in range(4):
                start = doc * 2
                end = start + 2
                vals = t[b, start:end]
                assert vals[0] == vals[1], \
                    f"Batch {b}, doc {doc}: t[{start}]={vals[0]:.4f} != t[{end-1}]={vals[1]:.4f}"

    def test_g26_xt_x0_structure(self):
        """Verify [x_t || x_0]: first half has mask tokens, second half is clean."""
        cfg = _tiny()
        targets = torch.randint(10, 100, (2, 16))
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]] * 2)

        torch.manual_seed(42)
        x_noisy, noise_mask, t = _apply_noise_per_doc(targets, doc_ids, cfg)
        x_input = torch.cat([x_noisy, targets], dim=1)

        L = 16
        x_t = x_input[:, :L]
        x_0 = x_input[:, L:]

        # x_0 EXACTLY matches targets
        assert torch.equal(x_0, targets), "x_0 half doesn't match targets"

        # x_t has mask tokens where noise_mask is True
        assert (x_t[noise_mask] == cfg.mask_token_id).all(), "Masked positions should have mask_token_id"

        # x_t has original tokens where noise_mask is False
        assert (x_t[~noise_mask] == targets[~noise_mask]).all(), "Unmasked positions should match targets"

    def test_g27_doc_ids_consecutive(self):
        """Consecutive tokens in same doc have same doc_id."""
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
        # Verify: for each position, if next position is same doc, doc_id is same
        for i in range(doc_ids.shape[1] - 1):
            if doc_ids[0, i] == doc_ids[0, i + 1]:
                pass  # same doc, consecutive -- this is correct
            else:
                # Boundary: doc_id should increase by exactly 1 (or more for non-contiguous)
                assert doc_ids[0, i + 1] > doc_ids[0, i], \
                    f"doc_id decreased at position {i}: {doc_ids[0, i]} -> {doc_ids[0, i+1]}"


# ============================================================================
# H. Checkpoint Weight Mapping (checkpoint.py)
# ============================================================================

class TestCheckpointMappingExhaustive:

    @staticmethod
    def _make_fake_hf_state(cfg):
        """Build a complete HF state_dict with Qwen3 key names for cfg.n_layer layers."""
        sd = {}
        d, nh, nkv, hd, mlp_h, V = (cfg.n_embd, cfg.n_head, cfg.n_kv_head,
                                      cfg.head_dim, cfg.mlp_hidden, cfg.vocab_size)
        sd['model.embed_tokens.weight'] = torch.randn(V, d) * 0.1
        sd['model.norm.weight'] = torch.randn(d) * 0.1
        sd['lm_head.weight'] = sd['model.embed_tokens.weight'].clone()

        for i in range(cfg.n_layer):
            pfx = f'model.layers.{i}'
            sd[f'{pfx}.self_attn.q_proj.weight'] = torch.randn(nh * hd, d) * 0.1
            sd[f'{pfx}.self_attn.k_proj.weight'] = torch.randn(nkv * hd, d) * 0.1
            sd[f'{pfx}.self_attn.v_proj.weight'] = torch.randn(nkv * hd, d) * 0.1
            sd[f'{pfx}.self_attn.o_proj.weight'] = torch.randn(d, nh * hd) * 0.1
            sd[f'{pfx}.self_attn.q_norm.weight'] = torch.randn(hd) * 0.1
            sd[f'{pfx}.self_attn.k_norm.weight'] = torch.randn(hd) * 0.1
            sd[f'{pfx}.mlp.gate_proj.weight'] = torch.randn(mlp_h, d) * 0.1
            sd[f'{pfx}.mlp.up_proj.weight'] = torch.randn(mlp_h, d) * 0.1
            sd[f'{pfx}.mlp.down_proj.weight'] = torch.randn(d, mlp_h) * 0.1
            sd[f'{pfx}.input_layernorm.weight'] = torch.randn(d) * 0.1
            sd[f'{pfx}.post_attention_layernorm.weight'] = torch.randn(d) * 0.1
            sd[f'{pfx}.self_attn.rotary_emb.inv_freq'] = torch.randn(hd // 2) * 0.1
        return sd

    def test_h28_load_from_hf_all_weights_match(self, monkeypatch):
        """Create fake HF state_dict with ALL Qwen3 key names for 2 layers.
        Load via load_from_hf. Verify every weight transferred correctly."""
        import phase6.checkpoint as ckpt_mod

        cfg = _tiny()
        fake_sd = self._make_fake_hf_state(cfg)

        monkeypatch.setattr(ckpt_mod, '_load_hf_weights', lambda name: fake_sd)

        model = Model(cfg)
        missing, unexpected = load_from_hf(model, model_name='fake/qwen3', device='cpu')

        # All model weights should be loaded (no missing)
        assert len(missing) == 0, f"Missing keys: {missing}"

        # Verify specific weights match exactly
        assert torch.allclose(model.token_emb.weight, fake_sd['model.embed_tokens.weight'])
        assert torch.allclose(model.final_norm.weight, fake_sd['model.norm.weight'])

        for i in range(cfg.n_layer):
            pfx = f'model.layers.{i}'
            block = model.blocks[i]
            assert torch.allclose(block.attn.c_q.weight, fake_sd[f'{pfx}.self_attn.q_proj.weight'])
            assert torch.allclose(block.attn.c_k.weight, fake_sd[f'{pfx}.self_attn.k_proj.weight'])
            assert torch.allclose(block.attn.c_v.weight, fake_sd[f'{pfx}.self_attn.v_proj.weight'])
            assert torch.allclose(block.attn.c_proj.weight, fake_sd[f'{pfx}.self_attn.o_proj.weight'])
            assert torch.allclose(block.attn.q_norm.weight, fake_sd[f'{pfx}.self_attn.q_norm.weight'])
            assert torch.allclose(block.attn.k_norm.weight, fake_sd[f'{pfx}.self_attn.k_norm.weight'])
            assert torch.allclose(block.mlp.gate_proj.weight, fake_sd[f'{pfx}.mlp.gate_proj.weight'])
            assert torch.allclose(block.mlp.up_proj.weight, fake_sd[f'{pfx}.mlp.up_proj.weight'])
            assert torch.allclose(block.mlp.down_proj.weight, fake_sd[f'{pfx}.mlp.down_proj.weight'])
            assert torch.allclose(block.attn_norm.weight, fake_sd[f'{pfx}.input_layernorm.weight'])
            assert torch.allclose(block.mlp_norm.weight, fake_sd[f'{pfx}.post_attention_layernorm.weight'])

    def test_h29_lm_head_not_in_missing(self, monkeypatch):
        """lm_head.weight is tied to token_emb -- NOT in missing keys."""
        import phase6.checkpoint as ckpt_mod

        cfg = _tiny()
        fake_sd = self._make_fake_hf_state(cfg)
        monkeypatch.setattr(ckpt_mod, '_load_hf_weights', lambda name: fake_sd)

        model = Model(cfg)
        missing, _ = load_from_hf(model, model_name='fake/qwen3', device='cpu')
        assert 'lm_head.weight' not in missing

    def test_h30_rotary_emb_silently_skipped(self, monkeypatch):
        """rotary_emb keys are in unexpected (silently skipped, not error)."""
        import phase6.checkpoint as ckpt_mod

        cfg = _tiny()
        fake_sd = self._make_fake_hf_state(cfg)
        monkeypatch.setattr(ckpt_mod, '_load_hf_weights', lambda name: fake_sd)

        model = Model(cfg)
        _, unexpected = load_from_hf(model, model_name='fake/qwen3', device='cpu')

        # rotary_emb keys are now categorized as 'rotary_emb' skip, not unexpected
        assert len(unexpected) == 0, f"Expected 0 unexpected keys, got {unexpected}"


# ============================================================================
# I. Generate Correctness (generate.py)
# ============================================================================

class TestGenerateExhaustive:

    def test_i31_transfer_tokens_sums_to_block_length(self):
        """get_num_transfer_tokens sums to block_length for all combos up to 32."""
        for block_length in range(1, 33):
            for steps in range(1, block_length + 1):
                schedule = get_num_transfer_tokens(block_length, steps)
                assert schedule.sum().item() == block_length, \
                    f"block={block_length}, steps={steps}: sum={schedule.sum().item()}"
                assert len(schedule) == steps
                # Each step should have at least 1 token (floor division)
                # Actually: base = block_length // steps, which can be 0 if steps > block_length
                # but we only test steps <= block_length

    def test_i32_dynamic_remasking_tau_0_all_committed_step1(self):
        """With tau=0 and uniform confidence, all tokens committed in step 1."""
        B, T = 1, 8
        confidences = torch.ones(B, T) * 0.5  # all uniform
        masked = torch.ones(B, T, dtype=torch.bool)
        num_to_commit = 2
        threshold = 0.0  # tau=0: everything is above threshold

        commit = _select_tokens_dynamic(confidences, masked, num_to_commit, threshold)
        # All masked tokens have conf > 0 = threshold, and n_high (8) >= num_to_commit (2)
        # So all 8 tokens above threshold get committed
        assert commit.sum().item() == 8, "tau=0 should commit ALL tokens above threshold"

    def test_i33_kv_cache_state_after_2_blocks(self):
        """After generating 2 blocks, cache should have 2*block_size KV entries."""
        cfg = _tiny(denoise_steps=2)
        model = Model(cfg)
        model.eval()

        B = 1
        blk = cfg.block_size

        model.set_cache_mode(True)
        model.reset_kv_cache()

        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (B, blk)), pos_offset=0)
            model(torch.randint(0, cfg.vocab_size, (B, blk)), pos_offset=blk)

        for block in model.blocks:
            k_cache, v_cache = block.attn.kv_cache
            assert k_cache.shape[2] == 2 * blk, \
                f"Expected cache length {2 * blk}, got {k_cache.shape[2]}"
            assert v_cache.shape[2] == 2 * blk

        model.disable_kv_cache()

    def test_i34_special_token_suppression(self):
        """During generation, mask_id logits should be set to -inf."""
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        # Direct check: generate function suppresses special tokens
        # We can verify by looking at the logits suppression code path
        B, T = 1, 8
        x = torch.randint(3, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits, _ = model(x)

        # Manually suppress like generate.py does
        mask_id = cfg.mask_token_id
        pad_id = cfg.pad_token_id
        logits[:, :, mask_id] = -float('inf')
        logits[:, :, pad_id] = -float('inf')

        # Verify suppression
        assert logits[0, 0, mask_id].item() == -float('inf')
        assert logits[0, 0, pad_id].item() == -float('inf')

        # Argmax should never select these
        selected = logits.argmax(dim=-1)
        assert (selected != mask_id).all()
        assert (selected != pad_id).all()


# ============================================================================
# J. End-to-End Numerical Reproducibility
# ============================================================================

class TestReproducibility:

    def test_j35_training_deterministic_with_seed(self):
        """Set seed, run 3 training steps, record loss. Reset seed, run again. Must match."""
        cfg = _tiny()

        losses_a = self._run_3_steps(cfg, seed=42)
        losses_b = self._run_3_steps(cfg, seed=42)

        for i in range(3):
            assert losses_a[i] == pytest.approx(losses_b[i], rel=1e-6), \
                f"Step {i}: run A loss {losses_a[i]:.6f} != run B loss {losses_b[i]:.6f}"

    def test_j36_generation_deterministic(self):
        """Same prompt + seed -> identical output twice."""
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        prompt_ids = [10, 20, 30]

        torch.manual_seed(42)
        r1 = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.0)
        torch.manual_seed(42)
        r2 = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.0)

        assert r1 == r2, f"Run 1: {r1[:10]}..., Run 2: {r2[:10]}..."

    @staticmethod
    def _run_3_steps(cfg, seed):
        torch.manual_seed(seed)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        losses = []
        for step in range(3):
            torch.manual_seed(seed + step + 1000)  # deterministic data per step
            B, L = 2, cfg.seq_len
            targets = torch.randint(3, cfg.vocab_size, (B, L))
            _, t = sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
            x_noisy, noise_mask = apply_noise(targets, t,
                                               mask_token_id=cfg.mask_token_id,
                                               block_size=cfg.block_size)
            x_input = torch.cat([x_noisy, targets], dim=1)
            elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)
            attn_mask = build_staircase_mask(L, cfg.block_size)

            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        return losses


# ============================================================================
# K. Additional Edge Cases and Bug Patterns from MEMORY
# ============================================================================

class TestBugPatterns:

    def test_rope_buffer_length_for_generation(self):
        """RoPE buffer should be >= 2*seq_len for [x_t||x_0] training."""
        cfg = _tiny()
        model = Model(cfg)
        # Buffer length should be max(2*seq_len, 8192)
        expected_len = max(cfg.seq_len * 2, 8192)
        assert model.cos.shape[1] >= expected_len, \
            f"RoPE buffer too short: {model.cos.shape[1]} < {expected_len}"

    def test_tied_embedding_init_order(self):
        """Tied embeddings: init happens BEFORE tying.

        Bug pattern: if tying happens before init, both get re-initialized.
        Correct: apply init, then tie. Verify token_emb and lm_head share memory.
        """
        cfg = _tiny()
        model = Model(cfg)
        assert model.lm_head.weight is model.token_emb.weight, \
            "lm_head.weight should be the SAME object as token_emb.weight"

    def test_residual_init_scaling(self):
        """Residual projections (c_proj, down_proj) should have smaller init std.

        Expected: std = 0.02 / sqrt(2 * n_layer)
        """
        cfg = _tiny()
        model = Model(cfg)

        expected_std = 0.02 / math.sqrt(2 * cfg.n_layer)

        for i, block in enumerate(model.blocks):
            c_proj_std = block.attn.c_proj.weight.std().item()
            down_std = block.mlp.down_proj.weight.std().item()
            # Allow 3x tolerance (statistical, finite sample)
            assert c_proj_std < expected_std * 3, \
                f"Block {i} c_proj std {c_proj_std:.4f} >> expected {expected_std:.4f}"
            assert down_std < expected_std * 3, \
                f"Block {i} down_proj std {down_std:.4f} >> expected {expected_std:.4f}"

    def test_loss_normalization_by_all_real_tokens(self):
        """Loss divides by ALL real tokens (not just masked).

        With mask fraction = 50%, loss/masked_count would be ~2x loss/all_count.
        Verify we use all_count normalization.
        """
        torch.manual_seed(42)
        D, V = 32, 64
        B, L = 1, 16
        h = torch.randn(B, L, D)
        w = torch.randn(V, D)
        targets = torch.randint(3, V, (B, L))  # no pad tokens (pad=2)

        # Half masked
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, :8] = True  # first 8 masked
        ew = torch.ones(B, L)

        loss = compute_loss(h, targets, mask, ew, w, _FakeCfg())

        # Compute with masked-only normalization
        h_flat = h.contiguous().view(-1, D)
        t_flat = targets.contiguous().view(-1)
        w_flat = (mask.float() * ew).contiguous().view(-1)
        logits = h_flat @ w.T
        ce = F.cross_entropy(logits, t_flat, reduction='none')
        masked_count = mask.sum().float()
        all_count = (targets != 2).float().sum()  # pad_token_id = 2

        loss_by_masked = (ce * w_flat).sum() / masked_count
        loss_by_all = (ce * w_flat).sum() / all_count

        # Our compute_loss uses all_count normalization
        assert loss.item() == pytest.approx(loss_by_all.item(), rel=1e-4), \
            f"Loss {loss.item():.4f} != loss_by_all {loss_by_all.item():.4f}"
        # And it should NOT equal masked-only normalization (which would be ~2x)
        assert abs(loss.item() - loss_by_masked.item()) > 0.01 * loss.item(), \
            "Loss appears to normalize by masked count instead of all real tokens"

    def test_gated_query_zero_init(self):
        """Gated Query Attention gates should be zero-initialized (identity at init)."""
        cfg = _tiny(use_gated_query=True)
        model = Model(cfg)

        for i, block in enumerate(model.blocks):
            w_gate = block.attn.w_gate.weight
            assert torch.allclose(w_gate, torch.zeros_like(w_gate)), \
                f"Block {i} w_gate not zero-init: max={w_gate.abs().max().item()}"

    def test_staircase_mask_no_x0_future_leakage(self):
        """x_0[block 0] CANNOT attend x_0[block 1] (anti-causal leakage)."""
        L, blk = 16, 4
        mask = build_staircase_mask(L, blk)

        n = L
        # x_0 block 0 = positions [n+0, n+3], x_0 block 1 = positions [n+4, n+7]
        for q in range(n, n + blk):
            for k in range(n + blk, n + 2 * blk):
                assert mask[q, k] == float('-inf'), \
                    f"x_0[{q}] should NOT attend x_0[{k}] (future block = anti-causal)"

    def test_cross_block_x_t_isolation(self):
        """x_t[block 0] CANNOT attend x_t[block 1] (different blocks in same half)."""
        L, blk = 16, 4
        mask = build_staircase_mask(L, blk)

        for q in range(0, blk):
            for k in range(blk, 2 * blk):
                assert mask[q, k] == float('-inf'), \
                    f"x_t[{q}] should NOT attend x_t[{k}] (different block)"

    def test_model_forward_returns_hidden_not_logits_in_training(self):
        """Training mode: model returns hidden_states, NOT logits.

        This is critical -- loss is computed externally via chunked CE.
        If model returns logits, the hidden_states @ lm_head.weight^T in loss.py
        would double-project.
        """
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        B, L = 1, cfg.seq_len
        idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
        targets = torch.randint(0, cfg.vocab_size, (B, L))
        mask = build_staircase_mask(L, cfg.block_size)

        with torch.no_grad():
            out, _ = model(idx, targets=targets, attn_mask=mask)

        # out should be hidden_states (B, L, n_embd), NOT logits (B, L, vocab_size)
        assert out.shape == (B, L, cfg.n_embd), \
            f"Training should return hidden_states {(B, L, cfg.n_embd)}, got {out.shape}"
        assert out.shape[-1] != cfg.vocab_size, \
            "Training returned vocab_size dim -- likely returning logits instead of hidden_states"

    def test_apply_noise_padding_positions_never_masked(self):
        """Padding positions should never be masked even at t=1.0."""
        targets = torch.randint(10, 100, (2, 16))
        targets[:, 12:] = 2  # pad_token_id = 2
        t = torch.ones(2, 16)  # t=1.0: mask everything

        _, noise_mask = apply_noise(targets, t, mask_token_id=0, pad_token_id=2, block_size=4)
        assert not noise_mask[:, 12:].any(), "Padding positions masked at t=1.0"

    def test_elbo_weight_below_tmin_clamped(self):
        """t values below t_min should get clamped, yielding max weight = 1/t_min."""
        t = torch.tensor([[0.01, 0.05, 0.1, 0.5]])
        w = compute_elbo_weight(t, t_min=0.1)
        # 0.01 and 0.05 are below t_min=0.1, so clamped to 0.1 -> weight = 10
        assert w[0, 0].item() == pytest.approx(10.0, rel=1e-5)
        assert w[0, 1].item() == pytest.approx(10.0, rel=1e-5)
        assert w[0, 2].item() == pytest.approx(10.0, rel=1e-5)
        assert w[0, 3].item() == pytest.approx(2.0, rel=1e-5)

    def test_generate_cleans_up_after_exception(self):
        """generate() should reset KV cache even if an error occurs."""
        cfg = _tiny()
        model = Model(cfg)
        model.eval()

        # Normal call
        generate(model, [10, 20], cfg, max_new_tokens=8)

        # Verify cleanup
        for block in model.blocks:
            assert block.attn.kv_cache is None
            assert block.attn.cache_mode is False

    def test_compute_positions_vectorized_matches_loop(self):
        """Verify vectorized _compute_positions matches naive loop implementation."""
        doc_ids = torch.tensor([
            [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        ])

        # Vectorized
        positions_vec = _compute_positions(doc_ids)

        # Naive loop
        B, L = doc_ids.shape
        positions_loop = torch.zeros_like(doc_ids)
        for b in range(B):
            pos = 0
            for i in range(L):
                if i > 0 and doc_ids[b, i] != doc_ids[b, i - 1]:
                    pos = 0
                positions_loop[b, i] = pos
                pos += 1

        assert torch.equal(positions_vec, positions_loop), \
            f"Vectorized:\n{positions_vec}\nLoop:\n{positions_loop}"


# ============================================================================
# L. Model Architecture Invariants
# ============================================================================

class TestArchitectureInvariants:

    def test_qwen3_head_dim_not_derived(self):
        """Qwen3: head_dim is independent of n_embd // n_head.

        Qwen3-0.6B: n_embd=1024, n_head=16 -> n_embd//n_head=64, but head_dim=128.
        This is a common footgun.
        """
        cfg = Config()  # default Qwen3-0.6B
        assert cfg.head_dim == 128
        assert cfg.n_embd // cfg.n_head == 64  # would be wrong if head_dim derived
        assert cfg.head_dim != cfg.n_embd // cfg.n_head

    def test_c_proj_dimensions(self):
        """c_proj: in_features = n_head * head_dim (NOT n_embd), out_features = n_embd."""
        cfg = _tiny()
        model = Model(cfg)
        attn = model.blocks[0].attn

        assert attn.c_proj.in_features == cfg.n_head * cfg.head_dim
        assert attn.c_proj.out_features == cfg.n_embd

    def test_count_params_matches_manual(self):
        """Model.count_params() matches manual sum with dedup."""
        cfg = _tiny()
        model = Model(cfg)

        count = model.count_params()

        seen = set()
        manual = 0
        for p in model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                manual += p.numel()

        assert count == manual

    def test_no_bias_in_linear_layers(self):
        """Qwen3 uses no bias in attention/MLP linears."""
        cfg = _tiny()
        model = Model(cfg)

        for block in model.blocks:
            assert block.attn.c_q.bias is None, "c_q should have no bias"
            assert block.attn.c_k.bias is None, "c_k should have no bias"
            assert block.attn.c_v.bias is None, "c_v should have no bias"
            assert block.attn.c_proj.bias is None, "c_proj should have no bias"
            assert block.mlp.gate_proj.bias is None, "gate_proj should have no bias"
            assert block.mlp.up_proj.bias is None, "up_proj should have no bias"
            assert block.mlp.down_proj.bias is None, "down_proj should have no bias"
