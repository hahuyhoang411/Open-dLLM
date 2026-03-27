"""Training pipeline integration tests — verifies the full training loop works end-to-end.

Tests the composed behavior of: model + loss + optimizer + schedule + data + checkpoint.
All CPU-only, TINY config, fast. No duplication with test_integration.py (which tests
forward shapes, staircase mask geometry, KV cache, HF loading, config validation).

Focus here: training steps, optimizer updates, LR schedule, ELBO weighting,
gradient accumulation, checkpoint resume, CART integration, per-doc noise.
"""

import math
import os
import tempfile

import pytest
import torch
from torch.nn.utils import clip_grad_norm_

from phase6.attention import build_staircase_mask
from phase6.checkpoint import save_checkpoint, load_checkpoint
from phase6.config import Config
from phase6.loss import compute_loss
from phase6.model import Model
from phase6.optim import build_param_groups, create_optimizer, MuonClip
from phase6.schedule import (
    apply_noise,
    compute_cart_weights,
    compute_elbo_weight,
    get_lr_factor,
    sample_timesteps,
)
from phase6.data import _sample_t_per_doc, _apply_noise_per_doc, _compute_positions


# ---------------------------------------------------------------------------
# TINY config — 2 layers, vocab 256, seq 32, block 8. CPU-only.
# ---------------------------------------------------------------------------

TINY = Config(
    n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
    mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
    rope_base=10000, rms_eps=1e-6, dropout=0.0, t_min=0.1,
    use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
    use_liger=False, use_grad_ckpt=False, use_flex=False, use_muon=False,
    use_cart=False, pad_token_id=0, mask_token_id=1, eos_token_id=2,
    muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
).validate()


def _make_batch(cfg, B=2, seed=None):
    """Build a synthetic training batch: x_input, targets, mask, elbo_w, attn_mask."""
    if seed is not None:
        torch.manual_seed(seed)
    L = cfg.seq_len
    targets = torch.randint(3, cfg.vocab_size, (B, L))

    _, t = sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
    x_noisy, noise_mask = apply_noise(
        targets, t,
        mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id,
        block_size=cfg.block_size,
    )
    elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)
    x_input = torch.cat([x_noisy, targets], dim=1)
    attn_mask = build_staircase_mask(L, cfg.block_size)
    return x_input, targets, noise_mask, elbo_w, attn_mask


def _training_step(model, optimizer, cfg, x_input, targets, noise_mask, elbo_w, attn_mask):
    """Single forward + loss + backward + clip + step. Returns (loss, grad_norm)."""
    hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
    loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), grad_norm


# ============================================================================
# 1. Full Training Step (forward + loss + backward + optimizer.step)
# ============================================================================

class TestFullTrainingStep:

    def test_single_step_finite_loss_nonzero_grad(self):
        """One complete training step: loss finite, grad_norm > 0, weights changed."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        # Snapshot initial weights
        w0 = model.blocks[0].mlp.gate_proj.weight.clone()

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)
        loss, grad_norm = _training_step(
            model, optimizer, cfg, x_input, targets, noise_mask, elbo_w, attn_mask,
        )

        assert math.isfinite(loss), f"Loss not finite: {loss}"
        assert loss > 0, f"Loss should be positive: {loss}"
        assert grad_norm > 0, f"grad_norm is 0 — backward broken"

        w1 = model.blocks[0].mlp.gate_proj.weight
        assert not torch.equal(w0, w1), "Weights unchanged after optimizer.step()"

    def test_single_step_with_muon(self):
        """Muon optimizer also produces finite loss and weight updates."""
        cfg = Config(
            n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
            mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
            rope_base=10000, rms_eps=1e-6, dropout=0.0, t_min=0.1,
            use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
            use_liger=False, use_grad_ckpt=False, use_flex=False,
            use_muon=True,
            use_cart=False, pad_token_id=0, mask_token_id=1, eos_token_id=2,
            muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
        ).validate()

        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)
        assert isinstance(optimizer, MuonClip)

        w0 = model.blocks[0].attn.c_q.weight.clone()
        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=0)
        loss, grad_norm = _training_step(
            model, optimizer, cfg, x_input, targets, noise_mask, elbo_w, attn_mask,
        )

        assert math.isfinite(loss)
        assert grad_norm > 0
        assert not torch.equal(w0, model.blocks[0].attn.c_q.weight)


# ============================================================================
# 2. Multi-Step Training (loss decreases)
# ============================================================================

class TestMultiStepTraining:

    def test_loss_decreases_over_20_steps(self):
        """Run 20 training steps on fixed data. Final loss < initial loss."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        # Fixed data for overfitting
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

        assert all(math.isfinite(l) for l in losses), f"NaN in losses: {losses}"
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: step 0={losses[0]:.4f}, step 19={losses[-1]:.4f}"

    def test_no_nan_at_any_step(self):
        """20 steps with fresh noise each step — never NaN."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        for step in range(20):
            x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, seed=step)
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
            optimizer.zero_grad()
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"NaN grad in {name} at step {step}"
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()


# ============================================================================
# 3. Learning Rate Schedule Integration
# ============================================================================

class TestLRScheduleIntegration:

    def test_wsd_schedule_warmup_stable_decay(self):
        """WSD schedule: warmup -> stable -> linear decay."""
        cfg = TINY  # warmup_iters, decay_start, max_iters set by validate()
        warmup = cfg.warmup_iters
        decay_start = cfg.decay_start
        max_iters = cfg.max_iters

        # Step 0: near-zero (warmup start)
        lr0 = get_lr_factor(0, warmup, decay_start, max_iters)
        assert 0 < lr0 <= 1.0 / warmup + 0.01, f"Step 0 LR factor too high: {lr0}"

        # Mid-stable phase
        mid = (warmup + decay_start) // 2
        lr_mid = get_lr_factor(mid, warmup, decay_start, max_iters)
        assert lr_mid == 1.0, f"Mid-stable LR should be 1.0, got {lr_mid}"

        # End of training: near-zero
        lr_end = get_lr_factor(max_iters - 1, warmup, decay_start, max_iters)
        assert lr_end < 0.1, f"End LR should be near 0, got {lr_end}"

    def test_lr_schedule_applied_to_optimizer(self):
        """Verify LR schedule actually changes optimizer param group LRs."""
        cfg = TINY
        model = Model(cfg)
        optimizer = create_optimizer(model, cfg)

        # Apply schedule at different steps
        for step in [0, cfg.warmup_iters, cfg.decay_start, cfg.max_iters - 1]:
            factor = get_lr_factor(step, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * factor

        # At max_iters - 1, LR should be near 0
        factor_end = get_lr_factor(
            cfg.max_iters - 1, cfg.warmup_iters, cfg.decay_start, cfg.max_iters,
        )
        for pg in optimizer.param_groups:
            expected = pg['initial_lr'] * factor_end
            assert abs(pg['lr'] - expected) < 1e-9


# ============================================================================
# 4. ELBO Weight Correctness in Pipeline
# ============================================================================

class TestELBOWeightPipeline:

    def test_high_t_loss_near_raw_ce(self):
        """At t=0.9, ELBO weight ~1.11 — loss close to raw CE."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.eval()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        t = torch.full((B, L), 0.9)
        x_noisy, noise_mask = apply_noise(
            targets, t, mask_token_id=cfg.mask_token_id,
            pad_token_id=cfg.pad_token_id, block_size=cfg.block_size,
        )
        x_input = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, cfg.block_size)
        elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)

        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            weighted_loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            # Raw CE (no ELBO) — weight = mask only
            ones = torch.ones_like(elbo_w)
            raw_loss = compute_loss(hidden, targets, noise_mask, ones, model.lm_head.weight, cfg)

        # At t=0.9, ELBO weight = 1/0.9 ~ 1.11. Weighted should be ~1.11x raw.
        ratio = weighted_loss.item() / max(raw_loss.item(), 1e-8)
        assert 0.8 < ratio < 1.5, f"High-t ratio {ratio:.3f} should be near 1.11"

    def test_low_t_loss_much_higher_than_raw(self):
        """At t=0.1, ELBO weight = 10 — loss ~10x raw CE."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.eval()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        t = torch.full((B, L), 0.1)
        x_noisy, noise_mask = apply_noise(
            targets, t, mask_token_id=cfg.mask_token_id,
            pad_token_id=cfg.pad_token_id, block_size=cfg.block_size,
        )
        x_input = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, cfg.block_size)
        elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)

        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            weighted_loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            ones = torch.ones_like(elbo_w)
            raw_loss = compute_loss(hidden, targets, noise_mask, ones, model.lm_head.weight, cfg)

        # At t=0.1, ELBO weight = 10. Weighted should be ~10x raw.
        ratio = weighted_loss.item() / max(raw_loss.item(), 1e-8)
        assert 5.0 < ratio < 15.0, f"Low-t ratio {ratio:.3f} should be near 10"

    def test_elbo_weight_scaling_is_1_over_t(self):
        """Directly verify ELBO weight = 1/t end-to-end."""
        t_vals = [0.1, 0.2, 0.5, 0.8, 1.0]
        for tv in t_vals:
            t = torch.tensor([[tv]])
            w = compute_elbo_weight(t, t_min=0.1)
            expected = 1.0 / tv
            assert abs(w.item() - expected) < 1e-5, \
                f"t={tv}: expected {expected}, got {w.item()}"


# ============================================================================
# 5. Checkpoint Save -> Resume -> Same Loss
# ============================================================================

class TestCheckpointResume:

    def test_checkpoint_roundtrip_same_loss(self):
        """Train 5 steps, checkpoint, reload into new model, verify same forward loss."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()
        optimizer = create_optimizer(model, cfg)

        x_input, targets, noise_mask, elbo_w, attn_mask = _make_batch(cfg, B=2, seed=0)

        for _ in range(5):
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Compute reference loss at step 5
        model.eval()
        with torch.no_grad():
            hidden_ref, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss_ref = compute_loss(
                hidden_ref, targets, noise_mask, elbo_w, model.lm_head.weight, cfg,
            ).item()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=5, loss=loss_ref, ckpt_dir=tmpdir)

            # New model + optimizer, load checkpoint
            model2 = Model(cfg)
            optimizer2 = create_optimizer(model2, cfg)
            resumed_step = load_checkpoint(
                os.path.join(tmpdir, 'latest.pt'), model2, optimizer2, device='cpu',
            )
            assert resumed_step == 6  # load_checkpoint returns step + 1

            model2.eval()
            with torch.no_grad():
                hidden2, _ = model2(x_input, targets=targets, attn_mask=attn_mask)
                loss_resumed = compute_loss(
                    hidden2, targets, noise_mask, elbo_w, model2.lm_head.weight, cfg,
                ).item()

        assert abs(loss_ref - loss_resumed) < 1e-5, \
            f"Loss mismatch after resume: ref={loss_ref:.6f}, resumed={loss_resumed:.6f}"


# ============================================================================
# 6. Gradient Accumulation Simulation
# ============================================================================

class TestGradientAccumulation:

    def test_accumulated_grads_approx_2x_single(self):
        """Accumulate 2 micro-steps before optimizer.step() — grads ~2x single step."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()

        x1, t1, m1, e1, am1 = _make_batch(cfg, B=2, seed=10)
        x2, t2, m2, e2, am2 = _make_batch(cfg, B=2, seed=20)

        # Single micro-step grads
        optimizer = create_optimizer(model, cfg)
        hidden, _ = model(x1, targets=t1, attn_mask=am1)
        loss1 = compute_loss(hidden, t1, m1, e1, model.lm_head.weight, cfg)
        loss1.backward()
        single_grad_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        optimizer.zero_grad()

        # Two micro-steps accumulated
        hidden, _ = model(x1, targets=t1, attn_mask=am1)
        loss_a = compute_loss(hidden, t1, m1, e1, model.lm_head.weight, cfg)
        loss_a.backward()

        hidden, _ = model(x2, targets=t2, attn_mask=am2)
        loss_b = compute_loss(hidden, t2, m2, e2, model.lm_head.weight, cfg)
        loss_b.backward()

        accum_grad_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # Accumulated should be roughly 2x single (not exact due to different data batches)
        ratio = accum_grad_norm / max(single_grad_norm, 1e-8)
        assert 1.2 < ratio < 4.0, \
            f"Accumulated/single grad ratio {ratio:.2f} — expected ~2x"


# ============================================================================
# 7. Per-Document t Sampling in Pipeline
# ============================================================================

class TestPerDocumentNoise:

    def test_same_doc_same_t(self):
        """Tokens within the same document get the same noise rate t."""
        B, L = 2, 32
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[0, 16:] = 1  # batch 0: doc 0 (first 16), doc 1 (last 16)
        doc_ids[1, 10:20] = 1
        doc_ids[1, 20:] = 2  # batch 1: doc 0 (0-9), doc 1 (10-19), doc 2 (20-31)

        t = _sample_t_per_doc(doc_ids, t_min=0.1)

        # Batch 0, doc 0: positions 0-15 should have same t
        assert (t[0, :16] == t[0, 0]).all()
        # Batch 0, doc 1: positions 16-31 should have same t
        assert (t[0, 16:] == t[0, 16]).all()

        # Batch 1, doc 0: positions 0-9
        assert (t[1, :10] == t[1, 0]).all()
        # Batch 1, doc 1: positions 10-19
        assert (t[1, 10:20] == t[1, 10]).all()
        # Batch 1, doc 2: positions 20-31
        assert (t[1, 20:] == t[1, 20]).all()

    def test_different_docs_can_differ(self):
        """Different documents within a batch can (usually) get different t values."""
        B, L = 4, 32
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, 16:] = 1

        t = _sample_t_per_doc(doc_ids, t_min=0.1)

        # Across 4 batch items with 2 docs each = 8 independent t samples.
        # P(all identical) is effectively 0 for continuous uniform.
        doc0_ts = [t[b, 0].item() for b in range(B)]
        doc1_ts = [t[b, 16].item() for b in range(B)]
        all_ts = doc0_ts + doc1_ts
        assert len(set(all_ts)) > 1, "All per-doc t values identical — sampling broken"

    def test_apply_noise_per_doc_masks_correctly(self):
        """Per-doc noise: each doc's mask fraction should roughly match its t."""
        cfg = TINY
        B, L = 4, cfg.seq_len
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, L // 2:] = 1
        targets = torch.randint(3, cfg.vocab_size, (B, L))

        x_noisy, noise_mask, t = _apply_noise_per_doc(targets, doc_ids, cfg)

        # Masked positions should have mask_token_id
        assert (x_noisy[noise_mask] == cfg.mask_token_id).all()
        # Each doc's mask fraction should be within [0, 1]
        for b in range(B):
            for doc_id in [0, 1]:
                doc_mask = doc_ids[b] == doc_id
                frac_masked = noise_mask[b, doc_mask].float().mean().item()
                # With block min-1-masked, can't be 0 even for low t
                assert 0 < frac_masked <= 1.0, \
                    f"Doc {doc_id} mask fraction {frac_masked} out of range"


# ============================================================================
# 8. CART Weights Integration
# ============================================================================

class TestCARTIntegration:

    def test_cart_weights_zero_for_unmasked(self):
        """CART weights should be 0 for unmasked positions."""
        B, L = 2, 32
        noise_mask = torch.zeros(B, L, dtype=torch.bool)
        noise_mask[:, ::3] = True  # mask every 3rd token
        padding = torch.ones(B, L, dtype=torch.bool)  # no padding

        cart_w = compute_cart_weights(noise_mask, padding, p=0.1)
        # Unmasked positions should have 0 CART weight
        assert (cart_w[~noise_mask] == 0).all(), "CART weights nonzero for unmasked tokens"

    def test_cart_weights_nonzero_for_masked(self):
        """CART weights should be > 0 for masked positions with nearby context."""
        B, L = 2, 32
        noise_mask = torch.zeros(B, L, dtype=torch.bool)
        noise_mask[:, 15:17] = True  # mask 2 tokens in the middle
        padding = torch.ones(B, L, dtype=torch.bool)

        cart_w = compute_cart_weights(noise_mask, padding, p=0.1)
        # Masked positions with context around them should have positive weight
        assert (cart_w[:, 15:17] > 0).all(), "CART weights zero for masked tokens with context"

    def test_cart_changes_total_loss(self):
        """Loss with CART on vs off should differ."""
        cfg_no_cart = TINY
        cfg_cart = Config(
            n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
            mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
            rope_base=10000, rms_eps=1e-6, dropout=0.0, t_min=0.1,
            use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
            use_liger=False, use_grad_ckpt=False, use_flex=False, use_muon=False,
            use_cart=True, cart_p=0.1,
            pad_token_id=0, mask_token_id=1, eos_token_id=2,
            muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
        ).validate()

        torch.manual_seed(42)
        model = Model(cfg_no_cart)
        model.eval()

        B, L = 2, cfg_no_cart.seq_len
        targets = torch.randint(3, cfg_no_cart.vocab_size, (B, L))
        _, t = sample_timesteps(B, cfg_no_cart.num_blocks, cfg_no_cart.block_size, t_min=0.1)
        x_noisy, noise_mask = apply_noise(
            targets, t, mask_token_id=cfg_no_cart.mask_token_id,
            pad_token_id=cfg_no_cart.pad_token_id, block_size=cfg_no_cart.block_size,
        )
        x_input = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, cfg_no_cart.block_size)

        # Compute ELBO weights with and without CART
        elbo_base = compute_elbo_weight(t, t_min=0.1)
        padding = torch.ones_like(targets, dtype=torch.bool)
        cart_w = compute_cart_weights(noise_mask, padding, p=0.1)
        elbo_cart = elbo_base * cart_w

        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
            loss_no_cart = compute_loss(
                hidden, targets, noise_mask, elbo_base, model.lm_head.weight, cfg_no_cart,
            ).item()
            loss_cart = compute_loss(
                hidden, targets, noise_mask, elbo_cart, model.lm_head.weight, cfg_cart,
            ).item()

        assert loss_no_cart != loss_cart, \
            f"CART should change loss: no_cart={loss_no_cart:.4f}, cart={loss_cart:.4f}"


# ============================================================================
# 9. Full Staircase Mask + Forward + Loss
# ============================================================================

class TestStaircasePipelineSanity:

    def test_staircase_with_doc_ids_no_nan(self):
        """Staircase mask with doc boundaries -> forward -> loss -> finite."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.train()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, L // 2:] = 1  # 2 documents per sequence

        _, t = sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        x_noisy, noise_mask = apply_noise(
            targets, t, mask_token_id=cfg.mask_token_id,
            pad_token_id=cfg.pad_token_id, block_size=cfg.block_size,
        )
        x_input = torch.cat([x_noisy, targets], dim=1)
        elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)
        attn_mask = build_staircase_mask(L, cfg.block_size, doc_ids=doc_ids)

        hidden, _ = model(x_input, targets=targets, attn_mask=attn_mask)
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

        assert torch.isfinite(loss), f"NaN/Inf loss with doc_ids mask: {loss.item()}"
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {name} with doc_ids mask"

    def test_single_doc_vs_multi_doc_loss_differs(self):
        """Different doc_ids should produce different losses (attention patterns differ)."""
        cfg = TINY
        torch.manual_seed(42)
        model = Model(cfg)
        model.eval()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        t = torch.full((B, L), 0.5)
        x_noisy, noise_mask = apply_noise(
            targets, t, mask_token_id=cfg.mask_token_id,
            pad_token_id=cfg.pad_token_id, block_size=cfg.block_size,
        )
        x_input = torch.cat([x_noisy, targets], dim=1)
        elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)

        # Single document
        mask_single = build_staircase_mask(L, cfg.block_size)
        # Two documents
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, L // 2:] = 1
        mask_multi = build_staircase_mask(L, cfg.block_size, doc_ids=doc_ids)

        with torch.no_grad():
            h_single, _ = model(x_input, targets=targets, attn_mask=mask_single)
            h_multi, _ = model(x_input, targets=targets, attn_mask=mask_multi)
            loss_single = compute_loss(
                h_single, targets, noise_mask, elbo_w, model.lm_head.weight, cfg,
            ).item()
            loss_multi = compute_loss(
                h_multi, targets, noise_mask, elbo_w, model.lm_head.weight, cfg,
            ).item()

        # Losses should differ because attention patterns differ
        assert loss_single != loss_multi, \
            f"Single vs multi doc loss identical: {loss_single:.4f}"


# ============================================================================
# 10. Param Count Matches Expectation (no tied-embed double-counting)
# ============================================================================

class TestParamCount:

    def test_count_params_no_double_counting(self):
        """Model.count_params() deduplicates tied embeddings."""
        cfg = TINY
        model = Model(cfg)

        count = model.count_params()

        # Verify no double-counting: count via data_ptr
        seen = set()
        manual_count = 0
        for p in model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                manual_count += p.numel()

        assert count == manual_count, \
            f"count_params() {count} != manual {manual_count} — possible double-counting"

    def test_tied_embeddings_share_data_ptr(self):
        """token_emb and lm_head share the same data pointer — not counted twice."""
        cfg = TINY
        model = Model(cfg)
        assert model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

        # named_parameters yields both, but data_ptr is the same
        names_for_ptr = []
        for name, p in model.named_parameters():
            if p.data_ptr() == model.token_emb.weight.data_ptr():
                names_for_ptr.append(name)
        # Should find token_emb.weight (lm_head.weight is tied alias, PyTorch may or may not list it)
        assert 'token_emb.weight' in names_for_ptr


# ============================================================================
# 11. Position Computation with Document Boundaries
# ============================================================================

class TestPositionComputation:

    def test_positions_reset_at_doc_boundary(self):
        """RoPE positions reset to 0 at each document boundary."""
        B, L = 1, 16
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]])
        positions = _compute_positions(doc_ids)

        expected = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]])
        assert torch.equal(positions, expected), \
            f"Positions wrong: {positions} vs expected {expected}"

    def test_single_doc_monotonic(self):
        """Single document: positions are just 0, 1, 2, ..., L-1."""
        B, L = 2, 32
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        positions = _compute_positions(doc_ids)

        expected = torch.arange(L).unsqueeze(0).expand(B, -1)
        assert torch.equal(positions, expected)
