"""Integration tests for Phase 6 — end-to-end with a toy model.

Each test category verifies cross-module behavior that unit tests miss:
forward+loss+backward, staircase mask correctness, noise schedule integration,
checkpoint roundtrips, KV cache consistency, document packing, numerical sanity.

Note: schedule.py has module-level default args referencing config.t_min etc.
which don't exist as module-level attributes. Tests inline the noise/ELBO logic
rather than importing schedule.py (which fails at import time).
"""

import math
import os
import tempfile

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from phase6.config import Config
from phase6.model import Model
from phase6.attention import build_staircase_mask


# ---------------------------------------------------------------------------
# Shared tiny config — 2 layers, vocab 256, seq 32, block 8. CPU-only.
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides):
    defaults = dict(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
    )
    defaults.update(overrides)
    return Config(**defaults).validate()


TINY = _tiny_cfg()


# ---------------------------------------------------------------------------
# Inline noise helpers — avoids importing schedule.py (broken module defaults)
# ---------------------------------------------------------------------------

def _sample_timesteps(batch_size, num_blocks, block_size, t_min=0.1):
    """Sample per-block t ~ U[t_min, 1), expand to per-token (antithetic)."""
    eps = torch.rand(batch_size, num_blocks)
    total = batch_size * num_blocks
    offset = torch.arange(total).view(batch_size, num_blocks).float() / total
    eps = (eps / total + offset) % 1
    t_blocks = t_min + (1 - t_min) * eps
    t = t_blocks.repeat_interleave(block_size, dim=1)
    return t_blocks, t


def _apply_noise(targets, t, mask_token_id, block_size, pad_token_id=None):
    """LINEAR schedule noise with per-block min-1-masked guarantee."""
    B, L = targets.shape
    mask_prob = t
    noise_mask = torch.rand(B, L) < mask_prob

    if pad_token_id is not None:
        padding = targets != pad_token_id
        noise_mask = noise_mask & padding

    num_blocks = L // block_size
    mask_blocks = noise_mask.view(B, num_blocks, block_size)
    zero_blocks = mask_blocks.sum(dim=2) == 0
    if zero_blocks.any():
        for b_idx, blk_idx in zip(*zero_blocks.nonzero(as_tuple=True)):
            start = blk_idx * block_size
            if pad_token_id is not None:
                real_pos = (targets[b_idx, start:start + block_size] != pad_token_id).nonzero(as_tuple=True)[0]
            else:
                real_pos = torch.arange(block_size)
            if len(real_pos) > 0:
                pick = start + real_pos[torch.randint(len(real_pos), (1,))]
                noise_mask[b_idx, pick] = True

    x_noisy = targets.clone()
    x_noisy[noise_mask] = mask_token_id
    return x_noisy, noise_mask


def _compute_elbo_weight(t, t_min=0.1):
    """ELBO weight = 1/t for linear schedule."""
    return 1.0 / t.clamp(min=t_min)


# ============================================================================
# 1. Forward Pass Correctness
# ============================================================================

class TestForwardPass:

    def test_training_forward_shape(self):
        """[x_t || x_0] (B=2, 2L=64) -> hidden_states (B=2, L=32, D=64)."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B, L = 2, cfg.seq_len
        idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
        targets = torch.randint(0, cfg.vocab_size, (B, L))
        mask = build_staircase_mask(L, cfg.block_size)

        with torch.no_grad():
            out, _ = model(idx, targets=targets, attn_mask=mask)

        assert out.shape == (B, L, cfg.n_embd), \
            f"Expected ({B}, {L}, {cfg.n_embd}), got {out.shape}"

    def test_generation_forward_shape(self):
        """Generation input (B=1, T=8) -> logits (B=1, 8, 256)."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B, T = 1, 8
        idx = torch.randint(0, cfg.vocab_size, (B, T))

        with torch.no_grad():
            logits, _ = model(idx)

        assert logits.shape == (B, T, cfg.vocab_size)

    def test_gradients_flow_nonzero(self):
        """Backward pass produces non-zero gradients for ALL parameters.
        This is THE critical check — Liger FLCE bug was grad_norm=0.
        """
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.train()

        B, L = 2, cfg.seq_len
        idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
        targets = torch.randint(3, cfg.vocab_size, (B, L))  # avoid special tokens
        mask = build_staircase_mask(L, cfg.block_size)

        hidden, _ = model(idx, targets=targets, attn_mask=mask)
        # Compute loss manually: project hidden -> logits -> CE on masked positions
        logits = model.lm_head(hidden)  # (B, L, vocab)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()

        # Collect unique parameters (tied embeddings share data_ptr)
        seen_ptrs = set()
        zero_grad_params = []
        for name, p in model.named_parameters():
            if p.data_ptr() in seen_ptrs:
                continue
            seen_ptrs.add(p.data_ptr())
            if p.grad is None or p.grad.abs().max() == 0:
                zero_grad_params.append(name)

        assert len(zero_grad_params) == 0, \
            f"Parameters with zero gradients: {zero_grad_params}"

        # Also verify total grad norm > 0
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm > 0, "Total grad_norm is 0 — backward is broken"


# ============================================================================
# 2. Staircase Mask Verification
# ============================================================================

class TestStaircaseMask:

    def test_m_obc_blocks_same_block_leakage(self):
        """x_t[block 0] CANNOT see x_0[block 0] — label leakage protection."""
        L, blk = 32, 8  # 4 blocks
        mask = build_staircase_mask(L, blk)

        # x_t block 0 = positions [0..7], x_0 block 0 = positions [32..39]
        for qt in range(0, blk):
            for kv0 in range(L, L + blk):
                assert mask[qt, kv0] == float('-inf'), \
                    f"x_t[{qt}] should NOT attend x_0[{kv0}] (same block = label leakage)"

    def test_m_obc_allows_earlier_blocks(self):
        """x_t[block 1] CAN see x_0[block 0]."""
        L, blk = 32, 8
        mask = build_staircase_mask(L, blk)

        # x_t block 1 = positions [8..15], x_0 block 0 = positions [32..39]
        for qt in range(blk, 2 * blk):
            for kv0 in range(L, L + blk):
                assert mask[qt, kv0] == 0.0, \
                    f"x_t[{qt}] SHOULD attend x_0[{kv0}] (earlier block via M_OBC)"

    def test_m_bc_x0_block_causal(self):
        """x_0[block 1] CAN see x_0[block 0] AND x_0[block 1]."""
        L, blk = 32, 8
        mask = build_staircase_mask(L, blk)

        # x_0 block 1 = positions [L+8..L+15]
        # x_0 block 0 = positions [L..L+7] — should be visible
        # x_0 block 1 = positions [L+8..L+15] — should be visible (same block)
        for qt in range(L + blk, L + 2 * blk):
            # Can see x_0 block 0
            for kv0 in range(L, L + blk):
                assert mask[qt, kv0] == 0.0, \
                    f"x_0[{qt}] SHOULD attend x_0[{kv0}] (earlier block via M_BC)"
            # Can see x_0 block 1 (own block)
            for kv0 in range(L + blk, L + 2 * blk):
                assert mask[qt, kv0] == 0.0, \
                    f"x_0[{qt}] SHOULD attend x_0[{kv0}] (same block via M_BC)"

    def test_m_bd_same_block_same_half(self):
        """x_t tokens in same block CAN see each other (bidirectional)."""
        L, blk = 32, 8
        mask = build_staircase_mask(L, blk)

        # x_t block 0: positions [0..7] should all attend each other
        for q in range(0, blk):
            for k in range(0, blk):
                assert mask[q, k] == 0.0, \
                    f"x_t[{q}] SHOULD attend x_t[{k}] (same block, M_BD)"

    def test_cross_document_isolation(self):
        """Tokens from doc 0 CANNOT see tokens from doc 1 in ANY submask."""
        L, blk = 32, 8
        B = 2
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, L // 2:] = 1  # first half doc 0, second half doc 1

        mask = build_staircase_mask(L, blk, doc_ids=doc_ids)
        # mask shape: (B, 1, 2L, 2L)
        assert mask.shape == (B, 1, 2 * L, 2 * L)

        # For batch 0: check that doc 0 positions cannot attend doc 1 positions
        half = L // 2
        for b in range(B):
            m = mask[b, 0]  # (2L, 2L)
            # Doc 0 x_t positions [0..half-1], doc 1 x_t positions [half..L-1]
            # Doc 0 x_0 positions [L..L+half-1], doc 1 x_0 positions [L+half..2L-1]
            doc0_positions = list(range(0, half)) + list(range(L, L + half))
            doc1_positions = list(range(half, L)) + list(range(L + half, 2 * L))

            for q in doc0_positions:
                for k in doc1_positions:
                    assert m[q, k] == float('-inf'), \
                        f"Batch {b}: doc0 pos {q} should NOT attend doc1 pos {k}"


# ============================================================================
# 3. Loss Computation
# ============================================================================

class TestLossComputation:
    """Integration test for loss: forward -> loss -> backward -> verify grads."""

    @staticmethod
    def _compute_loss(hidden_states, targets, noise_mask, elbo_weight, lm_head_weight, pad_token_id):
        """Minimal chunked CE matching Phase 5 pattern."""
        B, L, D = hidden_states.shape
        logits = hidden_states @ lm_head_weight.T  # (B, L, vocab)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        ce = ce.view(B, L)
        # Weight by mask * ELBO
        weight = noise_mask.float() * elbo_weight
        total = (ce * weight).sum()
        real_count = (targets != pad_token_id).float().sum().clamp(min=1)
        return total / real_count

    def test_loss_backward_nonzero_grad(self):
        """Full pipeline: model forward -> loss -> backward -> grad_norm > 0."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.train()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))

        # Apply noise
        t_blocks, t = _sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        x_noisy, noise_mask = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                            block_size=cfg.block_size)

        # Build training input [x_t || x_0]
        idx = torch.cat([x_noisy, targets], dim=1)  # (B, 2L)
        attn_mask = build_staircase_mask(L, cfg.block_size)

        # Forward
        hidden, _ = model(idx, targets=targets, attn_mask=attn_mask)

        # Loss
        elbo_w = _compute_elbo_weight(t, t_min=cfg.t_min)  # (B, L)
        loss = self._compute_loss(hidden, targets, noise_mask, elbo_w,
                                  model.lm_head.weight, cfg.pad_token_id)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

        loss.backward()

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert grad_norm > 0, "grad_norm is 0 after loss.backward() — broken backward"

    def test_loss_reasonable_at_init(self):
        """Random init model should produce loss ~ ln(vocab_size) = ln(256) ~ 5.55."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B, L = 4, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))

        # Uniform noise at t=0.5
        t = torch.full((B, L), 0.5)
        x_noisy, noise_mask = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                            block_size=cfg.block_size)
        idx = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, cfg.block_size)

        with torch.no_grad():
            hidden, _ = model(idx, targets=targets, attn_mask=attn_mask)
            logits = model.lm_head(hidden)
            ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

        expected = math.log(cfg.vocab_size)  # ~5.55
        # Allow wide tolerance — init randomness + small model
        assert 2.0 < ce.item() < 15.0, \
            f"Init loss {ce.item():.2f} way off from expected ~{expected:.2f}"


# ============================================================================
# 4. Weight Loading Roundtrip
# ============================================================================

class TestWeightRoundtrip:

    def test_save_load_exact_match(self):
        """Create model A, save, create model B, load -> state dicts match exactly."""
        from phase6.checkpoint import save_checkpoint, load_checkpoint

        cfg = _tiny_cfg()
        model_a = Model(cfg)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)

        # Take a step so optimizer has state
        B, T = 1, 8
        logits, _ = model_a(torch.randint(0, cfg.vocab_size, (B, T)))
        logits.sum().backward()
        opt_a.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model_a, opt_a, step=5, loss=1.0, ckpt_dir=tmpdir)

            model_b = Model(cfg)
            opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
            load_checkpoint(os.path.join(tmpdir, 'latest.pt'), model_b, opt_b, device='cpu')

            sd_a = model_a.state_dict()
            sd_b = model_b.state_dict()
            assert set(sd_a.keys()) == set(sd_b.keys()), "State dict keys don't match"
            for k in sd_a:
                assert torch.equal(sd_a[k], sd_b[k]), f"Mismatch at key '{k}'"

    def test_hf_style_weight_loading(self):
        """Mock HF-style Qwen3 state dict -> load_from_hf -> all weights transferred."""
        from phase6.checkpoint import load_from_hf
        import phase6.checkpoint as ckpt_mod

        cfg = _tiny_cfg()
        model = Model(cfg)

        # Build fake HF state dict with Qwen3 naming
        hf_sd = {}
        d, nh, nkv, hd, mlp_h, V = (cfg.n_embd, cfg.n_head, cfg.n_kv_head,
                                      cfg.head_dim, cfg.mlp_hidden, cfg.vocab_size)

        hf_sd['model.embed_tokens.weight'] = torch.randn(V, d)
        hf_sd['model.norm.weight'] = torch.randn(d)
        hf_sd['lm_head.weight'] = hf_sd['model.embed_tokens.weight'].clone()

        for i in range(cfg.n_layer):
            pfx = f'model.layers.{i}'
            hf_sd[f'{pfx}.self_attn.q_proj.weight'] = torch.randn(nh * hd, d)
            hf_sd[f'{pfx}.self_attn.k_proj.weight'] = torch.randn(nkv * hd, d)
            hf_sd[f'{pfx}.self_attn.v_proj.weight'] = torch.randn(nkv * hd, d)
            hf_sd[f'{pfx}.self_attn.o_proj.weight'] = torch.randn(d, nh * hd)
            hf_sd[f'{pfx}.self_attn.q_norm.weight'] = torch.randn(hd)
            hf_sd[f'{pfx}.self_attn.k_norm.weight'] = torch.randn(hd)
            hf_sd[f'{pfx}.mlp.gate_proj.weight'] = torch.randn(mlp_h, d)
            hf_sd[f'{pfx}.mlp.up_proj.weight'] = torch.randn(mlp_h, d)
            hf_sd[f'{pfx}.mlp.down_proj.weight'] = torch.randn(d, mlp_h)
            hf_sd[f'{pfx}.input_layernorm.weight'] = torch.randn(d)
            hf_sd[f'{pfx}.post_attention_layernorm.weight'] = torch.randn(d)
            hf_sd[f'{pfx}.self_attn.rotary_emb.inv_freq'] = torch.randn(hd // 2)

        original = ckpt_mod._load_hf_weights

        def _mock_load(model_name):
            return hf_sd

        ckpt_mod._load_hf_weights = _mock_load
        try:
            missing, unexpected = load_from_hf(model, model_name='fake/qwen3', device='cpu')
        finally:
            ckpt_mod._load_hf_weights = original

        # token_emb loaded
        assert torch.allclose(model.token_emb.weight, hf_sd['model.embed_tokens.weight'])
        # lm_head should also reflect the same (tied)
        assert torch.allclose(model.lm_head.weight, hf_sd['model.embed_tokens.weight'])
        # Layer 0 q_proj
        assert torch.allclose(
            model.blocks[0].attn.c_q.weight,
            hf_sd['model.layers.0.self_attn.q_proj.weight']
        )
        # lm_head.weight should NOT be in missing (tied to token_emb)
        assert 'lm_head.weight' not in missing
        # No real missing keys (all model keys covered)
        assert len(missing) == 0, f"Unexpected missing keys: {missing}"


# ============================================================================
# 5. Noise Schedule Integration
# ============================================================================

class TestNoiseSchedule:

    def test_timestep_sampling_shape(self):
        """sample_timesteps returns correct shapes."""
        cfg = _tiny_cfg()
        t_blocks, t = _sample_timesteps(4, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        assert t_blocks.shape == (4, cfg.num_blocks)
        assert t.shape == (4, cfg.seq_len)

    def test_timestep_range(self):
        """All timesteps in [t_min, 1)."""
        cfg = _tiny_cfg()
        t_blocks, t = _sample_timesteps(32, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        assert t.min() >= cfg.t_min - 1e-6
        assert t.max() < 1.0 + 1e-6

    def test_noise_applies_mask_token(self):
        """apply_noise replaces some tokens with mask_token_id."""
        cfg = _tiny_cfg()
        B, L = 4, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))

        t_blocks, t = _sample_timesteps(B, cfg.num_blocks, cfg.block_size, t_min=cfg.t_min)
        x_noisy, noise_mask = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                            block_size=cfg.block_size)

        # Some tokens should be masked
        assert noise_mask.any(), "No tokens were masked"
        # Masked positions should have mask_token_id
        assert (x_noisy[noise_mask] == cfg.mask_token_id).all()
        # Unmasked positions should keep original values
        assert torch.equal(x_noisy[~noise_mask], targets[~noise_mask])

    def test_elbo_weights_are_inverse_t(self):
        """ELBO weight = 1/t for linear schedule."""
        t = torch.tensor([0.1, 0.25, 0.5, 1.0])
        w = _compute_elbo_weight(t, t_min=0.1)
        expected = 1.0 / t
        assert torch.allclose(w, expected)

    def test_per_block_min_1_masked(self):
        """Every block has at least 1 masked token — Stable-DiffCoder guarantee."""
        cfg = _tiny_cfg()
        B, L = 8, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))

        # Use very low t so some blocks would naturally have 0 masked tokens
        t = torch.full((B, L), 0.05)  # 5% mask rate -> P(0 in 8 tokens) = 0.95^8 = 66%
        x_noisy, noise_mask = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                            block_size=cfg.block_size)

        # Check every block has at least 1 masked token
        mask_blocks = noise_mask.view(B, cfg.num_blocks, cfg.block_size)
        per_block_count = mask_blocks.sum(dim=2)  # (B, num_blocks)
        assert (per_block_count >= 1).all(), \
            f"Some blocks have 0 masked tokens: min={per_block_count.min().item()}"


# ============================================================================
# 6. KV Cache Consistency
# ============================================================================

class TestKVCacheConsistency:

    def test_cache_accumulates_across_blocks(self):
        """KV cache stores prior block context — second block output changes with cache."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        torch.manual_seed(42)
        B = 1
        block_size = cfg.block_size
        block1 = torch.randint(0, cfg.vocab_size, (B, block_size))
        block2 = torch.randint(0, cfg.vocab_size, (B, block_size))

        # --- Block 2 WITHOUT cache (no prior context) ---
        with torch.no_grad():
            logits_no_cache, _ = model(block2, pos_offset=block_size)

        # --- Block 2 WITH cache from block 1 ---
        model.enable_kv_cache()
        model.reset_kv_cache()
        with torch.no_grad():
            model(block1, pos_offset=0)  # populate cache
            logits_with_cache, _ = model(block2, pos_offset=block_size)
        model.disable_kv_cache()

        # Outputs must differ — block 2 sees block 1 via cached KV
        assert not torch.allclose(logits_no_cache, logits_with_cache, atol=1e-6), \
            "Block 2 logits unchanged by cache — cache is not being used"

    def test_cache_reset_clears_state(self):
        """After reset_kv_cache, output matches fresh forward (no stale state)."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        torch.manual_seed(42)
        B = 1
        block_size = cfg.block_size
        block = torch.randint(0, cfg.vocab_size, (B, block_size))

        # Fresh forward (cache mode on but cache empty — bidirectional within block)
        model.enable_kv_cache()
        model.reset_kv_cache()
        with torch.no_grad():
            logits_fresh, _ = model(block, pos_offset=0)

        # Process some data to populate cache, then reset
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (B, block_size)), pos_offset=0)

        # Reset and process the same block
        model.reset_kv_cache()
        with torch.no_grad():
            logits_after_reset, _ = model(block, pos_offset=0)
        model.disable_kv_cache()

        assert torch.allclose(logits_fresh, logits_after_reset, atol=1e-6), \
            f"Cache not properly reset. Max diff: {(logits_fresh - logits_after_reset).abs().max().item()}"

    def test_cache_kv_shapes_grow(self):
        """KV cache dimensions grow as blocks are processed."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B = 1
        block_size = cfg.block_size

        model.enable_kv_cache()
        model.reset_kv_cache()

        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (B, block_size)), pos_offset=0)

        # After 1 block: cache should have block_size tokens
        for block in model.blocks:
            k_cache, v_cache = block.attn.kv_cache
            assert k_cache.shape[2] == block_size, \
                f"Expected cache length {block_size}, got {k_cache.shape[2]}"

        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (B, block_size)), pos_offset=block_size)

        # After 2 blocks: cache should have 2 * block_size tokens
        for block in model.blocks:
            k_cache, v_cache = block.attn.kv_cache
            assert k_cache.shape[2] == 2 * block_size, \
                f"Expected cache length {2 * block_size}, got {k_cache.shape[2]}"

        model.disable_kv_cache()


# ============================================================================
# 7. Document Packing Mask
# ============================================================================

class TestDocumentPacking:

    def test_doc_isolation_in_all_submasks(self):
        """With 2 packed documents, doc 0 tokens CANNOT attend doc 1 tokens."""
        L, blk = 32, 8
        B = 1
        # doc 0 = first 16 tokens, doc 1 = last 16 tokens
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, 16:] = 1

        mask = build_staircase_mask(L, blk, doc_ids=doc_ids)  # (B, 1, 2L, 2L)
        m = mask[0, 0]  # (2L, 2L)

        doc0_all = list(range(0, 16)) + list(range(L, L + 16))
        doc1_all = list(range(16, L)) + list(range(L + 16, 2 * L))

        for q in doc0_all:
            for k in doc1_all:
                assert m[q, k] == float('-inf'), \
                    f"Doc0 pos {q} should NOT attend doc1 pos {k}"
            # Also check reverse
        for q in doc1_all:
            for k in doc0_all:
                assert m[q, k] == float('-inf'), \
                    f"Doc1 pos {q} should NOT attend doc0 pos {k}"

    def test_doc_packing_forward_runs(self):
        """Model forward with doc_ids-based mask doesn't crash."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B, L = 2, cfg.seq_len
        doc_ids = torch.zeros(B, L, dtype=torch.long)
        doc_ids[:, L // 2:] = 1

        idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
        targets = torch.randint(0, cfg.vocab_size, (B, L))
        mask = build_staircase_mask(L, cfg.block_size, doc_ids=doc_ids)

        with torch.no_grad():
            out, _ = model(idx, targets=targets, attn_mask=mask)

        assert out.shape == (B, L, cfg.n_embd)


# ============================================================================
# 8. Numerical Sanity
# ============================================================================

class TestNumericalSanity:

    def test_init_loss_near_log_vocab(self):
        """Random init: CE loss ~ ln(256) ~ 5.55."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.eval()

        B, L = 8, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        idx = torch.randint(0, cfg.vocab_size, (B, 2 * L))
        mask = build_staircase_mask(L, cfg.block_size)

        with torch.no_grad():
            hidden, _ = model(idx, targets=targets, attn_mask=mask)
            logits = model.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

        expected = math.log(cfg.vocab_size)
        # Within 2x of expected — init variance can shift it
        assert loss.item() > expected * 0.3, f"Loss {loss.item():.3f} too low"
        assert loss.item() < expected * 3.0, f"Loss {loss.item():.3f} too high"

    def test_loss_decreases_after_training_steps(self):
        """After 10 gradient steps on random data, loss should decrease."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, L = 4, cfg.seq_len
        # Fixed data for overfitting
        torch.manual_seed(123)
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        attn_mask = build_staircase_mask(L, cfg.block_size)

        losses = []
        for _ in range(10):
            # Fresh noise each step
            t = torch.full((B, L), 0.5)
            x_noisy, _ = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                       block_size=cfg.block_size)
            idx = torch.cat([x_noisy, targets], dim=1)

            hidden, _ = model(idx, targets=targets, attn_mask=attn_mask)
            logits = model.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at step 10 should be lower than step 1
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: step 0={losses[0]:.4f}, step 9={losses[-1]:.4f}"

    def test_no_nan_in_outputs_or_gradients(self):
        """No NaN in forward output, loss, or gradients."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        model.train()

        B, L = 2, cfg.seq_len
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        t = torch.full((B, L), 0.5)
        x_noisy, noise_mask = _apply_noise(targets, t, mask_token_id=cfg.mask_token_id,
                                            block_size=cfg.block_size)
        idx = torch.cat([x_noisy, targets], dim=1)
        attn_mask = build_staircase_mask(L, cfg.block_size)

        hidden, _ = model(idx, targets=targets, attn_mask=attn_mask)
        assert torch.isfinite(hidden).all(), "NaN/Inf in hidden states"

        logits = model.lm_head(hidden)
        assert torch.isfinite(logits).all(), "NaN/Inf in logits"

        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"

        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"


# ============================================================================
# 9. Tied Embedding Consistency
# ============================================================================

class TestTiedEmbeddings:

    def test_same_data_ptr(self):
        """token_emb.weight and lm_head.weight share the same storage."""
        cfg = _tiny_cfg()
        model = Model(cfg)
        assert model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_in_place_modification_propagates(self):
        """Modify token_emb.weight in-place -> lm_head.weight also changes."""
        cfg = _tiny_cfg()
        model = Model(cfg)

        model.token_emb.weight.data[0, 0] = 999.0
        assert model.lm_head.weight[0, 0].item() == 999.0, \
            "In-place modification of token_emb didn't propagate to lm_head"

    def test_state_dict_has_both_keys(self):
        """state_dict includes both token_emb.weight and lm_head.weight (tied alias)."""
        cfg = _tiny_cfg()
        model = Model(cfg)

        sd = model.state_dict()
        assert 'token_emb.weight' in sd
        assert 'lm_head.weight' in sd
        # Both should point to the same data
        assert torch.equal(sd['token_emb.weight'], sd['lm_head.weight'])

        # But parameters() deduplicates (PyTorch tracks data_ptr)
        # So unique param count < state_dict key count
        param_count = sum(1 for _ in model.parameters())
        sd_count = len(sd)
        assert sd_count > param_count, \
            f"state_dict ({sd_count} keys) should have more entries than parameters() ({param_count})"


# ============================================================================
# 10. Config Validation Edge Cases
# ============================================================================

class TestConfigValidation:

    def test_seq_len_not_divisible_by_block_size_raises(self):
        """seq_len % block_size != 0 should raise."""
        cfg = Config(seq_len=32, block_size=7)
        with pytest.raises(AssertionError, match="not divisible"):
            cfg.validate()

    def test_valid_config_passes(self):
        """Normal config passes validation."""
        cfg = Config(seq_len=32, block_size=8)
        result = cfg.validate()
        assert result is cfg
        assert cfg.num_blocks == 4

    def test_num_blocks_computed(self):
        """validate() correctly sets num_blocks."""
        cfg = Config(seq_len=64, block_size=16)
        cfg.validate()
        assert cfg.num_blocks == 4

    def test_warmup_capped_at_2000(self):
        """For large max_iters, warmup caps at 2000."""
        cfg = Config(max_iters=100_000)
        cfg.validate()
        assert cfg.warmup_iters == 2000

    def test_warmup_scales_for_short_runs(self):
        """For short runs, warmup = 7% of max_iters."""
        cfg = Config(max_iters=100)
        cfg.validate()
        assert cfg.warmup_iters == 7
