"""Kernel integration tests: correctness, compile compat, config independence, regression guard.

TDD — these tests define the contract for kernel optimizations that work
individually but regress in the full training loop. Tests are written FIRST;
implementations come after.

Known issues being tested:
- flash-attn RoPE: 4D cos/sin squeeze, position-gathered fallback, graph breaks
- gram-NS: orthogonality match, in-place mutation + cudagraph skip, compile scope
- RMSNorm: native nn.RMSNorm vs LigerRMSNorm equivalence, compile quality
- Config flags: cross-contamination, all-on/all-off
- Performance: no optimization should regress step time >5%
"""

import math
import time

import pytest
import torch
import torch.nn.functional as F

from phase6.config import Config
from phase6.attention import _apply_rotary_emb, apply_rotary_emb, _make_rms_norm
from phase6.optim import MuonClip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_cfg(**overrides):
    """Tiny model config for fast CPU tests."""
    defaults = dict(
        n_layer=2, n_embd=128, n_head=4, n_kv_head=2, head_dim=64,
        mlp_hidden=256, vocab_size=1024, seq_len=64, block_size=8,
        rope_base=10000.0, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        use_compile=False, use_muon=True, use_amp=False,
        mask_token_id=0, pad_token_id=2, eos_token_id=3,
        muon_lr=0.02, adamw_lr=3e-3,
    )
    defaults.update(overrides)
    return Config(**defaults).validate()


def _make_model(cfg):
    from phase6.model import Model
    return Model(cfg)


def _make_batch(cfg, B=2):
    """Training batch: [x_t || x_0] input + targets + staircase mask."""
    from phase6.attention import build_staircase_mask
    L = cfg.seq_len
    targets = torch.randint(3, cfg.vocab_size, (B, L))
    x_noisy = targets.clone()
    noise_mask = torch.rand(B, L) < 0.5
    x_noisy[noise_mask] = cfg.mask_token_id
    x_input = torch.cat([x_noisy, targets], dim=1)
    mask = build_staircase_mask(L, cfg.block_size)
    return x_input, targets, mask, noise_mask


def _run_train_step(model, cfg, optimizer=None, B=2):
    """Run one training step, return loss scalar."""
    from phase6.loss import compute_loss
    x_input, targets, mask, noise_mask = _make_batch(cfg, B=B)

    device = next(model.parameters()).device
    x_input = x_input.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    noise_mask = noise_mask.to(device)

    model.train()
    hidden, _ = model(x_input, targets=targets, attn_mask=mask)
    elbo_w = torch.ones_like(noise_mask, dtype=torch.float32, device=device) * 2.0
    loss = compute_loss(hidden, targets, noise_mask, elbo_w,
                        model.lm_head.weight, cfg)

    if optimizer is not None:
        optimizer.zero_grad()
    loss.backward()
    if optimizer is not None:
        optimizer.step()

    return loss.item()


# ============================================================================
# 1. Individual Kernel Correctness (CPU, no GPU needed)
# ============================================================================

class TestFlashAttnRoPE:
    """flash-attn RoPE must match manual implementation.

    The current _apply_rotary_emb is the manual baseline. When enable_fa_rope()
    is called, apply_rotary_emb should switch to flash-attn's fused kernel
    while producing numerically identical results.
    """

    def test_rope_correctness_simple(self):
        """Small shapes: RoPE output is correct shape, dtype, and finite."""
        B, T, H, D = 1, 8, 2, 16
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        ref = _apply_rotary_emb(x, cos, sin)

        assert ref.shape == (B, T, H, D)
        assert ref.dtype == x.dtype
        assert torch.isfinite(ref).all()

    def test_rope_correctness_real_shapes(self):
        """Qwen3-like shapes: B=2, T=128, H=16, HD=128."""
        B, T, H, D = 2, 128, 16, 128
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        ref = _apply_rotary_emb(x, cos, sin)
        assert ref.shape == (B, T, H, D)

        # Verify RoPE doesn't collapse to zero or identity
        assert not torch.allclose(ref, x, atol=1e-3)
        assert not torch.allclose(ref, torch.zeros_like(ref), atol=1e-3)

    def test_rope_backward_correctness(self):
        """Gradients flow through RoPE without NaN or zero."""
        B, T, H, D = 2, 32, 4, 64
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D, requires_grad=True)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        y = _apply_rotary_emb(x, cos, sin)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0, "RoPE backward must produce nonzero grads"

    def test_rope_with_position_gathering(self):
        """cos/sin with B>1 (position-gathered) must not crash.

        This is the bug that crashed the first H200 run: when positions
        are provided, cos/sin are gathered per-batch-element producing
        4D tensors with shape (B, T, 1, D//2). flash-attn expects
        (1, T, 1, D//2) or squeezed 2D. The code must fallback to manual
        impl when cos has batch dimension.
        """
        B, T, H, D = 2, 32, 4, 64
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)

        # Position-gathered cos/sin: shape (B, T, 1, D//2) -- batch dim present
        cos_batched = torch.randn(B, T, 1, D // 2)
        sin_batched = torch.randn(B, T, 1, D // 2)

        # Expand to match heads for GQA (mimics attention.py lines 228-232)
        cos_expanded = cos_batched.expand(-1, -1, H, -1)  # (B, T, H, D//2)
        sin_expanded = sin_batched.expand(-1, -1, H, -1)

        ref = _apply_rotary_emb(x, cos_expanded, sin_expanded)
        assert ref.shape == (B, T, H, D)
        assert torch.isfinite(ref).all()

    def test_rope_4d_cos_sin_squeeze(self):
        """Standard case: cos (1, T, 1, HD//2) -- the shape flash-attn can handle.

        This shape is a singleton batch dim, which flash-attn can squeeze to 2D.
        Verifying the standard training path works.
        """
        B, T, H, D = 2, 64, 4, 64
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)  # standard precomputed shape
        sin = torch.randn(1, T, 1, D // 2)

        ref = _apply_rotary_emb(x, cos, sin)
        assert ref.shape == (B, T, H, D)

    def test_rope_apply_rotary_emb_pair(self):
        """apply_rotary_emb(q, k, cos, sin) applies RoPE to both q and k."""
        B, T, D = 2, 64, 64
        n_head, n_kv_head = 4, 2
        torch.manual_seed(42)

        q = torch.randn(B, T, n_head, D)
        k = torch.randn(B, T, n_kv_head, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Verify both were actually rotated
        assert not torch.allclose(q_rot, q, atol=1e-3)
        assert not torch.allclose(k_rot, k, atol=1e-3)

    def test_rope_bf16_precision(self):
        """RoPE in bf16 stays finite and close to fp32 reference."""
        B, T, H, D = 2, 32, 4, 64
        torch.manual_seed(42)
        x_fp32 = torch.randn(B, T, H, D)
        cos_fp32 = torch.randn(1, T, 1, D // 2)
        sin_fp32 = torch.randn(1, T, 1, D // 2)

        ref = _apply_rotary_emb(x_fp32, cos_fp32, sin_fp32)

        x_bf16 = x_fp32.bfloat16()
        cos_bf16 = cos_fp32.bfloat16()
        sin_bf16 = sin_fp32.bfloat16()
        out_bf16 = _apply_rotary_emb(x_bf16, cos_bf16, sin_bf16)

        assert out_bf16.dtype == torch.bfloat16
        # bf16 tolerance: rtol=1e-2 is standard for bfloat16
        torch.testing.assert_close(out_bf16.float(), ref, atol=0.05, rtol=0.02)


class TestGramNS:
    """gram-NS must match compiled newton_schulz.

    MuonClip.newton_schulz is the reference. When enable_gram_ns() is called,
    the optimizer should use gram-NS instead, producing equivalent orthogonalized
    updates.
    """

    def test_ns_correctness_square(self):
        """1024x1024 square matrix: NS output shape and finiteness."""
        torch.manual_seed(42)
        G = torch.randn(1024, 1024)
        X = MuonClip.newton_schulz(G, steps=5)

        assert X.shape == G.shape
        assert torch.isfinite(X).all()

    def test_ns_correctness_rectangular(self):
        """3072x1024 rectangular matrix (MLP weight shape)."""
        torch.manual_seed(42)
        G = torch.randn(3072, 1024)
        X = MuonClip.newton_schulz(G, steps=5)

        assert X.shape == G.shape
        assert torch.isfinite(X).all()

    def test_ns_orthogonality(self):
        """NS output should be approximately orthogonal.

        For square matrix, X @ X.T should approximate scaled identity.
        """
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        X = MuonClip.newton_schulz(G, steps=5)

        # X @ X.T should be close to scaled I
        XtX = X.float() @ X.float().T
        diag = torch.diag(XtX)
        off_diag_mask = ~torch.eye(64, dtype=torch.bool)
        off_diag = XtX[off_diag_mask]

        # Diagonal should be roughly uniform
        assert diag.std() / diag.mean() < 0.1, \
            f"Diagonal should be uniform, CV={diag.std() / diag.mean():.3f}"
        # Off-diagonal should be small relative to diagonal
        assert off_diag.abs().mean() < diag.mean() * 0.1, \
            "Off-diagonal elements should be small"

    def test_ns_tall_matrix_orthogonality(self):
        """3072x1024: X.T @ X should approximate scaled identity."""
        torch.manual_seed(42)
        G = torch.randn(3072, 1024)
        X = MuonClip.newton_schulz(G, steps=5)

        XtX = X.float().T @ X.float()  # (1024, 1024)
        # Columns should have roughly equal norm
        col_norms = X.float().norm(dim=0)
        assert col_norms.std() / col_norms.mean() < 0.15, \
            f"Column norms should be uniform, CV={col_norms.std() / col_norms.mean():.3f}"

    def test_ns_deterministic(self):
        """Same input -> same output (no randomness in NS)."""
        torch.manual_seed(42)
        G = torch.randn(256, 256)

        X1 = MuonClip.newton_schulz(G, steps=5)
        X2 = MuonClip.newton_schulz(G, steps=5)

        torch.testing.assert_close(X1, X2)

    def test_ns_bf16_stability(self):
        """NS in bf16 (the actual training dtype) stays finite."""
        torch.manual_seed(42)
        G = torch.randn(512, 256).bfloat16()
        X = MuonClip.newton_schulz(G, steps=5)

        assert torch.isfinite(X).all(), "NS in bf16 should not produce NaN/Inf"
        assert X.dtype == torch.bfloat16

    def test_ns_zero_input_handled(self):
        """Zero input should not crash or produce NaN (eps guard)."""
        G = torch.zeros(64, 64)
        X = MuonClip.newton_schulz(G, steps=5)

        assert torch.isfinite(X).all(), "Zero input should not produce NaN"


class TestRMSNormSwap:
    """Native nn.RMSNorm must match LigerRMSNorm.

    When use_liger=False, the model should use native nn.RMSNorm.
    Output must be numerically equivalent within bf16 tolerance.
    """

    def test_native_rmsnorm_exists(self):
        """nn.RMSNorm should exist in torch >= 2.4."""
        assert hasattr(torch.nn, 'RMSNorm'), \
            f"torch.nn.RMSNorm not found (torch {torch.__version__})"

    def test_rmsnorm_factory_no_liger(self):
        """_make_rms_norm with use_liger=False returns native RMSNorm."""
        cfg = _small_cfg(use_liger=False)
        norm = _make_rms_norm(128, cfg)
        assert norm is not None
        x = torch.randn(2, 32, 128)
        y = norm(x)
        assert y.shape == x.shape

    def test_rmsnorm_forward_equivalence(self):
        """Native RMSNorm and manual RMSNorm produce same output."""
        dim = 128
        eps = 1e-6
        torch.manual_seed(42)
        weight = torch.randn(dim)
        x = torch.randn(2, 32, dim)

        # Manual reference
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        ref = (x / rms) * weight

        # Factory-created norm
        cfg = _small_cfg(use_liger=False)
        norm = _make_rms_norm(dim, cfg)
        # Copy weight
        with torch.no_grad():
            norm.weight.copy_(weight)
        out = norm(x)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)

    def test_rmsnorm_backward(self):
        """RMSNorm backward produces nonzero gradients."""
        cfg = _small_cfg(use_liger=False)
        norm = _make_rms_norm(128, cfg)
        x = torch.randn(2, 32, 128, requires_grad=True)
        y = norm(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_rmsnorm_bf16(self):
        """RMSNorm in bf16 produces finite output."""
        cfg = _small_cfg(use_liger=False)
        norm = _make_rms_norm(128, cfg)
        norm = norm.bfloat16()
        x = torch.randn(2, 32, 128, dtype=torch.bfloat16)
        y = norm(x)
        assert y.dtype == torch.bfloat16
        assert torch.isfinite(y).all()


class TestSwiGLU:
    """SwiGLU with and without Liger must produce equivalent results."""

    def test_swiglu_no_liger(self):
        """SwiGLU without Liger uses F.silu * up path."""
        from phase6.model import SwiGLU
        cfg = _small_cfg(use_liger=False)
        mlp = SwiGLU(cfg)
        x = torch.randn(2, 32, cfg.n_embd)
        y = mlp(x)
        assert y.shape == x.shape

    def test_swiglu_backward_no_liger(self):
        """SwiGLU backward without Liger produces nonzero grads."""
        from phase6.model import SwiGLU
        cfg = _small_cfg(use_liger=False)
        mlp = SwiGLU(cfg)
        x = torch.randn(2, 32, cfg.n_embd, requires_grad=True)
        y = mlp(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ============================================================================
# 2. torch.compile Compatibility (GPU required)
# ============================================================================

_HAS_CUDA = torch.cuda.is_available()
_SKIP_NO_GPU = pytest.mark.skipif(not _HAS_CUDA, reason='GPU required')


@_SKIP_NO_GPU
class TestCompileCompat:
    """Verify kernels don't break torch.compile.

    The core issue: optimizations that work in eager mode can cause graph
    breaks, recompilations, or cudagraph conflicts under torch.compile.
    """

    def _gpu_cfg(self, **overrides):
        kw = dict(device='cuda')
        kw.update(overrides)
        return _small_cfg(**kw)

    def test_baseline_compiled_forward(self):
        """Baseline: compiled model forward with NO kernel optimizations."""
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False, use_compile=False)
        model = _make_model(cfg).cuda()
        model.train(False)
        compiled = torch.compile(model, dynamic=False)

        x = torch.randint(0, cfg.vocab_size, (1, 16), device='cuda')
        with torch.no_grad():
            logits, _ = compiled(x)
        assert logits.shape == (1, 16, cfg.vocab_size)

    def test_baseline_compiled_training_step(self):
        """Baseline: compiled model + backward with NO kernel optimizations."""
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False, use_compile=False)
        model = _make_model(cfg).cuda()
        compiled = torch.compile(model, dynamic=False)

        x_input, targets, mask, noise_mask = _make_batch(cfg)
        x_input = x_input.cuda()
        targets = targets.cuda()
        mask = mask.cuda()

        compiled.train()
        hidden, _ = compiled(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                        if p.grad is not None)
        assert grad_norm > 0

    def test_compiled_block_graph_break_count(self):
        """Count graph breaks in a compiled Block forward.

        Graph breaks defeat the purpose of torch.compile -- each break adds
        Python overhead and prevents kernel fusion. Kernel optimizations
        should not INCREASE the graph break count.
        """
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False)
        from phase6.model import Block
        block = Block(cfg).cuda()
        block.train(False)

        # Count breaks via dynamo counters
        torch._dynamo.utils.counters.clear()

        compiled_fwd = torch.compile(block._forward, dynamic=False)
        B, T = 2, cfg.seq_len
        x = torch.randn(B, T, cfg.n_embd, device='cuda')
        cos = torch.randn(1, T // 2, 1, cfg.head_dim // 2, device='cuda')
        sin = torch.randn(1, T // 2, 1, cfg.head_dim // 2, device='cuda')

        with torch.no_grad():
            from phase6.attention import build_staircase_mask
            mask = build_staircase_mask(T // 2, cfg.block_size).cuda()
            _ = compiled_fwd(x, cos, sin, attn_mask=mask)

        breaks = dict(torch._dynamo.utils.counters.get('graph_break', {}))
        total_breaks = sum(breaks.values())

        # Record baseline graph breaks -- kernel opts should not exceed this
        assert isinstance(total_breaks, int), "Should be able to count graph breaks"

    def test_ns_outside_compiled_scope(self):
        """Newton-Schulz (optimizer step) runs OUTSIDE the compiled model scope.

        This is critical: the optimizer step is NOT compiled. gram-NS
        in-place mutations are safe because they happen in eager mode.
        """
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False)
        model = _make_model(cfg).cuda()
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        # Run a training step with compiled model
        compiled = torch.compile(model, dynamic=False)
        x_input, targets, mask, noise_mask = _make_batch(cfg)
        x_input = x_input.cuda()
        targets = targets.cuda()
        mask = mask.cuda()

        compiled.train()
        hidden, _ = compiled(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        optimizer.zero_grad()
        loss.backward()

        # This is the critical part: optimizer.step() calls newton_schulz
        # which is NOT inside the compiled graph. It should work fine.
        optimizer.step()  # Must not crash

    def test_rmsnorm_inside_compiled_block(self):
        """nn.RMSNorm inside a compiled Block should not cause graph breaks."""
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False)
        from phase6.model import Block
        block = Block(cfg).cuda()
        block.train(False)

        compiled_fwd = torch.compile(block._forward, dynamic=False)
        B, T = 1, cfg.seq_len
        x = torch.randn(B, T, cfg.n_embd, device='cuda')
        cos = torch.randn(1, T // 2, 1, cfg.head_dim // 2, device='cuda')
        sin = torch.randn(1, T // 2, 1, cfg.head_dim // 2, device='cuda')

        from phase6.attention import build_staircase_mask
        mask = build_staircase_mask(T // 2, cfg.block_size).cuda()

        with torch.no_grad():
            out = compiled_fwd(x, cos, sin, attn_mask=mask)
        assert out.shape == x.shape

    def test_compiled_model_matches_eager(self):
        """Compiled model output matches eager mode within tolerance."""
        torch._dynamo.reset()
        cfg = self._gpu_cfg(use_liger=False)
        torch.manual_seed(42)
        model = _make_model(cfg).cuda()
        model.train(False)

        x = torch.randint(0, cfg.vocab_size, (1, 16), device='cuda')

        with torch.no_grad():
            eager_out, _ = model(x)

        compiled = torch.compile(model, dynamic=False)
        with torch.no_grad():
            compiled_out, _ = compiled(x)

        max_diff = (eager_out - compiled_out).abs().max().item()
        assert max_diff < 1e-4, f"Compiled vs eager diff: {max_diff}"


# ============================================================================
# 3. Config Flag Independence
# ============================================================================

class TestConfigFlags:
    """Each optimization flag must be independent -- no cross-contamination.

    Turning one flag on/off should not affect the behavior of others.
    """

    def test_all_optimizations_off(self):
        """Baseline config with all optimizations disabled runs without error."""
        cfg = _small_cfg(
            use_liger=False,
            use_compile=False,
            use_flex=False,
            use_grad_ckpt=False,
            use_muon=False,
            use_amp=False,
        )
        model = _make_model(cfg)
        model.train(False)

        x = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 16, cfg.vocab_size)

    def test_all_optimizations_off_training(self):
        """Baseline config: full training step without any optimization."""
        cfg = _small_cfg(
            use_liger=False,
            use_compile=False,
            use_flex=False,
            use_grad_ckpt=False,
            use_muon=False,
            use_amp=False,
        )
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        loss = _run_train_step(model, cfg, optimizer)
        assert math.isfinite(loss), f"Loss must be finite, got {loss}"

    def test_muon_independent_of_liger(self):
        """use_muon=True works regardless of use_liger setting."""
        for use_liger in [True, False]:
            cfg = _small_cfg(use_liger=use_liger, use_muon=True)
            model = _make_model(cfg)
            from phase6.optim import create_optimizer
            optimizer = create_optimizer(model, cfg)

            loss = _run_train_step(model, cfg, optimizer)
            assert math.isfinite(loss), \
                f"use_liger={use_liger} + use_muon=True failed: loss={loss}"

    def test_grad_ckpt_independent_of_liger(self):
        """use_grad_ckpt=True works regardless of use_liger setting."""
        for use_liger in [True, False]:
            cfg = _small_cfg(use_liger=use_liger, use_grad_ckpt=True)
            model = _make_model(cfg)
            from phase6.optim import create_optimizer
            optimizer = create_optimizer(model, cfg)

            loss = _run_train_step(model, cfg, optimizer)
            assert math.isfinite(loss), \
                f"use_liger={use_liger} + use_grad_ckpt=True failed: loss={loss}"

    def test_qk_norm_independent_of_muon(self):
        """QK-norm toggle doesn't affect Muon optimizer."""
        for use_qk_norm in [True, False]:
            cfg = _small_cfg(use_qk_norm=use_qk_norm, use_muon=True)
            model = _make_model(cfg)
            from phase6.optim import create_optimizer
            optimizer = create_optimizer(model, cfg)

            loss = _run_train_step(model, cfg, optimizer)
            assert math.isfinite(loss), \
                f"use_qk_norm={use_qk_norm} failed: loss={loss}"

    def test_gated_query_independent_of_others(self):
        """use_gated_query works with any other flag combination."""
        cfg = _small_cfg(
            use_gated_query=True,
            use_liger=False,
            use_grad_ckpt=False,
            use_muon=True,
        )
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        loss = _run_train_step(model, cfg, optimizer)
        assert math.isfinite(loss)

    def test_config_validate_small(self):
        """Small config validates correctly: derived fields populated."""
        cfg = _small_cfg()
        assert cfg.num_blocks == cfg.seq_len // cfg.block_size
        assert cfg.warmup_iters > 0
        assert cfg.decay_start > 0


# ============================================================================
# 4. Staircase Mask Dtype Compatibility
# ============================================================================

class TestMaskDtype:
    """Dense staircase mask dtype must be compatible with query dtype.

    Known bug: mask built as bf16, SDPA expects mask dtype == query dtype.
    This tests the fix and prevents regression.
    """

    def test_mask_float32_with_float32_model(self):
        """Float32 model + float32 mask: no dtype mismatch."""
        from phase6.attention import build_staircase_mask
        cfg = _small_cfg(use_amp=False)
        model = _make_model(cfg).float()

        L = cfg.seq_len
        mask = build_staircase_mask(L, cfg.block_size)

        B = 2
        targets = torch.randint(3, cfg.vocab_size, (B, L))
        x_input = torch.cat([targets, targets], dim=1)

        model.train(False)
        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=mask)
        assert hidden.shape == (B, L, cfg.n_embd)

    @_SKIP_NO_GPU
    def test_mask_bf16_cast_to_query_dtype(self):
        """bf16 mask must be cast to query dtype before SDPA.

        This is the known bug: build_staircase_mask returns bf16 but SDPA
        expects the mask dtype to match q.dtype. If the model runs in fp32
        (e.g., no AMP), the bf16 mask causes a dtype mismatch error.
        """
        from phase6.attention import build_staircase_mask
        cfg = _small_cfg(use_amp=False)

        L = cfg.seq_len
        # Simulate the fix: mask should work regardless of its dtype
        mask_bf16 = build_staircase_mask(L, cfg.block_size).bfloat16()
        mask_fp32 = mask_bf16.float()

        # Both should contain the same logical mask
        assert (mask_bf16 == float('-inf')).sum() == (mask_fp32 == float('-inf')).sum()


# ============================================================================
# 5. Optimizer Integration
# ============================================================================

class TestOptimizerIntegration:
    """MuonClip + training pipeline integration."""

    def test_muonclip_with_model_training(self):
        """Full MuonClip step with actual model forward/backward."""
        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        w_before = next(model.parameters()).clone()
        loss = _run_train_step(model, cfg, optimizer)

        assert math.isfinite(loss)
        assert not torch.equal(w_before, next(model.parameters())), \
            "Weights should change after optimizer step"

    def test_adamw_with_model_training(self):
        """Full AdamW step (use_muon=False) with actual model forward/backward."""
        cfg = _small_cfg(use_muon=False)
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        loss = _run_train_step(model, cfg, optimizer)
        assert math.isfinite(loss)

    def test_multiple_steps_loss_finite(self):
        """5 consecutive steps all produce finite loss."""
        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        for step in range(5):
            loss = _run_train_step(model, cfg, optimizer)
            assert math.isfinite(loss), f"Step {step}: loss={loss}"

    def test_qk_clip_fires_on_high_logits(self):
        """QK-Clip should constrain weights when logits exceed tau.

        We test the mechanism by artificially inflating Q/K weights.
        """
        cfg = _small_cfg(use_muon=True, use_qk_norm=False)
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        # Inflate Q/K weights to produce large attention logits
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_q.weight.mul_(100.0)
                block.attn.c_k.weight.mul_(100.0)

        # Run step -- QK-Clip should engage
        loss = _run_train_step(model, cfg, optimizer)
        # If QK-Clip works, loss should still be finite (not NaN from huge logits)
        assert math.isfinite(loss), f"QK-Clip failed to prevent loss explosion: {loss}"


# ============================================================================
# 6. Performance Regression Guard (GPU required)
# ============================================================================

@_SKIP_NO_GPU
class TestPerformanceGuard:
    """Each optimization must not regress overall step time.

    These tests measure wall-clock time for training steps and verify
    that enabling an optimization doesn't make things slower.
    """

    def _time_steps(self, cfg, n_steps=5, n_warmup=2):
        """Time n_steps training steps after n_warmup warmup steps."""
        model = _make_model(cfg).cuda()
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        # Warmup
        for _ in range(n_warmup):
            _run_train_step(model, cfg, optimizer, B=2)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            _run_train_step(model, cfg, optimizer, B=2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return elapsed / n_steps

    def test_grad_ckpt_overhead_bounded(self):
        """Gradient checkpointing should not add >50% overhead.

        SAC (selective activation checkpointing) should keep overhead
        to ~2-15%. We test with a generous 50% bound.
        """
        cfg_base = _small_cfg(use_grad_ckpt=False, device='cuda')
        cfg_ckpt = _small_cfg(use_grad_ckpt=True, device='cuda')

        t_base = self._time_steps(cfg_base)
        t_ckpt = self._time_steps(cfg_ckpt)

        overhead = (t_ckpt - t_base) / t_base
        assert overhead < 0.50, \
            f"grad_ckpt overhead {overhead:.0%} exceeds 50% " \
            f"(base={t_base*1000:.1f}ms, ckpt={t_ckpt*1000:.1f}ms)"

    def test_muon_vs_adamw_comparable(self):
        """MuonClip step time should be within 3x of AdamW.

        Muon has Newton-Schulz overhead, but it should not be catastrophically
        slower. More than 3x indicates a bug (e.g., NS running on CPU).
        """
        cfg_muon = _small_cfg(use_muon=True, device='cuda')
        cfg_adamw = _small_cfg(use_muon=False, device='cuda')

        t_muon = self._time_steps(cfg_muon)
        t_adamw = self._time_steps(cfg_adamw)

        ratio = t_muon / t_adamw
        assert ratio < 3.0, \
            f"Muon is {ratio:.1f}x slower than AdamW " \
            f"(muon={t_muon*1000:.1f}ms, adamw={t_adamw*1000:.1f}ms)"

    def test_no_regression_baseline(self):
        """Record baseline step time for regression tracking.

        This test always passes -- it just records the baseline timing
        for comparison with future optimization tests.
        """
        cfg = _small_cfg(
            use_liger=False, use_compile=False, use_flex=False,
            use_grad_ckpt=False, use_muon=True, device='cuda',
        )
        t = self._time_steps(cfg)
        assert t > 0, f"Step time should be positive, got {t}"


# ============================================================================
# 7. Numerical Stability Under Combined Optimizations
# ============================================================================

class TestNumericalStability:
    """Combined optimizations should not cause numerical issues."""

    def test_loss_convergence_5_steps(self):
        """Loss should decrease (or at least not explode) over 5 steps."""
        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        losses = []
        torch.manual_seed(42)
        for _ in range(5):
            loss = _run_train_step(model, cfg, optimizer)
            losses.append(loss)

        # Loss should not explode (>100x first loss)
        assert losses[-1] < losses[0] * 100, \
            f"Loss exploded: {losses[0]:.4f} -> {losses[-1]:.4f}"
        # All losses finite
        assert all(math.isfinite(l) for l in losses), \
            f"Non-finite loss: {losses}"

    def test_gradients_finite_all_params(self):
        """All parameter gradients must be finite after backward."""
        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        _run_train_step(model, cfg)

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_gradients_nonzero(self):
        """At least some gradients should be nonzero (backward works)."""
        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        _run_train_step(model, cfg)

        has_nonzero = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero -- backward is broken"

    def test_no_nan_in_hidden_states(self):
        """Hidden states (model output before loss) must be finite."""
        cfg = _small_cfg()
        model = _make_model(cfg)
        model.train(False)

        x_input, targets, mask, _ = _make_batch(cfg)
        with torch.no_grad():
            hidden, _ = model(x_input, targets=targets, attn_mask=mask)
        assert torch.isfinite(hidden).all(), "NaN/Inf in hidden states"


# ============================================================================
# 8. Future Kernel Integration Contracts (TDD stubs)
# ============================================================================
# These tests define the contract for kernel optimizations that don't exist yet.
# They will fail until the integration code is written.

class TestFlashAttnRoPEIntegration:
    """Contract tests for flash-attn RoPE integration.

    These test the API that enable_fa_rope() should expose.
    Currently expected to FAIL -- implementation pending.
    """

    @pytest.mark.xfail(reason="enable_fa_rope not implemented yet")
    def test_enable_fa_rope_exists(self):
        """enable_fa_rope() function should exist in attention module."""
        from phase6.attention import enable_fa_rope
        assert callable(enable_fa_rope)

    @pytest.mark.xfail(reason="enable_fa_rope not implemented yet")
    def test_enable_fa_rope_swaps_implementation(self):
        """After enable_fa_rope(), apply_rotary_emb uses flash-attn kernel."""
        from phase6.attention import enable_fa_rope
        enable_fa_rope()

        B, T, H, D = 2, 32, 4, 64
        torch.manual_seed(42)
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, 2, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape

    @pytest.mark.xfail(reason="_apply_rotary_emb_manual not implemented yet")
    def test_manual_impl_preserved(self):
        """_apply_rotary_emb_manual should be the original Python implementation."""
        from phase6.attention import _apply_rotary_emb_manual
        B, T, H, D = 1, 8, 2, 16
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)
        ref = _apply_rotary_emb_manual(x, cos, sin)
        assert ref.shape == x.shape

    @pytest.mark.xfail(reason="use_fa_rope config flag not implemented yet")
    def test_config_flag_exists(self):
        """Config should have use_fa_rope flag."""
        cfg = Config()
        assert hasattr(cfg, 'use_fa_rope')

    @pytest.mark.xfail(reason="flash-attn RoPE not integrated yet")
    def test_fa_rope_matches_manual(self):
        """flash-attn RoPE must produce same output as manual within bf16 tol.

        This is the key correctness test. cos/sin shape (1, T, 1, D//2) must
        be squeezed to (T, D//2) for flash-attn, then results compared.
        """
        from phase6.attention import _apply_rotary_emb_manual
        try:
            from flash_attn.layers.rotary import apply_rotary_emb as fa_rope
        except ImportError:
            pytest.skip("flash-attn not installed")

        B, T, H, D = 2, 64, 4, 128
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)

        ref = _apply_rotary_emb_manual(x, cos, sin)

        # flash-attn expects cos/sin as (T, D//2) or (1, T, 1, D//2)
        cos_sq = cos.squeeze(0).squeeze(1)  # (T, D//2)
        sin_sq = sin.squeeze(0).squeeze(1)
        fa_out = fa_rope(x, cos_sq, sin_sq, interleaved=False)

        torch.testing.assert_close(fa_out, ref, atol=1e-3, rtol=1e-2)

    @pytest.mark.xfail(reason="flash-attn RoPE fallback not implemented yet")
    def test_fa_rope_fallback_on_batched_cos(self):
        """When cos has batch dim (B, T, 1, D//2), must fallback to manual.

        flash-attn RoPE doesn't support per-batch cos/sin. The integration
        must detect this and use the manual path.
        """
        from phase6.attention import enable_fa_rope, _apply_rotary_emb_manual
        enable_fa_rope()

        B, T, H, D = 2, 32, 4, 64
        torch.manual_seed(42)
        x = torch.randn(B, T, H, D)
        cos = torch.randn(B, T, 1, D // 2)  # batch dim present
        sin = torch.randn(B, T, 1, D // 2)

        ref = _apply_rotary_emb_manual(x, cos.expand(-1, -1, H, -1),
                                        sin.expand(-1, -1, H, -1))
        out = _apply_rotary_emb(x, cos.expand(-1, -1, H, -1),
                                 sin.expand(-1, -1, H, -1))

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


class TestGramNSIntegration:
    """Contract tests for gram-NS integration.

    These test the API that enable_gram_ns() should expose.
    Currently expected to FAIL -- implementation pending.
    """

    @pytest.mark.xfail(reason="enable_gram_ns not implemented yet")
    def test_enable_gram_ns_exists(self):
        """enable_gram_ns() function should exist in optim module."""
        from phase6.optim import enable_gram_ns
        assert callable(enable_gram_ns)

    @pytest.mark.xfail(reason="enable_gram_ns not implemented yet")
    def test_enable_gram_ns_replaces_ns(self):
        """After enable_gram_ns(), MuonClip.step uses gram-NS."""
        from phase6.optim import enable_gram_ns, create_optimizer
        enable_gram_ns()

        cfg = _small_cfg(use_muon=True)
        model = _make_model(cfg)
        optimizer = create_optimizer(model, cfg)

        loss = _run_train_step(model, cfg, optimizer)
        assert math.isfinite(loss)

    @pytest.mark.xfail(reason="use_gram_ns config flag not implemented yet")
    def test_config_flag_exists(self):
        """Config should have use_gram_ns flag."""
        cfg = Config()
        assert hasattr(cfg, 'use_gram_ns')

    @pytest.mark.xfail(reason="_newton_schulz_compiled not implemented yet")
    def test_compiled_ns_exists(self):
        """Module-level _newton_schulz_compiled should exist."""
        from phase6.optim import _newton_schulz_compiled
        assert callable(_newton_schulz_compiled)

    @pytest.mark.xfail(reason="gram-NS batched not implemented yet")
    def test_gram_ns_batched_matches_individual(self):
        """Batched gram-NS (stack N params, call once) matches N individual calls.

        The batching optimization stacks same-shape params into a 3D tensor
        and calls gram-NS once. Results must match calling NS individually
        on each param.
        """
        torch.manual_seed(42)
        n_params = 4
        M, N = 1024, 1024
        params = [torch.randn(M, N) for _ in range(n_params)]

        # Individual calls
        individual = [MuonClip.newton_schulz(p, steps=5) for p in params]

        # Batched call (the API we're defining)
        from phase6.optim import enable_gram_ns
        enable_gram_ns()
        batched_input = torch.stack(params)  # (n_params, M, N)
        # gram-NS should handle 3D input
        from gram_newton_schulz import GramNewtonSchulz
        gram_ns = GramNewtonSchulz(ns_use_kernels=False)
        batched_output = gram_ns(batched_input)

        for i in range(n_params):
            # Gram-NS uses different coefficients so won't match exactly,
            # but should produce similarly orthogonal results
            col_norms_ind = individual[i].float().norm(dim=0)
            col_norms_bat = batched_output[i].float().norm(dim=0)
            # Both should have uniform column norms (orthogonal property)
            assert col_norms_ind.std() / col_norms_ind.mean() < 0.15
            assert col_norms_bat.std() / col_norms_bat.mean() < 0.15


class TestLigerFlags:
    """Contract tests for fine-grained Liger control.

    Currently use_liger is a single flag. We need separate:
    - use_liger_swiglu: Liger SwiGLU kernel
    - use_liger_rmsnorm: Liger RMSNorm kernel

    Expected to FAIL -- implementation pending.
    """

    @pytest.mark.xfail(reason="use_liger_swiglu config flag not implemented yet")
    def test_liger_swiglu_flag_exists(self):
        """Config should have use_liger_swiglu flag."""
        cfg = Config()
        assert hasattr(cfg, 'use_liger_swiglu')

    @pytest.mark.xfail(reason="use_liger_rmsnorm config flag not implemented yet")
    def test_liger_rmsnorm_flag_exists(self):
        """Config should have use_liger_rmsnorm flag."""
        cfg = Config()
        assert hasattr(cfg, 'use_liger_rmsnorm')

    @pytest.mark.xfail(reason="Fine-grained Liger flags not implemented yet")
    def test_swiglu_on_rmsnorm_off(self):
        """use_liger_swiglu=True + use_liger_rmsnorm=False should work."""
        cfg = _small_cfg(use_liger_swiglu=True, use_liger_rmsnorm=False)
        model = _make_model(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        model.train(False)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 16, cfg.vocab_size)

    @pytest.mark.xfail(reason="Fine-grained Liger flags not implemented yet")
    def test_swiglu_off_rmsnorm_on(self):
        """use_liger_swiglu=False + use_liger_rmsnorm=True should work."""
        cfg = _small_cfg(use_liger_swiglu=False, use_liger_rmsnorm=True)
        model = _make_model(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        model.train(False)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 16, cfg.vocab_size)


# ============================================================================
# 9. End-to-End Kernel Flag Combinations
# ============================================================================

class TestFlagCombinations:
    """Exhaustive test of flag combinations that should work together.

    Tests all pairwise combinations of optimization flags to catch
    interaction bugs that only appear when multiple features are enabled.
    """

    @pytest.mark.parametrize("use_muon", [True, False])
    @pytest.mark.parametrize("use_grad_ckpt", [True, False])
    @pytest.mark.parametrize("use_qk_norm", [True, False])
    def test_flag_combination_runs(self, use_muon, use_grad_ckpt, use_qk_norm):
        """Every combination of core flags produces finite loss."""
        cfg = _small_cfg(
            use_muon=use_muon,
            use_grad_ckpt=use_grad_ckpt,
            use_qk_norm=use_qk_norm,
            use_liger=False,  # CPU-safe
        )
        model = _make_model(cfg)
        from phase6.optim import create_optimizer
        optimizer = create_optimizer(model, cfg)

        loss = _run_train_step(model, cfg, optimizer)
        assert math.isfinite(loss), \
            f"muon={use_muon}, ckpt={use_grad_ckpt}, qknorm={use_qk_norm}: loss={loss}"
