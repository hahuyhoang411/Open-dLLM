"""Test torch.compile compatibility: full-model vs per-block, with and without Liger.

Phase 5 used per-block compile when Liger or grad_ckpt was on, full-model otherwise.
SDAR uses @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs") on flex_attention only.

Key question: can we do full-model compile with Liger?
Phase 5 bug (Liger Issue #174): tl.constexpr tracing failure with whole-model compile.
This test checks if that's still the case.
"""

import pytest
import torch

from phase6.config import Config
from phase6.model import Model
from phase6.attention import build_staircase_mask


TINY = Config(
    n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
    mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
    rope_base=10000, rms_eps=1e-6, dropout=0.0,
    use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
    use_liger=False, use_grad_ckpt=False, use_flex=False,
    pad_token_id=0, mask_token_id=1, eos_token_id=2,
).validate()


def _make_batch(cfg):
    B, L = 2, cfg.seq_len
    targets = torch.randint(3, cfg.vocab_size, (B, L))
    t = torch.full((B, L), 0.5)
    noise_mask = torch.rand(B, L) < 0.5
    x_noisy = targets.clone()
    x_noisy[noise_mask] = cfg.mask_token_id
    x_input = torch.cat([x_noisy, targets], dim=1)
    mask_float = build_staircase_mask(L, cfg.block_size)
    return x_input, targets, mask_float


# --- Full-model compile (no Liger, no grad_ckpt) ---

def test_full_model_compile_forward():
    """torch.compile(model) works for forward pass."""
    cfg = TINY
    model = Model(cfg)
    model.train(False)
    compiled = torch.compile(model, dynamic=False)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = compiled(x)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_full_model_compile_training():
    """torch.compile(model) works for full training step (forward + backward)."""
    cfg = TINY
    model = Model(cfg)
    model.train()
    compiled = torch.compile(model, dynamic=False)
    x_input, targets, mask = _make_batch(cfg)
    hidden, _ = compiled(x_input, targets=targets, attn_mask=mask)
    loss = (hidden ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "Gradients must flow through compiled model"


def test_full_model_compile_with_loss():
    """Full pipeline: compiled model -> compute_loss -> backward."""
    from phase6.loss import compute_loss
    cfg = TINY
    model = Model(cfg)
    model.train()
    compiled = torch.compile(model, dynamic=False)
    x_input, targets, mask = _make_batch(cfg)

    noise_mask = torch.rand(2, cfg.seq_len) < 0.5
    elbo_w = torch.ones(2, cfg.seq_len) * 2.0

    hidden, _ = compiled(x_input, targets=targets, attn_mask=mask)
    loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    loss.backward()
    assert loss.isfinite()


# --- Per-block compile ---

def test_per_block_compile():
    """Per-block compile (Phase 5 default when grad_ckpt or Liger active)."""
    cfg = TINY
    model = Model(cfg)
    for block in model.blocks:
        block._forward = torch.compile(block._forward, dynamic=False)
    model.train()
    x_input, targets, mask = _make_batch(cfg)
    hidden, _ = model(x_input, targets=targets, attn_mask=mask)
    loss = (hidden ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


# --- Compile with grad_ckpt ---

def test_compile_with_grad_ckpt():
    """Per-block compile + gradient checkpointing."""
    cfg_ckpt = Config(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=True, use_flex=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
    ).validate()
    model = Model(cfg_ckpt)
    model.train()
    for block in model.blocks:
        block._forward = torch.compile(block._forward, dynamic=False)
    x_input, targets, mask = _make_batch(cfg_ckpt)
    hidden, _ = model(x_input, targets=targets, attn_mask=mask)
    loss = (hidden ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


# --- Compile with Liger (if available) ---

def test_compile_with_liger():
    """Test compile compatibility with Liger kernels."""
    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa
        from liger_kernel.transformers import LigerRMSNorm  # noqa
        liger_available = True
    except ImportError:
        liger_available = False

    if not liger_available:
        pytest.skip("Liger not installed")

    cfg_liger = Config(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=True, use_grad_ckpt=False, use_flex=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
    ).validate()
    model = Model(cfg_liger)
    model.train()

    # Try full-model compile with Liger
    full_model_works = False
    try:
        compiled = torch.compile(model, dynamic=False)
        x_input, targets, mask = _make_batch(cfg_liger)
        hidden, _ = compiled(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        loss.backward()
        full_model_works = True
        print("FULL-MODEL compile + Liger: WORKS")
    except Exception as e:
        print(f"FULL-MODEL compile + Liger: FAILS ({type(e).__name__}: {e})")

    # Per-block compile with Liger (should always work)
    model2 = Model(cfg_liger)
    model2.train()
    for block in model2.blocks:
        block._forward = torch.compile(block._forward, dynamic=False)
    x_input, targets, mask = _make_batch(cfg_liger)
    hidden, _ = model2(x_input, targets=targets, attn_mask=mask)
    loss = (hidden ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model2.parameters() if p.grad is not None)
    assert grad_norm > 0, "Per-block compile + Liger must work"
    print("PER-BLOCK compile + Liger: WORKS")

    if not full_model_works:
        print("CONCLUSION: Liger still incompatible with full-model compile. Use per-block.")


# --- Compare compiled vs eager outputs ---

def test_compiled_matches_eager():
    """Compiled model produces same output as eager model."""
    cfg = TINY
    torch.manual_seed(42)
    model = Model(cfg)
    model.train(False)
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        eager_logits, _ = model(x)

    compiled = torch.compile(model, dynamic=False)
    with torch.no_grad():
        compiled_logits, _ = compiled(x)

    max_diff = (eager_logits - compiled_logits).abs().max().item()
    assert max_diff < 1e-5, f"Compiled output differs from eager by {max_diff}"
