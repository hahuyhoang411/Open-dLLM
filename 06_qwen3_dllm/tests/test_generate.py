"""Tests for Phase 6 block-diffusion generation with dynamic remasking."""

import pytest
import torch

from phase6.config import Config
from phase6.model import Model


def small_cfg(**overrides):
    defaults = dict(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000.0, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        mask_token_id=0, eos_token_id=1, pad_token_id=2,
        denoise_steps=4, use_compile=False,
    )
    defaults.update(overrides)
    return Config(**defaults).validate()


@pytest.fixture
def cfg():
    return small_cfg()


@pytest.fixture
def model(cfg):
    m = Model(cfg)
    m.eval()
    return m


# ============================================================================
# 1. get_num_transfer_tokens
# ============================================================================

def test_get_num_transfer_tokens_even():
    from phase6.generate import get_num_transfer_tokens
    result = get_num_transfer_tokens(8, 4)
    assert result.tolist() == [2, 2, 2, 2]
    assert result.sum().item() == 8


def test_get_num_transfer_tokens_uneven():
    from phase6.generate import get_num_transfer_tokens
    result = get_num_transfer_tokens(7, 4)
    assert result.sum().item() == 7
    # Remainder 3 distributed to first 3 steps: [2, 2, 2, 1]
    assert result.tolist() == [2, 2, 2, 1]


def test_get_num_transfer_tokens_one_step():
    from phase6.generate import get_num_transfer_tokens
    result = get_num_transfer_tokens(8, 1)
    assert result.tolist() == [8]


# ============================================================================
# 2. Output is list of ints
# ============================================================================

def test_output_is_list_of_ints(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20, 30]
    result = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.7)
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


# ============================================================================
# 3. Output length respects max_new_tokens (within +/- block_size)
# ============================================================================

def test_output_length(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20, 30]
    max_new = 16
    result = generate(model, prompt_ids, cfg, max_new_tokens=max_new, temperature=0.7)
    # Output = prompt + generated. Generated should be <= max_new + block_size
    generated_len = len(result) - len(prompt_ids)
    assert generated_len <= max_new + cfg.block_size, \
        f"Generated {generated_len} tokens, expected <= {max_new + cfg.block_size}"


# ============================================================================
# 4. Prompt preserved
# ============================================================================

def test_prompt_preserved(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20, 30]
    result = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.7)
    assert result[:len(prompt_ids)] == prompt_ids


# ============================================================================
# 5. Dynamic remasking: tau=0.0 accepts everything -> block completes fast
# ============================================================================

def test_dynamic_remasking_tau_zero(model, cfg):
    """With tau=0.0, every token is above threshold -> all committed in step 1."""
    from phase6.generate import generate
    prompt_ids = [10, 20]
    # Should still produce output -- the speedup is that fewer denoise steps run
    result = generate(model, prompt_ids, cfg, max_new_tokens=8,
                      remasking='confidence_dynamic', confidence_threshold=0.0)
    assert isinstance(result, list)
    assert len(result) >= len(prompt_ids)


# ============================================================================
# 6. Static remasking: fixed tokens per step
# ============================================================================

def test_static_remasking(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20]
    result = generate(model, prompt_ids, cfg, max_new_tokens=8,
                      remasking='confidence_static')
    assert isinstance(result, list)
    assert len(result) >= len(prompt_ids)


# ============================================================================
# 7. Random remasking
# ============================================================================

def test_random_remasking(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20]
    result = generate(model, prompt_ids, cfg, max_new_tokens=8,
                      remasking='random')
    assert isinstance(result, list)
    assert len(result) >= len(prompt_ids)


# ============================================================================
# 8. KV cache cleaned up after generate()
# ============================================================================

def test_kv_cache_cleaned_up(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20]
    generate(model, prompt_ids, cfg, max_new_tokens=8)
    for block in model.blocks:
        assert block.attn.kv_cache is None, "KV cache should be None after generate()"
        assert block.attn.cache_mode is False, "cache_mode should be False after generate()"


# ============================================================================
# 9. EOS truncation
# ============================================================================

def test_eos_truncation(model, cfg):
    """If model produces eos_token_id, output stops there (no eos in output)."""
    from phase6.generate import generate
    # We can't force the model to produce EOS, but we verify that
    # eos_token_id is not in the returned list (it gets truncated)
    prompt_ids = [10, 20]
    result = generate(model, prompt_ids, cfg, max_new_tokens=16)
    assert cfg.eos_token_id not in result


# ============================================================================
# 10. Temperature=0 is deterministic
# ============================================================================

def test_temperature_zero_deterministic(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20, 30]
    torch.manual_seed(42)
    r1 = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.0)
    torch.manual_seed(42)
    r2 = generate(model, prompt_ids, cfg, max_new_tokens=16, temperature=0.0)
    assert r1 == r2


# ============================================================================
# 11. No special tokens in output
# ============================================================================

def test_no_special_tokens_in_output(model, cfg):
    from phase6.generate import generate
    prompt_ids = [10, 20, 30]
    result = generate(model, prompt_ids, cfg, max_new_tokens=16)
    assert cfg.mask_token_id not in result
    assert cfg.pad_token_id not in result
    assert cfg.eos_token_id not in result


# ============================================================================
# 12. Invalid remasking strategy raises ValueError
# ============================================================================

def test_invalid_remasking_raises(model, cfg):
    from phase6.generate import generate
    with pytest.raises(ValueError, match="Unknown remasking"):
        generate(model, prompt_ids=[10], cfg=cfg, max_new_tokens=8,
                 remasking='nonexistent')


# ============================================================================
# 13. Prompt longer than block_size
# ============================================================================

def test_long_prompt(model, cfg):
    """Prompt spanning multiple blocks is handled correctly."""
    from phase6.generate import generate
    # 20 tokens = 2 full blocks (8) + remainder (4)
    prompt_ids = list(range(10, 30))
    result = generate(model, prompt_ids, cfg, max_new_tokens=8)
    assert result[:len(prompt_ids)] == prompt_ids


# ============================================================================
# 14. Empty prompt
# ============================================================================

def test_empty_prompt(model, cfg):
    from phase6.generate import generate
    result = generate(model, [], cfg, max_new_tokens=8)
    assert isinstance(result, list)
    assert len(result) > 0


# ============================================================================
# 15. denoise_steps defaults to cfg.denoise_steps
# ============================================================================

def test_denoise_steps_default(cfg):
    """When denoise_steps=None, uses cfg.denoise_steps."""
    from phase6.generate import generate
    m = Model(cfg)
    m.eval()
    prompt_ids = [10, 20]
    # Should not crash -- denoise_steps derived from cfg
    result = generate(m, prompt_ids, cfg, max_new_tokens=8, denoise_steps=None)
    assert isinstance(result, list)
