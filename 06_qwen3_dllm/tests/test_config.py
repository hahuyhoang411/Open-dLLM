"""Tests for Phase 6 config — dataclass + factory, no module-level side effects."""

import sys
import pytest


# ---------------------------------------------------------------------------
# 1. Import safety — no argparse triggered at import time
# ---------------------------------------------------------------------------

def test_import_does_not_trigger_argparse():
    """Importing config must not call parse_args or touch sys.argv."""
    original_argv = sys.argv[:]
    sys.argv = ['test']  # minimal argv, no --train etc.
    try:
        # If config.py calls parse_args() at module level, this will raise
        # SystemExit or error because 'test' isn't a valid arg set.
        from phase6.config import Config  # noqa: F401
    finally:
        sys.argv = original_argv


def test_import_exposes_public_api():
    from phase6.config import Config, from_cli, setup_device, setup_features
    assert callable(from_cli)
    assert callable(setup_device)
    assert callable(setup_features)


# ---------------------------------------------------------------------------
# 2. Defaults match Qwen3-0.6B
# ---------------------------------------------------------------------------

def test_defaults_qwen3_06b():
    from phase6.config import Config
    c = Config()
    assert c.n_layer == 28
    assert c.n_embd == 1024
    assert c.n_head == 16
    assert c.n_kv_head == 8
    assert c.head_dim == 128
    assert c.mlp_hidden == 3072
    assert c.vocab_size == 151_936
    assert c.seq_len == 2048
    assert c.block_size == 8
    assert c.rope_base == 1_000_000
    assert c.rms_eps == 1e-6


def test_defaults_token_ids():
    from phase6.config import Config
    c = Config()
    assert c.mask_token_id == 151669
    assert c.eos_token_id == 151645
    assert c.pad_token_id == 151643


def test_defaults_arch_toggles():
    from phase6.config import Config
    c = Config()
    assert c.use_emb_norm is False
    assert c.use_gated_query is False
    assert c.use_qk_norm is True


# ---------------------------------------------------------------------------
# 3. head_dim is independent — NOT derived from n_embd // n_head
# ---------------------------------------------------------------------------

def test_head_dim_independent():
    from phase6.config import Config
    # Qwen3: n_head*head_dim = 16*128 = 2048 != n_embd = 1024
    c = Config(head_dim=128, n_embd=1024, n_head=16)
    assert c.head_dim == 128
    assert c.n_embd == 1024
    assert c.n_head == 16
    # head_dim should NOT be overwritten to n_embd // n_head = 64
    assert c.head_dim != c.n_embd // c.n_head


def test_head_dim_custom():
    from phase6.config import Config
    c = Config(head_dim=64, n_embd=512, n_head=8)
    assert c.head_dim == 64


# ---------------------------------------------------------------------------
# 4. validate() catches bad assertions
# ---------------------------------------------------------------------------

def test_validate_bad_kv_ratio():
    from phase6.config import Config
    c = Config(n_head=16, n_kv_head=5)  # 16 % 5 != 0
    with pytest.raises(AssertionError):
        c.validate()


def test_validate_bad_block_size():
    from phase6.config import Config
    c = Config(seq_len=2048, block_size=7)  # 2048 % 7 != 0
    with pytest.raises(AssertionError):
        c.validate()


# ---------------------------------------------------------------------------
# 5. validate() computes derived fields
# ---------------------------------------------------------------------------

def test_validate_derived_fields():
    from phase6.config import Config
    c = Config(seq_len=2048, block_size=8, max_iters=50_000)
    c.validate()
    assert c.num_blocks == 2048 // 8  # 256
    assert c.warmup_iters == min(2000, max(1, int(0.07 * 50_000)))  # 2000
    assert c.decay_start == int(0.8 * 50_000)  # 40_000


def test_validate_warmup_capped():
    from phase6.config import Config
    # Short run: warmup should be 7% of max_iters, not 2000
    c = Config(max_iters=100)
    c.validate()
    assert c.warmup_iters == 7  # int(0.07 * 100) = 7


def test_validate_returns_self():
    from phase6.config import Config
    c = Config()
    result = c.validate()
    assert result is c


# ---------------------------------------------------------------------------
# 6. from_cli() with mock sys.argv
# ---------------------------------------------------------------------------

def test_from_cli_defaults(monkeypatch):
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', ['train.py', '--train'])
    c = from_cli()
    assert c.n_layer == 28
    assert c.n_embd == 1024
    assert c.head_dim == 128
    assert c.vocab_size == 151_936


def test_from_cli_overrides(monkeypatch):
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--train',
        '--n-layer', '12',
        '--n-embd', '512',
        '--head-dim', '64',
        '--block-size', '16',
        '--vocab-size', '32000',
        '--rope-base', '10000',
        '--batch-size', '16',
        '--fp8',
        '--cart',
    ])
    c = from_cli()
    assert c.n_layer == 12
    assert c.n_embd == 512
    assert c.head_dim == 64
    assert c.block_size == 16
    assert c.vocab_size == 32000
    assert c.rope_base == 10000
    assert c.batch_size == 16
    assert c.use_fp8 is True
    assert c.use_cart is True


def test_from_cli_no_flags(monkeypatch):
    """--no-amp, --no-compile etc. should flip the flags."""
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--train', '--no-amp', '--no-compile', '--no-muon',
    ])
    c = from_cli()
    assert c.use_amp is False
    assert c.use_compile is False
    assert c.use_muon is False


def test_from_cli_hf_model_name(monkeypatch):
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--train', '--hf-model-name', 'Qwen/Qwen3-0.6B',
    ])
    c = from_cli()
    assert c.hf_model_name == 'Qwen/Qwen3-0.6B'


def test_from_cli_token_ids(monkeypatch):
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--train',
        '--mask-token-id', '0',
        '--eos-token-id', '1',
        '--pad-token-id', '2',
    ])
    c = from_cli()
    assert c.mask_token_id == 0
    assert c.eos_token_id == 1
    assert c.pad_token_id == 2


def test_from_cli_arch_toggles(monkeypatch):
    from phase6.config import from_cli
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--train',
        '--use-emb-norm', '--use-gated-query', '--no-qk-norm',
    ])
    c = from_cli()
    assert c.use_emb_norm is True
    assert c.use_gated_query is True
    assert c.use_qk_norm is False
