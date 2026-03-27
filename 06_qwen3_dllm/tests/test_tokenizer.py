"""Tests for Phase 6 tokenizer (Qwen3-0.6B + <|mask|>)."""
import pytest
from transformers import PreTrainedTokenizerBase

from phase6.tokenizer import load_tokenizer, encode, decode, get_mask_token_id, get_vocab_size


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset cached tokenizer between tests."""
    import phase6.tokenizer as mod
    mod._tokenizer = None
    yield
    mod._tokenizer = None


def test_load_returns_autotokenizer():
    tok = load_tokenizer()
    assert isinstance(tok, PreTrainedTokenizerBase)


def test_encode_decode_roundtrip():
    text = "Hello, world!"
    ids = encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0
    recovered = decode(ids)
    assert text in recovered or recovered.strip() == text


def test_mask_token_in_vocab():
    tok = load_tokenizer()
    mask_id = get_mask_token_id()
    assert isinstance(mask_id, int)
    assert mask_id >= 0
    # token should decode back to <|mask|>
    assert "<|mask|>" in tok.decode([mask_id])


def test_vocab_size():
    size = get_vocab_size()
    # Qwen3-0.6B base vocab + 1 for <|mask|>
    # len(tokenizer) reflects actual usable vocab (151,669 base + 1 mask = 151,670)
    assert size == 151_670


def test_encode_no_mask_in_normal_text():
    mask_id = get_mask_token_id()
    ids = encode("The quick brown fox jumps over the lazy dog.")
    assert mask_id not in ids


def test_eos_token_exists():
    tok = load_tokenizer()
    assert tok.eos_token is not None
    assert tok.eos_token_id is not None


def test_singleton_caching():
    tok1 = load_tokenizer()
    tok2 = load_tokenizer()
    assert tok1 is tok2
