"""Tokenizer utilities for Phase 6 (Qwen3-0.6B + <|mask|>)."""
from transformers import AutoTokenizer

_tokenizer = None

MASK_TOKEN = "<|mask|>"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def load_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    return _tokenizer


def encode(text: str) -> list[int]:
    return load_tokenizer().encode(text, add_special_tokens=False)


def decode(ids: list[int]) -> str:
    return load_tokenizer().decode(ids)


def get_mask_token_id() -> int:
    return load_tokenizer().convert_tokens_to_ids(MASK_TOKEN)


def get_vocab_size() -> int:
    return len(load_tokenizer())
