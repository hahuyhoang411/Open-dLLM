"""Tokenizer loading and encoding utilities for Phase 5."""
import os
from tokenizers import Tokenizer

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(DIR, "tokenizer.json")

SPECIAL_TOKENS = [
    "<|mask|>",          # 0
    "<|endoftext|>",     # 1
    "<|padding|>",       # 2
    "<|im_start|>",      # 3
    "<|im_end|>",        # 4
    "<|system|>",        # 5
    "<|user|>",          # 6
    "<|assistant|>",     # 7
    "<think>",           # 8
    "</think>",          # 9
    "<tool_call>",       # 10
    "</tool_call>",      # 11
    "<tool_response>",   # 12
    "</tool_response>",  # 13
]

_tokenizer = None


def load_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        assert _tokenizer.get_vocab_size() == 49_152
    return _tokenizer


def encode(text: str) -> list[int]:
    return load_tokenizer().encode(text).ids


def decode(ids: list[int]) -> str:
    return load_tokenizer().decode(ids)
