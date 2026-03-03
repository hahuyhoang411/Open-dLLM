"""Train a BPE tokenizer with Qwen3-style pre-tokenizer on multi-source 100B dataset.

Changes from Phase 2 tokenizer:
  - NFC Unicode normalization (canonical composition)
  - GPT-4/Qwen3 regex pre-tokenizer: splits contractions, words, individual
    digits, punctuation, and whitespace into atomic chunks before BPE merging
  - ByteLevel with use_regex=False (regex handled by the Split step)
  - Larger training sample (100K docs vs 50K) for better merge coverage

The digit-splitting behavior comes from \\p{N} matching single numeric characters,
so "2024" -> ["2","0","2","4"] before BPE, which improves number handling.
"""

import argparse
import os
import tempfile

from datasets import load_dataset
from tokenizers import (
    Tokenizer, Regex,
    models, pre_tokenizers, decoders, trainers, normalizers,
)

DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DIR, "tokenizer.json")
NUM_DOCS = 100_000
VOCAB_SIZE = 32_768

# GPT-4 / Qwen3 regex pattern for pre-tokenization word boundaries
PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"   # English contractions
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"       # optional punctuation + word
    r"|\p{N}"                            # single digit (digit splitting)
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"       # punctuation runs
    r"|\s*[\r\n]+"                       # newlines with leading whitespace
    r"|\s+(?!\S)"                        # trailing whitespace
    r"|\s+"                              # other whitespace
)


def download_texts(path: str) -> None:
    ds = load_dataset("HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", split="train", streaming=True)
    with open(path, "w") as f:
        for i, doc in enumerate(ds):
            if i >= NUM_DOCS:
                break
            f.write(doc["text"] + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"  downloaded {i + 1:,}/{NUM_DOCS:,} docs")
    print(f"  downloaded {min(i + 1, NUM_DOCS):,}/{NUM_DOCS:,} docs — done")


def train_tokenizer(text_path: str) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())

    # Qwen3-style: NFC normalization + regex split + byte-level encoding
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(PRETOKENIZER_REGEX),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    # SFT-ready special tokens: IDs assigned in list order (0, 1, 2, ...) before
    # BPE training. Qwen 3.5-style chat/think/tool tokens reserved now so the
    # vocabulary layout is stable before pretraining.
    special_tokens = [
        "<|mask|>",          # 0  — diffusion noise token
        "<|endoftext|>",     # 1  — EOS / document boundary
        "<|padding|>",       # 2  — right-padding
        "<|im_start|>",      # 3  — chat message start
        "<|im_end|>",        # 4  — chat message end
        "<|system|>",        # 5  — system role marker
        "<|user|>",          # 6  — user role marker
        "<|assistant|>",     # 7  — assistant role marker
        "<think>",           # 8  — reasoning start
        "</think>",          # 9  — reasoning end
        "<tool_call>",       # 10 — tool invocation start
        "</tool_call>",      # 11 — tool invocation end
        "<tool_response>",   # 12 — tool result start
        "</tool_response>",  # 13 — tool result end
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
    )
    tokenizer.train([text_path], trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer (Qwen3-style)")
    parser.add_argument("--force", action="store_true", help="Retrain even if tokenizer exists")
    args = parser.parse_args()

    if os.path.exists(TOKENIZER_PATH) and not args.force:
        print(f"Tokenizer already exists: {TOKENIZER_PATH}")
        print("Use --force to retrain.")
        return

    print("Downloading multi-source dataset sample...")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.close()
    try:
        download_texts(tmp.name)
        print("Training BPE tokenizer (Qwen3-style pre-tokenizer)...")
        tokenizer = train_tokenizer(tmp.name)
    finally:
        os.unlink(tmp.name)

    tokenizer.save(TOKENIZER_PATH)
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    # Verify special token IDs
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    expected = [
        ("<|mask|>", 0), ("<|endoftext|>", 1), ("<|padding|>", 2),
        ("<|im_start|>", 3), ("<|im_end|>", 4), ("<|system|>", 5),
        ("<|user|>", 6), ("<|assistant|>", 7), ("<think>", 8), ("</think>", 9),
        ("<tool_call>", 10), ("</tool_call>", 11),
        ("<tool_response>", 12), ("</tool_response>", 13),
    ]
    for name, expected_id in expected:
        actual = tokenizer.token_to_id(name)
        assert actual == expected_id, f"{name} should have id={expected_id}, got {actual}"
    print(f"Special tokens ({len(expected)}): " + ", ".join(f"{n}={i}" for n, i in expected))

    # Test digit splitting
    test = "The year 2024 has 365 days."
    encoded = tokenizer.encode(test)
    print(f"\nTest encode: {test!r}")
    print(f"  tokens ({len(encoded.ids)}): {encoded.tokens}")
    print(f"  decoded: {tokenizer.decode(encoded.ids)!r}")

    # Verify digit splitting works: "2024" should be split into 4+ tokens
    digit_test = "2024"
    digit_encoded = tokenizer.encode(digit_test)
    print(f"\nDigit test: {digit_test!r}")
    print(f"  tokens ({len(digit_encoded.ids)}): {digit_encoded.tokens}")


if __name__ == "__main__":
    main()
