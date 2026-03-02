"""Train a BPE tokenizer on FineWeb-Edu using HuggingFace tokenizers."""

import argparse
import os
import tempfile

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DIR, "tokenizer.json")
NUM_DOCS = 50_000
VOCAB_SIZE = 32_768


def download_texts(path: str) -> None:
    """Stream FineWeb-Edu and write document texts to a temp file."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    with open(path, "w") as f:
        for i, doc in enumerate(ds):
            if i >= NUM_DOCS:
                break
            f.write(doc["text"] + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"  downloaded {i + 1:,}/{NUM_DOCS:,} docs")
    print(f"  downloaded {min(i + 1, NUM_DOCS):,}/{NUM_DOCS:,} docs — done")


def train_tokenizer(text_path: str) -> Tokenizer:
    """Train a BPE tokenizer with ByteLevel pre-tokenizer."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[MASK]", "<|endoftext|>", "<|padding|>"],
    )
    tokenizer.train([text_path], trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on FineWeb-Edu")
    parser.add_argument("--force", action="store_true", help="Retrain even if tokenizer.json exists")
    args = parser.parse_args()

    if os.path.exists(TOKENIZER_PATH) and not args.force:
        print(f"Tokenizer already exists: {TOKENIZER_PATH}")
        print("Use --force to retrain.")
        return

    # Download texts to a temp file
    print("Downloading FineWeb-Edu sample...")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.close()
    try:
        download_texts(tmp.name)
        print("Training BPE tokenizer...")
        tokenizer = train_tokenizer(tmp.name)
    finally:
        os.unlink(tmp.name)

    tokenizer.save(TOKENIZER_PATH)
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    # Verify special token IDs
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    assert tokenizer.token_to_id("[MASK]") == 0, "[MASK] should have id=0"
    assert tokenizer.token_to_id("<|endoftext|>") == 1, "<|endoftext|> should have id=1"
    assert tokenizer.token_to_id("<|padding|>") == 2, "<|padding|> should have id=2"
    print("Special tokens: [MASK]=0, <|endoftext|>=1, <|padding|>=2")

    test = "Hello, world! This is a test of the BPE tokenizer."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Test encode: {test!r}")
    print(f"  tokens ({len(encoded.ids)}): {encoded.tokens[:12]}{'...' if len(encoded.tokens) > 12 else ''}")
    print(f"  decoded: {decoded!r}")


if __name__ == "__main__":
    main()
