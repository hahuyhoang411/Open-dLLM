"""Build Phase 5 tokenizer: SmolLM2 BPE merges + Qwen3 pre-tokenization + diffusion specials.

Downloads HuggingFaceTB/cosmo2-tokenizer, extracts trained BPE merges,
replaces pre-tokenizer with Qwen3-style (NFC + GPT-4 regex + ByteLevel),
and prepends our 14 diffusion/chat special tokens at IDs 0-13.

Final vocab size: exactly 49,152 (divisible by 64 for tensor core alignment).
"""
import argparse
import json
import os

from tokenizers import Tokenizer, Regex, AddedToken
from tokenizers import models, pre_tokenizers, decoders, normalizers

DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DIR, "tokenizer.json")
TARGET_VOCAB_SIZE = 49_152

# Qwen3/GPT-4/LLaDA regex pattern for pre-tokenization
# Ref: 04_modern_dllm/train_tokenizer.py:30-38
PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"   # English contractions
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"       # optional punctuation + word
    r"|\p{N}"                            # single digit (digit splitting)
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"       # punctuation runs
    r"|\s*[\r\n]+"                       # newlines with leading whitespace
    r"|\s+(?!\S)"                        # trailing whitespace
    r"|\s+"                              # other whitespace
)

# Our 14 special tokens (IDs 0-13), unchanged from Phase 4
SPECIAL_TOKENS = [
    "<|mask|>",          # 0  - diffusion noise token
    "<|endoftext|>",     # 1  - EOS / document boundary
    "<|padding|>",       # 2  - right-padding
    "<|im_start|>",      # 3  - chat message start
    "<|im_end|>",        # 4  - chat message end
    "<|system|>",        # 5  - system role marker
    "<|user|>",          # 6  - user role marker
    "<|assistant|>",     # 7  - assistant role marker
    "<think>",           # 8  - reasoning start
    "</think>",          # 9  - reasoning end
    "<tool_call>",       # 10 - tool invocation start
    "</tool_call>",      # 11 - tool invocation end
    "<tool_response>",   # 12 - tool result start
    "</tool_response>",  # 13 - tool result end
]


def download_cosmo2_tokenizer():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="HuggingFaceTB/cosmo2-tokenizer",
        filename="tokenizer.json",
    )
    with open(path) as f:
        return json.load(f)


def build_tokenizer(cosmo2_json):
    model = cosmo2_json["model"]
    assert model["type"] == "BPE", f"Expected BPE model, got {model['type']}"

    source_vocab = model["vocab"]
    merges = model["merges"]

    # Identify SmolLM2's special tokens to discard (17 tokens, IDs 0-16)
    smollm2_specials = {at["content"] for at in cosmo2_json["added_tokens"] if at.get("special")}
    print(f"  Discarding {len(smollm2_specials)} SmolLM2 specials")

    # Keep all non-special tokens, sorted by original ID to preserve merge order
    non_special = sorted(
        ((tok, tid) for tok, tid in source_vocab.items() if tok not in smollm2_specials),
        key=lambda x: x[1],
    )
    print(f"  Keeping {len(non_special)} non-special tokens + {len(merges)} merges")

    # Build new vocab: our 14 specials at IDs 0-13, then non-special tokens at 14+
    new_vocab = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        new_vocab[tok] = i

    next_id = len(SPECIAL_TOKENS)
    for tok, _ in non_special:
        new_vocab[tok] = next_id
        next_id += 1

    # Pad to target vocab size with reserved tokens
    # (cosmo2 has 17 specials, we have 14 → 3 tokens short)
    pad_count = 0
    while len(new_vocab) < TARGET_VOCAB_SIZE:
        new_vocab[f"<|reserved_{pad_count}|>"] = next_id
        next_id += 1
        pad_count += 1
    if pad_count:
        print(f"  Padded with {pad_count} reserved tokens to reach {TARGET_VOCAB_SIZE}")

    assert len(new_vocab) == TARGET_VOCAB_SIZE, \
        f"Vocab size {len(new_vocab)} != target {TARGET_VOCAB_SIZE}"

    # Merges in JSON are "tok1 tok2" strings; BPE constructor wants (tok1, tok2) tuples
    merge_tuples = [tuple(m.split(" ", 1)) for m in merges]

    # Create BPE tokenizer with remapped vocab + original merges
    tokenizer = Tokenizer(models.BPE(vocab=new_vocab, merges=merge_tuples))

    # Qwen3-style: NFC normalization + regex split + byte-level
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

    # Register special tokens (marks them as non-splittable, non-normalized)
    tokenizer.add_special_tokens([
        AddedToken(t, special=True) for t in SPECIAL_TOKENS
    ])

    return tokenizer


def verify(tokenizer):
    vs = tokenizer.get_vocab_size()
    assert vs == TARGET_VOCAB_SIZE, f"Vocab size {vs} != {TARGET_VOCAB_SIZE}"
    print(f"Vocab size: {vs}")

    # Special token IDs
    for i, tok in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(tok)
        assert actual == i, f"{tok} should be id={i}, got {actual}"
    print(f"Special tokens verified: IDs 0-{len(SPECIAL_TOKENS) - 1}")

    # Digit splitting: "2024" must produce 4+ tokens (one per digit)
    digit_enc = tokenizer.encode("2024")
    assert len(digit_enc.ids) >= 4, \
        f"Expected 4+ tokens for '2024', got {len(digit_enc.ids)}: {digit_enc.tokens}"
    print(f"Digit splitting: '2024' -> {digit_enc.tokens}")

    # Round-trip encode/decode
    test = "The year 2024 has 365 days."
    enc = tokenizer.encode(test)
    dec = tokenizer.decode(enc.ids)
    assert dec == test, f"Round-trip failed: {test!r} -> {dec!r}"
    print(f"Round-trip: '{test}' -> {len(enc.ids)} tokens -> OK")

    # Unicode round-trip (NFC normalization)
    unicode_test = "café résumé naïve"
    enc = tokenizer.encode(unicode_test)
    dec = tokenizer.decode(enc.ids)
    assert dec == unicode_test, f"Unicode round-trip failed: {unicode_test!r} -> {dec!r}"
    print(f"Unicode: '{unicode_test}' -> {len(enc.ids)} tokens -> OK")

    # Special tokens should not be split by regular encoding
    mask_enc = tokenizer.encode("<|mask|>")
    assert len(mask_enc.ids) == 1 and mask_enc.ids[0] == 0, \
        f"<|mask|> should encode to [0], got {mask_enc.ids}"
    print(f"Special token encoding: <|mask|> -> [0] -> OK")


def main():
    parser = argparse.ArgumentParser(description="Build Phase 5 hybrid tokenizer")
    parser.add_argument("--force", action="store_true", help="Rebuild even if exists")
    args = parser.parse_args()

    if os.path.exists(TOKENIZER_PATH) and not args.force:
        print(f"Tokenizer already exists: {TOKENIZER_PATH}")
        print("Use --force to rebuild.")
        return

    print("Downloading SmolLM2 cosmo2-tokenizer...")
    cosmo2_json = download_cosmo2_tokenizer()

    print("Building hybrid tokenizer...")
    tokenizer = build_tokenizer(cosmo2_json)

    print(f"Saving to {TOKENIZER_PATH}...")
    tokenizer.save(TOKENIZER_PATH)

    print("\nVerification:")
    verify(tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()
