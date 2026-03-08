"""Pre-tokenize the full dataset and push to HuggingFace Hub.

Tokenizes each document (variable-length, with trailing EOS) and stores as
uint16 on Hub. No packing — packing into seq_len happens at train time.

Usage:
    # On Modal (recommended for 100B tokens):
    modal run modal_train.py::pretokenize

    # With doc limit (for testing):
    modal run modal_train.py::pretokenize --max-docs 100000

    # Locally (small test):
    python 05_phase5_dllm/pretokenize.py --hub-repo HoangHa/test-tokenized --max-docs 1000
"""

import argparse
import os

from datasets import Features, Sequence, Value, load_dataset
from tokenizers import Tokenizer

DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'
HUB_REPO = 'HoangHa/100BT-dLLM-pretokenized'
EOS_ID = 1          # <|endoftext|>
VOCAB_SIZE = 49_152


def _load_tokenizer():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == VOCAB_SIZE
    return tok


def pretokenize(hub_repo=HUB_REPO, num_proc=16, max_docs=None):
    tok = _load_tokenizer()

    print(f'Loading source dataset: {DATASET}')
    ds = load_dataset(DATASET, split='train')

    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
        print(f'Limited to {len(ds):,} documents')

    print(f'Dataset: {len(ds):,} documents')

    features = Features({"input_ids": Sequence(Value("uint16"))})

    def tokenize_batch(batch):
        encoded = tok.encode_batch(batch["text"])
        return {"input_ids": [enc.ids + [EOS_ID] for enc in encoded]}

    print(f'Tokenizing with num_proc={num_proc}...')
    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=10000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        features=features,
        desc="Tokenizing",
    )

    print(f'Tokenized {len(tokenized):,} documents')
    print(f'Pushing to Hub: {hub_repo}')
    tokenized.push_to_hub(hub_repo, max_shard_size="2GB")

    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pre-tokenize dataset and push to HF Hub')
    p.add_argument('--hub-repo', type=str, default=HUB_REPO)
    p.add_argument('--num-proc', type=int, default=16)
    p.add_argument('--max-docs', type=int, default=None)
    args = p.parse_args()
    pretokenize(args.hub_repo, args.num_proc, args.max_docs)
