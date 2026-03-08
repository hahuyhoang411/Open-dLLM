"""Pre-tokenize the full dataset and push to HuggingFace Hub.

Tokenizes each document (variable-length, with trailing EOS) and stores as
uint16 on Hub. No packing — packing into seq_len happens at train time.

Processes in shards of 5M docs to avoid OOM on large datasets (62M+ docs).

Usage:
    # On Modal (recommended for 100B tokens):
    modal run modal_train.py::pretokenize

    # With doc limit (for testing):
    modal run modal_train.py::pretokenize --max-docs 100000

    # Locally (small test):
    python 05_phase5_dllm/pretokenize.py --hub-repo HoangHa/test-tokenized --max-docs 1000
"""

import argparse
import gc
import os
import shutil

from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Tokenizer

DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'
HUB_REPO = 'HoangHa/100BT-dLLM-pretokenized'
EOS_ID = 1          # <|endoftext|>
VOCAB_SIZE = 49_152
SHARD_SIZE = 5_000_000  # docs per shard — bounds peak memory


def _load_tokenizer():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == VOCAB_SIZE
    return tok


def pretokenize(hub_repo=HUB_REPO, num_proc=16, max_docs=None, work_dir='/tmp/tokenized_shards'):
    tok = _load_tokenizer()

    print(f'Loading source dataset: {DATASET}')
    ds = load_dataset(DATASET, split='train')

    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
        print(f'Limited to {len(ds):,} documents')

    n = len(ds)
    num_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f'Dataset: {n:,} documents → {num_shards} shards of {SHARD_SIZE:,}')

    features = Features({"input_ids": Sequence(Value("uint16"))})

    def tokenize_batch(batch):
        encoded = tok.encode_batch(batch["text"])
        return {"input_ids": [enc.ids + [EOS_ID] for enc in encoded]}

    os.makedirs(work_dir, exist_ok=True)

    for i in range(num_shards):
        start = i * SHARD_SIZE
        end = min(start + SHARD_SIZE, n)
        print(f'\nShard {i+1}/{num_shards}: docs {start:,}-{end-1:,}')

        shard = ds.select(range(start, end))
        tokenized = shard.map(
            tokenize_batch,
            batched=True,
            batch_size=10_000,
            num_proc=num_proc,
            remove_columns=shard.column_names,
            features=features,
            desc=f"Shard {i+1}/{num_shards}",
        )

        shard_path = os.path.join(work_dir, f'shard_{i:04d}')
        tokenized.save_to_disk(shard_path)
        print(f'Saved shard {i+1} → {shard_path}')

        # Free .map() cache files to reclaim disk
        for cf in tokenized.cache_files:
            try:
                os.unlink(cf['filename'])
            except OSError:
                pass

        del shard, tokenized
        gc.collect()

    # Load all shards (memory-mapped), concatenate, push
    print(f'\nConcatenating {num_shards} shards...')
    shards = [load_from_disk(os.path.join(work_dir, f'shard_{i:04d}')) for i in range(num_shards)]
    combined = concatenate_datasets(shards)

    print(f'Pushing {len(combined):,} documents to Hub: {hub_repo}')
    combined.push_to_hub(hub_repo, max_shard_size="2GB")

    shutil.rmtree(work_dir)
    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pre-tokenize dataset and push to HF Hub')
    p.add_argument('--hub-repo', type=str, default=HUB_REPO)
    p.add_argument('--num-proc', type=int, default=16)
    p.add_argument('--max-docs', type=int, default=None)
    p.add_argument('--work-dir', type=str, default='/tmp/tokenized_shards')
    args = p.parse_args()
    pretokenize(args.hub_repo, args.num_proc, args.max_docs, args.work_dir)
