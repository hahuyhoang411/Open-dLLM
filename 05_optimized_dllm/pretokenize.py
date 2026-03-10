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
    python 05_optimized_dllm/pretokenize.py --hub-repo HoangHa/test-tokenized --max-docs 1000
"""

import argparse
import gc
import os
import shutil
from glob import glob

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Tokenizer

DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'
HUB_REPO = 'HoangHa/100BT-dLLM-pretokenized'
EOS_ID = 1  # <|endoftext|>
VOCAB_SIZE = 49_152
SHARD_SIZE = 5_000_000  # docs per shard — bounds peak memory


def _load_tokenizer():
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
  tok = Tokenizer.from_file(path)
  assert tok.get_vocab_size() == VOCAB_SIZE
  return tok


def _discover_shards(work_dir):
  return sorted(p for p in glob(os.path.join(work_dir, 'shard_*')) if os.path.isdir(p))


def _push_with_fallback(dataset, hub_repo, max_shard_size, num_proc, split=None):
  try:
    kwargs = {'max_shard_size': max_shard_size, 'num_proc': num_proc}
    if split is not None:
      kwargs['split'] = split
    dataset.push_to_hub(hub_repo, **kwargs)
    return
  except RuntimeError as exc:
    message = str(exc)
    if 'MerkleDB Shard error' not in message and 'xet' not in message.lower():
      raise

  print('Xet upload failed, retrying with HF_HUB_DISABLE_XET=1 and num_proc=1')
  os.environ['HF_HUB_DISABLE_XET'] = '1'
  kwargs = {'max_shard_size': max_shard_size, 'num_proc': 1}
  if split is not None:
    kwargs['split'] = split
  dataset.push_to_hub(hub_repo, **kwargs)


def pretokenize(
  hub_repo=HUB_REPO,
  num_proc=16,
  max_docs=None,
  work_dir='/tmp/tokenized_shards',
  mode='full',
  resume=True,
  max_shard_size='2GB',
  cleanup_work_dir=False,
):
  if mode not in {'full', 'tokenize', 'upload', 'upload-splits'}:
    raise ValueError(f'Unsupported mode: {mode}')

  shards = []

  if mode in {'full', 'tokenize'}:
    tok = _load_tokenizer()

    print(f'Loading source dataset: {DATASET}')
    ds = load_dataset(DATASET, split='train')

    if max_docs:
      ds = ds.select(range(min(max_docs, len(ds))))
      print(f'Limited to {len(ds):,} documents')

    n = len(ds)
    num_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f'Dataset: {n:,} documents → {num_shards} shards of {SHARD_SIZE:,}')

    features = Features({'input_ids': Sequence(Value('uint16'))})

    def tokenize_batch(batch):
      encoded = tok.encode_batch(batch['text'])
      return {'input_ids': [enc.ids + [EOS_ID] for enc in encoded]}

    os.makedirs(work_dir, exist_ok=True)

    for i in range(num_shards):
      start = i * SHARD_SIZE
      end = min(start + SHARD_SIZE, n)
      shard_path = os.path.join(work_dir, f'shard_{i:04d}')

      if resume and os.path.isdir(shard_path):
        print(f'\nShard {i + 1}/{num_shards}: exists, skipping ({shard_path})')
        continue

      print(f'\nShard {i + 1}/{num_shards}: docs {start:,}-{end - 1:,}')
      shard = ds.select(range(start, end))
      tokenized = shard.map(
        tokenize_batch,
        batched=True,
        batch_size=10_000,
        num_proc=num_proc,
        remove_columns=shard.column_names,
        features=features,
        desc=f'Shard {i + 1}/{num_shards}',
      )

      tokenized.save_to_disk(shard_path)
      print(f'Saved shard {i + 1} → {shard_path}')

      for cf in tokenized.cache_files:
        try:
          os.unlink(cf['filename'])
        except OSError:
          pass

      del shard, tokenized
      gc.collect()

    if mode == 'tokenize':
      print('Tokenization completed.')
      return

  shard_paths = _discover_shards(work_dir)
  if not shard_paths:
    raise RuntimeError(f'No shard directories found in {work_dir}')

  if mode == 'upload-splits':
    for i, p in enumerate(shard_paths):
      split = f'train_{i:04d}'
      loaded = load_from_disk(p)
      if not isinstance(loaded, Dataset):
        raise TypeError(f'Expected Dataset shard at {p}, got {type(loaded).__name__}')
      print(f'Pushing {split} from {p}')
      _push_with_fallback(loaded, hub_repo, max_shard_size, num_proc, split=split)
    print('Done.')
    return

  print(f'\nConcatenating {len(shard_paths)} shards from {work_dir}...')
  shards = []
  for p in shard_paths:
    loaded = load_from_disk(p)
    if not isinstance(loaded, Dataset):
      raise TypeError(f'Expected Dataset shard at {p}, got {type(loaded).__name__}')
    shards.append(loaded)
  combined = concatenate_datasets(shards)

  print(f'Pushing {len(combined):,} documents to Hub: {hub_repo}')
  _push_with_fallback(combined, hub_repo, max_shard_size, num_proc)

  if cleanup_work_dir and os.path.isdir(work_dir):
    shutil.rmtree(work_dir)
  print('Done.')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description='Pre-tokenize dataset and push to HF Hub')
  p.add_argument('--hub-repo', type=str, default=HUB_REPO)
  p.add_argument('--num-proc', type=int, default=16)
  p.add_argument('--max-docs', type=int, default=None)
  p.add_argument('--work-dir', type=str, default='/tmp/tokenized_shards')
  p.add_argument('--mode', type=str, choices=['full', 'tokenize', 'upload', 'upload-splits'], default='full')
  p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--max-shard-size', type=str, default='2GB')
  p.add_argument('--cleanup-work-dir', action=argparse.BooleanOptionalAction, default=False)
  args = p.parse_args()
  pretokenize(
    args.hub_repo,
    args.num_proc,
    args.max_docs,
    args.work_dir,
    args.mode,
    args.resume,
    args.max_shard_size,
    args.cleanup_work_dir,
  )
