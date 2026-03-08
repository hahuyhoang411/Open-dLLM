"""Pre-tokenize the full dataset into sharded numpy files for fast training.

Eliminates the CPU bottleneck of on-the-fly HF streaming + tokenization.
Produces uint16 numpy shards that the training data loader can memmap.
Doc boundary IDs are reconstructed from EOS positions at load time.

Usage:
    # On Modal (recommended for 100B tokens):
    modal run modal_train.py::pretokenize

    # With token limit:
    modal run modal_train.py::pretokenize --max-tokens 20000000000

    # Locally (small test):
    python 05_phase5_dllm/pretokenize.py --out-dir ./tokenized --max-tokens 1_000_000

Storage: ~200 GB for 100B tokens (uint16 only, no doc_ids stored).
"""

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# ============================================================================
# Constants (match phase5/config.py — standalone to avoid parse_args import)
# ============================================================================

DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'
SEQ_LEN = 2048
EOS_ID = 1          # <|endoftext|>
VOCAB_SIZE = 49_152
SEQS_PER_SHARD = 100_000   # ~400 MB per shard
DOC_BATCH = 2000            # batch-encode this many docs (Rust parallelism)


# ============================================================================
# Standalone tokenizer (no config.py dependency)
# ============================================================================

def _load_tokenizer():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == VOCAB_SIZE, f'Expected {VOCAB_SIZE}, got {tok.get_vocab_size()}'
    return tok


# ============================================================================
# Core
# ============================================================================

def pretokenize(out_dir: str, max_tokens: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, 'meta.json')

    # Resume: count existing shards
    existing = sorted(f for f in os.listdir(out_dir) if f.startswith('shard_') and f.endswith('.npy'))
    start_shard = len(existing)
    n_docs_consumed = 0

    if start_shard > 0 and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_docs_consumed = meta.get('n_docs_consumed', 0)
        print(f'Resuming: {start_shard} shards exist, skipping {n_docs_consumed:,} docs')

    tok = _load_tokenizer()
    ds = load_dataset(DATASET, split='train', streaming=True)

    if n_docs_consumed > 0:
        t_skip = time.time()
        ds = ds.skip(n_docs_consumed)
        print(f'Skip took {time.time() - t_skip:.1f}s')

    ds_iter = iter(ds)

    # Pre-allocate buffer (2x shard size for overflow)
    shard_tokens = SEQS_PER_SHARD * SEQ_LEN
    buf = np.empty(shard_tokens * 2, dtype=np.uint16)
    buf_pos = 0

    shard_idx = start_shard
    total_tokens = start_shard * shard_tokens
    docs_this_run = 0
    t0 = time.time()
    done = False

    while not done:
        if max_tokens and total_tokens >= max_tokens:
            break

        # Batch-read documents
        texts = []
        for _ in range(DOC_BATCH):
            try:
                doc = next(ds_iter)
            except StopIteration:
                done = True
                break
            texts.append(doc['text'])

        if not texts:
            break

        # Batch encode (HF Tokenizer Rust parallelism)
        encoded = tok.encode_batch(texts)
        docs_this_run += len(texts)

        for enc in encoded:
            ids = enc.ids
            n = len(ids) + 1  # +1 for EOS

            # Grow buffer if needed
            if buf_pos + n > len(buf):
                new_buf = np.empty(max(len(buf) * 2, buf_pos + n), dtype=np.uint16)
                new_buf[:buf_pos] = buf[:buf_pos]
                buf = new_buf

            buf[buf_pos:buf_pos + len(ids)] = np.array(ids, dtype=np.uint16)
            buf[buf_pos + len(ids)] = EOS_ID
            buf_pos += n

        # Flush complete shards
        while buf_pos >= shard_tokens:
            if max_tokens and total_tokens >= max_tokens:
                done = True
                break

            shard = buf[:shard_tokens].reshape(SEQS_PER_SHARD, SEQ_LEN).copy()
            shard_path = os.path.join(out_dir, f'shard_{shard_idx:05d}.npy')
            np.save(shard_path, shard)

            # Shift remainder
            remain = buf_pos - shard_tokens
            if remain > 0:
                buf[:remain] = buf[shard_tokens:buf_pos]
            buf_pos = remain

            total_tokens += shard_tokens
            shard_idx += 1

            elapsed = time.time() - t0
            run_tokens = (shard_idx - start_shard) * shard_tokens
            tok_s = run_tokens / max(elapsed, 1)
            target = max_tokens if max_tokens else 100_000_000_000
            remaining_tok = target - total_tokens
            eta_s = remaining_tok / max(tok_s, 1)

            print(f'shard {shard_idx - 1:5d} | '
                  f'{total_tokens / 1e9:.2f}B tok | '
                  f'{tok_s / 1e6:.1f}M tok/s | '
                  f'{elapsed / 60:.0f}m elapsed | '
                  f'ETA {eta_s / 3600:.1f}h')

            # Save metadata for resumability
            _save_meta(meta_path, shard_idx, total_tokens,
                       n_docs_consumed + docs_this_run)

    # Partial last shard
    if buf_pos >= SEQ_LEN and not (max_tokens and total_tokens >= max_tokens):
        n_seqs = buf_pos // SEQ_LEN
        usable = n_seqs * SEQ_LEN
        shard = buf[:usable].reshape(n_seqs, SEQ_LEN).copy()
        shard_path = os.path.join(out_dir, f'shard_{shard_idx:05d}.npy')
        np.save(shard_path, shard)
        total_tokens += usable
        shard_idx += 1
        print(f'shard {shard_idx - 1:5d} (partial: {n_seqs:,} seqs) | '
              f'{total_tokens / 1e9:.2f}B tok')

    _save_meta(meta_path, shard_idx, total_tokens, n_docs_consumed + docs_this_run)

    total_time = time.time() - t0
    print(f'\nDone: {total_tokens / 1e9:.2f}B tokens in {shard_idx} shards. '
          f'Time: {total_time / 3600:.1f}h')


def _save_meta(path, n_shards, total_tokens, n_docs):
    meta = {
        'total_tokens': int(total_tokens),
        'total_sequences': int(n_shards * SEQS_PER_SHARD),
        'n_shards': n_shards,
        'seq_len': SEQ_LEN,
        'vocab_size': VOCAB_SIZE,
        'eos_token_id': EOS_ID,
        'seqs_per_shard': SEQS_PER_SHARD,
        'dataset': DATASET,
        'n_docs_consumed': int(n_docs),
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pre-tokenize dataset into numpy shards')
    p.add_argument('--out-dir', type=str, required=True, help='output directory for shards')
    p.add_argument('--max-tokens', type=int, default=None,
                    help='stop after N tokens (default: process all)')
    args = p.parse_args()
    pretokenize(args.out_dir, args.max_tokens)
