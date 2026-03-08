# Pre-tokenize 100B Tokens → HF Hub — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pre-tokenize the full 100B-token dataset using `datasets.map(num_proc=16)` and push to `HoangHa/100BT-dLLM-pretokenized` on HF Hub. Training loads from Hub cache with zero tokenization overhead.

**Architecture:** Rewrite `pretokenize.py` to use HF datasets pipeline (no numpy shards). Add `_PreTokenizedPacker` to `data.py` that loads cached HF dataset and packs at train time (any seq_len). Keep `_ShardedLoader` for backward compat with existing numpy shards.

**Tech Stack:** `datasets` (map, push_to_hub), `tokenizers` (encode_batch), `huggingface_hub` (auth)

---

### Task 1: Rewrite pretokenize.py — HF datasets.map + push_to_hub

**Files:**
- Rewrite: `05_phase5_dllm/pretokenize.py`

**Step 1: Replace pretokenize.py with HF Hub pipeline**

```python
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

    total_tokens = sum(len(row["input_ids"]) for row in tokenized.iter(batch_size=10000))
    print(f'Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)')

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
```

**Step 2: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('05_phase5_dllm/pretokenize.py').read()); print('OK')"`

---

### Task 2: Add _PreTokenizedPacker to data.py

**Files:**
- Modify: `05_phase5_dllm/phase5/data.py`

**Step 1: Add _PreTokenizedPacker class after _ShardedLoader**

The class loads an HF dataset (from Hub or cache), reads pre-tokenized documents
via batch random access, packs into fixed-length sequences at train time.
Same interface as `_DocumentPacker`: `get_sequence() -> (token_ids, doc_ids)`.

```python
class _PreTokenizedPacker:
    """Packs pre-tokenized HF dataset documents into fixed-length sequences.

    Reads from cached Arrow files (downloaded from Hub on first use).
    Each document is already tokenized with trailing EOS.
    Packing into config.seq_len happens here — works for any seq_len.
    """

    def __init__(self, dataset_id, rank=0, world_size=1, seed=42):
        from datasets import load_dataset as _load_ds
        ds = _load_ds(dataset_id, split='train')
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
        self._ds = ds
        self._n = len(ds)
        self._rng = np.random.RandomState(seed + rank)
        self._eos_id = config.eos_token_id
        self._buf = []

    def _refill(self):
        """Pull a batch of random documents into the buffer."""
        indices = self._rng.randint(0, self._n, size=100).tolist()
        batch = self._ds[indices]
        for ids in batch["input_ids"]:
            self._buf.extend(ids)  # already includes EOS

    def get_sequence(self):
        """Return one packed sequence of exactly seq_len tokens."""
        while len(self._buf) < config.seq_len:
            self._refill()

        token_ids = self._buf[:config.seq_len]
        self._buf = self._buf[config.seq_len:]

        # Reconstruct doc_ids from EOS positions
        doc_ids = []
        doc_id = 0
        for tid in token_ids:
            doc_ids.append(doc_id)
            if tid == self._eos_id:
                doc_id += 1

        return token_ids, doc_ids
```

Key design points:
- **Batch random access** (`ds[indices]`): reads 100 docs at once from Arrow cache. O(1) per lookup.
- **Random sampling with replacement**: avoids 4 GB shuffle array for 500M-row dataset. Replacement is fine for multi-epoch training.
- **DDP**: `ds.shard(contiguous=True)` gives each rank a contiguous chunk. Each rank's RNG is seeded differently.
- **Doc ID reconstruction**: inline from EOS positions. EOS gets same doc_id as preceding text (matches _DocumentPacker).

---

### Task 3: Update get_batch() auto-detection in data.py

**Files:**
- Modify: `05_phase5_dllm/phase5/data.py` (get_batch function, lines 247-264)

**Step 1: Update loader initialization to support 3 modes**

Replace the current `if _train_loader is None:` block with:

```python
    if split == 'train':
        if _train_loader is None:
            if config.data_dir:
                if os.path.isdir(config.data_dir) and os.path.exists(
                    os.path.join(config.data_dir, 'meta.json')
                ):
                    # Mode 1: local numpy shards (backward compat)
                    _train_loader = _ShardedLoader(
                        config.data_dir,
                        rank=config.ddp_rank,
                        world_size=config.ddp_world_size,
                    )
                    if config.master_process:
                        m = _train_loader.meta
                        print(f'[data] Sharded: {m["n_shards"]} shards, '
                              f'{m["total_tokens"]/1e9:.1f}B tokens')
                else:
                    # Mode 2: HF Hub dataset (or local HF cache)
                    _train_loader = _PreTokenizedPacker(
                        config.data_dir,
                        rank=config.ddp_rank,
                        world_size=config.ddp_world_size,
                    )
                    if config.master_process:
                        print(f'[data] Pre-tokenized: {_train_loader._n:,} docs '
                              f'from {config.data_dir}')
            else:
                # Mode 3: streaming + on-the-fly tokenization
                _train_loader = _DocumentPacker(_make_train_iter)
                if config.master_process:
                    print('[data] Streaming mode (no --data-dir)')
        loader = _train_loader
```

Detection logic:
- `--data-dir /data/tokenized` (local dir with meta.json) → _ShardedLoader (numpy)
- `--data-dir HoangHa/100BT-dLLM-pretokenized` (not a local dir) → _PreTokenizedPacker (HF Hub)
- No `--data-dir` → _DocumentPacker (streaming)

---

### Task 4: Update modal_train.py

**Files:**
- Modify: `modal_train.py`

**Step 1: Rewrite pretokenize() function**

Replace the current pretokenize function (lines 185-215) with:

```python
@app.function(
    image=image,
    cpu=16,
    timeout=86400,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pretokenize(max_docs: int = 0, hub_repo: str = "HoangHa/100BT-dLLM-pretokenized"):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data"
    os.environ["HF_DATASETS_CACHE"] = "/data/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        "python", "/root/05_phase5_dllm/pretokenize.py",
        f"--hub-repo={hub_repo}",
        "--num-proc=16",
    ]
    if max_docs > 0:
        cmd.extend(["--max-docs", str(max_docs)])

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    data_vol.commit()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Cleanup: remove intermediate Arrow cache to free volume space
    import shutil
    cache_dir = "/data/datasets"
    for item in os.listdir(cache_dir):
        path = os.path.join(cache_dir, item)
        if os.path.isdir(path) and "finepdfs" in item:
            print(f"Cleaning up source cache: {path}")
            shutil.rmtree(path)
    data_vol.commit()
```

**Step 2: Update Train.run() auto-detection**

Replace the current auto-detect block (lines 92-95) with:

```python
        # Auto-detect pre-tokenized data
        tokenized_dir = "/data/tokenized"
        hub_repo = "HoangHa/100BT-dLLM-pretokenized"
        if os.path.exists(os.path.join(tokenized_dir, "meta.json")):
            # Local numpy shards (legacy)
            cmd.append(f"--data-dir={tokenized_dir}")
        else:
            # Always try HF Hub dataset — load_dataset handles caching
            cmd.append(f"--data-dir={hub_repo}")
```

**Step 3: Add `huggingface_hub` to pip_install in image**

The image already has `datasets` which depends on `huggingface_hub`. But for `push_to_hub`
with auth, we need the HF token. The `huggingface-secret` Modal secret provides `HF_TOKEN`.
No additional pip install needed.

**Step 4: Update docstring**

Add to the module docstring:
```
    # Pre-tokenize dataset and push to HF Hub
    modal run modal_train.py::pretokenize
    modal run modal_train.py::pretokenize --max-docs 100000
```

---

### Task 5: Local end-to-end test

**Step 1: Test pretokenize on tiny dataset**

Run locally with 100 docs to verify the pipeline:

```bash
uv run python 05_phase5_dllm/pretokenize.py \
    --hub-repo HoangHa/test-dllm-tokenized \
    --max-docs 100 \
    --num-proc 1
```

Expected: pushes ~100 docs to `HoangHa/test-dllm-tokenized`.

**Step 2: Test _PreTokenizedPacker loads from Hub**

```bash
uv run python -c "
import sys
sys.argv = ['test', '--train', '--data-dir=HoangHa/test-dllm-tokenized', '--batch-size=4']
from phase5 import config
from phase5.data import get_batch

x_input, targets, noise_mask, elbo_w, doc_ids, positions = get_batch('train')
print(f'x_input: {x_input.shape}')
print(f'targets: {targets.shape}')
print(f'doc_ids max: {doc_ids.max().item()}')

# Verify [x_t || x_0] structure
x_0 = x_input[:, 2048:].cpu()
assert (x_0 == targets.cpu()).all(), 'x_0 should equal targets'
print('Integration test: PASSED')
"
```

**Step 3: Delete test dataset from Hub**

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().delete_repo('HoangHa/test-dllm-tokenized', repo_type='dataset')
print('Deleted test repo')
"
```

---

### Task 6: Commit

```bash
git add 05_phase5_dllm/pretokenize.py 05_phase5_dllm/phase5/data.py \
        05_phase5_dllm/phase5/config.py modal_train.py \
        docs/plans/2026-03-08-pretokenize-hub-design.md \
        docs/plans/2026-03-08-pretokenize-hub-plan.md
git commit -m "feat: pre-tokenize 100B dataset to HF Hub with datasets.map(num_proc=16)

- Rewrite pretokenize.py: datasets.map() + push_to_hub (uint16, variable-length docs)
- Add _PreTokenizedPacker to data.py: loads HF dataset, packs at train time
- 3 data modes: HF Hub (new), numpy shards (legacy), streaming (default)
- modal_train.py: pretokenize() pushes to Hub, training auto-detects

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
