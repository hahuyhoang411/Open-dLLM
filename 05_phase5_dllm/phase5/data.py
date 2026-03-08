"""Data pipeline with document packing for Phase 5 block diffusion LM.

No right-padding — every position is a real token. Multiple documents packed
per sequence with EOS boundaries, doc-aware attention masking, and RoPE
position reset at document boundaries.

Supports three modes:
1. Streaming: HF dataset -> tokenize on-the-fly -> pack (default, no --data-dir)
2. Sharded: pre-tokenized numpy files -> random-access batches (--data-dir with meta.json)
3. HF Hub: pre-tokenized HF dataset -> batch random access -> pack (--data-dir with HF repo ID)
"""

import glob
import json
import os
import threading

import numpy as np
import torch
from datasets import load_dataset

from . import config
from .schedule import apply_noise, compute_cart_weights, compute_elbo_weight, sample_timesteps
from .tokenizer import encode

# ============================================================================
# Streaming Dataset Iterators
# ============================================================================

_DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'

_train_loader = None
_val_packer = None


def _make_train_iter():
    ds = load_dataset(_DATASET, split='train', streaming=True)
    # DDP: shard so each rank reads disjoint data (avoids 7/8 wasted compute)
    if config.ddp_world_size > 1:
        ds = ds.shard(num_shards=config.ddp_world_size, index=config.ddp_rank)
    return iter(ds.shuffle(seed=42 + config.ddp_rank, buffer_size=10_000))


def _make_val_iter():
    ds = load_dataset(_DATASET, split='train', streaming=True)
    return iter(ds.skip(100_000))


# ============================================================================
# Document Packing (streaming mode)
# ============================================================================

class _DocumentPacker:
    """Packs streaming documents into fixed-length sequences with EOS boundaries.

    Yields (token_ids, doc_ids) tuples, both of length seq_len.
    doc_ids tracks which document each token belongs to (for attention masking).
    """

    def __init__(self, make_iter_fn):
        self._make_iter_fn = make_iter_fn
        self._iter = make_iter_fn()
        self._buf_ids = []
        self._buf_doc_ids = []
        self._current_doc = 0

    def _refill(self):
        """Pull one document into the buffer."""
        try:
            doc = next(self._iter)
        except StopIteration:
            self._iter = self._make_iter_fn()
            doc = next(self._iter)

        ids = encode(doc['text'])
        ids.append(config.eos_token_id)

        self._buf_ids.extend(ids)
        self._buf_doc_ids.extend([self._current_doc] * len(ids))
        self._current_doc += 1

    def get_sequence(self):
        """Return one packed sequence of exactly seq_len tokens."""
        while len(self._buf_ids) < config.seq_len:
            self._refill()

        token_ids = self._buf_ids[:config.seq_len]
        doc_ids = self._buf_doc_ids[:config.seq_len]

        self._buf_ids = self._buf_ids[config.seq_len:]
        self._buf_doc_ids = self._buf_doc_ids[config.seq_len:]

        # Renumber doc_ids to start from 0 within each sequence
        seen = {}
        remapped = []
        counter = 0
        for d in doc_ids:
            if d not in seen:
                seen[d] = counter
                counter += 1
            remapped.append(seen[d])

        return token_ids, remapped


# ============================================================================
# Sharded Loader (pre-tokenized mode)
# ============================================================================

class _ShardedLoader:
    """Fast random-access loader from pre-tokenized numpy shards.

    Shards are .npy files of shape (N, seq_len), dtype uint16.
    Doc boundaries are reconstructed from EOS positions at load time.
    Background thread preloads next shard to hide I/O latency.
    """

    def __init__(self, shard_dir, rank=0, world_size=1, seed=42):
        with open(os.path.join(shard_dir, 'meta.json')) as f:
            self.meta = json.load(f)

        self.eos_id = self.meta['eos_token_id']

        all_shards = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.npy')))
        self.shard_paths = all_shards[rank::world_size]
        assert self.shard_paths, f'No shards for rank {rank}/{world_size} ({len(all_shards)} total)'

        self._rng = np.random.RandomState(seed + rank)
        self._rng.shuffle(self.shard_paths)

        self._shard_idx = 0
        self._row_idx = 0
        self._current_data = None
        self._row_order = None
        self._epoch = 0

        self._next_data = None
        self._preload_thread = None
        self._load_current_shard()
        self._start_preload()

    def _load_current_shard(self):
        self._current_data = np.load(self.shard_paths[self._shard_idx])
        n = self._current_data.shape[0]
        self._row_order = self._rng.permutation(n)
        self._row_idx = 0

    def _start_preload(self):
        """Preload next shard in background thread."""
        next_idx = (self._shard_idx + 1) % len(self.shard_paths)
        path = self.shard_paths[next_idx]

        def _load():
            self._next_data = np.load(path)

        self._preload_thread = threading.Thread(target=_load, daemon=True)
        self._preload_thread.start()

    def _advance_shard(self):
        """Switch to next shard (preloaded in background)."""
        self._shard_idx = (self._shard_idx + 1) % len(self.shard_paths)
        if self._shard_idx == 0:
            self._epoch += 1
            self._rng.shuffle(self.shard_paths)

        if self._preload_thread:
            self._preload_thread.join()

        self._current_data = self._next_data
        n = self._current_data.shape[0]
        self._row_order = self._rng.permutation(n)
        self._row_idx = 0
        self._start_preload()

    def get_batch_tensors(self, batch_size):
        """Return (targets, doc_ids) tensors directly. Vectorized fast path."""
        chunks = []
        remaining = batch_size

        while remaining > 0:
            if self._row_idx >= len(self._row_order):
                self._advance_shard()

            available = len(self._row_order) - self._row_idx
            take = min(available, remaining)
            rows = self._row_order[self._row_idx:self._row_idx + take]
            chunks.append(self._current_data[rows])
            self._row_idx += take
            remaining -= take

        tokens = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        targets = torch.from_numpy(tokens.astype(np.int64))

        # Reconstruct doc_ids from EOS positions
        eos = (targets == self.eos_id)
        shifted = torch.zeros_like(eos)
        shifted[:, 1:] = eos[:, :-1]
        doc_ids = shifted.cumsum(dim=1)

        return targets, doc_ids


# ============================================================================
# Pre-Tokenized HF Hub Loader
# ============================================================================

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
        # Epoch-style iteration: permute indices, iterate without replacement
        self._order = self._rng.permutation(self._n)
        self._cursor = 0

    def _refill(self):
        """Pull next batch of documents (without replacement) into the buffer."""
        if self._cursor >= self._n:
            # Epoch boundary: reshuffle
            self._order = self._rng.permutation(self._n)
            self._cursor = 0
        end = min(self._cursor + 100, self._n)
        indices = self._order[self._cursor:end].tolist()
        self._cursor = end
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


# ============================================================================
# Position Computation (vectorized)
# ============================================================================

def _compute_positions(doc_ids):
    """Compute per-token RoPE positions that reset at document boundaries.

    Args:
        doc_ids: (B, L) int tensor — document index per token.

    Returns:
        positions: (B, L) int tensor — position within each document.
    """
    B, L = doc_ids.shape
    # Detect boundaries: where doc_id changes from previous position
    boundaries = torch.zeros_like(doc_ids, dtype=torch.bool)
    boundaries[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    boundaries[:, 0] = True  # first position is always a boundary

    # Vectorized segmented cumsum via cummax trick:
    # At each boundary, record the position index. cummax propagates
    # the latest boundary index forward. positions = arange - boundary_start.
    arange = torch.arange(L, device=doc_ids.device).unsqueeze(0).expand(B, -1)
    boundary_idx = torch.where(boundaries, arange, torch.zeros_like(arange))
    boundary_starts, _ = boundary_idx.cummax(dim=1)
    return arange - boundary_starts


# ============================================================================
# Batch Construction
# ============================================================================

def get_batch(split='train'):
    """Build a training batch from packed document streams or pre-tokenized shards.

    Returns:
        x_input:   (B, 2L) — [x_t || x_0] concatenation
        targets:   (B, L)  — original unmasked token IDs
        mask:      (B, L)  — True where tokens were noise-masked
        elbo_w:    (B, L)  — ELBO importance weights (1/t or CART)
        doc_ids:   (B, L)  — document index per token (for attention masking)
        positions: (B, L)  — per-token RoPE positions (reset at doc boundaries)
    """
    global _train_loader, _val_packer

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
    else:
        # Reset val packer each eval to measure on the same data every time.
        # Streaming from .skip(100_000) is deterministic — same seed, same order.
        _val_packer = _DocumentPacker(_make_val_iter)
        loader = _val_packer

    # Build (targets, doc_ids) — vectorized for sharded, loop for streaming
    if hasattr(loader, 'get_batch_tensors'):
        targets, doc_ids = loader.get_batch_tensors(config.batch_size)
    else:
        all_ids = []
        all_doc_ids = []
        for _ in range(config.batch_size):
            ids, dids = loader.get_sequence()
            all_ids.append(ids)
            all_doc_ids.append(dids)
        targets = torch.tensor(all_ids, dtype=torch.long)       # (B, L)
        doc_ids = torch.tensor(all_doc_ids, dtype=torch.long)    # (B, L)

    # Per-token positions (RoPE reset at doc boundaries)
    positions = _compute_positions(doc_ids)                       # (B, L)

    # Linear noise: sample per-block timesteps, expand to per-token
    t_blocks, t = sample_timesteps(config.batch_size, config.num_blocks, config.block_size)

    # Apply noise (mask_prob = t for linear schedule)
    x_noisy, noise_mask = apply_noise(targets, t, pad_token_id=config.pad_token_id)

    # ELBO weights
    if config.use_cart:
        # With doc packing, all tokens are real (no padding)
        padding = torch.ones_like(targets, dtype=torch.bool)
        elbo_w = compute_cart_weights(noise_mask, padding)
    else:
        elbo_w = compute_elbo_weight(t)                           # (B, L)

    # Build [x_t || x_0] input (2L tokens)
    x_input = torch.cat([x_noisy, targets], dim=1)               # (B, 2L)

    # Move to device (non_blocking overlaps HtoD with GPU compute)
    _nb = torch.cuda.is_available()
    x_input = x_input.to(config.device, non_blocking=_nb)
    targets = targets.to(config.device, non_blocking=_nb)
    noise_mask = noise_mask.to(config.device, non_blocking=_nb)
    elbo_w = elbo_w.to(config.device, non_blocking=_nb)
    doc_ids = doc_ids.to(config.device, non_blocking=_nb)
    positions = positions.to(config.device, non_blocking=_nb)

    return x_input, targets, noise_mask, elbo_w, doc_ids, positions
