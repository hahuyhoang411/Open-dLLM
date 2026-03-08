"""Data pipeline with document packing for Phase 5 block diffusion LM.

No right-padding — every position is a real token. Multiple documents packed
per sequence with EOS boundaries, doc-aware attention masking, and RoPE
position reset at document boundaries.

Supports two modes:
1. Streaming: HF dataset -> tokenize on-the-fly -> pack (default, no --data-dir)
2. HF Hub: pre-tokenized HF dataset -> batch random access -> pack (--data-dir with HF repo ID)
"""

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
                _train_loader = _PreTokenizedPacker(
                    config.data_dir,
                    rank=config.ddp_rank,
                    world_size=config.ddp_world_size,
                )
                if config.master_process:
                    print(f'[data] Pre-tokenized: {_train_loader._n:,} docs '
                          f'from {config.data_dir}')
            else:
                _train_loader = _DocumentPacker(_make_train_iter)
                if config.master_process:
                    print('[data] Streaming mode (no --data-dir)')
        loader = _train_loader
    else:
        # Reset val packer each eval to measure on the same data every time.
        # Streaming from .skip(100_000) is deterministic — same seed, same order.
        _val_packer = _DocumentPacker(_make_val_iter)
        loader = _val_packer

    # Build (targets, doc_ids) from packed sequences
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
