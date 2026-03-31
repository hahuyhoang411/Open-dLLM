"""Data pipeline with document packing for Phase 6 block diffusion LM.

No right-padding — every position is a real token. Multiple documents packed
per sequence with EOS boundaries, doc-aware attention masking, and RoPE
position reset at document boundaries.

Key change from Phase 5: per-document t sampling (SDAR pattern) instead of
per-batch. Each packed document gets its own noise rate.

Supports two modes:
1. Pre-tokenized: HF Hub Arrow files -> batch random access -> pack (default, --data-dir)
2. Streaming: on-the-fly tokenization -> pack (fallback, no --data-dir)

All functions take cfg parameter — no module-level config imports.
"""

import pathlib
import queue as _queue
import threading

import numpy as np
import torch
from datasets import load_dataset

from .schedule import apply_noise, compute_cart_weights, compute_elbo_weight
from .tokenizer import encode

# ============================================================================
# Streaming Dataset Iterators
# ============================================================================

_DATASET = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'

_loaders = {}  # keyed by (split, rank) to avoid globals


def _make_train_iter(cfg):
  ds = load_dataset(_DATASET, split='train', streaming=True)
  if cfg.ddp_world_size > 1:
    ds = ds.shard(num_shards=cfg.ddp_world_size, index=cfg.ddp_rank)
  return iter(ds.shuffle(seed=42 + cfg.ddp_rank, buffer_size=10_000))


def _make_val_iter(cfg):
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

  def __init__(self, make_iter_fn, cfg):
    self._make_iter_fn = make_iter_fn
    self._cfg = cfg
    self._iter = make_iter_fn(cfg)
    self._buf_ids = []
    self._buf_doc_ids = []
    self._current_doc = 0

  def _refill(self):
    try:
      doc = next(self._iter)
    except StopIteration:
      self._iter = self._make_iter_fn(self._cfg)
      doc = next(self._iter)

    ids = encode(doc['text'])
    ids.append(self._cfg.eos_token_id)

    self._buf_ids.extend(ids)
    self._buf_doc_ids.extend([self._current_doc] * len(ids))
    self._current_doc += 1

  def get_sequence(self):
    while len(self._buf_ids) < self._cfg.seq_len:
      self._refill()

    token_ids = self._buf_ids[: self._cfg.seq_len]
    doc_ids = self._buf_doc_ids[: self._cfg.seq_len]

    self._buf_ids = self._buf_ids[self._cfg.seq_len :]
    self._buf_doc_ids = self._buf_doc_ids[self._cfg.seq_len :]

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
  Packing into cfg.seq_len happens here — works for any seq_len.
  """

  def __init__(self, dataset_id, cfg, seed=42):
    import os

    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    from datasets import load_dataset as _load_ds

    rank = cfg.ddp_rank
    world_size = cfg.ddp_world_size

    if pathlib.Path(dataset_id).is_dir():
      shard_dirs = sorted(
        d for d in (os.path.join(dataset_id, x) for x in os.listdir(dataset_id)) if pathlib.Path(d).is_dir()
      )
      if shard_dirs:
        ds = concatenate_datasets([load_from_disk(s) for s in shard_dirs])
      else:
        ds = load_from_disk(dataset_id)
    else:
      loaded = _load_ds(dataset_id)
      if isinstance(loaded, DatasetDict):
        if 'train' in loaded:
          ds = loaded['train']
        else:
          split_names = sorted((k for k in loaded.keys() if str(k).startswith('train_')), key=str)
          if not split_names:
            raise RuntimeError(f'No train or train_* splits found in {dataset_id}')
          ds = concatenate_datasets([loaded[k] for k in split_names])
      else:
        ds = _load_ds(dataset_id, split='train')

    if world_size > 1:
      ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)

    self._ds = ds
    self._n = len(ds)
    self._cfg = cfg
    self._rng = np.random.RandomState(seed + rank)
    self._buf = []
    self._order = self._rng.permutation(self._n)
    self._cursor = 0

    # Sanity check: dataset token IDs must be within model vocab range
    sample_ids = ds[0]['input_ids'][:100] if len(ds) > 0 else []
    if sample_ids and max(sample_ids) >= cfg.vocab_size:
      raise ValueError(
        f'Dataset token IDs (max={max(sample_ids)}) exceed vocab_size={cfg.vocab_size}. '
        f'Dataset was likely tokenized with a different tokenizer. '
        f'Re-tokenize with Qwen3 tokenizer for Phase 6.'
      )

  def _refill(self):
    if self._cursor >= self._n:
      self._order = self._rng.permutation(self._n)
      self._cursor = 0
    end = min(self._cursor + 100, self._n)
    indices = self._order[self._cursor : end].tolist()
    self._cursor = end
    batch = self._ds[indices]
    for ids in batch['input_ids']:
      self._buf.extend(ids)  # already includes EOS

  def get_sequence(self):
    while len(self._buf) < self._cfg.seq_len:
      self._refill()

    token_ids = self._buf[: self._cfg.seq_len]
    self._buf = self._buf[self._cfg.seq_len :]

    # Reconstruct doc_ids from EOS positions
    doc_ids = []
    doc_id = 0
    for tid in token_ids:
      doc_ids.append(doc_id)
      if tid == self._cfg.eos_token_id:
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
# Per-Document Noise Sampling (SDAR pattern)
# ============================================================================


def _sample_t_per_doc(doc_ids, t_min=0.1):
  """Sample a separate t ~ U[t_min, 1) for each document within packed sequences.

  Vectorized: one gather op instead of Python loop over docs.

  Args:
      doc_ids: (B, L) int tensor — document index per token.
      t_min: minimum noise rate.

  Returns:
      t: (B, L) float tensor — per-token noise rate (constant within each doc).
  """
  B, _L = doc_ids.shape
  max_doc = doc_ids.max().item() + 1
  # Sample one t per (batch, doc) then scatter to token positions
  t_per_doc = t_min + (1.0 - t_min) * torch.rand(B, max_doc)
  t = t_per_doc.gather(1, doc_ids.long())
  return t


def _apply_noise_per_doc(targets, doc_ids, cfg):
  """Apply per-document noise: each doc gets its own t, then mask accordingly.

  Returns:
      x_noisy: (B, L) — targets with mask tokens applied
      noise_mask: (B, L) bool — True where tokens were masked
      t: (B, L) float — per-token noise rate (constant within each doc)
  """
  t = _sample_t_per_doc(doc_ids, t_min=cfg.t_min)
  x_noisy, noise_mask = apply_noise(
    targets,
    t,
    mask_token_id=cfg.mask_token_id,
    pad_token_id=cfg.pad_token_id,
    block_size=cfg.block_size,
  )
  return x_noisy, noise_mask, t


# ============================================================================
# Batch Construction
# ============================================================================


def get_batch(split, cfg, loader=None):
  """Build a training batch from packed document streams or pre-tokenized shards.

  Args:
      split: 'train' or 'val'
      cfg: Config dataclass
      loader: optional packer override (for testing)

  Returns:
      x_input:   (B, 2L) — [x_t || x_0] concatenation
      targets:   (B, L)  — original unmasked token IDs
      mask:      (B, L)  — True where tokens were noise-masked
      elbo_w:    (B, L)  — ELBO importance weights (1/t or CART)
      doc_ids:   (B, L)  — document index per token (for attention masking)
      positions: (B, L)  — per-token RoPE positions (reset at doc boundaries)
  """
  if loader is None:
    loader = _get_or_create_loader(split, cfg)

  # Build (targets, doc_ids) from packed sequences
  all_ids = []
  all_doc_ids = []
  for _ in range(cfg.batch_size):
    ids, dids = loader.get_sequence()
    all_ids.append(ids)
    all_doc_ids.append(dids)
  targets = torch.tensor(all_ids, dtype=torch.long)  # (B, L)
  doc_ids = torch.tensor(all_doc_ids, dtype=torch.long)  # (B, L)

  # Per-token positions (RoPE reset at doc boundaries)
  positions = _compute_positions(doc_ids)  # (B, L)

  # Per-document noise: each doc in the packed sequence gets its own t
  x_noisy, noise_mask, t = _apply_noise_per_doc(targets, doc_ids, cfg)

  # ELBO weights
  elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)  # (B, L) — 1/t base weight
  if cfg.use_cart:
    padding = torch.ones_like(targets, dtype=torch.bool)  # doc packing = no padding
    elbo_w = elbo_w * compute_cart_weights(noise_mask, padding, p=cfg.cart_p)

  # Build [x_t || x_0] input (2L tokens)
  x_input = torch.cat([x_noisy, targets], dim=1)  # (B, 2L)

  # Move to device (pin_memory + non_blocking overlaps HtoD with GPU compute)
  use_cuda = torch.cuda.is_available()
  if use_cuda:
    x_input = x_input.pin_memory().to(cfg.device, non_blocking=True)
    targets = targets.pin_memory().to(cfg.device, non_blocking=True)
    noise_mask = noise_mask.pin_memory().to(cfg.device, non_blocking=True)
    elbo_w = elbo_w.pin_memory().to(cfg.device, non_blocking=True)
    doc_ids = doc_ids.pin_memory().to(cfg.device, non_blocking=True)
    positions = positions.pin_memory().to(cfg.device, non_blocking=True)
  else:
    x_input = x_input.to(cfg.device)
    targets = targets.to(cfg.device)
    noise_mask = noise_mask.to(cfg.device)
    elbo_w = elbo_w.to(cfg.device)
    doc_ids = doc_ids.to(cfg.device)
    positions = positions.to(cfg.device)

  return x_input, targets, noise_mask, elbo_w, doc_ids, positions


def _get_or_create_loader(split, cfg):
  """Lazy-init loader, keyed by (split, rank) to support DDP."""
  key = (split, cfg.ddp_rank)
  if key not in _loaders:
    if split == 'train':
      if cfg.data_dir:
        _loaders[key] = _PreTokenizedPacker(cfg.data_dir, cfg)
        if cfg.master_process:
          print(f'[data] Pre-tokenized: {_loaders[key]._n:,} docs from {cfg.data_dir}')
      else:
        _loaders[key] = _DocumentPacker(_make_train_iter, cfg)
        if cfg.master_process:
          print('[data] Streaming mode (no --data-dir)')
    else:
      _loaders[key] = _DocumentPacker(_make_val_iter, cfg)
  return _loaders[key]


def reset_val_loader(cfg):
  """Reset validation loader for a fresh eval pass."""
  key = ('val', cfg.ddp_rank)
  _loaders.pop(key, None)


# ============================================================================
# Batch Prefetcher
# ============================================================================


class BatchPrefetcher:
  """Background thread that pre-builds batches while GPU computes.

  Usage:
      pf = BatchPrefetcher(lambda: get_batch('train', cfg), maxsize=2)
      for step in range(N):
          batch = pf.get()
          # ... forward, backward ...
      pf.stop()
  """

  def __init__(self, make_batch_fn, maxsize=2):
    self._q = _queue.Queue(maxsize=maxsize)
    self._stop = threading.Event()
    self._fn = make_batch_fn
    self._error = None
    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

  def _worker(self):
    while not self._stop.is_set():
      try:
        batch = self._fn()
        self._q.put(batch, timeout=1.0)
      except _queue.Full:
        continue
      except Exception as e:
        self._error = e
        break

  def get(self, timeout=60.0):
    if self._error is not None:
      raise self._error
    return self._q.get(timeout=timeout)

  def stop(self):
    self._stop.set()
    self._thread.join(timeout=5.0)
