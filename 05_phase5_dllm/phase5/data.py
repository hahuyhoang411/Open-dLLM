"""Data pipeline with document packing for Phase 5 block diffusion LM.

No right-padding — every position is a real token. Multiple documents packed
per sequence with EOS boundaries, doc-aware attention masking, and RoPE
position reset at document boundaries.
"""

import torch
from datasets import load_dataset

from . import config
from .tokenizer import encode
from .schedule import sample_timesteps, apply_noise, compute_elbo_weight, compute_cart_weights


# ============================================================================
# Streaming Dataset Iterators
# ============================================================================

_DATASET = "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"

_train_packer = None
_val_packer = None


def _make_train_iter():
    ds = load_dataset(_DATASET, split="train", streaming=True)
    return iter(ds.shuffle(seed=42 + config.ddp_rank, buffer_size=10_000))


def _make_val_iter():
    ds = load_dataset(_DATASET, split="train", streaming=True)
    return iter(ds.skip(100_000))


# ============================================================================
# Document Packing
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

        ids = encode(doc["text"])
        # Truncate individual docs to seq_len-1 (leave room for EOS)
        ids = ids[:config.seq_len - 1]
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

def get_batch(split="train"):
    """Build a training batch from packed document streams.

    Returns:
        x_input:   (B, 2L) — [x_t || x_0] concatenation
        targets:   (B, L)  — original unmasked token IDs
        mask:      (B, L)  — True where tokens were noise-masked
        elbo_w:    (B, L)  — ELBO importance weights (1/t or CART)
        doc_ids:   (B, L)  — document index per token (for attention masking)
        positions: (B, L)  — per-token RoPE positions (reset at doc boundaries)
    """
    global _train_packer, _val_packer

    if split == "train":
        if _train_packer is None:
            _train_packer = _DocumentPacker(_make_train_iter)
        packer = _train_packer
    else:
        if _val_packer is None:
            _val_packer = _DocumentPacker(_make_val_iter)
        packer = _val_packer

    # Pull batch_size packed sequences
    all_ids = []
    all_doc_ids = []
    for _ in range(config.batch_size):
        ids, dids = packer.get_sequence()
        all_ids.append(ids)
        all_doc_ids.append(dids)

    targets = torch.tensor(all_ids, dtype=torch.long)       # (B, L)
    doc_ids = torch.tensor(all_doc_ids, dtype=torch.long)    # (B, L)

    # Per-token positions (RoPE reset at doc boundaries)
    positions = _compute_positions(doc_ids)                   # (B, L)

    # Linear noise: sample per-block timesteps, expand to per-token
    t_blocks, t = sample_timesteps(config.batch_size, config.num_blocks, config.block_size)

    # Apply noise (mask_prob = t for linear schedule)
    x_noisy, noise_mask = apply_noise(targets, t)             # (B, L) each

    # ELBO weights
    if config.use_cart:
        # With doc packing, all tokens are real (no padding)
        padding = torch.ones_like(targets, dtype=torch.bool)
        elbo_w = compute_cart_weights(noise_mask, padding)
    else:
        elbo_w = compute_elbo_weight(t)                       # (B, L)

    # Build [x_t || x_0] input (2L tokens)
    x_input = torch.cat([x_noisy, targets], dim=1)           # (B, 2L)

    # Move to device
    x_input = x_input.to(config.device)
    targets = targets.to(config.device)
    noise_mask = noise_mask.to(config.device)
    elbo_w = elbo_w.to(config.device)
    doc_ids = doc_ids.to(config.device)
    positions = positions.to(config.device)

    return x_input, targets, noise_mask, elbo_w, doc_ids, positions
