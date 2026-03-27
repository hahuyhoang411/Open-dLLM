"""Loss computation for Phase 6 block diffusion LM.

Chunked CE with gradient checkpointing. Chunk size auto-scales based on
vocab_size to keep peak memory ~1.5 GB per chunk in bf16.

Liger FLCE reduction='none' backward is broken (linkedin/Liger-Kernel#488) —
this module uses plain F.cross_entropy instead.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

_TARGET_BYTES = 1_500_000_000  # ~1.5 GB peak per chunk in bf16


def _compute_chunk_size(vocab_size: int) -> int:
    """Auto-scale chunk_size so logits tensor stays under _TARGET_BYTES."""
    return max(1024, _TARGET_BYTES // (vocab_size * 2))


def _chunk_ce(h_chunk, weight, t_chunk, w_chunk):
    """Weighted CE for one token chunk. Re-executed during backward via grad_checkpoint."""
    logits = h_chunk @ weight.T  # (chunk, vocab)
    ce = F.cross_entropy(logits, t_chunk, reduction='none')  # (chunk,)
    return (ce * w_chunk).sum()


def compute_loss(hidden_states, targets, mask, elbo_weight, lm_head_weight, cfg):
    """Compute weighted cross-entropy loss for masked diffusion.

    Normalizes by ALL real tokens (not just masked). The 1/t ELBO weight
    already accounts for masking fraction.
    """
    B, L, D = hidden_states.shape
    vocab_size = lm_head_weight.shape[0]
    chunk_size = _compute_chunk_size(vocab_size)

    h_flat = hidden_states.contiguous().view(-1, D)
    t_flat = targets.contiguous().view(-1)
    w_flat = (mask.float() * elbo_weight).contiguous().view(-1)

    N = h_flat.shape[0]
    losses = []
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        chunk_loss = grad_checkpoint(
            _chunk_ce,
            h_flat[i:j], lm_head_weight, t_flat[i:j], w_flat[i:j],
            use_reentrant=False,
        )
        losses.append(chunk_loss)

    total_loss = torch.stack(losses).sum()
    real_count = (targets != cfg.pad_token_id).float().sum().clamp(min=1)
    return total_loss / real_count
