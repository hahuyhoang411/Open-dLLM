"""Loss computation for Phase 5 block diffusion LM.

Uses chunked CE with gradient checkpointing to avoid materializing the full
(B*L, vocab) logits tensor (~24 GB at batch=64, vocab=49152).

Note: Liger FLCE reduction='none' backward is broken (linkedin/Liger-Kernel#488).
Forward produces correct loss values but backward returns zero gradients.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from . import config

_CHUNK_SIZE = 16384  # tokens per chunk — peak ~1.5 GB bf16 for vocab=49152
# 16384 = 4x fewer chunks than 4096 → fewer kernel launches + Python overhead
# Each chunk: (16384, 49152) logits = 1.5 GB bf16, well within GPU headroom
# Recomputed during backward via grad_checkpoint (no persistent VRAM cost)


def _chunk_ce(h_chunk, weight, t_chunk, w_chunk):
    """Weighted CE for one token chunk. Re-executed during backward via grad_checkpoint."""
    logits = h_chunk @ weight.T  # (chunk, vocab)
    ce = F.cross_entropy(logits, t_chunk, reduction='none')  # (chunk,)
    return (ce * w_chunk).sum()


def compute_loss(hidden_states, targets, mask, elbo_weight, lm_head_weight, use_liger=False):
    """Compute weighted cross-entropy loss for masked diffusion.

    Normalizes by ALL real tokens (not just masked). The 1/t ELBO weight already
    accounts for the masking fraction. Ref: ZHZisZZ/dllm, Phase 4 fix [P4-19].
    """
    B, L, D = hidden_states.shape
    h_flat = hidden_states.contiguous().view(-1, D)
    t_flat = targets.contiguous().view(-1)
    w_flat = (mask.float() * elbo_weight).contiguous().view(-1)

    N = h_flat.shape[0]
    losses = []
    for i in range(0, N, _CHUNK_SIZE):
        j = min(i + _CHUNK_SIZE, N)
        chunk_loss = grad_checkpoint(
            _chunk_ce,
            h_flat[i:j], lm_head_weight, t_flat[i:j], w_flat[i:j],
            use_reentrant=False,
        )
        losses.append(chunk_loss)

    total_loss = torch.stack(losses).sum()
    real_count = (targets != config.pad_token_id).float().sum().clamp(min=1)
    return total_loss / real_count
