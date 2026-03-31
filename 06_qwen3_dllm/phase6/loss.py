"""Loss computation for Phase 6 block diffusion LM.

Two paths:
1. Chunked CE with gradient checkpointing (fallback, always available)
2. Liger FLCE — fused linear + cross-entropy, never materializes logits.
   The reduction='none' backward bug (linkedin/Liger-Kernel#488) was fixed
   in v0.5.9+ (PR #496, merged Dec 28 2024).
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
      h_flat[i:j],
      lm_head_weight,
      t_flat[i:j],
      w_flat[i:j],
      use_reentrant=False,
    )
    losses.append(chunk_loss)

  total_loss = torch.stack(losses).sum()
  real_count = (targets != cfg.pad_token_id).float().sum().clamp(min=1)
  return total_loss / real_count


def compute_loss_chunked_eager(hidden_states, targets, mask, elbo_weight, lm_head_weight, cfg):
  """Chunked CE with per-chunk backward — frees each chunk's graph eagerly.

  Does NOT return a loss tensor. Gradients are applied directly to hidden_states.
  Returns scalar loss value (float) for logging only.

  The caller MUST NOT call .backward() on the return value — it is already done.
  This is an alternative to compute_loss for memory-constrained setups, not a replacement.
  """
  B, L, D = hidden_states.shape
  vocab_size = lm_head_weight.shape[0]
  chunk_size = _compute_chunk_size(vocab_size)

  h_flat = hidden_states.contiguous().view(-1, D)
  t_flat = targets.contiguous().view(-1)
  w_flat = (mask.float() * elbo_weight).contiguous().view(-1)

  N = h_flat.shape[0]
  real_count = (targets != cfg.pad_token_id).float().sum().clamp(min=1)

  total_loss = 0.0
  for i in range(0, N, chunk_size):
    j = min(i + chunk_size, N)
    chunk_loss = grad_checkpoint(
      _chunk_ce,
      h_flat[i:j],
      lm_head_weight,
      t_flat[i:j],
      w_flat[i:j],
      use_reentrant=False,
    )
    scaled = chunk_loss / real_count
    scaled.backward()  # graph freed immediately after each chunk
    total_loss += chunk_loss.detach().item() / real_count.item()

  return total_loss


def compute_loss_flce(hidden_states, targets, mask, elbo_weight, lm_head_weight, cfg):
  """Compute weighted CE via Liger fused linear cross-entropy.

  Replaces the chunked path — no logit materialization, no grad_checkpoint
  per chunk. The Liger kernel tiles internally.

  API: LigerFusedLinearCrossEntropyLoss(reduction='none')
       forward(lin_weight, _input, target) → (N,) per-token CE
  """
  from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

  B, L, D = hidden_states.shape
  h_flat = hidden_states.contiguous().view(-1, D)  # (N, D)
  t_flat = targets.contiguous().view(-1)  # (N,)
  w_flat = (mask.float() * elbo_weight).contiguous().view(-1)  # (N,)

  flce = LigerFusedLinearCrossEntropyLoss(reduction='none')
  # Liger forward signature: (lin_weight, _input, target)
  per_token_loss = flce(lm_head_weight, h_flat, t_flat)  # (N,)

  weighted = per_token_loss * w_flat
  real_count = (targets != cfg.pad_token_id).float().sum().clamp(min=1)
  return weighted.sum() / real_count
