"""Sqrt-SAC: checkpoint only at sqrt(N) layer boundaries to reduce recompute overhead."""

import math

from torch.utils.checkpoint import checkpoint as grad_checkpoint


def get_sqrt_checkpoint_indices(n_layers, method='sqrt'):
  """Return sorted list of layer indices that should be checkpointed.

  method='sqrt': checkpoint every ceil(sqrt(N)) layers
  method=int: checkpoint every `method` layers
  Always includes layer 0.
  """
  if isinstance(method, int):
    step = max(1, method)
  elif method == 'sqrt':
    step = max(1, int(math.ceil(math.sqrt(n_layers))))
  else:
    raise ValueError(f'Unknown method: {method}')

  indices = list(range(0, n_layers, step))
  return sorted(set(indices))


def sqrt_sac_forward(block, block_idx, checkpoint_indices, *args, context_fn=None, **kwargs):
  """Run block forward, checkpointing only at checkpoint boundary layers.

  Layers at boundaries: full SAC (save matmuls, recompute norms).
  Other layers: standard forward (no recompute, more memory).
  """
  if block_idx in checkpoint_indices:
    ckpt_kwargs = dict(use_reentrant=False)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(block._forward, *args, **ckpt_kwargs)
  return block._forward(*args)
