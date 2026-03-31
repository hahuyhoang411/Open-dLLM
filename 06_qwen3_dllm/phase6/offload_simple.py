"""Simple CPU offload: raw .to('cpu') for checkpoint saved tensors."""

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def simple_offload_checkpoint(fn, *args, use_reentrant=False, context_fn=None, **kwargs):
  """Drop-in replacement for checkpoint() that moves saved tensors to CPU."""
  # Detect device from args — if not CUDA, fall back to standard checkpoint
  device = None
  for a in args:
    if isinstance(a, torch.Tensor) and a.is_cuda:
      device = a.device
      break

  if device is None:
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)

  def pack_hook(tensor):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
      return tensor
    return tensor.to('cpu', non_blocking=True)

  def unpack_hook(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.is_cuda:
      return tensor
    return tensor.to(device, non_blocking=True)

  with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)
