"""Activation compression: save checkpoint tensors in FP8 to halve memory."""

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# FP8 dtype (only available on recent torch + compatible hardware)
_FP8_DTYPE = None


def _get_fp8_dtype():
  global _FP8_DTYPE
  if _FP8_DTYPE is None:
    try:
      _FP8_DTYPE = torch.float8_e4m3fn
      # Verify it works
      torch.tensor([1.0], dtype=_FP8_DTYPE, device='cpu')
    except (AttributeError, RuntimeError):
      _FP8_DTYPE = False  # Not available
  return _FP8_DTYPE if _FP8_DTYPE is not False else None


def _pack_fp8(tensor):
  """Quantize tensor to FP8 with per-tensor scale."""
  fp8_dtype = _get_fp8_dtype()
  if fp8_dtype is None or not tensor.is_floating_point() or tensor.numel() < 1024:
    return tensor

  # Skip tensors with NaN/Inf or zero — FP8 can't represent these safely
  amax = tensor.abs().max()
  if not torch.isfinite(amax) or amax == 0:
    return tensor

  # float8_e4m3fn max value is 448.0
  scale = amax / 448.0
  scale = torch.clamp(scale, min=1e-12)

  quantized = (tensor / scale).to(fp8_dtype)
  return (quantized, scale, tensor.dtype, tensor.shape)


def _unpack_fp8(packed):
  """Dequantize FP8 tensor back to original dtype."""
  if isinstance(packed, torch.Tensor):
    return packed

  quantized, scale, orig_dtype, orig_shape = packed
  return (quantized.to(orig_dtype) * scale).reshape(orig_shape)


def compressed_checkpoint(fn, *args, use_reentrant=False, context_fn=None, **kwargs):
  """Drop-in replacement for checkpoint() with FP8 activation compression.

  Uses saved_tensors_hooks to compress saved activations to FP8.
  Falls back to standard checkpoint if FP8 not available.
  """
  fp8_dtype = _get_fp8_dtype()
  if fp8_dtype is None:
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)

  with torch.autograd.graph.saved_tensors_hooks(_pack_fp8, _unpack_fp8):
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)
