"""Smart CPU offload: pinned buffers + async CUDA stream for zero-overhead DMA."""

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Module-level state (lazily initialized per device)
_CPU_TENSORS: dict = {}  # unique_id -> pinned CPU tensor
_EXTRA_STREAMS: dict = {}  # device_idx -> Stream
_INITIALIZED_DEVICES: set = set()
_MIN_OFFLOAD_BYTES = 2 * 1024 * 1024  # 2 MB threshold
_tensor_counter = 0  # global counter for unique IDs


def _ensure_device_initialized(device):
  global _INITIALIZED_DEVICES
  dev_idx = _get_dev_idx(device)
  if dev_idx in _INITIALIZED_DEVICES:
    return
  _EXTRA_STREAMS[dev_idx] = torch.cuda.Stream(device=device)
  _INITIALIZED_DEVICES.add(dev_idx)


def _get_dev_idx(device):
  if hasattr(device, 'index') and device.index is not None:
    return device.index
  return 0


def reset_offload_state():
  """Free all pinned buffers and reset global state. Call between runs or in tests."""
  global _CPU_TENSORS, _EXTRA_STREAMS, _INITIALIZED_DEVICES, _tensor_counter
  _CPU_TENSORS.clear()
  _EXTRA_STREAMS.clear()
  _INITIALIZED_DEVICES.clear()
  _tensor_counter = 0


def smart_offload_checkpoint(fn, *args, use_reentrant=False, context_fn=None, is_last_layer=False, **kwargs):
  """Drop-in replacement for checkpoint() with pinned-buffer async offload."""
  # Detect device from args
  device = None
  for a in args:
    if isinstance(a, torch.Tensor) and a.is_cuda:
      device = a.device
      break

  # CPU or last layer: standard checkpoint (no offload benefit)
  if device is None or is_last_layer:
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)

  _ensure_device_initialized(device)
  dev_idx = _get_dev_idx(device)
  extra_stream = _EXTRA_STREAMS[dev_idx]

  def pack_hook(tensor):
    global _tensor_counter
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
      return tensor
    if tensor.nbytes < _MIN_OFFLOAD_BYTES:
      return tensor

    # Unique ID per tensor — no index collisions across blocks
    tid = _tensor_counter
    _tensor_counter += 1

    # Allocate pinned CPU buffer for this tensor
    cpu_buf = torch.empty(tensor.numel(), dtype=tensor.dtype, device='cpu', pin_memory=True)
    flat = cpu_buf.view(tensor.shape)

    # Async copy: sync extra stream with the *current* stream (may be backward
    # stream during recomputation, not the forward-time main_stream)
    extra_stream.wait_stream(torch.cuda.current_stream(tensor.device))
    with torch.cuda.stream(extra_stream):
      flat.copy_(tensor, non_blocking=True)

    _CPU_TENSORS[tid] = flat
    return (tid, tensor.shape, tensor.dtype, tensor.device)

  def unpack_hook(packed):
    if isinstance(packed, torch.Tensor):
      return packed  # was not offloaded (small tensor)

    tid, shape, dtype, orig_device = packed
    cpu_data = _CPU_TENSORS.pop(tid)  # remove to free pinned memory after use

    cur_main = torch.cuda.current_stream(orig_device)
    es = _EXTRA_STREAMS[_get_dev_idx(orig_device)]

    # Allocate fresh GPU tensor (no aliasing — each unpack gets its own)
    gpu_tensor = torch.empty(shape, dtype=dtype, device=orig_device)

    es.wait_stream(cur_main)
    with torch.cuda.stream(es):
      gpu_tensor.copy_(cpu_data, non_blocking=True)
    cur_main.wait_stream(es)

    return gpu_tensor

  with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    ckpt_kwargs = dict(use_reentrant=use_reentrant)
    if context_fn is not None:
      ckpt_kwargs['context_fn'] = context_fn
    return grad_checkpoint(fn, *args, **ckpt_kwargs, **kwargs)
