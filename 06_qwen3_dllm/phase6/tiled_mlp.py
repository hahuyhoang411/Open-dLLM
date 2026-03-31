"""TiledMLP: chunked MLP forward/backward to reduce intermediate activation memory."""

import torch
from torch.autograd import Function


class TiledMLP(Function):
  """Process MLP in chunks along flattened (B*T) dimension.

  Forward: split input, run MLP per chunk, concat output.
  Backward: recompute forward per chunk, get grads via torch.autograd.grad().
  """

  @staticmethod
  def forward(ctx, x, gate_weight, up_weight, down_weight, chunk_size, _use_liger_unused):
    # Save inputs for backward recomputation (NOT intermediates)
    ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
    ctx.chunk_size = chunk_size

    BT, D = x.shape
    output = torch.empty(BT, D, dtype=x.dtype, device=x.device)

    for start in range(0, BT, chunk_size):
      end = min(start + chunk_size, BT)
      x_chunk = x[start:end]
      gate = x_chunk @ gate_weight.T
      up = x_chunk @ up_weight.T
      # Always use native silu in tiled path — ensures backward recompute matches
      activated = torch.nn.functional.silu(gate) * up
      output[start:end] = activated @ down_weight.T
      # gate, up, activated freed here per chunk

    return output

  @staticmethod
  def backward(ctx, grad_output):
    x, gate_weight, up_weight, down_weight = ctx.saved_tensors
    chunk_size = ctx.chunk_size

    BT = x.shape[0]

    grad_x = torch.zeros_like(x)
    grad_gate_w = torch.zeros_like(gate_weight)
    grad_up_w = torch.zeros_like(up_weight)
    grad_down_w = torch.zeros_like(down_weight)

    for start in range(0, BT, chunk_size):
      end = min(start + chunk_size, BT)
      go_chunk = grad_output[start:end]

      # Detach and enable grad for recomputation graph
      x_c = x[start:end].detach().requires_grad_(True)
      gw = gate_weight.detach().requires_grad_(True)
      uw = up_weight.detach().requires_grad_(True)
      dw = down_weight.detach().requires_grad_(True)

      # Recompute forward under enable_grad — matches forward exactly (native silu)
      with torch.enable_grad():
        gate = x_c @ gw.T
        up = x_c @ uw.T
        activated = torch.nn.functional.silu(gate) * up
        out_chunk = activated @ dw.T

      grads = torch.autograd.grad(
        out_chunk,
        [x_c, gw, uw, dw],
        grad_outputs=go_chunk,
      )

      grad_x[start:end] = grads[0]
      grad_gate_w += grads[1]
      grad_up_w += grads[2]
      grad_down_w += grads[3]

    return grad_x, grad_gate_w, grad_up_w, grad_down_w, None, None


def _auto_chunk_size(BT, intermediate_size, element_size=2, target_gb=0.5):
  """Compute chunk size keeping MLP intermediates under target_gb.

  MLP intermediates per chunk: 3 * chunk * intermediate_size * element_size
  (gate, up, activated tensors)
  """
  target_bytes = int(target_gb * 1024**3)
  bytes_per_token = 3 * intermediate_size * element_size
  chunk = max(1, target_bytes // bytes_per_token)
  return min(chunk, BT)


def tiled_mlp_forward(mlp_module, x, chunk_size=None):
  """Convenience wrapper for TiledMLP.

  mlp_module: SwiGLU instance (has gate_proj, up_proj, down_proj, cfg, drop)
  x: (B, T, D) input (already normed by caller)
  chunk_size: tokens per chunk (None/0 = auto)
  """
  B, T, D = x.shape
  BT = B * T
  x_flat = x.reshape(BT, D)

  if chunk_size is None or chunk_size <= 0:
    chunk_size = _auto_chunk_size(BT, mlp_module.gate_proj.out_features, x.element_size())
  chunk_size = max(1, min(chunk_size, BT))

  out_flat = TiledMLP.apply(
    x_flat,
    mlp_module.gate_proj.weight,
    mlp_module.up_proj.weight,
    mlp_module.down_proj.weight,
    chunk_size,
    False,  # unused liger flag — always use native silu for consistency
  )
  # Apply dropout after tiled forward (H3 fix: SwiGLU.forward wraps down_proj with drop)
  out_flat = mlp_module.drop(out_flat)
  return out_flat.reshape(B, T, D)
