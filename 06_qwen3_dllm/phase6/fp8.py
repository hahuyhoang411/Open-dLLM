"""FP8 training — per-row + per-tensor dynamic scaling, adapted from nanochat.

Uses @allow_in_graph instead of torchao's tensor subclass approach.
torchao + gradient checkpointing causes a 3x slowdown (subclass decomposition
through every recomputed forward). The opaque-node approach avoids this entirely
while calling the same cuBLAS _scaled_mm kernel.

Forward uses per-row (rowwise) scaling for activations and weights — critical for
block diffusion where ELBO weighting creates 10x dynamic range across tokens.
Backward uses per-tensor scaling (transposed matmuls don't support rowwise cleanly).
"""

from contextlib import contextmanager

import torch
import torch.nn as nn

EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype, rowwise=False):
    fp8_max = torch.finfo(fp8_dtype).max
    if rowwise and x.ndim == 2:
        # Per-row: scale each row independently for better precision
        # _scaled_mm requires 2D scales: (M, 1) for scale_a, (1, N) for scale_b
        amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=EPS)
        scale = fp8_max / amax.double()
        scale = scale.float()
        x_fp8 = x.float().mul(scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return x_fp8, scale.reciprocal()  # (M, 1) inverse scale
    else:
        # Per-tensor: single scale for the whole tensor
        amax = x.float().abs().max()
        scale = fp8_max / amax.double().clamp(min=EPS)
        scale = scale.float()
        x_fp8 = x.float().mul(scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return x_fp8, scale.reciprocal()


def _to_col_major(x):
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, weight):
        # Forward: rowwise scaling for precision with ELBO-weighted tokens
        # _scaled_mm rowwise requires scale_a=(M,1), scale_b=(1,N), both float32
        in_fp8_row, in_inv_row = _to_fp8(input_2d, torch.float8_e4m3fn, rowwise=True)
        w_fp8_row, w_inv_row = _to_fp8(weight, torch.float8_e4m3fn, rowwise=True)
        # w_fp8_row.t() is (K,N), scale_b = w_inv_row.T is (1,N)
        out = torch._scaled_mm(
            in_fp8_row, w_fp8_row.t(),
            scale_a=in_inv_row, scale_b=w_inv_row.t(),
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )

        # Backward: per-tensor (transposed matmuls don't support rowwise)
        in_fp8, in_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        w_fp8, w_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(in_fp8, in_inv, w_fp8, w_inv)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # grad_input = grad_output @ weight  (e5m2 for grads, e4m3 for weights)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        grad_input = torch._scaled_mm(
            go_fp8, _to_col_major(w_fp8),
            scale_a=go_inv, scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # grad_weight = grad_output.T @ input
        grad_weight = torch._scaled_mm(
            go_fp8.t().contiguous(), _to_col_major(in_fp8),
            scale_a=go_inv, scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )
        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear that does FP8 matmuls. Weights stay in original precision."""

    def forward(self, input):
        input = input.to(torch.bfloat16)
        shape = input.shape
        output = _Float8Matmul.apply(input.reshape(-1, shape[-1]), self.weight)
        output = output.reshape(*shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        with torch.device("meta"):
            new = cls(mod.in_features, mod.out_features, bias=False)
        new.weight = mod.weight
        new.bias = mod.bias
        return new


def convert_to_float8_training(module, module_filter_fn=None):
    """Replace nn.Linear with Float8Linear throughout module (post-order walk).

    Shares weight/bias tensors — no copies.
    module_filter_fn: optional (module, fqn) -> bool filter.
    """
    def _walk(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _walk(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))
    _walk(module)
    return module


@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear -> nn.Linear for eval/generation. Restores on exit."""
    swapped = []
    # Collect first — named_modules() is a live generator, can't mutate during iteration
    fp8_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, Float8Linear)]
    for name, mod in fp8_modules:
            # Find parent + attr name
            parts = name.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            attr = parts[-1]
            # Build plain Linear on meta, share params
            with torch.device("meta"):
                lin = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            lin.weight = mod.weight
            lin.bias = mod.bias
            setattr(parent, attr, lin)
            swapped.append((parent, attr, mod))
    try:
        yield model
    finally:
        for parent, attr, fp8_mod in swapped:
            setattr(parent, attr, fp8_mod)
