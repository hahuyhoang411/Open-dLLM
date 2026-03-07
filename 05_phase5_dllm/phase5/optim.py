"""
Self-contained MuonClip optimizer with QK-Clip + AdamW fallback.

No external dependencies — only stdlib (math, threading) + torch.
Works in both single-GPU and DDP (no dist.all_gather dependency).
"""

import math
import threading
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


# ============================================================================
# Max Attention Logits Tracker (SDPA monkey-patch)
# ============================================================================

class _MaxLogitsTracker:
    """Track per-step max attention logit for QK-Clip.

    attention.py calls _update() directly with ||q||*||k||/sqrt(d) before
    any SDPA/FlexAttention dispatch. This works for all backends.

    Global scalar (not per-layer or per-head). Thread-safe via threading.local.
    """

    _tls = threading.local()

    @classmethod
    def _update(cls, v: float):
        cur = getattr(cls._tls, "max_logits", None)
        if cur is None or v > cur:
            cls._tls.max_logits = v

    @classmethod
    def enable(cls):
        pass  # tracking handled by attention.py direct call

    @classmethod
    def consume(cls):
        """Return and reset the tracked max logit value."""
        v = getattr(cls._tls, "max_logits", None)
        cls._tls.max_logits = None
        return v


# ============================================================================
# MuonClip Optimizer
# ============================================================================

class MuonClip(Optimizer):
    """Unified optimizer: Muon (Newton-Schulz) for 2D weights, AdamW for the rest.

    Per-group flags:
        apply_muon=True  → momentum + NS orthogonalization + RMS scaling
        apply_muon=False → standard AdamW with bias correction

    QK-Clip (is_qk=True groups only): if max attention logit > tau,
    scale weight and update by sqrt(tau / max_logit).
    """

    def __init__(self, params, **defaults):
        defaults.setdefault("lr", 0.02)
        defaults.setdefault("momentum", 0.95)
        defaults.setdefault("weight_decay", 0.1)
        defaults.setdefault("betas", (0.9, 0.95))
        defaults.setdefault("eps", 1e-8)
        super().__init__(params, defaults)
        _MaxLogitsTracker.enable()

    @staticmethod
    @torch.no_grad()
    def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """Moonlight polynomial Newton-Schulz iteration for approximate orthogonalization."""
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = G.bfloat16() / (G.norm() + eps)
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if transposed:
            X = X.T
        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        max_logits = _MaxLogitsTracker.consume()

        for group in self.param_groups:
            apply_muon = group.get("apply_muon", True)
            lr = group["lr"]
            wd = group["weight_decay"]

            if apply_muon:
                self._muon_step(group, lr, wd, max_logits)
            else:
                self._adamw_step(group, lr, wd)

        return loss

    def _muon_step(self, group, lr, wd, max_logits):
        momentum = group.get("momentum", 0.95)
        is_qk = group.get("is_qk", False)
        tau = group.get("qk_clip_tau", 100.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(p.grad)

            # Newton-Schulz orthogonalization + RMS scaling
            orth = self.newton_schulz(buf)
            n, m = p.shape[0], p.shape[1]
            rms_scale = math.sqrt(max(n, m)) * 0.2
            update = orth * rms_scale

            # Decoupled weight decay
            if wd != 0:
                p.mul_(1 - lr * wd)

            # QK-Clip
            if is_qk and max_logits is not None and max_logits > tau:
                gamma_sqrt = math.sqrt(tau / max_logits)
                p.mul_(gamma_sqrt)
                update = update * gamma_sqrt

            p.add_(update, alpha=-lr)

    def _adamw_step(self, group, lr, wd):
        betas = group.get("betas", (0.9, 0.95))
        eps = group.get("eps", 1e-8)
        beta1, beta2 = betas

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Moment estimates
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            # Decoupled weight decay
            if wd != 0:
                p.mul_(1 - lr * wd)

            # Update
            p.addcdiv_(exp_avg, denom, value=-step_size)


# ============================================================================
# Param Group Builders
# ============================================================================

def _is_qk_name(name: str) -> bool:
    return ".c_q." in name or ".c_k." in name


def _dedup_params(model):
    """Yield (name, param) with tied weights deduplicated via data_ptr()."""
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        yield name, p


def build_param_groups(model):
    """Create 3 param groups: QK Muon, Other Muon, AdamW.

    Returns a MuonClip optimizer instance.
    """
    embed_names = {"token_emb.weight", "lm_head.weight"}

    qk_muon, other_muon, adamw = [], [], []

    for name, p in _dedup_params(model):
        is_embed = name in embed_names
        if p.ndim >= 2 and not is_embed:
            if _is_qk_name(name):
                qk_muon.append(p)
            else:
                other_muon.append(p)
        else:
            adamw.append(p)

    groups = []

    if qk_muon:
        groups.append(dict(
            params=qk_muon, lr=0.02, momentum=0.95, weight_decay=0.1,
            apply_muon=True, is_qk=True, qk_clip_tau=100.0,
        ))

    if other_muon:
        groups.append(dict(
            params=other_muon, lr=0.02, momentum=0.95, weight_decay=0.1,
            apply_muon=True, is_qk=False,
        ))

    if adamw:
        groups.append(dict(
            params=adamw, lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01, apply_muon=False,
        ))

    optimizer = MuonClip(groups)

    # Store initial_lr for WSD schedule
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    return optimizer


def build_adamw_optimizer(model):
    """Plain AdamW fallback (--no-muon flag)."""
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        dict(params=decay, weight_decay=0.1),
        dict(params=no_decay, weight_decay=0.0),
    ]

    optimizer = torch.optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), eps=1e-8)

    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    return optimizer
