"""Self-contained MuonClip optimizer with QK-Clip + AdamW fallback.

No external dependencies — only stdlib (math, threading) + torch.
Works in both single-GPU and DDP (no dist.all_gather dependency).
"""

import math
import threading

import torch
from torch.optim import Optimizer

from .config import Config

# ============================================================================
# Max Attention Logits Tracker (for QK-Clip)
# ============================================================================


class _MaxLogitsTracker:
  """Track per-step max attention logit for QK-Clip.

  attention.py calls _update() directly with ||q||*||k||/sqrt(d) before
  any SDPA/FlexAttention dispatch. This works for all backends.

  Global scalar (not per-layer or per-head). Thread-safe via threading.local.
  """

  _tls = threading.local()

  @classmethod
  def _update(cls, v):
    """Accept 0-d GPU tensor or float. No .item() — stays on device."""
    cur = getattr(cls._tls, 'max_logits', None)
    if cur is None:
      cls._tls.max_logits = v
    elif isinstance(cur, torch.Tensor) and isinstance(v, torch.Tensor):
      cls._tls.max_logits = torch.maximum(cur, v)
    else:
      cur_f = cur.item() if isinstance(cur, torch.Tensor) else cur
      v_f = v.item() if isinstance(v, torch.Tensor) else v
      cls._tls.max_logits = max(cur_f, v_f)

  @classmethod
  def enable(cls):
    pass  # tracking handled by attention.py direct call

  @classmethod
  def consume(cls):
    """Return max logit as Python float (safe — called from optimizer.step(),
    outside compiled graph). Resets tracker for next forward pass.
    """
    v = getattr(cls._tls, 'max_logits', None)
    cls._tls.max_logits = None
    if v is None:
      return None
    return v.item() if isinstance(v, torch.Tensor) else v


# ============================================================================
# MuonClip Optimizer
# ============================================================================


class MuonClip(Optimizer):
  """Unified optimizer: Muon (Newton-Schulz) for 2D weights, AdamW for the rest.

  Per-group flags:
      apply_muon=True  -> momentum + NS orthogonalization + RMS scaling
      apply_muon=False -> standard AdamW with bias correction

  QK-Clip (is_qk=True groups only): if max attention logit > tau,
  scale weight and update by sqrt(tau / max_logit).
  """

  def __init__(self, params, **defaults):
    defaults.setdefault('lr', 0.02)
    defaults.setdefault('momentum', 0.95)
    defaults.setdefault('weight_decay', 0.1)
    defaults.setdefault('betas', (0.9, 0.95))
    defaults.setdefault('eps', 1e-8)
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
      apply_muon = group.get('apply_muon', True)
      lr = group['lr']
      wd = group['weight_decay']

      if apply_muon:
        self._muon_step(group, lr, wd, max_logits)
      else:
        self._adamw_step(group, lr, wd)

    return loss

  def _muon_step(self, group, lr, wd, max_logits):
    momentum = group.get('momentum', 0.95)
    is_qk = group.get('is_qk', False)
    tau = group.get('qk_clip_tau', 100.0)

    for p in group['params']:
      if p.grad is None or p.ndim < 2:
        continue

      state = self.state[p]
      if len(state) == 0:
        state['momentum_buffer'] = torch.zeros_like(p)

      buf = state['momentum_buffer']
      buf.mul_(momentum).add_(p.grad)

      # Newton-Schulz orthogonalization + RMS scaling
      orth = _newton_schulz_compiled(buf)
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
    betas = group.get('betas', (0.9, 0.95))
    eps = group.get('eps', 1e-8)
    beta1, beta2 = betas

    for p in group['params']:
      if p.grad is None:
        continue

      state = self.state[p]
      if len(state) == 0:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p)
        state['exp_avg_sq'] = torch.zeros_like(p)

      state['step'] += 1
      t = state['step']

      exp_avg = state['exp_avg']
      exp_avg_sq = state['exp_avg_sq']

      # Moment estimates
      exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

      # Bias correction
      bias_correction1 = 1 - beta1**t
      bias_correction2 = 1 - beta2**t

      step_size = lr / bias_correction1
      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

      # Decoupled weight decay
      if wd != 0:
        p.mul_(1 - lr * wd)

      # Update
      p.addcdiv_(exp_avg, denom, value=-step_size)


# Compile newton_schulz once at module level.
# dynamic=True handles varying parameter shapes (7+ unique shapes) with a single
# compiled graph instead of one per shape.
# Called as a plain function (not via self) to avoid Python injecting self as first arg.
try:
  _newton_schulz_compiled = torch.compile(MuonClip.newton_schulz, dynamic=True)
except Exception:
  _newton_schulz_compiled = MuonClip.newton_schulz


# ============================================================================
# Param Group Builders
# ============================================================================


def _is_qk_name(name: str) -> bool:
  return '.c_q.' in name or '.c_k.' in name


def _dedup_params(model):
  """Yield (name, param) with tied weights deduplicated via data_ptr()."""
  seen = set()
  for name, p in model.named_parameters():
    if not p.requires_grad or p.data_ptr() in seen:
      continue
    seen.add(p.data_ptr())
    yield name, p


def build_param_groups(model, cfg: Config):
  """Build optimizer param groups from model and config.

  Returns list of param group dicts for MuonClip:
      [0] QK Muon  — c_q, c_k projections (QK-Clip enabled)
      [1] Other Muon — all other 2D non-embed weights
      [2] AdamW — embeddings + 1D params (norms, biases)
  """
  embed_names = {'token_emb.weight', 'lm_head.weight'}

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
    groups.append(
      dict(
        params=qk_muon,
        lr=cfg.muon_lr,
        momentum=0.95,
        weight_decay=0.1,
        apply_muon=True,
        is_qk=True,
        qk_clip_tau=100.0,
      )
    )

  if other_muon:
    groups.append(
      dict(
        params=other_muon,
        lr=cfg.muon_lr,
        momentum=0.95,
        weight_decay=0.1,
        apply_muon=True,
        is_qk=False,
      )
    )

  if adamw:
    groups.append(
      dict(
        params=adamw,
        lr=cfg.adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        apply_muon=False,
      )
    )

  return groups


def _build_adamw_groups(model, cfg: Config):
  """Simple decay / no-decay split for plain AdamW."""
  decay, no_decay = [], []
  for name, p in _dedup_params(model):
    if p.ndim >= 2:
      decay.append(p)
    else:
      no_decay.append(p)
  groups = []
  if decay:
    groups.append(dict(params=decay, lr=cfg.adamw_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1))
  if no_decay:
    groups.append(dict(params=no_decay, lr=cfg.adamw_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))
  return groups


def create_optimizer(model, cfg: Config):
  """Create MuonClip or plain AdamW. Clean separation — no cross-contamination."""
  if cfg.use_muon:
    groups = build_param_groups(model, cfg)
    optimizer = MuonClip(groups)
  else:
    groups = _build_adamw_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups)

  for pg in optimizer.param_groups:
    pg['initial_lr'] = pg['lr']

  return optimizer
