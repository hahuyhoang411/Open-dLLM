"""Profiling toolkit for per-step performance measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class StepMetrics:
  step: int
  wall_ms: float
  peak_vram_gb: float
  alloc_gb: float
  freed_gb: float
  tokens_per_sec: float
  forward_ms: float
  backward_ms: float
  optimizer_ms: float
  data_ms: float
  tflops: float
  mem_bw_tb_s: float

  @property
  def gpu_util_pct(self) -> float:
    if self.wall_ms == 0.0:
      return 0.0
    return (self.forward_ms + self.backward_ms + self.optimizer_ms) / self.wall_ms * 100.0


class Profiler:
  def __init__(self, enabled: bool, device: str, warmup_steps: int = 3):
    self.enabled = enabled
    self.device = device
    self.warmup_steps = warmup_steps
    self.has_cuda = device.startswith('cuda') and torch.cuda.is_available()
    self.history: List[StepMetrics] = []
    self._region_times: Dict[str, float] = {}
    self._step_count = 0

  @contextmanager
  def step(self, step: int, num_tokens: int):
    if not self.enabled:
      yield
      return

    self._region_times = dict.fromkeys(('forward', 'backward', 'optimizer', 'data'), 0.0)

    if self.has_cuda:
      torch.cuda.reset_peak_memory_stats(self.device)
      alloc_before = torch.cuda.memory_allocated(self.device)
      freed_before = torch.cuda.memory_reserved(self.device)

    t0 = time.perf_counter()
    yield
    if self.has_cuda:
      torch.cuda.synchronize(self.device)
    wall_s = time.perf_counter() - t0
    wall_ms = wall_s * 1000.0

    if self.has_cuda:
      peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
      alloc_after = torch.cuda.memory_allocated(self.device)
      freed_after = torch.cuda.memory_reserved(self.device)
      alloc_gb = max(0.0, alloc_after - alloc_before) / 1e9
      freed_gb = max(0.0, freed_before - freed_after) / 1e9
    else:
      peak_vram_gb = 0.0
      alloc_gb = 0.0
      freed_gb = 0.0

    tokens_per_sec = num_tokens / wall_s if wall_s > 0 else 0.0

    cfg_placeholder = _DummyCfg(seq_len=num_tokens)
    tflops = 0.0
    mem_bw = 0.0

    m = StepMetrics(
      step=step,
      wall_ms=wall_ms,
      peak_vram_gb=peak_vram_gb,
      alloc_gb=alloc_gb,
      freed_gb=freed_gb,
      tokens_per_sec=tokens_per_sec,
      forward_ms=self._region_times.get('forward', 0.0),
      backward_ms=self._region_times.get('backward', 0.0),
      optimizer_ms=self._region_times.get('optimizer', 0.0),
      data_ms=self._region_times.get('data', 0.0),
      tflops=tflops,
      mem_bw_tb_s=mem_bw,
    )

    if self._step_count >= self.warmup_steps:
      self.history.append(m)
    self._step_count += 1

  @contextmanager
  def region(self, name: str):
    if not self.enabled:
      yield
      return
    if self.has_cuda:
      torch.cuda.synchronize(self.device)
    t0 = time.perf_counter()
    yield
    if self.has_cuda:
      torch.cuda.synchronize(self.device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    self._region_times[name] = self._region_times.get(name, 0.0) + elapsed_ms

  def summary(self) -> dict:
    if not self.history:
      return {}
    keys = [f for f in StepMetrics.__dataclass_fields__ if f != 'step']
    return {k: sum(getattr(m, k) for m in self.history) / len(self.history) for k in keys}

  def compare(self, baseline_dict: dict) -> dict:
    current = self.summary()
    result = {}
    for k, v in current.items():
      if k in baseline_dict and baseline_dict[k] != 0:
        result[k] = (v - baseline_dict[k]) / baseline_dict[k] * 100.0
    return result


class _DummyCfg:
  """Placeholder so step() compiles without a real Config."""

  def __init__(self, seq_len: int):
    self.seq_len = seq_len


def estimate_step_tflops(cfg, wall_s: float) -> float:
  """Estimate training TFLOPS from config and measured wall time."""
  B = cfg.batch_size
  # Input is [x_t || x_0] so sequence length doubles
  L = cfg.seq_len * 2
  D = cfg.n_embd
  H = cfg.n_head
  Hkv = cfg.n_kv_head
  Dh = cfg.head_dim
  Dmlp = cfg.mlp_hidden
  V = cfg.vocab_size
  NL = cfg.n_layer

  # Per layer FLOPs (matmuls only, 2 multiplies per MAC)
  # QKV projections: Q=(D->H*Dh), K=(D->Hkv*Dh), V=(D->Hkv*Dh)
  qkv_flops = 2 * B * L * D * (H * Dh + 2 * Hkv * Dh)
  # Attention: QK^T and softmax@V  (B, H, L, L)
  attn_qk = 2 * B * H * L * L * Dh
  attn_v = 2 * B * H * L * L * Dh
  # Output projection: H*Dh -> D
  out_proj = 2 * B * L * H * Dh * D
  # SwiGLU MLP: gate + up (D->Dmlp each) + down (Dmlp->D)
  mlp_flops = 2 * B * L * (D * Dmlp + D * Dmlp + Dmlp * D)

  per_layer = qkv_flops + attn_qk + attn_v + out_proj + mlp_flops
  total_layer_flops = NL * per_layer

  # LM head: h @ weight.T  (B*L, D) x (D, V)
  lm_head_flops = 2 * B * L * D * V

  fwd_flops = total_layer_flops + lm_head_flops

  # Training: fwd + bwd_data + bwd_weight ≈ 3× forward
  train_flops = 3 * fwd_flops

  return train_flops / wall_s / 1e12


def estimate_mem_bandwidth(cfg, wall_s: float) -> float:
  """Estimate bandwidth for bandwidth-bound ops (norms, RoPE, residuals) in TB/s."""
  B = cfg.batch_size
  L = cfg.seq_len * 2
  D = cfg.n_embd
  NL = cfg.n_layer
  bytes_per_elem = 2  # bf16

  # Per layer: 2 RMSNorm reads+writes + 2 residual adds
  norm_bw = NL * 4 * B * L * D * bytes_per_elem
  # RoPE: read/write Q and K  (using head_dim and n_head/n_kv_head)
  rope_bw = NL * 2 * B * L * (cfg.n_head + cfg.n_kv_head) * cfg.head_dim * bytes_per_elem
  # Residual adds: 2 per layer (post-attn, post-mlp) × read+write
  resid_bw = NL * 2 * 2 * B * L * D * bytes_per_elem

  total_bytes = norm_bw + rope_bw + resid_bw
  return total_bytes / wall_s / 1e12
