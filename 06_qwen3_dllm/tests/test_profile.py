"""Tests for profile.py and toy.py."""

import torch

# ============================================================================
# profile.py tests
# ============================================================================


def test_step_metrics_fields():
  from phase6.profile import StepMetrics

  m = StepMetrics(
    step=0,
    wall_ms=100.0,
    peak_vram_gb=1.0,
    alloc_gb=0.5,
    freed_gb=0.4,
    tokens_per_sec=10000.0,
    forward_ms=40.0,
    backward_ms=50.0,
    optimizer_ms=5.0,
    data_ms=5.0,
    tflops=2.0,
    mem_bw_tb_s=0.5,
  )
  assert m.step == 0
  assert m.wall_ms == 100.0
  assert m.peak_vram_gb == 1.0
  assert m.alloc_gb == 0.5
  assert m.freed_gb == 0.4
  assert m.tokens_per_sec == 10000.0
  assert m.forward_ms == 40.0
  assert m.backward_ms == 50.0
  assert m.optimizer_ms == 5.0
  assert m.data_ms == 5.0
  assert m.tflops == 2.0
  assert m.mem_bw_tb_s == 0.5


def test_step_metrics_gpu_util_pct():
  from phase6.profile import StepMetrics

  m = StepMetrics(
    step=1,
    wall_ms=200.0,
    peak_vram_gb=2.0,
    alloc_gb=1.0,
    freed_gb=0.8,
    tokens_per_sec=5000.0,
    forward_ms=50.0,
    backward_ms=60.0,
    optimizer_ms=10.0,
    data_ms=80.0,
    tflops=3.0,
    mem_bw_tb_s=0.8,
  )
  # gpu_util_pct = (forward + backward + optimizer) / wall * 100
  expected = (50.0 + 60.0 + 10.0) / 200.0 * 100.0
  assert abs(m.gpu_util_pct - expected) < 1e-6


def test_step_metrics_gpu_util_zero_wall():
  from phase6.profile import StepMetrics

  m = StepMetrics(
    step=0,
    wall_ms=0.0,
    peak_vram_gb=0.0,
    alloc_gb=0.0,
    freed_gb=0.0,
    tokens_per_sec=0.0,
    forward_ms=0.0,
    backward_ms=0.0,
    optimizer_ms=0.0,
    data_ms=0.0,
    tflops=0.0,
    mem_bw_tb_s=0.0,
  )
  # Should not raise ZeroDivisionError
  assert m.gpu_util_pct == 0.0


def test_profiler_context_cpu():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=0)
  with profiler.step(step=0, num_tokens=128):
    x = torch.randn(10, 10)
    y = x @ x.T

  assert len(profiler.history) == 1
  m = profiler.history[0]
  assert m.step == 0
  assert m.wall_ms > 0
  # CPU: GPU metrics are zeroed
  assert m.peak_vram_gb == 0.0
  assert m.tokens_per_sec > 0


def test_profiler_warmup_excludes():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=2)
  for i in range(5):
    with profiler.step(step=i, num_tokens=64):
      pass
  # warmup_steps=2 means steps 0,1 are excluded
  assert len(profiler.history) == 3
  assert profiler.history[0].step == 2


def test_profiler_disabled():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=False, device='cpu', warmup_steps=0)
  with profiler.step(step=0, num_tokens=64):
    pass
  assert len(profiler.history) == 0


def test_profiler_region():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=0)
  with profiler.step(step=0, num_tokens=64):
    with profiler.region('forward'):
      _ = torch.randn(10, 10)
    with profiler.region('backward'):
      _ = torch.randn(10, 10)
  m = profiler.history[0]
  assert m.forward_ms >= 0.0
  assert m.backward_ms >= 0.0


def test_profiler_summary():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=0)
  for i in range(3):
    with profiler.step(step=i, num_tokens=64):
      pass
  s = profiler.summary()
  assert 'wall_ms' in s
  assert 'tokens_per_sec' in s
  assert s['wall_ms'] > 0


def test_profiler_summary_empty():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=5)
  # All steps are warmup — history empty
  for i in range(3):
    with profiler.step(step=i, num_tokens=64):
      pass
  s = profiler.summary()
  assert s == {}


def test_profiler_compare():
  from phase6.profile import Profiler

  profiler = Profiler(enabled=True, device='cpu', warmup_steps=0)
  for i in range(3):
    with profiler.step(step=i, num_tokens=64):
      pass
  s = profiler.summary()
  baseline = {k: v * 1.1 for k, v in s.items()}  # baseline is 10% higher
  delta = profiler.compare(baseline)
  # Our values are ~10% lower than baseline -> delta should be negative
  assert 'wall_ms' in delta


def test_flops_estimate_positive():
  from phase6.config import Config
  from phase6.profile import estimate_step_tflops

  cfg = Config(
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_kv_head=2,
    head_dim=64,
    mlp_hidden=256,
    vocab_size=512,
    seq_len=64,
    block_size=8,
    batch_size=2,
  ).validate()
  tflops = estimate_step_tflops(cfg, wall_s=1.0)
  assert tflops > 0.0


def test_mem_bw_positive():
  from phase6.config import Config
  from phase6.profile import estimate_mem_bandwidth

  cfg = Config(
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_kv_head=2,
    head_dim=64,
    mlp_hidden=256,
    vocab_size=512,
    seq_len=64,
    block_size=8,
    batch_size=2,
  ).validate()
  bw = estimate_mem_bandwidth(cfg, wall_s=1.0)
  assert bw > 0.0


# ============================================================================
# toy.py tests
# ============================================================================


def test_toy_vocab_constant():
  from phase6.toy import TOY_VOCAB

  assert TOY_VOCAB == 512


def test_toy_config():
  from phase6.toy import toy_config

  cfg = toy_config()
  assert cfg.n_layer == 4
  assert cfg.n_embd == 128
  assert cfg.n_head == 4
  assert cfg.n_kv_head == 2
  assert cfg.head_dim == 64
  assert cfg.mlp_hidden == 256
  assert cfg.vocab_size == 512
  assert cfg.seq_len == 64
  assert cfg.block_size == 8


def test_toy_config_overrides():
  from phase6.toy import toy_config

  cfg = toy_config(batch_size=8, seq_len=32)
  assert cfg.batch_size == 8
  assert cfg.seq_len == 32


def test_qwen3_config():
  from phase6.toy import qwen3_config

  cfg = qwen3_config()
  assert cfg.n_layer == 28
  assert cfg.n_embd == 1024
  assert cfg.n_head == 16
  assert cfg.n_kv_head == 8
  assert cfg.head_dim == 128
  assert cfg.mlp_hidden == 3072
  assert cfg.vocab_size == 151936
  assert cfg.seq_len == 2048


def test_qwen3_config_overrides():
  from phase6.toy import qwen3_config

  cfg = qwen3_config(batch_size=4, seq_len=512)
  assert cfg.batch_size == 4
  assert cfg.seq_len == 512


def test_toy_batch_shapes():
  from phase6.toy import make_toy_batch, toy_config

  cfg = toy_config(batch_size=2)
  x_input, targets, noise_mask, elbo_w, doc_ids, positions = make_toy_batch(cfg, device='cpu')
  B, L = 2, cfg.seq_len
  assert x_input.shape == (B, 2 * L), f'x_input: {x_input.shape}'
  assert targets.shape == (B, L), f'targets: {targets.shape}'
  assert noise_mask.shape == (B, L), f'noise_mask: {noise_mask.shape}'
  assert elbo_w.shape == (B, L), f'elbo_w: {elbo_w.shape}'
  assert doc_ids.shape == (B, L), f'doc_ids: {doc_ids.shape}'
  assert positions.shape == (B, L), f'positions: {positions.shape}'


def test_toy_batch_noise_mask_bool():
  from phase6.toy import make_toy_batch, toy_config

  cfg = toy_config(batch_size=2)
  _, _, noise_mask, _, _, _ = make_toy_batch(cfg, device='cpu')
  assert noise_mask.dtype == torch.bool


def test_toy_batch_elbo_positive():
  from phase6.toy import make_toy_batch, toy_config

  cfg = toy_config(batch_size=2)
  _, _, _, elbo_w, _, _ = make_toy_batch(cfg, device='cpu')
  assert (elbo_w > 0).all()


def test_toy_batch_x_input_masked():
  """x_input[:, :L] should contain mask tokens where noise_mask is True."""
  from phase6.toy import make_toy_batch, toy_config

  cfg = toy_config(batch_size=2)
  x_input, targets, noise_mask, _, _, _ = make_toy_batch(cfg, device='cpu')
  L = cfg.seq_len
  x_noisy = x_input[:, :L]
  assert (x_noisy[noise_mask] == cfg.mask_token_id).all()
  assert (x_input[:, L:] == targets).all()


def test_run_train_step():
  from phase6.attention import build_staircase_mask
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.toy import make_toy_batch, run_train_step, toy_config

  cfg = toy_config(batch_size=2)
  cfg.use_liger = False
  cfg.use_compile = False
  cfg.use_flex = False
  cfg.use_amp = False
  model = Model(cfg).train()
  optimizer = create_optimizer(model, cfg)
  batch = make_toy_batch(cfg, device='cpu')
  attn_mask = build_staircase_mask(cfg.seq_len, cfg.block_size)
  loss_val, grad_norm = run_train_step(model, batch, cfg, optimizer, attn_mask)
  assert isinstance(loss_val, float)
  assert loss_val > 0
  assert isinstance(grad_norm, float)
  assert grad_norm >= 0
