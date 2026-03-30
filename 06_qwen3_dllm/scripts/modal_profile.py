"""Modal profiling harness for Phase 6 training optimization.

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_profile.py::profile_toy
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_profile.py::profile_full
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_profile.py::profile_ab --optimization flash
"""

import modal

app = modal.App('smoldlm-phase6-profile')

image = (
  modal.Image
  .debian_slim(python_version='3.11')
  .pip_install(
    'torch>=2.5.0',
    'transformers>=4.45.0',
    'datasets>=2.0.0',
    'liger-kernel>=0.5.9',
    'safetensors',
    'huggingface_hub',
    'numpy',
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


def _run_profiled_steps(cfg, num_steps: int = 20, warmup: int = 5, label: str = 'run'):
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import torch

  torch.set_float32_matmul_precision('high')

  from phase6.attention import build_staircase_mask
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.profile import Profiler, estimate_mem_bandwidth, estimate_step_tflops
  from phase6.toy import make_toy_batch

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  cfg.device = device

  model = Model(cfg).to(device).bfloat16().train()
  optimizer = create_optimizer(model, cfg)
  attn_mask = build_staircase_mask(cfg.seq_len, cfg.block_size).to(device)

  profiler = Profiler(enabled=True, device=device, warmup_steps=warmup)

  print(f'\n[{label}] {model.count_params() / 1e6:.1f}M params | {num_steps} steps | warmup={warmup}')
  print(f'  batch={cfg.batch_size} seq={cfg.seq_len} layers={cfg.n_layer} d={cfg.n_embd}')

  for step in range(num_steps):
    num_tokens = cfg.batch_size * cfg.seq_len

    with profiler.step(step=step, num_tokens=num_tokens):
      with profiler.region('data'):
        batch = make_toy_batch(cfg, device=device)

      with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=cfg.use_amp):
        with profiler.region('forward'):
          x_input, targets, noise_mask, elbo_w, doc_ids, positions = batch
          h, _ = model(x_input, targets=targets, attn_mask=attn_mask)

        from phase6.loss import compute_loss

        loss = compute_loss(h, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

      with profiler.region('backward'):
        loss.backward()

      with profiler.region('optimizer'):
        import torch.nn.utils as nn_utils

        nn_utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    if profiler.history:
      m = profiler.history[-1]
      tflops = estimate_step_tflops(cfg, m.wall_ms / 1000.0)
      mem_bw = estimate_mem_bandwidth(cfg, m.wall_ms / 1000.0)
      print(
        f'  step={step:3d}  loss={loss.item():.4f}'
        f'  wall={m.wall_ms:.0f}ms  tok/s={m.tokens_per_sec:.0f}'
        f'  vram={m.peak_vram_gb:.2f}GB'
        f'  fwd={m.forward_ms:.0f}ms bwd={m.backward_ms:.0f}ms opt={m.optimizer_ms:.0f}ms'
        f'  ~{tflops:.1f}TFLOPS  ~{mem_bw:.2f}TB/s'
      )
    elif step < warmup:
      print(f'  step={step:3d} [warmup]  loss={loss.item():.4f}')

  summary = profiler.summary()
  if summary:
    print(f'\n[{label}] Summary (avg over {len(profiler.history)} steps):')
    print(f'  wall_ms={summary["wall_ms"]:.1f}  tok/s={summary["tokens_per_sec"]:.0f}')
    print(f'  vram={summary["peak_vram_gb"]:.2f}GB')
    print(
      f'  fwd={summary["forward_ms"]:.1f}ms  bwd={summary["backward_ms"]:.1f}ms  opt={summary["optimizer_ms"]:.1f}ms'
    )
    print('  gpu_util ~ computed from summary above')

  return summary


@app.function(image=image, gpu='H100', timeout=600)
def profile_toy():
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  from phase6.toy import toy_config

  cfg = toy_config(batch_size=8)
  return _run_profiled_steps(cfg, num_steps=15, warmup=3, label='toy')


@app.function(image=image, gpu='H100', timeout=900)
def profile_full():
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  from phase6.toy import qwen3_config

  cfg = qwen3_config(batch_size=4, use_grad_ckpt=True)
  return _run_profiled_steps(cfg, num_steps=15, warmup=3, label='qwen3-full')


@app.function(image=image, gpu='H100', timeout=1200)
def profile_ab(optimization: str = 'baseline'):
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  from phase6.profile import Profiler
  from phase6.toy import toy_config

  # Baseline: no optimizations
  base_cfg = toy_config(
    batch_size=8,
    use_liger=False,
    use_compile=False,
    use_flex=False,
  )
  baseline = _run_profiled_steps(base_cfg, num_steps=15, warmup=3, label='baseline')

  # Optimized: with requested optimization
  opt_cfg = toy_config(batch_size=8)
  if optimization == 'liger':
    opt_cfg.use_liger = True
  elif optimization == 'compile':
    opt_cfg.use_compile = True
  elif optimization == 'flex':
    opt_cfg.use_flex = True
  # default: all on (toy_config defaults)

  optimized = _run_profiled_steps(opt_cfg, num_steps=15, warmup=3, label=f'optimized-{optimization}')

  # Delta table
  profiler = Profiler(enabled=True, device='cpu', warmup_steps=0)
  delta = {}
  if baseline and optimized:
    for k in baseline:
      if baseline[k] != 0:
        delta[k] = (optimized[k] - baseline[k]) / baseline[k] * 100.0

  print(f'\n[A/B delta] {optimization}:')
  for k, v in delta.items():
    sign = '+' if v >= 0 else ''
    print(f'  {k}: {sign}{v:.1f}%')

  return {'baseline': baseline, 'optimized': optimized, 'delta': delta}
