"""VRAM Reduction Methods — Comprehensive Comparison on H100/H200.

Tests 8 configurations back-to-back with the real training loop:
  1. SAC baseline (current)
  2. SAC + Simple CPU Offload
  3. SAC + Smart CPU Offload (pinned + async)
  4. SAC + TiledMLP
  5. SAC + Activation Compression (FP8)
  6. SAC + Sqrt-SAC
  7. SAC + Smart Offload + TiledMLP (recommended combo)
  8. SAC + Smart Offload + TiledMLP + Sqrt-SAC (triple)

For each: peak VRAM, step time, throughput, forward/backward breakdown.
Top configs get a batch-size sweep to find max B that fits.

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_vram_comparison.py
"""

import modal

app = modal.App('smoldlm-vram-comparison')

# Image: EXACT same base as kernel_validate (reuses Modal cache), plus training deps
image = (
  modal.Image
  # ---- layers below are IDENTICAL to modal_kernel_validate.py ----
  .from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
  .apt_install('git', 'build-essential', 'ninja-build')
  .env({
    'MAX_JOBS': '16',
    'TORCH_CUDA_ARCH_LIST': '9.0',
    'CUDA_HOME': '/usr/local/cuda',
    'CXX': 'g++',
    'CC': 'gcc',
  })
  .uv_pip_install(
    'torch',
    'numpy',
    'packaging',
    'setuptools',
    'wheel',
    'ninja',
    'psutil',
    'quack-kernels==0.3.7',
    'liger-kernel>=0.5.9',
  )
  .run_commands(
    '/.uv/uv pip install --system'
    " 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl'"
    " || echo 'FLASH_ATTN_BUILD_FAILED'",
  )
  .run_commands(
    "/.uv/uv pip install --system --no-deps git+https://github.com/Dao-AILab/gram-newton-schulz.git || echo 'GRAM_NS_INSTALL_FAILED'",
  )
  # ---- layers below are NEW (thin layer on cached base) ----
  .uv_pip_install(
    'datasets>=2.0.0',
    'transformers>=4.45.0',
    'safetensors',
    'huggingface_hub',
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


# ============================================================================
# Configuration matrix
# ============================================================================

CONFIGS = {
  'A_baseline': dict(
    label='SAC baseline (current)',
    use_grad_ckpt=True,
  ),
  'B_simple_offload': dict(
    label='SAC + Simple CPU Offload',
    use_grad_ckpt=True,
    use_offload_ckpt=True,
    offload_strategy='simple',
  ),
  'C_smart_offload': dict(
    label='SAC + Smart CPU Offload (pinned+async)',
    use_grad_ckpt=True,
    use_offload_ckpt=True,
    offload_strategy='smart',
  ),
  'D_tiled_mlp': dict(
    label='SAC + TiledMLP',
    use_grad_ckpt=True,
    use_tiled_mlp=True,
  ),
  'E_sqrt_sac': dict(
    label='SAC + Sqrt-SAC',
    use_grad_ckpt=True,
    use_sqrt_ckpt=True,
  ),
  'F_compress': dict(
    label='SAC + Activation Compression (FP8)',
    use_grad_ckpt=True,
    use_offload_ckpt=True,
    offload_strategy='compress',
  ),
  'G_smart_tiled': dict(
    label='SAC + Smart Offload + TiledMLP',
    use_grad_ckpt=True,
    use_offload_ckpt=True,
    offload_strategy='smart',
    use_tiled_mlp=True,
  ),
  'H_smart_tiled_sqrt': dict(
    label='SAC + Smart Offload + TiledMLP + Sqrt-SAC',
    use_grad_ckpt=True,
    use_offload_ckpt=True,
    offload_strategy='smart',
    use_tiled_mlp=True,
    use_sqrt_ckpt=True,
  ),
}

# Configs that get a batch-size sweep (top candidates)
SWEEP_CONFIGS = ['A_baseline', 'C_smart_offload', 'G_smart_tiled', 'H_smart_tiled_sqrt']
SWEEP_BATCHES = [1, 2, 4, 8, 16, 32, 48, 64]


def _stats(vals):
  n = len(vals)
  mean = sum(vals) / n
  std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
  return mean, std


def _make_cfg(batch_size=4, **vram_overrides):
  """Build production Qwen3-0.6B config with VRAM method flags."""
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')
  from phase6.config import Config, setup_features

  defaults = dict(
    n_layer=28,
    n_embd=1024,
    n_head=16,
    n_kv_head=8,
    head_dim=128,
    mlp_hidden=3072,
    vocab_size=151936,
    seq_len=8192,
    block_size=8,
    rope_base=1_000_000,
    rms_eps=1e-6,
    dropout=0.0,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=True,
    use_liger_swiglu=True,
    use_liger_rmsnorm=False,
    use_grad_ckpt=True,
    use_flex=True,
    use_muon=True,
    use_fp8=True,
    use_compile=True,
    use_amp=True,
    use_flce=True,
    use_fa_rope=True,
    use_gram_ns=True,
    pad_token_id=151643,
    mask_token_id=151669,
    eos_token_id=151645,
    batch_size=batch_size,
    max_iters=100,
    muon_lr=0.02,
    adamw_lr=3e-3,
    grad_clip=1.0,
    t_min=0.1,
    # VRAM method defaults (all off)
    use_offload_ckpt=False,
    offload_strategy='smart',
    use_tiled_mlp=False,
    tiled_mlp_chunk=0,
    use_sqrt_ckpt=False,
  )
  defaults.update(vram_overrides)

  cfg = Config(**defaults).validate()
  cfg.device = 'cuda'
  return setup_features(cfg)


def _run_single_config(config_name, config_dict, base_batch_size=4):
  """Run a single VRAM config: warmup + 15 timed steps. Returns results dict."""
  import time

  import torch
  from phase6.attention import build_staircase_block_mask, build_staircase_mask
  from phase6.loss import compute_loss, compute_loss_flce
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.toy import make_toy_batch
  from torch import nn

  label = config_dict.pop('label')
  cfg = _make_cfg(batch_size=base_batch_size, **config_dict)
  config_dict['label'] = label  # restore

  print(f'\n{"=" * 70}')
  print(f'  {config_name}: {label}')
  print(f'{"=" * 70}')
  flags = []
  if cfg.use_offload_ckpt:
    flags.append(f'offload={cfg.offload_strategy}')
  if cfg.use_tiled_mlp:
    flags.append(f'tiled_mlp(chunk={cfg.tiled_mlp_chunk or "auto"})')
  if cfg.use_sqrt_ckpt:
    flags.append('sqrt_sac')
  if cfg.use_fa_rope:
    flags.append('fa_rope')
  if cfg.use_gram_ns:
    flags.append('gram_ns')
  print(f'  Flags: {", ".join(flags) if flags else "baseline"}')
  print(f'  B={cfg.batch_size}, seq={cfg.seq_len}, effective_T={cfg.seq_len * 2}')

  # Build model
  model = Model(cfg).cuda()

  if cfg.use_fp8:
    try:
      from phase6.fp8 import convert_to_float8_training

      def fp8_ok(m, fqn):
        return isinstance(m, nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0 and fqn != 'lm_head'

      convert_to_float8_training(model, module_filter_fn=fp8_ok)
    except Exception as e:
      print(f'  FP8 skipped: {e}')

  if cfg.use_compile:
    for i, blk in enumerate(model.blocks):
      model.blocks[i] = torch.compile(blk, dynamic=False)

  optimizer = create_optimizer(model, cfg)
  model.train()
  params_m = model.count_params() / 1e6

  if cfg.use_flex:
    mask = build_staircase_block_mask(cfg.seq_len, cfg.block_size)
  else:
    mask = build_staircase_mask(cfg.seq_len, cfg.block_size).to('cuda')

  tokens_per_step = cfg.batch_size * cfg.seq_len * 2

  # Warmup (5 steps for JIT + CUDA graph warmup)
  print('  Warmup (5 steps)...')
  for step in range(5):
    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
      try:
        loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      except Exception:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

  # Reset smart offload state between configs
  try:
    from phase6.offload_smart import reset_offload_state

    reset_offload_state()
  except ImportError:
    pass

  # Timed steps (15 steps)
  torch.cuda.reset_peak_memory_stats()
  region_times = {'forward': [], 'loss': [], 'backward': [], 'optimizer': [], 'total': []}

  for step in range(15):
    torch.cuda.synchronize()

    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
    torch.cuda.synchronize()
    dt_fwd = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      try:
        loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      except Exception:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    torch.cuda.synchronize()
    dt_loss = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    dt_bwd = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt_opt = (time.perf_counter() - t0) * 1000

    dt_total = dt_fwd + dt_loss + dt_bwd + dt_opt
    region_times['forward'].append(dt_fwd)
    region_times['loss'].append(dt_loss)
    region_times['backward'].append(dt_bwd)
    region_times['optimizer'].append(dt_opt)
    region_times['total'].append(dt_total)

  peak_vram = torch.cuda.max_memory_allocated() / 1e9

  # Compute stats
  means = {k: _stats(v)[0] for k, v in region_times.items()}
  stds = {k: _stats(v)[1] for k, v in region_times.items()}
  avg_tps = int(tokens_per_step / (means['total'] / 1000))

  print('\n  Results (15 steps):')
  for region in ['forward', 'loss', 'backward', 'optimizer', 'total']:
    pct = means[region] / means['total'] * 100
    print(f'    {region:12s}: {means[region]:7.1f} +/- {stds[region]:5.1f} ms  ({pct:5.1f}%)')
  print(f'    Peak VRAM:    {peak_vram:.2f} GB')
  print(f'    Throughput:   {avg_tps:,} tok/s')
  print(f'    Params:       {params_m:.1f}M')

  del model, optimizer
  torch.cuda.empty_cache()

  return dict(
    name=config_name,
    label=label,
    batch_size=base_batch_size,
    peak_vram_gb=peak_vram,
    avg_step_ms=means['total'],
    std_step_ms=stds['total'],
    fwd_ms=means['forward'],
    loss_ms=means['loss'],
    bwd_ms=means['backward'],
    opt_ms=means['optimizer'],
    tok_per_s=avg_tps,
    params_m=params_m,
  )


def _run_batch_sweep(config_name, config_dict, batch_sizes):
  """Sweep batch sizes for a given config. Returns list of (B, peak_gb, tps, ms)."""
  import time

  import torch
  from phase6.attention import build_staircase_block_mask, build_staircase_mask
  from phase6.loss import compute_loss, compute_loss_flce
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.toy import make_toy_batch
  from torch import nn

  label = config_dict.get('label', config_name)
  results = []

  print(f'\n--- Batch sweep: {config_name} ({label}) ---')

  for bs in batch_sizes:
    try:
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      overrides = {k: v for k, v in config_dict.items() if k != 'label'}
      cfg = _make_cfg(batch_size=bs, **overrides)

      model = Model(cfg).cuda()

      if cfg.use_fp8:
        try:
          from phase6.fp8 import convert_to_float8_training

          def fp8_ok(m, fqn):
            return (
              isinstance(m, nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0 and fqn != 'lm_head'
            )

          convert_to_float8_training(model, module_filter_fn=fp8_ok)
        except Exception:
          pass

      if cfg.use_compile:
        for i, blk in enumerate(model.blocks):
          model.blocks[i] = torch.compile(blk, dynamic=False)

      opt = create_optimizer(model, cfg)
      model.train()

      if cfg.use_flex:
        mask = build_staircase_block_mask(cfg.seq_len, cfg.block_size)
      else:
        mask = build_staircase_mask(cfg.seq_len, cfg.block_size).to('cuda')

      step_times = []
      for i in range(8):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        batch = make_toy_batch(cfg, device='cuda')
        xi, tgt, nm, ew, _di, pos = batch
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
          h, _ = model(xi, tgt, attn_mask=mask, positions=pos)
          try:
            sl = compute_loss_flce(h, tgt, nm, ew, model.lm_head.weight, cfg)
          except Exception:
            sl = compute_loss(h, tgt, nm, ew, model.lm_head.weight, cfg)
        sl.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= 3:  # skip first 3 as warmup
          step_times.append(dt)

      peak_gb = torch.cuda.max_memory_allocated() / 1e9
      mean_dt = sum(step_times) / len(step_times)
      tps = int(bs * cfg.seq_len * 2 / (mean_dt / 1000))
      results.append((bs, peak_gb, tps, mean_dt))
      print(f'  B={bs:2d}: {peak_gb:6.1f} GB | {tps:7,} tok/s | {mean_dt:7.1f} ms/step')

      del model, opt
      torch.cuda.empty_cache()

      # Reset smart offload state
      try:
        from phase6.offload_smart import reset_offload_state

        reset_offload_state()
      except ImportError:
        pass

    except torch.cuda.OutOfMemoryError:
      print(f'  B={bs:2d}: OOM')
      results.append((bs, -1, -1, -1))
      torch.cuda.empty_cache()
    except Exception as e:
      print(f'  B={bs:2d}: ERROR - {e}')
      results.append((bs, -2, -2, -2))
      torch.cuda.empty_cache()

  return results


# ============================================================================
# Main entry point
# ============================================================================


@app.function(image=image, gpu='H100', timeout=7200, startup_timeout=1800)
def run_comparison():
  """Full VRAM method comparison."""
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import torch

  torch.set_float32_matmul_precision('high')

  props = torch.cuda.get_device_properties(0)
  total_mem = props.total_memory / 1e9
  print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
  print(f'{torch.cuda.get_device_name(0)} | {total_mem:.0f} GB')
  print('VRAM Reduction Methods — Comprehensive Comparison')
  print()

  # ================================================================
  # Part 1: Run all 8 configs at B=4
  # ================================================================
  print('#' * 70)
  print('#  PART 1: ALL CONFIGS AT B=4, seq=8192')
  print('#' * 70)

  all_results = {}
  for name, cfg_dict in CONFIGS.items():
    try:
      result = _run_single_config(name, cfg_dict.copy(), base_batch_size=4)
      all_results[name] = result
    except torch.cuda.OutOfMemoryError:
      print(f'\n  {name}: OOM at B=4!')
      all_results[name] = dict(name=name, peak_vram_gb=-1, tok_per_s=-1)
    except Exception as e:
      print(f'\n  {name}: ERROR - {e}')
      all_results[name] = dict(name=name, peak_vram_gb=-2, tok_per_s=-2)

  # ================================================================
  # Part 2: Comparison table
  # ================================================================
  print('\n\n')
  print('#' * 70)
  print('#  COMPARISON TABLE')
  print('#' * 70)

  baseline = all_results.get('A_baseline', {})
  base_vram = baseline.get('peak_vram_gb', 1)
  base_ms = baseline.get('avg_step_ms', 1)
  base_tps = baseline.get('tok_per_s', 1)

  print(f'\n  {"Config":<30s} | {"VRAM GB":>8s} | {"Saved":>6s} | {"ms/step":>8s} | {"Overhead":>8s} | {"tok/s":>9s}')
  print(f'  {"-" * 30}-+-{"-" * 8}-+-{"-" * 6}-+-{"-" * 8}-+-{"-" * 8}-+-{"-" * 9}')

  for name in CONFIGS:
    r = all_results.get(name, {})
    vram = r.get('peak_vram_gb', -1)
    ms = r.get('avg_step_ms', -1)
    tps = r.get('tok_per_s', -1)

    if vram < 0:
      print(f'  {name:<30s} | {"OOM/ERR":>8s} | {"—":>6s} | {"—":>8s} | {"—":>8s} | {"—":>9s}')
      continue

    saved = base_vram - vram
    overhead_pct = (ms - base_ms) / base_ms * 100 if base_ms > 0 else 0

    print(f'  {name:<30s} | {vram:8.2f} | {saved:+5.1f} | {ms:8.1f} | {overhead_pct:+7.1f}% | {tps:>9,}')

  # ================================================================
  # Part 3: Batch size sweep for top configs
  # ================================================================
  print('\n\n')
  print('#' * 70)
  print('#  PART 3: BATCH SIZE SWEEP (top configs)')
  print('#' * 70)

  sweep_results = {}
  for name in SWEEP_CONFIGS:
    cfg_dict = CONFIGS[name].copy()
    sweep_results[name] = _run_batch_sweep(name, cfg_dict, SWEEP_BATCHES)

  # Max batch size table
  print(f'\n\n  {"Config":<30s} | {"Max B":>5s} | {"VRAM @ max":>10s} | {"tok/s @ max":>11s}')
  print(f'  {"-" * 30}-+-{"-" * 5}-+-{"-" * 10}-+-{"-" * 11}')

  for name in SWEEP_CONFIGS:
    results = sweep_results.get(name, [])
    valid = [(b, v, t, m) for b, v, t, m in results if v > 0]
    if valid:
      best = valid[-1]  # highest B that didn't OOM
      print(f'  {name:<30s} | B={best[0]:>3d} | {best[1]:9.1f} GB | {best[2]:>9,} tok/s')
    else:
      print(f'  {name:<30s} | {"—":>5s} | {"—":>10s} | {"—":>11s}')

  # ================================================================
  # Part 4: Recommendations
  # ================================================================
  print('\n\n')
  print('#' * 70)
  print('#  RECOMMENDATIONS')
  print('#' * 70)
  print()
  print('  Based on empirical results above:')
  print('  - VRAM savings: check "Saved" column — which methods actually reduce peak memory')
  print('  - Throughput cost: check "Overhead" — methods with >5% overhead are suspect')
  print('  - Batch scaling: check max B — higher B = more throughput')
  print('  - Combo: G_smart_tiled is the expected winner (offload activations + chunk MLP)')
  print()
  print('  Production config recommendation:')
  print('    --grad-ckpt --offload-ckpt --offload-strategy smart --tiled-mlp')
  print('    --fa-rope --gram-ns --fp8 --no-liger')
  print()


@app.local_entrypoint()
def main():
  run_comparison.remote()
