"""H200 profiling: ISOLATION MATRIX benchmark — one variable at a time.

Runs 4 configs sequentially on the SAME H200 GPU for fair comparison:
  A: Baseline (Liger SwiGLU + RMSNorm, manual RoPE, compiled NS)
  B: --no-liger-rmsnorm only (native RMSNorm)
  C: --gram-ns only (gram-newton-schulz optimizer)
  D: All optimizations (native RMSNorm + flash-attn RoPE + gram-NS)

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_h200_profile.py
"""

import modal

app = modal.App('smoldlm-h200-profile')

image = (
  modal.Image
  .from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
  .apt_install('git', 'build-essential', 'ninja-build')
  .env({'TORCH_CUDA_ARCH_LIST': '9.0', 'CUDA_HOME': '/usr/local/cuda', 'CXX': 'g++', 'CC': 'gcc'})
  .uv_pip_install(
    'torch',
    'numpy',
    'packaging',
    'setuptools',
    'wheel',
    'ninja',
    'psutil',
    'liger-kernel>=0.5.9',
    'datasets>=2.0.0',
    'transformers>=4.45.0',
    'safetensors',
    'huggingface_hub',
  )
  .run_commands(
    '/.uv/uv pip install --system'
    " 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl'"
    " || echo 'FLASH_ATTN_BUILD_FAILED'",
  )
  .run_commands(
    '/.uv/uv pip install --system --no-deps git+https://github.com/Dao-AILab/gram-newton-schulz.git'
    " || echo 'GRAM_NS_INSTALL_FAILED'",
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


# ============================================================================
# H200 theoretical peaks
# ============================================================================
H200_BF16_TFLOPS = 990
H200_FP8_TFLOPS = 1979
H200_HBM_BW_TBS = 4.8
H200_VRAM_GB = 141


def _make_cfg(use_liger_swiglu=True, use_liger_rmsnorm=True, use_fa_rope=False, use_gram_ns=False):
  """Build production config with optional kernel optimizations."""
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')
  from phase6.config import Config, setup_features

  cfg = Config(
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
    use_liger_swiglu=use_liger_swiglu,
    use_liger_rmsnorm=use_liger_rmsnorm,
    use_grad_ckpt=True,
    use_flex=True,
    use_muon=True,
    use_fp8=True,
    use_compile=True,
    use_amp=True,
    use_flce=True,
    use_fa_rope=use_fa_rope,
    use_gram_ns=use_gram_ns,
    pad_token_id=151643,
    mask_token_id=151669,
    eos_token_id=151645,
    batch_size=4,
    max_iters=100,
    muon_lr=0.02,
    adamw_lr=3e-3,
    grad_clip=1.0,
    t_min=0.1,
  ).validate()
  cfg.device = 'cuda'
  return setup_features(cfg)


def _stats(vals):
  n = len(vals)
  mean = sum(vals) / n
  std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
  sorted_v = sorted(vals)
  median = sorted_v[n // 2]
  return mean, std, median


def _run_profile(cfg, label):
  """Run a full profile: compile warmup + 25 measured steps. Returns results dict."""
  import time

  import torch
  from phase6.attention import build_staircase_block_mask, build_staircase_mask
  from phase6.loss import compute_loss, compute_loss_flce
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.toy import make_toy_batch
  from torch import nn

  props = torch.cuda.get_device_properties(0)
  total_mem = props.total_memory / 1e9

  print('=' * 70)
  print(f' {label}')
  print('=' * 70)
  print(f'Hardware: {torch.cuda.get_device_name(0)} {total_mem:.0f}GB')
  print(f'Model: Qwen3-0.6B ({cfg.n_layer}L, {cfg.n_embd}d, {cfg.n_head}h, {cfg.n_kv_head}kv)')
  print(f'Batch: B={cfg.batch_size}, L={cfg.seq_len} (effective {cfg.batch_size * cfg.seq_len * 2} tokens/step)')
  print(
    f'Flags: liger_swiglu={cfg.use_liger_swiglu} liger_rmsnorm={cfg.use_liger_rmsnorm} '
    f'fa_rope={cfg.use_fa_rope} gram_ns={cfg.use_gram_ns} '
    f'compile={cfg.use_compile} flex={cfg.use_flex} fp8={cfg.use_fp8} flce={cfg.use_flce}'
  )
  print()

  # Build model
  model = Model(cfg).cuda()

  if cfg.use_fp8:
    from phase6.fp8 import Float8Linear, convert_to_float8_training

    def fp8_ok(m, fqn):
      return isinstance(m, nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0 and fqn != 'lm_head'

    convert_to_float8_training(model, module_filter_fn=fp8_ok)
    n8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    print(f'FP8 training: {n8} layers converted')

  compile_t0 = time.perf_counter()
  if cfg.use_compile:
    for i, blk in enumerate(model.blocks):
      model.blocks[i] = torch.compile(blk, dynamic=False)
    print('Per-block compile registered (will JIT on first forward)')

  optimizer = create_optimizer(model, cfg)
  model.train()

  params_m = model.count_params() / 1e6
  print(f'Parameters: {params_m:.1f}M (tied embeddings counted once)')
  print()

  # Build attention mask
  if cfg.use_flex:
    mask = build_staircase_block_mask(cfg.seq_len, cfg.block_size)
  else:
    mask = build_staircase_mask(cfg.seq_len, cfg.block_size).to('cuda')

  # ---- torch.compile warmup (10 steps) ----
  print('-' * 70)
  print('WARMUP (10 steps)')
  print('-' * 70)

  for step in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
      if cfg.use_flce:
        try:
          loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
        except Exception:
          loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      else:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    tps = int(cfg.batch_size * cfg.seq_len * 2 / (dt / 1000))
    print(f'  warmup step {step:2d} | {dt:8.1f} ms | {tps:,} tok/s')

  compile_wall_s = time.perf_counter() - compile_t0
  print(f'\n  Compile wall time: {compile_wall_s:.1f}s ({compile_wall_s / 60:.1f} min)')
  print()

  # ---- Measured steps (25 steps) ----
  print('-' * 70)
  print('MEASURED (25 steps)')
  print('-' * 70)

  torch.cuda.reset_peak_memory_stats()

  region_times = {'data': [], 'forward': [], 'loss': [], 'backward': [], 'optimizer': [], 'total': []}

  for step in range(25):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch
    torch.cuda.synchronize()
    dt_data = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
    torch.cuda.synchronize()
    dt_fwd = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      if cfg.use_flce:
        try:
          loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
        except Exception:
          loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      else:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    torch.cuda.synchronize()
    dt_loss = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    dt_bwd = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt_opt = (time.perf_counter() - t0) * 1000

    dt_total = dt_data + dt_fwd + dt_loss + dt_bwd + dt_opt

    region_times['data'].append(dt_data)
    region_times['forward'].append(dt_fwd)
    region_times['loss'].append(dt_loss)
    region_times['backward'].append(dt_bwd)
    region_times['optimizer'].append(dt_opt)
    region_times['total'].append(dt_total)

    tokens_per_step = cfg.batch_size * cfg.seq_len * 2
    tps = int(tokens_per_step / (dt_total / 1000))
    if step % 5 == 0:
      print(
        f'  step {step:2d} | total {dt_total:7.1f}ms | fwd {dt_fwd:6.1f} | loss {dt_loss:6.1f} | '
        f'bwd {dt_bwd:6.1f} | opt {dt_opt:6.1f} | data {dt_data:5.1f} | {tps:,} tok/s'
      )

  peak_vram = torch.cuda.max_memory_allocated() / 1e9

  print('\n--- Per-region stats (25 measured steps) ---')
  region_stats = {}
  for region in ['data', 'forward', 'loss', 'backward', 'optimizer', 'total']:
    mean, std, median = _stats(region_times[region])
    region_stats[region] = (mean, std, median)
    avg_total_ms = _stats(region_times['total'])[0]
    pct = mean / avg_total_ms * 100
    print(f'  {region:12s}: {mean:7.2f} +/- {std:5.2f} ms  (median {median:7.2f} ms, {pct:5.1f}%)')

  avg_total_ms = region_stats['total'][0]
  tokens_per_step = cfg.batch_size * cfg.seq_len * 2
  avg_tps = int(tokens_per_step / (avg_total_ms / 1000))
  print(f'\n  Throughput: {avg_tps:,} tok/s  (tokens_per_step={tokens_per_step:,} = B*2*seq_len)')

  # Roofline
  B = cfg.batch_size
  T = cfg.seq_len * 2
  D = cfg.n_embd
  H = cfg.n_head
  KV = cfg.n_kv_head
  HD = cfg.head_dim
  MLP = cfg.mlp_hidden
  V = cfg.vocab_size
  L = cfg.n_layer
  Lseq = cfg.seq_len

  qkv_flops = 2 * B * T * D * (H * HD + 2 * KV * HD)
  attn_flops_est = int(2 * B * H * T * T * HD * 0.75)
  out_proj_flops = 2 * B * T * H * HD * D
  mlp_flops = 2 * B * T * D * MLP * 3
  norm_flops_per_layer = 2 * B * T * D * 4
  rope_flops_per_layer = 2 * B * T * H * HD * 6
  per_layer_flops = qkv_flops + attn_flops_est + out_proj_flops + mlp_flops + norm_flops_per_layer + rope_flops_per_layer
  lm_head_flops = 2 * B * Lseq * D * V
  total_fwd_flops = L * per_layer_flops + lm_head_flops
  total_train_flops = 3 * total_fwd_flops

  achieved_tflops = total_train_flops / (avg_total_ms / 1000) / 1e12
  util_bf16 = achieved_tflops / H200_BF16_TFLOPS * 100

  results = {
    'step_ms': avg_total_ms,
    'forward_ms': region_stats['forward'][0],
    'loss_ms': region_stats['loss'][0],
    'backward_ms': region_stats['backward'][0],
    'optimizer_ms': region_stats['optimizer'][0],
    'data_ms': region_stats['data'][0],
    'tok_per_sec': avg_tps,
    'tokens_per_step': tokens_per_step,
    'tflops': achieved_tflops,
    'util_bf16_pct': util_bf16,
    'peak_vram_gb': peak_vram,
  }

  # Clean up
  del model, optimizer
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  return results


def _print_matrix(all_results):
  """Print the isolation matrix comparison table."""

  baseline = all_results.get('A_baseline')
  if baseline is None:
    print('ERROR: no baseline results')
    return

  base_step = baseline['step_ms']

  print()
  print()
  print('\u2550' * 70)
  print(' ISOLATION MATRIX \u2014 H200, seq_len=8192, B=4, FP8')
  print('\u2550' * 70)
  print()
  print(f'  {"Config":<26s} {"Step(ms)":>9s} {"fwd(ms)":>9s} {"bwd(ms)":>9s} {"opt(ms)":>9s} {"tok/s":>10s} {"vs baseline":>12s}')
  print(f'  {"\u2500" * 26} {"\u2500" * 9} {"\u2500" * 9} {"\u2500" * 9} {"\u2500" * 9} {"\u2500" * 10} {"\u2500" * 12}')

  labels = {
    'A_baseline': 'A: Baseline',
    'B_no_liger_rmsnorm': 'B: --no-liger-rmsnorm',
    'C_gram_ns': 'C: --gram-ns',
    'D_all': 'D: All optimizations',
  }

  for key in ['A_baseline', 'B_no_liger_rmsnorm', 'C_gram_ns', 'D_all']:
    r = all_results.get(key)
    if r is None:
      print(f'  {labels[key]:<26s} {"FAILED":>9s}')
      continue

    speedup = base_step / r['step_ms']
    print(
      f'  {labels[key]:<26s} {r["step_ms"]:9.0f} {r["forward_ms"]:9.0f} '
      f'{r["backward_ms"]:9.0f} {r["optimizer_ms"]:9.0f} '
      f'{r["tok_per_sec"]:>10,} {speedup:11.2f}x'
    )

  print()
  print('Optimizations applied per config:')
  print('  A: use_liger=True (SwiGLU+RMSNorm), manual RoPE, compiled NS')
  print('  B: use_liger_swiglu=True, use_liger_rmsnorm=False, manual RoPE, compiled NS')
  print('  C: use_liger=True (SwiGLU+RMSNorm), manual RoPE, gram-NS per-call')
  print('  D: use_liger_swiglu=True, use_liger_rmsnorm=False, flash-attn RoPE, gram-NS per-call')
  print()

  # GPU utilization summary
  print('-' * 70)
  print(' GPU UTILIZATION')
  print('-' * 70)
  for key in ['A_baseline', 'B_no_liger_rmsnorm', 'C_gram_ns', 'D_all']:
    r = all_results.get(key)
    if r is None:
      continue
    fwd_pct = r['forward_ms'] / r['step_ms'] * 100
    bwd_pct = r['backward_ms'] / r['step_ms'] * 100
    opt_pct = r['optimizer_ms'] / r['step_ms'] * 100
    print(
      f'  {labels[key]:<26s}: {r["tflops"]:.1f} TFLOPS ({r["util_bf16_pct"]:.1f}% bf16 peak) | '
      f'fwd {fwd_pct:.0f}% bwd {bwd_pct:.0f}% opt {opt_pct:.0f}% | '
      f'{r["peak_vram_gb"]:.1f} GB VRAM'
    )

  # Overall speedup
  d_results = all_results.get('D_all')
  if d_results:
    overall = base_step / d_results['step_ms']
    print()
    print(f'  Overall speedup (A->D): {overall:.2f}x ({base_step:.0f} ms -> {d_results["step_ms"]:.0f} ms)')
    print(f'  Throughput gain: {baseline["tok_per_sec"]:,} -> {d_results["tok_per_sec"]:,} tok/s')

  print()
  print('=' * 70)


@app.function(image=image, gpu='H200', timeout=7200, startup_timeout=1800)
def profile_h200():
  """Isolation matrix: 4 configs on the same H200 for fair comparison."""
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import torch

  torch.set_float32_matmul_precision('high')

  props = torch.cuda.get_device_properties(0)
  total_mem = props.total_memory / 1e9
  print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
  print(f'{torch.cuda.get_device_name(0)} | {total_mem:.0f} GB | SM {torch.cuda.get_device_capability(0)}')
  print()

  all_results = {}

  # ================================================================
  # Config A: Baseline — all defaults
  # ================================================================
  print('#' * 70)
  print('#  CONFIG A: BASELINE (Liger SwiGLU+RMSNorm, manual RoPE, compiled NS)')
  print('#' * 70)
  print()

  cfg_a = _make_cfg()
  all_results['A_baseline'] = _run_profile(cfg_a, 'A: BASELINE')

  # Clean GPU state between configs
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  # ================================================================
  # Config B: No Liger RMSNorm (only change: native RMSNorm)
  # ================================================================
  print()
  print('#' * 70)
  print('#  CONFIG B: --no-liger-rmsnorm (native RMSNorm, everything else same)')
  print('#' * 70)
  print()

  cfg_b = _make_cfg(use_liger_rmsnorm=False)
  all_results['B_no_liger_rmsnorm'] = _run_profile(cfg_b, 'B: --no-liger-rmsnorm')

  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  # ================================================================
  # Config C: gram-NS (only change: optimizer)
  # ================================================================
  print()
  print('#' * 70)
  print('#  CONFIG C: --gram-ns (gram-newton-schulz optimizer, everything else same)')
  print('#' * 70)
  print()

  cfg_c = _make_cfg(use_gram_ns=True)
  all_results['C_gram_ns'] = _run_profile(cfg_c, 'C: --gram-ns')

  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  # ================================================================
  # Config D: All optimizations
  # ================================================================
  print()
  print('#' * 70)
  print('#  CONFIG D: ALL (no-liger-rmsnorm + fa-rope + gram-ns)')
  print('#' * 70)
  print()

  cfg_d = _make_cfg(use_liger_rmsnorm=False, use_fa_rope=True, use_gram_ns=True)
  all_results['D_all'] = _run_profile(cfg_d, 'D: ALL OPTIMIZATIONS')

  # ================================================================
  # Comparison matrix
  # ================================================================
  _print_matrix(all_results)

  return all_results


@app.local_entrypoint()
def main():
  result = profile_h200.remote()
  print('\nDone. Results returned as dict with keys:', list(result.keys()))
