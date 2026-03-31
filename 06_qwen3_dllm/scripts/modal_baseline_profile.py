"""Stage 1+2: Baseline profiling + roofline analysis on H100.

Runs with the full V1 stack: Liger + torch.compile + FlexAttention + FP8 + FLCE + SAC.
Produces:
1. Per-op time breakdown (where does time go?) — 25 steps, 10 warmup
2. VRAM breakdown (params + grads + optimizer state + activations)
3. Roofline analysis (compute-bound vs memory-bound per op?)
4. Throughput baseline (tok/s, TFLOPS, GB/s)
5. Batch size sweep (VRAM + throughput for B=1..32)
6. Op microbenchmarks at real shapes

This is THE reference all optimizations compare against.

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_baseline_profile.py
"""

import modal

app = modal.App('smoldlm-baseline-profile')

image = (
  modal.Image
  # Match kernel test image: latest torch for apple-to-apple comparison
  .from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
  .uv_pip_install(
    'torch',  # latest (2.11+) — same as kernel benchmark
    'transformers>=4.45.0',
    'datasets>=2.0.0',
    'liger-kernel>=0.5.9',
    'safetensors>=0.4.0',
    'huggingface_hub>=0.20.0',
    'numpy',
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


@app.function(image=image, gpu='H100', timeout=3600)
def profile_baseline():
  """Full baseline profile at production shapes — V1 config."""
  import sys
  import time

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import torch

  torch.set_float32_matmul_precision('high')

  props = torch.cuda.get_device_properties(0)
  total_mem = props.total_memory / 1e9
  print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
  print(f'{torch.cuda.get_device_name(0)} | {total_mem:.0f} GB | SM {torch.cuda.get_device_capability(0)}')
  print()

  # ================================================================
  # H100 theoretical peaks
  # ================================================================
  H100_BF16_TFLOPS = 990  # tensor core bf16
  H100_FP8_TFLOPS = 1979  # tensor core fp8
  H100_HBM_BW_TBS = 3.35  # HBM3 bandwidth TB/s

  # ================================================================
  # V1 Config — must exactly match train.py production config
  # ================================================================
  from phase6.config import Config

  # Qwen3-0.6B block diffusion — 28L/1024d/16h/8kv
  # V1 flags: Liger + compile + FlexAttention + FP8 + FLCE + SAC (use_grad_ckpt)
  cfg = Config(
    n_layer=28,
    n_embd=1024,
    n_head=16,
    n_kv_head=8,
    head_dim=128,
    mlp_hidden=3072,
    vocab_size=151936,
    seq_len=2048,
    block_size=8,
    rope_base=1_000_000,
    rms_eps=1e-6,
    dropout=0.0,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=True,  # Liger RMSNorm + SwiGLU
    use_grad_ckpt=True,  # Selective Activation Checkpointing (SAC)
    use_flex=True,  # FlexAttention for staircase block mask
    use_muon=True,  # MuonClip optimizer
    use_fp8=True,  # FP8 matmuls (all Linear except lm_head)
    use_compile=True,  # per-block torch.compile
    use_amp=True,  # bfloat16 autocast
    use_flce=True,  # Liger Fused Linear Cross Entropy
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

  print('═' * 65)
  print(' BASELINE PROFILE — V1 (Liger+Compile+Flex+FP8+FLCE+SAC)')
  print('═' * 65)
  print(f'Hardware: {torch.cuda.get_device_name(0)} {total_mem:.0f}GB')
  print(f'Model: Qwen3-0.6B ({cfg.n_layer}L, {cfg.n_embd}d, {cfg.n_head}h, {cfg.n_kv_head}kv)')
  print(f'Batch: B={cfg.batch_size}, L={cfg.seq_len} (effective {cfg.batch_size * cfg.seq_len * 2} tokens/step)')
  print(
    f'Flags: liger={cfg.use_liger} compile={cfg.use_compile} flex={cfg.use_flex} '
    f'fp8={cfg.use_fp8} flce={cfg.use_flce} grad_ckpt={cfg.use_grad_ckpt}'
  )
  print()

  # ================================================================
  # Build model
  # ================================================================
  from phase6.attention import build_staircase_block_mask, build_staircase_mask
  from phase6.loss import compute_loss, compute_loss_flce
  from phase6.model import Model
  from phase6.optim import create_optimizer
  from phase6.toy import make_toy_batch
  from torch import nn

  model = Model(cfg).cuda()

  # FP8 conversion (identical to train.py)
  if cfg.use_fp8:
    from phase6.fp8 import Float8Linear, convert_to_float8_training

    def fp8_ok(m, fqn):
      return isinstance(m, nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0 and fqn != 'lm_head'

    convert_to_float8_training(model, module_filter_fn=fp8_ok)
    n8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    print(f'FP8 training: {n8} layers converted')

  # Per-block compile (train.py: use_grad_ckpt or use_liger → per-block)
  if cfg.use_compile:
    compile_t0 = time.perf_counter()
    for i, blk in enumerate(model.blocks):
      model.blocks[i] = torch.compile(blk, dynamic=False)
    print('Per-block compile registered (will JIT on first forward)')

  optimizer = create_optimizer(model, cfg)
  model.train()

  params_m = model.count_params() / 1e6
  print(f'Parameters: {params_m:.1f}M (tied embeddings counted once)')
  print()

  # ================================================================
  # Stage 0: torch.compile warmup timing
  # ================================================================
  print('=' * 65)
  print('STAGE 0: torch.compile JIT WARMUP (first 10 steps)')
  print('=' * 65)

  if cfg.use_flex:
    mask = build_staircase_block_mask(cfg.seq_len, cfg.block_size)
  else:
    mask = build_staircase_mask(cfg.seq_len, cfg.block_size).to('cuda')

  compile_step_times = []
  for step in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
      if cfg.use_flce:
        loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      else:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    compile_step_times.append(dt)
    tps = int(cfg.batch_size * cfg.seq_len * 2 / (dt / 1000))
    print(f'  warmup step {step:2d} | {dt:8.1f} ms | {tps:,} tok/s')

  compile_wall_s = time.perf_counter() - compile_t0
  print(f'\n  torch.compile JIT total wall time: {compile_wall_s:.1f}s ({compile_wall_s / 60:.1f} min)')
  print(f'  First step: {compile_step_times[0]:.0f} ms → last warmup: {compile_step_times[-1]:.0f} ms')
  print()

  # ================================================================
  # Stage 1: Per-region time breakdown (25 measurement steps)
  # ================================================================
  print('=' * 65)
  print('STAGE 1: PER-REGION TIME BREAKDOWN (25 steps, post-warmup)')
  print('=' * 65)

  torch.cuda.reset_peak_memory_stats()

  region_times = {'data': [], 'forward': [], 'loss': [], 'backward': [], 'optimizer': [], 'total': []}

  for step in range(25):
    # Data loading — should be ~0 (no real prefetcher in profiler, but isolate it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = make_toy_batch(cfg, device='cuda')
    x_input, targets, noise_mask, elbo_w, _doc_ids, positions = batch
    torch.cuda.synchronize()
    dt_data = (time.perf_counter() - t0) * 1000

    # Forward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      hidden, _ = model(x_input, targets, attn_mask=mask, positions=positions)
    torch.cuda.synchronize()
    dt_fwd = (time.perf_counter() - t0) * 1000

    # Loss (FLCE or chunked CE — separate from forward to isolate)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      if cfg.use_flce:
        loss = compute_loss_flce(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
      else:
        loss = compute_loss(hidden, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)
    torch.cuda.synchronize()
    dt_loss = (time.perf_counter() - t0) * 1000

    # Backward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    dt_bwd = (time.perf_counter() - t0) * 1000

    # Optimizer (MuonClip: Newton-Schulz + AdamW for embeddings/scalars)
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

    # tokens/step = B x 2 x seq_len (block diffusion: [x_t || x_0])
    tokens_per_step = cfg.batch_size * cfg.seq_len * 2
    tps = int(tokens_per_step / (dt_total / 1000))
    if step % 5 == 0:
      print(
        f'  step {step:2d} | total {dt_total:7.1f}ms | fwd {dt_fwd:6.1f} | loss {dt_loss:6.1f} | '
        f'bwd {dt_bwd:6.1f} | opt {dt_opt:6.1f} | data {dt_data:5.1f} | {tps:,} tok/s'
      )

  peak_vram_stage1 = torch.cuda.max_memory_allocated() / 1e9

  def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    sorted_v = sorted(vals)
    median = sorted_v[n // 2]
    return mean, std, median

  print('\n--- Per-region stats (25 measured steps) ---')
  region_stats = {}
  for region in ['data', 'forward', 'loss', 'backward', 'optimizer', 'total']:
    mean, std, median = _stats(region_times[region])
    region_stats[region] = (mean, std, median)
    avg_total_ms = _stats(region_times['total'])[0]
    pct = mean / avg_total_ms * 100
    print(f'  {region:12s}: {mean:7.2f} ± {std:5.2f} ms  (median {median:7.2f} ms, {pct:5.1f}%)')

  avg_total_ms = region_stats['total'][0]
  tokens_per_step = cfg.batch_size * cfg.seq_len * 2
  avg_tps = int(tokens_per_step / (avg_total_ms / 1000))
  print(f'\n  Throughput: {avg_tps:,} tok/s  (tokens_per_step={tokens_per_step:,} = B*2*seq_len)')

  # ================================================================
  # Stage 2: Roofline analysis
  # ================================================================
  print()
  print('=' * 65)
  print('STAGE 2: ROOFLINE ANALYSIS')
  print('=' * 65)

  B = cfg.batch_size
  T = cfg.seq_len * 2  # full input: [x_t || x_0]
  D = cfg.n_embd
  H = cfg.n_head
  KV = cfg.n_kv_head
  HD = cfg.head_dim
  MLP = cfg.mlp_hidden
  V = cfg.vocab_size
  L = cfg.n_layer
  Lseq = cfg.seq_len  # half-sequence (model outputs x_pred of this length)
  bpe = 2  # bytes per element (bf16)

  # Per-layer FLOPS (forward only, one layer)
  qkv_flops = 2 * B * T * D * (H * HD + 2 * KV * HD)
  # Staircase block mask: x_t half sees ~50% (causal blocks), x_0 half sees 100%.
  # Effective attention FLOPS ≈ 75% of full T*T — note this in the output.
  attn_flops_dense = 2 * B * H * T * T * HD  # full T*T upper bound
  attn_flops_est = int(attn_flops_dense * 0.75)  # staircase sparsity ~25% saving
  out_proj_flops = 2 * B * T * H * HD * D
  mlp_flops = 2 * B * T * D * MLP * 3  # gate + up + down (SwiGLU)
  norm_flops_per_layer = 2 * B * T * D * 4  # 2 RMSNorms x ~4 FLOPS/elem
  rope_flops_per_layer = 2 * B * T * H * HD * 6  # ~6 FLOPS/elem for Q+K

  # Per-layer bytes (forward only)
  qkv_bytes = (B * T * D + D * (H + 2 * KV) * HD + B * T * (H + 2 * KV) * HD) * bpe
  attn_bytes = (2 * B * H * T * HD + B * H * T * T + B * H * T * HD) * bpe
  out_proj_bytes = (B * T * H * HD + H * HD * D + B * T * D) * bpe
  mlp_bytes = (B * T * D * 3 + D * MLP * 3 + B * T * MLP * 2 + B * T * D) * bpe
  norm_bytes = 4 * B * T * D * bpe  # 2 norms per layer, read+write
  rope_bytes = 2 * B * T * H * HD * bpe  # Q+K read+write
  per_layer_flops = (
    qkv_flops + attn_flops_est + out_proj_flops + mlp_flops + norm_flops_per_layer + rope_flops_per_layer
  )

  # LM head: processes x_pred = x[:, :seq_len] — first half only
  lm_head_flops = 2 * B * Lseq * D * V
  lm_head_bytes = (B * Lseq * D + D * V + B * Lseq * V) * bpe

  total_fwd_flops = L * per_layer_flops + lm_head_flops
  total_train_flops = 3 * total_fwd_flops  # fwd + bwd_data + bwd_weight
  # H100 ridge point: 990 TFLOPS / 3.35 TB/s = 295 FLOPS/byte (bf16)
  ridge_point = H100_BF16_TFLOPS * 1e12 / (H100_HBM_BW_TBS * 1e12)

  print(f'\n  H100: {H100_BF16_TFLOPS} TFLOPS bf16 | {H100_FP8_TFLOPS} TFLOPS fp8 | {H100_HBM_BW_TBS} TB/s HBM3')
  print(f'  Ridge point (bf16): {ridge_point:.0f} FLOPS/byte')
  print('  Note: attention FLOPS use ~75% of dense T*T (staircase sparsity estimate)')
  print()

  ops = [
    ('QKV projections', qkv_flops, qkv_bytes),
    ('Attn QK+SV (est)', attn_flops_est, attn_bytes),
    ('Output projection', out_proj_flops, out_proj_bytes),
    ('MLP SwiGLU 3x', mlp_flops, mlp_bytes),
    ('RMSNorm (2/layer)', norm_flops_per_layer, norm_bytes),
    ('RoPE (Q+K)', rope_flops_per_layer, rope_bytes),
    ('LM head (loss)', lm_head_flops, lm_head_bytes),
  ]

  print(f'  {"Op":22s} | {"GFLOPS/layer":>12s} | {"GB/layer":>9s} | {"AI":>8s} | {"Bound":>10s} | {"% fwd":>8s}')
  print(f'  {"-" * 22}-+-{"-" * 12}-+-{"-" * 9}-+-{"-" * 8}-+-{"-" * 10}-+-{"-" * 8}')

  for name, flops, bytes_rw in ops:
    ai = flops / bytes_rw if bytes_rw > 0 else 0
    bound = 'COMPUTE' if ai > ridge_point else 'MEMORY'
    is_head = name.startswith('LM')
    total_flops_for_op = flops if is_head else flops * L
    pct = total_flops_for_op / total_fwd_flops * 100
    print(f'  {name:22s} | {flops / 1e9:12.1f} | {bytes_rw / 1e9:9.3f} | {ai:8.1f} | {bound:>10s} | {pct:7.1f}%')

  print(f'\n  Total forward FLOPS:   {total_fwd_flops / 1e12:.2f} TFLOPS')
  print(f'  Training FLOPS (3x):   {total_train_flops / 1e12:.2f} TFLOPS')

  achieved_tflops = total_train_flops / (avg_total_ms / 1000) / 1e12
  util_bf16 = achieved_tflops / H100_BF16_TFLOPS * 100
  util_fp8 = achieved_tflops / H100_FP8_TFLOPS * 100
  print(f'  Achieved:              {achieved_tflops:.1f} TFLOPS')
  print(f'  Utilization (bf16):    {util_bf16:.1f}% of {H100_BF16_TFLOPS} TFLOPS peak')
  print(f'  Utilization (fp8):     {util_fp8:.1f}% of {H100_FP8_TFLOPS} TFLOPS peak (matmuls)')

  # ================================================================
  # Stage 3: VRAM breakdown
  # ================================================================
  print()
  print('=' * 65)
  print('STAGE 3: VRAM BREAKDOWN')
  print('=' * 65)

  # Measure after a full step so optimizer state is populated
  param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
  grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
  opt_bytes = sum(
    sum(s.numel() * s.element_size() for s in state.values() if isinstance(s, torch.Tensor))
    for state in optimizer.state.values()
  )
  static_bytes = param_bytes + grad_bytes + opt_bytes
  activation_est = peak_vram_stage1 - static_bytes / 1e9

  print(f'  Model params:        {param_bytes / 1e9:.3f} GB')
  print(f'  Gradients:           {grad_bytes / 1e9:.3f} GB')
  print(f'  Optimizer state:     {opt_bytes / 1e9:.3f} GB')
  print(f'  Static total:        {static_bytes / 1e9:.3f} GB')
  print(f'  Activations (est):   {activation_est:.3f} GB  (peak - static)')
  print(f'  Peak allocated:      {peak_vram_stage1:.3f} GB')
  print(f'  H100 total:          {total_mem:.0f} GB')
  print(f'  Headroom:            {total_mem - peak_vram_stage1:.1f} GB')

  # ================================================================
  # Stage 4: Batch size sweep (VRAM + throughput)
  # ================================================================
  print()
  print('=' * 65)
  print('STAGE 4: BATCH SIZE SWEEP (VRAM + throughput at steady state)')
  print('=' * 65)

  sweep_results = {}
  for bs in [1, 2, 4, 8, 16, 32]:
    try:
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      sweep_cfg = Config(
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        head_dim=cfg.head_dim,
        mlp_hidden=cfg.mlp_hidden,
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        block_size=cfg.block_size,
        rope_base=cfg.rope_base,
        rms_eps=cfg.rms_eps,
        dropout=0.0,
        use_emb_norm=False,
        use_gated_query=False,
        use_qk_norm=cfg.use_qk_norm,
        use_liger=cfg.use_liger,
        use_grad_ckpt=cfg.use_grad_ckpt,
        use_flex=cfg.use_flex,
        use_muon=cfg.use_muon,
        use_fp8=cfg.use_fp8,
        use_compile=cfg.use_compile,
        use_amp=cfg.use_amp,
        use_flce=cfg.use_flce,
        pad_token_id=cfg.pad_token_id,
        mask_token_id=cfg.mask_token_id,
        eos_token_id=cfg.eos_token_id,
        batch_size=bs,
        max_iters=10,
        muon_lr=cfg.muon_lr,
        adamw_lr=cfg.adamw_lr,
        grad_clip=cfg.grad_clip,
        t_min=cfg.t_min,
      ).validate()

      sweep_model = Model(sweep_cfg).cuda()

      if sweep_cfg.use_fp8:
        convert_to_float8_training(sweep_model, module_filter_fn=fp8_ok)

      if sweep_cfg.use_compile:
        for i, blk in enumerate(sweep_model.blocks):
          sweep_model.blocks[i] = torch.compile(blk, dynamic=False)

      sweep_opt = create_optimizer(sweep_model, sweep_cfg)
      sweep_model.train()

      if sweep_cfg.use_flex:
        sweep_mask = build_staircase_block_mask(sweep_cfg.seq_len, sweep_cfg.block_size)
      else:
        sweep_mask = build_staircase_mask(sweep_cfg.seq_len, sweep_cfg.block_size).to('cuda')

      # Warm up 5 steps, measure 5
      step_times = []
      for i in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        b = make_toy_batch(sweep_cfg, device='cuda')
        xi, tgt, nm, ew, _di, pos = b
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
          h, _ = sweep_model(xi, tgt, attn_mask=sweep_mask, positions=pos)
          if sweep_cfg.use_flce:
            loss = compute_loss_flce(h, tgt, nm, ew, sweep_model.lm_head.weight, sweep_cfg)
          else:
            loss = compute_loss(h, tgt, nm, ew, sweep_model.lm_head.weight, sweep_cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sweep_model.parameters(), sweep_cfg.grad_clip)
        sweep_opt.step()
        sweep_opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= 5:
          step_times.append(dt)

      peak_gb = torch.cuda.max_memory_allocated() / 1e9
      mean_dt = sum(step_times) / len(step_times)
      sweep_tps = int(bs * sweep_cfg.seq_len * 2 / (mean_dt / 1000))
      sweep_results[bs] = (peak_gb, sweep_tps, mean_dt)
      print(f'  B={bs:2d}: {peak_gb:5.1f} GB | {sweep_tps:7,} tok/s | {mean_dt:7.1f} ms/step')

      del sweep_model, sweep_opt
      torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
      print(f'  B={bs:2d}: OOM')
      sweep_results[bs] = None
      torch.cuda.empty_cache()

  # ================================================================
  # Stage 5: Per-op microbenchmarks at real shapes
  # ================================================================
  print()
  print('=' * 65)
  print('STAGE 5: OP MICROBENCHMARKS (real shapes, 50 runs each)')
  print('=' * 65)

  def bench(fn, warmup=10, runs=50):
    for _ in range(warmup):
      fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      fn()
      torch.cuda.synchronize()
      times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std

  # RMSNorm at full T=2*seq_len
  x_norm = torch.randn(B * T, D, device='cuda', dtype=torch.bfloat16)
  w_norm = torch.ones(D, device='cuda', dtype=torch.bfloat16)
  m, s = bench(lambda: torch.nn.functional.rms_norm(x_norm, (D,), w_norm, 1e-6))
  print(f'  RMSNorm ({B * T}x{D}):             {m:.3f} ± {s:.3f} ms')

  # GEMM: Q projection (D → HxHD)
  x_q = torch.randn(B * T, D, device='cuda', dtype=torch.bfloat16)
  w_q = torch.randn(H * HD, D, device='cuda', dtype=torch.bfloat16)
  m, s = bench(lambda: x_q @ w_q.T)
  tflops_q = 2 * B * T * D * H * HD / (m / 1000) / 1e12
  print(f'  GEMM Q proj ({B * T}x{D})→({H * HD}):     {m:.3f} ± {s:.3f} ms  {tflops_q:.0f} TFLOPS')

  # GEMM: gate projection (D → MLP)
  w_gate = torch.randn(MLP, D, device='cuda', dtype=torch.bfloat16)
  m, s = bench(lambda: x_q @ w_gate.T)
  tflops_g = 2 * B * T * D * MLP / (m / 1000) / 1e12
  print(f'  GEMM gate  ({B * T}x{D})→({MLP}):  {m:.3f} ± {s:.3f} ms  {tflops_g:.0f} TFLOPS')

  # GEMM: down projection (MLP → D)
  x_mlp = torch.randn(B * T, MLP, device='cuda', dtype=torch.bfloat16)
  w_down = torch.randn(D, MLP, device='cuda', dtype=torch.bfloat16)
  m, s = bench(lambda: x_mlp @ w_down.T)
  tflops_d = 2 * B * T * MLP * D / (m / 1000) / 1e12
  print(f'  GEMM down  ({B * T}x{MLP})→({D}):  {m:.3f} ± {s:.3f} ms  {tflops_d:.0f} TFLOPS')

  # Attention (SDPA with GQA expansion — dense baseline, no FlexAttention)
  q_attn = torch.randn(B, H, T, HD, device='cuda', dtype=torch.bfloat16)
  k_attn = torch.randn(B, KV, T, HD, device='cuda', dtype=torch.bfloat16)
  v_attn = torch.randn(B, KV, T, HD, device='cuda', dtype=torch.bfloat16)
  repeats = H // KV

  def _sdpa_gqa():
    k_exp = k_attn.repeat_interleave(repeats, dim=1)
    v_exp = v_attn.repeat_interleave(repeats, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q_attn, k_exp, v_exp)

  m, s = bench(_sdpa_gqa)
  print(f'  SDPA GQA expand ({B}x{H}x{T}x{HD}): {m:.3f} ± {s:.3f} ms  (dense; FlexAttn handles sparsity)')

  # Cross-entropy at real vocab (chunked, matches compute_loss)
  logits_ce = torch.randn(B * Lseq, V, device='cuda', dtype=torch.bfloat16)
  targets_ce = torch.randint(0, V, (B * Lseq,), device='cuda')
  chunk_size_ce = max(1024, 1_500_000_000 // (V * 2))
  chunk_l = logits_ce[:chunk_size_ce]
  chunk_t = targets_ce[:chunk_size_ce]
  m, s = bench(lambda: torch.nn.functional.cross_entropy(chunk_l.float(), chunk_t, reduction='none'))
  print(f'  CE chunk ({chunk_size_ce}x{V}):  {m:.3f} ± {s:.3f} ms  (1 of {-(-B * Lseq // chunk_size_ce)} chunks)')

  # Newton-Schulz (single param, representative)
  from phase6.optim import MuonClip

  ns_input = torch.randn(MLP, D, device='cuda', dtype=torch.bfloat16)
  m, s = bench(lambda: MuonClip.newton_schulz(ns_input))
  print(f'  Newton-Schulz ({MLP}x{D}):   {m:.3f} ± {s:.3f} ms')

  # Full optimizer step (real gradients)
  def _full_opt_step():
    for p in model.parameters():
      if p.requires_grad:
        p.grad = torch.randn_like(p)
    optimizer.step()

  m, s = bench(_full_opt_step, warmup=3, runs=10)
  print(f'  MuonClip full step:          {m:.3f} ± {s:.3f} ms')

  # RoPE
  q_rope = torch.randn(B, T, H, HD, device='cuda', dtype=torch.bfloat16)
  cos_rope = torch.randn(1, T, 1, HD // 2, device='cuda', dtype=torch.bfloat16)
  sin_rope = torch.randn(1, T, 1, HD // 2, device='cuda', dtype=torch.bfloat16)
  from phase6.attention import _apply_rotary_emb

  m, s = bench(lambda: _apply_rotary_emb(q_rope, cos_rope, sin_rope))
  print(f'  RoPE ({B}x{T}x{H}x{HD}):       {m:.3f} ± {s:.3f} ms')

  # ================================================================
  # FINAL SUMMARY
  # ================================================================
  print()
  print('═' * 65)
  print(' BASELINE PROFILE SUMMARY — V1 (Liger+Compile+Flex+FP8+FLCE+SAC)')
  print('═' * 65)
  print(f'Hardware: {torch.cuda.get_device_name(0)}')
  print(f'Model: Qwen3-0.6B ({params_m:.0f}M params, {cfg.n_layer}L, {cfg.n_embd}d, {cfg.n_head}h, {cfg.n_kv_head}kv)')
  print(f'Batch: B={B}, L={cfg.seq_len} (effective {tokens_per_step:,} tokens/step = B*2*seq_len)')
  print()

  avg_fwd, fwd_std, _ = region_stats['forward']
  avg_loss, loss_std, _ = region_stats['loss']
  avg_bwd, bwd_std, _ = region_stats['backward']
  avg_opt, opt_std, _ = region_stats['optimizer']
  avg_data, data_std, _ = region_stats['data']
  avg_total, total_std, _ = region_stats['total']

  print('Per-step timing (25 steps, 10 compile warmup):')
  print(f'  Forward:    {avg_fwd:7.2f} ± {fwd_std:5.2f} ms')
  print(f'  Loss:       {avg_loss:7.2f} ± {loss_std:5.2f} ms')
  print(f'  Backward:   {avg_bwd:7.2f} ± {bwd_std:5.2f} ms')
  print(f'  Optimizer:  {avg_opt:7.2f} ± {opt_std:5.2f} ms')
  print(f'  Data:       {avg_data:7.2f} ± {data_std:5.2f} ms')
  print(f'  Total:      {avg_total:7.2f} ± {total_std:5.2f} ms')
  print()
  print(f'Throughput: {avg_tps:,} tok/s')
  print(f'FLOPS: {achieved_tflops:.1f} TFLOPS ({util_bf16:.1f}% of bf16 peak, {util_fp8:.1f}% of fp8 peak)')
  print(f'Peak VRAM: {peak_vram_stage1:.1f} / {total_mem:.0f} GB')
  print(f'torch.compile warmup: {compile_wall_s:.0f}s ({compile_wall_s / 60:.1f} min)')
  print()

  print('VRAM Batch Size Sweep:')
  for bs in [1, 2, 4, 8, 16, 32]:
    if bs in sweep_results and sweep_results[bs] is not None:
      gb, tps_bs, dt_bs = sweep_results[bs]
      print(f'  B={bs:2d}: {gb:5.1f} GB | {tps_bs:>8,} tok/s | {dt_bs:7.1f} ms/step')
    else:
      print(f'  B={bs:2d}: OOM')

  return {
    'step_ms': avg_total,
    'step_ms_std': total_std,
    'forward_ms': avg_fwd,
    'loss_ms': avg_loss,
    'backward_ms': avg_bwd,
    'optimizer_ms': avg_opt,
    'data_ms': avg_data,
    'tok_per_sec': avg_tps,
    'tokens_per_step': tokens_per_step,
    'tflops': achieved_tflops,
    'util_bf16_pct': util_bf16,
    'util_fp8_pct': util_fp8,
    'peak_vram_gb': peak_vram_stage1,
    'compile_warmup_s': compile_wall_s,
    'param_gb': param_bytes / 1e9,
    'grad_gb': grad_bytes / 1e9,
    'optimizer_gb': opt_bytes / 1e9,
    'activation_est_gb': activation_est,
    'batch_sweep': {
      bs: {'vram_gb': r[0], 'tok_per_sec': r[1], 'step_ms': r[2]} if r else None for bs, r in sweep_results.items()
    },
  }
