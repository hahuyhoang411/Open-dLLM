"""VRAM + throughput probe for B200 (192 GB).

No grad_ckpt, per-block compile + Liger + FP8 + AMP bf16.
Tests batch=64 with 5 warmup + 20 bench steps.

Usage:
    python 05_optimized_dllm/vram_probe.py
"""

import gc
import sys
import os
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn

# No grad_ckpt, compile + liger + fp8
sys.argv = [
    'vram_probe.py', '--train',
    '--no-grad-ckpt', '--fp8',
    '--batch-size', '1', '--max-iters', '1',
]

from phase5 import config
from phase5.model import Model
from phase5.loss import compute_loss
from phase5.attention import build_staircase_block_mask
from phase5.schedule import apply_noise, compute_elbo_weight, sample_timesteps
from phase5.fp8 import convert_to_float8_training

BATCH_SIZES = [96, 80, 64]
WARMUP_STEPS = 5
BENCH_STEPS = 20


def make_dummy_batch(bs):
    L = config.seq_len
    targets = torch.randint(3, config.vocab_size, (bs, L), device='cuda')
    doc_ids = torch.zeros(bs, L, dtype=torch.long, device='cuda')
    _, t = sample_timesteps(bs, config.num_blocks, config.block_size)
    x_noisy, noise_mask = apply_noise(targets, t, pad_token_id=config.pad_token_id)
    elbo_w = compute_elbo_weight(t)
    x_input = torch.cat([x_noisy, targets], dim=1).cuda()
    noise_mask = noise_mask.cuda()
    elbo_w = elbo_w.cuda()
    positions = torch.arange(L, device='cuda').unsqueeze(0).expand(bs, -1)
    return x_input, targets, noise_mask, elbo_w, doc_ids, positions


def bench_batch(model, lm_head_weight, optimizer, bs):
    """Run warmup + bench steps. Returns (tok_per_sec, peak_gb) or None on OOM."""
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    total_steps = WARMUP_STEPS + BENCH_STEPS
    step_times = []

    try:
        for step in range(total_steps):
            x_input, targets, mask, elbo_w, doc_ids, positions = make_dummy_batch(bs)
            attn_mask = build_staircase_block_mask(config.seq_len, config.block_size, doc_ids=doc_ids)

            torch.cuda.synchronize()
            t0 = time.time()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                hidden, _ = model(x_input, targets, attn_mask=attn_mask, positions=positions)
                loss = compute_loss(hidden, targets, mask, elbo_w, lm_head_weight, use_liger=config.use_liger)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            dt = time.time() - t0

            if step >= WARMUP_STEPS:
                step_times.append(dt)

            tokens = bs * config.seq_len
            marker = ' <-- bench starts' if step == WARMUP_STEPS else ''
            print(f'  step {step}: {dt:.3f}s ({int(tokens / dt):,} tok/s){marker}')

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        avg_dt = sum(step_times) / len(step_times)
        tps = bs * config.seq_len / avg_dt
        return tps, peak_gb, avg_dt

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


def main():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu_name} ({total_gb:.1f} GB)')
    print(f'Config: no_grad_ckpt, compile=per-block, fp8={config.use_fp8}, liger={config.use_liger}')
    print(f'Warmup={WARMUP_STEPS}, Bench={BENCH_STEPS}')
    print()

    model = Model().to('cuda')
    print(f'{model.count_params() / 1e6:.2f}M parameters')

    def _fp8_filter(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if fqn == 'lm_head':
            return False
        return True

    convert_to_float8_training(model, module_filter_fn=_fp8_filter)
    lm_head_weight = model.lm_head.weight

    if config.use_muon:
        from phase5.optim import build_param_groups
        optimizer = build_param_groups(model)
    else:
        from phase5.optim import build_adamw_optimizer
        optimizer = build_adamw_optimizer(model)

    for i, block in enumerate(model.blocks):
        model.blocks[i] = torch.compile(block, dynamic=True)
    print('Per-block compile ready.')
    print()

    results = []
    for bs in BATCH_SIZES:
        print(f'--- batch={bs} ---')
        result = bench_batch(model, lm_head_weight, optimizer, bs)
        if result:
            tps, peak, avg_dt = result
            headroom = total_gb - peak
            print(f'  → {int(tps):,} tok/s | {avg_dt:.3f}s/step | {peak:.1f}/{total_gb:.0f} GB ({headroom:.1f} GB free)')
            results.append((bs, tps, peak, avg_dt))
        else:
            print(f'  → OOM')
            results.append((bs, 0, 0, 0))
        print()

    # Summary
    print('=' * 70)
    print(f'{"Batch":>6} {"tok/s":>10} {"s/step":>8} {"Peak GB":>9} {"Headroom":>9} {"4-GPU tok/s":>12}')
    print('-' * 70)
    for bs, tps, peak, avg_dt in results:
        if tps > 0:
            headroom = total_gb - peak
            print(f'{bs:>6} {int(tps):>10,} {avg_dt:>8.3f} {peak:>7.1f} GB {headroom:>7.1f} GB {int(tps * 4):>12,}')
        else:
            print(f'{bs:>6} {"OOM":>10}')

    print()
    print('Projected training time (100B tokens, 4 GPUs):')
    for bs, tps, peak, avg_dt in results:
        if tps > 0:
            four_gpu_tps = tps * 4
            hours = 100_000_000_000 / four_gpu_tps / 3600
            cost_h100 = hours * 4 * 3.95
            print(f'  batch={bs}: {hours:.1f} hours (~${cost_h100:,.0f} at H100 rates)')


if __name__ == '__main__':
    main()
