"""Phase 5 training script — orchestrates all modules for block diffusion LM training.

Usage:
    uv run python 05_phase5_dllm/train.py --train
    uv run python 05_phase5_dllm/train.py --prompt "Once upon a time"
    torchrun --nproc_per_node=2 05_phase5_dllm/train.py --train

Architecture: 30L/576d/9h/3kv/1536MLP, ~144M params, linear noise schedule,
MuonClip optimizer, document packing, Gated Query Attention.
"""

import contextlib
import gc
import math
import os
import pathlib
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

# Phase 5 modules
from phase5 import config
from phase5.attention import build_staircase_block_mask, build_staircase_mask
from phase5.checkpoint import load_checkpoint, save_checkpoint
from phase5.generate import generate
from phase5.loss import compute_loss
from phase5.model import Model
from phase5.fp8 import convert_to_float8_training, disable_fp8
from phase5.optim import build_adamw_optimizer, build_param_groups
from phase5.schedule import get_lr_factor
from phase5.tokenizer import decode, encode
from torch.nn.parallel import DistributedDataParallel as DDP

try:
  import trackio

  _TRACKIO_AVAILABLE = config._TRACKIO_AVAILABLE
except ImportError:
  _TRACKIO_AVAILABLE = False


def _ddp_verify(raw_model, step, loss_val, grad_norm_val):
  """DDP health check: verify all ranks have identical model state. --debug only."""
  if not config.ddp:
    return
  param_sum = sum(p.data.double().sum().item() for p in raw_model.parameters())
  info = torch.tensor([param_sum, loss_val, grad_norm_val], dtype=torch.float64, device=config.device)
  gathered = [torch.zeros_like(info) for _ in range(config.ddp_world_size)]
  dist.all_gather(gathered, info)
  if config.master_process:
    for r, t in enumerate(gathered):
      ps, lo, gn = t[0].item(), t[1].item(), t[2].item()
      print(f'  [DDP] rank {r}: param_sum={ps:.4f} loss={lo:.6f} grad_norm={gn:.6f}')
    param_sums = [t[0].item() for t in gathered]
    grad_norms = [t[2].item() for t in gathered]
    max_p_delta = max(abs(p - param_sums[0]) for p in param_sums[1:])
    max_g_delta = max(abs(g - grad_norms[0]) for g in grad_norms[1:])
    if max_p_delta > 1e-2:
      print(f'  [DDP] WARNING: param_sum diverged! max_delta={max_p_delta:.6f}')
    else:
      print(f'  [DDP] OK: params in sync (max_delta={max_p_delta:.2e})')
    if step >= 0 and max_g_delta > 1e-4:
      print(f'  [DDP] WARNING: grad_norm diverged! max_delta={max_g_delta:.6f}')
    elif step >= 0:
      print(f'  [DDP] OK: grad_norm in sync (max_delta={max_g_delta:.2e})')


def _debug_grad_check(raw_model, loss, step):
  """Per-param gradient inspection. --debug only."""
  n_grad, n_none, n_zero = 0, 0, 0
  manual_sq = 0.0
  for name, p in raw_model.named_parameters():
    if p.grad is not None:
      n_grad += 1
      gnorm = p.grad.float().norm().item()
      manual_sq += gnorm**2
      if gnorm == 0:
        n_zero += 1
      if n_grad <= 3:
        print(f'  [GRAD] {name}: grad.norm={gnorm:.6e} dtype={p.grad.dtype} shape={list(p.grad.shape)}')
    else:
      n_none += 1
      if n_none <= 3:
        print(f'  [GRAD] {name}: grad=None')
  manual_norm = math.sqrt(manual_sq)
  print(f'  [GRAD] summary: {n_grad} with grad, {n_none} None, {n_zero} zero | manual_norm={manual_norm:.6e}')
  print(
    f'  [GRAD] loss: requires_grad={loss.requires_grad}, '
    f'grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}'
  )


def _build_attn_mask(doc_ids):
  """Build staircase attention mask, with doc boundary constraint if packing."""
  if config.use_flex and config._FLEX_AVAILABLE:
    return build_staircase_block_mask(config.seq_len, config.block_size, doc_ids=doc_ids)
  # Fallback: dense float mask with doc boundary constraint.
  return build_staircase_mask(config.seq_len, config.block_size, doc_ids=doc_ids).to(config.device)


@torch.no_grad()
def estimate_loss(model, get_batch_fn, *, eval_iters=None, splits=('train', 'val'), reset_val_fn=None):
  model_was_training = model.training
  model.eval()
  out = {}
  eval_steps = config.eval_iters if eval_iters is None else eval_iters
  for split in splits:
    if split == 'val' and reset_val_fn is not None:
      reset_val_fn()
    losses = torch.zeros(eval_steps)
    for k in range(eval_steps):
      x_input, targets, mask, elbo_w, doc_ids, positions = get_batch_fn(split)
      attn_mask = _build_attn_mask(doc_ids)
      with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_amp):
        hidden, _ = model(x_input, targets, attn_mask=attn_mask, positions=positions)
        loss = compute_loss(hidden, targets, mask, elbo_w, model.lm_head.weight, use_liger=config.use_liger)
      losses[k] = loss.item()
    out[split] = losses.mean().item()
  if model_was_training:
    model.train()
  return out


if __name__ == '__main__':
  # Lazy import data module (triggers dataset connection)
  from phase5.data import get_batch, reset_val_loader

  # --- 1. Print configuration ---
  if config.master_process:
    print('=' * 60)
    print('phase5_dllm — modern block diffusion LM (Phase 5)')
    print('=' * 60)
    print(f'  n_layer        = {config.n_layer}')
    print(f'  n_embd         = {config.n_embd}')
    print(f'  n_head         = {config.n_head}')
    print(f'  n_kv_head      = {config.n_kv_head} (GQA {config.n_head // config.n_kv_head}:1)')
    print(f'  mlp_hidden     = {config.mlp_hidden}')
    print(f'  seq_len        = {config.seq_len}')
    print(f'  block_size     = {config.block_size}')
    print(f'  num_blocks     = {config.num_blocks}')
    eff = config.batch_size * config.grad_accum_steps * config.ddp_world_size
    print(f'  batch_size     = {config.batch_size} (effective: {eff})')
    print(f'  vocab_size     = {config.vocab_size}')
    print(f'  max_iters      = {config.max_iters}')
    print(f'  warmup         = {config.warmup_iters}, decay_start = {config.decay_start}')
    print(f'  lr             = Muon {config.muon_lr} / AdamW {config.adamw_lr}')
    print(f'  dropout        = {config.dropout}')
    print(f'  device         = {config.device}')
    print(f'  ddp            = {config.ddp} (world_size={config.ddp_world_size})')
    print(f'  use_amp        = {config.use_amp}')
    print(f'  use_liger      = {config.use_liger}')
    print(f'  use_flex       = {config.use_flex}')
    print(f'  use_cart       = {config.use_cart}')
    compile_mode = ('whole-model' if not config.use_grad_ckpt else 'per-block') if config.use_compile else 'off'
    print(f'  use_compile    = {config.use_compile} ({compile_mode})')
    print(f'  use_grad_ckpt  = {config.use_grad_ckpt}')
    print(f'  use_muon       = {config.use_muon}')
    print(f'  use_fp8        = {config.use_fp8}')
    print('  noise_schedule = linear (ELBO weight = 1/t)')
    print('  doc_packing    = True')
    print('  gated_query    = True (arXiv:2505.06708)')
    print('=' * 60)

  # --- 2. Weights path ---
  script_dir = os.path.dirname(os.path.abspath(__file__))
  weights_path = os.path.join(script_dir, 'weights', f'phase5_dllm_b{config.block_size}.pt')
  pathlib.Path(os.path.dirname(weights_path)).mkdir(exist_ok=True, parents=True)

  # --- 3. Instantiate model ---
  model = Model().to(config.device)
  if config.master_process:
    print(f'{model.count_params() / 1e6:.2f}M parameters')
    if torch.cuda.is_available():
      print(f'Peak VRAM after model init: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

  raw_model = model

  # --- 4. Load or train ---
  if pathlib.Path(weights_path).exists() and not config.args.train:
    raw_model.load_state_dict(torch.load(weights_path, map_location=config.device, weights_only=True))
    if config.master_process:
      print(f'Loaded weights from {weights_path}')

  elif config.args.train:
    # Build optimizer on uncompiled model (clean param names)
    if config.use_muon:
      optimizer = build_param_groups(raw_model)
      if config.master_process:
        counts = [len(g['params']) for g in optimizer.param_groups]
        print(f'MuonClip optimizer: {counts} params per group (QK Muon / Other Muon / AdamW)')
    else:
      optimizer = build_adamw_optimizer(raw_model)
      if config.master_process:
        print('AdamW optimizer (--no-muon)')

    # Resume from checkpoint (into uncompiled model — clean keys)
    ckpt_dir = config.args.ckpt_dir or config.args.resume or os.path.join(script_dir, 'weights')
    start_step = 0
    if config.args.resume:
      start_step = load_checkpoint(config.args.resume, raw_model, optimizer, config.device)
      if start_step <= 0:
        start_step = 0
        if config.master_process:
          print(f'No checkpoint at {config.args.resume}, starting fresh')
      elif config.master_process:
        print(f'Resumed from step {start_step}')

    # FP8 conversion (before compile — Dynamo sees Float8Linear, not nn.Linear)
    if config.use_fp8:

      def _fp8_filter(mod, fqn):
        if not isinstance(mod, nn.Linear):
          return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
          return False
        if fqn == 'lm_head':
          return False
        return True

      from phase5.fp8 import Float8Linear

      num_linear = sum(1 for m in raw_model.modules() if isinstance(m, nn.Linear))
      convert_to_float8_training(raw_model, module_filter_fn=_fp8_filter)
      num_fp8 = sum(1 for m in raw_model.modules() if isinstance(m, Float8Linear))
      if config.master_process:
        print(f'FP8 training: converted {num_fp8}/{num_linear} linear layers')

    # Compile AFTER checkpoint loaded (torchtitan order: AC → load → compile → DDP)
    # Without grad_ckpt: whole-model compile (cross-block fusion, fewer kernel launches)
    #   - Liger in-model ops auto-disabled (Inductor fuses natively)
    # With grad_ckpt: per-block compile (checkpoint() inside compiled graph crashes)
    #   - Liger RMSNorm + SwiGLU used for eager-mode fusion
    if config.use_compile:
      if config.use_grad_ckpt:
        for i, block in enumerate(raw_model.blocks):
          raw_model.blocks[i] = torch.compile(block, dynamic=False)
        if config.master_process:
          print(f'torch.compile: per-block ({len(raw_model.blocks)} blocks)')
      else:
        model = torch.compile(model, dynamic=False)
        if config.master_process:
          print('torch.compile: whole-model (no grad_ckpt, no Liger in-model)')

    if config.ddp:
      model = DDP(
        model, device_ids=[int(config.device.split(':')[-1])], gradient_as_bucket_view=True, broadcast_buffers=False
      )

    effective_batch = config.batch_size * config.grad_accum_steps * config.ddp_world_size

    # Preload datasets on all ranks (avoids DDP timeout)
    if config.master_process:
      print('Preloading datasets...')
    get_batch('train')
    if config.ddp:
      dist.barrier()
    if config.master_process:
      print('Datasets ready.')
      if config.use_compile:
        print(
          'torch.compile warm-up — first forward triggers Triton compilation '
          '(expect 5-15 min, blocks cached after first)...'
        )
      if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # DDP: verify initial parameter sync
    if config.debug and config.ddp:
      _ddp_verify(raw_model, -1, 0.0, 0.0)
      if config.master_process:
        print('[DEBUG] DDP initial param sync verified.')

    # Trackio experiment tracking
    if config.master_process and _TRACKIO_AVAILABLE:
      trackio_cfg = {
        'n_layer': config.n_layer,
        'n_embd': config.n_embd,
        'n_head': config.n_head,
        'n_kv_head': config.n_kv_head,
        'mlp_hidden': config.mlp_hidden,
        'seq_len': config.seq_len,
        'block_size': config.block_size,
        'batch_size': effective_batch,
        'max_iters': config.max_iters,
        'use_muon': config.use_muon,
        'use_cart': config.use_cart,
        'ddp_world_size': config.ddp_world_size,
      }
      trackio_kw = {'project': 'open-dllm-phase5', 'config': trackio_cfg}
      if config.args.trackio_space:
        trackio_kw['space_id'] = config.args.trackio_space
      try:
        trackio.init(**trackio_kw)
      except Exception as e:
        print(f'WARNING: trackio init failed ({e}), continuing without tracking')
        _TRACKIO_AVAILABLE = False

    # --- Training loop ---
    losses = {}
    total_training_time = 0.0
    _WARMUP_STEPS = 10  # skip compilation warm-up for throughput measurement
    for step in range(start_step, config.max_iters):
      # WSD LR schedule
      lr_factor = get_lr_factor(step)
      for pg in optimizer.param_groups:
        pg['lr'] = pg['initial_lr'] * lr_factor

      # Periodic evaluation
      if step % config.eval_interval == 0 or step == config.max_iters - 1:
        # FP8 stays active during eval — quantization noise is negligible for
        # loss estimation, and swapping modules invalidates torch.compile cache
        # (forces full Triton recompilation with different module types).
        if step == 0:
          startup_eval_iters = max(1, min(8, config.eval_iters))
        else:
          startup_eval_iters = config.eval_iters
        losses = estimate_loss(
          raw_model,
          get_batch,
          eval_iters=startup_eval_iters,
          reset_val_fn=reset_val_loader,
        )
        # Step-0 sanity check: catch broken normalization/ELBO before burning GPU
        # With SmolLM2 init (std=1/√576), ELBO-weighted loss is ~19-20 at step 0
        # because mask_token_id=0 inputs produce higher CE than real tokens.
        # Raw CE is ~9.6 (correct, below ln(49152)=10.80).
        if step == 0 and losses.get('train', 0) > 25:
          raise RuntimeError(
            f'Step-0 train loss {losses["train"]:.2f} >> expected ~19-20. Check loss normalization and ELBO weighting.'
          )
        if config.master_process:
          lr = optimizer.param_groups[0]['lr']
          print(f'step {step:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f} | lr {lr:.6f}')
          if _TRACKIO_AVAILABLE:
            trackio.log({
              'eval_train_loss': losses['train'],
              'eval_val_loss': losses['val'],
              'eval_lr': lr,
            })
          if step > 0:
            with disable_fp8(raw_model) if config.use_fp8 else contextlib.nullcontext():
              sample = generate(
                raw_model,
                encode,
                decode,
                prompt='Vietnam, officially the Socialist Republic of Vietnam, is a country',
                max_new_tokens=64,
                temperature=0.8,
                top_k=5,
              )
            print(f'--- sample ---\n{sample[:300]}\n--- end sample ---')
          if torch.cuda.is_available():
            torch.cuda.empty_cache()

      # Checkpoint (decoupled from eval for preemption resilience)
      if step > 0 and step % config.args.ckpt_interval == 0 and config.master_process:
        ckpt_path = save_checkpoint(
          raw_model,
          optimizer,
          step,
          losses.get('val', 0.0) if step % config.eval_interval == 0 else 0.0,
          ckpt_dir,
        )
        print(f'Saved checkpoint: {ckpt_path}')

      # Per-step timing (nanochat pattern: synchronize at both ends)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      step_t0 = time.time()

      # Gradient accumulation with DDP no_sync
      accum_loss = 0.0
      for micro_step in range(config.grad_accum_steps):
        no_sync = config.ddp and micro_step < config.grad_accum_steps - 1
        ctx = model.no_sync() if no_sync else contextlib.nullcontext()
        with ctx:
          x_input, targets, mask, elbo_w, doc_ids, positions = get_batch('train')
          attn_mask = _build_attn_mask(doc_ids)
          with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=config.use_amp):
            hidden, _ = model(x_input, targets, attn_mask=attn_mask, positions=positions)
            loss = compute_loss(hidden, targets, mask, elbo_w, raw_model.lm_head.weight, use_liger=config.use_liger)
            loss = loss / config.grad_accum_steps
          accum_loss += loss.detach().item()
          if config.debug and step == start_step and config.master_process:
            print(
              f'  [PRE-BW] loss={loss.item():.4f} requires_grad={loss.requires_grad} '
              f'grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}'
            )
            print(
              f'  [PRE-BW] hidden: requires_grad={hidden.requires_grad} '
              f'grad_fn={type(hidden.grad_fn).__name__ if hidden.grad_fn else None} '
              f'shape={list(hidden.shape)} dtype={hidden.dtype}'
            )
          loss.backward()

      # Grad clip + optimizer step
      if config.debug and step == start_step and config.master_process:
        _debug_grad_check(raw_model, loss, step)
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      gn = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
      if config.debug and step == start_step and config.master_process:
        print(f'  [GRAD] clip_grad_norm_ returned: {gn:.6e}')
      if config.debug and step - start_step < 5:
        _ddp_verify(raw_model, step, accum_loss, gn)

      # Per-step timing (exclude compilation warm-up from average)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      step_dt = time.time() - step_t0
      steps_since_start = step - start_step
      if steps_since_start >= _WARMUP_STEPS:
        total_training_time += step_dt
      instant_tps = int(effective_batch * config.seq_len / max(step_dt, 1e-9))
      steady_steps = steps_since_start - _WARMUP_STEPS + 1
      avg_tps = (
        int((steady_steps * effective_batch * config.seq_len) / max(total_training_time, 1e-9))
        if total_training_time > 0
        else 0
      )

      # GC management (nanochat pattern: disable after first step, collect every 5K)
      if step == start_step:
        gc.collect()
        gc.freeze()
        gc.disable()
      elif (step - start_step) % 5000 == 0:
        gc.collect()

      # Progress logging every 10 steps
      if config.master_process and step % 10 == 0 and step > 0:
        vram = ''
        if torch.cuda.is_available():
          peak = torch.cuda.max_memory_allocated() / 1e9
          alloc = torch.cuda.memory_allocated() / 1e9
          vram = f' | VRAM {alloc:.1f}/{peak:.1f} GB'
        print(
          f'  step {step:5d} | loss {accum_loss:.4f} '
          f'| grad_norm {gn:.2f} | {instant_tps:,} tok/s '
          f'(avg {avg_tps:,}){vram}'
        )
        if _TRACKIO_AVAILABLE:
          trackio.log({
            'train_loss': accum_loss,
            'train_lr': optimizer.param_groups[0]['lr'],
            'train_grad_norm': gn,
            'train_tokens_per_sec': instant_tps,
          })

      # First step VRAM report
      if step == start_step and config.master_process and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
          f'VRAM after step 0: {alloc:.2f} GB alloc, '
          f'{peak:.2f} GB peak, {total:.1f} GB total '
          f'({peak / total * 100:.0f}% used)'
        )

    # Save final checkpoint + weights
    if config.master_process:
      save_checkpoint(raw_model, optimizer, config.max_iters, 0.0, ckpt_dir)
      torch.save(raw_model.state_dict(), weights_path)
      print(f'Saved weights to {weights_path}')
    if config.master_process and _TRACKIO_AVAILABLE:
      trackio.finish()

  elif config.args.prompt is not None:
    print(f'No weights found at {weights_path}')
    print('Run with --train to train from scratch.')
    sys.exit(1)

  # --- 5. Generate text ---
  if config.master_process and config.args.prompt is not None:
    with disable_fp8(raw_model) if config.use_fp8 else contextlib.nullcontext():
      sample = generate(
        raw_model,
        encode,
        decode,
        prompt=config.args.prompt,
        max_new_tokens=config.args.max_tokens,
        denoise_steps=config.args.denoise_steps,
        temperature=0.8,
        top_k=5,
      )
    print(sample)

  if config.ddp:
    dist.destroy_process_group()
