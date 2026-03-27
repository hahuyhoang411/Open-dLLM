"""Phase 6 training -- Qwen3 block diffusion LM.

Usage: uv run python 06_qwen3_dllm/train.py --train [--hf-model-name Qwen/Qwen3-0.6B]
       torchrun --nproc_per_node=N 06_qwen3_dllm/train.py --train
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
from torch.nn.parallel import DistributedDataParallel as DDP

from phase6.attention import build_staircase_block_mask, build_staircase_mask
from phase6.checkpoint import load_checkpoint, load_from_hf, save_checkpoint
from phase6.config import from_cli, setup_device, setup_features, _check_flex, _check_trackio
from phase6.fp8 import Float8Linear, convert_to_float8_training, disable_fp8
from phase6.loss import compute_loss
from phase6.model import Model
from phase6.optim import create_optimizer
from phase6.schedule import get_lr_factor
from phase6.tokenizer import decode, encode


# ============================================================================
# Helpers (extracted for testability)
# ============================================================================

def _build_attn_mask(doc_ids, cfg):
    """Build staircase attention mask, with doc boundary constraint if packing."""
    if cfg.use_flex and _check_flex():
        return build_staircase_block_mask(cfg.seq_len, cfg.block_size, doc_ids=doc_ids)
    return build_staircase_mask(cfg.seq_len, cfg.block_size, doc_ids=doc_ids).to(cfg.device)


def _apply_lr_schedule(optimizer, step, cfg):
    """Apply WSD LR schedule to all param groups. Returns current lr_factor."""
    factor = get_lr_factor(step, cfg.warmup_iters, cfg.decay_start, cfg.max_iters)
    for pg in optimizer.param_groups:
        pg['lr'] = pg['initial_lr'] * factor
    return factor


def _step0_sanity(loss_val, cfg):
    """Raise if step-0 loss is catastrophically wrong.

    Qwen3 vocab=151936: ln(151936) ~ 11.93 raw CE, ELBO-weighted ~ 24.
    With HF pretrained weights: expect lower (model knows language).
    Threshold: 30 for random init, anything above is broken normalization.
    """
    threshold = 30.0
    if loss_val > threshold:
        raise RuntimeError(
            f'Step-0 train loss {loss_val:.2f} >> expected ~24 (random) or lower (pretrained). '
            f'Check loss normalization and ELBO weighting.'
        )


def _generate_sample(model, cfg, prompt='The capital of France is', max_new_tokens=64,
                     temperature=0.8):
    """Try to generate text. Returns None if generate module is not available."""
    try:
        from phase6.generate import generate
        from phase6.tokenizer import encode, decode
        prompt_ids = encode(prompt)
        with disable_fp8(model) if cfg.use_fp8 else contextlib.nullcontext():
            token_ids = generate(model, prompt_ids, cfg,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature)
            return decode(token_ids)
    except (ImportError, AttributeError, TypeError):
        return None


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def estimate_loss(model, get_batch_fn, cfg, *, eval_iters=None, splits=('train', 'val'),
                  reset_val_fn=None):
    model_was_training = model.training
    model.eval()
    out = {}
    n_iters = eval_iters if eval_iters is not None else cfg.eval_iters
    for split in splits:
        if split == 'val' and reset_val_fn is not None:
            reset_val_fn()
        losses = torch.zeros(n_iters)
        for k in range(n_iters):
            x_input, targets, mask, elbo_w, doc_ids, positions = get_batch_fn(split, cfg)
            attn_mask = _build_attn_mask(doc_ids, cfg)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_amp):
                hidden, _ = model(x_input, targets, attn_mask=attn_mask, positions=positions)
                loss = compute_loss(hidden, targets, mask, elbo_w, model.lm_head.weight, cfg)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if model_was_training:
        model.train()
    return out


# ============================================================================
# Debug helpers
# ============================================================================

def _ddp_verify(raw_model, step, loss_val, grad_norm_val, cfg):
    """DDP health check: verify all ranks have identical model state. --debug only."""
    if not cfg.ddp:
        return
    param_sum = sum(p.data.double().sum().item() for p in raw_model.parameters())
    info = torch.tensor([param_sum, loss_val, grad_norm_val], dtype=torch.float64, device=cfg.device)
    gathered = [torch.zeros_like(info) for _ in range(cfg.ddp_world_size)]
    dist.all_gather(gathered, info)
    if cfg.master_process:
        psums = [t[0].item() for t in gathered]
        gnorms = [t[2].item() for t in gathered]
        pd = max(abs(p - psums[0]) for p in psums[1:])
        gd = max(abs(g - gnorms[0]) for g in gnorms[1:])
        status_p = 'WARN' if pd > 1e-2 else 'OK'
        status_g = 'WARN' if (step >= 0 and gd > 1e-4) else 'OK'
        print(f'  [DDP] params {status_p} (delta={pd:.2e}), grads {status_g} (delta={gd:.2e})')


def _debug_grad_check(raw_model, loss):
    """Per-param gradient inspection. --debug only."""
    n_grad, n_none, n_zero, sq = 0, 0, 0, 0.0
    for name, p in raw_model.named_parameters():
        if p.grad is not None:
            n_grad += 1
            gn = p.grad.float().norm().item()
            sq += gn**2
            n_zero += (gn == 0)
            if n_grad <= 3:
                print(f'  [GRAD] {name}: norm={gn:.6e}')
        else:
            n_none += 1
    print(f'  [GRAD] {n_grad} grads, {n_none} None, {n_zero} zero | total={math.sqrt(sq):.6e}')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    cfg = from_cli()
    cfg = setup_device(cfg)
    cfg = setup_features(cfg)

    trackio_ok = _check_trackio()
    if trackio_ok:
        import trackio
    from phase6.data import get_batch, reset_val_loader

    # --- 1. Print configuration ---
    if cfg.master_process:
        eff = cfg.batch_size * cfg.grad_accum_steps * cfg.ddp_world_size
        cm = ('per-block' if cfg.use_grad_ckpt or cfg.use_liger else 'whole-model') if cfg.use_compile else 'off'
        lines = [
            '=' * 60, 'phase6_dllm -- Qwen3 block diffusion LM (Phase 6)', '=' * 60,
            f'  arch           = {cfg.n_layer}L/{cfg.n_embd}d/{cfg.n_head}h/{cfg.n_kv_head}kv/{cfg.mlp_hidden}MLP (head_dim={cfg.head_dim})',
            f'  seq/block      = {cfg.seq_len}/{cfg.block_size} ({cfg.num_blocks} blocks)',
            f'  batch_size     = {cfg.batch_size} (effective: {eff})',
            f'  vocab_size     = {cfg.vocab_size}',
            f'  iters          = {cfg.max_iters} (warmup={cfg.warmup_iters}, decay={cfg.decay_start})',
            f'  lr             = Muon {cfg.muon_lr} / AdamW {cfg.adamw_lr}',
            f'  device         = {cfg.device}, ddp={cfg.ddp} (world_size={cfg.ddp_world_size})',
            f'  flags          = amp={cfg.use_amp} liger={cfg.use_liger} flex={cfg.use_flex} cart={cfg.use_cart}',
            f'                   compile={cfg.use_compile}({cm}) grad_ckpt={cfg.use_grad_ckpt} muon={cfg.use_muon} fp8={cfg.use_fp8}',
            f'                   qk_norm={cfg.use_qk_norm} dropout={cfg.dropout}',
            f'  hf_model       = {cfg.hf_model_name}',
            '  noise_schedule = linear (ELBO 1/t), doc_packing + per-doc t', '=' * 60,
        ]
        print('\n'.join(lines))

    # --- 2. Weights path + model ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'weights', f'phase6_dllm_b{cfg.block_size}.pt')
    pathlib.Path(os.path.dirname(weights_path)).mkdir(exist_ok=True, parents=True)

    model = Model(cfg).to(cfg.device)
    raw_model = model
    if cfg.master_process:
        print(f'{model.count_params() / 1e6:.2f}M parameters')

    # --- 3. Load HF weights if specified ---
    if cfg.hf_model_name:
        missing, unexpected = load_from_hf(model, cfg.hf_model_name, cfg.device)
        if cfg.master_process:
            print(f'HF load: {len(missing)} missing, {len(unexpected)} unexpected')

    # --- 4. Train or generate ---
    if pathlib.Path(weights_path).exists() and not cfg.train:
        raw_model.load_state_dict(torch.load(weights_path, map_location=cfg.device, weights_only=True))
    elif cfg.train:
        optimizer = create_optimizer(raw_model, cfg)
        if cfg.master_process:
            kind = f'MuonClip ({[len(g["params"]) for g in optimizer.param_groups]} groups)' if cfg.use_muon else 'AdamW'
            print(f'Optimizer: {kind}')

        # Resume from checkpoint
        ckpt_dir = cfg.ckpt_dir or cfg.resume or os.path.join(script_dir, 'weights')
        start_step = load_checkpoint(cfg.resume, raw_model, optimizer, cfg.device) if cfg.resume else 0
        if start_step <= 0:
            start_step = 0
        elif cfg.master_process:
            print(f'Resumed from step {start_step}')

        # FP8 conversion (before compile)
        if cfg.use_fp8:
            _fp8_ok = lambda m, fqn: (isinstance(m, nn.Linear) and m.in_features % 16 == 0
                                      and m.out_features % 16 == 0 and fqn != 'lm_head')
            convert_to_float8_training(raw_model, module_filter_fn=_fp8_ok)
            if cfg.master_process:
                n8 = sum(1 for m in raw_model.modules() if isinstance(m, Float8Linear))
                print(f'FP8 training: {n8} layers converted')

        # Compile (after checkpoint loaded)
        if cfg.use_compile:
            if cfg.use_grad_ckpt or cfg.use_liger:
                for i, blk in enumerate(raw_model.blocks):
                    raw_model.blocks[i] = torch.compile(blk, dynamic=False)
            else:
                model = torch.compile(model, dynamic=False)

        if cfg.ddp:
            model = DDP(model, device_ids=[int(cfg.device.split(':')[-1])],
                        gradient_as_bucket_view=True, broadcast_buffers=False)
        effective_batch = cfg.batch_size * cfg.grad_accum_steps * cfg.ddp_world_size

        # Preload datasets on all ranks
        get_batch('train', cfg)
        if cfg.ddp:
            dist.barrier()
        if cfg.master_process:
            print('Datasets ready.' + (' Compile warm-up may take 5-15 min...' if cfg.use_compile else ''))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        if cfg.debug and cfg.ddp:
            _ddp_verify(raw_model, -1, 0.0, 0.0, cfg)

        # Trackio experiment tracking
        if cfg.master_process and trackio_ok:
            tkw = {'project': 'smoldlm-phase6', 'config': {
                'arch': f'{cfg.n_layer}L/{cfg.n_embd}d/{cfg.n_head}h', 'seq_len': cfg.seq_len,
                'batch_size': effective_batch, 'max_iters': cfg.max_iters,
                'hf_model': cfg.hf_model_name, 'ddp': cfg.ddp_world_size,
            }}
            if cfg.trackio_space:
                tkw['space_id'] = cfg.trackio_space
            try:
                trackio.init(**tkw)
            except Exception as e:
                print(f'WARNING: trackio init failed ({e})'); trackio_ok = False

        # --- Training loop ---
        losses = {}
        total_training_time = 0.0
        _WARMUP_STEPS = 10

        for step in range(start_step, cfg.max_iters):
            # WSD LR schedule
            _apply_lr_schedule(optimizer, step, cfg)

            # Periodic evaluation
            if step % cfg.eval_every == 0 or step == cfg.max_iters - 1:
                startup_eval_iters = max(1, min(8, cfg.eval_iters)) if step == 0 else cfg.eval_iters
                losses = estimate_loss(
                    raw_model,
                    lambda split, c: get_batch(split, c),
                    cfg,
                    eval_iters=startup_eval_iters,
                    reset_val_fn=lambda: reset_val_loader(cfg),
                )

                # Step-0 sanity check
                if step == 0:
                    _step0_sanity(losses.get('train', 0), cfg)

                if cfg.master_process:
                    lr = optimizer.param_groups[0]['lr']
                    print(f'step {step:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f} | lr {lr:.6f}')
                    if trackio_ok:
                        trackio.log({
                            'eval_train_loss': losses['train'],
                            'eval_val_loss': losses['val'],
                            'eval_lr': lr,
                        })
                    if step > 0:
                        sample = _generate_sample(raw_model, cfg)
                        if sample:
                            print(f'--- sample ---\n{sample[:300]}\n--- end sample ---')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Checkpoint (decoupled from eval)
            if step > 0 and step % cfg.ckpt_interval == 0 and cfg.master_process:
                ckpt_path = save_checkpoint(
                    raw_model, optimizer, step,
                    losses.get('val', 0.0) if step % cfg.eval_every == 0 else 0.0,
                    ckpt_dir,
                )
                print(f'Saved checkpoint: {ckpt_path}')

            # Per-step timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_t0 = time.time()

            # Gradient accumulation with DDP no_sync
            accum_loss = 0.0
            for micro_step in range(cfg.grad_accum_steps):
                no_sync = cfg.ddp and micro_step < cfg.grad_accum_steps - 1
                ctx = model.no_sync() if no_sync else contextlib.nullcontext()
                with ctx:
                    x_input, targets, mask, elbo_w, doc_ids, positions = get_batch('train', cfg)
                    attn_mask = _build_attn_mask(doc_ids, cfg)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_amp):
                        hidden, _ = model(x_input, targets, attn_mask=attn_mask, positions=positions)
                        loss = compute_loss(hidden, targets, mask, elbo_w, raw_model.lm_head.weight, cfg)
                        loss = loss / cfg.grad_accum_steps
                    accum_loss += loss.detach().item()
                    if cfg.debug and step == start_step and cfg.master_process:
                        print(
                            f'  [PRE-BW] loss={loss.item():.4f} requires_grad={loss.requires_grad} '
                            f'grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}'
                        )
                    loss.backward()

            # Grad clip + optimizer step
            if cfg.debug and step == start_step and cfg.master_process:
                _debug_grad_check(raw_model, loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gn = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            if cfg.debug and step == start_step and cfg.master_process:
                print(f'  [GRAD] clip_grad_norm_ returned: {gn:.6e}')
            if cfg.debug and step - start_step < 5:
                _ddp_verify(raw_model, step, accum_loss, gn, cfg)

            # Per-step timing (exclude compilation warm-up)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_dt = time.time() - step_t0
            ss = step - start_step
            if ss >= _WARMUP_STEPS:
                total_training_time += step_dt
            tps = int(effective_batch * cfg.seq_len / max(step_dt, 1e-9))
            avg_tps = (int(((ss - _WARMUP_STEPS + 1) * effective_batch * cfg.seq_len) / max(total_training_time, 1e-9))
                       if total_training_time > 0 else 0)

            # GC management
            if ss == 0:
                gc.collect(); gc.freeze(); gc.disable()
            elif ss % 5000 == 0:
                gc.collect()

            # Progress logging every 10 steps
            if cfg.master_process and step % 10 == 0 and step > 0:
                vram = f' | VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB' if torch.cuda.is_available() else ''
                print(f'  step {step:5d} | loss {accum_loss:.4f} | gn {gn:.2f} | {tps:,} tok/s (avg {avg_tps:,}){vram}')
                if trackio_ok:
                    trackio.log({'train_loss': accum_loss, 'train_lr': optimizer.param_groups[0]['lr'],
                                 'train_grad_norm': gn, 'train_tokens_per_sec': tps})

            # First step VRAM report
            if step == start_step and cfg.master_process and torch.cuda.is_available():
                pk = torch.cuda.max_memory_allocated() / 1e9
                tot = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f'VRAM step 0: {pk:.2f}/{tot:.1f} GB ({pk/tot*100:.0f}%)')

        # Save final checkpoint + weights
        if cfg.master_process:
            save_checkpoint(raw_model, optimizer, cfg.max_iters, 0.0, ckpt_dir)
            torch.save(raw_model.state_dict(), weights_path)
            print(f'Saved weights to {weights_path}')
        if cfg.master_process and trackio_ok:
            trackio.finish()

    elif cfg.prompt is not None and not pathlib.Path(weights_path).exists() and not cfg.hf_model_name:
        print(f'No weights found at {weights_path}')
        print('Run with --train or --hf-model-name to load weights.')
        sys.exit(1)

    # --- 6. Generate text ---
    if cfg.master_process and cfg.prompt is not None:
        sample = _generate_sample(raw_model, cfg, prompt=cfg.prompt, max_new_tokens=cfg.max_tokens)
        if sample:
            print(sample)

    if cfg.ddp:
        dist.destroy_process_group()
