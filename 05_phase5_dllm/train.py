"""
Phase 5 training script — orchestrates all modules for block diffusion LM training.

Usage:
    uv run python 05_phase5_dllm/train.py --train
    uv run python 05_phase5_dllm/train.py --prompt "Once upon a time"
    torchrun --nproc_per_node=2 05_phase5_dllm/train.py --train

Architecture: 30L/576d/9h/3kv/1536MLP, ~144M params, linear noise schedule,
MuonClip optimizer, document packing, Gated Query Attention.
"""

import os
import sys
import time
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Phase 5 modules
from phase5 import config
from phase5.model import Model
from phase5.attention import build_staircase_block_mask, build_staircase_mask
from phase5.schedule import get_lr_factor
from phase5.optim import build_param_groups, build_adamw_optimizer
from phase5.checkpoint import save_checkpoint, load_checkpoint
from phase5.generate import generate
from phase5.tokenizer import encode, decode

try:
    import trackio
    _TRACKIO_AVAILABLE = config._TRACKIO_AVAILABLE
except ImportError:
    _TRACKIO_AVAILABLE = False


def _build_attn_mask(doc_ids):
    """Build staircase attention mask, with doc boundary constraint if packing."""
    if config.use_flex and config._FLEX_AVAILABLE:
        return build_staircase_block_mask(config.seq_len, config.block_size, doc_ids=doc_ids)
    else:
        # Fallback: float mask without doc boundary constraint.
        # Doc packing requires FlexAttention for correct cross-doc masking.
        # This path is only used on CPU/MPS where FlexAttention is unavailable.
        return build_staircase_mask(config.seq_len, config.block_size).to(config.device)


@torch.no_grad()
def estimate_loss(model, get_batch_fn):
    model_was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x_input, targets, mask, elbo_w, doc_ids, positions = get_batch_fn(split)
            attn_mask = _build_attn_mask(doc_ids)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_amp):
                _, loss = model(x_input, targets, mask, elbo_w,
                                attn_mask=attn_mask, positions=positions)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if model_was_training:
        model.train()
    return out


if __name__ == "__main__":
    # Lazy import data module (triggers dataset connection)
    from phase5.data import get_batch

    # --- 1. Print configuration ---
    if config.master_process:
        print("=" * 60)
        print("phase5_dllm — modern block diffusion LM (Phase 5)")
        print("=" * 60)
        print(f"  n_layer        = {config.n_layer}")
        print(f"  n_embd         = {config.n_embd}")
        print(f"  n_head         = {config.n_head}")
        print(f"  n_kv_head      = {config.n_kv_head} (GQA {config.n_head//config.n_kv_head}:1)")
        print(f"  mlp_hidden     = {config.mlp_hidden}")
        print(f"  seq_len        = {config.seq_len}")
        print(f"  block_size     = {config.block_size}")
        print(f"  num_blocks     = {config.num_blocks}")
        eff = config.batch_size * config.grad_accum_steps * config.ddp_world_size
        print(f"  batch_size     = {config.batch_size} (effective: {eff})")
        print(f"  vocab_size     = {config.vocab_size}")
        print(f"  max_iters      = {config.max_iters}")
        print(f"  warmup         = {config.warmup_iters}, decay_start = {config.decay_start}")
        print(f"  lr             = Muon 0.02 / AdamW 6e-4")
        print(f"  dropout        = {config.dropout}")
        print(f"  device         = {config.device}")
        print(f"  ddp            = {config.ddp} (world_size={config.ddp_world_size})")
        print(f"  use_amp        = {config.use_amp}")
        print(f"  use_liger      = {config.use_liger}")
        print(f"  use_flex       = {config.use_flex}")
        print(f"  use_cart       = {config.use_cart}")
        print(f"  use_compile    = {config.use_compile}")
        print(f"  use_muon       = {config.use_muon}")
        print(f"  noise_schedule = linear (ELBO weight = 1/t)")
        print(f"  doc_packing    = True")
        print(f"  gated_query    = True (arXiv:2505.06708)")
        print("=" * 60)

    # --- 2. Weights path ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", f"phase5_dllm_b{config.block_size}.pt")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # --- 3. Instantiate model ---
    model = Model().to(config.device)
    if config.master_process:
        print(f"{model.count_params() / 1e6:.2f}M parameters")
        if torch.cuda.is_available():
            print(f"Peak VRAM after model init: "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if config.use_compile:
        model = torch.compile(model, dynamic=False)
        if config.master_process:
            print("torch.compile enabled (default mode, dynamic=False)")

    if config.ddp:
        model = DDP(model, device_ids=[int(config.device.split(":")[-1])])

    raw_model = model.module if config.ddp else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model

    # --- 4. Load or train ---
    if os.path.exists(weights_path) and not config.args.train:
        raw_model.load_state_dict(
            torch.load(weights_path, map_location=config.device, weights_only=True)
        )
        if config.master_process:
            print(f"Loaded weights from {weights_path}")

    elif config.args.train:
        # Build optimizer
        if config.use_muon:
            optimizer = build_param_groups(raw_model)
            if config.master_process:
                counts = [len(g["params"]) for g in optimizer.param_groups]
                print(f"MuonClip optimizer: {counts} params per group "
                      f"(QK Muon / Other Muon / AdamW)")
        else:
            optimizer = build_adamw_optimizer(raw_model)
            if config.master_process:
                print("AdamW optimizer (--no-muon)")

        # Resume from checkpoint
        ckpt_dir = (config.args.ckpt_dir or config.args.resume
                    or os.path.join(script_dir, "weights"))
        start_step = 0
        if config.args.resume:
            start_step = load_checkpoint(config.args.resume, raw_model,
                                         optimizer, config.device)
            if start_step <= 0:
                start_step = 0
                if config.master_process:
                    print(f"No checkpoint at {config.args.resume}, starting fresh")
            elif config.master_process:
                print(f"Resumed from step {start_step}")

        effective_batch = (config.batch_size * config.grad_accum_steps
                           * config.ddp_world_size)

        # Preload datasets on all ranks (avoids DDP timeout)
        if config.master_process:
            print("Preloading datasets...")
        get_batch("train")
        get_batch("val")
        if config.ddp:
            dist.barrier()
        if config.master_process:
            print("Datasets ready.")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        # Trackio experiment tracking
        if config.master_process and _TRACKIO_AVAILABLE:
            trackio_cfg = {
                "n_layer": config.n_layer, "n_embd": config.n_embd,
                "n_head": config.n_head, "n_kv_head": config.n_kv_head,
                "mlp_hidden": config.mlp_hidden,
                "seq_len": config.seq_len, "block_size": config.block_size,
                "batch_size": effective_batch, "max_iters": config.max_iters,
                "use_muon": config.use_muon, "use_cart": config.use_cart,
                "ddp_world_size": config.ddp_world_size,
            }
            trackio_kw = {"project": "open-dllm-phase5", "config": trackio_cfg}
            if config.args.trackio_space:
                trackio_kw["space_id"] = config.args.trackio_space
            try:
                trackio.init(**trackio_kw)
            except Exception as e:
                print(f"WARNING: trackio init failed ({e}), continuing without tracking")
                _TRACKIO_AVAILABLE = False

        # --- Training loop ---
        losses = {}
        t0 = time.time()
        for step in range(start_step, config.max_iters):
            # WSD LR schedule
            lr_factor = get_lr_factor(step)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * lr_factor

            # Periodic evaluation
            if step % config.eval_interval == 0 or step == config.max_iters - 1:
                losses = estimate_loss(raw_model, get_batch)
                if config.master_process:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"step {step:5d} | train loss {losses['train']:.4f} | "
                          f"val loss {losses['val']:.4f} | lr {lr:.6f}")
                    if _TRACKIO_AVAILABLE:
                        trackio.log({
                            "eval_train_loss": losses["train"],
                            "eval_val_loss": losses["val"],
                            "eval_lr": lr, "step": step,
                        })
                    if step > 0:
                        sample = generate(raw_model, encode, decode,
                                          max_new_tokens=64, temperature=0.8, top_k=5)
                        print(f"--- sample ---\n{sample[:300]}\n--- end sample ---")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Checkpoint (decoupled from eval for preemption resilience)
            if (step > 0 and step % config.args.ckpt_interval == 0
                    and config.master_process):
                ckpt_path = save_checkpoint(
                    raw_model, optimizer, step,
                    losses.get("val", 0.0) if step % config.eval_interval == 0 else 0.0,
                    ckpt_dir,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            # Gradient accumulation with DDP no_sync
            for micro_step in range(config.grad_accum_steps):
                no_sync = config.ddp and micro_step < config.grad_accum_steps - 1
                ctx = model.no_sync() if no_sync else contextlib.nullcontext()
                with ctx:
                    x_input, targets, mask, elbo_w, doc_ids, positions = get_batch("train")
                    attn_mask = _build_attn_mask(doc_ids)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=config.use_amp):
                        _, loss = model(x_input, targets, mask, elbo_w,
                                        attn_mask=attn_mask, positions=positions)
                        loss = loss / config.grad_accum_steps
                    loss.backward()

            # Grad clip + optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Progress logging every 10 steps
            if config.master_process and step % 10 == 0 and step > 0:
                dt = time.time() - t0
                tps = ((step - start_step) * effective_batch * config.seq_len) / dt
                vram = ""
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    alloc = torch.cuda.memory_allocated() / 1e9
                    vram = f" | VRAM {alloc:.1f}/{peak:.1f} GB"
                gn = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                print(f"  step {step:5d} | loss {loss.item() * config.grad_accum_steps:.4f} "
                      f"| grad_norm {gn:.2f} | {tps:.0f} tok/s{vram}")
                if _TRACKIO_AVAILABLE:
                    trackio.log({
                        "train_loss": loss.item() * config.grad_accum_steps,
                        "train_lr": optimizer.param_groups[0]["lr"],
                        "train_grad_norm": gn,
                        "train_tokens_per_sec": tps, "step": step,
                    })

            # First step VRAM report
            if (step == start_step and config.master_process
                    and torch.cuda.is_available()):
                peak = torch.cuda.max_memory_allocated() / 1e9
                alloc = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"VRAM after step 0: {alloc:.2f} GB alloc, "
                      f"{peak:.2f} GB peak, {total:.1f} GB total "
                      f"({peak/total*100:.0f}% used)")

        # Save final checkpoint + weights
        if config.master_process:
            save_checkpoint(raw_model, optimizer, config.max_iters, 0.0, ckpt_dir)
            torch.save(raw_model.state_dict(), weights_path)
            print(f"Saved weights to {weights_path}")
        if config.master_process and _TRACKIO_AVAILABLE:
            trackio.finish()

    elif config.args.prompt is not None:
        print(f"No weights found at {weights_path}")
        print("Run with --train to train from scratch.")
        sys.exit(1)

    # --- 5. Generate text ---
    if config.master_process and config.args.prompt is not None:
        sample = generate(raw_model, encode, decode,
                          prompt=config.args.prompt,
                          max_new_tokens=config.args.max_tokens,
                          denoise_steps=config.args.denoise_steps,
                          temperature=0.8, top_k=5)
        print(sample)

    if config.ddp:
        dist.destroy_process_group()
