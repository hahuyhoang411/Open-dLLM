"""All-in-one Modal GPU test for Phase 6.

Tests:
  1. Liger + torch.compile compatibility (full-model vs per-block)
  2. Qwen3-0.6B weight loading + tied embedding verification
  3. AR greedy decode (untouched weights — proves model knows language)
  4. 5-step diffusion training (ELBO loss, MuonClip, bf16)
  5. Block-diffusion generate via our generate.py pipeline

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_gpu_test.py
"""

import modal

app = modal.App('smoldlm-phase6-gpu-test')

image = (
    modal.Image
    .debian_slim(python_version='3.11')
    .pip_install(
        'torch>=2.5.0',
        'transformers>=4.45.0',
        'datasets>=2.0.0',
        'liger-kernel>=0.5.0',
        'safetensors>=0.4.0',
        'huggingface_hub>=0.20.0',
        'numpy',
    )
    .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


def _section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def _qwen3_config(**overrides):
    from phase6.config import Config
    defaults = dict(
        n_layer=28, n_embd=1024, n_head=16, n_kv_head=8, head_dim=128,
        mlp_hidden=3072, vocab_size=151936, seq_len=512, block_size=8,
        rope_base=1_000_000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=True, use_grad_ckpt=False, use_flex=False,
        use_muon=True, use_fp8=False,
        pad_token_id=151643, mask_token_id=151669, eos_token_id=151645,
        batch_size=4, max_iters=100, muon_lr=0.02, adamw_lr=3e-3,
        grad_clip=1.0, t_min=0.1, denoise_steps=4,
    )
    defaults.update(overrides)
    return Config(**defaults).validate()


@app.function(image=image, gpu='H100', timeout=900)
def run_all():
    import sys
    import time
    sys.path.insert(0, '/root/06_qwen3_dllm')
    sys.path.insert(0, '/root')

    import torch
    torch.set_float32_matmul_precision('high')

    from phase6.config import Config
    from phase6.model import Model
    from phase6.attention import build_staircase_mask
    from phase6.checkpoint import load_from_hf
    from phase6.loss import compute_loss
    from phase6.optim import create_optimizer
    from phase6.schedule import compute_elbo_weight, get_lr_factor, apply_noise
    from phase6.data import _sample_t_per_doc

    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', 0)
    print(f"PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)} | {total_mem/1e9:.0f} GB")

    results = {}

    # ==================================================================
    # 1. Liger + compile
    # ==================================================================
    _section("1. LIGER + TORCH.COMPILE")

    tiny = Config(
        n_layer=4, n_embd=128, n_head=4, n_kv_head=2, head_dim=64,
        mlp_hidden=256, vocab_size=512, seq_len=64, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=True, use_grad_ckpt=False, use_flex=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
    ).validate()

    B, L = 2, tiny.seq_len
    targets = torch.randint(3, tiny.vocab_size, (B, L), device='cuda')
    x_noisy = targets.clone()
    x_noisy[torch.rand(B, L, device='cuda') < 0.5] = tiny.mask_token_id
    x_input = torch.cat([x_noisy, targets], dim=1)
    mask = build_staircase_mask(L, tiny.block_size).to('cuda')

    for label, liger_on in [("Liger ON", True), ("Liger OFF", False)]:
        cfg_i = Config(
            n_layer=4, n_embd=128, n_head=4, n_kv_head=2, head_dim=64,
            mlp_hidden=256, vocab_size=512, seq_len=64, block_size=8,
            rope_base=10000, rms_eps=1e-6, dropout=0.0,
            use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
            use_liger=liger_on, use_grad_ckpt=False, use_flex=False,
            pad_token_id=0, mask_token_id=1, eos_token_id=2,
        ).validate()
        try:
            m = Model(cfg_i).cuda().train()
            cm = torch.compile(m, dynamic=False)
            h, _ = cm(x_input, targets=targets, attn_mask=mask)
            (h**2).mean().backward()
            gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
            results[f'compile_{label}'] = 'PASS'
            print(f"  Full-model compile ({label}): PASS (grad_norm={gn:.4f})")
        except Exception as e:
            results[f'compile_{label}'] = f'FAIL: {e}'
            print(f"  Full-model compile ({label}): FAIL ({e})")
        del m, cm
    del x_input, mask, targets, x_noisy
    torch.cuda.empty_cache()

    # ==================================================================
    # 2. Load Qwen3-0.6B
    # ==================================================================
    _section("2. QWEN3-0.6B WEIGHT LOADING")

    qcfg = _qwen3_config()
    model = Model(qcfg).cuda().bfloat16()
    print(f"  Params: {model.count_params()/1e6:.1f}M")

    t0 = time.time()
    missing, unexpected = load_from_hf(model, 'Qwen/Qwen3-0.6B', device='cuda')
    print(f"  Loaded in {time.time()-t0:.1f}s | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    results['load_missing'] = len(missing)
    results['load_unexpected'] = len(unexpected)

    tied = model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print(f"  Tied embeddings: {tied}")
    results['tied_embeddings'] = tied

    # ==================================================================
    # 3. AR inference (untouched weights — should produce real language)
    # ==================================================================
    _section("3. AR INFERENCE (untouched Qwen3 weights)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    model.train(False)

    for prompt in [
        "The capital of Vietnam is",
        "The capital of France is",
        "1 + 1 =",
        "Python is a programming language that",
    ]:
        ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        gen = ids.clone()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(40):
                logits, _ = model(gen)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
                if nxt.item() == qcfg.eos_token_id:
                    break
        text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
        print(f"  '{prompt}'\n    -> {text[:150]}")

    vn_ids = tokenizer.encode("The capital of Vietnam is", return_tensors="pt").cuda()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        vn_logits, _ = model(vn_ids)
    top5 = torch.topk(vn_logits[0, -1].float(), 5)
    top5_words = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    print(f"\n  Top-5 'The capital of Vietnam is': {top5_words}")
    results['ar_top5_vietnam'] = top5_words

    # ==================================================================
    # 4. 5-step diffusion training
    # ==================================================================
    _section("4. 5-STEP DIFFUSION TRAINING")

    model.train()
    compiled = torch.compile(model, dynamic=False)
    optimizer = create_optimizer(model, qcfg)

    B_t, L_t = qcfg.batch_size, qcfg.seq_len
    losses, gnorms = [], []

    for step in range(5):
        t0 = time.time()
        tgt = torch.randint(3, qcfg.vocab_size, (B_t, L_t), device='cuda')
        doc_ids = torch.zeros(B_t, L_t, dtype=torch.long)
        t = _sample_t_per_doc(doc_ids, t_min=qcfg.t_min).to('cuda')
        xn, nmask = apply_noise(tgt, t, mask_token_id=qcfg.mask_token_id, block_size=qcfg.block_size)
        ew = compute_elbo_weight(t, t_min=qcfg.t_min).to('cuda')
        xi = torch.cat([xn, tgt], dim=1)
        am = build_staircase_mask(L_t, qcfg.block_size).to('cuda')

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h, _ = compiled(xi, targets=tgt, attn_mask=am)
            loss = compute_loss(h, tgt, nmask.to('cuda'), ew, model.lm_head.weight, qcfg)

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), qcfg.grad_clip).item()
        lr_f = get_lr_factor(step, qcfg.warmup_iters, qcfg.decay_start, qcfg.max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * lr_f
        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        losses.append(loss.item())
        gnorms.append(gn)
        tps = B_t * L_t / dt if dt > 0.01 else 0
        print(f"  Step {step}: loss={loss.item():.4f}  grad_norm={gn:.2f}  lr={lr_f:.4f}  {dt:.2f}s  {tps:.0f} tok/s")

    results['train_losses'] = [round(l, 4) for l in losses]
    results['train_finite'] = all(l == l for l in losses)

    # ==================================================================
    # 5. Block-diffusion generate (our generate.py)
    # ==================================================================
    _section("5. BLOCK-DIFFUSION GENERATE")

    model.train(False)
    try:
        from phase6.generate import generate

        for prompt in ["The capital of Vietnam is", "Python is"]:
            prompt_ids = tokenizer.encode(prompt)
            tok_ids = generate(
                model, prompt_ids, qcfg,
                max_new_tokens=24, temperature=0.8, denoise_steps=4,
            )
            text = tokenizer.decode(tok_ids, skip_special_tokens=True)
            print(f"  '{prompt}'\n    -> {text[:150]}")
        results['diffusion_generate'] = 'PASS'
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results['diffusion_generate'] = f'FAIL: {e}'

    # ==================================================================
    # Summary
    # ==================================================================
    _section("SUMMARY")
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM peak: {vram:.2f} GB")
    results['vram_gb'] = round(vram, 2)
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results
