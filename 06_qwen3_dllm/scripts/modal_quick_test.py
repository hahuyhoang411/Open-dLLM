"""Quick Modal GPU test: Liger + compile compat + 5-step training.

Usage:
    uv run modal run --profile meddiesresearch 06_qwen3_dllm/scripts/modal_quick_test.py
"""

import modal

app = modal.App('smoldlm-phase6-test')

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


@app.function(image=image, gpu='H100', timeout=600)
def test_gpu():
    import sys
    sys.path.insert(0, '/root/06_qwen3_dllm')
    sys.path.insert(0, '/root')

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', 0)
    print(f"GPU memory: {total_mem / 1e9:.1f} GB")

    results = {}

    # ================================================================
    # Test 1: Liger + full-model compile
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Liger + torch.compile compatibility")
    print("=" * 60)

    from phase6.config import Config
    from phase6.model import Model
    from phase6.attention import build_staircase_mask

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
    noise_mask = torch.rand(B, L, device='cuda') < 0.5
    x_noisy = targets.clone()
    x_noisy[noise_mask] = tiny.mask_token_id
    x_input = torch.cat([x_noisy, targets], dim=1)
    mask = build_staircase_mask(L, tiny.block_size).to('cuda')

    # Test A: Full-model compile + Liger
    try:
        model_a = Model(tiny).cuda()
        model_a.train()
        compiled_a = torch.compile(model_a, dynamic=False)
        hidden, _ = compiled_a(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        loss.backward()
        gn = sum(p.grad.norm().item() for p in model_a.parameters() if p.grad is not None)
        results['liger_full_compile'] = f"PASS (grad_norm={gn:.4f})"
        print(f"  Full-model compile + Liger: PASS (grad_norm={gn:.4f})")
    except Exception as e:
        results['liger_full_compile'] = f"FAIL ({type(e).__name__}: {e})"
        print(f"  Full-model compile + Liger: FAIL ({type(e).__name__}: {e})")

    # Test B: Per-block compile + Liger
    try:
        model_b = Model(tiny).cuda()
        model_b.train()
        for block in model_b.blocks:
            block._forward = torch.compile(block._forward, dynamic=False)
        hidden, _ = model_b(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        loss.backward()
        gn = sum(p.grad.norm().item() for p in model_b.parameters() if p.grad is not None)
        results['liger_perblock_compile'] = f"PASS (grad_norm={gn:.4f})"
        print(f"  Per-block compile + Liger: PASS (grad_norm={gn:.4f})")
    except Exception as e:
        results['liger_perblock_compile'] = f"FAIL ({type(e).__name__}: {e})"
        print(f"  Per-block compile + Liger: FAIL ({type(e).__name__}: {e})")

    # Test C: No Liger, full-model compile
    tiny_nl = Config(
        n_layer=4, n_embd=128, n_head=4, n_kv_head=2, head_dim=64,
        mlp_hidden=256, vocab_size=512, seq_len=64, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
    ).validate()
    try:
        model_c = Model(tiny_nl).cuda()
        model_c.train()
        compiled_c = torch.compile(model_c, dynamic=False)
        hidden, _ = compiled_c(x_input, targets=targets, attn_mask=mask)
        loss = (hidden ** 2).mean()
        loss.backward()
        gn = sum(p.grad.norm().item() for p in model_c.parameters() if p.grad is not None)
        results['noliger_full_compile'] = f"PASS (grad_norm={gn:.4f})"
        print(f"  Full-model compile (no Liger): PASS (grad_norm={gn:.4f})")
    except Exception as e:
        results['noliger_full_compile'] = f"FAIL ({type(e).__name__}: {e})"
        print(f"  Full-model compile (no Liger): FAIL ({type(e).__name__}: {e})")

    torch.cuda.empty_cache()

    # ================================================================
    # Test 2: 5-step training with full loss pipeline
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: 5-step training (small model, full pipeline)")
    print("=" * 60)

    from phase6.loss import compute_loss
    from phase6.optim import create_optimizer
    from phase6.schedule import compute_elbo_weight, get_lr_factor, apply_noise
    from phase6.data import _sample_t_per_doc

    train_cfg = Config(
        n_layer=4, n_embd=256, n_head=8, n_kv_head=4, head_dim=64,
        mlp_hidden=512, vocab_size=512, seq_len=128, block_size=8,
        rope_base=10000, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=True, use_grad_ckpt=False, use_flex=False,
        use_muon=True, use_fp8=False,
        pad_token_id=0, mask_token_id=1, eos_token_id=2,
        batch_size=4, max_iters=100, muon_lr=0.02, adamw_lr=3e-3,
        grad_clip=1.0, t_min=0.1,
    ).validate()

    model = Model(train_cfg).cuda()
    model.train()
    optimizer = create_optimizer(model, train_cfg)

    losses = []
    for step in range(5):
        B_sz, L_sz = train_cfg.batch_size, train_cfg.seq_len
        targets = torch.randint(3, train_cfg.vocab_size, (B_sz, L_sz), device='cuda')
        doc_ids = torch.zeros(B_sz, L_sz, dtype=torch.long)
        t = _sample_t_per_doc(doc_ids, t_min=train_cfg.t_min).to('cuda')
        x_noisy, nmask = apply_noise(targets, t, mask_token_id=train_cfg.mask_token_id,
                                      block_size=train_cfg.block_size)
        elbo_w = compute_elbo_weight(t, t_min=train_cfg.t_min).to('cuda')
        x_inp = torch.cat([x_noisy, targets], dim=1)
        attn_m = build_staircase_mask(L_sz, train_cfg.block_size).to('cuda')

        hidden, _ = model(x_inp, targets=targets, attn_mask=attn_m)
        loss = compute_loss(hidden, targets, nmask.to('cuda'), elbo_w,
                            model.lm_head.weight, train_cfg)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        lr_f = get_lr_factor(step, train_cfg.warmup_iters, train_cfg.decay_start, train_cfg.max_iters)
        losses.append(loss.item())
        print(f"  Step {step}: loss={loss.item():.4f}, grad_norm={gn:.4f}, lr_factor={lr_f:.4f}")

    results['training_losses'] = [f'{l:.2f}' for l in losses]
    results['loss_finite'] = all(not (l != l) for l in losses)  # NaN check
    print(f"\n  Losses: {' -> '.join(f'{l:.2f}' for l in losses)}")
    print(f"  All finite: {results['loss_finite']}")

    torch.cuda.empty_cache()

    # ================================================================
    # Test 3: Qwen3-0.6B weight loading on GPU
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Qwen3-0.6B weight loading + forward pass")
    print("=" * 60)

    from phase6.checkpoint import load_from_hf

    qwen_cfg = Config(
        n_layer=28, n_embd=1024, n_head=16, n_kv_head=8, head_dim=128,
        mlp_hidden=3072, vocab_size=151936, seq_len=2048, block_size=8,
        rope_base=1_000_000, rms_eps=1e-6,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        pad_token_id=151643, mask_token_id=151669, eos_token_id=151645,
    ).validate()

    qwen_model = Model(qwen_cfg).cuda().bfloat16()
    missing, unexpected = load_from_hf(qwen_model, 'Qwen/Qwen3-0.6B', device='cuda')
    non_rotary = [k for k in unexpected if 'rotary' not in k]
    print(f"  Missing keys: {len(missing)} — {missing[:5] if missing else '(none)'}")
    print(f"  Unexpected (non-rotary): {len(non_rotary)} — {non_rotary[:5]}")
    results['qwen3_missing'] = len(missing)
    results['qwen3_unexpected_real'] = len(non_rotary)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()

    qwen_model.train(False)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = qwen_model(input_ids)

    top5 = torch.topk(logits[0, -1].float(), 5)
    decoded = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    print(f"  Top-5 next tokens for 'The capital of France is': {decoded}")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    results['qwen3_top5'] = decoded
    results['qwen3_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results
