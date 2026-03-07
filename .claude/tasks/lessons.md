# Lessons Learned

## KV Cache Stale State (Phase 3)
- `generate()` resets KV cache at entry but not at exit
- When called mid-training (eval sample), stale batch=1 cache collides with training batch=32
- Fix: `model.reset_kv_cache()` after generation, before returning
- Pattern: any function that modifies shared model state (cache, mode) must clean up on exit

## Special Token ID Assignment (HuggingFace tokenizers)
- `BpeTrainer(special_tokens=[...])` assigns IDs in list order BEFORE training
- IDs are stable: [MASK]=0, <|endoftext|>=1, <|padding|>=2
- Retraining with more special tokens shifts BPE merge IDs (from 1→ to 3→), so old weights are incompatible

## Liger Kernel API (Phase 4)
- LigerFusedLinearCrossEntropyFunction.apply arg order: _input, weight, target, bias, ce_weight, ignore_index, lse_square_scale, label_smoothing, reduction
- **Return value is a 4-TUPLE**: `(loss, z_loss, token_accuracy, predicted_tokens)` — NOT a single tensor!
- Must unpack: `per_token_loss, _, _, _ = LigerFusedLinearCrossEntropyFunction.apply(...)`
- ce_weight must be None (not 0.0) — it expects a Tensor or None, not a float
- lse_square_scale=0.0 to disable z-loss (not -1; negative values corrupt gradients catastrophically)
- ignore_index=-100 is the standard PyTorch default (not -1)
- Always verify arg order against Liger source: github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

## Gradient Checkpointing Independence
- grad_checkpoint should be gated on `self.training` only, NOT on `use_amp`
- AMP and gradient checkpointing are orthogonal memory optimizations
- Tying them together silently disables checkpointing when AMP is off, causing OOM

## LigerRopeFunction Shape Mismatch (Phase 4)
- LigerRopeFunction.apply(q, k, cos, sin) expects q/k as (B, H, T, D) and cos/sin as (1, T, D)
- Our code has q/k as (B, T, H, D) at the RoPE application point, and cos/sin as (1, T, 1, D/2)
- Shape incompatibility makes integration non-trivial — extra transposes negate the speedup
- Decided to keep manual Python RoPE. RoPE is <5% of compute; the wins are in RMSNorm/SwiGLU/FusedLinearCE
- Lesson: always verify kernel API shapes against your data layout before integrating

## Kaggle Kernel Gotchas
- kernel-metadata.json `id` and `title` must match the slug exactly
- `code_file` must point to the right notebook filename
- Kernel outputs (logs, files) only available after COMPLETE status
- P100 has 16GB VRAM — depth=6 block_dllm (~36M params) fits fine
- `"enable_gpu": true` defaults to P100, NOT T4. Must set `"machine_shape": "NvidiaTeslaT4"` for T4
- There is NO single-T4 option on Kaggle — you get 2xT4. Use torchrun for DDP.
- Kaggle Docker may pre-install conflicting packages (e.g., bioinformatics `muon`). Defensive `pip uninstall` before install.

## PyPI Package Name Conflicts
- `muon` on PyPI = bioinformatics multi-omics package (imports scanpy)
- KellerJordan's Muon ML optimizer: `pip install git+https://github.com/KellerJordan/Muon`
- Package metadata name = `muon-optimizer`, import name = `muon` — same import conflicts with PyPI `muon`
- Always check package source: import a distinctive class and verify it exists

## Muon Optimizer API (KellerJordan)
- `MuonWithAuxAdam(param_groups)` — single list, each group has `use_muon` bool flag
- Muon groups: EXACTLY {params, lr, momentum, weight_decay, use_muon} — no extra keys
- Adam groups: EXACTLY {params, lr, betas, eps, weight_decay, use_muon} — no extra keys
- MuonWithAuxAdam.step() calls dist.all_gather — REQUIRES DDP. Use SingleDeviceMuonWithAuxAdam for single GPU
- Old kwargs API (muon_params=, adam_params=, adam_lr=) does NOT exist — was hallucinated in original code

## DDP Training Loop Patterns
- **Eval approach**: ALL ranks should run estimate_loss (uses raw_model, no DDP collectives). Only master prints. No barriers needed.
- **DO NOT use barriers around eval**: NCCL barriers have 600s timeout. If eval includes first-time lazy data loading (HF streaming takes minutes), the waiting rank times out. Barriers are for quick sync, not long waits.
- **Preload datasets before training loop**: Call `get_batch("train")` + `get_batch("val")` before the loop so lazy init happens in parallel on all ranks, then one barrier to sync.
- **Gradient accumulation no_sync**: DDP fires all_reduce on every backward(). During grad accum, use `model.no_sync()` on micro-steps 0..N-2, only sync on last micro-step. Without this: 2x NCCL traffic + incorrect gradient magnitudes.
- Pattern: `ctx = model.no_sync() if ddp and micro_step < grad_accum_steps - 1 else contextlib.nullcontext()`

## Liger Kernel Version Differences
- PyPI Liger (Kaggle): LigerFusedLinearCrossEntropyFunction.apply returns 3-tuple
- GitHub latest Liger: returns 4-tuple (loss, z_loss, token_accuracy, predicted_tokens)
- Version-agnostic: `liger_out = ...; loss = liger_out[0] if isinstance(liger_out, tuple) else liger_out`
- Never hardcode tuple unpacking length — always use indexing

## torch.inference_mode() vs torch.no_grad() — NEVER use inference_mode with caching
- `torch.inference_mode()` taints ALL tensors created within it as "inference tensors" — they can NEVER be used in autograd
- `torch.no_grad()` just disables gradient tracking — tensors remain normal and compatible with autograd
- CRITICAL: if any cached/lazy-init tensor (e.g., `_cached_staircase_mask`) is first created inside inference_mode, it becomes permanently unusable in training
- Our bug: `@torch.inference_mode()` on `estimate_loss()` → first `get_batch()` call caches the staircase mask as an inference tensor → training gradient checkpointing crashes
- Rule: ALWAYS use `@torch.no_grad()` for eval/generate. Never use `torch.inference_mode()` in code that shares model state with training.
- The dLLM reference repo (ZHZisZZ/dllm) uses `@torch.no_grad()` everywhere, never inference_mode

## CART Weight Explosion (Phase 4)
- `_compute_cart_weights()` returned `1/cart_scores.clamp(min=1e-4)` giving max weight 10,000
- When context is sparse (high masking), this inflates loss ~10x
- Neither ZHZisZZ/dllm nor JinjieNi/MegaDLMs uses CART — it's from Dream 7B (7B-scale model)
- Fix: disable by default (`--cart` opt-in) + cap max weight to `1/t_min` when enabled

## DLM Loss Normalization (Phase 4)
- Must divide by ALL real tokens (`targets != pad_token_id`), not just masked tokens
- The 1/mask_prob importance weight already accounts for the masked fraction
- Dividing by N_masked double-counts: loss inflated by E[1/mask_prob] for the given schedule
- Ref: ZHZisZZ/dllm uses `maskable_mask.sum()` (all valid tokens)
- Ref: JinjieNi/MegaDLMs uses `loss_mask.sum()` (similar effect)
- Correct formula: `weighted_loss.sum() / real_count` where `real_count = (targets != pad_token_id).sum()`

## ELBO Weight Must Match Noise Schedule (Phase 4) — CRITICAL
- **The ELBO importance weight must be `1/mask_prob`, NOT `1/t`**
- For LINEAR schedule (LLaDA): mask_prob = t, so 1/t is correct
- For COSINE schedule (our code): mask_prob = sin²(tπ/2) ≠ t, so 1/t is WRONG
- With 1/t on cosine schedule, low-noise timesteps (t=0.05) contribute 0.12x while high-noise contribute 1.08x
- This 8x imbalance causes loss to plateau at ~4.0 after 1000 steps — model never learns fine-grained patterns
- With 1/mask_prob, every timestep contributes exactly 1.0 (importance sampling cancellation)
- Verified: `mask_prob * (1/mask_prob) = 1.0` at all t values
- Ref: LLaDA (arXiv:2502.09992) uses `CE / p_mask` where p_mask = t (their linear schedule)
- Rule: when borrowing loss formulas across papers, verify the weight matches YOUR noise schedule

## Gradient Clipping Starvation Pattern
- Inflated loss → inflated gradients → grad_clip fires every step → very slow convergence
- Symptom: training runs without error but loss barely decreases
- Diagnostic: check step-0 loss. Should be ~ln(vocab_size). If 3x+ higher, check normalization.
- Example: vocab=32768, expected step-0 = ln(32768) ≈ 10.4. We observed ~104 (10x from CART + 3.3x from norm bug)

## FlexAttention Internal torch.compile
- FlexAttention (flex_attention, create_block_mask) internally uses torch._dynamo/Inductor
- This is SEPARATE from user-facing `torch.compile(model)` — `--no-compile` doesn't disable it
- On T4 (CC 7.5): hits Triton shared memory limit (65536 bytes)
- Must use `--no-flex` flag on T4 to fall back to SDPA with manual mask

## QK-Clip vs FlexAttention (Phase 5)
- QK-Clip (MuonClip) relies on tracking max attention logit per forward pass
- Original ms-swift approach: monkey-patch F.scaled_dot_product_attention to record ||q||*||k||*scale
- FlexAttention uses compiled Triton kernels that bypass SDPA → monkey-patch is blind
- Fix: compute upper bound directly in attention forward (after QK-norm, before dispatch)
- With QK-norm (per-head RMSNorm), max logit ≈ sqrt(head_dim) ≈ 8, well below tau=100
- QK-Clip is a safety net that rarely fires when QK-norm is active

## Phase 5 Code Review Bugs Found
- Missing `@torch.no_grad()` on `estimate_loss` → VRAM leak building computation graphs during eval
- Uninitialized `losses` dict → NameError when resuming from checkpoint not aligned to eval_interval
- Fallback attention mask (non-FlexAttention) doesn't enforce doc boundaries — only affects CPU/MPS testing
- CART weight direction was inverted vs Dream reference — FIXED: now matches Dream (more context = higher weight)
- Missing `set_cache_mode()` in model.py → `disable_kv_cache()` destroyed cache during generation — FIXED
- Named checkpoint not atomic (no tmp+replace) — FIXED
- `model.train()` not restored on exception in generate.py — FIXED: moved to finally block
- Dead SDPA monkey-patch in `_MaxLogitsTracker.enable()` — FIXED: replaced with pass
