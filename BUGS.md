# Bug Museum: Every Bug We Found Building a Diffusion Language Model

*23 bugs across 5 phases. Each one is a lesson.*

This isn't a changelog. It's a teaching document. We built a diffusion language model from scratch through 5 progressively complex phases, and we documented every significant bug we encountered along the way. Some took minutes to find, some took days. The critical ones teach more than any tutorial.

Most projects hide their bugs. We celebrate them.

## How to use this document

- **Building a dLLM?** Read bugs #1, #6, #16 first. They will bite you.
- **Debugging loss issues?** Bugs #6, #12, #13, #16, #17 cover common loss pathologies.
- **Working with fused kernels?** Bug #16 (Liger FLCE) is essential reading.
- **Setting up Muon optimizer?** Bugs #7, #8, #9 cover the complete gotcha list.
- **Using torch.compile for training?** Bug #11 explains why `reduce-overhead` is wrong. Bug #20 explains why whole-model compile crashes with grad checkpoint.
- **Implementing sampling?** Bug #22 (Gumbel-Max) catches a subtle but destructive math error.

## Hall of Fame

Three bugs that teach the most, hurt the worst, and are the easiest to reproduce in your own projects:

| Rank | Bug | Phase | Why it matters |
|------|-----|-------|----------------|
| 1 | [#6: ELBO Weight Schedule Mismatch](#bug-6-elbo-weight-schedule-mismatch) | 4 | Copied a formula between papers without checking a hidden assumption. Loss converged to the wrong value. Days to diagnose. |
| 2 | [#16: Liger FLCE Backward Returns Zeros](#bug-16-liger-flce-backward-returns-zeros) | 5 | Correct loss, zero gradients. Model never learns but nothing looks wrong unless you check grad_norm. |
| 3 | [#1: Staircase Mask Label Leakage](#bug-1-staircase-mask-label-leakage) | 3 | Off-by-one in an attention mask let the model see its own answers. Training looked perfect. Generation was garbage. |

All three share a trait: **the training loop reported no errors and the loss looked reasonable.** Silent correctness bugs are the most expensive kind.

---

## Phase 1: Character-Level dLLM

No significant bugs. The model is simple enough (character-level vocabulary, single-file implementation, ~1M parameters) that most mistakes surface immediately as crashes or obviously wrong output. This is the benefit of starting small.

---

## Phase 3: Block Diffusion

### Bug #1: Staircase Mask Label Leakage
**Phase:** 3 | **Severity:** critical | **Time to find:** days

**Symptom:** Model learned nothing useful. Loss decreased but generation was garbage. Training curves looked perfect — fast convergence, low final loss. But every generated sequence was incoherent.

**Root Cause:** The staircase attention mask uses block indices to determine which blocks can attend to which. The comparison `block_idx >= query_block_idx` allowed noisy block N to attend to its own clean tokens in the concatenated `[x_t || x_0]` input. The model learned to copy the answer from the clean side instead of learning to predict from context.

**Fix:** Changed `>=` to `>` in the mask comparison — strict inequality prevents self-attention to own clean tokens.

**Lesson:** Off-by-one errors in attention masks are label leakage. The training loop gives no indication anything is wrong (loss goes down because the model memorizes the leak). Always verify your attention mask with a hand-worked example on a 2-3 block toy input. If training loss drops suspiciously fast, suspect the mask before anything else.

---

### Bug #2: Tied Embedding Init Order
**Phase:** 3 | **Severity:** high | **Time to find:** hours

**Symptom:** Training loss plateau. Model wasn't learning at expected rate.

**Root Cause:** `self.apply(_init_weights)` came AFTER `self.lm_head.weight = self.token_emb.weight`. The `apply()` walks all modules and reinitializes all `nn.Linear` weights — including `lm_head`, which overwrites the tied embedding with a fresh random tensor, breaking the tie.

**Fix:** Call `self.apply(_init_weights)` BEFORE setting the weight tie.

**Lesson:** Weight tying is pointer aliasing. `nn.Parameter` assignment creates a reference, but init functions that write `param.data = ...` or construct new tensors break that reference silently. Init order matters: initialize first, tie second.

---

### Bug #3: RoPE Buffer Too Short
**Phase:** 3 | **Severity:** medium | **Time to find:** minutes

**Symptom:** `IndexError` during generation with KV cache, never during training.

**Root Cause:** RoPE frequencies were precomputed for exactly `seq_len` positions. Training always uses exactly `seq_len`. But block diffusion concatenates `[x_noisy || x_clean]`, doubling the effective length. KV caching during generation extends it further.

**Fix:** Precompute for `max(2*seq_len, 4096)`.

**Lesson:** Always allocate positional encodings for the worst-case generation length, not just training length. It's a buffer. Over-allocate and let the indexing slice what it needs.

---

### Bug #4: KV Cache Stale State
**Phase:** 3 | **Severity:** high | **Time to find:** hours

**Symptom:** Training crashed at step 500 (first eval callback) with a shape mismatch error. The error pointed to an attention operation expecting batch=32 but receiving batch=1.

**Root Cause:** `generate()` reset the KV cache at entry (to batch=1 for generation) but did NOT reset it at exit. When training resumed with batch=32, the stale batch=1 cache caused a dimension mismatch in the next forward pass.

**Fix:** Reset KV cache on both entry AND exit of `generate()`. Better: use try/finally to guarantee cleanup.

**Lesson:** Any function that mutates model state must restore it on exit. Reset-on-entry is necessary but not sufficient — you also need reset-on-exit. This applies to KV caches, dropout modes, compilation states, and any other mutable model attribute. Especially dangerous when eval runs periodically during training, because the mismatch only surfaces when training resumes.

---

### Bug #5: Qwen 3 Tokenizer OOM
**Phase:** 3 | **Severity:** medium | **Time to find:** minutes

**Symptom:** P100 (16 GB) OOM immediately at model init. Training never started.

**Root Cause:** Used Qwen 3's 151K vocabulary tokenizer. Embedding layer: 151K x 384 x 4 bytes = 232 MB. With tied embeddings counted as output projection too, plus the rest of the model, it exceeded P100's memory.

**Fix:** Built a custom 32K BPE tokenizer. Embedding: 32K x 384 x 4 = 49 MB. 4.7x reduction.

**Lesson:** Vocab size has a direct, large impact on memory. Each vocab entry costs `2 * hidden_dim * bytes_per_param` (embedding + lm_head). On constrained hardware, vocab size is the first knob to check. Do the arithmetic before choosing a tokenizer.

---

## Phase 4: Modern dLLM

### Bug #6: ELBO Weight Schedule Mismatch
**Phase:** 4 | **Severity:** critical | **Time to find:** days

**Symptom:** Loss dropped from ~10.4 to ~4.0 in the first few hundred steps, then plateaued and refused to decrease further. The model was clearly learning something (loss decreased), but converged to the wrong value. No NaN, no spikes, no obvious anomaly — just a premature plateau.

**Root Cause:** The ELBO loss for diffusion language models requires weighting each timestep by `1/mask_prob` (the inverse of the masking probability at that timestep). The code copied `elbo_weight = 1/t` from the LLaDA paper. This is correct for LLaDA because LLaDA uses a LINEAR noise schedule where `mask_prob = t`.

Phase 4 uses a COSINE noise schedule where `mask_prob = 1 - cos^2(t * pi/2)`. At low t (e.g., t=0.1), cosine gives mask_prob = 0.024. The correct weight is 1/0.024 = 41.7. We used 1/0.1 = 10. Low-noise timesteps — the refinement steps where the model learns fine-grained token prediction — were under-weighted by 4-8x.

The model learned coarse structure (high-t, high mask_prob, weights roughly correct) but couldn't learn refinement (low-t, weights too small). Hence the plateau: good enough at filling in heavily masked text, unable to improve at lightly masked text.

**Fix:** Changed `elbo_weight = 1/t` to `elbo_weight = 1/mask_prob` where `mask_prob` is computed from the actual noise schedule. Phase 5 solved this permanently by switching to a linear schedule where `mask_prob = t`, making the simplification trivially correct by construction.

**Lesson:** Never copy loss formulas between papers without verifying the full derivation chain. The `1/t` simplification is schedule-dependent — it's a special case, not a general rule. Verify: `mask_prob(t) * weight(t)` should equal 1.0 at every timestep.

---

### Bug #7: Muon pip Package Name Collision
**Phase:** 4 | **Severity:** medium | **Time to find:** hours

**Symptom:** `import muon` imported successfully but gave classes related to bioinformatics (single-cell RNA analysis) instead of an optimizer.

**Root Cause:** `pip install muon` installs a bioinformatics package from the scanpy ecosystem. The ML Muon optimizer by Keller Jordan is only available via `pip install git+https://github.com/KellerJordan/Muon`. Same import name, completely different packages.

**Fix:** Use the git URL for installation. On Kaggle Docker (which pre-installs the bioinformatics muon), add a defensive `pip uninstall muon -y` first. Phase 5 eliminated this dependency entirely by implementing MuonClip from scratch.

**Lesson:** Always verify package identity after install. PyPI has no namespace protection — anyone can claim any name. Import a distinctive class and confirm it exists before proceeding.

---

### Bug #8: Muon DDP Requirement
**Phase:** 4 | **Severity:** high | **Time to find:** hours

**Symptom:** Crash on single-GPU with `MuonWithAuxAdam`. Stack trace pointed to `dist.all_gather` — no process group initialized.

**Root Cause:** Muon's Newton-Schulz orthogonalization uses `all_gather` to average across ranks. On a single GPU with no DDP, there's no process group, so the call fails.

**Fix:** Use `SingleDeviceMuonWithAuxAdam` for single GPU. Phase 5 solved this permanently with a self-contained MuonClip that handles both cases internally.

**Lesson:** Check if an optimizer requires a distributed setup before using it on a single GPU. The dependency isn't always documented. Read the optimizer's `step()` method.

---

### Bug #9: Muon Param Group Key Mismatch
**Phase:** 4 | **Severity:** high | **Time to find:** hours

**Symptom:** Cryptic `KeyError` deep inside Muon's `step()` method.

**Root Cause:** `MuonWithAuxAdam` requires param groups with EXACTLY these keys — no more, no fewer. Muon groups: `{params, lr, momentum, weight_decay, use_muon}`. Adam groups: `{params, lr, betas, eps, weight_decay, use_muon}`. An extra key like `name` (commonly added for logging) or a missing key like `momentum` crashes inside `step()`.

**Fix:** Match the exact key set for each group type. Strip any extra keys before passing to the optimizer.

**Lesson:** Third-party optimizers often have strict, undocumented param group schemas. PyTorch's built-in optimizers are lenient (they ignore extra keys). Custom optimizers are not. Read the source code, not just the README.

---

### Bug #10: Tied Embedding Optimizer Deduplication
**Phase:** 4 | **Severity:** medium | **Time to find:** hours

**Symptom:** Parameter count in optimizer was higher than expected. Some weights appeared to get double updates. Sometimes manifested as NaN after many steps.

**Root Cause:** `model.named_parameters()` yields the same tensor twice when embeddings are tied — once as `token_emb.weight` and once as `lm_head.weight`. Without deduplication, the tensor ends up in two optimizer groups. Both optimizers apply different updates to the same memory, causing conflicting gradient steps.

**Fix:** Deduplicate parameters using a `data_ptr()` set before building optimizer groups:
```python
seen = set()
for name, param in model.named_parameters():
    if param.data_ptr() in seen:
        continue
    seen.add(param.data_ptr())
    # ... assign to appropriate group
```

**Lesson:** Tied weights appear multiple times in parameter iterators. This is by design in PyTorch — `named_parameters()` walks the module tree and doesn't know about ties. Always deduplicate by `data_ptr()` when building custom optimizer param groups.

---

### Bug #11: torch.compile reduce-overhead VRAM Explosion
**Phase:** 4 | **Severity:** high | **Time to find:** hours

**Symptom:** OOM on batch sizes that should fit based on parameter count arithmetic. A 125M model on an 80 GB A100 OOM'd at batch=16, which should need roughly 40 GB.

**Root Cause:** `torch.compile(model, mode="reduce-overhead")` uses CUDA graphs, which capture and replay entire execution traces. CUDA graphs require all intermediate tensors to remain allocated for the entire graph lifetime. For a 20-layer transformer in training mode, this means all 20 layers' forward activations are pinned simultaneously — they can't be freed and reused as in normal eager execution.

**Fix:** Use `mode="default"` (Inductor fusion). Every major training codebase in 2026 (Karpathy's nanochat, Meta's torchtitan, modded-nanogpt) uses default mode for training.

**Lesson:** `reduce-overhead` = inference. `default` = training. This is a universal rule. The name "reduce-overhead" sounds universally good — it's not. It trades memory for kernel launch latency, which is only a bottleneck at small batch sizes during inference.

---

### Bug #12: DLM Loss Normalization
**Phase:** 4 | **Severity:** medium | **Time to find:** hours

**Symptom:** Step-0 loss was ~34, significantly higher than the expected `ln(32768) ~ 10.4`.

**Root Cause:** Dividing by the number of masked tokens only, instead of all real tokens. But the `1/mask_prob` ELBO weight already accounts for the masking fraction. Dividing by `N_masked` double-counts: `(1/mask_prob) * loss / (mask_prob * N_total)` gives `loss / (mask_prob^2 * N_total)`.

**Fix:** Normalize by ALL real tokens: `weighted_loss.sum() / (targets != pad_token_id).sum()`.

**Lesson:** ELBO weight and loss normalization interact. If `weight = 1/mask_prob`, normalize by total tokens. If you normalize by masked tokens only, you'd need `weight = 1` (standard MLM). Pick one convention and be consistent. The correct formula: `sum(per_token_loss * (1/mask_prob)) / N_total_real`.

---

### Bug #13: CART Weight Explosion
**Phase:** 4 | **Severity:** medium | **Time to find:** hours

**Symptom:** Loss spikes and eventual NaN with CART (Context-Adaptive Reweighting) enabled.

**Root Cause:** Two problems compounding. First, CART weights were computed as `1/score.clamp(min=1e-4)`, giving a max weight of 10,000. A few outlier tokens with low context scores dominated the entire loss.

Second, CART replaces the ELBO `1/t` weight entirely (it doesn't multiply with it). This breaks the variational bound. At `t=0.99`, CART weight is ~0.01 while ELBO weight is ~1.01 — a 100x gradient starvation at high noise. Since generation starts from ~100% masked, the model never learns to denoise from scratch.

Dream 7B uses CART for SFT on a pretrained Qwen2.5, not for from-scratch pretraining.

**Fix:** CART off by default (`--cart` opt-in, SFT only). When enabled, cap max weight at `1/t_min = 20`.

**Lesson:** Any data-dependent loss weighting needs capping. Unbounded weights are a NaN generator. Always check if a technique was designed for pretraining vs fine-tuning — the context matters more than the technique.

---

### Bug #14: Modal Image Missing Git
**Phase:** 4 | **Severity:** low | **Time to find:** minutes

**Symptom:** `pip install git+https://...` fails with "git: command not found" on Modal.

**Root Cause:** `debian_slim` base image ships without git.

**Fix:** `.apt_install("git")` before any `pip_install` that uses git URLs.

**Lesson:** Minimal container images are minimal. Don't assume standard tools are present. Check transitive dependencies in container environments.

---

### Bug #15: Modal with\_options API
**Phase:** 4 | **Severity:** low | **Time to find:** minutes

**Symptom:** `TypeError` when trying to override GPU config on `@app.function`.

**Root Cause:** `with_options(gpu=...)` only works on `@app.cls`, NOT `@app.function`. Different decorators, different runtime override APIs.

**Fix:** Wrap in `@app.cls` and use `Cls.with_options(gpu=gpu)().method.remote()`.

**Lesson:** Modal's `@app.cls` and `@app.function` have asymmetric APIs. When you need runtime configurability (GPU type, timeout), use `@app.cls`.

---

## Phase 5: Production dLLM

### Bug #16: Liger FLCE Backward Returns Zeros
**Phase:** 5 | **Severity:** critical | **Time to find:** hours

**Symptom:** `loss=19.11` at step 0 (correct -- matches expected ELBO-weighted init loss). `grad_norm=0.000000` at every single step. Model weights identical at step 0 and step 1000. Loss barely changes -- the only movement comes from weight decay.

**Root Cause:** Liger Kernel's `LigerFusedLinearCrossEntropyFunction` with `reduction='none'` has a broken backward pass ([Liger-Kernel #488](https://github.com/linkedin/Liger-Kernel/issues/488)). The forward pass produces correct per-token loss values. The backward pass returns all-zero gradients for every parameter.

The autograd graph IS connected (`loss.requires_grad=True`, `grad_fn=DivBackward0`), so standard checks for detached tensors pass. The bug is inside the custom CUDA kernel's backward implementation.

This is the most insidious type of bug: the loss value is correct, so any check of "is the loss reasonable?" passes. The model simply never updates. You can run for 10,000 steps and burn hours of GPU time before noticing that nothing changed.

**Fix:** Replaced with chunked cross-entropy using `torch.utils.checkpoint`. Process 4096 tokens per chunk (~1.5 GB peak) instead of materializing full logits (24 GB for 49K vocab x 2048 seq x float32). Liger's RMSNorm and SwiGLU kernels are unaffected -- only FLCE has the backward bug.

**Lesson:** Always verify that `backward()` produces nonzero gradients, not just that `forward()` gives a correct loss. The simplest diagnostic: print `grad_norm` at step 0. If it's exactly 0.0, the backward pass is broken. This should be a mandatory check in every training loop -- one line of logging that saves hours of GPU time.

---

### Bug #17: Step-0 Loss Panic (False Alarm)
**Phase:** 5 | **Severity:** medium | **Time to find:** hours

**Symptom:** Phase 5 step-0 loss was ~19 (ELBO-weighted). Phase 4 step-0 was ~10. The 2x difference triggered investigation.

**Root Cause:** Not a bug. With 49,152 vocab, ELBO weighting at `t ~ U[0.1, 1.0]`, and SmolLM2's wider init std (`1/sqrt(576)`), the expected ELBO-weighted loss is ~19-20. Raw CE is ~9.6, safely below `ln(49152) = 10.80`. The mask token produces higher CE at init because the wider init std spreads probability mass differently across the vocabulary.

**Fix:** None needed. Established Phase 5 threshold at 25 (anything above indicates a real problem).

**Lesson:** Step-0 loss depends on vocab size, ELBO weighting scheme, and weight initialization. Always derive the expected value for YOUR configuration before judging. `ln(vocab_size)` is the unweighted baseline; ELBO weighting multiplies it by `E[1/mask_prob]`. Panicking at the right loss wastes time. Ignoring the wrong loss wastes GPU hours.

---

### Bug #18: WSD Schedule Factor Bug
**Phase:** 5 | **Severity:** medium | **Time to find:** hours

**Symptom:** Learning rate jump at the transition from warmup to stable phase.

**Root Cause:** `get_lr_factor()` returns a `[0, 1]` multiplier. Each param group stores `initial_lr`. If `initial_lr` was set incorrectly (e.g., to the Muon group LR instead of the AdamW group LR), the multiplier amplifies the error at phase transitions.

**Fix:** Ensure `initial_lr` in each param group matches the intended peak LR for that group.

**Lesson:** LR schedulers that use multipliers compound with the base LR. Always debug by printing the actual LR (not just the factor) at each phase transition. A correct factor times a wrong base gives a wrong LR.

---

### Bug #19: torch.inference\_mode Poisons Cached Tensors
**Phase:** 5 | **Severity:** high | **Time to find:** hours

**Symptom:** Training crashed with a cryptic autograd error inside gradient checkpointing, but only when eval ran before the first training step.

**Root Cause:** `estimate_loss()` was decorated with `@torch.inference_mode()`. The first call to `get_batch()` inside eval lazily created and cached the staircase attention mask (`_cached_staircase_mask`). `inference_mode` taints ALL tensors created within its scope as "inference tensors" -- they can never participate in autograd, permanently.

When training resumed, gradient checkpointing tried to use the cached (tainted) mask during recomputation. Autograd rejected it.

`torch.no_grad()` doesn't have this problem. It disables gradient tracking but leaves tensors normal and autograd-compatible.

**Fix:** Replace `@torch.inference_mode()` with `@torch.no_grad()` everywhere that shares state with training.

**Lesson:** Never use `torch.inference_mode()` in code that shares cached tensors with training. The "inference" in the name is literal -- tensors created under it are permanently barred from autograd. Use `torch.no_grad()` for eval and generate. The dLLM reference repo (ZHZisZZ/dllm) uses `@torch.no_grad()` everywhere, never `inference_mode`.

---

### Bug #20: Whole-Model Compile + Grad Checkpoint Crash
**Phase:** 5 | **Severity:** high | **Time to find:** hours

**Symptom:** `cudaErrorIllegalAddress` (XID 31 MMU fault) during training. GPU hard-faults, no Python traceback.

**Root Cause:** `torch.compile(model)` + `gradient_checkpoint()` inside `Block.forward()` + FlexAttention = GPU crash. The compiler traces through checkpoint boundaries and encounters FlexAttention (a higher-order op) inside the same graph, producing invalid CUDA code.

Nanochat and modded-nanogpt use whole-model compile but do NOT use gradient checkpointing. The combination is the problem.

**Fix:** Per-block compile (torchtitan pattern). Compile each transformer block individually, so the checkpoint wrapper stays OUTSIDE the compiled boundary:
```python
for i, block in enumerate(model.blocks):
    model.blocks[i] = torch.compile(block, dynamic=False)
```

**Lesson:** `torch.compile` and `gradient_checkpoint` are both graph-transforming operations. Nesting them creates interactions that the compiler doesn't handle. If you use grad checkpoint, compile at the block level, not the model level.

---

### Bug #21: RMSNorm Epsilon Mismatch
**Phase:** 5 | **Severity:** medium | **Time to find:** hours

**Symptom:** Subtle numerical differences between runs with and without Liger kernels. Loss curves diverged slightly after ~500 steps.

**Root Cause:** Three different epsilon values in play:
- LigerRMSNorm default: `1e-6`
- `nn.RMSNorm` bfloat16 fallback: `7.8e-3` (derived from dtype)
- SmolLM2 spec: `1e-5`

Without explicitly setting `eps`, the value depended on which code path ran. A 10x difference in epsilon changes normalization behavior in low-variance regions, causing slow numerical drift.

**Fix:** Pass `eps=1e-5` explicitly to all norm layers, matching the SmolLM2 spec.

**Lesson:** Always pass `eps=` explicitly to normalization layers. The default varies by implementation (Liger, PyTorch, and even dtype-dependent fallbacks all differ). One parameter, three different defaults, silent divergence. Never rely on defaults for numerical stability constants.

---

### Bug #22: Gumbel-Max Sampling Math Error
**Phase:** 5 | **Severity:** high | **Time to find:** hours

**Symptom:** Generated text was incoherent garbage at all temperatures except 0 (greedy), where it worked fine.

**Root Cause:** `_add_gumbel_noise(logits, temperature)` computed `logits / gumbel_noise` instead of `logits.exp() / gumbel_noise`. Without `.exp()`, the function divides log-space values by noise -- there's no theoretical basis for this operation. It produces garbage token rankings.

The bug is invisible at `temperature=0` because the function short-circuits and returns logits unchanged. So greedy generation worked fine, masking the bug during basic testing.

**Fix:** `return logits.exp() / gumbel_noise` (matches the reference implementation in `refs/dllm/dllm/core/samplers/utils.py`).

**Lesson:** When implementing sampling tricks from papers, verify against reference code line-by-line. Mathematical formulas in papers often omit the distinction between log-space and probability-space. The reference code doesn't lie. And always test non-greedy generation -- greedy is the one mode that hides sampling bugs.

---

### Bug #23: Doc Packing Truncation Defeated Its Own Purpose
**Phase:** 5 | **Severity:** medium | **Time to find:** hours

**Symptom:** Document packing showed no throughput improvement over padding. Sequences still had significant wasted space.

**Root Cause:** The packer truncated any document longer than `seq_len - 1` tokens. Long documents (which are common in the training data) got cut instead of being split across multiple packed sequences. The overflow that should have carried to the next sequence was discarded.

**Fix:** Removed truncation. The packing buffer carries overflow to the next packed sequence.

**Lesson:** Document packing exists to eliminate wasted padding tokens. Truncating long documents re-introduces waste through a different mechanism. The packer must handle overflow by carrying it forward, not by discarding it.

---

## Patterns

Looking across all 23 bugs, several patterns emerge:

### 1. Silent correctness bugs are the most expensive

Bugs #1, #6, and #16 share a critical property: the training loop reported no errors, the loss looked reasonable, and GPU time was burning. The only signal was that the final result was wrong (bad generation, premature plateau, zero learning). These bugs cost the most time because there's no error message pointing you to the problem.

**Defense:** Add invariant checks to every training loop. Print `grad_norm` at step 0 (catches #16). Compare step-0 loss to `ln(vocab_size)` (catches #12). If loss converges too fast, suspect label leakage (catches #1). If loss plateaus at an unexpected value, suspect loss weighting (catches #6).

### 2. Attention masks are treacherous

Bug #1 is the canonical example, but any mask bug in a diffusion LM is a leakage bug. The model should never see the tokens it's being asked to predict. Off-by-one errors, wrong inequality directions, and transposed mask dimensions all manifest the same way: suspiciously low training loss, useless generation.

### 3. Never copy loss functions across schedules

Bugs #6 and #13 both involve loss weights that are correct under one assumption and wrong under another. The ELBO weight `1/t` is correct for linear schedules only. CART weights are designed for SFT, not pretraining. The derivation matters more than the formula.

### 4. Fused kernels can have broken backward passes

Forward correctness does not imply backward correctness (#16). Always verify gradients after integrating custom CUDA kernels. One line of `grad_norm` logging catches this class of bug instantly.

### 5. Optimizer APIs are fragile

Wrong pip package (#7), DDP requirement (#8), exact key schemas (#9), tied-weight double-counting (#10). Read the source. The docs are incomplete. Phase 5 solved this by implementing MuonClip from scratch -- eliminating the dependency entirely.

### 6. Mutable state is a bug factory

KV cache (#4), tied embeddings (#2, #10), cached masks (#19), compile modes (#11, #20). Any function that touches shared state must restore it on exit. Any tensor created in a special context (`inference_mode`) inherits that context permanently.

### 7. Defaults are not universal

Epsilon values (#21), compile modes (#11), loss normalization denominators (#12). When a parameter has a "default," check what that default actually is in YOUR implementation. It varies across libraries, dtypes, and versions.

### 8. Check if the technique was designed for your setting

CART was designed for SFT, not pretraining (#13). Dream's `1/t` weight assumes a linear schedule (#6). Liger FLCE works with `reduction='mean'`, not `'none'` (#16). `reduce-overhead` is for inference, not training (#11). Context matters more than the technique itself.
