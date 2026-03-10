# The Bug Museum

Every significant bug found during Open-dLLM development, preserved for education. Most repositories hide their bugs. We celebrate them.

Bugs are the best teachers. A working codebase tells you what to do; a bug tells you what you misunderstood. Each entry here traces the full chain from symptom to root cause to fix, because "it broke and I changed something" teaches nothing — understanding the *mechanism* teaches everything.

15 bugs across 5 phases. 3 were critical (silent correctness failures that produce plausible-looking but wrong results). The rest range from "cost us a day" to "cost us a week." Total debugging time: roughly 30+ hours of GPU-time-aware panic.

---

## Bug Hall of Fame

Three bugs that teach the most, hurt the worst, and are the easiest to reproduce in your own projects:

| Rank | Bug | Phase | Why it matters |
|------|-----|-------|----------------|
| 1 | [#6: ELBO Weight Schedule Mismatch](#bug-6-elbo-weight-schedule-mismatch) | 4 | Copied a formula between papers without checking a hidden assumption. Loss converged to the wrong value. 12 hours to diagnose. |
| 2 | [#13: Liger FLCE Backward Returns Zeros](#bug-13-liger-flce-backward-returns-zeros) | 5 | Correct loss, zero gradients. Model never learns but nothing looks wrong unless you check grad_norm. 6 hours to diagnose. |
| 3 | [#1: Staircase Mask Label Leakage](#bug-1-staircase-mask-label-leakage) | 3 | Off-by-one in an attention mask let the model see its own answers. Training looked perfect. Generation was garbage. |

All three share a trait: **the training loop reported no errors and the loss looked reasonable.** Silent correctness bugs are the most expensive kind.

---

## Phase 1: Character-Level dLLM

No significant bugs. The model is simple enough (character-level vocabulary, single-file implementation, ~1M parameters) that most mistakes surface immediately as crashes or obviously wrong output. This is the benefit of starting small.

---

## Phase 3: Block Diffusion

### Bug 1: Staircase Mask Label Leakage
**Phase:** 3 | **Severity:** Critical | **Time to diagnose:** ~4 hours

**Symptom:** Model trained to very low loss very quickly but generated gibberish. Training curves looked perfect — fast convergence, low final loss. But every generated sequence was incoherent.

**Root cause:** The staircase attention mask uses block indices to determine which blocks can attend to which. The comparison `block_idx >= query_block_idx` allowed noisy block N to attend to its own clean tokens in the concatenated `[x_t || x_0]` input. The model learned to copy the answer from the clean side instead of learning to predict it from context.

**Fix:** Change `block_idx >= query_block_idx` to `block_idx > query_block_idx` — strict inequality prevents self-attention to own clean tokens.

**Lesson:** In attention masks, off-by-one errors are label leakage. If training loss drops suspiciously fast, suspect the mask before anything else. Always verify that the model cannot see the answer during training. A good diagnostic: if training loss is much lower than you'd expect from a model of this size, something is leaking.

---

### Bug 2: Tied Embedding Init Order
**Phase:** 3 | **Severity:** High | **Time to diagnose:** ~2 hours

**Symptom:** Model weights had unexpected initialization values despite custom init code running without errors.

**Root cause:** Code called `self.apply(_init_weights)` AFTER setting `self.lm_head.weight = self.token_emb.weight` (weight tying). The `apply()` walks all modules and reinitializes all `nn.Linear` weights — including `lm_head`, which overwrites the tied embedding with a fresh random tensor, breaking the tie.

**Fix:** Call `self.apply(_init_weights)` BEFORE `self.lm_head.weight = self.token_emb.weight`.

**Lesson:** Weight tying is pointer aliasing. Any operation that touches one side affects both — and any operation that *replaces* one side breaks the alias. Init order matters when you have shared parameters.

---

### Bug 3: RoPE Buffer Too Short
**Phase:** 3 | **Severity:** Medium | **Time to diagnose:** ~30 minutes

**Symptom:** `IndexError` during generation, never during training.

**Root cause:** RoPE frequencies were precomputed for exactly `seq_len` positions. Training always uses exactly `seq_len`. But generation with KV caching can produce sequences longer than `seq_len` (e.g., iterative refinement over 2x the training context).

**Fix:** Precompute RoPE for `max(2 * seq_len, 4096)` positions.

**Lesson:** Generation uses different sequence lengths than training. Positional encoding buffers, KV cache sizes, and attention masks all need to handle the generation case, not just the training case. Over-allocate and let the indexing slice what it needs.

---

### Bug 4: KV Cache Stale State
**Phase:** 3 | **Severity:** High | **Time to diagnose:** ~3 hours

**Symptom:** Training crashed with a shape mismatch error at step 500 — the first evaluation step that calls `generate()`. The error pointed to an attention operation expecting batch=32 but getting batch=1 somewhere.

**Root cause:** `generate()` reset the KV cache at entry (to batch=1 for generation) but did NOT reset it at exit. When training resumed with batch=32, the stale batch=1 cache from generation caused a dimension mismatch in the next forward pass.

**Fix:** Reset KV cache both at entry AND exit of `generate()`. Better yet: use a context manager that guarantees cleanup.

**Lesson:** Any function that modifies global model state must clean up after itself. Reset-on-entry is necessary but not sufficient — you also need reset-on-exit (or better, RAII-style cleanup). This applies to KV caches, dropout modes, compilation states, and any other mutable model attribute.

---

### Bug 5: Qwen3 Tokenizer OOM
**Phase:** 3 | **Severity:** Medium | **Time to diagnose:** ~1 hour

**Symptom:** P100 (16 GB) OOM immediately on model creation, before any data was loaded.

**Root cause:** First attempt used Qwen3's 151K vocabulary tokenizer. Embedding layer: 151K x 384 x 4 bytes = 232 MB. With tied embeddings counted twice in memory profiling, plus the output projection, the embedding alone consumed over 400 MB. Combined with the rest of the model, it exceeded the P100's memory.

**Fix:** Built a custom 32K BPE tokenizer. Embedding: 32K x 384 x 4 = 49 MB. 4.7x reduction.

**Lesson:** Vocabulary size is a direct multiplier on embedding memory. On constrained hardware, vocab size is the first knob to check. A 151K vocab is fine on an A100; it's a dealbreaker on a P100. Always do the arithmetic before choosing a tokenizer.

---

## Phase 4: Modern Block Diffusion

### Bug 6: ELBO Weight Schedule Mismatch
**Phase:** 4 | **Severity:** Critical | **Time to diagnose:** ~12 hours

**Symptom:** Loss dropped from ~10.4 to ~4.0 in the first few hundred steps, then plateaued and refused to decrease further. The model was clearly learning something (loss decreased), but converged to the wrong value. No NaN, no spikes, no obvious anomaly — just a premature plateau.

**Root cause:** The ELBO loss for diffusion language models requires weighting each timestep by `1/mask_prob` (the inverse of the masking probability at that timestep). The original code copied `elbo_weight = 1/t` from the LLaDA paper. This is correct for LLaDA because LLaDA uses a LINEAR noise schedule where `mask_prob = t`.

Phase 4 uses a COSINE noise schedule where `mask_prob = 1 - cos^2(t * pi/2)`. At low t (e.g., t=0.1), cosine gives mask_prob = 0.024. The correct weight is 1/0.024 = 41.7. We used 1/0.1 = 10. Low-noise timesteps — the refinement steps where the model learns fine-grained token prediction — were under-weighted by 4-8x.

The model learned coarse structure (high-t, high mask_prob, weights roughly correct) but couldn't learn refinement (low-t, weights too small). Hence the plateau: good enough at filling in heavily masked text, unable to improve at lightly masked text.

**Fix:** Change `elbo_weight = 1/t` to `elbo_weight = 1/mask_prob` where `mask_prob` is computed from the actual noise schedule.

**Lesson:** Never copy loss formulas between papers without verifying the full derivation chain. The `1/t` simplification is schedule-dependent — it's a special case, not a general rule. This bug is invisible in training metrics (loss converges, just to the wrong value). The only way to catch it is to verify the math or notice that the final loss is higher than expected. Phase 5 switched to a LINEAR schedule specifically so that `mask_prob = t` and the simplification holds by construction.

---

### Bug 7: Muon pip Package Confusion
**Phase:** 4 | **Severity:** Medium | **Time to diagnose:** ~30 minutes

**Symptom:** `import muon` imported successfully but gave classes related to bioinformatics (single-cell RNA analysis) instead of an optimizer.

**Root cause:** `pip install muon` installs a bioinformatics package from the scanpy ecosystem. The ML Muon optimizer by Keller Jordan is only available via `pip install git+https://github.com/KellerJordan/Muon`.

**Fix:** Use the git URL for installation. Phase 5 eliminated this dependency entirely by implementing MuonClip from scratch.

**Lesson:** Always verify package identity before installing. PyPI has no namespace protection — anyone can claim any name. `pip install X` and "the X library" are not necessarily the same thing.

---

### Bug 8: Muon Param Group Key Mismatch
**Phase:** 4 | **Severity:** High | **Time to diagnose:** ~2 hours

**Symptom:** Cryptic `KeyError` deep inside Muon optimizer's `step()` method, several stack frames into internal optimizer logic.

**Root cause:** `MuonWithAuxAdam` requires param groups with EXACTLY these keys — no more, no fewer. Muon groups: `{params, lr, momentum, weight_decay, use_muon}`. Adam groups: `{params, lr, betas, eps, weight_decay, use_muon}`. An extra key like `name` (commonly added for logging) or a missing key like `momentum` causes a crash inside `step()`.

**Fix:** Match the exact key set for each group type. Strip any extra keys before passing to the optimizer.

**Lesson:** Third-party optimizers often have strict, undocumented param group schemas. PyTorch's built-in optimizers are lenient (they ignore extra keys). Custom optimizers are not. Read the source code, not just the README.

---

### Bug 9: Tied Embedding Optimizer Deduplication
**Phase:** 4 | **Severity:** High | **Time to diagnose:** ~3 hours

**Symptom:** Training produced inconsistent gradient updates. Embedding weights diverged from lm_head weights despite being tied. Sometimes manifested as NaN after many steps.

**Root cause:** `model.named_parameters()` yields the tied embedding tensor twice — once as `token_emb.weight` and once as `lm_head.weight`. Without deduplication, the tensor ends up in both the Muon group (as a 2D non-embedding weight) and the AdamW group (as an embedding weight). Both optimizers apply different updates to the same memory, causing conflicting gradient steps.

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

### Bug 10: CART Weight Explosion
**Phase:** 4 | **Severity:** High | **Time to diagnose:** ~2 hours

**Symptom:** Loss suddenly spiked to extreme values early in training when CART (Context-Adaptive Rescheduling Training, from the Dream 7B paper) was enabled.

**Root cause:** CART replaces the standard ELBO weight with a context-dependent weight based on how "surprising" each token is. Without a cap, the maximum weight reached 10,000+ for rare tokens at low noise levels. A single "hard" token could dominate an entire batch's gradient, causing effective batch size to collapse to 1.

**Fix:** Cap CART weight at `1/t_min = 20`. Disabled CART by default (opt-in with `--cart`). CART is designed for supervised fine-tuning where the data distribution is narrow — for pretraining on diverse web text, the standard ELBO weight works better.

**Lesson:** Any adaptive weighting scheme needs an upper bound. Unbounded weights cause gradient explosion. This applies to CART, importance sampling, focal loss, and any other scheme that reweights examples based on difficulty or surprise.

---

### Bug 11: torch.compile reduce-overhead VRAM Explosion
**Phase:** 4 | **Severity:** High | **Time to diagnose:** ~3 hours

**Symptom:** OOM on batch sizes that should fit based on parameter count arithmetic. A 125M model on an 80 GB A100 OOM'd at batch=16, which should need roughly 40 GB.

**Root cause:** `torch.compile(model, mode="reduce-overhead")` uses CUDA graphs, which capture and replay entire execution traces. CUDA graphs require all intermediate tensors to remain allocated for the entire graph lifetime. For a 20-layer transformer in training mode, this means all 20 layers' forward activations are pinned simultaneously — they can't be freed and reused as in normal eager execution.

**Fix:** Use `mode="default"` for training. The default mode uses Triton kernel fusion (fusing small ops into fewer, larger kernels) without CUDA graphs. `reduce-overhead` is designed for inference with batch=1 where the overhead of kernel launches dominates — not for training where activation memory dominates.

**Lesson:** `torch.compile` modes have radically different memory profiles. Every major training codebase in 2026 (Karpathy's nanochat, Meta's torchtitan, modded-nanogpt) uses default mode for training. The name "reduce-overhead" sounds universally good — it's not. It trades memory for kernel launch latency, which is only a bottleneck at small batch sizes.

---

### Bug 12: Modal Image Missing git
**Phase:** 4 | **Severity:** Medium | **Time to diagnose:** ~15 minutes

**Symptom:** `pip install git+https://github.com/...` failed with "git: command not found" on Modal.

**Root cause:** Modal's `debian_slim` base image is truly minimal — no `git`, no `curl`, no build tools. The pip `git+https://` protocol shells out to the git binary, which doesn't exist.

**Fix:** Add `.apt_install("git")` to the Modal image definition BEFORE any `pip_install` that uses git URLs.

**Lesson:** Minimal container images lack tools you assume exist. The dependency chain is: your code -> pip -> git -> the package. If any link is missing, it fails. Always check transitive dependencies in container environments.

---

## Phase 5: Modular Block Diffusion

### Bug 13: Liger FLCE Backward Returns Zeros
**Phase:** 5 | **Severity:** Critical | **Time to diagnose:** ~6 hours

**Symptom:** `loss=19.11` at step 0 (correct — matches expected ELBO-weighted init loss). `grad_norm=0.000000` at every single step. Model weights identical at step 0 and step 1000. Loss constant at 19.11 forever.

**Root cause:** Liger Kernel's `LigerFusedLinearCrossEntropyFunction` computes the correct loss in the forward pass but returns zero gradients in the backward pass when called with `reduction='none'`. The dLLM training loop needs `reduction='none'` to apply per-token ELBO weighting before reducing. The bug is in Liger's custom CUDA backward kernel — tracked at [linkedin/Liger-Kernel#488](https://github.com/linkedin/Liger-Kernel/issues/488). Forward: correct. Backward: zeros.

This is the most insidious type of bug: the loss value is correct, so any check of "is the loss reasonable?" passes. The model simply never updates. You can run for 10,000 steps and burn hours of GPU time before noticing that nothing changed.

**Fix:** Replaced Liger FLCE with chunked cross-entropy using `torch.utils.checkpoint`:
```python
# Process 4096 tokens at a time instead of materializing full [batch, vocab] logits
for chunk in logits.split(chunk_size):
    loss += checkpoint(ce_fn, chunk, targets_chunk, weights_chunk)
```
Peak memory: ~1.5 GB vs 24 GB for full logit materialization. Liger's RMSNorm and SwiGLU kernels are unaffected — only FLCE has the backward bug.

**Lesson:** Always verify that `backward()` produces nonzero gradients, not just that `forward()` gives a correct loss. The simplest diagnostic: print `grad_norm` at step 0. If it's exactly 0.0, the backward pass is broken. This should be a mandatory check in any training loop — one line of logging that saves hours of debugging.

---

### Bug 14: WSD Schedule LR Factor Bug
**Phase:** 5 | **Severity:** Medium | **Time to diagnose:** ~1 hour

**Symptom:** Learning rate during the decay phase of the Warmup-Stable-Decay schedule didn't decrease as expected. LR was much higher than intended.

**Root cause:** PyTorch's `LambdaLR` scheduler expects the lambda function to return a multiplicative factor in [0, 1], which it multiplies by the `initial_lr` stored in each param group. The implementation was returning an absolute LR value instead of a factor, so the scheduler multiplied `initial_lr * absolute_lr` — squaring the learning rate effectively.

**Fix:** Return `current_lr / initial_lr` (the factor) instead of `current_lr` (the absolute value).

**Lesson:** PyTorch LR schedulers use multiplicative factors, not absolute values. The `LambdaLR` documentation says this, but it's easy to miss. When your LR schedule looks wrong, check whether you're returning a factor or an absolute value.

---

### Bug 15: Trackio HF Space Crash
**Phase:** 5 | **Severity:** Medium | **Time to diagnose:** ~20 minutes

**Symptom:** Training job crashed at step 0, before any training happened. Error pointed to `trackio.init()` failing to connect to a HuggingFace Space.

**Root cause:** `trackio.init()` raises an exception if the HF write token is missing, expired, or the target Space doesn't exist. This call was in the training setup path with no error handling, so a monitoring failure killed the entire training job.

**Fix:** Wrap `trackio.init()` in try/except. Log the failure, continue training without the dashboard.

**Lesson:** Non-essential services (monitoring, logging, dashboards) must never crash the main training loop. Use try/except, set a `dashboard_available` flag, and check it before logging calls. The training job is the expensive part — the dashboard is nice-to-have.

---

## Patterns

Across 15 bugs and 5 phases, several recurring themes emerge:

### 1. Silent correctness bugs are the most expensive

Bugs #1, #6, and #13 share a critical property: the training loop reported no errors, the loss looked reasonable, and GPU time was burning. The only signal was that the final result was wrong (bad generation, premature plateau, zero learning). These bugs cost the most time because there's no error message pointing you to the problem.

**Defense:** Add invariant checks to every training loop. Print `grad_norm` at step 0 (catches #13). Compare step-0 loss to `ln(vocab_size)` (catches normalization bugs). If loss converges too fast, suspect label leakage (catches #1). If loss plateaus at an unexpected value, suspect loss weighting (catches #6).

### 2. Attention mask bugs are label leakage

Bug #1 is the canonical example, but any mask bug in a diffusion LM is a leakage bug. The model should never see the tokens it's being asked to predict. Off-by-one errors, wrong inequality directions, and transposed mask dimensions all manifest the same way: suspiciously low training loss, useless generation.

**Defense:** Write explicit tests that verify the mask blocks the model from seeing target tokens. Don't trust visual inspection of mask matrices — test with actual forward passes.

### 3. Loss weighting bugs are schedule-dependent

Bugs #6 and #10 both involve loss weights that are correct under one assumption and wrong under another. The ELBO weight `1/t` is correct for linear schedules only. CART weights are correct for narrow distributions only.

**Defense:** Derive your loss from first principles for YOUR specific schedule and data distribution. Don't copy formulas between papers without checking every assumption. When in doubt, use the simplest schedule where the math is trivially correct (Phase 5 chose linear for exactly this reason).

### 4. Optimizer gotchas compound with tied weights

Bugs #2, #8, and #9 all involve optimizer-parameter interactions. Tied weights appear twice in iterators. Custom optimizers have strict schemas. Init order matters when parameters are aliased.

**Defense:** Always deduplicate by `data_ptr()`. Always read optimizer source code. Always init before tying.

### 5. Container and dependency assumptions break silently

Bugs #5, #7, and #12 are all "works on my machine" failures — wrong pip package, missing binary in container, unexpected vocab size. They're quick to fix once diagnosed but waste time when they hit in remote training environments (Modal, Kaggle) where debugging is slower.

**Defense:** Pin exact package versions and URLs. Test container builds locally before launching GPU jobs. Do arithmetic on memory requirements before choosing a model configuration.

### 6. torch.compile is not a free lunch

Bug #11 shows that compile mode selection has major consequences. "reduce-overhead" sounds like a universal improvement but trades memory for latency — the wrong tradeoff for training. Similarly, compile interacts badly with gradient checkpointing, custom autograd functions, and dynamic shapes.

**Defense:** Use `mode="default"` for training. Test VRAM at your target batch size before committing to a long run. Watch for recompilation warnings in logs.
