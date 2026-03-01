# Phase 2: nano_dllm — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a single-file BPE diffusion language model with cosine noise schedule and ELBO-weighted loss, trained on FineWeb-Edu via streaming, configurable via `--depth`.

**Architecture:** Same bidirectional transformer as Phase 1 (the 5 DIFF points are unchanged), upgraded with BPE tokenization, SwiGLU activation, cosine schedule masking, ELBO loss weighting, and configurable depth. Data streams from HuggingFace FineWeb-Edu — no local downloads needed for training.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `datasets` (streaming), HuggingFace `tokenizers` (BPE)

---

## Task 1: Setup and BPE Tokenizer Training

**Files:**
- Create: `02_nano_dllm/train_tokenizer.py`

**Step 1: Write train_tokenizer.py**

A small script (~60 lines) that:
1. Downloads a sample of FineWeb-Edu text (~50MB) using `datasets` library (not streaming — need files on disk for tokenizer training). Use `load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)`, iterate to collect ~50K documents, write their text to a temporary file.
2. Trains a BPE tokenizer using HuggingFace `tokenizers` library:
   - `vocab_size=32768` (32K, power of 2 for efficient embedding)
   - Special tokens: `["[MASK]"]` — mask token gets id=0
   - Pre-tokenizer: `ByteLevel` (handles all unicode, no UNK needed)
   - No post-processor (we don't need CLS/SEP tokens)
3. Saves the trained tokenizer to `02_nano_dllm/tokenizer.json`
4. Prints vocab size and encodes a test sentence to verify

Important details:
- Use `ByteLevelBPETokenizer` or the lower-level `Tokenizer(BPE())` API — either works, but the lower-level API gives more control over special tokens
- The `[MASK]` token must be registered so `tokenizer.token_to_id("[MASK]")` returns 0
- After training, verify: `tokenizer.get_vocab_size()` should be 32768

**Step 2: Run train_tokenizer.py and verify**

Run: `cd 02_nano_dllm && uv run python3 train_tokenizer.py`
Expected: Downloads ~50K FineWeb-Edu docs, trains BPE in ~30 seconds, saves `tokenizer.json`, prints vocab size and sample encoding.

**Step 3: Commit**

Stage `train_tokenizer.py` and `tokenizer.json`. Message: "feat: Phase 2 BPE tokenizer training on FineWeb-Edu sample"

---

## Task 2: Write nano_dllm.py — Imports, Config, Data Pipeline

**Files:**
- Create: `02_nano_dllm/nano_dllm.py`

**Step 1: Write the file header and data pipeline**

Start with:
- Module docstring with ASCII architecture diagram showing the Phase 2 upgrade path from Phase 1. Mark all 6 `[NEW]` additions (BPE, cosine schedule, ELBO weighting, --depth, FineWeb-Edu, SwiGLU). Keep the 5 `[DIFF]` markers from Phase 1.
- Imports: `os, sys, math, time, argparse, torch, torch.nn, F, tokenizers.Tokenizer, datasets.load_dataset`
- CLI argument parsing: `--train`, `--depth` (int, default=6), `--prompt` (str, optional), `--max-tokens` (int, default=512)
- Hyperparameters section derived from depth:
  - `n_layer = depth`
  - `n_embd = depth * 64`
  - `n_head = depth` (head_dim=64 always)
  - `block_size = 512` (fixed, BPE tokens)
  - `batch_size = 32` (tunable, smaller than Phase 1 due to larger vocab)
  - `max_iters = 20000`
  - `eval_interval = 500`
  - `eval_iters = 50`
  - `learning_rate = 1e-3`
  - `warmup_iters = 1000`
  - `min_lr = 1e-4` (for cosine LR decay)
- Device detection (cuda > mps > cpu)
- Tokenizer loading from `tokenizer.json` (located relative to script via `__file__`)
  - Extract `mask_token_id` from tokenizer
  - Define `vocab_size = tokenizer.get_vocab_size()`
  - `encode(text)` and `decode(ids)` wrapper functions
- `get_batch()` function with FineWeb-Edu streaming `[NEW]`:
  - Uses a global iterator over `load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).shuffle(buffer_size=10_000)`
  - Each call: pull `batch_size` documents, tokenize, truncate/pad to `block_size`, stack into tensor
  - Apply cosine schedule masking `[NEW]`:
    - Sample `t ~ U[0,1]` per sequence in batch
    - Compute `mask_prob = 1 - cos²(t · π/2)` per sequence
    - Apply per-token binary mask where `rand < mask_prob`
    - Replace masked positions with `mask_token_id`
  - Return: `(masked_input, original_targets, mask_bool, t_values)` — t_values needed for ELBO weighting
  - For validation: use a separate streaming iterator (`.skip(100_000)` to get non-overlapping data)

Include ASCII diagrams for:
- The cosine schedule curve: show `alpha_t` vs `t` with key points at t=0, 0.25, 0.5, 0.75, 1.0
- The data pipeline flow: FineWeb-Edu → tokenize → pad → mask (cosine) → batch

**Step 2: Verify data pipeline works**

Quick test: import the module, call `get_batch()` once, print shapes and mask statistics.

Run: `uv run python3 -c "...quick import test..."`
Expected: Prints batch shape `(32, 512)`, mask ratio close to expected cosine mean (~0.36).

**Step 3: Commit**

Message: "feat: nano_dllm data pipeline with FineWeb-Edu streaming and cosine masking"

---

## Task 3: Write nano_dllm.py — Model Architecture

**Files:**
- Modify: `02_nano_dllm/nano_dllm.py`

**Step 1: Add model components**

Add in order (same structure as Phase 1, with upgrades marked `[NEW]`):

1. `norm(x)` — functional RMSNorm (identical to Phase 1)
2. `apply_rotary_emb(x, cos, sin)` — RoPE helper (identical to Phase 1)
3. `MultiHeadAttention` class — Q/K/V projections, RoPE, QK-norm, `is_causal=False` `[DIFF 2]`
   - Same as Phase 1, but dimensions from `--depth` parametrization
4. `SwiGLU` class `[NEW]` — replaces Phase 1's ReLU² MLP:
   - Three linear projections: w1 (gate), w2 (up), w3 (down)
   - `hidden_dim = int(8/3 * n_embd)` rounded up to nearest multiple of 256
   - Forward: `F.silu(w1(x)) * w2(x)` then `w3(result)`
   - Include ASCII diagram comparing ReLU² MLP vs SwiGLU
5. `Block` class — pre-norm attention + pre-norm SwiGLU residual
6. `Model` class:
   - Token embedding (vocab_size × n_embd)
   - RoPE frequency buffers (precomputed cos/sin)
   - `n_layer` transformer blocks
   - Final RMSNorm + linear head (n_embd → vocab_size)
   - `init_weights()` with proper scaling
   - `forward(idx, targets=None, mask=None, t=None)`:
     - Embed, apply RoPE through blocks, project to vocab
     - If targets provided: compute ELBO-weighted loss `[NEW]`:
       - CE on masked positions only `[DIFF 4]`
       - Weight by `1/t` per sample (the ELBO weight from MDLM)
       - Clamp `t` to `[1e-4, 1.0]` for numerical stability
       - Final loss = mean of `(1/t_i) * CE_i` across batch
     - Return logits and loss

Include ASCII diagrams for:
- SwiGLU vs ReLU² architecture comparison
- ELBO loss weighting: show how `1/t` upweights low-masking samples (harder predictions)
- The depth → dimensions mapping table

**Step 2: Verify model instantiation**

Instantiate `Model()` with default depth=6, verify parameter count (~30M expected).
Also test with depth=12 (~150M expected).

**Step 3: Commit**

Message: "feat: nano_dllm model architecture with SwiGLU and ELBO loss"

---

## Task 4: Write nano_dllm.py — Generation / Inference

**Files:**
- Modify: `02_nano_dllm/nano_dllm.py`

**Step 1: Add the generate function**

Implement confidence-based parallel decoding `[DIFF 5]` (same algorithm as Phase 1, adapted for BPE):
- Accept: model, prompt text (optional), max_new_tokens, temperature, confidence_threshold, top_k
- If prompt given: encode with BPE tokenizer, use as prefix
- If no prompt: start with all-mask sequence
- Generate block-by-block (block_len = block_size - prompt_len, within block_size window)
- Within each block: iteratively unmask via confidence threshold
  - If no position exceeds threshold: unmask the single most confident one
  - Top-k sampling from normalized top-k probabilities
- BPE decode the final token IDs
- Print generation stats (total steps, avg decoded per step, tokens/sec)

Key difference from Phase 1: BPE decode instead of character decode. The generated token IDs go through `tokenizer.decode(ids)` which handles subword merging.

**Step 2: Add estimate_loss function**

Standard eval function: sample `eval_iters` batches for train/val, compute ELBO-weighted loss, average.

**Step 3: Add learning rate schedule helper**

Implement cosine decay with warmup:
- Linear warmup from 0 to `learning_rate` over `warmup_iters` steps
- Cosine decay from `learning_rate` to `min_lr` over remaining steps
- Return current lr given step number

**Step 4: Commit**

Message: "feat: nano_dllm generation with BPE decoding and LR schedule"

---

## Task 5: Write nano_dllm.py — Training Loop and Main

**Files:**
- Modify: `02_nano_dllm/nano_dllm.py`

**Step 1: Add the __main__ block**

Structure:
1. Parse CLI args (`--train`, `--depth`, `--prompt`, `--max-tokens`)
2. Print config: depth, n_layer, n_embd, n_head, block_size, device
3. Set `weights_path` to `weights/nano_dllm_d{depth}.pt` (relative to script via `__file__`)
4. Instantiate Model with depth-derived config, move to device, print param count
5. If weights exist and not `--train`: load weights
6. Else if `--train`: train from scratch:
   - AdamW optimizer with `weight_decay=0.1` (exclude biases and norms)
   - Loop `max_iters` iterations:
     - Compute learning rate via schedule, update optimizer param groups
     - Get batch (streaming from FineWeb-Edu)
     - Forward pass (returns ELBO-weighted loss)
     - Backward, grad clip (max_norm=1.0), optimizer step
     - Every `eval_interval`: estimate_loss, generate sample, print progress
     - Print step, loss, lr, tokens/sec periodically
   - Save weights after training
7. Generate final output:
   - If `--prompt`: use prompt as prefix
   - Else: generate unconditionally
   - Generate `max_tokens` tokens with temp=0.8, top_k=5
8. Print generation time and stats

**Step 2: Smoke test (short run)**

Temporarily reduce `max_iters` to 100 in the CLI or via a quick hack, run training to verify no crashes.

Run: `cd 02_nano_dllm && uv run python3 nano_dllm.py --train --depth 6`
Expected: Streams data, trains 100 iterations, loss decreases, generates a sample (likely gibberish at 100 steps), no errors.

**Step 3: Restore defaults and commit**

Message: "feat: nano_dllm training loop — Phase 2 complete"

---

## Task 6: Write the Phase 2 Theory README

**Files:**
- Create: `02_nano_dllm/README.md`

**Step 1: Write the educational README**

Structure:

1. Title and one-liner: "Phase 2: nano_dllm — A real diffusion language model"
2. "What You'll Learn" bullet list: BPE tokenization, cosine noise schedule, ELBO-weighted loss, scaling via depth, streaming data
3. "What's New from Phase 1" — table showing the 6 upgrades with rationale
4. "Cosine Noise Schedule" — the math, ASCII diagram of the curve, why it's Fisher-Rao optimal, intuition for why mid-range masking ratios are most informative
5. "ELBO-Weighted Loss" — derive `L = (1/t) · CE` from the continuous-time ELBO integral, explain the MDLM simplification, ASCII diagram showing weight distribution across t
6. "BPE Tokenization for Diffusion" — why BPE matters for dLLMs (larger vocab = fewer tokens = less long-range dependency), how [MASK] token is added, how masking works at subword level
7. "SwiGLU Activation" — the formula, comparison with ReLU², why it helps (gated + smooth), ASCII diagram
8. "Scaling via --depth" — the dimension mapping table, compute-optimal intuition
9. "Quick Start" — commands to train tokenizer, train model, generate
10. "What's Next" — preview of Phase 3 (block diffusion)
11. "References" — MDLM, LLaDA, cosine schedule optimality paper

Target: ~200-300 lines of markdown with ASCII diagrams.

**Step 2: Commit**

Message: "docs: Phase 2 theory README with cosine schedule and ELBO explanation"

---

## Task 7: End-to-End Validation

**Step 1: Full pipeline test**

Run the full pipeline:
1. `cd 02_nano_dllm && uv run python3 train_tokenizer.py` (if tokenizer.json not yet created)
2. `uv run python3 nano_dllm.py --train --depth 6`

Expected:
- Prints depth=6, ~30M parameters
- Streams FineWeb-Edu data
- Loss decreases over training (from ~10-11 → ~5-6 expected for BPE)
- Sample text progressively improves
- Saves weights to `weights/nano_dllm_d6.pt`

**Step 2: Test generation from saved weights**

Run: `uv run python3 nano_dllm.py --depth 6 --prompt "The meaning of life is"`
Expected: Loads weights, generates text continuing the prompt, shows parallel decoding stats.

**Step 3: Test unconditional generation**

Run: `uv run python3 nano_dllm.py --depth 6`
Expected: Generates text from all-mask sequence, shows parallel decoding efficiency.

**Step 4: Verify the NEW and DIFF markers**

Grep for `[NEW]` and `[DIFF]` markers in `nano_dllm.py`. Verify:
- 5 [DIFF] markers preserved (mask token, bidirectional attention, random masking, masked loss, confidence-based decoding)
- 6 [NEW] markers present (BPE, cosine schedule, ELBO weighting, --depth, FineWeb-Edu, SwiGLU)

**Step 5: Commit**

Message: "feat: Phase 2 validated end-to-end"

---

## Task 8: Update Root README

**Files:**
- Modify: `README.md`

**Step 1: Update Phase 2 status in learning roadmap**

Change Phase 2 row from "Coming soon" to "Done" in the roadmap table.

**Step 2: Add Phase 2 quick start commands**

Add a section or update the Quick Start to mention Phase 2:
```
# Phase 2 (requires pip install datasets tokenizers)
pip install -e ".[phase2]"
cd 02_nano_dllm
python train_tokenizer.py
python nano_dllm.py --train --depth 6
```

**Step 3: Update project structure tree**

Add the Phase 2 files to the tree diagram.

**Step 4: Commit**

Message: "docs: update root README for Phase 2 completion"

---

## Summary

| Task | What | Files | Est. Effort |
|------|------|-------|-------------|
| 1 | BPE tokenizer training script | train_tokenizer.py | Small |
| 2 | Data pipeline + config | nano_dllm.py (partial) | Medium |
| 3 | Model architecture | nano_dllm.py (partial) | Medium |
| 4 | Generation + LR schedule | nano_dllm.py (partial) | Medium |
| 5 | Training loop + main | nano_dllm.py (complete) | Medium |
| 6 | Phase 2 theory README | README.md | Medium |
| 7 | End-to-end validation | Run training | Medium + training time |
| 8 | Update root README | README.md | Small |
