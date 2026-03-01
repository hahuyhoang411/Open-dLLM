# Phase 1: Hello Diffusion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a self-contained, annotated ~400-line diffusion language model that trains on Tiny Shakespeare and generates recognizable text — the "hello world" of dLLMs.

**Architecture:** A standard 6-layer transformer with bidirectional attention that predicts original tokens from partially masked sequences. Exactly 5 surgical changes from a GPT: add mask token, bidirectional attention, masked training objective, loss on masked positions only, confidence-based parallel decoding.

**Tech Stack:** Python 3.10+, PyTorch (only dependency)

---

## Task 1: Project Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `01_hello_diffusion/download_data.py`

**Step 1: Update pyproject.toml**

Replace the current contents with proper project metadata and torch dependency. Remove kaggle dependencies. Add optional `phase2` extras for `datasets` and `tokenizers`.

**Step 2: Write download_data.py**

A tiny script (~15 lines) that downloads Tiny Shakespeare from Karpathy's char-rnn repo to `01_hello_diffusion/data.txt` using only `urllib.request`. Check if file exists first to skip re-download.

**Step 3: Verify data download works**

Run: `python 01_hello_diffusion/download_data.py`
Expected: Downloads `data.txt` (~1.1MB), prints byte count.

**Step 4: Commit**

Stage pyproject.toml, download_data.py. Message: "feat: project scaffolding and Phase 1 data download"

---

## Task 2: Write the Phase 1 Theory README

**Files:**
- Create: `01_hello_diffusion/README.md`

**Step 1: Write the educational README**

This is a critical deliverable. Structure:

1. Title and one-liner
2. "What You'll Learn" bullet list
3. "The Big Idea" — ASCII diagram: clean text -> mask -> model predicts -> unmask iteratively
4. "How Autoregressive Models Work" — brief review with diagram
5. "How Diffusion Models Work for Text" — forward process (masking) and reverse process (denoising) with ASCII diagrams
6. "The 5 Changes from GPT to dLLM" — table comparing the two
7. "Training" — explain random masking ratio, CE on masked positions, with ASCII diagram
8. "Inference: Confidence-Based Parallel Decoding" — step-by-step walkthrough with ASCII
9. "Quick Start" — commands to download, train, generate
10. "What's Next" — preview of Phase 2
11. "References" — links to papers and repos

Target: ~200-300 lines of markdown with multiple ASCII diagrams.

**Step 2: Commit**

Stage README. Message: "docs: Phase 1 theory README with ASCII diagrams"

---

## Task 3: Write hello_diffusion.py — Tokenizer and Data

**Files:**
- Create: `01_hello_diffusion/hello_diffusion.py`

**Step 1: Write the file**

Start with:
- Module docstring with ASCII architecture diagram, references, and explanation of the 5 DIFF markers
- Imports (os, sys, time, torch, torch.nn, F)
- Hyperparameters section (batch_size=64, block_size=256, max_iters=10000, etc.)
- Device detection (cuda > mps > cpu)
- Data loading from data.txt
- Character-level tokenizer with `_` as mask token (id=0), marked [DIFF]
- encode/decode functions
- Train/val split (90/10)
- `get_batch` function with random masking per sample, marked [DIFF]

Include ASCII diagrams for:
- The tokenizer character mapping (showing mask token at position 0)
- The get_batch data flow (text -> random mask -> masked input + targets + mask)

Mark every diffusion-specific piece with `[DIFF]` in comments.

**Step 2: Verify data section runs**

Run a quick import/execution test to ensure data loads correctly.

**Step 3: Commit**

Message: "feat: hello_diffusion tokenizer and data loading"

---

## Task 4: Write hello_diffusion.py — Model Architecture

**Files:**
- Modify: `01_hello_diffusion/hello_diffusion.py`

**Step 1: Add model components**

Add in order:
- `norm(x)` — functional RMSNorm (no learnable params)
- `apply_rotary_emb(x, cos, sin)` — RoPE helper
- `MultiHeadAttention` class — Q/K/V projections, RoPE, QK-norm, `is_causal=False` [DIFF]
- `MLP` class — Linear -> ReLU squared -> Linear
- `Block` class — pre-norm residual
- `Model` class — token embedding + RoPE buffers + blocks + lm_head + init_weights

Include ASCII diagrams for:
- Bidirectional vs causal attention masks (showing the filled squares pattern)
- Why bidirectional attention matters for predicting masked tokens
- The loss computation (CE only on masked positions, marked [DIFF])

The attention class must use `is_causal=False` in `F.scaled_dot_product_attention`. This is THE key [DIFF] from GPT.

The forward method must:
1. Accept idx, targets (optional), mask (optional)
2. Embed tokens, apply RoPE, run through blocks, project to vocab
3. If targets provided: compute CE loss on masked positions only [DIFF]
4. Return logits and loss

**Step 2: Verify model instantiation**

Instantiate Model(), verify ~10.7M parameters.

**Step 3: Commit**

Message: "feat: hello_diffusion model architecture with educational annotations"

---

## Task 5: Write hello_diffusion.py — Generation / Inference

**Files:**
- Modify: `01_hello_diffusion/hello_diffusion.py`

**Step 1: Add the generate function**

Implement confidence-based parallel decoding [DIFF]:
- Accept model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3
- Use first prompt_len chars from data as context
- Generate block-by-block (block_len=240 within block_size=256)
- Within each block: iteratively unmask via confidence threshold
- If no position exceeds threshold: unmask the single most confident one
- Top-k sampling from normalized top-k probabilities
- Print generation stats (total steps, avg decoded per step)

Include a thorough ASCII diagram showing the iterative unmasking process step by step, contrasting with GPT's sequential generation.

**Step 2: Add estimate_loss function**

Standard eval function: sample `eval_iters` batches for train/val, average losses.

**Step 3: Commit**

Message: "feat: hello_diffusion generation with confidence-based decoding"

---

## Task 6: Write hello_diffusion.py — Training Loop and Main

**Files:**
- Modify: `01_hello_diffusion/hello_diffusion.py`

**Step 1: Add the __main__ block**

Structure:
1. Parse `--train` flag from sys.argv
2. Set weights_path to `weights/diffusion.pt`
3. Instantiate Model, move to device, print param count
4. If weights exist and not --train: load weights
5. Else: train from scratch:
   - AdamW optimizer, lr=3e-4
   - Loop max_iters iterations
   - Every eval_interval: estimate_loss, generate sample, print
   - Standard: zero_grad, forward, loss.backward, optimizer.step
   - Save weights after training
6. Generate final output (2000 chars, temp=0.8, top_k=2)
7. Print generation time

Include an ASCII diagram showing the training loop data flow.

**Step 2: Smoke test (short run)**

Temporarily reduce max_iters to 100, run training to verify no crashes. Check that loss prints and a sample generates.

Run: `cd 01_hello_diffusion && python hello_diffusion.py --train`
Expected: Trains 100 iterations, prints losses, generates gibberish sample, no errors.

**Step 3: Restore max_iters to 10000 and commit**

Message: "feat: hello_diffusion training loop — Phase 1 complete"

---

## Task 7: End-to-End Validation

**Step 1: Clean run from scratch**

Run the full pipeline:
1. `python 01_hello_diffusion/download_data.py`
2. `cd 01_hello_diffusion && python hello_diffusion.py --train`

Expected:
- Downloads data (~1.1MB)
- Prints ~10.7M parameters
- Trains 10K iterations
- Loss decreases from ~4.0 to ~2.0 over training
- Sample text progressively looks more like Shakespeare
- Saves weights to weights/diffusion.pt

**Step 2: Test generation from saved weights**

Run: `python hello_diffusion.py`
Expected: Loads weights, generates 2000 characters, prints generation stats showing parallel decoding efficiency (avg decoded per step > 1).

**Step 3: Verify the 5 diff points**

Manually verify that the code has exactly 5 [DIFF]-marked changes from a standard GPT:
1. Mask token added to charset
2. is_causal=False in attention
3. get_batch applies random masking
4. forward computes loss on masked positions only
5. generate uses confidence-based parallel decoding

**Step 4: Commit**

Message: "feat: Phase 1 complete — hello_diffusion validated end-to-end"

---

## Task 8: Root README and Cleanup

**Files:**
- Modify: `README.md`
- Delete: `main.py`

**Step 1: Remove placeholder main.py**

Delete the hello-world main.py.

**Step 2: Write the full root README**

Contents:
- Title: Open-dLLM
- Tagline: "Learn diffusion language models from scratch, one phase at a time"
- What this project teaches (2-3 sentences)
- Learning roadmap table (Phase 1-4 with status, description, prerequisites)
- Quick start for Phase 1 (download data, train, generate)
- Project structure tree
- Key references (papers, repos)
- Credits (tiny-diffusion, nanochat, MDLM, LLaDA, Mercury)
- License (MIT)

**Step 3: Commit**

Message: "docs: root README with learning roadmap, remove placeholder main.py"

---

## Summary

| Task | What | Files | Est. Time |
|------|------|-------|-----------|
| 1 | Project scaffolding + data download | pyproject.toml, download_data.py | 5 min |
| 2 | Phase 1 theory README | 01_hello_diffusion/README.md | 15 min |
| 3 | Tokenizer and data section | hello_diffusion.py (partial) | 10 min |
| 4 | Model architecture | hello_diffusion.py (partial) | 15 min |
| 5 | Generation / inference | hello_diffusion.py (partial) | 10 min |
| 6 | Training loop and main | hello_diffusion.py (complete) | 10 min |
| 7 | End-to-end validation | Run training | 20-30 min |
| 8 | Root README + cleanup | README.md, rm main.py | 5 min |

**Total implementation time**: ~90 minutes + training time
