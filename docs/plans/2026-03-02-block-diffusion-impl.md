# Phase 3: Block Diffusion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `03_block_diffusion/block_dllm.py` — a single-file block diffusion language model with configurable block size, KV caching, and variable-length generation.

**Architecture:** Port Phase 2's transformer (RMSNorm, SwiGLU, RoPE) and replace `is_causal=False` with an explicit staircase attention mask from BD3-LMs. Use Qwen 3 tokenizer (152K vocab) with added [MASK] token. Training uses `[x_t || x_0]` concatenation with shared position IDs. Inference uses block-by-block generation with KV caching.

**Tech Stack:** Python 3.10+, PyTorch, transformers (Qwen 3 tokenizer), datasets (FineWeb-Edu)

**Reference:** BD3-LMs (github.com/kuleshov-group/bd3lms), Phase 2 (02_nano_dllm/nano_dllm.py)

---

## Task 1: Scaffold — CLI, Config, Tokenizer

**Files:**
- Create: `03_block_diffusion/block_dllm.py`

**Step 1: Write the CLI and configuration section**

Port from Phase 2 (nano_dllm.py lines 98-142) with these changes:
- Add `--block-size` argument (int, default=4, choices include 1,2,4,8,16 and special value 0 meaning "full sequence")
- Keep `--train`, `--depth`, `--prompt`, `--max-tokens`
- Add `--denoise-steps` argument (int, default=10, number of denoising steps per block during generation)
- Depth scaling: same as Phase 2 (`n_layer=depth`, `n_embd=depth*64`, `n_head=depth`, `head_dim=64`)
- `block_size_seq = 512` (sequence length, same as Phase 2)
- When `--block-size 0`, set `block_size_blk = block_size_seq` (full-sequence diffusion, equivalent to Phase 2)

**Step 2: Write the tokenizer section**

Replace Phase 2's custom BPE tokenizer with Qwen 3:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
vocab_size = len(tokenizer)  # 151937 (151936 + 1 for [MASK])
mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
eos_token_id = tokenizer.eos_token_id

def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)

def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=False)
```

Note: `add_special_tokens=False` in encode — we don't want BOS/EOS injected automatically. The [MASK] token will be the last token in the vocabulary.

**Step 3: Verify CLI and tokenizer work**

Run: `uv run python 03_block_diffusion/block_dllm.py --help`
Expected: Shows help with all arguments including --block-size and --denoise-steps.

Run: `uv run python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); t.add_special_tokens({'additional_special_tokens': ['[MASK]']}); print(len(t), t.convert_tokens_to_ids('[MASK]'))"`
Expected: Prints vocab size and [MASK] token ID.

**Step 4: Commit**

Message: "feat: Phase 3 scaffold — CLI, config, Qwen 3 tokenizer"

---

## Task 2: Model Architecture — Transformer with Explicit Attention Mask

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Write the model components**

Port directly from Phase 2 (nano_dllm.py lines 338-510) with ONE change — the attention mechanism:

1. `norm(x)` — RMSNorm (identical to Phase 2 line 338-340)
2. `apply_rotary_emb(x, cos, sin)` — RoPE (identical to Phase 2 lines 350-357)
3. `SwiGLU` class — MLP (identical to Phase 2 lines 435-444)
4. `Block` class — Pre-norm attention + SwiGLU (modified to pass attn_mask)
5. `MultiHeadAttention` class — Modified to accept explicit `attn_mask` parameter

For `MultiHeadAttention.forward(x, cos_sin, attn_mask=None)`:
- Replace `F.scaled_dot_product_attention(q, k, v, is_causal=False)` with:
  `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)`
- The attn_mask is a float tensor where `-inf` means "don't attend" and `0.0` means "attend"
- When attn_mask is None (inference with KV cache), use `is_causal=False`

For KV cache support in `MultiHeadAttention`:
- Add `kv_cache` attribute (initialized to None)
- In forward, when `kv_cache` is not None:
  - Compute Q, K, V for current block only
  - Prepend cached K, V from previous blocks
  - Apply attention (no mask needed — cached tokens are all finalized)
  - After block is finalized, update cache with new K, V
- Add `reset_cache()` and `update_cache(k, v)` methods

For `Block.forward(x, cos_sin, attn_mask=None)`:
- Thread attn_mask through to attention

**Step 2: Write the Model class**

Port from Phase 2 (lines 490-510) with changes:
- `self.token_emb = nn.Embedding(vocab_size, n_embd)` — now 152K vocab
- Tied embeddings: `self.lm_head.weight = self.token_emb.weight`
- RoPE precomputed for `block_size_seq * 2` (to handle `[x_t || x_0]` during training)
- `forward(self, idx, targets=None, mask=None, t=None, attn_mask=None)`:
  - Same as Phase 2 but pass attn_mask through blocks
  - During training: extract logits from first half only `[:, :L]`
  - Loss computation: same ELBO-weighted CE on masked positions

**Step 3: Verify model instantiates and has expected param count**

Run a quick import test to check param count.
Expected: ~100-120M params (78M embeddings + transformer layers at depth=6).

**Step 4: Commit**

Message: "feat: Phase 3 transformer architecture with explicit attention mask support"

---

## Task 3: Staircase Attention Mask

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Write the staircase mask builder**

Add `build_staircase_mask(seq_len, block_size_blk)` function:

Given a doubled sequence `[x_t || x_0]` of length `2 * seq_len`:
- `n = seq_len`
- For each pair of positions `(q, kv)` in `[0, 2n)`:
  - `x0_flag_q = (q >= n)`, `x0_flag_kv = (kv >= n)`
  - `block_q = (q - n) // block_size_blk if x0_flag_q else q // block_size_blk`
  - `block_kv = (kv - n) // block_size_blk if x0_flag_kv else kv // block_size_blk`
  - **M_BD** (block diagonal): `(block_q == block_kv) AND (x0_flag_q == x0_flag_kv)`
  - **M_OBC** (offset block-causal): `(block_q > block_kv) AND x0_flag_kv AND NOT x0_flag_q`
  - **M_BC** (block-causal): `(block_q >= block_kv) AND x0_flag_kv AND x0_flag_q`
  - `allow = M_BD OR M_OBC OR M_BC`

Return a `(2n, 2n)` float tensor: `0.0` where allowed, `-inf` where blocked.

Special case: when `block_size_blk >= seq_len` (full-sequence mode), return a simple mask that is bidirectional for the x_t half attending to x_t, and allows x_t to see x_0 (equivalent to Phase 2 behavior).

**Step 2: Write a visualization test**

Add a small function that prints the mask as a character grid (e.g., `.` for attend, `X` for blocked) to visually verify the staircase pattern. Run it with `seq_len=8, block_size_blk=4` (2 blocks) and verify the pattern matches BD3-LMs paper Figure 2.

**Step 3: Verify mask shape and pattern**

Run a quick test to build the mask and verify shape is `(16, 16)` with the staircase pattern visible.

**Step 4: Commit**

Message: "feat: Phase 3 staircase attention mask for block diffusion"

---

## Task 4: Data Pipeline — Block-Aware Batching

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Write the data pipeline**

Port from Phase 2 (lines 221-329) with these changes:

`get_batch(split)` returns `(x_input, targets, mask, t, attn_mask)`:

1. **Document loading**: Same as Phase 2 — stream FineWeb-Edu, tokenize, truncate to `block_size_seq`, pad with [MASK].
2. **Block-aware timestep sampling**: Sample one `t` per block (not per sequence). For a sequence with `num_blocks = block_size_seq // block_size_blk` blocks, sample `t_blocks ~ U[0,1]` of shape `(B, num_blocks)`, repeat to per-token shape `(B, block_size_seq)`.
3. **Cosine noise masking**: Same formula `mask_prob = 1 - cos^2(t * pi/2)`, but now `t` varies per block within each sequence.
4. **Concatenation**: Build `x_input = cat([x_t, x_0], dim=1)` of shape `(B, 2 * block_size_seq)`.
5. **Position IDs**: Not explicitly needed — RoPE will be sliced to `block_size_seq` and applied to both halves identically.
6. **Staircase mask**: Call `build_staircase_mask(block_size_seq, block_size_blk)` — this is the same for all sequences, so compute once and cache.
7. **Return**: `(x_input, targets, mask, t, attn_mask)` where:
   - `x_input`: `(B, 2*block_size_seq)` — `[x_t || x_0]`
   - `targets`: `(B, block_size_seq)` — original tokens
   - `mask`: `(B, block_size_seq)` — True where noise-masked AND real token
   - `t`: `(B, block_size_seq)` — per-token timestep (for ELBO weighting)
   - `attn_mask`: `(2*block_size_seq, 2*block_size_seq)` — staircase mask

**Step 2: Verify batch shapes**

Run a quick test that calls `get_batch("train")` and prints all tensor shapes.
Expected: x_input is `(32, 1024)`, targets is `(32, 512)`, mask is `(32, 512)`, t is `(32, 512)`, attn_mask is `(1024, 1024)`.

**Step 3: Commit**

Message: "feat: Phase 3 block-aware data pipeline with per-block timesteps"

---

## Task 5: Forward Pass — Training with [x_t || x_0]

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Implement the training forward pass**

Modify `Model.forward()` to handle the doubled input:

1. **Embedding**: `x = self.token_emb(idx)` — shape `(B, 2L, D)` during training
2. **RoPE**: Compute for `L` positions only. Apply to both halves separately:
   - Split: `x_t_emb = x[:, :L]`, `x_0_emb = x[:, L:]`
   - Apply RoPE with same `(cos[:L], sin[:L])` to both
   - Recombine: `x = cat([x_t_emb, x_0_emb], dim=1)`
3. **Transformer blocks**: Pass through with staircase `attn_mask`
4. **Output**: Extract logits from first half only: `logits = self.lm_head(x[:, :L])` — shape `(B, L, V)`
5. **Loss**: Same ELBO-weighted CE on masked positions as Phase 2

Note: RoPE is applied in the attention layer (per Q, K), not at the embedding level. So the split-RoPE logic goes in MultiHeadAttention.forward, not Model.forward. The attention layer should:
- Split Q, K into first-half and second-half
- Apply same RoPE (positions 0..L-1) to both halves
- Recombine before computing attention with the staircase mask

**Step 2: Verify forward pass runs without errors**

Run a quick test: create model, create random input of shape `(2, 1024)`, random targets `(2, 512)`, random mask `(2, 512)`, random t `(2, 512)`, build staircase mask, call forward.
Expected: Returns logits `(2, 512, vocab_size)` and scalar loss.

**Step 3: Commit**

Message: "feat: Phase 3 training forward pass with [x_t || x_0] concatenation"

---

## Task 6: Training Loop

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Write the training loop**

Port from Phase 2 (lines 834-917) with these changes:

1. **Config printing**: Add block_size_blk to the printed config
2. **Weights path**: `weights/block_dllm_d{depth}_b{block_size_blk}.pt`
3. **Optimizer**: Same AdamW with weight decay on 2D+ params
4. **LR schedule**: Same cosine decay with warmup
5. **Training step**: Call `get_batch("train")` then `model(x_input, targets, mask, t, attn_mask)` then backward
6. **Eval**: Same `estimate_loss()` averaging over eval_iters batches
7. **Sampling during training**: Call a simple version of generation (can be just the first block) to show progress

**Step 2: Write the `estimate_loss()` function**

Same as Phase 2 but uses the new get_batch signature.

**Step 3: Verify training runs for a few steps**

Run (locally, quick sanity check only — not real training): `uv run python 03_block_diffusion/block_dllm.py --train --depth 4 --block-size 4`
Expected: Prints config, model size, starts training loop, shows loss decreasing. Kill after ~10 steps.

**Step 4: Commit**

Message: "feat: Phase 3 training loop with block diffusion"

---

## Task 7: Inference — Block-by-Block Generation with KV Cache

**Files:**
- Modify: `03_block_diffusion/block_dllm.py`

**Step 1: Write the generation function**

`generate(model, max_new_tokens, prompt, denoise_steps, temp, top_k, confidence_threshold)`:

1. **Initialization**:
   - Encode prompt to token IDs
   - Reset KV cache in all attention layers
   - Determine how many blocks to generate: `num_blocks = ceil(max_new_tokens / block_size_blk)`

2. **Prompt handling**:
   - Fill initial blocks with prompt tokens
   - For each full prompt block: run a forward pass to populate KV cache, no denoising needed
   - The last (partial) prompt block may need denoising for remaining positions

3. **Block-by-block generation loop** — for each new block:
   a. Initialize block with all [MASK] tokens: `block = [MASK] * block_size_blk`
   b. Run `denoise_steps` iterations:
      - Forward pass: attend to current block (bidirectional) + KV cache (previous blocks)
      - Use Phase 2's confidence-based unmasking: unmask positions above `confidence_threshold`, force-unmask best if none qualify
   c. After block is fully denoised:
      - Update KV cache with this block's K, V
      - Check for EOS token — if found, stop generation
      - Append block tokens to output

4. **Return**: Decoded text

**Step 2: Implement KV cache integration in MultiHeadAttention**

During generation:
- `forward(x_block, cos_sin, attn_mask=None)`:
  - Q, K, V computed from current block only
  - K_full = cat([cached_K, K_block]), V_full = cat([cached_V, V_block])
  - Attention: Q attends to K_full, V_full (no mask — all cached tokens are finalized)
  - RoPE: position offset = len(cached_tokens), so current block gets positions after the cache

**Step 3: Test generation**

Run (quick test with random weights): `uv run python 03_block_diffusion/block_dllm.py --depth 4 --block-size 4 --prompt "Hello" --max-tokens 32`
Expected: Generates garbage (untrained) but runs without errors. Prints generation stats.

**Step 4: Commit**

Message: "feat: Phase 3 block-by-block generation with KV caching"

---

## Task 8: Kaggle Training Notebook

**Files:**
- Create: `kaggle/train_phase3.ipynb`

**Step 1: Create the Kaggle notebook**

Cells:
1. Install deps: `!pip install -q torch datasets transformers`
2. Clone repo: `!git clone https://github.com/hahuyhoang411/Open-dLLM.git && cd Open-dLLM`
3. Check GPU (P100 16GB)
4. Train with block_size=4: `!python 03_block_diffusion/block_dllm.py --train --depth 8 --block-size 4`
5. Generate text: `!python 03_block_diffusion/block_dllm.py --depth 8 --block-size 4 --prompt "The meaning of life is"`
6. Save weights to `/kaggle/working/`

**Step 2: Commit**

Message: "feat: Kaggle Phase 3 training notebook"

---

## Task 9: DCLM CORE Evaluation Support

**Files:**
- Modify: `eval/base_eval.py`

**Step 1: Add block_dllm model loading**

Add `load_block_dllm_model(depth, block_size, weights_path, device)` to `base_eval.py`:
- Similar to `load_dllm_model` but imports from `03_block_diffusion/block_dllm`
- Patches sys.argv for the module-level parse_args
- Returns `(model_fn, tokenize_fn, mask_token_id, max_seq_len)`

Add CLI arg: `--model block_dllm` with `--block-size` parameter.

**Step 2: Test eval integration**

Run: `uv run python eval/base_eval.py --model block_dllm --depth 8 --block-size 4 --max-per-task 5`
Expected: Loads model, runs eval (scores will be near-random without training).

**Step 3: Commit**

Message: "feat: DCLM CORE eval support for block diffusion model"

---

## Task 10: Documentation and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update pyproject.toml**

Add `transformers` to the phase2 or create a phase3 deps group. Since we already have `transformers` in the eval group, and phase3 needs it for the tokenizer, either:
- Add a `phase3` group: `phase3 = ["datasets>=2.0.0", "transformers>=4.30.0"]`
- Or ensure phase2 + transformers covers it

**Step 2: Update README.md**

- Change Phase 3 status from "Coming soon" to "Done"
- Add Phase 3 quick start commands
- Update project structure tree

**Step 3: Update CLAUDE.md**

Add Phase 3 gotchas:
- `03_block_diffusion/block_dllm.py` has module-level `parse_args()` — same gotcha as Phase 2
- Qwen 3 tokenizer downloaded on first run (~500MB)
- Weights at `03_block_diffusion/weights/block_dllm_d{depth}_b{block_size}.pt`

**Step 4: Commit**

Message: "docs: Phase 3 dependencies, README, and CLAUDE.md updates"

---

## Task 11: End-to-End Validation on Kaggle

**Step 1: Push to GitHub**

Push all Phase 3 commits to origin.

**Step 2: Run Kaggle training**

Push notebook to Kaggle, run on P100. Verify:
- Training completes without OOM
- Loss decreases over training
- Generation produces coherent-ish text

**Step 3: Run DCLM CORE eval on Kaggle**

Evaluate the trained model and compare with Phase 2 and GPT-2.

**Step 4: Final commit**

Message: "feat: Phase 3 block diffusion validated end-to-end"

---

## Summary

| Task | What | Files | Key Change |
|------|------|-------|------------|
| 1 | CLI + tokenizer | block_dllm.py (new) | Qwen 3 tokenizer, --block-size flag |
| 2 | Model architecture | block_dllm.py | Explicit attn_mask in attention, KV cache support |
| 3 | Staircase mask | block_dllm.py | build_staircase_mask() — the core BD3-LMs innovation |
| 4 | Data pipeline | block_dllm.py | Per-block timesteps, [x_t \|\| x_0] concatenation |
| 5 | Training forward | block_dllm.py | Split RoPE, extract first-half logits |
| 6 | Training loop | block_dllm.py | Same as Phase 2, new batch format |
| 7 | Generation | block_dllm.py | Block-by-block with KV cache + confidence decoding |
| 8 | Kaggle notebook | train_phase3.ipynb | Training on P100 |
| 9 | Eval support | eval/base_eval.py | load_block_dllm_model() |
| 10 | Docs + deps | pyproject.toml, README, CLAUDE.md | Phase 3 updates |
| 11 | Validation | Kaggle | End-to-end training + eval |
