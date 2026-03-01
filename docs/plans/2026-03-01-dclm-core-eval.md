# DCLM CORE Evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight DCLM CORE benchmark (22 tasks, centered accuracy) that evaluates both our masked diffusion models and HuggingFace AR models, enabling direct dLLM vs GPT comparison.

**Architecture:** Port nanochat's dependency-free eval engine (`core_eval.py`) and adapt it for dLLMs by adding Monte Carlo likelihood estimation. Two files in `eval/` — the evaluation engine and the CLI entry point. Eval data downloaded from nanochat's S3 bundle.

**Tech Stack:** Python 3.10+, PyTorch, jinja2 (prompt templates), transformers (optional, for HF models)

---

## Task 1: Create eval/core_eval.py — Prompt Rendering and Tokenization

**Files:**
- Create: `eval/core_eval.py`

**Step 1: Write the prompt rendering and sequence batching utilities**

Port these functions from nanochat's `core_eval.py` (MIT license):

1. `render_prompts_mc(item, continuation_delimiter, fewshot_examples)` — Render prompts for multiple-choice tasks. Uses jinja2 template: fewshot examples → query → each choice option. Returns list of prompt strings (one per choice).

2. `render_prompts_schema(item, continuation_delimiter, fewshot_examples)` — Render prompts for schema tasks (Winograd-style). Different contexts, same continuation. Returns list of prompt strings (one per context option).

3. `render_prompts_lm(item, continuation_delimiter, fewshot_examples)` — Render prompts for language modeling tasks (LAMBADA-style). Returns two prompts: context-only and context+continuation.

4. `find_common_length(token_sequences, direction)` — Find length of common prefix (direction='left') or suffix (direction='right') across token sequences. Used to identify where the "answer" region starts/ends.

5. `stack_sequences(tokens, pad_token_id)` — Pad token sequences to equal length, stack into tensor.

6. `batch_sequences_mc(tokenize_fn, prompts)` — Tokenize MC prompts, find common prefix (answer starts after it). Returns tokens, start_indices, end_indices.

7. `batch_sequences_schema(tokenize_fn, prompts)` — Tokenize schema prompts, find common suffix (continuation). Returns tokens, start_indices, end_indices.

8. `batch_sequences_lm(tokenize_fn, prompts)` — Tokenize LM prompts. Returns tokens (the context+continuation), start/end for continuation region.

**Key difference from nanochat:** Our tokenize_fn interface is simpler. nanochat uses a custom `HuggingFaceTokenizer` class with `prepend=bos_token_id`. We'll accept a callable `tokenize_fn(text) -> list[int]` that both our BPE tokenizer and HF tokenizers can satisfy. No BOS token for our dLLM (we don't use one), optional for HF models.

The jinja2 templates are identical to nanochat:

For `multiple_choice`:
```
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}
```

For `schema`:
```
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}
```

For `language_modeling`:
```
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}
```

**Step 2: Verify the module imports cleanly**

Run: `python -c "import eval.core_eval; print('OK')"`
Expected: Prints "OK" with no import errors.

**Step 3: Commit**

Message: "feat: eval core_eval.py prompt rendering and tokenization utilities"

---

## Task 2: Add eval/core_eval.py — AR Forward and Scoring

**Files:**
- Modify: `eval/core_eval.py`

**Step 1: Add the AR (autoregressive) forward and scoring functions**

Add these functions:

1. `forward_model_ar(model_fn, input_ids)` — Standard AR forward pass:
   - Call `model_fn(input_ids)` → logits `(B, T, V)`
   - Compute per-token CE: `logits[t]` predicts `input_ids[t+1]` (shift by 1)
   - Return `(losses, predictions)` where losses is `(B, T)` and predictions is argmax `(B, T)`
   - Set `losses[:, -1] = nan` (last position has no target)

2. `forward_model_dllm(model_fn, input_ids, answer_start_idxs, answer_end_idxs, mask_token_id, mc_num=64)` — MC likelihood for dLLM:
   - For each sequence in the batch, identify the answer region `[start:end]`
   - Run `mc_num` iterations:
     a. Sample `t ~ U[0,1]`
     b. Compute `mask_prob = 1 - cos²(t · π/2)` (cosine schedule)
     c. Mask random subset of answer tokens (Bernoulli with `mask_prob`)
     d. If no tokens masked, force-mask at least one random answer token
     e. Forward pass: `model_fn(masked_input)` → logits
     f. CE on masked answer positions, importance-weighted by `1/max(t, 1e-4)`
   - Average losses across MC samples
   - Return `(mean_losses_per_position, predictions_from_fully_masked)` matching AR output shape
   - `predictions`: run one forward pass with ALL answer tokens masked, return argmax (for LM tasks)

3. `evaluate_example(idx, model_fn, tokenize_fn, data, device, task_meta, mode='ar', mask_token_id=None, mc_num=64, max_seq_len=None)` — Evaluate a single example:
   - Render prompts based on task type
   - Batch and tokenize sequences
   - Truncate to `max_seq_len` if needed (crop from left, like nanochat)
   - If `mode='ar'`: call `forward_model_ar`, score by min mean loss (MC/schema) or exact argmax match (LM)
   - If `mode='dllm'`: call `forward_model_dllm`, score by min mean MC loss (MC/schema) or argmax match on fully-masked forward (LM)
   - Return `is_correct` boolean

4. `evaluate_task(model_fn, tokenize_fn, data, device, task_meta, mode='ar', mask_token_id=None, mc_num=64, max_seq_len=None)` — Evaluate all examples in a task:
   - Loop through data, call `evaluate_example` for each
   - Return mean accuracy

**Key design for dLLM MC scoring:**

For `multiple_choice` and `schema` tasks:
- Each choice gets its own MC loss estimate
- The choice with lowest mean MC loss wins (same as AR but with MC loss instead of AR next-token loss)

For `language_modeling` tasks:
- Mask ALL continuation tokens
- Single forward pass
- Check if argmax at each position matches target token
- (No MC needed — one-shot prediction from fully masked)

**Step 2: Verify with a simple test**

Run: `python -c "from eval.core_eval import evaluate_example; print('OK')"`
Expected: Prints "OK"

**Step 3: Commit**

Message: "feat: eval core_eval.py AR and dLLM forward pass with MC likelihood"

---

## Task 3: Create eval/base_eval.py — CLI, Model Loading, and Eval Bundle

**Files:**
- Create: `eval/base_eval.py`
- Create: `eval/__init__.py` (empty)

**Step 1: Write the CLI entry point and model loading**

Structure of `eval/base_eval.py`:

1. **Imports**: os, csv, time, json, yaml, zipfile, argparse, torch, urllib.request

2. **Constants**:
   - `EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"`
   - `CACHE_DIR = os.path.expanduser("~/.cache/open-dllm")`

3. **`download_eval_bundle()`** — Download and extract eval bundle:
   - Check if `~/.cache/open-dllm/eval_bundle/` exists
   - If not: download zip from S3 via `urllib.request.urlretrieve`, extract, place in cache dir
   - Return path to eval_bundle directory

4. **`load_dllm_model(depth, weights_path, device)`** — Load our nano_dllm model:
   - Import from `02_nano_dllm/nano_dllm.py` by adding the script's parent dir to sys.path
   - Set the module-level globals (`depth`, `n_layer`, `n_embd`, etc.) by calling the module's config setup
   - Instantiate `Model()`, load weights, set to eval mode
   - Return a tuple: `(model_fn, tokenize_fn, mask_token_id, max_seq_len=512)`
   - `model_fn(input_ids)` should return `(B, T, V)` logits
   - `tokenize_fn(text)` should return `list[int]`

5. **`load_hf_model(hf_path, device)`** — Load a HuggingFace AR model:
   - Use `transformers.AutoModelForCausalLM.from_pretrained(hf_path)`
   - Use `transformers.AutoTokenizer.from_pretrained(hf_path)`
   - Wrap model to return `(B, T, V)` logits
   - `tokenize_fn(text)` wraps the HF tokenizer `.encode(text)`
   - Return: `(model_fn, tokenize_fn, None, max_seq_len)` (no mask_token_id for AR)
   - `max_seq_len = 1024` for GPT-2, `None` otherwise

6. **`evaluate_core(model_fn, tokenize_fn, device, mode, mask_token_id, mc_num, max_seq_len, max_per_task)`** — Run all CORE tasks:
   - Load `core.yaml` for task list
   - Load `eval_meta_data.csv` for random baselines
   - For each task: load JSONL data, shuffle with seed 1337, optionally limit to `max_per_task`
   - Call `evaluate_task()` from core_eval.py
   - Compute centered accuracy: `(acc - 0.01*baseline) / (1 - 0.01*baseline)`
   - Print results table
   - Print final CORE score

7. **`main()`** — CLI argument parsing:
   - `--model dllm` — evaluate our dLLM model
   - `--depth N` — dLLM depth (default 6)
   - `--weights PATH` — path to weights file (default: auto-detect from depth)
   - `--hf-model NAME` — evaluate a HuggingFace model (e.g. `gpt2`)
   - `--mc-num N` — MC samples for dLLM (default 64)
   - `--max-per-task N` — limit examples per task for quick eval (default -1 = all)
   - `--device` — device override (default: auto-detect)

**Step 2: Create `eval/__init__.py`**

Empty file, just needed for imports.

**Step 3: Verify the CLI parses**

Run: `python eval/base_eval.py --help`
Expected: Shows help with all arguments.

**Step 4: Commit**

Message: "feat: eval base_eval.py CLI with dLLM and HF model loading"

---

## Task 4: Integration Testing — Eval Bundle Download and AR Eval

**Files:**
- Modify: `eval/base_eval.py` (if needed for bug fixes)

**Step 1: Test eval bundle download**

Run: `python -c "from eval.base_eval import download_eval_bundle; p = download_eval_bundle(); print(p)"`
Expected: Downloads ~50MB zip, extracts to `~/.cache/open-dllm/eval_bundle/`, prints path.

**Step 2: Run quick GPT-2 eval (limited examples)**

Run: `pip install transformers && python eval/base_eval.py --hf-model gpt2 --max-per-task 10`

Expected: Loads GPT-2, runs 22 tasks with 10 examples each, prints results table with CORE score. Should take ~2-5 minutes on CPU. The quick score won't match GPT-2's full 0.257 but should be in the right ballpark.

**Step 3: Fix any bugs found during integration**

Debug any issues with tokenization, data loading, or scoring.

**Step 4: Commit**

Message: "feat: eval integration tested with GPT-2 quick eval"

---

## Task 5: dLLM Model Adapter and MC Evaluation

**Files:**
- Modify: `eval/base_eval.py` (refine dLLM loading if needed)
- Modify: `eval/core_eval.py` (refine MC forward if needed)

**Step 1: Test dLLM model loading**

Run: `python -c "from eval.base_eval import load_dllm_model; m = load_dllm_model(6, '02_nano_dllm/weights/nano_dllm_d6.pt', 'cpu'); print('OK')"`

Expected: Loads model successfully, prints "OK".

Note: If we don't have local weights (trained on Kaggle), download from Kaggle output:
```bash
export KAGGLE_API_TOKEN=... && uv run kaggle kernels output hhoang41/open-dllm-phase2-training -p ./kaggle/output
cp kaggle/output/weights/nano_dllm_d6.pt 02_nano_dllm/weights/
```

**Step 2: Run quick dLLM eval**

Run: `python eval/base_eval.py --model dllm --depth 6 --max-per-task 10 --mc-num 16`

Expected: Loads nano_dllm_d6, runs 22 tasks with MC likelihood, prints results table with CORE score. Score will be very low for a 35M model with only 20K steps of training — expected.

**Step 3: Debug and fix any issues**

Common issues to watch for:
- Tokenizer mismatch: our BPE tokenizer doesn't have a BOS token. Make sure tokenize_fn doesn't prepend one.
- Sequence length: our model's block_size is 512. Some fewshot prompts may exceed this — verify left-cropping works.
- MC masking: ensure we only mask answer tokens, never context tokens.
- Loss computation: dLLM forward returns logits for ALL positions, but we only want loss on masked answer positions.

**Step 4: Commit**

Message: "feat: dLLM MC evaluation working end-to-end"

---

## Task 6: Kaggle Evaluation Notebook

**Files:**
- Create: `kaggle/eval_phase2.ipynb`
- Modify: `kaggle/kernel-metadata.json` (if we want a separate kernel, or add eval cells to existing)

**Step 1: Create Kaggle eval notebook**

Cells:
1. Install deps: `!pip install -q torch jinja2 pyyaml transformers`
2. Clone repo: `!git clone https://github.com/hahuyhoang411/Open-dLLM.git && cd Open-dLLM`
3. Check GPU
4. Download weights from Kaggle dataset or train (if not available):
   ```python
   import os
   weights = '02_nano_dllm/weights/nano_dllm_d6.pt'
   if not os.path.exists(weights):
       print("Weights not found. Training first...")
       !python 02_nano_dllm/nano_dllm.py --train --depth 6
   ```
5. Run dLLM eval: `!python eval/base_eval.py --model dllm --depth 6`
6. Run GPT-2 eval for comparison: `!python eval/base_eval.py --hf-model gpt2`
7. Save results to `/kaggle/working/`

**Step 2: Commit**

Message: "feat: Kaggle evaluation notebook for DCLM CORE benchmark"

---

## Task 7: Update Dependencies and Documentation

**Files:**
- Modify: `pyproject.toml` (add jinja2, pyyaml to eval deps)
- Modify: `README.md` (add eval section)

**Step 1: Update pyproject.toml**

Add an `eval` optional dependency group:
```toml
[project.optional-dependencies]
phase2 = ["datasets>=2.0.0", "tokenizers>=0.15.0"]
eval = ["jinja2>=3.0.0", "pyyaml>=6.0.0", "transformers>=4.30.0"]
```

**Step 2: Update root README**

Add a brief "Evaluation" section after Quick Start:
```markdown
## Evaluation

Run the DCLM CORE benchmark (22 tasks, centered accuracy):

    # Evaluate our dLLM
    pip install -e ".[eval]"
    python eval/base_eval.py --model dllm --depth 6

    # Compare with GPT-2
    python eval/base_eval.py --hf-model gpt2

    # Quick eval (limited examples)
    python eval/base_eval.py --model dllm --depth 6 --max-per-task 100
```

Update project structure tree to include `eval/` directory.

**Step 3: Commit**

Message: "docs: add DCLM CORE eval section to README and pyproject.toml"

---

## Task 8: End-to-End Validation

**Step 1: Full GPT-2 eval (or a substantial subset)**

Run: `python eval/base_eval.py --hf-model gpt2 --max-per-task 500`

Expected: CORE score should be in the range 0.20-0.30 (full dataset gives 0.257). Verify individual task accuracies are reasonable.

**Step 2: Full dLLM eval**

Run: `python eval/base_eval.py --model dllm --depth 6 --mc-num 64`

Expected: CORE score likely in range -0.05 to 0.10 for our small 35M model. The score may be near or below 0 — that's expected and educational ("this is why you need more parameters and training").

**Step 3: Push to Kaggle and verify**

Push the eval notebook to Kaggle, verify it runs on GPU.

**Step 4: Final commit**

Message: "feat: DCLM CORE evaluation validated end-to-end"

---

## Summary

| Task | What | Files | Est. Effort |
|------|------|-------|-------------|
| 1 | Prompt rendering + tokenization | eval/core_eval.py | Medium |
| 2 | AR and dLLM forward + scoring | eval/core_eval.py | Medium |
| 3 | CLI, model loading, eval bundle | eval/base_eval.py | Medium |
| 4 | Integration test with GPT-2 | eval/base_eval.py | Medium |
| 5 | dLLM adapter + MC eval | eval/*.py | Medium |
| 6 | Kaggle eval notebook | kaggle/eval_phase2.ipynb | Small |
| 7 | Dependencies + docs | pyproject.toml, README.md | Small |
| 8 | End-to-end validation | Run evals | Medium + compute |
