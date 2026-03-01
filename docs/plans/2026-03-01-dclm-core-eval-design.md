# DCLM CORE Evaluation — Design Document

**Date**: 2026-03-01
**Status**: Approved

## Objective

Add a lightweight, dependency-free DCLM CORE evaluation to Open-dLLM that works for both our masked diffusion models and standard HuggingFace AR models, enabling direct dLLM vs GPT comparison on the same 22-task benchmark.

## Background

DCLM CORE is a curated set of 22 low-variance tasks from the DCLM benchmark suite. The score is "centered accuracy": per-task accuracy rescaled so 0 = random guessing, 1 = perfect, then averaged. GPT-2 (124M) scores ~0.257.

nanochat implements this as a single custom file (~262 lines) with no lm-evaluation-harness dependency. We adapt this pattern for dLLMs by adding Monte Carlo likelihood estimation.

## The dLLM Evaluation Problem

AR models compute `log p(answer | context)` via next-token logits — straightforward. dLLMs predict masked positions in parallel with no sequential factorization, so the standard loglikelihood trick doesn't work.

**Solution: Monte Carlo likelihood estimation** (from LLaDA, MDLM):

```
1. Concatenate [context_tokens | answer_tokens]
2. For mc_num samples (default 128):
   a. Sample t ~ U[0,1]
   b. Mask floor(t * len(answer)) random answer tokens with [MASK]
   c. Forward pass → logits for masked positions
   d. CE on masked positions, importance-weighted by 1/t
3. Return -mean(weighted_losses) as log-likelihood estimate
```

For multiple-choice: pick answer with highest estimated log-likelihood.
For language modeling (LAMBADA): check if model's argmax at each masked answer position matches the target.

## Architecture

### File Structure

```
eval/
├── core_eval.py      # Evaluation engine (~300 lines)
│                      # - 3 task types: multiple_choice, schema, language_modeling
│                      # - Prompt rendering with few-shot templates
│                      # - Scoring: min-loss selection (MC/schema), argmax match (LM)
│                      # - Centered accuracy computation
├── base_eval.py      # CLI entry point + model loading (~200 lines)
│                      # - --model dllm: loads nano_dllm, MC likelihood
│                      # - --hf-model: loads HuggingFace AR model, standard logits
│                      # - Downloads eval bundle from S3
│                      # - Prints results table + CORE score
└── README.md         # How to run, interpret results
```

Plus: `kaggle/eval_phase2.ipynb` — Kaggle notebook for running eval on GPU.

### Model Interface

Both backends expose the same interface to `core_eval.py`:

```python
class ModelAdapter:
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits for all positions. Shape: (1, seq_len, vocab_size)"""

    def get_log_likelihood(self, context_ids, answer_ids) -> float:
        """Return log p(answer | context). AR: chain rule. dLLM: MC estimation."""

    @property
    def max_seq_len(self) -> int | None:
        """Context window limit (512 for nano_dllm, 1024 for GPT-2, None for unlimited)"""
```

### AR Adapter (HuggingFace)

```python
class ARAdapter(ModelAdapter):
    def get_log_likelihood(self, context_ids, answer_ids):
        full = concat(context_ids, answer_ids)
        logits = model(full).logits
        # Standard: sum log-probs of answer tokens given context
        return sum(log_softmax(logits)[answer_positions])
```

### dLLM Adapter (nano_dllm)

```python
class DLLMAdapter(ModelAdapter):
    def get_log_likelihood(self, context_ids, answer_ids, mc_num=128):
        full = concat(context_ids, answer_ids)
        losses = []
        for _ in range(mc_num):
            t = random.uniform(0, 1)
            # Mask random subset of answer tokens
            masked = apply_mask(full, answer_positions, t)
            logits = model(masked)
            ce = cross_entropy(logits[masked_positions], full[masked_positions])
            losses.append(ce / max(t, 1e-4))  # importance weight
        return -mean(losses)
```

### DCLM CORE Tasks (22)

| Category | Tasks |
|----------|-------|
| Commonsense | hellaswag, hellaswag_zeroshot, copa, commonsense_qa, piqa, openbook_qa, winograd, winogrande |
| Knowledge | jeopardy, bigbench_qa_wikidata, arc_easy, arc_challenge |
| Language | lambada_openai, bigbench_language_identification |
| Symbolic | bigbench_dyck_languages, bigbench_cs_algorithms, bigbench_operators, bigbench_repeat_copy_logic |
| Reasoning | agi_eval_lsat_ar |
| Reading | squad, coqa, boolq |

### Centered Accuracy Formula

```
centered = (raw_accuracy - random_baseline) / (1.0 - random_baseline)
CORE = mean(centered for all 22 tasks)
```

Random baselines: 25% for 4-way MC, 50% for binary, 0% for open-ended generation tasks.

### Eval Data

Download nanochat's eval bundle from S3:
- URL: `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`
- Cache: `~/.cache/open-dllm/eval_bundle/`
- Contents: `core.yaml` (task metadata), `eval_data/` (task datasets), `eval_meta_data.csv` (random baselines)

## CLI

```bash
# Eval our dLLM
python eval/base_eval.py --model dllm --depth 6 --weights 02_nano_dllm/weights/nano_dllm_d6.pt

# Eval GPT-2 for comparison
python eval/base_eval.py --hf-model gpt2

# Quick eval (limit examples per task)
python eval/base_eval.py --model dllm --depth 6 --max-per-task 100

# Specific eval types
python eval/base_eval.py --model dllm --depth 6 --eval core
```

## Dependencies

- `torch` (already have)
- `jinja2` (for prompt templates — lightweight)
- `transformers` (only for --hf-model, optional)
- `requests` (for downloading eval bundle — stdlib-adjacent)

## Scope

**In scope**: CORE score computation, MC likelihood for dLLMs, AR model comparison, Kaggle eval notebook
**Out of scope**: MMLU (tracked separately in DCLM), chat evals, bits-per-byte, distributed evaluation, wandb integration

## Success Criteria

1. `python eval/base_eval.py --hf-model gpt2` produces CORE score close to nanochat's 0.257
2. `python eval/base_eval.py --model dllm --depth 6` produces a meaningful (likely lower) CORE score
3. Results table clearly shows per-task comparison
4. Kaggle notebook runs eval end-to-end on GPU

## References

- [nanochat](https://github.com/karpathy/nanochat) — `core_eval.py`, `base_eval.py`
- [DCLM](https://github.com/datacomplm/DCLM) — `eval/mmlu_and_lowvar.yaml`, `eval/eval_meta_data.csv`
- [LLaDA](https://github.com/ML-GSAI/LLaDA) — `get_log_likelihood.py` (MC estimation for dLLMs)
- [dLLM framework](https://github.com/ZHZisZZ/dllm) — `MDLMEvalHarness` (harness adapter pattern)
