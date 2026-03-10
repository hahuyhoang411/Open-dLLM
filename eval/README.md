# Evaluation: DCLM CORE Benchmark for Diffusion LLMs

## The Problem

Evaluating a dLLM is fundamentally different from evaluating an autoregressive LM. AR models compute per-token likelihood in a single forward pass — the model sees all previous tokens and predicts the next one. dLLMs don't have a "next token." They see a partially masked sequence and predict all masked positions simultaneously. Each forward pass sees a *different* random mask pattern, so a single pass gives a noisy likelihood estimate. You need many passes to get a stable one.

## How AR Evaluation Works

```
Question: "The capital of France is ___"
Choices: (A) Paris  (B) London  (C) Berlin

AR model: One forward pass -> compute P(choice | context) -> pick highest
```

The model processes the context left-to-right. At each answer token position, it has seen all prior tokens. One pass, one score per choice. Done.

## How dLLM Evaluation Works

```
Question: "The capital of France is ___"
Choices: (A) Paris  (B) London  (C) Berlin

dLLM model:
  For each choice:
    Repeat N times (default N=64):
      1. Sample a masking rate t ~ U(0, 1]
      2. Randomly mask ~t fraction of the answer tokens
      3. Forward pass -> cross-entropy at masked positions only
      4. Weight by 1/t (ELBO importance weight)
    Average weighted CE across N samples
  Pick choice with lowest average CE
```

The N=64 Monte Carlo samples are needed because:
- Each mask pattern reveals different positions to the model
- A single pattern might mask exactly the discriminative tokens (e.g., "Par" in "Paris")
- Averaging over many patterns gives a stable ELBO likelihood estimate
- The 1/t weighting corrects for the linear noise schedule (mask more -> observe less -> weight less)

This uses antithetic sampling: `t = (u0 + i/N) % 1` with a single random offset `u0`, giving uniform coverage of the noise schedule with lower variance than i.i.d. sampling.

## Centered Accuracy

Raw accuracy is misleading when tasks have different numbers of choices (2-choice vs 5-choice). Centered accuracy normalizes this:

```
centered_accuracy = (accuracy - random_baseline) / (1 - random_baseline)
```

- 0.0 = random guessing
- 1.0 = perfect
- Negative = worse than random

The CORE score is the mean centered accuracy across all 22 tasks. GPT-2 (124M) scores ~0.257.

## The 22 Tasks

| Category | Tasks | Type |
|---|---|---|
| **Language Understanding** | hellaswag (0-shot, 10-shot), lambada_openai, winograd, winogrande, bigbench_language_identification | MC, LM, Schema |
| **World Knowledge** | jeopardy, bigbench_qa_wikidata, arc_easy, arc_challenge | LM, MC |
| **Commonsense Reasoning** | copa, commonsense_qa, piqa, openbook_qa | MC |
| **Symbolic Problem Solving** | bigbench_dyck_languages, agi_eval_lsat_ar, bigbench_cs_algorithms, bigbench_operators, bigbench_repeat_copy_logic | LM, MC |
| **Reading Comprehension** | squad, coqa, boolq | LM, MC |

Three task types, each scored differently:
- **Multiple choice (MC):** Same context, different continuations. Pick lowest-loss choice.
- **Schema:** Different contexts, same continuation (Winograd-style). Pick lowest-loss context.
- **Language modeling (LM):** Predict the final word(s). Exact match on continuation tokens.

## Usage

```bash
pip install -e ".[eval]"

# Evaluate a dLLM (Phase 2, depth=6)
python eval/base_eval.py --model dllm --depth 6

# Phase 5 model
python eval/base_eval.py --model phase5 --weights path/to/ckpt.pt

# Compare with AR baseline
python eval/base_eval.py --hf-model gpt2

# Quick eval (fewer examples per task)
python eval/base_eval.py --model dllm --depth 6 --max-per-task 100

# More MC samples (slower, lower variance)
python eval/base_eval.py --model dllm --depth 6 --mc-num 128
```

## Files

- `base_eval.py` -- CLI entry point, model loading (dllm/block_dllm/modern_dllm/phase5/HF), result formatting
- `core_eval.py` -- Evaluation engine: jinja2 prompt rendering, MC sampling, AR/dLLM forward passes, scoring

## Technical Notes

- Eval bundle cached at `~/.cache/open-dllm/eval_bundle/`
- YAML key is `icl_tasks` (list), CSV column is `"Eval Task"` for random baselines
- MC sampling uses antithetic masks (stratified t values), not i.i.d. random
- Few-shot examples are sampled deterministically per item (seed = 1234 + index)
- MPS (Apple Silicon): `torch.mps.empty_cache()` called between tasks to prevent OOM
- Sequences exceeding `max_seq_len` are left-cropped (preserves the answer region)
