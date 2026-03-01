# Phase 2: nano_dllm — Design Document

**Date**: 2026-03-01
**Status**: Approved

## Objective

Build a single-file BPE diffusion language model trained on FineWeb-Edu with proper MDLM cosine noise schedule and ELBO-weighted loss. Configurable via `--depth` flag (6-24 layers). Generates coherent English text.

## What Phase 2 Adds Over Phase 1

| # | Phase 1 | Phase 2 | Why It Matters |
|---|---------|---------|----------------|
| 1 | Char tokenizer (66 tokens) | BPE tokenizer (~32K vocab) | Real-world tokenization, handles open vocabulary |
| 2 | Uniform masking `t ~ U[0,1]` | Cosine schedule `alpha_t = cos²(t·pi/2)` | Fisher-Rao optimal, spends budget on informative masking ratios |
| 3 | Unweighted CE loss | ELBO-weighted `(1/t) · CE` | Proper continuous-time ELBO from MDLM theory |
| 4 | Fixed 6 layers | `--depth` configurable (6-24) | Compute-optimal scaling, one codebase for quick/serious training |
| 5 | Local Tiny Shakespeare (1MB) | FineWeb-Edu streaming (multi-TB) | Real web-scale data, proper educational content |
| 6 | ReLU² activation | SwiGLU activation | Matches LLaDA/LLaMA architecture, better performance |

The 5 core DIFF markers (mask token, bidirectional attention, random masking, masked loss, confidence-based decoding) remain identical to Phase 1.

## Architecture

### Model

Standard transformer (LLaMA-style) with bidirectional attention:
- **RMSNorm** (functional, no learnable params)
- **RoPE** (rotary position embeddings)
- **SwiGLU** activation (upgrade from Phase 1's ReLU²)
- **Bidirectional attention** (`is_causal=False`)
- **No timestep conditioning** (model infers t from mask count — RADD, ICLR 2025)

### Depth Parametrization

Single `--depth` argument controls everything:
- `model_dim = depth * 64`
- `n_heads = depth`
- `head_dim = 64` (fixed)
- `n_layers = depth`
- `mlp_dim = model_dim * 8/3` (SwiGLU standard)

| `--depth` | Layers | Dim | Heads | ~Params | T4 Time (~100M tokens) |
|-----------|--------|-----|-------|---------|------------------------|
| 6 | 6 | 384 | 6 | ~30M | ~1 hour |
| 12 | 12 | 768 | 12 | ~150M | ~4 hours |
| 24 | 24 | 1536 | 24 | ~850M | Needs A100+ |

### Tokenizer

- Train BPE on ~50MB FineWeb-Edu sample using HuggingFace `tokenizers` library
- ~32K vocab with `[MASK]` as special token
- Separate `train_tokenizer.py` script (run once, save `tokenizer.json`)
- Main script loads saved tokenizer

### Cosine Noise Schedule

```
alpha_t = cos²(t · pi / 2)
mask_prob = 1 - alpha_t
```

- At t=0: alpha=1, mask_prob=0 (clean)
- At t=0.5: alpha≈0.5, mask_prob≈0.5
- At t=1: alpha≈0, mask_prob≈1 (fully masked)
- S-curve spends more budget at mid-range masking (richest signal)

### ELBO-Weighted Loss

```
L = (1/t) · CE[logits_masked, targets_masked]
```

- `t` sampled from `U[0,1]`, clamped to `[eps, 1]` for numerical stability
- Low t (less masking) → higher weight (harder predictions are more valuable)
- This is the proper continuous-time ELBO from MDLM (NeurIPS 2024)

## Data Pipeline

1. **Streaming**: `load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)`
2. **Tokenize**: BPE encode on-the-fly per batch
3. **Batch**: Pack sequences to `block_size` (default 512), pad shorter ones
4. **Mask**: Sample `t ~ U[0,1]` per sequence, compute `mask_prob = 1 - cos²(t·pi/2)`, apply per-token

No disk storage needed — data streams directly from HuggingFace.

## Training

- **Optimizer**: AdamW with learning rate warmup + cosine decay
- **Default hyperparameters**: lr=1e-3, warmup=1000 steps, weight_decay=0.1
- **Eval**: Every N steps, compute validation loss on held-out streaming data, generate sample
- **Checkpoints**: Save to `weights/nano_dllm_d{depth}.pt`
- **CLI**: `python nano_dllm.py --train --depth 6`

## Generation / Inference

Same confidence-based parallel decoding as Phase 1:
- Start with all-mask sequence (or prompt + masks)
- Iteratively unmask highest-confidence positions
- Top-k sampling with temperature
- Block-by-block for long sequences
- BPE decode at the end

Support text prompts: `python nano_dllm.py --prompt "The meaning of life is"`

## File Structure

```
02_nano_dllm/
├── README.md              # Theory: cosine schedule, ELBO, BPE, scaling
├── nano_dllm.py           # Single file (~800-1000 lines)
├── train_tokenizer.py     # BPE tokenizer training (~50 lines)
├── tokenizer.json         # Trained BPE tokenizer (committed)
└── weights/               # Saved checkpoints (gitignored)
```

## Dependencies

- `torch>=2.0.0` (same as Phase 1)
- `datasets>=2.0.0` (FineWeb-Edu streaming)
- `tokenizers>=0.15.0` (BPE training/encoding)

Already configured in `pyproject.toml` under `[project.optional-dependencies] phase2`.

Install: `pip install -e ".[phase2]"`

## Success Criteria

1. `--depth 6` trains in ~1 hour on T4, generates recognizable English
2. `--depth 12` trains in ~4 hours on T4, generates coherent sentences
3. Loss decreases meaningfully over training (expect ~10 → ~4-5 with BPE)
4. Generation shows parallel decoding efficiency (avg tokens decoded per step > 1)
5. Code has educational ASCII diagrams for cosine schedule, ELBO, BPE pipeline
6. All 5 DIFF markers preserved, plus [NEW] markers for Phase 2 additions

## Scope

**In scope**: BPE tokenizer training, cosine schedule, ELBO loss, FineWeb-Edu streaming, configurable depth, generation with prompts
**Out of scope**: Multi-GPU training, mixed precision, gradient accumulation, learning rate finder, model parallelism, evaluation benchmarks

## Key References

- MDLM (Sahoo et al., NeurIPS 2024) — Cosine schedule, ELBO theory
- LLaDA (Nie et al., 2025) — Architecture recipe, no timestep conditioning
- RADD (ICLR 2025) — Time-agnostic training (model infers t from mask count)
- Zhang & Syed (2025) — Cosine schedule Fisher-Rao optimality proof
- nanochat (Karpathy) — `--depth` parametrization pattern
