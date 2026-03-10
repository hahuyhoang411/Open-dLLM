# Phase 5: SmolDLM -- Production Block Diffusion LM

**A modular 144M parameter block diffusion language model.**

---

## What You'll Learn

- How to structure a training codebase as a Python package (12 modules, 1935 lines)
- FP8 mixed-precision training on H100+
- MuonClip optimizer (self-contained, no external dependencies)
- Linear noise schedule (and why it eliminates an entire class of bugs)
- Document packing with attention boundary masking
- Gated Query Attention
- VRAM benchmarking and throughput optimization

---

## Why Phase 5

Phase 4 was 1525 lines in a single file. It worked, but scaling further requires
modularity. Phase 5 refactors into a Python package:

```
phase5/
├── config.py      # CLI args, DDP setup, hyperparameters      (201 lines)
├── model.py       # Model, Block, SwiGLU, RMSNorm             (267 lines)
├── attention.py   # GQA, staircase masks, FlexAttention, RoPE (327 lines)
├── optim.py       # MuonClip + AdamW param grouping           (288 lines)
├── schedule.py    # Linear noise, ELBO weight, WSD scheduler  (108 lines)
├── loss.py        # Chunked CE with grad_checkpoint           (53 lines)
├── data.py        # Document packing, streaming, pre-tokenized(295 lines)
├── generate.py    # Block-by-block denoising, top-k sampling  (166 lines)
├── checkpoint.py  # Save/load, DDP-aware, latest.pt symlink   (57 lines)
├── tokenizer.py   # Encode/decode wrapper                     (42 lines)
├── fp8.py         # Float8Linear, tensorwise scaling           (129 lines)
└── __init__.py    #                                            (2 lines)
                                                         Total: 1935 lines
```

Each module has one job. No circular imports, no god objects. The training
orchestrator (`train.py`, 494 lines) wires them together.

---

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 30 |
| Dim | 576 |
| Query heads | 9 |
| KV heads | 3 (GQA 3:1) |
| MLP hidden | 1536 |
| Params | 144.47M (tied embeddings) |
| Vocab | 49,152 (SmolLM2 merges + Qwen3 pre-tokenizer) |
| Block size | 32 |
| Seq len | 2048 |
| Noise | Linear (mask_prob = t) |
| Optimizer | MuonClip (Muon lr=0.02) + AdamW (lr=3e-3) |

---

## What Changed from Phase 4

| Feature | Phase 4 | Phase 5 |
|---------|---------|---------|
| Code | 1525 lines, single file | 12 modules, 1935 lines |
| Architecture | 20L/768d/12h | 30L/576d/9h (narrower, deeper) |
| Params | 125M | 144M |
| Tokenizer | 32K BPE (custom) | 49K (SmolLM2 cosmo2 + Qwen3 pre-tok) |
| Noise schedule | Cosine | Linear (mask_prob = t) |
| ELBO weight | 1/mask_prob (bug-fixed) | 1/t (trivially correct with linear) |
| Optimizer | External Muon + AdamW | Self-contained MuonClip + AdamW |
| FP8 | No | Yes (nanochat-style, H100+) |
| Data | Streaming, right-padded | Document packing, no padding |
| Attention | FlexAttention | FlexAttention + Gated Query |
| Loss | Liger FLCE | Chunked CE + grad_ckpt (Liger FLCE broken) |

---

## Linear Noise Schedule

Phase 5's simplest and most important change:

```
Cosine: mask_prob = 1 - cos^2(t * pi/2)    ELBO weight = 1/mask_prob (complex)
Linear: mask_prob = t                        ELBO weight = 1/t          (trivial)
```

With the cosine schedule, `mask_prob` and `t` are different quantities. Phase 4
had a bug where `1/t` was used instead of `1/mask_prob` -- the ELBO weight was
wrong by up to 8x at low noise, causing loss to plateau at 4.0. Fixing it
required tracing the math back through the MDLM derivation.

With the linear schedule, `mask_prob = t`, so `1/mask_prob = 1/t`. The bug is
impossible. There is no distinction between the two quantities.

Quokka (2026) confirmed: linear > cosine > poly2 at convergence. The simpler
schedule is also the better one.

```
ELBO weight comparison:

  Cosine schedule:                    Linear schedule:
  weight = 1/mask_prob                weight = 1/t
         = 1/(1 - cos^2(t*pi/2))            = 1/t  (same thing)

  At t=0.1:                           At t=0.1:
    mask_prob = 0.024                   mask_prob = 0.1
    weight = 41.1                       weight = 10.0
    EASY TO GET WRONG                   TRIVIALLY CORRECT
```

---

## The Liger FLCE Bug

WARNING: Liger-Kernel's `FusedLinearCrossEntropyFunction` with `reduction='none'`
has a broken backward pass. This is [linkedin/Liger-Kernel#488](https://github.com/linkedin/Liger-Kernel/issues/488).

**Symptom:** loss=19 (correct CE at init), `grad_norm=0.000000` every single step.
The model never learns. Loss never changes. Everything looks correct except
gradients are all zeros.

**Root cause:** The forward pass computes correct loss values. The backward pass
returns zero gradients. The bug is entirely in the backward kernel -- it silently
produces zeros instead of real gradients when `reduction='none'` is used.

**Why it's hard to catch:** The loss value is correct. Step-0 loss matches the
expected `~ln(49152) = 10.8` for raw CE. Only `grad_norm` reveals the problem,
and many training loops don't log it.

**Fix:** Chunked cross-entropy with `torch.utils.checkpoint`:

```python
# Process 4096 tokens per chunk instead of full (B*L, vocab) logits
# ~1.5 GB peak vs 24 GB for full materialization
for chunk in hidden.split(chunk_size):
    with grad_checkpoint:
        logits = chunk @ lm_head_weight.T
        loss += CE(logits, targets_chunk)
```

Liger's RMSNorm and SwiGLU fused kernels still work correctly. Only FLCE is
broken.

**Lesson:** A correct forward pass does NOT guarantee a correct backward pass.
Always check `grad_norm` at step 0. If it's zero or near-zero, the backward
pass is broken -- regardless of what the loss says.

---

## FP8 Training

Nanochat-style tensorwise FP8 scaling, NOT torchao. Torchao's FP8 causes a 3x
slowdown when combined with gradient checkpointing due to graph breaks.

- 240 `Float8Linear` layers (8 per block x 30 blocks)
- `lm_head` skipped -- its weight is accessed directly in the chunked CE loss
- `disable_fp8()` context manager disables FP8 during generation
- All layer dims divisible by 16 (required for Float8 tensor cores)
- Enable with `--fp8` flag; requires H100+ (SM90)

```
Without FP8:  bf16 matmuls in all linear layers
With --fp8:   FP8 matmuls in attention projections + MLP layers
              bf16 for embeddings, norms, and lm_head
```

---

## Document Packing

Phase 4 right-padded every sequence to `seq_len=2048`. Short documents wasted
compute on padding tokens.

Phase 5 packs documents end-to-end with no padding:

```
Phase 4 (padded):
  [doc1 tokens...] [PAD PAD PAD PAD PAD PAD PAD PAD PAD]  <- waste
  [doc2 tokens............] [PAD PAD PAD PAD PAD PAD PAD]  <- waste
  [doc3 tokens.......................] [PAD PAD PAD PAD]    <- waste

Phase 5 (packed):
  [doc1 tokens...] [doc2 tokens............] [doc3 tok...]
  [...ens.......................] [doc4 tokens............]
  Every position is a real token. Zero waste.
```

Three things make packing work correctly:

1. **Attention boundaries.** Documents within the same packed sequence must not
   attend to each other. FlexAttention masks cross-document attention.
2. **RoPE resets.** Position IDs reset to 0 at each document boundary. Without
   this, the second document in a packed sequence would have positions starting
   at, say, 847 instead of 0.
3. **Loss boundaries.** Loss is computed per-document. Mask tokens from one
   document don't contribute to another document's loss.

---

## Gated Query Attention

Standard multi-head attention computes:

```
output = softmax(QK^T / sqrt(d)) * V
```

Gated Query Attention adds a learned gate on the attention output:

```
output = sigmoid(gate(x)) * attn(x)
```

The gate is initialized to produce zeros, so `sigmoid(0) = 0.5`. At training
start, the attention output is dampened by half. This prevents attention from
dominating the residual stream before the model has learned meaningful attention
patterns.

As training progresses, the gate learns which dimensions of the attention output
to amplify and which to suppress, giving the model finer control over information
flow.

---

## VRAM Benchmarks (1x H100 80GB)

| Mode | Max Batch | Peak VRAM | Throughput | Relative |
|------|-----------|-----------|------------|----------|
| No grad_ckpt | 28 | 77 GB | 55,786 tok/s | 1.0x |
| SAC grad_ckpt | 40 | 79 GB | 22,087 tok/s | 0.40x |
| Regular grad_ckpt | 160 | 76 GB | 16,648 tok/s | 0.30x |

**Winner: no grad_ckpt** -- 3.4x faster despite 5.7x smaller batch.

Why? H100 is compute-bound at 144M params. The GPU is already well-saturated at
batch=28. Recomputing all 30 layers in the backward pass costs more wall-clock
time than the throughput gained from fitting larger batches.

SAC (Selective Activation Checkpointing) is worst of both worlds for this model
size: it still recomputes some ops (slower than no-ckpt) but saves enough
matmul/attention outputs across 30 layers to only allow batch=40 (not much
bigger than 28). SAC shines on deeper/larger models where saved activations are
small relative to total memory.

**Rule of thumb:** At 144M params on H100, skip gradient checkpointing. The
compute cost of recomputation exceeds the throughput benefit of larger batches.

---

## Training Results (Run 1: 4x H100, 1500 steps)

| Metric | Value |
|--------|-------|
| Step-0 loss (ELBO) | 19.11 |
| Step-1500 val loss | 3.36 |
| Step-1500 train loss | 3.58 |
| Throughput | 189K tok/s |
| VRAM peak | 57.9 / 85 GB (68%) |
| Tokens seen | ~1.57B |
| Grad norm | 9.5 -> 0.05 (settled) |

Loss curve: monotonic decrease, no anomalies, no grad clipping issues.

```
Loss
  19 |*
     | **
     |   ***
  10 |      ****
     |          *****
   5 |               *********
     |                        **************
 3.4 |                                      ****
     +------------------------------------------
     0        375       750      1125      1500
                        step
```

Generation at step 750 produced "Vietnam, officially..." -- the model learned
language structure but still degenerates into repetition. This is expected at
1500 steps. Quokka scaling laws predict ~5x more data is needed at 144M params
for meaningful generation quality (need 5K-10K+ steps).

---

## Module Map

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | 201 | CLI argument parsing, DDP initialization, all hyperparameters as a single dataclass |
| `model.py` | 267 | Transformer backbone: `DiffusionLM`, `Block`, `SwiGLU`, `RMSNorm`. Tied embeddings, per-block compile |
| `attention.py` | 327 | GQA with gated queries, staircase block masks via FlexAttention, RoPE with per-document position reset |
| `optim.py` | 288 | Self-contained MuonClip (Newton-Schulz + QK-Clip tau=100) with AdamW fallback. No external muon dependency |
| `schedule.py` | 108 | Linear noise schedule, ELBO weighting (1/t, capped at 10), WSD learning rate scheduler |
| `loss.py` | 53 | Chunked cross-entropy with gradient checkpointing. Processes 4096 tokens per chunk to avoid materializing full logits |
| `data.py` | 295 | Document packing (no padding), HuggingFace streaming, pre-tokenized dataset loading, collation with boundary tracking |
| `generate.py` | 166 | Block-by-block denoising with confidence-based unmasking, top-k sampling, temperature control |
| `checkpoint.py` | 57 | Save/load model + optimizer state, DDP-aware (unwraps module), `latest.pt` symlink for resumption |
| `tokenizer.py` | 42 | Thin wrapper around HuggingFace tokenizer. Encode/decode with special token handling |
| `fp8.py` | 129 | `Float8Linear` with tensorwise scaling, `convert_to_fp8()` for model conversion, `disable_fp8()` context manager |

---

## Quick Start

```bash
# Install Phase 5 dependencies
pip install -e ".[phase5]"

# Local training (single GPU)
python 05_optimized_dllm/train.py --train

# Multi-GPU
torchrun --nproc_per_node=4 05_optimized_dllm/train.py --train --fp8

# Modal cloud (8x H100)
modal run modal_train.py

# Generate
python 05_optimized_dllm/train.py --prompt "The meaning of life is"
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/modal_train.py` | Modal cloud training orchestrator. Configures volumes, secrets, GPU count |
| `scripts/vram_probe.py` | VRAM + throughput benchmark. Tests batch sizes with/without grad_ckpt, FP8 |
| `scripts/debug_generate.py` | Generation diagnostic. Prints logit distributions, tests prompted/unprompted |

---

## References

- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024) -- Masked diffusion theory, continuous-time ELBO
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) -- First 8B dLLM, proves masked diffusion scales
- [BD3-LMs](https://arxiv.org/abs/2503.09573) (Arriola et al., ICLR 2025 Oral) -- Block diffusion theory
- [Quokka](https://arxiv.org/abs/2502.09992) (2026) -- dLLM scaling laws, schedule comparison (linear > cosine)
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (HuggingFace, 2025) -- Tokenizer source (cosmo2 merges)
- [nanochat](https://github.com/karpathy/nanochat) (Karpathy) -- FP8 training reference, clean training style
- [MuonClip](https://github.com/modelscope/ms-swift) -- QK-Clip + Moonlight Newton-Schulz + RMS scaling
- [Liger-Kernel#488](https://github.com/linkedin/Liger-Kernel/issues/488) -- FLCE backward bug with reduction='none'
