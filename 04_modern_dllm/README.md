# Phase 4: Modern dLLM

**A modern block diffusion language model with fused kernels, in a single 1720-line file.**

---

## What You'll Learn

- FlexAttention: compiled block-sparse attention (no mask tensor in memory)
- Grouped Query Attention (GQA) -- 4:1 head ratio
- Muon optimizer for transformer weights
- Liger fused kernels (RMSNorm, SwiGLU, FusedLinearCrossEntropy)
- torch.compile + DDP multi-GPU training
- WSD learning rate schedule
- **The ELBO weight bug** -- the most important lesson in this project

---

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 20 |
| Dim | 768 |
| Heads | 12 (query) / 3 (KV) |
| MLP hidden | 1536 |
| Params | ~125M (tied embeddings) |
| Vocab | 32,768 (BPE, 14 special tokens) |
| Block size | 32 |
| Seq len | 2048 |

---

## What Changed from Phase 3

| Feature | Phase 3 | Phase 4 |
|---------|---------|---------|
| Architecture | 6L/384d/6h | 20L/768d/12h/3kv |
| Params | 36M | 125M |
| Attention | Manual staircase tensor O(n^2) | FlexAttention (compiled Triton kernel) |
| GQA | No (all MHA) | 4:1 ratio (12 query, 3 KV heads) |
| Optimizer | AdamW | Muon + AdamW |
| Fused kernels | None | Liger (RMSNorm, SwiGLU, FLCE) |
| Compile | None | torch.compile(mode="default") |
| Multi-GPU | No | DDP via torchrun |
| Precision | float32 | AMP bf16 |
| Grad checkpointing | No | Per-block |
| Scheduler | Cosine decay | WSD (warmup-stable-decay) |
| Dataset | FineWeb-Edu (streaming) | 100B multi-source |
| Tokenizer | 32K BPE (3 specials) | 32K BPE (14 specials, Qwen3-style) |
| Normalization | LayerNorm | Learnable RMSNorm (Liger-fused) |
| Noise schedule | Cosine | Cosine (with bandwidth clamp) |
| ELBO weight | 1/t (buggy) | 1/mask_prob (correct) |

Phase 3 proved block diffusion works. Phase 4 makes it fast, scales it 3.5x, and fixes the
math along the way.

---

## The ELBO Weight Bug

This is the most important lesson in this entire project. If you read one section, read this.

### The Setup

Phase 2 used a cosine noise schedule with ELBO weight `1/t`, following LLaDA's implementation.
It looked clean. It ran fine. The loss decreased. And it was wrong.

### The Math

The continuous-time ELBO for masked diffusion is:

```
L_ELBO = E_t[ (1/mask_prob(t)) * CE_masked(t) ]
```

The correct importance weight is `1/mask_prob`, not `1/t`. These are only the same when
`mask_prob = t` -- i.e., with a **linear** schedule.

LLaDA uses a linear schedule: `mask_prob = t`, so `1/mask_prob = 1/t`. Correct.

We used a cosine schedule: `mask_prob = 1 - cos^2(t * pi/2)`. So `1/mask_prob != 1/t`. Wrong.

### Concrete Numbers

```
At t = 0.1 (low noise):
  Cosine: mask_prob = 1 - cos^2(0.1 * pi/2) = 0.024
  Correct weight: 1/0.024 = 41.7
  Our weight:     1/0.1   = 10.0    <-- 4x too low

At t = 0.3 (moderate noise):
  Cosine: mask_prob = 1 - cos^2(0.3 * pi/2) = 0.206
  Correct weight: 1/0.206 = 4.85
  Our weight:     1/0.3   = 3.33    <-- 1.5x too low

At t = 0.9 (high noise):
  Cosine: mask_prob = 1 - cos^2(0.9 * pi/2) = 0.976
  Correct weight: 1/0.976 = 1.02
  Our weight:     1/0.9   = 1.11    <-- slightly too high
```

### The Effect

Low-noise timesteps (where the model learns to do final refinement) are systematically
under-weighted by up to 4x. High-noise timesteps (where the model is mostly guessing)
are slightly over-weighted. The model never properly learns the refinement steps that
matter most during generation.

Symptom: loss plateaus at ~4.0 and stops improving.

### The Fix

Two options:

1. **Fix the weight**: use `1/mask_prob` instead of `1/t` (what Phase 4 does)
2. **Fix the schedule**: switch to linear where `mask_prob = t` (what Phase 5 does)

Phase 5 chose option 2 because the linear schedule is trivially correct -- no room for
this class of bug. `mask_prob = t`, weight = `1/t`, done.

### The Lesson

The ELBO derivation is schedule-specific. The `1/t` weighting from MDLM only works when
`mask_prob = t`. LLaDA's code uses `1/t` because their schedule is linear. We copied from
LLaDA but used a cosine schedule. The code looked right. The math was wrong.

**Always derive the ELBO weight for YOUR specific schedule.**

The relevant code in `modern_dllm.py` (lines 519-530):

```python
# The correct importance weight for the cosine schedule is 1/mask_prob, NOT 1/t.
# With mask_prob = sin^2(t*pi/2), using 1/t under-weights low-noise timesteps
# by 4-8x, causing loss plateau at ~4.0.
if use_cart:
    elbo_w = _compute_cart_weights(mask, padding, cart_p)
else:
    elbo_w = 1.0 / mask_prob.clamp(min=1e-4)
```

---

## FlexAttention

The staircase mask is the core of block diffusion. It controls information flow in the
doubled training sequence `[x_t || x_0]`:

```
                x_t half          x_0 half
           +------------------+------------------+
   x_t     |  M_BD            |  M_OBC           |
   half    |  (bidirectional  |  (attend to x_0  |
           |   within block)  |   from EARLIER    |
           |                  |   blocks only)    |
           +------------------+------------------+
   x_0     |                  |  M_BC            |
   half    |    (nothing)     |  (block-causal:  |
           |                  |   current +      |
           |                  |   earlier)       |
           +------------------+------------------+
```

Phase 3 built this as a dense float tensor -- O(n^2) memory, materialized every forward pass.

Phase 4 defines the same mask as a Python function and lets PyTorch compile it into a
Triton kernel:

```python
def staircase_mask_mod(b, h, q_idx, kv_idx):
    x0_q = (q_idx >= n)
    x0_kv = (kv_idx >= n)
    blk_q = (q_idx % n) // blk_size
    blk_kv = (kv_idx % n) // blk_size

    m_bd = (blk_q == blk_kv) & (x0_q == x0_kv)
    m_obc = (blk_q > blk_kv) & x0_kv & ~x0_q
    m_bc = (blk_q >= blk_kv) & x0_kv & x0_q

    return m_bd | m_obc | m_bc

mask = create_block_mask(staircase_mask_mod, ...)
```

`create_block_mask` analyzes which 128x128 tiles are entirely masked and skips them.
For the staircase pattern with ~50% sparsity, this means ~half the tiles are never loaded
or computed. Result: ~2x faster than SDPA with an explicit mask, and zero mask memory.

---

## GQA (Grouped Query Attention)

Standard multi-head attention: every query head has its own key and value head. GQA
shares key/value heads across groups of query heads.

```
MHA (Phase 3):                GQA 4:1 (Phase 4):
  Q heads: 12                   Q heads: 12
  K heads: 12                   K heads:  3
  V heads: 12                   V heads:  3

  Q[0..3]  share K[0], V[0]
  Q[4..7]  share K[1], V[1]
  Q[8..11] share K[2], V[2]
```

Benefits:
- KV projection parameters cut by 4x
- KV cache memory cut by 4x during generation
- Minimal quality impact at 125M scale (validated empirically)

PyTorch 2.5+ handles the expansion in `scaled_dot_product_attention` via `enable_gqa=True`.

---

## Muon Optimizer

Standard AdamW tracks first and second moment estimates for each parameter -- 2x model
size in optimizer states. Muon replaces this for 2D weight matrices with Newton-Schulz
orthogonalization, cutting optimizer memory by ~50% for those params.

The split:
- **Muon** (lr=0.02): all 2D weight matrices in attention and MLP
- **AdamW** (lr=6e-4): embeddings, normalization parameters, biases

`MuonWithAuxAdam` handles both groups in one optimizer. It requires DDP
(`dist.all_gather` in the step). For single GPU, use `SingleDeviceMuonWithAuxAdam`.

Gotchas:
- `pip install muon` installs a bioinformatics package. The correct install:
  `pip install git+https://github.com/KellerJordan/Muon`
- Muon param groups have a strict key schema: `{params, lr, momentum, weight_decay, use_muon}`.
  AdamW groups: `{params, lr, betas, eps, weight_decay, use_muon}`. Extra or missing keys crash.
- Tied embeddings create duplicate parameters in `named_parameters()`. Deduplicate with
  `data_ptr()` before building param groups.

---

## WSD Schedule

Warmup-Stable-Decay replaces cosine annealing. Three phases, sharp transitions:

```
LR factor
  1.0 |        ________________________________
      |       /                                 \
      |      /                                    \
      |     /                                       \
      |    /                                          \
  0.0 |___/                                            \___
      0  warmup(2K)     stable plateau      decay(40K)  50K
```

- **Warmup** (0 to 2K steps): linear ramp from 0 to peak LR
- **Stable** (2K to 40K steps): hold at peak LR
- **Decay** (40K to 50K steps): linear decay from peak to 0

The schedule returns a `[0, 1]` multiplier. Each param group stores its own `initial_lr`,
so Muon (0.02) and AdamW (6e-4) share the same schedule shape but different peak values.

WSD is simpler than cosine (no hyperparameter tuning for the decay shape) and allows
extending training by appending more stable steps before decay.

---

## Training Infrastructure

### Modal (Cloud GPU)

```bash
modal run 05_phase5_dllm/scripts/modal_train.py --gpu A100-40GB:2
```

Uses Modal volumes `smoldlm-checkpoints` + `smoldlm-data`, requires `huggingface-secret`.

### Local (Multi-GPU)

```bash
torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train
```

DDP auto-detects via the `RANK` environment variable set by torchrun.

### torch.compile

```python
model = torch.compile(model, dynamic=False)
```

`mode="default"` (Inductor fusion). NOT `reduce-overhead` -- CUDA graphs pin all layer
activations simultaneously, causing 7-10x memory overhead vs eager. Only useful for
inference at batch=1. Every major training repo (Karpathy's nanochat, torchtitan, Meta)
uses default mode.

First-run compilation takes 5-15 minutes (Triton kernel generation). Normal behavior.

### AMP bf16

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, loss = model(x_input, targets, mask, t, attn_mask)
```

No `GradScaler` needed -- bf16 has the same exponent range as float32 (unlike fp16).
Requires Ampere+ GPU (A10G, A100, H100).

---

## Quick Start

```bash
# Install Phase 4 dependencies
pip install -e ".[phase4]"

# Local training (2 GPUs)
torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train

# Modal cloud training
modal run 05_phase5_dllm/scripts/modal_train.py --gpu A100-40GB:2

# Generate text
python 04_modern_dllm/modern_dllm.py --prompt "The quick brown fox"
```

---

## What's Next

**Phase 5** refactors the single file into a modular package (`phase5/`) and makes several
key changes:

- **Linear noise schedule**: `mask_prob = t`, eliminating the ELBO weight bug class entirely
- **SmolLM2 tokenizer**: 49,152 vocab (cosmo2 merges + Qwen3 pre-tokenizer)
- **FP8 matmuls**: nanochat-style tensorwise scaling on H100+
- **MuonClip**: self-contained optimizer (no external dependency, QK-Clip tau=100)
- **Document packing**: no right-padding, attention masked at doc boundaries
- **Gated Query Attention**: replaces standard GQA

---

## References

- [BD3-LMs](https://arxiv.org/abs/2503.09573) (Arriola et al., ICLR 2025 Oral) -- Block diffusion, staircase attention mask
- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024) -- Continuous-time ELBO for masked diffusion
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) -- First 8B dLLM, linear schedule with 1/t weight
- [Fast-dLLM v2](https://arxiv.org/abs/2509.26328) (NVIDIA, 2025) -- Complementary masking
- [Liger Kernel](https://arxiv.org/abs/2410.10989) (LinkedIn, 2024) -- Fused RMSNorm, SwiGLU, FLCE
- [Muon](https://github.com/KellerJordan/Muon) (Keller Jordan) -- Newton-Schulz optimizer for transformers
- [nanochat](https://github.com/karpathy/nanochat) (Karpathy) -- Training recipe reference (WSD, compile, AMP)
- [Dream](https://arxiv.org/abs/2412.06264) (Dream Team, 2024) -- CART noise rescheduling
