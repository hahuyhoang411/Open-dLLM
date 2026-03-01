# Phase 2: nano_dllm

**A real diffusion language model, in a single file.**

---

## What You'll Learn

- BPE tokenization for diffusion models
- Cosine noise schedule and why it's optimal
- ELBO-weighted loss (the proper training objective from MDLM)
- SwiGLU activation (the modern LLM standard)
- Scaling via `--depth` (one dial controls everything)
- Streaming data from HuggingFace

---

## What's New from Phase 1

| # | Phase 1 | Phase 2 | Why |
|---|---------|---------|-----|
| 1 | Char tokenizer (66 tokens) | BPE tokenizer (~32K vocab) | Handles real text, reduces sequence length |
| 2 | Uniform masking | Cosine schedule: `alpha_t = cos^2(t * pi/2)` | Fisher-Rao optimal, better gradient variance |
| 3 | Unweighted CE | ELBO-weighted: `(1/t) * CE` | Proper continuous-time ELBO from MDLM |
| 4 | Fixed 6 layers | `--depth` configurable (6-24) | One codebase from quick experiments to serious runs |
| 5 | Tiny Shakespeare (1MB) | FineWeb-Edu (streaming) | Real web-scale educational text |
| 6 | ReLU^2 activation | SwiGLU activation | Gated mechanism, better performance per parameter |

Phase 1 taught you the five changes that turn a GPT into a dLLM. Phase 2 upgrades
every component to match what real research papers use.

---

## Cosine Noise Schedule

### The Formula

In Phase 1, we sampled a masking ratio `t` and masked tokens with probability `t`
(linear schedule). Phase 2 replaces this with the cosine schedule:

```
alpha_t = cos^2(t * pi/2)          — fraction of tokens kept clean
mask_prob = 1 - alpha_t             — fraction of tokens masked
```

Where `t ~ U[0, 1]` is the timestep sampled during training.

### The Curve

```
alpha_t
(clean)
  1.0 |*
      | **
      |   **
      |     ***
  0.5 |        ***
      |           ****
      |               *****
      |                    ********
  0.0 |                            *****
      +-----------------------------------
     t=0         t=0.5              t=1.0
    (clean)     (half)            (all masked)

  Cosine:   alpha_t = cos^2(t * pi/2)
  Linear:   alpha_t = 1 - t  (Phase 1)
```

### Why Cosine Is Better

The linear schedule spends equal training budget at every noise level. But not all
noise levels are equally informative:

- **t near 0** (few masks): trivial predictions, little learning signal
- **t near 1** (all masks): random guessing, little learning signal
- **t near 0.5** (half masked): rich context, maximum learning signal

The cosine schedule naturally spends more time in the mid-range where the model
learns the most. It moves slowly through the productive middle region and quickly
through the uninformative extremes.

### Why It's Optimal

The cosine schedule is not just a good heuristic -- it is the **Fisher-Rao geodesic**
in information geometry. Zhang & Syed (2025) proved that among all monotone schedules
mapping `[0,1] -> [0,1]`, the cosine schedule minimizes the total information-geometric
path length. This means it distributes gradient variance most uniformly across
timesteps, giving each training step maximum learning efficiency.

### Comparison with Phase 1

```
Phase 1 (uniform):                   Phase 2 (cosine):
  mask_prob = t                        mask_prob = 1 - cos^2(t * pi/2)

  Training budget:                     Training budget:
  |====|====|====|====|====|           |==|====|========|====|==|
  t=0  0.2  0.4  0.6  0.8  1.0        t=0 0.2  0.4  0.6  0.8 1.0
  Equal time at all levels             More time at mid-range noise
```

---

## ELBO-Weighted Loss

### The Theory

Phase 1 used a simple unweighted cross-entropy on masked positions:

```
L_phase1 = CE_masked / num_masked_tokens
```

Phase 2 derives the loss from the continuous-time Evidence Lower Bound (ELBO).
The ELBO for masked diffusion (MDLM, NeurIPS 2024) is:

```
                    1
L_ELBO = integral      alpha'(t)
                    0  ----------  *  E[ sum_masked  -log p(x_0^i | x_t) ]  dt
                    1 - alpha(t)
```

For the cosine schedule, the ratio `alpha'(t) / (1 - alpha(t))` simplifies. When we
estimate this integral by sampling `t ~ U[0,1]`, the loss becomes:

```
L = E_t[ (1/t) * CE_masked(t) ]
```

That's it. Sample a timestep, compute cross-entropy on masked tokens, multiply by `1/t`.

### The 1/t Weighting Curve

```
weight
  10 |*
     | *
     |  *
   5 |   *
     |    **
     |      ***
   1 |         ************************
     +-----------------------------------
    t=0         t=0.5              t=1.0
   (few masks)                   (all masks)
```

### Intuition

Why does `1/t` make sense?

- **Low t** (few tokens masked): the model sees almost the full sentence. Predicting
  the few remaining tokens is hard -- it requires precise knowledge. These harder
  predictions get **upweighted**.
- **High t** (many tokens masked): the model is nearly blind. Predictions are
  largely random. These easier-to-guess predictions get **downweighted**.

The `1/t` weight makes the model focus on the refinement steps (low noise) where
quality matters most during generation.

### Practical Implementation

The `1/t` weight diverges as `t -> 0`, so we clamp:

```python
elbo_weight = 1.0 / t.clamp(min=1e-4)
```

This caps the maximum weight at 10,000 and prevents numerical instability.

This ELBO weighting is THE key theoretical contribution of MDLM (Sahoo et al.,
NeurIPS 2024). It turns a heuristic training objective into a proper variational
bound, giving theoretically grounded training for masked diffusion.

---

## BPE Tokenization for Diffusion

### Why BPE Matters

Phase 1 used character-level tokenization: each character is one token. This means
a 512-character sequence has 512 tokens. With BPE (Byte Pair Encoding), common
subwords like "the", "ing", "tion" become single tokens.

```
Character-level (Phase 1):
  "The meaning of" -> ['T','h','e',' ','m','e','a','n','i','n','g',' ','o','f']
  = 14 tokens for 14 characters

BPE (Phase 2):
  "The meaning of" -> ['The', ' meaning', ' of']
  = 3 tokens for 14 characters
```

Fewer tokens per document means:

- **Shorter sequences**: ~4x fewer tokens than char-level, so the same `block_size`
  covers ~4x more text
- **Less long-range dependency**: the model doesn't waste attention learning that
  "t-h-e" is a word
- **Larger effective context**: 512 BPE tokens covers roughly 2000 characters

### How the MASK Token Works

We train a custom BPE tokenizer with vocabulary size 32,768. The `[MASK]` token is
added as a special token at index 0:

```
Token ID 0:     [MASK]        (special)
Token IDs 1+:   BPE subwords  (learned from data)
```

At the subword level, masking works identically to character-level. Each BPE token
is independently masked or kept -- the model doesn't know or care whether a token
represents one character or five.

### Why Train Our Own

We train a BPE tokenizer from scratch on FineWeb-Edu (`train_tokenizer.py`) rather
than reusing a pre-trained one. This is for educational value: you see exactly how
the vocabulary is built from byte pairs. In production, you would typically reuse
a tokenizer from an existing model.

---

## SwiGLU Activation

### The Formula

Phase 1 used ReLU^2 in the feed-forward network:

```
ReLU^2(x) = max(0, x)^2
MLP(x) = W2 * ReLU^2(W1 * x)
```

Phase 2 uses SwiGLU, the gated activation used in LLaMA, Gemma, and most modern LLMs:

```
SwiGLU(x) = W3 * (SiLU(W1 * x) * W2 * x)

where SiLU(x) = x * sigmoid(x)
```

### Architecture Comparison

```
Phase 1 (ReLU^2 MLP):              Phase 2 (SwiGLU):

  x                                   x
  |                                   |
  v                                   +--------+
  W1 (D -> 4D)                        |        |
  |                                   v        v
  ReLU^2                             W1(D->8D/3)  W2(D->8D/3)
  |                                   |        |
  v                                  SiLU      |
  W2 (4D -> D)                        |        |
  |                                   * (gate) *
  v                                   |
  output                              v
                                     W3 (8D/3 -> D)
                                      |
                                      v
                                      output

  Parameters: 2 * D * 4D              Parameters: 3 * D * 8D/3
            = 8D^2                              ~ 8D^2
```

### Why Gating Helps

The key difference is the **gating mechanism**: `W1` produces a gate signal (passed
through SiLU), and `W2` produces values. The gate selectively amplifies or suppresses
information from the values path. This gives:

- **Smoother gradients**: SiLU is smooth everywhere (unlike ReLU's hard zero)
- **More expressive per parameter**: the gate can learn to route information
- **Better training dynamics**: gated networks converge faster in practice

### The 8/3 Expansion Ratio

Standard MLPs use a 4x expansion: `D -> 4D -> D` (2 matrices, `8D^2` parameters).
SwiGLU has 3 matrices, so to keep the same parameter count, the expansion is `8/3`x:
`D -> 8D/3 -> D` (3 matrices, `3 * D * 8D/3 = 8D^2` parameters).

In practice, the hidden dimension is rounded to a multiple of 256 for GPU alignment.

---

## Scaling via --depth

### The Depth-Dimensions Table

A single `--depth` argument controls everything: number of layers, embedding
dimension, and number of attention heads.

```
depth    n_layer    n_embd    n_head    head_dim    ~Params
-----    -------    ------    ------    --------    -------
  6        6         384        6         64         ~26M
 12       12         768       12         64        ~150M
 24       24        1536       24         64        ~800M
```

The formulas:

```
n_layer = depth
n_embd  = depth * 64
n_head  = depth
head_dim = 64          (fixed across all depths)
```

### Why Single-Dial Scaling

Most transformer configs have 4+ knobs (layers, width, heads, MLP ratio). Compute-
optimal scaling (Chinchilla) shows these should grow together. By tying them to one
number, you get:

- **Easy experimentation**: `--depth 6` for quick debugging, `--depth 12` for real runs
- **Compute-optimal ratios**: width and depth scale together, matching best practices
- **No config files**: the model is fully defined by one integer

Phase 1 was fixed at 6 layers. Phase 2 lets you dial from a small model (26M) to
a serious one (800M) with a single flag.

---

## Quick Start

```bash
# Install Phase 2 dependencies
pip install -e ".[phase2]"

# Train BPE tokenizer on FineWeb-Edu sample
python 02_nano_dllm/train_tokenizer.py

# Train (depth=6, ~26M params, ~1-2 hours on T4 GPU)
python 02_nano_dllm/nano_dllm.py --train --depth 6

# Generate text
python 02_nano_dllm/nano_dllm.py --depth 6 --prompt "The meaning of life is"
```

---

## What's Next

**Phase 3: Block Diffusion** turns nano_dllm into an autoregressive-diffusion hybrid
that generates text in variable-length blocks:

1. **Block-causal attention**: bidirectional within each block, causal between blocks
2. **Variable-length generation**: EOS token ends blocks naturally (no fixed lengths)
3. **KV caching across blocks**: reuse past block keys/values for fast generation
4. **The architecture behind Mercury** (BD3-LMs, ICLR 2025 Oral)

---

## References

- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024) -- Masked diffusion theory, continuous-time ELBO, cosine schedule
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) -- First 8B dLLM, proves masked diffusion scales
- [RADD](https://arxiv.org/abs/2406.03736) (Zhu et al., ICLR 2025) -- Time-agnostic reparameterized absorbing discrete diffusion
- [Cosine schedule optimality](https://arxiv.org/abs/2508.04884) (Zhang & Syed, 2025) -- Fisher-Rao geodesic proof
- [SwiGLU](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) -- GLU variants for transformers
- [nanochat](https://github.com/karpathy/nanochat) (Karpathy) -- Clean training script style reference
