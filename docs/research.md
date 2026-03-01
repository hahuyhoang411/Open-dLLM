# dLLM Research Compilation

Compiled 2026-03-01 from deep research across ~50 papers, 15+ codebases, and latest 2025-2026 developments.

---

## The Core Idea

A diffusion language model generates text by iteratively denoising a fully masked sequence, rather than predicting one token at a time. The key insight: **a dLLM is BERT with two twists** — (1) train across all masking ratios (0%-100%) instead of fixed 15%, and (2) iteratively denoise at inference.

## Evolution Timeline

```
BERT MLM (2018)
    └── D3PM (Austin et al., 2021) — First principled discrete diffusion
         ├── SEDD (Lou et al., ICML 2024 Best Paper) — Score entropy for discrete spaces
         └── MDLM (Sahoo et al., NeurIPS 2024) — Simplified masked diffusion = weighted MLM
              ├── LLaDA 8B (Nie et al., Feb 2025) — First 8B dLLM, competitive with LLaMA 3
              ├── BD3-LMs (Arriola et al., ICLR 2025 Oral) — Block diffusion
              ├── Mercury (Inception Labs, Jun 2025) — 1000+ TPS commercial dLLM
              ├── d1 (Apr 2025) — RL (diffu-GRPO) for dLLM reasoning
              ├── TiDAR (Nov 2025) — Hybrid diffusion+AR
              ├── Mercury 2 (Feb 2026) — First reasoning dLLM, 5x faster than speed-optimized AR
              └── LLaDA 2.1 (Feb 2026) — T2T editing, 1587 TPS peak
```

---

## Three Families of Approaches

### 1. Absorbing/Masking Diffusion (Simplest, Best at Scale)

**How**: Replace tokens with [MASK], train a bidirectional transformer to predict original tokens.

**Forward process**: Each token independently masked with probability `(1 - alpha_t)`:
```
q(x_t | x_0) = alpha_t * delta(x_t, x_0) + (1 - alpha_t) * delta(x_t, MASK)
```

**Loss**: Cross-entropy on masked positions:
```
L = integral_0^1 (alpha'_t / (1 - alpha_t)) * E[sum_{masked i} -log p_theta(x_0^i | x_t)] dt
```

For uniform `t ~ U[0,1]`, simplifies to: `L = (1/t) * CE on masked positions`

**Key papers**: MDLM (NeurIPS 2024), LLaDA (2025), BD3-LMs (ICLR 2025 Oral)

### 2. Score Entropy Discrete Diffusion (SEDD)

**How**: Learn probability ratios `p(y)/p(x)` (concrete scores) instead of direct token probabilities.

**Score entropy loss**:
```
L = sum_{y neighbor of x} [s_theta(x)_y - (p(y|x_0)/p(x|x_0)) * log s_theta(x)_y]
```

**Key paper**: SEDD (Lou et al., ICML 2024 Best Paper). Reduces perplexity by 25-75% vs prior diffusion. Outperforms GPT-2.

**Insight from RADD (ICLR 2025)**: For absorbing diffusion, SEDD's score network learns the same thing as a masked prediction network, just parameterized differently.

### 3. Discrete Flow Matching

**How**: Learn a velocity field transporting noise distribution to data distribution on the probability simplex.

**Key paper**: Gat et al. (NeurIPS 2024). Most general framework but in practice reduces to something similar to masking.

---

## Key Papers Deep Dive

### MDLM (NeurIPS 2024) — The Foundation

**Core insight**: The SUBS parameterization simplifies the absorbing diffusion ELBO to a **weighted mixture of classical MLM losses**. No CTMC theory needed.

**SUBS parameterization**:
1. Replace logit for [MASK] with -infinity (masked probability = 0)
2. Unmasked tokens copy directly (carry-over)

**Training recipe improvements**:
- Larger vocabularies reduce long-range dependencies
- DiT architecture (RoPE + timestep conditioning) > T5
- Low-discrepancy time sampling reduces ELBO variance
- Cosine noise schedule

**Results**: PPL 23.00 on LM1B (vs AR baseline 20.86) at 110M params.

### LLaDA 8B (Feb 2025)

**Architecture**: Standard transformer (RMSNorm, SwiGLU, RoPE) with bidirectional attention. Vanilla MHA (no GQA — KV caching incompatible with full-sequence diffusion). 8B params, trained on 2.3T tokens.

**Key finding**: Emergent capabilities (in-context learning, instruction following) are NOT exclusive to AR models.

**Training**: mask_ratio ~ U[0,1], cross-entropy on masked positions, 0.13M H800 GPU hours.

**No timestep conditioning** — model infers noise level from mask count.

### Mercury / Mercury 2 (Inception Labs, 2025-2026)

**Block diffusion**: Divide output into fixed blocks, generate tokens within each block in parallel via diffusion, process blocks sequentially.
- Between blocks: autoregressive (causal attention, KV cacheable)
- Within blocks: diffusion (bidirectional attention)

**Mercury 2** (Feb 2026):
- 1,009 tokens/sec on NVIDIA Blackwell GPUs
- 128K context window
- Reasoning capabilities (AIME: 91.1, GPQA: 73.6, LiveCodeBench: 67.3)
- 5x faster than Claude 4.5 Haiku, GPT-5 Mini
- $0.25/$0.75 per million tokens (input/output)

### BD3-LMs (ICLR 2025 Oral)

**Formalization**: Block diffusion interpolates between pure AR (block_size=1) and pure diffusion (block_size=full_sequence). Block-causal attention.

**Benefits**: Arbitrary-length generation, KV caching across blocks, SOTA likelihoods among diffusion models.

### d1 — diffu-GRPO (Apr 2025)

**RL for dLLMs**: Group generation → relative scoring → policy optimization.

**Key innovation**: Mean-field approximation for sequence log-probability (dLLMs can't factorize probability autoregressively). Random prompt masking as regularization.

**Results**: +3.9% GSM8K, +26.2% Countdown on LLaDA 8B.

### LLaDA 2.1 (Feb 2026)

**T2T (Token-to-Token) editing**: Already-placed tokens can be revised. Dual thresholds (tau_mask, tau_edit) control unmasking vs editing. Multiple Block Editing (MBE) revises previous blocks based on new context.

**Speed**: 1587 TPS peak.

### TiDAR (Nov 2025)

**Hybrid**: Draft tokens via diffusion (parallel), verify via AR (quality guarantee), single forward pass.
- Causal attention for prefix (KV cacheable)
- Bidirectional attention for decoding block
- Joint NTP + M2T loss

**Results**: 4.7-5.9x speedup over pure AR with near-identical quality.

---

## Noise Schedules

### Cosine Schedule (Recommended, Provably Optimal)

```
alpha_t = cos^2(t * pi / 2)
```

- Smooth, symmetric S-curve
- Spends more budget at mid-range masking ratios (richest learning signal)
- **Proven Fisher-Rao optimal** (Zhang & Syed, 2025): The geodesic in information geometry produces the cosine schedule

### Linear Schedule

```
alpha_t = 1 - t
```

Simple but over-weights easy predictions (low masking). Suboptimal.

### Key Insight

The ELBO is invariant to schedule choice at endpoints, but gradient **variance** depends on weighting. Cosine minimizes this variance.

---

## Architecture Recipe

The proven recipe for a dLLM (from LLaDA, tiny-diffusion, MDLM):

1. Standard transformer (LLaMA-style): RMSNorm, SwiGLU (or ReLU²), RoPE
2. Remove causal mask → bidirectional attention (`is_causal=False`)
3. Vanilla MHA (no GQA — KV cache doesn't apply to full-sequence diffusion)
4. Add [MASK] token to vocabulary
5. No explicit timestep conditioning needed (RADD, ICLR 2025: model infers noise from mask count)

### The 5 Changes from GPT → dLLM

1. **Add mask token** to vocab
2. **`is_causal=False`** — bidirectional attention
3. **Training objective**: randomly mask tokens, predict originals
4. **Loss masking**: only compute loss on [MASK] positions
5. **Generation**: confidence-based iterative unmasking

---

## Inference Strategies

### Confidence-Based Unmasking

At each step, unmask positions where the model is most confident. Threshold-based: all tokens above confidence threshold unmask simultaneously; if none, unmask the most confident one.

### Number of Steps

Practical range: 10-64 steps. 50-100 steps near-optimal. 1-10 steps possible with consistency distillation.

### Variable-Length Generation

Block diffusion: generate block-by-block, stop at EOS. Alternatively, dLLM-Var: train model to predict EOS token accurately.

---

## What's Failing / Open Problems

1. **No KV cache** in full-sequence mode (block diffusion partially solves)
2. **Length prediction** required upfront
3. **Sequential reasoning**: For perfect-sequence tasks, dLLMs need as many steps as AR needs tokens
4. **Tokens are permanent** in absorbing diffusion (T2T editing addresses this)
5. **Numerical precision bug** in 32-bit Gumbel-max sampling
6. **Chain-of-thought** reasoning flow not natural for diffusion

---

## Existing Implementations (Simplest → Most Complex)

| Implementation | Params | Tokenizer | Lines | Hardware | URL |
|---|---|---|---|---|---|
| tiny-diffusion | 10.7M | char | 365 | Any | github.com/nathan-barry/tiny-diffusion |
| diffusion-gpt | 7.2M | char | 1 notebook | Colab | github.com/ash80/diffusion-gpt |
| TextDiffusionSEDD | small | char/BPE | 3 files | A10 | github.com/Oxen-AI/TextDiffusionSEDD |
| LLaDA-from-scratch | 100-310M | BPE | multi-file | A100 | github.com/F4k3r22/LLaDA-from-scratch |
| SEDD official | GPT-2 | BPE | ~2K+ | 8x80GB | github.com/louaaron/Score-Entropy-Discrete-Diffusion |
| MDLM official | varied | BPE | moderate | A5000 | github.com/kuleshov-group/mdlm |
| BD3-LMs | varied | BPE | moderate | multi-GPU | github.com/kuleshov-group/bd3lms |
| dLLM library | any | any | framework | multi-GPU | github.com/ZHZisZZ/dllm |
| Open-dLLM/dCoder | 0.5B | BPE | full stack | 4+ GPU | github.com/pengzhangzhi/Open-dLLM |
| Dream 7B | 7B | BPE | moderate | 8+ GPU | github.com/DreamLM/Dream |
| LLaDA official | 8B | BPE | inference only | - | github.com/ML-GSAI/LLaDA |

---

## Key Academic Resources

### Course
- **Cornell Deep Generative Models** (Kuleshov group): kuleshov-group.github.io/dgm-website/
  - Lecture 13: Diffusion Models
  - Lecture 16: Discrete Deep Generative Models

### Papers
- D3PM: arxiv.org/abs/2107.03006
- SEDD: arxiv.org/abs/2310.16834
- MDLM: arxiv.org/abs/2406.07524
- LLaDA: arxiv.org/abs/2502.09992
- LLaDA 2.1: arxiv.org/abs/2602.08676
- Mercury: arxiv.org/abs/2506.17298
- BD3-LMs: arxiv.org/abs/2503.09573
- d1: arxiv.org/abs/2504.12216
- TiDAR: arxiv.org/abs/2511.08923
- Discrete Flow Matching: arxiv.org/abs/2407.15595
- Cosine Schedule Optimality: arxiv.org/abs/2508.04884
- RADD (time-agnostic): ICLR 2025

### Blog Posts
- SEDD blog: aaronlou.com/blog/2024/discrete-diffusion/
- Oxen.ai training: ghost.oxen.ai/how-to-train-diffusion-for-text-from-scratch/
- dLLM overview: spacehunterinf.github.io/blog/2025/diffusion-language-models/
- Sean Goedecke limitations: seangoedecke.com/limitations-of-text-diffusion-models/
