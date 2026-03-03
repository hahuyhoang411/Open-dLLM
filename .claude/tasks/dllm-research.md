# Diffusion Language Models (dLLMs) -- Deep Research Report

Date: 2026-03-01

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [MDLM -- The Foundation](#1-mdlm-masked-diffusion-language-models)
3. [SEDD -- Score Entropy Alternative](#2-sedd-score-entropy-discrete-diffusion)
4. [LLaDA -- Scaling Masked Diffusion](#3-llada-large-language-diffusion-with-masking)
5. [LLaDA 2.1 -- Token Editing](#4-llada-21-token-editing)
6. [Mercury -- Block Diffusion](#5-mercury-inception-labs)
7. [d1 -- RL for Reasoning](#6-d1-reasoning-via-reinforcement-learning)
8. [TiDAR -- Hybrid Diffusion-AR](#7-tidar-think-in-diffusion-talk-in-autoregression)
9. [Discrete Flow Matching](#8-discrete-flow-matching)
10. [tiny-diffusion Reference Implementation](#9-tiny-diffusion-reference-implementation)
11. [Minimum Viable dLLM](#10-minimum-viable-dllm)
12. [Common Pitfalls](#11-common-pitfalls)
13. [Sources](#sources)

---

## Executive Summary

Diffusion Language Models (dLLMs) generate text by iteratively denoising a corrupted sequence rather than predicting one token at a time left-to-right. The dominant paradigm is **masked diffusion**: corrupt text by randomly replacing tokens with [MASK], then train a bidirectional transformer to predict the original tokens. At inference, start from all-[MASK] and iteratively unmask tokens in parallel.

**Key insight**: A masked diffusion LLM is remarkably similar to a standard GPT. The tiny-diffusion project demonstrates that only **19 lines of code** differ between a GPT and a dLLM implementation. The 5 changes are: (1) add mask token to vocabulary, (2) bidirectional attention instead of causal, (3) parallel confidence-based decoding instead of sequential, (4) training objective is denoising masked tokens instead of next-token prediction, (5) only masked tokens contribute to loss.

The field has progressed rapidly:
- **2023**: SEDD (ICML 2024 Best Paper) -- score entropy for discrete diffusion
- **2024**: MDLM (NeurIPS 2024) -- simplified masked diffusion to weighted MLM losses
- **2025 Feb**: LLaDA -- first 8B dLLM trained from scratch, competitive with LLaMA3
- **2025 Jun**: Mercury -- 1100+ tokens/sec via block diffusion
- **2025 Apr**: d1 -- first RL (diffu-GRPO) applied to dLLMs for reasoning
- **2025 Nov**: TiDAR -- hybrid diffusion+AR, 5.9x speedup with AR quality
- **2026 Feb**: LLaDA 2.1 -- token editing, 1587 TPS peak speed

---

## 1. MDLM (Masked Diffusion Language Models)

**Paper**: "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
**Authors**: Sahoo, Arriola, Gokaslan, et al. (Kuleshov group, Cornell)
**Code**: https://github.com/kuleshov-group/mdlm

### Core Idea

Masked diffusion for text is far simpler and more effective than previously believed. By restricting discrete diffusion to masking-only (absorbing state) processes, the ELBO simplifies to a **weighted mixture of classical masked language modeling (MLM) losses**. This Rao-Blackwellized objective has lower variance and better performance than general discrete diffusion approaches like D3PM.

### Forward Process (Noising)

Each token is independently masked with time-dependent probability:

```
q(z_t | x) = Cat(z_t; alpha_t * x + (1 - alpha_t) * m)
```

Where:
- `x` = one-hot original token
- `m` = one-hot mask token
- `alpha_t` in [0, 1] strictly decreasing: alpha_0 ~ 1 (clean), alpha_1 ~ 0 (all masked)
- At time t, each token stays original with probability alpha_t, becomes [MASK] with probability (1 - alpha_t)

**Absorbing property**: Once a token is masked, it stays masked for all subsequent times.

### Reverse Process (Denoising)

When z_t = [MASK], the reverse posterior is:

```
q(z_s | z_t=MASK, x) = Cat(z_s; [(1-alpha_s)*m + (alpha_s - alpha_t)*x] / (1-alpha_t))
```

The model x_theta(z_t, t) predicts the clean token distribution for each masked position.

### Training Objective

The continuous-time NELBO simplifies to:

```
L = integral_0^1 (alpha'_t / (1 - alpha_t)) * sum_{l=1}^{L} log <x_theta^l(z_t), x^l> dt
```

Where alpha'_t is the time derivative of alpha_t, and `<.,.>` is the inner product (dot product of predicted probability with one-hot ground truth).

**Key property**: The ELBO is invariant to the functional form of alpha_t (the noise schedule). This means the specific schedule shape doesn't matter much -- only the weighting induced by `alpha'_t / (1 - alpha_t)` matters.

In practice, this is estimated via Monte Carlo by sampling random timesteps t ~ U[0,1].

### SUBS Parameterization

Two critical constraints on x_theta:
1. **Zero Masking Probability**: Set logit of [MASK] token to -infinity (model never predicts [MASK] as a real token)
2. **Carry-Over Unmasking**: If z_t^l is NOT masked, copy it through directly (don't re-predict already visible tokens)

### Noise Schedule

The paper shows ELBO is invariant to alpha_t's functional form. In practice they use a log-linear schedule. The schedule only affects the weighting of which masking levels are emphasized during training.

### Architecture

- **Diffusion Transformer (DiT)**: Encoder-only transformer with timestep conditioning
- **Rotary Positional Embeddings (RoPE)**
- ~110M parameters (comparable to GPT-2 Small)
- Context sizes: 128 tokens (LM1B), 1024 tokens (OpenWebText)
- No causal mask (bidirectional attention)

### Key Results

| Model | Params | Dataset | PPL (NELBO) |
|-------|--------|---------|-------------|
| D3PM (absorb) | 70M | LM1B | <=76.90 |
| DiffusionBERT | 110M | LM1B | <=63.78 |
| SEDD | 110M | LM1B | <=32.79 |
| **MDLM** | **110M** | **LM1B** | **<=23.00** |
| AR Transformer | 110M | LM1B | 20.86 |

MDLM closes the gap to AR models to ~10% on LM1B (23.00 vs 20.86).
On OpenWebText: MDLM <=23.21 vs AR 17.54.

### Sampling Algorithm

1. Start with z_1 = all [MASK] tokens
2. For each step t -> s (decreasing):
   - Run x_theta(z_t, t) to get predicted clean token distributions
   - For each masked position, sample from reverse posterior
   - Already-unmasked tokens are kept fixed
3. **Semi-autoregressive**: Generate blocks of L tokens, use tail as prefix for next block
4. **Low-discrepancy sampler**: Correlated time samples across batch to reduce variance

### What Worked / Didn't

**Worked**:
- Restricting to masking-only (vs general discrete diffusion) enabled massive simplification
- Modern architecture (DiT, RoPE) over older T5-based designs
- Larger vocabularies (vs D3PM's 8k vocab) crucial for good perplexity
- Computing KL only over masked indices (numerical stability)

**Didn't work as well**:
- Still ~10-25% gap to AR models on perplexity
- Requires more training steps than AR (only masked tokens contribute to loss, so ~2x)
- Fixed context length (no flexible generation without semi-autoregressive tricks)

### Simplest Implementation

The core training loop is:
```
1. Sample batch of clean sequences x
2. Sample t ~ U[0,1]
3. For each token, mask with probability (1-alpha_t)
4. Forward pass: logits = model(masked_x, t)
5. Loss = cross_entropy(logits[masked_positions], x[masked_positions])
   weighted by alpha'_t / (1 - alpha_t)
6. Backprop
```

---

## 2. SEDD (Score Entropy Discrete Diffusion)

**Paper**: "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024 Best Paper)
**Authors**: Aaron Lou, Chenlin Meng, Stefano Ermon (Stanford)
**Code**: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

### Core Idea

Instead of predicting the clean token directly (like MDLM), SEDD learns **probability ratios** between neighboring discrete states. This is the discrete analogue of score matching in continuous diffusion. The key insight: by modeling ratios p(y)/p(x) rather than absolute probabilities p(x), the intractable normalizing constant Z cancels out.

### The Concrete Score

For discrete data, define the "concrete score" as:

```
s_theta(x)_y = p_data(y) / p_data(x)
```

for states y that are Hamming-distance-1 neighbors of x (differ at exactly one position). The network outputs these ratios for all neighbor states from a single forward pass.

### Score Entropy Loss

The training objective (score entropy):

```
L = E[ sum_{y~x} s_theta(x)_y - (p(y|x_0) / p(x|x_0)) * log s_theta(x)_y ]
```

Where y~x means y differs from x at one position, and the expectation is over noisy samples from the forward process. The true ratios p(y|x_0)/p(x|x_0) are tractable given the clean data x_0.

### Noise Process (CTMC)

Uses a Continuous-Time Markov Chain (CTMC):

```
dp_t/dt = Q * p_t
```

where Q is a transition rate matrix. Unlike MDLM's absorbing (masking-only) process, SEDD can use more general noise processes including:
- **Absorbing**: tokens jump to [MASK] (same as MDLM)
- **Uniform**: tokens jump to any random token
- The paper found absorbing works best for text

### Reverse Process

```
p(x_{t-dt} = a | x_t = b) ~ delta_b(a) + (p_t(a)/p_t(b)) * Q(b,a) * dt
```

The learned score s_theta replaces the unknown ratio p_t(a)/p_t(b).

### Architecture

- Encoder-only transformer (similar to BERT)
- SEDD-small: ~90M params, SEDD-medium: ~320M non-embedding params
- Matches GPT-2 sizes for fair comparison
- Time conditioning injected into transformer
- Batch size 512, learning rate 3e-4, gradient norm clipped to 1

### Key Results

- Reduces perplexity by **25-75%** compared to previous diffusion approaches
- **Outperforms GPT-2** at comparable model sizes
- 6-8x better generative perplexity than un-annealed GPT-2
- Achieves similar quality with **32x fewer network evaluations** (compute-quality tradeoff)
- Enables controllable infilling

### What Worked / Didn't

**Worked**:
- Ratio estimation avoids normalizing constant problem
- Score entropy is a principled extension of continuous score matching
- Compute-quality tradeoff: can trade inference steps for quality
- No need for temperature scaling at generation time

**Didn't**:
- More complex than MDLM (CTMC theory, ratio estimation)
- Time-dependent rates prevent caching optimizations that MDLM can do
- MDLM's simpler approach ultimately achieved better perplexity

### Simplest Version

The conceptual jump from MDLM to SEDD: instead of predicting "what was the original token?", predict "how likely is each neighbor state relative to the current state?". In practice, MDLM's absorbing variant and SEDD-absorbing are closely related -- the key difference is the loss function (cross-entropy vs score entropy).

---

## 3. LLaDA (Large Language Diffusion with mAsking)

**Paper**: "Large Language Diffusion Models" (NeurIPS 2025 Oral)
**Authors**: Shen Nie, Fengqi Zhu, et al.
**Code**: https://github.com/ML-GSAI/LLaDA

### Core Idea

LLaDA proves that the emergent capabilities of LLMs (in-context learning, instruction following, reasoning) are NOT exclusive to autoregressive models. By scaling masked diffusion to 8B parameters and training on 2.3T tokens, LLaDA matches LLaMA3 8B on many benchmarks. This is the first evidence that dLLMs can be a viable alternative paradigm at scale.

### Forward Process

Identical to MDLM conceptually:

```
q(x_t | x_0) = product_{i=1}^{L} q(x_t^i | x_0^i)
```

Each token independently: stays as x_0^i with probability (1-t), becomes [MASK] with probability t.

- t ~ U[0, 1] during training
- At t=0: sequence is clean
- At t=1: sequence is fully masked
- Crucially, LLaDA uses **uniform t**, meaning the masking ratio varies uniformly from 0% to 100%

### Training Objective

Cross-entropy loss only on masked positions:

```
L = E_{t~U[0,1]} [ E_{x_t ~ q(x_t|x_0)} [ -sum_{i in masked(t)} log p_theta(x_0^i | x_t) ] ]
```

The model predicts the original token for each masked position, ignoring unmasked positions in the loss.

### Architecture

LLaDA uses a standard transformer, nearly identical to LLaMA3 architecture with key modifications:

- **8B parameters** (also 1B for ablations)
- Standard Transformer: RMSNorm, SwiGLU FFN, RoPE
- **No causal mask** -- full bidirectional attention
- **Vanilla multi-head attention** (not grouped query attention) because KV caching is incompatible
- Reduced FFN dimension to keep total parameter count comparable (since attention has more params without GQA)
- Sequence length: 4096 tokens
- 1% of sequences use random shorter lengths (for variable-length handling)

### Training Details

- **Pre-training data**: 2.3 trillion tokens (general text, code, math, multilingual)
- **Compute**: 0.13 million H800 GPU hours
- **Learning rate**: Linear warmup to 4e-4 over 2000 steps, held at 4e-4, decay to 1e-4 after 1.2T tokens, linear decay to 1e-5 for final 0.3T tokens
- **SFT**: 4.5 million instruction pairs. Only response tokens can be masked (prompt tokens stay visible). Standard cross-entropy on masked response tokens.

### Inference (Reverse Process)

1. Start with fully masked sequence (all [MASK])
2. For T steps (e.g., T=64 or T=128):
   - Run model on current sequence
   - Get predicted token probabilities for all masked positions
   - Unmask a fraction of positions (those with highest confidence)
   - **Low-confidence remasking**: positions that were unmasked but have low confidence can be re-masked for the next step
3. Prompt handling: prompt tokens are never masked, only response tokens go through diffusion

### Key Results

| Benchmark | LLaDA 8B | LLaMA3 8B |
|-----------|----------|-----------|
| MMLU | Competitive | Baseline |
| ARC-C | Competitive | Baseline |
| GSM8K | Competitive | Baseline |
| HumanEval | Competitive | Baseline |
| Reversal Poem | **Surpasses GPT-4o** | N/A |

LLaDA uniquely solves the **reversal curse** -- it can complete poems given later lines because it's not constrained to left-to-right generation.

### What Worked / Didn't

**Worked**:
- Standard transformer architecture (minimal changes from AR)
- Scaling laws hold: performance improves predictably with scale
- SFT works naturally (just mask response tokens)
- Bidirectional attention gives advantages for some tasks (reversal, infilling)

**Didn't work as well**:
- No KV caching support (cannot use GQA optimization)
- Inference is slower than AR for same model size (need multiple denoising steps)
- Still slightly behind AR on some reasoning benchmarks

---

## 4. LLaDA 2.1 (Token Editing)

**Paper**: "LLaDA2.1: Speeding Up Text Diffusion via Token Editing" (Feb 2026)
**Authors**: LLaDA team
**arXiv**: 2602.08676

### Core Idea

Standard masked diffusion only does Mask-to-Token (M2T) transitions: a [MASK] gets replaced with a real token, and that decision is final. LLaDA 2.1 adds **Token-to-Token (T2T) editing**: already-placed tokens can be swapped for better ones as context improves. This is like a human drafting text and then revising it.

### Two Operations

1. **M2T (Mask-to-Token)**: Standard unmasking. A [MASK] position gets filled with a predicted token. This is "drafting."
2. **T2T (Token-to-Token)**: An already-committed token gets replaced with a different, better token. This is "editing."

### Threshold Decoding

At each denoising step, two sets are identified:

- **Unmasking Set (Gamma_t)**: Masked positions where predicted token probability exceeds tau_mask
- **Editing Set (Delta_t)**: Non-masked positions where the top predicted token differs from current token AND exceeds tau_edit

State evolution:
```
x^i_{t-1} = v^i_t    if i in Gamma_t union Delta_t
x^i_{t-1} = x^i_t    otherwise
```

### Two Operating Modes

1. **Speedy Mode (S Mode)**: Low tau_mask (aggressively accept drafts), rely on T2T editing to fix mistakes later. Tokens-per-forward nearly doubles (5.93 vs 3.08 TPF).
2. **Quality Mode (Q Mode)**: Conservative thresholds for both M2T and T2T. Better benchmark scores.

### Training Objective

Unified mixture of M2T and T2T losses:
- **M2T stream**: Standard masked token prediction (drafting)
- **T2T stream**: Recover correct tokens from randomly perturbed inputs (editing capability)
- **MTF (Multi-Turn Forward)** data augmentation: expose model to diverse editing scenarios

### RL Integration (EBPO)

ELBO-based Block-level Policy Optimization:
```
J_EBPO(theta) = E[min(rho(y|x) * A_hat, clip(rho(y|x), 1-eps_low, 1+eps_high) * A_hat)]
```

Uses block-level log-likelihood contributions with vectorized likelihood estimation.

### Speed Results

| Model | TPS (HumanEval+) | TPS (BigCodeBench) |
|-------|-------------------|--------------------|
| LLaDA2.1-Flash (100B) | 891.74 | 801 |
| LLaDA2.1-Mini (16B) | 1586.93 | - |
| Qwen3-30B-A3B (AR) | 240 | - |

### What Worked / Didn't

**Worked**:
- T2T editing significantly improves both speed AND quality (not a tradeoff)
- Speed mode enables rough-then-refine workflow
- RL training stabilizes with EBPO
- Multi-block editing enhances reasoning and coding

**Didn't**:
- Additional complexity in training (need M2T + T2T streams)
- Threshold tuning (tau_mask, tau_edit) requires careful calibration

---

## 5. Mercury (Inception Labs)

**Paper**: "Mercury: Ultra-Fast Language Models Based on Diffusion" (Jun 2025)
**Authors**: Inception Labs
**arXiv**: 2506.17298

### Core Idea

**Block diffusion**: divide the output sequence into fixed-size blocks. Generate each block via masked diffusion (parallel token prediction within the block), but process blocks sequentially (each block is conditioned on all previous blocks). This combines the parallelism of diffusion within blocks with the sequential coherence of autoregressive generation across blocks.

### Block Diffusion Mechanism

1. **Divide sequence into blocks** of fixed size B
2. For each block (sequentially):
   a. Initialize block as all [MASK] tokens
   b. Run K denoising steps:
      - Forward pass predicts all tokens in the block in parallel
      - Unmask tokens based on confidence (highest confidence first)
   c. Cache the block's KV states for subsequent blocks
3. Each block is conditioned on all previously generated blocks via KV cache

### Training

Two-pass transformer regime:
1. **First pass**: Compute and cache keys, values, hidden states for the whole sequence
2. **Second pass**: Denoise each block's masked tokens independently, conditioning on cached context from preceding blocks

The model is trained with a masking-based corruption process. Noise schedule calibrated for language tokens: corruption probability increases from 0 (clean) to 1 (fully masked).

### Architecture

- Transformer backbone (standard architecture)
- Mercury Coder Mini and Mercury Coder Small
- Supports **KV caching** (unlike pure dLLMs like LLaDA) because blocks are processed sequentially
- The hybrid block approach enables standard AR optimizations

### Key Results

| Model | TPS (H100) | Quality |
|-------|------------|---------|
| Mercury Coder Mini | 1109 | Competitive |
| Mercury Coder Small | 737 | Competitive |
| Mercury 2 | ~1000 | AIME 91.1, GPQA 73.6 |
| Typical AR frontier | ~100-200 | Comparable |

Mercury achieves **5-10x** throughput improvement over comparable AR models.

### What Worked / Didn't

**Worked**:
- Block decomposition enables KV caching (key practical advantage)
- Confidence-ordered unmasking within blocks maintains coherence
- Hybrid approach gets best of both worlds: parallel within block, sequential across blocks
- Commercial-scale deployment proves viability

**Didn't**:
- Block size is a hyperparameter that affects quality/speed tradeoff
- Still need multiple denoising steps per block
- Complexity of implementation vs pure AR

---

## 6. d1 (Reasoning via Reinforcement Learning)

**Paper**: "d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning" (Apr 2025)
**Authors**: Zhao, Gupta, Zheng, Grover (UCLA)
**Code**: https://github.com/dllm-reasoning/d1

### Core Idea

Apply reinforcement learning to pre-trained masked dLLMs (specifically LLaDA 8B) to improve reasoning capabilities. The key challenge: existing RL algorithms (PPO, GRPO) require computing log-probabilities of generated sequences, which dLLMs cannot do directly (they don't factorize probability autoregressively). Solution: **diffu-GRPO**, a novel RL algorithm using mean-field approximation.

### The Log-Probability Problem

AR models compute log p(sequence) = sum of log p(token_i | previous tokens). dLLMs generate all tokens simultaneously via iterative denoising -- there's no natural sequential factorization.

### Mean-Field Approximation

Approximate the joint log-probability as:

```
log p(y | x) ~ sum_{i=1}^{L} log p_theta(y_i | x_masked)
```

Where x_masked is the prompt with stochastic masking applied. This decomposes the joint probability into independent per-token contributions.

### diffu-GRPO Algorithm

1. **Generate completions**: Run multiple denoising trajectories for each prompt
2. **Compute rewards**: Use task-specific reward function (e.g., correctness for math)
3. **Estimate log-probabilities**: Use mean-field approximation with random prompt masking
4. **Policy gradient update**: GRPO-style update using estimated advantages

**Random prompt masking**: During log-probability estimation, randomly mask parts of the prompt. This creates perturbed views of the same (prompt, completion) pairs, serving as regularization. Crucially, this allows scaling the number of inner gradient updates (mu) to much higher values while maintaining stability, reducing computational cost.

### Training Pipeline

1. **Masked SFT**: Distill reasoning knowledge from curated dataset (s1k: 1000 high-quality reasoning questions with step-by-step solutions). Standard masked diffusion SFT.
2. **diffu-GRPO**: Apply RL with reward signals from task correctness.

### Key Results

| Benchmark | LLaDA 8B (base) | d1-LLaDA 8B | Improvement |
|-----------|-----------------|-------------|-------------|
| GSM8K | Baseline | +3.9% | Modest |
| MATH500 | Baseline | +4.0% | Modest |
| Countdown | Baseline | +26.2% | Large |
| Sudoku | Baseline | +10.0% | Large |

### What Worked / Didn't

**Worked**:
- Mean-field approximation is tractable and effective
- Random prompt masking is critical for stable RL training
- Larger gains on logical reasoning tasks (Countdown, Sudoku) than pure math
- First successful RL integration with dLLMs

**Didn't**:
- Modest improvements on GSM8K/MATH (possibly because base model already decent)
- Mean-field is an approximation -- joint probability estimation is inherently noisy
- Requires careful tuning of masking ratio for prompt masking

---

## 7. TiDAR (Think in Diffusion, Talk in Autoregression)

**Paper**: "TiDAR: Think in Diffusion, Talk in Autoregression" (Nov 2025)
**Authors**: Jingyu Liu et al.

### Core Idea

A **hybrid architecture** that drafts tokens using diffusion (parallel, fast) and then verifies/samples final outputs using autoregressive decoding (sequential, high quality) -- all in a single forward pass. This exploits free GPU compute slots: modern GPUs are memory-bound during AR decoding, so the extra diffusion computation fits into "free" compute slots without adding latency.

### How It Works

1. **Prefix section**: Standard causal (AR) attention on already-accepted clean tokens
2. **Diffusion section**: Bidirectional attention on [MASK] tokens within each block
3. **Single forward pass**:
   - Current step's tokens undergo rejection sampling against AR distribution
   - Next step's tokens are simultaneously pre-drafted via diffusion
   - Results handed off regardless of acceptance length

### Structured Attention Masks

Novel hybrid attention pattern within the same transformer:
- Clean prefix tokens: causal attention (can only see past)
- Mask tokens: bidirectional attention within block (see all mask tokens + prefix)
- This enables simultaneous AR verification + diffusion drafting

### Training

- **Continual pretraining** from existing AR models (Qwen2.5 1.5B, Qwen3 8B)
- Combined loss: `L_TiDAR = (1/(1+alpha)) * [L_AR + alpha * L_diffusion]`
- **Full masking strategy**: All diffusion tokens set to [MASK] during training (not random masking)
- 50B tokens for 1.5B model, 150B for 8B
- BFloat16, cosine LR schedule, max LR 1e-5

### Architecture

- Supports exact KV caching (unlike pure dLLMs)
- One-step diffusion at inference (single denoising step)
- Generates 7-8 tokens per network evaluation

### Key Results

| Model | Speedup | HumanEval | GSM8K |
|-------|---------|-----------|-------|
| TiDAR 1.5B | 4.71x | 43.29% | 53.90% |
| Qwen2.5 1.5B (AR) | 1x | 48.17% | 54.74% |
| TiDAR 8B | 5.91x | - | - |
| Qwen3 8B (AR) | 1x | - | - |

First architecture to close the quality gap with AR while delivering 4.7-5.9x speedup.

### What Worked / Didn't

**Worked**:
- Exploiting free GPU compute slots (zero-overhead parallel computation)
- Full masking during training (denser loss signal than random masking)
- KV cache support (unlike pure dLLMs)
- Single denoising step is sufficient when combined with AR verification

**Didn't**:
- Still slight quality gap on some benchmarks
- Requires continual pre-training from AR checkpoint
- Block size tuning required

---

## 8. Discrete Flow Matching

**Paper**: "Discrete Flow Matching" (NeurIPS 2024)
**Authors**: Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, et al. (Meta FAIR)

### Core Idea

Extends continuous flow matching to discrete data (text). Instead of learning a denoising model, learn a **probability velocity field** that governs how probability mass flows from a source distribution (noise) to the target distribution (data). This provides a more general framework than masking-based diffusion.

### How It Differs from Masking-Based Approaches

| Aspect | Masking (MDLM/LLaDA) | Flow Matching |
|--------|----------------------|---------------|
| Noise process | Tokens -> [MASK] | Probability flow between distributions |
| Training | Predict original token | Match velocity field |
| Framework | Absorbing CTMC | General probability paths |
| Flexibility | Fixed masking process | Arbitrary couplings & schedulers |
| Complexity | Simpler | More general but more complex |

### Probability Path

```
p_t(x | x_0, x_1) = sum_j kappa_{i,j}(t) * w_j(x | x_0, x_1)
```

Interpolates between source (x_0, noise) and target (x_1, data) through a family of conditional probability paths parameterized by schedulers.

### Velocity Field

The probability velocity governing mass flow:

```
u_t(x_i, z) = sum_j a_{i,j}(t) * w_tilde_j(x_i | x_0, x_1) + b_i(t) * delta_z(x_i)
```

Satisfies a discrete continuity equation: `dot{p}_t(x) + div_x(p_t * u_t) = 0`

### Sampling

Supports both x-prediction (probability denoiser) and epsilon-prediction (noise prediction), analogous to continuous flow matching. Can use different schedulers to trade off quality and speed.

### Key Results

Scaled to 1.7B parameters:
- HumanEval: 6.7% Pass@1, 13.4% Pass@10
- MBPP (1-shot): 6.7% Pass@1, 20.6% Pass@10

### What Worked / Didn't

**Worked**:
- Most general framework -- subsumes MDLM and SEDD as special cases
- Enables new scheduling strategies
- Theoretically elegant unification

**Didn't**:
- More complex to implement than MDLM
- Results are not dramatically better than simpler masking approaches
- Practical advantages over MDLM are limited for most use cases

### For Our Implementation

For a minimal educational implementation, **masking-based approaches (MDLM/LLaDA style) are preferred** over flow matching. The conceptual overhead of flow matching doesn't translate to meaningful practical gains at small scale.

---

## 9. tiny-diffusion Reference Implementation

**Repository**: https://github.com/nathan-barry/tiny-diffusion
**Author**: Nathan Barry

### Overview

Character-level masked diffusion model trained on Tiny Shakespeare. 10.7M parameters. **364 lines of code** in a single file. ~80% identical to a GPT implementation (312 lines), differing in only **19 lines of code** across two functions (`get_batch` and `generate`).

### The 5 Key Changes from GPT

1. **Add mask token to vocabulary** (`_`): `chars = ["_"] + chars`
2. **Bidirectional attention**: `is_causal=False` (instead of `is_causal=True`)
3. **Confidence-based parallel decoding** (instead of sequential left-to-right)
4. **Training objective: denoising** (predict original tokens at masked positions, not next token)
5. **Loss only on masked tokens**: `loss = (loss * mask_flat).sum() / mask_flat.sum()`

### get_batch Implementation

```python
def get_batch(split):
    # Sample random sequences from data
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = x.clone()  # original tokens = targets

    # Random masking ratio per sample (LLaDA-style uniform)
    mask_probs = torch.rand(batch_size, 1)  # t ~ U[0,1]
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id  # replace with [MASK]

    return x, y, mask
```

Key: Each sample gets a random masking ratio t ~ U[0,1], then each token is independently masked with probability t.

### generate Implementation

Confidence-based parallel decoding in blocks:

```python
1. Start with prompt + block of [MASK] tokens
2. While any positions are masked:
   a. Forward pass -> get logits
   b. Compute confidences (sum of top-k probabilities)
   c. Unmask positions where confidence >= threshold (0.95)
   d. If none exceed threshold, unmask the single most confident position
   e. Sample from top-k distribution at unmasked positions
3. Move to next block (slide prompt window)
```

### Architecture

- 6 layers, 6 heads, 384 embedding dim
- RoPE positional embeddings
- RMSNorm (functional, no learnable params)
- SwiGLU-like activation: `F.relu(x).square()` (ReGLU variant)
- No bias in linear layers
- 10.7M parameters

### Training

- 10,000 iterations (~20 min on A100) -- 2x longer than GPT equivalent
- Reason: only masked tokens contribute to loss, so ~2x less signal per batch
- AdamW optimizer, lr=3e-4
- Block size 256, batch size 64

### Key Insight

The diffusion model trains for **twice as long** because only ~half the tokens are masked on average (mask ratio ~ U[0,1], so E[mask_ratio] = 0.5), meaning only half the tokens contribute to the loss in each batch.

---

## 10. Minimum Viable dLLM

### Absolute Minimum Components

To build the simplest possible dLLM from scratch:

1. **Tokenizer**: Character-level (simplest) or BPE
2. **Vocabulary**: Original vocab + 1 [MASK] token
3. **Model**: Standard transformer with bidirectional attention (just set `is_causal=False`)
4. **Forward process**: Random masking with uniform ratio t ~ U[0,1]
5. **Loss**: Cross-entropy on masked positions only
6. **Sampling**: Iterative unmasking (confidence-based or fixed-fraction-per-step)

### Training Loop (Pseudocode)

```
for each batch:
    x = sample_sequences()           # clean text
    t = uniform(0, 1)                # random masking ratio
    mask = random(shape) < t         # per-token mask
    x_noisy = x.copy()
    x_noisy[mask] = MASK_TOKEN       # apply noise

    logits = model(x_noisy)          # bidirectional transformer
    loss = cross_entropy(logits[mask], x[mask])  # loss on masked only
    loss.backward()
    optimizer.step()
```

### Sampling Loop (Pseudocode)

```
x = all_mask_tokens(length)          # start fully masked
for step in range(num_steps):
    logits = model(x)
    probs = softmax(logits)

    # Determine how many to unmask this step
    n_unmask = length // num_steps    # or use confidence threshold

    # Find most confident masked positions
    confidences = probs.max(dim=-1)
    top_positions = topk(confidences[masked_positions], n_unmask)

    # Sample tokens at those positions
    x[top_positions] = sample(probs[top_positions])
```

### What You Don't Need

- No noise schedule parameter (uniform masking ratio works)
- No timestep conditioning in the model (optional, tiny-diffusion doesn't use it)
- No CTMC theory
- No flow matching
- No special parameterization (SUBS helps but isn't required for a first version)
- No remasking at inference (simpler: once unmasked, stay unmasked)

### What Helps But Isn't Required

- **Timestep conditioning**: Tell the model what masking ratio was used. MDLM showed that removing time conditioning gives 2x inference speedup with minimal quality loss.
- **SUBS parameterization**: Force mask logit to -inf and copy through unmasked tokens. Reduces trivial errors.
- **Low-discrepancy sampling**: Use quasi-random timesteps across the batch for lower variance.
- **RoPE**: Better than learned positional embeddings for generalization.

---

## 11. Common Pitfalls

### Training Pitfalls

1. **Loss only on masked tokens**: Forgetting to mask the loss leads to the model learning to copy input tokens (trivial solution). Only masked positions should contribute to the loss.

2. **Causal attention left on**: The single most impactful change is switching from causal to bidirectional attention. If you leave causal masking, the model can only see left context and loses the key advantage of diffusion.

3. **Mask token in predictions**: The model should never predict [MASK] as an output token. Set the mask token logit to -infinity in the output.

4. **Training twice as long**: Since only ~50% of tokens contribute to loss on average (E[t] = 0.5), you need roughly 2x the iterations to see the same number of effective training tokens as AR.

5. **Numerical stability**: Compute cross-entropy only over masked indices. Computing over all positions and then masking can cause numerical issues.

6. **Noise schedule doesn't matter much**: MDLM proved the ELBO is invariant to the schedule. Don't over-engineer this. Uniform masking ratio t ~ U[0,1] is fine.

### Inference Pitfalls

7. **Too few denoising steps**: Quality degrades sharply below ~16 steps. Start with 64-128 steps and reduce.

8. **Fixed unmasking fraction**: Using a fixed fraction per step (e.g., unmask 1/T positions each step) is worse than confidence-based unmasking. High-confidence tokens should be unmasked first.

9. **No remasking**: Once you unmask a token, errors propagate. Consider remasking low-confidence tokens for refinement (but this adds complexity).

10. **Temperature**: Unlike AR models that often need temperature < 1.0 for coherent text, dLLMs (especially SEDD) generate faithful text without temperature scaling.

### Architecture Pitfalls

11. **Small vocabulary**: D3PM used 8k vocabulary and had terrible perplexity. Use standard BPE tokenizers (32k-50k vocab) for competitive results.

12. **No KV caching**: Pure dLLMs (LLaDA-style) cannot use KV caching because attention is bidirectional and the sequence changes each step. This makes inference slower per step. Mercury's block diffusion solves this.

13. **Wrong positional encoding**: Absolute positional encodings don't generalize well for variable-length generation. Use RoPE.

---

## Sources

### Papers
- MDLM: https://arxiv.org/abs/2406.07524 | Code: https://github.com/kuleshov-group/mdlm
- SEDD: https://arxiv.org/abs/2310.16834 | Code: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- LLaDA: https://arxiv.org/abs/2502.09992 | Code: https://github.com/ML-GSAI/LLaDA
- LLaDA 2.1: https://arxiv.org/abs/2602.08676
- Mercury: https://arxiv.org/abs/2506.17298
- d1: https://arxiv.org/abs/2504.12216 | Code: https://github.com/dllm-reasoning/d1
- TiDAR: https://arxiv.org/abs/2511.08923
- Discrete Flow Matching: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html

### Implementations
- tiny-diffusion: https://github.com/nathan-barry/tiny-diffusion
- MDLM project page: https://s-sahoo.com/mdlm/
- LLaDA demo: https://ml-gsai.github.io/LLaDA-demo/
- d1 project page: https://dllm-reasoning.github.io/
- TiDAR project page: https://tidarlm.github.io/
- Inception Labs: https://www.inceptionlabs.ai/

### Blog Posts & Explainers
- SEDD blog by Aaron Lou: https://aaronlou.com/blog/2024/discrete-diffusion/
- dLLM overview: https://spacehunterinf.github.io/blog/2025/diffusion-language-models/
- LLaDA explainer: https://medium.com/@krishnanp2001/llada-large-language-diffusion-model-with-masking-vs-autoregressive-language-models-2c01844b2f39
- LLaDA deep dive (TDS): https://towardsdatascience.com/llada-the-diffusion-model-that-could-redefine-language-generation/
- LLaDA 2.1 review: https://qubytes.substack.com/p/llada21-introduces-a-draft-and-edit
