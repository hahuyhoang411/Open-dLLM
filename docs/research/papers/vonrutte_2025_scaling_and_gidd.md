# Von Rutte et al. 2025: Scaling Laws + GIDD

Two papers from the same ETH Zurich group (von Rutte, Fluri, Orvieto, Scholkopf, Hofmann) that form a theoretical-empirical pair: GIDD gives you the math, Scaling gives you the compute curves. The scaling paper explicitly builds on the GIDD framework by reparameterizing it through log-SNR.

---

# Paper 1: Scaling Behavior of Discrete Diffusion Language Models

**arXiv:2512.10858** | ICLR 2025
Von Rutte, Fluri, Pooladzandi, Scholkopf, Hofmann, Orvieto

## TIER 1 -- Summary

### One-Sentence Takeaway

Uniform diffusion LMs scale more parameter-hungry than masked (M* ~ C^0.589 vs C^0.566) but converge toward AR-competitive loss at 10^22 FLOPs, while all DLM variants require significantly more parameters per FLOP than Chinchilla-optimal AR models (C^0.49).

### Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Uniform M* exponent | 0.589 +/- 0.019 | vs Chinchilla AR 0.49 -- DLMs are 20% more parameter-hungry |
| Masked M* exponent | 0.566 +/- 0.022 | Closer to AR but still above DeepSeek's 0.5243 |
| Uniform L* exponent | -0.0522 +/- 0.00029 | Best loss scaling of any noise type tested |
| Likelihood gap at 10^21 FLOPs | 1.7% | Down from 3.2% at 10^18 FLOPs -- gap shrinks with scale |
| Optimal batch size exponent | B* ~ D^0.82 | Nearly linear in training tokens, independent of model size |

### Red Flags

1. **No absolute loss numbers for large runs.** The 3B and 10B models are validated by "closely matching predictions" but they never report the actual ELBO values. This makes it impossible to compare directly against AR baselines at the same FLOP budget. The claim that DLMs "match DeepSeek scaling trend" is based on visual curve alignment, not numeric comparison.

2. **Loss floor hand-waved away.** They fit power laws without constant offset (implicitly assuming irreducible loss is zero), then acknowledge this differs from AR practice. The justification -- that the offset is "too small to accurately model" -- is unconvincing at 510 runs. If you can't measure it, you can't claim it's negligible.

3. **Single dataset (Nemotron-CC).** They acknowledge "scaling coefficients can fluctuate across datasets depending on data composition" but only train on one. The 0% HumanEval and 2.43% GSM8k results at 10B params reflect data composition, not model capability, but they still report them.

4. **No generation quality evaluation.** For diffusion models that can generate in parallel, the paper never measures generation quality (perplexity of samples, MAUVE, etc.). This is the actual selling point of DLMs and it's entirely absent.

5. **Sub-epoch regime only.** All experiments use single-pass training. Modern practice involves multi-epoch, and they explicitly disclaim generalization to that setting.

### Bottom Line

**Implement (the methodology), cite (the exponents), read carefully (the batch size analysis).** The CompleteP + iso-FLOP methodology is the real contribution -- clean hyperparameter transfer without annealing. The headline claim about DLMs being "competitive at scale" rests on extrapolation, not observation. Still, this is the first serious Chinchilla-style analysis for discrete diffusion and the exponents are useful for planning compute budgets.

---

## TIER 2 -- Details

### Section-by-Section

**Section 2 -- GIDD Background + SNR Reformulation.**
The key move: reparameterize the GIDD ELBO from time t to log-SNR lambda = log(alpha / (1 - alpha)). This gives noise-schedule invariance (the bound depends only on SNR, not on how you traverse it). They define a sigmoid-controlled mixing distribution:

pi_lambda = sigma(a*lambda + b) * u + (1 - sigma(a*lambda + b)) * m

where u is uniform over vocabulary, m is the mask token. Parameters a and b control where the transition from masking to uniform noise happens. For "balanced" hybrid: a=1, b=0 (transition at SNR=1).

This is elegant. It means you get a continuum of noise types parameterized by two scalars, and the ELBO doesn't care which schedule you use to traverse the SNR range.

**Section 3 -- CompleteP Implementation.**
They adopt CompleteP (Dey et al., 2025), a variant of muP that transfers learning rates across both width AND depth. Key hyperparameters found empirically:

- sigma_base = 0.4 (bulk parameter init variance)
- sigma_aux = 0.02 (auxiliary parameter init variance)
- eta_base = 0.3 (base learning rate at batch size 64)
- eta_aux = 0.02 * eta_base

The critical practical insight: "optimal batch size being a function of dataset size, optimal learning rate being a function of (optimal) batch size, and both being largely independent of model size." This means you can sweep batch size and learning rate once at small scale, then transfer to large models. They estimate this saves ~2.45% of total scaling-law compute.

**Section 4 -- Scaling Laws.**
Methodology: 510 total runs across 5 noise types, 5 model sizes (25M to 567M non-embedding params), 7 batch sizes per model (2^14 to 2^20 tokens), 2-3 learning rates per batch size. Sequence length 2048. Iso-FLOP profiles a la Hoffmann et al. "Approach 2."

Five noise types tested, forming a spectrum:
1. Masked (pi = m always)
2. Low-uniform (slight uniform mixing)
3. Balanced (a=1, b=0)
4. High-uniform (more uniform)
5. Uniform (pi = u always)

Full exponent table:

| Noise Type | M* ~ C^a | D* ~ C^b | L* ~ C^c |
|------------|----------|----------|----------|
| Masked | 0.566 | 0.434 | -0.0496 |
| Low-uniform | 0.535 | 0.465 | -0.0509 |
| Balanced | 0.534 | 0.466 | -0.0512 |
| High-uniform | 0.573 | 0.427 | -0.0514 |
| Uniform | 0.589 | 0.411 | -0.0522 |
| Chinchilla (AR) | 0.49 | 0.51 | -- |
| DeepSeek (AR) | 0.524 | 0.476 | -- |

Fascinating non-monotonicity: the parameter exponent dips at low-uniform/balanced, then climbs for high-uniform/uniform. The loss exponent monotonically improves toward uniform. This suggests a phase transition in what the model needs to learn.

**Section 4.5 -- Batch Size Scaling.**
The batch size-step count relationship follows a hyperbola:

(floor(S/S_min)^alpha - 1) * (floor(B/B_min)^alpha - 1) = 1

with alpha ~ 0.1-0.2. The optimal batch size B* ~ D^0.82 (nearly linear in training tokens). The optimal learning rate eta* ~ B*^0.34. Both are independent of model size -- a huge practical win.

**Section 5 -- Large-Scale Validation.**
3B model at 10^21 FLOPs: predictions match observations, gap between masked and uniform drops from 3.2% to 1.7%.
10B model at 10^22 FLOPs (uniform only): "matches the scaling trend of DeepSeek."

Downstream results for 10B uniform model:
- ARC-E: 61.8%, ARC-C: 35.7%, WinoGrande: 55.5%, PIQA: 66.3%
- OpenBookQA: 32.8%, BoolQ: 60.3%, GSM8k: 2.43%
- HumanEval: 0% (no code in training data)

The GSM8k and HumanEval numbers are embarrassing but fair -- Nemotron-CC has no math/code data.

### Key Equation

The SNR-reparameterized ELBO:

```
-log p(x) <= E_{lambda, z}[ w_lambda(x)_z / p(lambda) * (D_KL + D_IS) ]
```

| Variable | Meaning |
|----------|---------|
| lambda | log-SNR = log(alpha / (1 - alpha)), the noise level |
| z | noised token at SNR level lambda |
| w_lambda | importance weight for token z at noise level lambda |
| p(lambda) | noise schedule density (cancels in expectation -- schedule invariance) |
| D_KL | KL divergence between data and model posteriors |
| D_IS | residual term (Itakura-Saito-like: a/b - log(a/b) - 1) |

The schedule invariance is the key insight: p(lambda) in denominator cancels with the sampling distribution. You can use any schedule and get the same bound. This means schedule choice is purely a variance-reduction decision, not a bias decision.

### Tricks Worth Reusing

1. **CompleteP for DLMs.** The hyperparameter transfer recipe (sigma_base=0.4, sigma_aux=0.02, eta_base=0.3) is directly applicable. If you're doing muP already, this is a strict upgrade.

2. **Batch size ~ tokens, not model size.** For compute budgeting: scale batch size with dataset, not with model. At B* ~ D^0.82, if you double your data, increase batch ~1.77x.

3. **SNR parameterization for hybrid noise.** The sigmoid mixing pi_lambda = sigma(a*lambda+b)*u + (1-sigma(a*lambda+b))*m is a clean two-parameter family. Worth adopting even if you only use pure masking -- it's a free axis of variation for ablations.

4. **No learning rate annealing needed for scaling law estimation.** They claim this saves 2.45% of compute. Small but meaningful at scale.

### Critical Analysis

**Baseline fairness.** The comparison to Chinchilla/DeepSeek is apples-to-oranges: different datasets, different tokenizers (131K vocab vs 32K/100K), different architectures. The exponent comparison is valid in spirit but not in absolute loss terms.

**Statistical rigor.** The 2-sigma confidence intervals on exponents are commendable (most scaling law papers don't report them). The intervals are tight enough to be meaningful. 510 runs is substantial.

**Reproducibility.** Good: they report all hyperparameters, use a public dataset (Nemotron-CC), and specify the CompleteP recipe. Bad: no code release mentioned, the custom 131K BPE tokenizer is not described in enough detail to reproduce, and the 3B/10B results lack absolute numbers.

**Missing analysis.** No discussion of inference compute. DLMs trade training efficiency for inference parallelism, but the scaling laws only cover training. At what scale does the inference advantage kick in? Unanswered.

---

## TIER 3 -- Expand

### Field Context

This paper sits at the intersection of two lines of work:
- **Scaling laws** (Kaplan 2020, Hoffmann/Chinchilla 2022, DeepSeek 2024): established that AR models have predictable scaling behavior. This is the first equivalent analysis for discrete diffusion.
- **Discrete diffusion LMs** (Austin D3PM 2021, Campbell CTMC 2022, Sahoo MDLM 2024, Shi SEDD 2024, Arriola BD3-LMs 2025): the model family being scaled. The paper uses the GIDD framework from their own companion paper.

The key predecessor is Hoffmann et al. (Chinchilla): same iso-FLOP methodology, same goal (compute-optimal allocation between parameters and data). The DLM-specific contribution is showing that noise type fundamentally changes the optimal allocation.

### Replication Blueprint

- **Small-scale sweep (to fit exponents):** 5 model sizes x 5 noise types x 7 batch sizes x 2-3 LRs = ~510 runs. Model sizes: 25M, ~60M, ~130M, ~270M, 567M (non-embedding). Each run: enough steps to reach target FLOP budget. Estimated cost: order of 10^19 total FLOPs across all runs.
- **Large-scale validation:** 3B at 10^21 FLOPs, 10B at 10^22 FLOPs. These are expensive: 10^22 FLOPs ~ 1000 A100-hours at half precision.
- **Key settings:** Seq length 2048, vocab 131K (custom BPE on Nemotron-CC), CompleteP with stated hyperparameters, Adam with beta=(0.9, 0.99).
- **Hardware not specified** in the fetched content, but 510 runs at this scale likely required significant cluster time (estimate: 2000-5000 A100-hours for the full sweep).

### Phase 2 Reading List

1. **Hoffmann et al. 2022 (Chinchilla)** -- The direct AR precedent for this methodology.
2. **Dey et al. 2025 (CompleteP)** -- The muP variant they use. Critical for understanding their hyperparameter transfer.
3. **Su et al. 2024 (Nemotron-CC)** -- The training dataset. Composition matters for downstream results.
4. **McCandlish et al. 2018** -- The batch size scaling framework they build on (the hyperbola equation).
5. **The companion GIDD paper (2503.04482)** -- For the theoretical foundation underneath the SNR reparameterization.

---

# Paper 2: Generalized Interpolating Discrete Diffusion (GIDD)

**arXiv:2503.04482** | ICML 2025
Von Rutte, Fluri, Ding, Orvieto, Scholkopf, Hofmann

## TIER 1 -- Summary

### One-Sentence Takeaway

GIDD provides a clean continuous-time framework that unifies masked and uniform discrete diffusion through a time-varying mixing distribution pi_t, achieving 22.29 PPL (vs 24.37 for MDM reimplementation) at 92M params, and discovering that uniform noise (p_u >= 0.1) enables self-correction (up to 55% generative PPL improvement) at the cost of worse raw likelihood.

### Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| GIDD+ (p_u=0.0) val PPL | 22.29 | vs MDM reimpl 24.37, vs Llama-110M 16.11 -- still 38% behind AR |
| Self-correction improvement | up to 55% | Generative PPL drop via fixed-point iteration, only works with p_u >= 0.1 |
| 7-task downstream average | 40.46% | GIDD+ base model, vs GPT2-small 38.77%, vs retrained Llama 41.04% |
| p_u=0.2 overtakes p_u=0.0 | ~10^21 FLOPs (projected) | "Highly unreliable" prediction per the authors themselves |
| Global minimum guarantee | ELBO = 0 (+ constant C) | Iff model perfectly matches data -- schedule doesn't limit performance |

### Red Flags

1. **The 22.29 PPL is GIDD+ with p_u=0.0, which is... masked diffusion.** The best-performing variant is just a better-engineered masked diffusion model. The interesting uniform noise variants (p_u > 0) actually hurt likelihood: p_u=0.2 gives 24.38 PPL at 262B tokens. The self-correction capability is real, but you pay for it in raw quality.

2. **Self-correction evaluation uses GPT2-large as grader.** The authors themselves flag this: GPT2-large assigns high likelihood to low-diversity repetitive text. They recommend Gemma 2 9B but don't use it for their main results. The 55% improvement number should be taken with a grain of salt.

3. **Scaling experiments stop at 3.3x10^20 FLOPs.** That's "2 orders of magnitude below modern LLM training." The prediction that p_u > 0 overtakes at 10^21 FLOPs is extrapolation they call "highly unreliable."

4. **No comparison with SEDD or BD3-LMs.** The baseline is their own MDM reimplementation. We don't know how GIDD compares to other recent discrete diffusion approaches.

### Bottom Line

**Read (the theory), cite (the framework), implement (selectively).** The theoretical contribution -- unifying masked and uniform under one ELBO with schedule invariance -- is genuine and will likely become the standard reference formulation. The self-correction discovery is the most interesting empirical finding. But if you just want the best DLM today, GIDD+ at p_u=0.0 is basically an improved MDM. The uniform noise regime is a research bet on scale, not a current win.

---

## TIER 2 -- Details

### Section-by-Section

**Section 3.1 -- Forward Process.**
The core generalization: define a forward process where each token independently transitions via

q_t(z_t | x) = Cat(z_t; alpha_t * x + beta_t * pi_t)

where x is the one-hot data token, alpha_t is the signal retention probability decaying from 1 to 0, and pi_t is an arbitrary time-varying probability distribution over the vocabulary. When pi_t = m (mask token), you recover masked diffusion. When pi_t = u (uniform), you get uniform diffusion. Any mixture works.

The conditional transitions (Proposition 3.3) follow as:

Q_{t|s} = alpha_{t|s} * I + beta_{t|s} * pi_{t|s} * 1^T

where alpha_{t|s} = alpha_t / alpha_s, and beta_{t|s} * pi_{t|s} = beta_t * pi_t - (alpha_t / alpha_s) * beta_s * pi_s. This is a rank-1-plus-identity matrix -- computationally cheap.

**Section 3.2 -- ELBO Derivation (Theorem 3.7).**
The continuous-time negative ELBO decomposes into a weighted expectation:

-log p(x) <= E_{t, z_t}[ w_t(z_t, x) * (D_KL(q_t(z_s|x) || q_t(z_s|x_theta)) + r_theta(z_t, x)) ] + E_t[alpha_t'/alpha_t] + C

The weight function:

w_t(z_t, x) = [1 / q_t(z_t | x)] * z_t^T * ((beta_t * pi_t)' - (alpha_t'/alpha_t) * beta_t * pi_t)

The residual term:

r_theta(z_t, x) = q_t(z_t | x) / q_t(z_t | x_theta) - log(q_t(z_t | x) / q_t(z_t | x_theta))

This residual has the form a/b - log(a/b), which is the Itakura-Saito divergence (plus a constant -1). So the ELBO has a two-divergence structure: KL between posteriors (how well the model predicts clean tokens given noisy ones) plus an I-S-like term (how well the model predicts the noise level of the current token). The KL term dominates for masked tokens; the I-S term matters for uniform noise where the model must distinguish clean from noisy tokens.

**Proposition 3.9 -- Global Optimum.**
The CT-NELBO achieves zero (up to constant C) iff q_t(z_t | x) = q_t(z_t | x_theta) for all x, t, z_t. This means the mixing schedule pi_t does NOT limit achievable performance -- a perfect model scores equally well regardless of noise type. The practical gap is about model capacity, not theoretical ceiling.

**Section 4 -- Mixing Schedules.**
The concrete hybrid parameterization:

alpha_t = (1 - t) / C_t
beta_t * pi_t = (t / C_t) * m + (c_t / C_t) * 1

where:
- c_t = B * t^(gamma/2) * (1-t)^(gamma/2) controls uniform noise proportion
- C_t = 1 + N * c_t is the normalization
- B = 2^gamma * p_u / (N * (1 - p_u)) converts p_u (target uniform fraction) to the schedule parameter
- gamma = 1 (default)
- N = vocabulary size

Setting p_u = 0 recovers pure masked diffusion. Increasing p_u adds uniform noise, peaking at mid-diffusion (t=0.5 when gamma=1).

**Section 5 -- Loss Weighting.**
Raw ELBO weights w_t explode at extreme noise levels. Two solutions:

Clamped: w_clamp_t = min(w_max, w_t) with w_max = 1

Dynamic: w_dyn_t(z_t, x) = w_max * (1 + delta_{z_t, m} + (B * e^{-lambda_t/2} - 1) * delta_{z_t, x})

where lambda_t is the log-SNR. Dynamic weighting preserves relative weights between masked, clean, and noisy tokens. Table 3 shows this is critical for p_u > 0: without it, uniform noise variants degrade badly.

**Section 6 -- Self-Correction.**
The most surprising result. Models trained with p_u >= 0.1 can iteratively improve their own outputs:

1. Generate sample normally
2. For each token position: compute model's prediction at t=0
3. Sample alternative token with temperature tau in [0.1, 0.5]
4. If new token has higher model likelihood, replace
5. Repeat with patience 32 (stop if no improvement for 32 steps)
6. Up to 10% of tokens resampled per iteration

Result: generative PPL improves up to 55% for base models at p_u=0.2. Mask-only models (p_u=0.0) show zero self-correction ability. The authors argue this is because uniform noise training forces the model to learn which tokens are wrong (distinguishing clean from corrupted), while mask-only training only learns to fill blanks.

This is genuinely interesting. It means uniform noise DLMs have a built-in "editing" capability that masked DLMs lack. Whether this matters at scale -- where raw quality might be high enough that self-correction is unnecessary -- remains open.

### Key Equations

**The Forward Marginal (the foundational equation):**

```
q_t(z_t | x) = Cat(z_t; alpha_t * x + beta_t * pi_t)
```

| Variable | Meaning |
|----------|---------|
| z_t | Token state at time t (one-hot vector over vocabulary) |
| x | Original clean token (one-hot) |
| alpha_t | Signal retention probability, decays from 1 to 0 |
| beta_t | Noise injection probability = 1 - alpha_t |
| pi_t | Time-varying noise distribution (mask, uniform, or any mixture) |

Plain English: at time t, each token is either still clean (probability alpha_t) or has been replaced by a sample from pi_t (probability beta_t).

**The ELBO Two-Divergence Structure:**

```
ELBO term for token z_t ~ w_t * [D_KL(posterior_data || posterior_model) + r_theta(z_t, x)]
```

where r_theta = a/b - log(a/b) with a = q_t(z_t|x), b = q_t(z_t|x_theta)

| Term | What it measures |
|------|-----------------|
| D_KL | How well model predicts clean token given noisy observation |
| r_theta | How well model estimates the corruption state of the current token |

For masked diffusion (p_u=0), all noisy tokens are [MASK] and r_theta vanishes -- the model only needs to fill blanks. For uniform diffusion, r_theta is non-zero because the model must detect which tokens are corrupted. This is why uniform is "strictly harder."

### Tricks Worth Reusing

1. **Dynamic loss weighting (Eq. 9).** If you're training with any uniform noise, the clamped weighting is insufficient. The dynamic scheme preserves relative importance across token types and is easy to implement.

2. **Self-correction as free post-processing.** If your model was trained with p_u >= 0.1, you get iterative refinement for free at inference. No architecture changes needed. Temperature 0.1-0.5, patience 32, resample up to 10% of tokens.

3. **The rank-1-plus-identity transition matrix.** Q_{t|s} = alpha * I + beta * pi * 1^T means you never need to materialize a V x V matrix. All operations are O(V), not O(V^2). Essential for large vocabularies.

4. **Schedule invariance.** Once you parameterize by log-SNR, you can use any schedule for training and any (possibly different) schedule for inference. This decouples two design decisions that were previously entangled.

### Critical Analysis

**Baseline fairness.** The MDM baseline is their own reimplementation, not the original. It gets 24.37 PPL while GIDD+ (p_u=0.0) gets 22.29 -- a 9% improvement. But we don't know if their MDM reimplementation is faithful. No comparison with SEDD, BD3-LMs, or LLaDA.

**Statistical rigor.** No error bars on the main PPL results. The self-correction experiments seem to be single runs. The ablation table (Table 3) is useful but also lacks confidence intervals.

**Reproducibility.** Good: code released at github.com/dvruette/gidd/, DiT architecture fully specified (Tiny/Small/Base configs), training recipe detailed (Adam, lr=5e-4, warmup 10k, cosine decay, batch 512, seq 512, bfloat16). Bad: OpenWebText at 262B tokens -- need to verify this matches the standard OWT version.

**The AR gap.** Best GIDD+ at 92M params: 22.29 PPL. Retrained Llama at 110M params: 16.11 PPL. That's a 38% gap. On downstream tasks, GIDD+ base (321M) at 40.46% is competitive with GPT2-small (38.77%) but below retrained Llama (41.04%). The gap is real, persistent, and not closing fast at current scales.

---

## TIER 3 -- Expand

### Field Context

GIDD builds on the continuous-time discrete diffusion lineage:
- **Austin et al. 2021 (D3PM):** Discrete diffusion with structured transition matrices. GIDD generalizes beyond the specific noise types D3PM considered.
- **Campbell et al. 2022:** Extended to continuous time with CTMCs. GIDD's derivation follows this continuous-time approach.
- **Sahoo et al. 2024 (MDLM):** Showed masked diffusion is surprisingly competitive. GIDD recovers MDLM as a special case (Corollary 3.8) and improves on it.
- **Shi et al. 2024 (SEDD):** Score-based approach to discrete diffusion. Different theoretical angle, similar practical regime.
- **Ou et al. 2024:** Extended masked diffusion analysis. GIDD subsumes this work.

The key novelty: previous work either did masked-only (MDLM, SEDD) or uniform-only (D3PM absorbing). GIDD provides the first framework where you can smoothly interpolate and the theory stays clean (closed-form transitions, tractable ELBO, schedule invariance).

### Replication Blueprint

**Compute requirements:**
- Small model (92M, 131B tokens): ~10^19 FLOPs. On 8x A100: ~1-2 days.
- Base model (321M, 262B tokens): ~3.3x10^20 FLOPs. On 8x A100: ~1-2 weeks.
- Full ablation suite (all p_u values, all sizes): estimate 8-16x base cost.

**Key settings to reproduce:**
- Architecture: DiT (not GPT), L/H/d as specified per size
- Optimizer: Adam, beta=(0.9, 0.99), eps=1e-9
- LR: 5e-4, linear warmup 10k steps, cosine decay to 10%
- Batch size: 512 sequences, seq length 512
- Weight decay: 0.02 (GIDD+ only)
- Gradient clipping: norm 1.0
- Precision: bfloat16
- Data: OpenWebText, 262B tokens (need custom BPE tokenizer -- not specified if they share it)
- Dynamic weighting with w_max=1 for p_u > 0

**Pitfalls:**
- Numerical instability in sampling at extreme noise levels (Appendix G). They use Gumbel-max trick for stability.
- ELBO weight explosion requires the dynamic weighting scheme -- clamping alone is not enough for uniform noise.
- Self-correction only works if trained with p_u >= 0.1 from scratch; you cannot fine-tune a masked-only model to gain this capability (not tested, but implied by the mechanism).

### Phase 2 Reading List

1. **Campbell et al. 2022** -- The continuous-time CTMC framework GIDD builds on. Required for understanding the rate matrix derivation.
2. **Sahoo et al. 2024 (MDLM)** -- The masked diffusion baseline GIDD generalizes. Important for understanding what GIDD adds beyond the special case.
3. **Arriola et al. 2025 (BD3-LMs)** -- Block diffusion, a complementary approach. Not compared in GIDD but architecturally relevant.
4. **Zheng et al. 2024 (LLaDA)** -- Large-scale masked DLM. The scaling regime GIDD aspires to.
5. **Lou et al. 2024** -- Score-based discrete diffusion. Alternative theoretical framework that may connect to GIDD's I-S divergence term.

---

# Cross-Paper Analysis

## How They Fit Together

The scaling paper explicitly builds on GIDD: it takes the GIDD ELBO and reparameterizes it through log-SNR to get schedule invariance. The sigmoid mixing distribution pi_lambda = sigma(a*lambda+b)*u + (1-sigma(a*lambda+b))*m is used to define the five noise types for the scaling study. GIDD provides the theory; the scaling paper provides the empirical validation at scale.

## Key Tensions

1. **Uniform noise: better scaling but worse absolute performance.** GIDD shows p_u > 0 hurts likelihood at current scales (22.29 vs 24.38 PPL). The scaling paper shows uniform has better loss exponents (-0.0522 vs -0.0496). Both predict crossover around 10^21 FLOPs but call this "highly unreliable." Nobody has observed the crossover.

2. **Self-correction vs. raw quality.** GIDD shows uniform noise enables self-correction (55% gen PPL improvement). The scaling paper doesn't evaluate generation quality at all. These two capabilities may trade off, or self-correction might close the raw quality gap -- nobody has checked.

3. **Loss floor.** GIDD proves the theoretical minimum is zero (Proposition 3.9). The scaling paper fits power laws without offset, implicitly assuming this. But neither paper discusses the practical loss floor imposed by finite model capacity and training compute. The AR comparison is incomplete without this.

## Implications for Open-dLLM (Phase 4)

1. **Noise type matters for scaling.** If training at ~10^19 FLOPs (Phase 4 scale), masked diffusion is strictly better on likelihood. Uniform only wins at >> 10^21 FLOPs. Stick with masked or very low p_u.

2. **Batch size scaling.** B* ~ D^0.82 and independent of model size. For Phase 4's ~50M params on FineWeb-Edu subset, the batch size should be set by dataset size, not model size.

3. **CompleteP hyperparameters.** If adopting muP: sigma_base=0.4, sigma_aux=0.02, eta_base=0.3 are empirically validated starting points.

4. **Dynamic weighting if using any uniform noise.** Even small p_u requires the dynamic weighting scheme from GIDD Eq. 9. Clamped weighting is not enough.

5. **Self-correction as evaluation signal.** Even if training masked-only, it's worth training one run with p_u=0.1 to see if self-correction emerges. It's a qualitative capability that downstream benchmarks may not capture.
