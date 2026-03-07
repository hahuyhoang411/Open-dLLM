# Diffusion LMs: Reasoning and Data Efficiency

Three-paper deep read. Date: 2026-03-02.

---

## Paper 1: Beyond Autoregression — Discrete Diffusion for Complex Reasoning and Planning

**arXiv 2410.14157** | Ye, Gao, Gong, Zheng, Jiang, Li, Kong (HKU + Huawei Noah's Ark) | ICLR 2025

### Summary

**One-Sentence Takeaway:** A 6M-parameter discrete diffusion model solves Sudoku at 100% accuracy where LLaMA-13B scores 32.9%, because diffusion decomposes hard subgoals into multiple denoising views instead of forcing left-to-right sequential commitment.

| Metric | Value |
|---|---|
| MDM (85M) vs GPT-2 (85M) on Countdown-5 | 46.6% vs 5.1% (9x gap) |
| MDM (6M) vs LLaMA-13B on Sudoku | 100% vs 32.9% (2000x fewer params) |
| MDM vs GPT-4 ToT on Game of 24 | 76.0% vs 74.0% at 186x fewer tokens |
| MDM 1-step speed vs AR | 10x faster, still 75% on Countdown-4 |

**Red Flags:**
- All benchmarks are synthetic constraint-satisfaction tasks (Countdown, Sudoku, SAT, Game of 24). No natural language reasoning (GSM8K, MATH, ARC). The gap between "planning over known constraints" and "reasoning in natural language" is enormous and unaddressed.
- The 6M-vs-13B Sudoku comparison is misleading: LLaMA-13B was fine-tuned on 100k Sudoku examples, but it was pretrained on text, not grid puzzles. The architecture mismatch (causal attention on a 2D grid problem) is doing most of the work, not "diffusion vs AR" per se.
- No explicit limitations section. The paper quietly notes LLaMA-7B with fine-tuning resolves synthetic planning at all distances, undermining the "AR fundamentally can't plan" narrative.

**Bottom Line:** Convincing evidence that bidirectional attention + iterative refinement beats left-to-right generation on constraint satisfaction. The "subgoal imbalance" theory (hard subgoals are exponentially data-hungry for AR) is well-motivated. But the claim stops at synthetic tasks. Extrapolating to "complex reasoning" broadly is a stretch the paper doesn't earn.

### Key Findings

The core experiment is a synthetic planning task with tunable difficulty (planning distance). AR models need ~50k examples for distance=2 but plateau near random for distance 3-5. Diffusion keeps improving with the same data. The proposed mechanism: AR must predict hard intermediate steps using only left context; diffusion sees the problem from multiple noise levels, effectively decomposing each hard subgoal into easier partial-denoising problems.

Multi-granularity Diffusion Modeling (MDM) adds two tricks: (1) token-level reweighting v(x) that emphasizes harder tokens during training, (2) "easy-first TopK" decoding that unmasks confident predictions first. Both are simple and effective — Countdown-5 jumps from ~30% (vanilla diffusion) to 46.6% (MDM).

The Game of 24 result (76% vs GPT-4 ToT's 74% at 186x fewer tokens) is the flashiest number but also the least controlled: MDM is trained specifically on Game of 24 data, GPT-4 is prompted zero-shot. Apples and oranges.

**Fairness of comparisons:** Model sizes are matched for GPT-2 baselines (85M vs 85M), which is fair. The LLaMA comparisons are unfair in both directions — LLaMA has more parameters but wrong inductive bias for grid tasks. The synthetic data regime (100k-500k examples) is well-controlled but narrow.

### Connections

This paper provides the *mechanistic explanation* for why diffusion should be better: subgoal decomposition via multi-view denoising. Paper 2 (Prabhudesai et al.) provides the *data efficiency explanation*: implicit augmentation via random masking orderings. Paper 3 (Ni et al.) scales both ideas to 1.7B parameters and real benchmarks. Together they form a coherent story — diffusion gets more "views" per training example — but each paper tests in a different regime, making direct comparison impossible.

---

## Paper 2: Diffusion Beats Autoregressive in Data-Constrained Settings

**arXiv 2507.15857** | Prabhudesai, Wu, Zadeh, Fragkiadaki, Pathak (CMU + Lambda) | NeurIPS 2025

### Summary

**One-Sentence Takeaway:** When training data is fixed and you repeat it for many epochs, masked diffusion models resist overfitting far longer than AR — diffusion's data reuse half-life is 513 epochs vs AR's 32 epochs, a 16x difference.

| Metric | Value |
|---|---|
| Diffusion R_D* (data reuse half-life) | 512.85 epochs |
| AR R_D* (data reuse half-life) | 31.93 epochs |
| Val loss at 100M unique tokens (multi-epoch best) | Diffusion 3.55 vs AR 3.71 |
| SciQ accuracy (500M unique tokens) | Diffusion 79.1% vs AR 67.8% |

**Red Flags:**
- The single-epoch (compute-optimal) regime still favors AR: val loss 7.07 vs diffusion's 10.65. The paper buries this. Diffusion only wins after ~50 epochs of repetition on 100M tokens. In the real world, most pretraining runs see each token once.
- Scaling laws are fit over 25M-100M unique tokens only. Whether the 16x overfitting resistance holds at 10B or 100B tokens is entirely unknown. The extrapolation is optimistic.
- 200 models trained but the largest is 2.5B parameters. No evidence the phenomenon persists at frontier scale (7B+).
- Downstream accuracy numbers are modest in absolute terms (SciQ 79%, HellaSwag 35%). These are small models on small data — the question is whether the relative advantage scales.

**Bottom Line:** The most rigorous of the three papers. Clean experimental design: same architecture, same hyperparameters, same data, only attention mask and objective differ. The R_D* metric (data reuse half-life) is a genuinely useful contribution. The limitation is regime: this is a story about small data repeated many times, not about frontier pretraining.

### Key Findings

The paper trains 200 models (100 diffusion, 100 AR) across sweeps of model size (7M-2.5B), data budget (25M-100M unique tokens), and epoch count (up to 800). Architecture is identical except causal vs. bidirectional attention and next-token vs. masked-token objective.

The central result: AR models' validation loss diverges after ~4 epochs of data repetition. Diffusion models maintain nearly overlapping training curves up to ~100 epochs. This is quantified via R_D*, the number of epochs at which repeated data is worth half as much as fresh data. Diffusion's 513 vs AR's 32 is stark.

The proposed mechanism: diffusion provides "implicit data augmentation" because each training step samples a random subset of tokens to mask. A sequence of length L generates 2^L possible masking patterns, each a different training signal. AR always sees the same left-to-right factorization. The paper tests explicit augmentation for AR (input masking, attention dropout) and shows it helps but cannot close the gap — the factorization diversity itself is the key.

The critical compute threshold C_crit(U) proportional to U^2.174 predicts exactly when diffusion starts winning for a given dataset size. Below that compute budget, AR wins. Above it, diffusion wins. This is the paper's most actionable contribution.

**Fairness of comparisons:** Exceptionally fair. Same optimizer, same schedule, same architecture backbone. The only flag: hyperparameters were "primarily tuned for AR models," which the authors acknowledge gives AR a slight advantage. If anything, this makes the diffusion wins more impressive.

### Connections

Paper 1 (Ye et al.) shows *why* diffusion helps reasoning (subgoal decomposition). This paper shows *when* diffusion helps (data-constrained regime with high epoch count). Paper 3 (Ni et al.) independently confirms the epoch-resistance finding at larger scale (1B-8B params, up to 480 epochs) and adds the "3x data efficiency" claim. The R_D* metric from this paper and the crossover analysis from Paper 3 are measuring the same phenomenon from different angles.

Key tension: Paper 2 says diffusion wins only after ~50 epochs of repetition. Paper 3 claims crossover happens even earlier with larger models. Neither addresses the elephant in the room: most serious pretraining uses each token once or a handful of times. The "data-constrained" framing is forward-looking (anticipating data exhaustion) but not yet the dominant regime.

---

## Paper 3: Diffusion Language Models Are Super Data Learners

**arXiv 2511.03276** | Ni, Liu, Dou, Du, Wang, Yan, Pang, Shieh (NUS + Sea AI Lab + StepFun) | 2025

### Summary

**One-Sentence Takeaway:** A 1B diffusion LM trained on 1B unique tokens (96 epochs) hits 56% HellaSwag where the matched AR gets 41%, and a 1.7B diffusion coder trained on 10B Python tokens overtakes its AR twin — but at >100x training FLOPs.

| Metric | Value |
|---|---|
| 1B DLM vs 1B AR on HellaSwag (1B unique tokens, 96 epochs) | 56% vs 41% |
| Data efficiency ratio claimed | ~3x (DLM on 0.5B tokens matches AR on 1.5B) |
| Training FLOPs overhead for DLM | >100x vs AR |
| Inference FLOPs overhead at seq_len=4096 | ~4700x vs AR |

**Red Flags:**
- The 100x training FLOPs overhead is staggering. "Data efficiency" is meaningless in isolation — what matters is the Pareto frontier of performance vs. total cost (data + compute). If you need 100x more FLOPs to save 3x data, you're only ahead when data is literally 33x more expensive per token than compute. The paper acknowledges this but doesn't quantify the crossover.
- The 4700x inference overhead at seq_len=4096 makes deployment impractical at scale. KV-cache for AR vs. full bidirectional recomputation is a fundamental gap, not an engineering detail.
- Hyperparameters are "primarily optimized for AR models" — the authors call their own setup "inherently unfair for diffusion." This cuts both ways: maybe diffusion would do even better with tuned hyperparams, or maybe the AR baseline is artificially weak.
- The "validation loss does not predict downstream performance" finding (Section 5) is important but also convenient — it makes it hard to compare models on any single metric.
- MoE experiments show DLM MoE < DLM dense at same param count, suggesting the dense compute itself (not parameter count) drives gains. This is consistent with "just throw more FLOPs at it" rather than "diffusion is fundamentally smarter."

**Bottom Line:** The largest-scale study of the three, with the most practical benchmarks (HellaSwag, MMLU, HumanEval, MBPP). The 3x data efficiency claim is real but comes with 100x compute tax. The paper is refreshingly honest about limitations. Most valuable contribution: the finding that crossover timing shifts systematically with data size, model scale, and data quality — giving practitioners a framework to decide when diffusion is worth the cost.

### Key Findings

The paper runs controlled experiments at 1B-8B parameters on Nemotron-CC and C4 data. The "intelligence crossover" — the epoch at which DLM downstream accuracy surpasses AR — is the central object of study.

Three mechanisms are proposed for DLM's data efficiency:
1. **Any-order modeling**: Bidirectional attention sees 2^L masking patterns per sequence of length L, vs AR's single left-to-right factorization. This is the same argument as Paper 2.
2. **Super-dense compute**: Each token is attended to ~N times per sequence (bidirectional) vs once (AR with KV cache). DLMs convert FLOPs into signal more aggressively.
3. **Monte Carlo augmentation**: The diffusion objective averages over random corruption patterns, creating diverse training signals from each example.

The ablation studies are informative: adding input masking (10-90%) to AR helps modestly but collapses above 30% masking rate. Dropout on AR similarly helps but cannot close the gap. This suggests the any-order factorization is the primary driver, not just the noise.

The code experiment (1.7B params, 10B Python tokens, ~150 epochs) is the strongest result: DLM overtakes AR on MBPP/HumanEval with matched settings. Crossover timing varies by benchmark — MBPP crosses early, HumanEval late — suggesting evaluation protocol sensitivity.

The "validation loss vs downstream accuracy" disconnect (Section 5) deserves attention: DLMs can have rising val loss while improving on benchmarks, because multiple-choice accuracy depends on relative NLL across options, not absolute NLL. This makes standard scaling law analysis (which tracks val loss) unreliable for DLMs.

**Fairness of comparisons:** Architecture is identical except attention mask and objective. Data, optimizer, schedule all matched. The self-acknowledged hyperparameter bias toward AR is a legitimate concern. The FLOPs comparison is the most important missing piece — performance-per-FLOP would likely favor AR in most regimes.

### Connections

Paper 3 is the natural scale-up of Paper 2. Both find the same phenomenon (diffusion resists overfitting under data repetition), both attribute it to masking diversity, and both acknowledge the compute overhead. Paper 3 adds: (a) scaling to 8B parameters, (b) real benchmarks beyond validation loss, (c) the data quality dimension, (d) the MoE angle.

Paper 1 provides the theoretical grounding (subgoal decomposition) that Papers 2 and 3 lack. Papers 2 and 3 are empirical scaling studies that confirm the phenomenon at larger scale but don't explain *why* bidirectional denoising helps reasoning specifically — they only show it helps data efficiency generally.

---

## Cross-Paper Synthesis

### The Emerging Consensus

All three papers agree on three points:

1. **Diffusion LMs extract more signal per unique training example than AR.** Paper 1 shows this on synthetic planning (10k vs 50k examples for equal accuracy). Paper 2 quantifies it as 16x longer data reuse half-life (R_D*=513 vs 32 epochs). Paper 3 puts a ratio on it: ~3x data efficiency at 1B-scale models.

2. **The mechanism is masking diversity.** AR sees one factorization (left-to-right). Diffusion sees 2^L possible masking patterns per sequence. This is "implicit data augmentation" (Paper 2), "Monte Carlo augmentation" (Paper 3), or "multi-view subgoal decomposition" (Paper 1). All three names describe the same phenomenon.

3. **The advantage is conditional on regime.** Paper 2 is clearest: below a critical compute threshold, AR wins. Above it (when you're repeating data many times), diffusion wins. Paper 3 confirms: with enough unique data and single-epoch training, AR is better. Paper 1 shows the advantage is strongest on tasks requiring non-sequential planning.

### Are the Claims Consistent?

Mostly yes, with tensions:

- **Magnitude:** Paper 2 claims 16x longer data reuse. Paper 3 claims 3x data efficiency. These aren't contradictory (different metrics) but the 3x number is more practically meaningful and more modest.
- **Regime:** Paper 2 says the crossover needs ~50 epochs of repetition at 100M tokens. Paper 3 says larger models shift the crossover earlier. Neither tests at frontier data scales (100B+ unique tokens). The extrapolation to "data exhaustion era" is plausible but unproven.
- **Compute cost:** Paper 1 ignores it. Paper 2 acknowledges it implicitly. Paper 3 is blunt: >100x training FLOPs, up to 4700x inference FLOPs. The "data efficiency" story falls apart if compute is the bottleneck, not data.

### What Gaps Remain

1. **No frontier-scale evidence.** The largest model is 8B parameters on 96B tokens. Whether diffusion's advantage holds at 70B+ parameters on 1T+ unique tokens is unknown. The scaling laws from Paper 2 are fit on 25M-100M tokens — four orders of magnitude below current pretraining scale.

2. **Compute-adjusted comparisons are missing.** All three papers compare at matched model sizes and data. None compare at matched total FLOPs. Given diffusion's >100x compute overhead, the question "which model wins at the same total cost?" remains unanswered. Paper 2's critical compute threshold is the closest attempt but doesn't incorporate inference cost.

3. **Natural language reasoning is untested.** Paper 1's tasks are synthetic constraint satisfaction. Papers 2 and 3 test on multiple-choice benchmarks (HellaSwag, MMLU, ARC) but not on chain-of-thought reasoning (GSM8K, MATH, GPQA). The claim that diffusion helps "reasoning" specifically (vs. just general language modeling) is supported only by Paper 1's grid puzzles.

4. **Inference cost is the real blocker.** Even if diffusion is 3x more data-efficient, 4700x inference overhead at seq_len=4096 makes it unusable for serving. Speculative decoding, distillation to AR, or block diffusion (as in this project's Phase 3-4) are potential mitigations, but none of these papers address deployment.

5. **Hyperparameter optimization asymmetry.** All three papers acknowledge that training recipes are optimized for AR. DLM-specific schedules, masking strategies, and loss weighting are underexplored. The true gap between AR and diffusion may be larger or smaller than reported.

### The Combined Picture for Open-dLLM

For this project's Phase 4 (modern block diffusion at ~58M params on limited Kaggle data), the papers paint an encouraging picture:

- At small scale (sub-1B params) and limited data (Kaggle's constraints), diffusion's data efficiency advantage should be strongest (Paper 2's critical compute threshold is easily crossed).
- Block diffusion with staircase masking (Phase 3-4) partially addresses inference cost by generating blocks in parallel rather than token-by-token.
- The "any-order modeling" advantage (all three papers) is exactly what block diffusion preserves within each block.
- The warning: compute overhead is real. Training at >100x FLOPs per token means Kaggle T4 time is the bottleneck, not data volume. Phase 4's AMP + gradient checkpointing + Liger kernels are essential mitigations.
