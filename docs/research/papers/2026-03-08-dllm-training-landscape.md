# dLLM Training Landscape — Cross-Paper Analysis (March 2026)

**Date:** 2026-03-08
**Papers analyzed:** 11 primary + 22 discovered via search
**Focus:** Training objectives, loss functions, noise schedules, and what Phase 5 might be missing

---

## Universal Training Objective

Every paper in the field uses the same fundamental loss:

$$\mathcal{L} = \mathbb{E}_{t \sim p(t)} \left[ \frac{\alpha'_t}{1 - \alpha_t} \sum_{i: x_t^i = \text{MASK}} -\log p_\theta(x_0^i | x_t) \right]$$

For linear schedule ($\alpha_t = 1-t$, mask_prob = $t$): weight = $1/t$.

No paper proposes a fundamentally different pre-training objective.

## Papers Analyzed

### 1. LLaDA (2502.09992) — Nie et al., Feb 2025
- Linear schedule, 1/t ELBO weight, 8B params, 2.3T tokens
- Loss normalized by `1/(t*L)` (Algorithm 1) — divides by ALL tokens
- 1% random-length training trick: sample seq_len ∈ [1, 4096]
- No GQA (argues KV cache irrelevant for bidirectional)
- Eq. 14: deterministic mask count form for lower-variance evaluation

### 2. Dream 7B (2508.15487) — Ye et al., Jul 2025
- Linear schedule, initialized from Qwen2.5-7B (AR→dLLM)
- **CART replaces 1/t entirely** (not multiplied) — geometric kernel w/ p=0.1
- Loss normalized by masked token count (not all tokens)
- Shift operation: logits[i] predicts token[i+1] (AR convention preserved)
- Optional focal-loss token reweighting: `alpha * (1 - exp(-loss))^gamma * loss`
- All-or-nothing EOS masking for SFT

### 3. Dream-Coder 7B (2509.01142) — Xie et al., Jul 2025
- Same loss as Dream 7B, initialized from Qwen2.5-Coder
- 322B tokens, CART with geometric distance weighting
- SFT tricks: random truncation, padding penalty (inference only)
- GRPO for diffusion: no entropy/KL loss, asymmetric clipping (ε_low=0.2, ε_high=0.28)
- Coupled sampling + informative substitution for RL

### 4. BD3-LMs (2503.09573) — Arriola et al., ICLR 2025
- Block diffusion ELBO: sum of per-block NELBOs
- **Clipped schedules**: U[β,ω] instead of U[0,1], reduces gradient variance 2-6x
- Optimal for block_size=16: U[0.3, 0.8]; for block_size=4: U[0.45, 0.95]
- Variance is the key bottleneck: standard linear has ~14x higher variance than AR
- No timestep conditioning needed (time-agnostic property confirmed)
- Pre-train at max block size, fine-tune at target block size

### 5. Seed Diffusion (2508.02193) — ByteDance, Jul 2025
- Three-stage pipeline: mask-only → mask+edit → constrained-order + on-policy
- Edit-based corruption (≤10% deletions/insertions/substitutions) prevents carry-over bias
- On-policy trajectory optimization minimizes denoising steps
- Block-level semi-AR inference with KV cache
- Core ELBO is standard; innovations are in pipeline and inference

### 6. LLaDA 2.0 (2512.15745) — inclusionAI, Dec 2025
- WSD Block Size Schedule: AR(B=1) → B=4 → B=32 → B=4096 → back down
- Complementary masking: SFT only (negative result for pre-training)
- CAP training: entropy minimization on correctly predicted tokens
- DPO with ELBO as likelihood surrogate
- Masked embedding stabilization: Gaussian noise for AR→dLLM conversion
- Top-k checkpoint merge
- Mask ratio bandwidth [α_min, α_max] during SFT

### 7. Ni et al. "Super Data Learners" (2511.03276) — Nov 2025
- Standard MDLM loss, no algorithmic novelty
- DLMs resist overfitting 16x longer than AR (masking diversity = implicit augmentation)
- Data efficiency crossover: DLMs overtake AR when data is scarce + many epochs
- DLMs require >100x more FLOPs for optimal performance

### 8. Quokka (2510.03280) — Ni et al., Sep 2025
- **Linear schedule consistently outperforms cosine and poly2**
- ELBO loss beats MaskGIT loss at convergence
- DLMs need ~5x more data than AR at 144M scale (ratio → 3.3x at 67B)
- Multi-epoch tolerance: 144M can train 1000+ epochs before overfitting
- Clean-to-noisy curriculum: marginal gains, under-specified
- Hyperparameters transfer from AR models
- Weight decay important for multi-epoch training

### 9. Implicit Regularizer (2601.22450) — Huang & Mirzasoleiman, Jan 2026
- MDM loss = Signal regime (feature learning) + Noise regime (implicit regularizer)
- Both regimes necessary — eliminating noise regime causes gradient collapse
- Optimal t for discriminative tasks: [0.45, 0.55]
- Optimal t for generative/reasoning: [0.5, 1.0]
- Change is one line: restrict t sampling interval
- **+8.8% is cherry-picked** (ARC-Easy, 8B, 15K steps)
- No discussion of how restricted training affects generation
- Theory only proven for k-parity; language extension is empirical

### 10. Stable-DiffCoder (2601.15892) — ByteDance, Jan 2026
- Block-wise clipped noise: q_blk(t) = min(1, max(u(t), 1/B))
- Eliminates 1/(B+1) wasted training steps (20% at B=4, 3% at B=32)
- Noise warmup for AR→dLLM: cap u_max, drop ELBO weight during warmup
- Per-block min-1-masked guarantee
- Beats AR counterpart at 8B on code generation (+1.9 HumanEval, +5.5 HE+)

---

## Key Findings for Phase 5

### Validated (keep as-is)
- Linear noise schedule (mask_prob = t)
- ELBO weight 1/t
- Loss normalized by all real tokens
- No complementary masking for pretraining
- No timestep conditioning
- Block diffusion with staircase attention
- Document-level attention masking

### Actionable Improvements
1. **Per-block min-1-masked guarantee** — check per-block, not per-sequence (Stable-DiffCoder)
2. **Raise t_min to ~0.2** — eliminates wasteful near-zero masking, caps ELBO weight at 5 (BD3-LMs, Implicit Regularizer)
3. **Upper-clip t_max to ~0.9** — eliminates pure-noise regime at t→1 (BD3-LMs)
4. **Data budget: plan for 15B+ tokens** or aggressive multi-epoch (Quokka scaling law)

### Future Work (post-Phase 5)
- CART (Dream-style geometric kernel) for SFT
- CAP training for parallel decoding quality (LLaDA 2.0)
- DPO with ELBO as likelihood surrogate (LLaDA 2.0)
- Edit-based corruption for self-correction (Seed Diffusion)
- GRPO for diffusion RL (Dream-Coder)
- Soft masking (Hersche et al.)

---

## Additional Papers Discovered (Not Yet Deep-Read)

| Paper | arXiv | Key Contribution |
|---|---|---|
| SDAR-VL | 2512.14068 | Async block noise, mask ratio scaling, beta curriculum |
| Soft-Masked DLMs | 2510.17206 | Soft blending replaces binary mask, 169M SOTA |
| Diffusion Duality | 2506.10892 | Gaussian-guided curriculum doubles speed |
| GIDD | 2503.04482 | Hybrid mask+uniform, self-correction +55% |
| XDLM | 2602.01362 | Unifies MDLM and UDLM |
| ProSeCo | 2602.11590 | Self-correcting MDM, +14% HumanEval |
| ARMD | 2601.16971 | Causal MDM with permutation-equivariant arch |
| VMD | 2510.23606 | Global latent variable for token dependencies |
| Frequency-Informed | 2509.05056 | Rare-token-prioritized masking |
| DiffuCoder | 2506.20639 | Complementary mask GRPO (Apple) |
| Reweighted Bounds | 2511.19664 | Theoretical: reweighted losses are better bounds |
| Simplified MDLM | 2406.04329 | Foundation: ELBO = weighted CE integral |
| Scaling Beyond | 2602.15014 | Masked > uniform at scale, but GSM8K exception |
| Data-Constrained | 2507.15857 | Closed-form compute threshold for diffusion > AR |
| Time-Agnostic MDMs | 2409.02908 | First-hitting sampler, 20x speedup |
| Attention Floating | 2601.07894 | MDM attention dynamics, 2x AR on knowledge tasks |
| Train for Worst | 2502.06768 | Adaptive token ordering: 6.88% → 89.49% Sudoku |
