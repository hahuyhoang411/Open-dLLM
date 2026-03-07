# Phase 4 dLLM Architecture Research

**Query:** What architectural improvements, kernel optimizations, and training techniques should Phase 4 adopt?
**Date:** 2026-03-02 | **Sources analyzed:** 30+ top-tier + 50+ second-tier | **Languages:** EN, CN
**Research scope:** Academic papers (OpenAlex, arXiv), English communities (HN, Reddit, Bluesky), Chinese communities (Zhihu, CSDN, 机器之心), practitioner sources (GitHub, blogs)

---

## State of the Art (March 2026)

The dLLM field moved from research curiosity to commercial viability in 12 months. LLaDA scaled to 8B (Feb 2025), then 100B MoE (LLaDA 2.0). Mercury and Gemini Diffusion broke 1000 tok/s commercially. Dream 7B (Aug 2025) proved AR-to-diffusion initialization works. Consistency Diffusion LMs (Feb 2026) cut sampling steps 4-7x via post-training distillation.

Three results matter most for small-model training: (1) dLLMs show 3x+ data efficiency over AR models at small scale (arXiv:2511.03276), meaning our limited compute budget goes further. (2) CART noise rescheduling (Dream 7B) uses per-token adaptive noise — Sudoku accuracy jumped 21% → 81% vs uniform noise. (3) Soft masking adds 3 learnable parameters for ~6.5% perplexity gain with zero architectural change.

## Key Approaches

| Approach | Key Players | Best Result | Status |
|----------|------------|-------------|--------|
| Block Diffusion (BD3-LMs) | Kuleshov group, ICLR 2025 | Block-by-block parallel generation with KV cache | proven (our Phase 3 foundation) |
| CART Noise Rescheduling | Dream 7B (HKU) | Sudoku 81% vs AR 21%; code gen competitive | proven at 7B scale |
| Soft Masking | arXiv:2510.17206 | PPL 23.14 → 21.63 (~6.5% gain), 3 params | proven on MDLM/GPT-2 scale |
| Complementary Masking | Fast-dLLM v2 (NVIDIA) | 2x training signal per sample | proven, training-free |
| Liger Fused Kernels | LinkedIn | 20% throughput, 60% memory reduction | proven, production-grade |
| FlexAttention | PyTorch team | ~90% of FlashAttention2 speed, custom patterns | proven, PyTorch 2.5+ |
| GQA | GLM-4, Qwen3, LLaMA 2+ | 4x KV cache reduction | standard practice |
| DualCache Inference | Fast-dLLM v2 | 5-11x inference speedup | proven for block diffusion |

## What Works

Liger Kernel's FusedLinearCrossEntropy is the single highest-impact kernel optimization. Instead of materializing a (batch × seq_len × vocab_size) logit tensor, it chunks the lm_head projection and cross-entropy into micro-batches. For a 32K vocab at batch=32, seq_len=512, this saves ~2GB of peak memory. The fused RMSNorm (7x speed), RoPE (8x speed), and SwiGLU (1.6x memory) are drop-in replacements. All require CC 7.0+ (T4 qualifies). End-to-end: 20% throughput increase, 60% memory reduction on LLaMA-scale models.

FlexAttention replaces our hand-crafted 1024×1024 float staircase mask with a compiled kernel. You define a `mask_mod(batch, head, q_idx, kv_idx) → bool` function and `create_block_mask()` compiles it into a sparse block pattern. The BD3-LM authors list FlexAttention as "crucial" for training efficiency. Performance: ~90% of FlashAttention2 for custom patterns. The staircase mask maps naturally: attend if `block_id(q) == block_id(kv)` (bidirectional within block) OR `block_id(kv) < block_id(q)` AND `is_x0_half(kv)` (causal across blocks, attending only to clean tokens). Cache the `BlockMask` object — `create_block_mask` is expensive.

CART noise rescheduling (Dream 7B, arXiv:2508.15487) replaces uniform `t ~ U[0,1]` per block with context-adaptive timesteps. A geometric distribution measures how much contextual information each token position has — tokens near known context get less noise (easier), isolated tokens get more. This concentrates training signal where it matters most. Dream 7B trained with CART on 1/4 of LLaDA's data budget and matched or exceeded it.

Complementary masking (Fast-dLLM v2, arXiv:2509.26328) duplicates each training sample with complementary masks: if position i is masked in copy 1, it's unmasked in copy 2. This guarantees every position gets learned at least once per batch, reducing gradient variance — the single biggest identified training problem for block diffusion.

## What Doesn't Work

Gated DeltaNet (used by Qwen3.5 at 75% ratio) is causal-only. Its linear recurrent mechanism computes a running state that depends on sequential order. Bidirectional diffusion attention — where tokens attend freely within blocks — breaks this dependency. Multiple Chinese community discussions confirm: "DeltaNet works for AR, not for diffusion."

MLA (DeepSeek Multi-head Latent Attention) compresses KV cache 93% via low-rank projection. Powerful at scale (DeepSeek V3, 671B MoE), but at 35-100M params the KV cache isn't the bottleneck — activations and logits are. The implementation complexity (custom low-rank Q/KV projections, absorption trick for efficient inference) doesn't pay off at our scale.

AR-to-diffusion initialization (DiffuLLaMA, Dream 7B) gives ~2x training speedup by starting from pretrained AR weights. But our project is educational — training from scratch teaches more about diffusion dynamics, and we don't have a pretrained 100M AR model to convert from.

## Hardware: P100 vs T4

**This is the single most important finding.** P100 (Pascal, CC 6.0) blocks ALL Triton-based libraries:
- FlexAttention: NOT compatible (Triton kernels)
- Liger Kernel: NOT compatible (Triton >= CC 7.0)
- Unsloth: NOT compatible (min CC 7.0)
- Flash Attention: NOT compatible (min CC 8.0)

T4 (Turing, CC 7.5, 16GB, tensor cores) unlocks everything. Same free tier on Kaggle, same 30h/week quota. The only P100-safe optimizations: AMP fp16 (via CUDA cores, not tensor cores), gradient checkpointing, chunked cross-entropy (pure PyTorch loop), xformers (explicitly supports SM60).

Kaggle also offers dual T4 (2×16GB) on the free tier, enabling DDP for 2x effective batch size.

## Open Questions

- Does CART noise rescheduling adapt well from per-token (Dream, full-sequence diffusion) to per-block (our block diffusion)? Dream operates on individual tokens; we assign one timestep per 4-32 token block. Averaging CART scores across block positions is the natural adaptation, but no paper has tested this
- Soft masking claims ~6.5% PPL improvement but at ~2x training cost (needs a preliminary forward pass to get predictions before creating the soft mask). Is this worth it for a Kaggle training budget of ~5 hours?
- Liger Kernel on T4: academic scout says Triton compiles for SM60+ (element-wise ops only); practitioner scout says CC 7.0+ required. T4 (CC 7.5) is safe regardless, but the P100 fallback path is uncertain for Liger

## Recommended Reading

1. [DEEP-READ] **Dream 7B** (arXiv:2508.15487) — CART noise rescheduling algorithm, ablations vs uniform noise
2. [DEEP-READ] **Fast-dLLM v2** (arXiv:2509.26328) — complementary masking, DualCache, DPad, block_size=32
3. [DEEP-READ] **Soft-Masked Diffusion LMs** (arXiv:2510.17206) — 3-parameter soft masking implementation
4. [DEEP-READ] **Liger Kernel** (arXiv:2410.10989) — fused kernel implementations and benchmarks
5. [SKIM] **dLLMs are Super Data Learners** (arXiv:2511.03276) — data efficiency analysis
6. [SKIM] **LLaDA 2.0** — 100B MoE dLLM, WSD training, block_size=32
7. [SKIM] **CDLM** (arXiv:2511.19269) — consistency distillation post-training recipe
8. [SKIM] **DiffuLLaMA** (arXiv:2410.17891) — AR-to-diffusion conversion at 127M params
9. [REFERENCE] **BD3-LMs** (ICLR 2025 Oral) — our foundation architecture
10. [REFERENCE] **ZHZisZZ/dllm** — reference library with A2D, CFG, eval framework

## Chinese Community Highlights

**LLaDA 2.0 (100B MoE)**: First 100B dLLM. Scores 73.18 across 47 benchmarks (vs Qwen3-30B at 73.60). Block size = 32. Three-phase WSD training schedule. Source: 36Kr, Zhihu.

**dInfer inference framework** (Ant Group): 1011 tok/s on HumanEval, 10.7x faster than Fast-dLLM. Commercial-grade, integrated with SGLang.

**Community prediction**: 2026 will see industrial-grade, consumer-facing text diffusion models for cheap, reliable, low-latency inference.

**"Parallel-performance fundamental contradiction"**: Explicitly named in Chinese literature, underappreciated in English. Generating too many tokens at once breaks inter-token dependencies. Larger block sizes = more parallelism but worse quality. Block diffusion's block_size controls this tradeoff.

**Qwen3.5 hybrid ratio**: 75% Gated DeltaNet + 25% full attention (3:1 ratio, confirmed via ablation). DeltaNet's key: transforms linear attention's "write-only" memory into erasable memory via gating. Qwen3-Next-80B-A3B matches Qwen3-32B quality with <10% training compute.

**Entropy Sink** (DiffuCoder, Apple + HKU): dLLMs show higher prediction confidence near prompt boundaries. Novel Coupled-GRPO RL algorithm.

## Key GitHub Repos

- [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) — ICLR 2025, supports flex/sdpa attention backends
- [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) — unified dLLM library (LLaDA, Dream, A2D, Fast-dLLM)
- [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — fused Triton kernels for training
- [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) — inference acceleration + training tricks
- [VILA-Lab/Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) — survey paper companion
- [meta-pytorch/attention-gym](https://github.com/meta-pytorch/attention-gym) — FlexAttention examples
- [mgmalek/efficient_cross_entropy](https://github.com/mgmalek/efficient_cross_entropy) — chunked CE (pure PyTorch)
- [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) — torch.compile CE variant
