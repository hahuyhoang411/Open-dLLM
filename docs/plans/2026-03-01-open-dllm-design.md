# Open-dLLM Design Document

**Date**: 2026-03-01
**Status**: Approved

## Objective

Build a progressive, educational repository for learning diffusion language models (dLLMs) from scratch, structured as self-contained phases that go from "hello world" to production-grade techniques.

## Project Identity

- **Name**: Open-dLLM
- **Tagline**: "Learn diffusion language models from scratch, one phase at a time"
- **Philosophy**: Like Karpathy's nanoGPT/nanochat but for diffusion LLMs. Each phase is a self-contained lesson. The code IS the textbook.
- **Target audience**: ML practitioners who understand transformers/GPT but want to learn dLLMs from fundamentals.
- **References**: tiny-diffusion (Nathan Barry), nanochat (Karpathy), MDLM (Kuleshov group), LLaDA, Mercury

## Phase Structure

### Phase 1: `01_hello_diffusion/`

**Goal**: The "hello world" of dLLMs. Prove the concept in ~400 lines.

- **Tokenizer**: Character-level (65 chars from Tiny Shakespeare)
- **Model**: 6 layers, 6 heads, 384 dim = ~10.7M params
- **Attention**: Bidirectional (RoPE, `is_causal=False`)
- **Activation**: ReLU squared
- **Norm**: RMSNorm (no learnable params)
- **Mask token**: `_` (id=0)
- **Training**: mask_ratio ~ U[0,1] per sample, CE loss on masked positions, 10K iterations
- **Inference**: Confidence-based unmasking, block-by-block (240 tokens), top-k sampling
- **Dataset**: Tiny Shakespeare (~1MB)
- **Hardware**: Any (CPU, MPS, CUDA)
- **Time**: ~20 min on GPU, ~2 hours on CPU
- **Files**: `hello_diffusion.py` (single file, ~400 lines)

**README teaches**: What is diffusion for text, forward/reverse process, why bidirectional attention, confidence-based decoding

### Phase 2: `02_nano_dllm/`

**Goal**: Train a real diffusion LLM on real data. The main event.

- **Tokenizer**: BPE (~32K vocab, HuggingFace tokenizers)
- **Model**: Configurable via `--depth` (6-24 layers), model_dim = depth * 64
- **Attention**: Bidirectional (RoPE)
- **Noise schedule**: Cosine: alpha_t = cos^2(t * pi/2)
- **Loss**: (1/t) * CE on masked positions (proper ELBO weighting)
- **Dataset**: FineWeb-Edu (HuggingFace streaming)
- **Hardware**: Kaggle T4/P100 (16GB VRAM)
- **Time**: Hours to days depending on depth
- **Files**: `nano_dllm.py` (model), `train.py` (training), `dataloader.py` (data pipeline), `generate.py` (inference)

**README teaches**: Cosine schedule (math + Fisher-Rao optimality), loss weighting, MDLM ELBO, BPE for diffusion, scaling

### Phase 3: `03_block_diffusion/`

**Goal**: Implement block diffusion (BD3-LMs) — the architecture behind Mercury.

- **Attention**: Block-causal (bidirectional within block, causal between blocks)
- **T2T editing**: Revise committed tokens based on confidence
- **Variable-length generation**: Block-by-block with EOS
- **KV caching**: Across blocks (key practical advantage)
- **Files**: `block_dllm.py`, `train.py`, `generate.py`

**README teaches**: Block diffusion, T2T editing, Mercury architecture, variable-length generation

### Phase 4: `04_advanced/` (Future)

Planned scripts for cutting-edge techniques:
- `flow_matching.py` — Discrete flow matching
- `diffu_grpo.py` — RL for dLLMs
- `tidar.py` — Think in Diffusion, Talk in AR (hybrid)
- `sedd.py` — Score Entropy Discrete Diffusion

## Repo Structure

```
Open-dLLM/
├── README.md                              # Project overview + learning roadmap
├── LICENSE
├── pyproject.toml
├── 01_hello_diffusion/
│   ├── README.md                          # Theory: What is text diffusion?
│   ├── hello_diffusion.py                 # ~400 lines, self-contained
│   └── download_data.py
├── 02_nano_dllm/
│   ├── README.md                          # Theory: Noise schedules, BPE, ELBO
│   ├── nano_dllm.py
│   ├── train.py
│   ├── dataloader.py
│   └── generate.py
├── 03_block_diffusion/
│   ├── README.md                          # Theory: Block diffusion, T2T, Mercury
│   ├── block_dllm.py
│   ├── train.py
│   └── generate.py
├── 04_advanced/
│   └── README.md
└── docs/
    ├── plans/
    │   └── 2026-03-01-open-dllm-design.md
    └── research.md
```

## Tech Stack

- Python 3.10+
- PyTorch (only required ML dependency)
- HuggingFace `datasets` (Phase 2+ for FineWeb-Edu streaming)
- HuggingFace `tokenizers` (Phase 2+ for BPE)
- No training frameworks (no Lightning, no DeepSpeed, no Hydra)
- Phase 1 has zero dependencies beyond PyTorch

## Code Style

- **Annotated code with rich comments**: The code is the learning material
- **ASCII diagrams**: Inline visual explanations of data flow and architecture
- **Self-contained phases**: Each folder runs independently
- **nanochat-style**: Linear scripts, no Trainer classes, no config files
- **Educational comments**: Explain WHY, not just WHAT. Mark diff from GPT with `[DIFF]`

## Success Criteria

1. Phase 1 trains in <30 min on any GPU and produces recognizable text patterns
2. Phase 2 trains a BPE model on FineWeb-Edu that produces coherent text
3. Phase 3 demonstrates block diffusion with variable-length generation
4. Each phase README explains theory clearly enough to learn from
5. Code is annotated with ASCII diagrams and educational comments
6. Anyone who understands nanoGPT can learn dLLMs from this repo

## Scope

**In scope**: Training from scratch, inference, theory documentation, educational annotations
**Out of scope**: Fine-tuning existing models, deployment/serving, multi-node training, RLHF (except Phase 4 diffu-GRPO), model evaluation benchmarks beyond basic perplexity/samples

## Key Research References

- MDLM (Sahoo et al., NeurIPS 2024) — Foundational theory
- LLaDA (Nie et al., 2025) — First 8B dLLM
- LLaDA 2.1 (2026) — T2T editing
- Mercury/Mercury 2 (Inception Labs) — Block diffusion, 1000+ TPS
- SEDD (Lou et al., ICML 2024 Best Paper) — Score entropy approach
- BD3-LMs (Arriola et al., ICLR 2025 Oral) — Block diffusion theory
- d1 (2025) — RL for dLLMs
- TiDAR (2025) — Hybrid diffusion+AR
- tiny-diffusion (Nathan Barry) — Educational reference implementation
- nanochat (Karpathy) — Repo structure reference
