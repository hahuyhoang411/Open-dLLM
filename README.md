# Open-dLLM

**Learn diffusion language models from scratch, one phase at a time.**

An educational repository that progressively builds diffusion language models -- from a simple character-level "hello world" to advanced techniques like block diffusion and RL fine-tuning. Each phase is self-contained with annotated code and theory documentation. Think nanoGPT, but for diffusion LLMs.

---

## The Big Idea

A standard LLM (GPT) generates text left-to-right, one token at a time. A **diffusion language model** (dLLM) starts with a fully masked sequence and iteratively fills in tokens -- all at once.

```
Forward process (masking):

  "The quick brown fox jumps"   -->   "The ___ brown ___ jumps"   -->   "___ ___ ___ ___ ___"
   t=0.0 (clean)                      t=0.5 (partial)                  t=1.0 (all masked)

Reverse process (denoising):

  "___ ___ ___ ___ ___"   -->   "___ quick ___ fox ___"   -->   "The quick brown fox jumps"
   t=1.0 (all masked)          t=0.5 (most confident           t=0.0 (fully denoised)
                                       filled first)
```

The model sees ALL positions at once (bidirectional attention) and fills in the ones it is most confident about first. Multiple tokens are revealed per step, enabling parallel generation.

---

## Learning Roadmap

| Phase | Status | What You'll Build | What You'll Learn | Prerequisites |
|-------|--------|-------------------|-------------------|---------------|
| `01_hello_diffusion` | Done | ~10M param char-level dLLM on Tiny Shakespeare | Masked diffusion basics, bidirectional attention, confidence-based decoding | Python, PyTorch basics |
| `02_nano_dllm` | Done | BPE model on FineWeb-Edu | Cosine noise schedule, ELBO weighting, BPE tokenization, scaling | Phase 1 |
| `03_block_diffusion` | Done | Block diffusion model (BD3-LM style) | Staircase attention mask, KV caching, block-by-block generation, per-block timesteps | Phase 2 |
| `04_advanced` | Coming soon | Advanced techniques | Qwen 3 tokenizer, flow matching, diffu-GRPO (RL), TiDAR (hybrid AR+diffusion), SEDD | Phase 3 |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Open-dLLM.git
cd Open-dLLM

# Install dependencies
pip install torch

# Download Tiny Shakespeare (~1.1MB)
python 01_hello_diffusion/download_data.py

# Train the model (~20 min on GPU, ~2 hours on CPU)
cd 01_hello_diffusion && python hello_diffusion.py --train

# Generate text (loads saved weights)
python hello_diffusion.py

# Phase 2: BPE diffusion model on FineWeb-Edu
pip install datasets tokenizers   # or: pip install -e ".[phase2]"
cd ../02_nano_dllm
python train_tokenizer.py          # Train BPE tokenizer (~30s)
python nano_dllm.py --train --depth 6   # ~1 hour on T4 GPU
python nano_dllm.py --depth 6 --prompt "The meaning of life is"

# Phase 3: Block diffusion on FineWeb-Edu
pip install -e ".[phase3]"    # datasets, tokenizers
cd ../03_block_diffusion
python block_dllm.py --train --depth 6 --block-size 4    # ~2 hours on P100
python block_dllm.py --depth 6 --block-size 4 --prompt "The meaning of life is"
```

---

## Evaluation

Run the DCLM CORE benchmark (22 tasks, centered accuracy) to compare dLLM vs autoregressive models:

```bash
# Install eval dependencies
pip install -e ".[eval]"

# Evaluate our dLLM (needs trained weights)
python eval/base_eval.py --model dllm --depth 6

# Compare with GPT-2
python eval/base_eval.py --hf-model gpt2

# Quick eval (limited examples per task)
python eval/base_eval.py --model dllm --depth 6 --max-per-task 100
```

DCLM CORE score is centered accuracy averaged over 22 tasks (0 = random guessing, 1 = perfect). GPT-2 (124M) scores ~0.257. See [`eval/`](eval/) for details.

---

## The 5 Changes from GPT to dLLM

A diffusion LLM is a GPT with exactly 5 surgical modifications. Everything else (transformer, RMSNorm, RoPE, MLP) stays the same.

| # | Change | GPT | dLLM |
|---|--------|-----|------|
| 1 | Vocabulary | Standard charset | Add `[MASK]` token |
| 2 | Attention | Causal (lower triangle) | Bidirectional (full) |
| 3 | Training data | Clean text sequences | Randomly masked sequences |
| 4 | Loss | All positions | Masked positions only |
| 5 | Generation | Sequential (1 token/step) | Parallel (confidence-based) |

See [`01_hello_diffusion/README.md`](01_hello_diffusion/README.md) for the full theory walkthrough with code examples and diagrams.

---

## Project Structure

```
Open-dLLM/
├── README.md                           # This file
├── LICENSE                             # MIT
├── pyproject.toml                      # Project config
├── 01_hello_diffusion/
│   ├── README.md                       # Theory: the 5 changes, training, inference
│   ├── hello_diffusion.py              # Complete dLLM in ~640 lines of annotated Python
│   ├── download_data.py                # Downloads Tiny Shakespeare
│   └── data.txt                        # Tiny Shakespeare (gitignored)
├── 02_nano_dllm/
│   ├── README.md                       # Theory: cosine schedule, ELBO, BPE, SwiGLU
│   ├── nano_dllm.py                    # BPE diffusion LLM in ~930 lines
│   ├── train_tokenizer.py              # BPE tokenizer training on FineWeb-Edu
│   └── tokenizer.json                  # Trained 32K BPE tokenizer
├── eval/
│   ├── core_eval.py                    # DCLM CORE evaluation engine (22 tasks)
│   └── base_eval.py                    # CLI: --model dllm or --hf-model gpt2
├── 03_block_diffusion/
│   └── block_dllm.py                 # Block diffusion with staircase mask (~1180 lines)
├── 04_advanced/                        # Coming soon
└── docs/
    ├── plans/                          # Design documents
    └── research.md                     # Paper notes and references
```

---

## Key References

- **[MDLM](https://arxiv.org/abs/2406.07524)** -- Sahoo et al., NeurIPS 2024. Foundational theory for masked diffusion language models.
- **[SEDD](https://arxiv.org/abs/2310.16834)** -- Lou et al., ICML 2024 Best Paper. Score entropy for discrete diffusion.
- **[LLaDA](https://arxiv.org/abs/2502.09992)** -- Nie et al., 2025. First 8B parameter dLLM, proves the approach scales.
- **[BD3-LMs](https://arxiv.org/abs/2503.09573)** -- Arriola et al., ICLR 2025 Oral. Block diffusion for efficient generation.
- **[Mercury](https://arxiv.org/abs/2506.17298)** -- Inception Labs, 2025. Production-grade diffusion LLM with KV caching.
- **[d1](https://arxiv.org/abs/2504.12216)** -- Zhao et al., 2025. Reinforcement learning for diffusion LLMs (diffu-GRPO).
- **[TiDAR](https://arxiv.org/abs/2511.08923)** -- 2025. Hybrid diffusion + autoregressive decoding.

---

## Credits

This project builds on the work of many others:

- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) by Nathan Barry -- educational diffusion LLM that inspired Phase 1
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy -- clean training script style we follow
- [MDLM](https://github.com/kuleshov-group/mdlm) by the Kuleshov group -- foundational masked diffusion implementation
- [LLaDA](https://github.com/ML-GSAI/LLaDA) -- scaling dLLMs to 8B parameters
- [Mercury](https://github.com/Inception-Labs/Mercury) by Inception Labs -- production diffusion LLM

---

## License

MIT
