# SmolDLM

**Build diffusion language models from scratch, one phase at a time.**

A diffusion LLM is a GPT with 5 surgical changes. This repo teaches you all of them -- from a 10M char-level toy to a 144M production model trained on H100s. Think nanoGPT, but for discrete diffusion.

```
GPT (autoregressive):

  [The] --> [The][quick] --> [The][quick][brown] --> [The][quick][brown][fox]

  Left to right. One token at a time. Sequential.


dLLM (diffusion):

  [____][____][____][____]      all masked
  [____][quick][____][____]     fill the most confident
  [The][quick][____][fox]       fill the next most confident
  [The][quick][brown][fox]      done

  All positions at once. Parallel. Bidirectional.
```

---

## Five Changes from GPT to dLLM

A dLLM is not a new architecture. It is a standard transformer with exactly five modifications. Everything else -- RMSNorm, RoPE, SwiGLU, tied embeddings -- stays identical.

```
                  GPT                              dLLM
          +------------------+              +------------------+
          |   Vocabulary      |              |   Vocabulary      |
          |   V tokens        |     [1]      |   V + [MASK]      |
          +--------+---------+              +--------+---------+
                   |                                 |
          +--------v---------+              +--------v---------+
          |   Attention       |              |   Attention       |
          |   Causal (>)      |     [2]      |   Bidirectional   |
          +--------+---------+              +--------+---------+
                   |                                 |
          +--------v---------+              +--------v---------+
          |   Training        |              |   Training        |
          |   Clean text      |     [3]      |   Randomly mask   |
          |   next-token      |              |   fraction t      |
          +--------+---------+              +--------+---------+
                   |                                 |
          +--------v---------+              +--------v---------+
          |   Loss            |              |   Loss            |
          |   All positions   |     [4]      |   Masked only     |
          +--------+---------+              +--------+---------+
                   |                                 |
          +--------v---------+              +--------v---------+
          |   Decoding        |              |   Decoding        |
          |   Left-to-right   |     [5]      |   Confidence      |
          |   1 tok/step      |              |   parallel        |
          +------------------+              +------------------+
```

| # | What changes | GPT | dLLM |
|---|---|---|---|
| 1 | **Vocabulary** | Standard tokens | Add `[MASK]` token |
| 2 | **Attention** | Causal (lower triangle) | Bidirectional (full) |
| 3 | **Training** | Clean sequences, predict next token | Randomly masked sequences, predict masked tokens |
| 4 | **Loss** | Cross-entropy on all positions | Cross-entropy on masked positions only |
| 5 | **Decoding** | Sequential, 1 token/step | Parallel, fill most confident first |

Phase 1 implements all five in ~640 lines.

---

## Learning Roadmap

Five phases. Each builds on the last. Each is self-contained.

| Phase | What You Build | What You Learn | Hardware |
|---|---|---|---|
| `01_hello_diffusion` | ~10M char-level dLLM on Tiny Shakespeare | The 5 GPT-to-dLLM changes, confidence decoding | Any GPU / CPU |
| `02_nano_dllm` | ~26M BPE model on FineWeb-Edu | Cosine noise schedule, ELBO loss weighting, BPE tokenization, SwiGLU | Kaggle T4 |
| `03_block_diffusion` | ~36M block diffusion model | Staircase attention mask, KV caching, block-by-block generation | Kaggle P100 |
| `04_modern_dllm` | ~125M production model (single file) | FlexAttention, GQA, Muon optimizer, Liger kernels, DDP, torch.compile | 2xA10G / A100 |
| `05_optimized_dllm` | ~144M modular dLLM | FP8 training, MuonClip, document packing, linear noise schedule, Gated Query Attention | 4-8xH100 |

### Architecture Evolution

```
Phase 1    6L / 384d / 6h          10M params    char-level, uniform noise
                                                  |
Phase 2    6L / 384d / 6h          26M params    + BPE 32K, cosine schedule, ELBO
                                                  |
Phase 3    6L / 384d / 6h          36M params    + staircase mask, KV cache, block diffusion
                                                  |
Phase 4   20L / 768d / 12h / 3kv  125M params   + FlexAttention, GQA, Muon, Liger, DDP
                                                  |
Phase 5   30L / 576d / 9h / 3kv   144M params   + FP8, MuonClip, doc packing, linear noise
```

---

## Quick Start

### Phase 1: Hello Diffusion (start here)

```bash
git clone https://github.com/hahuyhoang411/SmolDLM.git
cd SmolDLM

pip install torch

# Download Tiny Shakespeare (~1.1 MB)
python 01_hello_diffusion/download_data.py

# Train (~20 min on GPU, ~2 hours on CPU)
python 01_hello_diffusion/hello_diffusion.py --train

# Generate
python 01_hello_diffusion/hello_diffusion.py
```

### Phase 2-3: BPE, Block Diffusion

```bash
pip install -e ".[phase2]"

python 02_nano_dllm/train_tokenizer.py              # Train BPE tokenizer
python 02_nano_dllm/nano_dllm.py --train --depth 6  # ~1 hour on Kaggle T4

pip install -e ".[phase3]"

python 03_block_diffusion/block_dllm.py --train --depth 6 --block-size 4  # ~5 hours on P100
```

### Phase 4-5: Modern Training

```bash
pip install -e ".[phase4]"

# Local (2 GPUs)
torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train

# Modal cloud (A100/H100)
pip install -e ".[cloud]"
modal run modal_train.py

# Phase 5 (8xH100)
pip install -e ".[phase5]"
modal run modal_train.py
```

---

## Training Results

### Phase 5: 144M on 4xH100

| Metric | Value |
|---|---|
| Steps | 1,500 |
| Loss | 19.11 --> 3.36 (val) |
| Throughput | 189K tok/s |
| Tokens seen | ~1.57B |
| VRAM peak | 57.9 / 85 GB (68%) |
| Hardware | 4x H100 80GB, bf16, DDP |

Loss curve monotonically decreasing, no anomalies. The model learns language structure by step 750 (generates coherent openings like "Vietnam, officially...") and has substantial remaining capacity -- no plateau in sight.

---

## Evaluation

DCLM CORE benchmark -- 22 tasks, centered accuracy (0 = random, 1 = perfect).

dLLMs need a different eval protocol than autoregressive models. Instead of computing perplexity in one forward pass, we use **Monte Carlo likelihood estimation**: mask the answer region, run N forward passes with different random masks, average the cross-entropy, pick the lowest-loss answer.

```bash
pip install -e ".[eval]"

python eval/base_eval.py --model dllm --depth 6          # Evaluate a dLLM
python eval/base_eval.py --hf-model gpt2                 # Compare with GPT-2
python eval/base_eval.py --model dllm --max-per-task 100  # Quick subset
```

---

## How dLLM Generation Works

Autoregressive models generate left-to-right. dLLMs denoise -- starting from all `[MASK]` tokens, revealing the most confident predictions first:

```
Step 0:  [____][____][____][____][____][____]    all masked
Step 1:  [____][____][ , ][____][____][ . ]     punctuation first (highest confidence)
Step 2:  [ The][____][ , ][ the][ on ][ . ]     function words next
Step 3:  [ The][ cat][ , ][ the][ on ][ . ]     content words last
```

Block diffusion (Phases 3-5) adds structure: generate one block of B tokens at a time, with KV-cached context from prior blocks. `block_size=1` is autoregressive. `block_size=seq_len` is full-sequence diffusion. The sweet spot is in between.

```
Block 1: [____]x32  -->  denoise  -->  [The cat sat on the ...]
Block 2: [____]x32  -->  denoise  -->  [mat. It was a dark ...]    ^ KV cache from block 1
Block 3: [____]x32  -->  denoise  -->  [and stormy night.   ]    ^ KV cache from blocks 1-2
```

---

## The Theory in 60 Seconds

**Masked diffusion = weighted MLM.** Sample a noise level `t ~ U[0,1]`, mask that fraction of tokens, predict the originals, weight the loss by `1/t`.

```
L = E_t[ (1/t) * CrossEntropy(predicted, original) | masked positions ]
```

The `1/t` weight upweights low-noise timesteps (few masks, hard predictions) and downweights high-noise timesteps (many masks, easy guessing). This gives a proper variational lower bound (ELBO) on log-likelihood.

**Why `1/t`?** For a linear schedule where mask_prob = t, the rate of information revealed is constant and the weight simplifies to `1/t`. The [JOURNEY.md](JOURNEY.md) has the full derivation.

---

## Project Structure

```
SmolDLM/
├── 01_hello_diffusion/         # Phase 1: char-level dLLM (~640 lines)
│   ├── hello_diffusion.py
│   └── download_data.py
├── 02_nano_dllm/               # Phase 2: BPE + ELBO + cosine schedule (~930 lines)
│   ├── nano_dllm.py
│   └── train_tokenizer.py
├── 03_block_diffusion/         # Phase 3: staircase mask + KV cache (~1180 lines)
│   └── block_dllm.py
├── 04_modern_dllm/             # Phase 4: Muon, FlexAttention, GQA, 125M
│   └── modern_dllm.py
├── 05_optimized_dllm/             # Phase 5: production 144M model
│   ├── train.py                #   Training orchestrator
│   ├── build_tokenizer.py      #   SmolLM2/Qwen3 hybrid tokenizer
│   └── phase5/                 #   Modular package
│       ├── config.py model.py attention.py
│       ├── optim.py schedule.py loss.py
│       ├── data.py checkpoint.py generate.py
│       └── tokenizer.py fp8.py
├── eval/                       # DCLM CORE benchmark (22 tasks)
├── modal_train.py              # Modal cloud training launcher
└── kaggle/                     # Kaggle training notebooks
```

---

## Companion Documents

- **[JOURNEY.md](JOURNEY.md)** -- The full learning narrative across all five phases: architecture decisions, training curves, what worked and what didn't. Start here if you want the story, not just the code.
- **[BUGS.md](BUGS.md)** -- Every significant bug found during development, preserved for education. 15 bugs, 3 critical (silent correctness failures). Each traces symptom to root cause to fix. Most repos hide their bugs. We celebrate them.

---

## Key References

```
BERT MLM (2018) -- masked token prediction
  └── D3PM (Austin et al., 2021) -- discrete diffusion formalized
      ├── SEDD (Lou et al., ICML 2024 Best Paper) -- score entropy
      └── MDLM (Sahoo et al., NeurIPS 2024) -- masked diffusion = weighted MLM
          ├── LLaDA 8B (Nie et al., Feb 2025) -- first large-scale dLLM
          ├── BD3-LMs (Arriola et al., ICLR 2025 Oral) -- block diffusion
          ├── Mercury (Inception Labs, Jun 2025) -- 1000+ TPS production dLLM
          └── LLaDA 2.1 (Feb 2026) -- T2T editing, 1587 TPS
```

- **[MDLM](https://arxiv.org/abs/2406.07524)** (Sahoo et al., NeurIPS 2024) -- Foundational theory: masked diffusion is weighted MLM
- **[SEDD](https://arxiv.org/abs/2310.16834)** (Lou et al., ICML 2024 Best Paper) -- Score entropy for discrete diffusion
- **[LLaDA](https://arxiv.org/abs/2502.09992)** (Nie et al., 2025) -- First 8B dLLM, proves the approach scales
- **[BD3-LMs](https://arxiv.org/abs/2503.09573)** (Arriola et al., ICLR 2025 Oral) -- Block diffusion theory and staircase mask
- **[Mercury](https://arxiv.org/abs/2506.17298)** (Inception Labs, 2025) -- Production dLLM, 1000+ tokens/sec
- **[d1](https://arxiv.org/abs/2504.12216)** (Zhao et al., 2025) -- Reinforcement learning for dLLMs

---

## Credits

- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) by Nathan Barry -- educational dLLM that inspired Phase 1
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy -- training script style and engineering patterns
- [MDLM](https://github.com/kuleshov-group/mdlm) by the Kuleshov group -- foundational masked diffusion implementation
- [LLaDA](https://github.com/ML-GSAI/LLaDA) by ML-GSAI -- scaling dLLMs to 8B
- [Mercury](https://github.com/Inception-Labs/Mercury) by Inception Labs -- production dLLM, KV caching for diffusion
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by HuggingFace -- architecture and tokenizer foundation for Phase 5

---

## License

Apache 2.0
