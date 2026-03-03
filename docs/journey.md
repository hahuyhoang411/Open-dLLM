# Open-dLLM: Building Diffusion Language Models from Scratch

*A learning journey from "what is a dLLM?" to block diffusion with KV caching — in 3 phases.*

## The Premise

Standard LLMs (GPT, LLaMA) generate text left-to-right, one token at a time. Diffusion LLMs start with a fully masked sequence and iteratively fill in tokens **in parallel** — like solving a crossword puzzle instead of writing a story.

This is an educational rebuild of that idea, from first principles. Think nanoGPT, but for dLLMs.

```
GPT (autoregressive):  [The] → [The][cat] → [The][cat][sat] → ...
dLLM (diffusion):      [___][___][___] → [The][___][sat] → [The][cat][sat]
```

## The Research Landscape

Before writing code, we surveyed the field. The key lineage:

```
BERT MLM (2018) — masked token prediction
  └── D3PM (Austin et al., 2021) — discrete diffusion formalized
      ├── SEDD (Lou et al., ICML 2024 Best Paper) — score entropy
      └── MDLM (Sahoo et al., NeurIPS 2024) — masked diffusion = weighted MLM
          ├── LLaDA 8B (Nie et al., Feb 2025) — first large-scale dLLM
          ├── BD3-LMs (Arriola et al., ICLR 2025 Oral) — block diffusion theory
          ├── Mercury (Inception Labs, Jun 2025) — 1000+ TPS production dLLM
          └── LLaDA 2.1 (Feb 2026) — T2T editing, 1587 TPS
```

The core insight from MDLM: masked diffusion is just **weighted masked language modeling**. The loss is `L = (1/t) * CE_masked(t)` — sample a noise level `t`, mask that fraction of tokens, predict the originals, weight by `1/t`. That's it. No complex noise processes, no score matching — just weighted MLM with a schedule.

Three families exist (absorbing/masking, score entropy, flow matching), but absorbing is simplest and scales best. We went with that.

## Phase 1: Hello Diffusion

**Goal**: Prove that 5 changes turn a GPT into a dLLM.

| | GPT | dLLM (Phase 1) |
|---|---|---|
| Attention | Causal (triangular mask) | Bidirectional (full attention) |
| Training | Next-token prediction | Masked-token prediction |
| Loss | CE on all positions | CE on masked positions only |
| Vocab | Characters | Characters + [MASK] |
| Generation | Left-to-right, 1 tok/step | Parallel, confidence-based |

### Setup

| Detail | Value |
|---|---|
| Dataset | Tiny Shakespeare (1.1 MB) |
| Vocab | 66 tokens (65 chars + [MASK]) |
| Model | 6 layers, 384 dim, 6 heads |
| Params | ~10.7M |
| Context | 256 characters |
| Training | 10K steps, batch=64, lr=3e-4 |
| Time | ~20 min on GPU |
| Activation | ReLU squared |
| Positional encoding | RoPE |
| Normalization | RMSNorm (no learnable params) |

### The 5 Changes (annotated in code as [DIFF 1-5])

1. **Add [MASK] token** (id=0) to vocabulary
2. **Bidirectional attention** — `is_causal=False` instead of causal mask
3. **Random masking** — sample `t ~ U[0,1]`, mask that fraction of tokens
4. **Loss on masked positions only** — don't penalize predictions at visible positions
5. **Confidence-based decoding** — unmask positions where the model is most confident first

### Generation: How Parallel Decoding Works

```
Step 0: [___][___][___][___][___][___]   (all masked)
Step 1: [___][___][ , ][___][___][___]   (confident about comma)
Step 2: [The][___][ , ][___][sat][___]   (fill high-confidence)
Step 3: [The][cat][ , ][the][sat][mat]   (done)
```

At each step: forward pass → softmax → top-k sampling → unmask positions above confidence threshold. If nothing qualifies, force-unmask the single most confident position.

### What We Learned

- dLLMs work. 5 changes is genuinely all it takes.
- The model learned Shakespeare-like patterns (character names, verse structure).
- Char-level tokenization limits practical quality — you need BPE for real text.
- Uniform noise schedule (Phase 1) works but isn't optimal — cosine is better (Phase 2).

## Phase 2: Nano dLLM

**Goal**: Scale to real data with proper theory (cosine schedule, ELBO loss, BPE).

### What Changed from Phase 1

| Change | Phase 1 | Phase 2 |
|---|---|---|
| Tokenizer | 66 chars | 32,768 BPE tokens |
| Dataset | Tiny Shakespeare (1.1MB) | FineWeb-Edu (1.5T tokens, streaming) |
| Noise schedule | Uniform `t ~ U[0,1]` | Cosine `alpha_t = cos^2(t * pi/2)` |
| Loss weighting | Unweighted CE | ELBO: `(1/t) * CE_masked` |
| Activation | ReLU squared | SwiGLU |
| Architecture | Fixed 6-layer | Depth-parametrized (`--depth N`) |
| Training | 10K steps | 20K steps |
| Hardware | Local GPU | Kaggle T4 |

### The Depth Dial

One `--depth` parameter controls everything:

| Depth | Layers | Dim | Heads | Params | Hardware |
|---|---|---|---|---|---|
| 6 | 6 | 384 | 6 | ~26M | Kaggle T4 |
| 12 | 12 | 768 | 12 | ~150M | Kaggle T4 |
| 24 | 24 | 1536 | 24 | ~850M | A100+ |

Formula: `n_layer = depth`, `n_embd = depth * 64`, `n_head = depth`, `head_dim = 64` (fixed).

### Cosine Noise Schedule (Why It's Optimal)

The uniform schedule from Phase 1 spends equal time at all noise levels. But learning is richest at intermediate levels — very low noise (almost clean) gives trivial predictions, very high noise (almost fully masked) gives random guessing. The cosine schedule spends more budget in the sweet spot:

```
alpha_t = cos^2(t * pi/2)    -- fraction of tokens kept clean
mask_prob = 1 - alpha_t       -- fraction of tokens masked

  mask_prob
  1.0 |                          *
      |                       **
      |                    **
  0.5 |              ****
      |          **
      |       *
  0.0 |*****
      +--|---------|---------|-->
      0         0.5         1.0  t
```

Zhang & Syed (2025) proved this is the Fisher-Rao geodesic on the probability simplex — the mathematically optimal path between clean and fully masked.

### ELBO Loss Weighting

Phase 1 used unweighted cross-entropy. Phase 2 uses the MDLM evidence lower bound:

```
L = E_t[ (1/t) * CE_masked(t) ]
```

The `1/t` weight upweights low-noise predictions (few masks, hard predictions) and downweights high-noise predictions (many masks, easy/random). This gives a proper variational bound on log-likelihood.

### Training Results (Phase 2, depth=6)

| Metric | Value |
|---|---|
| Final train loss | ~8.2 |
| Final val loss | ~8.1 |
| Throughput | ~15K tok/s (Kaggle T4) |
| Training time | ~1 hour |
| Weight file | 102 MB |

### The BPE Tokenizer

Trained on 50K FineWeb-Edu documents. Special tokens:
- `[MASK]` = 0 (diffusion noise)
- `<|endoftext|>` = 1 (document boundary / EOS)
- `<|padding|>` = 2 (right-padding for short sequences)

Why 3 tokens, not 1? In Phase 1 we used `[MASK]` for both noise and padding. But these are different roles — noise means "predict what goes here," padding means "nothing here." Separating them gives the model a cleaner signal.

No BOS token needed — dLLMs see all positions simultaneously, there's no "first position" that needs a seed.

## Phase 3: Block Diffusion

**Goal**: Bridge the AR-diffusion spectrum with block-causal attention and KV caching.

### The Core Idea

Full-sequence diffusion (Phase 2) denoises all 512 tokens at once — great for parallelism, but can't do variable-length generation or KV caching. Autoregressive models generate 1 token at a time — great for caching, but fully serial.

Block diffusion is the middle ground: **denoise a block of B tokens in parallel, then move to the next block**. With B=1 it's AR. With B=512 it's Phase 2. With B=4 (default), it's the sweet spot.

```
Block diffusion (block_size=4):

Block 1: [___][___][___][___]  → denoise → [The][cat][sat][on ]
Block 2: [___][___][___][___]  → denoise → [the][mat][ . ][\n ]
Block 3: [___][___][___][___]  → denoise → [It ][was][a  ][dar]
                                              ↑ KV cache from blocks 1-2
```

Each block sees:
- **Bidirectional** within itself (all positions in the block see each other)
- **Causal** across blocks (block 3 sees blocks 1-2 via KV cache, but not vice versa)

This is the **staircase attention mask** from BD3-LMs (Arriola et al., ICLR 2025 Oral).

### Staircase Mask

The input is doubled: `[x_noisy || x_clean]` (2L tokens total). The mask has a specific structure:

```
Staircase mask (block_size=4, 3 blocks shown):

         Noisy half        Clean half
       b1    b2    b3    b1    b2    b3
  b1 [ ████  ....  ....  ████  ....  .... ]  noisy b1 sees: own noisy + own clean
  b2 [ ....  ████  ....  ████  ████  .... ]  noisy b2 sees: own noisy + clean b1,b2
  b3 [ ....  ....  ████  ████  ████  ████ ]  noisy b3 sees: own noisy + clean b1,b2,b3
  b1 [ ....  ....  ....  ████  ....  .... ]  clean b1 sees: own clean only
  b2 [ ....  ....  ....  ████  ████  .... ]  clean b2 sees: clean b1,b2
  b3 [ ....  ....  ....  ████  ████  ████ ]  clean b3 sees: clean b1,b2,b3
```

Critical bug found during implementation: the mask must use strict `>` (not `>=`) for block indexing — `>=` leaks the current block's clean tokens into its own noisy prediction, which is label leakage.

### What Changed from Phase 2

| Change | Phase 2 | Phase 3 |
|---|---|---|
| Attention mask | Bidirectional (all-to-all) | Staircase (block-causal) |
| Noise sampling | Per-sequence `t` | Per-block `t` (each block gets own noise level) |
| Input format | `x_noisy` only | `[x_noisy \|\| x_clean]` (concatenated) |
| Generation | Full-sequence parallel | Block-by-block with KV cache |
| Variable length | No (fixed 512) | Yes (generate until EOS or max) |
| KV caching | No | Yes (across blocks) |
| EOS stopping | No | Yes (`<\|endoftext\|>` triggers stop) |

### Training Results (Phase 3, depth=6, block_size=4)

| Metric | Value |
|---|---|
| Params | 35.78M |
| GPU | Kaggle P100 (16 GB) |
| Steps | 20,000 |
| Throughput | ~15,575 tok/s |
| Final train loss | 7.880 |
| Final val loss | 7.742 |
| Training time | ~5 hours |
| Weight file | 137 MB |
| Generation speed | ~155-199 tok/s (on P100) |

### Loss Curve Progression

```
Loss
18 |*
   |
15 |
   |  *
12 |     *
   |       * *
 9 |           * * * * *
   |                     * * * * * *
 7 |                                 * * * *
   +--|-----|-----|-----|-----|-----|----> steps
   0  2K    5K    8K   12K   16K   20K
```

The loss dropped from 17.2 → 7.7 over 20K steps — a 10-point improvement. The curve shows healthy convergence with no divergence or plateaus.

### Generation Samples (step 20K, temp=0.8, top_k=5)

At step 500 (early):
> "The name of the world is the of the world history of the one of a few in the world of the world, the time of the world..."

At step 18500:
> "The name of the group of men in the history of the United States is to the same, and it is not a common name of women."

At step 19000:
> "The health and our body are all vital. There are many different..."

At step 20000 (final, prompted "The meaning of life is"):
> "The meaning of life is the meaning of life. It is the meaning of the meaning of life..."

The model learned English grammar, factual patterns, and list/heading structure. Repetition is expected for a 36M param model at 20K steps — quality improves significantly with more depth and training.

### Bugs Found Along the Way

1. **Staircase mask label leakage**: Used `>=` instead of `>` for block boundary comparison. The model saw the answer during training, learned nothing useful.

2. **Tied embedding init order**: `self.apply(_init_weights)` must come BEFORE `self.lm_head.weight = self.token_emb.weight`, otherwise the init overwrites the tie.

3. **RoPE buffer too short**: Generation with KV caching exceeds training sequence length. Precompute for `max(2*seq_len, 4096)`.

4. **KV cache stale state**: `generate()` reset the cache at entry but not exit. When called mid-training (step 500 eval sample), stale batch_size=1 cache crashed the next training forward pass (batch_size=32).

5. **Qwen 3 tokenizer OOM**: First attempt used Qwen 3's 151K vocab tokenizer. P100 OOMed on the embedding layer alone (~151K * 384 * 4 bytes = 232 MB just for embeddings). Switched to our 32K BPE tokenizer.

## Evaluation: DCLM CORE Benchmark

We evaluate on the DCLM CORE benchmark (22 tasks, centered accuracy scoring). For dLLMs, evaluation uses Monte Carlo likelihood estimation:

1. Mask the answer region
2. Run the forward pass N times (default: 64 MC samples)
3. Average the cross-entropy loss over samples
4. Pick the answer choice with lowest average loss

This is different from AR models (which just compute perplexity in one forward pass) — dLLMs need multiple samples because each forward pass sees a different random mask pattern.

Centered accuracy subtracts random baselines: `centered = (acc - baseline) / (1 - baseline)`, so 0 = random guessing, 1 = perfect.

## Architecture Summary

All three phases share the same transformer backbone:

| Component | Choice | Why |
|---|---|---|
| Normalization | RMSNorm (no learnable params) | Simpler, works as well as LayerNorm |
| Positional encoding | RoPE | Handles unseen lengths, no learned params |
| Activation | SwiGLU (Phase 2+) | Modern standard, gated mechanism |
| MLP ratio | 8/3 expansion | SwiGLU standard (same param count as 4x ReLU) |
| Attention | Bidirectional / Staircase | No causal mask (dLLM sees all positions) |
| Embeddings | Tied (input = output) | Halves embedding params, standard practice |
| Weight init | Xavier uniform, embedding scaled | Stable training from step 0 |
| Optimizer | AdamW (weight_decay=0.1) | Standard for transformers |
| LR schedule | Linear warmup → cosine decay | 1K warmup steps, decay to 0.1x |

## Training Loss Comparison

### Important: Why Raw Losses Differ Across Phases

The absolute loss values are **not directly comparable** between phases:

- **Phase 1**: Unweighted CE, 66-token vocabulary → loss ~4.0 → ~2.0
- **Phase 2**: ELBO-weighted (`1/t * CE`), 32K vocabulary → loss ~42 → ~19
- **Phase 3**: ELBO-weighted with per-block timesteps, 32K vocabulary → loss ~17 → ~7.7

Phase 2 losses are highest because ELBO weighting amplifies low-noise predictions by up to 10,000x. Phase 3 starts lower because the staircase mask gives the model more context (clean tokens from earlier blocks), making predictions easier.

The meaningful comparison is the **learning curve shape** — how fast each model learns and whether it converges cleanly.

### Phase 2: Nano dLLM (depth=6, 26M params, Kaggle T4)

```
Val Loss (ELBO-weighted)
 40 |*
    |
 30 | *  *
    |      * * * * *
 25 |               * * *
    |                    * * * * * * * * * * *
 20 |                                         * * * * * * * * * * * * * * *
    |
    +--|-----|-----|-----|-----|-----|-----|-----|----> steps (x1000)
    0     2     4     6     8    10    12    14   16    18    20
```

| Step | Train Loss | Val Loss | LR |
|------|-----------|---------|-----|
| 0 | 41.66 | 38.07 | 0.000001 |
| 1000 | 29.30 | 28.14 | 0.001000 |
| 2000 | 25.35 | 26.73 | 0.000994 |
| 5000 | 24.12 | 22.83 | 0.000905 |
| 10000 | 22.53 | 22.02 | 0.000587 |
| 15000 | 23.38 | 22.13 | 0.000245 |
| 19999 | 18.80 | 20.83 | 0.000100 |

**Observations**:
- Fast initial drop (41 → 28 in first 1K steps) during warmup
- Slow, noisy convergence after step 5K — loss oscillates between 19-24
- Train-val gap stays small (~1-2 points) — no overfitting
- Final val loss 20.8, with the ELBO weighting making this number artificially large

### Phase 3: Block Diffusion (depth=6, 36M params, Kaggle P100)

```
Val Loss (ELBO-weighted)
 18 |*
    |
 14 |
    |
 10 |   * *
    |       * * * *
  8 |             * * * * * * * * * * * * * * * * * * * * * * * * * * *
    |
    +--|-----|-----|-----|-----|-----|-----|-----|----> steps (x1000)
    0     2     4     6     8    10    12    14   16    18    20
```

| Step | Train Loss | Val Loss | LR |
|------|-----------|---------|-----|
| 0 | 17.25 | 17.24 | 0.000001 |
| 1000 | 10.27 | 10.31 | 0.001000 |
| 2000 | 9.56 | 9.54 | 0.000994 |
| 5000 | 8.84 | 8.70 | 0.000905 |
| 10000 | 8.35 | 8.30 | 0.000587 |
| 15000 | 8.02 | 8.09 | 0.000245 |
| 19999 | 7.88 | 7.74 | 0.000100 |

**Observations**:
- Much steeper initial drop (17.2 → 10.3 in first 1K steps)
- Smoother convergence than Phase 2 — less oscillation
- Train-val gap is negligible (~0.1 points) — very well regularized
- Still improving at step 20K — more steps would likely push below 7.5

### Phase 2 vs Phase 3: Side-by-Side

| Metric | Phase 2 (Nano dLLM) | Phase 3 (Block Diffusion) |
|---|---|---|
| Params | 26M | 36M |
| Tokenizer | BPE 32K (`[MASK]` only) | BPE 32K (`[MASK]`, EOS, PAD) |
| Attention | Full bidirectional | Staircase (block-causal) |
| Noise | Per-sequence | Per-block |
| Input format | `x_noisy` | `[x_noisy \|\| x_clean]` |
| Start val loss | 38.07 | 17.24 |
| Final val loss | 20.83 | 7.74 |
| Relative drop | 45% | 55% |
| Convergence | Noisy, oscillating | Smooth, monotonic |
| Train-val gap | ~1-2 points | ~0.1 points |
| Throughput | ~15K tok/s (T4) | ~15.5K tok/s (P100) |
| Training time | ~5 hours | ~5 hours |

**Why Phase 3 converges better:**
1. **Staircase mask gives more context** — when denoising block N, the model sees clean tokens from blocks 1..N-1. Phase 2 sees only noisy tokens everywhere.
2. **Per-block timesteps** — each block gets its own noise level, creating a richer training signal per sample.
3. **Separate PAD token** — no longer conflating padding with noise masking.
4. **10M more params** — Phase 3 has 36M vs 26M due to the doubled input length (embedding overhead).

## What's Next: Phase 4

Planned techniques:
- **FlexAttention** (PyTorch native) — replace manual staircase mask with torch.compile'd attention
- **RL for dLLMs** (diffu-GRPO from d1) — reward-guided generation
- **Hybrid AR+diffusion** (TiDAR-style) — 4.7x speedup
- **Scaling** — depth=12+ on better hardware

## Key Takeaways

1. **A dLLM is just 5 changes from GPT.** Bidirectional attention, mask token, random masking, masked loss, parallel decoding. The code diff is surprisingly small.

2. **Masked diffusion = weighted MLM.** The MDLM insight simplifies everything. No need for complex noise processes — just weight your MLM loss by `1/t`.

3. **Block size controls the AR↔diffusion spectrum.** block_size=1 is autoregressive, block_size=N is full diffusion. Block_size=4 gives the best of both: parallel within blocks, causal across blocks, KV caching enabled.

4. **Separate your special tokens.** Using [MASK] for both noise and padding conflates two signals. Dedicated PAD and EOS tokens give the model cleaner training data.

5. **The staircase mask is the key innovation.** It enables block-by-block generation with KV caching while preserving bidirectional attention within blocks — the architecture behind Mercury's 1000+ TPS.

---

*Built with PyTorch, trained on Kaggle, explained with ASCII diagrams. Code at [github.com/hahuyhoang411/Open-dLLM](https://github.com/hahuyhoang411/Open-dLLM).*
