# SmolDLM: Building Diffusion Language Models from Scratch

*A learning journey from "what is a dLLM?" to a 144M-param production block diffusion LM — in 5 phases.*

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

## Phase 4: Modern dLLM

**Goal:** Scale to 125M params with modern training infrastructure — Muon optimizer, FlexAttention, Liger kernels, multi-GPU DDP, and a 100B-token dataset.

### What Changed from Phase 3

| Change | Phase 3 | Phase 4 | Why |
|---|---|---|---|
| Architecture | 6L/384d/6h (depth dial) | 20L/768d/12h/3kv/1536MLP (fixed) | Production-scale, GQA |
| Attention | Manual staircase mask | FlexAttention (torch.compile'd) | 2-5x faster, no manual mask code |
| GQA | Multi-head (1:1) | Grouped query (4:1) | 4x less KV memory |
| Optimizer | AdamW | Muon + AdamW (2D->Muon, 1D/embed->AdamW) | Newton-on-manifold, faster convergence |
| Kernels | Pure PyTorch | Liger (RMSNorm, SwiGLU) | Fused ops, less memory |
| LR schedule | Cosine LR | WSD (warmup-stable-decay) | Simpler, easier resumption |
| Dataset | FineWeb-Edu only | 100B tokens (FinePDFs 50B + DCLM 30B + FineWeb-Edu 20B) | Multi-source, better diversity |
| Training | Single GPU | DDP multi-GPU + torch.compile | Scale to 8xA100 |
| Params | 36M | ~125M | Real scale |
| Tokenizer | BPE 32K | Qwen3-style BPE 32K with digit splitting, NFC | Better number handling |

### The ELBO Weight Bug — The Most Important Bug in the Repo

This is the single most educational bug we hit. The original LLaDA paper uses `1/t` as the ELBO weight. We copied it. But LLaDA uses a LINEAR noise schedule where `mask_prob = t`. With our COSINE schedule, `mask_prob != t`.

The correct ELBO weight is `1/mask_prob`, not `1/t`.

```
With the bug (cosine schedule):

  t=0.1  ->  mask_prob = 1 - cos^2(0.1 * pi/2) = 0.024
              Correct weight: 1/0.024 = 41.7
              Bug weight:     1/0.1   = 10.0     (4.2x under-weighted)

  t=0.01 ->  mask_prob = 1 - cos^2(0.01 * pi/2) = 0.00025
              Correct weight: 1/0.00025 = 4000
              Bug weight:     1/0.01    = 100     (40x under-weighted)
```

Effect: Loss plateaued at ~4.0 and refused to decrease further. The model couldn't learn the fine-grained refinement steps — the low-noise timesteps where only a few tokens are masked (the "polishing" phase of diffusion) were barely contributing to the gradient.

Fix: Change `elbo_weight = 1/t` to `elbo_weight = 1/mask_prob` where `mask_prob = 1 - cos^2(t * pi/2)`.

Lesson: When copying loss formulas from papers, verify the FULL derivation chain. The `1/t` simplification only holds for linear schedules. This bug doesn't appear in any paper's errata — you'd only catch it by re-deriving the ELBO from scratch.

### Muon Optimizer

Muon ("Momentum Orthogonalized Update") is a Newton-on-Stiefel-manifold optimizer. For 2D weight matrices, it applies momentum then orthogonalizes the update via Newton-Schulz iterations. This gives Newton-like curvature information without computing the full Hessian.

The param group split:
```
2D weight matrices (Q, K, V, MLP)  ->  Muon  (lr=0.02, momentum=0.95)
1D params + embeddings             ->  AdamW (lr=6e-4, betas=(0.9, 0.95))
```

Key gotchas we hit:
- `pip install muon` installs a bioinformatics package (scanpy), NOT the ML optimizer. Correct: `pip install git+https://github.com/KellerJordan/Muon`
- Muon requires DDP — it calls `dist.all_gather` internally. Single GPU needs `SingleDeviceMuonWithAuxAdam`.
- Param group keys must be EXACT: Muon groups take `{params, lr, momentum, weight_decay, use_muon}`, Adam groups take `{params, lr, betas, eps, weight_decay, use_muon}`. Extra keys crash it silently.
- Tied embeddings: `named_parameters()` yields the same tensor twice for `token_emb.weight` and `lm_head.weight`. Deduplicate with `data_ptr()` set, or the same tensor gets two optimizer states.

### FlexAttention

PyTorch's FlexAttention compiles custom attention masks into efficient CUDA kernels. Instead of materializing the full `(seq_len x seq_len)` mask tensor, you write a `score_mod` function that FlexAttention JIT-compiles into a fused kernel.

```
Before (Phase 3):
  mask = torch.zeros(2*L, 2*L)       # Materialize full mask
  for i, j in all_positions:          # O(L^2) Python loop
      mask[i][j] = staircase_rule(i, j, block_size)
  attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

After (Phase 4):
  def score_mod(score, b, h, q_idx, kv_idx):
      return staircase_rule(q_idx, kv_idx, block_size)  # Pure function
  attn_out = flex_attention(q, k, v, score_mod=score_mod)  # Compiled CUDA kernel
```

For our staircase mask, this eliminates manual mask construction entirely and runs 2-5x faster. The mask logic is the same — FlexAttention just compiles it into hardware-efficient code.

### Training Infrastructure

```
Phase 3:  python block_dllm.py --train           (single Kaggle P100)
Phase 4:  modal run modal_train.py               (8xA100-80GB, DDP)
          torchrun --nproc_per_node=2 modern_dllm.py --train  (local multi-GPU)
```

- **Modal cloud:** Serverless GPUs, volumes for checkpoints + data, HuggingFace secrets
- **torch.compile:** `mode="default"` for training (Inductor fusion). NOT `reduce-overhead` — that pins all layer activations for CUDA graphs, only useful for inference. First compilation takes 5-15 minutes (Triton kernel generation).
- **WSD scheduler:** warmup 2K steps, stable LR until step 40K, linear decay to step 50K. Simpler than cosine and allows easy checkpoint resumption (just resume in the stable phase).

### What We Learned

- **ELBO weight derivation depends on the noise schedule.** This is NOT stated in most papers. `1/t` is only correct for linear schedules.
- **Muon gives faster convergence** than AdamW for large 2D matrices but needs careful param grouping.
- **FlexAttention is a game-changer** for custom attention patterns. Write a Python function, get CUDA speed.
- **WSD scheduler > cosine LR** for training: simpler, and the stable phase allows easy resumption.
- **torch.compile with `mode="default"`** is the right choice for training. `reduce-overhead` is for inference.
- **Multi-source data matters.** FinePDFs + DCLM + FineWeb-Edu gives better diversity than FineWeb-Edu alone.

## Phase 5: Production Block Diffusion LM

**Goal:** Build a modular, production-quality dLLM — the final evolution. Every lesson from Phases 1-4 distilled into a clean, correct, fast system.

### What Changed from Phase 4

| Change | Phase 4 | Phase 5 | Why |
|---|---|---|---|
| Architecture | 20L/768d/12h | 30L/576d/9h (SmolLM2 shape) | Deeper+narrower, better scaling |
| Code structure | Single 1525-line file | Modular package (12 files, ~2000 lines) | Maintainability, testing |
| Tokenizer | 32K BPE custom | 49,152 (SmolLM2 merges + Qwen3 pre-tokenizer + 14 specials) | Proven tokenizer, wider vocab |
| Noise schedule | Cosine | LINEAR (mask_prob=t) | Trivially correct ELBO weight (1/t) |
| ELBO weight | 1/mask_prob (fixed from bug) | 1/t capped at 10 | Linear schedule: mask_prob=t, so 1/t is exact |
| Optimizer | External Muon package | Self-contained MuonClip (no external dep) | QK-Clip tau=100, cleaner |
| Attention | FlexAttention + MHA | FlexAttention + Gated Query Attention | GQA + learned gate dampens attention at init |
| Data | Streaming from HF | Document packing (no padding waste) | 10-15% throughput boost |
| FP8 | No | nanochat-style Float8Linear (H100+) | 15-30% faster matmuls |
| Loss | Liger FLCE | Chunked CE with grad_checkpoint | Liger FLCE backward BROKEN with reduction='none' |

### The Liger FLCE Bug

Liger Kernel's `FusedLinearCrossEntropyFunction` never materializes the full `(B*L, vocab)` logit tensor — a massive memory win. We adopted it. It worked... until it didn't.

```
loss.py with Liger FLCE:
  loss = LigerFLCE(logits, targets, reduction='none')  # Per-token loss for ELBO weighting
  loss = loss * elbo_weight                              # Weight by 1/t
  loss = loss.sum() / num_real_tokens                    # Normalize

Step 0:  loss=19.11   grad_norm=0.000000
Step 1:  loss=19.11   grad_norm=0.000000
Step 2:  loss=19.11   grad_norm=0.000000
...
Step 500: loss=19.11  grad_norm=0.000000    <-- model never learns
```

Forward pass: correct. Backward pass with `reduction='none'`: returns zeros. The bug is in Liger's backward kernel — it only handles `reduction='mean'` and `reduction='sum'` correctly. With `reduction='none'` (which we need for per-token ELBO weighting), the gradient is all zeros.

Tracked at `linkedin/Liger-Kernel#488`. Forward is correct; only backward is broken.

Fix: Replaced with chunked cross-entropy + `torch.utils.checkpoint`. Process 4096 tokens at a time (~1.5 GB peak) instead of full logits (24 GB). Slower than a working FLCE would be, but correct:

```
Chunked CE (loss.py):
  for chunk in tokens.split(chunk_size):
      chunk_logits = lm_head(chunk)            # Only materialize one chunk
      chunk_loss = F.cross_entropy(..., reduction='none')
      chunk_loss = chunk_loss * elbo_weight     # Per-token ELBO weighting
      total += chunk_loss.sum()
  loss = total / num_real_tokens
```

Lesson: Always verify `backward()`, not just `forward()`. A correct loss value means nothing if gradients are zero.

### Linear Noise Schedule — Learning from the ELBO Bug

Phase 5 deliberately uses a LINEAR noise schedule (`mask_prob = t`) instead of cosine. The ELBO bug from Phase 4 literally cannot happen:

```
Cosine schedule:  mask_prob = 1 - cos^2(t * pi/2)
                  ELBO weight = 1/mask_prob  (must derive, easy to get wrong)

Linear schedule:  mask_prob = t
                  ELBO weight = 1/t          (trivially correct)
```

This is a design decision born from a bug. The cosine schedule is theoretically "optimal" (Fisher-Rao geodesic), but the linear schedule is practically safer. And Quokka (2026) showed that linear actually outperforms cosine at convergence — the theoretical optimality of cosine doesn't survive to practice.

Other noise schedule details:
- `t ~ U[0.1, 1.0]` — skip the near-zero noise regime where predictions are trivial
- ELBO weight capped at `1/t_min = 10` — prevents gradient explosion at low noise
- Per-block min-1-masked guarantee — every block always has at least one masked token (Stable-DiffCoder insight)

### MuonClip Optimizer

Phase 5 uses a self-contained MuonClip implementation — no external dependency, no `pip install git+...` fragility.

```
MuonClip improvements over Phase 4's Muon:
  - QK-Clip (tau=100): caps the Q*K dot product to prevent attention logit explosion
  - Moonlight-style Newton-Schulz iterations for orthogonalization
  - RMS scaling factor for the update magnitude
  - Single param group list with `use_muon` flag per group (cleaner API)

Param split (same principle, cleaner implementation):
  2D weight matrices  ->  MuonClip  (lr=0.02)
  1D params + embed   ->  AdamW     (lr=3e-3, calibrated to SmolLM2-135M)
```

### Document Packing

Phase 4 right-padded short documents to `seq_len` with `<|padding|>` tokens. Padding tokens consume compute but contribute nothing to the loss — pure waste.

```
Phase 4 (right-padding):
  [doc1 tokens...][PAD][PAD][PAD][PAD][PAD][PAD]  <- 30% padding waste
  [doc2 tokens...][PAD][PAD][PAD]                  <- 15% padding waste

Phase 5 (document packing):
  [doc1 tokens...][EOS][doc2 tokens...][EOS][doc3] <- 0% padding waste
```

Attention is masked at document boundaries — each document only attends to itself. RoPE positions reset at each document boundary. This gives 10-15% more useful tokens per batch for no additional compute.

### FP8 Training

nanochat-style tensorwise FP8 scaling on H100+. Every `nn.Linear` (except `lm_head`) gets converted to `Float8Linear`. The matmul runs in FP8 with dynamic scaling, giving 15-30% faster throughput.

Requirements:
- All layer dimensions divisible by 16 (576d, 1536 MLP, 9 heads: all clean)
- Uses `@allow_in_graph` to avoid torchao + `grad_checkpoint` 3x slowdown
- `lm_head` skipped — its weight is accessed directly in `loss.py` for chunked CE
- `disable_fp8()` context manager for generation (FP8 introduces noise incompatible with iterative refinement)

### Modular Code Structure

Phase 4 was a single 1525-line file (`modern_dllm.py`). Phase 5 is a proper Python package:

```
05_optimized_dllm/
  phase5/
    config.py      -- TrainConfig dataclass, CLI parsing
    model.py       -- DiffusionLM (Transformer backbone)
    attention.py   -- FlexAttention + Gated Query Attention
    optim.py       -- MuonClip + AdamW param group setup
    schedule.py    -- WSD LR scheduler
    loss.py        -- Chunked CE with ELBO weighting
    data.py        -- Document packing, streaming, DataLoader
    checkpoint.py  -- Save/load, auto-resume
    generate.py    -- Block-by-block diffusion generation
    tokenizer.py   -- SmolLM2/Qwen3 hybrid tokenizer
    fp8.py         -- Float8Linear, conversion, disable context
    __init__.py
  train.py         -- Training orchestrator (imports from phase5/)
  build_tokenizer.py
```

Debugging a 1525-line file meant binary-searching for bugs across attention, loss, data, and optimizer code interleaved together. With 12 focused files, each module can be tested and verified independently.

### Training Run 1 Results (March 8, 2026)

| Metric | Value |
|---|---|
| Hardware | 4xH100 (85 GB each), bf16 AMP, DDP |
| Batch size | 128/GPU (effective 512) |
| Steps | 1,500 |
| Warmup | 105 steps |
| Decay start | Step 1,200 |
| Step-0 loss | 19.11 (ELBO-weighted; raw CE ~9.6) |
| Final val loss | 3.36 |
| Final train loss | 3.58 |
| Throughput | ~189K tok/s average |
| VRAM peak | 57.9/85 GB (68%) |
| Tokens seen | ~1.57B |
| Grad norm | 9.5 (step 0) -> 0.05 (converged) |

```
Loss (ELBO-weighted)
 19 |*
    |
 15 |
    | *
 12 |
    |   *
  9 |
    |     *
  6 |       * *
    |           * * * *
  3 |                   * * * * * *
    +--|------|------|------|------|----> steps
    0    300    600    900   1200  1500
```

Monotonic decrease, no anomalies, no loss spikes. The WSD decay phase (step 1200-1500) delivers a clean final drop.

### Generation Samples

At step 750 (temp=0.8):
> "Vietnam, officially the Socialist Republic of Vietnam, is a country in Southeast Asia. It is bordered by China to the north, Laos and Cambodia to the west, and the South China Sea to the east..."

Learned language structure, factual content, proper sentence construction. But degenerates into repetition after 2-3 sentences — expected at 1500 steps.

Greedy generation (temp=0) produces "0000..." — mode collapse at low temperature. This is a known dLLM artifact in early training: the model hasn't learned enough diversity to avoid the dominant mode.

### VRAM Benchmarks (1xH100, Phase 5 model)

```
                  Max Batch    Throughput     Peak VRAM
No grad_ckpt:        28        55,786 tok/s    77 GB     <- WINNER
SAC grad_ckpt:       40        22,087 tok/s    79 GB
Regular grad_ckpt:  160        16,648 tok/s    76 GB
```

The counterintuitive result: gradient checkpointing HURTS throughput on this model. Why? At 144M params on H100, the GPU is compute-bound, not memory-bound. Recomputing 30 layers of forward pass costs more time than the throughput gained from fitting larger batches. The GPU is already well-saturated at batch=28.

SAC (Selective Activation Checkpointing) is the worst of both worlds here: it still recomputes some ops (slower than no-ckpt) but saves enough activations to restrict batch severely. SAC shines on deeper/larger models where saved activations are small relative to total memory.

Rule of thumb for H100: if your model is under ~300M params, skip gradient checkpointing. The compute cost of recomputation exceeds the throughput benefit of larger batches.

### What We Learned

- **Linear noise schedule eliminates an entire class of bugs** (ELBO weight mismatch). Design for correctness, not theoretical optimality.
- **Liger FLCE backward is broken with `reduction='none'`** — always verify `backward()` not just `forward()`. A correct loss value means nothing if gradients are zero.
- **Chunked CE with grad_checkpoint** is the robust fallback for per-token loss weighting. Not as fast as a working fused kernel, but always correct.
- **Document packing eliminates padding waste** but requires careful attention masking + RoPE reset per document.
- **FP8 is free throughput on H100+** but needs careful exclusion of lm_head and a disable context for generation.
- **For 144M models on H100, gradient checkpointing HURTS throughput.** The GPU is compute-bound, not memory-bound. Measure before assuming ckpt helps.
- **Modular code (12 files) is infinitely easier to debug** than a single 1525-line file. The time spent refactoring was repaid tenfold during the Liger FLCE investigation.

## Evaluation: DCLM CORE Benchmark

We evaluate on the DCLM CORE benchmark (22 tasks, centered accuracy scoring). For dLLMs, evaluation uses Monte Carlo likelihood estimation:

1. Mask the answer region
2. Run the forward pass N times (default: 64 MC samples)
3. Average the cross-entropy loss over samples
4. Pick the answer choice with lowest average loss

This is different from AR models (which just compute perplexity in one forward pass) — dLLMs need multiple samples because each forward pass sees a different random mask pattern.

Centered accuracy subtracts random baselines: `centered = (acc - baseline) / (1 - baseline)`, so 0 = random guessing, 1 = perfect.

## Architecture Summary

All five phases share the same transformer backbone with progressive improvements:

| Component | Phase 1-3 | Phase 4 | Phase 5 |
|---|---|---|---|
| Normalization | RMSNorm | RMSNorm + Liger fused | RMSNorm + Liger fused |
| Positional encoding | RoPE | RoPE | RoPE (reset per doc) |
| Activation | SwiGLU | SwiGLU + Liger fused | SwiGLU + Liger fused |
| Attention | Manual masks | FlexAttention + GQA 4:1 | FlexAttention + Gated Query Attn |
| Embeddings | Tied | Tied | Tied |
| Optimizer | AdamW | Muon + AdamW | MuonClip + AdamW |
| LR schedule | Cosine decay | WSD | WSD |
| Compile | No | torch.compile (default) | torch.compile (per-block) |
| Precision | fp32 | bf16 AMP | bf16 AMP + FP8 matmuls |
| Noise schedule | Uniform/Cosine | Cosine | Linear |

## Training Loss Comparison

### Important: Why Raw Losses Differ Across Phases

The absolute loss values are **not directly comparable** between phases:

- **Phase 1**: Unweighted CE, 66-token vocabulary -> loss ~4.0 -> ~2.0
- **Phase 2**: ELBO-weighted (`1/t * CE`), 32K vocabulary -> loss ~42 -> ~19
- **Phase 3**: ELBO-weighted with per-block timesteps, 32K vocabulary -> loss ~17 -> ~7.7
- **Phase 4**: ELBO-weighted, cosine schedule, 32K vocabulary, 125M params
- **Phase 5**: ELBO-weighted, linear schedule, 49K vocabulary, 144M params -> loss ~19 -> ~3.4

Phase 2 losses are highest because ELBO weighting amplifies low-noise predictions by up to 10,000x. Phase 3 starts lower because the staircase mask gives the model more context (clean tokens from earlier blocks). Phase 5's step-0 loss (~19) is higher than Phase 3's final loss (7.7) because the vocabulary is larger (49K vs 32K) and the ELBO cap of 10 changes the loss magnitude.

The meaningful comparison is the **learning curve shape** — how fast each model learns and whether it converges cleanly.

## The Journey in Numbers

```
Phase       Params    Vocab    Dataset         Hardware       Best Val Loss
1 (hello)    10.7M      66    Tiny Shak (1MB)  Local GPU      ~2.0
2 (nano)     26M     32,768   FineWeb-Edu      Kaggle T4      20.8 (ELBO)
3 (block)    36M     32,768   FineWeb-Edu      Kaggle P100    7.74 (ELBO)
4 (modern)  125M     32,768   100B multi-src   8xA100-80GB    --
5 (prod)    144M     49,152   100B multi-src   4xH100-85GB    3.36 (ELBO)
```

From 66 characters on Tiny Shakespeare to 49,152 BPE tokens on 100B multi-source data. From a single local GPU to distributed training across 4-8 datacenter GPUs. From 10.7M parameters to 144M. Each phase earned its complexity by solving a real limitation of the previous one.

## Key Takeaways

1. **A dLLM is just 5 changes from GPT.** Bidirectional attention, mask token, random masking, masked loss, parallel decoding. The code diff is surprisingly small.

2. **Masked diffusion = weighted MLM.** The MDLM insight simplifies everything. No need for complex noise processes — just weight your MLM loss by `1/t`.

3. **Block size controls the AR-diffusion spectrum.** block_size=1 is autoregressive, block_size=N is full diffusion. Block_size=32 gives the best of both: parallel within blocks, causal across blocks, KV caching enabled.

4. **Separate your special tokens.** Using [MASK] for both noise and padding conflates two signals. Dedicated PAD and EOS tokens give the model cleaner training data.

5. **The staircase mask is the key innovation.** It enables block-by-block generation with KV caching while preserving bidirectional attention within blocks — the architecture behind Mercury's 1000+ TPS.

6. **Verify the full derivation chain when copying formulas.** The ELBO weight `1/t` is only correct for linear noise schedules. With cosine schedule, you need `1/mask_prob`. This single bug plateaued training for days. No paper warns about this.

7. **Always verify backward(), not just forward().** Liger FLCE produces correct loss values but zero gradients with `reduction='none'`. The model appears to train (loss prints) but never learns. Check `grad_norm` from step 0.

8. **Design for correctness, then optimize.** Phase 5 chose linear noise schedule over the "theoretically optimal" cosine specifically because it eliminates an entire class of bugs. The 0.1% theoretical loss difference is irrelevant when the alternative risks silent training failure.

9. **Measure before assuming.** Gradient checkpointing "obviously" helps training throughput — except when the GPU is compute-bound, where it actively hurts by 3.4x. VRAM benchmarking on the actual hardware is non-negotiable.

10. **Modular code pays for itself.** The refactor from 1525 lines to 12 files took a day. It saved a week of debugging when hunting the Liger FLCE backward bug — testing loss.py in isolation immediately isolated the problem.

## Bug Hall of Fame

Every bug here was a learning experience. Sorted by educational value:

| Bug | Phase | Impact | Root Cause |
|---|---|---|---|
| ELBO weight `1/t` vs `1/mask_prob` | 4 | Loss plateaued at 4.0 | Schedule-dependent derivation copied without verification |
| Liger FLCE backward zeros | 5 | Model never learned (grad_norm=0) | Upstream kernel bug with `reduction='none'` |
| Staircase mask `>=` vs `>` | 3 | Label leakage, trivial loss | Off-by-one in block boundary comparison |
| KV cache stale state | 3 | Training crash after eval generation | Cache not reset on `generate()` exit |
| Tied embedding init order | 3 | Random init overwritten | `apply(init)` must precede `lm_head.weight = token_emb.weight` |
| `pip install muon` wrong package | 4 | ImportError | PyPI `muon` is bioinformatics, not the ML optimizer |
| Tied embedding duplicate in optimizer | 4 | Same tensor, two optimizer states | `named_parameters()` yields tied weight twice |
| RoPE buffer too short | 3 | Generation crash at long sequences | Buffer precomputed for training length only |

## What's Next

Phase 5's loss curve shows no plateau at 1500 steps — the model is still learning. The immediate next step is a longer training run (5K-10K+ steps) to see where the loss floor actually is. At 1.57B tokens seen, the model has consumed barely 1.5% of the 100B-token dataset. Quokka scaling laws suggest 144M params can productively train on 15B+ tokens before diminishing returns.

Beyond longer training:
- **CAP training** (LLaDA 2.0): confidence-aware parallel decoding — a separate post-training stage that teaches the model to predict multiple tokens per diffusion step based on its own confidence. This is how Mercury hits 1000+ TPS.
- **Larger scale**: The Phase 5 architecture is designed to scale. 576d/30L at 144M is the base; 768d/30L at ~250M and 1024d/30L at ~450M are natural next steps. The modular codebase makes this a config change, not a rewrite.
- **Evaluation depth**: DCLM CORE gives us a number, but we need generation quality benchmarks (MAUVE, diversity metrics) to understand where the model actually fails.

---

*Built with PyTorch. Trained on Kaggle, Modal, and H100 clusters. Explained with ASCII diagrams. From 10.7M params on Tiny Shakespeare to 144M params on 100B tokens — every bug documented, every lesson earned.*

*Code at [github.com/hahuyhoang411/SmolDLM](https://github.com/hahuyhoang411/SmolDLM).*
