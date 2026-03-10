# Phase 3: Block Diffusion

**A block diffusion language model with staircase attention and KV caching, in ~1189 lines.**

---

## What You'll Learn

- Block diffusion: the middle ground between autoregressive and full-sequence diffusion
- The staircase attention mask (from BD3-LMs, ICLR 2025 Oral)
- KV caching for efficient block-by-block generation
- Per-block noise levels vs per-sequence noise
- Variable-length generation with EOS stopping

---

## The Core Idea

Phase 2 denoises all 512 tokens at once -- great parallelism, but no way to cache
previous computation. Autoregressive models generate one token at a time -- great
for KV caching, but fully serial. Block diffusion sits in the middle: denoise a
block of B tokens in parallel via diffusion, then advance to the next block
autoregressively.

```
block_size=1   -> fully autoregressive (1 token per diffusion step)
block_size=4   -> 4 tokens denoised in parallel per block (default)
block_size=512 -> equivalent to Phase 2 (full-sequence diffusion)
```

One parameter controls the entire AR-diffusion spectrum:

```
AR (B=1):               Block Diffusion (B=4):       Full Diffusion (B=512):
  token by token           block by block               all at once
  +-+-+-+-+-+-+-+          +----+----+----+             +------------------------+
  |1|2|3|4|5|6|7|          | 1  | 2  | 3  |             |           1            |
  +-+-+-+-+-+-+-+          +----+----+----+             +------------------------+
  7 serial steps           3 block steps                 1 parallel step
  KV cache: yes            KV cache: yes                 KV cache: no
  variable length: yes     variable length: yes          variable length: no
```

At block_size=4, the model generates 4 tokens simultaneously within each block,
then slides forward. It gets the caching benefits of AR (only the current block
is active) and the parallelism of diffusion (4 tokens decoded at once).

---

## What Changed from Phase 2

| Feature | Phase 2 | Phase 3 | Why |
|---------|---------|---------|-----|
| Attention | Bidirectional (`is_causal=False`) | Staircase mask (block-causal) | Enables KV caching across blocks |
| Noise | Per-sequence `t` | Per-block `t` | Richer training signal per sequence |
| Input shape | `x_noisy` (B, L) | `[x_noisy \|\| x_clean]` (B, 2L) | Clean half provides conditioning |
| Generation | Full sequence, iterative unmasking | Block-by-block + KV cache | No recomputation of past blocks |
| Variable length | No (fixed 512 tokens) | Yes (EOS stops generation) | Natural document boundaries |
| Special tokens | `[MASK]` only | `[MASK]`, `<\|endoftext\|>`, `<\|padding\|>` | EOS for stopping, PAD for short docs |
| Tokenizer | BPE (32,768 vocab) | Same BPE (32,768 vocab) | Reused from Phase 2 |

Phase 2 had two problems that block diffusion solves. First, full-sequence diffusion
recomputes attention over all 512 tokens at every denoising step -- no way to reuse
past computation. Second, generation always produces a fixed-length sequence with no
mechanism to stop early. Block diffusion fixes both: past blocks are cached, and
EOS within a block ends generation.

---

## The Staircase Attention Mask

This is the core innovation from BD3-LMs. The training input is doubled:
`[x_noisy || x_clean]`, giving a sequence of length 2L. The staircase mask
controls exactly what each position can attend to.

### The Mask Pattern

For a sequence with 3 blocks (seq_len=12, block_size=4):

```
              x_noisy (noisy half)        x_clean (clean half)
             b0     b1     b2           b0     b1     b2
x_t   b0 [ ####   ....   ....        ....   ....   .... ]
      b1 [ ....   ####   ....        ####   ....   .... ]
      b2 [ ....   ....   ####        ####   ####   .... ]
x_0   b0 [ ....   ....   ....        ####   ....   .... ]
      b1 [ ....   ....   ....        ####   ####   .... ]
      b2 [ ....   ....   ....        ####   ####   #### ]

  #### = attend (0.0)       .... = blocked (-inf)
```

Three components compose the full mask:

| Component | Name | Rule |
|-----------|------|------|
| M_BD | Block-diagonal | Same block, same half -- full bidirectional within a block |
| M_OBC | Offset block-causal | Noisy queries attend to clean keys from STRICTLY EARLIER blocks |
| M_BC | Block-causal | Clean queries attend to clean keys from current + earlier blocks |

### What Each Block Sees

Reading the mask row by row:

- **Noisy b0**: sees only its own noisy tokens. No context from other blocks.
  This is the hardest block to denoise -- pure diffusion with no conditioning.

- **Noisy b1**: sees its own noisy tokens + clean b0. It knows what block 0
  says, so it can continue the text coherently.

- **Noisy b2**: sees its own noisy tokens + clean b0 and b1. Most context,
  easiest to denoise.

- **Clean blocks**: standard block-causal attention. Clean b1 sees clean b0 and
  b1. Clean b2 sees all clean blocks. This is the autoregressive half.

The critical detail: noisy block i sees clean blocks `0..i-1` but NOT clean block
i. If it could see its own clean tokens, it would copy the answers -- label leakage.
This is enforced by strict `>` in the mask construction (see Bugs Found below).

### Why the Doubled Input

The clean half serves as a conditioning signal. Without it, each noisy block would
only see its own masked tokens -- no way to condition on earlier text.

```
Without doubling:   [noisy_b0 | noisy_b1 | noisy_b2]
                     Block b2 sees b0 and b1, but they're NOISY -- bad context.

With doubling:      [noisy_b0 | noisy_b1 | noisy_b2 | clean_b0 | clean_b1 | clean_b2]
                     Block b2 sees clean_b0 and clean_b1 -- perfect context.
                     (Staircase mask prevents it from seeing clean_b2.)
```

### The Mask Code

```python
def build_staircase_mask(seq_len, block_size_blk):
    n = seq_len
    total = 2 * n

    pos = torch.arange(total)
    q = pos.unsqueeze(1)    # (2n, 1)
    kv = pos.unsqueeze(0)   # (1, 2n)

    x0_flag_q = (q >= n)                    # True if query is in clean half
    x0_flag_kv = (kv >= n)                  # True if key is in clean half
    block_q = (q % n) // block_size_blk     # block index of query
    block_kv = (kv % n) // block_size_blk   # block index of key

    # M_BD: same block, same half -> bidirectional
    m_bd = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # M_OBC: noisy attends to clean from STRICTLY EARLIER blocks
    m_obc = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q    # > not >=

    # M_BC: clean attends to clean from current or earlier blocks
    m_bc = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    allow = m_bd | m_obc | m_bc
    mask = torch.where(allow, 0.0, float('-inf'))
    return mask
```

The mask is computed once and cached for all batches. It is a float tensor where
`0.0` means "attend" and `-inf` means "blocked" -- passed directly to
`F.scaled_dot_product_attention(attn_mask=mask)`.

---

## Per-Block Noise Levels

Phase 2 sampled one timestep `t` per sequence. Phase 3 samples one `t` per block:

```
Phase 2 (per-sequence):
  t = 0.6 for entire sequence
  mask_prob = 1 - cos^2(0.6 * pi/2) = 0.69
  [MASK MASK the  MASK fox  MASK MASK MASK]
   0.69 probability applied uniformly

Phase 3 (per-block, block_size=4):
  Block 0: t=0.3, mask_prob=0.21
  Block 1: t=0.8, mask_prob=0.90
  Block 2: t=0.5, mask_prob=0.50
  [The  quick MASK fox | MASK MASK MASK MASK | MASK the  MASK dog ]
   low noise             high noise             medium noise
```

Each block trains at a different difficulty level within the same sequence. The model
learns to denoise easy blocks and hard blocks simultaneously, and learns to use clean
context from earlier blocks to help denoise later ones.

The cosine schedule and ELBO weighting from Phase 2 are preserved:

```
mask_prob = 1 - cos^2(t * pi/2)
ELBO weight = 1/t per token
```

The only difference: `t` is now (B, num_blocks) instead of (B,), expanded to
(B, L) by repeating each block's `t` across its tokens.

---

## Block-by-Block Generation with KV Cache

Generation proceeds one block at a time. Each completed block's keys and values
are cached, so the model never recomputes past blocks.

```
Prompt: "The quick brown fox"   block_size=4   denoise_steps=10

Step 0: Warm up KV cache with prompt blocks
+-----------+
| The quick |  -> forward pass, cache K0, V0
+-----------+
| brown fox |  -> forward pass (attends to K0, V0), cache K1, V1
+-----------+

Step 1: Generate block 2
KV cache: [K0, V0 | K1, V1]
Block input: [____] [____] [____] [____]   (all [MASK])

  Denoise iteration 1:
    Input:      [____] [____] [____] [____]   pos_offset=8
    Q from block 2, K/V = [K_cached | K_current]
    Confidence:  0.45   0.98   0.72   0.91
    Unmask:       no    YES     no     no
    Result:     [____] [jumped] [____] [____]

  Denoise iteration 2:
    Input:      [____] [jumped] [____] [____]
    Confidence:  0.97    --     0.88   0.96
    Unmask:     YES     --      no    YES
    Result:     [ ] [jumped] [____] [over]

  Denoise iteration 3:
    Input:      [ ] [jumped] [____] [over]
    Confidence:  --    --     0.99    --
    Result:     [ ] [jumped] [the] [over]   <- block done!

  Final pass with cache_mode=True -> save K2, V2

Step 2: Generate block 3
KV cache: [K0, V0 | K1, V1 | K2, V2]
Block input: [____] [____] [____] [____]
...continue until max_new_tokens or EOS...
```

### How Denoising Works Within a Block

Within each block, the denoising loop is the same confidence-based decoding from
Phase 2:

1. Forward pass through the model (current block queries + cached context)
2. Compute confidence (max softmax probability) at each masked position
3. Unmask positions above `confidence_threshold` (default 0.95)
4. If nothing qualifies, force-unmask the single most confident position
5. Repeat until all positions in the block are filled, or `denoise_steps` reached

After a block is fully denoised, one final forward pass with `cache_mode=True`
appends its K, V to the cache. The model then moves to the next block.

### KV Cache Mechanics

The KV cache is what makes block diffusion practical. Without it, generating block
N would require recomputing attention over all N-1 previous blocks at every
denoising step. With caching:

```
Block 0: compute from scratch (no cache)
Block 1: compute only block 1, attend to cached K0, V0
Block 2: compute only block 2, attend to cached K0, V0, K1, V1
Block N: compute only block N, attend to cached K0..K_{N-1}
```

This is the same caching strategy AR models use, applied at the block level.
The attention computation is: `Q_current @ [K_cached | K_current]^T` -- the
query is the current block, the key/value is the concatenation of all cached
past blocks plus the current block.

### Variable-Length Generation

Phase 2 always generates exactly 512 tokens. Phase 3 checks for `<|endoftext|>`
after each block -- if any generated token is EOS, the output is truncated there
and generation stops. This gives natural document boundaries without padding.

---

## Training Results

Trained on Kaggle P100 (16 GB) with depth=6 (384-dim, 6 heads, 6 layers).

| Metric | Value |
|--------|-------|
| Parameters | ~35.78M |
| Tokenizer | BPE (32,768 vocab) |
| Sequence length | 512 tokens |
| Block size | 4 tokens (128 blocks/sequence) |
| Batch size | 32 |
| GPU | Kaggle P100 (16 GB) |
| Steps | 20,000 |
| Throughput | ~15,575 tok/s |
| Final train loss | 7.880 |
| Final val loss | 7.742 |
| Training time | ~5 hours |

### Loss Curve

```
loss
  17 |*
     | **
     |   ***
  14 |      ***
     |         ****
     |             ***
  11 |                ***
     |                   ****
     |                       ****
   9 |                           ****
     |                               ****
   8 |                                   ********
     |                                           *********
 7.7 |                                                    ****
     +------|------|------|------|------|------|------|------|
     0    2500   5000   7500  10000  12500  15000  17500  20000
                               step
```

The loss drops steeply from ~17.2 to ~11 in the first 2500 steps (basic token
distributions learned), then gradually refines to 7.7. The slight gap between
train (7.88) and val (7.74) suggests no overfitting -- the streaming FineWeb-Edu
data is effectively infinite.

### Generation Samples

Step 500 (early) -- learned common words, produces repetitive loops:

```
The name of the world is the of the world the of the world...
```

Step 20,000 (final) -- grammatical but repetitive:

```
The meaning of life is the meaning of life...
```

The repetition at 35M parameters and 20K steps is expected. Phase 3 is about
the architecture (staircase mask, KV cache, block structure), not generation
quality. Phase 4 scales to 125M parameters with better optimizers and longer
training to address this.

---

## Bugs Found

Four bugs discovered during Phase 3 development. Each one silently degrades
training or breaks generation without obvious error messages.

### 1. Staircase mask: `>` vs `>=` (CRITICAL)

```python
# WRONG: >= lets noisy block i see its own clean tokens -> label leakage
m_obc = (block_q >= block_kv) & x0_flag_kv & ~x0_flag_q

# RIGHT: > ensures noisy block i only sees clean from EARLIER blocks
m_obc = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
```

With `>=`, the model cheats by reading the clean version of the tokens it should
predict. Loss drops fast during training but the model learns nothing useful --
it just copies from the clean half. Suspiciously low training loss + garbage
generation is the symptom.

### 2. Tied embedding init order

```python
# WRONG: tie first, then init overwrites the shared weight
self.lm_head.weight = self.token_emb.weight
self.apply(self._init_weights)     # re-inits lm_head.weight, breaking the tie

# RIGHT: init first, then tie
self.apply(self._init_weights)
self.lm_head.weight = self.token_emb.weight
```

### 3. RoPE buffer too short

```python
# WRONG: only enough for training sequences
self.rotary_seq_len = block_size_seq * 2    # 1024

# RIGHT: account for long generation with KV cache
self.rotary_seq_len = max(block_size_seq * 2, 4096)
```

During generation, each new block gets a RoPE offset equal to its position in
the full sequence. After 256 blocks of size 4, the offset hits 1024 -- past
the precomputed buffer. The fix: precompute for at least 4096 positions.

### 4. KV cache stale state

```python
# WRONG: only reset at entry
def generate(model, ...):
    model.reset_kv_cache()
    ...
    return output    # KV cache still holds batch=1 tensors

# Next training step crashes: cache has batch=1, training has batch=32

# RIGHT: reset on BOTH entry and exit
def generate(model, ...):
    model.reset_kv_cache()
    ...
    model.reset_kv_cache()    # clean up before returning
    return output
```

---

## Quick Start

```bash
# Install Phase 3 dependencies
pip install -e ".[phase3]"

# Train (depth=6, ~36M params, ~5 hours on P100)
cd 03_block_diffusion
python block_dllm.py --train --depth 6 --block-size 4

# Generate text
python block_dllm.py --depth 6 --block-size 4 --prompt "The meaning of life is"

# Adjust denoising steps per block (default: 10)
python block_dllm.py --depth 6 --block-size 4 --denoise-steps 20 --prompt "Once upon a time"

# Verify staircase mask (no weights needed, prints mask visualization)
python block_dllm.py --depth 6 --block-size 4
```

---

## What's Next

**Phase 4: Modern Block Diffusion** scales up to 125M parameters with
production-grade training infrastructure:

1. **FlexAttention** for the staircase mask (fused kernels, no explicit mask tensor)
2. **Muon optimizer** (momentum + Newton-Schulz orthogonalization)
3. **GQA 4:1** (grouped-query attention, 4x fewer KV heads)
4. **Liger kernels** (fused RMSNorm + SwiGLU) for throughput
5. **DDP multi-GPU** training on Modal (8x A100-80GB)
6. **125M parameters** (20L/768d/12h, up from 36M at 6L/384d/6h)

---

## References

- [BD3-LMs](https://arxiv.org/abs/2503.09573) (Arriola et al., ICLR 2025 Oral) -- Block diffusion theory, staircase mask, the paper this phase implements
- [Mercury](https://arxiv.org/abs/2506.17298) (Inception Labs, 2025) -- Production-scale block diffusion using the same staircase mask
- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024) -- Continuous-time ELBO, cosine schedule (carried over from Phase 2)
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) -- First 8B dLLM, proves masked diffusion scales
- [nanochat](https://github.com/karpathy/nanochat) (Karpathy) -- Clean training script style reference
