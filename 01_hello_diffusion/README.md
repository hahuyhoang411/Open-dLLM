# Phase 1: Hello Diffusion

**The simplest possible diffusion language model, in ~400 lines of annotated Python.**

---

## What You'll Learn

- What diffusion means for text (not images -- discrete tokens, not continuous pixels)
- The 5 exact code changes that turn a GPT into a diffusion language model
- Masked denoising training: randomly mask tokens, predict the originals
- Confidence-based parallel decoding: generate multiple tokens per step

---

## The Big Idea

A GPT generates text left-to-right, one token at a time. A diffusion language model
(dLLM) starts with a fully masked sequence and iteratively fills in tokens -- in parallel.

```
GPT (autoregressive):
  Step 1:  The ________________________________________
  Step 2:  The quick ___________________________________
  Step 3:  The quick brown ______________________________
  Step 4:  The quick brown fox __________________________
  ...20 more steps, one token each...

dLLM (diffusion):
  Step 1:  ___ _____ _____ ___ _____ ____ ___ ____ ___
  Step 2:  The _____ _____ fox _____ over ___ ____ dog
  Step 3:  The quick _____ fox jumps over the ____ dog
  Step 4:  The quick brown fox jumps over the lazy dog
  ...done in ~4 steps instead of ~24!
```

The dLLM sees ALL positions at once (bidirectional attention) and fills in the
ones it is most confident about first.

---

## How Autoregressive Models Work (Review)

A GPT predicts the next token given all previous tokens:

```
P(x_1, x_2, ..., x_n) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ...
```

At each step, the model sees only the past (causal attention mask):

```
Input:    The  quick  brown  fox
           |     |      |     |
           v     v      v     v
Attends:  [The] [The ] [The ] [The  ]
                [quick] [quick] [quick]
                        [brown] [brown]
                                [fox  ]
           |     |      |     |
           v     v      v     v
Predicts: quick brown   fox   jumps
```

This is sequential by design. Token N cannot be generated until tokens 1..N-1 exist.

---

## How Diffusion Models Work for Text

### Forward Process: Adding Noise (Masking)

Take clean text and randomly replace tokens with `[MASK]`. The masking ratio
controls how much noise is applied. More masks = more noise.

```
t=0.0 (clean):   The quick brown fox jumps over the lazy dog
t=0.3 (light):   The _____ brown fox jumps ____ the lazy dog
t=0.6 (medium):  ___ _____ brown ___ _____ ____ the ____ ___
t=0.9 (heavy):   ___ _____ _____ ___ _____ ____ ___ ____ ___
t=1.0 (noise):   ___ _____ _____ ___ _____ ____ ___ ____ ___
```

Each token is independently masked with probability `t`.

### Reverse Process: Removing Noise (Denoising)

The model predicts original tokens from the masked input. Iterate to denoise:

```
Step 0:  ___ _____ _____ ___ _____ ____ ___ ____ ___
              model predicts all positions...
              unmask the most confident ones:
Step 1:  ___ _____ _____ ___ jumps ____ the ____ ___
Step 2:  The _____ _____ fox jumps over the ____ dog
Step 3:  The quick _____ fox jumps over the lazy dog
Step 4:  The quick brown fox jumps over the lazy dog
```

At each step, multiple tokens can be revealed simultaneously -- this is the
source of the speed advantage over autoregressive models.

### Full Denoising Sequence

```
  ALL MASKED         Step 1           Step 2           Step 3         CLEAN
+------------+  +------------+  +------------+  +------------+  +------------+
| _ _ _ _ _  |  | _ _ _ _ _  |  | _ w _ _ _  |  | t w _ s _  |  | t w a s a  |
| _ _ _ _ _  |  | _ _ _ t _  |  | d _ r _ t  |  | d a r k t  |  | d a r k a  |
| _ _ _ _ _  |  | _ _ o _ _  |  | _ t o _ _ |  | s t o r m  |  | s t o r m  |
| _ _ _ _ _  |  | _ i _ _ _  |  | _ i g h _  |  | n i g h t  |  | n i g h t  |
+------------+  +------------+  +------------+  +------------+  +------------+
  t=1.0            t=0.75           t=0.50           t=0.25          t=0.0
  (all noise)                                                     (all clean)
```

---

## The 5 Changes from GPT to dLLM

A dLLM is a GPT with exactly 5 surgical modifications. Everything else
(transformer architecture, RMSNorm, RoPE, MLP) stays the same.

### Change 1: Add Mask Token to Vocabulary

```python
# GPT: vocabulary = dataset characters only
chars = sorted(set(text))                      # 65 chars
vocab_size = len(chars)                        # 65

# dLLM: add a mask token at position 0                          [DIFF 1]
MASK_TOKEN = "_"
chars = [MASK_TOKEN] + sorted(set(text) - {MASK_TOKEN})
MASK_ID = 0
vocab_size = len(chars)                        # 65 (mask replaces underscore)
```

The mask token is just another token in the vocabulary. Nothing special
architecturally -- the model treats it like any other input.

### Change 2: Bidirectional Attention

```python
# GPT: causal attention (can only look at past tokens)
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# dLLM: bidirectional attention (can look everywhere)           [DIFF 2]
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

This is the single most important change. A GPT uses a causal mask so token N
can only attend to tokens 1..N. A dLLM needs to see the ENTIRE sequence to
predict what goes in the masked positions.

```
Causal (GPT):              Bidirectional (dLLM):
+--------+                 +--------+
|X . . . |                 |X X X X |
|X X . . |                 |X X X X |
|X X X . |                 |X X X X |
|X X X X |                 |X X X X |
+--------+                 +--------+
  "Can only see past"        "Can see everything"
```

### Change 3: Training -- Randomly Mask Input Tokens

```python
# GPT: feed clean text, predict next token
# Input:  [The] [quick] [brown] [fox]
# Target: [quick] [brown] [fox] [jumps]

# dLLM: randomly mask tokens, predict originals                [DIFF 3]
t = torch.rand(batch_size)                     # random ratio per sample
mask = torch.rand(seq_len) < t                 # which positions to mask
masked_input = input.clone()
masked_input[mask] = MASK_ID                   # replace with mask token
# Input:  [___] [quick] [___] [fox]
# Target: [The] [quick] [brown] [fox]
```

The masking ratio `t` is sampled uniformly from `[0, 1]` for each sample.
This means the model trains on all difficulty levels -- from nearly clean
(t close to 0) to nearly all masked (t close to 1).

### Change 4: Loss -- Only on Masked Positions

```python
# GPT: cross-entropy on ALL positions (next-token prediction)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# dLLM: cross-entropy on MASKED positions only                 [DIFF 4]
loss = F.cross_entropy(
    logits[mask],          # predictions at masked positions
    targets[mask]          # true tokens at masked positions
)
```

We only penalize the model for its predictions at positions that were masked.
The model's predictions at already-visible positions are irrelevant.

### Change 5: Generation -- Confidence-Based Parallel Decoding

```python
# GPT: generate one token at a time (sequential)
for i in range(max_tokens):
    logits = model(tokens[:i])
    next_token = sample(logits[-1])
    tokens[i] = next_token

# dLLM: generate multiple tokens per step (parallel)           [DIFF 5]
tokens = [MASK_ID] * max_tokens                # start fully masked
while any(t == MASK_ID for t in tokens):
    logits = model(tokens)                     # predict all positions
    probs = softmax(logits)
    confidence = probs.max(dim=-1)             # how sure at each position
    # unmask positions where confidence > threshold
    for pos in masked_positions:
        if confidence[pos] > threshold:
            tokens[pos] = sample(probs[pos])   # fill in this token
```

Instead of generating tokens one-by-one, the model predicts ALL masked
positions simultaneously, then fills in the ones it is most confident about.
This naturally produces multiple tokens per step.

---

## Training

### Overview

Training is simple: take a text chunk, randomly mask some tokens, and train the
model to predict the original tokens at the masked positions.

```
Training Data Flow
==================

  Raw text: "First Citizen:\nBefore we proceed..."
       |
       v
  +------------------+
  | Sample chunk      |    Random chunk of block_size characters
  | (256 chars)       |    from training data
  +------------------+
       |
       v
  +------------------+
  | Encode to IDs     |    Character-level: each char -> integer
  | [23, 8, 41, ...]  |
  +------------------+
       |
       v
  +------------------+
  | Sample t ~ U[0,1] |    e.g. t = 0.4 (mask 40% of tokens)
  +------------------+
       |
       v
  +------------------+
  | Apply mask         |    Randomly replace 40% of tokens with MASK_ID
  | [23, 0, 41, 0, .] |    (0 = mask token)
  +------------------+
       |
       +-- masked_input: [23,  0, 41,  0, ...]  (model input)
       +-- targets:      [23,  8, 41, 15, ...]  (original tokens)
       +-- mask:         [ 0,  1,  0,  1, ...]  (where to compute loss)
       |
       v
  +------------------+
  | Model forward      |    Bidirectional transformer predicts all positions
  | logits = model(    |
  |   masked_input)    |
  +------------------+
       |
       v
  +------------------+
  | Loss (CE)          |    Cross-entropy ONLY at masked positions
  | loss = CE(         |
  |   logits[mask],    |    [DIFF] GPT computes loss at ALL positions
  |   targets[mask])   |
  +------------------+
       |
       v
  loss.backward()
  optimizer.step()
```

### Masking Ratio

Each training sample gets its own masking ratio `t ~ U[0, 1]`. This is critical:

- `t` near 0: very few masks, easy predictions (model learns basic patterns)
- `t` near 0.5: half masked, rich learning signal (most information to learn)
- `t` near 1: nearly all masked, hard predictions (model learns global structure)

By sampling `t` uniformly, the model learns to denoise at all difficulty levels,
which is exactly what it needs to do during iterative generation.

---

## Inference: Confidence-Based Parallel Decoding

Generation works by starting with a fully masked sequence and iteratively
unmasking tokens based on model confidence.

### Step-by-Step Walkthrough

```
Generate 12 tokens, confidence threshold = 0.95

Step 0: Input to model
  Tokens: [_] [_] [_] [_] [_] [_] [_] [_] [_] [_] [_] [_]

  Model predicts probabilities at every position:
  Pos 0:  T=0.42  h=0.38  e=0.05  ...  (not confident)
  Pos 1:  h=0.97  e=0.01  a=0.01  ...  confidence=0.97 > 0.95 --> UNMASK
  Pos 2:  e=0.98  a=0.01  i=0.00  ...  confidence=0.98 > 0.95 --> UNMASK
  Pos 3:  space=0.55 a=0.30 ...         (not confident)
  Pos 7:  space=0.96 ...                confidence=0.96 > 0.95 --> UNMASK
  ...

Step 1: After unmasking confident positions
  Tokens: [_] [h] [e] [_] [_] [_] [_] [ ] [_] [_] [_] [_]
  Remaining masks: 9

  Model re-predicts with new context...
  Now seeing "h", "e", and space, more positions become confident:
  Pos 0:  T=0.96  ...  --> UNMASK
  Pos 3:  space=0.97   --> UNMASK
  Pos 4:  q=0.40       (not confident)
  ...

Step 2:
  Tokens: [T] [h] [e] [ ] [_] [_] [_] [ ] [_] [_] [_] [_]
  Remaining masks: 7

  ...continue until all masks are filled...

Step 5:
  Tokens: [T] [h] [e] [ ] [q] [u] [i] [ ] [b] [r] [o] [w]
  Remaining masks: 0 --> DONE

Total: 5 steps to generate 12 tokens (2.4 tokens/step average)
GPT would need 12 steps (1 token/step).
```

### The Confidence Threshold

- High threshold (0.95): conservative, more iterations but higher quality.
  Only fills in positions where the model is very sure.
- Low threshold (0.5): aggressive, fewer iterations but lower quality.
  Fills in uncertain positions that may be wrong (and cannot be revised).

If NO position exceeds the threshold at a given step, we unmask the single
most confident position to guarantee progress.

### Block-by-Block Generation

For sequences longer than the model's context window (`block_size`), we
generate in blocks:

```
block_size = 256, prompt = 16 chars

Block 1: [prompt: 16 chars] [generate: 240 chars via iterative unmasking]
Block 2: [context: last 16 chars of block 1] [generate: 240 chars]
Block 3: [context: last 16 chars of block 2] [generate: 240 chars]
...
```

Each block uses the end of the previous block as context (prompt), then
fills in the remaining positions through iterative confidence-based unmasking.

---

## Quick Start

```bash
# 1. Download Tiny Shakespeare (~1.1MB)
python 01_hello_diffusion/download_data.py

# 2. Train the model (~20 min on GPU, ~2 hours on CPU)
cd 01_hello_diffusion && python hello_diffusion.py --train

# 3. Generate text (loads saved weights)
python hello_diffusion.py
```

---

## What's Next

**Phase 2: Nano dLLM** builds on everything here with four upgrades:

1. **BPE tokenizer** (~32K vocabulary) instead of character-level
2. **Cosine noise schedule** (`alpha_t = cos^2(t * pi/2)`) instead of uniform
3. **FineWeb-Edu** (real web text) instead of Tiny Shakespeare
4. **Proper ELBO weighting** (`(1/t) * CE`) for mathematically correct training

These changes turn the toy model into a real language model that generates
coherent English paragraphs.

---

## References

- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) -- The educational reference this phase is modeled after
- [nanochat](https://github.com/karpathy/nanochat) -- Karpathy's clean training script style we follow
- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024) -- The foundational theory paper for masked diffusion
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) -- First 8B dLLM, proves the approach scales
- [SEDD](https://arxiv.org/abs/2310.16834) (Lou et al., ICML 2024 Best Paper) -- Score entropy for discrete diffusion
- [BD3-LMs](https://arxiv.org/abs/2503.09573) (Arriola et al., ICLR 2025 Oral) -- Block diffusion theory
