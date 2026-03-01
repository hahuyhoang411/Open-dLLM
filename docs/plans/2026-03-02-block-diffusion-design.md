# Phase 3: Block Diffusion — Design Document

**Date**: 2026-03-02
**Status**: Approved

## Objective

Build a block diffusion language model that bridges autoregressive and diffusion paradigms, with configurable block size to explore the full AR↔diffusion spectrum.

## Scope

**In scope:**
- Block-causal attention (staircase mask) — explicit attention mask approach
- KV caching across blocks for inference
- Variable-length generation (EOS-driven block termination)
- Configurable `--block-size {1,4,8,16,full}` CLI flag
- Qwen 3.5 tokenizer (production tokenizer, ~150K vocab)
- Tied embeddings (input + output share weights)
- Single file: `03_block_diffusion/block_dllm.py`
- FineWeb-Edu streaming data (same as Phase 2)
- Kaggle P100 training

**Out of scope:**
- T2T editing (separate concept, can be added to any phase later)
- FlexAttention (reserved for Phase 4)
- Custom tokenizer training

## Architecture

### The 1 Change from Phase 2

Replace `is_causal=False` (fully bidirectional) with an explicit **staircase attention mask**:
- Bidirectional within each block
- Causal across blocks

Everything else stays identical to Phase 2: RMSNorm, SwiGLU, RoPE, cosine noise schedule, ELBO-weighted loss, confidence-based decoding.

### Model Size

- ~200M param budget (with tied embeddings)
- Qwen 3.5 tokenizer (~150K vocab) → ~75M params in tied embedding table
- Remaining ~125M in transformer layers
- Depth-based scaling dial (like Phase 2): `--depth N` controls n_layer, n_embd, n_head

### Training

1. **Input**: For sequence of length L, concatenate `[x_t || x_0]` (noisy + clean) → effective length 2L. Both halves share position IDs (RoPE).
2. **Staircase mask**: `(2L, 2L)` boolean mask with:
   - Block-diagonal: same-block + same-half tokens see each other
   - Offset block-causal: noisy tokens see clean tokens from earlier blocks
   - Block-causal: clean tokens see clean tokens causally
3. **Loss**: CE on masked positions in noisy half `[:, :L]` only, weighted by `1/t`
4. **Data**: FineWeb-Edu streaming, seq_len=512, effective 1024 with concatenation
5. **Compute**: Kaggle P100 (16GB VRAM)

### Inference

Block-by-block generation with KV caching:
1. Encode prompt → fill initial block(s)
2. For each new block: run T denoising steps with confidence-based unmasking
   - Bidirectional attention within current block
   - Attend to KV cache from all previous blocks (causal)
3. Cache finalized block's KV
4. Stop on EOS or max length

### CLI

```
python block_dllm.py --train --depth 10 --block-size 4
python block_dllm.py --depth 10 --block-size 4 --prompt "The meaning of life is"
```

Flags:
- `--block-size {1,4,8,16,full}` — AR↔diffusion spectrum
- `--depth N` — model size
- `--train` — training mode
- `--prompt "text"` — prompted generation
- `--max-tokens N` — max output length
- `--denoise-steps T` — denoising steps per block (default ~10)

## Key References

- **BD3-LMs** (arxiv.org/abs/2503.09573) — ICLR 2025 Oral. Block diffusion formalization.
- **Mercury** (arxiv.org/abs/2506.17298) — Inception Labs. Production block diffusion at 1000+ TPS.
- Reference implementation: github.com/kuleshov-group/bd3lms
