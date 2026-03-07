# Phase 5 Design: Modern Block Diffusion LM (~144M)

**Date:** 2026-03-07
**Status:** Approved
**Builds on:** Phase 4 (`04_modern_dllm/modern_dllm.py`)
**Target directory:** `05_phase5_dllm/`

---

## Motivation

Phase 4 validated block diffusion at 125M scale but exposed a critical ELBO weight bug (cosine schedule with `1/t` instead of `1/mask_prob`, $200 wasted compute). Phase 5 incorporates lessons from 7 reference implementations and recent papers to build a correct, modern block diffusion LM.

Key changes: deeper/narrower architecture (SmolLM2-validated), linear noise schedule (eliminates ELBO weight complexity), MuonClip optimizer (QK stability), document packing (no padding waste), Gated Query Attention (attention sink elimination).

---

## 1. Architecture

Match SmolLM2-135M's proven deeper/narrower layout, plus Gated Query Attention.

| Param | Phase 4 | Phase 5 | Rationale |
|---|---|---|---|
| Layers | 20 | **30** | SmolLM2 validated deeper is better at 135M |
| Hidden dim | 768 | **576** | Narrower compensates for more layers |
| Heads | 12 | **9** | 576/9 = 64 head_dim (standard) |
| KV heads | 3 | 3 | GQA 3:1 ratio preserved |
| FFN dim | 1536 | 1536 | Same |
| Vocab | 32,768 | **49,152** | SmolLM2 merges (divisible by 64) |
| Params | ~125M | **~144M** | +10M from Gated Query Attention gate |
| Tied embed | Yes | Yes | Same |
| Activation | SwiGLU | SwiGLU | Same |
| Norm | RMSNorm (eps=1e-5) | RMSNorm (eps=1e-5) | Same |
| RoPE theta | 10,000 | 10,000 | Same |
| seq_len | 2,048 | 2,048 | Same |
| block_size | 32 | 32 | Same |

### Param count breakdown

- Embedding: 49,152 x 576 = 28.3M (tied with lm_head)
- Per attention layer: Q(576x576) + K(576x192) + V(576x192) + O(576x576) + Gate(576x576) = 1.22M
- Per FFN layer: gate_proj(576x1536) + up_proj(576x1536) + down_proj(1536x576) = 2.65M
- Per layer total: 1.22M + 2.65M + 2 x RMSNorm(576) = 3.88M
- 30 layers: 116.4M
- Total: 28.3M + 116.4M = ~144M

### Gated Query Attention

**Source:** Qiu et al. "ADOPT" (arXiv:2505.06708), NeurIPS 2025 Best Paper.

After SDPA output Y and before output projection W_O, apply a sigmoid gate:

```python
# In attention forward, after sdpa_output (B, H, T, D):
gate = torch.sigmoid(x @ self.w_gate)  # x is pre-attention input, (B, T, D)
y = y * gate.unsqueeze(1)              # broadcast over heads
y = y @ self.w_o                       # output projection
```

**Reference implementation pattern:**
- `w_gate` shape: `(d_model, d_model)`, initialized to zeros (identity gate at init)
- Applied element-wise after reshaping gate to match head layout
- Eliminates attention sinks (46.7% -> 4.8%), reduces max activation (1053 -> 94)
- May be extra useful for diffusion where mask tokens create artificial attention sinks

**Ref file:** `refs/REFERENCE_GUIDE.md` (Section on future optimizations in memory)

---

## 2. Tokenizer

Hybrid: SmolLM2's trained BPE merges + Qwen3-style pre-tokenization + our diffusion special tokens.

### Build process

1. Download `HuggingFaceTB/cosmo2-tokenizer` from HuggingFace
2. Extract BPE merge rules + 256 byte-level base tokens
3. Discard SmolLM2's 17 special tokens (IDs 0-16)
4. Build new tokenizer:
   - **Normalizer:** NFC (from Qwen3/LLaDA, not present in SmolLM2)
   - **Pre-tokenizer:** Qwen3/GPT-4 regex split + ByteLevel
   - **Special tokens (IDs 0-13):** our 14 diffusion/chat tokens
   - **BPE merges (IDs 14+):** SmolLM2's trained merges
5. Pad to exactly **49,152** vocab size (divisible by 64 for GPU tensor core alignment)

### Special token mapping

SmolLM2 tokens with same purpose as ours — use our ID assignment, drop theirs:

| SmolLM2 token | SmolLM2 ID | Our token | Our ID | Action |
|---|---|---|---|---|
| `<\|endoftext\|>` | 0 | `<\|endoftext\|>` | 1 | Same purpose, keep our ID |
| `<\|im_start\|>` | 1 | `<\|im_start\|>` | 3 | Same purpose, keep our ID |
| `<\|im_end\|>` | 2 | `<\|im_end\|>` | 4 | Same purpose, keep our ID |
| `<repo_name>` etc. | 3-16 | — | — | Drop (code-specific) |

Our full special token layout (unchanged from Phase 4):

```
ID 0:  <|mask|>          — diffusion noise token
ID 1:  <|endoftext|>     — EOS / document boundary
ID 2:  <|padding|>       — right-padding (unused with doc packing, kept for compat)
ID 3:  <|im_start|>      — chat message start
ID 4:  <|im_end|>        — chat message end
ID 5:  <|system|>        — system role marker
ID 6:  <|user|>          — user role marker
ID 7:  <|assistant|>     — assistant role marker
ID 8:  <think>           — reasoning start
ID 9:  </think>          — reasoning end
ID 10: <tool_call>       — tool invocation start
ID 11: </tool_call>      — tool invocation end
ID 12: <tool_response>   — tool result start
ID 13: </tool_response>  — tool result end
```

### Pre-tokenizer regex

Same as Phase 4 (`train_tokenizer.py:30-38`), identical to LLaDA-8B and Qwen3:

```python
PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"   # English contractions
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"       # optional punctuation + word
    r"|\p{N}"                            # single digit (digit splitting)
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"       # punctuation runs
    r"|\s*[\r\n]+"                       # newlines with leading whitespace
    r"|\s+(?!\S)"                        # trailing whitespace
    r"|\s+"                              # other whitespace
)
```

**Ref files:**
- Our current tokenizer: `04_modern_dllm/train_tokenizer.py`
- SmolLM2 tokenizer: `https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer`
- LLaDA tokenizer (same regex): `https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/raw/main/tokenizer.json`
- SmolLM2 training config: `refs/smollm/text/pretraining/smollm2/config_smollm2_135M.yaml`

---

## 3. Noise Schedule & Loss

### Linear noise schedule

Switch from cosine to **linear**. This is the single biggest correctness simplification.

```python
# Linear schedule
t = eps + (1 - eps) * torch.rand(B)   # eps = 1e-3
mask_prob = t                           # linear: mask_prob = t
weight = 1.0 / t                        # ELBO: 1/mask_prob = 1/t (trivially correct)
```

Why linear:
- LLaDA 8B uses linear (arXiv:2502.09992, GUIDELINES.md:28)
- BD3-LMs uses loglinear where p=t (refs/bd3lms/noise_schedule.py:64-86)
- Dream 7B uses linear with 1/t weight (refs/Dream/src/trainer/fsdp_sft_trainer.py:147)
- Fast-dLLM uses linear (refs/Fast-dLLM/dream/eval.py:278)
- dllm reference defaults to linear (refs/dllm/dllm/core/schedulers/alpha.py:102)
- Cosine's ELBO weight `1/sin^2(t*pi/2)` caused our $200 bug — linear makes it `1/t`, trivial

### Loss computation

Unchanged from corrected Phase 4. Normalize by ALL real tokens, not just masked:

```python
token_nll = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
token_nll = token_nll * weight[:, None] * masked_mask.float()
loss = token_nll.sum() / real_token_count.clamp_min(1)
```

- `real_token_count = (targets != pad_token_id).sum()` — all non-padding tokens
- `weight = 1/t` — ELBO importance weight (per-sample, broadcast to tokens)
- `masked_mask` — binary mask of which tokens were actually masked

**Expected step-0 loss:** ln(49152) = **10.80**

**Ref files:**
- dllm loss: `refs/dllm/dllm/core/trainers/mdlm.py:187-237`
- dllm ELBO weight: `refs/dllm/dllm/core/schedulers/alpha.py:86-88`
- dllm masking: `refs/dllm/dllm/core/trainers/mdlm.py:149-155`
- LLaDA loss: `refs/LLaDA/GUIDELINES.md:50-51`
- BD3-LMs loss: `refs/bd3lms/diffusion.py:819-891`
- Cross-reference table: `refs/REFERENCE_GUIDE.md` (Loss Formula Comparison)

---

## 4. Optimizer: MuonClip + AdamW

Replace external `muon-optimizer` package with self-contained MuonClip (from ms-swift) + AdamW.

### Param groups

| Group | Which params | Optimizer | Settings |
|---|---|---|---|
| QK Muon | `q_proj.weight`, `k_proj.weight` | Muon + QK-Clip | lr=0.02, mom=0.95, wd=0.1, tau=100 |
| Other Muon | other 2D non-embed weights (v_proj, o_proj, gate_proj, up_proj, down_proj, w_gate) | Muon | lr=0.02, mom=0.95, wd=0.1 |
| AdamW | embeddings, lm_head, all 1D (norms) | AdamW | lr=6e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.01 |

### MuonClip implementation

Self-contained, ~150 lines. Key components:

**Newton-Schulz orthogonalization** (Moonlight polynomial, 5 iterations):
```python
a, b, c = (3.4445, -4.7750, 2.0315)
X = G.bfloat16() / (G.norm() + eps)
if G.size(0) > G.size(1): X = X.T; transposed = True
for _ in range(5):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

**RMS scaling** (post-orthogonalization):
```python
rms_scale = sqrt(max(n, m)) * 0.2
update = orthogonalized * rms_scale
```

**QK-Clip** (per-step, scalar):
```python
# Track max attention logit via SDPA monkey-patch:
#   max_logit_bound = max||q|| * max||k|| * scale
# At optimizer step:
if max_logit > tau:
    gamma_sqrt = sqrt(tau / max_logit)
    p.mul_(gamma_sqrt)        # scale weight
    update = update * gamma_sqrt  # scale update
```

**Ref files:**
- MuonClip full implementation: `refs/ms-swift/muonclip.py:183-313`
- MaxLogitsTracker (SDPA patch): `refs/ms-swift/muonclip.py:15-181`
- MuonClip param grouping: `refs/ms-swift/muonclip.py:316-457`
- NS coefficients source: Moonlight (Liu et al. 2025)
- Karpathy's Muon reference: `refs/modded-nanogpt/train_gpt.py` (NorMuonAndAdam class)
- nanochat Muon: `refs/nanochat/nanochat/gpt.py:356-394`

### Why replace external Muon

- KellerJordan's `MuonWithAuxAdam` requires DDP (`dist.all_gather` in step). `SingleDeviceMuonWithAuxAdam` exists but is a separate class.
- PyPI name conflict: `pip install muon` = bioinformatics package, NOT ML optimizer.
- MuonClip is self-contained, adds QK-Clip for free, works in both single-GPU and DDP.
- We already have the full source at `refs/ms-swift/muonclip.py`.

---

## 5. Document Packing

Replace right-padding with multi-document packing. Every position is a real token.

### Data pipeline

```python
# Pack documents into fixed-length sequences
def pack_documents(doc_iterator, seq_len, eos_id):
    buffer_ids = []
    buffer_doc_ids = []
    current_doc = 0
    for doc_tokens in doc_iterator:
        tokens = doc_tokens + [eos_id]  # append EOS
        buffer_ids.extend(tokens)
        buffer_doc_ids.extend([current_doc] * len(tokens))
        current_doc += 1
        while len(buffer_ids) >= seq_len:
            yield buffer_ids[:seq_len], buffer_doc_ids[:seq_len]
            buffer_ids = buffer_ids[seq_len:]
            buffer_doc_ids = buffer_doc_ids[seq_len:]
```

### Attention mask integration

AND existing staircase mask with document boundary constraint:

```python
def mask_mod(b, h, q_idx, kv_idx):
    staircase = staircase_mask(b, h, q_idx, kv_idx)  # existing M_BD | M_OBC | M_BC
    same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    return staircase & same_doc
```

### RoPE position reset

Position IDs reset at each document boundary:

```python
# positions[i] = i - start_of_current_doc
positions = torch.zeros_like(input_ids)
for b in range(B):
    pos = 0
    for i in range(seq_len):
        positions[b, i] = pos
        pos += 1
        if i + 1 < seq_len and doc_ids[b, i+1] != doc_ids[b, i]:
            pos = 0
```

### Complementary masking: REMOVED

LLaDA 2.0 (arXiv:2512.15745v2) found complementary masking does not help in pre-training at >100B tokens. The diversity from random masking across epochs is sufficient.

> "We found that complementary masking provides no benefit during pre-training when the dataset is large enough (>100B tokens)." — LLaDA 2.0

**Ref files:**
- LLaDA 2.0 packing: arXiv:2512.15745v2 (Section 3.2, document-level attention)
- LLaDA 2.0 complementary masking result: arXiv:2512.15745v2 (Section 4.1, ablation)
- Zhao et al. ACL 2024 packing benefits: +7-37% downstream, -8% perplexity
- Our staircase mask: `04_modern_dllm/modern_dllm.py` (staircase_mask_mod function)
- BD3-LMs staircase: `refs/bd3lms/models/dit.py:30-74`

---

## 6. Training Configuration

| Setting | Value | Source |
|---|---|---|
| seq_len | 2,048 | SmolLM2-135M |
| block_size | 32 | Phase 4 |
| batch_size per GPU | TBD by VRAM | — |
| effective batch tokens | ~1M/step | SmolLM2 (~1,048,576) |
| grad_clip | 1.0 | SmolLM2 |
| time_epsilon | 1e-3 | dllm reference |
| noise_schedule | linear | LLaDA, BD3-LMs, Dream, dllm |
| MLP dropout | 0.0 | SmolLM2 (no dropout) |
| AMP dtype | bfloat16 | SmolLM2 |
| torch.compile | mode="default" | Karpathy, torchtitan |
| Expected step-0 loss | ~10.8 (ln(49152)) | — |

### WSD LR schedule

```
|-- warmup --|-------- stable --------|-- decay --|
0          2,000                    80%         100%
             LR                  full LR      -> 0
```

- Warmup: 2,000 steps (linear ramp)
- Stable: constant LR until 80% of total steps
- Decay: linear to 0 over final 20% (SmolLM2 uses 20% decay)

**Ref files:**
- SmolLM2 WSD: `refs/smollm/text/pretraining/smollm2/config_smollm2_135M.yaml`
- nanochat WSD: `refs/nanochat/scripts/base_train.py:362-381`
- Phase 4 WSD: `04_modern_dllm/modern_dllm.py` (get_lr_factor function)

---

## 7. Kept from Phase 4

These components are validated and unchanged:

- **FlexAttention** with block staircase mask (M_BD | M_OBC | M_BC with strict `>` for M_OBC)
  - Ref: `refs/bd3lms/models/dit.py:30-74`, `04_modern_dllm/modern_dllm.py`
- **Block diffusion** timestep sampling (per-block t, broadcast to tokens)
  - Ref: `refs/bd3lms/diffusion.py:768-789`
- **AMP** (bfloat16 autocast + GradScaler)
- **Liger kernels** (LigerRMSNorm, LigerSwiGLUMLP, LigerFusedLinearCrossEntropyFunction)
  - Caveat: FLCE has 3 `.item()` graph breaks (Liger v0.7.0), minimal impact
- **DDP** with no_sync for gradient accumulation
- **Gradient checkpointing** (optional `--no-grad-ckpt`)
- **CART** (optional `--cart`, off by default, capped to 1/t_min)

---

## 8. NOT in Phase 5 (Future)

| Feature | Why deferred |
|---|---|
| FP8 matmuls | Needs H100. Add when hardware available. Ref: `refs/modded-nanogpt/train_gpt.py` |
| CAP training | Separate inference optimization stage. Ref: LLaDA 2.0 Section 3.3 |
| (1+w) zero-init RMSNorm | Small gain, add incrementally |
| Gated DeltaNet | Causal-only, incompatible with bidirectional diffusion |
| Top-k checkpoint merge | Post-training technique, not pre-training |

---

## 9. Pre-Training Verification Checklist

Run these checks before committing to a full training run:

- [ ] Step-0 loss ~ ln(49152) = 10.80 (+/- 0.5)
- [ ] ELBO weight = 1/t (linear schedule, trivially correct)
- [ ] Loss normalized by all real tokens (not just masked)
- [ ] Gradient clipping: first 10 steps are NOT all clipped
- [ ] Staircase mask: M_OBC uses strict `>` (not `>=`)
- [ ] Document packing: doc boundary attention mask works (tokens don't attend across docs)
- [ ] RoPE resets at document boundaries
- [ ] QK-Clip tau=100: verify MaxLogitsTracker produces values
- [ ] Tokenizer: `<|mask|>`=0, `<|endoftext|>`=1, `<|padding|>`=2
- [ ] Tokenizer: digit splitting works ("2024" -> 4+ tokens)
- [ ] Tied embeddings: lm_head.weight is token_emb.weight (same data_ptr)
- [ ] Optimizer param dedup: tied weight appears in exactly one param group

---

## 10. Reference File Index

Quick lookup for implementation agents:

### Architecture references
- SmolLM2-135M config: `refs/smollm/text/pretraining/smollm2/config_smollm2_135M.yaml`
- SmolLM2 paper: arXiv:2502.02737
- Gated Query Attention: arXiv:2505.06708

### Loss & schedule references
- dllm loss + ELBO weight + masking: `refs/dllm/dllm/core/trainers/mdlm.py:149-237`
- dllm noise schedules: `refs/dllm/dllm/core/schedulers/alpha.py:86-116`
- LLaDA loss pseudocode: `refs/LLaDA/GUIDELINES.md:28-51`
- BD3-LMs loss + schedule: `refs/bd3lms/diffusion.py:819-891`, `refs/bd3lms/noise_schedule.py:64-86`
- Dream loss + time weights: `refs/Dream/src/trainer/fsdp_sft_trainer.py:722-835`
- Cross-reference table: `refs/REFERENCE_GUIDE.md` (Loss Formula Comparison)

### Optimizer references
- MuonClip full source: `refs/ms-swift/muonclip.py`
  - MuonClip optimizer: lines 183-313
  - MaxLogitsTracker: lines 15-181
  - Param grouping callback: lines 316-457
- Karpathy NorMuon: `refs/modded-nanogpt/train_gpt.py` (search NorMuonAndAdam)
- nanochat Muon+AdamW: `refs/nanochat/nanochat/gpt.py:356-394`

### Attention & masking references
- BD3-LMs staircase mask: `refs/bd3lms/models/dit.py:30-74`
- BD3-LMs block timestep sampling: `refs/bd3lms/diffusion.py:768-789`
- Phase 4 staircase: `04_modern_dllm/modern_dllm.py` (staircase_mask_mod)

### Tokenizer references
- Our current tokenizer trainer: `04_modern_dllm/train_tokenizer.py`
- SmolLM2 tokenizer: `https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer`
- LLaDA tokenizer (regex match): `https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/raw/main/tokenizer.json`

### Training loop references
- Karpathy training loop: `refs/modded-nanogpt/train_gpt.py:1986-2030`
- nanochat WSD schedule: `refs/nanochat/scripts/base_train.py:362-381`
- LLaDA 2.0 document packing: arXiv:2512.15745v2 (Section 3.2)
- LLaDA 2.0 complementary masking ablation: arXiv:2512.15745v2 (Section 4.1)

### Phase 4 (base to build on)
- Full training script: `04_modern_dllm/modern_dllm.py` (1720 lines)
- Lessons learned: `.claude/tasks/lessons.md`
