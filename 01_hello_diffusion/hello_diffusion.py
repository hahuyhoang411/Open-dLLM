"""
hello_diffusion.py — A ~400-line diffusion language model from scratch.

This is a character-level masked diffusion model trained on Tiny Shakespeare.
It is a GPT with exactly 5 surgical modifications, each marked with [DIFF].

Architecture Overview
=====================

    Input: "Th_s is _ te_t"          (partially masked character sequence)
           |
           v
    +-------------------+
    | Token Embedding    |            vocab_size=66 (65 chars + 1 mask token)
    | (66 -> 384)        |            n_embd=384
    +-------------------+
           |
           v
    +-------------------+
    | RMSNorm            |            Functional, no learnable params
    +-------------------+
           |
           v
    +-------------------+
    | Transformer Block  |  x6        n_layer=6
    | +----- Attention -+|            n_head=6, head_dim=64
    | | Bidirectional!   ||            [DIFF] is_causal=False
    | | + RoPE           ||            Rotary positional embeddings
    | | + QK Norm        ||
    | +-----------------+|
    | +----- MLP -------+|
    | | Linear(384,1536) ||
    | | ReLU^2           ||            F.relu(x).square()
    | | Linear(1536,384) ||
    | +-----------------+|
    +-------------------+
           |
           v
    +-------------------+
    | RMSNorm            |
    +-------------------+
           |
           v
    +-------------------+
    | LM Head            |            Linear(384 -> 66)
    | (384 -> vocab)     |
    +-------------------+
           |
           v
    Output: logits (B, T, 66)         Predict original token at EVERY position

    Total: ~10.7M parameters

References:
    - tiny-diffusion: github.com/nathan-barry/tiny-diffusion
    - nanochat: github.com/karpathy/nanochat
    - MDLM (Sahoo et al., NeurIPS 2024): arxiv.org/abs/2406.07524
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Hyperparameters
# ============================================================================
# These are module-level constants, not wrapped in a config object.
# Matches the nanoGPT / tiny-diffusion convention for simplicity.

batch_size = 64      # independent sequences processed in parallel
block_size = 256     # maximum context length (characters)
max_iters = 10000    # total training iterations
eval_interval = 500  # evaluate every N steps
learning_rate = 3e-4 # AdamW learning rate
eval_iters = 200     # batches to average for evaluation loss

n_embd = 384         # embedding dimension
n_head = 6           # number of attention heads
n_layer = 6          # number of transformer blocks
head_dim = n_embd // n_head  # 64 per head

# Device selection: CUDA > MPS > CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)

# ============================================================================
# Data Loading
# ============================================================================

with open(os.path.join(os.path.dirname(__file__) or ".", "data.txt"), "r", encoding="utf-8") as f:
    text = f.read()

# ============================================================================
# Tokenizer: Character-Level with Mask Token
# ============================================================================
#
# [DIFF 1] Add mask token to vocabulary
#
# A GPT tokenizer just maps characters to integers. A diffusion model needs
# one extra token: the MASK token, used to represent "unknown" positions.
#
# Character Mapping
# =================
#
#   Index:  0    1    2    3    4    ...   65
#   Char:   _   \n   ' '   !    &   ...    z
#           ^
#           |
#      MASK TOKEN (always index 0)
#
# The underscore `_` serves double duty: it IS the mask token. We prepend it
# to the sorted character set so it always gets index 0. Since `_` doesn't
# appear naturally in Shakespeare, this works cleanly.

chars = sorted(list(set(text)))
chars = ["_"] + chars  # [DIFF 1]: Prepend mask token at index 0
vocab_size = len(chars)
mask_token_id = 0  # [DIFF 1]: The mask token is always at index 0

# Character <-> integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """String -> list of integers."""
    return [stoi[ch] for ch in s]


def decode(l):
    """List of integers -> string."""
    return "".join([itos[n] for n in l])


# Train/val split (90/10)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ============================================================================
# Batch Construction
# ============================================================================
#
# [DIFF 3] Training uses random masking instead of next-token prediction
#
# Training Data Flow
# ==================
#
#   Raw text:  "First Citizen:\nBefore we proceed..."
#        |
#        v
#   Sample random chunk of block_size=256 characters
#        |
#        v
#   Encode to token IDs:  [23, 8, 41, 52, ...]
#        |
#        +---> y (targets) = original IDs    [23,  8, 41, 52, ...]
#        |
#        v
#   Sample mask_prob ~ U[0,1] per batch element (e.g., t=0.4)
#        |
#        v
#   Create boolean mask:  rand < t           [ 0,  1,  0,  1, ...]
#        |
#        v
#   Replace masked positions with mask_token  [23,  0, 41,  0, ...]
#        |
#        v
#   Return: x (masked input), y (targets), mask (boolean)
#
# Key insight: each batch element gets its OWN masking ratio t ~ U[0,1].
# This trains the model at all noise levels simultaneously.

def get_batch(split):
    """Sample a batch of masked sequences for diffusion training."""
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = x.clone()  # Targets are the original (clean) tokens

    # [DIFF 3]: Random masking — each sample gets a different mask ratio
    # mask_probs shape: (batch_size, 1) — broadcasts across sequence length
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id  # Replace masked positions with [MASK]

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


# ============================================================================
# RMSNorm (Functional)
# ============================================================================
# No learnable parameters — purely functional normalization.
# This is simpler than LayerNorm and works just as well for transformers.

def norm(x):
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================
# RoPE encodes position by rotating pairs of dimensions in the query/key
# vectors. Unlike learned position embeddings, RoPE naturally handles
# relative positions and generalizes to unseen sequence lengths.

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to queries or keys."""
    assert x.ndim == 4  # (B, T, H, D) from multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


# ============================================================================
# Multi-Head Attention
# ============================================================================
#
# [DIFF 2] Bidirectional attention (is_causal=False)
#
# Causal vs Bidirectional Attention Masks
# ========================================
#
#   Causal (GPT):              Bidirectional (dLLM):
#   +--------+                 +--------+
#   |X . . . |                 |X X X X |
#   |X X . . |                 |X X X X |
#   |X X X . |                 |X X X X |
#   |X X X X |                 |X X X X |
#   +--------+                 +--------+
#     "See past only"            "See everything"
#
# Why bidirectional? A dLLM must see ALL tokens (including other mask tokens)
# to figure out what goes in each masked position. If token 5 is masked, the
# model needs context from BOTH sides to predict it — unlike GPT which only
# needs the left context.

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to queries, keys, values then reshape for multi-head
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply RoPE for relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # QK-norm: stabilizes training at scale
        q, k = norm(q), norm(k)

        # Transpose to (B, H, T, D) for attention computation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # [DIFF 2]: Bidirectional attention — every token sees every other token
        # GPT would use is_causal=True here
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# ============================================================================
# MLP (Feed-Forward Network)
# ============================================================================
# Standard expand-contract MLP with ReLU^2 activation.
# ReLU^2 (squared ReLU) gives sparser activations than GELU, which can
# improve efficiency. Used in PaLM and other modern architectures.

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.c_proj(x)
        return x


# ============================================================================
# Transformer Block
# ============================================================================
# Pre-norm architecture: normalize BEFORE attention and MLP, not after.
# Residual connections around both sub-layers.

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)  # Attention with pre-norm
        x = x + self.mlp(norm(x))            # MLP with pre-norm
        return x


# ============================================================================
# Model
# ============================================================================
#
# The full diffusion language model. Architecturally, this is a standard
# transformer (like GPT) with two key differences:
#   1. Bidirectional attention [DIFF 2]
#   2. Loss computed only on masked positions [DIFF 4]
#
# Everything else — embeddings, RoPE, RMSNorm, MLP, residuals — is identical.

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings: each of the 66 tokens gets a 384-dim vector
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Precompute rotary embeddings for up to 2x block_size positions
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output projection: predict which of the 66 tokens belongs at each position
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Normal(0, 0.02) initialization for Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        """Precompute cos/sin tables for RoPE."""
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Add batch and head dims: (1, T, 1, D//2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, mask=None):
        """
        Forward pass.

        Args:
            idx: input token IDs, shape (B, T)
            targets: ground-truth token IDs, shape (B, T). None during generation.
            mask: boolean mask, shape (B, T). True where tokens were masked.
                  None for standard CE loss (backward compatibility).

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.size()

        # Embed tokens and normalize
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Slice rotary embeddings to current sequence length
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # [DIFF 4]: Loss only on masked positions
            #
            # Loss Computation
            # ================
            #
            #   Position:   0     1     2     3     4     5
            #   Input:     [The] [_]  [brown] [_]  [jumps] [_]
            #   Target:    [The] [quick] [brown] [fox] [jumps] [over]
            #   Mask:        0     1      0      1      0      1
            #
            #   CE loss:    --   2.31    --    3.45    --    1.87
            #   Weighted:    0   2.31     0    3.45     0    1.87
            #                         sum = 7.63
            #                         count = 3
            #                         loss = 7.63 / 3 = 2.54
            #
            # GPT computes loss at ALL positions. The dLLM only computes loss
            # where tokens were masked — the model shouldn't be penalized for
            # positions where it already saw the answer.
            if mask is not None:
                mask_flat = mask.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                # Fallback: standard CE on all positions (for compatibility)
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ============================================================================
# Generation: Confidence-Based Parallel Decoding
# ============================================================================
#
# [DIFF 5] Parallel decoding instead of sequential next-token prediction
#
# Generation Process (Iterative Unmasking)
# =========================================
#
#   Start:  [prompt tokens] [_] [_] [_] [_] [_] [_] [_] [_]
#
#   Step 1: Forward pass -> get logits at all positions
#           Softmax -> probabilities
#           Top-k probs -> confidence = sum(top_k_probs)
#
#           Position:  prompt  p+1  p+2  p+3  p+4  p+5  p+6  p+7  p+8
#           Confidence:  --   0.92 0.97 0.45 0.98 0.88 0.71 0.96 0.55
#           Threshold:                        0.95
#           Decode?:     --    no   YES   no   YES  no   no   YES  no
#
#           After:  [prompt tokens] [_] [e] [_] [,] [_] [_] [t] [_]
#                                        ^       ^              ^
#                                   3 tokens decoded in one step!
#
#   Step 2: Re-run model with updated context...
#           Now more positions become confident (they see neighbors)
#
#           After:  [prompt tokens] [h] [e] [_] [,] [_] [_] [t] [h]
#
#   Step 3: ...continue until all masks are filled...
#
#   Result: [prompt tokens] [h] [e] [r] [,] [ ] [t] [t] [h] [e]
#
# If NO position exceeds the threshold, we force-decode the SINGLE most
# confident position to guarantee progress.

@torch.no_grad()
def generate(model, max_new_tokens, prompt_len=16, temp=1.0,
             confidence_threshold=0.95, top_k=3):
    """
    Generate text using confidence-based parallel decoding.

    Unlike GPT (one token per step), this generates MULTIPLE tokens per step
    by unmasking all positions where the model is confident enough.

    Args:
        model: the diffusion language model
        max_new_tokens: how many new tokens to generate
        prompt_len: number of tokens from data to use as initial context
        temp: sampling temperature (lower = more deterministic)
        confidence_threshold: minimum confidence to unmask a position
        top_k: number of top candidates to consider for sampling
    """
    # Start with prompt tokens from the dataset
    all_tokens = data[:prompt_len].tolist()
    total_steps = 0

    # Generate one block at a time (for sequences longer than block_size)
    while len(all_tokens) - prompt_len < max_new_tokens:
        # How many tokens to generate in this block
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))

        # Initialize: prompt + all masks within block_size window
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        # Track which positions still need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        # Iterative unmasking loop
        while masked.any():
            total_steps += 1

            # 1. Forward pass to get predictions
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)

            # 2. Get top-k candidates and compute confidence
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)  # Confidence = P(top_k)

            # 3. Decide which masked positions to decode
            decode_mask = (confidences >= confidence_threshold) & masked

            # 4. If nothing exceeds threshold, force-decode the best one
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"))
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            # 5. Sample from normalized top-k distribution
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(
                top_k_probs_norm.view(-1, top_k), 1
            ).view(1, block_size)
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            # 6. Update: place sampled tokens, clear mask at decoded positions
            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        # Extract generated tokens from this block
        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    tokens_generated = len(all_tokens) - prompt_len
    print(f"Total steps: {total_steps} for {tokens_generated} tokens")
    print(f"Avg decoded per step: {tokens_generated / total_steps:.2f}")
    return decode(all_tokens)


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def estimate_loss(model):
    """Estimate train and val loss by averaging over eval_iters batches."""
    out = {}
    model.training = False
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.training = True
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    train_flag = "--train" in sys.argv
    weights_path = "weights/diffusion.pt"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    model = Model()
    m = model.to(device)

    # Print parameter count
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Load existing weights or train from scratch
    if os.path.exists(weights_path) and not train_flag:
        print(f"Loading weights from {weights_path}")
        m.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Training from scratch")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(max_iters):
            # Evaluate periodically
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(m)
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}, "
                    f"time {time.time() - start:.2f}s"
                )
                # Generate a sample to monitor progress
                sample = generate(m, max_new_tokens=240)
                print(f"Sample:\n{sample}\n")

            # Training step
            xb, yb, mb = get_batch("train")
            logits, loss = model(xb, yb, mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Total training time: {time.time() - start:.2f}s")
        print(f"Saving weights to {weights_path}")
        torch.save(m.state_dict(), weights_path)

    # Final generation: 2000 characters
    start = time.time()
    output = generate(
        m, max_new_tokens=2000, temp=0.8, confidence_threshold=0.95, top_k=2
    )
    print(f"Total generation time: {time.time() - start:.2f}s")
    print(f"\nOutput:\n{output}")
