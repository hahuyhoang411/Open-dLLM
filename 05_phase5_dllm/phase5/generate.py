"""Block-by-block generation with KV cache and confidence-based unmasking.

Sampling uses Gumbel-max (ref: dllm/core/samplers/utils.py:add_gumbel_noise).
Confidence = argmax token probability (ref: dllm/core/samplers/bd3lm.py:108).
"""

import time

import torch
import torch.nn.functional as F

from .config import block_size, device, eos_token_id, mask_token_id, pad_token_id


def _add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. Ref: dllm/core/samplers/utils.py:72-83."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise.clamp(min=1e-20))) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate(model, encode_fn, decode_fn, prompt='', max_new_tokens=512,
             denoise_steps=10, temperature=0.7, top_k=50,
             confidence_threshold=0.5):
    was_training = model.training
    model.eval()
    model.reset_kv_cache()
    model.set_cache_mode(False)

    try:
        # Encode prompt
        prompt_ids = encode_fn(prompt) if prompt else []
        prompt_len = len(prompt_ids)
        all_tokens = list(prompt_ids)

        # Split prompt into full blocks for KV cache warmup
        n_full_prompt_blocks = prompt_len // block_size
        prompt_remainder = prompt_len % block_size
        pos_offset = 0

        model.set_cache_mode(True)
        for i in range(n_full_prompt_blocks):
            start = i * block_size
            end = start + block_size
            block_ids = torch.tensor(
                [prompt_ids[start:end]], dtype=torch.long, device=device
            )
            model(block_ids, pos_offset=pos_offset)
            pos_offset += block_size
        model.set_cache_mode(False)

        # Block-by-block generation
        tokens_generated = 0
        total_steps = 0
        done = False
        t_start = time.time()

        while tokens_generated < max_new_tokens and not done:
            fill_from_prompt = min(prompt_remainder, block_size)

            # Initialize block: prompt remainder (if any) + mask tokens
            block = torch.full(
                (1, block_size), mask_token_id, dtype=torch.long, device=device
            )
            if fill_from_prompt > 0:
                remainder_start = n_full_prompt_blocks * block_size
                block[0, :fill_from_prompt] = torch.tensor(
                    prompt_ids[remainder_start:remainder_start + fill_from_prompt],
                    dtype=torch.long, device=device,
                )

            # Track which positions need decoding
            masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
            masked[0, fill_from_prompt:] = True

            # Precompute how many tokens to unmask per step (uniform schedule)
            n_masked = masked.sum().item()
            tokens_per_step = max(1, n_masked // denoise_steps)

            # Denoise loop: iteratively unmask within this block
            for step in range(denoise_steps):
                if not masked.any():
                    break
                total_steps += 1

                logits, _ = model(block, pos_offset=pos_offset)

                # Suppress mask/padding tokens
                logits[:, :, mask_token_id] = -float('inf')
                logits[:, :, pad_token_id] = -float('inf')

                # Gumbel-max sampling (ref: dllm samplers)
                noisy_logits = _add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(noisy_logits, dim=-1)  # (1, block_size)

                # Per-token confidence = probability of predicted token
                # Ref: dllm/core/samplers/bd3lm.py:108
                probs = F.softmax(logits.float(), dim=-1)
                confidences = torch.gather(
                    probs, -1, x0.unsqueeze(-1)
                ).squeeze(-1)  # (1, block_size)

                # Determine which positions to unmask this step
                masked_confidences = torch.where(
                    masked, confidences,
                    torch.tensor(-float('inf'), device=device),
                )

                # Unmask top-k most confident masked positions
                n_to_unmask = min(tokens_per_step, int(masked.sum().item()))
                if step == denoise_steps - 1:
                    n_to_unmask = int(masked.sum().item())
                n_to_unmask = max(1, n_to_unmask)

                _, top_indices = torch.topk(
                    masked_confidences.view(-1), k=n_to_unmask
                )
                decode_mask = torch.zeros_like(masked.view(-1))
                decode_mask[top_indices] = True
                decode_mask = decode_mask.view(1, block_size).bool()

                block = torch.where(decode_mask, x0, block)
                masked = masked & ~decode_mask

            # Cache the finalized block
            model.set_cache_mode(True)
            model(block, pos_offset=pos_offset)
            model.set_cache_mode(False)
            pos_offset += block_size

            # Consume prompt remainder (only affects first generated block)
            prompt_remainder = 0

            # Extract generated tokens (skip prompt remainder portion)
            new_tokens = block[0, fill_from_prompt:].tolist()

            # Truncate at EOS
            for i, tok in enumerate(new_tokens):
                if tok == eos_token_id:
                    new_tokens = new_tokens[:i]
                    done = True
                    break

            all_tokens.extend(new_tokens)
            tokens_generated += len(new_tokens)

        elapsed = time.time() - t_start
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float('inf')
        print(f'Generation: {total_steps} denoise steps, {tokens_generated} tokens, '
              f'{tok_per_sec:.1f} tok/s')

        # Filter special tokens before decoding
        decoded_tokens = [t for t in all_tokens if t >= 14]  # skip special IDs 0-13
        return decode_fn(decoded_tokens)

    finally:
        # Reset cache_mode AND KV cache so stale state doesn't corrupt training
        model.set_cache_mode(False)
        model.reset_kv_cache()
        if was_training:
            model.train()
