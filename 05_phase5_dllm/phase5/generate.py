"""Block-by-block generation with KV cache and confidence-based unmasking."""

import time
import torch
import torch.nn.functional as F

from .config import mask_token_id, eos_token_id, pad_token_id, device, block_size


@torch.no_grad()
def generate(model, encode_fn, decode_fn, prompt="", max_new_tokens=512,
             denoise_steps=10, temperature=0.7, top_k=50,
             confidence_threshold=0.9):
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

            # Denoise loop: iteratively unmask within this block
            for step in range(denoise_steps):
                if not masked.any():
                    break
                total_steps += 1

                logits, _ = model(block, pos_offset=pos_offset)
                probs = F.softmax(logits / temperature, dim=-1)

                # Never generate mask or padding tokens
                probs[:, :, mask_token_id] = 0.0
                probs[:, :, pad_token_id] = 0.0

                # Top-k confidence-based unmasking
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                confidences = top_k_probs.sum(dim=-1)

                decode_mask = (confidences >= confidence_threshold) & masked

                # If nothing qualifies, force-unmask most confident masked position
                if not decode_mask.any():
                    masked_confidences = torch.where(
                        masked, confidences,
                        torch.tensor(-float("inf"), device=device),
                    )
                    decode_mask.view(-1)[masked_confidences.argmax()] = True

                # Sample from normalized top-k distribution
                top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                sampled_k = torch.multinomial(
                    top_k_probs_norm.view(-1, top_k), 1
                ).view(1, block_size)
                sampled_tokens = torch.gather(
                    top_k_indices, -1, sampled_k.unsqueeze(-1)
                ).squeeze(-1)

                block = torch.where(decode_mask, sampled_tokens, block)
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
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float("inf")
        print(f"Generation: {total_steps} denoise steps, {tokens_generated} tokens, "
              f"{tok_per_sec:.1f} tok/s")

        return decode_fn(all_tokens)

    finally:
        # CRITICAL: reset KV cache so stale batch=1 state doesn't crash training
        model.reset_kv_cache()
        if was_training:
            model.train()
