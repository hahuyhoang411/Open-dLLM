"""Block-by-block generation with KV cache and SDAR dynamic remasking.

Three remasking strategies:
  - confidence_dynamic: commit all tokens above tau if enough, else top-k (SDAR)
  - confidence_static: always commit exactly top-k by confidence
  - random: random token selection (baseline)

Sampling uses Gumbel-max (ref: dllm/core/samplers/utils.py:add_gumbel_noise).
"""

import time

import torch
import torch.nn.functional as F

from .config import Config


def get_num_transfer_tokens(block_length, steps):
    """Distribute block_length tokens evenly across steps. Ref: SDAR generate.py:49-54."""
    base = block_length // steps
    remainder = block_length % steps
    schedule = torch.zeros(steps, dtype=torch.int64) + base
    schedule[:remainder] += 1
    return schedule


def _add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. Ref: dllm/core/samplers/utils.py:72-83."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise.clamp(min=1e-20))) ** temperature
    return logits.exp() / gumbel_noise


def _select_tokens_dynamic(confidences, masked, num_to_commit, threshold):
    """SDAR dynamic remasking: commit high-confidence tokens or fall back to top-k.

    If count(conf > threshold among masked) >= num_to_commit -> commit ALL above threshold.
    Else -> commit exactly num_to_commit by confidence (top-k fallback).
    Minimum 1 token committed per call.
    """
    masked_conf = torch.where(masked, confidences, torch.tensor(-float('inf'), device=confidences.device))
    flat_conf = masked_conf.view(-1)

    high_conf_mask = (flat_conf > threshold)
    n_high = high_conf_mask.sum().item()

    if n_high >= num_to_commit:
        commit = high_conf_mask.view(masked.shape)
    else:
        k = max(1, num_to_commit)
        _, top_idx = torch.topk(flat_conf, k=k)
        commit_flat = torch.zeros_like(flat_conf, dtype=torch.bool)
        commit_flat[top_idx] = True
        commit = commit_flat.view(masked.shape)
    return commit


def _select_tokens_static(confidences, masked, num_to_commit):
    """Static remasking: always commit exactly num_to_commit by confidence."""
    masked_conf = torch.where(masked, confidences, torch.tensor(-float('inf'), device=confidences.device))
    k = max(1, num_to_commit)
    _, top_idx = torch.topk(masked_conf.view(-1), k=k)
    commit_flat = torch.zeros(masked_conf.numel(), dtype=torch.bool, device=masked.device)
    commit_flat[top_idx] = True
    return commit_flat.view(masked.shape)


def _select_tokens_random(masked, num_to_commit):
    """Random remasking: randomly select num_to_commit masked positions."""
    masked_indices = masked.view(-1).nonzero(as_tuple=False).squeeze(-1)
    k = min(max(1, num_to_commit), masked_indices.size(0))
    perm = torch.randperm(masked_indices.size(0), device=masked.device)[:k]
    selected = masked_indices[perm]
    commit_flat = torch.zeros(masked.numel(), dtype=torch.bool, device=masked.device)
    commit_flat[selected] = True
    return commit_flat.view(masked.shape)


@torch.no_grad()
def generate(model, prompt_ids, cfg, max_new_tokens=512, temperature=0.7,
             top_k=50, denoise_steps=None, remasking='confidence_dynamic',
             confidence_threshold=0.9):
    """Block-by-block diffusion generation with dynamic remasking.

    Args:
        model: Phase 6 Model instance.
        prompt_ids: list[int] of prompt token IDs.
        cfg: Config instance (provides mask_token_id, eos_token_id, block_size, etc.).
        max_new_tokens: maximum tokens to generate.
        temperature: sampling temperature (0 = greedy argmax).
        top_k: unused placeholder for future top-k filtering.
        denoise_steps: denoising steps per block. None -> cfg.denoise_steps.
        remasking: 'confidence_dynamic' | 'confidence_static' | 'random'.
        confidence_threshold: tau for dynamic remasking.

    Returns:
        list[int] of token IDs (prompt + generated, special tokens filtered).
    """
    if remasking not in ('confidence_dynamic', 'confidence_static', 'random'):
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    if denoise_steps is None:
        denoise_steps = cfg.denoise_steps

    mask_id = cfg.mask_token_id
    eos_id = cfg.eos_token_id
    pad_id = cfg.pad_token_id
    blk = cfg.block_size
    special_ids = {mask_id, eos_id, pad_id}

    was_training = model.training
    model.eval()
    model.reset_kv_cache()
    model.set_cache_mode(False)

    device = next(model.parameters()).device

    try:
        prompt_len = len(prompt_ids)
        all_tokens = list(prompt_ids)

        # Split prompt into full blocks for KV cache warmup
        n_full_prompt_blocks = prompt_len // blk
        prompt_remainder = prompt_len % blk
        pos_offset = 0

        model.set_cache_mode(True)
        for i in range(n_full_prompt_blocks):
            start = i * blk
            end = start + blk
            block_ids = torch.tensor(
                [prompt_ids[start:end]], dtype=torch.long, device=device
            )
            model(block_ids, pos_offset=pos_offset)
            pos_offset += blk
        model.set_cache_mode(False)

        # Transfer token schedule (same for every generated block)
        num_transfer = get_num_transfer_tokens(blk, denoise_steps)

        # Block-by-block generation
        tokens_generated = 0
        total_steps = 0
        done = False
        t_start = time.time()

        while tokens_generated < max_new_tokens and not done:
            fill_from_prompt = min(prompt_remainder, blk)

            # Initialize block: prompt remainder (if any) + mask tokens
            block = torch.full(
                (1, blk), mask_id, dtype=torch.long, device=device
            )
            if fill_from_prompt > 0:
                remainder_start = n_full_prompt_blocks * blk
                block[0, :fill_from_prompt] = torch.tensor(
                    prompt_ids[remainder_start:remainder_start + fill_from_prompt],
                    dtype=torch.long, device=device,
                )

            # Track which positions need decoding
            masked = torch.zeros(1, blk, dtype=torch.bool, device=device)
            masked[0, fill_from_prompt:] = True

            # Denoise loop
            for step in range(denoise_steps):
                if not masked.any():
                    break
                total_steps += 1

                logits, _ = model(block, pos_offset=pos_offset)

                # Suppress special tokens (mask, pad, and Qwen3 specials 151643-151668)
                logits[:, :, mask_id] = -float('inf')
                logits[:, :, pad_id] = -float('inf')
                # Suppress Qwen3 special tokens range (except eos which model should predict)
                for sid in range(151643, 151669):
                    if sid != eos_id and sid < logits.shape[-1]:
                        logits[:, :, sid] = -float('inf')

                # Gumbel-max sampling
                noisy_logits = _add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(noisy_logits, dim=-1)  # (1, blk)

                # Per-token confidence = probability of predicted token
                probs = F.softmax(logits.float(), dim=-1)
                confidences = torch.gather(
                    probs, -1, x0.unsqueeze(-1)
                ).squeeze(-1)  # (1, blk)

                # How many tokens to commit this step
                n_masked = int(masked.sum().item())
                n_budget = int(num_transfer[step].item())
                is_last = (step == denoise_steps - 1)

                if is_last:
                    n_budget = n_masked  # commit everything on last step

                n_budget = min(n_budget, n_masked)
                n_budget = max(1, n_budget)

                # Select which positions to commit
                if remasking == 'confidence_dynamic':
                    commit = _select_tokens_dynamic(
                        confidences, masked, n_budget, confidence_threshold)
                elif remasking == 'confidence_static':
                    commit = _select_tokens_static(confidences, masked, n_budget)
                else:  # random
                    commit = _select_tokens_random(masked, n_budget)

                block = torch.where(commit, x0, block)
                masked = masked & ~commit

            # Cache the finalized block
            model.set_cache_mode(True)
            model(block, pos_offset=pos_offset)
            model.set_cache_mode(False)
            pos_offset += blk

            # Consume prompt remainder (only affects first generated block)
            prompt_remainder = 0

            # Extract generated tokens (skip prompt remainder portion)
            new_tokens = block[0, fill_from_prompt:].tolist()

            # Truncate at EOS
            for i, tok in enumerate(new_tokens):
                if tok == eos_id:
                    new_tokens = new_tokens[:i]
                    done = True
                    break

            all_tokens.extend(new_tokens)
            tokens_generated += len(new_tokens)

        elapsed = time.time() - t_start
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float('inf')
        print(f'Generation: {total_steps} denoise steps, {tokens_generated} tokens, '
              f'{tok_per_sec:.1f} tok/s')

        # Filter special tokens
        return [t for t in all_tokens if t not in special_ids]

    finally:
        model.set_cache_mode(False)
        model.reset_kv_cache()
        if was_training:
            model.train()
