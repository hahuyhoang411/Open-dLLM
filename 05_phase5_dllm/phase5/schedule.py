"""Noise schedule, ELBO weighting, LR schedule, and CART rescheduling."""

import torch
import torch.nn.functional as F

from . import config

_cart_kernel_cache = {}


def sample_timesteps(batch_size, num_blocks, block_size, t_min=config.t_min):
    """Sample per-block t ~ U[t_min, 1), expand to per-token.

    Uses antithetic (stratified) sampling across batch*blocks for
    variance reduction. Ref: bd3lms/diffusion.py:776-780.
    """
    eps = torch.rand(batch_size, num_blocks)
    # Antithetic sampling: stratify across batch*blocks
    total = batch_size * num_blocks
    offset = torch.arange(total).view(batch_size, num_blocks).float() / total
    eps = (eps / total + offset) % 1
    t_blocks = t_min + (1 - t_min) * eps                                 # (B, num_blocks)
    t = t_blocks.repeat_interleave(block_size, dim=1)                    # (B, L)
    return t_blocks, t


def apply_noise(targets, t, mask_token_id=config.mask_token_id, pad_token_id=None,
                block_size=config.block_size):
    """LINEAR schedule: mask_prob = t. Apply noise mask to targets.

    Per-block min-1-masked guarantee (Stable-DiffCoder): every block of
    block_size tokens has at least one masked real token, eliminating wasted
    supervision. At t=0.05, B=32: P(zero masks per block) = 0.95^32 = 19%.
    """
    B, L = targets.shape
    mask_prob = t  # linear schedule: trivially mask_prob = t

    noise_mask = torch.rand(B, L, device=targets.device) < mask_prob.to(targets.device)

    # Don't mask padding positions
    if pad_token_id is not None:
        padding = targets != pad_token_id
        noise_mask = noise_mask & padding

    # Per-block min-1-masked guarantee (replaces per-sequence check)
    num_blocks = L // block_size
    mask_blocks = noise_mask.view(B, num_blocks, block_size)
    zero_blocks = mask_blocks.sum(dim=2) == 0  # (B, num_blocks)
    if zero_blocks.any():
        for b_idx, blk_idx in zip(*zero_blocks.nonzero(as_tuple=True)):
            start = blk_idx * block_size
            if pad_token_id is not None:
                real_pos = (targets[b_idx, start:start + block_size] != pad_token_id).nonzero(as_tuple=True)[0]
            else:
                real_pos = torch.arange(block_size, device=targets.device)
            if len(real_pos) > 0:
                pick = start + real_pos[torch.randint(len(real_pos), (1,))]
                noise_mask[b_idx, pick] = True

    x_noisy = targets.clone()
    x_noisy[noise_mask] = mask_token_id
    return x_noisy, noise_mask


def compute_elbo_weight(t, t_min=config.t_min):
    """LINEAR schedule ELBO weight: 1/t (mask_prob = t, so 1/mask_prob = 1/t)."""
    return 1.0 / t.clamp(min=t_min)


def get_lr_factor(step, warmup_iters=config.warmup_iters,
                  decay_start=config.decay_start, max_iters=config.max_iters):
    """WSD learning rate schedule. Returns multiplier in [0, 1]."""
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    if step < decay_start:
        return 1.0
    return max(0.0, 1.0 - (step - decay_start) / (max_iters - decay_start))


def compute_cart_weights(mask, padding, p=config.cart_p):
    """CART context-adaptive ELBO weights (optional, off by default).

    More context around a masked token → higher weight (stronger penalty).
    Matches Dream 7B (arXiv:2412.06264) exactly: geometric kernel convolved
    over unmasked positions, masked positions keep weight, unmasked get 0.
    """
    B, L = mask.shape
    context = padding & ~mask  # unmasked real tokens

    key = (L, p)
    if key not in _cart_kernel_cache:
        max_dist = min(L, 100)
        d = torch.arange(1, max_dist + 1, dtype=torch.float32)
        geo = 0.5 * p * ((1 - p) ** (d - 1))
        kernel = torch.cat([geo.flip(0), torch.zeros(1), geo]).view(1, 1, -1)
        _cart_kernel_cache[key] = (kernel, max_dist)
    kernel, max_dist = _cart_kernel_cache[key]

    kernel = kernel.to(mask.device)

    ctx = F.pad(context.float().unsqueeze(1), (max_dist, max_dist))
    cart_scores = F.conv1d(ctx, kernel).squeeze(1)  # (B, L)

    # Raw scores: more context → higher weight (Dream 7B semantics).
    # Tokens with more unmasked neighbors get stronger gradient signal.
    # Unmasked positions get 0.
    cart_scores = cart_scores * mask.float()
    return cart_scores
