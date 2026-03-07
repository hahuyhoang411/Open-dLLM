"""Noise schedule, ELBO weighting, LR schedule, and CART rescheduling."""

import torch
import torch.nn.functional as F

from . import config

_cart_kernel_cache = {}


def sample_timesteps(batch_size, num_blocks, block_size, t_min=config.t_min):
    """Sample per-block t ~ U[t_min, 1], expand to per-token."""
    t_blocks = t_min + (1 - t_min) * torch.rand(batch_size, num_blocks)  # (B, num_blocks)
    t = t_blocks.repeat_interleave(block_size, dim=1)                    # (B, L)
    return t_blocks, t


def apply_noise(targets, t, mask_token_id=config.mask_token_id, pad_token_id=None):
    """LINEAR schedule: mask_prob = t. Apply noise mask to targets."""
    B, L = targets.shape
    mask_prob = t  # linear schedule: trivially mask_prob = t

    noise_mask = torch.rand(B, L, device=targets.device) < mask_prob.to(targets.device)

    # Don't mask padding positions
    if pad_token_id is not None:
        padding = targets != pad_token_id
        noise_mask = noise_mask & padding

    # Min-1-masked guarantee per sequence
    zero_masked = noise_mask.sum(dim=1) == 0
    if zero_masked.any():
        for idx in zero_masked.nonzero(as_tuple=True)[0]:
            if pad_token_id is not None:
                real_pos = (targets[idx] != pad_token_id).nonzero(as_tuple=True)[0]
            else:
                real_pos = torch.arange(L, device=targets.device)
            if len(real_pos) > 0:
                pick = real_pos[torch.randint(len(real_pos), (1,))]
                noise_mask[idx, pick] = True

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

    # Only masked positions get CART weight; unmasked get 0 (matching Dream)
    cart_scores = cart_scores * mask.float()
    return cart_scores
