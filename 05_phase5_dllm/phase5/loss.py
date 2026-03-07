"""Loss computation for Phase 5 block diffusion LM."""

import torch
import torch.nn.functional as F

from . import config

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    _LIGER_FLCE_AVAILABLE = True
except ImportError:
    _LIGER_FLCE_AVAILABLE = False


def compute_loss(hidden_states, targets, mask, elbo_weight, lm_head_weight, use_liger=False):
    """Compute weighted cross-entropy loss for masked diffusion.

    Normalizes by ALL real tokens (not just masked). The 1/t ELBO weight already
    accounts for the masking fraction. Ref: ZHZisZZ/dllm, Phase 4 fix [P4-19].

    Expected step-0 loss: ln(49152) ~ 10.80.
    """
    B, L, D = hidden_states.shape

    if use_liger and _LIGER_FLCE_AVAILABLE:
        liger_out = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_states.contiguous().view(-1, D),   # (B*L, D)
            lm_head_weight,                            # (vocab, D)
            targets.contiguous().view(-1),             # (B*L,)
            None,                                      # bias
            None,                                      # ce_weight
            -100,                                      # ignore_index
            0.0,                                       # lse_square_scale
            0.0,                                       # label_smoothing
            "none",                                    # reduction
        )
        per_token_loss = (liger_out[0] if isinstance(liger_out, tuple) else liger_out).view(B, L)
    else:
        logits = hidden_states @ lm_head_weight.T      # (B, L, vocab)
        per_token_loss = F.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none"
        )  # (B, L)

    # Weight: per-token CE * mask indicator * ELBO weight
    weighted = per_token_loss * mask.float() * elbo_weight

    # Normalize by all real tokens (with doc packing, all are real;
    # pad check for safety/generality)
    real_count = (targets != config.pad_token_id).float().sum().clamp(min=1)
    return weighted.sum() / real_count
