"""Toy model factory for fast kernel testing."""

from __future__ import annotations

import torch

from .config import Config
from .data import _compute_positions, _sample_t_per_doc
from .schedule import apply_noise, compute_elbo_weight

TOY_VOCAB = 512


def toy_config(**overrides) -> Config:
  defaults = dict(
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_kv_head=2,
    head_dim=64,
    mlp_hidden=256,
    vocab_size=TOY_VOCAB,
    seq_len=64,
    block_size=8,
    rope_base=10_000,
    rms_eps=1e-6,
    dropout=0.0,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=True,
    use_grad_ckpt=False,
    use_flex=False,
    use_muon=True,
    use_fp8=False,
    use_amp=False,
    use_compile=False,
    pad_token_id=0,
    mask_token_id=1,
    eos_token_id=2,
    batch_size=2,
    max_iters=100,
    muon_lr=0.02,
    adamw_lr=3e-3,
    grad_clip=1.0,
    t_min=0.1,
  )
  defaults.update(overrides)
  return Config(**defaults).validate()


def qwen3_config(**overrides) -> Config:
  defaults = dict(
    n_layer=28,
    n_embd=1024,
    n_head=16,
    n_kv_head=8,
    head_dim=128,
    mlp_hidden=3072,
    vocab_size=151_936,
    seq_len=2048,
    block_size=8,
    rope_base=1_000_000,
    rms_eps=1e-6,
    dropout=0.0,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=True,
    use_grad_ckpt=False,
    use_flex=False,
    use_muon=True,
    use_fp8=False,
    use_amp=True,
    use_compile=True,
    pad_token_id=151_643,
    mask_token_id=151_669,
    eos_token_id=151_645,
    batch_size=4,
    max_iters=50_000,
    muon_lr=0.02,
    adamw_lr=3e-3,
    grad_clip=1.0,
    t_min=0.1,
  )
  defaults.update(overrides)
  return Config(**defaults).validate()


def make_toy_batch(cfg: Config, device: str):
  """Build a synthetic training batch with realistic noise/ELBO structure.

  Returns (x_input, targets, noise_mask, elbo_w, doc_ids, positions).
  """
  B, L = cfg.batch_size, cfg.seq_len

  # Targets: random real tokens (avoid special ids 0,1,2)
  targets = torch.randint(3, cfg.vocab_size, (B, L))

  # One document per sequence
  doc_ids = torch.zeros(B, L, dtype=torch.long)
  positions = _compute_positions(doc_ids)

  t = _sample_t_per_doc(doc_ids, t_min=cfg.t_min)
  x_noisy, noise_mask = apply_noise(
    targets,
    t,
    mask_token_id=cfg.mask_token_id,
    pad_token_id=cfg.pad_token_id,
    block_size=cfg.block_size,
  )
  elbo_w = compute_elbo_weight(t, t_min=cfg.t_min)

  x_input = torch.cat([x_noisy, targets], dim=1)

  return (
    x_input.to(device),
    targets.to(device),
    noise_mask.to(device),
    elbo_w.to(device),
    doc_ids.to(device),
    positions.to(device),
  )


def run_train_step(model, batch, cfg: Config, optimizer, attn_mask):
  """Run one full training step. Returns (loss_val, grad_norm)."""
  from torch.nn.utils import clip_grad_norm_

  from .loss import compute_loss

  x_input, targets, noise_mask, elbo_w, doc_ids, positions = batch

  optimizer.zero_grad()

  h, _ = model(x_input, targets=targets, attn_mask=attn_mask)
  loss = compute_loss(h, targets, noise_mask, elbo_w, model.lm_head.weight, cfg)

  loss.backward()
  grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
  optimizer.step()

  return loss.item(), grad_norm
