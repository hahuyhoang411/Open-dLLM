"""Phase 6 configuration: dataclass + factory functions.

No module-level side effects. Import freely without triggering argparse or DDP.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


@dataclass
class Config:
  # --- Mode ---
  train: bool = False
  prompt: Optional[str] = None
  max_tokens: int = 512

  # --- Architecture (Qwen3-0.6B defaults) ---
  n_layer: int = 28
  n_embd: int = 1024
  n_head: int = 16
  n_kv_head: int = 8
  head_dim: int = 128  # independent — Qwen3 has n_head*head_dim != n_embd
  mlp_hidden: int = 3072
  vocab_size: int = 151_936
  seq_len: int = 2048
  block_size: int = 8
  rope_base: float = 1_000_000
  rms_eps: float = 1e-6

  # --- Token IDs ---
  mask_token_id: int = 151669  # added to Qwen3 vocab
  eos_token_id: int = 151645  # <|im_end|>
  pad_token_id: int = 151643  # <|endoftext|>

  # --- Architecture toggles ---
  use_emb_norm: bool = False  # Qwen3 doesn't use it
  use_gated_query: bool = False  # Qwen3 doesn't have it
  use_qk_norm: bool = True  # Qwen3 has it

  # --- Training ---
  batch_size: int = 32
  max_iters: int = 50_000
  dropout: float = 0.0
  grad_accum_steps: int = 1
  muon_lr: float = 0.02
  adamw_lr: float = 3e-3
  grad_clip: float = 1.0
  t_min: float = 0.1
  use_cart: bool = False
  cart_p: float = 0.1
  denoise_steps: int = 10

  # --- Feature flags ---
  use_amp: bool = True
  use_liger: bool = True
  use_flce: bool = True
  use_flex: bool = True
  use_grad_ckpt: bool = False
  use_compile: bool = True
  use_muon: bool = True
  use_fp8: bool = False

  # --- Data ---
  # NOTE: Phase 5 data ('HoangHa/100BT-dLLM-pretokenized') uses vocab=49152 tokenizer.
  # Phase 6 uses Qwen3 vocab=151936. Default empty → streaming mode with Qwen3 tokenizer.
  # Set to a Qwen3-tokenized dataset when available.
  data_dir: str = ''

  # --- Checkpointing ---
  resume: Optional[str] = None
  ckpt_dir: Optional[str] = None
  eval_every: int = 1000
  eval_iters: int = 50
  ckpt_interval: int = 250

  # --- Tracking ---
  trackio_space: Optional[str] = None

  # --- Debug ---
  debug: bool = False

  # --- Device / DDP (set by setup_device) ---
  device: str = 'cpu'
  ddp: bool = False
  ddp_rank: int = 0
  ddp_local_rank: int = 0
  ddp_world_size: int = 1
  master_process: bool = True

  # --- HF weight loading ---
  hf_model_name: Optional[str] = None

  # --- Derived (set by validate) ---
  num_blocks: int = 0
  warmup_iters: int = 0
  decay_start: int = 0

  def validate(self) -> Config:
    assert self.n_head % self.n_kv_head == 0, f'n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}'
    assert self.seq_len % self.block_size == 0, f'seq_len={self.seq_len} not divisible by block_size={self.block_size}'
    self.num_blocks = self.seq_len // self.block_size
    self.warmup_iters = min(2000, max(1, int(0.07 * self.max_iters)))
    self.decay_start = int(0.8 * self.max_iters)
    return self


# ---------------------------------------------------------------------------
# Feature availability checks (lazy — only run when called)
# ---------------------------------------------------------------------------

_LIGER_AVAILABLE: Optional[bool] = None
_FLCE_AVAILABLE: Optional[bool] = None
_FLEX_AVAILABLE: Optional[bool] = None
_TRACKIO_AVAILABLE: Optional[bool] = None


def _check_liger() -> bool:
  global _LIGER_AVAILABLE
  if _LIGER_AVAILABLE is None:
    try:
      from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
      from liger_kernel.transformers import LigerRMSNorm  # noqa: F401

      _LIGER_AVAILABLE = True
    except ImportError:
      _LIGER_AVAILABLE = False
  return _LIGER_AVAILABLE


def _check_flce() -> bool:
  global _FLCE_AVAILABLE
  if _FLCE_AVAILABLE is None:
    try:
      from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: F401

      _FLCE_AVAILABLE = True
    except ImportError:
      _FLCE_AVAILABLE = False
  return _FLCE_AVAILABLE


def _check_flex() -> bool:
  global _FLEX_AVAILABLE
  if _FLEX_AVAILABLE is None:
    try:
      from torch.nn.attention.flex_attention import (  # noqa: F401
        BlockMask,
        create_block_mask,
        flex_attention,
      )

      _FLEX_AVAILABLE = True
    except (ImportError, AttributeError):
      _FLEX_AVAILABLE = False
  return _FLEX_AVAILABLE


def _check_trackio() -> bool:
  global _TRACKIO_AVAILABLE
  if _TRACKIO_AVAILABLE is None:
    try:
      import trackio  # noqa: F401

      _TRACKIO_AVAILABLE = True
    except ImportError:
      _TRACKIO_AVAILABLE = False
  return _TRACKIO_AVAILABLE


# ---------------------------------------------------------------------------
# Factory: CLI -> Config
# ---------------------------------------------------------------------------


def from_cli() -> Config:
  """Parse CLI args into a Config. Only call from train.py main."""
  p = argparse.ArgumentParser(description='phase6_dllm - Qwen3 block diffusion LM')

  # Mode
  p.add_argument('--train', action='store_true')
  p.add_argument('--prompt', type=str, default=None)
  p.add_argument('--max-tokens', type=int, default=512)

  # Architecture
  p.add_argument('--n-layer', type=int, default=28)
  p.add_argument('--n-embd', type=int, default=1024)
  p.add_argument('--n-head', type=int, default=16)
  p.add_argument('--n-kv-head', type=int, default=8)
  p.add_argument('--head-dim', type=int, default=128)
  p.add_argument('--mlp-hidden', type=int, default=3072)
  p.add_argument('--vocab-size', type=int, default=151_936)
  p.add_argument('--seq-len', type=int, default=2048)
  p.add_argument('--block-size', type=int, default=8)
  p.add_argument('--rope-base', type=float, default=1_000_000)
  p.add_argument('--rms-eps', type=float, default=1e-6)

  # Token IDs
  p.add_argument('--mask-token-id', type=int, default=151669)
  p.add_argument('--eos-token-id', type=int, default=151645)
  p.add_argument('--pad-token-id', type=int, default=151643)

  # Architecture toggles
  p.add_argument('--use-emb-norm', action='store_true')
  p.add_argument('--use-gated-query', action='store_true')
  p.add_argument('--no-qk-norm', action='store_true')

  # Training
  p.add_argument('--batch-size', type=int, default=32)
  p.add_argument('--max-iters', type=int, default=50_000)
  p.add_argument('--dropout', type=float, default=0.0)
  p.add_argument('--grad-accum-steps', type=int, default=1)
  p.add_argument('--muon-lr', type=float, default=0.02)
  p.add_argument('--adamw-lr', type=float, default=3e-3)
  p.add_argument('--grad-clip', type=float, default=1.0)
  p.add_argument('--t-min', type=float, default=0.1)
  p.add_argument('--cart', action='store_true')
  p.add_argument('--cart-p', type=float, default=0.1)
  p.add_argument('--denoise-steps', type=int, default=10)

  # Feature flags
  p.add_argument('--no-amp', action='store_true')
  p.add_argument('--no-liger', action='store_true')
  p.add_argument('--no-flce', action='store_true')
  p.add_argument('--no-flex', action='store_true')
  p.add_argument('--grad-ckpt', action='store_true')
  p.add_argument('--no-compile', action='store_true')
  p.add_argument('--no-muon', action='store_true')
  p.add_argument('--fp8', action='store_true')

  # Data
  p.add_argument(
    '--data-dir', type=str, default='', help='HF Hub dataset or local path. Empty = streaming with Qwen3 tokenizer'
  )

  # Checkpointing
  p.add_argument('--resume', type=str, default=None)
  p.add_argument('--ckpt-dir', type=str, default=None)
  p.add_argument('--eval-every', type=int, default=1000)
  p.add_argument('--ckpt-interval', type=int, default=250)

  # Tracking
  p.add_argument('--trackio-space', type=str, default=None)

  # Debug
  p.add_argument('--debug', action='store_true')

  # HF loading
  p.add_argument('--hf-model-name', type=str, default=None)

  a = p.parse_args()

  cfg = Config(
    # Mode
    train=a.train,
    prompt=a.prompt,
    max_tokens=a.max_tokens,
    # Architecture
    n_layer=a.n_layer,
    n_embd=a.n_embd,
    n_head=a.n_head,
    n_kv_head=a.n_kv_head,
    head_dim=a.head_dim,
    mlp_hidden=a.mlp_hidden,
    vocab_size=a.vocab_size,
    seq_len=a.seq_len,
    block_size=a.block_size,
    rope_base=a.rope_base,
    rms_eps=a.rms_eps,
    # Token IDs
    mask_token_id=a.mask_token_id,
    eos_token_id=a.eos_token_id,
    pad_token_id=a.pad_token_id,
    # Architecture toggles
    use_emb_norm=a.use_emb_norm,
    use_gated_query=a.use_gated_query,
    use_qk_norm=not a.no_qk_norm,
    # Training
    batch_size=a.batch_size,
    max_iters=a.max_iters,
    dropout=a.dropout,
    grad_accum_steps=a.grad_accum_steps,
    muon_lr=a.muon_lr,
    adamw_lr=a.adamw_lr,
    grad_clip=a.grad_clip,
    t_min=a.t_min,
    use_cart=a.cart,
    cart_p=a.cart_p,
    denoise_steps=a.denoise_steps,
    # Feature flags
    use_amp=not a.no_amp,
    use_liger=not a.no_liger,
    use_flce=not a.no_flce,
    use_flex=not a.no_flex,
    use_grad_ckpt=a.grad_ckpt,
    use_compile=not a.no_compile,
    use_muon=not a.no_muon,
    use_fp8=a.fp8,
    # Data
    data_dir=a.data_dir,
    # Checkpointing
    resume=a.resume,
    ckpt_dir=a.ckpt_dir,
    eval_every=a.eval_every,
    ckpt_interval=a.ckpt_interval,
    # Tracking
    trackio_space=a.trackio_space,
    # Debug
    debug=a.debug,
    # HF loading
    hf_model_name=a.hf_model_name,
  )
  cfg.validate()
  return cfg


# ---------------------------------------------------------------------------
# Device / DDP setup — call explicitly from train.py
# ---------------------------------------------------------------------------


def setup_device(cfg: Config) -> Config:
  import torch
  import torch.distributed as dist

  cfg.ddp = int(os.environ.get('RANK', -1)) != -1
  if cfg.ddp:
    dist.init_process_group(backend='nccl')
    cfg.ddp_rank = int(os.environ['RANK'])
    cfg.ddp_local_rank = int(os.environ['LOCAL_RANK'])
    cfg.ddp_world_size = int(os.environ['WORLD_SIZE'])
    cfg.device = f'cuda:{cfg.ddp_local_rank}'
    torch.cuda.set_device(cfg.device)
    cfg.master_process = cfg.ddp_rank == 0
  else:
    cfg.ddp_rank = 0
    cfg.ddp_local_rank = 0
    cfg.ddp_world_size = 1
    cfg.master_process = True
    cfg.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

  torch.manual_seed(1337 + cfg.ddp_rank)
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

  return cfg


# ---------------------------------------------------------------------------
# Feature availability — call explicitly from train.py
# ---------------------------------------------------------------------------


def setup_features(cfg: Config) -> Config:
  import torch

  has_cuda = torch.cuda.is_available()
  cfg.use_amp = cfg.use_amp and has_cuda
  cfg.use_liger = cfg.use_liger and _check_liger()
  cfg.use_flce = cfg.use_flce and _check_flce()
  cfg.use_flex = cfg.use_flex and _check_flex() and has_cuda
  cfg.use_compile = cfg.use_compile and has_cuda
  cfg.use_fp8 = cfg.use_fp8 and has_cuda
  return cfg
