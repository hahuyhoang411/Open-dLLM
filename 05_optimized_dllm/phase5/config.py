"""Phase 5 configuration: CLI arguments, hyperparameters, device setup.

All modules import from here. No circular dependencies.
"""

import argparse
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.distributed as dist

# ============================================================================
# CLI Arguments
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description='phase5_dllm - modern block diffusion LM')
    # Mode
    p.add_argument('--train', action='store_true', help='train from scratch')
    p.add_argument('--prompt', type=str, default=None, help='text prompt for generation')
    p.add_argument('--max-tokens', type=int, default=512, help='max tokens to generate')
    # Architecture (SmolLM2-135M aligned + Gated Query Attention)
    p.add_argument('--n-layer', type=int, default=30, help='transformer layers')
    p.add_argument('--n-embd', type=int, default=576, help='embedding dimension')
    p.add_argument('--n-head', type=int, default=9, help='query heads')
    p.add_argument('--n-kv-head', type=int, default=3, help='KV heads for GQA (3:1)')
    p.add_argument('--mlp-hidden', type=int, default=1536, help='MLP hidden dim')
    p.add_argument('--seq-len', type=int, default=2048, help='sequence length')
    p.add_argument('--block-size', type=int, default=32,
                    choices=[0, 1, 2, 4, 8, 16, 32], help='diffusion block size')
    # Training
    p.add_argument('--batch-size', type=int, default=64, help='batch size per GPU')
    p.add_argument('--grad-accum-steps', type=int, default=1, help='gradient accumulation')
    p.add_argument('--dropout', type=float, default=0.0, help='MLP dropout (0=disabled)')
    p.add_argument('--denoise-steps', type=int, default=10, help='denoising steps per block')
    # Feature flags
    p.add_argument('--no-amp', action='store_true', help='disable AMP bf16')
    p.add_argument('--no-liger', action='store_true', help='disable Liger fused kernels')
    p.add_argument('--no-flex', action='store_true', help='disable FlexAttention')
    p.add_argument('--no-grad-ckpt', action='store_true', help='disable gradient checkpointing')
    p.add_argument('--no-compile', action='store_true', help='disable torch.compile')
    p.add_argument('--no-muon', action='store_true', help='use plain AdamW (no MuonClip)')
    p.add_argument('--fp8', action='store_true', help='enable FP8 training (requires H100+ GPU)')
    p.add_argument('--cart', action='store_true', help='enable CART noise rescheduling (SFT only)')
    p.add_argument('--cart-p', type=float, default=0.1, help='CART geometric param')
    p.add_argument('--t-min', type=float, default=0.1, help='min timestep (caps ELBO weight at 1/t_min)')
    # Data
    p.add_argument('--data-dir', type=str, default='HoangHa/100BT-dLLM-pretokenized',
                    help='HF Hub dataset ID or local path (default: HoangHa/100BT-dLLM-pretokenized)')
    # Checkpointing
    p.add_argument('--resume', type=str, default=None, help='checkpoint dir to resume from')
    p.add_argument('--max-iters', type=int, default=50_000, help='total training steps')
    p.add_argument('--ckpt-dir', type=str, default=None, help='checkpoint output dir')
    p.add_argument('--eval-every', type=int, default=1000, help='eval interval')
    p.add_argument('--ckpt-interval', type=int, default=250, help='checkpoint interval')
    # Tracking
    p.add_argument('--trackio-space', type=str, default=None, help='HF Space for trackio')
    # Debug
    p.add_argument('--debug', action='store_true', help='verbose DDP/grad diagnostics')
    return p.parse_args()


args = parse_args()


# ============================================================================
# Device Detection + DDP
# ============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = (
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

torch.manual_seed(1337 + ddp_rank)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


# ============================================================================
# Architecture Dimensions
# ============================================================================

n_layer = args.n_layer          # 30 (SmolLM2-135M)
n_embd = args.n_embd            # 576
n_head = args.n_head             # 9
head_dim = n_embd // n_head      # 64
n_kv_head = args.n_kv_head      # 3 (GQA 3:1)
mlp_hidden = args.mlp_hidden    # 1536

assert n_embd % n_head == 0, f'n_embd={n_embd} not divisible by n_head={n_head}'
assert n_head % n_kv_head == 0, f'n_head={n_head} not divisible by n_kv_head={n_kv_head}'

# Sequence / block dimensions
seq_len = args.seq_len
block_size = seq_len if args.block_size == 0 else args.block_size
assert seq_len % block_size == 0, f'seq_len={seq_len} not divisible by block_size={block_size}'
num_blocks = seq_len // block_size

# Vocab (fixed — tokenizer must match)
vocab_size = 49_152   # SmolLM2 merges + our 14 specials, divisible by 64

# Special token IDs
mask_token_id = 0     # <|mask|>
eos_token_id = 1      # <|endoftext|>
pad_token_id = 2      # <|padding|>


# ============================================================================
# Training Hyperparameters
# ============================================================================

batch_size = args.batch_size
max_iters = args.max_iters
eval_interval = args.eval_every
eval_iters = 50
grad_accum_steps = args.grad_accum_steps
dropout = args.dropout

# LR — single source of truth (optim.py reads these)
muon_lr = 0.02                   # Muon (Newton-Schulz) for 2D weights
adamw_lr = 3e-3                  # AdamW for embeddings/1D params (SmolLM2-135M: 3e-3)
warmup_iters = min(2000, max(1, int(0.07 * max_iters)))  # 7% of run, capped at 2K
decay_start = int(0.8 * max_iters)
grad_clip = 1.0

# Noise schedule
t_min = args.t_min  # clipped lower bound (caps ELBO weight at 1/t_min)

# CART (off by default — only useful for SFT on pretrained models, not from-scratch)
use_cart = args.cart
cart_p = args.cart_p

# Debug mode
debug = args.debug

# Data (pre-tokenized shards)
data_dir = args.data_dir


# ============================================================================
# Feature Flags
# ============================================================================

# Liger fused kernels
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction  # noqa: F401
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
    from liger_kernel.transformers import LigerRMSNorm  # noqa: F401
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

# FlexAttention
try:
    from torch.nn.attention.flex_attention import (  # noqa: F401
        BlockMask,
        create_block_mask,
        flex_attention,
    )
    _compiled_flex = torch.compile(flex_attention)
    _FLEX_AVAILABLE = True
except (ImportError, AttributeError):
    _FLEX_AVAILABLE = False
    BlockMask = type(None)

# Trackio
try:
    import trackio  # noqa: F401
    _TRACKIO_AVAILABLE = True
except ImportError:
    _TRACKIO_AVAILABLE = False

use_amp = not args.no_amp and torch.cuda.is_available()
use_grad_ckpt = not args.no_grad_ckpt
use_compile = not args.no_compile and torch.cuda.is_available()
# Liger in-model ops (RMSNorm, SwiGLU) are incompatible with whole-model compile
# (Liger Issue #174: tl.constexpr tracing failure). Inductor fuses these natively,
# so Liger only helps in eager/per-block compile mode (i.e., when grad_ckpt is on).
use_liger = not args.no_liger and _LIGER_AVAILABLE
use_flex = not args.no_flex and _FLEX_AVAILABLE and torch.cuda.is_available()
use_muon = not args.no_muon
use_fp8 = args.fp8 and torch.cuda.is_available()
