"""Checkpoint save/load with atomic writes + HF weight loading for Qwen3."""

import os
import pathlib
import random
import re
import shutil

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Save / Load (verbatim from Phase 5)
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, loss, ckpt_dir):
    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    # Strip _orig_mod from per-block compiled keys for portable checkpoints
    sd = {k.replace('._orig_mod.', '.'): v for k, v in raw_model.state_dict().items()}
    ckpt = {
        'step': step,
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': {
            'python': random.getstate(),
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'loss': loss,
    }
    pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
    # Named checkpoint (atomic: tmp then replace, survives preemption mid-write)
    path = os.path.join(ckpt_dir, f'ckpt_{step:06d}.pt')
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    pathlib.Path(tmp_path).replace(path)
    # Atomic latest: copy from the already-written named checkpoint (no double serialize)
    latest = os.path.join(ckpt_dir, 'latest.pt')
    tmp_latest = latest + '.tmp'
    shutil.copy2(path, tmp_latest)
    pathlib.Path(tmp_latest).replace(latest)
    return path


def load_checkpoint(ckpt_path, model, optimizer=None, device='cuda'):
    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if pathlib.Path(ckpt_path).is_dir():
        ckpt_path = os.path.join(ckpt_path, 'latest.pt')
    if not pathlib.Path(ckpt_path).exists():
        return 0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # Restore RNG states
    rng = ckpt['rng_state']
    random.setstate(rng['python'])
    torch.random.set_rng_state(rng['torch'].cpu().byte())
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().byte() for s in rng['cuda']])
    return ckpt['step'] + 1


# ---------------------------------------------------------------------------
# HF → SmolDLM key mapping
# ---------------------------------------------------------------------------

_HF_TO_OURS = {
    'model.embed_tokens.weight': 'token_emb.weight',
    'model.norm.weight': 'final_norm.weight',
}

_LAYER_MAP = {
    'self_attn.q_proj.weight': 'attn.c_q.weight',
    'self_attn.k_proj.weight': 'attn.c_k.weight',
    'self_attn.v_proj.weight': 'attn.c_v.weight',
    'self_attn.o_proj.weight': 'attn.c_proj.weight',
    'self_attn.q_norm.weight': 'attn.q_norm.weight',
    'self_attn.k_norm.weight': 'attn.k_norm.weight',
    'mlp.gate_proj.weight': 'mlp.gate_proj.weight',
    'mlp.up_proj.weight': 'mlp.up_proj.weight',
    'mlp.down_proj.weight': 'mlp.down_proj.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'mlp_norm.weight',
}


def _map_hf_key(hf_key):
    """Map a single HF key to our naming convention. Returns None if unmapped."""
    if hf_key in _HF_TO_OURS:
        return _HF_TO_OURS[hf_key]
    if hf_key == 'lm_head.weight':
        return None  # tied to token_emb, skip
    m = re.match(r'model\.layers\.(\d+)\.(.*)', hf_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest in _LAYER_MAP:
            return f'blocks.{idx}.{_LAYER_MAP[rest]}'
    return None  # unknown (rotary_emb, etc.)


# ---------------------------------------------------------------------------
# Download + load HF safetensors
# ---------------------------------------------------------------------------

def _load_hf_weights(model_name):
    """Download and merge all safetensors shards from HF Hub into one state_dict."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    files = [f for f in list_repo_files(model_name) if f.endswith('.safetensors')]
    merged = {}
    for fname in files:
        path = hf_hub_download(model_name, fname)
        merged.update(load_file(path))
    return merged


def load_from_hf(model, model_name='Qwen/Qwen3-0.6B', device='cpu'):
    """Load Qwen3 HF weights into our model.
    Returns (missing_keys, unexpected_hf_keys) for verification."""
    hf_sd = _load_hf_weights(model_name)

    # Remap keys
    mapped = {}
    unexpected = []
    for hf_key, tensor in hf_sd.items():
        our_key = _map_hf_key(hf_key)
        if our_key is not None:
            mapped[our_key] = tensor
        else:
            unexpected.append(hf_key)

    # Detect tied weights BEFORE load_state_dict breaks ties.
    # named_parameters() deduplicates tied params — a key in state_dict() but
    # absent from named_parameters() is a tied alias. Exclude from missing if
    # the canonical key (in named_parameters()) was loaded.
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(mapped.keys())
    params = dict(model.named_parameters())
    loaded_ptrs = {params[k].data_ptr() for k in loaded_keys if k in params}
    # state_dict includes tied aliases; resolve them via the underlying storage
    sd_tensors = model.state_dict()
    missing = sorted(
        k for k in (model_keys - loaded_keys)
        if sd_tensors[k].data_ptr() not in loaded_ptrs
    )

    # Actually load the weights
    model.load_state_dict(mapped, strict=False)

    # Move to device
    model.to(device)

    n_loaded = len(loaded_keys & model_keys)
    n_total = len(model_keys)
    print(f'[checkpoint] Loaded {n_loaded}/{n_total} params from {model_name}')
    if missing:
        print(f'[checkpoint] Missing ({len(missing)}): {missing[:5]}...' if len(missing) > 5 else f'[checkpoint] Missing ({len(missing)}): {missing}')
    if unexpected:
        print(f'[checkpoint] Skipped HF keys ({len(unexpected)}): {unexpected[:5]}...' if len(unexpected) > 5 else f'[checkpoint] Skipped HF keys ({len(unexpected)}): {unexpected}')

    return missing, unexpected
