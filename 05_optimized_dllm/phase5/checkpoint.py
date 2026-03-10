"""Checkpoint save/load with atomic writes and RNG state preservation."""

import os
import pathlib
import random
import shutil

import torch
from torch import nn


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
