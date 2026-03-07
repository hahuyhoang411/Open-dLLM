"""Checkpoint save/load with atomic writes and RNG state preservation."""

import os
import random
import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, step, loss, ckpt_dir):
    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    ckpt = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "loss": loss,
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    # Named checkpoint (atomic: tmp then replace, survives preemption mid-write)
    path = os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt")
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    # Atomic latest symlink
    latest = os.path.join(ckpt_dir, "latest.pt")
    tmp = latest + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, latest)
    return path


def load_checkpoint(ckpt_path, model, optimizer=None, device="cuda"):
    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "latest.pt")
    if not os.path.exists(ckpt_path):
        return 0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Restore RNG states
    rng = ckpt["rng_state"]
    random.setstate(rng["python"])
    torch.random.set_rng_state(rng["torch"].cpu().byte())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().byte() for s in rng["cuda"]])
    return ckpt["step"] + 1
