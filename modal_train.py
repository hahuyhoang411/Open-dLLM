"""
modal_train.py — Modal training infrastructure for Phase 5 of Open-dLLM.

Usage:
    # Launch training (default: 8×A100-80GB)
    modal run modal_train.py

    # Override GPU type
    modal run modal_train.py --gpu H100:8
    modal run modal_train.py --gpu A10G       # single GPU (auto-detects, skips DDP)

    # With trackio dashboard
    modal run modal_train.py --trackio-space HoangHa/open-dllm

    # Custom architecture
    modal run modal_train.py --extra-args "--n-layer 12 --n-embd 576 --dropout 0.0"

    # Check training status
    modal run modal_train.py::status

    # Download latest checkpoint
    modal volume get dllm-checkpoints latest.pt ./latest.pt
"""

import modal

app = modal.App("open-dllm-phase5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "datasets>=2.0.0",
        "tokenizers>=0.15.0",
        "liger-kernel>=0.4.0",
        "trackio>=0.18.0",
    )
    .add_local_dir("05_phase5_dllm", "/root/05_phase5_dllm")
)

ckpt_vol = modal.Volume.from_name("dllm-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("dllm-data", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100-80GB:8",
    timeout=86400,
    retries=modal.Retries(max_retries=4, initial_delay=0.0),
    volumes={"/checkpoints": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Train:
    @modal.method()
    def run(
        self,
        trackio_space: str = "",
        extra_args: str = "",
    ):
        import os
        import subprocess
        import torch

        os.environ["HF_HOME"] = "/data"
        os.environ["HF_DATASETS_CACHE"] = "/data/datasets"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        n_gpus = torch.cuda.device_count()
        launcher = (
            ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}"]
            if n_gpus > 1
            else ["python"]
        )

        cmd = [
            *launcher,
            "/root/05_phase5_dllm/train.py", "--train",
            "--ckpt-dir=/checkpoints",
            "--resume=/checkpoints",
        ]

        if trackio_space:
            cmd.append(f"--trackio-space={trackio_space}")

        if extra_args:
            cmd.extend(extra_args.split())

        print(f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()

        ckpt_vol.commit()
        data_vol.commit()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("torch>=2.5.0"),
    volumes={"/checkpoints": ckpt_vol},
)
def status():
    import os
    import torch

    ckpt_vol.reload()
    ckpt_path = "/checkpoints/latest.pt"

    if not os.path.exists(ckpt_path):
        print("No checkpoint found.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Step: {ckpt.get('step', '?')}")
    print(f"Loss: {ckpt.get('loss', '?')}")
    print()

    print("Checkpoints:")
    for f in sorted(os.listdir("/checkpoints")):
        path = os.path.join("/checkpoints", f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {f:40s} {size_mb:8.1f} MB")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/checkpoints": ckpt_vol},
)
def download():
    print("Use the Modal CLI to download checkpoints:")
    print("  modal volume get dllm-checkpoints latest.pt ./latest.pt")


@app.local_entrypoint()
def main(
    gpu: str = "A100-80GB:8",
    trackio_space: str = "",
    extra_args: str = "",
):
    Train.with_options(gpu=gpu)().run.remote(trackio_space=trackio_space, extra_args=extra_args)
