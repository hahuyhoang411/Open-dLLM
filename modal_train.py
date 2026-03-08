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

    # Resume a specific run
    modal run modal_train.py --run-id 20260307_212400

    # Download checkpoint from a run
    modal volume get dllm-checkpoints phase5/20260307_212400/latest.pt ./latest.pt

    # Data sources (--data-dir flag):
    modal run modal_train.py                                              # HoangHa/100BT-dLLM-pretokenized (default)
    modal run modal_train.py --data-dir streaming                         # HuggingFaceFW/finepdfs_50BT-... on-the-fly
    modal run modal_train.py --data-dir HoangHa/other-dataset             # any other HF dataset

    # Pre-tokenize dataset and push to HF Hub
    modal run modal_train.py::pretokenize
    modal run modal_train.py::pretokenize --max-docs 100000
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
    gpu="H100:8",
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
        run_id: str = "",
        data_dir: str = "HoangHa/100BT-dLLM-pretokenized",
    ):
        import os
        import subprocess
        import torch

        os.environ["HF_HOME"] = "/data"
        os.environ["HF_DATASETS_CACHE"] = "/data/datasets"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        ckpt_dir = f"/checkpoints/phase5/{run_id}" if run_id else "/checkpoints/phase5"

        n_gpus = torch.cuda.device_count()
        launcher = (
            ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}"]
            if n_gpus > 1
            else ["python"]
        )

        cmd = [
            *launcher,
            "/root/05_phase5_dllm/train.py", "--train",
            "--fp8",
            f"--ckpt-dir={ckpt_dir}",
            f"--resume={ckpt_dir}",
        ]

        # Data source: "streaming" = on-the-fly, anything else = --data-dir value
        if data_dir and data_dir != "streaming":
            cmd.append(f"--data-dir={data_dir}")

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
    base = "/checkpoints/phase5"
    if not os.path.exists(base):
        print("No phase5 checkpoints found.")
        return

    for run_dir in sorted(os.listdir(base)):
        run_path = os.path.join(base, run_dir)
        if not os.path.isdir(run_path):
            continue
        latest = os.path.join(run_path, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            print(f"Run {run_dir}: step {ckpt.get('step', '?')}, loss {ckpt.get('loss', '?'):.4f}")
        files = sorted(os.listdir(run_path))
        for f in files:
            path = os.path.join(run_path, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {f:40s} {size_mb:8.1f} MB")


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": ckpt_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def debug(run_id: str = "", ckpt_name: str = ""):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data"
    os.environ["HF_DATASETS_CACHE"] = "/data/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ckpt_vol.reload()
    base = f"/checkpoints/phase5/{run_id}" if run_id else "/checkpoints/phase5"

    # Find checkpoint
    if ckpt_name:
        ckpt_path = os.path.join(base, ckpt_name)
    else:
        # Find latest checkpoint
        candidates = sorted(f for f in os.listdir(base) if f.startswith("ckpt_"))
        if candidates:
            ckpt_path = os.path.join(base, candidates[-1])
        else:
            print(f"No checkpoints found in {base}")
            return
    print(f"Using checkpoint: {ckpt_path}")

    cmd = [
        "python", "/root/05_phase5_dllm/debug_generate.py",
        f"--ckpt={ckpt_path}", "--skip-data",
    ]
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        print(f"Debug script failed with code {proc.returncode}")


@app.function(
    image=image,
    cpu=16,
    memory=64 * 1024,
    timeout=86400,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pretokenize(max_docs: int = 0, hub_repo: str = "HoangHa/100BT-dLLM-pretokenized"):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data"
    os.environ["HF_DATASETS_CACHE"] = "/data/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        "python", "/root/05_phase5_dllm/pretokenize.py",
        f"--hub-repo={hub_repo}",
        "--num-proc=16",
        "--work-dir=/data/tmp_shards",
    ]
    if max_docs > 0:
        cmd.extend(["--max-docs", str(max_docs)])

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    data_vol.commit()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Cleanup: remove intermediate caches to free volume space
    import shutil
    cache_dir = "/data/datasets"
    for item in os.listdir(cache_dir):
        path = os.path.join(cache_dir, item)
        if os.path.isdir(path) and "finepdfs" in item:
            print(f"Cleaning up source cache: {path}")
            shutil.rmtree(path)
    if os.path.isdir("/data/tmp_shards"):
        print("Cleaning up tokenized shards work dir")
        shutil.rmtree("/data/tmp_shards")
    data_vol.commit()


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/checkpoints": ckpt_vol},
)
def download():
    print("Use the Modal CLI to download checkpoints:")
    print("  modal volume get dllm-checkpoints latest.pt ./latest.pt")


@app.local_entrypoint()
def main(
    gpu: str = "H100:8",
    trackio_space: str = "",
    extra_args: str = "",
    run_id: str = "",
    data_dir: str = "HoangHa/100BT-dLLM-pretokenized",
):
    if not run_id:
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}  (checkpoints at /checkpoints/phase5/{run_id}/)")
    Train.with_options(gpu=gpu)().run.remote(
        trackio_space=trackio_space, extra_args=extra_args, run_id=run_id,
        data_dir=data_dir,
    )
