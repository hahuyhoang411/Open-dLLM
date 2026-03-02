"""
base_eval.py — DCLM CORE evaluation CLI for Open-dLLM.

Downloads the eval bundle (tasks + metadata), loads either a dLLM or
HuggingFace AR model, and runs all 22 DCLM CORE benchmark tasks.

Usage examples:

    # Evaluate a dLLM model (depth=6, auto-detect weights)
    python eval/base_eval.py --model dllm --depth 6

    # Evaluate a dLLM model with explicit weights path
    python eval/base_eval.py --model dllm --depth 6 --weights path/to/weights.pt

    # Evaluate a block diffusion model (depth=10, block_size=4)
    python eval/base_eval.py --model block_dllm --depth 10 --block-size 4

    # Evaluate a HuggingFace AR model (e.g., GPT-2)
    python eval/base_eval.py --hf-model gpt2

    # Limit examples per task for quick sanity check
    python eval/base_eval.py --hf-model gpt2 --max-per-task 50

    # Evaluate dLLM with more MC samples (slower, more accurate)
    python eval/base_eval.py --model dllm --depth 6 --mc-num 128
"""

import os
import csv
import sys
import time
import json
import yaml
import random
import zipfile
import argparse
import urllib.request

import torch

try:
    from core_eval import evaluate_task
except ImportError:
    from eval.core_eval import evaluate_task

# =============================================================================
# Constants
# =============================================================================

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
CACHE_DIR = os.path.expanduser("~/.cache/open-dllm")


# =============================================================================
# Download + Extract
# =============================================================================

def download_eval_bundle():
    """Download and extract the DCLM CORE bundle if not already cached.

    The bundle contains:
      - core.yaml: task definitions (task type, delimiter, fewshot count)
      - eval_meta_data.csv: random baselines per task
      - eval_data/: JSONL files for each task

    Returns: path to the eval_bundle directory.
    """
    bundle_dir = os.path.join(CACHE_DIR, "eval_bundle")
    marker = os.path.join(bundle_dir, "core.yaml")

    if os.path.exists(marker):
        print(f"Bundle already cached at {bundle_dir}")
        return bundle_dir

    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "eval_bundle.zip")

    print(f"Downloading bundle from {EVAL_BUNDLE_URL} ...")
    urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
    print(f"Download complete ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

    print("Extracting bundle ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    print(f"Bundle extracted to {bundle_dir}")

    # Clean up zip
    os.remove(zip_path)

    return bundle_dir


# =============================================================================
# Model Loading — dLLM (nano_dllm)
# =============================================================================

def load_dllm_model(depth, weights_path, device):
    """Load our nano_dllm model for scoring.

    Adds 02_nano_dllm/ to sys.path, overrides the module-level config globals
    for the given depth, instantiates the model, and loads weights.

    Returns: (model_fn, tokenize_fn, mask_token_id, block_size)
    """
    # The script lives at eval/base_eval.py; repo root is one level up
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nano_dllm_dir = os.path.join(repo_root, "02_nano_dllm")

    # Patch sys.argv before importing nano_dllm so its parse_args() doesn't
    # choke on our CLI flags. Save and restore afterwards.
    saved_argv = sys.argv
    sys.argv = ["nano_dllm.py", f"--depth={depth}"]

    if nano_dllm_dir not in sys.path:
        sys.path.insert(0, nano_dllm_dir)

    import nano_dllm as ndllm

    # Restore original argv
    sys.argv = saved_argv

    # Override module-level config for the requested depth
    ndllm.depth = depth
    ndllm.n_layer = depth
    ndllm.n_embd = depth * 64
    ndllm.n_head = depth
    ndllm.device = device

    # Instantiate model and load weights
    model = ndllm.Model().to(device)
    print(f"Loading dLLM weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.set_default_dtype = None  # no-op, just ensuring clean state
    model.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"dLLM model loaded: depth={depth}, params={param_count:,}, device={device}")

    def model_fn(input_ids):
        logits, _ = model(input_ids)
        return logits

    def tokenize_fn(text):
        return ndllm.encode(text)

    return model_fn, tokenize_fn, ndllm.mask_token_id, ndllm.block_size


# =============================================================================
# Model Loading — Block dLLM (block_dllm)
# =============================================================================

def load_block_dllm_model(depth, block_size, weights_path, device):
    """Load the Phase 3 block diffusion model for scoring.

    Same pattern as load_dllm_model: patches sys.argv before importing
    block_dllm (which has module-level parse_args()), loads weights.

    Returns: (model_fn, tokenize_fn, mask_token_id, max_seq_len)
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    block_dllm_dir = os.path.join(repo_root, "03_block_diffusion")

    # Patch sys.argv before importing block_dllm so its parse_args() works
    saved_argv = sys.argv
    sys.argv = ["block_dllm.py", f"--depth={depth}", f"--block-size={block_size}"]

    if block_dllm_dir not in sys.path:
        sys.path.insert(0, block_dllm_dir)

    import importlib
    if "block_dllm" in sys.modules:
        importlib.reload(sys.modules["block_dllm"])
    import block_dllm as bdllm

    sys.argv = saved_argv

    # Instantiate model and load weights
    model = bdllm.Model().to(device)
    print(f"Loading block_dllm weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"block_dllm model loaded: depth={depth}, block_size={block_size}, "
          f"params={param_count:,}, device={device}")

    def model_fn(input_ids):
        logits, _ = model(input_ids)
        return logits

    def tokenize_fn(text):
        return bdllm.encode(text)

    return model_fn, tokenize_fn, bdllm.mask_token_id, bdllm.block_size_seq


# =============================================================================
# Model Loading — HuggingFace AR
# =============================================================================

def load_hf_model(hf_path, device):
    """Load a HuggingFace autoregressive model for scoring.

    Returns: (model_fn, tokenize_fn, None, max_seq_len)
        mask_token_id is None (AR models don't use masking).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"HF model loaded: {hf_path}, params={param_count:,}, device={device}")

    def model_fn(input_ids):
        return model(input_ids).logits

    def tokenize_fn(text):
        return tokenizer.encode(text)

    max_seq_len = 1024 if "gpt2" in hf_path else None
    return model_fn, tokenize_fn, None, max_seq_len


# =============================================================================
# CORE Scoring Loop
# =============================================================================

def run_core(model_fn, tokenize_fn, device, mode, mask_token_id,
             mc_num, max_seq_len, max_per_task):
    """Run all 22 DCLM CORE tasks and compute the CORE score.

    Returns: dict with per-task results and overall CORE score.
    """
    bundle_dir = download_eval_bundle()

    # Load task definitions from core.yaml
    with open(os.path.join(bundle_dir, "core.yaml")) as f:
        task_config = yaml.safe_load(f)

    # Load random baselines from eval_meta_data.csv
    baselines = {}
    with open(os.path.join(bundle_dir, "eval_meta_data.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            baselines[row["Eval Task"]] = float(row["Random baseline"])

    # Build task list: normalize icl_tasks list into (name, meta) pairs
    # YAML has: label, dataset_uri, num_fewshot (list), icl_task_type, continuation_delimiter
    # core_eval expects: task_type, continuation_delimiter, num_fewshot (int)
    tasks = []
    for entry in task_config["icl_tasks"]:
        task_meta = {
            "task_type": entry["icl_task_type"],
            "dataset_uri": entry["dataset_uri"],
            "num_fewshot": entry["num_fewshot"][0],  # YAML stores as list [N]
            "continuation_delimiter": entry.get("continuation_delimiter", " "),
        }
        tasks.append((entry["label"], task_meta))

    results = {}
    centered_scores = []

    print()
    print("=" * 60)
    print("DCLM CORE Scoring")
    print("=" * 60)
    print(f"Mode: {mode} | MC samples: {mc_num} | Max per task: {max_per_task}")
    print("=" * 60)
    print()

    for task_name, task_meta in tasks:
        dataset_uri = task_meta["dataset_uri"]
        data_path = os.path.join(bundle_dir, "eval_data", dataset_uri)

        # Load JSONL data
        data = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # Shuffle deterministically and limit
        rng = random.Random(1337)
        rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # Score this task
        t0 = time.time()
        acc = evaluate_task(
            model_fn, tokenize_fn, data, device, task_meta,
            mode=mode, mask_token_id=mask_token_id,
            mc_num=mc_num, max_seq_len=max_seq_len,
        )
        elapsed = time.time() - t0

        # Centered accuracy: (acc - baseline) / (1 - baseline)
        baseline = 0.01 * baselines.get(task_name, 0.0)
        if baseline < 1.0:
            centered = (acc - baseline) / (1.0 - baseline)
        else:
            centered = 0.0
        centered_scores.append(centered)

        results[task_name] = {
            "accuracy": acc,
            "centered_accuracy": centered,
            "baseline": baseline,
            "num_examples": len(data),
            "elapsed_seconds": elapsed,
        }

        # Free GPU memory between tasks to prevent OOM on MPS
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        print(f"  {task_name:40s} | acc={acc:.4f} | centered={centered:.4f} "
              f"| n={len(data):4d} | {elapsed:.1f}s")

    # CORE score = mean of centered accuracies
    core_score = sum(centered_scores) / len(centered_scores) if centered_scores else 0.0
    results["__core_score__"] = core_score

    # Print final table
    print()
    print("=" * 60)
    print("DCLM CORE Results")
    print("=" * 60)
    print(f"{'Task':34s} | {'Acc':>6s} | {'Centered':>8s}")
    print("-" * 34 + "-+-" + "-" * 6 + "-+-" + "-" * 8)
    for task_name, _ in tasks:
        r = results[task_name]
        print(f"{task_name:34s} | {r['accuracy']:.4f} | {r['centered_accuracy']:>8.4f}")
    print("-" * 34 + "-+-" + "-" * 6 + "-+-" + "-" * 8)
    print(f"{'CORE Score':34s} | {'':>6s} | {core_score:>8.4f}")
    print("=" * 60)

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DCLM CORE benchmark for Open-dLLM and HuggingFace models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python eval/base_eval.py --model dllm --depth 6
  python eval/base_eval.py --model block_dllm --depth 10 --block-size 4
  python eval/base_eval.py --hf-model gpt2
  python eval/base_eval.py --hf-model gpt2 --max-per-task 50
""",
    )

    parser.add_argument(
        "--model", choices=["dllm", "block_dllm"],
        help="Model type to score (dllm or block_dllm)",
    )
    parser.add_argument(
        "--depth", type=int, default=6,
        help="dLLM depth parameter (default: 6)",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to dLLM weights file (default: auto from depth)",
    )
    parser.add_argument(
        "--block-size", type=int, default=4,
        help="Block size for block_dllm model (default: 4)",
    )
    parser.add_argument(
        "--hf-model", type=str, default=None,
        help="HuggingFace model name or path (e.g., gpt2)",
    )
    parser.add_argument(
        "--mc-num", type=int, default=64,
        help="Number of MC samples for dLLM scoring (default: 64)",
    )
    parser.add_argument(
        "--max-per-task", type=int, default=-1,
        help="Limit examples per task, -1 for all (default: -1)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: auto-detect cuda > mps > cpu)",
    )

    args = parser.parse_args()

    # Validate: must specify exactly one of --model or --hf-model
    if args.model and args.hf_model:
        parser.error("Specify only one of --model or --hf-model, not both.")
    if not args.model and not args.hf_model:
        parser.error("Must specify one of --model or --hf-model.")

    # Device detection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    if args.model == "dllm":
        weights_path = args.weights
        if weights_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(
                repo_root, "02_nano_dllm", "weights", f"nano_dllm_d{args.depth}.pt"
            )
        model_fn, tokenize_fn, mask_token_id, max_seq_len = load_dllm_model(
            args.depth, weights_path, device
        )
        mode = "dllm"
    elif args.model == "block_dllm":
        weights_path = args.weights
        if weights_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(
                repo_root, "03_block_diffusion", "weights",
                f"block_dllm_d{args.depth}_b{args.block_size}.pt",
            )
        model_fn, tokenize_fn, mask_token_id, max_seq_len = load_block_dllm_model(
            args.depth, args.block_size, weights_path, device
        )
        mode = "dllm"
    else:
        model_fn, tokenize_fn, mask_token_id, max_seq_len = load_hf_model(
            args.hf_model, device
        )
        mode = "ar"

    # Run scoring
    results = run_core(
        model_fn, tokenize_fn, device, mode, mask_token_id,
        mc_num=args.mc_num, max_seq_len=max_seq_len,
        max_per_task=args.max_per_task,
    )

    print(f"\nFinal CORE score: {results['__core_score__']:.4f}")


if __name__ == "__main__":
    main()
