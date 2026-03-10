"""
modal_train.py — Modal training infrastructure for Phase 5 of SmolDLM.

Usage:
    # Launch training (default: 8×A100-80GB)
    modal run modal_train.py

    # Override GPU type
    modal run modal_train.py --gpu H100:8
    modal run modal_train.py --gpu A10G       # single GPU (auto-detects, skips DDP)

    # With trackio dashboard
    modal run modal_train.py --trackio-space HoangHa/smoldlm

    # Custom architecture
    modal run modal_train.py --extra-args "--n-layer 12 --n-embd 576 --dropout 0.0"

    # Check training status
    modal run modal_train.py::status

    # Resume a specific run
    modal run modal_train.py --run-id 20260307_212400

    # Download checkpoint from a run
    modal volume get smoldlm-checkpoints phase5/20260307_212400/latest.pt ./latest.pt

    # Data sources (--data-dir flag):
    modal run modal_train.py                                              # HoangHa/100BT-dLLM-pretokenized (default, from HF Hub)
    modal run modal_train.py --data-dir streaming                         # HuggingFaceFW/finepdfs_50BT-... on-the-fly
    modal run modal_train.py --data-dir /data/tmp_shards                  # local Modal volume (after pretokenize)
    modal run modal_train.py --data-dir HoangHa/other-dataset             # any other HF dataset

    # Pre-tokenize dataset and push to HF Hub
    modal run modal_train.py::pretokenize
    modal run modal_train.py::pretokenize --max-docs 100000

    # Resume/upload from existing /data/tmp_shards only (skip tokenization)
    modal run modal_train.py::upload_from_shards
    modal run modal_train.py::upload_from_shards --num-proc 8 --max-shard-size 8GB
"""

import modal

app = modal.App('smoldlm-phase5')

image = (
  modal.Image
  .debian_slim(python_version='3.11')
  .pip_install(
    'torch>=2.5.0',
    'datasets>=2.0.0',
    'tokenizers>=0.15.0',
    'liger-kernel>=0.4.0',
    'trackio>=0.18.0',
    'jinja2>=3.0.0',
    'pyyaml>=6.0',
  )
  .add_local_dir('05_optimized_dllm', '/root/05_optimized_dllm')
  .add_local_dir('eval', '/root/eval')
)

ckpt_vol = modal.Volume.from_name('smoldlm-checkpoints', create_if_missing=True)
data_vol = modal.Volume.from_name('smoldlm-data', create_if_missing=True)


@app.cls(
  image=image,
  gpu='H100:8',
  timeout=86400,
  retries=modal.Retries(max_retries=6, initial_delay=0.0),
  volumes={'/checkpoints': ckpt_vol, '/data': data_vol},
  secrets=[modal.Secret.from_name('huggingface-secret')],
)
class Train:
  @modal.method()
  def run(
    self,
    trackio_space: str = '',
    extra_args: str = '',
    run_id: str = '',
    data_dir: str = 'HoangHa/100BT-dLLM-pretokenized',
  ):
    import os
    import subprocess
    import torch

    os.environ['HF_HOME'] = '/data'
    os.environ['HF_DATASETS_CACHE'] = '/data/datasets'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    ckpt_dir = f'/checkpoints/phase5/{run_id}' if run_id else '/checkpoints/phase5'

    n_gpus = torch.cuda.device_count()
    launcher = ['torchrun', '--standalone', f'--nproc_per_node={n_gpus}'] if n_gpus > 1 else ['python']

    cmd = [
      *launcher,
      '/root/05_optimized_dllm/train.py',
      '--train',
      '--fp8',
      f'--ckpt-dir={ckpt_dir}',
      f'--resume={ckpt_dir}',
    ]

    # Data source: "streaming" = on-the-fly, anything else = --data-dir value
    if data_dir and data_dir != 'streaming':
      cmd.append(f'--data-dir={data_dir}')

    if trackio_space:
      cmd.append(f'--trackio-space={trackio_space}')

    if extra_args:
      cmd.extend(extra_args.split())

    print(f'Running: {" ".join(cmd)}')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout is None:
      raise RuntimeError('Failed to capture training logs')
    for line in proc.stdout:
      print(line, end='')
    proc.wait()

    ckpt_vol.commit()
    data_vol.commit()

    if proc.returncode != 0:
      raise subprocess.CalledProcessError(proc.returncode, cmd)


@app.function(
  image=modal.Image.debian_slim(python_version='3.11').pip_install('torch>=2.5.0'),
  volumes={'/checkpoints': ckpt_vol},
)
def status():
  import os
  import torch

  ckpt_vol.reload()
  base = '/checkpoints/phase5'
  if not os.path.exists(base):
    print('No phase5 checkpoints found.')
    return

  for run_dir in sorted(os.listdir(base)):
    run_path = os.path.join(base, run_dir)
    if not os.path.isdir(run_path):
      continue
    latest = os.path.join(run_path, 'latest.pt')
    if os.path.exists(latest):
      ckpt = torch.load(latest, map_location='cpu', weights_only=False)
      print(f'Run {run_dir}: step {ckpt.get("step", "?")}, loss {ckpt.get("loss", "?"):.4f}')
    files = sorted(os.listdir(run_path))
    for f in files:
      path = os.path.join(run_path, f)
      size_mb = os.path.getsize(path) / (1024 * 1024)
      print(f'  {f:40s} {size_mb:8.1f} MB')


@app.function(
  image=image,
  gpu='A10G',
  timeout=600,
  volumes={'/checkpoints': ckpt_vol, '/data': data_vol},
  secrets=[modal.Secret.from_name('huggingface-secret')],
)
def debug(run_id: str = '', ckpt_name: str = ''):
  import os
  import subprocess

  os.environ['HF_HOME'] = '/data'
  os.environ['HF_DATASETS_CACHE'] = '/data/datasets'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'

  ckpt_vol.reload()
  base = f'/checkpoints/phase5/{run_id}' if run_id else '/checkpoints/phase5'

  # Find checkpoint
  if ckpt_name:
    ckpt_path = os.path.join(base, ckpt_name)
  else:
    # Find latest checkpoint
    candidates = sorted(f for f in os.listdir(base) if f.startswith('ckpt_'))
    if candidates:
      ckpt_path = os.path.join(base, candidates[-1])
    else:
      print(f'No checkpoints found in {base}')
      return
  print(f'Using checkpoint: {ckpt_path}')

  cmd = [
    'python',
    '/root/05_optimized_dllm/debug_generate.py',
    f'--ckpt={ckpt_path}',
    '--skip-data',
  ]
  proc = subprocess.run(cmd, capture_output=False)
  if proc.returncode != 0:
    print(f'Debug script failed with code {proc.returncode}')


@app.function(
  image=image,
  gpu='A10G',
  timeout=7200,
  volumes={'/checkpoints': ckpt_vol},
)
def score(
  run_id: str = '20260309_221412',
  step: int = 0,
  mc_num: int = 128,
  mc_batch_size: int = 64,
  max_per_task: int = -1,
):
  import os
  import sys
  import torch

  ckpt_vol.reload()

  base = f'/checkpoints/phase5/{run_id}'
  if step > 0:
    ckpt_path = os.path.join(base, f'ckpt_{step:06d}.pt')
  else:
    ckpt_path = os.path.join(base, 'latest.pt')
  if not os.path.exists(ckpt_path):
    print(f'Checkpoint not found: {ckpt_path}')
    available = sorted(f for f in os.listdir(base) if f.startswith('ckpt_')) if os.path.isdir(base) else []
    print(f'Available: {available}')
    return

  sys.path.insert(0, '/root/05_optimized_dllm')
  sys.path.insert(0, '/root/eval')

  from phase5.model import Model
  from phase5.tokenizer import encode

  device = 'cuda'
  model = Model().to(device)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
  model.load_state_dict(state_dict)
  model.requires_grad_(False)
  model.eval()

  ckpt_step = ckpt.get('step', step)
  ckpt_loss = ckpt.get('loss', 0.0)
  param_count = sum(p.numel() for p in model.parameters())
  print(f'Loaded: step={ckpt_step}, loss={ckpt_loss:.4f}, params={param_count:,}')

  def model_fn(input_ids):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      logits, _ = model(input_ids)
    return logits

  def tokenize_fn(text):
    return encode(text)

  from base_eval import run_core

  results = run_core(
    model_fn, tokenize_fn, device, mode='dllm', mask_token_id=0,
    mc_num=mc_num, max_seq_len=2048, max_per_task=max_per_task,
    pad_token_id=2, mc_batch_size=mc_batch_size,
  )
  print(f'\n=== Step {ckpt_step} | CORE score: {results["__core_score__"]:.4f} ===')


@app.function(
  image=image,
  cpu=16,
  memory=64 * 1024,
  timeout=86400,
  volumes={'/data': data_vol},
  secrets=[modal.Secret.from_name('huggingface-secret')],
)
def pretokenize(
  max_docs: int = 0,
  hub_repo: str = 'HoangHa/100BT-dLLM-pretokenized',
  num_proc: int = 16,
  max_shard_size: str = '2GB',
):
  import os
  import subprocess

  os.environ['HF_HOME'] = '/data'
  os.environ['HF_DATASETS_CACHE'] = '/data/datasets'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'

  cmd = [
    'python',
    '/root/05_optimized_dllm/pretokenize.py',
    '--mode=full',
    f'--hub-repo={hub_repo}',
    f'--num-proc={num_proc}',
    f'--max-shard-size={max_shard_size}',
    '--work-dir=/data/tmp_shards',
  ]
  if max_docs > 0:
    cmd.extend(['--max-docs', str(max_docs)])

  print(f'Running: {" ".join(cmd)}')
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  if proc.stdout is None:
    raise RuntimeError('Failed to capture pretokenize logs')
  for line in proc.stdout:
    print(line, end='')
  proc.wait()

  data_vol.commit()

  if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, cmd)

  # Cleanup: remove intermediate caches to free volume space
  import shutil

  cache_dir = '/data/datasets'
  for item in os.listdir(cache_dir):
    path = os.path.join(cache_dir, item)
    if os.path.isdir(path) and 'finepdfs' in item:
      print(f'Cleaning up source cache: {path}')
      shutil.rmtree(path)
  if os.path.isdir('/data/tmp_shards'):
    print('Cleaning up tokenized shards work dir')
    shutil.rmtree('/data/tmp_shards')
  data_vol.commit()


@app.function(
  image=image,
  cpu=32,
  memory=128 * 1024,
  timeout=86400,
  volumes={'/data': data_vol},
  secrets=[modal.Secret.from_name('huggingface-secret')],
)
def upload_from_shards(
  hub_repo: str = 'HoangHa/100BT-dLLM-pretokenized',
  work_dir: str = '/data/tmp_shards',
  num_proc: int = 8,
  max_shard_size: str = '8GB',
  per_shard: bool = True,
  cleanup_after_upload: bool = False,
):
  import os
  import shutil
  import subprocess

  os.environ['HF_HOME'] = '/data'
  os.environ['HF_DATASETS_CACHE'] = '/data/datasets'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'

  mode = 'upload-splits' if per_shard else 'upload'
  cmd = [
    'python',
    '/root/05_optimized_dllm/pretokenize.py',
    f'--mode={mode}',
    f'--hub-repo={hub_repo}',
    f'--num-proc={num_proc}',
    f'--max-shard-size={max_shard_size}',
    f'--work-dir={work_dir}',
  ]

  print(f'Running: {" ".join(cmd)}')
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  if proc.stdout is None:
    raise RuntimeError('Failed to capture upload logs')
  for line in proc.stdout:
    print(line, end='')
  proc.wait()

  data_vol.commit()

  if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, cmd)

  if cleanup_after_upload and os.path.isdir(work_dir):
    print(f'Cleaning up upload source work dir: {work_dir}')
    shutil.rmtree(work_dir)
    data_vol.commit()


@app.function(
  image=image,
  gpu='H100',
  timeout=3600,
)
def vram_probe():
  import subprocess

  proc = subprocess.run(
    ['python', '/root/05_optimized_dllm/vram_probe.py'],
    capture_output=False,
  )
  if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, 'vram_probe')


@app.function(
  image=modal.Image.debian_slim(python_version='3.11').pip_install(
    'torch>=2.5.0', 'numpy', 'safetensors>=0.4.0', 'huggingface_hub>=0.20.0',
  ),
  volumes={'/checkpoints': ckpt_vol},
  secrets=[modal.Secret.from_name('huggingface-secret')],
  timeout=7200,
)
def push_to_hub(
  run_id: str = '20260309_221412',
  hub_repo: str = 'HoangHa/smoldlm-144m',
  every_n: int = 5000,
  batch_size: int = 5,
):
  """Push model checkpoints to HuggingFace Hub (model weights only, no optimizer).

  Batches uploads via upload_folder to avoid HF rate limits.
  Skips checkpoints already on Hub. Retries on 429.
  """
  import json
  import os
  import shutil
  import time

  import torch
  from huggingface_hub import HfApi
  from safetensors.torch import save_file

  ckpt_vol.reload()
  base = f'/checkpoints/phase5/{run_id}'
  if not os.path.exists(base):
    print(f'Run dir not found: {base}')
    return

  # Model config
  config = {
    'model_type': 'smoldlm',
    'architecture': 'block_diffusion_lm',
    'n_layer': 30,
    'n_embd': 576,
    'n_head': 9,
    'n_kv_head': 3,
    'mlp_hidden': 1536,
    'vocab_size': 49152,
    'seq_len': 2048,
    'block_size': 32,
    'num_blocks': 64,
    'tied_embeddings': True,  # lm_head.weight = token_emb.weight (omitted from safetensors)
    'attention': 'gated_query_attention',
    'noise_schedule': 'linear',
    'optimizer': 'muonclip_adamw',
    'total_params': '144.47M',
    'tokenizer': 'smoldlm',
    'mask_token_id': 0,
    'eos_token_id': 1,
    'pad_token_id': 2,
  }

  api = HfApi()
  api.create_repo(hub_repo, exist_ok=True)

  # Check which files already exist on Hub
  existing = set(api.list_repo_files(hub_repo))
  print(f'Found {len(existing)} files already on Hub')

  # Upload config (only if missing)
  if 'config.json' not in existing:
    config_path = '/tmp/config.json'
    with open(config_path, 'w') as f:
      json.dump(config, f, indent=2)
    api.upload_file(path_or_fileobj=config_path, path_in_repo='config.json', repo_id=hub_repo)
    print(f'Uploaded config.json to {hub_repo}')

  # Find checkpoints to upload, skipping existing
  ckpt_files = sorted(f for f in os.listdir(base) if f.startswith('ckpt_') and f.endswith('.pt'))
  selected = []
  for f in ckpt_files:
    step = int(f.split('_')[1].split('.')[0])
    if step % every_n == 0 or f == ckpt_files[-1]:
      repo_path = f'checkpoints/model_step_{step:06d}.safetensors'
      if repo_path not in existing:
        selected.append((step, f))

  print(f'Total checkpoints: {len(ckpt_files)}, to upload: {len(selected)} (skipped {len(ckpt_files) - len(selected)} existing)')
  if not selected:
    print('Nothing to upload!')

  # Process in batches with retry
  staging = '/tmp/hub_staging'
  for batch_start in range(0, len(selected), batch_size):
    batch = selected[batch_start:batch_start + batch_size]
    if os.path.exists(staging):
      shutil.rmtree(staging)
    os.makedirs(f'{staging}/checkpoints', exist_ok=True)

    for step, fname in batch:
      ckpt_path = os.path.join(base, fname)
      print(f'  Converting {fname}...')
      ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
      sd = {k: v for k, v in ckpt['model_state_dict'].items() if k != 'lm_head.weight'}
      ckpt_step = ckpt.get('step', step)
      train_loss = ckpt.get('loss', 0.0)
      save_file(sd, f'{staging}/checkpoints/model_step_{ckpt_step:06d}.safetensors')
      with open(f'{staging}/checkpoints/model_step_{ckpt_step:06d}.json', 'w') as f:
        json.dump({'step': ckpt_step, 'loss': float(train_loss)}, f)
      del ckpt, sd

    steps = [s for s, _ in batch]
    msg = f'Add checkpoints: steps {steps[0]}-{steps[-1]}'
    print(f'Uploading batch: {msg} ({len(batch)} checkpoints)...')

    for attempt in range(5):
      try:
        api.upload_folder(folder_path=staging, repo_id=hub_repo, commit_message=msg)
        print(f'  Batch uploaded ({len(batch)} checkpoints)')
        break
      except Exception as e:
        if '429' in str(e) and attempt < 4:
          wait = 300 * (attempt + 1)  # 5, 10, 15, 20 min
          print(f'  Rate limited, waiting {wait}s (attempt {attempt + 1}/5)...')
          time.sleep(wait)
        else:
          raise
    shutil.rmtree(staging)

  # Upload latest as top-level model.safetensors
  if 'model.safetensors' not in existing:
    latest_path = os.path.join(base, 'latest.pt')
    if os.path.exists(latest_path):
      ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
      latest_sd = {k: v for k, v in ckpt['model_state_dict'].items() if k != 'lm_head.weight'}
      save_file(latest_sd, '/tmp/model.safetensors')
      api.upload_file(path_or_fileobj='/tmp/model.safetensors', path_in_repo='model.safetensors', repo_id=hub_repo)
      print(f'Uploaded model.safetensors (latest, step {ckpt.get("step", "?")})')

  print(f'\nDone! https://huggingface.co/{hub_repo}')


@app.function(
  image=modal.Image.debian_slim(python_version='3.11'),
  volumes={'/checkpoints': ckpt_vol},
)
def download():
  print('Use the Modal CLI to download checkpoints:')
  print('  modal volume get smoldlm-checkpoints latest.pt ./latest.pt')


@app.local_entrypoint()
def main(
  gpu: str = 'H100:8',
  trackio_space: str = '',
  extra_args: str = '',
  run_id: str = '',
  data_dir: str = 'HoangHa/100BT-dLLM-pretokenized',
):
  if not run_id:
    import datetime

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  print(f'Run ID: {run_id}  (checkpoints at /checkpoints/phase5/{run_id}/)')
  train_with_options = getattr(Train, 'with_options')
  train_with_options(gpu=gpu)().run.remote(
    trackio_space=trackio_space,
    extra_args=extra_args,
    run_id=run_id,
    data_dir=data_dir,
  )
