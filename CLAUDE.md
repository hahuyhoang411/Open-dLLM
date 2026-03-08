# CLAUDE.md

## Rules
- **Never run heavy computation locally** — use Kaggle for all training, eval, and GPU work. Local machine is for code editing, quick imports, and tests only.
- Use `uv run python ...` to execute scripts (project uses uv package manager)
- Match existing code style — no docstrings on obvious functions, minimal comments

## Project Structure
- `01_hello_diffusion/` — Phase 1: char-level dLLM on Tiny Shakespeare
- `02_nano_dllm/` — Phase 2: BPE dLLM on FineWeb-Edu (~35M params at depth=6)
- `03_block_diffusion/` — Phase 3: Block diffusion with staircase mask, KV caching
- `04_modern_dllm/` — Phase 4: Modern block diffusion (~125M, 20L/768d/12h/3kv/1536MLP, WSD, Muon, multi-source 100B)
- `05_phase5_dllm/` — Phase 5: Modular block diffusion (~144M, 30L/576d/9h/3kv/1536MLP, SmolLM2 tokenizer, linear noise, MuonClip, doc packing, Gated Query Attention)
  - `phase5/` — package: config, model, attention, optim, schedule, loss, data, checkpoint, generate, tokenizer
  - `train.py` — training orchestrator
  - `build_tokenizer.py` — builds hybrid SmolLM2/Qwen3 tokenizer
- `eval/` — DCLM CORE benchmark (22 tasks, centered accuracy scoring)
- `kaggle/` — Kaggle notebooks for training and eval on GPU
- `docs/plans/` — Design docs and implementation plans

## Key Gotchas
- `02_nano_dllm/nano_dllm.py` has module-level `parse_args()` — must patch `sys.argv` before importing
- Eval bundle (`~/.cache/open-dllm/eval_bundle/`): YAML key is `icl_tasks` (list), CSV column is `"Eval Task"`
- Weights live at `02_nano_dllm/weights/nano_dllm_d{depth}.pt`
- MPS (Apple Silicon) fragments memory over many forward passes — call `torch.mps.empty_cache()` between tasks
- `03_block_diffusion/block_dllm.py` has module-level `parse_args()` — same gotcha as Phase 2
- Phase 3 reuses Phase 2's BPE tokenizer (tokenizer.json copied to each dir)
- Weights at `03_block_diffusion/weights/block_dllm_d{depth}_b{block_size}.pt`
- `04_modern_dllm/modern_dllm.py` has module-level `parse_args()` — same gotcha as Phases 2-3
- Phase 4 requires Ampere+ GPU (SM86+) for Liger Kernel + FlexAttention (A10G/A100)
- Phase 4 architecture: 20L/768d/12h/3kv/1536MLP, tied embeddings, ~125M params
- Phase 4 tokenizer: 14 special tokens (IDs 0-13), BPE merges start at ID 14
- Phase 4 tokenizer must be retrained after `train_tokenizer.py` changes (old tokenizer.json incompatible)
- Phase 4 dataset: `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (100B tokens)
- Weights at `04_modern_dllm/weights/modern_dllm_b{block_size}.pt`
- Phase 4 CART weighting is OFF by default (enable with `--cart`). Loss normalization divides by all real tokens (not just masked) — [P4-19]
- Phase 4 expected step-0 loss: ~10-11 (ln(32768)). If higher, check loss normalization and ELBO weighting
- Modal training: `modal run modal_train.py` — uses volumes `dllm-checkpoints` + `dllm-data`, requires `huggingface-secret`
- Phase 5 data: `--data-dir=HoangHa/100BT-dLLM-pretokenized` (HF Hub) or omit for streaming. Modal default is HF Hub, `train.py` default is streaming
- Phase 5 pre-tokenize: `modal run modal_train.py::pretokenize` pushes to HF Hub (cpu=16, datasets.map)
- Phase 5 is a package (`phase5/`) — no module-level parse_args() gotcha (config.py handles it)
- Phase 5 uses self-contained MuonClip (no external muon-optimizer dependency)
- Phase 5 tokenizer: SmolLM2 cosmo2 merges + Qwen3 pre-tokenizer + 14 specials = 49,152 vocab
- Phase 5 noise: LINEAR schedule (mask_prob=t, ELBO weight=1/t — trivially correct)
- Phase 5 doc packing: no right-padding, attention masked at doc boundaries, RoPE resets per-doc
- Phase 5 noise schedule: t ~ U[0.1, 1.0] (default `--t-min 0.1`), ELBO weight capped at 10. Per-block min-1-masked guarantee.
- Phase 5 expected step-0 loss: ~19-20 (ELBO-weighted; raw CE ~9.6 < ln(49152)=10.80). Threshold is 25. Mask tokens produce higher CE at init due to SmolLM2's wider init std
- Phase 5 FP8: `--fp8` enables FP8 matmuls on H100+ (nanochat-style, NOT torchao). All layer dims divisible by 16. Uses `@allow_in_graph` to avoid torchao + grad_ckpt 3x slowdown.
- Phase 5 FP8: lm_head skipped in FP8 conversion (weight accessed directly in loss.py). `disable_fp8()` context manager for generation.
- Weights at `05_phase5_dllm/weights/phase5_dllm_b{block_size}.pt`

## Training Run Monitoring — MANDATORY
When monitoring a training run (Modal, Kaggle, or any GPU), follow these rules. GPU time costs real money — a bad run left unchecked is money burned.

### Step-0 Sanity Check
- Phase 5 expected step-0 loss: ~19-20 (ELBO-weighted; raw CE ~9.6). Phase 4: ~10.4 (ln(32768)).
- If step-0 loss is >25 (Phase 5) or >15 (Phase 4): **STOP immediately.** Loss normalization or weighting is broken.
- If step-0 loss is >40: **STOP immediately.** Something catastrophic (wrong labels, broken masking, CART explosion).

### Loss Trajectory Red Flags — Act Within 2 Check-ins
These patterns mean something is wrong. Do NOT "wait and see." Flag to user with diagnosis and recommend stopping:
1. **Flat loss**: loss doesn't decrease for >200 steps after initial drop → likely weight/schedule bug (e.g., ELBO weight mismatch)
2. **Loss plateau at unexpected value**
3. **Loss spikes >2x current value**: transient spikes (1 step) can be data noise, but repeated spikes → LR too high or numerical instability
4. **Loss increases monotonically for >50 steps**: divergence. Stop immediately.
5. **grad_norm clipped every single step**: inflated loss → inflated gradients → clipping starves learning. Check normalization.
6. **NaN/Inf in loss or grad_norm**: stop immediately, no exceptions

### Mandatory Actions When Anomaly Detected
1. **Flag immediately** — don't wait for more data points to "confirm." Say: "Loss looks anomalous: [specific observation]. Recommend stopping to investigate."
2. **Provide diagnosis** — cross-reference against known bugs in lessons.md and CLAUDE.md (ELBO weight, CART explosion, loss normalization, grad clip starvation)
3. **Default to STOP** — when uncertain whether a pattern is a bug or normal, recommend stopping. A stopped run can be resumed from checkpoint; a wasted run cannot be un-billed.

### Cost Awareness
- Rule of thumb: if loss hasn't improved in 30 min of GPU time. Flag it.

## Kaggle
- Push notebooks: `uv run kaggle kernels push -p kaggle/`
- kernel-metadata.json `title` must exactly match `id` slug
- Needs `KAGGLE_API_TOKEN` env var for auth
- GPU selection: `"machine_shape": "NvidiaTeslaT4"` for 2xT4, default `"enable_gpu": true` gives P100

## Dependencies
- Core: `torch`
- Phase 2: `pip install -e ".[phase2]"` (datasets, tokenizers)
- Phase 3: `pip install -e ".[phase3]"` (datasets, tokenizers)
- Phase 4: `pip install -e ".[phase4]"` (datasets, tokenizers, liger-kernel, muon-optimizer from GitHub)
- Phase 4 training: `modal run modal_train.py` (default: 2×A10G, override: `--gpu A100-40GB:2`)
- Phase 4 DDP: `torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train`
- Phase 5: `pip install -e ".[phase5]"` (datasets, tokenizers, liger-kernel) — no external muon dependency
- Phase 5 training: `modal run modal_train.py` (default: 8×A100-80GB)
- Phase 5 DDP: `torchrun --nproc_per_node=2 05_phase5_dllm/train.py --train`
- Phase 5 Modal training with live dashboard: `modal run modal_train.py --trackio-space HoangHa/open-dllm`
- Trackio dashboard URL: https://huggingface.co/spaces/HoangHa/open-dllm
- Eval: `pip install -e ".[eval]"` (jinja2, pyyaml, transformers)
