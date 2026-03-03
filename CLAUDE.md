# CLAUDE.md

## Rules
- **Never run heavy computation locally** — use Kaggle for all training, eval, and GPU work. Local machine is for code editing, quick imports, and tests only.
- Use `uv run python ...` to execute scripts (project uses uv package manager)
- Match existing code style — no docstrings on obvious functions, minimal comments

## Project Structure
- `01_hello_diffusion/` — Phase 1: char-level dLLM on Tiny Shakespeare
- `02_nano_dllm/` — Phase 2: BPE dLLM on FineWeb-Edu (~35M params at depth=6)
- `03_block_diffusion/` — Phase 3: Block diffusion with staircase mask, KV caching
- `04_modern_dllm/` — Phase 4: Modern block diffusion (~213M, 16L/1024d/16h/4kv/2816MLP, WSD, Muon, multi-source 100B)
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
- Phase 4 requires T4 GPU (CC 7.5) for Liger Kernel + FlexAttention
- Phase 4 architecture: 16L/1024d/16h/4kv/2816MLP, tied embeddings, ~213M params
- Phase 4 tokenizer: 14 special tokens (IDs 0-13), BPE merges start at ID 14
- Phase 4 tokenizer must be retrained after `train_tokenizer.py` changes (old tokenizer.json incompatible)
- Phase 4 dataset: `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (100B tokens)
- Weights at `04_modern_dllm/weights/modern_dllm_b{block_size}.pt`
- Phase 4 CART weighting is OFF by default (enable with `--cart`). Loss normalization divides by all real tokens (not just masked) — [P4-19]
- Phase 4 expected step-0 loss: ~10-11 (ln(32768)). If higher, check loss normalization and ELBO weighting

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
- Phase 4 DDP: `torchrun --nproc_per_node=2 04_modern_dllm/modern_dllm.py --train`
- Eval: `pip install -e ".[eval]"` (jinja2, pyyaml, transformers)
