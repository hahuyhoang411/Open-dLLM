# CLAUDE.md

## Rules
- **Never run heavy computation locally** — use Kaggle for all training, eval, and GPU work. Local machine is for code editing, quick imports, and tests only.
- Use `uv run python ...` to execute scripts (project uses uv package manager)
- Match existing code style — no docstrings on obvious functions, minimal comments

## Project Structure
- `01_hello_diffusion/` — Phase 1: char-level dLLM on Tiny Shakespeare
- `02_nano_dllm/` — Phase 2: BPE dLLM on FineWeb-Edu (~35M params at depth=6)
- `03_block_diffusion/` — Phase 3: Block diffusion with staircase mask, KV caching
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

## Kaggle
- Push notebooks: `uv run kaggle kernels push -p kaggle/`
- kernel-metadata.json `title` must exactly match `id` slug
- Needs `KAGGLE_API_TOKEN` env var for auth

## Dependencies
- Core: `torch`
- Phase 2: `pip install -e ".[phase2]"` (datasets, tokenizers)
- Phase 3: `pip install -e ".[phase3]"` (datasets, tokenizers)
- Eval: `pip install -e ".[eval]"` (jinja2, pyyaml, transformers)
