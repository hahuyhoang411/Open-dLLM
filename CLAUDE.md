# CLAUDE.md

## Rules
- **Never run heavy computation locally** — use Modal/Kaggle for training, eval, GPU work
- Use `uv run python ...` to execute scripts (uv package manager)
- Match existing code style — no docstrings on obvious functions, minimal comments
- Phase 5 is the active development target: `05_optimized_dllm/`

## Project Structure
- `01_hello_diffusion/` — Phase 1: char-level dLLM (~10M)
- `02_nano_dllm/` — Phase 2: BPE dLLM (~26M)
- `03_block_diffusion/` — Phase 3: block diffusion (~36M)
- `04_modern_dllm/` — Phase 4: modern dLLM (~125M)
- `05_optimized_dllm/` — Phase 5: production dLLM (~144M, 30L/576d/9h/3kv)
  - `phase5/` — modular package (config, model, attention, optim, schedule, loss, data, checkpoint, generate, tokenizer, fp8)
  - `train.py` — training orchestrator
  - `scripts/modal_train.py` — Modal cloud training
  - `scripts/vram_probe.py` — VRAM benchmarking
- `eval/` — DCLM CORE benchmark (22 tasks)
- `kaggle/` — training notebooks
- See each phase's README.md for details, BUGS.md for known issues

## Commands
- `modal run 05_optimized_dllm/scripts/modal_train.py` — train on 8×H100
- `modal run 05_optimized_dllm/scripts/modal_train.py::status` — check training status
- `modal run 05_optimized_dllm/scripts/modal_train.py::score` — eval checkpoint
- `torchrun --nproc_per_node=N 05_optimized_dllm/train.py --train --fp8` — local DDP
- `uv run kaggle kernels push -p kaggle/` — push Kaggle notebooks
- Deps: `pip install -e ".[phase5]"`, `pip install -e ".[eval]"`, `pip install -e ".[cloud]"`

## Autonomous Training Monitor — MANDATORY

You are the training run watchdog. GPU time = real money. Act decisively.

### How to Read Logs
1. **Modal dashboard:** `modal run 05_optimized_dllm/scripts/modal_train.py::status` — shows step, loss per run
2. **Trackio dashboard:** https://huggingface.co/spaces/HoangHa/open-dllm — live loss/grad/throughput curves
3. **Checkpoint inspection:** `modal volume get dllm-checkpoints phase5/<run_id>/latest.pt` — download and inspect

### Step-0 Sanity (first log line)
| Phase | Expected step-0 loss | STOP threshold |
|-------|---------------------|----------------|
| Phase 4 | ~10.4 (ln(32768)) | >15 |
| Phase 5 | ~19-20 (ELBO-weighted) | >25 |
| Any phase | — | >40 (catastrophic) |

Action: If step-0 exceeds threshold → flag immediately, recommend stopping, diagnose (ELBO weight? loss normalization? broken masking?).

### Red Flags — Act Within 2 Check-ins
| Pattern | Likely Cause | Action |
|---------|-------------|--------|
| Loss flat >200 steps after initial drop | ELBO weight bug, schedule mismatch | STOP + diagnose |
| Loss spikes >2x repeatedly | LR too high, numerical instability | STOP + reduce LR |
| Loss increases >50 steps | Divergence | STOP immediately |
| grad_norm=0.000000 | Broken backward (Liger FLCE bug) | STOP immediately |
| grad_norm clipped every step | Loss normalization inflating gradients | Flag + investigate |
| NaN/Inf in loss or grad_norm | — | STOP, no exceptions |
| Loss hasn't improved in 30 min GPU time | Plateau or bug | Flag it |

### Decision Protocol
1. **Flag immediately** — don't wait for more data. Say: "Loss anomaly: [observation]. Recommend stopping."
2. **Diagnose** — cross-reference BUGS.md known issues (ELBO weight, CART explosion, Liger FLCE, loss normalization)
3. **Default to STOP** — a stopped run resumes from checkpoint. A wasted run cannot be un-billed.
4. **If healthy** — report: step, loss, grad_norm, throughput, VRAM, estimated completion time

### Known Healthy Signatures
- Phase 5 Run 1: step-0=19.11 → step-1500=3.36 (val). Monotonic decrease. grad_norm 9.5→0.05. 189K tok/s, 57.9/85 GB VRAM.
- Loss should decrease sharply in first 200 steps, then steady decline
- Occasional 1-step spikes (<2x) are normal data noise — only flag repeated spikes

### Proactive Actions
- When user shares logs or asks to check a run: parse the numbers, compare against known healthy signatures, flag deviations
- When a run finishes: suggest next steps (eval, longer training, hyperparameter changes)
- When starting a new run: verify config makes sense (batch size × GPUs × seq_len = expected tokens/step)
- Track cost: H100 ~$3.95/hr/GPU. Flag if spend exceeds expected budget.

## Phase 6 Target
- Goal: match SmolLM2-135M on standard benchmarks (HellaSwag 42%, PIQA 68%, ARC 44%)
- Current: 1500 steps, ~1.57B tokens. Need 15B+ tokens (Quokka: dLLMs need ~5x AR data at 144M)
- Roadmap: extended pretraining → eval infra → SFT → DPO/GRPO
- See `docs/plans/phase6-roadmap.md` for full plan
