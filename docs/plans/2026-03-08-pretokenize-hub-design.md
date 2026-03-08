# Pre-tokenize 100B Tokens → HF Hub

**Date:** 2026-03-08
**Status:** Approved

## Goal

Pre-tokenize the full 100B-token training dataset and store on HuggingFace Hub as `HoangHa/100BT-dLLM-pretokenized` (public). Eliminates the CPU bottleneck of on-the-fly HF streaming + tokenization during training. Flexible: any seq_len at train time without re-processing.

## Why

- 1500-step Phase 5 run hit ~190K tok/s on 4×H100 (~50% GPU utilization). Bottleneck: HF streaming + tokenization.
- Quokka scaling law: 144M dLLMs need ~14-17B tokens (5x AR Chinchilla). We want 100B unique tokens for multi-epoch training.
- Pre-tokenized data eliminates encoding overhead entirely. Training just reads int arrays and packs.

## Design

### Pipeline (runs on Modal, 16-core CPU)

1. `load_dataset(non-streaming)` — download source Parquets to Modal `/data` volume (~30 min, cached)
2. `ds.map(tokenize, batched=True, batch_size=10000, num_proc=16)` — parallel tokenization (~1-2h)
3. `push_to_hub("HoangHa/100BT-dLLM-pretokenized")` — upload to Hub (~30-60 min)
4. Delete intermediates from Modal volume (cleanup)

Total: ~3 hours.

### Hub Dataset Format

- **Repo:** `HoangHa/100BT-dLLM-pretokenized` (public)
- **Rows:** ~500M (one per document, variable length)
- **Schema:** `{"input_ids": Sequence(Value("uint16"))}` — token IDs with trailing EOS per doc
- **Storage:** ~120-140 GB (Parquet compressed from ~200 GB raw uint16)
- **No packing** — documents stored at original length. Packing into seq_len happens at train time.
- **Tokenizer:** SmolLM2 cosmo2 merges + Qwen3 pre-tokenizer + 14 specials = 49,152 vocab

### Tokenization Function

```python
def tokenize(batch):
    encoded = tok.encode_batch(batch["text"])
    return {"input_ids": [enc.ids + [EOS_ID] for enc in encoded]}

tokenized = ds.map(
    tokenize, batched=True, batch_size=10000,
    num_proc=16, remove_columns=ds.column_names,
)
```

### Training Integration

New `_PreTokenizedPacker` in `data.py`:
- Loads HF dataset with `load_dataset(repo_id, cache_dir="/data")` — first run downloads ~140 GB, subsequent runs use Arrow cache
- Same interface as `_DocumentPacker` but reads pre-tokenized `input_ids` instead of calling `encode()`
- Packs into any `seq_len` at train time (2048, 4096, etc.)
- Doc boundaries reconstructed from EOS tokens
- DDP: each rank reads different shard via `ds.shard(num_shards, index)`

### Modal Infrastructure

- `pretokenize()` function: CPU-only (no GPU), 16 cores, 86400s timeout
- Uses `dllm-data` volume for caching source dataset + intermediates
- Requires `huggingface-secret` for Hub upload (write token)
- Auto-detect in training: if HF dataset exists, use `_PreTokenizedPacker`

## Storage

| Location | Size | Duration |
|---|---|---|
| Modal volume (temp) | ~1 TB peak | During processing only |
| HF Hub (permanent) | ~120-140 GB | Permanent, free |
| Training cache on Modal | ~140 GB | Downloaded once per training infra |

## Flexibility

- **Change seq_len?** No re-processing. Pack at train time.
- **Change tokenizer?** Must re-tokenize (push new dataset).
- **Change dataset?** Must re-tokenize (push new dataset).

## Decisions

- **Variable-length docs over packed sequences**: flexibility for seq_len changes outweighs the minor packing overhead at train time
- **uint16 over int32**: vocab 49,152 fits in uint16 (max 65,535), halves storage
- **Public dataset**: enables community reuse for dLLM research
- **`map(num_proc=16)` over streaming**: ~4-8x speedup from process-level parallelism, worth the temp disk cost
