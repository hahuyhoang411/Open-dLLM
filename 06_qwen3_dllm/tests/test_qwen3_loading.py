"""Verify real Qwen3-0.6B weights load correctly and produce matching outputs."""

import pytest
import torch

# Mark all tests as slow (require network + ~1.2GB download)
pytestmark = pytest.mark.slow


def _make_cfg():
  """Qwen3-0.6B config for our model."""
  from phase6.config import Config

  return Config(
    n_layer=28,
    n_embd=1024,
    n_head=16,
    n_kv_head=8,
    head_dim=128,
    mlp_hidden=3072,
    vocab_size=151936,
    seq_len=2048,
    block_size=8,
    rope_base=1_000_000,
    rms_eps=1e-6,
    use_emb_norm=False,
    use_gated_query=False,
    use_qk_norm=True,
    use_liger=False,
    use_grad_ckpt=False,
    use_flex=False,
    pad_token_id=151643,
  ).validate()


def _load_our_model_from_hf(hf_model):
  """Load weights from an already-loaded HF model into our model (no API calls)."""
  from phase6.checkpoint import _map_hf_key
  from phase6.model import Model

  cfg = _make_cfg()
  our_model = Model(cfg)
  hf_sd = hf_model.state_dict()
  mapped = {}
  for hf_key, tensor in hf_sd.items():
    our_key, _ = _map_hf_key(hf_key)
    if our_key is not None:
      mapped[our_key] = tensor
  our_model.load_state_dict(mapped, strict=False)
  return our_model


def test_load_qwen3_no_missing_keys():
  """Load Qwen3-0.6B -> our model. No missing keys (except optional extras)."""
  from phase6.checkpoint import load_from_hf
  from phase6.model import Model

  cfg = _make_cfg()
  model = Model(cfg)
  missing, unexpected = load_from_hf(model, 'Qwen/Qwen3-0.6B', device='cpu')

  # No missing keys (all Qwen3 weights should map)
  assert len(missing) == 0, f'Missing keys: {missing}'
  # Unexpected = rotary_emb buffers (we precompute our own) + lm_head (tied) -- these are OK
  for k in unexpected:
    assert 'rotary_emb' in k or k == 'lm_head.weight', f'Truly unexpected key: {k}'


def test_forward_matches_hf():
  """Our model produces same logits as HuggingFace Qwen3 with causal mask."""
  from transformers import AutoModelForCausalLM, AutoTokenizer

  # Load HF model
  hf_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32)
  hf_model.eval()

  # Load our model with same weights (directly from HF state_dict)
  our_model = _load_our_model_from_hf(hf_model)
  our_model.eval()

  # Same input
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
  input_ids = tokenizer.encode('The capital of France is', return_tensors='pt')
  T = input_ids.size(1)

  # HF uses causal mask by default. Our dLLM model uses bidirectional by default,
  # so we pass an explicit causal mask to match HF behavior.
  causal_mask = torch.where(torch.tril(torch.ones(T, T, dtype=torch.bool)), 0.0, float('-inf'))

  with torch.no_grad():
    hf_logits = hf_model(input_ids).logits  # (1, T, V)
    our_logits, _ = our_model(input_ids, attn_mask=causal_mask)  # (1, T, V)

  # Shape check
  assert hf_logits.shape == our_logits.shape, f'Shape mismatch: {hf_logits.shape} vs {our_logits.shape}'

  # Top-1 predictions must match (strongest signal)
  hf_top1 = hf_logits.argmax(dim=-1)
  our_top1 = our_logits.argmax(dim=-1)
  assert torch.equal(hf_top1, our_top1), f'Top-1 predictions differ:\nHF:  {hf_top1}\nOurs: {our_top1}'

  # Numerical closeness (float32 on CPU should be exact or near-exact)
  max_diff = (hf_logits - our_logits).abs().max().item()
  assert max_diff < 0.01, f'Max logit difference: {max_diff} (threshold: 0.01)'
  print(f'\nMax logit difference: {max_diff:.6f} -- MATCH')


def test_embedding_values_match():
  """Spot-check that specific weight values match between HF and our model."""
  from transformers import AutoModelForCausalLM

  hf_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32)
  our_model = _load_our_model_from_hf(hf_model)

  # Embedding weights
  assert torch.equal(hf_model.model.embed_tokens.weight, our_model.token_emb.weight), "Embedding weights don't match"

  # Layer 0 Q projection
  assert torch.equal(hf_model.model.layers[0].self_attn.q_proj.weight, our_model.blocks[0].attn.c_q.weight), (
    "Layer 0 Q proj weights don't match"
  )

  # Last layer MLP gate
  assert torch.equal(hf_model.model.layers[27].mlp.gate_proj.weight, our_model.blocks[27].mlp.gate_proj.weight), (
    "Layer 27 gate_proj weights don't match"
  )

  # Final norm
  assert torch.equal(hf_model.model.norm.weight, our_model.final_norm.weight), "Final norm weights don't match"

  # QK norms
  assert torch.equal(hf_model.model.layers[0].self_attn.q_norm.weight, our_model.blocks[0].attn.q_norm.weight), (
    "Layer 0 QK-norm weights don't match"
  )

  print('\nAll spot-checked weights match exactly')
