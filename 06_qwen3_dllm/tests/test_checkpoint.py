"""Tests for Phase 6 checkpoint — save/load roundtrip + HF weight mapping."""

import os
import random
import tempfile

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# 1. _map_hf_key — embedding, final norm, static mappings
# ---------------------------------------------------------------------------

def test_map_embed_tokens():
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key('model.embed_tokens.weight') == 'token_emb.weight'


def test_map_final_norm():
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key('model.norm.weight') == 'final_norm.weight'


def test_map_lm_head_skipped():
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key('lm_head.weight') is None


# ---------------------------------------------------------------------------
# 2. _map_hf_key — layer keys at various indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('idx', ['0', '15', '27'])
def test_map_layer_attn_projs(idx):
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key(f'model.layers.{idx}.self_attn.q_proj.weight') == f'blocks.{idx}.attn.c_q.weight'
    assert _map_hf_key(f'model.layers.{idx}.self_attn.k_proj.weight') == f'blocks.{idx}.attn.c_k.weight'
    assert _map_hf_key(f'model.layers.{idx}.self_attn.v_proj.weight') == f'blocks.{idx}.attn.c_v.weight'
    assert _map_hf_key(f'model.layers.{idx}.self_attn.o_proj.weight') == f'blocks.{idx}.attn.c_proj.weight'


@pytest.mark.parametrize('idx', ['0', '15', '27'])
def test_map_layer_qk_norms(idx):
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key(f'model.layers.{idx}.self_attn.q_norm.weight') == f'blocks.{idx}.attn.q_norm.weight'
    assert _map_hf_key(f'model.layers.{idx}.self_attn.k_norm.weight') == f'blocks.{idx}.attn.k_norm.weight'


@pytest.mark.parametrize('idx', ['0', '15', '27'])
def test_map_layer_mlp(idx):
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key(f'model.layers.{idx}.mlp.gate_proj.weight') == f'blocks.{idx}.mlp.gate_proj.weight'
    assert _map_hf_key(f'model.layers.{idx}.mlp.up_proj.weight') == f'blocks.{idx}.mlp.up_proj.weight'
    assert _map_hf_key(f'model.layers.{idx}.mlp.down_proj.weight') == f'blocks.{idx}.mlp.down_proj.weight'


@pytest.mark.parametrize('idx', ['0', '15', '27'])
def test_map_layer_norms(idx):
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key(f'model.layers.{idx}.input_layernorm.weight') == f'blocks.{idx}.attn_norm.weight'
    assert _map_hf_key(f'model.layers.{idx}.post_attention_layernorm.weight') == f'blocks.{idx}.mlp_norm.weight'


# ---------------------------------------------------------------------------
# 3. _map_hf_key — unknown keys return None
# ---------------------------------------------------------------------------

def test_map_unknown_key():
    from phase6.checkpoint import _map_hf_key
    assert _map_hf_key('model.layers.0.self_attn.rotary_emb.inv_freq') is None
    assert _map_hf_key('some.random.key') is None


# ---------------------------------------------------------------------------
# 4. save_checkpoint / load_checkpoint roundtrip
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def test_save_load_roundtrip():
    from phase6.checkpoint import save_checkpoint, load_checkpoint

    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Take a step so optimizer has state
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, opt, step=42, loss=1.23, ckpt_dir=tmpdir)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(tmpdir, 'latest.pt'))

        # Load into fresh model
        model2 = _TinyModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        start_step = load_checkpoint(path, model2, opt2, device='cpu')
        assert start_step == 43  # step + 1

        # Weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_load_checkpoint_missing_returns_zero():
    from phase6.checkpoint import load_checkpoint
    model = _TinyModel()
    assert load_checkpoint('/nonexistent/path.pt', model, device='cpu') == 0


def test_load_checkpoint_dir_resolves_latest():
    from phase6.checkpoint import save_checkpoint, load_checkpoint

    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, opt, step=10, loss=0.5, ckpt_dir=tmpdir)
        # Pass directory, should resolve to latest.pt
        model2 = _TinyModel()
        start_step = load_checkpoint(tmpdir, model2, device='cpu')
        assert start_step == 11


# ---------------------------------------------------------------------------
# 5. load_from_hf with mock state_dict (no real downloads)
# ---------------------------------------------------------------------------

class _FakeNorm(nn.Module):
    """1D weight like RMSNorm (not nn.Linear which has 2D weight)."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _FakeSmolDLM(nn.Module):
    """Minimal model mimicking Phase 6 naming for 2 layers."""
    def __init__(self, d=16, n_head=2, n_kv_head=1, head_dim=8, mlp_hidden=32, vocab=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([self._block(d, n_head, n_kv_head, head_dim, mlp_hidden) for _ in range(2)])
        self.final_norm = _FakeNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.token_emb.weight  # tied

    def _block(self, d, n_head, n_kv_head, head_dim, mlp_hidden):
        block = nn.Module()
        # attn sub-module
        attn = nn.Module()
        attn.c_q = nn.Linear(d, n_head * head_dim, bias=False)
        attn.c_k = nn.Linear(d, n_kv_head * head_dim, bias=False)
        attn.c_v = nn.Linear(d, n_kv_head * head_dim, bias=False)
        attn.c_proj = nn.Linear(n_head * head_dim, d, bias=False)
        attn.q_norm = _FakeNorm(head_dim)
        attn.k_norm = _FakeNorm(head_dim)
        block.attn = attn
        block.attn_norm = _FakeNorm(d)
        # mlp sub-module
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(d, mlp_hidden, bias=False)
        mlp.up_proj = nn.Linear(d, mlp_hidden, bias=False)
        mlp.down_proj = nn.Linear(mlp_hidden, d, bias=False)
        block.mlp = mlp
        block.mlp_norm = _FakeNorm(d)
        return block


def _build_fake_hf_state(d=16, n_head=2, n_kv_head=1, head_dim=8, mlp_hidden=32, vocab=64, n_layers=2):
    """Build a state_dict using Qwen3 HF naming convention."""
    sd = {}
    sd['model.embed_tokens.weight'] = torch.randn(vocab, d)
    sd['model.norm.weight'] = torch.randn(d)  # RMSNorm: 1D shape (d,)
    sd['lm_head.weight'] = sd['model.embed_tokens.weight'].clone()  # tied, should be skipped

    for i in range(n_layers):
        pfx = f'model.layers.{i}'
        sd[f'{pfx}.self_attn.q_proj.weight'] = torch.randn(n_head * head_dim, d)
        sd[f'{pfx}.self_attn.k_proj.weight'] = torch.randn(n_kv_head * head_dim, d)
        sd[f'{pfx}.self_attn.v_proj.weight'] = torch.randn(n_kv_head * head_dim, d)
        sd[f'{pfx}.self_attn.o_proj.weight'] = torch.randn(d, n_head * head_dim)
        sd[f'{pfx}.self_attn.q_norm.weight'] = torch.randn(head_dim)
        sd[f'{pfx}.self_attn.k_norm.weight'] = torch.randn(head_dim)
        sd[f'{pfx}.mlp.gate_proj.weight'] = torch.randn(mlp_hidden, d)
        sd[f'{pfx}.mlp.up_proj.weight'] = torch.randn(mlp_hidden, d)
        sd[f'{pfx}.mlp.down_proj.weight'] = torch.randn(d, mlp_hidden)
        sd[f'{pfx}.input_layernorm.weight'] = torch.randn(d)
        sd[f'{pfx}.post_attention_layernorm.weight'] = torch.randn(d)
        # rotary_emb — should be skipped gracefully
        sd[f'{pfx}.self_attn.rotary_emb.inv_freq'] = torch.randn(head_dim // 2)
    return sd


def test_load_from_hf_mock(monkeypatch):
    from phase6.checkpoint import load_from_hf

    fake_sd = _build_fake_hf_state()
    model = _FakeSmolDLM()

    # Mock hf_hub_download + safetensors to return our fake state_dict
    import phase6.checkpoint as ckpt_mod

    def _mock_load_hf_weights(model_name):
        return fake_sd

    monkeypatch.setattr(ckpt_mod, '_load_hf_weights', _mock_load_hf_weights)

    missing, unexpected = load_from_hf(model, model_name='fake/model', device='cpu')

    # lm_head.weight is skipped (tied) → appears in unexpected HF keys
    assert 'lm_head.weight' in unexpected
    # lm_head.weight should NOT be in missing (tied to token_emb, loaded via token_emb)
    assert 'lm_head.weight' not in missing
    # token_emb should have been loaded
    assert torch.allclose(model.token_emb.weight, fake_sd['model.embed_tokens.weight'])
    # Check a layer weight transferred correctly
    assert torch.allclose(
        model.blocks[0].attn.c_q.weight,
        fake_sd['model.layers.0.self_attn.q_proj.weight']
    )
    assert torch.allclose(
        model.blocks[1].mlp.down_proj.weight,
        fake_sd['model.layers.1.mlp.down_proj.weight']
    )


def test_load_from_hf_reports_missing_keys(monkeypatch):
    """Model has keys not in HF (e.g., gated_query, emb_norm) → reported as missing."""
    from phase6.checkpoint import load_from_hf
    import phase6.checkpoint as ckpt_mod

    # Build HF state with only embed + norm (no layer keys)
    sparse_sd = {
        'model.embed_tokens.weight': torch.randn(64, 16),
        'model.norm.weight': torch.randn(16),
    }

    def _mock_load(model_name):
        return sparse_sd

    monkeypatch.setattr(ckpt_mod, '_load_hf_weights', _mock_load)

    model = _FakeSmolDLM()
    missing, unexpected = load_from_hf(model, model_name='fake/model', device='cpu')

    # All block keys should be missing
    assert any('blocks.0.attn.c_q.weight' in k for k in missing)
    assert any('blocks.1.mlp.gate_proj.weight' in k for k in missing)


def test_load_from_hf_skips_rotary_emb(monkeypatch):
    """rotary_emb.inv_freq in HF state should be silently skipped (not error)."""
    from phase6.checkpoint import load_from_hf
    import phase6.checkpoint as ckpt_mod

    fake_sd = _build_fake_hf_state()
    # Confirm rotary_emb keys exist
    assert any('rotary_emb' in k for k in fake_sd)

    def _mock_load(model_name):
        return fake_sd

    monkeypatch.setattr(ckpt_mod, '_load_hf_weights', _mock_load)

    model = _FakeSmolDLM()
    missing, unexpected = load_from_hf(model, model_name='fake/model', device='cpu')

    # rotary_emb keys should be in unexpected (unmapped HF keys)
    assert any('rotary_emb' in k for k in unexpected)
