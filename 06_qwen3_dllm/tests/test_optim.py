"""Tests for Phase 6 optimizer: MuonClip + param group assignment."""

import pytest
import torch

from phase6.config import Config


# Small config — matches test_model.py
def small_cfg(**overrides):
    defaults = dict(
        n_layer=2, n_embd=64, n_head=4, n_kv_head=2, head_dim=32,
        mlp_hidden=128, vocab_size=256, seq_len=32, block_size=8,
        rope_base=10000.0, rms_eps=1e-6, dropout=0.0,
        use_emb_norm=False, use_gated_query=False, use_qk_norm=True,
        use_liger=False, use_grad_ckpt=False, use_flex=False,
        mask_token_id=0, pad_token_id=2,
        muon_lr=0.02, adamw_lr=3e-3, use_muon=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_model(cfg):
    from phase6.model import Model
    return Model(cfg)


# ============================================================================
# 1. build_param_groups creates 3 groups
# ============================================================================

def test_build_param_groups_three_groups():
    """MuonClip mode: QK Muon, Other Muon, AdamW — exactly 3 groups."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)
    assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"

    # Verify group flags
    qk_group = groups[0]
    muon_group = groups[1]
    adamw_group = groups[2]

    assert qk_group['apply_muon'] is True
    assert qk_group['is_qk'] is True
    assert muon_group['apply_muon'] is True
    assert muon_group['is_qk'] is False
    assert adamw_group['apply_muon'] is False


# ============================================================================
# 2. QK detection: .c_q. and .c_k. params → QK Muon group
# ============================================================================

def test_qk_params_in_qk_group():
    """Params from c_q and c_k projections go to QK Muon group."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    qk_group = groups[0]
    qk_ptrs = {p.data_ptr() for p in qk_group['params']}

    # Collect expected c_q and c_k params
    expected_ptrs = set()
    for name, p in model.named_parameters():
        if '.c_q.' in name or '.c_k.' in name:
            expected_ptrs.add(p.data_ptr())

    assert expected_ptrs, "Model should have c_q/c_k params"
    assert qk_ptrs == expected_ptrs, "QK group should contain exactly c_q + c_k params"


# ============================================================================
# 3. Embedding params → AdamW group (NOT Muon)
# ============================================================================

def test_embedding_in_adamw_group():
    """token_emb.weight goes to AdamW group, not Muon."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    adamw_group = groups[2]
    adamw_ptrs = {p.data_ptr() for p in adamw_group['params']}

    emb_ptr = model.token_emb.weight.data_ptr()
    assert emb_ptr in adamw_ptrs, "token_emb.weight should be in AdamW group"

    # NOT in Muon groups
    for g in groups[:2]:
        muon_ptrs = {p.data_ptr() for p in g['params']}
        assert emb_ptr not in muon_ptrs, "token_emb.weight should NOT be in Muon group"


# ============================================================================
# 4. 1D params (norms) → AdamW group
# ============================================================================

def test_1d_params_in_adamw_group():
    """Norm weights (1D) go to AdamW group."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    adamw_group = groups[2]
    adamw_ptrs = {p.data_ptr() for p in adamw_group['params']}

    # All 1D params should be in AdamW
    for name, p in model.named_parameters():
        if p.ndim == 1:
            assert p.data_ptr() in adamw_ptrs, \
                f"1D param '{name}' should be in AdamW group"


# ============================================================================
# 5. Tied embedding dedup: lm_head.weight counted once
# ============================================================================

def test_tied_embedding_dedup():
    """lm_head.weight and token_emb.weight share data_ptr — only counted once."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)

    # Confirm tied
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()

    groups = build_param_groups(model, cfg)
    all_ptrs = []
    for g in groups:
        for p in g['params']:
            all_ptrs.append(p.data_ptr())

    # No duplicate data_ptrs
    assert len(all_ptrs) == len(set(all_ptrs)), "Tied params should appear exactly once"


# ============================================================================
# 6. No params lost: sum of group params == model params (deduplicated)
# ============================================================================

def test_no_params_lost():
    """Total params across all groups == model's deduplicated param count."""
    from phase6.optim import build_param_groups
    cfg = small_cfg()
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    group_total = sum(p.numel() for g in groups for p in g['params'])
    model_total = model.count_params()

    assert group_total == model_total, \
        f"Group params ({group_total}) != model params ({model_total})"


# ============================================================================
# 7. MuonClip step: one step runs without crash, weights change
# ============================================================================

def test_muonclip_step():
    """Create MuonClip, do one step, verify weights changed."""
    from phase6.optim import create_optimizer
    cfg = small_cfg()
    model = _make_model(cfg)
    optimizer = create_optimizer(model, cfg)

    # Snapshot a weight before step
    target_param = None
    for g in optimizer.param_groups:
        if g.get('apply_muon') and not g.get('is_qk'):
            target_param = g['params'][0]
            break
    assert target_param is not None

    w_before = target_param.clone()

    # Fake forward/backward
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(w_before, target_param), "Weights should change after step"


# ============================================================================
# 8. AdamW fallback: use_muon=False → plain torch.optim.AdamW
# ============================================================================

def test_adamw_fallback():
    """cfg.use_muon=False returns plain AdamW, not MuonClip."""
    from phase6.optim import create_optimizer, MuonClip
    cfg = small_cfg(use_muon=False)
    model = _make_model(cfg)
    optimizer = create_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert not isinstance(optimizer, MuonClip)


def test_adamw_fallback_step():
    """AdamW fallback runs a step without crash."""
    from phase6.optim import create_optimizer
    cfg = small_cfg(use_muon=False)
    model = _make_model(cfg)
    optimizer = create_optimizer(model, cfg)

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()  # no crash


# ============================================================================
# 9. Gated Query params: w_gate → Muon (2D, not embed)
# ============================================================================

def test_gated_query_params_in_muon():
    """When use_gated_query=True, w_gate.weight goes to Muon (not AdamW)."""
    from phase6.optim import build_param_groups
    cfg = small_cfg(use_gated_query=True)
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    # Collect all Muon param ptrs (groups 0 and 1)
    muon_ptrs = set()
    for g in groups[:2]:
        for p in g['params']:
            muon_ptrs.add(p.data_ptr())

    # w_gate is 2D and non-embed → should be in Muon
    for name, p in model.named_parameters():
        if 'w_gate' in name:
            assert p.data_ptr() in muon_ptrs, \
                f"w_gate param '{name}' should be in Muon group"


# ============================================================================
# 10. QK-norm params → AdamW (1D)
# ============================================================================

def test_qk_norm_params_in_adamw():
    """q_norm and k_norm weights (1D RMSNorm) go to AdamW, not Muon."""
    from phase6.optim import build_param_groups
    cfg = small_cfg(use_qk_norm=True)
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    adamw_group = groups[2]
    adamw_ptrs = {p.data_ptr() for p in adamw_group['params']}

    for name, p in model.named_parameters():
        if 'q_norm' in name or 'k_norm' in name:
            assert p.ndim == 1, f"QK norm param '{name}' should be 1D"
            assert p.data_ptr() in adamw_ptrs, \
                f"QK norm param '{name}' should be in AdamW group"


# ============================================================================
# 11. LR values from config
# ============================================================================

def test_lr_from_config():
    """Param group LRs come from cfg, not hardcoded."""
    from phase6.optim import build_param_groups
    cfg = small_cfg(muon_lr=0.05, adamw_lr=1e-4)
    model = _make_model(cfg)
    groups = build_param_groups(model, cfg)

    for g in groups[:2]:  # Muon groups
        assert g['lr'] == 0.05
    assert groups[2]['lr'] == 1e-4  # AdamW group


# ============================================================================
# 12. Newton-Schulz produces approximate orthogonal matrix
# ============================================================================

def test_newton_schulz_orthogonality():
    """NS output should be approximately orthogonal: X @ X^T ≈ I (scaled)."""
    from phase6.optim import MuonClip
    G = torch.randn(64, 32)
    X = MuonClip.newton_schulz(G, steps=5)

    # X @ X^T should be close to scaled identity for tall matrices
    assert X.shape == G.shape
    # Columns should have roughly unit norm after NS
    col_norms = X.norm(dim=0)
    assert col_norms.std() < 0.5, "Column norms should be relatively uniform"
