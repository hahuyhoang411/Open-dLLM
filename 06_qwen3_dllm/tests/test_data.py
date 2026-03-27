"""Tests for Phase 6 data pipeline — packing, positions, per-document noise."""

import torch
import pytest
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tiny config for testing (no real dataset, no GPU)
# ---------------------------------------------------------------------------

@dataclass
class _TinyCfg:
    seq_len: int = 16
    block_size: int = 4
    num_blocks: int = 4
    batch_size: int = 2
    mask_token_id: int = 999
    eos_token_id: int = 998
    pad_token_id: int = 997
    t_min: float = 0.1
    use_cart: bool = False
    cart_p: float = 0.1
    data_dir: str = ''
    device: str = 'cpu'
    ddp_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True


# ---------------------------------------------------------------------------
# Mock packer: returns fixed sequences without real data
# ---------------------------------------------------------------------------

class _MockPacker:
    """Returns deterministic packed sequences for testing."""

    def __init__(self, sequences):
        self._seqs = sequences
        self._idx = 0

    def get_sequence(self):
        seq = self._seqs[self._idx % len(self._seqs)]
        self._idx += 1
        return seq


# ---------------------------------------------------------------------------
# Tests: _compute_positions
# ---------------------------------------------------------------------------

def test_compute_positions_basic():
    """doc_ids [0,0,0,1,1,1,1,2] -> positions [0,1,2,0,1,2,3,0]."""
    from phase6.data import _compute_positions

    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2]])
    positions = _compute_positions(doc_ids)
    expected = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0]])
    assert torch.equal(positions, expected)


def test_compute_positions_single_doc():
    """Single document -> 0..L-1."""
    from phase6.data import _compute_positions

    doc_ids = torch.zeros(1, 8, dtype=torch.long)
    positions = _compute_positions(doc_ids)
    expected = torch.arange(8).unsqueeze(0)
    assert torch.equal(positions, expected)


def test_compute_positions_all_different():
    """Every token is a new doc -> all zeros."""
    from phase6.data import _compute_positions

    doc_ids = torch.arange(8).unsqueeze(0)
    positions = _compute_positions(doc_ids)
    expected = torch.zeros(1, 8, dtype=torch.long)
    assert torch.equal(positions, expected)


def test_compute_positions_batch():
    """Batched computation works correctly."""
    from phase6.data import _compute_positions

    doc_ids = torch.tensor([
        [0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 1, 1, 2, 2],
    ])
    positions = _compute_positions(doc_ids)
    expected = torch.tensor([
        [0, 1, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 0, 1, 0, 1],
    ])
    assert torch.equal(positions, expected)


# ---------------------------------------------------------------------------
# Tests: document packing
# ---------------------------------------------------------------------------

def test_document_packing_doc_ids():
    """3 short docs packed into one sequence -> doc_ids track boundaries."""
    # Simulate: doc0=[10,11,12,EOS], doc1=[20,21,EOS], doc2=[30,31,32,33,34,35,36,37,38,39,EOS]
    # Total: 4+3+11 = 18 tokens. seq_len=16 -> first 16 taken.
    # doc0: ids 0-3 (tokens 10,11,12,998), doc1: ids 4-6 (20,21,998),
    # doc2: ids 7-15 (30,31,32,33,34,35,36,37,38)
    cfg = _TinyCfg()
    token_ids = [10, 11, 12, cfg.eos_token_id, 20, 21, cfg.eos_token_id,
                 30, 31, 32, 33, 34, 35, 36, 37, 38]
    doc_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    assert len(token_ids) == cfg.seq_len
    assert len(doc_ids) == cfg.seq_len
    # Verify EOS positions mark boundaries
    for i, tid in enumerate(token_ids):
        if tid == cfg.eos_token_id:
            if i + 1 < len(doc_ids):
                assert doc_ids[i + 1] > doc_ids[i], f"doc_id should increase after EOS at {i}"


# ---------------------------------------------------------------------------
# Tests: get_batch output shapes
# ---------------------------------------------------------------------------

def test_get_batch_shapes():
    """All tensors from get_batch have correct shapes."""
    from phase6.data import get_batch, _apply_noise_per_doc, _compute_positions

    cfg = _TinyCfg()
    # Create mock data: 2 sequences, each seq_len=16
    token_ids_1 = list(range(100, 116))
    doc_ids_1 = [0] * 8 + [1] * 8
    token_ids_2 = list(range(200, 216))
    doc_ids_2 = [0] * 4 + [1] * 4 + [2] * 8

    mock = _MockPacker([(token_ids_1, doc_ids_1), (token_ids_2, doc_ids_2)])

    x_input, targets, mask, elbo_w, doc_ids, positions = get_batch('train', cfg, loader=mock)

    B, L = cfg.batch_size, cfg.seq_len
    assert x_input.shape == (B, 2 * L)
    assert targets.shape == (B, L)
    assert mask.shape == (B, L)
    assert elbo_w.shape == (B, L)
    assert doc_ids.shape == (B, L)
    assert positions.shape == (B, L)


# ---------------------------------------------------------------------------
# Tests: [x_t || x_0] concatenation
# ---------------------------------------------------------------------------

def test_xt_x0_concatenation():
    """x_input[:, :L] has mask tokens, x_input[:, L:] matches targets."""
    from phase6.data import get_batch, _apply_noise_per_doc

    cfg = _TinyCfg()
    token_ids = list(range(100, 116))
    doc_ids = [0] * 8 + [1] * 8
    mock = _MockPacker([(token_ids, doc_ids)] * 2)

    x_input, targets, mask, elbo_w, d_ids, positions = get_batch('train', cfg, loader=mock)

    L = cfg.seq_len
    x_t = x_input[:, :L]
    x_0 = x_input[:, L:]

    # x_0 half matches targets exactly
    assert torch.equal(x_0, targets)

    # Masked positions in x_t have mask_token_id
    assert (x_t[mask] == cfg.mask_token_id).all()

    # Unmasked positions in x_t match targets
    assert (x_t[~mask] == targets[~mask]).all()


# ---------------------------------------------------------------------------
# Tests: per-document t sampling
# ---------------------------------------------------------------------------

def test_per_document_t_same_within_doc():
    """Tokens within the same doc should have the same noise rate t."""
    from phase6.data import _sample_t_per_doc

    # 3 docs: [0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2]
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]])

    torch.manual_seed(42)
    t = _sample_t_per_doc(doc_ids, t_min=0.1)

    # Within doc 0 (positions 0-2): all same t
    assert torch.allclose(t[0, 0:3], t[0, 0].expand(3))
    # Within doc 1 (positions 3-6): all same t
    assert torch.allclose(t[0, 3:7], t[0, 3].expand(4))
    # Within doc 2 (positions 7-15): all same t
    assert torch.allclose(t[0, 7:16], t[0, 7].expand(9))


def test_per_document_t_different_docs():
    """Different docs CAN have different t values (statistical, not guaranteed per sample)."""
    from phase6.data import _sample_t_per_doc

    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

    # Run multiple times — at least once the two docs should differ
    found_different = False
    for seed in range(100):
        torch.manual_seed(seed)
        t = _sample_t_per_doc(doc_ids, t_min=0.1)
        if not torch.allclose(t[0, 0], t[0, 4]):
            found_different = True
            break
    assert found_different, "Per-doc t should vary between documents"


def test_per_document_t_range():
    """All t values in [t_min, 1)."""
    from phase6.data import _sample_t_per_doc

    doc_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]] * 4)
    torch.manual_seed(0)
    t = _sample_t_per_doc(doc_ids, t_min=0.1)
    assert t.min() >= 0.1
    assert t.max() < 1.0


# ---------------------------------------------------------------------------
# Tests: noise mask
# ---------------------------------------------------------------------------

def test_noise_mask_matches_mask_token():
    """Masked positions in x_noisy correspond to mask_token_id."""
    from phase6.data import _apply_noise_per_doc

    cfg = _TinyCfg()
    targets = torch.randint(10, 100, (2, 16))
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]] * 2)

    torch.manual_seed(42)
    x_noisy, mask, t = _apply_noise_per_doc(targets, doc_ids, cfg)

    assert (x_noisy[mask] == cfg.mask_token_id).all()
    assert (x_noisy[~mask] == targets[~mask]).all()


# ---------------------------------------------------------------------------
# Tests: ELBO weights
# ---------------------------------------------------------------------------

def test_elbo_weights_correct():
    """w = 1/t for masked positions, within correct range."""
    from phase6.data import _sample_t_per_doc
    from phase6.schedule import compute_elbo_weight

    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    torch.manual_seed(42)
    t = _sample_t_per_doc(doc_ids, t_min=0.1)
    w = compute_elbo_weight(t, t_min=0.1)

    # w = 1/t, t in [0.1, 1) -> w in (1, 10]
    assert w.min() >= 1.0
    assert w.max() <= 10.0 + 1e-6

    # Verify w = 1/t.clamp(min=0.1)
    expected = 1.0 / t.clamp(min=0.1)
    assert torch.allclose(w, expected)


# ---------------------------------------------------------------------------
# Tests: padding handling
# ---------------------------------------------------------------------------

def test_padding_not_masked():
    """Pad tokens should never be masked."""
    from phase6.data import _apply_noise_per_doc

    cfg = _TinyCfg()
    targets = torch.randint(10, 100, (1, 16))
    targets[0, 12:] = cfg.pad_token_id  # last 4 are padding
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    torch.manual_seed(42)
    x_noisy, mask, t = _apply_noise_per_doc(targets, doc_ids, cfg)

    assert not mask[0, 12:].any(), "padding positions should never be masked"


# ---------------------------------------------------------------------------
# Tests: position reset at doc boundaries
# ---------------------------------------------------------------------------

def test_position_reset_at_boundaries():
    """Positions reset to 0 at each new document."""
    from phase6.data import _compute_positions

    # 2 docs in a sequence
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    positions = _compute_positions(doc_ids)

    # First doc: 0,1,2,3,4
    assert torch.equal(positions[0, :5], torch.arange(5))
    # Second doc: resets to 0,1,2,...
    assert torch.equal(positions[0, 5:], torch.arange(11))


def test_position_reset_in_batch():
    """Full get_batch returns positions with doc boundary resets."""
    from phase6.data import get_batch

    cfg = _TinyCfg()
    token_ids = list(range(100, 116))
    doc_ids_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    mock = _MockPacker([(token_ids, doc_ids_list)] * 2)
    _, _, _, _, doc_ids, positions = get_batch('train', cfg, loader=mock)

    # Verify reset at boundary (position 5 is doc 1, should be 0)
    assert positions[0, 5] == 0
    assert positions[0, 0] == 0
    assert positions[0, 4] == 4
