# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
"""Unit tests for exp2 gradient effective rank probe functions.

Tests cover:
1. Default model is NOT 8B (smallest-first policy per AGENTS.md)
2. _get_grad_by_key navigates flattened MLX grad tree by dotted key
3. _build_grad_rank_hook records per-step ranks correctly
4. _compute_cumulative_union_rank stacks and ranks correctly

Run with:
    poetry run pytest tests/experiments/test_continual_learning_exp2_grad_rank.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import helpers directly from the experiment script.
_EXP2_PATH = Path(__file__).parents[2] / "scripts" / "continual_learning"
if str(_EXP2_PATH) not in sys.path:
    sys.path.insert(0, str(_EXP2_PATH))

from exp2_null_space_capacity import (  # noqa: E402
    MODEL_PATH_DEFAULT,
    _build_grad_rank_hook,
    _compute_cumulative_union_rank,
    _get_grad_by_key,
    _rank_eps,
)


def _get_backend():
    try:
        from modelcypher.backends import get_backend
        return get_backend("mlx")
    except Exception:
        pytest.skip("MLX backend not available")


# ---------------------------------------------------------------------------
# Test 1: default model compliance
# ---------------------------------------------------------------------------

def test_default_model_is_not_8b() -> None:
    """MODEL_PATH_DEFAULT must not be an 8B model.

    Per AGENTS.md: 'Do NOT run 8B models for research iteration.'
    """
    assert "8B" not in MODEL_PATH_DEFAULT, (
        f"MODEL_PATH_DEFAULT points to an 8B model: {MODEL_PATH_DEFAULT!r}. "
        "Use a ≤4B model for research iteration (AGENTS.md: smallest-first policy)."
    )


# ---------------------------------------------------------------------------
# Test 2: _get_grad_by_key
# ---------------------------------------------------------------------------

@pytest.mark.mlx
def test_get_grad_by_key_finds_nested_value() -> None:
    """_get_grad_by_key returns the correct tensor for a known dotted key.

    The MLX grad tree is a nested dict (with list entries for layers).
    mlx_tree_flatten flattens to (path_str, value) pairs where list
    indices become positional integers in the path.
    """
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")

    sentinel = mx.array([1.0, 2.0, 3.0])
    # Nested structure mirroring real grad tree: model.layers.0.self_attn.q_proj.A_tilde
    grad_tree = {
        "model": {
            "layers": [
                {
                    "self_attn": {
                        "q_proj": {"A_tilde": sentinel}
                    }
                }
            ]
        }
    }

    result = _get_grad_by_key(grad_tree, "model.layers.0.self_attn.q_proj.A_tilde")
    assert result is not None, "Expected to find 'A_tilde' in grad tree"
    assert result.tolist() == sentinel.tolist()


@pytest.mark.mlx
def test_get_grad_by_key_returns_none_for_missing_key() -> None:
    """_get_grad_by_key returns None when the key is not present."""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")

    grad_tree = {"model": {"layers": [{"self_attn": {"q_proj": {"A_tilde": mx.array([1.0])}}}]}}
    result = _get_grad_by_key(grad_tree, "model.layers.0.self_attn.k_proj.A_tilde")
    assert result is None


# ---------------------------------------------------------------------------
# Test 3: _build_grad_rank_hook records per-step ranks
# ---------------------------------------------------------------------------

@pytest.mark.mlx
def test_build_grad_rank_hook_records_per_step_ranks() -> None:
    """Hook accumulates grad_rank per step and populates step_data in-place.

    Setup:
      Layer: model.layers.0.self_attn.q_proj.weight
      V_tail: [4, 3] — 4-dim input space, 3 tail dims
      G_A:    [1, 4] — rank-1 adapter, one gradient row
      C = G_A @ V_tail: [1, 3] — rank 1 by construction (one row)
      Expected: each step records grad_rank = 1
    """
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")

    backend = _get_backend()
    eps = float(backend.finfo().eps)

    layer_name = "model.layers.0.self_attn.q_proj.weight"

    # V_tail [4, 3]: first 3 standard basis columns → orthonormal
    V_tail = backend.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype="float32")
    tail_bases = {layer_name: (V_tail, 3)}

    # G_A [1, 4]: gradient projects onto all 3 non-zero V_tail columns
    G_A = mx.array([[1.0, 1.0, 1.0, 0.0]])  # [1, 4]
    # C = G_A @ V_tail = [[1, 1, 1]] → rank 1

    grad_tree = {
        "model": {
            "layers": [
                {
                    "self_attn": {
                        "q_proj": {"A_tilde": G_A}
                    }
                }
            ]
        }
    }

    hook, step_data = _build_grad_rank_hook(backend, tail_bases, n_probe_steps=2, eps=eps)

    # Call hook twice — should record both steps
    hook(grad_tree)
    hook(grad_tree)
    # Third call should be a no-op (beyond n_probe_steps)
    hook(grad_tree)

    assert len(step_data) == 2, f"Expected 2 recorded steps, got {len(step_data)}"

    for t, record in enumerate(step_data):
        assert record["step"] == t
        assert layer_name in record["per_layer"]
        layer_rec = record["per_layer"][layer_name]
        assert layer_rec["tail_dims"] == 3
        # C = [[1, 1, 1]] has rank 1
        assert layer_rec["grad_rank"] == 1, (
            f"Step {t}: expected grad_rank=1, got {layer_rec['grad_rank']}"
        )
        # C matrix should be [1, 3]
        assert layer_rec["C"] is not None
        c_shape = tuple(layer_rec["C"].shape)
        assert c_shape == (1, 3), f"Expected C shape (1, 3), got {c_shape}"


# ---------------------------------------------------------------------------
# Test 4: _compute_cumulative_union_rank
# ---------------------------------------------------------------------------

@pytest.mark.mlx
def test_compute_cumulative_union_rank_single_layer() -> None:
    """Cumulative union rank across two orthogonal gradient steps equals 2.

    Setup:
      Layer: layer_a with tail_dims=3
      Step 1: C_1 = [[1, 0, 0]]  → spans e1
      Step 2: C_2 = [[0, 1, 0]]  → spans e2
      Union of {e1, e2} has rank 2.
    """
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")

    backend = _get_backend()
    eps = float(backend.finfo().eps)

    layer_name = "layer_a"
    V_tail = backend.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0]], dtype="float32")
    tail_bases = {layer_name: (V_tail, 3)}

    C1 = mx.array([[1.0, 0.0, 0.0]])  # spans e1
    C2 = mx.array([[0.0, 1.0, 0.0]])  # spans e2

    step_data = [
        {"step": 0, "per_layer": {layer_name: {"tail_dims": 3, "grad_rank": 1, "C": C1}}},
        {"step": 1, "per_layer": {layer_name: {"tail_dims": 3, "grad_rank": 1, "C": C2}}},
    ]

    cumulative = _compute_cumulative_union_rank(backend, step_data, tail_bases, eps)

    assert layer_name in cumulative
    assert cumulative[layer_name] == 2, (
        f"Expected cumulative_union_rank=2, got {cumulative[layer_name]}"
    )
