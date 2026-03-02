# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
"""Unit tests for exp1 continual learning metric functions.

Focuses on _cumulative_utilization_by_layer: verifies that
rank([C_1..C_T]) / tail_dims is computed correctly and that
per-task overlap does NOT inflate the cumulative count.

Run with:
    poetry run pytest tests/experiments/test_continual_learning_exp1_metrics.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Import the helpers directly from the experiment script.
# The script is not a package, so we add its parent to sys.path.
_EXP1_PATH = Path(__file__).parents[2] / "scripts" / "continual_learning"
if str(_EXP1_PATH) not in sys.path:
    sys.path.insert(0, str(_EXP1_PATH))

from exp1_sequential_forgetting import (  # noqa: E402
    _cumulative_utilization_by_layer,
    _rank_eps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_backend():
    try:
        from modelcypher.backends import get_backend
        return get_backend("mlx")
    except Exception:
        pytest.skip("MLX backend not available")


# ---------------------------------------------------------------------------
# Tests for _rank_eps
# ---------------------------------------------------------------------------

def test_rank_eps_zero_vector() -> None:
    assert _rank_eps([], 1e-7) == 0
    assert _rank_eps([0.0, 0.0], 1e-7) == 0


def test_rank_eps_full_rank() -> None:
    # All above threshold
    assert _rank_eps([1.0, 0.5, 0.25], 1e-7) == 3


def test_rank_eps_threshold_cutoff() -> None:
    eps = 1.192e-7  # float32 eps
    sqrt_eps = math.sqrt(eps)
    # s1=1.0, threshold = 1.0 * sqrt_eps ≈ 3.45e-4
    s = [1.0, 0.5, sqrt_eps * 0.5]  # last one is below threshold
    assert _rank_eps(s, eps) == 2


# ---------------------------------------------------------------------------
# Tests for _cumulative_utilization_by_layer
# ---------------------------------------------------------------------------

@pytest.mark.mlx
def test_cumulative_utilization_orthogonal_tasks() -> None:
    """Two tasks with non-overlapping null-space projections sum correctly.

    Setup: 4×4 identity V_tail (tail_dims=4).
    Task 1: delta whose projection has rank 2 (directions e1, e2).
    Task 2: delta whose projection has rank 2 (directions e3, e4).
    Cumulative rank = 4 → utilization = 1.0
    """
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    # V_tail: identity [4, 4] — each column is a standard basis vector
    V_tail_np = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
    V_tail = backend.array(V_tail_np, dtype="float32")

    # Task 1 delta [out=2, in=4]: projects onto e1 and e2 only
    delta1 = backend.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]], dtype="float32")

    # Task 2 delta [out=2, in=4]: projects onto e3 and e4 only
    delta2 = backend.array([[0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype="float32")

    tail_bases = {"layer_a": (V_tail, 4)}
    all_deltas = [{"layer_a": delta1}, {"layer_a": delta2}]

    util = _cumulative_utilization_by_layer(backend, all_deltas, tail_bases, eps)

    assert "layer_a" in util
    assert util["layer_a"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.mlx
def test_cumulative_utilization_overlapping_tasks() -> None:
    """Two tasks with identical projections do NOT double-count.

    Same delta used for both tasks → cumulative rank = 2, not 4.
    utilization = 2/4 = 0.5
    """
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    V_tail = backend.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype="float32")

    delta = backend.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype="float32")

    tail_bases = {"layer_a": (V_tail, 4)}
    all_deltas = [{"layer_a": delta}, {"layer_a": delta}]

    util = _cumulative_utilization_by_layer(backend, all_deltas, tail_bases, eps)

    # Cumulative rank = 2 (same subspace both times), not 4
    assert util["layer_a"] == pytest.approx(0.5, abs=1e-5)


@pytest.mark.mlx
def test_cumulative_utilization_missing_layer_in_one_task() -> None:
    """Layer present in task 1 but absent from task 2 uses only task 1 delta."""
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    V_tail = backend.array([[1.0, 0.0],
                             [0.0, 1.0]], dtype="float32")

    delta1 = backend.array([[1.0, 0.0]], dtype="float32")  # rank 1 in null space

    tail_bases = {"layer_a": (V_tail, 2)}
    all_deltas = [
        {"layer_a": delta1},
        {},  # task 2 has no adapter for this layer
    ]

    util = _cumulative_utilization_by_layer(backend, all_deltas, tail_bases, eps)
    assert util["layer_a"] == pytest.approx(0.5, abs=1e-5)


@pytest.mark.mlx
def test_cumulative_utilization_no_tasks() -> None:
    """Empty task list returns 0.0 for all layers."""
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    V_tail = backend.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    tail_bases = {"layer_a": (V_tail, 2)}

    util = _cumulative_utilization_by_layer(backend, [], tail_bases, eps)
    assert util["layer_a"] == 0.0
