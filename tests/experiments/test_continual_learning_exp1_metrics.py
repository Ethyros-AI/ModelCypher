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
    _compute_sigma_k_ref,
    _cumulative_utilization_by_layer,
    _precompute_tail_bases,
    _rank_eps,
    _read_quant_config,
    _spectral_norm,
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


# ---------------------------------------------------------------------------
# Tests for _precompute_tail_bases quantization handling
# ---------------------------------------------------------------------------


class _FakeModelLoader:
    """Minimal model_loader stub: yields from a fixed dict."""

    def __init__(self, weights: dict):
        self._weights = weights

    def iter_weights(self, _model_path: str):
        yield from sorted(self._weights.items())


def test_precompute_tail_bases_float_weight() -> None:
    """Non-quantized (float32) weight goes straight to SVD without dequantize."""
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    # Use a rank-deficient weight so we get tail_dims > 0
    W_rank2 = backend.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]], dtype="float32"
    )

    loader = _FakeModelLoader({"layer.weight": W_rank2})
    # Provide a temp dir that has no config.json so quant_cfg is empty
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = _precompute_tail_bases(
            backend, loader, tmp, {"layer.weight"}, eps
        )

    assert "layer.weight" in result
    V_tail, tail_dims = result["layer.weight"]
    assert tail_dims == 2
    assert V_tail.shape[0] == 4  # V_tail lives in input space (4-dim)
    assert V_tail.shape[1] == 2  # 2 tail dimensions


@pytest.mark.mlx
def test_precompute_tail_bases_quantized_weight() -> None:
    """Quantized (uint32) weight is dequantized before SVD.

    Constructs a 4×64 weight (MLX minimum group_size=64), quantizes it via
    backend.quantize, passes the packed tensor to _precompute_tail_bases, and
    verifies:
    - V_tail lives in the unpacked 64-dim input space (not 8-dim packed space)
    - tail_dims > 0 (confirming SVD ran on the dequantized matrix)
    """
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    # 4×64 weight: rows 0–1 are identical → rank ≤ 3, tail_dims ≥ 1.
    # MLX quantize requires group_size ∈ {32, 64, 128} and in_features % group_size == 0.
    # Use group_size=64 (exactly one group per row).
    row0 = [float(i % 8 + 1) for i in range(64)]
    row1 = list(row0)                        # identical → linear dependence
    row2 = [float((i + 1) % 8 + 1) for i in range(64)]
    row3 = [float((i + 3) % 8 + 1) for i in range(64)]
    W_fp = backend.array([row0, row1, row2, row3], dtype="float32")
    backend.eval(W_fp)

    w_q, scales, biases = backend.quantize(W_fp, group_size=64, bits=4, mode="affine")
    backend.eval(w_q, scales)
    if biases is not None:
        backend.eval(biases)

    weights: dict = {"layer.weight": w_q, "layer.scales": scales}
    if biases is not None:
        weights["layer.biases"] = biases

    import json
    import tempfile
    quant_cfg = {"bits": 4, "group_size": 64, "mode": "affine"}
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(json.dumps({"quantization": quant_cfg}))

        loader = _FakeModelLoader(weights)
        result = _precompute_tail_bases(
            backend, loader, tmp, {"layer.weight"}, eps
        )

    assert "layer.weight" in result, "quantized layer must be processed"
    V_tail, tail_dims = result["layer.weight"]
    # V_tail input dimension must match unpacked in_features=64, not packed=8
    assert V_tail.shape[0] == 64, (
        f"V_tail.shape[0]={V_tail.shape[0]}, expected 64 (unpacked in_features)"
    )
    assert tail_dims >= 1


@pytest.mark.mlx
def test_precompute_tail_bases_missing_scales_raises() -> None:
    """Quantized weight without scales raises RuntimeError (fail-closed).

    Partial geometry (missing scales → can't dequantize) would silently bias
    capacity_total and the stop rule. Must fail closed.
    """
    backend = _get_backend()
    eps = float(backend.finfo().eps)

    # Produce a real packed uint32 tensor via quantize (group_size=64 required by MLX).
    W_fp = backend.array([[float(i % 8 + 1) for i in range(64)]], dtype="float32")
    backend.eval(W_fp)
    w_q, _scales, _biases = backend.quantize(W_fp, group_size=64, bits=4, mode="affine")
    backend.eval(w_q)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        loader = _FakeModelLoader({"layer.weight": w_q})
        with pytest.raises(RuntimeError, match="no scales tensor"):
            _precompute_tail_bases(
                backend, loader, tmp, {"layer.weight"}, eps
            )


@pytest.mark.mlx
def test_compute_sigma_k_ref_ignores_non_weight_entries() -> None:
    """sigma_k_ref must come from a 2D .weight tensor, not .biases/.scales."""
    backend = _get_backend()
    W = backend.array([[3.0, 0.0], [0.0, 4.0]], dtype="float32")
    b = backend.array([1.0, 2.0], dtype="float32")
    backend.eval(W, b)

    loader = _FakeModelLoader(
        {
            "layer.biases": b,
            "layer.weight": W,
        }
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        layer_name, sigma = _compute_sigma_k_ref(backend, loader, tmp)

    assert layer_name == "layer.weight"
    assert sigma == pytest.approx(4.0, abs=1e-6)


@pytest.mark.mlx
def test_compute_sigma_k_ref_quantized_matches_dequantized_weight() -> None:
    """Quantized reference must dequantize before spectral norm."""
    backend = _get_backend()

    # 2×64 is valid for MLX quantize(group_size=64).
    W_fp = backend.array(
        [[float((i % 7) + 1) for i in range(64)],
         [float(((i + 3) % 7) + 1) for i in range(64)]],
        dtype="float32",
    )
    backend.eval(W_fp)
    w_q, scales, biases = backend.quantize(W_fp, group_size=64, bits=4, mode="affine")
    backend.eval(w_q, scales)
    if biases is not None:
        backend.eval(biases)

    weights: dict = {
        "layer.scales": scales,
        "layer.weight": w_q,
    }
    if biases is not None:
        weights["layer.biases"] = biases

    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(
            json.dumps({"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}})
        )

        loader = _FakeModelLoader(weights)
        layer_name, sigma = _compute_sigma_k_ref(backend, loader, tmp)

    W_deq = backend.dequantize(w_q, scales, biases, group_size=64, bits=4, mode="affine")
    backend.eval(W_deq)
    expected = _spectral_norm(backend, W_deq)

    assert layer_name == "layer.weight"
    assert sigma == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.mlx
def test_compute_sigma_k_ref_quantized_missing_scales_raises() -> None:
    """Missing scales on quantized reference must fail closed."""
    backend = _get_backend()
    W_fp = backend.array([[float(i % 8 + 1) for i in range(64)]], dtype="float32")
    backend.eval(W_fp)
    w_q, _scales, _biases = backend.quantize(W_fp, group_size=64, bits=4, mode="affine")
    backend.eval(w_q)

    loader = _FakeModelLoader({"layer.weight": w_q})
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="sigma_k_ref"):
            _compute_sigma_k_ref(backend, loader, tmp)
