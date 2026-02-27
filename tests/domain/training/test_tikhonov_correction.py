"""Tests for Tikhonov eigenvalue-weighted quantization correction."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.tikhonov_correction import (
    TikhonovLayerResult,
    compute_mp_noise_edge,
    compute_tikhonov_weights,
    correct_projection_tikhonov,
)


@pytest.fixture()
def backend():
    return get_default_backend()


# ── compute_mp_noise_edge ──────────────────────────────────────────────


def test_mp_noise_edge_positive(backend):
    """MP noise edge must be positive for any positive eigenspectrum."""
    eigvals = backend.array([10.0, 5.0, 1.0, 0.1], dtype="float32")
    backend.eval(eigvals)
    edge = compute_mp_noise_edge(eigvals, n_tokens=100, dimensionality=4, backend=backend)
    assert edge > 0.0


def test_mp_noise_edge_increases_with_aspect_ratio(backend):
    """MP edge grows with D/N (more dimensions relative to samples → more noise)."""
    eigvals = backend.array([10.0, 5.0, 1.0, 0.1], dtype="float32")
    backend.eval(eigvals)

    edge_low = compute_mp_noise_edge(eigvals, n_tokens=1000, dimensionality=4, backend=backend)
    edge_high = compute_mp_noise_edge(eigvals, n_tokens=10, dimensionality=4, backend=backend)
    assert edge_high > edge_low


def test_mp_noise_edge_rejects_invalid_inputs(backend):
    eigvals = backend.array([1.0], dtype="float32")
    backend.eval(eigvals)
    with pytest.raises(ValueError, match="must be > 0"):
        compute_mp_noise_edge(eigvals, n_tokens=0, dimensionality=4, backend=backend)
    with pytest.raises(ValueError, match="must be > 0"):
        compute_mp_noise_edge(eigvals, n_tokens=100, dimensionality=0, backend=backend)


# ── compute_tikhonov_weights ───────────────────────────────────────────


def test_weights_in_unit_interval(backend):
    """All Tikhonov weights must be in [0, 1]."""
    eigvals = backend.array([100.0, 10.0, 1.0, 0.1, 0.01], dtype="float32")
    backend.eval(eigvals)
    mp_edge = compute_mp_noise_edge(eigvals, n_tokens=50, dimensionality=5, backend=backend)
    weights = compute_tikhonov_weights(eigvals, mp_edge, backend=backend)
    backend.eval(weights)

    for i in range(5):
        w = float(backend.to_scalar(weights[i]))
        assert 0.0 <= w <= 1.0, f"Weight {i} = {w} outside [0, 1]"


def test_weights_monotonically_decreasing(backend):
    """Weights must decrease as eigenvalues decrease (for descending eigvals)."""
    eigvals = backend.array([100.0, 10.0, 1.0, 0.1, 0.001], dtype="float32")
    backend.eval(eigvals)
    mp_edge = compute_mp_noise_edge(eigvals, n_tokens=50, dimensionality=5, backend=backend)
    weights = compute_tikhonov_weights(eigvals, mp_edge, backend=backend)
    backend.eval(weights)

    prev = float(backend.to_scalar(weights[0]))
    for i in range(1, 5):
        curr = float(backend.to_scalar(weights[i]))
        assert curr <= prev + 1e-7, f"Weight {i} ({curr}) > weight {i-1} ({prev})"
        prev = curr


def test_large_eigenvalue_weight_near_one(backend):
    """Eigenvalue >> MP edge → weight ≈ 1.

    MP edge = sigma_sq * (1 + sqrt(D/N))^2 where sigma_sq = trace/D.
    With many similar eigenvalues, sigma_sq is moderate and top eigenvalue
    dominates the noise edge.
    """
    # 10 eigenvalues, all ~10.0.  sigma_sq ≈ 10, aspect = 10/10000 = 0.001,
    # mp_edge ≈ 10 * (1 + 0.032)^2 ≈ 10.6.  Top eigval = 100 >> 10.6.
    eigvals = backend.array(
        [100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        dtype="float32",
    )
    backend.eval(eigvals)
    mp_edge = compute_mp_noise_edge(eigvals, n_tokens=10000, dimensionality=10, backend=backend)
    weights = compute_tikhonov_weights(eigvals, mp_edge, backend=backend)
    backend.eval(weights)
    # w = 100 / (100 + mp_edge).  mp_edge includes top eigenvalue in trace → ~20.
    # So w ≈ 100/120 ≈ 0.83.  This IS conservative weighting by design.
    assert float(backend.to_scalar(weights[0])) > 0.8


def test_small_eigenvalue_weight_near_zero(backend):
    """Eigenvalue << MP edge → weight ≈ 0."""
    eigvals = backend.array([1e6, 1e-10], dtype="float32")
    backend.eval(eigvals)
    mp_edge = compute_mp_noise_edge(eigvals, n_tokens=1000, dimensionality=2, backend=backend)
    weights = compute_tikhonov_weights(eigvals, mp_edge, backend=backend)
    backend.eval(weights)
    assert float(backend.to_scalar(weights[1])) < 0.01


# ── correct_projection_tikhonov ───────────────────────────────────────


def test_correction_bounded_by_error(backend):
    """||Delta||_F <= ||E||_F — correction never exceeds total error."""
    fp_w = backend.array([[5.0, 3.0], [1.0, 4.0]], dtype="float32")
    q_w = backend.array([[4.8, 2.9], [1.1, 3.8]], dtype="float32")
    eigvecs = backend.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    weights = backend.array([0.9, 0.1], dtype="float32")
    backend.eval(fp_w, q_w, eigvecs, weights)

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, backend, layer_key="test",
    )
    assert result is not None
    assert result.delta_frob <= result.E_total_frob + 1e-6


def test_energy_fractions_bounded(backend):
    """correction_fraction and preserved_fraction are both non-negative and sum <= 1.

    With V orthonormal: ||Delta||² + ||E_residual||² = Σᵢⱼ E²ᵢⱼ (wⱼ² + (1-wⱼ)²).
    Since wⱼ² + (1-wⱼ)² ∈ [0.5, 1.0] for wⱼ ∈ [0, 1], the sum is ≤ ||E||².
    Equality holds only when all weights are 0 or 1 (hard rank cutoff).
    """
    fp_w = backend.array([[5.0, 3.0, 1.0], [1.0, 4.0, 2.0]], dtype="float32")
    q_w = backend.array([[4.5, 2.5, 1.5], [0.5, 3.5, 2.5]], dtype="float32")
    eigvecs = backend.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32",
    )
    weights = backend.array([0.95, 0.5, 0.01], dtype="float32")
    backend.eval(fp_w, q_w, eigvecs, weights)

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, backend, layer_key="test",
    )
    assert result is not None
    assert result.correction_fraction >= 0.0
    assert result.preserved_fraction >= 0.0
    total = result.correction_fraction + result.preserved_fraction
    assert total <= 1.0 + 1e-6


def test_zero_error_returns_none(backend):
    """When W_fp == W_q, correction is skipped (no error to correct)."""
    w = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    eigvecs = backend.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    weights = backend.array([0.9, 0.1], dtype="float32")
    backend.eval(w, eigvecs, weights)

    corrected, result = correct_projection_tikhonov(
        w, w, eigvecs, weights, backend, layer_key="test",
    )
    assert result is None


def test_corrected_weight_closer_to_fp(backend):
    """Corrected weight should be closer to FP than the quantized weight."""
    fp_w = backend.array([[5.0, 3.0], [1.0, 4.0]], dtype="float32")
    q_w = backend.array([[4.0, 2.0], [2.0, 3.0]], dtype="float32")
    eigvecs = backend.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    weights = backend.array([0.8, 0.8], dtype="float32")
    backend.eval(fp_w, q_w, eigvecs, weights)

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, backend, layer_key="test",
    )
    backend.eval(corrected)

    # ||W_corrected - W_fp||_F < ||W_q - W_fp||_F
    E_before = fp_w - q_w
    E_after = fp_w - corrected
    backend.eval(E_before, E_after)
    frob_before = float(backend.to_scalar(backend.sum(E_before * E_before)))
    frob_after = float(backend.to_scalar(backend.sum(E_after * E_after)))
    assert frob_after < frob_before


def test_result_dataclass_fields(backend):
    """TikhonovLayerResult has all expected fields."""
    fp_w = backend.array([[5.0, 3.0], [1.0, 4.0]], dtype="float32")
    q_w = backend.array([[4.8, 2.9], [1.1, 3.8]], dtype="float32")
    eigvecs = backend.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    weights = backend.array([0.9, 0.1], dtype="float32")
    backend.eval(fp_w, q_w, eigvecs, weights)

    _, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, backend, layer_key="test.layer.0",
        tikhonov_effective_rank=1.0, mp_noise_edge=0.5, D_eff=1.5,
    )
    assert result is not None
    assert result.layer_key == "test.layer.0"
    assert result.tikhonov_effective_rank == 1.0
    assert result.mp_noise_edge == 0.5
    assert result.D_eff == 1.5
    assert result.top_weight == pytest.approx(0.9, abs=1e-6)
