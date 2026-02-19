# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.spectral_capacity import (
    SpectralCapacityAnalyzer,
    compute_full_energy_curve,
    find_energy_inflection_points,
)


def test_identity_matrix_capacity(any_backend) -> None:
    b = any_backend
    analyzer = SpectralCapacityAnalyzer(b)

    identity = b.eye(5)
    report = analyzer.analyze("identity", identity)

    assert report.weight_shape == (5, 5)
    assert report.spectral_norm == pytest.approx(1.0)
    assert report.nuclear_norm == pytest.approx(5.0)
    assert report.frobenius_norm == pytest.approx(math.sqrt(5.0))
    assert report.effective_rank == pytest.approx(5.0)
    assert report.stable_rank == pytest.approx(5.0)
    assert report.numerical_rank_f32 == 5
    assert report.numerical_rank_f16 == 5
    assert report.null_space_dim_f32 == 0
    assert report.null_space_fraction == pytest.approx(0.0)
    assert report.capacity_utilization == pytest.approx(1.0)


def test_rank_one_outer_product_capacity(any_backend) -> None:
    b = any_backend
    analyzer = SpectralCapacityAnalyzer(b)

    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    v = [2.0, 0.0, -1.0, 1.0, 3.0]
    rank_one = b.array([[ui * vj for vj in v] for ui in u], dtype="float32")
    report = analyzer.analyze("rank_one", rank_one)

    assert report.effective_rank == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert report.stable_rank == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert report.numerical_rank_f32 == 1
    assert report.null_space_dim_f32 == 4
    assert report.null_space_fraction == pytest.approx(4.0 / 5.0)


def test_known_gap_spectrum_recommendation(any_backend) -> None:
    b = any_backend
    analyzer = SpectralCapacityAnalyzer(b)

    # Singular values are exactly [10, 10, 10, 0.001, 0.001].
    matrix = b.diag(b.array([10.0, 10.0, 10.0, 0.001, 0.001], dtype="float32"))
    report = analyzer.analyze("known_gap", matrix)

    assert report.effective_rank == pytest.approx(3.0, rel=1e-3, abs=1e-3)
    assert report.recommended_rank == 3
    assert report.spectral_gap_at_rank == pytest.approx(10000.0, rel=1e-3)
    assert report.numerical_rank_f32 == 3
    assert report.numerical_rank_f16 == 3
    assert report.null_space_dim_f32 == 2
    assert report.null_space_fraction == pytest.approx(2.0 / 5.0)


def test_linear_decay_diagonal_effective_rank(any_backend) -> None:
    b = any_backend
    analyzer = SpectralCapacityAnalyzer(b)

    diag_values = [5.0, 4.0, 3.0, 2.0, 1.0]
    matrix = b.diag(b.array(diag_values, dtype="float32"))
    report = analyzer.analyze("linear_decay", matrix)

    # Closed form for diagonal matrix with singular values s_i:
    # effective_rank = (sum(s_i)^2) / sum(s_i^2)
    expected_effective_rank = (sum(diag_values) ** 2) / sum(v * v for v in diag_values)
    assert report.effective_rank == pytest.approx(expected_effective_rank, rel=1e-6)


class _FailingSVDBackend:
    def __init__(self, backend):
        self._backend = backend

    def svd(self, array, compute_uv=True):
        raise RuntimeError("forced svd failure")

    def __getattr__(self, name):
        return getattr(self._backend, name)


class _FailingSVDEigBackend:
    def __init__(self, backend):
        self._backend = backend

    def svd(self, array, compute_uv=True):
        raise RuntimeError("forced svd failure")

    def eigvalsh(self, array):
        raise RuntimeError("forced eigvalsh failure")

    def __getattr__(self, name):
        return getattr(self._backend, name)


def test_fallback_uses_gram_eigh_when_svd_fails(any_backend) -> None:
    wrapped = _FailingSVDBackend(any_backend)
    analyzer = SpectralCapacityAnalyzer(wrapped)

    matrix = wrapped.eye(4)
    report = analyzer.analyze("fallback_gram", matrix)

    assert report.computation_method == "gram_eigh"
    assert report.effective_rank == pytest.approx(4.0, rel=1e-5)
    assert report.numerical_rank_f32 == 4


def test_fallback_uses_iterative_when_svd_and_eigh_fail(any_backend) -> None:
    wrapped = _FailingSVDEigBackend(any_backend)
    analyzer = SpectralCapacityAnalyzer(wrapped)

    matrix = wrapped.eye(4)
    report = analyzer.analyze("fallback_iterative", matrix)

    assert report.computation_method in {"power_deflation", "power_iteration"}
    assert report.spectral_norm > 0.0
    assert report.numerical_rank_f32 >= 1


# --- Tests for compute_full_energy_curve ---


def test_energy_curve_empty_input() -> None:
    assert compute_full_energy_curve([]) == []


def test_energy_curve_single_value() -> None:
    curve = compute_full_energy_curve([5.0])
    assert len(curve) == 1
    assert curve[0] == pytest.approx(1.0)


def test_energy_curve_monotonic_and_converges() -> None:
    """Energy curve must be monotonically non-decreasing and converge to 1.0."""
    sv = [10.0, 5.0, 2.0, 1.0, 0.5]
    curve = compute_full_energy_curve(sv)
    assert len(curve) == len(sv)
    for i in range(1, len(curve)):
        assert curve[i] >= curve[i - 1]
    assert curve[-1] == pytest.approx(1.0)


def test_energy_curve_known_values() -> None:
    """Verify energy fractions for a known diagonal spectrum."""
    sv = [3.0, 2.0, 1.0]
    # energies: 9, 4, 1; total=14
    curve = compute_full_energy_curve(sv)
    assert curve[0] == pytest.approx(9.0 / 14.0)
    assert curve[1] == pytest.approx(13.0 / 14.0)
    assert curve[2] == pytest.approx(1.0)


def test_energy_curve_all_zeros() -> None:
    curve = compute_full_energy_curve([0.0, 0.0, 0.0])
    assert all(v == 0.0 for v in curve)


# --- Tests for find_energy_inflection_points ---


def test_inflection_points_too_short() -> None:
    """Curves shorter than 4 elements cannot have inflection points."""
    assert find_energy_inflection_points([]) == []
    assert find_energy_inflection_points([0.5, 0.8, 1.0]) == []


def test_inflection_points_detects_cliff() -> None:
    """A sharp cliff in the spectrum produces a high-prominence inflection."""
    sv = [10.0, 10.0, 10.0, 0.001, 0.001, 0.001]
    curve = compute_full_energy_curve(sv)
    inflections = find_energy_inflection_points(curve)
    assert len(inflections) > 0
    # The top inflection should be near the cliff (rank 3-4)
    top = inflections[0]
    assert 2 <= top["rank"] <= 5


def test_inflection_points_sorted_by_prominence() -> None:
    """Inflection points are returned sorted by |d2E| descending."""
    sv = [10.0, 8.0, 5.0, 1.0, 0.5, 0.1]
    curve = compute_full_energy_curve(sv)
    inflections = find_energy_inflection_points(curve)
    for i in range(1, len(inflections)):
        assert inflections[i]["abs_d2E"] <= inflections[i - 1]["abs_d2E"]


def test_inflection_points_min_prominence_filters() -> None:
    """Raising min_prominence should reduce the number of detected inflections."""
    sv = [10.0, 8.0, 5.0, 1.0, 0.5, 0.1, 0.01]
    curve = compute_full_energy_curve(sv)
    all_inflections = find_energy_inflection_points(curve, min_prominence=0.0)
    filtered = find_energy_inflection_points(curve, min_prominence=0.01)
    assert len(filtered) <= len(all_inflections)
