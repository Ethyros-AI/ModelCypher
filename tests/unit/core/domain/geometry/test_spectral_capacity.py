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

from modelcypher.core.domain.geometry.spectral_capacity import SpectralCapacityAnalyzer


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
