# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Tests for Geodesic RBF CKA module.

Tests cover:
- HSICEstimator enum values
- CKAResult dataclass
- RBF Gram matrix computation
- CKA computation and invariance properties
- Feature bias correction
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.cka import (
    CKAResult,
    HSICEstimator,
    _center_gram_matrix,
    _feature_sampling_correction,
    _participation_ratio,
    compute_cka,
    compute_cka_from_grams,
    rbf_gram_matrix,
)
from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _scalar_tol(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _array_tol(backend: "Backend", array) -> float:
    return division_epsilon(backend, array)


def _assert_unit_interval(value: float, tol: float) -> None:
    assert -tol <= value <= 1.0 + tol


def _max_abs(backend: "Backend", array) -> float:
    diff = backend.max(backend.abs(array))
    backend.eval(diff)
    return float(backend.to_scalar(diff))


# =============================================================================
# HSICEstimator Enum Tests
# =============================================================================


class TestHSICEstimator:
    """Tests for HSICEstimator enum."""

    def test_enum_values(self) -> None:
        """Verify enum string values."""
        assert HSICEstimator.BIASED.value == "biased"
        assert HSICEstimator.UNBIASED.value == "unbiased"
        assert HSICEstimator.AUTO.value == "auto"

    def test_enum_members(self) -> None:
        """Verify all enum members exist."""
        members = list(HSICEstimator)
        assert len(members) == 3


# =============================================================================
# CKAResult Dataclass Tests
# =============================================================================


class TestCKAResult:
    """Tests for CKAResult dataclass."""

    def test_basic_construction(self) -> None:
        """Test basic CKAResult construction."""
        result = CKAResult(
            cka=0.85,
            hsic_xy=0.5,
            hsic_xx=0.6,
            hsic_yy=0.65,
            sample_count=100,
        )
        assert result.cka == 0.85
        assert result.hsic_xy == 0.5
        assert result.sample_count == 100
        assert result.cka_corrected is None

    def test_is_valid_true(self) -> None:
        """Test is_valid property for valid results."""
        result = CKAResult(cka=0.5, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50)
        assert result.is_valid is True

    def test_is_valid_false_nan(self) -> None:
        """Test is_valid property for NaN CKA."""
        result = CKAResult(
            cka=float("nan"), hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result.is_valid is False

    def test_best_property(self) -> None:
        """Test best property returns corrected CKA when available."""
        result = CKAResult(
            cka=0.75,
            hsic_xy=0.3,
            hsic_xx=0.4,
            hsic_yy=0.4,
            sample_count=50,
            cka_corrected=0.88,
        )
        assert result.best == 0.88


# =============================================================================
# Gram Matrix Tests
# =============================================================================


class TestCenterGramMatrix:
    """Tests for _center_gram_matrix function."""

    def test_centering_ones(self, any_backend: "Backend") -> None:
        """Centering all-ones matrix should produce all zeros."""
        backend = any_backend
        n = 4
        gram = backend.ones((n, n))
        centered = _center_gram_matrix(gram, backend)

        tol = _array_tol(backend, centered)
        sum_arr = backend.sum(centered)
        backend.eval(sum_arr)
        assert abs(float(backend.to_scalar(sum_arr))) <= tol

    def test_centering_row_and_col_sums_zero(self, any_backend: "Backend") -> None:
        """Centered Gram matrix should have row and column sums = 0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((5, 8))
        gram = backend.matmul(X, backend.transpose(X))
        centered = _center_gram_matrix(gram, backend)

        row_sums = backend.sum(centered, axis=1)
        col_sums = backend.sum(centered, axis=0)

        tol = _array_tol(backend, centered)
        assert _max_abs(backend, row_sums) <= tol
        assert _max_abs(backend, col_sums) <= tol


class TestRBFGramMatrix:
    """Tests for RBF Gram matrix computation."""

    def test_diagonal_is_one(self, any_backend: "Backend") -> None:
        """RBF Gram diagonal should be 1 (K(x,x) = 1)."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 5))
        gram = rbf_gram_matrix(X, backend)
        diag = backend.diag(gram)
        ones = backend.ones_like(diag)

        diff = backend.max(backend.abs(diag - ones))
        backend.eval(diff)
        tol = _scalar_tol(backend)
        assert float(backend.to_scalar(diff)) <= tol

    def test_symmetric(self, any_backend: "Backend") -> None:
        """RBF Gram matrix should be symmetric."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 5))
        gram = rbf_gram_matrix(X, backend)
        gram_T = backend.transpose(gram)

        diff = backend.max(backend.abs(gram - gram_T))
        backend.eval(diff)
        tol = _scalar_tol(backend)
        assert float(backend.to_scalar(diff)) <= tol

    def test_values_in_zero_one(self, any_backend: "Backend") -> None:
        """RBF kernel values should be in (0, 1]."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 5))
        gram = rbf_gram_matrix(X, backend)

        min_val = backend.min(gram)
        max_val = backend.max(gram)
        backend.eval(min_val, max_val)

        tol = _scalar_tol(backend)
        assert float(backend.to_scalar(min_val)) >= -tol
        assert float(backend.to_scalar(max_val)) <= 1.0 + tol


# =============================================================================
# CKA Computation Tests
# =============================================================================


class TestComputeCKA:
    """Tests for main compute_cka function."""

    def test_identical_activations_cka_one(self, any_backend: "Backend") -> None:
        """CKA of identical activations should be ~1.0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((20, 16))

        result = compute_cka(X, X, backend)

        assert result.is_valid
        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol

    def test_cka_in_valid_range(self, any_backend: "Backend") -> None:
        """CKA should be in [0, 1]."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        _assert_unit_interval(result.cka, _scalar_tol(backend))

    def test_cka_single_sample_returns_zero(self, any_backend: "Backend") -> None:
        """CKA with n=1 should return 0.0."""
        backend = any_backend
        X = backend.random_normal((1, 8))
        Y = backend.random_normal((1, 8))

        result = compute_cka(X, Y, backend)

        tol = _scalar_tol(backend)
        assert abs(result.cka) <= tol
        assert result.sample_count == 1

    def test_cka_cross_dimensional(self, any_backend: "Backend") -> None:
        """CKA should work with different feature dimensions."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))  # 10 features
        Y = backend.random_normal((15, 20))  # 20 features

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        _assert_unit_interval(result.cka, _scalar_tol(backend))

    def test_cka_with_bias_correction(self, any_backend: "Backend") -> None:
        """CKA with feature bias correction should compute corrected value."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 30))
        Y = backend.random_normal((10, 30))

        result = compute_cka(X, Y, backend, feature_bias_correction=True)

        assert result.is_valid
        assert result.cka_corrected is not None
        assert result.correction_factor is not None


# =============================================================================
# CKA Invariance Properties Tests
# =============================================================================


class TestCKAInvariance:
    """Tests for CKA mathematical invariance properties."""

    def test_rotation_invariance(self, any_backend: "Backend") -> None:
        """CKA should be invariant to orthogonal rotation."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        Y = backend.random_normal((15, 8))

        # Create orthogonal matrix via QR decomposition
        random_matrix = backend.random_normal((8, 8))
        Q, _ = backend.qr(random_matrix)

        X_rotated = backend.matmul(X, Q)
        backend.eval(X_rotated)

        result_original = compute_cka(X, Y, backend)
        result_rotated = compute_cka(X_rotated, Y, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_rotated.cka) <= tol

    def test_scale_sensitivity(self, any_backend: "Backend") -> None:
        """Geodesic RBF CKA is NOT scale-invariant.

        Because sigma is derived from data distribution, scaling inputs
        changes the manifold structure. This is intentional - geodesic
        distances depend on the actual data scale.

        This test verifies that both original and scaled inputs produce
        valid CKA values in [0, 1].
        """
        from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        Y = backend.random_normal((15, 8))

        X_scaled = X * 10.0
        backend.eval(X_scaled)

        cka_original = compute_geodesic_cka(X, Y, backend)
        cka_scaled = compute_geodesic_cka(X_scaled, Y, backend)

        # Both should be valid CKA values, but NOT necessarily equal
        assert 0.0 <= cka_original <= 1.0
        assert 0.0 <= cka_scaled <= 1.0

    def test_symmetry(self, any_backend: "Backend") -> None:
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X)."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 12))

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        tol = _scalar_tol(backend)
        assert abs(result_xy.cka - result_yx.cka) <= tol


# =============================================================================
# Feature Sampling Correction Tests
# =============================================================================


class TestFeatureSamplingCorrection:
    """Tests for feature-sampling bias correction."""

    def test_participation_ratio_uniform(self, any_backend: "Backend") -> None:
        """Uniform eigenvalues should give participation ratio = n."""
        backend = any_backend
        eigvals = backend.ones((5,))
        pr = _participation_ratio(eigvals, backend)

        tol = _array_tol(backend, eigvals)
        assert abs(pr - 5.0) <= tol

    def test_correction_with_valid_gram(self, any_backend: "Backend") -> None:
        """Correction should return factor >= 1.0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 20))
        gram = backend.matmul(X, backend.transpose(X))
        centered = _center_gram_matrix(gram, backend)

        correction, gamma = _feature_sampling_correction(centered, 20, backend)

        tol = _array_tol(backend, centered)
        assert correction >= 1.0 - tol


# =============================================================================
# CKA from Grams Tests
# =============================================================================


class TestComputeCKAFromGrams:
    """Tests for compute_cka_from_grams."""

    def test_from_grams_identical(self, any_backend: "Backend") -> None:
        """CKA from identical Grams should be 1.0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))
        gram = rbf_gram_matrix(X, backend)

        cka = compute_cka_from_grams(gram, gram, backend)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_from_grams_in_range(self, any_backend: "Backend") -> None:
        """CKA from Grams should be in [0, 1]."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 12))
        gram_x = rbf_gram_matrix(X, backend)
        gram_y = rbf_gram_matrix(Y, backend)

        cka = compute_cka_from_grams(gram_x, gram_y, backend)

        _assert_unit_interval(cka, _scalar_tol(backend))


