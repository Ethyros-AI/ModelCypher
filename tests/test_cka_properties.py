# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Property-based tests for Geodesic RBF CKA.

Tests mathematical invariants:
- CKA ∈ [0, 1]
- CKA(X, X) = 1.0 (self-similarity)
- Rotation invariance
- Scale invariance
- Symmetry
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import (
    _center_gram_matrix,
    compute_cka,
    geodesic_squared_distances,
    rbf_gram_matrix,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _scalar_tol(backend) -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _all_close(backend, arr1, arr2) -> bool:
    """Check if two arrays are element-wise close."""
    diff = backend.abs(arr1 - arr2)
    max_diff = backend.max(diff)
    backend.eval(max_diff)
    tol = division_epsilon(backend, diff)
    return float(backend.to_scalar(max_diff)) <= tol


def _all_non_negative(backend, arr) -> bool:
    """Check if all elements are >= -tol."""
    min_val = backend.min(arr)
    backend.eval(min_val)
    tol = division_epsilon(backend, arr)
    return float(backend.to_scalar(min_val)) >= -tol


# =============================================================================
# CKA Bounds Tests
# =============================================================================


class TestCKABounds:
    """Tests for CKA value bounds."""

    @pytest.mark.parametrize("seed", range(10))
    def test_cka_in_zero_one(self, seed: int):
        """CKA must be in [0, 1]."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result = compute_cka(x, y, backend)
        tol = _scalar_tol(backend)
        assert -tol <= result.cka <= 1.0 + tol

    @pytest.mark.parametrize("seed", range(10))
    def test_cka_self_is_one(self, seed: int):
        """CKA(X, X) = 1.0."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        result = compute_cka(x, x, backend)
        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol


# =============================================================================
# Invariance Tests
# =============================================================================


class TestCKAInvariance:
    """Tests for CKA invariance properties."""

    @pytest.mark.parametrize("seed", range(5))
    def test_rotation_invariance(self, seed: int):
        """CKA should be invariant to orthogonal transformations."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        # Generate random orthogonal matrix via QR decomposition
        random_mat = _random_matrix(backend, 10, 10, seed + 2000)
        q, _ = backend.qr(random_mat)

        y_rotated = backend.matmul(y, q)

        result_original = compute_cka(x, y, backend)
        result_rotated = compute_cka(x, y_rotated, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_rotated.cka) <= tol

    def test_scale_sensitivity(self):
        """Geodesic RBF CKA is NOT scale-invariant.

        Because sigma is derived from geodesic distances, scaling inputs
        changes the manifold structure. Both original and scaled should
        produce valid CKA values.
        """
        from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        y = _random_matrix(backend, 20, 10, 43)
        x_scaled = x * 5.0

        cka_original = compute_geodesic_cka(x, y, backend)
        cka_scaled = compute_geodesic_cka(x_scaled, y, backend)

        # Both should be valid but not necessarily equal
        assert 0.0 <= cka_original <= 1.0
        assert 0.0 <= cka_scaled <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_symmetry(self, seed: int):
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X)."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result_xy = compute_cka(x, y, backend)
        result_yx = compute_cka(y, x, backend)

        tol = _scalar_tol(backend)
        assert abs(result_xy.cka - result_yx.cka) <= tol


# =============================================================================
# Gram Matrix Tests
# =============================================================================


class TestGramMatrix:
    """Tests for Gram matrix properties."""

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_matrix_symmetric(self, seed: int):
        """Geodesic distance matrix should be symmetric."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        distances = geodesic_squared_distances(x, backend)
        distances_T = backend.transpose(distances)

        assert _all_close(backend, distances, distances_T)

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_matrix_non_negative(self, seed: int):
        """Geodesic distances should be non-negative."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        distances = geodesic_squared_distances(x, backend)
        assert _all_non_negative(backend, distances)

    @pytest.mark.parametrize("seed", range(10))
    def test_centered_gram_row_sum_zero(self, seed: int):
        """Centered Gram matrix rows should sum to approximately zero."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        gram = rbf_gram_matrix(x, backend)
        centered = _center_gram_matrix(gram, backend)

        row_sums = backend.sum(centered, axis=1)
        max_row_sum = backend.max(backend.abs(row_sums))
        backend.eval(max_row_sum)

        tol = division_epsilon(backend, centered)
        assert float(backend.to_scalar(max_row_sum)) <= tol


# =============================================================================
# Edge Cases
# =============================================================================


class TestCKAEdgeCases:
    """Edge case tests for CKA."""

    def test_single_sample_returns_zero_cka(self):
        """Single sample should return CKA = 0."""
        backend = get_default_backend()
        x = backend.array([[1.0, 2.0, 3.0]])
        result = compute_cka(x, x, backend)
        tol = _scalar_tol(backend)
        assert abs(result.cka) <= tol
        assert result.sample_count == 1

    def test_two_samples_valid(self):
        """Two samples should produce valid CKA."""
        backend = get_default_backend()
        x = backend.array([[1.0, 2.0], [3.0, 4.0]])
        y = backend.array([[1.1, 2.1], [3.1, 4.1]])
        result = compute_cka(x, y, backend)
        tol = _scalar_tol(backend)
        assert -tol <= result.cka <= 1.0 + tol
        assert result.sample_count == 2

    def test_different_feature_dimensions(self):
        """CKA should work with different feature dimensions."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        y = _random_matrix(backend, 20, 15, 43)

        result = compute_cka(x, y, backend)
        tol = _scalar_tol(backend)
        assert -tol <= result.cka <= 1.0 + tol
        assert result.is_valid


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestCKAHypothesis:
    """Hypothesis-based property tests."""

    @given(
        n_samples=st.integers(min_value=4, max_value=50),
        n_features=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None)
    def test_self_similarity_hypothesis(self, n_samples: int, n_features: int, seed: int):
        """CKA(X, X) = 1.0 for all valid matrices."""
        backend = get_default_backend()
        X = _random_matrix(backend, n_samples, n_features, seed)

        result = compute_cka(X, X, backend)

        assert result.is_valid
        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol

    @given(
        n_samples=st.integers(min_value=4, max_value=30),
        n_features_x=st.integers(min_value=2, max_value=20),
        n_features_y=st.integers(min_value=2, max_value=20),
        seed_x=st.integers(min_value=0, max_value=10000),
        seed_y=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None)
    def test_symmetry_hypothesis(
        self,
        n_samples: int,
        n_features_x: int,
        n_features_y: int,
        seed_x: int,
        seed_y: int,
    ):
        """CKA(X, Y) = CKA(Y, X)."""
        assume(seed_x != seed_y)
        backend = get_default_backend()
        X = _random_matrix(backend, n_samples, n_features_x, seed_x)
        Y = _random_matrix(backend, n_samples, n_features_y, seed_y)

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        assert result_xy.is_valid and result_yx.is_valid
        tol = _scalar_tol(backend)
        assert abs(result_xy.cka - result_yx.cka) <= tol

    @given(
        n_samples=st.integers(min_value=4, max_value=50),
        n_features_x=st.integers(min_value=2, max_value=32),
        n_features_y=st.integers(min_value=2, max_value=32),
        seed_x=st.integers(min_value=0, max_value=10000),
        seed_y=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_bounded_hypothesis(
        self,
        n_samples: int,
        n_features_x: int,
        n_features_y: int,
        seed_x: int,
        seed_y: int,
    ):
        """CKA always in [0, 1]."""
        backend = get_default_backend()
        X = _random_matrix(backend, n_samples, n_features_x, seed_x)
        Y = _random_matrix(backend, n_samples, n_features_y, seed_y)

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0
