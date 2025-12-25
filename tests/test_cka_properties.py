# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Property-based tests for CKA (Centered Kernel Alignment).

Tests mathematical invariants:
- CKA ∈ [0, 1]
- CKA(X, X) = 1.0 (self-similarity)
- Rotation invariance: CKA(X @ R, Y) = CKA(X, Y) for orthogonal R
- Scale invariance: CKA(α * X, Y) = CKA(X, Y)
- HSIC ≥ 0 (non-negative)
- Symmetry: CKA(X, Y) = CKA(Y, X)

NOTE: All tests use the Backend protocol exclusively. No numpy.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import (
    _center_gram_matrix,
    _compute_pairwise_squared_distances,
    compute_cka,
)


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _is_close(a: float, b: float, atol: float = 1e-5) -> bool:
    """Check if two floats are close."""
    return abs(a - b) <= atol


def _all_close(backend, arr1, arr2, atol: float = 1e-5) -> bool:
    """Check if two arrays are element-wise close using backend."""
    diff = backend.abs(arr1 - arr2)
    max_diff = float(backend.to_numpy(backend.max(diff)))
    return max_diff <= atol


def _all_non_negative(backend, arr, tol: float = 1e-10) -> bool:
    """Check if all elements are >= -tol using backend."""
    min_val = float(backend.to_numpy(backend.min(arr)))
    return min_val >= -tol


# =============================================================================
# CKA Bounds Tests
# =============================================================================


class TestCKABounds:
    """Tests for CKA value bounds."""

    @pytest.mark.parametrize("seed", range(10))
    def test_cka_in_zero_one(self, seed: int):
        """CKA must be in [0, 1].

        Mathematical property: CKA is normalized by sqrt(HSIC_xx * HSIC_yy),
        making it a correlation-like measure bounded in [0, 1].
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result = compute_cka(x, y, backend)
        assert 0.0 <= result.cka <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_cka_self_is_one(self, seed: int):
        """CKA(X, X) = 1.0.

        Mathematical property: HSIC(X, X) / sqrt(HSIC(X, X) * HSIC(X, X)) = 1.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        result = compute_cka(x, x, backend)
        assert result.cka == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# Invariance Tests
# =============================================================================


class TestCKAInvariance:
    """Tests for CKA invariance properties."""

    @pytest.mark.parametrize("seed", range(5))
    def test_rotation_invariance(self, seed: int):
        """CKA should be invariant to orthogonal transformations.

        Mathematical property: CKA(X @ R, Y) = CKA(X, Y) for orthogonal R.
        This holds because RBF/linear kernels depend on distances/inner products,
        both preserved under orthogonal transforms.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        # Generate random orthogonal matrix via QR decomposition
        random_mat = _random_matrix(backend, 10, 10, seed + 2000)
        q, _ = backend.qr(random_mat)

        y_rotated = backend.matmul(y, q)

        result_original = compute_cka(x, y, backend)
        result_rotated = compute_cka(x, y_rotated, backend)

        assert result_original.cka == pytest.approx(result_rotated.cka, abs=0.05)

    @pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 5.0, 10.0])
    def test_scale_invariance(self, scale: float):
        """CKA should be invariant to positive scaling.

        Mathematical property: CKA(α * X, Y) = CKA(X, Y) for α > 0.
        The normalization by sqrt(HSIC_xx) cancels out scale factors.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        y = _random_matrix(backend, 20, 10, 43)
        x_scaled = x * scale

        result_original = compute_cka(x, y, backend)
        result_scaled = compute_cka(x_scaled, y, backend)

        assert result_original.cka == pytest.approx(result_scaled.cka, abs=0.05)

    @pytest.mark.parametrize("seed", range(10))
    def test_symmetry(self, seed: int):
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X).

        Mathematical property: HSIC is symmetric in its arguments.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result_xy = compute_cka(x, y, backend)
        result_yx = compute_cka(y, x, backend)

        assert result_xy.cka == pytest.approx(result_yx.cka, abs=1e-6)


# =============================================================================
# HSIC Tests
# =============================================================================


class TestHSIC:
    """Tests for HSIC (Hilbert-Schmidt Independence Criterion)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_hsic_self_non_negative(self, seed: int):
        """HSIC(X, X) >= 0.

        Mathematical property: HSIC is a squared norm in RKHS,
        hence always non-negative.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        result = compute_cka(x, x, backend)
        assert result.hsic_xx >= 0.0

    @pytest.mark.parametrize("seed", range(10))
    def test_hsic_xy_bounded(self, seed: int):
        """HSIC(X, Y) should satisfy Cauchy-Schwarz.

        Mathematical property: |HSIC(X, Y)| <= sqrt(HSIC(X, X) * HSIC(Y, Y)).
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result = compute_cka(x, y, backend)

        # Cauchy-Schwarz bound
        max_hsic = math.sqrt(result.hsic_xx * result.hsic_yy)
        if max_hsic > 1e-10:
            assert abs(result.hsic_xy) <= max_hsic + 1e-6


# =============================================================================
# Gram Matrix Tests
# =============================================================================


class TestGramMatrix:
    """Tests for Gram matrix properties."""

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_matrix_symmetric(self, seed: int):
        """Pairwise distance matrix should be symmetric.

        Mathematical property: ||x_i - x_j|| = ||x_j - x_i||.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        distances = _compute_pairwise_squared_distances(x, backend)
        distances_T = backend.transpose(distances)

        assert _all_close(backend, distances, distances_T)

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_matrix_non_negative(self, seed: int):
        """Pairwise distances should be non-negative.

        Mathematical property: ||x_i - x_j||^2 >= 0.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        distances = _compute_pairwise_squared_distances(x, backend)
        assert _all_non_negative(backend, distances)

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_diagonal_zero(self, seed: int):
        """Diagonal of distance matrix should be zero.

        Mathematical property: ||x_i - x_i|| = 0.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        distances = _compute_pairwise_squared_distances(x, backend)
        diag = backend.diag(distances)
        diag_max = float(backend.to_numpy(backend.max(backend.abs(diag))))

        assert diag_max < 1e-5

    @pytest.mark.parametrize("seed", range(10))
    def test_centered_gram_row_sum_zero(self, seed: int):
        """Centered Gram matrix rows should sum to approximately zero.

        Mathematical property: H @ K @ H has zero row/column sums
        where H is the centering matrix.
        """
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)

        # Gram matrix = X @ X^T
        gram = backend.matmul(x, backend.transpose(x))
        centered = _center_gram_matrix(gram, backend)

        row_sums = backend.sum(centered, axis=1)
        max_row_sum = float(backend.to_numpy(backend.max(backend.abs(row_sums))))

        assert max_row_sum < 1e-5


# =============================================================================
# CKAResult Tests
# =============================================================================


class TestCKAResult:
    """Tests for CKAResult dataclass."""

    @pytest.mark.parametrize("seed", range(10))
    def test_result_is_valid(self, seed: int):
        """CKAResult.is_valid should be True for valid inputs."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, seed)
        y = _random_matrix(backend, 20, 10, seed + 1000)

        result = compute_cka(x, y, backend)
        assert result.is_valid

    @pytest.mark.parametrize("n_samples", [5, 10, 20, 50])
    def test_sample_count_correct(self, n_samples: int):
        """sample_count should match input."""
        backend = get_default_backend()
        x = _random_matrix(backend, n_samples, 10, 42)

        result = compute_cka(x, x, backend)
        assert result.sample_count == n_samples


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
        assert result.cka == 0.0
        assert result.sample_count == 1

    def test_two_samples_valid(self):
        """Two samples should produce valid CKA."""
        backend = get_default_backend()
        x = backend.array([[1.0, 2.0], [3.0, 4.0]])
        y = backend.array([[1.1, 2.1], [3.1, 4.1]])
        result = compute_cka(x, y, backend)
        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 2

    def test_different_feature_dimensions(self):
        """CKA should work with different feature dimensions."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        y = _random_matrix(backend, 20, 15, 43)

        result = compute_cka(x, y, backend)
        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_zero_matrix_returns_zero_cka(self):
        """Zero matrix should return CKA = 0."""
        backend = get_default_backend()
        x = backend.zeros((10, 5))
        y = _random_matrix(backend, 10, 5, 42)

        result = compute_cka(x, y, backend)
        assert result.cka == 0.0

    def test_rbf_kernel_bounds(self):
        """RBF kernel CKA should also be in [0, 1]."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        y = _random_matrix(backend, 20, 10, 43)

        result = compute_cka(x, y, backend, use_linear_kernel=False)
        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_identical_with_noise(self):
        """Near-identical matrices should have high CKA."""
        backend = get_default_backend()
        x = _random_matrix(backend, 20, 10, 42)
        noise = _random_matrix(backend, 20, 10, 43) * 0.01
        y = x + noise

        result = compute_cka(x, y, backend)
        assert result.cka > 0.99

    def test_orthogonal_activations_low_cka(self):
        """Orthogonal activations should have low CKA."""
        backend = get_default_backend()

        # Create two orthogonal activation patterns using backend
        x = backend.zeros((10, 4))
        y = backend.zeros((10, 4))

        # Fill with random values in non-overlapping regions
        backend.random_seed(42)
        x_patch = backend.random_normal(shape=(5, 2))
        y_patch = backend.random_normal(shape=(5, 2))

        # Use slicing to set values - construct full arrays
        x_full = backend.zeros((10, 4))
        y_full = backend.zeros((10, 4))

        # Build x: first 5 rows, first 2 cols have random values
        x_row1 = backend.concatenate([x_patch[0:1], backend.zeros((1, 2))], axis=1)
        x_row2 = backend.concatenate([x_patch[1:2], backend.zeros((1, 2))], axis=1)
        x_row3 = backend.concatenate([x_patch[2:3], backend.zeros((1, 2))], axis=1)
        x_row4 = backend.concatenate([x_patch[3:4], backend.zeros((1, 2))], axis=1)
        x_row5 = backend.concatenate([x_patch[4:5], backend.zeros((1, 2))], axis=1)
        x_zeros = backend.zeros((5, 4))
        x = backend.concatenate(
            [x_row1, x_row2, x_row3, x_row4, x_row5, x_zeros], axis=0
        )

        # Build y: last 5 rows, last 2 cols have random values
        y_zeros = backend.zeros((5, 4))
        y_row6 = backend.concatenate([backend.zeros((1, 2)), y_patch[0:1]], axis=1)
        y_row7 = backend.concatenate([backend.zeros((1, 2)), y_patch[1:2]], axis=1)
        y_row8 = backend.concatenate([backend.zeros((1, 2)), y_patch[2:3]], axis=1)
        y_row9 = backend.concatenate([backend.zeros((1, 2)), y_patch[3:4]], axis=1)
        y_row10 = backend.concatenate([backend.zeros((1, 2)), y_patch[4:5]], axis=1)
        y = backend.concatenate(
            [y_zeros, y_row6, y_row7, y_row8, y_row9, y_row10], axis=0
        )

        result = compute_cka(x, y, backend)
        # Should be low due to orthogonal structure
        assert result.cka < 0.5
