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
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.geometry.cka import (
    _center_gram_matrix,
    _compute_pairwise_squared_distances,
    compute_cka,
)

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
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_cka_self_is_one(self, seed: int):
        """CKA(X, X) = 1.0.

        Mathematical property: HSIC(X, X) / sqrt(HSIC(X, X) * HSIC(X, X)) = 1.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, x)
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
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        # Generate random orthogonal matrix via QR
        q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
        y_rotated = y @ q.astype(np.float32)

        result_original = compute_cka(x, y)
        result_rotated = compute_cka(x, y_rotated)

        assert result_original.cka == pytest.approx(result_rotated.cka, abs=0.05)

    @pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 5.0, 10.0])
    def test_scale_invariance(self, scale: float):
        """CKA should be invariant to positive scaling.

        Mathematical property: CKA(α * X, Y) = CKA(X, Y) for α > 0.
        The normalization by sqrt(HSIC_xx) cancels out scale factors.
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        x_scaled = x * scale

        result_original = compute_cka(x, y)
        result_scaled = compute_cka(x_scaled, y)

        assert result_original.cka == pytest.approx(result_scaled.cka, abs=0.05)

    @pytest.mark.parametrize("seed", range(10))
    def test_symmetry(self, seed: int):
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X).

        Mathematical property: HSIC is symmetric in its arguments.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        result_xy = compute_cka(x, y)
        result_yx = compute_cka(y, x)

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
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, x)
        assert result.hsic_xx >= 0.0

    @pytest.mark.parametrize("seed", range(10))
    def test_hsic_xy_bounded(self, seed: int):
        """HSIC(X, Y) should satisfy Cauchy-Schwarz.

        Mathematical property: |HSIC(X, Y)| <= sqrt(HSIC(X, X) * HSIC(Y, Y)).
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, y)

        # Cauchy-Schwarz bound
        max_hsic = np.sqrt(result.hsic_xx * result.hsic_yy)
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
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        distances = _compute_pairwise_squared_distances(x)
        assert np.allclose(distances, distances.T)

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_matrix_non_negative(self, seed: int):
        """Pairwise distances should be non-negative.

        Mathematical property: ||x_i - x_j||^2 >= 0.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        distances = _compute_pairwise_squared_distances(x)
        assert np.all(distances >= -1e-10)

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_diagonal_zero(self, seed: int):
        """Diagonal of distance matrix should be zero.

        Mathematical property: ||x_i - x_i|| = 0.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        distances = _compute_pairwise_squared_distances(x)
        assert np.allclose(np.diag(distances), 0.0, atol=1e-5)

    @pytest.mark.parametrize("seed", range(10))
    def test_centered_gram_row_sum_zero(self, seed: int):
        """Centered Gram matrix rows should sum to approximately zero.

        Mathematical property: H @ K @ H has zero row/column sums
        where H is the centering matrix.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)

        gram = x @ x.T
        centered = _center_gram_matrix(gram)

        row_sums = np.sum(centered, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-5)


# =============================================================================
# CKAResult Tests
# =============================================================================


class TestCKAResult:
    """Tests for CKAResult dataclass."""

    @pytest.mark.parametrize("seed", range(10))
    def test_result_is_valid(self, seed: int):
        """CKAResult.is_valid should be True for valid inputs."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, y)
        assert result.is_valid

    @pytest.mark.parametrize("n_samples", [5, 10, 20, 50])
    def test_sample_count_correct(self, n_samples: int):
        """sample_count should match input."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((n_samples, 10)).astype(np.float32)

        result = compute_cka(x, x)
        assert result.sample_count == n_samples


# =============================================================================
# Edge Cases
# =============================================================================


class TestCKAEdgeCases:
    """Edge case tests for CKA."""

    def test_single_sample_returns_zero_cka(self):
        """Single sample should return CKA = 0."""
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = compute_cka(x, x)
        assert result.cka == 0.0
        assert result.sample_count == 1

    def test_two_samples_valid(self):
        """Two samples should produce valid CKA."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 2

    def test_different_feature_dimensions(self):
        """CKA should work with different feature dimensions."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 15)).astype(np.float32)

        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_zero_matrix_returns_zero_cka(self):
        """Zero matrix should return CKA = 0."""
        x = np.zeros((10, 5), dtype=np.float32)
        y = np.random.randn(10, 5).astype(np.float32)

        result = compute_cka(x, y)
        assert result.cka == 0.0

    def test_rbf_kernel_bounds(self):
        """RBF kernel CKA should also be in [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal((20, 10)).astype(np.float32)

        result = compute_cka(x, y, use_linear_kernel=False)
        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_identical_with_noise(self):
        """Near-identical matrices should have high CKA."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 10)).astype(np.float32)
        y = x + rng.standard_normal((20, 10)).astype(np.float32) * 0.01

        result = compute_cka(x, y)
        assert result.cka > 0.99

    def test_orthogonal_activations_low_cka(self):
        """Orthogonal activations should have low CKA."""
        # Create two orthogonal activation patterns
        x = np.zeros((10, 4), dtype=np.float32)
        x[:5, :2] = np.random.randn(5, 2)  # First 5 samples use first 2 features

        y = np.zeros((10, 4), dtype=np.float32)
        y[5:, 2:] = np.random.randn(5, 2)  # Last 5 samples use last 2 features

        result = compute_cka(x, y)
        # Should be low due to orthogonal structure
        assert result.cka < 0.5
