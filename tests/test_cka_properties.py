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
from hypothesis import given, settings, assume, strategies as st

from modelcypher.core.domain.geometry.cka import (
    compute_cka,
    CKAResult,
    _compute_hsic,
    _center_gram_matrix,
    _compute_pairwise_squared_distances,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def activation_matrix(draw, n_samples: int = 20, n_features: int = 10):
    """Generate a random activation matrix."""
    data = [
        [draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
         for _ in range(n_features)]
        for _ in range(n_samples)
    ]
    return np.array(data, dtype=np.float32)


@st.composite
def random_orthogonal_matrix(draw, size: int = 10):
    """Generate a random orthogonal matrix via QR decomposition."""
    data = [
        [draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
         for _ in range(size)]
        for _ in range(size)
    ]
    arr = np.array(data, dtype=np.float64)
    q, _ = np.linalg.qr(arr)
    return q.astype(np.float32)


@st.composite
def positive_scale(draw):
    """Generate a positive scaling factor."""
    return draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))


# =============================================================================
# CKA Bounds Tests
# =============================================================================


class TestCKABounds:
    """Tests for CKA value bounds."""

    @given(activation_matrix(), activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_cka_in_zero_one(self, x: np.ndarray, y: np.ndarray):
        """CKA must be in [0, 1].

        Mathematical property: CKA is normalized by sqrt(HSIC_xx * HSIC_yy),
        making it a correlation-like measure bounded in [0, 1].
        """
        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_cka_self_is_one(self, x: np.ndarray):
        """CKA(X, X) = 1.0.

        Mathematical property: HSIC(X, X) / sqrt(HSIC(X, X) * HSIC(X, X)) = 1.
        """
        # Ensure non-trivial matrix
        x_norm = np.linalg.norm(x)
        assume(x_norm > 1e-6)

        result = compute_cka(x, x)
        assert result.cka == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# Invariance Tests
# =============================================================================


class TestCKAInvariance:
    """Tests for CKA invariance properties."""

    @given(activation_matrix(), activation_matrix(), random_orthogonal_matrix())
    @settings(max_examples=20, deadline=None)
    def test_rotation_invariance(self, x: np.ndarray, y: np.ndarray, r: np.ndarray):
        """CKA should be invariant to orthogonal transformations.

        Mathematical property: CKA(X @ R, Y) = CKA(X, Y) for orthogonal R.
        This holds because RBF/linear kernels depend on distances/inner products,
        both preserved under orthogonal transforms.
        """
        # Ensure matrices are non-trivial
        assume(np.linalg.norm(x) > 1e-6)
        assume(np.linalg.norm(y) > 1e-6)

        # Apply rotation to one matrix
        y_rotated = y @ r

        result_original = compute_cka(x, y)
        result_rotated = compute_cka(x, y_rotated)

        assert result_original.cka == pytest.approx(result_rotated.cka, abs=0.05)

    @given(activation_matrix(), activation_matrix(), positive_scale())
    @settings(max_examples=30, deadline=None)
    def test_scale_invariance(self, x: np.ndarray, y: np.ndarray, scale: float):
        """CKA should be invariant to positive scaling.

        Mathematical property: CKA(α * X, Y) = CKA(X, Y) for α > 0.
        The normalization by sqrt(HSIC_xx) cancels out scale factors.
        """
        assume(np.linalg.norm(x) > 1e-6)
        assume(np.linalg.norm(y) > 1e-6)

        x_scaled = x * scale

        result_original = compute_cka(x, y)
        result_scaled = compute_cka(x_scaled, y)

        assert result_original.cka == pytest.approx(result_scaled.cka, abs=0.05)

    @given(activation_matrix(), activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_symmetry(self, x: np.ndarray, y: np.ndarray):
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X).

        Mathematical property: HSIC is symmetric in its arguments.
        """
        result_xy = compute_cka(x, y)
        result_yx = compute_cka(y, x)

        assert result_xy.cka == pytest.approx(result_yx.cka, abs=1e-6)


# =============================================================================
# HSIC Tests
# =============================================================================


class TestHSIC:
    """Tests for HSIC (Hilbert-Schmidt Independence Criterion)."""

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_hsic_self_non_negative(self, x: np.ndarray):
        """HSIC(X, X) >= 0.

        Mathematical property: HSIC is a squared norm in RKHS,
        hence always non-negative.
        """
        result = compute_cka(x, x)
        assert result.hsic_xx >= 0.0

    @given(activation_matrix(), activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_hsic_xy_bounded(self, x: np.ndarray, y: np.ndarray):
        """HSIC(X, Y) should satisfy Cauchy-Schwarz.

        Mathematical property: |HSIC(X, Y)| <= sqrt(HSIC(X, X) * HSIC(Y, Y)).
        """
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

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_distance_matrix_symmetric(self, x: np.ndarray):
        """Pairwise distance matrix should be symmetric.

        Mathematical property: ||x_i - x_j|| = ||x_j - x_i||.
        """
        distances = _compute_pairwise_squared_distances(x)
        assert np.allclose(distances, distances.T)

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_distance_matrix_non_negative(self, x: np.ndarray):
        """Pairwise distances should be non-negative.

        Mathematical property: ||x_i - x_j||^2 >= 0.
        """
        distances = _compute_pairwise_squared_distances(x)
        assert np.all(distances >= -1e-10)  # Small tolerance for floating point

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_distance_diagonal_zero(self, x: np.ndarray):
        """Diagonal of distance matrix should be zero.

        Mathematical property: ||x_i - x_i|| = 0.
        """
        distances = _compute_pairwise_squared_distances(x)
        # Diagonal should be zero (self-distance)
        assert np.allclose(np.diag(distances), 0.0, atol=1e-5)

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_centered_gram_row_sum_zero(self, x: np.ndarray):
        """Centered Gram matrix rows should sum to approximately zero.

        Mathematical property: H @ K @ H has zero row/column sums
        where H is the centering matrix.
        """
        # Need non-trivial matrix for meaningful centering
        assume(np.linalg.norm(x) > 1e-6)

        gram = x @ x.T
        centered = _center_gram_matrix(gram)

        row_sums = np.sum(centered, axis=1)
        # Use higher tolerance for float32 operations
        assert np.allclose(row_sums, 0.0, atol=1e-4)


# =============================================================================
# CKAResult Tests
# =============================================================================


class TestCKAResult:
    """Tests for CKAResult dataclass."""

    @given(activation_matrix(), activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_result_is_valid(self, x: np.ndarray, y: np.ndarray):
        """CKAResult.is_valid should be True for valid inputs."""
        result = compute_cka(x, y)
        assert result.is_valid

    @given(activation_matrix())
    @settings(max_examples=30, deadline=None)
    def test_sample_count_correct(self, x: np.ndarray):
        """sample_count should match input."""
        result = compute_cka(x, x)
        assert result.sample_count == x.shape[0]


# =============================================================================
# Edge Cases
# =============================================================================


class TestCKAEdgeCases:
    """Edge case tests for CKA."""

    def test_single_sample_returns_zero_cka(self):
        """Single sample should return CKA = 0."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = compute_cka(x, x)
        assert result.cka == 0.0
        assert result.sample_count == 1

    def test_two_samples_valid(self):
        """Two samples should produce valid CKA."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.1, 2.1], [3.1, 4.1]])
        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 2

    def test_different_feature_dimensions(self):
        """CKA should work with different feature dimensions."""
        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 15).astype(np.float32)

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
        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 10).astype(np.float32)

        result = compute_cka(x, y, use_linear_kernel=False)
        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid
