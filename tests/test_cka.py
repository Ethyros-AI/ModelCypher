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

"""Comprehensive tests for CKA (Centered Kernel Alignment) module.

Tests cover:
- HSICEstimator enum values
- CKAResult dataclass validation and properties
- HSIC computation (biased and unbiased)
- Gram matrix centering
- CKA invariance properties (rotation, scale, permutation)
- Edge cases (small samples, empty inputs, numerical stability)
- Feature bias correction (Chun et al., 2025)
- Cross-dimensional comparison via Gram matrices
- CKAComputer class operations
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.cka import (
    CKAComputer,
    CKAResult,
    HSICEstimator,
    _center_gram_matrix,
    _compute_hsic,
    _compute_hsic_dispatch,
    _compute_hsic_unbiased,
    _feature_sampling_correction,
    _participation_ratio,
    compute_cka,
    compute_cka_backend,
    compute_cka_from_grams,
    compute_cka_from_lists,
    compute_cka_matrix,
    compute_layer_cka,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _scalar_tol(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _array_tol(backend: "Backend", array) -> float:
    return division_epsilon(backend, array)


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
        assert HSICEstimator.BIASED in members
        assert HSICEstimator.UNBIASED in members
        assert HSICEstimator.AUTO in members

    def test_enum_is_string_enum(self) -> None:
        """Verify HSICEstimator is a string enum."""
        assert isinstance(HSICEstimator.BIASED, str)
        assert HSICEstimator.BIASED == "biased"


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
        assert result.hsic_xx == 0.6
        assert result.hsic_yy == 0.65
        assert result.sample_count == 100
        assert result.cka_corrected is None
        assert result.correction_factor is None

    def test_full_construction(self) -> None:
        """Test full CKAResult construction with all fields."""
        result = CKAResult(
            cka=0.85,
            hsic_xy=0.5,
            hsic_xx=0.6,
            hsic_yy=0.65,
            sample_count=100,
            cka_corrected=0.92,
            correction_factor=1.08,
            intrinsic_dim_x=15.3,
            intrinsic_dim_y=18.7,
            feature_dim_x=256,
            feature_dim_y=512,
        )
        assert result.cka_corrected == 0.92
        assert result.correction_factor == 1.08
        assert result.intrinsic_dim_x == 15.3
        assert result.intrinsic_dim_y == 18.7
        assert result.feature_dim_x == 256
        assert result.feature_dim_y == 512

    def test_is_valid_true(self) -> None:
        """Test is_valid property for valid results."""
        result = CKAResult(cka=0.5, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50)
        assert result.is_valid is True

    def test_is_valid_false_nan_cka(self) -> None:
        """Test is_valid property for NaN CKA."""
        result = CKAResult(
            cka=float("nan"), hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result.is_valid is False

    def test_is_valid_false_inf_hsic(self) -> None:
        """Test is_valid property for infinite HSIC."""
        result = CKAResult(
            cka=0.5, hsic_xy=float("inf"), hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result.is_valid is False

    def test_is_valid_false_cka_out_of_range(self) -> None:
        """Test is_valid property for CKA out of [0, 1] range."""
        result = CKAResult(cka=1.5, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50)
        assert result.is_valid is False

        result_neg = CKAResult(
            cka=-0.1, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result_neg.is_valid is False

    def test_is_valid_boundary_values(self) -> None:
        """Test is_valid at boundary values 0 and 1."""
        result_zero = CKAResult(
            cka=0.0, hsic_xy=0.0, hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result_zero.is_valid is True

        result_one = CKAResult(
            cka=1.0, hsic_xy=0.4, hsic_xx=0.4, hsic_yy=0.4, sample_count=50
        )
        assert result_one.is_valid is True

    def test_best_property_no_correction(self) -> None:
        """Test best property returns raw CKA when no correction."""
        result = CKAResult(cka=0.75, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50)
        assert result.best == 0.75

    def test_best_property_with_correction(self) -> None:
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

    def test_frozen_dataclass(self) -> None:
        """Test that CKAResult is frozen (immutable)."""
        result = CKAResult(cka=0.5, hsic_xy=0.3, hsic_xx=0.4, hsic_yy=0.4, sample_count=50)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.cka = 0.9  # type: ignore


# =============================================================================
# Gram Matrix Centering Tests
# =============================================================================


class TestCenterGramMatrix:
    """Tests for _center_gram_matrix function."""

    def test_identity_matrix_centering(self, any_backend: "Backend") -> None:
        """Centering identity matrix should produce specific pattern."""
        backend = any_backend
        n = 5
        gram = backend.eye(n)
        centered = _center_gram_matrix(gram, backend)

        # Centered identity: K - col_mean - row_mean + grand_mean
        # For identity: grand_mean = 1/n, col/row means are 1/n each
        # Diagonal entries: 1 - 1/n - 1/n + 1/n = 1 - 1/n
        # Off-diagonal: 0 - 1/n - 1/n + 1/n = -1/n

        centered_np = backend.to_numpy(centered)
        expected_diag = 1.0 - 1.0 / n
        expected_off = -1.0 / n
        tol = _array_tol(backend, centered)

        for i in range(n):
            for j in range(n):
                if i == j:
                    assert abs(centered_np[i, j] - expected_diag) <= tol
                else:
                    assert abs(centered_np[i, j] - expected_off) <= tol

    def test_centering_zeros(self, any_backend: "Backend") -> None:
        """Centering zero matrix should stay zero."""
        backend = any_backend
        gram = backend.zeros((4, 4))
        centered = _center_gram_matrix(gram, backend)

        centered_np = backend.to_numpy(centered)
        tol = _array_tol(backend, centered)
        assert abs(centered_np.sum()) <= tol

    def test_centering_ones(self, any_backend: "Backend") -> None:
        """Centering all-ones matrix should produce all zeros."""
        backend = any_backend
        n = 4
        gram = backend.ones((n, n))
        centered = _center_gram_matrix(gram, backend)

        # All-ones: col_mean = 1, row_mean = 1, grand_mean = 1
        # Centered = 1 - 1 - 1 + 1 = 0
        centered_np = backend.to_numpy(centered)
        tol = _array_tol(backend, centered)
        assert abs(centered_np.sum()) <= tol

    def test_centering_preserves_shape(self, any_backend: "Backend") -> None:
        """Centering should preserve matrix shape."""
        backend = any_backend
        backend.random_seed(42)
        gram = backend.random_normal((7, 7))
        centered = _center_gram_matrix(gram, backend)

        assert centered.shape == gram.shape

    def test_centering_double_centering_idempotent(self, any_backend: "Backend") -> None:
        """Centering an already centered matrix should be idempotent."""
        backend = any_backend
        backend.random_seed(42)
        gram = backend.random_normal((6, 6))

        centered_once = _center_gram_matrix(gram, backend)
        centered_twice = _center_gram_matrix(centered_once, backend)

        c1 = backend.to_numpy(centered_once)
        c2 = backend.to_numpy(centered_twice)

        # Should be numerically identical (up to floating point)
        tol = _array_tol(backend, centered_once)
        assert abs(c1 - c2).max() <= tol

    def test_centering_row_and_col_sums_zero(self, any_backend: "Backend") -> None:
        """Centered Gram matrix should have row and column sums = 0."""
        backend = any_backend
        backend.random_seed(42)

        # Create a symmetric positive semi-definite Gram matrix
        X = backend.random_normal((5, 8))
        gram = backend.matmul(X, backend.transpose(X))
        centered = _center_gram_matrix(gram, backend)

        centered_np = backend.to_numpy(centered)
        row_sums = centered_np.sum(axis=1)
        col_sums = centered_np.sum(axis=0)

        tol = _array_tol(backend, centered)
        assert abs(row_sums).max() <= tol
        assert abs(col_sums).max() <= tol

    def test_centering_empty_matrix(self, any_backend: "Backend") -> None:
        """Centering empty matrix should return empty."""
        backend = any_backend
        gram = backend.zeros((0, 0))
        centered = _center_gram_matrix(gram, backend)
        assert centered.shape == (0, 0)


# =============================================================================
# HSIC Computation Tests
# =============================================================================


class TestComputeHSIC:
    """Tests for HSIC computation functions."""

    def test_hsic_identical_grams(self, any_backend: "Backend") -> None:
        """HSIC of identical Gram matrices should be positive."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic(gram, gram, backend)
        assert hsic > 0.0

    def test_hsic_different_grams(self, any_backend: "Backend") -> None:
        """HSIC of different Gram matrices should be finite."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        Y = backend.random_normal((10, 8))
        gram_x = backend.matmul(X, backend.transpose(X))
        gram_y = backend.matmul(Y, backend.transpose(Y))

        hsic = _compute_hsic(gram_x, gram_y, backend)
        assert math.isfinite(hsic)

    def test_hsic_single_sample_returns_zero(self, any_backend: "Backend") -> None:
        """HSIC with n=1 should return 0."""
        backend = any_backend
        gram = backend.ones((1, 1))
        hsic = _compute_hsic(gram, gram, backend)
        assert hsic == 0.0

    def test_hsic_two_samples(self, any_backend: "Backend") -> None:
        """HSIC with n=2 should return finite value."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((2, 5))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic(gram, gram, backend)
        assert math.isfinite(hsic)

    def test_hsic_zero_gram(self, any_backend: "Backend") -> None:
        """HSIC with zero Gram matrix should return 0."""
        backend = any_backend
        gram = backend.zeros((5, 5))
        hsic = _compute_hsic(gram, gram, backend)
        assert hsic == 0.0


class TestComputeHSICUnbiased:
    """Tests for unbiased HSIC estimator (Song et al., 2012)."""

    def test_unbiased_requires_n_ge_4(self, any_backend: "Backend") -> None:
        """Unbiased HSIC requires n >= 4."""
        backend = any_backend

        # n=3 should return 0
        gram_3 = backend.ones((3, 3))
        hsic_3 = _compute_hsic_unbiased(gram_3, gram_3, backend)
        assert hsic_3 == 0.0

        # n=4 should compute
        backend.random_seed(42)
        X_4 = backend.random_normal((4, 5))
        gram_4 = backend.matmul(X_4, backend.transpose(X_4))
        hsic_4 = _compute_hsic_unbiased(gram_4, gram_4, backend)
        assert hsic_4 >= 0.0

    def test_unbiased_identical_grams_positive(self, any_backend: "Backend") -> None:
        """Unbiased HSIC of identical Gram matrices should be non-negative."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic_unbiased(gram, gram, backend)
        assert hsic >= 0.0

    def test_unbiased_non_negative(self, any_backend: "Backend") -> None:
        """Unbiased HSIC should be clamped to non-negative."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((8, 5))
        Y = backend.random_normal((8, 5))
        gram_x = backend.matmul(X, backend.transpose(X))
        gram_y = backend.matmul(Y, backend.transpose(Y))

        hsic = _compute_hsic_unbiased(gram_x, gram_y, backend)
        assert hsic >= 0.0

    def test_unbiased_uses_zero_diagonal(self, any_backend: "Backend") -> None:
        """Unbiased HSIC zeros out diagonal before computation."""
        backend = any_backend
        # Create a diagonal-only matrix
        gram = backend.eye(5) * 10.0

        # With diagonal zeroed, all entries become 0, so HSIC should be 0
        hsic = _compute_hsic_unbiased(gram, gram, backend)
        assert hsic == 0.0


class TestComputeHSICDispatch:
    """Tests for HSIC dispatch function."""

    def test_dispatch_biased(self, any_backend: "Backend") -> None:
        """Dispatch with BIASED should use biased estimator."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((6, 5))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic_dispatch(
            gram, gram, backend, HSICEstimator.BIASED, n_features_x=5, n_features_y=5
        )
        assert hsic >= 0.0

    def test_dispatch_unbiased(self, any_backend: "Backend") -> None:
        """Dispatch with UNBIASED should use unbiased estimator."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((6, 5))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic_dispatch(
            gram, gram, backend, HSICEstimator.UNBIASED, n_features_x=5, n_features_y=5
        )
        assert hsic >= 0.0

    def test_dispatch_unbiased_fallback_for_small_n(self, any_backend: "Backend") -> None:
        """Dispatch with UNBIASED but n<4 should fall back to biased."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((3, 5))
        gram = backend.matmul(X, backend.transpose(X))

        # UNBIASED with n=3 should fall back to biased
        hsic = _compute_hsic_dispatch(
            gram, gram, backend, HSICEstimator.UNBIASED, n_features_x=5, n_features_y=5
        )
        # Should use biased estimator, which works for n=3
        assert math.isfinite(hsic)

    def test_dispatch_auto_uses_unbiased_high_dim(self, any_backend: "Backend") -> None:
        """Dispatch with AUTO should use UNBIASED when features >> samples."""
        backend = any_backend
        backend.random_seed(42)
        n_samples = 5
        n_features = 100  # features >> samples
        X = backend.random_normal((n_samples, n_features))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic_dispatch(
            gram,
            gram,
            backend,
            HSICEstimator.AUTO,
            n_features_x=n_features,
            n_features_y=n_features,
        )
        assert hsic >= 0.0

    def test_dispatch_auto_uses_biased_low_dim(self, any_backend: "Backend") -> None:
        """Dispatch with AUTO should use BIASED when samples >= features."""
        backend = any_backend
        backend.random_seed(42)
        n_samples = 20
        n_features = 5  # samples > features
        X = backend.random_normal((n_samples, n_features))
        gram = backend.matmul(X, backend.transpose(X))

        hsic = _compute_hsic_dispatch(
            gram,
            gram,
            backend,
            HSICEstimator.AUTO,
            n_features_x=n_features,
            n_features_y=n_features,
        )
        assert hsic >= 0.0


# =============================================================================
# Feature Sampling Correction Tests
# =============================================================================


class TestParticipationRatio:
    """Tests for participation ratio (effective rank)."""

    def test_uniform_eigenvalues(self, any_backend: "Backend") -> None:
        """Uniform eigenvalues should give participation ratio = n."""
        backend = any_backend
        # 5 equal eigenvalues of 1.0
        eigvals = backend.ones((5,))
        pr = _participation_ratio(eigvals, backend)

        # sum^2 / sum_sq = (5*1)^2 / (5*1^2) = 25/5 = 5
        tol = _array_tol(backend, eigvals)
        assert abs(pr - 5.0) <= tol

    def test_single_nonzero_eigenvalue(self, any_backend: "Backend") -> None:
        """Single non-zero eigenvalue should give participation ratio = 1."""
        backend = any_backend
        eigvals = backend.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pr = _participation_ratio(eigvals, backend)

        # sum^2 / sum_sq = 1^2 / 1^2 = 1
        tol = _array_tol(backend, eigvals)
        assert abs(pr - 1.0) <= tol

    def test_negative_eigenvalues_clamped(self, any_backend: "Backend") -> None:
        """Negative eigenvalues should be clamped to zero."""
        backend = any_backend
        eigvals = backend.array([1.0, 1.0, -0.5, -0.1])
        pr = _participation_ratio(eigvals, backend)

        # After clamping: [1, 1, 0, 0]
        # sum^2 / sum_sq = 4 / 2 = 2
        tol = _array_tol(backend, eigvals)
        assert abs(pr - 2.0) <= tol

    def test_all_zero_eigenvalues(self, any_backend: "Backend") -> None:
        """All zero eigenvalues should give participation ratio = 0."""
        backend = any_backend
        eigvals = backend.zeros((5,))
        pr = _participation_ratio(eigvals, backend)
        assert pr == 0.0


class TestFeatureSamplingCorrection:
    """Tests for feature-sampling bias correction."""

    def test_correction_with_valid_gram(self, any_backend: "Backend") -> None:
        """Correction should return factor >= 1.0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 20))
        gram = backend.matmul(X, backend.transpose(X))
        centered = _center_gram_matrix(gram, backend)

        correction, intrinsic_dim = _feature_sampling_correction(centered, 20, backend)

        assert correction >= 1.0
        assert intrinsic_dim >= 0.0

    def test_correction_zero_feature_dim(self, any_backend: "Backend") -> None:
        """Zero feature dimension should return correction = 1.0."""
        backend = any_backend
        gram = backend.eye(5)
        centered = _center_gram_matrix(gram, backend)

        correction, intrinsic_dim = _feature_sampling_correction(centered, 0, backend)
        assert correction == 1.0
        assert intrinsic_dim == 0.0

    def test_correction_negative_feature_dim(self, any_backend: "Backend") -> None:
        """Negative feature dimension should return correction = 1.0."""
        backend = any_backend
        gram = backend.eye(5)
        centered = _center_gram_matrix(gram, backend)

        correction, _ = _feature_sampling_correction(centered, -5, backend)
        assert correction == 1.0


# =============================================================================
# compute_cka Tests
# =============================================================================


class TestComputeCKA:
    """Tests for main compute_cka function."""

    def test_identical_activations_cka_one(self, any_backend: "Backend") -> None:
        """CKA of identical activations should be 1.0."""
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
        assert 0.0 <= result.cka <= 1.0

    def test_cka_sample_count_mismatch_error(self, any_backend: "Backend") -> None:
        """CKA should raise ValueError for sample count mismatch."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        Y = backend.random_normal((15, 8))

        with pytest.raises(ValueError, match="Sample count mismatch"):
            compute_cka(X, Y, backend)

    def test_cka_single_sample_returns_zero(self, any_backend: "Backend") -> None:
        """CKA with n=1 should return 0.0."""
        backend = any_backend
        X = backend.random_normal((1, 8))
        Y = backend.random_normal((1, 8))

        result = compute_cka(X, Y, backend)

        assert result.cka == 0.0
        assert result.sample_count == 1

    def test_cka_linear_kernel(self, any_backend: "Backend") -> None:
        """CKA with linear kernel should work."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((12, 8))
        Y = backend.random_normal((12, 8))

        result = compute_cka(X, Y, backend, use_linear_kernel=True)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_cka_rbf_kernel(self, any_backend: "Backend") -> None:
        """CKA with RBF kernel should work."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((12, 8))
        Y = backend.random_normal((12, 8))

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

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
        assert result.cka_corrected >= result.cka  # Correction increases CKA

    def test_cka_returns_sample_count(self, any_backend: "Backend") -> None:
        """CKA result should contain correct sample count."""
        backend = any_backend
        backend.random_seed(42)
        n_samples = 25
        X = backend.random_normal((n_samples, 10))
        Y = backend.random_normal((n_samples, 10))

        result = compute_cka(X, Y, backend)

        assert result.sample_count == n_samples

    def test_cka_unbiased_estimator(self, any_backend: "Backend") -> None:
        """CKA with unbiased estimator should work."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        Y = backend.random_normal((10, 8))

        result = compute_cka(X, Y, backend, estimator=HSICEstimator.UNBIASED)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_cka_auto_estimator(self, any_backend: "Backend") -> None:
        """CKA with AUTO estimator should work."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 50))  # features > samples
        Y = backend.random_normal((10, 50))

        result = compute_cka(X, Y, backend, estimator=HSICEstimator.AUTO)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_cka_cross_dimensional(self, any_backend: "Backend") -> None:
        """CKA should work with different feature dimensions."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))  # 10 features
        Y = backend.random_normal((15, 20))  # 20 features

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0
        assert result.feature_dim_x == 10
        assert result.feature_dim_y == 20


# =============================================================================
# CKA Invariance Properties Tests
# =============================================================================


class TestCKAInvarianceProperties:
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

        # Rotate X
        X_rotated = backend.matmul(X, Q)
        backend.eval(X_rotated)

        # CKA should be invariant
        result_original = compute_cka(X, Y, backend)
        result_rotated = compute_cka(X_rotated, Y, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_rotated.cka) <= tol

    def test_scale_invariance(self, any_backend: "Backend") -> None:
        """CKA should be invariant to scaling."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        Y = backend.random_normal((15, 8))

        # Scale X by constant
        scale = 5.0
        X_scaled = X * scale
        backend.eval(X_scaled)

        result_original = compute_cka(X, Y, backend)
        result_scaled = compute_cka(X_scaled, Y, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_scaled.cka) <= tol

    def test_permutation_invariance(self, any_backend: "Backend") -> None:
        """CKA should be invariant to sample permutation (when applied to both)."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        Y = backend.random_normal((15, 8))

        # Permute both X and Y with same permutation
        perm = [7, 2, 11, 0, 5, 3, 14, 8, 1, 12, 4, 9, 6, 13, 10]
        X_perm = backend.array([backend.to_numpy(X[i]) for i in perm])
        Y_perm = backend.array([backend.to_numpy(Y[i]) for i in perm])

        result_original = compute_cka(X, Y, backend)
        result_permuted = compute_cka(X_perm, Y_perm, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_permuted.cka) <= tol

    def test_symmetry(self, any_backend: "Backend") -> None:
        """CKA(X, Y) should equal CKA(Y, X)."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        Y = backend.random_normal((15, 10))

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        tol = _scalar_tol(backend)
        assert abs(result_xy.cka - result_yx.cka) <= tol


# =============================================================================
# compute_cka_matrix Tests
# =============================================================================


class TestComputeCKAMatrix:
    """Tests for compute_cka_matrix function."""

    def test_basic_matrix_computation(self, any_backend: "Backend") -> None:
        """Compute CKA matrix between activation sets."""
        backend = any_backend
        backend.random_seed(42)

        source = {
            "probe1": backend.random_normal((10, 8)),
            "probe2": backend.random_normal((10, 8)),
        }
        target = {
            "probeA": backend.random_normal((10, 8)),
            "probeB": backend.random_normal((10, 8)),
        }

        matrix, src_ids, tgt_ids = compute_cka_matrix(source, target, backend)

        assert matrix.shape == (2, 2)
        assert src_ids == ["probe1", "probe2"]
        assert tgt_ids == ["probeA", "probeB"]

    def test_empty_source(self, any_backend: "Backend") -> None:
        """Empty source should return empty matrix."""
        backend = any_backend
        backend.random_seed(42)

        source: dict = {}
        target = {"probeA": backend.random_normal((10, 8))}

        matrix, src_ids, tgt_ids = compute_cka_matrix(source, target, backend)

        assert matrix.shape == (0, 0)
        assert src_ids == []
        assert tgt_ids == []

    def test_empty_target(self, any_backend: "Backend") -> None:
        """Empty target should return empty matrix."""
        backend = any_backend
        backend.random_seed(42)

        source = {"probe1": backend.random_normal((10, 8))}
        target: dict = {}

        matrix, src_ids, tgt_ids = compute_cka_matrix(source, target, backend)

        assert matrix.shape == (0, 0)
        assert src_ids == []
        assert tgt_ids == []

    def test_matrix_values_in_range(self, any_backend: "Backend") -> None:
        """All CKA matrix values should be in [0, 1]."""
        backend = any_backend
        backend.random_seed(42)

        source = {
            f"probe{i}": backend.random_normal((10, 8)) for i in range(3)
        }
        target = {
            f"target{i}": backend.random_normal((10, 8)) for i in range(3)
        }

        matrix, _, _ = compute_cka_matrix(source, target, backend)

        matrix_np = backend.to_numpy(matrix)
        assert (matrix_np >= 0.0).all()
        assert (matrix_np <= 1.0).all()

    def test_insufficient_samples_returns_zero(self, any_backend: "Backend") -> None:
        """CKA matrix should return 0 for probes with < 2 samples."""
        backend = any_backend
        backend.random_seed(42)

        source = {"probe1": backend.random_normal((1, 8))}  # Only 1 sample
        target = {"probeA": backend.random_normal((10, 8))}

        matrix, _, _ = compute_cka_matrix(source, target, backend)

        matrix_np = backend.to_numpy(matrix)
        assert matrix_np[0, 0] == 0.0


# =============================================================================
# compute_layer_cka Tests
# =============================================================================


class TestComputeLayerCKA:
    """Tests for compute_layer_cka function."""

    def test_same_shape_weights(self, any_backend: "Backend") -> None:
        """CKA between same-shape weight matrices."""
        backend = any_backend
        backend.random_seed(42)
        source = backend.random_normal((64, 32))
        target = backend.random_normal((64, 32))

        result = compute_layer_cka(source, target, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_identical_weights_cka_one(self, any_backend: "Backend") -> None:
        """Identical weights should have CKA = 1.0."""
        backend = any_backend
        backend.random_seed(42)
        weights = backend.random_normal((64, 32))

        result = compute_layer_cka(weights, weights, backend)

        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol

    def test_different_shape_weights_aligned(self, any_backend: "Backend") -> None:
        """Weights with different shapes should be aligned to common dimensions."""
        backend = any_backend
        backend.random_seed(42)
        source = backend.random_normal((64, 32))
        target = backend.random_normal((48, 24))

        result = compute_layer_cka(source, target, backend)

        assert result.is_valid
        # Aligned to min(64, 48) x min(32, 24) = 48 x 24
        assert result.sample_count == 48

    def test_layer_cka_with_bias_correction(self, any_backend: "Backend") -> None:
        """Layer CKA with feature bias correction."""
        backend = any_backend
        backend.random_seed(42)
        source = backend.random_normal((32, 64))
        target = backend.random_normal((32, 64))

        result = compute_layer_cka(
            source, target, backend, feature_bias_correction=True
        )

        assert result.is_valid
        assert result.cka_corrected is not None


# =============================================================================
# compute_cka_backend Tests
# =============================================================================


class TestComputeCKABackend:
    """Tests for compute_cka_backend function."""

    def test_identical_arrays_returns_one(self, any_backend: "Backend") -> None:
        """Identical arrays should return CKA = 1.0."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))

        cka = compute_cka_backend(X, X, backend)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_basic_computation(self, any_backend: "Backend") -> None:
        """Basic CKA computation should work."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        cka = compute_cka_backend(X, Y, backend)

        assert 0.0 <= cka <= 1.0

    def test_insufficient_samples_returns_zero(self, any_backend: "Backend") -> None:
        """Less than 2 samples should return 0.0."""
        backend = any_backend
        X = backend.random_normal((1, 10))
        Y = backend.random_normal((1, 10))

        cka = compute_cka_backend(X, Y, backend)
        assert cka == 0.0

    def test_with_unbiased_estimator(self, any_backend: "Backend") -> None:
        """CKA backend with unbiased estimator."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 8))
        Y = backend.random_normal((10, 8))

        cka = compute_cka_backend(X, Y, backend, estimator=HSICEstimator.UNBIASED)

        assert 0.0 <= cka <= 1.0

    def test_with_feature_bias_correction(self, any_backend: "Backend") -> None:
        """CKA backend with feature bias correction."""
        backend = any_backend
        backend.random_seed(42)
        X = backend.random_normal((10, 30))
        Y = backend.random_normal((10, 30))

        cka = compute_cka_backend(X, Y, backend, feature_bias_correction=True)

        assert 0.0 <= cka <= 1.0


# =============================================================================
# compute_cka_from_lists Tests
# =============================================================================


class TestComputeCKAFromLists:
    """Tests for compute_cka_from_lists function."""

    def test_basic_list_computation(self, any_backend: "Backend") -> None:
        """Compute CKA from nested lists."""
        backend = any_backend

        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        y = [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]]

        cka = compute_cka_from_lists(x, y, backend)

        assert 0.0 <= cka <= 1.0

    def test_identical_lists_returns_one(self, any_backend: "Backend") -> None:
        """Identical lists should return CKA close to 1.0."""
        backend = any_backend

        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        cka = compute_cka_from_lists(x, x, backend)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_insufficient_samples(self, any_backend: "Backend") -> None:
        """Less than 2 samples should return 0.0."""
        backend = any_backend

        x = [[1.0, 2.0]]
        y = [[3.0, 4.0]]

        cka = compute_cka_from_lists(x, y, backend)
        assert cka == 0.0

    def test_different_list_lengths(self, any_backend: "Backend") -> None:
        """Lists with different lengths should use minimum."""
        backend = any_backend

        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]  # 4 samples
        y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3 samples

        cka = compute_cka_from_lists(x, y, backend)

        assert 0.0 <= cka <= 1.0


# =============================================================================
# compute_cka_from_grams Tests
# =============================================================================


class TestComputeCKAFromGrams:
    """Tests for compute_cka_from_grams function."""

    def test_from_2d_grams(self, any_backend: "Backend") -> None:
        """Compute CKA from 2D Gram matrices."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((10, 8))
        Y = backend.random_normal((10, 8))
        gram_a = backend.matmul(X, backend.transpose(X))
        gram_b = backend.matmul(Y, backend.transpose(Y))

        cka = compute_cka_from_grams(gram_a, gram_b, backend=backend)

        assert 0.0 <= cka <= 1.0

    def test_from_flattened_grams(self, any_backend: "Backend") -> None:
        """Compute CKA from flattened Gram matrices."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((5, 8))
        gram = backend.matmul(X, backend.transpose(X))
        gram_flat = backend.reshape(gram, (-1,))

        cka = compute_cka_from_grams(gram_flat, gram_flat, n=5, backend=backend)

        # Same Gram matrix should give CKA = 1.0
        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_identical_grams_returns_one(self, any_backend: "Backend") -> None:
        """Identical Gram matrices should return CKA = 1.0."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((8, 6))
        gram = backend.matmul(X, backend.transpose(X))

        cka = compute_cka_from_grams(gram, gram, backend=backend)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_mismatched_gram_shapes(self, any_backend: "Backend") -> None:
        """Mismatched Gram shapes should return 0.0."""
        backend = any_backend

        gram_a = backend.zeros((5, 5))
        gram_b = backend.zeros((6, 6))

        cka = compute_cka_from_grams(gram_a, gram_b, backend=backend)

        assert cka == 0.0

    def test_non_square_gram_returns_zero(self, any_backend: "Backend") -> None:
        """Non-square Gram matrices should return 0.0."""
        backend = any_backend

        gram_a = backend.zeros((5, 5))
        gram_b = backend.zeros((5, 6))

        cka = compute_cka_from_grams(gram_a, gram_b, backend=backend)

        assert cka == 0.0

    def test_single_sample_gram_returns_zero(self, any_backend: "Backend") -> None:
        """Gram matrices with n=1 should return 0.0."""
        backend = any_backend

        gram = backend.ones((1, 1))
        cka = compute_cka_from_grams(gram, gram, backend=backend)

        assert cka == 0.0

    def test_from_python_lists(self, any_backend: "Backend") -> None:
        """Compute CKA from Python lists representing Gram matrices."""
        backend = any_backend

        # 3x3 Gram matrix as flat list
        gram_list = [1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0]

        cka = compute_cka_from_grams(gram_list, gram_list, n=3, backend=backend)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_with_unbiased_estimator(self, any_backend: "Backend") -> None:
        """CKA from Grams with unbiased estimator."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((8, 6))
        gram = backend.matmul(X, backend.transpose(X))

        cka = compute_cka_from_grams(
            gram, gram, backend=backend, estimator=HSICEstimator.UNBIASED
        )

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_with_feature_bias_correction(self, any_backend: "Backend") -> None:
        """CKA from Grams with feature bias correction."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((8, 20))
        gram = backend.matmul(X, backend.transpose(X))

        cka = compute_cka_from_grams(
            gram,
            gram,
            backend=backend,
            feature_dim_a=20,
            feature_dim_b=20,
            feature_bias_correction=True,
        )

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol


# =============================================================================
# CKAComputer Class Tests
# =============================================================================


class TestCKAComputer:
    """Tests for CKAComputer class."""

    def test_initialization_default(self, any_backend: "Backend") -> None:
        """Test default initialization."""
        computer = CKAComputer(any_backend)

        assert computer.backend == any_backend
        assert computer.estimator == HSICEstimator.BIASED

    def test_initialization_with_estimator(self, any_backend: "Backend") -> None:
        """Test initialization with custom estimator."""
        computer = CKAComputer(any_backend, estimator=HSICEstimator.UNBIASED)

        assert computer.estimator == HSICEstimator.UNBIASED

    def test_linear_cka(self, any_backend: "Backend") -> None:
        """Test linear_cka method."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        cka = computer.linear_cka(X, Y)

        assert 0.0 <= cka <= 1.0

    def test_linear_cka_identical(self, any_backend: "Backend") -> None:
        """Linear CKA of identical arrays should be 1.0."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((15, 10))

        cka = computer.linear_cka(X, X)

        tol = _scalar_tol(backend)
        assert abs(cka - 1.0) <= tol

    def test_rbf_cka(self, any_backend: "Backend") -> None:
        """Test rbf_cka method."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        cka = computer.rbf_cka(X, Y)

        assert 0.0 <= cka <= 1.0

    def test_full_returns_cka_result(self, any_backend: "Backend") -> None:
        """Test full method returns CKAResult."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        result = computer.full(X, Y)

        assert isinstance(result, CKAResult)
        assert result.is_valid

    def test_full_with_rbf_kernel(self, any_backend: "Backend") -> None:
        """Test full method with RBF kernel."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        result = computer.full(X, Y, use_linear=False)

        assert isinstance(result, CKAResult)
        assert result.is_valid

    def test_from_grams(self, any_backend: "Backend") -> None:
        """Test from_grams method."""
        backend = any_backend
        backend.random_seed(42)

        computer = CKAComputer(backend)
        X = backend.random_normal((10, 8))
        gram = backend.matmul(X, backend.transpose(X))

        cka = computer.from_grams(gram, gram)

        assert abs(cka - 1.0) < 1e-4


# =============================================================================
# Edge Cases and Numerical Stability Tests
# =============================================================================


class TestEdgeCasesAndNumericalStability:
    """Tests for edge cases and numerical stability."""

    def test_very_small_values(self, any_backend: "Backend") -> None:
        """CKA should handle very small activation values."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((10, 8)) * 1e-10
        Y = backend.random_normal((10, 8)) * 1e-10

        result = compute_cka(X, Y, backend)

        # Should handle gracefully
        assert math.isfinite(result.cka)

    def test_very_large_values(self, any_backend: "Backend") -> None:
        """CKA should handle very large activation values."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((10, 8)) * 1e6
        Y = backend.random_normal((10, 8)) * 1e6

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_mixed_scales(self, any_backend: "Backend") -> None:
        """CKA should handle activations with very different scales."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((10, 8)) * 1e-5
        Y = backend.random_normal((10, 8)) * 1e5

        result = compute_cka(X, Y, backend)

        assert math.isfinite(result.cka)

    def test_constant_activations(self, any_backend: "Backend") -> None:
        """CKA with constant activations should handle gracefully."""
        backend = any_backend

        X = backend.ones((10, 8))
        Y = backend.ones((10, 8)) * 2.0

        result = compute_cka(X, Y, backend)

        # Constant activations have no variance, CKA may be 0 or undefined
        assert math.isfinite(result.cka)

    def test_sparse_activations(self, any_backend: "Backend") -> None:
        """CKA should handle sparse activations (many zeros)."""
        backend = any_backend
        backend.random_seed(42)

        # Create sparse matrices
        X = backend.random_normal((10, 8))
        Y = backend.random_normal((10, 8))

        # Zero out most entries (copy needed for JAX read-only arrays)
        X_np = backend.to_numpy(X).copy()
        Y_np = backend.to_numpy(Y).copy()
        X_np[X_np < 0.5] = 0.0
        Y_np[Y_np < 0.5] = 0.0
        X_sparse = backend.array(X_np)
        Y_sparse = backend.array(Y_np)

        result = compute_cka(X_sparse, Y_sparse, backend)

        assert math.isfinite(result.cka)

    def test_high_dimensional_features(self, any_backend: "Backend") -> None:
        """CKA should handle high-dimensional features."""
        backend = any_backend
        backend.random_seed(42)

        # Features >> samples
        X = backend.random_normal((10, 500))
        Y = backend.random_normal((10, 500))

        result = compute_cka(X, Y, backend, estimator=HSICEstimator.AUTO)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0

    def test_minimal_samples(self, any_backend: "Backend") -> None:
        """CKA with exactly 2 samples."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((2, 8))
        Y = backend.random_normal((2, 8))

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert result.sample_count == 2

    def test_exactly_4_samples_for_unbiased(self, any_backend: "Backend") -> None:
        """Unbiased estimator requires exactly n >= 4."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((4, 8))
        Y = backend.random_normal((4, 8))

        result = compute_cka(X, Y, backend, estimator=HSICEstimator.UNBIASED)

        assert result.is_valid


# =============================================================================
# Caching Behavior Tests
# =============================================================================


class TestCachingBehavior:
    """Tests for Gram matrix caching behavior."""

    def test_repeated_calls_same_result(self, any_backend: "Backend") -> None:
        """Repeated calls with same data should give same result."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((15, 10))
        Y = backend.random_normal((15, 10))

        result1 = compute_cka(X, Y, backend)
        result2 = compute_cka(X, Y, backend)

        assert abs(result1.cka - result2.cka) < 1e-10

    def test_self_similarity_cached(self, any_backend: "Backend") -> None:
        """Self-similarity should leverage cache for identical Gram matrices."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((15, 10))

        # First call computes and caches
        result1 = compute_cka(X, X, backend)
        # Second call should use cache
        result2 = compute_cka(X, X, backend)

        assert abs(result1.cka - 1.0) < 1e-5
        assert abs(result2.cka - 1.0) < 1e-5


# =============================================================================
# Cross-Dimensional Comparison Tests
# =============================================================================


class TestCrossDimensionalComparison:
    """Tests for cross-dimensional comparison capabilities."""

    def test_different_feature_dimensions(self, any_backend: "Backend") -> None:
        """CKA should work with different feature dimensions."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((20, 32))  # 32 features
        Y = backend.random_normal((20, 64))  # 64 features

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0
        assert result.feature_dim_x == 32
        assert result.feature_dim_y == 64

    def test_gram_comparison_cross_dimensional(self, any_backend: "Backend") -> None:
        """Gram-based CKA works across any dimensions."""
        backend = any_backend
        backend.random_seed(42)

        # Different feature dims, same sample count
        X = backend.random_normal((15, 100))
        Y = backend.random_normal((15, 50))

        # Compute Gram matrices
        gram_x = backend.matmul(X, backend.transpose(X))  # [15, 15]
        gram_y = backend.matmul(Y, backend.transpose(Y))  # [15, 15]

        cka = compute_cka_from_grams(gram_x, gram_y, backend=backend)

        # Same Gram shape, should work
        assert 0.0 <= cka <= 1.0

    def test_very_different_dimensions(self, any_backend: "Backend") -> None:
        """CKA should handle very different feature dimensions."""
        backend = any_backend
        backend.random_seed(42)

        X = backend.random_normal((10, 16))  # Small
        Y = backend.random_normal((10, 256))  # Much larger

        result = compute_cka(X, Y, backend)

        assert result.is_valid
        assert 0.0 <= result.cka <= 1.0
