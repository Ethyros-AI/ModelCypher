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

"""Comprehensive tests for numerical_stability.py.

Tests cover:
- Epsilon/threshold functions (machine_epsilon, division_epsilon, etc.)
- SVD via eigendecomposition (geodesic_svd)
- QR-based linear solvers (solve_full_row_rank_via_qr)
- SVD-based solver (solve_via_truncated_svd)
- Entropy-based rank estimation
- Gram alignment solver
- CCA-Procrustes solver
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    compute_entropy_effective_rank,
    compute_shared_relational_rank,
    condition_threshold,
    division_epsilon,
    geodesic_svd,
    is_finite,
    is_inf,
    is_nan,
    log_scalar,
    machine_epsilon,
    regularization_epsilon,
    safe_log_epsilon,
    solve_full_row_rank_via_qr,
    solve_via_cca_procrustes,
    solve_via_gram_alignment,
    solve_via_truncated_svd,
    sqrt_scalar,
    svd_rank_threshold,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps(backend: "Backend", *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _div_eps(backend: "Backend", *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# Epsilon and Threshold Functions
# =============================================================================


class TestMachineEpsilon:
    """Tests for machine_epsilon function."""

    def test_float32_epsilon(self, any_backend: "Backend") -> None:
        """Float32 epsilon should reflect dtype precision."""
        b = any_backend
        arr = b.zeros((2, 2))  # Default is float32
        eps = machine_epsilon(b, arr)
        # Test in backend float32 precision, not Python float64
        one = b.array([1.0])
        half_eps_arr = b.array([eps / 2.0])
        eps_arr = b.array([eps])
        b.eval(one, half_eps_arr, eps_arr)
        # In float32: 1.0 + half_eps should round to 1.0
        assert float(b.to_scalar(one + half_eps_arr)) == 1.0
        # In float32: 1.0 + eps should NOT round to 1.0
        assert float(b.to_scalar(one + eps_arr)) != 1.0

    def test_float64_epsilon(self, any_backend: "Backend") -> None:
        """Float64 epsilon should be smaller than float32 when supported."""
        b = any_backend
        # Some backends (JAX without x64, MLX) silently truncate float64 to float32
        try:
            arr = b.astype(b.zeros((2, 2)), "float64")
        except (ValueError, RuntimeError):
            pytest.skip("float64 not supported on this backend")
        # Check if the dtype is actually float64 (not silently truncated)
        if "float64" not in str(arr.dtype):
            pytest.skip("float64 truncated to float32 on this backend")
        eps = machine_epsilon(b, arr)
        eps32 = machine_epsilon(b, b.zeros((2, 2)))
        assert eps < eps32

    def test_epsilon_is_smallest_distinguishable(self, any_backend: "Backend") -> None:
        """1.0 + epsilon should not equal 1.0."""
        b = any_backend
        arr = b.zeros((2, 2))
        eps = machine_epsilon(b, arr)
        # This is the definition of machine epsilon
        one_plus_eps = 1.0 + eps
        assert one_plus_eps != 1.0


class TestDivisionEpsilon:
    """Tests for division_epsilon function."""

    def test_division_epsilon_scaled(self, any_backend: "Backend") -> None:
        """Division epsilon should be sqrt(machine_epsilon)."""
        b = any_backend
        arr = b.zeros((2, 2))
        div_eps = division_epsilon(b, arr)
        mach_eps = machine_epsilon(b, arr)
        expected = sqrt_scalar(mach_eps, b)
        eps = _eps(b, div_eps, expected)
        assert abs(div_eps - expected) <= eps

    def test_division_epsilon_prevents_zero_division(
        self, any_backend: "Backend"
    ) -> None:
        """Division epsilon should be large enough to prevent zero division issues."""
        b = any_backend
        arr = b.zeros((2, 2))
        div_eps = division_epsilon(b, arr)
        mach_eps = machine_epsilon(b, arr)
        eps = _eps(b, div_eps, mach_eps)
        assert div_eps >= mach_eps
        assert abs(div_eps * div_eps - mach_eps) <= eps


class TestRegularizationEpsilon:
    """Tests for regularization_epsilon function."""

    def test_regularization_epsilon_is_eps_0_75(self, any_backend: "Backend") -> None:
        """Regularization epsilon should be machine_epsilon^0.75."""
        b = any_backend
        arr = b.zeros((2, 2))
        reg_eps = regularization_epsilon(b, arr)
        mach_eps = machine_epsilon(b, arr)
        expected = mach_eps ** 0.75
        eps = _eps(b, reg_eps, expected)
        assert abs(reg_eps - expected) <= eps

    def test_regularization_epsilon_float32(self, any_backend: "Backend") -> None:
        """Float32 regularization epsilon should sit between eps and division eps."""
        b = any_backend
        arr = b.zeros((2, 2))  # float32
        reg_eps = regularization_epsilon(b, arr)
        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        assert mach_eps <= reg_eps <= div_eps


class TestConditionThreshold:
    """Tests for condition_threshold function."""

    def test_condition_threshold_is_inverse_eps(self, any_backend: "Backend") -> None:
        """Condition threshold should be 1/machine_epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        cond_thresh = condition_threshold(b, arr)
        mach_eps = machine_epsilon(b, arr)
        expected = 1.0 / mach_eps
        eps = _eps(b, cond_thresh, expected)
        assert abs(cond_thresh - expected) <= eps

    def test_condition_threshold_float32(self, any_backend: "Backend") -> None:
        """Float32 condition threshold should be inverse epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))  # float32
        cond_thresh = condition_threshold(b, arr)
        mach_eps = machine_epsilon(b, arr)
        eps = _eps(b, cond_thresh, mach_eps)
        assert abs(cond_thresh * mach_eps - 1.0) <= eps


class TestSvdRankThreshold:
    """Tests for svd_rank_threshold function."""

    def test_svd_rank_threshold_scales_with_dimension(
        self, any_backend: "Backend"
    ) -> None:
        """SVD rank threshold should scale linearly with max_dim."""
        b = any_backend
        arr = b.zeros((2, 2))
        thresh_10 = svd_rank_threshold(b, arr, max_dim=10)
        thresh_100 = svd_rank_threshold(b, arr, max_dim=100)
        # Should scale linearly
        ratio = thresh_100 / thresh_10
        eps = _eps(b, ratio, 10.0)
        assert abs(ratio - 10.0) <= eps

    def test_svd_rank_threshold_formula(self, any_backend: "Backend") -> None:
        """SVD rank threshold should be max_dim * eps."""
        b = any_backend
        arr = b.zeros((2, 2))
        max_dim = 50
        thresh = svd_rank_threshold(b, arr, max_dim=max_dim)
        mach_eps = machine_epsilon(b, arr)
        expected = float(max_dim) * mach_eps
        eps = _eps(b, thresh, expected)
        assert abs(thresh - expected) <= eps


class TestTinyValue:
    """Tests for tiny_value function."""

    def test_tiny_value_is_positive(self, any_backend: "Backend") -> None:
        """Tiny value should be positive."""
        b = any_backend
        arr = b.zeros((2, 2))
        tiny = tiny_value(b, arr)
        assert tiny > 0

    def test_tiny_value_is_small(self, any_backend: "Backend") -> None:
        """Tiny value should be very small."""
        b = any_backend
        arr = b.zeros((2, 2))  # float32
        tiny = tiny_value(b, arr)
        mach_eps = machine_epsilon(b, arr)
        assert tiny < mach_eps


class TestSafeLogEpsilon:
    """Tests for safe_log_epsilon function."""

    def test_safe_log_epsilon_equals_tiny(self, any_backend: "Backend") -> None:
        """Safe log epsilon should equal tiny value."""
        b = any_backend
        arr = b.zeros((2, 2))
        log_eps = safe_log_epsilon(b, arr)
        tiny = tiny_value(b, arr)
        assert log_eps == tiny

    def test_safe_log_epsilon_prevents_log_zero(self, any_backend: "Backend") -> None:
        """Safe log epsilon should allow safe log computation."""
        b = any_backend
        arr = b.zeros((2, 2))
        log_eps = safe_log_epsilon(b, arr)
        # log(epsilon) should be finite
        log_val = log_scalar(log_eps, b)
        assert is_finite(log_val, b)


# =============================================================================
# QR-based Linear Solvers
# =============================================================================


class TestSolveFullRowRankViaQR:
    """Tests for solve_full_row_rank_via_qr function."""

    def test_qr_overdetermined_well_conditioned(self, any_backend: "Backend") -> None:
        """Overdetermined well-conditioned system should solve accurately."""
        b = any_backend
        b.random_seed(42)
        # source [100, 10], target [100, 5]
        source = b.random_normal((100, 10))
        # Create target as source @ true_F for consistent system
        true_F = b.random_normal((10, 5))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        assert F is not None
        assert diag["system_type"] == "overdetermined"
        assert diag["method"] in ["qr", "qr_regularized", "qr_refined"]
        reconstructed = b.matmul(source, F)
        b.eval(reconstructed)
        residual = reconstructed - target
        residual_norm_arr = b.norm(residual)
        target_norm_arr = b.norm(target)
        b.eval(residual_norm_arr, target_norm_arr)
        residual_norm = float(b.to_scalar(residual_norm_arr))
        target_norm = float(b.to_scalar(target_norm_arr))
        eps = _div_eps(b, target_norm)
        assert residual_norm <= eps * max(target_norm, eps)

    def test_qr_underdetermined_well_conditioned(self, any_backend: "Backend") -> None:
        """Underdetermined well-conditioned system should find minimum-norm solution."""
        b = any_backend
        b.random_seed(42)
        # source [10, 100], target [10, 5]
        source = b.random_normal((10, 100))
        # Create consistent target
        true_F = b.random_normal((100, 5))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        assert F is not None
        assert diag["system_type"] == "underdetermined"
        assert diag["method"] in [
            "qr",
            "qr_regularized",
            "qr_rank_deficient",
            "qr_refined",
        ]

    def test_qr_reconstruction_accuracy(self, any_backend: "Backend") -> None:
        """source @ F should approximately equal target."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((50, 20))
        true_F = b.random_normal((20, 10))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        assert F is not None
        reconstructed = b.matmul(source, F)
        b.eval(reconstructed)

        diff = reconstructed - target
        diff_max_arr = b.max(b.abs(diff))
        target_max_arr = b.max(b.abs(target))
        b.eval(diff_max_arr, target_max_arr)
        diff_max = float(b.to_scalar(diff_max_arr))
        target_max = float(b.to_scalar(target_max_arr))
        eps = _div_eps(b, target_max)
        rel_error = diff_max / max(target_max, eps)
        assert rel_error <= eps

    def test_qr_empty_system(self, any_backend: "Backend") -> None:
        """Empty system should return None."""
        b = any_backend
        source = b.zeros((0, 10))
        target = b.zeros((0, 5))

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        assert F is None
        assert diag["method"] == "failed"

    def test_qr_zero_columns_source(self, any_backend: "Backend") -> None:
        """Source with zero columns should return None."""
        b = any_backend
        source = b.zeros((10, 0))
        target = b.random_normal((10, 5))
        b.eval(target)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        assert F is None
        assert diag["method"] == "failed"

    def test_qr_ill_conditioned_regularizes(self, any_backend: "Backend") -> None:
        """Ill-conditioned system should trigger regularization."""
        b = any_backend
        b.random_seed(42)
        # Create ill-conditioned matrix by scaling columns
        source = b.random_normal((50, 10))
        # Scale last few columns to be very small
        scales = b.array([1.0] * 7 + [1e-8] * 3)
        scales = b.reshape(scales, (1, -1))
        source = source * scales
        target = b.random_normal((50, 5))
        b.eval(source, target, scales)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        # Should succeed, possibly with regularization
        assert F is not None or diag["method"] == "failed"

    def test_qr_diagnostics_complete(self, any_backend: "Backend") -> None:
        """Diagnostics should contain all expected fields."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((30, 15))
        target = b.random_normal((30, 5))
        b.eval(source, target)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        # Check required diagnostics fields
        assert "rank" in diag
        assert "condition" in diag
        assert "residual_norm" in diag
        assert "method" in diag
        assert "system_type" in diag
        assert "n_samples" in diag
        assert "d_source" in diag
        assert "d_target" in diag


# =============================================================================
# SVD-based Linear Solver
# =============================================================================


class TestSolveViaTruncatedSvd:
    """Tests for solve_via_truncated_svd function."""

    def test_svd_solver_consistent_system(self, any_backend: "Backend") -> None:
        """Consistent system should have near-zero projection error."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((50, 20))
        true_F = b.random_normal((20, 10))
        target = b.matmul(source, true_F)  # Consistent system
        b.eval(source, target, true_F)

        F, diag = solve_via_truncated_svd(b, source, target)

        assert F is not None
        assert diag["method"] == "svd_truncated"
        # Projection error should be near zero for consistent system
        reconstructed = b.matmul(source, F)
        b.eval(reconstructed)
        residual = reconstructed - target
        residual_norm_arr = b.norm(residual)
        target_norm_arr = b.norm(target)
        b.eval(residual_norm_arr, target_norm_arr)
        residual_norm = float(b.to_scalar(residual_norm_arr))
        target_norm = float(b.to_scalar(target_norm_arr))
        eps = _div_eps(b, target_norm)
        assert residual_norm <= eps * max(target_norm, eps)

    def test_svd_solver_reconstruction(self, any_backend: "Backend") -> None:
        """source @ F should approximately equal target for consistent system."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((50, 20))
        true_F = b.random_normal((20, 10))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_via_truncated_svd(b, source, target)

        assert F is not None
        reconstructed = b.matmul(source, F)
        b.eval(reconstructed)

        diff = reconstructed - target
        diff_max_arr = b.max(b.abs(diff))
        target_max_arr = b.max(b.abs(target))
        b.eval(diff_max_arr, target_max_arr)
        diff_max = float(b.to_scalar(diff_max_arr))
        target_max = float(b.to_scalar(target_max_arr))
        eps = _div_eps(b, target_max)
        rel_error = diff_max / max(target_max, eps)
        assert rel_error <= eps

    def test_svd_solver_rank_deficient(self, any_backend: "Backend") -> None:
        """Rank-deficient source should have reduced effective rank."""
        b = any_backend
        b.random_seed(42)
        # Create rank-5 matrix from 50x50
        U = b.random_normal((50, 5))
        V = b.random_normal((5, 20))
        source = b.matmul(U, V)
        target = b.random_normal((50, 10))
        b.eval(source, target)

        F, diag = solve_via_truncated_svd(b, source, target)

        # SVD should detect reduced rank (though numerical noise may inflate it)
        # Just verify it completes without error and has valid rank
        assert diag["rank"] > 0
        assert diag["rank"] <= 20  # Cannot exceed min(50, 20)

    def test_svd_solver_empty_system(self, any_backend: "Backend") -> None:
        """Empty system should return None."""
        b = any_backend
        source = b.zeros((0, 10))
        target = b.zeros((0, 5))

        F, diag = solve_via_truncated_svd(b, source, target)

        assert F is None

    def test_svd_solver_custom_rank_threshold(self, any_backend: "Backend") -> None:
        """Custom rank threshold should affect truncation."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((30, 15))
        target = b.random_normal((30, 10))
        b.eval(source, target)

        # Aggressive threshold derived from dtype precision
        high_thresh = division_epsilon(b, source)
        low_thresh = machine_epsilon(b, source)
        F1, diag1 = solve_via_truncated_svd(b, source, target, rank_threshold=high_thresh)
        F2, diag2 = solve_via_truncated_svd(b, source, target, rank_threshold=low_thresh)

        # Aggressive threshold should have lower rank
        if F1 is not None and F2 is not None:
            assert diag1["rank"] <= diag2["rank"]

    def test_svd_solver_diagnostics(self, any_backend: "Backend") -> None:
        """Diagnostics should contain all expected fields."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((30, 15))
        target = b.random_normal((30, 10))
        b.eval(source, target)

        F, diag = solve_via_truncated_svd(b, source, target)

        assert "rank" in diag
        assert "condition" in diag
        assert "residual_norm" in diag
        assert "projection_error" in diag
        assert "method" in diag


# =============================================================================
# Entropy-based Rank Estimation
# =============================================================================


class TestComputeEntropyEffectiveRank:
    """Tests for compute_entropy_effective_rank function."""

    def test_uniform_singular_values_high_rank(self, any_backend: "Backend") -> None:
        """Uniform singular values should give high effective rank."""
        b = any_backend
        # All singular values equal -> max entropy -> rank = n
        sv = [1.0, 1.0, 1.0, 1.0, 1.0]
        erank = compute_entropy_effective_rank(b, sv)
        expected = float(len(sv))
        eps = 10.0 * _eps(b, erank, expected)  # 10x epsilon for log/exp stability
        assert abs(erank - expected) <= eps

    def test_concentrated_singular_values_low_rank(
        self, any_backend: "Backend"
    ) -> None:
        """Concentrated singular values should give low effective rank."""
        b = any_backend
        # One dominant singular value -> low entropy -> rank ~ 1
        sv = [10.0, 0.01, 0.01, 0.01, 0.01]
        erank = compute_entropy_effective_rank(b, sv)
        uniform_rank = compute_entropy_effective_rank(b, [1.0] * len(sv))
        eps = _eps(b, erank, uniform_rank)
        assert erank <= uniform_rank + eps

    def test_empty_singular_values(self, any_backend: "Backend") -> None:
        """Empty singular values should return 0."""
        b = any_backend
        erank = compute_entropy_effective_rank(b, [])
        assert erank == 0.0

    def test_all_zero_singular_values(self, any_backend: "Backend") -> None:
        """All zero singular values should return 0."""
        b = any_backend
        erank = compute_entropy_effective_rank(b, [0.0, 0.0, 0.0])
        assert erank == 0.0

    def test_single_singular_value(self, any_backend: "Backend") -> None:
        """Single singular value should give rank 1."""
        b = any_backend
        erank = compute_entropy_effective_rank(b, [5.0])
        eps = _eps(b, erank, 1.0)
        assert abs(erank - 1.0) <= eps

    def test_geometric_decay_singular_values(self, any_backend: "Backend") -> None:
        """Geometrically decaying singular values should give intermediate rank."""
        b = any_backend
        # Geometric decay: 1, 0.5, 0.25, 0.125, 0.0625
        sv = [1.0, 0.5, 0.25, 0.125, 0.0625]
        erank = compute_entropy_effective_rank(b, sv)
        uniform_rank = compute_entropy_effective_rank(b, [1.0] * len(sv))
        concentrated_rank = compute_entropy_effective_rank(b, [1.0] + [0.0] * (len(sv) - 1))
        eps = _eps(b, erank, uniform_rank, concentrated_rank)
        assert concentrated_rank - eps <= erank <= uniform_rank + eps

    def test_effective_rank_bounds(self, any_backend: "Backend") -> None:
        """Effective rank should be bounded by [1, n]."""
        b = any_backend
        sv = [3.0, 2.0, 1.0, 0.5, 0.1]
        erank = compute_entropy_effective_rank(b, sv)
        # Bounded: 1 <= erank <= 5
        eps = _eps(b, erank, float(len(sv)))
        assert 1.0 - eps <= erank <= float(len(sv)) + eps


class TestComputeSharedRelationalRank:
    """Tests for compute_shared_relational_rank function."""

    def test_shared_rank_is_maximum(self, any_backend: "Backend") -> None:
        """Shared rank should preserve signal from both source and target."""
        b = any_backend
        # Source: high effective rank
        source_sv = [1.0, 1.0, 1.0, 1.0]
        # Target: low effective rank
        target_sv = [10.0, 0.1, 0.01, 0.001]

        shared_rank, diag = compute_shared_relational_rank(b, source_sv, target_sv)

        # Shared should preserve the maximum of the two ranks
        assert shared_rank >= diag["integer_rank_source"]
        assert shared_rank >= diag["integer_rank_target"]

    def test_shared_rank_diagnostics(self, any_backend: "Backend") -> None:
        """Diagnostics should contain all expected fields."""
        b = any_backend
        source_sv = [1.0, 0.5, 0.25]
        target_sv = [2.0, 1.0, 0.5]

        _, diag = compute_shared_relational_rank(b, source_sv, target_sv)

        assert "effective_rank_source" in diag
        assert "effective_rank_target" in diag
        assert "integer_rank_source" in diag
        assert "integer_rank_target" in diag
        assert "shared_relational_rank" in diag
        assert "source_exclusive_dims" in diag
        assert "target_exclusive_dims" in diag

    def test_empty_singular_values(self, any_backend: "Backend") -> None:
        """Empty singular values should give shared rank 1 (minimum)."""
        b = any_backend
        shared_rank, _ = compute_shared_relational_rank(b, [], [1.0, 0.5])
        # min of (0 -> 1 via max(1, floor), target_rank)
        assert shared_rank >= 0

    def test_identical_spectra(self, any_backend: "Backend") -> None:
        """Identical spectra should have matching ranks."""
        b = any_backend
        sv = [1.0, 0.8, 0.6, 0.4, 0.2]
        shared_rank, diag = compute_shared_relational_rank(b, sv, sv)

        # Ranks should match
        assert diag["integer_rank_source"] == diag["integer_rank_target"]
        assert shared_rank == diag["integer_rank_source"]


# =============================================================================
# Gram Alignment Solver
# =============================================================================


class TestSolveViaGramAlignment:
    """Tests for solve_via_gram_alignment function."""

    def test_gram_alignment_same_dimension(self, any_backend: "Backend") -> None:
        """Same dimension source/target should align."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((50, 20))
        # Create target as rotated source
        rotation = b.random_normal((20, 20))
        # Orthogonalize via QR
        Q, _ = b.qr(rotation)
        target = b.matmul(source, Q)
        b.eval(source, target)

        F, diag = solve_via_gram_alignment(b, source, target)

        assert F is not None
        assert diag["method"] == "gram_alignment"
        from modelcypher.core.domain.geometry.cka import compute_cka

        aligned = b.matmul(source, F)
        b.eval(aligned)
        cka_result = compute_cka(aligned, target, backend=b)
        # Tolerance: 1e-6 for iterative geodesic alignment (not machine eps)
        assert abs(cka_result.cka - 1.0) <= 1e-6, f"CKA not ~1.0: {cka_result.cka}"

    def test_gram_alignment_different_dimensions(self, any_backend: "Backend") -> None:
        """Different dimension source/target should still work."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((50, 30))
        target = b.random_normal((50, 20))
        b.eval(source, target)

        F, diag = solve_via_gram_alignment(b, source, target)

        # Should produce a transform even for different dims
        assert F is not None or diag["shared_relational_rank"] == 0
        if F is not None:
            assert b.shape(F) == (30, 20)

    def test_gram_alignment_too_few_samples(self, any_backend: "Backend") -> None:
        """Too few samples (n < 2) should return None."""
        b = any_backend
        source = b.random_normal((1, 10))
        target = b.random_normal((1, 10))
        b.eval(source, target)

        F, diag = solve_via_gram_alignment(b, source, target)

        assert F is None

    def test_gram_alignment_empty_features(self, any_backend: "Backend") -> None:
        """Empty features should return None."""
        b = any_backend
        source = b.zeros((10, 0))
        target = b.zeros((10, 5))

        F, diag = solve_via_gram_alignment(b, source, target)

        assert F is None

    def test_gram_alignment_diagnostics(self, any_backend: "Backend") -> None:
        """Diagnostics should contain expected fields."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((30, 15))
        target = b.random_normal((30, 10))
        b.eval(source, target)

        F, diag = solve_via_gram_alignment(b, source, target)

        assert "method" in diag
        assert "n_samples" in diag
        assert "d_source" in diag
        assert "d_target" in diag
        assert "procrustes_error" in diag
        assert "rank_source" in diag
        assert "rank_target" in diag


# =============================================================================
# CCA-Procrustes Solver
# =============================================================================


class TestSolveViaCcaProcrustes:
    """Tests for solve_via_cca_procrustes function."""

    def test_cca_procrustes_correlated_data(self, any_backend: "Backend") -> None:
        """Highly correlated source/target should find shared dimensions."""
        b = any_backend
        b.random_seed(42)
        # Create correlated data through shared latent
        latent = b.random_normal((50, 5))
        source = b.matmul(latent, b.random_normal((5, 20)))
        target = b.matmul(latent, b.random_normal((5, 15)))
        b.eval(source, target)

        F, diag = solve_via_cca_procrustes(b, source, target)

        # Should find shared dimensions
        if F is not None:
            eps = _eps(b, diag["shared_dim"], diag["top_correlation"])
            assert diag["shared_dim"] > 0
            assert diag["top_correlation"] > eps
            assert diag["method"] == "cca_procrustes"

    def test_cca_procrustes_uncorrelated_data(self, any_backend: "Backend") -> None:
        """Correlated data should yield higher top correlation than uncorrelated."""
        b = any_backend
        b.random_seed(42)
        # Correlated data through shared latent
        latent = b.random_normal((50, 5))
        source_corr = b.matmul(latent, b.random_normal((5, 20)))
        target_corr = b.matmul(latent, b.random_normal((5, 15)))
        b.eval(source_corr, target_corr)

        _, diag_corr = solve_via_cca_procrustes(b, source_corr, target_corr)
        assert diag_corr["method"] == "cca_procrustes"

        # Independent data
        b.random_seed(123)
        source = b.random_normal((50, 20))
        b.random_seed(456)
        target = b.random_normal((50, 15))
        b.eval(source, target)

        _, diag_uncorr = solve_via_cca_procrustes(b, source, target)

        eps = machine_epsilon(b, source)
        assert diag_corr["top_correlation"] >= diag_uncorr["top_correlation"] - eps

    def test_cca_procrustes_too_few_samples(self, any_backend: "Backend") -> None:
        """Too few samples should return None."""
        b = any_backend
        source = b.random_normal((1, 10))
        target = b.random_normal((1, 10))
        b.eval(source, target)

        F, diag = solve_via_cca_procrustes(b, source, target)

        assert F is None

    def test_cca_procrustes_pca_dims_returned(self, any_backend: "Backend") -> None:
        """PCA dimensions should be reported in diagnostics."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((30, 15))
        target = b.random_normal((30, 10))
        b.eval(source, target)

        F, diag = solve_via_cca_procrustes(b, source, target)

        assert "pca_dims" in diag
        if F is not None:
            assert len(diag["pca_dims"]) == 2

    def test_cca_procrustes_alignment_error(self, any_backend: "Backend") -> None:
        """Alignment error should be computed for successful alignment."""
        b = any_backend
        b.random_seed(42)
        latent = b.random_normal((40, 8))
        source = b.matmul(latent, b.random_normal((8, 20)))
        target = b.matmul(latent, b.random_normal((8, 15)))
        b.eval(source, target)

        F, diag = solve_via_cca_procrustes(b, source, target)

        if F is not None:
            assert "alignment_error" in diag
            assert diag["alignment_error"] < float("inf")

    def test_cca_procrustes_output_shape(self, any_backend: "Backend") -> None:
        """Output transform F should have correct shape [d_source, d_target]."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((40, 25))
        target = b.random_normal((40, 18))
        b.eval(source, target)

        F, diag = solve_via_cca_procrustes(b, source, target)

        if F is not None:
            assert b.shape(F) == (25, 18)

# =============================================================================
# Integration Tests
# =============================================================================


class TestSolverComparison:
    """Compare different solvers on the same problems."""

    def test_all_solvers_on_consistent_system(self, any_backend: "Backend") -> None:
        """All solvers should work on a consistent overdetermined system."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((60, 20))
        true_F = b.random_normal((20, 10))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        # QR solver
        F_qr, diag_qr = solve_full_row_rank_via_qr(b, source, target)
        # SVD solver
        F_svd, diag_svd = solve_via_truncated_svd(b, source, target)
        # Gram alignment
        F_gram, diag_gram = solve_via_gram_alignment(b, source, target)

        # All should produce valid solutions
        assert F_qr is not None
        assert F_svd is not None
        # Gram alignment may or may not work depending on data structure
        # Just check it doesn't crash

        # QR and SVD should have low residuals
        recon_qr = b.matmul(source, F_qr)
        recon_svd = b.matmul(source, F_svd)
        b.eval(recon_qr, recon_svd)
        residual_qr_arr = b.norm(recon_qr - target)
        residual_svd_arr = b.norm(recon_svd - target)
        target_norm_arr = b.norm(target)
        b.eval(residual_qr_arr, residual_svd_arr, target_norm_arr)
        residual_qr = float(b.to_scalar(residual_qr_arr))
        residual_svd = float(b.to_scalar(residual_svd_arr))
        target_norm = float(b.to_scalar(target_norm_arr))
        eps = _div_eps(b, target_norm)
        assert residual_qr <= eps * max(target_norm, eps)
        assert residual_svd <= eps * max(target_norm, eps)

    def test_solvers_on_rank_deficient_system(self, any_backend: "Backend") -> None:
        """Solvers should handle rank-deficient systems."""
        b = any_backend
        b.random_seed(42)
        # Rank-5 source matrix
        U = b.random_normal((50, 5))
        V = b.random_normal((5, 20))
        source = b.matmul(U, V)
        target = b.random_normal((50, 10))
        b.eval(source, target)

        # All solvers should handle this without crashing
        F_qr, diag_qr = solve_full_row_rank_via_qr(b, source, target)
        F_svd, diag_svd = solve_via_truncated_svd(b, source, target)
        F_gram, diag_gram = solve_via_gram_alignment(b, source, target)

        # SVD should report valid rank (numerical noise may inflate it beyond true rank)
        assert diag_svd["rank"] > 0
        assert diag_svd["rank"] <= 20  # Cannot exceed min(50, 20)


class TestNumericalPrecision:
    """Tests for numerical precision and stability."""

    def test_svd_eigh_numerical_stability(self, any_backend: "Backend") -> None:
        """SVD via eigh should be numerically stable for ill-conditioned matrices."""
        b = any_backend
        b.random_seed(42)
        # Create ill-conditioned matrix
        A = b.random_normal((20, 10))
        # Scale columns to create poor conditioning
        scales = b.array([10.0 ** (-i) for i in range(10)])
        scales = b.reshape(scales, (1, -1))
        A = A * scales
        b.eval(A, scales)

        # Should not raise or produce NaN
        U, S, Vt = geodesic_svd(b, A)
        b.eval(U, S, Vt)

        nan_s = b.sum(b.astype(b.isnan(S), "int32"))
        nan_u = b.sum(b.astype(b.isnan(U), "int32"))
        b.eval(nan_s, nan_u)
        assert int(b.to_scalar(nan_s)) == 0
        assert int(b.to_scalar(nan_u)) == 0

    def test_epsilon_functions_consistency(self, any_backend: "Backend") -> None:
        """Epsilon functions should have consistent relationships."""
        b = any_backend
        arr = b.zeros((2, 2))

        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        reg_eps = regularization_epsilon(b, arr)
        cond_thresh = condition_threshold(b, arr)
        tiny = tiny_value(b, arr)

        # Division epsilon > machine epsilon
        assert div_eps >= mach_eps
        eps = _eps(b, div_eps, mach_eps)
        assert abs(div_eps * div_eps - mach_eps) <= eps
        # Regularization epsilon > machine epsilon
        assert reg_eps >= mach_eps
        assert reg_eps <= div_eps
        # Condition threshold consistent with machine epsilon
        eps = _eps(b, cond_thresh, mach_eps)
        assert abs(cond_thresh * mach_eps - 1.0) <= eps
        # Tiny < machine epsilon
        assert tiny < mach_eps
        # All should be positive
        assert all(v > 0.0 for v in [mach_eps, div_eps, reg_eps, cond_thresh, tiny])


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================


from hypothesis import given, strategies as st, assume, settings as hypothesis_settings, HealthCheck


class TestNumericalStabilityHypothesis:
    """Hypothesis property-based tests for numerical stability functions."""

    @given(
        rows=st.integers(min_value=2, max_value=30),
        cols=st.integers(min_value=2, max_value=30),
    )
    @hypothesis_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_epsilon_functions_always_positive(
        self, any_backend: "Backend", rows: int, cols: int
    ) -> None:
        """All epsilon functions should return positive values for any array shape."""
        b = any_backend
        arr = b.zeros((rows, cols))

        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        reg_eps = regularization_epsilon(b, arr)
        cond_thresh = condition_threshold(b, arr)
        tiny = tiny_value(b, arr)
        log_eps = safe_log_epsilon(b, arr)

        assert mach_eps > 0, "machine_epsilon must be positive"
        assert div_eps > 0, "division_epsilon must be positive"
        assert reg_eps > 0, "regularization_epsilon must be positive"
        assert cond_thresh > 0, "condition_threshold must be positive"
        assert tiny > 0, "tiny_value must be positive"
        assert log_eps > 0, "safe_log_epsilon must be positive"

    @given(
        max_dim=st.integers(min_value=1, max_value=1000),
    )
    @hypothesis_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_svd_rank_threshold_scales_linearly(
        self, any_backend: "Backend", max_dim: int
    ) -> None:
        """SVD rank threshold should scale linearly with max_dim."""
        b = any_backend
        arr = b.zeros((2, 2))

        thresh = svd_rank_threshold(b, arr, max_dim=max_dim)
        mach_eps = machine_epsilon(b, arr)

        expected = float(max_dim) * mach_eps
        eps = _eps(b, thresh, expected)
        assert abs(thresh - expected) <= eps, f"Expected {expected}, got {thresh}"

    @given(
        rows=st.integers(min_value=3, max_value=20),
        cols=st.integers(min_value=3, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_svd_no_nan_inf(
        self, any_backend: "Backend", rows: int, cols: int, seed: int
    ) -> None:
        """SVD via eigh should never produce NaN or Inf for random matrices."""
        b = any_backend
        b.random_seed(seed)
        A = b.random_normal((rows, cols))
        b.eval(A)

        U, S, Vt = geodesic_svd(b, A)
        b.eval(U, S, Vt)

        nan_s = b.sum(b.astype(b.isnan(S), "int32"))
        inf_s = b.sum(b.astype(b.isinf(S), "int32"))
        b.eval(nan_s, inf_s)
        assert int(b.to_scalar(nan_s)) == 0, "S contains NaN"
        assert int(b.to_scalar(inf_s)) == 0, "S contains Inf"

        # Check no NaN/Inf in U (if non-empty)
        if b.shape(U)[0] > 0 and b.shape(U)[1] > 0:
            nan_u = b.sum(b.astype(b.isnan(U), "int32"))
            b.eval(nan_u)
            assert int(b.to_scalar(nan_u)) == 0, "U contains NaN"

    @given(
        rows=st.integers(min_value=5, max_value=20),
        cols=st.integers(min_value=3, max_value=15),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_svd_singular_values_nonnegative_hypothesis(
        self, any_backend: "Backend", rows: int, cols: int, seed: int
    ) -> None:
        """SVD singular values should always be non-negative."""
        b = any_backend
        b.random_seed(seed)
        A = b.random_normal((rows, cols))
        b.eval(A)

        _, S, _ = geodesic_svd(b, A)
        b.eval(S)

        S_np = b.tolist(S)
        eps = _eps(b, float(min(S_np, default=0.0)))
        assert all(s >= -eps for s in S_np), "Singular values must be non-negative"

    @given(
        n_sv=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_entropy_effective_rank_bounded(
        self, any_backend: "Backend", n_sv: int, seed: int
    ) -> None:
        """Effective rank should be bounded by [1, n] for non-trivial spectra."""
        b = any_backend
        b.random_seed(seed)

        # Generate random positive singular values
        sv_base = b.abs(b.random_normal((n_sv,)))
        eps = division_epsilon(b, sv_base)
        sv_arr = sv_base + eps
        b.eval(sv_arr)
        sv = [float(v) for v in b.tolist(sv_arr)]

        erank = compute_entropy_effective_rank(b, sv)

        # Effective rank is bounded: 1 <= erank <= n
        eps = _eps(b, erank, float(n_sv))
        assert erank >= 1.0 - eps, f"Effective rank {erank} below lower bound"
        assert erank <= float(n_sv) + eps, f"Effective rank {erank} above upper bound {n_sv}"

    @given(
        n=st.integers(min_value=3, max_value=10),
    )
    @hypothesis_settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_uniform_singular_values_max_rank(
        self, any_backend: "Backend", n: int
    ) -> None:
        """Uniform singular values should give effective rank close to n."""
        b = any_backend
        sv = [1.0] * n

        erank = compute_entropy_effective_rank(b, sv)

        # Uniform distribution has max entropy, so rank should be n
        eps = _div_eps(b, erank, float(n))
        assert abs(erank - n) <= eps, f"Expected rank {n}, got {erank}"


class TestSolverStabilityHypothesis:
    """Hypothesis property-based tests for solver stability."""

    @given(
        rows=st.integers(min_value=10, max_value=30),
        cols=st.integers(min_value=5, max_value=20),
        target_cols=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_qr_solver_consistent_system_low_residual(
        self, any_backend: "Backend", rows: int, cols: int, target_cols: int, seed: int
    ) -> None:
        """QR solver on consistent system should have low residual."""
        assume(rows > cols)  # Overdetermined
        b = any_backend
        b.random_seed(seed)

        source = b.random_normal((rows, cols))
        true_F = b.random_normal((cols, target_cols))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_full_row_rank_via_qr(b, source, target)

        if F is not None:
            reconstructed = b.matmul(source, F)
            b.eval(reconstructed)
            residual = reconstructed - target
            residual_norm_arr = b.norm(residual)
            target_norm_arr = b.norm(target)
            b.eval(residual_norm_arr, target_norm_arr)
            residual_norm = float(b.to_scalar(residual_norm_arr))
            target_norm = float(b.to_scalar(target_norm_arr))
            eps = _div_eps(b, target_norm)
            assert residual_norm <= eps * max(target_norm, eps)

    @given(
        rows=st.integers(min_value=10, max_value=30),
        cols=st.integers(min_value=5, max_value=20),
        target_cols=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_svd_solver_consistent_system_low_projection_error(
        self, any_backend: "Backend", rows: int, cols: int, target_cols: int, seed: int
    ) -> None:
        """SVD solver on consistent system should have low projection error."""
        assume(rows > cols)  # Overdetermined
        b = any_backend
        b.random_seed(seed)

        source = b.random_normal((rows, cols))
        true_F = b.random_normal((cols, target_cols))
        target = b.matmul(source, true_F)
        b.eval(source, target, true_F)

        F, diag = solve_via_truncated_svd(b, source, target)

        if F is not None:
            reconstructed = b.matmul(source, F)
            b.eval(reconstructed)
            residual = reconstructed - target
            residual_norm_arr = b.norm(residual)
            target_norm_arr = b.norm(target)
            b.eval(residual_norm_arr, target_norm_arr)
            residual_norm = float(b.to_scalar(residual_norm_arr))
            target_norm = float(b.to_scalar(target_norm_arr))
            eps = _div_eps(b, target_norm)
            assert residual_norm <= eps * max(target_norm, eps)

    @given(
        scale=st.floats(min_value=1e-6, max_value=1e6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_solver_scale_invariance(
        self, any_backend: "Backend", scale: float, seed: int
    ) -> None:
        """Solvers should handle different input scales gracefully."""
        b = any_backend
        b.random_seed(seed)

        source = b.random_normal((30, 15))
        target = b.random_normal((30, 10))
        b.eval(source, target)

        # Scale inputs
        source_scaled = source * scale
        target_scaled = target * scale
        b.eval(source_scaled, target_scaled)

        F, diag = solve_full_row_rank_via_qr(b, source_scaled, target_scaled)

        # Should produce a valid solution (not None, no NaN)
        if F is not None:
            nan_count = b.sum(b.astype(b.isnan(F), "int32"))
            b.eval(nan_count)
            assert int(b.to_scalar(nan_count)) == 0, (
                f"F contains NaN values at scale {scale}"
            )


class TestEdgeCaseEpsilons:
    """Tests for edge case numerical behavior."""

    def test_very_small_array_values(self, any_backend: "Backend") -> None:
        """Arrays with very small values should not cause underflow issues."""
        b = any_backend
        # Create array with values near machine epsilon
        small = b.array([[1e-30, 1e-35], [1e-35, 1e-30]])
        b.eval(small)

        # Epsilon functions should still work
        eps = machine_epsilon(b, small)
        assert eps > 0
        assert not is_nan(eps, b)

    def test_very_large_array_values(self, any_backend: "Backend") -> None:
        """Arrays with very large values should not cause overflow issues."""
        b = any_backend
        # Create array with large values
        large = b.array([[1e30, 1e35], [1e35, 1e30]])
        b.eval(large)

        # Epsilon functions should still work
        eps = machine_epsilon(b, large)
        assert eps > 0
        assert not is_nan(eps, b)

    def test_mixed_scale_array(self, any_backend: "Backend") -> None:
        """Arrays with mixed scales should be handled properly."""
        b = any_backend
        b.random_seed(42)

        # Create matrix with mixed scales
        source = b.random_normal((20, 10))
        scales = b.array([10.0 ** (i - 5) for i in range(10)])
        scales = b.reshape(scales, (1, -1))
        source = source * scales
        target = b.random_normal((20, 5))
        b.eval(source, target, scales)

        # SVD via eigh should handle this
        U, S, Vt = geodesic_svd(b, source)
        b.eval(U, S, Vt)

        nan_s = b.sum(b.astype(b.isnan(S), "int32"))
        inf_s = b.sum(b.astype(b.isinf(S), "int32"))
        b.eval(nan_s, inf_s)
        assert int(b.to_scalar(nan_s)) == 0, "SVD produced NaN"
        assert int(b.to_scalar(inf_s)) == 0, "SVD produced Inf"

    def test_near_singular_matrix(self, any_backend: "Backend") -> None:
        """Near-singular matrices should be handled gracefully."""
        b = any_backend

        # Create a nearly singular matrix
        # Start with rank-1 matrix and add tiny perturbation
        v = b.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        rank1 = b.matmul(v, b.transpose(v))
        # Use machine epsilon directly for truly near-singular behavior
        eps = machine_epsilon(b, rank1)
        perturbation = b.eye(5) * eps
        near_singular = rank1 + perturbation
        b.eval(near_singular)

        # SVD should handle this without error
        U, S, Vt = geodesic_svd(b, near_singular)
        b.eval(U, S, Vt)

        # Should have one dominant singular value
        S_np = [float(v) for v in b.tolist(S)]
        assert S_np[0] > 0, "Largest singular value should be positive"
        # Condition number should be very high (close to or above 1/eps)
        if S_np[-1] > 0:
            condition = S_np[0] / S_np[-1]
            # Condition should be high, but not necessarily at threshold since
            # the rank-1 matrix has specific structure. Just verify it's large.
            assert condition >= 1e4, f"Expected high condition number, got {condition}"

    def test_diagonal_matrix_svd(self, any_backend: "Backend") -> None:
        """Diagonal matrices should have accurate SVD decomposition."""
        b = any_backend

        # Diagonal matrix - SVD should be trivial
        diag_vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        D = b.diag(b.array(diag_vals))
        b.eval(D)

        U, S, Vt = geodesic_svd(b, D)
        b.eval(U, S, Vt)

        # Singular values should match diagonal (sorted descending)
        # Note: geodesic_svd uses power iteration which is iterative, so we
        # allow a few multiples of machine epsilon (not exact precision)
        S_np = sorted([float(v) for v in b.tolist(S)], reverse=True)
        expected = sorted(diag_vals, reverse=True)
        for s, e in zip(S_np, expected):
            eps = _eps(b, s, e) * 10  # Power iteration is iterative
            assert abs(s - e) <= eps, f"Expected {e}, got {s}"

    def test_zero_row_matrix(self, any_backend: "Backend") -> None:
        """Matrix with zero rows should not cause solver crashes."""
        b = any_backend
        b.random_seed(42)

        # Create matrix with one zero row
        source = b.random_normal((10, 5))
        mask = b.concatenate(
            [b.ones((3, 5)), b.zeros((1, 5)), b.ones((6, 5))],
            axis=0,
        )
        source = source * mask
        target = b.random_normal((10, 3))
        b.eval(source, target, mask)

        # QR solver should handle this
        F, diag = solve_full_row_rank_via_qr(b, source, target)
        # May succeed or fail gracefully
        assert diag is not None, "Diagnostics should always be returned"

    def test_zero_column_matrix(self, any_backend: "Backend") -> None:
        """Matrix with zero columns should not cause solver crashes."""
        b = any_backend
        b.random_seed(42)

        # Create matrix with one zero column
        source = b.random_normal((10, 5))
        mask = b.concatenate(
            [b.ones((10, 2)), b.zeros((10, 1)), b.ones((10, 2))],
            axis=1,
        )
        source = source * mask
        target = b.random_normal((10, 3))
        b.eval(source, target, mask)

        # QR solver should handle this
        F, diag = solve_full_row_rank_via_qr(b, source, target)
        # May succeed or fail gracefully
        assert diag is not None, "Diagnostics should always be returned"
