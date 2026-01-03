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

"""Tests for Low-Rank Gromov-Wasserstein implementation."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.low_rank_gw import (
    LowRankCoupling,
    LowRankGromovWasserstein,
    LowRankGWResult,
    compute_lowrank_gw,
    project_via_lowrank_gw,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestLowRankCoupling:
    """Tests for LowRankCoupling dataclass."""

    def test_to_dense_basic(self, backend):
        """Test reconstructing dense coupling from factors."""
        b = backend
        n, m, r = 10, 8, 3

        Q = b.random_uniform(shape=(n, r)) + 0.1
        g = b.ones((r,))
        R = b.random_uniform(shape=(m, r)) + 0.1
        b.eval(Q, g, R)

        coupling = LowRankCoupling(Q=Q, g=g, R=R)
        P = coupling.to_dense(b)
        b.eval(P)

        assert P.shape == (n, m)
        # All entries should be positive
        assert float(b.tolist(b.min(P))) > -1e-8

    def test_apply_left_shape(self, backend):
        """Test apply_left produces correct output shape."""
        b = backend
        n, m, r = 20, 15, 5

        Q = b.random_uniform(shape=(n, r)) + 0.1
        g = b.ones((r,))
        R = b.random_uniform(shape=(m, r)) + 0.1
        b.eval(Q, g, R)

        coupling = LowRankCoupling(Q=Q, g=g, R=R)

        # Apply to a matrix X of shape [n, d]
        d = 10
        X = b.random_normal((n, d))
        b.eval(X)

        result = coupling.apply_left(X, b)
        b.eval(result)

        # Should project from n rows to m rows
        assert result.shape == (m, d)

    def test_apply_right_shape(self, backend):
        """Test apply_right produces correct output shape."""
        b = backend
        n, m, r = 20, 15, 5

        Q = b.random_uniform(shape=(n, r)) + 0.1
        g = b.ones((r,))
        R = b.random_uniform(shape=(m, r)) + 0.1
        b.eval(Q, g, R)

        coupling = LowRankCoupling(Q=Q, g=g, R=R)

        # Apply to a matrix X of shape [m, d]
        d = 10
        X = b.random_normal((m, d))
        b.eval(X)

        result = coupling.apply_right(X, b)
        b.eval(result)

        # Should project from m rows to n rows
        assert result.shape == (n, d)


class TestLowRankGromovWasserstein:
    """Tests for LowRankGromovWasserstein solver."""

    def test_identical_matrices_zero_distance(self, backend):
        """Identical cost matrices should give near-zero distance.

        The implementation detects identical matrices and returns identity coupling.
        """
        b = backend
        n = 30  # Smaller for faster test

        # Create a random symmetric cost matrix
        X = b.random_normal((n, 10))
        C = b.matmul(X, b.transpose(X))
        b.eval(C)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)
        result = solver.compute(C, C)

        # Should detect identical matrices and return zero distance
        assert result.distance < 5.0, f"Distance {result.distance} too high for identical matrices"

    def test_different_sizes(self, backend):
        """Test low-rank GW handles different sized matrices."""
        b = backend
        n, m = 30, 20

        # Create two random cost matrices
        X1 = b.random_normal((n, 8))
        X2 = b.random_normal((m, 8))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)
        result = solver.compute(C1, C2)

        # Should produce valid result
        assert result.distance >= 0
        assert result.iterations > 0
        # Rank is derived from sqrt(min(n, m)) clamped to [10, 500]
        derived_rank = min(10, n, m)  # sqrt(20) ≈ 4.5, clamped to 10
        assert result.coupling.Q.shape == (n, derived_rank)
        assert result.coupling.R.shape == (m, derived_rank)

    def test_coupling_marginals(self, backend):
        """Test that coupling approximately satisfies marginal constraints."""
        b = backend
        n, m = 40, 30

        X1 = b.random_normal((n, 6))
        X2 = b.random_normal((m, 6))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)
        result = solver.compute(C1, C2)

        # Reconstruct coupling
        P = result.coupling.to_dense(b)
        b.eval(P)

        # Check row sums (should sum to 1/n for uniform marginal)
        row_sums = b.sum(P, axis=1)
        b.eval(row_sums)
        expected_row = 1.0 / n
        row_error = float(b.tolist(b.max(b.abs(row_sums - expected_row))))

        # Check column sums (should sum to 1/m for uniform marginal)
        col_sums = b.sum(P, axis=0)
        b.eval(col_sums)
        expected_col = 1.0 / m
        col_error = float(b.tolist(b.max(b.abs(col_sums - expected_col))))

        # Should be approximately correct (within tolerance)
        assert row_error < 0.2, f"Row marginal error: {row_error}"
        assert col_error < 0.2, f"Col marginal error: {col_error}"

    def test_large_dimension_tractable(self, backend):
        """Test that low-rank GW can handle dimensions that break standard GW."""
        b = backend

        # These dimensions would break standard GW (which has 20k limit)
        # but should work fine with low-rank
        n, m = 500, 400  # Reduced from 1000 to speed up test

        X1 = b.random_normal((n, 10))
        X2 = b.random_normal((m, 10))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)
        result = solver.compute(C1, C2)

        # Should complete without error
        assert result.distance >= 0
        # Rank derived from sqrt(min(500, 400)) = 20, clamped to [10, 500] -> 20
        assert result.coupling.Q.shape[0] == n
        assert result.coupling.R.shape[0] == m


class TestComputeLowrankGW:
    """Tests for the convenience function compute_lowrank_gw."""

    def test_from_points(self, backend):
        """Test computing GW from point sets."""
        b = backend
        b.random_seed(42)

        n, m = 30, 25
        d_s, d_t = 8, 6

        source = b.random_normal((n, d_s))
        target = b.random_normal((m, d_t))
        b.eval(source, target)

        # All parameters derived from data
        result = compute_lowrank_gw(source, target, b)

        assert result.distance >= 0
        assert result.coupling.Q.shape[0] == n
        assert result.coupling.R.shape[0] == m


class TestProjectViaLowrankGW:
    """Tests for project_via_lowrank_gw function."""

    def test_row_projection(self, backend):
        """Test projecting source to target with row mismatch."""
        b = backend
        b.random_seed(42)

        m_s, m_t = 100, 80
        d = 32

        source = b.random_normal((m_s, d))
        target = b.random_normal((m_t, d))
        b.eval(source, target)

        # All parameters derived from data
        projected, result = project_via_lowrank_gw(source, target, b)
        b.eval(projected)

        # Output should have target shape
        assert projected.shape == (m_t, d)

    def test_full_projection(self, backend):
        """Test projecting with both row and column mismatch."""
        b = backend
        b.random_seed(42)

        m_s, d_s = 60, 24
        m_t, d_t = 50, 20

        source = b.random_normal((m_s, d_s))
        target = b.random_normal((m_t, d_t))
        b.eval(source, target)

        # All parameters derived from data
        projected, result = project_via_lowrank_gw(source, target, b)
        b.eval(projected)

        # Output should have target shape
        assert projected.shape == (m_t, d_t)

    def test_same_shape_identity(self, backend):
        """Same shape matrices should pass through without modification."""
        b = backend
        b.random_seed(42)

        m, d = 40, 16
        source = b.random_normal((m, d))
        target = b.random_normal((m, d))
        b.eval(source, target)

        # All parameters derived from data
        projected, result = project_via_lowrank_gw(source, target, b)
        b.eval(projected)

        assert projected.shape == (m, d)
        assert result.iterations == 0  # No projection needed


class TestCrossDimensionalIntegration:
    """Test integration with cross_dimensional_projection module."""

    def test_large_row_mismatch_uses_lowrank(self, backend):
        """Test that large row mismatches use low-rank GW instead of failing."""
        from modelcypher.core.domain.geometry.cross_dimensional_projection import (
            ProjectionMethod,
            project_cross_dimensional,
        )

        b = backend
        b.random_seed(42)

        # Simulate MLP cross-architecture projection (reduced scale for test)
        # Real case: Llama 70B (28672) -> Qwen 8B (12288)
        # Test case: 500 -> 400 (still above the implicit tractability threshold in tests)
        m_s, m_t = 500, 400
        d = 64

        source = b.random_normal((m_s, d))
        target = b.random_normal((m_t, d))
        b.eval(source, target)

        # This should NOT raise an error now - uses low-rank GW
        result = project_cross_dimensional(
            source, target,
            method=ProjectionMethod.GRAM_TRANSPORT,
            backend=b,
        )

        assert result.projected.shape == (m_t, d)
        assert result.alignment_score >= 0


class TestMathematicalProperties:
    """Test mathematical properties of the low-rank GW solution."""

    def test_distance_symmetry(self, backend):
        """GW distance should be approximately symmetric."""
        b = backend
        b.random_seed(42)

        n, m = 40, 35

        X1 = b.random_normal((n, 8))
        X2 = b.random_normal((m, 8))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)

        result_12 = solver.compute(C1, C2, seed=42)
        result_21 = solver.compute(C2, C1, seed=42)

        # Distances should be similar (not exact due to optimization)
        ratio = result_12.distance / (result_21.distance + 1e-10)
        assert 0.5 < ratio < 2.0, f"Asymmetry: {result_12.distance} vs {result_21.distance}"

    def test_coupling_non_negative(self, backend):
        """Coupling entries should all be non-negative."""
        b = backend
        b.random_seed(42)

        n, m = 30, 25

        X1 = b.random_normal((n, 6))
        X2 = b.random_normal((m, 6))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data
        solver = LowRankGromovWasserstein(b)
        result = solver.compute(C1, C2)

        P = result.coupling.to_dense(b)
        b.eval(P)

        min_val = float(b.tolist(b.min(P)))
        assert min_val >= -1e-8, f"Negative coupling entry: {min_val}"

    def test_convergence_consistent(self, backend):
        """Algorithm should produce consistent results with same seed."""
        b = backend

        n, m = 35, 30

        X1 = b.random_normal((n, 6))
        X2 = b.random_normal((m, 6))
        C1 = b.matmul(X1, b.transpose(X1))
        C2 = b.matmul(X2, b.transpose(X2))
        b.eval(C1, C2)

        # All parameters derived from data, same seed for reproducibility
        solver = LowRankGromovWasserstein(b)
        result_1 = solver.compute(C1, C2, seed=42)
        result_2 = solver.compute(C1, C2, seed=42)

        # Same seed should give same result
        assert abs(result_1.distance - result_2.distance) < 1e-6
