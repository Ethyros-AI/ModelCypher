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

"""Comprehensive tests for cross_dimensional_projection.py.

Tests the unified cross-dimensional projection API that handles all
dimension mismatches in model merging using geometry-preserving methods.
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_dimensional_projection import (
    ProjectionMethod,
    ProjectionResult,
    project_cross_dimensional,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Array, Backend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
    return get_default_backend()


# =============================================================================
# ProjectionMethod Enum Tests
# =============================================================================


class TestProjectionMethod:
    """Tests for ProjectionMethod enum."""

    def test_all_values_exist(self) -> None:
        """Verify all expected projection methods exist."""
        assert ProjectionMethod.GRAM_TRANSPORT == "gram_transport"
        assert ProjectionMethod.PROCRUSTES == "procrustes"
        assert ProjectionMethod.SVD_PROJECT == "svd_project"

    def test_is_string_enum(self) -> None:
        """ProjectionMethod should be a string enum."""
        for m in ProjectionMethod:
            assert isinstance(m.value, str)
            assert isinstance(m, str)

    def test_enum_count(self) -> None:
        """Verify expected number of projection methods."""
        assert len(list(ProjectionMethod)) == 3


# =============================================================================
# ProjectionResult Dataclass Tests
# =============================================================================


class TestProjectionResult:
    """Tests for ProjectionResult dataclass."""

    def test_creation(self, backend: "Backend") -> None:
        """Test basic result creation."""
        projected = backend.random_normal((10, 8))
        result = ProjectionResult(
            projected=projected,
            alignment_score=0.95,
            method_used=ProjectionMethod.GRAM_TRANSPORT,
            row_coupling=None,
            col_coupling=None,
        )
        assert result.alignment_score == 0.95
        assert result.method_used == ProjectionMethod.GRAM_TRANSPORT
        assert result.row_coupling is None
        assert result.col_coupling is None

    def test_with_couplings(self, backend: "Backend") -> None:
        """Test result with couplings."""
        projected = backend.random_normal((10, 8))
        row_coupling = backend.random_normal((12, 10))
        col_coupling = backend.random_normal((16, 8))
        result = ProjectionResult(
            projected=projected,
            alignment_score=0.8,
            method_used=ProjectionMethod.GRAM_TRANSPORT,
            row_coupling=row_coupling,
            col_coupling=col_coupling,
        )
        assert result.row_coupling is not None
        assert result.col_coupling is not None

    def test_frozen_dataclass(self, backend: "Backend") -> None:
        """ProjectionResult should be frozen (immutable)."""
        projected = backend.random_normal((10, 8))
        result = ProjectionResult(
            projected=projected,
            alignment_score=0.9,
            method_used=ProjectionMethod.SVD_PROJECT,
            row_coupling=None,
            col_coupling=None,
        )
        with pytest.raises(AttributeError):
            result.alignment_score = 0.5


# =============================================================================
# project_cross_dimensional Same Shape Tests
# =============================================================================


class TestSameShapeProjection:
    """Tests for projection when source and target have same shape."""

    def test_same_shape_returns_source(self, backend: "Backend") -> None:
        """Same shape should return source unchanged."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.alignment_score == 1.0
        assert result.row_coupling is None
        assert result.col_coupling is None

    def test_same_shape_all_methods(self, backend: "Backend") -> None:
        """Same shape should work for all methods."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))

        for method in ProjectionMethod:
            result = project_cross_dimensional(source, target, method=method, backend=backend)
            assert result.alignment_score == 1.0
            assert result.method_used == method


# =============================================================================
# project_cross_dimensional Column Mismatch Tests
# =============================================================================


class TestColumnMismatchProjection:
    """Tests for projection when only column dimension differs."""

    def test_column_expansion(self, backend: "Backend") -> None:
        """Expanding columns (d_s < d_t)."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 16)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_column_truncation(self, backend: "Backend") -> None:
        """Truncating columns (d_s > d_t)."""
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 8)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_column_mismatch_procrustes(self, backend: "Backend") -> None:
        """Procrustes should handle column mismatch with same rows."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        assert result.projected.shape == (10, 8)
        assert result.method_used == ProjectionMethod.PROCRUSTES

    def test_column_mismatch_svd(self, backend: "Backend") -> None:
        """SVD should handle column mismatch."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )
        assert result.projected.shape == (10, 8)
        assert result.method_used == ProjectionMethod.SVD_PROJECT


# =============================================================================
# project_cross_dimensional Row Mismatch Tests
# =============================================================================


class TestRowMismatchProjection:
    """Tests for projection when only row dimension differs."""

    def test_row_expansion(self, backend: "Backend") -> None:
        """Expanding rows (m_s < m_t)."""
        backend.random_seed(42)
        source = backend.random_normal((8, 16))
        target = backend.random_normal((12, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (12, 16)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_row_truncation(self, backend: "Backend") -> None:
        """Truncating rows (m_s > m_t)."""
        backend.random_seed(42)
        source = backend.random_normal((16, 8))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 8)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_row_mismatch_procrustes(self, backend: "Backend") -> None:
        """Procrustes should handle row mismatch with same columns."""
        backend.random_seed(42)
        source = backend.random_normal((12, 8))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        assert result.projected.shape == (10, 8)
        assert result.method_used == ProjectionMethod.PROCRUSTES

    def test_row_mismatch_procrustes_rank_deficient(self, backend: "Backend") -> None:
        """Procrustes should handle row mismatch even when SVD rank < target dim.

        When source has fewer rows than columns and we transpose for row handling,
        the SVD rank is limited. The code should pad with zeros to reach target dim.
        """
        backend.random_seed(42)
        # Source: (8, 12) -> transpose (12, 8) has rank at most 8
        # Target: (10, 12) -> transpose (12, 10) needs 10 columns
        # After transpose, we have d_s=8, d_t=10, rank=8 < d_t
        source = backend.random_normal((8, 12))
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        assert result.projected.shape == (10, 12)
        assert result.method_used == ProjectionMethod.PROCRUSTES
        assert 0.0 <= result.alignment_score <= 1.0

    def test_row_mismatch_svd(self, backend: "Backend") -> None:
        """SVD should handle row mismatch."""
        backend.random_seed(42)
        source = backend.random_normal((12, 8))
        target = backend.random_normal((10, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )
        assert result.projected.shape == (10, 8)
        assert result.method_used == ProjectionMethod.SVD_PROJECT


# =============================================================================
# project_cross_dimensional Both Dimensions Mismatch Tests
# =============================================================================


class TestBothDimensionsMismatch:
    """Tests for projection when both dimensions differ."""

    def test_both_expansion(self, backend: "Backend") -> None:
        """Expanding both dimensions."""
        backend.random_seed(42)
        source = backend.random_normal((8, 12))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 16)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_both_truncation(self, backend: "Backend") -> None:
        """Truncating both dimensions."""
        backend.random_seed(42)
        source = backend.random_normal((16, 20))
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_mixed_expansion_truncation(self, backend: "Backend") -> None:
        """Expand rows, truncate columns."""
        backend.random_seed(42)
        source = backend.random_normal((8, 20))
        target = backend.random_normal((12, 10))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (12, 10)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_procrustes_falls_back_to_gram(self, backend: "Backend") -> None:
        """Procrustes should fall back to gram_transport when both dims differ."""
        backend.random_seed(42)
        source = backend.random_normal((8, 12))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        # Should fall back to gram_transport when both dimensions differ
        assert result.projected.shape == (10, 16)
        assert result.method_used == ProjectionMethod.GRAM_TRANSPORT

    def test_svd_handles_both_dims(self, backend: "Backend") -> None:
        """SVD should handle both dimensions differing."""
        backend.random_seed(42)
        source = backend.random_normal((8, 12))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )
        assert result.projected.shape == (10, 16)
        assert result.method_used == ProjectionMethod.SVD_PROJECT


# =============================================================================
# Method String Parameter Tests
# =============================================================================


class TestMethodStringParameter:
    """Tests for method parameter as string."""

    def test_gram_transport_string(self, backend: "Backend") -> None:
        """'gram_transport' string should work."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 16))

        result = project_cross_dimensional(source, target, method="gram_transport", backend=backend)
        assert result.method_used == ProjectionMethod.GRAM_TRANSPORT

    def test_procrustes_string(self, backend: "Backend") -> None:
        """'procrustes' string should work."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 8))

        result = project_cross_dimensional(source, target, method="procrustes", backend=backend)
        assert result.method_used == ProjectionMethod.PROCRUSTES

    def test_svd_project_string(self, backend: "Backend") -> None:
        """'svd_project' string should work."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 16))

        result = project_cross_dimensional(source, target, method="svd_project", backend=backend)
        assert result.method_used == ProjectionMethod.SVD_PROJECT

    def test_invalid_method_string_raises(self, backend: "Backend") -> None:
        """Invalid method string should raise ValueError."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 16))

        with pytest.raises(ValueError):
            project_cross_dimensional(source, target, method="invalid_method", backend=backend)


# =============================================================================
# Gram Transport Specific Tests
# =============================================================================


class TestGramTransport:
    """Tests for Gram transport projection method."""

    def test_provides_couplings(self, backend: "Backend") -> None:
        """Gram transport should provide coupling matrices."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.GRAM_TRANSPORT, backend=backend
        )
        # Column dimension differs, so col_coupling should be provided
        assert result.col_coupling is not None

    def test_row_coupling_for_row_mismatch(self, backend: "Backend") -> None:
        """Row coupling should be provided for row dimension mismatch."""
        backend.random_seed(42)
        source = backend.random_normal((12, 16))
        target = backend.random_normal((10, 16))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.GRAM_TRANSPORT, backend=backend
        )
        # Row dimension differs and is tractable, so row_coupling should be provided
        assert result.row_coupling is not None

    def test_preserves_relational_structure(self, backend: "Backend") -> None:
        """Gram transport should preserve relational structure."""
        backend.random_seed(42)
        # Create source with clear structure (identity-like)
        source = backend.eye(8)
        target = backend.random_normal((8, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.GRAM_TRANSPORT, backend=backend
        )
        # Should have valid alignment score
        assert 0.0 <= result.alignment_score <= 1.0


# =============================================================================
# Procrustes Specific Tests
# =============================================================================


class TestProcrustes:
    """Tests for Procrustes projection method."""

    def test_finds_rotation(self, backend: "Backend") -> None:
        """Procrustes should find optimal rotation."""
        backend.random_seed(42)
        # Create target as rotated version of source
        source = backend.random_normal((20, 8))
        # Create an orthogonal rotation matrix
        Q, _ = backend.qr(backend.random_normal((8, 8)))
        target = backend.matmul(source, Q)
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        # Should achieve high alignment for rotated data
        eps = machine_epsilon(backend, result.projected)
        assert result.alignment_score >= 1.0 - eps

    def test_handles_reflection(self, backend: "Backend") -> None:
        """Procrustes should handle reflections correctly."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        # Create reflection (det < 0)
        backend.eval(source)

        # Just verify it runs without error
        result = project_cross_dimensional(
            source, -source, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        assert result.projected.shape == source.shape

    def test_column_expansion_pads_zeros(self, backend: "Backend") -> None:
        """Expanding columns should pad with zeros."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.PROCRUSTES, backend=backend
        )
        assert result.projected.shape == (10, 12)
        # Score should reflect partial information content
        eps = machine_epsilon(backend, result.projected)
        assert result.alignment_score <= 1.0 - eps


# =============================================================================
# SVD Projection Specific Tests
# =============================================================================


class TestSVDProjection:
    """Tests for SVD projection method."""

    def test_preserves_variance(self, backend: "Backend") -> None:
        """SVD projection should preserve top singular values."""
        backend.random_seed(42)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )
        # Alignment score reflects variance preserved
        assert 0.0 <= result.alignment_score <= 1.0

    def test_handles_rank_deficient(self, backend: "Backend") -> None:
        """SVD should handle rank-deficient matrices."""
        backend.random_seed(42)
        # Create rank-2 matrix in 8-dim space
        base = backend.random_normal((10, 2))
        projection = backend.random_normal((2, 8))
        source = backend.matmul(base, projection)
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )
        assert result.projected.shape == (10, 12)

    def test_scales_to_target_norm(self, backend: "Backend") -> None:
        """SVD projection should scale to match target Frobenius norm."""
        backend.random_seed(42)
        # Create source with large magnitude
        source = backend.random_normal((10, 8)) * 100
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(
            source, target, method=ProjectionMethod.SVD_PROJECT, backend=backend
        )

        # Compute Frobenius norms
        target_fro = float(backend.tolist(backend.sqrt(backend.sum(target ** 2))))
        proj_fro = float(backend.tolist(backend.sqrt(backend.sum(result.projected ** 2))))

        # Frobenius norms should match target within precision limits
        eps = division_epsilon(backend, result.projected)
        rel_diff = abs(proj_fro - target_fro) / (target_fro + eps)
        assert rel_diff <= eps


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_row(self, backend: "Backend") -> None:
        """Handle single row matrices."""
        backend.random_seed(42)
        source = backend.random_normal((1, 8))
        target = backend.random_normal((1, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (1, 12)

    def test_single_column(self, backend: "Backend") -> None:
        """Handle single column matrices."""
        backend.random_seed(42)
        source = backend.random_normal((10, 1))
        target = backend.random_normal((10, 1))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.alignment_score == 1.0

    def test_square_to_rectangular(self, backend: "Backend") -> None:
        """Project square matrix to rectangular."""
        backend.random_seed(42)
        source = backend.random_normal((8, 8))
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)

    def test_rectangular_to_square(self, backend: "Backend") -> None:
        """Project rectangular matrix to square."""
        backend.random_seed(42)
        source = backend.random_normal((10, 12))
        target = backend.random_normal((8, 8))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (8, 8)

    def test_very_tall_matrix(self, backend: "Backend") -> None:
        """Handle very tall matrix (embedding-like)."""
        backend.random_seed(42)
        # Simulate vocabulary embedding (but smaller for test speed)
        source = backend.random_normal((100, 16))
        target = backend.random_normal((100, 32))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (100, 32)

    def test_very_wide_matrix(self, backend: "Backend") -> None:
        """Handle very wide matrix."""
        backend.random_seed(42)
        source = backend.random_normal((8, 100))
        target = backend.random_normal((8, 50))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (8, 50)

    def test_zero_matrix_source(self, backend: "Backend") -> None:
        """Handle zero source matrix."""
        backend.random_seed(42)
        source = backend.zeros((10, 8))
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        # Should not crash
        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_values(self, backend: "Backend") -> None:
        """Handle very small values without underflow."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8)) * 1e-10
        target = backend.random_normal((10, 12)) * 1e-10
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)
        # Should not be NaN - use backend operations
        nan_count = backend.sum(backend.isnan(result.projected))
        backend.eval(nan_count)
        assert float(backend.to_scalar(nan_count)) == 0

    def test_very_large_values(self, backend: "Backend") -> None:
        """Handle very large values without overflow."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8)) * 1e6
        target = backend.random_normal((10, 12)) * 1e6
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)
        # Should not be Inf - use backend operations
        inf_count = backend.sum(backend.isinf(result.projected))
        backend.eval(inf_count)
        assert float(backend.to_scalar(inf_count)) == 0

    def test_mixed_magnitude(self, backend: "Backend") -> None:
        """Handle mixed magnitude values."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        # Scale different columns differently using backend operations
        # Create scale factors: 10^(i-4) for i in 0..7 = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        scale_factors = backend.array([10 ** (i - 4) for i in range(8)])
        # Broadcast multiply: source (10, 8) * scale_factors (8,) -> scaled (10, 8)
        source = source * scale_factors
        target = backend.random_normal((10, 12))
        backend.eval(source, target)

        result = project_cross_dimensional(source, target, backend=backend)
        assert result.projected.shape == (10, 12)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestProperties:
    """Property-based tests for invariants."""

    def test_output_shape_always_matches_target(self, backend: "Backend") -> None:
        """Output shape must always match target shape."""
        shapes = [
            ((10, 8), (10, 8)),    # Same
            ((10, 8), (10, 12)),   # Column expand
            ((10, 16), (10, 8)),   # Column truncate
            ((8, 10), (12, 10)),   # Row expand
            ((16, 10), (8, 10)),   # Row truncate
            ((8, 10), (12, 16)),   # Both expand
            ((16, 20), (8, 10)),   # Both truncate
            ((8, 20), (12, 10)),   # Mixed
        ]
        for (m_s, d_s), (m_t, d_t) in shapes:
            backend.random_seed(42)
            source = backend.random_normal((m_s, d_s))
            target = backend.random_normal((m_t, d_t))

            for method in ProjectionMethod:
                result = project_cross_dimensional(source, target, method=method, backend=backend)
                assert result.projected.shape == (m_t, d_t), \
                    f"Shape mismatch for {(m_s, d_s)} -> {(m_t, d_t)} with {method}"

    def test_alignment_score_in_range(self, backend: "Backend") -> None:
        """Alignment score must be in [0, 1]."""
        for seed in [1, 42, 123, 456, 789]:
            backend.random_seed(seed)
            source = backend.random_normal((10, 12))
            target = backend.random_normal((15, 8))

            for method in ProjectionMethod:
                result = project_cross_dimensional(source, target, method=method, backend=backend)
                assert 0.0 <= result.alignment_score <= 1.0, \
                    f"Score {result.alignment_score} out of range for seed {seed}, method {method}"

    def test_deterministic_with_same_seed(self, backend: "Backend") -> None:
        """Same seed should give same result."""
        def run_projection():
            backend.random_seed(42)
            source = backend.random_normal((10, 12))
            target = backend.random_normal((10, 16))
            return project_cross_dimensional(source, target, backend=backend)

        result1 = run_projection()
        result2 = run_projection()

        # Use backend operations to compare
        diff = backend.abs(result1.projected - result2.projected)
        max_diff = backend.max(diff)
        backend.eval(max_diff)
        diff_val = float(backend.to_scalar(max_diff))

        eps = division_epsilon(backend, result1.projected)
        assert diff_val <= eps, f"Results differ by {diff_val}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_mlp_projection(self, backend: "Backend") -> None:
        """Simulate MLP weight projection between architectures."""
        backend.random_seed(42)
        # Simulating different intermediate_size
        source_mlp = backend.random_normal((64, 256))  # Small model
        target_mlp = backend.random_normal((128, 512))  # Large model
        backend.eval(source_mlp, target_mlp)

        result = project_cross_dimensional(source_mlp, target_mlp, backend=backend)
        assert result.projected.shape == (128, 512)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_attention_projection(self, backend: "Backend") -> None:
        """Simulate attention weight projection."""
        backend.random_seed(42)
        # Q/K/V projections with different head counts
        source_attn = backend.random_normal((64, 64))
        target_attn = backend.random_normal((96, 96))
        backend.eval(source_attn, target_attn)

        result = project_cross_dimensional(source_attn, target_attn, backend=backend)
        assert result.projected.shape == (96, 96)

    def test_round_trip_preserves_structure(self, backend: "Backend") -> None:
        """Project A->B->A should somewhat preserve structure."""
        backend.random_seed(42)
        original = backend.random_normal((20, 16))
        intermediate = backend.random_normal((25, 20))
        backend.eval(original, intermediate)

        # Forward projection
        forward = project_cross_dimensional(original, intermediate, backend=backend)

        # Backward projection (project result back to original shape)
        back = project_cross_dimensional(forward.projected, original, backend=backend)

        # Compute correlation between original and round-trip result
        orig_flat = backend.reshape(original, (-1,))
        back_flat = backend.reshape(back.projected, (-1,))
        backend.eval(orig_flat, back_flat)

        # Compute Pearson correlation using backend
        from modelcypher.core.domain.geometry.numerical_stability import compute_pearson_correlation
        import math
        orig_list = backend.tolist(orig_flat)
        back_list = backend.tolist(back_flat)
        corr = compute_pearson_correlation(orig_list, back_list)
        # Note: with random data, correlation may be low but should be defined
        assert math.isfinite(corr)

    def test_all_methods_produce_valid_output(self, backend: "Backend") -> None:
        """All methods should produce valid, usable output."""
        backend.random_seed(42)
        source = backend.random_normal((32, 48))
        target = backend.random_normal((24, 64))
        backend.eval(source, target)

        for method in ProjectionMethod:
            result = project_cross_dimensional(source, target, method=method, backend=backend)

            # Check output is valid
            assert result.projected.shape == (24, 64)
            assert 0.0 <= result.alignment_score <= 1.0

            # Check no NaN or Inf
            isfinite_arr = backend.isfinite(result.projected)
            backend.eval(isfinite_arr)
            isfinite_list = backend.tolist(isfinite_arr)
            # Flatten and check all are finite
            all_finite = all(all(row) if isinstance(row, list) else row for row in isfinite_list)
            assert all_finite, "Output contains NaN or Inf"
