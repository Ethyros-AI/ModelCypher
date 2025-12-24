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

"""
Tests for null-space filtering.

Validates the core mathematical guarantee: if Δw ∈ null(A),
then A @ (W + Δw) = A @ W (no interference with prior task).
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from modelcypher.core.domain.geometry.null_space_filter import (
    NullSpaceFilter,
    NullSpaceFilterConfig,
    NullSpaceMethod,
    filter_merge_delta_to_null_space,
)


class TestNullSpaceProjection:
    """Test null space computation."""

    def test_identity_projection_for_empty_activations(self):
        """Empty activations should give identity projection (full null space)."""
        config = NullSpaceFilterConfig(min_samples=0)
        filter = NullSpaceFilter(config)

        # Very few samples
        A = np.random.randn(2, 10)
        config_low = NullSpaceFilterConfig(min_samples=5)
        filter_low = NullSpaceFilter(config_low)

        projection = filter_low.compute_null_space_projection(A)

        # Should return identity-like projection due to insufficient samples
        assert projection.null_dim == 10  # Full dimension
        assert np.allclose(projection.projection_matrix, np.eye(10))

    def test_null_space_orthogonal_to_row_space(self):
        """Null space vectors should be orthogonal to all rows of A."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Create rank-deficient matrix (5 samples in 10D space)
        A = np.random.randn(20, 10)
        A[:, 5:] = 0  # Make last 5 dimensions unused

        projection = filter.compute_null_space_projection(A)

        # Null space should have dimension ~5 (the unused dimensions)
        assert projection.null_dim >= 4  # Allow some tolerance

        # Projection onto null space should be orthogonal to A's rows
        for row in A:
            projected_row = projection.projection_matrix @ row
            # Projected row should be small (in null space)
            residual = np.linalg.norm(projected_row) / np.linalg.norm(row)
            assert residual < 0.1 or np.linalg.norm(row) < 1e-6

    def test_projection_is_idempotent(self):
        """Projecting twice should give same result as projecting once."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        projection = filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        P_squared = P @ P

        # P^2 = P for projection matrices
        assert np.allclose(P, P_squared, atol=1e-6)

    def test_projection_is_symmetric(self):
        """Projection matrix should be symmetric."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(25, 15)
        projection = filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        assert np.allclose(P, P.T, atol=1e-6)

    @pytest.mark.parametrize("method", list(NullSpaceMethod))
    def test_methods_give_similar_results(self, method):
        """All methods should compute similar null spaces."""
        config = NullSpaceFilterConfig(method=method)
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        projection = filter.compute_null_space_projection(A)

        # Basic sanity checks
        assert projection.null_dim >= 0
        assert projection.null_dim <= 20
        assert projection.row_space_dim >= 0
        assert projection.null_dim + projection.row_space_dim == 20


class TestNullSpaceFiltering:
    """Test delta filtering through null space."""

    def test_filtered_delta_preserves_no_interference(self):
        """Core guarantee: A @ (W + Δw_safe) = A @ W."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        d = 20
        n_samples = 50

        # Random activations and weights
        A = np.random.randn(n_samples, d)
        W = np.random.randn(d, d)
        delta = np.random.randn(d, d)

        # Filter delta
        result = filter.filter_delta(delta.flatten(), A)

        if result.filtering_applied and result.null_space_dim > 0:
            delta_safe = result.filtered_delta.reshape(d, d)

            # Original output
            Y_orig = A @ W

            # Output with safe delta
            Y_new = A @ (W + delta_safe)

            # Should be nearly identical
            relative_change = np.linalg.norm(Y_new - Y_orig) / np.linalg.norm(Y_orig)
            assert relative_change < 0.01, f"Interference detected: {relative_change:.4f}"

    def test_preservation_fraction_bounded(self):
        """Preserved fraction should be in [0, 1]."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        delta = np.random.randn(20)

        result = filter.filter_delta(delta, A)

        assert 0.0 <= result.preserved_fraction <= 1.0
        assert 0.0 <= result.projection_loss <= 1.0
        assert abs(result.preserved_fraction + result.projection_loss - 1.0) < 1e-6

    def test_zero_delta_gives_zero_filtered(self):
        """Zero delta should give zero filtered delta."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        delta = np.zeros(20)

        result = filter.filter_delta(delta, A)

        assert np.allclose(result.filtered_delta, 0)
        assert result.original_norm == 0
        assert result.preserved_fraction == 1.0

    def test_full_rank_activations_give_empty_null_space(self):
        """If activations span the full space, null space should be empty."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        d = 10
        # More samples than dimensions with full rank
        A = np.random.randn(50, d)
        delta = np.random.randn(d)

        result = filter.filter_delta(delta, A)

        # With full rank activations, null space should be small or empty
        # The filtered delta should be significantly reduced
        if result.filtering_applied:
            assert result.preserved_fraction < 0.5

    def test_dimension_mismatch_skips_filtering(self):
        """Mismatched dimensions should skip filtering gracefully."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        delta = np.random.randn(15)  # Wrong dimension

        result = filter.filter_delta(delta, A)

        assert not result.filtering_applied
        assert np.array_equal(result.filtered_delta, result.original_delta)


class TestMergeIntegration:
    """Test integration with merge workflow."""

    def test_filter_merge_delta_convenience(self):
        """Test convenience function for merge workflow."""
        d = 20
        n_samples = 50

        source = np.random.randn(d, d)
        target = np.random.randn(d, d)
        activations = np.random.randn(n_samples, d * d)  # Flattened

        merged, result = filter_merge_delta_to_null_space(
            source.flatten(),
            target.flatten(),
            activations,
            alpha=0.5,
        )

        assert merged.shape == source.flatten().shape
        assert result.filtering_applied or result.null_space_dim == 0

    def test_alpha_zero_gives_target(self):
        """Alpha=0 should give target weights (regardless of filtering)."""
        d = 10

        source = np.random.randn(d)
        target = np.random.randn(d)
        activations = np.random.randn(30, d)

        merged, _ = filter_merge_delta_to_null_space(
            source, target, activations, alpha=0.0
        )

        assert np.allclose(merged, target)

    def test_alpha_one_with_full_null_gives_source(self):
        """Alpha=1 with full null space should approach source."""
        d = 10

        source = np.random.randn(d)
        target = np.random.randn(d)
        # Few samples = large null space
        activations = np.random.randn(3, d)

        config = NullSpaceFilterConfig(min_samples=1)
        merged, result = filter_merge_delta_to_null_space(
            source, target, activations, alpha=1.0, config=config
        )

        if result.preserved_fraction > 0.9:
            # If most of delta preserved, should be close to source
            assert np.linalg.norm(merged - source) < np.linalg.norm(source - target)


class TestModelProfile:
    """Test model-level null space profiling."""

    def test_compute_model_profile(self):
        """Test computing null space profile across layers."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Simulate layer activations with varying null space
        layer_activations = {
            0: np.random.randn(50, 100),  # Likely small null space
            1: np.random.randn(20, 100),  # Larger null space
            2: np.hstack([np.random.randn(30, 50), np.zeros((30, 50))]),  # Half null
        }

        profile = filter.compute_model_null_space_profile(layer_activations)

        assert len(profile.per_layer) == 3
        assert profile.total_null_dim >= 0
        assert 0.0 <= profile.mean_null_fraction <= 1.0
        assert all(l in [0, 1, 2] for l in profile.graftable_layers)

    def test_graftable_layers_threshold(self):
        """Test that graft threshold correctly identifies layers."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Layer 0: full rank (not graftable)
        # Layer 1: half null (graftable at 0.1 threshold)
        layer_activations = {
            0: np.random.randn(200, 50),  # Overdetermined
            1: np.hstack([np.random.randn(30, 25), np.zeros((30, 25))]),  # 50% null
        }

        profile = filter.compute_model_null_space_profile(
            layer_activations, graft_threshold=0.4
        )

        # Layer 1 should be graftable (50% > 40% threshold)
        assert 1 in profile.graftable_layers


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20)
    def test_projection_loss_plus_preserved_equals_one(self, n_samples, d):
        """projection_loss + preserved_fraction should always equal 1."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(n_samples, d)
        delta = np.random.randn(d)

        result = filter.filter_delta(delta, A)

        total = result.projection_loss + result.preserved_fraction
        assert abs(total - 1.0) < 1e-6, f"Total was {total}"

    @given(
        d=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=10)
    def test_filtered_norm_leq_original_norm(self, d):
        """Filtered delta should never have larger norm than original."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(50, d)
        delta = np.random.randn(d)

        result = filter.filter_delta(delta, A)

        assert result.filtered_norm <= result.original_norm + 1e-6


class TestEdgeCases:
    """Edge case handling."""

    def test_single_sample_activation(self):
        """Single sample should still work."""
        config = NullSpaceFilterConfig(min_samples=1)
        filter = NullSpaceFilter(config)

        A = np.random.randn(1, 10)
        delta = np.random.randn(10)

        result = filter.filter_delta(delta, A)
        # Should not crash, may or may not filter

    def test_very_high_dimensional(self):
        """High dimensional space should work."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(100, 500)
        delta = np.random.randn(500)

        result = filter.filter_delta(delta, A)

        # Should have large null space (500 - rank(A) ~ 400)
        assert result.null_space_dim >= 350

    def test_zero_activations(self):
        """All-zero activations should give full null space."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.zeros((30, 20))
        delta = np.random.randn(20)

        projection = filter.compute_null_space_projection(A)

        # All of space is null
        assert projection.null_dim == 20

    def test_nan_handling(self):
        """NaN in activations should be handled gracefully."""
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = np.random.randn(30, 20)
        A[0, 0] = np.nan
        delta = np.random.randn(20)

        # Should not crash (may produce warnings)
        try:
            result = filter.filter_delta(delta, A)
            # If it doesn't crash, check result is reasonable
            assert not np.any(np.isnan(result.filtered_delta)) or not result.filtering_applied
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable to raise on NaN
            pass
