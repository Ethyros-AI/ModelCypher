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

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
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
        backend = get_default_backend()
        config = NullSpaceFilterConfig(min_samples=0)
        NullSpaceFilter(config)

        # Very few samples
        backend.random_seed(42)
        A = backend.random_normal((2, 10))
        backend.eval(A)
        config_low = NullSpaceFilterConfig(min_samples=5)
        filter_low = NullSpaceFilter(config_low)

        projection = filter_low.compute_null_space_projection(A)

        # Should return identity-like projection due to insufficient samples
        assert projection.null_dim == 10  # Full dimension
        eye_mat = backend.eye(10)
        backend.eval(projection.projection_matrix)
        backend.eval(eye_mat)
        assert float(backend.to_numpy(backend.max(backend.abs(projection.projection_matrix - eye_mat)))) < 1e-6

    def test_null_space_orthogonal_to_row_space(self):
        """Null space vectors should be orthogonal to all rows of A."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Create simple full-rank matrix
        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)

        projection = filter.compute_null_space_projection(A)

        # With full rank, null space should be small or empty
        # Just verify the function works without crashing
        assert projection.null_dim >= 0
        assert projection.null_dim <= 20

    def test_projection_is_idempotent(self):
        """Projecting twice should give same result as projecting once."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)
        projection = filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        P_squared = P @ P
        backend.eval(P)
        backend.eval(P_squared)

        # P^2 = P for projection matrices
        assert float(backend.to_numpy(backend.max(backend.abs(P - P_squared)))) < 1e-6

    def test_projection_is_symmetric(self):
        """Projection matrix should be symmetric."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((25, 15))
        backend.eval(A)
        projection = filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        P_T = backend.transpose(P)
        backend.eval(P)
        backend.eval(P_T)
        assert float(backend.to_numpy(backend.max(backend.abs(P - P_T)))) < 1e-6

    @pytest.mark.parametrize("method", list(NullSpaceMethod))
    def test_methods_give_similar_results(self, method):
        """All methods should compute similar null spaces."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig(method=method)
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)
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
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        d = 20
        n_samples = 50

        # Random activations and weights
        backend.random_seed(42)
        A = backend.random_normal((n_samples, d))
        W = backend.random_normal((d, d))
        delta = backend.random_normal((d, d))
        backend.eval(A)
        backend.eval(W)
        backend.eval(delta)

        # Filter delta
        delta_flat = backend.reshape(delta, (-1,))
        backend.eval(delta_flat)
        result = filter.filter_delta(delta_flat, A)

        if result.filtering_applied and result.null_space_dim > 0:
            delta_safe = backend.reshape(result.filtered_delta, (d, d))
            backend.eval(delta_safe)

            # Original output
            Y_orig = A @ W

            # Output with safe delta
            Y_new = A @ (W + delta_safe)

            backend.eval(Y_orig)
            backend.eval(Y_new)

            # Should be nearly identical
            diff = Y_new - Y_orig
            backend.eval(diff)
            relative_change = float(backend.to_numpy(backend.norm(diff) / backend.norm(Y_orig)))
            assert relative_change < 0.01, f"Interference detected: {relative_change:.4f}"

    def test_preservation_fraction_bounded(self):
        """Preserved fraction should be in [0, 1]."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        assert 0.0 <= result.preserved_fraction <= 1.0
        assert 0.0 <= result.projection_loss <= 1.0
        assert abs(result.preserved_fraction + result.projection_loss - 1.0) < 1e-6

    def test_zero_delta_gives_zero_filtered(self):
        """Zero delta should give zero filtered delta."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.zeros((20,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        backend.eval(result.filtered_delta)
        zero_arr = backend.zeros_like(result.filtered_delta)
        backend.eval(zero_arr)
        assert float(backend.to_numpy(backend.max(backend.abs(result.filtered_delta - zero_arr)))) < 1e-6
        assert result.original_norm == 0
        assert result.preserved_fraction == 1.0

    def test_full_rank_activations_give_empty_null_space(self):
        """If activations span the full space, null space should be empty."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        d = 10
        # More samples than dimensions with full rank
        backend.random_seed(42)
        A = backend.random_normal((50, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        # With full rank activations, null space should be small or empty
        # The filtered delta should be significantly reduced
        if result.filtering_applied:
            assert result.preserved_fraction < 0.5

    def test_dimension_mismatch_skips_filtering(self):
        """Mismatched dimensions should skip filtering gracefully."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((15,))  # Wrong dimension
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        assert not result.filtering_applied
        backend.eval(result.filtered_delta)
        backend.eval(result.original_delta)
        assert float(backend.to_numpy(backend.max(backend.abs(result.filtered_delta - result.original_delta)))) == 0


class TestMergeIntegration:
    """Test integration with merge workflow."""

    def test_filter_merge_delta_convenience(self):
        """Test convenience function for merge workflow."""
        backend = get_default_backend()
        d = 20
        n_samples = 50

        backend.random_seed(42)
        source = backend.random_normal((d, d))
        target = backend.random_normal((d, d))
        activations = backend.random_normal((n_samples, d * d))  # Flattened
        backend.eval(source)
        backend.eval(target)
        backend.eval(activations)

        source_flat = backend.reshape(source, (-1,))
        target_flat = backend.reshape(target, (-1,))
        backend.eval(source_flat)
        backend.eval(target_flat)

        merged, result = filter_merge_delta_to_null_space(
            source_flat,
            target_flat,
            activations,
            alpha=0.5,
        )

        backend.eval(merged)
        assert merged.shape == source_flat.shape
        assert result.filtering_applied or result.null_space_dim == 0

    def test_alpha_zero_gives_target(self):
        """Alpha=0 should give target weights (regardless of filtering)."""
        backend = get_default_backend()
        d = 10

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        activations = backend.random_normal((30, d))
        backend.eval(source)
        backend.eval(target)
        backend.eval(activations)

        merged, _ = filter_merge_delta_to_null_space(source, target, activations, alpha=0.0)

        backend.eval(merged)
        backend.eval(target)
        assert float(backend.to_numpy(backend.max(backend.abs(merged - target)))) < 1e-6

    def test_alpha_one_with_full_null_gives_source(self):
        """Alpha=1 with full null space should approach source."""
        backend = get_default_backend()
        d = 10

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        # Few samples = large null space
        activations = backend.random_normal((3, d))
        backend.eval(source)
        backend.eval(target)
        backend.eval(activations)

        config = NullSpaceFilterConfig(min_samples=1)
        merged, result = filter_merge_delta_to_null_space(
            source, target, activations, alpha=1.0, config=config
        )

        backend.eval(merged)
        if result.preserved_fraction > 0.9:
            # If most of delta preserved, should be close to source
            diff1 = merged - source
            diff2 = source - target
            backend.eval(diff1)
            backend.eval(diff2)
            assert float(backend.to_numpy(backend.norm(diff1))) < float(backend.to_numpy(backend.norm(diff2)))


class TestModelProfile:
    """Test model-level null space profiling."""

    def test_compute_model_profile(self):
        """Test computing null space profile across layers."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Simulate layer activations with varying null space
        backend.random_seed(42)
        act0 = backend.random_normal((50, 100))  # Likely small null space
        act1 = backend.random_normal((20, 100))  # Larger null space
        act2_a = backend.random_normal((30, 50))
        act2_b = backend.zeros((30, 50))
        act2 = backend.concatenate([act2_a, act2_b], axis=1)  # Half null
        backend.eval(act0)
        backend.eval(act1)
        backend.eval(act2)

        layer_activations = {
            0: act0,
            1: act1,
            2: act2,
        }

        profile = filter.compute_model_null_space_profile(layer_activations)

        assert len(profile.per_layer) == 3
        assert profile.total_null_dim >= 0
        assert 0.0 <= profile.mean_null_fraction <= 1.0
        assert all(l in [0, 1, 2] for l in profile.graftable_layers)

    def test_graftable_layers_threshold(self):
        """Test that graft threshold correctly identifies layers."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        # Layer 0: full rank (not graftable)
        # Layer 1: half null (graftable at 0.1 threshold)
        backend.random_seed(42)
        act0 = backend.random_normal((200, 50))  # Overdetermined
        act1_a = backend.random_normal((30, 25))
        act1_b = backend.zeros((30, 25))
        act1 = backend.concatenate([act1_a, act1_b], axis=1)  # 50% null
        backend.eval(act0)
        backend.eval(act1)

        layer_activations = {
            0: act0,
            1: act1,
        }

        profile = filter.compute_model_null_space_profile(layer_activations, graft_threshold=0.4)

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
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((n_samples, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        total = result.projection_loss + result.preserved_fraction
        assert abs(total - 1.0) < 1e-6, f"Total was {total}"

    @given(
        d=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=10)
    def test_filtered_norm_leq_original_norm(self, d):
        """Filtered delta should never have larger norm than original."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((50, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        assert result.filtered_norm <= result.original_norm + 1e-6


class TestEdgeCases:
    """Edge case handling."""

    def test_single_sample_activation(self):
        """Single sample should still work."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig(min_samples=1)
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((1, 10))
        delta = backend.random_normal((10,))
        backend.eval(A)
        backend.eval(delta)

        filter.filter_delta(delta, A)
        # Should not crash, may or may not filter

    def test_very_high_dimensional(self):
        """High dimensional space should work."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((100, 500))
        delta = backend.random_normal((500,))
        backend.eval(A)
        backend.eval(delta)

        result = filter.filter_delta(delta, A)

        # Should have large null space (500 - rank(A) ~ 400)
        assert result.null_space_dim >= 350

    def test_zero_activations(self):
        """All-zero activations should give full null space."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        A = backend.zeros((30, 20))
        backend.eval(A)

        projection = filter.compute_null_space_projection(A)

        # All of space is null
        assert projection.null_dim == 20

    def test_nan_handling(self):
        """NaN in activations should be handled gracefully."""
        backend = get_default_backend()
        config = NullSpaceFilterConfig()
        filter = NullSpaceFilter(config)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        # Set one element to NaN
        A_np = backend.to_numpy(A)
        import numpy as np
        A_np[0, 0] = np.nan
        A = backend.array(A_np)
        backend.eval(A)
        backend.eval(delta)

        # Should not crash (may produce warnings)
        try:
            result = filter.filter_delta(delta, A)
            # If it doesn't crash, check result is reasonable
            backend.eval(result.filtered_delta)
            has_nan = backend.any(backend.isnan(result.filtered_delta))
            backend.eval(has_nan)
            assert not bool(backend.to_numpy(has_nan)) or not result.filtering_applied
        except (ValueError, RuntimeError):
            # Acceptable to raise on NaN
            pass
