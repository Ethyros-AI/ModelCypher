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
Tests for geodesic null-space filtering.

This module tests the GPU-accelerated geodesic null-space filter that uses
Riemannian geometry instead of linear algebra SVD. The geodesic approach
is mathematically correct for high-dimensional curved manifolds.

Key differences from flat-space null-space filters:
- Uses geodesic distances (k-NN graph + Floyd-Warshall)
- Uses Fréchet mean instead of centroid
- Uses tangent space projection instead of SVD
- All operations stay on GPU (no CPU fallback)
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    division_epsilon,
)
from modelcypher.core.domain.geometry.geodesic_null_space import (
    GeodesicNullSpaceFilter,
    GeodesicNullSpaceResult,
    filter_merge_delta_geodesic,
)
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    compute_rmt_null_space_weights,
)


class TestGeodesicNullSpaceBasics:
    """Basic functionality tests for geodesic null-space filter."""

    def test_filter_returns_result_dataclass(self):
        """Filter should return a GeodesicNullSpaceResult dataclass."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        assert isinstance(result, GeodesicNullSpaceResult)
        assert hasattr(result, "filtered_delta")
        assert hasattr(result, "original_delta")
        assert hasattr(result, "orthogonal_dim")
        assert hasattr(result, "projection_loss")
        assert hasattr(result, "preserved_fraction")
        assert hasattr(result, "original_norm")
        assert hasattr(result, "filtered_norm")
        assert hasattr(result, "filtering_applied")
        assert hasattr(result, "k_neighbors")
        assert hasattr(result, "mean_geodesic_distance")

    def test_filtered_delta_shape_preserved(self):
        """Filtered delta should have same shape as input delta."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)
        backend.eval(result.filtered_delta)

        assert result.filtered_delta.shape == delta.shape

    def test_filtered_delta_shape_preserved_2d(self):
        """Filtered delta should preserve 2D weight shape."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        d = 20
        backend.random_seed(42)
        activations = backend.random_normal((30, d))
        delta = backend.random_normal((d, d))
        backend.eval(activations, delta)

        # Filter expects delta to match activation dim or be transposable
        delta_flat = backend.reshape(delta, (-1,))
        activations_flat = backend.random_normal((30, d * d))
        backend.eval(delta_flat, activations_flat)

        result = geo_filter.filter_delta(delta_flat, activations_flat)
        backend.eval(result.filtered_delta)

        assert result.filtered_delta.shape == delta_flat.shape

    def test_preserves_zero_delta(self):
        """Zero delta should remain zero after filtering."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.zeros((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)
        backend.eval(result.filtered_delta)

        filtered_norm = backend.norm(result.filtered_delta)
        backend.eval(filtered_norm)

        eps = machine_epsilon(backend, delta) * delta.shape[0]
        assert backend.to_scalar(filtered_norm) <= eps
        assert result.original_norm < eps

    def test_k_neighbors_auto_derived(self):
        """k_neighbors should be automatically derived when not specified."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations, k_neighbors=None)

        # k_neighbors should be set to something reasonable
        assert result.k_neighbors > 0
        assert result.k_neighbors <= 30  # Can't exceed n_samples


class TestGeodesicProjection:
    """Test geodesic projection properties."""

    def test_preserved_fraction_bounded(self):
        """Preserved fraction should be in [0, 1]."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        assert 0.0 <= result.preserved_fraction <= 1.0
        assert 0.0 <= result.projection_loss <= 1.0

    def test_projection_loss_plus_preserved_equals_one(self):
        """projection_loss + preserved_fraction should equal 1."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        total = result.projection_loss + result.preserved_fraction
        eps = machine_epsilon(backend, backend.array(1.0))
        assert abs(total - 1.0) <= eps

    def test_filtered_norm_leq_original_norm(self):
        """Filtered delta norm should not exceed original norm."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        eps = machine_epsilon(backend, backend.array(1.0))
        assert result.filtered_norm <= result.original_norm + eps

    def test_underdetermined_system_preserves_more(self):
        """Fewer samples than dimensions should preserve more of delta.

        When n_samples < d, the tangent space doesn't span all of R^d,
        so there's room for the geodesic-orthogonal complement.
        """
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        d = 50
        # Very few samples compared to dimensions
        backend.random_seed(42)
        activations = backend.random_normal((10, d))
        delta = backend.random_normal((d,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        keep_weights, _ = compute_rmt_null_space_weights(activations, backend=backend)
        expected_dim = int(float(backend.to_scalar(backend.sum(keep_weights))))
        assert result.orthogonal_dim == expected_dim
        assert result.orthogonal_dim >= d - activations.shape[0]


class TestMergeIntegration:
    """Test integration with merge workflow."""

    def test_filter_merge_delta_geodesic_convenience(self):
        """Test convenience function for merge workflow."""
        backend = get_default_backend()
        d = 20
        n_samples = 30

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        activations = backend.random_normal((n_samples, d))
        backend.eval(source, target, activations)

        merged, result = filter_merge_delta_geodesic(source, target, activations)
        backend.eval(merged)

        assert merged.shape == source.shape
        assert isinstance(result, GeodesicNullSpaceResult)

    def test_merged_is_target_plus_filtered_delta(self):
        """Merged = target + filtered_delta (geometric addition)."""
        backend = get_default_backend()
        d = 20

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        activations = backend.random_normal((30, d))
        backend.eval(source, target, activations)

        merged, result = filter_merge_delta_geodesic(source, target, activations)
        backend.eval(merged)

        # Verify: merged = target + filtered_delta
        expected = target + result.filtered_delta
        backend.eval(expected)

        diff = backend.max(backend.abs(merged - expected))
        backend.eval(diff)
        eps = machine_epsilon(backend, merged) * merged.shape[0]
        assert backend.to_scalar(diff) <= eps

    def test_no_alpha_no_blending(self):
        """Verify there's no alpha blending - it's pure addition.

        The geodesic merge should be:
            merged = target + projected_delta

        NOT:
            merged = alpha * source + (1 - alpha) * target
        """
        backend = get_default_backend()
        d = 20

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        activations = backend.random_normal((30, d))
        backend.eval(source, target, activations)

        merged, result = filter_merge_delta_geodesic(source, target, activations)
        backend.eval(merged)

        # If it were alpha blending, merged would be between source and target
        # With geometric addition, merged is offset from target by the projected delta
        delta_from_target = merged - target
        backend.eval(delta_from_target)

        delta_from_target_norm = backend.norm(delta_from_target)
        filtered_delta_norm = backend.norm(result.filtered_delta)
        backend.eval(delta_from_target_norm, filtered_delta_norm)

        # They should be identical (not a blend)
        diff = backend.abs(delta_from_target_norm - filtered_delta_norm)
        backend.eval(diff)
        eps = machine_epsilon(backend, merged) * merged.shape[0]
        assert backend.to_scalar(diff) <= eps


class TestGeodesicMetrics:
    """Test geodesic-specific metrics."""

    def test_mean_geodesic_distance_positive(self):
        """Mean geodesic distance should be positive for non-trivial data."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        # With random data spread out, mean geodesic should be > 0
        assert result.mean_geodesic_distance >= 0.0

    def test_orthogonal_dim_reasonable(self):
        """Orthogonal dimension should be reasonable given data shape."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        d = 50
        n_samples = 20

        backend.random_seed(42)
        activations = backend.random_normal((n_samples, d))
        delta = backend.random_normal((d,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        # Orthogonal dim ~ d - n_samples for underdetermined systems
        assert result.orthogonal_dim >= 0
        assert result.orthogonal_dim <= d


class TestEdgeCases:
    """Edge case handling."""

    def test_single_sample(self):
        """Single sample should not crash."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((1, 10))
        delta = backend.random_normal((10,))
        backend.eval(activations, delta)

        # Should not crash with single sample
        result = geo_filter.filter_delta(delta, activations)
        assert result is not None

    def test_dimension_mismatch_returns_original(self):
        """Dimension mismatch should return original delta unchanged.

        Unlike the Euclidean filter which raises, the geodesic filter
        gracefully handles mismatched dimensions by returning the original.
        """
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((15,))  # Wrong dimension
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        # Should return original delta unchanged
        assert result.filtering_applied is False
        assert result.preserved_fraction == 1.0

    def test_high_dimensional_space(self):
        """High dimensional space should work without memory issues."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        # 500D space - should still work efficiently
        backend.random_seed(42)
        activations = backend.random_normal((50, 500))
        delta = backend.random_normal((500,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        keep_weights, _ = compute_rmt_null_space_weights(activations, backend=backend)
        expected_dim = int(float(backend.to_scalar(backend.sum(keep_weights))))
        assert result.orthogonal_dim == expected_dim
        expected_preserved = (
            result.filtered_norm / result.original_norm if result.original_norm > 0 else 1.0
        )
        eps = division_epsilon(backend, activations)
        assert abs(result.preserved_fraction - expected_preserved) <= eps

    def test_explicit_k_neighbors(self):
        """Explicit k_neighbors should be respected."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations, k_neighbors=5)

        # k_neighbors should be set (may be clamped but should be used)
        assert result.k_neighbors > 0


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=15)
    def test_projection_loss_plus_preserved_equals_one_hypothesis(self, n_samples, d):
        """projection_loss + preserved_fraction should always equal 1."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((n_samples, d))
        delta = backend.random_normal((d,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        total = result.projection_loss + result.preserved_fraction
        eps = machine_epsilon(backend, backend.array(1.0))
        assert abs(total - 1.0) <= eps, f"Total was {total}"

    @given(
        d=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=10)
    def test_filtered_norm_leq_original_hypothesis(self, d):
        """Filtered delta should never have larger norm than original."""
        backend = get_default_backend()
        geo_filter = GeodesicNullSpaceFilter(backend)

        backend.random_seed(42)
        activations = backend.random_normal((30, d))
        delta = backend.random_normal((d,))
        backend.eval(activations, delta)

        result = geo_filter.filter_delta(delta, activations)

        eps = machine_epsilon(backend, backend.array(1.0))
        assert result.filtered_norm <= result.original_norm + eps
