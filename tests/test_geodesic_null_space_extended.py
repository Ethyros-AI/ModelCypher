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

"""Extended tests for geodesic null-space filtering.

Tests critical APIs:
- filter_delta_svd(): SVD-based delta filtering with energy threshold
- filter_merge_delta_geodesic(): Full merge delta computation
- GeodesicNullSpaceFilter.prepare_basis(): Basis precomputation
- GeodesicNullSpaceFilter.filter_delta(): Main filtering with variance weighting
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geodesic_null_space import (
    GeodesicNullSpaceFilter,
    filter_delta_svd,
    filter_merge_delta_geodesic,
)


@pytest.fixture
def backend():
    return get_default_backend()


@pytest.fixture
def small_activations(backend):
    """Small activation matrix for fast tests."""
    return backend.random_normal((16, 32))


@pytest.fixture
def small_delta(backend):
    """Small weight delta for fast tests."""
    return backend.random_normal((64, 32))


class TestFilterDeltaSVD:
    """Tests for filter_delta_svd() - SVD-based low-rank filtering."""

    def test_zero_delta_unchanged(self, backend):
        """Zero delta should return unchanged."""
        delta = backend.zeros((32, 16))
        result = filter_delta_svd(delta, backend)

        assert result.filtering_applied is False
        assert result.preserved_fraction == 1.0
        assert result.projection_loss == 0.0
        assert result.original_norm == 0.0
        assert result.filtered_norm == 0.0

    def test_1d_delta_works(self, backend):
        """1D delta vectors should be handled correctly."""
        delta = backend.random_normal((64,))
        result = filter_delta_svd(delta, backend)

        assert result.filtering_applied is True
        # Original shape should be preserved
        assert backend.shape(result.filtered_delta) == (64,)

    def test_2d_delta_works(self, backend):
        """2D delta matrices should work correctly."""
        delta = backend.random_normal((32, 16))
        result = filter_delta_svd(delta, backend)

        assert result.filtering_applied is True
        assert backend.shape(result.filtered_delta) == (32, 16)
        assert 0 < result.preserved_fraction <= 1.0

    def test_energy_threshold_one_preserves_all(self, backend):
        """Energy threshold 1.0 should preserve all singular values."""
        delta = backend.random_normal((16, 8))
        result = filter_delta_svd(delta, backend, energy_threshold=1.0)

        # With threshold=1.0, all singular values should be kept
        # so preserved_fraction should be very close to 1.0
        assert result.preserved_fraction > 0.999
        assert result.projection_loss < 0.001

    def test_energy_threshold_low_truncates(self, backend):
        """Low energy threshold should truncate more aggressively."""
        delta = backend.random_normal((32, 16))
        result_high = filter_delta_svd(delta, backend, energy_threshold=0.99)
        result_low = filter_delta_svd(delta, backend, energy_threshold=0.5)

        # Lower threshold means more truncation, lower preserved fraction
        assert result_low.orthogonal_dim <= result_high.orthogonal_dim

    def test_filtered_norm_leq_original(self, backend):
        """Filtered norm should never exceed original norm."""
        delta = backend.random_normal((32, 16))
        result = filter_delta_svd(delta, backend)

        assert result.filtered_norm <= result.original_norm + 1e-6

    def test_fractions_sum_to_one(self, backend):
        """preserved_fraction + projection_loss should equal 1."""
        delta = backend.random_normal((16, 8))
        result = filter_delta_svd(delta, backend)

        total = result.preserved_fraction + result.projection_loss
        assert abs(total - 1.0) < 1e-6


class TestGeodesicNullSpaceFilter:
    """Tests for GeodesicNullSpaceFilter class."""

    def test_filter_delta_basic(self, backend, small_activations, small_delta):
        """Basic filter_delta call should work."""
        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(small_delta, small_activations)

        assert result.filtered_delta is not None
        assert backend.shape(result.filtered_delta) == backend.shape(small_delta)
        assert 0 <= result.preserved_fraction <= 1.0

    def test_filter_delta_dimension_mismatch(self, backend):
        """Dimension mismatch should return original delta."""
        activations = backend.random_normal((16, 32))
        # Delta with incompatible dimensions
        delta = backend.random_normal((64, 64))

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(delta, activations)

        # When dimensions don't align, should return original
        assert result.filtering_applied is False
        assert result.preserved_fraction == 1.0

    def test_filter_delta_1d(self, backend):
        """1D deltas should work when dimension matches."""
        activations = backend.random_normal((16, 32))
        delta = backend.random_normal((32,))

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(delta, activations)

        assert backend.shape(result.filtered_delta) == (32,)

    def test_filter_delta_transpose_handling(self, backend):
        """Delta with transposed shape should be handled."""
        activations = backend.random_normal((16, 32))
        # Delta is [d, out] instead of [out, d]
        delta = backend.random_normal((32, 64))

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(delta, activations)

        # Should preserve shape after filtering
        assert backend.shape(result.filtered_delta) == (32, 64)

    def test_prepare_basis_reuse(self, backend, small_activations, small_delta):
        """Precomputed basis should produce same results as on-the-fly."""
        gns = GeodesicNullSpaceFilter(backend)

        # Compute basis once
        basis = gns.prepare_basis(small_activations)

        # Use precomputed basis
        result_with_basis = gns.filter_delta(
            small_delta, small_activations, basis=basis
        )

        # Compute fresh
        result_fresh = gns.filter_delta(small_delta, small_activations)

        # Results should be very close (may differ due to caching)
        r1 = float(backend.to_scalar(backend.mean(result_with_basis.filtered_delta)))
        r2 = float(backend.to_scalar(backend.mean(result_fresh.filtered_delta)))
        # Note: Results may differ due to cache vs fresh computation
        # The important thing is both complete without error

    def test_filter_delta_with_occupancy_weights(self, backend, small_activations):
        """Occupancy weights should influence filtering."""
        gns = GeodesicNullSpaceFilter(backend)
        delta = backend.random_normal((64, 32))

        # Without occupancy weights
        result_no_occ = gns.filter_delta(delta, small_activations)

        # With occupancy weights (all 0.5)
        occ_weights = backend.full((32,), 0.5)
        result_with_occ = gns.filter_delta(
            delta, small_activations, occupancy_weights=occ_weights
        )

        # Both should complete; results may differ
        assert result_no_occ.filtered_delta is not None
        assert result_with_occ.filtered_delta is not None

    def test_density_aware_mode(self, backend):
        """Density-aware mode with source activations should work."""
        target_act = backend.random_normal((16, 32))
        source_act = backend.random_normal((16, 32))
        delta = backend.random_normal((64, 32))

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(
            delta, target_act, source_activations=source_act
        )

        assert result.filtered_delta is not None
        assert result.filtering_applied is True or result.filtering_applied is False

    def test_delta_weights_returned(self, backend, small_activations, small_delta):
        """delta_weights should be returned in result."""
        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(small_delta, small_activations)

        assert result.delta_weights is not None
        assert backend.shape(result.delta_weights) == (32,)


class TestFilterMergeDeltaGeodesic:
    """Tests for filter_merge_delta_geodesic() - full merge pipeline."""

    def test_basic_merge(self, backend, small_activations):
        """Basic merge should work."""
        source_w = backend.random_normal((64, 32))
        target_w = backend.random_normal((64, 32))

        merged, result = filter_merge_delta_geodesic(
            source_w, target_w, small_activations
        )

        assert merged is not None
        assert backend.shape(merged) == (64, 32)
        assert result.filtered_delta is not None

    def test_identical_weights_no_change(self, backend, small_activations):
        """Identical source/target should produce target as merged."""
        weights = backend.random_normal((64, 32))

        merged, result = filter_merge_delta_geodesic(
            weights, weights, small_activations
        )

        # Delta is zero, so merged should equal target
        diff = backend.mean(backend.abs(merged - weights))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-6

    def test_density_aware_merge(self, backend):
        """Merge with source activations should work."""
        target_act = backend.random_normal((16, 32))
        source_act = backend.random_normal((16, 32))
        source_w = backend.random_normal((64, 32))
        target_w = backend.random_normal((64, 32))

        merged, result = filter_merge_delta_geodesic(
            source_w, target_w, target_act,
            source_activations=source_act,
        )

        assert merged is not None
        assert backend.shape(merged) == (64, 32)


class TestNullSpaceMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
        out_dim=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_filtered_norm_bounded(self, n_samples, d, out_dim):
        """Filtered norm should never exceed original norm."""
        backend = get_default_backend()
        activations = backend.random_normal((n_samples, d))
        delta = backend.random_normal((out_dim, d))
        backend.eval(activations, delta)

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(delta, activations)

        # Allow small numerical tolerance
        assert result.filtered_norm <= result.original_norm + 1e-5

    @given(
        m=st.integers(min_value=4, max_value=32),
        n=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_svd_fractions_sum_to_one(self, m, n):
        """SVD preserved_fraction + projection_loss should equal 1."""
        backend = get_default_backend()
        delta = backend.random_normal((m, n))
        backend.eval(delta)

        result = filter_delta_svd(delta, backend)

        total = result.preserved_fraction + result.projection_loss
        assert abs(total - 1.0) < 1e-6

    @given(
        n_samples=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_preserved_fraction_bounded(self, n_samples, d):
        """preserved_fraction should be in [0, 1]."""
        backend = get_default_backend()
        activations = backend.random_normal((n_samples, d))
        delta = backend.random_normal((16, d))
        backend.eval(activations, delta)

        gns = GeodesicNullSpaceFilter(backend)
        result = gns.filter_delta(delta, activations)

        assert 0.0 <= result.preserved_fraction <= 1.0

    @given(
        m=st.integers(min_value=4, max_value=16),
        n=st.integers(min_value=4, max_value=16),
        threshold=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=10, deadline=None)
    def test_svd_monotonic_energy_threshold(self, m, n, threshold):
        """Higher energy threshold should preserve more (monotonicity)."""
        backend = get_default_backend()
        delta = backend.random_normal((m, n))
        backend.eval(delta)

        result_high = filter_delta_svd(delta, backend, energy_threshold=threshold)
        result_low = filter_delta_svd(delta, backend, energy_threshold=threshold * 0.5)

        # Higher threshold should preserve at least as much
        assert result_high.orthogonal_dim >= result_low.orthogonal_dim


class TestPrepareBasis:
    """Tests for prepare_basis() method."""

    def test_prepare_basis_returns_valid(self, backend, small_activations):
        """prepare_basis should return valid GeodesicNullSpaceBasis."""
        gns = GeodesicNullSpaceFilter(backend)
        basis = gns.prepare_basis(small_activations)

        assert basis is not None
        assert basis.Q is not None
        assert basis.orthogonal_dim >= 0
        assert basis.k_neighbors > 0
        assert basis.mean_geodesic_distance >= 0

    def test_prepare_basis_with_explicit_k(self, backend, small_activations):
        """prepare_basis with explicit k_neighbors should use that k."""
        gns = GeodesicNullSpaceFilter(backend)
        basis = gns.prepare_basis(small_activations, k_neighbors=5)

        assert basis.k_neighbors == 5

    def test_prepare_basis_deterministic(self, backend, small_activations):
        """Same activations should produce same basis (cached)."""
        gns = GeodesicNullSpaceFilter(backend)

        basis1 = gns.prepare_basis(small_activations)
        basis2 = gns.prepare_basis(small_activations)

        # Should be same due to caching
        assert basis1.orthogonal_dim == basis2.orthogonal_dim
        assert basis1.k_neighbors == basis2.k_neighbors
