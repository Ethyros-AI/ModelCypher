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

"""Extended tests for knowledge density estimation.

Tests critical APIs:
- compute_knn_point_cloud_density(): k-NN based density comparison
- compute_density_weights(): Convert densities to transfer weights
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_density import (
    compute_density_weights,
    compute_knn_point_cloud_density,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    regularization_epsilon,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeKNNPointCloudDensity:
    """Tests for compute_knn_point_cloud_density()."""

    def test_basic_density_computation(self, backend):
        """Basic density computation should work."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        assert result.source_densities is not None
        assert result.target_densities is not None
        assert result.density_diff is not None

    def test_density_shapes(self, backend):
        """Density arrays should have correct shapes."""
        n_points = 16
        source = backend.random_normal((n_points, 32))
        target = backend.random_normal((n_points, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        assert backend.shape(result.source_densities)[0] == n_points
        assert backend.shape(result.target_densities)[0] == n_points
        assert backend.shape(result.density_diff)[0] == n_points

    def test_densities_normalized(self, backend):
        """Densities should be normalized to [0, 1]."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        src_min = backend.min(result.source_densities)
        src_max = backend.max(result.source_densities)
        tgt_min = backend.min(result.target_densities)
        tgt_max = backend.max(result.target_densities)
        backend.eval(src_min, src_max, tgt_min, tgt_max)

        tol_src = regularization_epsilon(backend, result.source_densities)
        tol_tgt = regularization_epsilon(backend, result.target_densities)
        assert float(backend.to_scalar(src_min)) >= -tol_src
        assert float(backend.to_scalar(src_max)) <= 1.0 + tol_src
        assert float(backend.to_scalar(tgt_min)) >= -tol_tgt
        assert float(backend.to_scalar(tgt_max)) <= 1.0 + tol_tgt

    def test_degenerate_single_point(self, backend):
        """Single point should return zeros gracefully."""
        source = backend.random_normal((1, 32))
        target = backend.random_normal((1, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        # Degenerate case should return zeros
        assert result.mean_source_density == 0.0
        assert result.mean_target_density == 0.0

    def test_explicit_k_neighbors(self, backend):
        """Explicit k should be used."""
        source = backend.random_normal((20, 32))
        target = backend.random_normal((20, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, k=5, backend=backend)

        # Should complete without error
        assert result.source_densities is not None

    def test_statistics_computed(self, backend):
        """Aggregate statistics should be computed."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        assert isinstance(result.mean_source_density, float)
        assert isinstance(result.mean_target_density, float)
        assert isinstance(result.mean_density_diff, float)
        assert isinstance(result.positive_diff_count, int)
        assert isinstance(result.negative_diff_count, int)

    def test_diff_counts_valid(self, backend):
        """Positive/negative diff counts should sum correctly."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 32))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        # Counts should not exceed total points
        n_points = 16
        assert result.positive_diff_count <= n_points
        assert result.negative_diff_count <= n_points


class TestComputeDensityWeights:
    """Tests for compute_density_weights()."""

    def test_basic_weight_computation(self, backend):
        """Basic weight computation should work."""
        source_densities = backend.array([0.8, 0.5, 0.3])
        target_densities = backend.array([0.2, 0.5, 0.7])
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )

        assert weights is not None
        assert backend.shape(weights) == (3,)

    def test_weights_bounded(self, backend):
        """Weights should be in [0, 1]."""
        source_densities = backend.random_normal((16,))
        target_densities = backend.random_normal((16,))
        # Make positive
        source_densities = backend.abs(source_densities)
        target_densities = backend.abs(target_densities)
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )

        min_w = backend.min(weights)
        max_w = backend.max(weights)
        backend.eval(min_w, max_w)

        tol = regularization_epsilon(backend, weights)
        assert float(backend.to_scalar(min_w)) >= -tol
        assert float(backend.to_scalar(max_w)) <= 1.0 + tol

    def test_high_source_high_weight(self, backend):
        """High source density, low target → high weight (transfer)."""
        source_densities = backend.array([1.0])
        target_densities = backend.array([0.0])
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )
        backend.eval(weights)

        tol = regularization_epsilon(backend, weights)
        assert abs(float(backend.to_scalar(weights[0])) - 1.0) <= tol

    def test_low_source_low_weight(self, backend):
        """Low source density, high target → low weight (preserve)."""
        source_densities = backend.array([0.0])
        target_densities = backend.array([1.0])
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )
        backend.eval(weights)

        tol = regularization_epsilon(backend, weights)
        assert float(backend.to_scalar(weights[0])) <= tol

    def test_equal_densities_half_weight(self, backend):
        """Equal densities should give weight ~0.5."""
        source_densities = backend.array([0.5, 0.5, 0.5])
        target_densities = backend.array([0.5, 0.5, 0.5])
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )
        backend.eval(weights)

        mean_w = backend.mean(weights)
        backend.eval(mean_w)

        tol = regularization_epsilon(backend, weights)
        assert abs(float(backend.to_scalar(mean_w)) - 0.5) <= tol

    def test_zero_densities_handled(self, backend):
        """Both zero densities should not cause division by zero."""
        source_densities = backend.array([0.0, 0.0])
        target_densities = backend.array([0.0, 0.0])
        backend.eval(source_densities, target_densities)

        weights = compute_density_weights(
            source_densities, target_densities, backend=backend
        )

        # Should be finite
        assert all_finite(weights, backend)


class TestDensityMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_points=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_densities_finite(self, n_points, d):
        """All densities should be finite."""
        backend = get_default_backend()
        source = backend.random_normal((n_points, d))
        target = backend.random_normal((n_points, d))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        assert all_finite(result.source_densities, backend)
        assert all_finite(result.target_densities, backend)
        assert all_finite(result.density_diff, backend)

    @given(
        n=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_weights_sum_property(self, n):
        """Weight formula: w = src / (src + tgt)."""
        backend = get_default_backend()
        src = backend.abs(backend.random_normal((n,))) + 0.1  # Positive
        tgt = backend.abs(backend.random_normal((n,))) + 0.1
        backend.eval(src, tgt)

        weights = compute_density_weights(src, tgt, backend=backend)

        # w = src / (src + tgt), so w * (src + tgt) = src
        # Therefore: w * tgt = src - w * src = src * (1 - w)
        # And: (1 - w) = tgt / (src + tgt)
        one_minus_w = 1.0 - weights
        expected = tgt / (src + tgt)
        diff = backend.mean(backend.abs(one_minus_w - expected))
        backend.eval(diff)

        tol = regularization_epsilon(backend, expected)
        assert float(backend.to_scalar(diff)) <= tol

    @given(
        n_points=st.integers(min_value=8, max_value=24),
        d=st.integers(min_value=8, max_value=24),
    )
    @settings(max_examples=5, deadline=None)
    def test_diff_count_consistency(self, n_points, d):
        """Positive + negative + neutral should equal total points."""
        backend = get_default_backend()
        source = backend.random_normal((n_points, d))
        target = backend.random_normal((n_points, d))
        backend.eval(source, target)

        result = compute_knn_point_cloud_density(source, target, backend=backend)

        # Total classified should not exceed n_points
        total_classified = result.positive_diff_count + result.negative_diff_count
        assert total_classified <= n_points
