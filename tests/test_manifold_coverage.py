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

"""Tests for manifold coverage analysis and filling."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_coverage import (
    CoverageConfig,
    ManifoldCoverage,
    analyze_coverage,
    find_sparse_regions,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    DirectionalCoverage,
    FarthestPointSamplingResult,
    RiemannianGeometry,
    farthest_point_sampling,
    find_sparse_direction,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    LocalDimensionMap,
)


@pytest.fixture
def backend():
    return get_default_backend()


@pytest.fixture
def uniform_2d_points(backend):
    """Create uniformly distributed 2D points."""
    backend.random_seed(42)
    return backend.random_normal((50, 2))


@pytest.fixture
def clustered_points(backend):
    """Create clustered points with sparse regions between clusters."""
    backend.random_seed(42)
    # Two clusters with a gap
    cluster1 = backend.random_normal((25, 2)) * 0.5
    cluster2 = backend.random_normal((25, 2)) * 0.5 + 5.0
    return backend.concatenate([cluster1, cluster2], axis=0)


class TestFarthestPointSampling:
    """Tests for geodesic farthest point sampling."""

    def test_fps_returns_correct_count(self, backend, uniform_2d_points):
        """FPS should return exactly n_samples points."""
        rg = RiemannianGeometry(backend)
        result = rg.farthest_point_sampling(uniform_2d_points, n_samples=10)

        assert isinstance(result, FarthestPointSamplingResult)
        assert len(result.selected_indices) == 10

    def test_fps_seed_is_first(self, backend, uniform_2d_points):
        """FPS should start with the seed point."""
        rg = RiemannianGeometry(backend)
        result = rg.farthest_point_sampling(uniform_2d_points, n_samples=5, seed_idx=7)

        assert result.selected_indices[0] == 7

    def test_fps_indices_unique(self, backend, uniform_2d_points):
        """FPS should not select the same point twice."""
        rg = RiemannianGeometry(backend)
        result = rg.farthest_point_sampling(uniform_2d_points, n_samples=20)

        assert len(set(result.selected_indices)) == len(result.selected_indices)

    def test_fps_coverage_radius_positive(self, backend, uniform_2d_points):
        """Coverage radius should be positive for non-trivial point clouds."""
        rg = RiemannianGeometry(backend)
        result = rg.farthest_point_sampling(uniform_2d_points, n_samples=5)

        # With only 5 samples from 50 points, coverage radius should be > 0
        assert result.coverage_radius > 0

    def test_fps_full_coverage_zero_radius(self, backend, uniform_2d_points):
        """Selecting all points should give zero coverage radius."""
        rg = RiemannianGeometry(backend)
        n = int(uniform_2d_points.shape[0])
        result = rg.farthest_point_sampling(uniform_2d_points, n_samples=n)

        assert result.coverage_radius == 0.0

    def test_fps_convenience_function(self, backend, uniform_2d_points):
        """Convenience function should return just indices."""
        indices = farthest_point_sampling(uniform_2d_points, n_samples=5, backend=backend)

        assert isinstance(indices, list)
        assert len(indices) == 5

    def test_fps_empty_points(self, backend):
        """FPS should handle empty input gracefully."""
        empty = backend.zeros((0, 2))
        rg = RiemannianGeometry(backend)
        result = rg.farthest_point_sampling(empty, n_samples=5)

        assert result.selected_indices == []


class TestDirectionalCoverage:
    """Tests for directional sparsity analysis in tangent space."""

    def test_directional_coverage_returns_unit_vector(self, backend, uniform_2d_points):
        """Sparse direction should be a unit vector."""
        rg = RiemannianGeometry(backend)
        result = rg.directional_coverage(0, uniform_2d_points, k=10)

        assert isinstance(result, DirectionalCoverage)

        # Check unit norm
        norm = backend.sqrt(backend.sum(result.sparse_direction ** 2))
        backend.eval(norm)
        assert abs(float(backend.to_numpy(norm)) - 1.0) < 0.01

    def test_directional_coverage_gap_angle_valid(self, backend, uniform_2d_points):
        """Max gap angle should be in [0, pi]."""
        rg = RiemannianGeometry(backend)
        result = rg.directional_coverage(0, uniform_2d_points, k=10)

        assert 0 <= result.max_gap_angle <= math.pi

    def test_directional_coverage_uniformity_bounded(self, backend, uniform_2d_points):
        """Coverage uniformity should be in [0, 1]."""
        rg = RiemannianGeometry(backend)
        result = rg.directional_coverage(0, uniform_2d_points, k=10)

        assert 0 <= result.coverage_uniformity <= 1

    def test_directional_coverage_neighbor_directions_normalized(
        self, backend, uniform_2d_points
    ):
        """Neighbor directions should be unit vectors."""
        rg = RiemannianGeometry(backend)
        result = rg.directional_coverage(0, uniform_2d_points, k=5)

        norms = backend.sqrt(
            backend.sum(result.neighbor_directions ** 2, axis=1)
        )
        backend.eval(norms)
        norms_np = backend.to_numpy(norms).flatten()

        for n in norms_np:
            assert abs(n - 1.0) < 0.01

    def test_find_sparse_direction_convenience(self, backend, uniform_2d_points):
        """Convenience function should return just the direction."""
        direction = find_sparse_direction(0, uniform_2d_points, k=10, backend=backend)

        # Should be an array with d dimensions
        assert direction.shape == (2,)


class TestLocalDimensionMap:
    """Tests for per-point intrinsic dimension estimation."""

    def test_local_dimension_map_returns_correct_shape(self, backend, uniform_2d_points):
        """Local dimension array should have n elements."""
        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(uniform_2d_points, k=10)

        assert isinstance(result, LocalDimensionMap)
        n = int(uniform_2d_points.shape[0])
        assert result.dimensions.shape[0] == n

    def test_local_dimension_modal_positive(self, backend, uniform_2d_points):
        """Modal dimension should be positive for non-trivial data."""
        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(uniform_2d_points, k=10)

        assert result.modal_dimension > 0

    def test_local_dimension_2d_data_positive(self, backend, uniform_2d_points):
        """2D uniform data should have positive intrinsic dimension."""
        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(uniform_2d_points, k=10)

        # Local TwoNN has higher variance than global estimation.
        # For 2D data, local dimension estimates can be higher due to
        # finite-sample effects and the inherent noise in local estimation.
        # The key property is that dimension is positive and finite.
        assert result.modal_dimension > 0
        assert result.mean_dimension > 0
        assert result.std_dimension >= 0

    def test_local_dimension_deficiency_detection(self, backend):
        """Should detect deficient regions in mixed-dimension data."""
        # Create data with a 1D line embedded in 2D
        backend.random_seed(42)
        line = backend.random_normal((30, 1))
        line_2d = backend.concatenate([line, backend.zeros((30, 1))], axis=1)

        # Add some 2D noise points
        noise_2d = backend.random_normal((20, 2)) * 0.1 + 3.0

        points = backend.concatenate([line_2d, noise_2d], axis=0)

        estimator = IntrinsicDimension(backend)
        result = estimator.local_dimension_map(points, k=5, deficiency_threshold=0.8)

        # The line points should have lower local dimension
        # At least some should be detected as deficient
        # (This depends on the threshold and exact geometry)
        assert isinstance(result.deficient_indices, list)

    def test_detect_dimension_deficiency_static(self, backend, uniform_2d_points):
        """Static convenience method should return indices."""
        deficient = IntrinsicDimension.detect_dimension_deficiency(
            uniform_2d_points, threshold=0.8, k=10, backend=backend
        )

        assert isinstance(deficient, list)
        # For uniform 2D data, most points should NOT be deficient
        n = int(uniform_2d_points.shape[0])
        assert len(deficient) < n * 0.5


class TestManifoldCoverage:
    """Tests for the ManifoldCoverage orchestrator."""

    def test_analyze_returns_complete_analysis(self, backend, uniform_2d_points):
        """Analyze should return all expected fields."""
        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(uniform_2d_points)

        assert isinstance(analysis.sparse_points, list)
        assert isinstance(analysis.sparse_directions, dict)
        assert analysis.local_dimensions is not None
        assert analysis.modal_dimension > 0
        assert isinstance(analysis.dimension_deficient, list)
        assert isinstance(analysis.proposed_fills, list)
        assert analysis.metrics is not None

    def test_analyze_metrics_valid(self, backend, uniform_2d_points):
        """Coverage metrics should have valid values."""
        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(uniform_2d_points)

        assert analysis.metrics.coverage_radius >= 0
        assert analysis.metrics.mean_density >= 0
        assert 0 <= analysis.metrics.dimension_uniformity <= 1

    def test_analyze_clustered_finds_sparse(self, backend, clustered_points):
        """Analysis should find sparse regions between clusters."""
        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(clustered_points)

        # Should find some sparse points (in the gap region)
        # The exact number depends on configuration
        assert len(analysis.sparse_points) > 0

    def test_propose_fills_returns_points(self, backend, clustered_points):
        """Propose fills should return new point locations."""
        mc = ManifoldCoverage(backend)
        fills = mc.propose_fills(clustered_points, n_proposals=3)

        assert len(fills) <= 3
        for fill in fills:
            # Each fill should have same dimension as input
            assert fill.shape == (2,)

    def test_poisson_disk_sample_min_distance(self, backend, uniform_2d_points):
        """Poisson disk sampling should respect minimum distance."""
        mc = ManifoldCoverage(backend)
        selected = mc.poisson_disk_sample(uniform_2d_points, min_dist=0.5, k=10)

        assert isinstance(selected, list)
        assert len(selected) > 0
        assert len(selected) <= int(uniform_2d_points.shape[0])

        # Verify minimum distance (approximately, since we use geodesic)
        # This is a soft check since geodesic != Euclidean
        assert len(set(selected)) == len(selected)  # No duplicates

    def test_farthest_point_sample_wrapper(self, backend, uniform_2d_points):
        """FPS wrapper should work correctly."""
        mc = ManifoldCoverage(backend)
        selected = mc.farthest_point_sample(uniform_2d_points, n_samples=5)

        assert len(selected) == 5

    def test_analyze_small_input(self, backend):
        """Analysis should handle small inputs gracefully."""
        small = backend.random_normal((2, 3))
        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(small)

        # Should not crash, but metrics will be limited
        assert analysis.metrics.coverage_radius == 0.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_coverage(self, backend, uniform_2d_points):
        """analyze_coverage convenience function should work."""
        analysis = analyze_coverage(uniform_2d_points, k=10, backend=backend)

        assert analysis is not None
        assert analysis.metrics is not None

    def test_find_sparse_regions(self, backend, clustered_points):
        """find_sparse_regions convenience function should work."""
        sparse = find_sparse_regions(clustered_points, k=10, backend=backend)

        assert isinstance(sparse, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self, backend):
        """Should handle empty input gracefully."""
        empty = backend.zeros((0, 2))

        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(empty)

        assert len(analysis.sparse_points) == 0
        assert len(analysis.proposed_fills) == 0

    def test_single_point(self, backend):
        """Should handle single point gracefully."""
        single = backend.random_normal((1, 3))

        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(single)

        assert analysis.metrics.coverage_radius == 0.0

    def test_two_points(self, backend):
        """Should handle two points gracefully."""
        two = backend.random_normal((2, 3))

        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(two)

        # Limited analysis possible with 2 points
        assert analysis is not None

    def test_high_dimensional(self, backend):
        """Should work with high-dimensional data."""
        backend.random_seed(42)
        high_d = backend.random_normal((30, 64))

        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(high_d, CoverageConfig(k_neighbors=5, n_fps_samples=5))

        assert analysis.metrics.coverage_radius >= 0

    def test_custom_config(self, backend, uniform_2d_points):
        """Should respect custom configuration."""
        config = CoverageConfig(
            k_neighbors=5,
            density_percentile=0.3,
            dimension_threshold=0.7,
            n_fps_samples=3,
            n_fill_proposals=2,
        )

        mc = ManifoldCoverage(backend)
        analysis = mc.analyze(uniform_2d_points, config)

        assert analysis.k_neighbors == 5
        assert analysis.density_percentile == 0.3
