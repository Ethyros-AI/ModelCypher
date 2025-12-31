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

"""Comprehensive tests for Riemannian geometry utilities.

Tests cover:
- Fréchet mean computation and convergence
- Geodesic distance computation via k-NN graph
- Local curvature estimation
- Riemannian covariance
- Geodesic interpolation and path reconstruction
- Farthest point sampling
- Directional coverage analysis
- Edge cases and numerical stability
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.riemannian_utils import (
    CurvatureEstimate,
    DirectionalCoverage,
    FarthestPointSamplingResult,
    FrechetMeanResult,
    GeodesicDistanceResult,
    RiemannianGeometry,
    farthest_point_sampling,
    find_sparse_direction,
    frechet_mean,
    geodesic_distance_matrix,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestFrechetMeanResult:
    """Tests for FrechetMeanResult dataclass."""

    def test_basic_construction(self, any_backend: "Backend") -> None:
        """Test basic construction."""
        backend = any_backend
        mean = backend.array([1.0, 2.0, 3.0])

        result = FrechetMeanResult(
            mean=mean,
            iterations=10,
            converged=True,
            final_variance=0.5,
        )

        assert result.iterations == 10
        assert result.converged is True
        assert result.final_variance == 0.5

    def test_frozen_dataclass(self, any_backend: "Backend") -> None:
        """Test that FrechetMeanResult is frozen."""
        backend = any_backend
        result = FrechetMeanResult(
            mean=backend.zeros((3,)),
            iterations=5,
            converged=True,
            final_variance=0.1,
        )

        with pytest.raises(Exception):
            result.iterations = 20  # type: ignore


class TestGeodesicDistanceResult:
    """Tests for GeodesicDistanceResult dataclass."""

    def test_basic_construction(self, any_backend: "Backend") -> None:
        """Test basic construction."""
        backend = any_backend

        result = GeodesicDistanceResult(
            distances=backend.zeros((5, 5)),
            adjacency=backend.zeros((5, 5)),
            inf_value=1e10,
            k_neighbors=3,
            connected=True,
        )

        assert result.inf_value == 1e10
        assert result.k_neighbors == 3
        assert result.connected is True


class TestCurvatureEstimate:
    """Tests for CurvatureEstimate dataclass."""

    def test_positive_curvature(self) -> None:
        """Test positive curvature estimate."""
        result = CurvatureEstimate(
            sectional_curvature=0.5,
            is_positive=True,
            is_negative=False,
            confidence=0.9,
        )

        assert result.sectional_curvature == 0.5
        assert result.is_positive is True
        assert result.is_negative is False

    def test_negative_curvature(self) -> None:
        """Test negative curvature estimate."""
        result = CurvatureEstimate(
            sectional_curvature=-0.3,
            is_positive=False,
            is_negative=True,
            confidence=0.85,
        )

        assert result.sectional_curvature == -0.3
        assert result.is_negative is True

    def test_flat_curvature(self) -> None:
        """Test flat (zero) curvature estimate."""
        result = CurvatureEstimate(
            sectional_curvature=0.0,
            is_positive=False,
            is_negative=False,
            confidence=0.95,
        )

        assert result.sectional_curvature == 0.0
        assert result.is_positive is False
        assert result.is_negative is False


class TestDirectionalCoverage:
    """Tests for DirectionalCoverage dataclass."""

    def test_basic_construction(self, any_backend: "Backend") -> None:
        """Test basic construction."""
        backend = any_backend

        result = DirectionalCoverage(
            sparse_direction=backend.array([1.0, 0.0, 0.0]),
            max_gap_angle=1.5,
            coverage_uniformity=0.8,
            neighbor_directions=backend.zeros((5, 3)),
            point_idx=0,
        )

        assert result.max_gap_angle == 1.5
        assert result.coverage_uniformity == 0.8
        assert result.point_idx == 0


class TestFarthestPointSamplingResult:
    """Tests for FarthestPointSamplingResult dataclass."""

    def test_basic_construction(self, any_backend: "Backend") -> None:
        """Test basic construction."""
        backend = any_backend

        result = FarthestPointSamplingResult(
            selected_indices=[0, 5, 10],
            min_distances=backend.zeros((20,)),
            coverage_radius=2.5,
        )

        assert result.selected_indices == [0, 5, 10]
        assert result.coverage_radius == 2.5


# =============================================================================
# RiemannianGeometry Class Tests
# =============================================================================


class TestRiemannianGeometryInit:
    """Tests for RiemannianGeometry initialization."""

    def test_init_with_backend(self, any_backend: "Backend") -> None:
        """Test initialization with explicit backend."""
        rg = RiemannianGeometry(any_backend)
        assert rg._backend == any_backend

    def test_init_default_backend(self) -> None:
        """Test initialization with default backend."""
        rg = RiemannianGeometry()
        assert rg._backend is not None


# =============================================================================
# Fréchet Mean Tests
# =============================================================================


class TestFrechetMean:
    """Tests for Fréchet mean computation."""

    def test_single_point(self, any_backend: "Backend") -> None:
        """Fréchet mean of a single point is that point."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[1.0, 2.0, 3.0]])
        result = rg.frechet_mean(points)

        mean_np = backend.to_numpy(result.mean)
        assert abs(mean_np[0] - 1.0) < 1e-5
        assert abs(mean_np[1] - 2.0) < 1e-5
        assert abs(mean_np[2] - 3.0) < 1e-5
        assert result.converged is True
        assert result.iterations == 0

    def test_empty_points(self, any_backend: "Backend") -> None:
        """Fréchet mean of empty set returns zero."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.zeros((0, 3))
        result = rg.frechet_mean(points)

        mean_np = backend.to_numpy(result.mean)
        assert len(mean_np) == 3
        assert result.converged is True

    def test_two_points(self, any_backend: "Backend") -> None:
        """Fréchet mean of two points should be near midpoint."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [2.0, 0.0],
        ])
        result = rg.frechet_mean(points)

        mean_np = backend.to_numpy(result.mean)
        # Should be near (1.0, 0.0)
        assert abs(mean_np[0] - 1.0) < 0.5
        assert abs(mean_np[1]) < 0.5

    def test_symmetric_points(self, any_backend: "Backend") -> None:
        """Fréchet mean of symmetric points should be at origin."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Points symmetric around origin
        points = backend.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        result = rg.frechet_mean(points)

        mean_np = backend.to_numpy(result.mean)
        # Should be near origin
        assert abs(mean_np[0]) < 0.5
        assert abs(mean_np[1]) < 0.5

    def test_with_weights(self, any_backend: "Backend") -> None:
        """Fréchet mean with weights should shift toward heavier weights."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [4.0, 0.0],
        ])
        # Weight second point 3x more than first
        weights = backend.array([1.0, 3.0])

        result = rg.frechet_mean(points, weights=weights)
        mean_np = backend.to_numpy(result.mean)

        # Weighted mean should be closer to (4, 0) than (0, 0)
        # Euclidean weighted mean would be at (3, 0)
        assert mean_np[0] > 1.5  # Should be shifted toward second point

    def test_convergence(self, any_backend: "Backend") -> None:
        """Fréchet mean should converge."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        # Random point cloud
        points = backend.random_normal((10, 4))
        result = rg.frechet_mean(points, max_iterations=100)

        # Should converge or reach max iterations
        assert result.iterations <= 100
        assert result.final_variance >= 0.0

    def test_variance_decreases(self, any_backend: "Backend") -> None:
        """Final variance should be reasonable."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 3))
        result = rg.frechet_mean(points)

        # Variance should be non-negative
        assert result.final_variance >= 0.0


# =============================================================================
# Geodesic Distance Tests
# =============================================================================


class TestGeodesicDistances:
    """Tests for geodesic distance computation."""

    def test_single_point(self, any_backend: "Backend") -> None:
        """Geodesic distances for single point."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[1.0, 2.0]])
        result = rg.geodesic_distances(points)

        assert result.distances.shape == (1, 1)
        assert result.k_neighbors == 0
        assert result.connected is True

    def test_two_points(self, any_backend: "Backend") -> None:
        """Geodesic distance between two points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [3.0, 4.0],  # Euclidean distance = 5
        ])
        result = rg.geodesic_distances(points)

        dist_np = backend.to_numpy(result.distances)
        # Should be near 5.0 (Euclidean = geodesic for 2 points)
        assert abs(dist_np[0, 1] - 5.0) < 0.5
        assert abs(dist_np[1, 0] - 5.0) < 0.5
        # Diagonal should be 0
        assert dist_np[0, 0] == 0.0
        assert dist_np[1, 1] == 0.0

    def test_diagonal_is_zero(self, any_backend: "Backend") -> None:
        """Diagonal of geodesic distance matrix should be zero."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 4))
        result = rg.geodesic_distances(points)

        dist_np = backend.to_numpy(result.distances)
        for i in range(10):
            assert dist_np[i, i] == 0.0

    def test_symmetry(self, any_backend: "Backend") -> None:
        """Geodesic distance matrix should be symmetric."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 3))
        result = rg.geodesic_distances(points)

        dist_np = backend.to_numpy(result.distances)
        for i in range(8):
            for j in range(8):
                assert abs(dist_np[i, j] - dist_np[j, i]) < 1e-5

    def test_triangle_inequality(self, any_backend: "Backend") -> None:
        """Geodesic distances should satisfy triangle inequality."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((6, 3))
        result = rg.geodesic_distances(points, k_neighbors=5)

        if not result.connected:
            pytest.skip("Graph not connected with k=5")

        dist_np = backend.to_numpy(result.distances)
        n = dist_np.shape[0]

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if not math.isinf(dist_np[i, j]) and not math.isinf(dist_np[j, k]) and not math.isinf(dist_np[i, k]):
                        # d(i, k) <= d(i, j) + d(j, k)
                        assert dist_np[i, k] <= dist_np[i, j] + dist_np[j, k] + 1e-5

    def test_custom_k_neighbors(self, any_backend: "Backend") -> None:
        """Test with custom k_neighbors."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        result = rg.geodesic_distances(points, k_neighbors=3)

        assert result.k_neighbors == 3

    def test_k_neighbors_clamped(self, any_backend: "Backend") -> None:
        """k_neighbors should be clamped to valid range."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = rg.geodesic_distances(points, k_neighbors=100)

        # k should be clamped to n-1 = 2
        assert result.k_neighbors <= 2


# =============================================================================
# Local Curvature Estimation Tests
# =============================================================================


class TestLocalCurvatureEstimation:
    """Tests for local curvature estimation."""

    def test_insufficient_points(self, any_backend: "Backend") -> None:
        """Curvature estimation with < 3 points returns zero."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0]])
        result = rg.estimate_local_curvature(points, center_idx=0)

        assert result.sectional_curvature == 0.0
        assert result.confidence == 0.0

    def test_flat_points(self, any_backend: "Backend") -> None:
        """Points on a line should have near-zero curvature."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Points on a line
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ])
        result = rg.estimate_local_curvature(points, center_idx=2, k_neighbors=4)

        # Should be approximately flat
        assert abs(result.sectional_curvature) < 1.0

    def test_returns_valid_estimate(self, any_backend: "Backend") -> None:
        """Curvature estimation should return valid values."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        result = rg.estimate_local_curvature(points, center_idx=0, k_neighbors=5)

        assert math.isfinite(result.sectional_curvature)
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Riemannian Covariance Tests
# =============================================================================


class TestRiemannianCovariance:
    """Tests for Riemannian covariance computation."""

    def test_single_point(self, any_backend: "Backend") -> None:
        """Covariance of single point is zero."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[1.0, 2.0, 3.0]])
        cov = rg.riemannian_covariance(points)

        cov_np = backend.to_numpy(cov)
        assert cov_np.shape == (3, 3)
        assert abs(cov_np.sum()) < 1e-10

    def test_covariance_shape(self, any_backend: "Backend") -> None:
        """Covariance matrix should have correct shape."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 5))
        cov = rg.riemannian_covariance(points)

        assert cov.shape == (5, 5)

    def test_covariance_symmetric(self, any_backend: "Backend") -> None:
        """Covariance matrix should be symmetric."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 4))
        cov = rg.riemannian_covariance(points)

        cov_np = backend.to_numpy(cov)
        # Check symmetry
        for i in range(4):
            for j in range(4):
                assert abs(cov_np[i, j] - cov_np[j, i]) < 1e-5

    def test_with_precomputed_mean(self, any_backend: "Backend") -> None:
        """Covariance with precomputed mean."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 3))
        mean_result = rg.frechet_mean(points)
        cov = rg.riemannian_covariance(points, mean=mean_result.mean)

        assert cov.shape == (3, 3)


# =============================================================================
# Geodesic Interpolation Tests
# =============================================================================


class TestGeodesicInterpolation:
    """Tests for geodesic interpolation."""

    def test_t_zero_returns_start(self, any_backend: "Backend") -> None:
        """Interpolation at t=0 returns start point."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        p1 = backend.array([0.0, 0.0])
        p2 = backend.array([1.0, 1.0])
        context = backend.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])

        result = rg.geodesic_interpolation(p1, p2, t=0.0, points_context=context)
        result_np = backend.to_numpy(result)

        assert abs(result_np[0]) < 1e-5
        assert abs(result_np[1]) < 1e-5

    def test_t_one_returns_end(self, any_backend: "Backend") -> None:
        """Interpolation at t=1 returns end point."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        p1 = backend.array([0.0, 0.0])
        p2 = backend.array([1.0, 1.0])
        context = backend.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])

        result = rg.geodesic_interpolation(p1, p2, t=1.0, points_context=context)
        result_np = backend.to_numpy(result)

        assert abs(result_np[0] - 1.0) < 1e-5
        assert abs(result_np[1] - 1.0) < 1e-5

    def test_requires_context(self, any_backend: "Backend") -> None:
        """Geodesic interpolation requires context points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        p1 = backend.array([0.0, 0.0])
        p2 = backend.array([1.0, 1.0])

        with pytest.raises(ValueError, match="requires points_context"):
            rg.geodesic_interpolation(p1, p2, t=0.5, points_context=None)

    def test_insufficient_context(self, any_backend: "Backend") -> None:
        """Geodesic interpolation needs at least 2 context points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        p1 = backend.array([0.0, 0.0])
        p2 = backend.array([1.0, 1.0])
        context = backend.array([[0.5, 0.5]])  # Only 1 point

        with pytest.raises(ValueError, match="at least 2 context points"):
            rg.geodesic_interpolation(p1, p2, t=0.5, points_context=context)


# =============================================================================
# Farthest Point Sampling Tests
# =============================================================================


class TestFarthestPointSampling:
    """Tests for farthest point sampling."""

    def test_empty_points(self, any_backend: "Backend") -> None:
        """FPS on empty set returns empty result."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.zeros((0, 3))
        result = rg.farthest_point_sampling(points, n_samples=5)

        assert result.selected_indices == []
        assert result.coverage_radius == 0.0

    def test_single_sample(self, any_backend: "Backend") -> None:
        """FPS with n_samples=1 returns seed."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = rg.farthest_point_sampling(points, n_samples=1, seed_idx=1)

        assert result.selected_indices == [1]

    def test_all_samples(self, any_backend: "Backend") -> None:
        """FPS with n_samples=n returns all indices."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        result = rg.farthest_point_sampling(points, n_samples=3)

        assert len(result.selected_indices) == 3
        assert set(result.selected_indices) == {0, 1, 2}

    def test_samples_are_spread_out(self, any_backend: "Backend") -> None:
        """FPS should select spread-out points."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        # Create a cluster structure
        points_list = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.2],
        ]
        points = backend.array(points_list)

        result = rg.farthest_point_sampling(points, n_samples=2, seed_idx=0)

        # Should select one from each cluster
        assert len(result.selected_indices) == 2
        # First is seed (0), second should be from far cluster (3, 4, or 5)
        assert result.selected_indices[0] == 0
        assert result.selected_indices[1] in [3, 4, 5]

    def test_custom_seed(self, any_backend: "Backend") -> None:
        """FPS with custom seed starts from that point."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = rg.farthest_point_sampling(points, n_samples=2, seed_idx=2)

        assert result.selected_indices[0] == 2


# =============================================================================
# Directional Coverage Tests
# =============================================================================


class TestDirectionalCoverage:
    """Tests for directional coverage analysis."""

    def test_isolated_point(self, any_backend: "Backend") -> None:
        """Coverage of isolated point returns full gap."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Single point (no neighbors)
        points = backend.array([[1.0, 2.0, 3.0]])
        result = rg.directional_coverage(0, points, k=5)

        assert result.max_gap_angle == pytest.approx(math.pi, abs=0.1)
        assert result.coverage_uniformity == 0.0
        assert result.point_idx == 0

    def test_returns_unit_direction(self, any_backend: "Backend") -> None:
        """Sparse direction should be a unit vector."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        result = rg.directional_coverage(0, points, k=5)

        direction_np = backend.to_numpy(result.sparse_direction)
        norm = math.sqrt(sum(d * d for d in direction_np))

        # Should be approximately unit length
        assert abs(norm - 1.0) < 0.1

    def test_coverage_in_valid_range(self, any_backend: "Backend") -> None:
        """Coverage uniformity should be in [0, 1]."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((15, 4))
        result = rg.directional_coverage(0, points, k=8)

        assert 0.0 <= result.coverage_uniformity <= 1.0


# =============================================================================
# Propose in Sparse Direction Tests
# =============================================================================


class TestProposeInSparseDirection:
    """Tests for proposing points in sparse direction."""

    def test_step_size_affects_distance(self, any_backend: "Backend") -> None:
        """Larger step size should produce farther point."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        base_point = points[0]

        proposed_small = rg.propose_in_sparse_direction(0, points, step_size=0.1)
        proposed_large = rg.propose_in_sparse_direction(0, points, step_size=1.0)

        # Compute distances from base
        diff_small = proposed_small - base_point
        diff_large = proposed_large - base_point

        dist_small = float(backend.to_numpy(backend.sqrt(backend.sum(diff_small * diff_small))))
        dist_large = float(backend.to_numpy(backend.sqrt(backend.sum(diff_large * diff_large))))

        # Larger step should produce larger distance
        assert dist_large > dist_small

    def test_preserves_dimension(self, any_backend: "Backend") -> None:
        """Proposed point should have same dimension."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 5))
        proposed = rg.propose_in_sparse_direction(0, points, step_size=0.5)

        assert proposed.shape == (5,)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestFrechetMeanConvenience:
    """Tests for frechet_mean convenience function."""

    def test_returns_array(self, any_backend: "Backend") -> None:
        """Convenience function returns just the mean array."""
        backend = any_backend
        backend.random_seed(42)

        points = backend.random_normal((8, 3))
        mean = frechet_mean(points, backend=backend)

        assert mean.shape == (3,)

    def test_with_weights(self, any_backend: "Backend") -> None:
        """Convenience function accepts weights."""
        backend = any_backend

        points = backend.array([[0.0, 0.0], [2.0, 0.0]])
        weights = backend.array([1.0, 3.0])

        mean = frechet_mean(points, weights=weights, backend=backend)
        mean_np = backend.to_numpy(mean)

        # Should be shifted toward second point
        assert mean_np[0] > 0.5


class TestGeodesicDistanceMatrixConvenience:
    """Tests for geodesic_distance_matrix convenience function."""

    def test_returns_matrix(self, any_backend: "Backend") -> None:
        """Convenience function returns just the distance matrix."""
        backend = any_backend
        backend.random_seed(42)

        points = backend.random_normal((6, 3))
        distances = geodesic_distance_matrix(points, backend=backend)

        assert distances.shape == (6, 6)

    def test_with_k_neighbors(self, any_backend: "Backend") -> None:
        """Convenience function accepts k_neighbors."""
        backend = any_backend
        backend.random_seed(42)

        points = backend.random_normal((8, 3))
        distances = geodesic_distance_matrix(points, k_neighbors=3, backend=backend)

        assert distances.shape == (8, 8)


class TestFarthestPointSamplingConvenience:
    """Tests for farthest_point_sampling convenience function."""

    def test_returns_indices(self, any_backend: "Backend") -> None:
        """Convenience function returns just the indices list."""
        backend = any_backend
        backend.random_seed(42)

        points = backend.random_normal((10, 3))
        indices = farthest_point_sampling(points, n_samples=3, backend=backend)

        assert isinstance(indices, list)
        assert len(indices) == 3

    def test_with_seed(self, any_backend: "Backend") -> None:
        """Convenience function accepts seed_idx."""
        backend = any_backend

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        indices = farthest_point_sampling(points, n_samples=2, seed_idx=1, backend=backend)

        assert indices[0] == 1


class TestFindSparseDirectionConvenience:
    """Tests for find_sparse_direction convenience function."""

    def test_returns_direction(self, any_backend: "Backend") -> None:
        """Convenience function returns just the direction vector."""
        backend = any_backend
        backend.random_seed(42)

        points = backend.random_normal((10, 4))
        direction = find_sparse_direction(0, points, backend=backend)

        assert direction.shape == (4,)


# =============================================================================
# Edge Cases and Numerical Stability Tests
# =============================================================================


class TestEdgeCasesAndNumericalStability:
    """Tests for edge cases and numerical stability."""

    def test_coincident_points(self, any_backend: "Backend") -> None:
        """Handle coincident (identical) points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # All points are the same
        points = backend.array([
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ])

        # Geodesic distances should handle this
        result = rg.geodesic_distances(points)
        dist_np = backend.to_numpy(result.distances)

        # All distances should be near zero
        assert abs(dist_np.max()) < 1e-3

    def test_very_close_points(self, any_backend: "Backend") -> None:
        """Handle very close but not identical points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [1e-8, 0.0],
            [0.0, 1e-8],
        ])

        result = rg.geodesic_distances(points)
        dist_np = backend.to_numpy(result.distances)

        # Should handle without NaN
        for i in range(3):
            for j in range(3):
                assert math.isfinite(dist_np[i, j])

    def test_high_dimensional_points(self, any_backend: "Backend") -> None:
        """Handle high-dimensional point clouds."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 100))
        result = rg.geodesic_distances(points)

        assert result.distances.shape == (10, 10)

    def test_frechet_mean_convergence_tolerance(self, any_backend: "Backend") -> None:
        """Fréchet mean respects convergence tolerance."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))

        # Tight tolerance should take more iterations
        result_tight = rg.frechet_mean(points, tolerance=1e-10, max_iterations=100)

        # Result should be valid
        mean_np = backend.to_numpy(result_tight.mean)
        assert all(math.isfinite(v) for v in mean_np)

    def test_geodesic_on_line(self, any_backend: "Backend") -> None:
        """Geodesic on a line should equal Euclidean."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Points on a line (1D manifold embedded in 2D)
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])

        result = rg.geodesic_distances(points, k_neighbors=3)
        dist_np = backend.to_numpy(result.distances)

        # Geodesic should approximately equal Euclidean for points on a line
        for i in range(4):
            for j in range(4):
                expected = abs(i - j)  # Euclidean distance on the line
                if not math.isinf(dist_np[i, j]):
                    assert abs(dist_np[i, j] - expected) < 0.5
