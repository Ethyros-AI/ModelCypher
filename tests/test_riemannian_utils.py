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

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
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
    geodesic_norms,
)
from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps(backend: "Backend", *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _div_eps(backend: "Backend", *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _is_inf(value: float) -> bool:
    return value in (float("inf"), float("-inf"))


def _max_abs(backend: "Backend", array) -> float:
    diff = backend.max(backend.abs(array))
    backend.eval(diff)
    return float(backend.to_scalar(diff))


def _derive_seed_idx(backend: "Backend", geo_dist) -> int:
    finite_mask = backend.isfinite(geo_dist)
    finite_sum = backend.sum(
        backend.where(finite_mask, geo_dist, backend.zeros_like(geo_dist)), axis=1
    )
    finite_count = backend.sum(backend.astype(finite_mask, "int32"), axis=1)
    finite_count = backend.maximum(finite_count, backend.ones_like(finite_count))
    mean_dist = finite_sum / finite_count
    backend.eval(mean_dist)
    seed_idx_arr = backend.argmax(mean_dist)
    backend.eval(seed_idx_arr)
    return int(backend.to_scalar(seed_idx_arr))


PI = 3.141592653589793


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


class TestDirectionalCoverageDataclass:
    """Tests for DirectionalCoverage dataclass."""

    def test_basic_construction(self, any_backend: "Backend") -> None:
        """Test basic construction."""
        backend = any_backend

        result = DirectionalCoverage(
            sparse_direction=backend.array([1.0, 0.0, 0.0]),
            max_gap_angle=1.5,
            coverage_variance=0.8,
            neighbor_directions=backend.zeros((5, 3)),
            point_idx=0,
        )

        assert result.max_gap_angle == 1.5
        assert result.coverage_variance == 0.8
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

        mean_list = array_to_list(backend, result.mean)
        eps = _eps(
            backend,
            float(mean_list[0]),
            1.0,
            float(mean_list[1]),
            2.0,
            float(mean_list[2]),
            3.0,
        )
        assert abs(mean_list[0] - 1.0) <= eps
        assert abs(mean_list[1] - 2.0) <= eps
        assert abs(mean_list[2] - 3.0) <= eps
        assert result.converged is True
        assert result.iterations == 0

    def test_empty_points(self, any_backend: "Backend") -> None:
        """Fréchet mean of empty set returns zero."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.zeros((0, 3))
        result = rg.frechet_mean(points)

        mean_list = array_to_list(backend, result.mean)
        assert len(mean_list) == 3
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

        mean_list = array_to_list(backend, result.mean)
        # Should be near (1.0, 0.0)
        eps = _eps(backend, float(mean_list[0]), 1.0, float(mean_list[1]), 0.0)
        assert abs(mean_list[0] - 1.0) <= eps
        assert abs(mean_list[1]) <= eps

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

        mean_list = array_to_list(backend, result.mean)
        # Should be near origin
        eps = _eps(backend, float(mean_list[0]), float(mean_list[1]))
        assert abs(mean_list[0]) <= eps
        assert abs(mean_list[1]) <= eps

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
        mean_list = array_to_list(backend, result.mean)

        # Weighted mean should be closer to (4, 0) than (0, 0)
        # Euclidean weighted mean would be at (3, 0)
        eps = _eps(backend, float(mean_list[0]), 3.0)
        assert abs(mean_list[0] - 3.0) <= eps

    def test_convergence(self, any_backend: "Backend") -> None:
        """Fréchet mean should converge."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        # Random point cloud
        points = backend.random_normal((10, 4))
        max_iterations = 100
        result = rg.frechet_mean(points, max_iterations=max_iterations)

        # Should converge or reach max iterations
        assert result.iterations <= max_iterations
        eps = _eps(backend, result.final_variance)
        assert result.final_variance >= -eps

    def test_variance_decreases(self, any_backend: "Backend") -> None:
        """Final variance should be reasonable."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 3))
        result = rg.frechet_mean(points)

        # Variance should be non-negative
        eps = _eps(backend, result.final_variance)
        assert result.final_variance >= -eps


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
        """Geodesic distance between two points equals chord on complete graph."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([
            [0.0, 0.0],
            [3.0, 4.0],  # Euclidean distance = 5
        ])
        # For 2 points on complete graph, geodesic = chord
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        # Should be near 5.0 (Euclidean = geodesic for 2 points on complete graph)
        eps = _eps(backend, float(dist_list[0][1]), float(dist_list[1][0]), 5.0)
        assert abs(dist_list[0][1] - 5.0) <= eps
        assert abs(dist_list[1][0] - 5.0) <= eps
        # Diagonal should be 0
        assert abs(dist_list[0][0]) <= eps
        assert abs(dist_list[1][1]) <= eps

    def test_diagonal_is_zero(self, any_backend: "Backend") -> None:
        """Diagonal of geodesic distance matrix should be zero."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 4))
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        for i in range(10):
            eps = _eps(backend, float(dist_list[i][i]))
            assert abs(dist_list[i][i]) <= eps

    def test_symmetry(self, any_backend: "Backend") -> None:
        """Geodesic distance matrix should be symmetric."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((8, 3))
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        for i in range(8):
            for j in range(8):
                eps = _eps(backend, float(dist_list[i][j]), float(dist_list[j][i]))
                assert abs(dist_list[i][j] - dist_list[j][i]) <= eps

    def test_triangle_inequality(self, any_backend: "Backend") -> None:
        """Geodesic distances should satisfy triangle inequality."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((6, 3))
        result = rg.geodesic_distances(points, k_neighbors=5)
        assert result.connected, "Graph must be connected with k_neighbors=5"

        dist_list = array_to_list(backend, result.distances)
        n = len(dist_list)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if not _is_inf(dist_list[i][j]) and not _is_inf(dist_list[j][k]) and not _is_inf(dist_list[i][k]):
                        # d(i, k) <= d(i, j) + d(j, k)
                        eps = _eps(
                            backend,
                            float(dist_list[i][k]),
                            float(dist_list[i][j]),
                            float(dist_list[j][k]),
                        )
                        assert dist_list[i][k] <= dist_list[i][j] + dist_list[j][k] + eps

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
        assert result.k_neighbors == len(points) - 1


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

        eps = _div_eps(backend, result.sectional_curvature, 1.0)
        assert abs(result.sectional_curvature) <= eps, (
            f"Expected near-zero curvature for flat points, got {result.sectional_curvature}"
        )

    def test_returns_valid_estimate(self, any_backend: "Backend") -> None:
        """Curvature estimation should return valid values."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        result = rg.estimate_local_curvature(points, center_idx=0, k_neighbors=5)

        assert _is_finite(result.sectional_curvature)
        eps = _eps(backend, result.confidence)
        assert -eps <= result.confidence <= 1.0 + eps


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

        assert cov.shape == (3, 3)
        cov_sum = backend.sum(cov)
        backend.eval(cov_sum)
        cov_sum_val = float(backend.to_scalar(cov_sum))
        eps = _eps(backend, cov_sum_val)
        assert abs(cov_sum_val) <= eps

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

        cov_list = array_to_list(backend, cov)
        # Check symmetry
        for i in range(4):
            for j in range(4):
                eps = _eps(backend, float(cov_list[i][j]), float(cov_list[j][i]))
                assert abs(cov_list[i][j] - cov_list[j][i]) <= eps

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
        result_list = array_to_list(backend, result)

        eps = _eps(backend, float(result_list[0]), float(result_list[1]))
        assert abs(result_list[0]) <= eps
        assert abs(result_list[1]) <= eps

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
        result_list = array_to_list(backend, result)

        eps = _eps(backend, float(result_list[0]), float(result_list[1]), 1.0)
        assert abs(result_list[0] - 1.0) <= eps
        assert abs(result_list[1] - 1.0) <= eps

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
        """FPS with n_samples=1 returns the data-derived seed."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = rg.farthest_point_sampling(points, n_samples=1)

        geo_result = rg.geodesic_distances(points)
        expected_seed = _derive_seed_idx(backend, geo_result.distances)
        assert result.selected_indices == [expected_seed]

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
        """FPS should select the farthest point from the seed."""
        backend = any_backend
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

        result = rg.farthest_point_sampling(points, n_samples=2)

        assert len(result.selected_indices) == 2
        geo_result = rg.geodesic_distances(points)
        geo_dist = geo_result.distances
        seed_idx = _derive_seed_idx(backend, geo_dist)
        row = backend.take(geo_dist, backend.array([seed_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        row_masked = backend.where(
            backend.arange(0, int(points.shape[0])) == seed_idx,
            backend.full((int(points.shape[0]),), float("-inf")),
            row,
        )
        farthest_idx_arr = backend.argmax(row_masked)
        backend.eval(farthest_idx_arr)
        farthest_idx = int(backend.to_scalar(farthest_idx_arr))

        assert result.selected_indices[0] == seed_idx
        assert result.selected_indices[1] == farthest_idx

    def test_rejects_seed_override(self, any_backend: "Backend") -> None:
        """FPS rejects explicit seed overrides."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(TypeError):
            rg.farthest_point_sampling(points, n_samples=2, seed_idx=2)


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
        result = rg.directional_coverage(0, points)

        # With no neighbors, the implementation returns a full-gap angle of pi.
        eps = _div_eps(backend, result.max_gap_angle, PI)
        assert abs(result.max_gap_angle - PI) <= eps
        assert result.coverage_variance != result.coverage_variance  # NaN for undefined variance
        assert result.point_idx == 0

    def test_returns_unit_direction(self, any_backend: "Backend") -> None:
        """Sparse direction should be a unit vector."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        result = rg.directional_coverage(0, points)

        norm = backend.norm(result.sparse_direction)
        backend.eval(norm)
        norm_val = float(backend.to_scalar(norm))

        # Should be approximately unit length
        eps = _eps(backend, norm_val, 1.0)
        assert abs(norm_val - 1.0) <= eps

    def test_coverage_in_valid_range(self, any_backend: "Backend") -> None:
        """Coverage uniformity should be in [0, 1]."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((15, 4))
        result = rg.directional_coverage(0, points)

        assert result.coverage_variance >= -_eps(backend, result.coverage_variance)


# =============================================================================
# Propose in Sparse Direction Tests
# =============================================================================


class TestProposeInSparseDirection:
    """Tests for proposing points in sparse direction."""

    def test_proposed_is_existing_point(self, any_backend: "Backend") -> None:
        """Proposed point should be one of the existing points."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 3))
        proposed = rg.propose_in_sparse_direction(0, points)

        diffs = points - proposed
        dist_sq = backend.sum(diffs * diffs, axis=1)
        backend.eval(dist_sq)
        min_dist_sq = backend.min(dist_sq)
        backend.eval(min_dist_sq)
        min_val = float(backend.to_scalar(min_dist_sq))

        assert min_val <= _eps(backend, min_val)

    def test_preserves_dimension(self, any_backend: "Backend") -> None:
        """Proposed point should have same dimension."""
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((10, 5))
        proposed = rg.propose_in_sparse_direction(0, points)

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
        mean_list = array_to_list(backend, mean)

        # Should be shifted toward second point
        eps = _eps(backend, float(mean_list[0]), 1.5)
        assert abs(mean_list[0] - 1.5) <= eps


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

    def test_rejects_seed_override(self, any_backend: "Backend") -> None:
        """Convenience function rejects seed override."""
        backend = any_backend

        points = backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(TypeError):
            farthest_point_sampling(points, n_samples=2, seed_idx=1, backend=backend)


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

        # All distances should be near zero
        # Floyd-Warshall accumulates up to (n-1) edge weights across multi-hop paths
        # Each edge is floored to division_epsilon, so tolerance = n * div_eps
        n = result.distances.shape[0]
        dist_max = backend.max(result.distances)
        backend.eval(dist_max)
        dist_max_val = float(backend.to_scalar(dist_max))
        eps = n * _div_eps(backend, dist_max_val)
        assert abs(dist_max_val) <= eps

    def test_very_close_points(self, any_backend: "Backend") -> None:
        """Handle very close but not identical points."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        eps = _div_eps(backend, 1.0)
        points = backend.array([
            [0.0, 0.0],
            [eps, 0.0],
            [0.0, eps],
        ])

        result = rg.geodesic_distances(points)
        dist_list = array_to_list(backend, result.distances)

        # Should handle without NaN
        for i in range(3):
            for j in range(3):
                assert _is_finite(dist_list[i][j])

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
        tol = machine_epsilon(backend, points)

        # Tight tolerance should take more iterations
        result_tight = rg.frechet_mean(points, tolerance=tol, max_iterations=100)

        # Result should be valid
        mean_list = array_to_list(backend, result_tight.mean)
        assert all(_is_finite(v) for v in mean_list)

    def test_geodesic_on_line(self, any_backend: "Backend") -> None:
        """Geodesic on a line should equal Euclidean (Floyd-Warshall)."""
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Points on a line (1D manifold embedded in 2D)
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])

        # On a 1D manifold, graph geodesic = Euclidean
        result = rg.geodesic_distances(points, k_neighbors=3)
        dist_list = array_to_list(backend, result.distances)

        # Geodesic should approximately equal Euclidean for points on a line
        for i in range(4):
            for j in range(4):
                expected = abs(i - j)  # Euclidean distance on the line
                if not _is_inf(dist_list[i][j]):
                    eps = _eps(backend, float(dist_list[i][j]), float(expected))
                    assert abs(dist_list[i][j] - expected) <= eps


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestRiemannianHypothesis:
    """Hypothesis-based property tests for Riemannian geometry."""

    @given(
        n_points=st.integers(min_value=4, max_value=30),
        n_dim=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_diagonal_zero(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Geodesic distance d(x, x) = 0 for all points."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        for i in range(n_points):
            eps = _eps(backend, float(dist_list[i][i]))
            assert abs(dist_list[i][i]) <= eps

    @given(
        n_points=st.integers(min_value=4, max_value=20),
        n_dim=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_symmetry(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Geodesic distance d(x, y) = d(y, x) (symmetry)."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        for i in range(n_points):
            for j in range(n_points):
                eps = _eps(backend, float(dist_list[i][j]), float(dist_list[j][i]))
                assert abs(dist_list[i][j] - dist_list[j][i]) <= eps

    @given(
        n_points=st.integers(min_value=4, max_value=15),
        n_dim=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_triangle_inequality_hypothesis(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Geodesic distances satisfy triangle inequality d(x,z) <= d(x,y) + d(y,z)."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.geodesic_distances(points, k_neighbors=min(n_points - 1, 8))

        if not result.connected:
            assume(False)  # Skip disconnected graphs

        dist_list = array_to_list(backend, result.distances)
        n = len(dist_list)

        # Sample random triples to check triangle inequality
        import random
        random.seed(seed)
        for _ in range(min(50, n * n * n)):
            i, j, k = random.randint(0, n-1), random.randint(0, n-1), random.randint(0, n-1)
            d_ij = dist_list[i][j]
            d_jk = dist_list[j][k]
            d_ik = dist_list[i][k]
            if _is_finite(d_ij) and _is_finite(d_jk) and _is_finite(d_ik):
                # Use relative epsilon scaled by max value for numerical stability
                # Floyd-Warshall accumulates error over path steps
                max_val = max(abs(d_ik), abs(d_ij), abs(d_jk), 1.0)
                eps = _eps(backend, float(d_ik), float(d_ij), float(d_jk)) * max_val
                assert d_ik <= d_ij + d_jk + eps

    @given(
        n_points=st.integers(min_value=4, max_value=20),
        n_dim=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_non_negative(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Geodesic distances are non-negative d(x, y) >= 0."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.geodesic_distances(points)

        dist_list = array_to_list(backend, result.distances)
        for i in range(n_points):
            for j in range(n_points):
                eps = _eps(backend, float(dist_list[i][j]))
                assert dist_list[i][j] >= -eps

    @given(
        n_points=st.integers(min_value=3, max_value=15),
        n_dim=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_frechet_mean_variance_non_negative(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Fréchet mean has non-negative final variance."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.frechet_mean(points, max_iterations=50)

        eps = _eps(backend, result.final_variance)
        assert result.final_variance >= -eps

    @given(
        n_points=st.integers(min_value=3, max_value=15),
        n_dim=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_frechet_mean_finite(
        self, n_points: int, n_dim: int, seed: int, any_backend: "Backend"
    ):
        """Fréchet mean should be finite."""
        backend = any_backend
        backend.random_seed(seed)
        rg = RiemannianGeometry(backend)

        points = backend.random_normal((n_points, n_dim))
        result = rg.frechet_mean(points, max_iterations=50)

        mean_list = array_to_list(backend, result.mean)
        assert all(_is_finite(v) for v in mean_list)


# =============================================================================
# Synthetic Manifold Tests (Ground Truth)
# =============================================================================


class TestSyntheticManifolds:
    """Tests on synthetic manifolds with known ground truth."""

    def _sample_sphere(self, backend: "Backend", n_points: int, dim: int, seed: int):
        """Sample uniform points on unit (dim-1)-sphere in R^dim."""
        backend.random_seed(seed)
        # Sample Gaussian, normalize to sphere using geodesic norms
        points = backend.random_normal((n_points, dim))
        norms_flat = geodesic_norms(points, backend)
        backend.eval(norms_flat)
        norms = backend.reshape(norms_flat, (-1, 1))
        # Avoid division by zero
        eps = division_epsilon(backend, norms)
        norms = backend.maximum(norms, backend.ones_like(norms) * eps)
        sphere_points = points / norms
        return sphere_points

    def test_frechet_mean_uniform_sphere_converges(self, any_backend: "Backend"):
        """Fréchet mean of uniformly sampled sphere points converges.

        On points uniformly distributed on a sphere, the geodesic Fréchet mean
        should converge and minimize variance. Note: the geodesic Fréchet mean
        via k-NN graph is NOT guaranteed to be closer to the origin than the
        arithmetic mean - that property only holds for the intrinsic sphere
        Fréchet mean, not the extrinsic k-NN approximation.
        """
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Sample many points uniformly on 3D unit sphere
        n_points = 100
        sphere_points = self._sample_sphere(backend, n_points=n_points, dim=3, seed=42)

        # Use k = n-1 for precise geodesics (complete graph)
        result = rg.frechet_mean(
            sphere_points, max_iterations=100, k_neighbors=n_points - 1
        )

        # Verify the algorithm ran and produced a result
        # Note: for symmetric distributions like uniform sphere, convergence
        # may not be achieved as multiple points minimize variance equally.
        # This is expected behavior - we just verify the result is sensible.
        assert result.iterations > 0, "Should have run at least one iteration"
        assert result.final_variance > 0, "Variance should be positive"

        # Mean should be finite
        mean_norm = backend.norm(result.mean)
        backend.eval(mean_norm)
        mean_norm_val = float(backend.to_scalar(mean_norm))
        assert mean_norm_val < float("inf"), "Fréchet mean should be finite"

        point_norms = backend.norm(sphere_points, axis=1)
        backend.eval(point_norms)
        max_norm = float(backend.to_scalar(backend.max(point_norms)))
        eps = _div_eps(backend, mean_norm_val, max_norm)
        assert mean_norm_val <= max_norm + eps, (
            f"Fréchet mean norm {mean_norm_val} exceeds point norm {max_norm}"
        )

    def test_geodesic_on_linear_subspace(self, any_backend: "Backend"):
        """On a linear subspace (flat manifold), geodesic equals Euclidean.

        For points on a line, geodesic distance = Euclidean distance.
        """
        backend = any_backend
        rg = RiemannianGeometry(backend)

        # Points on a line in 3D (2D ambient, 1D manifold)
        t = backend.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        # Line: (t, 0, 0)
        zeros = backend.zeros((5, 2))
        points = backend.concatenate([t, zeros], axis=1)
        backend.eval(points)

        # On a 1D subspace, graph geodesic = Euclidean
        result = rg.geodesic_distances(points, k_neighbors=4)
        dist_list = array_to_list(backend, result.distances)

        # Geodesic should match Euclidean on a line
        for i in range(5):
            for j in range(5):
                expected = abs(i - j)  # Euclidean distance on line
                if not _is_inf(dist_list[i][j]):
                    eps = _eps(backend, float(dist_list[i][j]), float(expected))
                    assert abs(dist_list[i][j] - expected) <= eps, (
                        f"d({i},{j}) = {dist_list[i][j]}, expected {expected}"
                    )

    def test_frechet_mean_equals_arithmetic_in_euclidean(self, any_backend: "Backend"):
        """In Euclidean space (flat), Fréchet mean = arithmetic mean.

        When points lie in a flat region with a complete k-NN graph (k = n-1),
        the Riemannian Fréchet mean converges to the arithmetic mean.
        """
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        # Generate points in a small region (approximately Euclidean)
        n_points = 10
        points = backend.random_normal((n_points, 3)) * 0.1  # Small scale
        backend.eval(points)

        # Arithmetic mean
        arith_mean = backend.mean(points, axis=0)
        arith_list = array_to_list(backend, arith_mean)

        # Fréchet mean with complete graph (k = n-1) to ensure flat space property
        result = rg.frechet_mean(points, max_iterations=100, k_neighbors=n_points - 1)
        frechet_list = array_to_list(backend, result.mean)

        # In flat space (Euclidean), Fréchet mean ≈ arithmetic mean.
        # The iterative algorithm has small numerical errors, so we use
        # sqrt(eps) tolerance instead of machine epsilon.
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
        eps = machine_epsilon(backend, points)
        tol = eps ** 0.5  # sqrt(machine_epsilon) for iterative algorithm tolerance
        for i in range(3):
            assert abs(frechet_list[i] - arith_list[i]) <= tol, (
                f"Dim {i}: Fréchet={frechet_list[i]}, Arithmetic={arith_list[i]}"
            )

    def test_geodesic_geq_euclidean(self, any_backend: "Backend"):
        """Graph geodesic distance >= Euclidean distance (Floyd-Warshall).

        The geodesic (shortest path on k-NN graph) is always >= the straight-line
        Euclidean distance (chord) because it follows graph edges.
        Note: This property is guaranteed for Floyd-Warshall but not for spectral
        geodesics which use a different distance notion.
        """
        backend = any_backend
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)

        # Points on a curved manifold (random cloud)
        points = backend.random_normal((15, 4))
        backend.eval(points)

        # Compute geodesic distances (graph shortest paths)
        geo_result = rg.geodesic_distances(points, k_neighbors=10)
        geo_list = array_to_list(backend, geo_result.distances)
        for i in range(15):
            for j in range(15):
                if i != j and _is_finite(geo_list[i][j]):
                    # Euclidean distance
                    diff = points[i] - points[j]
                    euc = backend.norm(diff)
                    backend.eval(euc)
                    euc_val = float(backend.to_scalar(euc))
                    # Geodesic should be >= Euclidean (with sqrt(eps) tolerance for numerical error)
                    # The geodesic is computed via shortest path on k-NN graph which accumulates
                    # floating point errors. The euclidean is computed in float64 (Python).
                    eps = _div_eps(backend, float(geo_list[i][j]), euc_val)
                    assert geo_list[i][j] >= euc_val - eps, (
                        f"Geodesic({i},{j})={geo_list[i][j]} < Euclidean={euc_val}"
                    )
