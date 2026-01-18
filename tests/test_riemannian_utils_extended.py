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

"""Extended tests for Riemannian geometry utilities.

Tests critical APIs:
- frechet_mean(): Compute Frechet mean of point cloud
- geodesic_distance_matrix(): Compute pairwise geodesic distances
- farthest_point_sampling(): Geodesic farthest point sampling
- RiemannianGeometry: Main class for geodesic operations
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    farthest_point_sampling,
    frechet_mean,
    geodesic_distance_matrix,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestFrechetMean:
    """Tests for frechet_mean()."""

    def test_basic_computation(self, backend):
        """Basic Frechet mean should work."""
        points = backend.random_normal((16, 32))
        backend.eval(points)

        mean = frechet_mean(points, backend=backend)

        assert mean is not None
        assert backend.shape(mean) == (32,)
        assert all_finite(mean, backend)

    def test_single_point_returns_itself(self, backend):
        """Single point's Frechet mean should be itself."""
        point = backend.random_normal((1, 32))
        backend.eval(point)

        mean = frechet_mean(point, backend=backend)

        diff = backend.mean(backend.abs(mean - backend.squeeze(point, axis=0)))
        backend.eval(diff)
        tol = regularization_epsilon(backend, mean)
        assert float(backend.to_scalar(diff)) <= tol

    def test_with_weights(self, backend):
        """Weighted Frechet mean should work."""
        points = backend.random_normal((16, 32))
        weights = backend.abs(backend.random_normal((16,)))
        backend.eval(points, weights)

        mean = frechet_mean(points, weights=weights, backend=backend)

        assert mean is not None
        assert backend.shape(mean) == (32,)
        assert all_finite(mean, backend)

    def test_mean_is_centroid_like(self, backend):
        """Frechet mean should be close to geometric centroid."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        fmean = frechet_mean(points, backend=backend)
        arithmetic_mean = backend.mean(points, axis=0)
        backend.eval(fmean, arithmetic_mean)

        # Frechet mean should be within the data's own spread
        diff = backend.mean(backend.abs(fmean - arithmetic_mean))
        spread = backend.mean(backend.abs(points - arithmetic_mean))
        backend.eval(diff, spread)
        tol = regularization_epsilon(backend, points)
        assert float(backend.to_scalar(diff)) <= float(backend.to_scalar(spread)) + tol


class TestGeodesicDistanceMatrix:
    """Tests for geodesic_distance_matrix()."""

    def test_basic_computation(self, backend):
        """Basic geodesic distance matrix should work."""
        points = backend.random_normal((16, 32))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        assert dist_matrix is not None
        assert backend.shape(dist_matrix) == (16, 16)
        assert all_finite(dist_matrix, backend)

    def test_diagonal_is_zero(self, backend):
        """Diagonal of distance matrix should be zero."""
        points = backend.random_normal((16, 32))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        diagonal = backend.diag(dist_matrix)
        backend.eval(diagonal)
        max_diag = backend.max(backend.abs(diagonal))
        backend.eval(max_diag)
        tol = regularization_epsilon(backend, dist_matrix)
        assert float(backend.to_scalar(max_diag)) <= tol

    def test_symmetric(self, backend):
        """Distance matrix should be symmetric."""
        points = backend.random_normal((16, 32))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        diff = dist_matrix - backend.transpose(dist_matrix)
        max_diff = backend.max(backend.abs(diff))
        backend.eval(max_diff)
        tol = regularization_epsilon(backend, dist_matrix)
        assert float(backend.to_scalar(max_diff)) <= tol

    def test_non_negative(self, backend):
        """All distances should be non-negative."""
        points = backend.random_normal((16, 32))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        min_dist = backend.min(dist_matrix)
        backend.eval(min_dist)
        tol = regularization_epsilon(backend, dist_matrix)
        assert float(backend.to_scalar(min_dist)) >= -tol

    def test_explicit_k_neighbors(self, backend):
        """Explicit k_neighbors should be used."""
        points = backend.random_normal((20, 32))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, k_neighbors=5, backend=backend)

        assert dist_matrix is not None
        assert backend.shape(dist_matrix) == (20, 20)


class TestFarthestPointSampling:
    """Tests for farthest_point_sampling()."""

    def test_basic_sampling(self, backend):
        """Basic farthest point sampling should work."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        indices = farthest_point_sampling(points, n_samples=5, backend=backend)

        assert len(indices) == 5
        assert all(0 <= i < 32 for i in indices)

    def test_indices_unique(self, backend):
        """Selected indices should be unique."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        indices = farthest_point_sampling(points, n_samples=10, backend=backend)

        assert len(indices) == len(set(indices))

    def test_seed_idx_rejected(self, backend):
        """Seed overrides are not supported."""
        points = backend.random_normal((32, 16))
        backend.eval(points)

        with pytest.raises(TypeError):
            farthest_point_sampling(points, n_samples=5, seed_idx=5, backend=backend)

    def test_sample_all_points(self, backend):
        """Sampling all points should return all indices."""
        n_points = 16
        points = backend.random_normal((n_points, 8))
        backend.eval(points)

        indices = farthest_point_sampling(points, n_samples=n_points, backend=backend)

        assert len(indices) == n_points
        assert set(indices) == set(range(n_points))


class TestRiemannianGeometry:
    """Tests for RiemannianGeometry class."""

    def test_geodesic_distances_result(self, backend):
        """geodesic_distances should return full result."""
        rg = RiemannianGeometry(backend)
        points = backend.random_normal((16, 32))
        backend.eval(points)

        result = rg.geodesic_distances(points)

        assert result.distances is not None
        assert result.k_neighbors > 0
        assert backend.shape(result.distances) == (16, 16)

    def test_frechet_mean_result(self, backend):
        """frechet_mean should return full result."""
        rg = RiemannianGeometry(backend)
        points = backend.random_normal((16, 32))
        backend.eval(points)

        result = rg.frechet_mean(points)

        assert result.mean is not None
        assert result.iterations >= 0
        assert result.converged is True or result.converged is False

    def test_farthest_point_sampling_result(self, backend):
        """farthest_point_sampling should return full result."""
        rg = RiemannianGeometry(backend)
        points = backend.random_normal((32, 16))
        backend.eval(points)

        result = rg.farthest_point_sampling(points, n_samples=5)

        assert len(result.selected_indices) == 5
        assert result.min_distances is not None


class TestRiemannianMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_points=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_frechet_mean_finite(self, n_points, d):
        """Frechet mean should always be finite."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        mean = frechet_mean(points, backend=backend)

        assert all_finite(mean, backend)

    @given(
        n_points=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_distance_matrix_symmetric(self, n_points, d):
        """Distance matrix should be symmetric."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        diff = dist_matrix - backend.transpose(dist_matrix)
        max_diff = backend.max(backend.abs(diff))
        backend.eval(max_diff)
        tol = regularization_epsilon(backend, dist_matrix)
        assert float(backend.to_scalar(max_diff)) <= tol

    @given(
        n_points=st.integers(min_value=8, max_value=32),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_distance_matrix_non_negative(self, n_points, d):
        """All geodesic distances should be non-negative."""
        backend = get_default_backend()
        points = backend.random_normal((n_points, d))
        backend.eval(points)

        dist_matrix = geodesic_distance_matrix(points, backend=backend)

        min_dist = backend.min(dist_matrix)
        backend.eval(min_dist)
        tol = regularization_epsilon(backend, dist_matrix)
        assert float(backend.to_scalar(min_dist)) >= -tol

    @given(
        n_points=st.integers(min_value=8, max_value=32),
        n_samples=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=5, deadline=None)
    def test_fps_indices_valid(self, n_points, n_samples):
        """FPS indices should be valid and unique."""
        n_samples = min(n_samples, n_points)
        backend = get_default_backend()
        points = backend.random_normal((n_points, 16))
        backend.eval(points)

        indices = farthest_point_sampling(points, n_samples=n_samples, backend=backend)

        assert len(indices) == n_samples
        assert len(set(indices)) == n_samples
        assert all(0 <= i < n_points for i in indices)
