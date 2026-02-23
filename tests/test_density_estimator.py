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

"""Tests for density_estimator.py.

Tests the k-NN density estimator used for manifold visualization.

Key properties tested:
- Density computation is consistent
- Normalization works correctly
- Grid density computation produces valid shapes
- Edge cases are handled properly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.density_estimator import (
    DensityEstimator,
    DensityResult,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
    return get_default_backend()


def _eps(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


@pytest.fixture
def random_points_2d(backend: "Backend") -> "Array":
    """Generate random 2D points for testing."""
    backend.random_seed(42)
    return backend.random_normal((50, 2))


@pytest.fixture
def random_points_3d(backend: "Backend") -> "Array":
    """Generate random 3D points for testing."""
    backend.random_seed(42)
    return backend.random_normal((50, 3))


@pytest.fixture
def clustered_points(backend: "Backend") -> "Array":
    """Generate clustered points for testing density variation."""
    backend.random_seed(42)
    # Create two clusters
    cluster1 = backend.random_normal((25, 3)) * 0.1  # Tight cluster
    cluster2 = backend.random_normal((25, 3)) * 0.1 + 5.0  # Another cluster
    return backend.concatenate([cluster1, cluster2], axis=0)


# =============================================================================
# DensityResult Tests
# =============================================================================


class TestDensityResult:
    """Tests for DensityResult dataclass."""

    def test_creation(self, backend: "Backend") -> None:
        """Test basic result creation."""
        densities = backend.random_normal((50,))
        radii = backend.random_normal((50,))
        neighbors = backend.zeros((50, 10))

        result = DensityResult(
            densities=densities,
            radii=radii,
            neighbors=neighbors,
            k_neighbors=10,
        )

        assert result.k_neighbors == 10
        assert result.densities.shape == (50,)
        assert result.radii.shape == (50,)
        assert result.neighbors.shape == (50, 10)


# =============================================================================
# DensityEstimator Tests
# =============================================================================


class TestDensityEstimator:
    """Tests for DensityEstimator class."""

    def test_initialization(self, backend: "Backend") -> None:
        """Test estimator initialization."""
        estimator = DensityEstimator(backend)
        assert estimator.backend is backend

    def test_compute_basic(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test basic density computation."""
        estimator = DensityEstimator(backend)
        result = estimator.compute(random_points_3d)

        assert isinstance(result, DensityResult)
        assert result.densities.shape == (50,)
        assert result.radii.shape == (50,)
        assert result.neighbors.shape == (50, result.k_neighbors)

    def test_k_derived_from_data(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test k is derived from Berry & Sauer 2016: k >= ceil(log(n))."""
        import math

        estimator = DensityEstimator(backend)
        result = estimator.compute(random_points_3d)

        # k should be ceil(log(n)) for n=50: ceil(log(50)) = ceil(3.91) = 4
        expected_k = int(math.ceil(math.log(50)))
        assert result.k_neighbors == expected_k

    def test_raw_densities_positive(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test raw densities are positive (no normalization)."""
        estimator = DensityEstimator(backend)
        result = estimator.compute(random_points_3d)

        # All densities should be positive
        min_val = float(backend.to_scalar(backend.min(result.densities)))
        assert min_val > _eps(backend)

    def test_radii_positive(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test that radii are positive."""
        estimator = DensityEstimator(backend)
        result = estimator.compute(random_points_3d)

        min_val = float(backend.to_scalar(backend.min(result.radii)))
        assert min_val > _eps(backend)

    def test_clustered_density_variation(
        self, backend: "Backend", clustered_points: "Array"
    ) -> None:
        """Test that clustered points show density variation."""
        estimator = DensityEstimator(backend)
        result = estimator.compute(clustered_points)

        # Should have variation in density - use backend std
        std_val = float(backend.to_scalar(backend.std(result.densities)))
        assert std_val > _eps(backend)

    def test_very_small_point_cloud(self, backend: "Backend") -> None:
        """Test with a very small point cloud where k derives properly."""
        import math

        estimator = DensityEstimator(backend)
        # With 5 points, k = ceil(log(5)) = ceil(1.6) = 2
        small_points = backend.random_normal((5, 3))
        result = estimator.compute(small_points)

        expected_k = int(math.ceil(math.log(5)))
        assert result.k_neighbors == expected_k
        assert result.densities.shape == (5,)


# =============================================================================
# Grid Density Tests
# =============================================================================


def _has_meshgrid(backend: "Backend") -> bool:
    """Check if backend supports meshgrid."""
    return hasattr(backend, "meshgrid")


class TestGridDensity:
    """Tests for compute_grid_density method.

    Note: These tests require the meshgrid backend method which may not be
    available in all backends. Tests are skipped if meshgrid is unavailable.
    """

    def test_grid_density_basic(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test basic grid density computation."""
        assert _has_meshgrid(backend), "Backend does not support meshgrid"

        estimator = DensityEstimator(backend)
        X, Y, Z, density = estimator.compute_grid_density(
            random_points_3d, grid_size=10
        )

        assert X.shape == (10, 10, 10)
        assert Y.shape == (10, 10, 10)
        assert Z.shape == (10, 10, 10)
        assert density.shape == (10, 10, 10)

    def test_grid_density_values_positive(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test that grid densities are positive."""
        assert _has_meshgrid(backend), "Backend does not support meshgrid"

        estimator = DensityEstimator(backend)
        _, _, _, density = estimator.compute_grid_density(
            random_points_3d, grid_size=5
        )

        min_val = float(backend.to_scalar(backend.min(density)))
        assert min_val > 0

    def test_grid_density_non_3d_error(
        self, backend: "Backend", random_points_2d: "Array"
    ) -> None:
        """Test error when points are not 3D."""
        assert _has_meshgrid(backend), "Backend does not support meshgrid"

        estimator = DensityEstimator(backend)

        with pytest.raises(ValueError, match="3D"):
            estimator.compute_grid_density(random_points_2d)

    def test_grid_density_small_grid(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test grid density with different grid size."""
        assert _has_meshgrid(backend), "Backend does not support meshgrid"

        estimator = DensityEstimator(backend)
        X, Y, Z, density = estimator.compute_grid_density(
            random_points_3d, grid_size=5
        )

        assert density.shape == (5, 5, 5)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestDensityConsistency:
    """Tests for density computation consistency."""

    def test_deterministic_results(
        self, backend: "Backend"
    ) -> None:
        """Test that density computation is deterministic."""
        backend.random_seed(42)
        points = backend.random_normal((50, 3))

        estimator = DensityEstimator(backend)
        result1 = estimator.compute(points)
        result2 = estimator.compute(points)

        # Use backend operations to compare
        diff = backend.abs(result1.densities - result2.densities)
        max_diff = backend.max(diff)
        backend.eval(max_diff)
        assert float(backend.to_scalar(max_diff)) == 0.0

    def test_different_dimensions(self, backend: "Backend") -> None:
        """Test density works for different dimensions."""
        estimator = DensityEstimator(backend)

        for d in [2, 3, 4, 10]:
            backend.random_seed(42)
            points = backend.random_normal((50, d))
            result = estimator.compute(points)
            assert result.densities.shape == (50,)
