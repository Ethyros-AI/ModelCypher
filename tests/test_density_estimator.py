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
    DensityConfiguration,
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
# DensityConfiguration Tests
# =============================================================================


class TestDensityConfiguration:
    """Tests for DensityConfiguration dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values.

        k_neighbors defaults to None (derived from sqrt(n) at runtime).
        """
        config = DensityConfiguration()
        assert config.k_neighbors is None  # Derived from sqrt(n) at runtime
        assert config.normalize is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DensityConfiguration(
            k_neighbors=20,
            normalize=False,
        )
        assert config.k_neighbors == 20
        assert config.normalize is False


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

    def test_compute_with_config(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test density computation with explicit config."""
        estimator = DensityEstimator(backend)
        config = DensityConfiguration(k_neighbors=5, normalize=True)
        result = estimator.compute(random_points_3d, config)

        assert result.k_neighbors == 5

    def test_normalized_densities(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test that normalized densities are in [0, 1]."""
        estimator = DensityEstimator(backend)
        config = DensityConfiguration(normalize=True)
        result = estimator.compute(random_points_3d, config)

        eps = _eps(backend)
        min_val = float(backend.to_scalar(backend.min(result.densities)))
        max_val = float(backend.to_scalar(backend.max(result.densities)))
        assert min_val >= -eps
        assert max_val <= 1.0 + eps

    def test_unnormalized_densities(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test unnormalized densities."""
        estimator = DensityEstimator(backend)
        config = DensityConfiguration(normalize=False)
        result = estimator.compute(random_points_3d, config)

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
        config = DensityConfiguration(k_neighbors=5, normalize=True)
        result = estimator.compute(clustered_points, config)

        # Should have variation in density - use backend std
        std_val = float(backend.to_scalar(backend.std(result.densities)))
        assert std_val > _eps(backend)

    def test_too_few_points_error(self, backend: "Backend") -> None:
        """Test error when too few points for k-NN."""
        estimator = DensityEstimator(backend)
        small_points = backend.random_normal((10, 3))
        config = DensityConfiguration(k_neighbors=small_points.shape[0] + 1)

        with pytest.raises(ValueError, match="more than"):
            estimator.compute(small_points, config)

    def test_local_density_convenience(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test local_density convenience method."""
        estimator = DensityEstimator(backend)
        densities = estimator.local_density(random_points_3d, k=5)

        assert densities.shape == (50,)
        # Should be normalized
        eps = _eps(backend)
        min_val = float(backend.to_scalar(backend.min(densities)))
        max_val = float(backend.to_scalar(backend.max(densities)))
        assert min_val >= -eps
        assert max_val <= 1.0 + eps


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
        if not _has_meshgrid(backend):
            pytest.skip("Backend does not support meshgrid")

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
        if not _has_meshgrid(backend):
            pytest.skip("Backend does not support meshgrid")

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
        if not _has_meshgrid(backend):
            pytest.skip("Backend does not support meshgrid")

        estimator = DensityEstimator(backend)

        with pytest.raises(ValueError, match="3D"):
            estimator.compute_grid_density(random_points_2d)

    def test_grid_density_with_config(
        self, backend: "Backend", random_points_3d: "Array"
    ) -> None:
        """Test grid density with explicit config."""
        if not _has_meshgrid(backend):
            pytest.skip("Backend does not support meshgrid")

        estimator = DensityEstimator(backend)
        config = DensityConfiguration(k_neighbors=3)
        X, Y, Z, density = estimator.compute_grid_density(
            random_points_3d, grid_size=5, config=config
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
