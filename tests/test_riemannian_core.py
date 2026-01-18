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

"""Tests for riemannian_core.py - Core Riemannian geometry operations.

Tests cover:
- RiemannianGeometry initialization and backend handling
- Fréchet mean computation (single and batch)
- Geodesic distance matrix computation (Isomap-style)
- Graph connectivity checks (minimum connected k)
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_core import (
    RiemannianGeometry,
    _get_riemannian_geometry,
)
from modelcypher.core.domain.geometry.riemannian_types import GeodesicDistanceResult


# =============================================================================
# RiemannianGeometry Initialization Tests
# =============================================================================


class TestRiemannianGeometryInit:
    """Tests for RiemannianGeometry initialization."""

    def test_singleton_behavior(self):
        """_get_riemannian_geometry returns same instance for same backend."""
        backend = get_default_backend()
        geom1 = _get_riemannian_geometry(backend)
        geom2 = _get_riemannian_geometry(backend)
        assert geom1 is geom2

    def test_init_with_backend(self):
        """RiemannianGeometry initializes with provided backend."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        assert geom._backend is backend


# =============================================================================
# Fréchet Mean Tests
# =============================================================================


class TestFrechetMean:
    """Tests for Fréchet mean computation."""

    def test_frechet_mean_single_point(self):
        """Fréchet mean of a single point is the point itself."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        point = backend.array([[1.0, 2.0, 3.0]])
        result = geom.frechet_mean(point)
        
        # Should be identical
        diff = backend.to_scalar(backend.sum(backend.abs(point - result.mean)))
        eps = division_epsilon(backend, point)
        assert diff < eps

    def test_frechet_mean_identical_points(self):
        """Fréchet mean of identical points is the point."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        points = backend.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        result = geom.frechet_mean(points)
        
        result_list = backend.tolist(result.mean)
        eps = division_epsilon(backend, points)
        assert abs(result_list[0] - 1.0) < eps

    def test_frechet_mean_euclidean_approx(self):
        """test that it runs (approximation checks are tricky without known curvature)."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        # Simple points in a line
        points = backend.array([[0.0, 0.0], [1.0, 1.0]])
        result = geom.frechet_mean(points)
        
        # Expect [0.5, 0.5] within precision
        result_list = backend.tolist(result.mean)
        eps = division_epsilon(backend, points)
        assert abs(result_list[0] - 0.5) <= eps


# =============================================================================
# Geodesic Distance Tests
# =============================================================================


class TestGeodesicDistances:
    """Tests for geodesic distance computation."""

    def test_geodesic_distances_small_set(self):
        """geodesic_distances returns result with correct shape."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        # 4 points
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        result = geom.geodesic_distances(points, k_neighbors=3)
        
        assert isinstance(result, GeodesicDistanceResult)
        # Should be 4x4
        shape = backend.shape(result.distances)
        assert shape == (4, 4)

    def test_minimum_connected_k(self):
        """_minimum_connected_k finds usable k."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        points = backend.array([
            [0.0, 0.0],
            [0.1, 0.0],  # Close to 1
            [5.0, 0.0],  # Far
            [5.1, 0.0],  # Close to 3
        ])
        
        # With k=1, we might have 2 separate components {(0,1), (2,3)}
        # It should try to find a k that connects them or fallback
        k, _, _ = geom._minimum_connected_k(points)
        assert k >= 1


# =============================================================================
# Batch Fréchet Mean Tests
# =============================================================================


class TestFrechetMeanBatch:
    """Tests for batch Fréchet mean computation."""

    def test_frechet_mean_batch_output_shape(self):
        """frechet_mean_batch returns [B, D] array."""
        backend = get_default_backend()
        geom = RiemannianGeometry(backend)
        
        # Batch of 2 sets of points, each set has 3 points of dim 2
        batch_points = backend.array([
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # Set 1
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], # Set 2
        ])
        
        means = geom.frechet_mean_batch(batch_points)
        
        shape = backend.shape(means)
        assert shape == (2, 2)
