# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for Grassmann distance on subspace manifolds."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.subspace import (
    GrassmannDistanceResult,
    compute_grassmann_distance,
    grassmann_log,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestGrassmannDistance:
    """Tests for Grassmann geodesic distance."""

    def test_identical_subspaces_zero_distance(self, backend):
        """Identical subspaces should have zero distance."""
        b = backend
        # Orthonormal basis for 2D subspace of R^3
        Q = b.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        result = compute_grassmann_distance(Q, Q, backend=b)

        assert abs(result.geodesic_distance) < 1e-10
        assert abs(result.chordal_distance) < 1e-10

    def test_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        Q1 = b.array([[1.0, 0.0]])
        Q2 = b.array([[1.0, 0.0]])

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        assert hasattr(result, "geodesic_distance")
        assert hasattr(result, "principal_angles")
        assert hasattr(result, "chordal_distance")
        assert hasattr(result, "subspace_dims")

    def test_orthogonal_subspaces_max_distance(self, backend):
        """Orthogonal subspaces should have maximal distance."""
        b = backend
        # Two orthogonal 1D subspaces in R^2
        Q1 = b.array([[1.0, 0.0]])  # x-axis
        Q2 = b.array([[0.0, 1.0]])  # y-axis

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        # Principal angle should be pi/2
        angles = b.tolist(result.principal_angles)
        assert len(angles) == 1
        assert abs(angles[0] - 1.5707963) < 0.01  # pi/2 ≈ 1.5708

        # Geodesic distance for orthogonal 1D subspaces is pi/2
        assert abs(result.geodesic_distance - 1.5707963) < 0.01

    def test_same_line_rotated(self, backend):
        """Rotated line should have distance proportional to angle."""
        b = backend
        import math

        theta = 0.3  # 0.3 radians
        Q1 = b.array([[1.0, 0.0]])
        Q2 = b.array([[math.cos(theta), math.sin(theta)]])

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        # For 1D subspaces, geodesic distance equals the angle
        assert abs(result.geodesic_distance - theta) < 0.01

    def test_higher_dimensional_subspaces(self, backend):
        """Test with 2D subspaces in R^4."""
        b = backend
        # Two 2D subspaces in R^4
        Q1 = b.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        Q2 = b.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        # Principal angles should both be pi/2
        angles = b.tolist(result.principal_angles)
        assert len(angles) == 2

        # Distance should be sqrt(2) * pi/2 ≈ 2.22
        expected = 1.5707963 * 1.414  # sqrt(2 * (pi/2)^2)
        assert abs(result.geodesic_distance - expected) < 0.1

    def test_chordal_vs_geodesic(self, backend):
        """Chordal distance should be <= geodesic distance."""
        b = backend
        import math

        theta = 0.5
        Q1 = b.array([[1.0, 0.0]])
        Q2 = b.array([[math.cos(theta), math.sin(theta)]])

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        # Chordal = sin(theta), Geodesic = theta
        # For theta in (0, pi), sin(theta) <= theta
        assert result.chordal_distance <= result.geodesic_distance + 1e-10

    def test_column_format_subspaces(self, backend):
        """Test with column-format subspaces [n, k]."""
        b = backend
        # Subspaces in column format
        Q1 = b.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])  # [3, 2]
        Q2 = b.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ])  # [3, 2]

        result = compute_grassmann_distance(Q1, Q2, backend=b)

        # Should auto-detect and handle column format
        assert result.geodesic_distance >= 0


class TestGrassmannLog:
    """Tests for Grassmann logarithm."""

    def test_zero_at_identity(self, backend):
        """Log of same subspace should be zero."""
        b = backend
        Q = b.array([[1.0, 0.0]])

        tangent = grassmann_log(Q, Q, backend=b)
        b.eval(tangent)

        norm = b.sqrt(b.sum(tangent * tangent))
        b.eval(norm)

        assert float(b.to_scalar(norm)) < 1e-10

    def test_tangent_vector_shape(self, backend):
        """Tangent vector should match subspace shape."""
        b = backend
        Q1 = b.array([[1.0, 0.0, 0.0]])
        Q2 = b.array([[0.707, 0.707, 0.0]])

        tangent = grassmann_log(Q1, Q2, backend=b)
        b.eval(tangent)

        assert b.shape(tangent) == b.shape(Q1)
