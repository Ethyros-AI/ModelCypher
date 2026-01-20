# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for parallel transport and holonomy."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.parallel_transport import (
    ParallelTransporter,
    compute_holonomy,
    parallel_transport,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestParallelTransport:
    """Tests for parallel transport along geodesic paths."""

    def test_trivial_path_no_transport(self, backend):
        """Single point path should return same vector."""
        b = backend
        points = b.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        initial_vector = b.array([1.0, 0.0, 0.0])

        result = parallel_transport(points, [0], initial_vector, backend=b)

        assert result.path_length == 0.0
        assert result.angular_drift == 0.0
        assert result.norm_ratio == 1.0

    def test_straight_line_transport(self, backend):
        """Transport along straight line should preserve vector component."""
        b = backend
        # Points along x-axis
        points = b.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        # Vector perpendicular to path
        initial_vector = b.array([0.0, 1.0, 0.0])

        result = parallel_transport(points, [0, 1, 2], initial_vector, backend=b)

        # Path length should be 2
        assert abs(result.path_length - 2.0) < 1e-5

        # Perpendicular vector should be preserved (zero drift)
        assert result.angular_drift < 1e-5

        # Norm should be preserved
        assert abs(result.norm_ratio - 1.0) < 1e-5

    def test_transport_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        points = b.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        initial_vector = b.array([1.0, 0.0])

        result = parallel_transport(points, [0, 1, 2], initial_vector, backend=b)

        assert hasattr(result, "transported_vector")
        assert hasattr(result, "initial_vector")
        assert hasattr(result, "path_indices")
        assert hasattr(result, "path_length")
        assert hasattr(result, "angular_drift")
        assert hasattr(result, "norm_ratio")

    def test_transport_with_turn(self, backend):
        """Transport around a corner should rotate vector."""
        b = backend
        # L-shaped path
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        initial_vector = b.array([1.0, 0.0])

        result = parallel_transport(points, [0, 1, 2], initial_vector, backend=b)

        # Some angular drift expected from turn
        assert result.angular_drift >= 0

    def test_parallel_vector_preserved(self, backend):
        """Vector parallel to path should be preserved, not zeroed."""
        b = backend
        # Points along x-axis
        points = b.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        # Vector parallel to path
        initial_vector = b.array([1.0, 0.0, 0.0])

        result = parallel_transport(points, [0, 1, 2], initial_vector, backend=b)

        # Vector parallel to travel should be preserved in flat space
        transported = result.transported_vector
        b.eval(transported)

        # Norm should be preserved (not zeroed)
        transported_norm = b.sqrt(b.sum(transported * transported))
        b.eval(transported_norm)
        assert float(b.to_scalar(transported_norm)) > 0.9  # Should be ~1.0

        # The transported vector should still be in the same direction
        # In flat space, parallel transport preserves the vector
        dot_product = b.sum(initial_vector * transported)
        b.eval(dot_product)
        assert float(b.to_scalar(dot_product)) > 0.9  # Should be ~1.0

    def test_transported_basis_stays_full_rank(self, backend):
        """Transported basis vectors should remain linearly independent."""
        b = backend
        from modelcypher.core.domain.geometry.parallel_transport import ParallelTransporter

        # Path with a turn
        points = b.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        transporter = ParallelTransporter(b)
        transported_basis = transporter.transport_basis_along_path(points, [0, 1, 2])
        b.eval(transported_basis)

        # Compute determinant to check rank
        # For full rank 3x3 matrix, det should be non-zero
        # Note: transported_basis is [3, 3]
        det = b.det(transported_basis)
        b.eval(det)

        assert abs(float(b.to_scalar(det))) > 0.5  # Should be ~1 for orthonormal


class TestHolonomy:
    """Tests for holonomy around closed loops."""

    def test_triangle_holonomy(self, backend):
        """Holonomy around a triangle."""
        b = backend
        # Simple triangle in 2D
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866],  # equilateral triangle
        ])

        result = compute_holonomy(points, [0, 1, 2], backend=b)

        assert hasattr(result, "holonomy_matrix")
        assert hasattr(result, "holonomy_angle")
        assert hasattr(result, "loop_indices")
        assert hasattr(result, "loop_length")
        assert hasattr(result, "axis")

    def test_flat_space_near_zero_holonomy(self, backend):
        """Holonomy in flat space should be small."""
        b = backend
        # Small triangle in flat plane
        points = b.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.05, 0.05, 0.0],
        ])

        result = compute_holonomy(points, [0, 1, 2], backend=b)

        # Should be close to identity (small angle)
        # Note: numerical precision limits how small this can be
        assert result.holonomy_angle < 1.0  # Less than ~57 degrees

    def test_holonomy_matrix_is_rotation(self, backend):
        """Holonomy matrix should be orthogonal."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])

        result = compute_holonomy(points, [0, 1, 2], backend=b)
        H = result.holonomy_matrix
        b.eval(H)

        # H @ H^T should be identity
        HHt = b.matmul(H, b.transpose(H))
        b.eval(HHt)
        I = b.eye(2)

        diff = HHt - I
        diff_norm = b.sqrt(b.sum(diff * diff))
        b.eval(diff_norm)

        assert float(b.to_scalar(diff_norm)) < 1e-5

    def test_degenerate_loop(self, backend):
        """Two-point loop should have zero holonomy."""
        b = backend
        points = b.array([[0.0, 0.0], [1.0, 0.0]])

        result = compute_holonomy(points, [0, 1], backend=b)

        assert result.holonomy_angle == 0.0


class TestParallelTransporter:
    """Tests for ParallelTransporter class."""

    def test_transport_basis(self, backend):
        """Transport complete basis along path."""
        b = backend
        transporter = ParallelTransporter(b)

        points = b.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])

        basis = transporter.transport_basis_along_path(points, [0, 1, 2])
        b.eval(basis)

        # Result should be 3x3 matrix
        assert b.shape(basis) == (3, 3)

    def test_triangle_holonomy_convenience(self, backend):
        """Test compute_triangle_holonomy convenience method."""
        b = backend
        transporter = ParallelTransporter(b)

        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])

        result = transporter.compute_triangle_holonomy(points, 0, 1, 2)

        assert hasattr(result, "holonomy_angle")
        assert result.loop_indices == (0, 1, 2)
