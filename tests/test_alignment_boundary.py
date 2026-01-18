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

"""Tests for alignment boundary geometric guardrails."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_boundary import (
    AlignmentBoundary,
    BoundaryViolationType,
    batch_check_boundary,
    check_boundary,
    compute_alignment_boundary,
    steer_to_boundary,
)
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


@pytest.fixture
def backend():
    """Get default backend for tests."""
    return get_default_backend()


@pytest.fixture
def simple_boundary(backend):
    """Create a simple alignment boundary for testing."""
    # Refusal direction: unit vector along first dimension
    refusal_direction = backend.zeros((10,))
    refusal_direction = backend.put_along_axis(
        refusal_direction, backend.array([0]), backend.array([1.0]), axis=0
    )
    backend.eval(refusal_direction)

    # Safe centroid: origin
    safe_centroid = backend.zeros((10,))
    backend.eval(safe_centroid)

    return AlignmentBoundary(
        refusal_direction=refusal_direction,
        safe_centroid=safe_centroid,
        refusal_threshold=0.5,  # Must have projection >= 0.5 onto refusal direction
        safe_radius=2.0,  # Must be within distance 2.0 of centroid
        layer_index=0,
    )


class TestComputeAlignmentBoundary:
    """Tests for compute_alignment_boundary function."""

    def test_computes_boundary_from_safe_activations(self, backend):
        """Test that boundary is computed from safe activation statistics."""
        # Create synthetic safe activations
        n_samples = 100
        hidden_dim = 10

        # Safe activations clustered around origin with refusal-direction component
        safe_acts = backend.random_normal((n_samples, hidden_dim)) * 0.5
        # Add positive component along first dimension (refusal direction)
        # Create offset array: [1.0, 0, 0, 0, ...] to add to each row
        offset = backend.zeros((hidden_dim,))
        offset = backend.put_along_axis(offset, backend.array([0]), backend.array([1.0]), axis=0)
        safe_acts = safe_acts + offset  # Broadcasting adds to each row
        backend.eval(safe_acts)

        # Refusal direction
        refusal_dir = backend.zeros((hidden_dim,))
        refusal_dir = backend.put_along_axis(
            refusal_dir, backend.array([0]), backend.array([1.0]), axis=0
        )
        backend.eval(refusal_dir)

        boundary = compute_alignment_boundary(
            refusal_direction=refusal_dir,
            safe_activations=safe_acts,
            refusal_percentile=5.0,
            distance_percentile=95.0,
            layer_index=5,
            backend=backend,
        )

        # Check that boundary has reasonable values
        assert boundary.layer_index == 5
        assert boundary.refusal_threshold > 0  # Should be positive since safe acts have positive projection
        assert boundary.safe_radius > 0  # Should be positive

    def test_percentile_affects_threshold(self, backend):
        """Test that percentile parameters affect threshold values."""
        n_samples = 100
        hidden_dim = 10

        safe_acts = backend.random_normal((n_samples, hidden_dim))
        backend.eval(safe_acts)

        refusal_dir = backend.zeros((hidden_dim,))
        refusal_dir = backend.put_along_axis(
            refusal_dir, backend.array([0]), backend.array([1.0]), axis=0
        )
        backend.eval(refusal_dir)

        # Stricter boundary (lower refusal percentile = higher threshold requirement)
        strict_boundary = compute_alignment_boundary(
            refusal_direction=refusal_dir,
            safe_activations=safe_acts,
            refusal_percentile=10.0,
            distance_percentile=90.0,
            backend=backend,
        )

        # Looser boundary
        loose_boundary = compute_alignment_boundary(
            refusal_direction=refusal_dir,
            safe_activations=safe_acts,
            refusal_percentile=1.0,
            distance_percentile=99.0,
            backend=backend,
        )

        # Looser boundary should have lower threshold and larger radius
        assert loose_boundary.refusal_threshold < strict_boundary.refusal_threshold
        assert loose_boundary.safe_radius > strict_boundary.safe_radius


class TestCheckBoundary:
    """Tests for check_boundary function."""

    def test_safe_activation_passes(self, backend, simple_boundary):
        """Test that activation within boundary passes."""
        # Create activation with high refusal projection, near centroid
        activation = backend.zeros((10,))
        activation = backend.put_along_axis(
            activation, backend.array([0]), backend.array([1.0]), axis=0
        )
        backend.eval(activation)

        result = check_boundary(activation, simple_boundary, backend=backend)

        assert result.is_within_boundary
        assert result.violation_type == BoundaryViolationType.NONE
        assert result.refusal_projection >= simple_boundary.refusal_threshold
        assert result.distance_to_centroid <= simple_boundary.safe_radius

    def test_low_refusal_projection_fails(self, backend, simple_boundary):
        """Test that activation with low refusal projection fails."""
        # Create activation perpendicular to refusal direction
        activation = backend.zeros((10,))
        activation = backend.put_along_axis(
            activation, backend.array([1]), backend.array([0.1]), axis=0
        )
        backend.eval(activation)

        result = check_boundary(activation, simple_boundary, backend=backend)

        assert not result.is_within_boundary
        assert result.violation_type == BoundaryViolationType.LOW_REFUSAL_PROJECTION
        assert result.refusal_projection < simple_boundary.refusal_threshold

    def test_high_distance_fails(self, backend, simple_boundary):
        """Test that activation far from centroid fails."""
        # Create activation with good refusal projection but far from centroid
        activation = backend.zeros((10,))
        # Good refusal projection
        activation = backend.put_along_axis(
            activation, backend.array([0]), backend.array([1.0]), axis=0
        )
        # But far away in another dimension
        activation = backend.put_along_axis(
            activation, backend.array([1]), backend.array([10.0]), axis=0
        )
        backend.eval(activation)

        result = check_boundary(activation, simple_boundary, backend=backend)

        assert not result.is_within_boundary
        assert result.violation_type == BoundaryViolationType.HIGH_DISTANCE
        assert result.distance_to_centroid > simple_boundary.safe_radius

    def test_both_violations(self, backend, simple_boundary):
        """Test that both violations are detected."""
        # Create activation with low refusal projection AND far from centroid
        activation = backend.zeros((10,))
        # Zero refusal projection (perpendicular)
        activation = backend.put_along_axis(
            activation, backend.array([1]), backend.array([10.0]), axis=0
        )
        backend.eval(activation)

        result = check_boundary(activation, simple_boundary, backend=backend)

        assert not result.is_within_boundary
        assert result.violation_type == BoundaryViolationType.BOTH


class TestSteerToBoundary:
    """Tests for steer_to_boundary function."""

    def test_safe_activation_unchanged(self, backend, simple_boundary):
        """Test that safe activation is not modified."""
        activation = backend.zeros((10,))
        activation = backend.put_along_axis(
            activation, backend.array([0]), backend.array([1.0]), axis=0
        )
        backend.eval(activation)

        steered = steer_to_boundary(activation, simple_boundary, backend=backend)
        backend.eval(steered)

        # Should be unchanged
        diff = backend.sum(backend.abs(steered - activation))
        backend.eval(diff)
        tol = regularization_epsilon(backend, diff)
        assert float(backend.to_scalar(diff)) <= tol

    def test_low_projection_steered_up(self, backend, simple_boundary):
        """Test that low projection activation gets refusal direction added."""
        # Start with zero projection
        activation = backend.zeros((10,))
        activation = backend.put_along_axis(
            activation, backend.array([1]), backend.array([0.1]), axis=0
        )
        backend.eval(activation)

        steered = steer_to_boundary(activation, simple_boundary, backend=backend)
        backend.eval(steered)

        # Check that steered has higher projection
        result_before = check_boundary(activation, simple_boundary, backend=backend)
        result_after = check_boundary(steered, simple_boundary, backend=backend)

        assert result_after.refusal_projection > result_before.refusal_projection


class TestBatchCheckBoundary:
    """Tests for batch_check_boundary function."""

    def test_batch_check(self, backend, simple_boundary):
        """Test batch checking multiple activations."""
        # Create batch with mix of safe and unsafe activations
        n_samples = 10
        hidden_dim = 10

        # Build activations list then stack
        activation_list = []

        # First half: safe (high refusal projection along dim 0)
        for _ in range(5):
            act = backend.zeros((hidden_dim,))
            act = backend.put_along_axis(act, backend.array([0]), backend.array([1.0]), axis=0)
            backend.eval(act)
            activation_list.append(act)

        # Second half: unsafe (low refusal projection, only dim 1 has value)
        for _ in range(5):
            act = backend.zeros((hidden_dim,))
            act = backend.put_along_axis(act, backend.array([1]), backend.array([0.1]), axis=0)
            backend.eval(act)
            activation_list.append(act)

        activations = backend.stack(activation_list, axis=0)
        backend.eval(activations)

        results, violation_rate = batch_check_boundary(
            activations, simple_boundary, backend=backend
        )

        assert len(results) == n_samples
        assert 0.0 <= violation_rate <= 1.0
        # Should have approximately 50% violation rate
        assert 0.3 <= violation_rate <= 0.7
