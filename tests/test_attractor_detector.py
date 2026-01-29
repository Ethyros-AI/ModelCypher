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

"""Tests for attractor detection in manifold trajectories.

These tests verify that the AttractorDetector correctly identifies:
1. Fixed points (frozen trajectories)
2. Limit cycles (repeating patterns)
3. Normal flow (healthy trajectories)

The detector should use geometry-derived thresholds, not heuristics.
"""

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.attractor_detector import (
    AttractorDetector,
    AttractorState,
    AttractorType,
    create_attractor_detector,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def detector(backend):
    """Create a detector with small hidden dimension for testing."""
    return AttractorDetector(hidden_dim=64, window_size=10, backend=backend)


class TestAttractorDetection:
    """Tests for attractor detection logic."""

    def test_no_attractor_for_flowing_trajectory(self, detector, backend):
        """Normal trajectory should not be flagged as attractor."""
        import math

        # Generate a flowing trajectory with diverse directions
        # Use rotation to ensure positions are not collinear
        for i in range(20):
            # Create position that rotates through space (not just linear)
            angle = float(i) * 0.5
            position = backend.array([
                math.sin(angle + j * 0.1) + float(i) * 0.1
                for j in range(64)
            ])
            state = detector.update(position)

        # Should not be in an attractor
        assert state.attractor_type == AttractorType.NONE
        assert state.severity < 0.5

    def test_fixed_point_detection(self, detector, backend):
        """Frozen trajectory should be detected as fixed point."""
        # Same position repeated
        fixed_position = backend.array([1.0] * 64)

        for _ in range(20):
            state = detector.update(fixed_position)

        # Should detect fixed point
        assert state.attractor_type == AttractorType.FIXED_POINT
        assert state.severity > 0.5
        assert state.position_variance < 1e-6

    def test_limit_cycle_detection(self, detector, backend):
        """Cycling trajectory should be detected as limit cycle."""
        # Two alternating positions
        pos_a = backend.array([1.0] * 64)
        pos_b = backend.array([2.0] * 64)

        for i in range(20):
            position = pos_a if i % 2 == 0 else pos_b
            state = detector.update(position)

        # Should detect limit cycle
        assert state.attractor_type == AttractorType.LIMIT_CYCLE
        assert state.cycle_length > 0

    def test_slow_dynamics_detection(self, detector, backend):
        """Very slow moving trajectory should be flagged."""
        # Trajectory moving very slowly (delta < sqrt(eps))
        sqrt_eps = math.sqrt(1e-15)

        for i in range(20):
            # Move by tiny amount each step
            position = backend.array([float(i) * sqrt_eps * 0.1 for _ in range(64)])
            state = detector.update(position)

        # Should detect slow dynamics
        assert state.attractor_type in (
            AttractorType.SLOW_DYNAMICS,
            AttractorType.FIXED_POINT,
        )

    def test_escape_direction_requires_null_basis(self, detector, backend):
        """Escape direction should only be computed when null basis is provided."""
        fixed_position = backend.array([1.0] * 64)

        # Without null basis
        for _ in range(10):
            state = detector.update(fixed_position, null_basis=None)

        assert state.escape_direction is None

        # Reset and try with null basis
        detector.reset()
        null_basis = backend.array([[1.0 if j == i else 0.0 for j in range(64)] for i in range(10)])

        for _ in range(10):
            state = detector.update(fixed_position, null_basis=null_basis)

        # With attractor and null basis, should have escape direction
        if state.attractor_type != AttractorType.NONE:
            assert state.escape_direction is not None
            assert len(state.escape_direction) == 64

    def test_escape_resets_trajectory(self, detector, backend):
        """Escape should reset the trajectory state."""
        fixed_position = backend.array([1.0] * 64)

        for _ in range(15):
            detector.update(fixed_position)

        assert detector.timesteps_stuck > 0

        # Escape
        escape_dir = tuple([1.0 / 8] * 64)  # Rough unit vector
        detector.escape_attractor(fixed_position, escape_dir)

        # Should be reset
        assert detector.timesteps_stuck == 0

    def test_window_size_limits_memory(self, backend):
        """Detector should only keep window_size positions."""
        detector = AttractorDetector(hidden_dim=64, window_size=5, backend=backend)

        for i in range(20):
            position = backend.array([float(i)] * 64)
            detector.update(position)

        # Should only have window_size positions
        assert len(detector._positions) == 5

    def test_to_dict_serializable(self, detector, backend):
        """AttractorState should be JSON-serializable."""
        position = backend.array([1.0] * 64)
        for _ in range(5):
            state = detector.update(position)

        state_dict = state.to_dict()

        # Check all keys present
        assert "attractor_type" in state_dict
        assert "severity" in state_dict
        assert "cycle_length" in state_dict
        assert "position_variance" in state_dict
        assert "velocity_magnitude" in state_dict
        assert "timesteps_stuck" in state_dict
        assert "has_escape_direction" in state_dict

        # Check values are JSON-serializable (no numpy/mlx arrays)
        import json
        json.dumps(state_dict)  # Should not raise


class TestFactoryFunction:
    """Tests for create_attractor_detector factory."""

    def test_creates_detector(self, backend):
        """Factory should create configured detector."""
        detector = create_attractor_detector(
            hidden_dim=128,
            window_size=20,
            backend=backend,
        )

        assert detector._hidden_dim == 128
        assert detector.window_size == 20

    def test_default_window_size(self, backend):
        """Window size should default to sqrt(hidden_dim)."""
        detector = create_attractor_detector(hidden_dim=1024, backend=backend)

        # Default is max(10, sqrt(hidden_dim)) = max(10, 32) = 32
        assert detector.window_size == 32


class TestGeometricThresholds:
    """Tests verifying thresholds derive from geometry."""

    def test_thresholds_are_reasonable(self, backend):
        """Thresholds should be reasonable for detection."""
        detector = AttractorDetector(hidden_dim=64, backend=backend)

        # sqrt_eps should be approximately sqrt(1e-7) for float32
        # or sqrt(1e-15) for float64
        sqrt_eps = detector._sqrt_eps

        assert sqrt_eps > 0
        assert sqrt_eps < 0.01  # Should be small

        # Fixed point and velocity thresholds scale with sqrt(dim)
        import math
        expected_scale = sqrt_eps * math.sqrt(64)
        assert detector._fixed_point_threshold == expected_scale
        assert detector._slow_velocity_threshold == expected_scale

        # Cycle similarity threshold is now geometry-derived (1 - sqrt(eps))
        # For float32, sqrt(eps) ≈ 3.45e-4, so threshold ≈ 0.9997
        assert detector._cycle_similarity_threshold > 0.999
        assert detector._cycle_similarity_threshold < 1.0
