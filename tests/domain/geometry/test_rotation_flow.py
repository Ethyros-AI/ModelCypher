# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for rotation flow analysis."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.generalized_procrustes import (
    RotationFlowAnalyzer,
    compute_rotation_flow,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestRotationFlow:
    """Tests for rotation flow computation."""

    def test_identity_rotations_zero_flow(self, backend):
        """Identity rotations should have zero flow."""
        b = backend
        d = 4

        rotations = [b.eye(d) for _ in range(5)]

        result = compute_rotation_flow(rotations, backend=b)

        # All speeds should be near zero
        speeds = b.tolist(result.rotation_speeds)
        for speed in speeds:
            assert speed < 1e-5

    def test_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        rotations = [b.eye(3), b.eye(3), b.eye(3)]

        result = compute_rotation_flow(rotations, backend=b)

        assert hasattr(result, "layer_indices")
        assert hasattr(result, "rotation_speeds")
        assert hasattr(result, "rotation_accelerations")
        assert hasattr(result, "cumulative_rotation")
        assert hasattr(result, "lie_algebra_norms")
        assert hasattr(result, "max_jump_layer")
        assert hasattr(result, "max_jump_magnitude")

    def test_single_rotation(self, backend):
        """Single rotation should return empty flow arrays."""
        b = backend
        rotations = [b.eye(3)]

        result = compute_rotation_flow(rotations, backend=b)

        assert result.rotation_speeds.shape[0] == 0
        assert result.rotation_accelerations.shape[0] == 0

    def test_two_rotations(self, backend):
        """Two rotations give one speed, no acceleration."""
        b = backend
        rotations = [b.eye(3), b.eye(3)]

        result = compute_rotation_flow(rotations, backend=b)

        assert result.rotation_speeds.shape[0] == 1
        assert result.rotation_accelerations.shape[0] == 0

    def test_varying_rotations(self, backend):
        """Different rotations should produce non-zero flow."""
        b = backend

        # Create rotation matrices
        theta1 = 0.1
        theta2 = 0.2
        theta3 = 0.5

        def rotation_z(theta):
            c, s = b.cos(b.array([theta])), b.sin(b.array([theta]))
            c, s = float(b.to_scalar(c)), float(b.to_scalar(s))
            return b.array([
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ])

        rotations = [rotation_z(theta1), rotation_z(theta2), rotation_z(theta3)]

        result = compute_rotation_flow(rotations, backend=b)

        # Should have non-zero speeds (rotations differ)
        speeds = b.tolist(result.rotation_speeds)
        assert any(s > 0 for s in speeds)

    def test_cumulative_rotation_monotonic(self, backend):
        """Cumulative rotation should be monotonically increasing."""
        b = backend

        rotations = [b.eye(3) for _ in range(5)]
        result = compute_rotation_flow(rotations, backend=b)

        cumulative = b.tolist(result.cumulative_rotation)
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]

    def test_layer_indices_custom(self, backend):
        """Custom layer indices should be preserved."""
        b = backend
        rotations = [b.eye(3), b.eye(3), b.eye(3)]
        layer_indices = [0, 5, 10]

        result = compute_rotation_flow(rotations, layer_indices, backend=b)

        assert result.layer_indices == tuple(layer_indices)


class TestRotationFlowAnalyzer:
    """Tests for RotationFlowAnalyzer class."""

    def test_analyze_layer_alignments_matching_layers(self, backend):
        """Test alignment analysis with matching layer activations."""
        b = backend
        analyzer = RotationFlowAnalyzer(b)

        n_samples = 10
        d = 5

        # Create synthetic activations
        source_acts = {
            0: b.array([[float(i + j) for j in range(d)] for i in range(n_samples)]),
            1: b.array([[float(i + j + 1) for j in range(d)] for i in range(n_samples)]),
            2: b.array([[float(i + j + 2) for j in range(d)] for i in range(n_samples)]),
        }
        target_acts = {
            0: b.array([[float(i + j) for j in range(d)] for i in range(n_samples)]),
            1: b.array([[float(i + j + 1) for j in range(d)] for i in range(n_samples)]),
            2: b.array([[float(i + j + 2) for j in range(d)] for i in range(n_samples)]),
        }

        result = analyzer.analyze_layer_alignments(source_acts, target_acts)

        assert result is not None
        assert len(result.layer_indices) == 3

    def test_analyze_layer_alignments_no_common_layers(self, backend):
        """Test with no common layers returns None."""
        b = backend
        analyzer = RotationFlowAnalyzer(b)

        source_acts = {0: b.array([[1.0, 2.0]])}
        target_acts = {1: b.array([[1.0, 2.0]])}

        result = analyzer.analyze_layer_alignments(source_acts, target_acts)

        assert result is None
