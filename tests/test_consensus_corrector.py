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

"""Unit tests for consensus correction (requires MLX)."""

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.consensus_corrector import ConsensusCorrector
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestConsensusCorrector:
    """Tests for consensus correction computation."""

    def test_correction_moves_toward_consensus(self):
        """Correction should move target position toward consensus."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        # Simple 2D example
        # Target is at (0, 0), consensus stress implies position at (5, 5)
        target_position = backend.array([0.0, 0.0])

        # Set up anchors at corners of a square
        target_anchors = {
            "anchor_0": backend.array([0.0, 0.0]),
            "anchor_1": backend.array([10.0, 0.0]),
            "anchor_2": backend.array([0.0, 10.0]),
            "anchor_3": backend.array([10.0, 10.0]),
        }

        # Consensus stress: distances from (5, 5) to each anchor
        # Distance to (0,0): sqrt(50) ≈ 7.07
        # Distance to (10,0): sqrt(25+25) = sqrt(50) ≈ 7.07
        # Distance to (0,10): sqrt(50) ≈ 7.07
        # Distance to (10,10): sqrt(50) ≈ 7.07
        import math
        d = math.sqrt(50)
        consensus_stress = backend.array([d, d, d, d])

        # Simple activations and weights for lstsq
        target_activations = backend.array([[1.0, 0.0], [0.0, 1.0]])
        target_weights = backend.array([[1.0, 0.0], [0.0, 1.0]])

        result = corrector.compute_correction_delta(
            target_position=target_position,
            consensus_stress=consensus_stress,
            target_anchors=target_anchors,
            target_activations=target_activations,
            target_weights=target_weights,
        )

        # Activation delta should point toward (5, 5)
        delta_x = float(result.activation_delta[0])
        delta_y = float(result.activation_delta[1])

        eps = _div_eps()
        assert delta_x > 0  # Should move in positive x direction
        assert delta_y > 0  # Should move in positive y direction

    def test_apply_correction(self):
        """Applying correction should add delta to weights."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        target_weights = backend.array([[1.0, 2.0], [3.0, 4.0]])
        weight_delta = backend.array([[0.1, 0.2], [0.3, 0.4]])

        corrected = corrector.apply_correction(target_weights, weight_delta)

        eps = _div_eps()
        assert abs(float(corrected[0, 0]) - 1.1) < eps
        assert abs(float(corrected[0, 1]) - 2.2) < eps
        assert abs(float(corrected[1, 0]) - 3.3) < eps
        assert abs(float(corrected[1, 1]) - 4.4) < eps

    def test_zero_correction_when_at_consensus(self):
        """No correction needed when target is already at consensus."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        # Target is already at (5, 5)
        target_position = backend.array([5.0, 5.0])

        target_anchors = {
            "anchor_0": backend.array([0.0, 0.0]),
            "anchor_1": backend.array([10.0, 0.0]),
            "anchor_2": backend.array([0.0, 10.0]),
            "anchor_3": backend.array([10.0, 10.0]),
        }

        # Consensus stress matches target position
        import math
        d = math.sqrt(50)
        consensus_stress = backend.array([d, d, d, d])

        target_activations = backend.array([[1.0, 0.0], [0.0, 1.0]])
        target_weights = backend.array([[1.0, 0.0], [0.0, 1.0]])

        result = corrector.compute_correction_delta(
            target_position=target_position,
            consensus_stress=consensus_stress,
            target_anchors=target_anchors,
            target_activations=target_activations,
            target_weights=target_weights,
        )

        # Activation delta should be near zero
        delta_norm = float(backend.sqrt(backend.sum(result.activation_delta ** 2)))
        assert delta_norm < 1.0  # Small correction

    def test_stress_from_position(self):
        """Stress computation should return correct distances."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        position = backend.array([5.0, 5.0])
        anchor_positions = {
            "anchor_0": backend.array([0.0, 0.0]),
            "anchor_1": backend.array([10.0, 10.0]),
        }

        stress = corrector._compute_stress_from_position(position, anchor_positions)

        import math
        expected_d0 = math.sqrt(50)  # Distance from (5,5) to (0,0)
        expected_d1 = math.sqrt(50)  # Distance from (5,5) to (10,10)

        eps = _div_eps()
        assert abs(float(stress[0]) - expected_d0) < 0.1
        assert abs(float(stress[1]) - expected_d1) < 0.1

    def test_multilateration_accuracy(self):
        """Multilateration should recover position from distances."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        # Known position
        true_position = backend.array([3.0, 7.0])

        # Anchors
        anchor_positions = {
            "a0": backend.array([0.0, 0.0]),
            "a1": backend.array([10.0, 0.0]),
            "a2": backend.array([0.0, 10.0]),
            "a3": backend.array([10.0, 10.0]),
        }

        # Compute exact distances from true position
        import math
        stress = [
            math.sqrt(3**2 + 7**2),      # to (0,0)
            math.sqrt(7**2 + 7**2),      # to (10,0)
            math.sqrt(3**2 + 3**2),      # to (0,10)
            math.sqrt(7**2 + 3**2),      # to (10,10)
        ]
        stress_arr = backend.array(stress)

        # Solve for position
        recovered = corrector._solve_position_from_stress(stress_arr, anchor_positions)

        # Should recover approximately the true position
        dist = float(backend.sqrt(backend.sum((recovered - true_position) ** 2)))
        assert dist < 1.0  # Should be close


class TestCorrectionVsAddition:
    """Tests verifying correction differs from addition (no null-space)."""

    def test_correction_changes_behavior(self):
        """Correction should change output, unlike null-space addition."""
        backend = get_default_backend()
        corrector = ConsensusCorrector(backend)

        # Simple weights and activations
        target_weights = backend.array([[1.0, 0.0], [0.0, 1.0]])
        target_activations = backend.array([[1.0, 0.5], [0.5, 1.0]])

        # Target at origin, move to (5, 5)
        target_position = backend.array([0.0, 0.0])
        target_anchors = {
            "a0": backend.array([0.0, 0.0]),
            "a1": backend.array([10.0, 0.0]),
            "a2": backend.array([0.0, 10.0]),
        }

        import math
        d = math.sqrt(50)
        consensus_stress = backend.array([d, d, d])

        result = corrector.compute_correction_delta(
            target_position=target_position,
            consensus_stress=consensus_stress,
            target_anchors=target_anchors,
            target_activations=target_activations,
            target_weights=target_weights,
        )

        # Weight delta should be non-zero
        delta_norm = float(backend.sqrt(backend.sum(result.weight_delta ** 2)))
        assert delta_norm > 0.01  # Non-trivial correction

        # Apply correction
        corrected_weights = corrector.apply_correction(target_weights, result.weight_delta)

        # Output on same activations should change
        original_output = backend.matmul(target_activations, backend.transpose(target_weights))
        corrected_output = backend.matmul(target_activations, backend.transpose(corrected_weights))

        output_diff = float(backend.sqrt(backend.sum((corrected_output - original_output) ** 2)))
        assert output_diff > 0.01  # Behavior changed
