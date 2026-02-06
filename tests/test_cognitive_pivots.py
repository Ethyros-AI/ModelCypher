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

"""Tests for cognitive pivot detection (STARS method).

Tests mathematical properties:
- L2 distance computation between known vectors
- Spike detection with synthetic data
- Full trajectory analysis
- Edge cases (empty, single token)
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cognitive_pivots import (
    CognitivePivot,
    CognitivePivotDetector,
    ReasoningTrajectory,
    detect_cognitive_pivots,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestL2DistanceComputation:
    """Tests for L2 distance between consecutive hidden states."""

    def test_identical_states_zero_distance(self, backend) -> None:
        """Identical hidden states should have zero L2 distance."""
        detector = CognitivePivotDetector(backend=backend)
        state = {0: backend.array([1.0, 2.0, 3.0])}
        distances = detector.compute_l2_distances(state, state)
        assert abs(distances[0]) < 1e-6

    def test_known_distance(self, backend) -> None:
        """L2 distance between [0,0,0] and [3,4,0] should be 5.0."""
        detector = CognitivePivotDetector(backend=backend)
        current = {0: backend.array([3.0, 4.0, 0.0])}
        previous = {0: backend.array([0.0, 0.0, 0.0])}
        distances = detector.compute_l2_distances(current, previous)
        assert abs(distances[0] - 5.0) < 1e-5

    def test_first_token_zero_distance(self, backend) -> None:
        """First token (no previous) should return zero distances."""
        detector = CognitivePivotDetector(backend=backend)
        state = {0: backend.array([1.0, 2.0, 3.0])}
        distances = detector.compute_l2_distances(state, None)
        assert distances[0] == 0.0

    def test_multi_layer_distances(self, backend) -> None:
        """L2 distances should be computed independently per layer."""
        detector = CognitivePivotDetector(backend=backend)
        current = {
            0: backend.array([3.0, 4.0]),
            1: backend.array([0.0, 1.0]),
        }
        previous = {
            0: backend.array([0.0, 0.0]),
            1: backend.array([0.0, 0.0]),
        }
        distances = detector.compute_l2_distances(current, previous)
        assert abs(distances[0] - 5.0) < 1e-5
        assert abs(distances[1] - 1.0) < 1e-5


class TestSpikeDetection:
    """Tests for L2 spike detection."""

    def test_no_spikes_in_uniform_trajectory(self, backend) -> None:
        """Uniform distances should produce no spikes."""
        detector = CognitivePivotDetector(backend=backend)
        # All distances approximately equal -> no spikes
        all_distances = [
            {0: 1.0, 1: 1.0},
            {0: 1.01, 1: 0.99},
            {0: 1.02, 1: 1.01},
            {0: 0.99, 1: 1.0},
            {0: 1.0, 1: 1.01},
        ]
        spikes = detector.detect_spikes(all_distances)
        assert len(spikes) == 0

    def test_single_spike_detected(self, backend) -> None:
        """A large distance spike should be detected."""
        detector = CognitivePivotDetector(backend=backend, spike_threshold_k=2.0)
        # Normal distances ~1.0, then a spike at position 3
        all_distances = [
            {0: 1.0},
            {0: 1.01},
            {0: 0.99},
            {0: 10.0},  # Spike: ~9 std devs above mean
            {0: 1.02},
            {0: 0.98},
        ]
        spikes = detector.detect_spikes(all_distances)
        assert len(spikes) >= 1
        # The spike should be at token index 3
        spike_indices = [s[0] for s in spikes]
        assert 3 in spike_indices

    def test_spike_threshold_respected(self, backend) -> None:
        """Higher threshold should detect fewer spikes."""
        all_distances = [
            {0: 1.0},
            {0: 1.01},
            {0: 3.0},  # Moderate spike
            {0: 1.02},
            {0: 0.99},
        ]
        # Low threshold -> detect
        detector_low = CognitivePivotDetector(backend=backend, spike_threshold_k=1.0)
        spikes_low = detector_low.detect_spikes(all_distances)

        # High threshold -> might not detect
        detector_high = CognitivePivotDetector(backend=backend, spike_threshold_k=5.0)
        spikes_high = detector_high.detect_spikes(all_distances)

        assert len(spikes_low) >= len(spikes_high)

    def test_too_few_tokens_no_spikes(self, backend) -> None:
        """Single token cannot have spikes."""
        detector = CognitivePivotDetector(backend=backend)
        spikes = detector.detect_spikes([{0: 1.0}])
        assert len(spikes) == 0


class TestTrajectoryAnalysis:
    """Tests for full trajectory analysis."""

    def test_empty_trajectory(self, backend) -> None:
        """Empty trajectory should return empty result."""
        detector = CognitivePivotDetector(backend=backend)
        trajectory = detector.analyze_trajectory([], [])
        assert trajectory.pivots == ()
        assert trajectory.points == ()
        assert trajectory.mean_l2_distance == 0.0

    def test_trajectory_preserves_tokens(self, backend) -> None:
        """Trajectory should preserve token strings."""
        detector = CognitivePivotDetector(backend=backend)
        states = [
            {0: backend.array([1.0, 0.0])},
            {0: backend.array([1.0, 1.0])},
        ]
        tokens = ["hello", "world"]
        trajectory = detector.analyze_trajectory(states, tokens)
        assert trajectory.generated_tokens == ("hello", "world")

    def test_trajectory_with_known_pivot(self, backend) -> None:
        """Trajectory with deliberate spike should detect pivot."""
        detector = CognitivePivotDetector(backend=backend, spike_threshold_k=2.0)
        # Create 20 tokens: 19 with small moves, 1 with large move at position 10
        # Need enough normal tokens so the spike is clearly above mean + 2*std
        states = []
        for i in range(20):
            if i == 10:
                # Large jump - will create spike at position 10
                states.append({0: backend.array([100.0, 100.0])})
            elif i == 11:
                # Return from spike - will also be a spike
                states.append({0: backend.array([1.1, 0.0])})
            else:
                states.append({0: backend.array([float(i) * 0.1, 0.0])})
        tokens = [f"t{i}" for i in range(20)]
        trajectory = detector.analyze_trajectory(states, tokens)
        # Should detect at least one pivot (the spike at 10 or return at 11)
        assert len(trajectory.pivots) >= 1

    def test_trajectory_correctness_label(self, backend) -> None:
        """is_correct label should be preserved."""
        detector = CognitivePivotDetector(backend=backend)
        states = [{0: backend.array([1.0])}]
        trajectory = detector.analyze_trajectory(states, ["x"], is_correct=True)
        assert trajectory.is_correct is True

    def test_trajectory_mean_l2_positive(self, backend) -> None:
        """Non-trivial trajectory should have positive mean L2."""
        detector = CognitivePivotDetector(backend=backend)
        states = [
            {0: backend.array([0.0, 0.0])},
            {0: backend.array([1.0, 0.0])},
            {0: backend.array([1.0, 1.0])},
        ]
        trajectory = detector.analyze_trajectory(states, ["a", "b", "c"])
        assert trajectory.mean_l2_distance > 0.0


class TestConvenienceFunction:
    """Tests for detect_cognitive_pivots convenience function."""

    def test_returns_list_of_pivots(self, backend) -> None:
        """Convenience function should return a list."""
        states = [
            {0: backend.array([0.0, 0.0])},
            {0: backend.array([1.0, 0.0])},
        ]
        pivots = detect_cognitive_pivots(states, ["a", "b"], backend=backend)
        assert isinstance(pivots, list)

    def test_detects_spike_via_convenience(self, backend) -> None:
        """Convenience function should detect spikes."""
        # Need enough normal tokens so spike is clearly above mean + 2*std
        states = []
        for i in range(20):
            if i == 10:
                states.append({0: backend.array([50.0, 50.0])})
            elif i == 11:
                states.append({0: backend.array([1.1, 0.0])})
            else:
                states.append({0: backend.array([float(i) * 0.01, 0.0])})
        tokens = [f"t{i}" for i in range(20)]
        pivots = detect_cognitive_pivots(states, tokens, backend=backend)
        assert len(pivots) >= 1
        assert all(isinstance(p, CognitivePivot) for p in pivots)
