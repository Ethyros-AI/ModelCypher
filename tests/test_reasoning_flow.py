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

"""Tests for reasoning flow geometry (Zhou et al., ICLR 2026)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.reasoning_flow import (
    FlowMetrics,
    ReasoningFlowAnalyzer,
    TokenCurvatureProfile,
    analyze_multilayer_flow,
    analyze_token_curvature,
)


@pytest.fixture
def backend():
    return get_default_backend()


@pytest.fixture
def analyzer(backend):
    return ReasoningFlowAnalyzer(backend)


class TestReasoningFlowAnalyzer:
    """Tests for ReasoningFlowAnalyzer."""

    def test_compute_velocities_basic(self, backend, analyzer):
        """Velocities are first differences of positions."""
        positions = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        velocities = analyzer.compute_velocities(positions)
        backend.eval(velocities)

        assert backend.shape(velocities) == (2, 2)
        # First velocity: [1, 0] - [0, 0] = [1, 0]
        assert abs(float(velocities[0, 0]) - 1.0) < 1e-5
        assert abs(float(velocities[0, 1]) - 0.0) < 1e-5
        # Second velocity: [1, 1] - [1, 0] = [0, 1]
        assert abs(float(velocities[1, 0]) - 0.0) < 1e-5
        assert abs(float(velocities[1, 1]) - 1.0) < 1e-5

    def test_compute_velocities_empty(self, backend, analyzer):
        """Empty or single-point trajectories have no velocities."""
        # Single point
        positions = backend.array([[1.0, 2.0]])
        velocities = analyzer.compute_velocities(positions)
        assert backend.shape(velocities)[0] == 0

    def test_compute_menger_curvature_straight_line(self, backend, analyzer):
        """Straight line has zero curvature."""
        # Points on a straight line
        positions = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        curvatures = analyzer.compute_menger_curvature(positions)
        backend.eval(curvatures)

        assert backend.shape(curvatures) == (2,)
        # Curvature should be ~0 for straight line
        assert float(backend.max(curvatures)) < 0.01

    def test_compute_menger_curvature_sharp_turn(self, backend, analyzer):
        """Sharp 90-degree turn has high curvature."""
        positions = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        curvatures = analyzer.compute_menger_curvature(positions)
        backend.eval(curvatures)

        assert backend.shape(curvatures) == (1,)
        # 90-degree turn should have significant curvature
        assert float(curvatures[0]) > 0.5

    def test_compute_arc_length(self, backend, analyzer):
        """Arc length is sum of step distances."""
        # Unit steps along x-axis
        positions = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        arc_length = analyzer.compute_arc_length(positions)
        assert abs(arc_length - 2.0) < 1e-5

    def test_analyze_flow_returns_metrics(self, backend, analyzer):
        """analyze_flow returns a FlowMetrics object with all fields."""
        positions = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ])
        metrics = analyzer.analyze_flow(positions)

        assert isinstance(metrics, FlowMetrics)
        assert metrics.arc_length > 0
        assert metrics.mean_speed > 0
        assert metrics.mean_curvature >= 0
        assert metrics.max_curvature >= 0
        assert 0 < metrics.smoothness <= 1
        assert 0 < metrics.directness <= 1

    def test_flow_smoothness_inversely_related_to_curvature(self, backend, analyzer):
        """Higher curvature means lower smoothness."""
        # Straight line (low curvature, high smoothness)
        straight = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        straight_metrics = analyzer.analyze_flow(straight)

        # Zigzag (high curvature, low smoothness)
        zigzag = backend.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
        ])
        zigzag_metrics = analyzer.analyze_flow(zigzag)

        assert straight_metrics.smoothness > zigzag_metrics.smoothness

    def test_flow_directness(self, backend, analyzer):
        """Directness measures how direct the path is."""
        # Direct path (high directness)
        direct = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        direct_metrics = analyzer.analyze_flow(direct)

        # Indirect path (low directness)
        indirect = backend.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [2.0, 0.0],
        ])
        indirect_metrics = analyzer.analyze_flow(indirect)

        assert direct_metrics.directness > indirect_metrics.directness


class TestMultilayerFlow:
    """Tests for analyze_multilayer_flow."""

    def test_analyze_multilayer_flow(self, backend):
        """Multilayer analysis returns one profile per layer."""
        layer_positions = {
            0: backend.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            1: backend.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]),
        }
        profiles = analyze_multilayer_flow(backend, layer_positions)

        assert len(profiles) == 2
        assert profiles[0].layer_idx == 0
        assert profiles[1].layer_idx == 1
        # Layer 1 has sharper turn, so higher curvature
        assert profiles[1].metrics.mean_curvature > profiles[0].metrics.mean_curvature


class TestTokenCurvature:
    """Tests for analyze_token_curvature (Zhou et al. methodology)."""

    def test_analyze_token_curvature_returns_profile(self, backend):
        """Token curvature returns a profile with per-token values."""
        # 5 tokens, 2 layers
        layer_positions = {
            0: backend.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],  # sharp turn at token 2
                [2.0, 1.0],
                [3.0, 1.0],
            ]),
            1: backend.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 2.0],  # even sharper turn at token 2
                [2.0, 2.0],
                [3.0, 2.0],
            ]),
        }
        profile = analyze_token_curvature(backend, layer_positions)

        assert isinstance(profile, TokenCurvatureProfile)
        assert profile.total_tokens == 5
        # Should have 3 curvature values (5 tokens - 2 = 3 triplets)
        assert len(profile.token_curvatures) == 3
        # Token indices should be 1, 2, 3 (interior points)
        assert profile.token_indices == [1, 2, 3]

    def test_token_curvature_finds_peak(self, backend):
        """Token curvature correctly identifies peak curvature region."""
        # Create a trajectory with a sharp turn followed by gentle motion
        layer_positions = {
            0: backend.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 5.0],  # Sharp turn here
                [2.0, 5.0],
                [3.0, 5.0],
                [4.0, 5.0],
                [5.0, 5.0],  # Straight section
            ]),
        }
        profile = analyze_token_curvature(backend, layer_positions)

        # Peak should be in the early part where the sharp turn is
        # (tokens 1 or 2 - both triplets include the sharp turn)
        assert profile.peak_token_idx in [1, 2]
        assert profile.peak_curvature > 0
        # The final tokens (straight line) should have zero curvature
        assert profile.token_curvatures[-1] < 0.01

    def test_token_curvature_insufficient_tokens(self, backend):
        """Token curvature handles insufficient tokens gracefully."""
        layer_positions = {
            0: backend.array([[0.0, 0.0], [1.0, 0.0]]),  # Only 2 tokens
        }
        profile = analyze_token_curvature(backend, layer_positions)

        assert profile.total_tokens == 2
        assert len(profile.token_curvatures) == 0
        assert profile.peak_token_idx == -1
