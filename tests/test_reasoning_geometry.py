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

"""Tests for unified reasoning geometry analysis.

Tests that the ReasoningGeometryAnalyzer correctly composes:
1. Topology signal (β₁ from persistent homology)
2. Cognitive pivot signal (L2 spike detection)
3. Linear probe signal (correctness prediction)

Uses synthetic data throughout - no real models needed.
"""

from __future__ import annotations

import math
import random

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.reasoning_geometry import (
    PivotSignal,
    ProbeSignal,
    ReasoningGeometryAnalyzer,
    ReasoningGeometryResult,
    TopologySignal,
    analyze_reasoning_geometry,
)


@pytest.fixture
def backend():
    return get_default_backend()


def _make_synthetic_trajectory(backend, n_tokens: int = 10, n_layers: int = 4, dim: int = 8, seed: int = 42):
    """Create a synthetic trajectory with smooth hidden states.

    Returns (layer_hidden_states_sequence, tokens).
    """
    rng = random.Random(seed)
    states_seq = []
    tokens = []

    for t in range(n_tokens):
        layer_states = {}
        for layer in range(n_layers):
            # Smooth trajectory: each layer adds structure
            base = [math.sin(t * 0.5 + layer * 0.3 + d * 0.1) + rng.gauss(0, 0.01)
                    for d in range(dim)]
            layer_states[layer] = backend.array(base)
        states_seq.append(layer_states)
        tokens.append(f"tok_{t}")

    return states_seq, tokens


def _make_trajectory_with_spike(backend, n_tokens: int = 10, n_layers: int = 4, dim: int = 8, spike_pos: int = 5):
    """Create trajectory with a deliberate L2 spike at spike_pos."""
    rng = random.Random(42)
    states_seq = []
    tokens = []

    for t in range(n_tokens):
        layer_states = {}
        for layer in range(n_layers):
            if t == spike_pos:
                # Large jump
                vals = [50.0 + rng.gauss(0, 0.1) for _ in range(dim)]
            else:
                vals = [float(t) * 0.1 + rng.gauss(0, 0.01) for _ in range(dim)]
            layer_states[layer] = backend.array(vals)
        states_seq.append(layer_states)
        tokens.append(f"tok_{t}")

    return states_seq, tokens


def _make_cycle_trajectory(backend, n_layers: int = 4, dim: int = 3):
    """Create trajectory with 4 tokens forming a square plus an outlier.

    A square's cycle dies at the diagonal distance. Adding a 5th distant point
    ensures max_filtration > diagonal, so the cycle persists in the diagram.

    The Vietoris-Rips filtration detects β₁ = 1 when:
    - Cycle born at side length (all 4 sides connect vertices)
    - Cycle dies at diagonal (triangle fills the cycle)
    - max_filtration > diagonal (so cycle is counted)
    """
    # Square with side 1, plus an outlier at distance 5
    p1 = [0.0, 0.0] + [0.0] * (dim - 2)
    p2 = [1.0, 0.0] + [0.0] * (dim - 2)
    p3 = [1.0, 1.0] + [0.0] * (dim - 2)
    p4 = [0.0, 1.0] + [0.0] * (dim - 2)
    p5 = [5.0, 5.0] + [0.0] * (dim - 2)  # Outlier ensures max_filtration > sqrt(2)

    states_seq = []
    for point in [p1, p2, p3, p4, p5]:
        layer_states = {}
        for layer in range(n_layers):
            layer_states[layer] = backend.array(point)
        states_seq.append(layer_states)

    return states_seq, ["a", "b", "c", "d", "e"]


def _make_line_trajectory(backend, n_tokens: int = 5, n_layers: int = 4, dim: int = 3):
    """Create trajectory with collinear tokens (β₁ should be 0)."""
    states_seq = []
    for t in range(n_tokens):
        layer_states = {}
        for layer in range(n_layers):
            vals = [float(t)] + [0.0] * (dim - 1)
            layer_states[layer] = backend.array(vals)
        states_seq.append(layer_states)

    return states_seq, [f"tok_{t}" for t in range(n_tokens)]


class TestEmptyAndEdgeCases:
    """Tests for edge cases in ReasoningGeometryAnalyzer."""

    def test_empty_trajectory(self, backend) -> None:
        """Empty trajectory should return empty result."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        result = analyzer.analyze_trajectory([], [])
        assert result.n_tokens == 0
        assert result.n_layers == 0
        assert result.pivots.pivot_count == 0
        assert result.topology.delta_beta1 == 0.0
        assert result.probe is None

    def test_single_token(self, backend) -> None:
        """Single token should return result with zero pivots."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states = [{0: backend.array([1.0, 2.0]), 1: backend.array([3.0, 4.0])}]
        result = analyzer.analyze_trajectory(states, ["hello"])
        assert result.n_tokens == 1
        assert result.n_layers == 2
        assert result.pivots.pivot_count == 0

    def test_two_tokens(self, backend) -> None:
        """Two tokens should work (minimum for topology)."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states = [
            {0: backend.array([0.0, 0.0])},
            {0: backend.array([1.0, 1.0])},
        ]
        result = analyzer.analyze_trajectory(states, ["a", "b"])
        assert result.n_tokens == 2
        assert result.n_layers == 1


class TestTopologySignal:
    """Tests for the β₁ topology signal."""

    def test_line_has_no_loops(self, backend) -> None:
        """Collinear tokens should have β₁ = 0 at all layers."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_line_trajectory(backend)
        result = analyzer.analyze_trajectory(states, tokens)

        # All β₁ should be 0 for collinear points
        for beta1 in result.topology.beta1_by_layer:
            assert beta1 == 0

    def test_cycle_has_loop(self, backend) -> None:
        """Square + outlier should have β₁ >= 1 (cycle from square persists)."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_cycle_trajectory(backend, n_layers=4, dim=3)
        result = analyzer.analyze_trajectory(states, tokens)

        # At least one layer should show β₁ > 0
        total_beta1 = sum(result.topology.beta1_by_layer)
        assert total_beta1 >= 1

    def test_delta_beta1_computed(self, backend) -> None:
        """Δβ₁ should be the difference between late and early layers."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=5, n_layers=6)
        result = analyzer.analyze_trajectory(states, tokens)

        # Δβ₁ should be a finite number
        assert math.isfinite(result.topology.delta_beta1)

        # Verify manually: early = first third, late = last third
        n_layers = len(result.topology.beta1_by_layer)
        third = max(1, n_layers // 3)
        early = result.topology.beta1_by_layer[:third]
        late = result.topology.beta1_by_layer[-third:]
        expected = sum(late) / len(late) - sum(early) / len(early)
        assert abs(result.topology.delta_beta1 - expected) < 1e-10

    def test_layer_indices_preserved(self, backend) -> None:
        """Layer indices should match input layers."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        # Use non-contiguous layer indices
        states = [
            {0: backend.array([1.0, 0.0]), 5: backend.array([0.0, 1.0])},
            {0: backend.array([0.0, 1.0]), 5: backend.array([1.0, 0.0])},
            {0: backend.array([1.0, 1.0]), 5: backend.array([1.0, 1.0])},
        ]
        result = analyzer.analyze_trajectory(states, ["a", "b", "c"])
        assert result.topology.layer_indices == (0, 5)

    def test_persistence_entropy_per_layer(self, backend) -> None:
        """Each layer should have a persistence entropy value."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_layers=3)
        result = analyzer.analyze_trajectory(states, tokens)
        assert len(result.topology.persistence_entropy_by_layer) == 3


class TestPivotSignal:
    """Tests for the cognitive pivot signal."""

    def test_smooth_trajectory_few_pivots(self, backend) -> None:
        """Smooth trajectory should have few or no pivots."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=10)
        result = analyzer.analyze_trajectory(states, tokens)
        # Smooth trajectory - pivots should be rare
        # (not asserting zero because noise can occasionally trigger)
        assert result.pivots.pivot_count < len(tokens)

    def test_spike_detected_as_pivot(self, backend) -> None:
        """Trajectory with deliberate spike should detect pivots."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_trajectory_with_spike(backend, n_tokens=10, spike_pos=5)
        result = analyzer.analyze_trajectory(states, tokens)
        assert result.pivots.pivot_count >= 1

    def test_pivot_statistics(self, backend) -> None:
        """Pivot signal should have valid statistics."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=5)
        result = analyzer.analyze_trajectory(states, tokens)
        assert result.pivots.mean_l2_distance >= 0.0
        assert result.pivots.std_l2_distance >= 0.0


class TestProbeSignal:
    """Tests for the linear probe signal via batch analysis."""

    def test_no_labels_no_probe(self, backend) -> None:
        """Unlabeled trajectories should not produce a probe signal."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=5, n_layers=2, dim=4)
        trajectories = [(states, tokens, None)]
        results, probe = analyzer.analyze_batch(trajectories)
        assert probe is None
        assert len(results) == 1
        assert results[0].probe is None

    def test_labeled_batch_produces_probe(self, backend) -> None:
        """Labeled trajectories should produce per-layer probe signal."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        rng = random.Random(42)

        # Create trajectories where correct ones have positive bias
        # and incorrect ones have negative bias
        trajectories = []
        for i in range(20):
            is_correct = i < 10
            bias = 2.0 if is_correct else -2.0
            states_seq = []
            for t in range(5):
                layer_states = {}
                for layer in range(3):
                    vals = [bias + rng.gauss(0, 0.3) for _ in range(4)]
                    layer_states[layer] = backend.array(vals)
                states_seq.append(layer_states)
            tokens = [f"tok_{t}" for t in range(5)]
            trajectories.append((states_seq, tokens, is_correct))

        results, probe = analyzer.analyze_batch(trajectories)
        assert probe is not None
        assert isinstance(probe, ProbeSignal)
        assert len(probe.auroc_by_layer) == 3
        assert probe.best_auroc >= 0.5
        # With strong separation, AUROC should be good
        assert probe.best_auroc > 0.6

    def test_all_correct_no_probe(self, backend) -> None:
        """All-correct batch should not produce probe (no negative examples)."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=3, n_layers=2, dim=4)
        trajectories = [(states, tokens, True), (states, tokens, True)]
        _, probe = analyzer.analyze_batch(trajectories)
        assert probe is None


class TestReasoningGeometryResult:
    """Tests for the result data class."""

    def test_as_dict(self, backend) -> None:
        """as_dict should produce a serializable dictionary."""
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=5, n_layers=3)
        result = analyzer.analyze_trajectory(states, tokens)

        d = result.as_dict()
        assert isinstance(d, dict)
        assert "topology" in d
        assert "pivots" in d
        assert "n_layers" in d
        assert "n_tokens" in d
        assert d["n_layers"] == 3
        assert d["n_tokens"] == 5
        # probe should not be in dict when None
        assert "probe" not in d

    def test_as_dict_with_probe(self, backend) -> None:
        """as_dict with probe should include probe section."""
        result = ReasoningGeometryResult(
            topology=TopologySignal(
                betti_by_layer=({0: 1, 1: 0},),
                beta1_by_layer=(0,),
                delta_beta1=0.0,
                mean_beta1=0.0,
                persistence_entropy_by_layer=(0.5,),
                layer_indices=(0,),
            ),
            probe=ProbeSignal(
                auroc_by_layer=(0.7,),
                best_layer=0,
                best_auroc=0.7,
                separation_by_layer=(1.5,),
                layer_indices=(0,),
            ),
            pivots=PivotSignal(
                pivot_count=2,
                mean_l2_distance=1.0,
                std_l2_distance=0.5,
                pivots=(),
            ),
            n_layers=1,
            n_tokens=5,
        )
        d = result.as_dict()
        assert "probe" in d
        assert d["probe"]["best_auroc"] == 0.7


class TestConvenienceFunction:
    """Tests for the analyze_reasoning_geometry convenience function."""

    def test_convenience_matches_class(self, backend) -> None:
        """Convenience function should produce same result as class."""
        states, tokens = _make_synthetic_trajectory(backend, n_tokens=5, n_layers=2, dim=4)

        result1 = analyze_reasoning_geometry(states, tokens, backend=backend)
        analyzer = ReasoningGeometryAnalyzer(backend=backend)
        result2 = analyzer.analyze_trajectory(states, tokens)

        assert result1.n_tokens == result2.n_tokens
        assert result1.n_layers == result2.n_layers
        assert result1.pivots.pivot_count == result2.pivots.pivot_count
        assert result1.topology.delta_beta1 == result2.topology.delta_beta1


class TestLazyLoading:
    """Tests that modules are properly registered for lazy loading."""

    def test_import_reasoning_geometry(self) -> None:
        """ReasoningGeometryAnalyzer should be importable from geometry package."""
        from modelcypher.core.domain.geometry import ReasoningGeometryAnalyzer
        assert ReasoningGeometryAnalyzer is not None

    def test_import_cognitive_pivots(self) -> None:
        """CognitivePivotDetector should be importable from geometry package."""
        from modelcypher.core.domain.geometry import CognitivePivotDetector
        assert CognitivePivotDetector is not None

    def test_import_linear_probe(self) -> None:
        """CorrectnessProbe should be importable from geometry package."""
        from modelcypher.core.domain.geometry import CorrectnessProbe
        assert CorrectnessProbe is not None
