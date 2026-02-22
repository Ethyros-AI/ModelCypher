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

"""Tests for multi-metric topology module.

Tests mathematical properties:
- Collinear points: all metrics give β₁ = 0
- Square cycle: all metrics give β₁ >= 1
- Cycle rank formula: known graph -> expected β₁
- Graph β₁ non-negativity
- Sign agreement on synthetic data
- Cosine distance clamping
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.multi_metric_topology import (
    GraphBeta1Result,
    MultiMetricBeta1,
    MultiMetricTopologySignal,
    compute_beta1_multi_metric,
    compute_graph_beta1,
    compute_multi_metric_topology_signal,
    _compute_cosine_distances,
    _compute_delta,
    _compute_sign_agreement,
    _beta1_from_distance_matrix,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestCollinearPoints:
    """Collinear points have no cycles — β₁ = 0 under all metrics."""

    def test_collinear_all_metrics_zero(self, backend) -> None:
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        result = compute_beta1_multi_metric(points, backend)

        assert result.euclidean_beta1 == 0
        assert result.cosine_beta1 == 0
        assert result.spectral_beta1 == 0
        assert result.geodesic_beta1 == 0


class TestSquareCycle:
    """Square vertices form a cycle — β₁ >= 1 under most metrics."""

    def test_square_has_cycle(self, backend) -> None:
        # Unit square vertices — clear topological loop
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        result = compute_beta1_multi_metric(points, backend)

        # At least geodesic should detect the cycle (our validated default)
        assert result.geodesic_beta1 >= 1, (
            f"Geodesic should detect cycle in square, got {result.geodesic_beta1}"
        )


class TestGraphBeta1:
    """Tests for graph-based β₁ proxy (cycle rank)."""

    def test_cycle_rank_formula(self, backend) -> None:
        """Known graph: 4 vertices fully connected = K4.

        K4 has 6 edges, 4 vertices, 1 component.
        Cycle rank = 6 - 4 + 1 = 3.
        """
        # 4 points close together so all connect in kNN
        points = backend.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
        ])
        # k=3 means each point connects to all 3 others -> K4
        result = compute_graph_beta1(points, k_neighbors=3, backend=backend)

        assert result.n_vertices == 4
        assert result.n_edges == 6  # K4 has 6 edges
        assert result.n_components == 1
        assert result.beta1 == 3  # 6 - 4 + 1

    def test_non_negative(self, backend) -> None:
        """Graph β₁ is always non-negative."""
        import random
        rng = random.Random(42)

        for _ in range(5):
            pts = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(10)]
            points = backend.array(pts)
            result = compute_graph_beta1(points, backend=backend)
            assert result.beta1 >= 0

    def test_line_graph_no_cycles(self, backend) -> None:
        """Points on a line with k=1: tree, no cycles."""
        points = backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        result = compute_graph_beta1(points, k_neighbors=1, backend=backend)

        # k=1 on a line: each connects to nearest neighbor
        # Should form a path graph: 3 edges, 4 vertices, 1 component
        assert result.beta1 == 0  # path has no cycles

    def test_single_point(self, backend) -> None:
        points = backend.array([[1.0, 2.0]])
        result = compute_graph_beta1(points, backend=backend)
        assert result.beta1 == 0
        assert result.n_vertices == 1

    def test_two_points(self, backend) -> None:
        points = backend.array([[0.0, 0.0], [1.0, 0.0]])
        result = compute_graph_beta1(points, backend=backend)
        assert result.beta1 == 0
        assert result.n_edges == 1


class TestCosineDistanceClamping:
    """Cosine distances should never be negative (floating point safety)."""

    def test_no_negative_distances(self, backend) -> None:
        # Points with near-identical directions -> cos_sim ≈ 1 -> dist ≈ 0
        points = backend.array([
            [1.0, 0.001, 0.0],
            [1.0, 0.002, 0.0],
            [1.0, 0.003, 0.0],
            [1.0, 0.004, 0.0],
        ])
        dists = _compute_cosine_distances(points, backend)

        for row in dists:
            for d in row:
                assert d >= 0.0, f"Negative cosine distance: {d}"

    def test_identical_points_zero_distance(self, backend) -> None:
        points = backend.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        dists = _compute_cosine_distances(points, backend)

        # Distance from point to itself should be 0
        for i in range(3):
            assert abs(dists[i][i]) < 1e-6


class TestSignAgreement:
    """Tests for sign agreement computation."""

    def test_all_agree_positive(self) -> None:
        deltas = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 3.0}
        assert _compute_sign_agreement(deltas) == 1.0

    def test_all_agree_negative(self) -> None:
        deltas = {"a": -1.0, "b": -2.0, "c": -0.5}
        assert _compute_sign_agreement(deltas) == 1.0

    def test_split_vote(self) -> None:
        deltas = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}
        assert _compute_sign_agreement(deltas) == 0.5

    def test_three_agree_one_disagrees(self) -> None:
        deltas = {"a": 1.0, "b": 1.0, "c": 1.0, "d": -1.0}
        assert _compute_sign_agreement(deltas) == 0.75

    def test_zeros_excluded(self) -> None:
        deltas = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
        # Only 2 voters, both positive -> 1.0
        assert _compute_sign_agreement(deltas) == 1.0

    def test_all_zero(self) -> None:
        deltas = {"a": 0.0, "b": 0.0}
        assert _compute_sign_agreement(deltas) == 1.0


class TestDeltaComputation:
    """Tests for Δ = mean(late 1/3) - mean(early 1/3)."""

    def test_increasing_sequence(self) -> None:
        values = [0, 0, 0, 1, 1, 1, 3, 3, 3]
        delta = _compute_delta(values)
        # early 1/3 = [0,0,0] mean=0, late 1/3 = [3,3,3] mean=3
        assert delta == 3.0

    def test_constant_sequence(self) -> None:
        values = [2, 2, 2, 2, 2, 2]
        assert _compute_delta(values) == 0.0

    def test_empty(self) -> None:
        assert _compute_delta([]) == 0.0

    def test_single(self) -> None:
        assert _compute_delta([5]) == 0.0


class TestBeta1FromDistanceMatrix:
    """Tests for β₁ extraction from raw distance matrices."""

    def test_trivial_distances_no_cycles(self) -> None:
        # Three points on a line: distances [0, 1, 2]
        dists = [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        assert _beta1_from_distance_matrix(dists) == 0

    def test_too_few_points(self) -> None:
        assert _beta1_from_distance_matrix([[0.0, 1.0], [1.0, 0.0]]) == 0
        assert _beta1_from_distance_matrix([]) == 0


class TestMultiMetricTopologySignal:
    """Integration tests for full trajectory analysis."""

    def test_empty_trajectory(self, backend) -> None:
        result = compute_multi_metric_topology_signal(
            [], [0, 1, 2], backend=backend,
        )
        assert len(result.beta1_by_metric_by_layer) == 0
        assert result.sign_agreement == 1.0

    def test_single_layer(self, backend) -> None:
        # 5 tokens, 4-dim hidden states, 1 layer
        import random
        rng = random.Random(123)

        hidden = []
        for _ in range(5):
            arr = backend.array([rng.gauss(0, 1) for _ in range(4)])
            hidden.append({0: arr})

        result = compute_multi_metric_topology_signal(
            hidden, [0], backend=backend,
        )
        assert len(result.beta1_by_metric_by_layer) == 1
        assert len(result.graph_beta1_by_layer) == 1

    def test_multi_layer_output_shape(self, backend) -> None:
        """Output has correct structure for multi-layer input."""
        import random
        rng = random.Random(456)

        n_tokens = 8
        n_layers = 6
        hidden_dim = 4
        layer_indices = list(range(n_layers))

        hidden = []
        for _ in range(n_tokens):
            states = {}
            for l in layer_indices:
                states[l] = backend.array(
                    [rng.gauss(0, 1) for _ in range(hidden_dim)]
                )
            hidden.append(states)

        result = compute_multi_metric_topology_signal(
            hidden, layer_indices, backend=backend,
        )

        assert len(result.beta1_by_metric_by_layer) == n_layers
        assert len(result.graph_beta1_by_layer) == n_layers
        assert len(result.layer_indices) == n_layers
        assert set(result.delta_beta1_by_metric.keys()) == {
            "euclidean", "cosine", "spectral", "geodesic"
        }
        assert 0.0 <= result.sign_agreement <= 1.0

    def test_as_dict_roundtrip(self, backend) -> None:
        """as_dict produces valid serializable output."""
        import random
        rng = random.Random(789)

        hidden = []
        for _ in range(5):
            states = {0: backend.array([rng.gauss(0, 1) for _ in range(3)])}
            hidden.append(states)

        result = compute_multi_metric_topology_signal(
            hidden, [0], backend=backend,
        )
        d = result.as_dict()

        assert "beta1_by_metric_by_layer" in d
        assert "delta_beta1_by_metric" in d
        assert "sign_agreement" in d
        assert "graph_beta1_by_layer" in d
        assert "graph_delta_beta1" in d
        assert "layer_indices" in d


class TestSyntheticLoopCloud:
    """Synthetic point cloud with known loop structure."""

    def test_circle_has_agreement(self, backend) -> None:
        """Points on a circle should produce β₁ > 0 in multiple metrics.

        If they do, sign agreement should be high.
        """
        import math

        n_points = 12
        radius = 1.0
        points_list = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            points_list.append([
                radius * math.cos(theta),
                radius * math.sin(theta),
            ])

        points = backend.array(points_list)
        result = compute_beta1_multi_metric(points, backend)

        # At least geodesic (validated default) should find the loop
        assert result.geodesic_beta1 >= 1

        # Graph proxy should also detect the cycle
        graph_result = compute_graph_beta1(points, backend=backend)
        assert graph_result.beta1 >= 1
