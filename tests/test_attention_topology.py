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

"""Tests for attention topology: distance conversion, Betti curves, barcode stats."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.attention_topology import (
    AttentionTopologySignal,
    attention_to_distance,
    barcode_statistics,
    betti_curve_statistics,
    compute_attention_topology,
    compute_betti_curve,
    _compute_head_diagram,
    _mean_pairwise_wasserstein,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    PersistenceDiagram,
    PersistencePoint,
)


class TestAttentionToDistance:
    """Tests for attention_to_distance()."""

    def test_identity_matrix(self) -> None:
        """Identity attention: token attends only to itself.

        A[i,i] = 1, A[i,j!=i] = 0.
        D[i,j] = 1 - max(0, 0) = 1.0 for i != j.
        D[i,i] = 0.
        """
        attn = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        dist = attention_to_distance(attn)
        assert dist[0][0] == 0.0
        assert dist[1][1] == 0.0
        assert dist[2][2] == 0.0
        assert dist[0][1] == 1.0
        assert dist[0][2] == 1.0
        assert dist[1][2] == 1.0
        # Symmetric
        assert dist[1][0] == dist[0][1]
        assert dist[2][0] == dist[0][2]
        assert dist[2][1] == dist[1][2]

    def test_uniform_attention(self) -> None:
        """Uniform attention: every token attends equally to all.

        A[i,j] = 1/n for all i,j.
        D[i,j] = 1 - max(1/n, 1/n) = 1 - 1/n.
        """
        n = 4
        a = 1.0 / n
        attn = [[a] * n for _ in range(n)]
        dist = attention_to_distance(attn)
        expected = 1.0 - a
        for i in range(n):
            assert dist[i][i] == 0.0
            for j in range(i + 1, n):
                assert abs(dist[i][j] - expected) < 1e-10
                assert dist[i][j] == dist[j][i]

    def test_asymmetric_attention(self) -> None:
        """Asymmetric A[i,j] != A[j,i] → takes max for symmetrization."""
        attn = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.1, 0.5]]
        dist = attention_to_distance(attn)
        # D[0,1] = 1 - max(A[0,1], A[1,0]) = 1 - max(0.3, 0.1) = 0.7
        assert abs(dist[0][1] - 0.7) < 1e-10
        # D[0,2] = 1 - max(0.2, 0.4) = 0.6
        assert abs(dist[0][2] - 0.6) < 1e-10
        # D[1,2] = 1 - max(0.1, 0.1) = 0.9
        assert abs(dist[1][2] - 0.9) < 1e-10
        # Symmetric
        assert dist[0][1] == dist[1][0]

    def test_full_attention_pair(self) -> None:
        """If two tokens attend fully to each other, distance = 0."""
        attn = [[0.0, 1.0], [1.0, 0.0]]
        dist = attention_to_distance(attn)
        assert dist[0][1] == 0.0
        assert dist[1][0] == 0.0

    def test_single_token(self) -> None:
        """Single token: 1x1 matrix → empty distance matrix."""
        dist = attention_to_distance([[1.0]])
        assert dist == [[0.0]]

    def test_values_in_unit_interval(self) -> None:
        """All distances should be in [0, 1]."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        dist = attention_to_distance(attn)
        for i in range(3):
            for j in range(3):
                assert 0.0 <= dist[i][j] <= 1.0


class TestBettiCurve:
    """Tests for compute_betti_curve()."""

    def test_empty_diagram(self) -> None:
        """Empty diagram → flat zero curve."""
        diag = PersistenceDiagram([])
        curve = compute_betti_curve(diag, dimension=1, n_steps=50)
        assert len(curve) == 50
        assert all(v == 0.0 for v in curve)

    def test_no_matching_dimension(self) -> None:
        """Diagram with only H0 features → H1 curve is zero."""
        diag = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        curve = compute_betti_curve(diag, dimension=1, n_steps=50)
        assert all(v == 0.0 for v in curve)

    def test_single_bar(self) -> None:
        """Single H1 bar [0.2, 0.8) → curve is 1 in that range, 0 outside."""
        diag = PersistenceDiagram([PersistencePoint(0.2, 0.8, 1)])
        curve = compute_betti_curve(diag, dimension=1, n_steps=100)
        # Curve should be 0 at t < 0.2, 1 for 0.2 <= t < 0.8, 0 at t >= 0.8
        # Since n_steps=100 and range is [0.2, 0.8], all internal points alive
        # First point is at birth, last at death — only 1 bar so peak = 1
        assert max(curve) == 1.0

    def test_two_overlapping_bars(self) -> None:
        """Two H1 bars with overlap → peak > 1."""
        diag = PersistenceDiagram([
            PersistencePoint(0.1, 0.6, 1),
            PersistencePoint(0.3, 0.8, 1),
        ])
        curve = compute_betti_curve(diag, dimension=1, n_steps=100)
        assert max(curve) == 2.0  # Both alive in [0.3, 0.6)

    def test_h0_curve(self) -> None:
        """Can compute H0 curve too."""
        diag = PersistenceDiagram([
            PersistencePoint(0.0, 0.5, 0),
            PersistencePoint(0.0, 1.0, 0),
        ])
        curve = compute_betti_curve(diag, dimension=0, n_steps=50)
        assert curve[0] == 2.0  # Both alive at start


class TestBettiCurveStatistics:
    """Tests for betti_curve_statistics()."""

    def test_zero_curve(self) -> None:
        """All-zero curve → all stats are 0."""
        w, c, p, s = betti_curve_statistics([0.0] * 50)
        assert w == 0.0
        assert c == 0.0
        assert p == 0.0
        assert s == 0.0

    def test_empty_curve(self) -> None:
        w, c, p, s = betti_curve_statistics([])
        assert w == 0.0

    def test_full_width(self) -> None:
        """All-nonzero curve → width = 1.0."""
        w, _, _, _ = betti_curve_statistics([1.0] * 10)
        assert w == 1.0

    def test_half_width(self) -> None:
        """Half nonzero → width = 0.5."""
        curve = [0.0] * 5 + [1.0] * 5
        w, _, _, _ = betti_curve_statistics(curve)
        assert w == 0.5

    def test_peak(self) -> None:
        """Peak is the maximum value."""
        curve = [0.0, 1.0, 3.0, 2.0, 0.0]
        _, _, p, _ = betti_curve_statistics(curve)
        assert p == 3.0

    def test_centroid_symmetry(self) -> None:
        """Symmetric curve → centroid at 0.5."""
        curve = [1.0, 1.0, 1.0, 1.0, 1.0]
        _, c, _, _ = betti_curve_statistics(curve)
        assert abs(c - 0.5) < 1e-10


class TestBarcodeStatistics:
    """Tests for barcode_statistics()."""

    def test_empty_diagram(self) -> None:
        total, max_p, ent, n0, n1 = barcode_statistics(PersistenceDiagram([]))
        assert total == 0.0
        assert max_p == 0.0
        assert ent == 0.0
        assert n0 == 0
        assert n1 == 0

    def test_single_h0_bar(self) -> None:
        diag = PersistenceDiagram([PersistencePoint(0.0, 0.5, 0)])
        total, max_p, ent, n0, n1 = barcode_statistics(diag)
        assert total == 0.5
        assert max_p == 0.5
        assert n0 == 1
        assert n1 == 0
        # Single bar → entropy = 0 (p=1, log(1)=0)
        assert ent == 0.0

    def test_two_equal_bars(self) -> None:
        diag = PersistenceDiagram([
            PersistencePoint(0.0, 1.0, 1),
            PersistencePoint(0.0, 1.0, 1),
        ])
        total, max_p, ent, n0, n1 = barcode_statistics(diag)
        assert total == 2.0
        assert max_p == 1.0
        assert n0 == 0
        assert n1 == 2
        # Two equal bars → max entropy = log(2)
        assert abs(ent - math.log(2)) < 1e-10

    def test_mixed_dimensions(self) -> None:
        diag = PersistenceDiagram([
            PersistencePoint(0.0, 1.0, 0),
            PersistencePoint(0.0, 0.5, 0),
            PersistencePoint(0.1, 0.4, 1),
        ])
        _, _, _, n0, n1 = barcode_statistics(diag)
        assert n0 == 2
        assert n1 == 1


class TestComputeHeadDiagram:
    """Tests for _compute_head_diagram()."""

    def test_identity_attention(self) -> None:
        """Identity attention: all distances = 1, no cycles expected at low filtration."""
        attn = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        diag = _compute_head_diagram(attn)
        assert isinstance(diag, PersistenceDiagram)
        # Should have at least H0 points (connected components)
        h0 = [p for p in diag.points if p.dimension == 0]
        assert len(h0) > 0

    def test_small_matrix(self) -> None:
        """2x2 attention → valid diagram."""
        attn = [[0.5, 0.5], [0.3, 0.7]]
        diag = _compute_head_diagram(attn)
        assert isinstance(diag, PersistenceDiagram)

    def test_single_token(self) -> None:
        """1x1 attention → empty diagram."""
        diag = _compute_head_diagram([[1.0]])
        assert diag.points == []


class TestMeanPairwiseWasserstein:
    """Tests for _mean_pairwise_wasserstein()."""

    def test_single_diagram(self) -> None:
        """Single diagram → 0 (no pairs)."""
        diag = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        assert _mean_pairwise_wasserstein([diag]) == 0.0

    def test_identical_diagrams(self) -> None:
        """Two identical diagrams → Wasserstein = 0."""
        diag = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        result = _mean_pairwise_wasserstein([diag, diag])
        assert result == 0.0

    def test_different_diagrams(self) -> None:
        """Two different diagrams → positive Wasserstein."""
        d1 = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        d2 = PersistenceDiagram([PersistencePoint(0.0, 0.5, 0)])
        result = _mean_pairwise_wasserstein([d1, d2])
        assert result > 0


class TestComputeAttentionTopology:
    """Tests for compute_attention_topology() end-to-end."""

    def test_empty_input(self) -> None:
        """No attention matrices → default signal."""
        signal = compute_attention_topology({})
        assert signal.betti_curve_width == 0.0
        assert signal.diagrams == {}

    def test_single_layer_single_head(self) -> None:
        """Single layer, single head → valid signal with all fields."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        assert 0 in signal.diagrams
        assert len(signal.diagrams[0]) == 1
        # Cross-head should be 0 (single head)
        assert signal.cross_head_wasserstein[0] == 0.0
        # No cross-layer (single layer)
        assert signal.cross_layer_wasserstein == []

    def test_two_layers(self) -> None:
        """Two layers → cross-layer Wasserstein computed."""
        attn1 = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        attn2 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        signal = compute_attention_topology({0: [attn1], 1: [attn2]})
        assert len(signal.cross_layer_wasserstein) == 1

    def test_two_heads(self) -> None:
        """Two heads in one layer → cross-head Wasserstein computed."""
        attn1 = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        attn2 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        signal = compute_attention_topology({0: [attn1, attn2]})
        assert len(signal.diagrams[0]) == 2
        # Cross-head should be computed (may or may not be > 0 depending on diagrams)
        assert 0 in signal.cross_head_wasserstein

    def test_expansion_ratio_passthrough(self) -> None:
        """Expansion ratio is passed through to signal."""
        signal = compute_attention_topology({}, expansion_ratio=1.23)
        assert signal.expansion_ratio == 1.23

    def test_feature_vector_length(self) -> None:
        """Feature vector has expected length."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        vec = signal.feature_vector()
        assert len(vec) == len(AttentionTopologySignal.feature_names())
        assert len(vec) == 12

    def test_feature_vector_all_finite(self) -> None:
        """All feature vector values should be finite."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        vec = signal.feature_vector()
        for v in vec:
            assert math.isfinite(v), f"Non-finite value in feature vector: {v}"
