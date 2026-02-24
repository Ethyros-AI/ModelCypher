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

"""Tests for attention topology: distance conversion, barcode stats, Wasserstein."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.attention_topology import (
    AttentionTopologySignal,
    _compute_head_diagram,
    _mean_pairwise_wasserstein,
    attention_to_distance,
    barcode_statistics,
    compute_attention_topology,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    PersistenceDiagram,
    PersistencePoint,
)


class TestAttentionToDistance:
    """Tests for attention_to_distance().

    D[i,j] = 1 - max(A[i,j], A[j,i]) — first-principles derivation.
    """

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
        """Asymmetric A[i,j] != A[j,i] -> takes max for symmetrization."""
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
        """Single token: 1x1 matrix -> 1x1 distance matrix."""
        dist = attention_to_distance([[1.0]])
        assert dist == [[0.0]]

    def test_values_in_unit_interval(self) -> None:
        """All distances should be in [0, 1]."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        dist = attention_to_distance(attn)
        for i in range(3):
            for j in range(3):
                assert 0.0 <= dist[i][j] <= 1.0


class TestBarcodeStatistics:
    """Tests for barcode_statistics() — direct from birth/death pairs."""

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
        # Single bar -> entropy = 0 (p=1, log(1)=0)
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
        # Two equal bars -> max entropy = log(2)
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
        """Identity attention: all distances = 1, produces valid diagram."""
        attn = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        diag = _compute_head_diagram(attn)
        assert isinstance(diag, PersistenceDiagram)
        # Should have at least H0 points (connected components)
        h0 = [p for p in diag.points if p.dimension == 0]
        assert len(h0) > 0

    def test_small_matrix(self) -> None:
        """2x2 attention -> valid diagram."""
        attn = [[0.5, 0.5], [0.3, 0.7]]
        diag = _compute_head_diagram(attn)
        assert isinstance(diag, PersistenceDiagram)

    def test_single_token(self) -> None:
        """1x1 attention -> empty diagram."""
        diag = _compute_head_diagram([[1.0]])
        assert diag.points == []


class TestMeanPairwiseWasserstein:
    """Tests for _mean_pairwise_wasserstein()."""

    def test_single_diagram(self) -> None:
        """Single diagram -> 0 (no pairs)."""
        diag = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        assert _mean_pairwise_wasserstein([diag]) == 0.0

    def test_identical_diagrams(self) -> None:
        """Two identical diagrams -> Wasserstein = 0."""
        diag = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        result = _mean_pairwise_wasserstein([diag, diag])
        assert result == 0.0

    def test_different_diagrams(self) -> None:
        """Two different diagrams -> positive Wasserstein."""
        d1 = PersistenceDiagram([PersistencePoint(0.0, 1.0, 0)])
        d2 = PersistenceDiagram([PersistencePoint(0.0, 0.5, 0)])
        result = _mean_pairwise_wasserstein([d1, d2])
        assert result > 0


class TestComputeAttentionTopology:
    """Tests for compute_attention_topology() end-to-end."""

    def test_empty_input(self) -> None:
        """No attention matrices -> default signal."""
        signal = compute_attention_topology({})
        assert signal.total_persistence == 0.0
        assert signal.diagrams == {}

    def test_single_layer_single_head(self) -> None:
        """Single layer, single head -> valid signal with all fields."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        assert 0 in signal.diagrams
        assert len(signal.diagrams[0]) == 1
        # Cross-head should be 0 (single head)
        assert signal.cross_head_wasserstein[0] == 0.0
        # No cross-layer (single layer)
        assert signal.cross_layer_wasserstein == []

    def test_two_layers(self) -> None:
        """Two layers -> cross-layer Wasserstein computed."""
        attn1 = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        attn2 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        signal = compute_attention_topology({0: [attn1], 1: [attn2]})
        assert len(signal.cross_layer_wasserstein) == 1

    def test_two_heads(self) -> None:
        """Two heads in one layer -> cross-head Wasserstein computed."""
        attn1 = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        attn2 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        signal = compute_attention_topology({0: [attn1, attn2]})
        assert len(signal.diagrams[0]) == 2
        assert 0 in signal.cross_head_wasserstein

    def test_expansion_ratio_passthrough(self) -> None:
        """Expansion ratio is passed through to signal."""
        signal = compute_attention_topology({}, expansion_ratio=1.23)
        assert signal.expansion_ratio == 1.23

    def test_barcode_stats_populated(self) -> None:
        """Barcode statistics are computed from the aggregate diagram."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        # Should have some H0 bars (connected components)
        assert signal.n_bars_h0 > 0
        # Total persistence should be positive
        assert signal.total_persistence > 0
        # Max persistence should be positive and <= total
        assert 0 < signal.max_persistence <= signal.total_persistence

    def test_all_measurements_finite(self) -> None:
        """All scalar measurements should be finite."""
        attn = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.1, 0.6]]
        signal = compute_attention_topology({0: [attn]})
        assert math.isfinite(signal.total_persistence)
        assert math.isfinite(signal.max_persistence)
        assert math.isfinite(signal.persistence_entropy)
        assert math.isfinite(float(signal.n_bars_h0))
        assert math.isfinite(float(signal.n_bars_h1))
        for v in signal.cross_head_wasserstein.values():
            assert math.isfinite(v)
        for v in signal.cross_layer_wasserstein:
            assert math.isfinite(v)

    def test_no_betti_curve_attributes(self) -> None:
        """Signal should NOT have Betti curve attributes (removed — n_steps was a guess)."""
        signal = compute_attention_topology({})
        assert not hasattr(signal, "betti_curve_width")
        assert not hasattr(signal, "betti_curve_centroid")
        assert not hasattr(signal, "betti_curve_peak")
        assert not hasattr(signal, "betti_curve_spread")

    def test_no_feature_vector_method(self) -> None:
        """Signal should NOT have feature_vector() (removed — 0.0 imputation was a lie)."""
        signal = compute_attention_topology({})
        assert not hasattr(signal, "feature_vector")
