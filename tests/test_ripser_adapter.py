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

import pytest

from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)


def test_ripser_filtration_0dim():
    """Test 0-dimensional persistence (connected components)."""
    # 3 points, two very close, one far
    points = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]]

    distances = TopologicalFingerprint._compute_pairwise_distances(points)
    diagram = TopologicalFingerprint._vietoris_rips_filtration(
        distances=distances, min_filtration=0.0, max_filtration=20.0, num_steps=100, max_dimension=0
    )

    # Dimension 0 points should track merges
    points0 = [p for p in diagram.points if p.dimension == 0]
    # One point lives forever (death = max_filtration)
    # Others die when they merge.
    # P0 and P1 merge at dist 0.1
    # Then P0+P1 and P2 merge at dist 9.9

    deaths = sorted([p.death for p in points0])
    # Use approximate comparison due to floating point
    assert any(d == pytest.approx(0.1, rel=0.1) for d in deaths)
    assert any(d == pytest.approx(9.9, rel=0.1) for d in deaths)
    assert any(d == pytest.approx(20.0, rel=0.1) for d in deaths)


def test_ripser_filtration_1dim_cycle():
    """Test 1-dimensional persistence (cycles)."""
    # 4 points in a square forming a loop
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    distances = TopologicalFingerprint._compute_pairwise_distances(points)
    diagram = TopologicalFingerprint._vietoris_rips_filtration(
        distances=distances, min_filtration=0.0, max_filtration=10.0, num_steps=50, max_dimension=1
    )

    points1 = [p for p in diagram.points if p.dimension == 1]
    assert len(points1) > 0
    # Birth should be the distance that completes the cycle (side length 1.0)
    assert any(p.birth == pytest.approx(1.0) for p in points1)


def test_ripser_bottleneck_distance():
    """Test bottleneck distance calculation."""
    # Create two identical diagrams
    diag_a = TopologicalFingerprint.compute([[0, 0], [1, 0]]).diagram
    diag_b = TopologicalFingerprint.compute([[0, 0], [1, 0]]).diagram

    dist = TopologicalFingerprint._bottleneck_distance(diag_a, diag_b)
    assert dist == pytest.approx(0.0)


def test_ripser_wasserstein_distance():
    """Test Wasserstein distance calculation."""
    diag_a = TopologicalFingerprint.compute([[0, 0], [1, 0]]).diagram
    diag_b = TopologicalFingerprint.compute([[0, 0], [1, 0]]).diagram

    dist = TopologicalFingerprint._wasserstein_distance(diag_a, diag_b)
    assert dist == pytest.approx(0.0)


def test_ripser_adapter_elder_rule():
    """Test the elder rule in filtration (older component survives)."""
    # Points P0, P1, P2
    # P0 and P1 merge first at t=1
    # P2 merges with P0+P1 at t=2
    distances = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    diagram = TopologicalFingerprint._vietoris_rips_filtration(
        distances=distances, min_filtration=0.0, max_filtration=10.0, num_steps=10, max_dimension=0
    )

    points0 = [p for p in diagram.points if p.dimension == 0]
    deaths = sorted([p.death for p in points0])

    # 3 points start. 2 merges happen.
    # t=1: P1 dies (born at 0)
    # t=2: P2 dies (born at 0)
    # P0 survives to 10.
    assert deaths == [1.0, 2.0, 10.0]
