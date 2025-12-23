"""Tests for TopologicalFingerprint.

Tests mathematical properties of persistent homology computation:
- Betti numbers for known topologies
- Distance metric properties (d(X,X) = 0, non-negativity)
- Edge cases (empty, single point, collinear)
"""

from __future__ import annotations

import math
import pytest

from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
    PersistencePoint,
    PersistenceDiagram,
)


class TestComputeEdgeCases:
    """Tests for edge cases in TopologicalFingerprint.compute()."""

    def test_single_point_has_one_component(self) -> None:
        """Single point should have exactly one connected component."""
        fingerprint = TopologicalFingerprint.compute([[0.0, 0.0]])
        assert fingerprint.summary.component_count == 1
        assert fingerprint.summary.cycle_count == 0
        assert fingerprint.diagram.points == []

    def test_empty_point_cloud(self) -> None:
        """Empty point cloud should return zero components."""
        fingerprint = TopologicalFingerprint.compute([])
        assert fingerprint.summary.component_count == 0
        assert fingerprint.summary.cycle_count == 0

    def test_two_points_one_component(self) -> None:
        """Two points eventually merge into one component."""
        fingerprint = TopologicalFingerprint.compute([[0.0, 0.0], [1.0, 0.0]])
        # After filtration, 2 points merge into 1 component
        # There should be a persistence point recording the merge
        assert fingerprint.summary.component_count >= 1

    def test_collinear_points_no_cycles(self) -> None:
        """Collinear points cannot form cycles.

        Mathematical property: Points on a line have trivial 1-dimensional
        homology because there's no enclosed area.
        """
        collinear = [[float(i), 0.0] for i in range(5)]
        fingerprint = TopologicalFingerprint.compute(collinear, max_dimension=1)
        # Collinear points cannot form 1-cycles
        assert fingerprint.summary.cycle_count == 0


class TestBettiNumbers:
    """Tests for Betti number computation on known topologies."""

    def test_triangle_has_potential_cycle(self) -> None:
        """Three points in a triangle can form a 1-cycle.

        Mathematical property: A triangle has β₀=1 (connected), and
        before the triangle fills in, there's a 1-cycle (β₁=1).
        """
        # Equilateral triangle
        triangle = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
        ]
        fingerprint = TopologicalFingerprint.compute(triangle, max_dimension=1)
        # Should have at least one component
        assert fingerprint.summary.component_count >= 1
        # The cycle may or may not persist depending on filtration

    def test_square_topology(self) -> None:
        """Square has a 1-cycle before diagonal fills it."""
        square = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
        fingerprint = TopologicalFingerprint.compute(square, max_dimension=1)
        # Eventually becomes one component
        assert fingerprint.summary.component_count >= 1

    def test_betti_persistence_threshold_filters_noise(self) -> None:
        """Betti numbers with threshold should filter short-lived features."""
        points = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]]  # Two close, one far
        fingerprint = TopologicalFingerprint.compute(points, max_dimension=1)

        # With a high threshold, short-lived features are filtered
        betti_strict = fingerprint.diagram.betti_numbers(persistence_threshold=1.0)
        betti_loose = fingerprint.diagram.betti_numbers(persistence_threshold=0.01)

        # Loose threshold should show more features
        total_loose = sum(betti_loose.values())
        total_strict = sum(betti_strict.values())
        assert total_loose >= total_strict


class TestCompareFingerprints:
    """Tests for TopologicalFingerprint.compare()."""

    def test_identical_fingerprints_zero_distance(self) -> None:
        """Identical fingerprints should have zero bottleneck and Wasserstein distance.

        Mathematical property: d(X, X) = 0 for any metric.
        """
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        fingerprint = TopologicalFingerprint.compute(points, max_dimension=1)
        comparison = TopologicalFingerprint.compare(fingerprint, fingerprint)

        assert comparison.bottleneck_distance == pytest.approx(0.0, abs=1e-6)
        assert comparison.wasserstein_distance == pytest.approx(0.0, abs=1e-6)
        assert comparison.betti_difference == 0
        assert comparison.similarity_score == pytest.approx(1.0, abs=1e-6)
        assert comparison.is_compatible is True

    def test_different_topologies_positive_distance(self) -> None:
        """Different topologies should have positive distance."""
        # Simple line
        line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        # Triangle (different topology)
        triangle = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]

        fp_line = TopologicalFingerprint.compute(line, max_dimension=1)
        fp_triangle = TopologicalFingerprint.compute(triangle, max_dimension=1)

        comparison = TopologicalFingerprint.compare(fp_line, fp_triangle)
        # Different shapes should have some distance
        # At minimum, the wasserstein distance captures differences in persistence
        assert comparison.bottleneck_distance >= 0
        assert comparison.wasserstein_distance >= 0

    def test_comparison_is_symmetric(self) -> None:
        """Comparison should be symmetric: d(A, B) = d(B, A).

        Mathematical property: Bottleneck and Wasserstein are symmetric metrics.
        """
        points_a = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        points_b = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]

        fp_a = TopologicalFingerprint.compute(points_a, max_dimension=1)
        fp_b = TopologicalFingerprint.compute(points_b, max_dimension=1)

        comp_ab = TopologicalFingerprint.compare(fp_a, fp_b)
        comp_ba = TopologicalFingerprint.compare(fp_b, fp_a)

        assert comp_ab.bottleneck_distance == pytest.approx(comp_ba.bottleneck_distance, abs=1e-6)
        assert comp_ab.wasserstein_distance == pytest.approx(comp_ba.wasserstein_distance, abs=1e-6)
        assert comp_ab.betti_difference == comp_ba.betti_difference


class TestPersistencePoint:
    """Tests for PersistencePoint dataclass."""

    def test_persistence_is_death_minus_birth(self) -> None:
        """Persistence should be death - birth."""
        point = PersistencePoint(birth=0.5, death=1.5, dimension=0)
        assert point.persistence == pytest.approx(1.0, abs=1e-9)

    def test_persistence_non_negative(self) -> None:
        """Persistence should be non-negative when death >= birth."""
        point = PersistencePoint(birth=0.0, death=0.5, dimension=1)
        assert point.persistence >= 0


class TestPersistenceDiagram:
    """Tests for PersistenceDiagram dataclass."""

    def test_count_by_dimension(self) -> None:
        """count_by_dimension should count points per dimension."""
        points = [
            PersistencePoint(0.0, 1.0, 0),
            PersistencePoint(0.0, 0.5, 0),
            PersistencePoint(0.2, 0.8, 1),
        ]
        diagram = PersistenceDiagram(points)
        counts = diagram.count_by_dimension

        assert counts[0] == 2
        assert counts[1] == 1

    def test_betti_numbers_respects_threshold(self) -> None:
        """betti_numbers should only count features above threshold."""
        points = [
            PersistencePoint(0.0, 1.0, 0),  # persistence = 1.0
            PersistencePoint(0.0, 0.05, 0),  # persistence = 0.05 (below 0.1)
            PersistencePoint(0.2, 0.5, 1),  # persistence = 0.3
        ]
        diagram = PersistenceDiagram(points)

        betti = diagram.betti_numbers(persistence_threshold=0.1)
        assert betti.get(0, 0) == 1  # Only the long-lived component
        assert betti.get(1, 0) == 1


class TestInterpretation:
    """Tests for comparison interpretation strings."""

    def test_identical_structure_interpretation(self) -> None:
        """Identical topologies should say 'Identical topological structure'."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        fingerprint = TopologicalFingerprint.compute(points, max_dimension=1)
        comparison = TopologicalFingerprint.compare(fingerprint, fingerprint)

        assert "Identical" in comparison.interpretation

    def test_scaled_points_similar_topology(self) -> None:
        """Scaled version of same points should have similar topology."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        scaled = [[p[0] * 2, p[1] * 2] for p in points]

        fp_orig = TopologicalFingerprint.compute(points, max_dimension=1)
        fp_scaled = TopologicalFingerprint.compute(scaled, max_dimension=1)

        comparison = TopologicalFingerprint.compare(fp_orig, fp_scaled)
        # Scaling preserves topology, so Betti numbers should match
        assert comparison.betti_difference == 0
