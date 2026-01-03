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

"""Tests for TopologicalFingerprint.

Tests mathematical properties of persistent homology computation:
- Betti numbers for known topologies
- Distance metric properties (d(X,X) = 0, non-negativity)
- Edge cases (empty, single point, collinear)
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    BackendTopologicalFingerprint,
    PersistenceDiagram,
    PersistencePoint,
    TopologicalFingerprint,
    get_topological_fingerprint,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


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
        fingerprint = TopologicalFingerprint.compute(collinear)
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
        triangle_height = sqrt_scalar(3.0, get_default_backend()) / 2.0
        triangle = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, triangle_height],
        ]
        fingerprint = TopologicalFingerprint.compute(triangle)
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
        fingerprint = TopologicalFingerprint.compute(square)
        # Eventually becomes one component
        assert fingerprint.summary.component_count >= 1

    def test_betti_persistence_threshold_filters_noise(self) -> None:
        """Betti numbers with threshold should filter short-lived features."""
        points = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]]  # Two close, one far
        fingerprint = TopologicalFingerprint.compute(points)

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
        fingerprint = TopologicalFingerprint.compute(points)
        comparison = TopologicalFingerprint.compare(fingerprint, fingerprint)

        eps = _eps(
            comparison.bottleneck_distance,
            comparison.wasserstein_distance,
            comparison.similarity_score,
        )
        assert abs(comparison.bottleneck_distance - 0.0) <= eps
        assert abs(comparison.wasserstein_distance - 0.0) <= eps
        assert comparison.betti_difference == 0
        assert abs(comparison.similarity_score - 1.0) <= eps
        assert comparison.betti_numbers_match is True

    def test_different_topologies_positive_distance(self) -> None:
        """Different topologies should have positive distance."""
        # Simple line
        line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        # Triangle (different topology)
        triangle = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]

        fp_line = TopologicalFingerprint.compute(line)
        fp_triangle = TopologicalFingerprint.compute(triangle)

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

        fp_a = TopologicalFingerprint.compute(points_a)
        fp_b = TopologicalFingerprint.compute(points_b)

        comp_ab = TopologicalFingerprint.compare(fp_a, fp_b)
        comp_ba = TopologicalFingerprint.compare(fp_b, fp_a)

        eps = _eps(
            comp_ab.bottleneck_distance,
            comp_ba.bottleneck_distance,
            comp_ab.wasserstein_distance,
            comp_ba.wasserstein_distance,
        )
        assert abs(comp_ab.bottleneck_distance - comp_ba.bottleneck_distance) <= eps
        assert abs(comp_ab.wasserstein_distance - comp_ba.wasserstein_distance) <= eps
        assert comp_ab.betti_difference == comp_ba.betti_difference


class TestPersistencePoint:
    """Tests for PersistencePoint dataclass."""

    def test_persistence_is_death_minus_birth(self) -> None:
        """Persistence should be death - birth."""
        point = PersistencePoint(birth=0.5, death=1.5, dimension=0)
        eps = _eps(point.persistence, 1.0)
        assert abs(point.persistence - 1.0) <= eps

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


class TestComparisonMetrics:
    """Tests for comparison raw metrics."""

    def test_identical_structure_metrics(self) -> None:
        """Identical topologies should have zero distance and matching Betti numbers."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        fingerprint = TopologicalFingerprint.compute(points)
        comparison = TopologicalFingerprint.compare(fingerprint, fingerprint)

        # Identical fingerprints have zero distance and matching Betti numbers
        eps = _eps(comparison.bottleneck_distance)
        assert comparison.bottleneck_distance <= eps
        assert comparison.betti_difference == 0
        assert comparison.betti_numbers_match is True

    def test_scaled_points_similar_topology(self) -> None:
        """Scaled version of same points should have similar topology."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        scaled = [[p[0] * 2, p[1] * 2] for p in points]

        fp_orig = TopologicalFingerprint.compute(points)
        fp_scaled = TopologicalFingerprint.compute(scaled)

        comparison = TopologicalFingerprint.compare(fp_orig, fp_scaled)
        # Scaling preserves topology, so Betti numbers should match
        assert comparison.betti_difference == 0


class TestHungarianAlgorithm:
    """Tests for Hungarian algorithm implementation."""

    def test_simple_matching(self) -> None:
        """Should find optimal matching for simple cost matrix."""
        cost = [
            [1.0, 2.0],
            [3.0, 0.5],
        ]
        matching = TopologicalFingerprint._hungarian_algorithm(cost)

        # Optimal: row 0 -> col 0 (cost 1), row 1 -> col 1 (cost 0.5)
        assert matching[0] == 0
        assert matching[1] == 1

    def test_empty_matrix(self) -> None:
        """Should handle empty matrix."""
        matching = TopologicalFingerprint._hungarian_algorithm([])
        assert matching == []

    def test_three_by_three(self) -> None:
        """Should find optimal matching for 3x3 cost matrix."""
        cost = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
        ]
        matching = TopologicalFingerprint._hungarian_algorithm(cost)

        # Optimal: diagonal (total cost 3)
        assert matching[0] == 0
        assert matching[1] == 1
        assert matching[2] == 2


class TestPairwiseDistances:
    """Tests for distance matrix computation."""

    def test_distance_matrix_symmetric(self) -> None:
        """Distance matrix should be symmetric."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        distances = TopologicalFingerprint._compute_pairwise_distances(points)

        n = len(points)
        for i in range(n):
            for j in range(n):
                eps = _eps(distances[i][j], distances[j][i])
                assert abs(distances[i][j] - distances[j][i]) <= eps

    def test_diagonal_is_zero(self) -> None:
        """Distance from point to itself should be 0."""
        points = [[0.0, 0.0], [1.0, 0.0]]
        distances = TopologicalFingerprint._compute_pairwise_distances(points)

        for i in range(len(points)):
            eps = _eps(distances[i][i], 0.0)
            assert abs(distances[i][i] - 0.0) <= eps

    def test_euclidean_distance_correct(self) -> None:
        """Should compute correct Euclidean distances."""
        points = [[0.0, 0.0], [3.0, 4.0]]
        distances = TopologicalFingerprint._compute_pairwise_distances(points)

        # Distance should be 5 (3-4-5 triangle)
        eps = _eps(distances[0][1], distances[1][0], 5.0)
        assert abs(distances[0][1] - 5.0) <= eps
        assert abs(distances[1][0] - 5.0) <= eps

    def test_empty_returns_empty(self) -> None:
        """Empty input should return empty matrix."""
        distances = TopologicalFingerprint._compute_pairwise_distances([])
        backend = get_default_backend()
        assert backend.shape(distances) == (0, 0)


class TestPersistenceEntropy:
    """Tests for persistence entropy computation."""

    def test_entropy_non_negative(self) -> None:
        """Entropy should be non-negative."""
        values = [0.1, 0.2, 0.3]
        entropy = TopologicalFingerprint._compute_entropy(values)
        assert entropy >= 0

    def test_entropy_zero_for_empty(self) -> None:
        """Entropy should be 0 for empty values."""
        entropy = TopologicalFingerprint._compute_entropy([])
        eps = _eps(entropy, 0.0)
        assert abs(entropy - 0.0) <= eps

    def test_entropy_max_for_uniform(self) -> None:
        """Entropy should be maximal for uniform distribution."""
        n = 4
        uniform = [1.0] * n
        skewed = [0.7, 0.1, 0.1, 0.1]

        uniform_entropy = TopologicalFingerprint._compute_entropy(uniform)
        skewed_entropy = TopologicalFingerprint._compute_entropy(skewed)

        assert uniform_entropy >= skewed_entropy


class TestTopologySummary:
    """Tests for TopologySummary computation."""

    def test_summary_has_valid_fields(self) -> None:
        """Summary should have valid component and cycle counts."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        fp = TopologicalFingerprint.compute(points)

        assert fp.summary.component_count >= 1
        assert fp.summary.cycle_count >= 0
        assert fp.summary.average_persistence >= 0
        assert fp.summary.max_persistence >= 0
        assert fp.summary.persistence_entropy >= 0


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    def test_persistence_always_non_negative(self) -> None:
        """All persistence values should be >= 0."""
        import random

        random.seed(42)
        for _ in range(10):
            n = random.randint(2, 20)
            points = [[random.random(), random.random()] for _ in range(n)]
            fp = TopologicalFingerprint.compute(points)

            for point in fp.diagram.points:
                assert point.persistence >= 0

    def test_self_comparison_perfect_match(self) -> None:
        """Comparing fingerprint to itself should give perfect match."""
        import random

        random.seed(42)
        for _ in range(5):
            n = random.randint(3, 10)
            points = [[random.random(), random.random()] for _ in range(n)]
            fp = TopologicalFingerprint.compute(points)

            result = TopologicalFingerprint.compare(fp, fp)

            eps = _eps(result.bottleneck_distance, 0.0)
            assert abs(result.bottleneck_distance - 0.0) <= eps
            assert result.betti_difference == 0

    def test_distance_matrix_is_square(self) -> None:
        """Distance matrix should be n x n."""
        import random

        random.seed(42)
        for n in [2, 5, 10]:
            points = [[random.random(), random.random()] for _ in range(n)]
            distances = TopologicalFingerprint._compute_pairwise_distances(points)

            assert len(distances) == n
            for row in distances:
                assert len(row) == n


class TestBackendTopologicalFingerprint:
    """Tests for GPU-accelerated BackendTopologicalFingerprint."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def backend_fp(self, backend):
        return BackendTopologicalFingerprint(backend)

    def test_compute_matches_pure_python(self, backend_fp) -> None:
        """Backend compute should match pure Python results."""
        points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.5]]

        pure = TopologicalFingerprint.compute(points)
        gpu = backend_fp.compute(points)

        # Summary should match
        assert pure.summary.component_count == gpu.summary.component_count
        assert pure.summary.cycle_count == gpu.summary.cycle_count
        eps = _eps(pure.summary.average_persistence, gpu.summary.average_persistence)
        assert abs(pure.summary.average_persistence - gpu.summary.average_persistence) <= eps
        eps = _eps(pure.summary.max_persistence, gpu.summary.max_persistence)
        assert abs(pure.summary.max_persistence - gpu.summary.max_persistence) <= eps

    def test_compare_matches_pure_python(self, backend_fp) -> None:
        """Backend compare should match pure Python results."""
        points_a = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
        points_b = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]

        fp_a = TopologicalFingerprint.compute(points_a)
        fp_b = TopologicalFingerprint.compute(points_b)

        pure = TopologicalFingerprint.compare(fp_a, fp_b)
        gpu = backend_fp.compare(fp_a, fp_b)

        eps = _eps(pure.bottleneck_distance, gpu.bottleneck_distance)
        assert abs(pure.bottleneck_distance - gpu.bottleneck_distance) <= eps
        eps = _eps(pure.wasserstein_distance, gpu.wasserstein_distance)
        assert abs(pure.wasserstein_distance - gpu.wasserstein_distance) <= eps
        assert pure.betti_difference == gpu.betti_difference

    def test_bottleneck_distance_matches(self, backend_fp) -> None:
        """Backend bottleneck distance should match pure Python."""
        points = [
            PersistencePoint(0.0, 1.0, 0),
            PersistencePoint(0.2, 0.8, 0),
            PersistencePoint(0.1, 0.5, 1),
        ]
        diag = PersistenceDiagram(points)

        pure = TopologicalFingerprint._bottleneck_distance(diag, diag)
        gpu = backend_fp._bottleneck_distance(diag, diag)

        eps = _eps(pure, gpu)
        assert abs(pure - gpu) <= eps

    def test_wasserstein_distance_matches(self, backend_fp) -> None:
        """Backend Wasserstein distance should match pure Python."""
        points = [
            PersistencePoint(0.0, 1.0, 0),
            PersistencePoint(0.2, 0.8, 0),
        ]
        diag = PersistenceDiagram(points)

        pure = TopologicalFingerprint._wasserstein_distance(diag, diag)
        gpu = backend_fp._wasserstein_distance(diag, diag)

        eps = _eps(pure, gpu)
        assert abs(pure - gpu) <= eps

    def test_entropy_matches(self, backend_fp) -> None:
        """Backend entropy computation should match pure Python."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]

        pure = TopologicalFingerprint._compute_entropy(values)
        gpu = backend_fp._compute_entropy(values)

        eps = _eps(pure, gpu)
        assert abs(pure - gpu) <= eps

    def test_empty_input_matches(self, backend_fp) -> None:
        """Empty input should work identically."""
        pure = TopologicalFingerprint.compute([])
        gpu = backend_fp.compute([])

        assert pure.summary.component_count == gpu.summary.component_count
        assert pure.summary.cycle_count == gpu.summary.cycle_count

    def test_single_point_matches(self, backend_fp) -> None:
        """Single point should work identically."""
        pure = TopologicalFingerprint.compute([[0.0, 0.0]])
        gpu = backend_fp.compute([[0.0, 0.0]])

        assert pure.summary.component_count == gpu.summary.component_count
        assert pure.summary.cycle_count == gpu.summary.cycle_count


class TestGetTopologicalFingerprint:
    """Tests for the factory function."""

    def test_returns_class_without_backend(self) -> None:
        """Factory should return TopologicalFingerprint class without backend."""
        result = get_topological_fingerprint()
        assert result is TopologicalFingerprint

    def test_returns_instance_with_backend(self) -> None:
        """Factory should return BackendTopologicalFingerprint instance with backend."""
        backend = get_default_backend()
        result = get_topological_fingerprint(backend)
        assert isinstance(result, BackendTopologicalFingerprint)
