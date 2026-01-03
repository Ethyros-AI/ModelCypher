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

"""Property-based tests for geometry operations using Hypothesis."""

import pytest
from hypothesis import assume, given, settings

pytestmark = pytest.mark.property
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.generalized_procrustes import GeneralizedProcrustes
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
)


# Strategy for generating valid 2D matrices
@st.composite
def matrix_2d(draw, rows=st.integers(2, 10), cols=st.integers(2, 10)):
    """Generate a 2D matrix with random floats."""
    n_rows = draw(rows)
    n_cols = draw(cols)
    data = []
    for _ in range(n_rows):
        row = [
            draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
            for _ in range(n_cols)
        ]
        data.append(row)
    return data


@st.composite
def point_cloud(draw, n_points=st.integers(3, 20), dims=st.integers(2, 5)):
    """Generate a point cloud with random floats."""
    n = draw(n_points)
    d = draw(dims)
    points = []
    for _ in range(n):
        point = [
            draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
            for _ in range(d)
        ]
        points.append(point)
    return points


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def has_distinct_points(
    points: list[list[float]],
    min_separation: float | None = None,
) -> bool:
    """Check if point cloud has at least 2 distinct points.

    When all points are identical, geodesic distances become infinite
    because there's no edge in the k-NN graph (self-loops don't count).
    This filters out such degenerate cases for property-based tests.
    """
    if len(points) < 2:
        return False
    if min_separation is None:
        backend = get_default_backend()
        min_separation = machine_epsilon(backend, backend.array(points))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist_sq = sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
            if dist_sq > min_separation * min_separation:
                return True
    return False


class TestProcrustesProperties:
    """Property-based tests for Procrustes alignment."""

    @given(matrix_2d())
    @settings(max_examples=50, deadline=None)
    def test_self_alignment_is_perfect(self, matrix):
        """Aligning a matrix with itself should give exactly zero error.

        Mathematical property: GPA(X, X) = 0 because the optimal rotation
        is the identity matrix and both matrices are identical.
        """
        assume(len(matrix) >= 2)
        assume(len(matrix[0]) >= 2)

        # All parameters are now derived from data at runtime
        result = GeneralizedProcrustes().align([matrix, matrix])

        if result is not None:
            # Self-alignment must be exactly 0 (within floating point tolerance)
            # This tests the core mathematical property: d(X, X) = 0
            assert abs(result.alignment_error - 0.0) <= _eps(result.alignment_error)

    @given(matrix_2d(), matrix_2d())
    @settings(max_examples=30, deadline=None)
    def test_alignment_error_is_non_negative(self, matrix_a, matrix_b):
        """Alignment error should always be non-negative."""
        assume(len(matrix_a) >= 2 and len(matrix_b) >= 2)
        assume(len(matrix_a[0]) >= 2 and len(matrix_b[0]) >= 2)

        # All parameters are now derived from data at runtime
        result = GeneralizedProcrustes().align([matrix_a, matrix_b])

        if result is not None:
            eps = _eps(result.alignment_error)
            assert result.alignment_error >= -eps

    @given(st.lists(matrix_2d(), min_size=2, max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_model_count_matches_input(self, matrices):
        """Result should report correct model count."""
        # Ensure all matrices have compatible dimensions
        if not matrices:
            return

        assume(all(len(m) >= 2 for m in matrices))
        assume(all(len(m[0]) >= 2 for m in matrices if m))

        # All parameters are now derived from data at runtime
        result = GeneralizedProcrustes().align(matrices)

        if result is not None:
            assert result.model_count == len(matrices)

    @given(matrix_2d(rows=st.just(3), cols=st.just(3)))
    @settings(max_examples=30, deadline=None)
    def test_consensus_variance_ratio_bounded(self, matrix):
        """Consensus variance ratio should be in [0, 1]."""
        assume(len(matrix) >= 2)

        # All parameters are now derived from data at runtime
        result = GeneralizedProcrustes().align([matrix, matrix])

        if result is not None:
            eps = _eps(result.consensus_variance_ratio, 1.0)
            assert result.consensus_variance_ratio >= -eps
            assert result.consensus_variance_ratio <= 1.0 + eps


class TestGromovWassersteinProperties:
    """Property-based tests for Gromov-Wasserstein distance."""

    @given(point_cloud())
    @settings(max_examples=30, deadline=None)
    def test_self_distance_is_zero(self, points):
        """Distance from a point cloud to itself should be zero.

        Mathematical property: GW(D, D) = 0 when comparing identical
        distance matrices, because the identity coupling is optimal.
        """
        assume(len(points) >= 2)
        assume(has_distinct_points(points))  # Filter degenerate cases

        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(points)

        result = gw.compute(distances, distances)

        # Implementation has fast-path for identical matrices returning 0
        # This is a fundamental mathematical property: d(X, X) = 0
        assert abs(result.distance - 0.0) <= _eps(result.distance)

    @given(point_cloud(), point_cloud())
    @settings(max_examples=20, deadline=None)
    def test_distance_is_non_negative(self, points_a, points_b):
        """Distance should always be non-negative."""
        assume(len(points_a) >= 2 and len(points_b) >= 2)
        assume(has_distinct_points(points_a) and has_distinct_points(points_b))

        gw = GromovWassersteinDistance()
        distances_a = gw.compute_pairwise_distances(points_a)
        distances_b = gw.compute_pairwise_distances(points_b)

        result = gw.compute(distances_a, distances_b)

        eps = _eps(result.distance)
        assert result.distance >= -eps

    @given(point_cloud(), point_cloud())
    @settings(max_examples=20, deadline=None)
    def test_normalized_distance_bounded(self, points_a, points_b):
        """Normalized distance should be in [0, 1]."""
        assume(len(points_a) >= 2 and len(points_b) >= 2)
        assume(has_distinct_points(points_a) and has_distinct_points(points_b))

        gw = GromovWassersteinDistance()
        distances_a = gw.compute_pairwise_distances(points_a)
        distances_b = gw.compute_pairwise_distances(points_b)

        result = gw.compute(distances_a, distances_b)

        eps = _eps(result.normalized_distance, 1.0)
        assert result.normalized_distance >= -eps
        assert result.normalized_distance <= 1.0 + eps

    @given(point_cloud(), point_cloud())
    @settings(max_examples=20, deadline=None)
    def test_alignment_score_bounded(self, points_a, points_b):
        """Compatibility score should be in [0, 1]."""
        assume(len(points_a) >= 2 and len(points_b) >= 2)
        assume(has_distinct_points(points_a) and has_distinct_points(points_b))

        gw = GromovWassersteinDistance()
        distances_a = gw.compute_pairwise_distances(points_a)
        distances_b = gw.compute_pairwise_distances(points_b)

        result = gw.compute(distances_a, distances_b)

        eps = _eps(result.alignment_score, 1.0)
        assert result.alignment_score >= -eps
        assert result.alignment_score <= 1.0 + eps

    @given(point_cloud())
    @settings(max_examples=30, deadline=None)
    def test_coupling_has_correct_shape(self, points):
        """Coupling matrix should have correct dimensions."""
        assume(len(points) >= 2)
        assume(has_distinct_points(points))

        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(points)

        result = gw.compute(distances, distances)

        n = len(points)
        if result.coupling is not None and hasattr(result.coupling, 'shape'):
            assert result.coupling.shape[0] == n
            assert result.coupling.shape[1] == n

    @given(point_cloud())
    @settings(max_examples=30, deadline=None)
    def test_pairwise_distances_symmetric(self, points):
        """Pairwise distance matrix should be symmetric."""
        assume(len(points) >= 2)
        assume(has_distinct_points(points))

        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(points)

        # Convert to numpy for element-wise comparison
        if hasattr(distances, 'tolist'):
            dist_np = distances.tolist() if hasattr(distances, 'tolist') else list(distances)
        else:
            dist_np = [[float(distances[i, j]) for j in range(len(points))] for i in range(len(points))]

        n = len(points)
        for i in range(n):
            for j in range(n):
                eps = _eps(dist_np[i][j], dist_np[j][i])
                assert abs(dist_np[i][j] - dist_np[j][i]) <= eps

    @given(point_cloud())
    @settings(max_examples=30, deadline=None)
    def test_pairwise_distances_diagonal_zero(self, points):
        """Diagonal of distance matrix should be zero (within floating point tolerance)."""
        assume(len(points) >= 2)
        assume(has_distinct_points(points))

        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(points)

        for i in range(len(points)):
            # Extract scalar value from array
            diag_val = float(distances[i, i]) if hasattr(distances, '__getitem__') else distances[i][i]
            assert abs(diag_val) <= _eps(diag_val)
