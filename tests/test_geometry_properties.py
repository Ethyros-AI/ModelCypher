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
from modelcypher.core.domain.geometry.generalized_procrustes import GeneralizedProcrustes
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


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
    """Tests for Gromov-Wasserstein distance mathematical properties.

    These properties are enforced by construction in the Result dataclass.
    Fixed inputs verify the implementation without expensive random generation.
    """

    # Fixed test point clouds (small, distinct points)
    POINTS_A = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    POINTS_B = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    POINTS_C = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

    def test_self_distance_is_zero(self):
        """GW(D, D) = 0 for identical distance matrices (fast path)."""
        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(self.POINTS_A)
        result = gw.compute(distances, distances)
        assert abs(result.distance) <= _eps(result.distance)

    def test_distance_is_non_negative(self):
        """Distance >= 0 (enforced by Result.__post_init__)."""
        gw = GromovWassersteinDistance()
        dist_a = gw.compute_pairwise_distances(self.POINTS_A)
        dist_b = gw.compute_pairwise_distances(self.POINTS_B)
        result = gw.compute(dist_a, dist_b)
        assert result.distance >= -_eps(result.distance)

    def test_normalized_distance_bounded(self):
        """Normalized distance in [0, 1] (by definition: 1 - exp(-d))."""
        gw = GromovWassersteinDistance()
        dist_a = gw.compute_pairwise_distances(self.POINTS_A)
        dist_c = gw.compute_pairwise_distances(self.POINTS_C)
        result = gw.compute(dist_a, dist_c)
        eps = _eps(result.normalized_distance, 1.0)
        assert result.normalized_distance >= -eps
        assert result.normalized_distance <= 1.0 + eps

    def test_aligned_is_boolean(self):
        """aligned property returns bool."""
        gw = GromovWassersteinDistance()
        dist_a = gw.compute_pairwise_distances(self.POINTS_A)
        dist_b = gw.compute_pairwise_distances(self.POINTS_B)
        result = gw.compute(dist_a, dist_b)
        assert isinstance(result.aligned, bool)

    def test_coupling_has_correct_shape(self):
        """Coupling matrix has [n, m] shape."""
        gw = GromovWassersteinDistance()
        dist_a = gw.compute_pairwise_distances(self.POINTS_A)
        dist_b = gw.compute_pairwise_distances(self.POINTS_B)
        result = gw.compute(dist_a, dist_b)
        assert result.coupling.shape == (len(self.POINTS_A), len(self.POINTS_B))

    def test_pairwise_distances_symmetric(self):
        """Distance matrix D[i,j] = D[j,i]."""
        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(self.POINTS_C)
        backend = get_default_backend()
        diff = backend.abs(distances - backend.transpose(distances))
        max_diff = float(backend.to_scalar(backend.max(diff)))
        assert max_diff <= _eps(max_diff)

    def test_pairwise_distances_diagonal_zero(self):
        """Distance matrix diagonal D[i,i] = 0."""
        gw = GromovWassersteinDistance()
        distances = gw.compute_pairwise_distances(self.POINTS_A)
        backend = get_default_backend()
        diag = backend.diag(distances)
        max_diag = float(backend.to_scalar(backend.max(backend.abs(diag))))
        assert max_diag <= _eps(max_diag)
