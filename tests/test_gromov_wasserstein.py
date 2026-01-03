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

"""Comprehensive tests for gromov_wasserstein.py.

Tests:
- Result dataclass (properties, __post_init__)
- GromovWassersteinDistance class (compute, Frank-Wolfe, Sinkhorn, helpers)
- Edge cases (empty, single point, identical matrices)

All solver parameters are derived from dtype - no configuration needed.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
    Result,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps(backend: "Backend", *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))

# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestResult:
    """Tests for Result dataclass."""

    def test_basic_creation(self, any_backend: "Backend") -> None:
        """Should create result with all fields."""
        b = any_backend
        coupling = b.eye(3) / 3
        b.eval(coupling)

        result = Result(
            distance=0.5,
            coupling=coupling,
            converged=True,
            iterations=10,
        )

        assert result.distance == 0.5
        assert result.converged is True
        assert result.iterations == 10

    def test_negative_distance_clamped(self, any_backend: "Backend") -> None:
        """Negative distance should be clamped to 0."""
        b = any_backend
        coupling = b.zeros((2, 2))
        b.eval(coupling)

        result = Result(
            distance=-0.5,
            coupling=coupling,
            converged=True,
            iterations=1,
        )

        # __post_init__ should clamp to 0
        assert result.distance == 0.0

    def test_inf_distance_not_clamped(self, any_backend: "Backend") -> None:
        """Infinite distance should not be clamped."""
        b = any_backend
        coupling = b.zeros((2, 2))
        b.eval(coupling)

        result = Result(
            distance=float("inf"),
            coupling=coupling,
            converged=False,
            iterations=0,
        )

        assert result.distance == float("inf")

    def test_normalized_distance_zero(self, any_backend: "Backend") -> None:
        """normalized_distance should be 0 for distance=0."""
        b = any_backend
        coupling = b.eye(2) / 2
        b.eval(coupling)

        result = Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        # 1 - exp(-0) = 1 - 1 = 0
        eps = _eps(b, result.normalized_distance, 0.0)
        assert abs(result.normalized_distance - 0.0) <= eps

    def test_normalized_distance_large(self, any_backend: "Backend") -> None:
        """normalized_distance should approach 1 for large distance."""
        b = any_backend
        coupling = b.eye(2) / 2
        b.eval(coupling)

        result = Result(distance=10.0, coupling=coupling, converged=True, iterations=0)

        neg_distance = b.array([-result.distance])
        exp_val = b.exp(neg_distance)
        b.eval(exp_val)
        expected = 1.0 - float(b.to_scalar(exp_val))
        eps = _eps(b, result.normalized_distance, expected)
        assert abs(result.normalized_distance - expected) <= eps

    def test_normalized_distance_inf(self, any_backend: "Backend") -> None:
        """normalized_distance should be 1 for infinite distance."""
        b = any_backend
        coupling = b.zeros((2, 2))
        b.eval(coupling)

        result = Result(distance=float("inf"), coupling=coupling, converged=False, iterations=0)

        eps = _eps(b, result.normalized_distance, 1.0)
        assert abs(result.normalized_distance - 1.0) <= eps

    def test_alignment_score_zero_distance(self, any_backend: "Backend") -> None:
        """alignment_score should be 1 for distance=0."""
        b = any_backend
        coupling = b.eye(2) / 2
        b.eval(coupling)

        result = Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        # exp(-0) = 1
        eps = _eps(b, result.alignment_score, 1.0)
        assert abs(result.alignment_score - 1.0) <= eps

    def test_alignment_score_large_distance(self, any_backend: "Backend") -> None:
        """alignment_score should approach 0 for large distance."""
        b = any_backend
        coupling = b.eye(2) / 2
        b.eval(coupling)

        result = Result(distance=10.0, coupling=coupling, converged=True, iterations=0)

        neg_distance = b.array([-result.distance])
        exp_val = b.exp(neg_distance)
        b.eval(exp_val)
        expected = float(b.to_scalar(exp_val))
        eps = _eps(b, result.alignment_score, expected)
        assert abs(result.alignment_score - expected) <= eps

    def test_alignment_score_inf(self, any_backend: "Backend") -> None:
        """alignment_score should be 0 for infinite distance."""
        b = any_backend
        coupling = b.zeros((2, 2))
        b.eval(coupling)

        result = Result(distance=float("inf"), coupling=coupling, converged=False, iterations=0)

        eps = _eps(b, result.alignment_score, 0.0)
        assert abs(result.alignment_score - 0.0) <= eps

    def test_frozen(self, any_backend: "Backend") -> None:
        """Result should be frozen (immutable)."""
        b = any_backend
        coupling = b.eye(2) / 2
        b.eval(coupling)

        result = Result(distance=0.5, coupling=coupling, converged=True, iterations=5)

        with pytest.raises(Exception):  # FrozenInstanceError
            result.distance = 1.0  # type: ignore


# =============================================================================
# GromovWassersteinDistance Tests - Basic Operations
# =============================================================================


class TestGromovWassersteinDistanceBasic:
    """Basic tests for GromovWassersteinDistance class."""

    def test_creation_with_backend(self, any_backend: "Backend") -> None:
        """Should create with provided backend."""
        gw = GromovWassersteinDistance(backend=any_backend)
        assert gw._backend is any_backend

    def test_creation_without_backend(self) -> None:
        """Should use default backend when none provided."""
        gw = GromovWassersteinDistance()
        assert gw._backend is not None

    def test_compute_empty_matrices(self, any_backend: "Backend") -> None:
        """Should handle empty distance matrices."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        empty = b.zeros((0, 0))
        b.eval(empty)

        result = gw.compute(empty, empty)

        assert result.distance == float("inf")
        assert result.converged is False
        assert result.iterations == 0

    def test_compute_identical_matrices(self, any_backend: "Backend") -> None:
        """Identical matrices should have distance 0."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        # Create a distance matrix
        dist = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        b.eval(dist)

        result = gw.compute(dist, dist)

        assert result.distance == 0.0
        assert result.converged is True
        assert result.iterations == 0

    def test_compute_pairwise_distances_simple(self, any_backend: "Backend") -> None:
        """Should compute pairwise distances for simple points."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        points = b.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        b.eval(points)

        distances = gw.compute_pairwise_distances(points)
        b.eval(distances)

        # Should be symmetric
        shape = b.shape(distances)
        assert shape[0] == 3
        assert shape[1] == 3

        # Diagonal should be 0
        for i in range(3):
            diag = float(distances[i, i])
            eps = _eps(b, diag, 0.0)
            assert abs(diag - 0.0) <= eps

    def test_compute_pairwise_distances_empty(self, any_backend: "Backend") -> None:
        """Should handle empty point set."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        empty = b.zeros((0, 2))
        b.eval(empty)

        distances = gw.compute_pairwise_distances(empty)
        b.eval(distances)

        shape = b.shape(distances)
        assert shape[0] == 0
        assert shape[1] == 0


# =============================================================================
# GromovWassersteinDistance Tests - Identity and Permutation
# =============================================================================


class TestGromovWassersteinDistanceIdentity:
    """Tests for identity distance cases."""

    def test_gw_identity_distance(self, any_backend: "Backend") -> None:
        """Self-comparison should give zero GW distance with uniform coupling."""
        gw = GromovWassersteinDistance(backend=any_backend)
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        distances = gw.compute_pairwise_distances(points)
        result = gw.compute(distances, distances)

        eps = _eps(any_backend, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps
        assert result.converged is True
        assert result.iterations == 0
        b = any_backend
        coupling = result.coupling
        shape = b.shape(coupling)
        assert shape[0] == 3
        assert shape[1] == 3
        expected = 1.0 / 3.0
        diag = b.diag(coupling)
        b.eval(diag)
        diag_list = array_to_list(b, diag)
        for value in diag_list:
            eps = _eps(any_backend, value, expected)
            assert abs(value - expected) <= eps

    def test_gw_permutation_distance_small(self, any_backend: "Backend") -> None:
        """Permuted points should have near-zero GW distance (same shape)."""
        gw = GromovWassersteinDistance(backend=any_backend)
        points_a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        permutation = [2, 0, 1]
        points_b = [points_a[idx] for idx in permutation]

        dist_a = gw.compute_pairwise_distances(points_a)
        dist_b = gw.compute_pairwise_distances(points_b)

        # All params derived from dtype - no config needed
        result = gw.compute(dist_a, dist_b)

        identity = gw.compute(dist_a, dist_a)
        eps = _eps(any_backend, result.distance, identity.distance)
        assert abs(result.distance - identity.distance) <= eps

        # Verify coupling marginals sum to uniform distribution
        coupling = result.coupling
        row_sums = any_backend.sum(coupling, axis=1)
        col_sums = any_backend.sum(coupling, axis=0)
        any_backend.eval(row_sums)
        any_backend.eval(col_sums)
        row_mass = array_to_list(any_backend, row_sums)
        col_mass = array_to_list(any_backend, col_sums)
        expected = 1.0 / 3.0
        for value in row_mass:
            eps = _eps(any_backend, value, expected)
            assert abs(value - expected) <= eps
        for value in col_mass:
            eps = _eps(any_backend, value, expected)
            assert abs(value - expected) <= eps


# =============================================================================
# GromovWassersteinDistance Tests - Random Coupling
# =============================================================================


class TestRandomCoupling:
    """Tests for random coupling generation."""

    def test_random_coupling_valid_marginals(self, any_backend: "Backend") -> None:
        """Random coupling should have valid marginals."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend
        b.random_seed(42)

        n, m = 5, 5
        coupling = gw._random_coupling(n, m, b)
        b.eval(coupling)

        # Check row sums (should be 1/n)
        row_sums = b.sum(coupling, axis=1)
        b.eval(row_sums)
        row_list = array_to_list(b, row_sums)
        for value in row_list:
            expected = 1.0 / n
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

        # Check column sums (should be 1/m)
        col_sums = b.sum(coupling, axis=0)
        b.eval(col_sums)
        col_list = array_to_list(b, col_sums)
        for value in col_list:
            expected = 1.0 / m
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

    def test_random_coupling_positive(self, any_backend: "Backend") -> None:
        """Random coupling should be non-negative."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend
        b.random_seed(42)

        coupling = gw._random_coupling(4, 4, b)
        b.eval(coupling)

        min_val = float(b.min(coupling))
        assert min_val >= 0.0


# =============================================================================
# GromovWassersteinDistance Tests - Small Matrix Permutation Search
# =============================================================================


class TestPermutationSearch:
    """Tests for exhaustive permutation search on small matrices."""

    def test_permutation_search_identity(self, any_backend: "Backend") -> None:
        """Permutation search on identical matrices should find distance 0."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        # 3x3 matrix (within permutation search threshold of n<=8)
        dist = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        b.eval(dist)

        result = gw._solve_by_permutation_search(dist, dist, 3, b)

        eps = _eps(b, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps
        assert result.converged is True

    def test_permutation_search_permuted(self, any_backend: "Backend") -> None:
        """Permutation search on permuted matrix should find near-zero distance."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        # Original
        dist1 = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        b.eval(dist1)

        # Permuted (swap rows 0,2 and cols 0,2)
        dist2 = b.array([[0.0, 1.5, 2.0], [1.5, 0.0, 1.0], [2.0, 1.0, 0.0]])
        b.eval(dist2)

        result = gw._solve_by_permutation_search(dist1, dist2, 3, b)

        eps = _eps(b, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps


# =============================================================================
# GromovWassersteinDistance Tests - Loss and Gradient
# =============================================================================


class TestLossAndGradient:
    """Tests for loss computation and gradient."""

    def test_init_loss_matrices(self, any_backend: "Backend") -> None:
        """_init_loss_matrices should create valid matrices."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        n, m = 3, 4
        C1 = b.random_normal((n, n))
        C2 = b.random_normal((m, m))
        p = b.ones((n,)) / n
        q = b.ones((m,)) / m
        b.eval(C1, C2, p, q)

        constC, hC1, hC2 = gw._init_loss_matrices(C1, C2, p, q)
        b.eval(constC, hC1, hC2)

        # Check shapes
        assert b.shape(constC) == (n, m)
        assert b.shape(hC1) == (n, n)
        assert b.shape(hC2) == (m, m)

    def test_tensor_product_shape(self, any_backend: "Backend") -> None:
        """_tensor_product should return correct shape."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        n, m = 3, 4
        C1 = b.random_normal((n, n))
        C2 = b.random_normal((m, m))
        p = b.ones((n,)) / n
        q = b.ones((m,)) / m
        T = b.matmul(b.reshape(p, (n, 1)), b.reshape(q, (1, m)))
        b.eval(C1, C2, p, q, T)

        constC, hC1, hC2 = gw._init_loss_matrices(C1, C2, p, q)
        tens = gw._tensor_product(constC, hC1, hC2, T)
        b.eval(tens)

        assert b.shape(tens) == (n, m)

    def test_gw_loss_non_negative(self, any_backend: "Backend") -> None:
        """GW loss should be non-negative for squared loss."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend
        b.random_seed(42)

        n, m = 4, 4
        C1 = b.abs(b.random_normal((n, n)))  # Non-negative distances
        C2 = b.abs(b.random_normal((m, m)))
        p = b.ones((n,)) / n
        q = b.ones((m,)) / m
        T = b.matmul(b.reshape(p, (n, 1)), b.reshape(q, (1, m)))
        b.eval(C1, C2, p, q, T)

        constC, hC1, hC2 = gw._init_loss_matrices(C1, C2, p, q)
        loss = gw._gw_loss(constC, hC1, hC2, T)

        # Loss can be slightly negative due to numerical precision
        eps = _eps(b, float(loss))
        assert float(loss) >= -eps

    def test_gw_gradient_shape(self, any_backend: "Backend") -> None:
        """GW gradient should have same shape as coupling."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        n, m = 3, 4
        C1 = b.random_normal((n, n))
        C2 = b.random_normal((m, m))
        p = b.ones((n,)) / n
        q = b.ones((m,)) / m
        T = b.matmul(b.reshape(p, (n, 1)), b.reshape(q, (1, m)))
        b.eval(C1, C2, p, q, T)

        constC, hC1, hC2 = gw._init_loss_matrices(C1, C2, p, q)
        grad = gw._gw_gradient(constC, hC1, hC2, T)
        b.eval(grad)

        assert b.shape(grad) == (n, m)


# =============================================================================
# GromovWassersteinDistance Tests - Sinkhorn
# =============================================================================


class TestSinkhorn:
    """Tests for Sinkhorn algorithm."""

    def test_solve_linear_ot_marginals(self, any_backend: "Backend") -> None:
        """Sinkhorn should produce valid marginals."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        n, m = 4, 5
        cost = b.abs(b.random_normal((n, m)))
        p = b.ones((n,)) / n
        q = b.ones((m,)) / m
        b.eval(cost, p, q)

        epsilon = division_epsilon(b, cost)
        threshold = division_epsilon(b, cost)
        max_iterations = n * m
        G = gw._solve_linear_ot(cost, p, q, epsilon=epsilon, max_iterations=max_iterations, threshold=threshold)
        b.eval(G)

        # Check row sums
        row_sums = b.sum(G, axis=1)
        b.eval(row_sums)
        for i in range(n):
            expected = 1.0 / n
            value = float(row_sums[i])
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

        # Check column sums
        col_sums = b.sum(G, axis=0)
        b.eval(col_sums)
        for j in range(m):
            expected = 1.0 / m
            value = float(col_sums[j])
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

    def test_solve_linear_ot_empty(self, any_backend: "Backend") -> None:
        """Sinkhorn should handle empty inputs."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        cost = b.zeros((0, 0))
        p = b.zeros((0,))
        q = b.zeros((0,))
        b.eval(cost, p, q)

        epsilon = division_epsilon(b, cost)
        threshold = division_epsilon(b, cost)
        max_iterations = 0
        G = gw._solve_linear_ot(cost, p, q, epsilon=epsilon, max_iterations=max_iterations, threshold=threshold)
        b.eval(G)

        assert b.shape(G) == (0, 0)


# =============================================================================
# GromovWassersteinDistance Tests - Step Size
# =============================================================================


class TestStepSize:
    """Tests for step size computation."""

    def test_compute_step_size_bounded(self, any_backend: "Backend") -> None:
        """Step size should be in [0, 1]."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend
        b.random_seed(42)

        n = 4
        C1 = b.abs(b.random_normal((n, n)))
        C2 = b.abs(b.random_normal((n, n)))
        p = b.ones((n,)) / n
        q = b.ones((n,)) / n
        T = b.matmul(b.reshape(p, (n, 1)), b.reshape(q, (1, n)))
        G = gw._random_coupling(n, n, b)
        b.eval(C1, C2, p, q, T, G)

        constC, hC1, hC2 = gw._init_loss_matrices(C1, C2, p, q)
        alpha = gw._compute_step_size(constC, hC1, hC2, T, G)

        eps = _eps(b, alpha, 1.0, 0.0)
        assert alpha + eps >= 0.0
        assert alpha <= 1.0 + eps


# =============================================================================
# GromovWassersteinDistance Tests - Different Sizes
# =============================================================================


class TestDifferentSizes:
    """Tests for matrices of different sizes."""

    def test_compute_different_sizes(self, any_backend: "Backend") -> None:
        """Should handle distance matrices of different sizes."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        # 3x3 and 4x4 distance matrices
        dist1 = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        dist2 = b.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.5, 2.5],
                [2.0, 1.5, 0.0, 1.0],
                [3.0, 2.5, 1.0, 0.0],
            ]
        )
        b.eval(dist1, dist2)

        # All params derived from dtype - no config needed
        result = gw.compute(dist1, dist2)

        # Should return valid result
        assert _is_finite(result.distance)
        assert result.coupling.shape == (3, 4)


# =============================================================================
# Point Cloud GW Tests
# =============================================================================


class TestPointCloudGW:
    """Tests for point-cloud GW distance via pairwise geodesic distances."""

    def test_compute_identical_points(self, any_backend: "Backend") -> None:
        """Identical point sets should have near-zero distance."""
        b = any_backend

        points = b.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        b.eval(points)

        # All params derived from dtype - no config needed
        gw = GromovWassersteinDistance(backend=b)
        source_dist = gw.compute_pairwise_distances(points)
        target_dist = gw.compute_pairwise_distances(points)
        result = gw.compute(source_dist, target_dist)

        eps = _eps(b, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps

    def test_compute_with_default_backend(self) -> None:
        """Should work with default backend."""
        points_a = [[0.0, 0.0], [1.0, 0.0]]
        points_b = [[0.0, 0.0], [1.0, 0.0]]

        # All params derived from dtype - no config needed
        gw = GromovWassersteinDistance()
        source_dist = gw.compute_pairwise_distances(points_a)
        target_dist = gw.compute_pairwise_distances(points_b)
        result = gw.compute(source_dist, target_dist)

        backend = gw._backend
        eps = _eps(backend, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_point(self, any_backend: "Backend") -> None:
        """Should handle single-point distance matrices."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        dist = b.array([[0.0]])
        b.eval(dist)

        result = gw.compute(dist, dist)

        assert result.distance == 0.0
        assert result.converged is True

    def test_two_points(self, any_backend: "Backend") -> None:
        """Should handle two-point distance matrices."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        dist = b.array([[0.0, 1.0], [1.0, 0.0]])
        b.eval(dist)

        result = gw.compute(dist, dist)

        assert result.distance == 0.0
        assert result.converged is True

    def test_scaled_distances(self, any_backend: "Backend") -> None:
        """GW should be sensitive to scale differences."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        # Original distances
        dist1 = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])

        # Scaled by 2
        dist2 = dist1 * 2.0
        b.eval(dist1, dist2)

        # All params derived from dtype - no config needed
        result = gw.compute(dist1, dist2)

        # Should have non-zero distance due to scale difference
        eps = _eps(b, result.distance, 0.0)
        assert result.distance > eps

    def test_identity_distance_small_matrix(self, any_backend: "Backend") -> None:
        """Identical matrices should have zero distance."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        dist = b.array([[0.0, 1.0], [1.0, 0.0]])
        b.eval(dist)

        result = gw.compute(dist, dist)

        # Should converge to zero distance
        assert result.distance == 0.0

    def test_identical_larger_matrix(self, any_backend: "Backend") -> None:
        """Larger identical matrices should have zero distance."""
        gw = GromovWassersteinDistance(backend=any_backend)
        b = any_backend

        dist = b.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        b.eval(dist)

        result = gw.compute(dist, dist)

        # Should produce valid result with zero distance
        assert result.distance == 0.0


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestGromovWassersteinHypothesis:
    """Hypothesis-based property tests for Gromov-Wasserstein distance."""

    @given(
        n_points=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_self_distance_zero_hypothesis(
        self, n_points: int, seed: int, any_backend: "Backend"
    ):
        """GW distance from a set to itself should be 0."""
        b = any_backend
        b.random_seed(seed)
        gw = GromovWassersteinDistance(backend=b)

        # Generate random points
        points = b.random_normal((n_points, 3))
        b.eval(points)

        # Compute distance matrix
        dist = gw.compute_pairwise_distances(points)

        # Self-distance should be 0
        result = gw.compute(dist, dist)
        eps = _eps(b, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps
        assert result.converged is True

    @given(
        n_points=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_symmetry_hypothesis(
        self, n_points: int, seed: int, any_backend: "Backend"
    ):
        """GW(A, B) should equal GW(B, A) (symmetry)."""
        b = any_backend
        b.random_seed(seed)
        gw = GromovWassersteinDistance(backend=b)

        # Generate two different point sets
        points_a = b.random_normal((n_points, 3))
        b.random_seed(seed + 1000)
        points_b = b.random_normal((n_points, 3))
        b.eval(points_a, points_b)

        dist_a = gw.compute_pairwise_distances(points_a)
        dist_b = gw.compute_pairwise_distances(points_b)

        # All params derived from dtype - no config needed
        result_ab = gw.compute(dist_a, dist_b)
        result_ba = gw.compute(dist_b, dist_a)

        # Should be symmetric within tolerance
        eps = _eps(b, result_ab.distance, result_ba.distance)
        assert abs(result_ab.distance - result_ba.distance) <= eps

    @given(
        n_points=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_distance_non_negative_hypothesis(
        self, n_points: int, seed: int, any_backend: "Backend"
    ):
        """GW distance should always be non-negative."""
        b = any_backend
        b.random_seed(seed)
        gw = GromovWassersteinDistance(backend=b)

        points_a = b.random_normal((n_points, 3))
        b.random_seed(seed + 1000)
        points_b = b.random_normal((n_points, 3))
        b.eval(points_a, points_b)

        dist_a = gw.compute_pairwise_distances(points_a)
        dist_b = gw.compute_pairwise_distances(points_b)

        # All params derived from dtype - no config needed
        result = gw.compute(dist_a, dist_b)

        eps = _eps(b, result.distance, 0.0)
        assert result.distance + eps >= 0.0

    @given(
        n_points=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coupling_valid_marginals_hypothesis(
        self, n_points: int, seed: int, any_backend: "Backend"
    ):
        """Coupling matrix should have valid marginals (row and column sums)."""
        b = any_backend
        b.random_seed(seed)
        gw = GromovWassersteinDistance(backend=b)

        points_a = b.random_normal((n_points, 3))
        b.random_seed(seed + 1000)
        points_b = b.random_normal((n_points, 3))
        b.eval(points_a, points_b)

        dist_a = gw.compute_pairwise_distances(points_a)
        dist_b = gw.compute_pairwise_distances(points_b)

        # All params derived from dtype - no config needed
        result = gw.compute(dist_a, dist_b)

        if not result.converged:
            assume(False)

        coupling = result.coupling
        n = any_backend.shape(coupling)[0]

        # Check row sums ≈ 1/n
        row_sums = any_backend.sum(coupling, axis=1)
        any_backend.eval(row_sums)
        row_list = array_to_list(any_backend, row_sums)
        for value in row_list:
            expected = 1.0 / n
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

        # Check column sums ≈ 1/n
        col_sums = any_backend.sum(coupling, axis=0)
        any_backend.eval(col_sums)
        col_list = array_to_list(any_backend, col_sums)
        for value in col_list:
            expected = 1.0 / n
            eps = _eps(b, value, expected)
            assert abs(value - expected) <= eps

    @given(
        n_points=st.integers(min_value=3, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_permutation_invariance_hypothesis(
        self, n_points: int, seed: int, any_backend: "Backend"
    ):
        """GW distance should be invariant to point permutation."""
        b = any_backend
        b.random_seed(seed)
        gw = GromovWassersteinDistance(backend=b)

        # Generate points
        points = b.random_normal((n_points, 3))
        b.eval(points)

        # Create permutation
        rng = random.Random(seed)
        perm = list(range(n_points))
        rng.shuffle(perm)

        # Permute points
        perm_array = b.array(perm)
        permuted = b.take(points, perm_array, axis=0)
        b.eval(permuted)

        dist_orig = gw.compute_pairwise_distances(points)
        dist_perm = gw.compute_pairwise_distances(permuted)

        # All params derived from dtype - no config needed
        result = gw.compute(dist_orig, dist_perm)

        # Should be near 0 since it's just a permutation
        eps = _eps(b, result.distance, 0.0)
        assert abs(result.distance - 0.0) <= eps
