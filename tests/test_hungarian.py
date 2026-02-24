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

"""Tests for hungarian.py - Hungarian algorithm for optimal assignment.

Tests cover:
- hungarian_assignment() GPU-accelerated function
- hungarian_assignment_list() CPU fallback
- clear_hungarian_cache() cache management
- Edge cases: 1x1, 2x2, identity costs
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.hungarian import (
    clear_hungarian_cache,
    hungarian_assignment,
    hungarian_assignment_list,
)

# =============================================================================
# hungarian_assignment Tests
# =============================================================================


class TestHungarianAssignment:
    """Tests for hungarian_assignment function."""

    def test_identity_cost_matrix(self):
        """Identity cost matrix assigns i->i."""
        backend = get_default_backend()
        # Cost[i][i] = 0, else = 1 => optimal is diagonal
        cost = backend.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])

        result = hungarian_assignment(cost, backend, use_cache=False)

        assert list(result) == [0, 1, 2]

    def test_2x2_matrix(self):
        """2x2 matrix optimal assignment."""
        backend = get_default_backend()
        # Cost[0][1] = 0, Cost[1][0] = 0 => swap is optimal
        cost = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        result = hungarian_assignment(cost, backend, use_cache=False)

        # Optimal: 0->1, 1->0
        assert list(result) == [1, 0]

    def test_1x1_matrix(self):
        """1x1 matrix trivial assignment."""
        backend = get_default_backend()
        cost = backend.array([[5.0]])

        result = hungarian_assignment(cost, backend, use_cache=False)

        assert list(result) == [0]

    def test_result_is_permutation(self):
        """Result is a valid permutation (each target used once)."""
        backend = get_default_backend()
        n = 5
        cost_list = [[(i + j) % n for j in range(n)] for i in range(n)]
        cost = backend.array(cost_list)

        result = hungarian_assignment(cost, backend, use_cache=False)

        # Check permutation: all targets 0..n-1 appear exactly once
        assert sorted(result) == list(range(n))


# =============================================================================
# hungarian_assignment_list Tests
# =============================================================================


class TestHungarianAssignmentList:
    """Tests for hungarian_assignment_list function (CPU)."""

    def test_identity_cost_matrix(self):
        """Identity cost matrix assigns i->i."""
        cost = [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]

        result = hungarian_assignment_list(cost)

        assert result == [0, 1, 2]

    def test_2x2_swap(self):
        """2x2 matrix with swap optimal."""
        cost = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        result = hungarian_assignment_list(cost)

        assert result == [1, 0]

    def test_result_is_permutation(self):
        """Result is a valid permutation."""
        n = 4
        cost = [[(i * j) % n for j in range(n)] for i in range(n)]

        result = hungarian_assignment_list(cost)

        assert sorted(result) == list(range(n))


# =============================================================================
# Cache Tests
# =============================================================================


class TestHungarianCache:
    """Tests for Hungarian algorithm cache."""

    def test_clear_cache(self):
        """clear_hungarian_cache clears the cache."""
        backend = get_default_backend()
        cost = backend.array([[0.0, 1.0], [1.0, 0.0]])

        # Run once to populate cache
        hungarian_assignment(cost, backend, use_cache=True)

        # Clear and verify no error
        clear_hungarian_cache()

    def test_cache_consistent_results(self):
        """Cached results match uncached results."""
        backend = get_default_backend()
        cost = backend.array([[0.0, 1.0], [1.0, 0.0]])

        clear_hungarian_cache()
        result1 = hungarian_assignment(cost, backend, use_cache=True)
        result2 = hungarian_assignment(cost, backend, use_cache=True)

        assert list(result1) == list(result2)
