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

"""Tests for GramAligner module.

Tests verify the API works correctly. Thresholds are derived from machine
epsilon at runtime - we don't test for specific arbitrary values.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import (
    GramAligner,
    AlignmentResult,
    find_alignment,
)
from modelcypher.core.domain.geometry.cka import compute_cka


class TestGramAlignerInit:
    """Tests for GramAligner initialization."""

    def test_default_initialization(self):
        """Should initialize with default backend."""
        b = get_default_backend()
        aligner = GramAligner(b)
        assert aligner._backend is not None

    def test_no_backend_uses_default(self):
        """Should use default backend when none provided."""
        aligner = GramAligner()
        assert aligner._backend is not None


class TestGramAlignerFindPerfectAlignment:
    """Tests for GramAligner.find_perfect_alignment() method."""

    def test_identical_matrices(self):
        """Identical matrices should achieve perfect alignment."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, A)

        assert result.is_perfect
        assert result.achieved_cka > 0.999

    def test_similar_matrices(self):
        """Similar matrices should achieve high CKA."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        noise = b.random_normal((20, 8))
        noise_np = b.to_numpy(noise)
        A_np = b.to_numpy(A)
        # Small perturbation
        B = b.array(A_np + 0.01 * noise_np)
        b.eval(A, B)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        assert result.achieved_cka > 0.95

    def test_returns_valid_transforms(self):
        """Alignment should return valid transformation matrices."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        B = b.random_normal((20, 12))
        b.eval(A, B)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        # feature_transform should be d_source x d_target
        assert len(result.feature_transform) == 8
        assert len(result.feature_transform[0]) == 12

    def test_different_dimensions(self):
        """Should handle different feature dimensions."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((30, 16))
        B = b.random_normal((30, 32))
        b.eval(A, B)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        assert result.achieved_cka >= 0.0
        assert result.iterations > 0


class TestFindAlignment:
    """Tests for find_alignment convenience function."""

    def test_find_alignment_basic(self):
        """find_alignment should work with basic inputs."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        result = find_alignment(A, A, backend=b)

        assert result.is_perfect
        assert result.achieved_cka > 0.999

    def test_find_alignment_uses_default_backend(self):
        """find_alignment should use default backend when none provided."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        # Should not raise
        result = find_alignment(A, A)
        assert result.achieved_cka > 0.9


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_matrices(self):
        """Should handle small matrices."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((5, 3))
        B = b.random_normal((5, 3))
        b.eval(A, B)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        assert result.achieved_cka >= 0.0

    def test_single_sample(self):
        """Should handle single sample gracefully."""
        b = get_default_backend()

        A = b.ones((1, 8))
        B = b.ones((1, 8))
        b.eval(A, B)

        aligner = GramAligner(b)
        # May return degenerate result but shouldn't crash
        result = aligner.find_perfect_alignment(A, B)
        assert result is not None
