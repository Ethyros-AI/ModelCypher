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

"""
VectorMath Unit Tests.

Tests for vector math utilities that support both Python lists and MLX arrays.
Ported from Swift VectorMath tests to ensure parity.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.vector_math import (
    SparseVectorMath,
    VectorMath,
    _len,
    _to_list,
)


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_to_list_from_list(self):
        """_to_list returns list unchanged."""
        lst = [1.0, 2.0, 3.0]
        assert _to_list(lst) == lst

    def test_to_list_from_mlx_array(self):
        """_to_list converts MLX array to list."""
        try:
            import mlx.core as mx

            arr = mx.array([1.0, 2.0, 3.0])
            result = _to_list(arr)
            assert result == [1.0, 2.0, 3.0]
        except ImportError:
            pytest.skip("MLX not available")

    def test_len_from_list(self):
        """_len returns correct length for list."""
        assert _len([1.0, 2.0, 3.0]) == 3
        assert _len([]) == 0

    def test_len_from_mlx_array(self):
        """_len returns correct length for MLX array."""
        try:
            import mlx.core as mx

            arr = mx.array([1.0, 2.0, 3.0, 4.0])
            assert _len(arr) == 4
        except ImportError:
            pytest.skip("MLX not available")


class TestVectorMathDot:
    """Tests for VectorMath.dot()."""

    def test_dot_basic(self):
        """Basic dot product computation."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = VectorMath.dot(a, b)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert result == pytest.approx(32.0)

    def test_dot_orthogonal(self):
        """Dot product of orthogonal vectors is zero."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = VectorMath.dot(a, b)
        assert result == pytest.approx(0.0)

    def test_dot_empty_vectors(self):
        """Dot product of empty vectors returns None."""
        assert VectorMath.dot([], []) is None

    def test_dot_mismatched_lengths(self):
        """Dot product of mismatched lengths returns None."""
        assert VectorMath.dot([1.0, 2.0], [1.0]) is None

    def test_dot_with_mlx_arrays(self):
        """Dot product works with MLX arrays."""
        try:
            import mlx.core as mx

            a = mx.array([1.0, 2.0, 3.0])
            b = mx.array([4.0, 5.0, 6.0])
            result = VectorMath.dot(a, b)
            assert result == pytest.approx(32.0)
        except ImportError:
            pytest.skip("MLX not available")


class TestVectorMathL2Norm:
    """Tests for VectorMath.l2_norm()."""

    def test_l2_norm_unit_vector(self):
        """L2 norm of unit vector is 1."""
        a = [1.0, 0.0, 0.0]
        assert VectorMath.l2_norm(a) == pytest.approx(1.0)

    def test_l2_norm_simple(self):
        """L2 norm computation - 3-4-5 triangle."""
        a = [3.0, 4.0]
        assert VectorMath.l2_norm(a) == pytest.approx(5.0)

    def test_l2_norm_empty(self):
        """L2 norm of empty vector returns None."""
        assert VectorMath.l2_norm([]) is None

    def test_l2_norm_zero_vector(self):
        """L2 norm of zero vector returns None."""
        assert VectorMath.l2_norm([0.0, 0.0, 0.0]) is None

    def test_l2_norm_with_mlx_array(self):
        """L2 norm works with MLX arrays."""
        try:
            import mlx.core as mx

            a = mx.array([3.0, 4.0])
            result = VectorMath.l2_norm(a)
            assert result == pytest.approx(5.0)
        except ImportError:
            pytest.skip("MLX not available")


class TestVectorMathL2Normalized:
    """Tests for VectorMath.l2_normalized()."""

    def test_l2_normalized_basic(self):
        """L2 normalized vector has unit norm."""
        a = [3.0, 4.0]
        result = VectorMath.l2_normalized(a)
        # Should be [0.6, 0.8]
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.8)
        # Norm should be 1
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)

    def test_l2_normalized_zero_vector(self):
        """L2 normalizing zero vector returns original."""
        a = [0.0, 0.0]
        result = VectorMath.l2_normalized(a)
        assert result == [0.0, 0.0]

    def test_l2_normalized_empty(self):
        """L2 normalizing empty vector returns empty."""
        result = VectorMath.l2_normalized([])
        assert result == []


class TestVectorMathCosineSimilarity:
    """Tests for VectorMath.cosine_similarity()."""

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors is 1."""
        a = [1.0, 2.0, 3.0]
        result = VectorMath.cosine_similarity(a, a)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_opposite(self):
        """Cosine similarity of opposite vectors is -1."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = VectorMath.cosine_similarity(a, b)
        assert result == pytest.approx(-1.0)

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors is 0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = VectorMath.cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        """Cosine similarity of empty vectors returns None."""
        assert VectorMath.cosine_similarity([], []) is None

    def test_cosine_similarity_mismatched(self):
        """Cosine similarity of mismatched lengths returns None."""
        assert VectorMath.cosine_similarity([1.0, 2.0], [1.0]) is None

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector returns None."""
        assert VectorMath.cosine_similarity([0.0, 0.0], [1.0, 2.0]) is None

    def test_cosine_similarity_with_mlx_arrays(self):
        """Cosine similarity works with MLX arrays."""
        try:
            import mlx.core as mx

            a = mx.array([1.0, 0.0])
            b = mx.array([1.0, 0.0])
            result = VectorMath.cosine_similarity(a, b)
            assert result == pytest.approx(1.0)
        except ImportError:
            pytest.skip("MLX not available")

    def test_cosine_similarity_mixed_list_and_mlx(self):
        """Cosine similarity works with mixed list and MLX array."""
        try:
            import mlx.core as mx

            a = [1.0, 2.0, 3.0]
            b = mx.array([1.0, 2.0, 3.0])
            result = VectorMath.cosine_similarity(a, b)
            assert result == pytest.approx(1.0)
        except ImportError:
            pytest.skip("MLX not available")


class TestSparseVectorMathL2Norm:
    """Tests for SparseVectorMath.l2_norm()."""

    def test_sparse_l2_norm_basic(self):
        """Basic sparse L2 norm."""
        v = {"a": 3.0, "b": 4.0}
        result = SparseVectorMath.l2_norm(v)
        assert result == pytest.approx(5.0)

    def test_sparse_l2_norm_empty(self):
        """Sparse L2 norm of empty dict returns None."""
        assert SparseVectorMath.l2_norm({}) is None


class TestSparseVectorMathCosineSimilarity:
    """Tests for SparseVectorMath.cosine_similarity()."""

    def test_sparse_cosine_identical(self):
        """Sparse cosine similarity of identical vectors is 1."""
        v = {"a": 1.0, "b": 2.0}
        result = SparseVectorMath.cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_sparse_cosine_orthogonal(self):
        """Sparse cosine similarity of orthogonal vectors is 0."""
        a = {"x": 1.0}
        b = {"y": 1.0}
        result = SparseVectorMath.cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_sparse_cosine_partial_overlap(self):
        """Sparse cosine similarity with partial overlap."""
        a = {"x": 1.0, "y": 1.0}
        b = {"x": 1.0, "z": 1.0}
        # dot = 1*1 = 1, norms = sqrt(2) each
        # cos = 1 / (sqrt(2) * sqrt(2)) = 1/2
        result = SparseVectorMath.cosine_similarity(a, b)
        assert result == pytest.approx(0.5)

    def test_sparse_cosine_empty(self):
        """Sparse cosine similarity of empty dicts returns None."""
        assert SparseVectorMath.cosine_similarity({}, {}) is None
        assert SparseVectorMath.cosine_similarity({"a": 1.0}, {}) is None
