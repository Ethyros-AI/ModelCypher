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
    BackendVectorMath,
    SparseVectorMath,
    VectorMath,
    _len,
    _to_list,
    get_vector_math,
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


class TestVectorMathSlerp:
    """Tests for VectorMath.slerp() - Spherical Linear Interpolation."""

    def test_slerp_orthogonal_midpoint(self):
        """SLERP between orthogonal vectors at t=0.5 gives 45-degree result."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]
        result = VectorMath.slerp(v0, v1, 0.5)
        # At 45 degrees: [cos(45), sin(45), 0] = [0.7071, 0.7071, 0]
        assert result is not None
        assert result[0] == pytest.approx(0.7071, rel=0.01)
        assert result[1] == pytest.approx(0.7071, rel=0.01)
        assert result[2] == pytest.approx(0.0)

    def test_slerp_t0_returns_v0(self):
        """SLERP at t=0 returns the first vector."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]
        result = VectorMath.slerp(v0, v1, 0.0)
        assert result is not None
        assert result == [1.0, 0.0, 0.0]

    def test_slerp_t1_returns_v1(self):
        """SLERP at t=1 returns the second vector."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]
        result = VectorMath.slerp(v0, v1, 1.0)
        assert result is not None
        assert result == [0.0, 1.0, 0.0]

    def test_slerp_preserves_unit_norm(self):
        """SLERP of unit vectors has unit norm when interpolate_magnitude=True."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = VectorMath.slerp(v0, v1, t)
            assert result is not None
            norm = VectorMath.l2_norm(result)
            assert norm == pytest.approx(1.0, rel=0.001)

    def test_slerp_magnitude_interpolation(self):
        """SLERP interpolates magnitudes linearly."""
        v0 = [2.0, 0.0, 0.0]  # magnitude 2
        v1 = [0.0, 4.0, 0.0]  # magnitude 4
        result = VectorMath.slerp(v0, v1, 0.5)
        assert result is not None
        # Expected magnitude: (2 + 4) / 2 = 3
        norm = VectorMath.l2_norm(result)
        assert norm == pytest.approx(3.0, rel=0.001)

    def test_slerp_no_magnitude_interpolation(self):
        """SLERP without magnitude interpolation returns unit vector."""
        v0 = [2.0, 0.0, 0.0]
        v1 = [0.0, 4.0, 0.0]
        result = VectorMath.slerp(v0, v1, 0.5, interpolate_magnitude=False)
        assert result is not None
        norm = VectorMath.l2_norm(result)
        assert norm == pytest.approx(1.0, rel=0.001)

    def test_slerp_near_parallel_fallback(self):
        """SLERP falls back to linear for near-parallel vectors."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [1.0, 0.0001, 0.0]  # Almost parallel
        result = VectorMath.slerp(v0, v1, 0.5)
        assert result is not None
        # Should be close to linear interpolation
        assert result[0] == pytest.approx(1.0, rel=0.01)
        assert result[1] == pytest.approx(0.00005, rel=0.1)

    def test_slerp_empty_vectors(self):
        """SLERP of empty vectors returns None."""
        assert VectorMath.slerp([], [], 0.5) is None

    def test_slerp_mismatched_lengths(self):
        """SLERP of mismatched lengths returns None."""
        assert VectorMath.slerp([1.0, 2.0], [1.0], 0.5) is None

    def test_slerp_zero_vector(self):
        """SLERP with zero vector returns None."""
        assert VectorMath.slerp([0.0, 0.0], [1.0, 0.0], 0.5) is None

    def test_slerp_with_mlx_arrays(self):
        """SLERP works with MLX arrays."""
        try:
            import mlx.core as mx

            v0 = mx.array([1.0, 0.0, 0.0])
            v1 = mx.array([0.0, 1.0, 0.0])
            result = VectorMath.slerp(v0, v1, 0.5)
            assert result is not None
            assert result[0] == pytest.approx(0.7071, rel=0.01)
            assert result[1] == pytest.approx(0.7071, rel=0.01)
        except ImportError:
            pytest.skip("MLX not available")


class TestVectorMathSlerpBatch:
    """Tests for VectorMath.slerp_batch() - batch SLERP for weight merging."""

    def test_slerp_batch_basic(self):
        """Basic batch SLERP on matching weight dicts."""
        weights_a = {"layer1": [1.0, 0.0], "layer2": [0.0, 1.0]}
        weights_b = {"layer1": [0.0, 1.0], "layer2": [1.0, 0.0]}
        result = VectorMath.slerp_batch(weights_a, weights_b, 0.5)
        assert "layer1" in result
        assert "layer2" in result
        # Both should be 45-degree interpolations
        assert result["layer1"][0] == pytest.approx(0.7071, rel=0.01)
        assert result["layer1"][1] == pytest.approx(0.7071, rel=0.01)

    def test_slerp_batch_missing_key_in_b(self):
        """Batch SLERP includes keys only in weights_a unchanged."""
        weights_a = {"layer1": [1.0, 0.0], "layer2": [0.5, 0.5]}
        weights_b = {"layer1": [0.0, 1.0]}
        result = VectorMath.slerp_batch(weights_a, weights_b, 0.5)
        # layer1 should be interpolated
        assert result["layer1"][0] == pytest.approx(0.7071, rel=0.01)
        # layer2 should be unchanged from weights_a
        assert result["layer2"] == [0.5, 0.5]

    def test_slerp_batch_missing_key_in_a(self):
        """Batch SLERP includes keys only in weights_b unchanged."""
        weights_a = {"layer1": [1.0, 0.0]}
        weights_b = {"layer1": [0.0, 1.0], "layer2": [0.3, 0.7]}
        result = VectorMath.slerp_batch(weights_a, weights_b, 0.5)
        # layer2 should be unchanged from weights_b
        assert result["layer2"] == [0.3, 0.7]

    def test_slerp_batch_empty_dicts(self):
        """Batch SLERP of empty dicts returns empty dict."""
        result = VectorMath.slerp_batch({}, {}, 0.5)
        assert result == {}


class TestBackendVectorMath:
    """Tests for GPU-accelerated BackendVectorMath."""

    @pytest.fixture
    def backend(self):
        """Get the default backend for testing."""
        from modelcypher.core.domain._backend import get_default_backend
        return get_default_backend()

    @pytest.fixture
    def bvm(self, backend):
        """Get BackendVectorMath instance."""
        return BackendVectorMath(backend)

    def test_dot_product(self, backend, bvm):
        """Test GPU-accelerated dot product."""
        v1 = backend.array([1.0, 2.0, 3.0])
        v2 = backend.array([4.0, 5.0, 6.0])
        result = bvm.dot(v1, v2)
        assert result == pytest.approx(32.0)

    def test_l2_norm(self, backend, bvm):
        """Test GPU-accelerated L2 norm."""
        v = backend.array([3.0, 4.0])
        result = bvm.l2_norm(v)
        assert result == pytest.approx(5.0)

    def test_cosine_similarity(self, backend, bvm):
        """Test GPU-accelerated cosine similarity."""
        v1 = backend.array([1.0, 0.0])
        v2 = backend.array([0.0, 1.0])
        result = bvm.cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0)

    def test_slerp_orthogonal(self, backend, bvm):
        """Test GPU-accelerated SLERP on orthogonal vectors."""
        v0 = backend.array([1.0, 0.0, 0.0])
        v1 = backend.array([0.0, 1.0, 0.0])
        result = bvm.slerp(v0, v1, 0.5)
        backend.eval(result)
        result_list = result.tolist()
        assert result_list[0] == pytest.approx(0.7071, rel=0.01)
        assert result_list[1] == pytest.approx(0.7071, rel=0.01)
        assert result_list[2] == pytest.approx(0.0)

    def test_slerp_magnitude_interpolation(self, backend, bvm):
        """Test SLERP interpolates magnitudes correctly."""
        v0 = backend.array([2.0, 0.0, 0.0])
        v1 = backend.array([0.0, 4.0, 0.0])
        result = bvm.slerp(v0, v1, 0.5)
        norm = bvm.l2_norm(result)
        assert norm == pytest.approx(3.0, rel=0.001)

    def test_slerp_from_list(self, backend, bvm):
        """Test SLERP auto-converts lists to backend arrays."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]
        result = bvm.slerp(v0, v1, 0.5)
        backend.eval(result)
        result_list = result.tolist()
        assert result_list[0] == pytest.approx(0.7071, rel=0.01)

    def test_precision_matches_pure_python(self, backend, bvm):
        """Verify Backend results match pure Python within precision."""
        v0 = [1.0, 0.0, 0.0]
        v1 = [0.0, 1.0, 0.0]

        # Pure Python
        py_result = VectorMath.slerp(v0, v1, 0.5)

        # Backend
        v0_arr = backend.array(v0)
        v1_arr = backend.array(v1)
        gpu_result = bvm.slerp(v0_arr, v1_arr, 0.5)
        backend.eval(gpu_result)
        gpu_result_list = gpu_result.tolist()

        # Check precision (should be within float32 epsilon)
        for py_val, gpu_val in zip(py_result, gpu_result_list):
            assert py_val == pytest.approx(gpu_val, abs=1e-6)


class TestGetVectorMath:
    """Tests for the get_vector_math factory function."""

    def test_returns_vectormath_without_backend(self):
        """Without backend, returns pure Python VectorMath."""
        vm = get_vector_math(None)
        assert isinstance(vm, VectorMath)

    def test_returns_backendvectormath_with_backend(self):
        """With backend, returns GPU-accelerated BackendVectorMath."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()
        vm = get_vector_math(backend)
        assert isinstance(vm, BackendVectorMath)
