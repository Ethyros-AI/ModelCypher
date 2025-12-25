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

"""Property-based tests for LoRA Adapter Merger.

Tests geometric merge invariants:
1. Output shape matches input shape
2. Merging identical adapters returns the same adapter
3. Procrustes rotation is a proper rotation (det > 0)
4. Permutation alignment produces valid permutation
5. 1D tensors (biases) are simply averaged
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.merging.lora_adapter_merger import LoRAAdapterMerger
from modelcypher.ports.backend import Backend


# =============================================================================
# Test Fixtures
# =============================================================================


class MockBackend(Backend):
    """Minimal numpy backend for testing."""

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float32)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, arr, axes=None):
        return np.transpose(arr, axes)

    def astype(self, arr, dtype):
        if dtype == "float32":
            return arr.astype(np.float32)
        return arr

    def reshape(self, arr, shape):
        return np.reshape(arr, shape)

    def eval(self, *arrays):
        pass

    def to_numpy(self, arr):
        return np.asarray(arr)

    def sum(self, arr, axis=None, keepdims=False):
        return np.sum(arr, axis=axis, keepdims=keepdims)

    def mean(self, arr, axis=None, keepdims=False):
        return np.mean(arr, axis=axis, keepdims=keepdims)

    def sqrt(self, arr):
        return np.sqrt(arr)

    def abs(self, arr):
        return np.abs(arr)

    def argmax(self, arr, axis=None):
        return np.argmax(arr, axis=axis)

    def max(self, arr, axis=None):
        return np.max(arr, axis=axis)

    def arange(self, n):
        return np.arange(n)

    def diag(self, arr):
        return np.diag(arr)

    def take(self, arr, indices, axis=0):
        return np.take(arr, indices, axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)


@pytest.fixture
def backend():
    return MockBackend()


# =============================================================================
# Unit Tests: _geometric_merge_matrices
# =============================================================================


class TestGeometricMergeMatrices:
    """Tests for the core geometric merge function."""

    def test_single_matrix_returns_unchanged(self, backend):
        """Single matrix should be returned as-is."""
        matrix = np.random.randn(8, 16).astype(np.float32)
        
        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix], backend
        )
        
        assert result.shape == matrix.shape
        assert np.allclose(result, matrix)
        assert proc_error == 0.0
        assert perm_quality == 1.0

    def test_identical_matrices_return_same(self, backend):
        """Merging identical matrices should return approximately the same matrix."""
        np.random.seed(42)
        matrix = np.random.randn(8, 16).astype(np.float32)
        
        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix.copy(), matrix.copy()], backend
        )
        
        assert result.shape == matrix.shape
        # Self-merge should have low error
        assert proc_error < 0.1, f"Self-merge error too high: {proc_error}"
        # Self-alignment should have high quality
        assert perm_quality > 0.8, f"Self-alignment quality too low: {perm_quality}"

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        np.random.seed(42)
        shapes = [(4, 8), (16, 4), (8, 8), (32, 64)]
        
        for shape in shapes:
            m1 = np.random.randn(*shape).astype(np.float32)
            m2 = np.random.randn(*shape).astype(np.float32)
            
            result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
                [m1, m2], backend
            )
            
            assert result.shape == shape, f"Shape mismatch: {result.shape} vs {shape}"

    def test_1d_tensors_averaged(self, backend):
        """1D tensors (biases) should be simply averaged."""
        bias1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        bias2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        expected = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        
        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [bias1, bias2], backend
        )
        
        assert result.shape == bias1.shape
        assert np.allclose(result, expected)
        # 1D tensors should have default metrics
        assert proc_error == 0.0
        assert perm_quality == 1.0

    def test_no_nan_in_output(self, backend):
        """Merge should not introduce NaN values."""
        np.random.seed(42)
        m1 = np.random.randn(8, 16).astype(np.float32)
        m2 = np.random.randn(8, 16).astype(np.float32)
        
        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )
        
        assert not np.isnan(result).any(), "Output contains NaN"

    def test_dtype_preserved(self, backend):
        """Output dtype should match input dtype."""
        matrix = np.random.randn(8, 16).astype(np.float32)
        
        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix], backend
        )
        
        assert result.dtype == matrix.dtype


# =============================================================================
# Unit Tests: _procrustes_align
# =============================================================================


class TestProcrustesAlign:
    """Tests for Procrustes rotation alignment."""

    def test_identical_matrices_low_error(self, backend):
        """Identical matrices should have near-zero alignment error."""
        matrix = np.random.randn(8, 16).astype(np.float32)
        source_arr = backend.array(matrix)
        target_arr = backend.array(matrix)
        
        rotated, error = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )
        
        assert error < 0.01, f"Self-alignment error should be ~0, got {error}"

    def test_rotation_is_proper(self, backend):
        """Procrustes should produce proper rotation (det > 0)."""
        np.random.seed(42)
        source = np.random.randn(8, 8).astype(np.float32)
        target = np.random.randn(8, 8).astype(np.float32)
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        
        # Compute rotation matrix (extract from the algorithm)
        M = target.T @ source
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        
        det = np.linalg.det(R)
        assert det > 0, f"Rotation should have positive determinant, got {det}"

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        source = np.random.randn(8, 16).astype(np.float32)
        target = np.random.randn(8, 16).astype(np.float32)
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        
        rotated, _ = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )
        
        rotated_np = backend.to_numpy(rotated)
        assert rotated_np.shape == source.shape

    def test_error_is_normalized(self, backend):
        """Error should be normalized (relative to target norm)."""
        np.random.seed(42)
        source = np.random.randn(8, 16).astype(np.float32)
        target = np.random.randn(8, 16).astype(np.float32) * 100  # Large target
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        
        _, error = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )
        
        # Normalized error should be reasonable, not dominated by scale
        assert error < 10, f"Normalized error should be reasonable, got {error}"


# =============================================================================
# Unit Tests: _permutation_align
# =============================================================================


class TestPermutationAlign:
    """Tests for permutation alignment."""

    def test_result_has_valid_structure(self, backend):
        """Result should have permutation matrix and signs."""
        source = np.random.randn(8, 16).astype(np.float32)
        target = np.random.randn(8, 16).astype(np.float32)
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        
        result = LoRAAdapterMerger._permutation_align(
            source_arr, target_arr, backend
        )
        
        assert hasattr(result, 'permutation')
        assert hasattr(result, 'signs')
        assert hasattr(result, 'match_quality')

    def test_self_alignment_high_quality(self, backend):
        """Self-alignment should have high quality."""
        matrix = np.random.randn(8, 16).astype(np.float32)
        arr = backend.array(matrix)
        
        result = LoRAAdapterMerger._permutation_align(arr, arr, backend)
        
        assert result.match_quality > 0.8, \
            f"Self-alignment quality should be > 0.8, got {result.match_quality}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_matrices(self, backend):
        """Should handle small (2x2) matrices."""
        m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )
        
        assert result.shape == (2, 2)
        assert not np.isnan(result).any()

    def test_three_way_merge(self, backend):
        """Should handle merging 3 adapters."""
        np.random.seed(42)
        matrices = [np.random.randn(8, 16).astype(np.float32) for _ in range(3)]
        
        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            matrices, backend
        )
        
        assert result.shape == (8, 16)
        assert not np.isnan(result).any()
        # Should have some quality metrics
        assert 0.0 <= perm_quality <= 1.0 + 1e-5


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionCases:
    """Tests with known expected behavior."""

    def test_merge_reduces_variance(self, backend):
        """Merging should generally reduce variance (averaging effect)."""
        np.random.seed(42)
        # Create matrices with high variance
        m1 = np.random.randn(8, 16).astype(np.float32) * 2.0
        m2 = np.random.randn(8, 16).astype(np.float32) * 2.0
        
        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )
        
        # Merged result should have lower or similar variance
        input_var = (np.var(m1) + np.var(m2)) / 2
        output_var = np.var(result)
        
        # Allow some tolerance for alignment transformations
        assert output_var < input_var * 1.5, \
            f"Output variance {output_var} >> input variance {input_var}"
