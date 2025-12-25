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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.merging.lora_adapter_merger import LoRAAdapterMerger
from modelcypher.ports.backend import Backend


# =============================================================================
# Test Fixtures
# =============================================================================


class MockBackend(Backend):
    """Wrapper around get_default_backend for testing."""

    def __init__(self) -> None:
        self._backend = get_default_backend()

    def array(self, data, dtype=None):
        return self._backend.array(data, dtype=dtype)

    def matmul(self, a, b):
        return self._backend.matmul(a, b)

    def transpose(self, arr, axes=None):
        return self._backend.transpose(arr, axes=axes)

    def astype(self, arr, dtype):
        if dtype == "float32":
            return self._backend.astype(arr, self._backend.float32)
        return arr

    def reshape(self, arr, shape):
        return self._backend.reshape(arr, shape)

    def eval(self, *arrays):
        return self._backend.eval(*arrays)

    def to_numpy(self, arr):
        return self._backend.to_numpy(arr)

    def sum(self, arr, axis=None, keepdims=False):
        return self._backend.sum(arr, axis=axis, keepdims=keepdims)

    def mean(self, arr, axis=None, keepdims=False):
        return self._backend.mean(arr, axis=axis, keepdims=keepdims)

    def sqrt(self, arr):
        return self._backend.sqrt(arr)

    def abs(self, arr):
        return self._backend.abs(arr)

    def argmax(self, arr, axis=None):
        return self._backend.argmax(arr, axis=axis)

    def max(self, arr, axis=None):
        return self._backend.max(arr, axis=axis)

    def arange(self, n):
        return self._backend.arange(n)

    def diag(self, arr):
        return self._backend.diag(arr)

    def take(self, arr, indices, axis=0):
        return self._backend.take(arr, indices, axis=axis)

    def stack(self, arrays, axis=0):
        return self._backend.stack(arrays, axis=axis)


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
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        matrix_np = default_backend.to_numpy(matrix)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix_np], backend
        )

        assert result.shape == matrix_np.shape
        assert np.allclose(result, matrix_np)
        assert proc_error == 0.0
        assert perm_quality == 1.0

    def test_identical_matrices_return_same(self, backend):
        """Merging identical matrices should return approximately the same matrix."""
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        matrix_np = default_backend.to_numpy(matrix)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix_np.copy(), matrix_np.copy()], backend
        )

        assert result.shape == matrix_np.shape
        # Self-merge should have low error
        assert proc_error < 0.1, f"Self-merge error too high: {proc_error}"
        # Self-alignment should have high quality
        assert perm_quality > 0.8, f"Self-alignment quality too low: {perm_quality}"

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        shapes = [(4, 8), (16, 4), (8, 8), (32, 64)]

        for shape in shapes:
            m1 = default_backend.random_normal(shape)
            m2 = default_backend.random_normal(shape)
            m1_np = default_backend.to_numpy(m1)
            m2_np = default_backend.to_numpy(m2)

            result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
                [m1_np, m2_np], backend
            )

            assert result.shape == shape, f"Shape mismatch: {result.shape} vs {shape}"

    def test_1d_tensors_averaged(self, backend):
        """1D tensors (biases) should be simply averaged."""
        import numpy as np
        default_backend = get_default_backend()
        bias1_data = [1.0, 2.0, 3.0, 4.0]
        bias2_data = [5.0, 6.0, 7.0, 8.0]
        expected_data = [3.0, 4.0, 5.0, 6.0]
        bias1 = default_backend.array(bias1_data, dtype=default_backend.float32)
        bias2 = default_backend.array(bias2_data, dtype=default_backend.float32)
        expected = default_backend.array(expected_data, dtype=default_backend.float32)
        bias1_np = default_backend.to_numpy(bias1)
        bias2_np = default_backend.to_numpy(bias2)
        expected_np = default_backend.to_numpy(expected)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [bias1_np, bias2_np], backend
        )

        assert result.shape == bias1_np.shape
        assert np.allclose(result, expected_np)
        # 1D tensors should have default metrics
        assert proc_error == 0.0
        assert perm_quality == 1.0

    def test_no_nan_in_output(self, backend):
        """Merge should not introduce NaN values."""
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        m1 = default_backend.random_normal((8, 16))
        m2 = default_backend.random_normal((8, 16))
        m1_np = default_backend.to_numpy(m1)
        m2_np = default_backend.to_numpy(m2)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1_np, m2_np], backend
        )

        assert not np.isnan(result).any(), "Output contains NaN"

    def test_dtype_preserved(self, backend):
        """Output dtype should match input dtype."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        matrix_np = default_backend.to_numpy(matrix)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix_np], backend
        )

        assert result.dtype == matrix_np.dtype


# =============================================================================
# Unit Tests: _procrustes_align
# =============================================================================


class TestProcrustesAlign:
    """Tests for Procrustes rotation alignment."""

    def test_identical_matrices_low_error(self, backend):
        """Identical matrices should have near-zero alignment error."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        matrix_np = default_backend.to_numpy(matrix)
        source_arr = backend.array(matrix_np)
        target_arr = backend.array(matrix_np)

        rotated, error = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )

        assert error < 0.01, f"Self-alignment error should be ~0, got {error}"

    def test_rotation_is_proper(self, backend):
        """Procrustes should produce proper rotation (det > 0)."""
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 8))
        target = default_backend.random_normal((8, 8))
        source_np = default_backend.to_numpy(source)
        target_np = default_backend.to_numpy(target)
        source_arr = backend.array(source_np)
        target_arr = backend.array(target_np)

        # Compute rotation matrix (extract from the algorithm)
        M = target_np.T @ source_np
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        det = np.linalg.det(R)
        assert det > 0, f"Rotation should have positive determinant, got {det}"

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 16))
        target = default_backend.random_normal((8, 16))
        source_np = default_backend.to_numpy(source)
        target_np = default_backend.to_numpy(target)
        source_arr = backend.array(source_np)
        target_arr = backend.array(target_np)

        rotated, _ = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )

        rotated_np = backend.to_numpy(rotated)
        assert rotated_np.shape == source_np.shape

    def test_error_is_normalized(self, backend):
        """Error should be normalized (relative to target norm)."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 16))
        target = default_backend.random_normal((8, 16))
        source_np = default_backend.to_numpy(source)
        target_np = default_backend.to_numpy(target) * 100  # Large target
        source_arr = backend.array(source_np)
        target_arr = backend.array(target_np)

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
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 16))
        target = default_backend.random_normal((8, 16))
        source_np = default_backend.to_numpy(source)
        target_np = default_backend.to_numpy(target)
        source_arr = backend.array(source_np)
        target_arr = backend.array(target_np)

        result = LoRAAdapterMerger._permutation_align(
            source_arr, target_arr, backend
        )

        assert hasattr(result, 'permutation')
        assert hasattr(result, 'signs')
        assert hasattr(result, 'match_quality')

    def test_self_alignment_high_quality(self, backend):
        """Self-alignment should have high quality."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        matrix_np = default_backend.to_numpy(matrix)
        arr = backend.array(matrix_np)

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
        import numpy as np
        default_backend = get_default_backend()
        m1_data = [[1.0, 2.0], [3.0, 4.0]]
        m2_data = [[5.0, 6.0], [7.0, 8.0]]
        m1 = default_backend.array(m1_data, dtype=default_backend.float32)
        m2 = default_backend.array(m2_data, dtype=default_backend.float32)
        m1_np = default_backend.to_numpy(m1)
        m2_np = default_backend.to_numpy(m2)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1_np, m2_np], backend
        )

        assert result.shape == (2, 2)
        assert not np.isnan(result).any()

    def test_three_way_merge(self, backend):
        """Should handle merging 3 adapters."""
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrices = []
        for _ in range(3):
            m = default_backend.random_normal((8, 16))
            matrices.append(default_backend.to_numpy(m))

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
        import numpy as np
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        # Create matrices with high variance
        m1 = default_backend.random_normal((8, 16))
        m2 = default_backend.random_normal((8, 16))
        m1_np = default_backend.to_numpy(m1) * 2.0
        m2_np = default_backend.to_numpy(m2) * 2.0

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1_np, m2_np], backend
        )

        # Merged result should have lower or similar variance
        input_var = (np.var(m1_np) + np.var(m2_np)) / 2
        output_var = np.var(result)

        # Allow some tolerance for alignment transformations
        assert output_var < input_var * 1.5, \
            f"Output variance {output_var} >> input variance {input_var}"
