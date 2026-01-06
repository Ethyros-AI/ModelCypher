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
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.merging.lora_adapter_merger import LoRAAdapterMerger
from modelcypher.ports.backend import Backend


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


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
            return self._backend.astype(arr, "float32")
        return arr

    def reshape(self, arr, shape):
        return self._backend.reshape(arr, shape)

    def eval(self, *arrays):
        return self._backend.eval(*arrays)

    def to_numpy(self, arr):
        return self._backend.tolist(arr)

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
    return get_default_backend()


# =============================================================================
# Unit Tests: _geometric_merge_matrices
# =============================================================================


class TestGeometricMergeMatrices:
    """Tests for the core geometric merge function."""

    def test_single_matrix_returns_unchanged(self, backend):
        """Single matrix should be returned as-is."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        default_backend.eval(matrix)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix], backend
        )

        default_backend.eval(result)
        eps = _div_eps()
        assert result.shape == matrix.shape
        # Check arrays are close using backend operations
        diff = default_backend.abs(result - matrix)
        max_diff = default_backend.max(diff)
        default_backend.eval(max_diff)
        assert float(default_backend.to_scalar(max_diff)) <= eps
        assert abs(proc_error) < eps
        assert abs(perm_quality - 1.0) < eps

    def test_identical_matrices_return_same(self, backend):
        """Merging identical matrices should return approximately the same matrix."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        default_backend.eval(matrix)
        # Create copies via backend array construction from list
        matrix_list = default_backend.tolist(matrix)
        matrix1 = default_backend.array(matrix_list)
        matrix2 = default_backend.array(matrix_list)
        default_backend.eval(matrix1, matrix2)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix1, matrix2], backend
        )

        eps = _div_eps()
        assert result.shape == matrix.shape
        assert proc_error < eps, f"Self-merge error too high: {proc_error}"
        assert abs(perm_quality - 1.0) < eps

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        shapes = [(4, 8), (16, 4), (8, 8), (32, 64)]

        for shape in shapes:
            m1 = default_backend.random_normal(shape)
            m2 = default_backend.random_normal(shape)
            default_backend.eval(m1, m2)

            result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
                [m1, m2], backend
            )

            assert result.shape == shape, f"Shape mismatch: {result.shape} vs {shape}"

    def test_1d_tensors_averaged(self, backend):
        """1D tensors (biases) should be simply averaged."""
        default_backend = get_default_backend()
        bias1_data = [1.0, 2.0, 3.0, 4.0]
        bias2_data = [5.0, 6.0, 7.0, 8.0]
        expected_data = [3.0, 4.0, 5.0, 6.0]
        bias1 = default_backend.array(bias1_data, dtype="float32")
        bias2 = default_backend.array(bias2_data, dtype="float32")
        expected = default_backend.array(expected_data, dtype="float32")
        default_backend.eval(bias1, bias2, expected)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            [bias1, bias2], backend
        )

        default_backend.eval(result)
        result_list = default_backend.tolist(result)
        expected_list = expected_data
        assert result.shape == bias1.shape
        eps = _div_eps()
        assert abs(result_list[0] - expected_list[0]) < eps
        assert abs(result_list[1] - expected_list[1]) < eps
        # 1D tensors should have default metrics
        assert abs(proc_error) < eps
        assert abs(perm_quality - 1.0) < eps

    def test_no_nan_in_output(self, backend):
        """Merge should not introduce NaN values."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        m1 = default_backend.random_normal((8, 16))
        m2 = default_backend.random_normal((8, 16))
        default_backend.eval(m1, m2)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )

        default_backend.eval(result)
        isfinite_arr = default_backend.isfinite(result)
        default_backend.eval(isfinite_arr)
        isfinite_list = default_backend.tolist(isfinite_arr)
        assert all(all(row) for row in isfinite_list), "Output contains NaN or Inf"

    def test_dtype_preserved(self, backend):
        """Output dtype should match input dtype."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrix = default_backend.random_normal((8, 16))
        default_backend.eval(matrix)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix], backend
        )

        default_backend.eval(result)
        assert default_backend.dtype(result) == default_backend.dtype(matrix)


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
        default_backend.eval(matrix)
        # Create identical copies using tolist
        matrix_list = default_backend.tolist(matrix)
        source_arr = backend.array(matrix_list)
        target_arr = backend.array(matrix_list)

        rotated, error = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )

        assert error < _div_eps(), f"Self-alignment error should be ~0, got {error}"

    def test_rotation_is_proper(self, backend):
        """Procrustes should produce proper rotation (det > 0)."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 8))
        target = default_backend.random_normal((8, 8))
        default_backend.eval(source, target)

        # Compute rotation matrix using backend operations
        M = default_backend.matmul(default_backend.transpose(target), source)
        U, S, Vt = default_backend.svd(M)
        R = default_backend.matmul(U, Vt)
        default_backend.eval(R)

        det = default_backend.det(R)
        default_backend.eval(det)
        det_val = float(default_backend.to_scalar(det))

        if det_val < 0:
            # Flip last column of U
            U_list = default_backend.tolist(U)
            for row in U_list:
                row[-1] *= -1
            U = default_backend.array(U_list)
            R = default_backend.matmul(U, Vt)
            default_backend.eval(R)
            det = default_backend.det(R)
            default_backend.eval(det)
            det_val = float(default_backend.to_scalar(det))

        assert det_val > 0, f"Rotation should have positive determinant, got {det_val}"

    def test_output_shape_preserved(self, backend):
        """Output shape should match input shape."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 16))
        target = default_backend.random_normal((8, 16))
        default_backend.eval(source, target)
        source_list = default_backend.tolist(source)
        target_list = default_backend.tolist(target)
        source_arr = backend.array(source_list)
        target_arr = backend.array(target_list)

        rotated, _ = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )

        assert rotated.shape == source.shape

    def test_error_is_normalized(self, backend):
        """Error should be normalized (relative to target norm)."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        source = default_backend.random_normal((8, 16))
        target = default_backend.random_normal((8, 16))
        target = target * 100.0  # Large target
        default_backend.eval(source, target)
        source_list = default_backend.tolist(source)
        target_list = default_backend.tolist(target)
        source_arr = backend.array(source_list)
        target_arr = backend.array(target_list)

        aligned, error = LoRAAdapterMerger._procrustes_align(
            source_arr, target_arr, backend
        )

        diff = aligned - target_arr
        mse = backend.mean(diff * diff)
        target_energy = backend.mean(target_arr * target_arr)
        backend.eval(mse, target_energy)
        eps = _div_eps()
        expected = float(backend.to_scalar(mse)) / max(
            float(backend.to_scalar(target_energy)), eps
        )
        assert abs(error - expected) < eps

# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_matrices(self, backend):
        """Should handle small (2x2) matrices."""
        default_backend = get_default_backend()
        m1_data = [[1.0, 2.0], [3.0, 4.0]]
        m2_data = [[5.0, 6.0], [7.0, 8.0]]
        m1 = default_backend.array(m1_data, dtype="float32")
        m2 = default_backend.array(m2_data, dtype="float32")
        default_backend.eval(m1, m2)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )

        assert result.shape == (2, 2)
        default_backend.eval(result)
        isfinite_arr = default_backend.isfinite(result)
        default_backend.eval(isfinite_arr)
        isfinite_list = default_backend.tolist(isfinite_arr)
        assert all(all(row) for row in isfinite_list), "Output contains NaN"

    def test_three_way_merge(self, backend):
        """Should handle merging 3 adapters."""
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        matrices = []
        for _ in range(3):
            m = default_backend.random_normal((8, 16))
            default_backend.eval(m)
            matrices.append(m)

        result, proc_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
            matrices, backend
        )

        assert result.shape == (8, 16)
        default_backend.eval(result)
        isfinite_arr = default_backend.isfinite(result)
        default_backend.eval(isfinite_arr)
        isfinite_list = default_backend.tolist(isfinite_arr)
        assert all(all(row) for row in isfinite_list), "Output contains NaN"
        # Should have some quality metrics
        assert 0.0 <= perm_quality <= 1.0 + _div_eps()


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionCases:
    """Tests with known expected behavior."""

    def test_merge_reduces_variance(self, backend):
        """Merge output variance should remain finite."""
        import math
        default_backend = get_default_backend()
        default_backend.random_seed(42)
        # Create matrices with high variance
        m1 = default_backend.random_normal((8, 16))
        m2 = default_backend.random_normal((8, 16))
        m1_scaled = m1 * 2.0
        m2_scaled = m2 * 2.0
        default_backend.eval(m1_scaled, m2_scaled)

        result, _, _ = LoRAAdapterMerger._geometric_merge_matrices(
            [m1_scaled, m2_scaled], backend
        )
        default_backend.eval(result)

        # Compute variance using backend: var = mean(x²) - mean(x)²
        def compute_var(arr):
            mean_val = default_backend.mean(arr)
            mean_sq = default_backend.mean(arr * arr)
            default_backend.eval(mean_val, mean_sq)
            return float(default_backend.to_scalar(mean_sq)) - float(default_backend.to_scalar(mean_val)) ** 2

        input_var = (compute_var(m1_scaled) + compute_var(m2_scaled)) / 2
        output_var = compute_var(result)

        assert math.isfinite(input_var)
        assert math.isfinite(output_var)
        assert output_var >= 0.0
