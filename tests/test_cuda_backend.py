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

"""Mock-based tests for CUDA backend.

Tests verify CUDA backend methods work correctly without requiring actual GPU hardware.
Uses unittest.mock to patch torch with a mock module that simulates PyTorch behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modelcypher.core.support.array_utils import array_to_list

_BACKEND = None


def _get_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from modelcypher.backends.mlx_backend import MLXBackend

        _BACKEND = MLXBackend()
    except Exception:
        try:
            from modelcypher.backends.jax_backend import JAXBackend

            _BACKEND = JAXBackend()
        except Exception:
            pytest.skip(
                "MLX or JAX backend required for CUDA backend tests.",
                allow_module_level=True,
            )
    return _BACKEND


def _is_inf(value: float) -> bool:
    return value == float("inf") or value == float("-inf")


def _is_finite(value: float) -> bool:
    return value == value and not _is_inf(value)


class MockTensor:
    """Mock tensor that simulates PyTorch tensor behavior."""

    def __init__(self, data, dtype=None, device=None):
        backend = _get_backend()
        self._data = backend.array(data)
        self._dtype = dtype
        self._device = device

    def dim(self) -> int:
        return self._data.ndim

    @property
    def ndim(self) -> int:
        return len(_get_backend().shape(self._data))

    @property
    def shape(self):
        return _get_backend().shape(self._data)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def squeeze(self, dim=None):
        backend = _get_backend()
        if dim is None:
            return MockTensor(backend.squeeze(self._data), self._dtype, self._device)
        return MockTensor(backend.squeeze(self._data, axis=dim), self._dtype, self._device)

    def unsqueeze(self, dim):
        backend = _get_backend()
        return MockTensor(backend.expand_dims(self._data, dim), self._dtype, self._device)

    def tolist(self):
        return array_to_list(_get_backend(), self._data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.tolist()

    def _binary_op(self, other, op):
        backend = _get_backend()
        other_data = other._data if isinstance(other, MockTensor) else backend.array(other)
        result = op(self._data, other_data)
        dtype = self._dtype if self._dtype is not None else getattr(other, "_dtype", None)
        return MockTensor(result, dtype, self._device)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: b * a)


def create_mock_torch():
    """Create a mock torch module with required functions."""
    backend = _get_backend()
    mock_torch = MagicMock()

    # Mock dtypes
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.int32 = "int32"
    mock_torch.int64 = "int64"
    mock_torch.bool = "bool"

    # Mock torch.triu - creates upper triangular matrix
    def mock_triu(tensor, diagonal=0):
        data = tensor._data if isinstance(tensor, MockTensor) else backend.array(tensor)
        rows, cols = backend.shape(data)
        row_idx = backend.reshape(backend.arange(rows), (rows, 1))
        col_idx = backend.reshape(backend.arange(cols), (1, cols))
        row_grid = backend.broadcast_to(row_idx, (rows, cols))
        col_grid = backend.broadcast_to(col_idx, (rows, cols))
        mask = col_grid >= row_grid + diagonal
        zeros = backend.zeros_like(data)
        result = backend.where(mask, data, zeros)
        return MockTensor(result, tensor._dtype if isinstance(tensor, MockTensor) else None, "cuda")

    mock_torch.triu = mock_triu

    def mock_tril(tensor, diagonal=0):
        data = tensor._data if isinstance(tensor, MockTensor) else backend.array(tensor)
        rows, cols = backend.shape(data)
        row_idx = backend.reshape(backend.arange(rows), (rows, 1))
        col_idx = backend.reshape(backend.arange(cols), (1, cols))
        row_grid = backend.broadcast_to(row_idx, (rows, cols))
        col_grid = backend.broadcast_to(col_idx, (rows, cols))
        mask = col_grid <= row_grid + diagonal
        zeros = backend.zeros_like(data)
        result = backend.where(mask, data, zeros)
        return MockTensor(result, tensor._dtype if isinstance(tensor, MockTensor) else None, "cuda")

    mock_torch.tril = mock_tril

    # Mock torch.full - creates tensor filled with value
    def mock_full(shape, fill_value, dtype=None, device=None):
        data = backend.full(shape, fill_value)
        return MockTensor(data, dtype, device)

    mock_torch.full = mock_full

    # Mock torch.softmax - applies softmax along axis
    def mock_softmax(tensor, dim=-1):
        data = tensor._data if isinstance(tensor, MockTensor) else backend.array(tensor)
        shifted = data - backend.max(data, axis=dim, keepdims=True)
        exp_data = backend.exp(shifted)
        probs = exp_data / backend.sum(exp_data, axis=dim, keepdims=True)
        return MockTensor(probs, tensor._dtype if isinstance(tensor, MockTensor) else None, "cuda")

    mock_torch.softmax = mock_softmax

    def mock_einsum(pattern, a, b):
        a_arr = a._data if isinstance(a, MockTensor) else backend.array(a)
        b_arr = b._data if isinstance(b, MockTensor) else backend.array(b)
        if len(backend.shape(a_arr)) != 4 or len(backend.shape(b_arr)) != 4:
            raise ValueError("Mock einsum expects rank-4 tensors for attention")
        if pattern == "...qhd,...khd->...hqk":
            a_trans = backend.transpose(a_arr, axes=(0, 2, 1, 3))
            b_trans = backend.transpose(b_arr, axes=(0, 2, 1, 3))
            b_t = backend.transpose(b_trans, axes=(0, 1, 3, 2))
            scores = backend.matmul(a_trans, b_t)
            dtype = a._dtype if isinstance(a, MockTensor) else None
            return MockTensor(scores, dtype, "cuda")
        if pattern == "...hqk,...khd->...qhd":
            v_trans = backend.transpose(b_arr, axes=(0, 2, 1, 3))
            out = backend.matmul(a_arr, v_trans)
            out = backend.transpose(out, axes=(0, 2, 1, 3))
            dtype = a._dtype if isinstance(a, MockTensor) else None
            return MockTensor(out, dtype, "cuda")
        raise ValueError(f"Unsupported einsum pattern: {pattern}")

    mock_torch.einsum = mock_einsum

    # Mock torch.multinomial - samples from categorical distribution
    def mock_multinomial(probs_tensor, num_samples=1, replacement=True):
        probs = probs_tensor._data if isinstance(probs_tensor, MockTensor) else backend.array(
            probs_tensor
        )

        was_1d = len(backend.shape(probs)) == 1
        if was_1d:
            probs = backend.expand_dims(probs, 0)

        probs = probs / backend.sum(probs, axis=1, keepdims=True)
        cdf = backend.cumsum(probs, axis=1)

        batch_size, num_categories = backend.shape(cdf)
        samples = backend.random_uniform(shape=(batch_size, num_samples))
        cdf_exp = backend.expand_dims(cdf, -1)
        samples_exp = backend.expand_dims(samples, 1)
        mask = cdf_exp >= samples_exp
        indices = backend.argmax(mask * 1, axis=1)

        if was_1d:
            indices = backend.squeeze(indices, axis=0)

        return MockTensor(indices, None, "cuda")

    mock_torch.multinomial = mock_multinomial

    # Mock torch.tensor
    def mock_tensor(data, dtype=None, device=None):
        return MockTensor(data, dtype, device)

    mock_torch.tensor = mock_tensor
    mock_torch.as_tensor = mock_tensor

    def mock_ones(shape, dtype=None, device=None):
        data = backend.ones(shape)
        return MockTensor(data, dtype, device)

    mock_torch.ones = mock_ones

    def mock_where(condition, x, y):
        cond_data = condition._data if isinstance(condition, MockTensor) else backend.array(condition)
        x_data = x._data if isinstance(x, MockTensor) else backend.array(x)
        y_data = y._data if isinstance(y, MockTensor) else backend.array(y)
        result = backend.where(cond_data, x_data, y_data)
        return MockTensor(result, None, "cuda")

    mock_torch.where = mock_where

    def mock_finfo(dtype=None):
        info = backend.finfo()
        return info

    mock_torch.finfo = mock_finfo

    # Mock CUDA module
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.synchronize = MagicMock()
    mock_torch.cuda.manual_seed = MagicMock()

    # Mock manual_seed
    mock_torch.manual_seed = MagicMock()

    return mock_torch


class TestCUDABackendCreateCausalMask:
    """Tests for CUDA backend create_causal_mask method."""

    def test_causal_mask_shape(self):
        """Mask should have shape (seq_len, seq_len)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Need to reimport after patching
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(5)

            assert mask.shape == (5, 5)

    def test_causal_mask_diagonal_zero(self):
        """Diagonal elements should be 0 (attend to current position)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.tolist()

            # Check diagonal is 0
            for i in range(4):
                assert data[i][i] == 0.0

    def test_causal_mask_lower_triangular_zero(self):
        """Lower triangular (below diagonal) should be 0 (attend to past)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.tolist()

            # Check lower triangle is 0
            for i in range(4):
                for j in range(i):
                    assert data[i][j] == 0.0

    def test_causal_mask_upper_triangular_neginf(self):
        """Upper triangular (above diagonal) should be -inf (block future)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.tolist()

            # Check upper triangle is -inf
            for i in range(4):
                for j in range(i + 1, 4):
                    assert data[i][j] == float("-inf")

    def test_causal_mask_seq_len_1(self):
        """Mask for seq_len=1 should be [[0.0]] (no masking needed)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(1)
            data = mask.tolist()

            assert len(data) == 1
            assert len(data[0]) == 1
            assert data[0][0] == 0.0

    def test_causal_mask_large_sequence(self):
        """Test mask works for larger sequences."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(128)
            data = mask.tolist()

            assert len(data) == 128
            assert len(data[0]) == 128
            # Spot check
            assert data[0][0] == 0.0
            assert data[0][1] == float("-inf")
            assert data[127][0] == 0.0
            assert data[127][127] == 0.0


class TestCUDABackendRandomCategorical:
    """Tests for CUDA backend random_categorical method."""

    def test_categorical_single_sample_1d(self):
        """Single sample from 1D logits should return shape (1,)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # Create mock logits tensor
            logits = MockTensor([1.0, 2.0, 3.0])
            samples = backend.random_categorical(logits, num_samples=1)

            # Should return indices in valid range
            sample_data = samples.tolist()
            assert len(sample_data) == 1
            assert 0 <= sample_data[0] < 3

    def test_categorical_multiple_samples_1d(self):
        """Multiple samples from 1D logits."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            logits = MockTensor([1.0, 2.0, 3.0])
            samples = backend.random_categorical(logits, num_samples=10)

            sample_data = samples.tolist()
            assert len(sample_data) == 10
            # All samples should be valid indices
            assert all(0 <= s < 3 for s in sample_data)

    def test_categorical_batch_2d(self):
        """Sampling from batch of distributions (2D logits)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # 3 distributions, each with 4 categories
            logits = MockTensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
            samples = backend.random_categorical(logits, num_samples=5)

            sample_data = samples.tolist()
            assert len(sample_data) == 3
            assert len(sample_data[0]) == 5
            # All samples should be valid indices
            assert all(0 <= s < 4 for row in sample_data for s in row)

    def test_categorical_deterministic_with_extreme_logits(self):
        """With extreme logits, sampling should strongly favor highest logit."""
        mock_torch = create_mock_torch()
        _get_backend().random_seed(42)  # For reproducibility
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # Make category 2 overwhelmingly likely
            logits = MockTensor([-100.0, -100.0, 100.0])
            samples = backend.random_categorical(logits, num_samples=100)

            sample_data = samples.tolist()
            # With such extreme logits, all samples should be index 2
            assert all(s == 2 for s in sample_data)

    def test_categorical_respects_probability_distribution(self):
        """Samples should roughly follow the probability distribution."""
        mock_torch = create_mock_torch()
        _get_backend().random_seed(42)  # For reproducibility
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # Log probabilities that create a skewed distribution
            # After softmax: roughly [0.09, 0.24, 0.67]
            logits = MockTensor([0.0, 1.0, 2.0])
            samples = backend.random_categorical(logits, num_samples=1000)

            sample_data = samples.tolist()
            counts = [0, 0, 0]
            for value in sample_data:
                counts[value] += 1
            backend_ref = _get_backend()
            backend_ref.random_seed(42)
            logits_arr = backend_ref.array([0.0, 1.0, 2.0])
            shifted = logits_arr - backend_ref.max(logits_arr)
            exp_data = backend_ref.exp(shifted)
            probs = exp_data / backend_ref.sum(exp_data)
            probs = backend_ref.expand_dims(probs, 0)
            cdf = backend_ref.cumsum(probs, axis=1)
            samples_ref = backend_ref.random_uniform(shape=(1, 1000))
            cdf_exp = backend_ref.expand_dims(cdf, -1)
            samples_exp = backend_ref.expand_dims(samples_ref, 1)
            mask = cdf_exp >= samples_exp
            indices = backend_ref.argmax(mask * 1, axis=1)
            expected_indices = backend_ref.squeeze(indices, axis=0)
            expected_counts = [0, 0, 0]
            for value in array_to_list(backend_ref, expected_indices):
                expected_counts[value] += 1

            assert counts == expected_counts


class TestCUDABackendIntegration:
    """Integration tests verifying backend methods work together."""

    def test_mask_structure_matches_numpy_reference(self):
        """Verify mask matches expected mathematical structure."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.tolist()

            # Expected structure for seq_len=4
            expected = [
                [0.0, float("-inf"), float("-inf"), float("-inf")],
                [0.0, 0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0, float("-inf")],
                [0.0, 0.0, 0.0, 0.0],
            ]

            assert data == expected

    def test_mask_can_be_used_for_attention(self):
        """Verify mask values work correctly with softmax attention pattern."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(3)
            mask_data = mask.tolist()

            # Simulate attention scores + mask
            scores = [[1.0, 1.0, 1.0] for _ in range(3)]
            masked_scores = [
                [scores[i][j] + mask_data[i][j] for j in range(3)] for i in range(3)
            ]

            # After masking, upper triangle should be -inf
            assert _is_inf(masked_scores[0][1])
            assert _is_inf(masked_scores[0][2])
            assert _is_inf(masked_scores[1][2])

            # Lower triangle and diagonal should be finite
            assert _is_finite(masked_scores[0][0])
            assert _is_finite(masked_scores[1][0])
            assert _is_finite(masked_scores[1][1])
            assert _is_finite(masked_scores[2][0])
            assert _is_finite(masked_scores[2][1])
            assert _is_finite(masked_scores[2][2])


class TestCUDABackendAttentionSinks:
    """Tests for attention sink handling."""

    def test_scaled_dot_product_attention_with_sinks_biases_output(self):
        """Sinks should bias attention toward higher-sink keys."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            q = mock_torch.tensor([[[[1.0]], [[1.0]]]])
            k = mock_torch.tensor([[[[1.0]], [[1.0]]]])
            v = mock_torch.tensor([[[[1.0]], [[3.0]]]])

            out_no_sinks = backend.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=1.0,
                sinks=[[0.0, 0.0], [0.0, 0.0]],
            )
            out_with_sinks = backend.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=1.0,
                sinks=[[0.0, 1.0], [0.0, 1.0]],
            )

            out_no_val = out_no_sinks.tolist()[0][0][0][0]
            out_sink_val = out_with_sinks.tolist()[0][0][0][0]

            assert out_sink_val > out_no_val
