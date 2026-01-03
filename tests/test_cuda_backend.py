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


def create_mock_torch():
    """Create a mock torch module with required functions."""
    backend = _get_backend()
    mock_torch = MagicMock()

    # Mock dtypes
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.int32 = "int32"
    mock_torch.int64 = "int64"

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

            # Category 2 should have most samples, category 0 least
            assert counts[2] > counts[1] > counts[0]


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
