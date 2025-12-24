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
import numpy as np
import pytest


class MockTensor:
    """Mock tensor that simulates PyTorch tensor behavior."""

    def __init__(self, data, dtype=None, device=None):
        self._data = np.array(data)
        self._dtype = dtype
        self._device = device

    def dim(self) -> int:
        return self._data.ndim

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    def squeeze(self, dim=None):
        return MockTensor(np.squeeze(self._data, axis=dim), self._dtype, self._device)

    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self._data, axis=dim), self._dtype, self._device)

    def to_numpy(self):
        return self._data.copy()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data.copy()


def create_mock_torch():
    """Create a mock torch module with required functions."""
    mock_torch = MagicMock()

    # Mock dtypes
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.int32 = "int32"
    mock_torch.int64 = "int64"

    # Mock torch.triu - creates upper triangular matrix
    def mock_triu(tensor, diagonal=0):
        data = tensor._data if isinstance(tensor, MockTensor) else np.array(tensor)
        result = np.triu(data, k=diagonal)
        return MockTensor(result, tensor._dtype if isinstance(tensor, MockTensor) else None, "cuda")

    mock_torch.triu = mock_triu

    # Mock torch.full - creates tensor filled with value
    def mock_full(shape, fill_value, dtype=None, device=None):
        data = np.full(shape, fill_value)
        return MockTensor(data, dtype, device)

    mock_torch.full = mock_full

    # Mock torch.softmax - applies softmax along axis
    def mock_softmax(tensor, dim=-1):
        data = tensor._data if isinstance(tensor, MockTensor) else np.array(tensor)
        # Numerically stable softmax
        shifted = data - np.max(data, axis=dim, keepdims=True)
        exp_data = np.exp(shifted)
        probs = exp_data / np.sum(exp_data, axis=dim, keepdims=True)
        return MockTensor(probs, tensor._dtype if isinstance(tensor, MockTensor) else None, "cuda")

    mock_torch.softmax = mock_softmax

    # Mock torch.multinomial - samples from categorical distribution
    def mock_multinomial(probs_tensor, num_samples=1, replacement=True):
        probs = probs_tensor._data if isinstance(probs_tensor, MockTensor) else np.array(probs_tensor)

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        batch_size = probs.shape[0]
        num_categories = probs.shape[1]

        samples = []
        for i in range(batch_size):
            p = probs[i]
            # Normalize to ensure valid probability distribution
            p = p / p.sum()
            sample = np.random.choice(num_categories, size=num_samples, replace=replacement, p=p)
            samples.append(sample)

        result = np.array(samples)
        return MockTensor(result, None, "cuda")

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
            data = mask.to_numpy()

            # Check diagonal is 0
            for i in range(4):
                assert data[i, i] == 0.0

    def test_causal_mask_lower_triangular_zero(self):
        """Lower triangular (below diagonal) should be 0 (attend to past)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.to_numpy()

            # Check lower triangle is 0
            for i in range(4):
                for j in range(i):
                    assert data[i, j] == 0.0

    def test_causal_mask_upper_triangular_neginf(self):
        """Upper triangular (above diagonal) should be -inf (block future)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(4)
            data = mask.to_numpy()

            # Check upper triangle is -inf
            for i in range(4):
                for j in range(i + 1, 4):
                    assert data[i, j] == float("-inf")

    def test_causal_mask_seq_len_1(self):
        """Mask for seq_len=1 should be [[0.0]] (no masking needed)."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(1)
            data = mask.to_numpy()

            assert data.shape == (1, 1)
            assert data[0, 0] == 0.0

    def test_causal_mask_large_sequence(self):
        """Test mask works for larger sequences."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(128)
            data = mask.to_numpy()

            assert data.shape == (128, 128)
            # Spot check
            assert data[0, 0] == 0.0
            assert data[0, 1] == float("-inf")
            assert data[127, 0] == 0.0
            assert data[127, 127] == 0.0


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
            sample_data = samples.to_numpy()
            assert sample_data.shape == (1,)
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

            sample_data = samples.to_numpy()
            assert sample_data.shape == (10,)
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
            logits = MockTensor([
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ])
            samples = backend.random_categorical(logits, num_samples=5)

            sample_data = samples.to_numpy()
            assert sample_data.shape == (3, 5)
            # All samples should be valid indices
            assert all(0 <= s < 4 for row in sample_data for s in row)

    def test_categorical_deterministic_with_extreme_logits(self):
        """With extreme logits, sampling should strongly favor highest logit."""
        mock_torch = create_mock_torch()
        np.random.seed(42)  # For reproducibility
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # Make category 2 overwhelmingly likely
            logits = MockTensor([-100.0, -100.0, 100.0])
            samples = backend.random_categorical(logits, num_samples=100)

            sample_data = samples.to_numpy()
            # With such extreme logits, all samples should be index 2
            assert all(s == 2 for s in sample_data)

    def test_categorical_respects_probability_distribution(self):
        """Samples should roughly follow the probability distribution."""
        mock_torch = create_mock_torch()
        np.random.seed(42)  # For reproducibility
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()

            # Log probabilities that create a skewed distribution
            # After softmax: roughly [0.09, 0.24, 0.67]
            logits = MockTensor([0.0, 1.0, 2.0])
            samples = backend.random_categorical(logits, num_samples=1000)

            sample_data = samples.to_numpy()
            counts = np.bincount(sample_data, minlength=3)

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
            data = mask.to_numpy()

            # Expected structure for seq_len=4
            expected = np.array([
                [0.0, -np.inf, -np.inf, -np.inf],
                [0.0, 0.0, -np.inf, -np.inf],
                [0.0, 0.0, 0.0, -np.inf],
                [0.0, 0.0, 0.0, 0.0],
            ])

            np.testing.assert_array_equal(data, expected)

    def test_mask_can_be_used_for_attention(self):
        """Verify mask values work correctly with softmax attention pattern."""
        mock_torch = create_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            import modelcypher.backends.cuda_backend as cuda_module

            importlib.reload(cuda_module)
            backend = cuda_module.CUDABackend()
            mask = backend.create_causal_mask(3)
            mask_data = mask.to_numpy()

            # Simulate attention scores + mask
            scores = np.ones((3, 3))
            masked_scores = scores + mask_data

            # After masking, upper triangle should be -inf
            assert np.isinf(masked_scores[0, 1])
            assert np.isinf(masked_scores[0, 2])
            assert np.isinf(masked_scores[1, 2])

            # Lower triangle and diagonal should be finite
            assert np.isfinite(masked_scores[0, 0])
            assert np.isfinite(masked_scores[1, 0])
            assert np.isfinite(masked_scores[1, 1])
            assert np.isfinite(masked_scores[2, 0])
            assert np.isfinite(masked_scores[2, 1])
            assert np.isfinite(masked_scores[2, 2])
