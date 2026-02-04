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

"""Tests for LayerEntropyProjector (Entropy-Lens implementation).

These tests verify that the layer entropy projector correctly computes
real per-layer entropy by projecting hidden states through the unembedding
matrix, rather than fabricating entropy values.

References:
    Ali et al. (2025) "Entropy-Lens: The Information Signature of
    Transformer Computations" arXiv:2502.16570
"""

import pytest

from modelcypher.core.domain.entropy.layer_entropy_projector import (
    LayerEntropyProjector,
    LayerEntropyResult,
    ModelLayerEntropyProfile,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    log_scalar,
    machine_epsilon,
    sqrt_scalar,
)


def _eps(backend) -> float:
    return machine_epsilon(backend, backend.array([1.0]))


class TestLayerEntropyResult:
    """Tests for LayerEntropyResult dataclass."""

    def test_creation(self, any_backend):
        """Should create result with all fields."""
        result = LayerEntropyResult(
            layer_index=5,
            layer_name="layers.5",
            mean_entropy=3.5,
            entropy_variance=0.2,
            sample_count=10,
            min_entropy=3.0,
            max_entropy=4.0,
        )

        assert result.layer_index == 5
        assert result.layer_name == "layers.5"
        assert result.mean_entropy == 3.5
        assert result.entropy_variance == 0.2
        assert result.sample_count == 10


class TestModelLayerEntropyProfile:
    """Tests for ModelLayerEntropyProfile dataclass."""

    def test_entropy_trajectory(self, any_backend):
        """Should return entropy values in layer order."""
        results = {
            2: LayerEntropyResult(2, "layers.2", 3.0, 0.1, 5, 2.8, 3.2),
            0: LayerEntropyResult(0, "layers.0", 5.0, 0.2, 5, 4.8, 5.2),
            1: LayerEntropyResult(1, "layers.1", 4.0, 0.15, 5, 3.9, 4.1),
        }
        profile = ModelLayerEntropyProfile(
            model_name="test",
            layer_results=results,
            probe_prompts=["test"],
            unembedding_source="lm_head",
            vocab_size=32000,
            max_possible_entropy=log_scalar(32000.0, any_backend),
        )

        trajectory = profile.entropy_trajectory()
        # Should be sorted by layer index: 0, 1, 2
        assert trajectory == [5.0, 4.0, 3.0]

    def test_normalized_trajectory(self, any_backend):
        """Should normalize entropy to [0, 1] by max possible entropy."""
        max_entropy = log_scalar(32000.0, any_backend)
        results = {
            0: LayerEntropyResult(0, "layers.0", max_entropy / 2, 0.1, 5, 0, 0),
        }
        profile = ModelLayerEntropyProfile(
            model_name="test",
            layer_results=results,
            probe_prompts=["test"],
            unembedding_source="lm_head",
            vocab_size=32000,
            max_possible_entropy=max_entropy,
        )

        normalized = profile.normalized_trajectory()
        eps = _eps(any_backend)
        assert len(normalized) == 1
        assert abs(normalized[0] - 0.5) <= eps * max(1.0, abs(normalized[0]))


class TestLayerEntropyProjector:
    """Tests for LayerEntropyProjector."""

    def test_initialization(self, any_backend):
        """Should initialize with backend."""
        projector = LayerEntropyProjector(backend=any_backend)

        assert projector._backend is any_backend
        assert projector._unembedding_matrix is None

    def test_uniform_logits_gives_max_entropy(self, any_backend):
        """Uniform distribution should give H = log(vocab_size)."""
        vocab_size = 1000
        hidden_dim = 64

        # Create uniform unembedding matrix (all ones)
        # This will project any hidden state to uniform logits
        scale = sqrt_scalar(float(hidden_dim), any_backend)
        unembedding = any_backend.ones((vocab_size, hidden_dim)) / scale
        projector = LayerEntropyProjector(backend=any_backend)
        projector._unembedding_matrix = unembedding
        projector._vocab_size = vocab_size
        projector._hidden_dim = hidden_dim

        # Create a hidden state
        hidden_state = any_backend.ones((hidden_dim,))

        entropy, _ = projector.compute_layer_entropy(hidden_state)

        # Entropy of uniform distribution = log(vocab_size)
        expected = log_scalar(float(vocab_size), any_backend)
        eps = _eps(any_backend)
        assert abs(entropy - expected) <= eps * max(1.0, expected)

    def test_concentrated_logits_gives_low_entropy(self, any_backend):
        """One-hot distribution should give near-zero entropy."""
        vocab_size = 1000
        hidden_dim = 64

        # Create an unembedding matrix that strongly favors one token
        # One row has high values, others have low values
        high_values = any_backend.ones((1, hidden_dim)) * 100.0
        rest = any_backend.zeros((vocab_size - 1, hidden_dim))
        unembedding = any_backend.concatenate([high_values, rest], axis=0)

        projector = LayerEntropyProjector(backend=any_backend)
        projector._unembedding_matrix = unembedding
        projector._vocab_size = vocab_size
        projector._hidden_dim = hidden_dim

        # Create a hidden state that will project to concentrated distribution
        hidden_state = any_backend.ones((hidden_dim,))

        entropy, _ = projector.compute_layer_entropy(hidden_state)

        # Entropy should be very low (near 0) for concentrated distribution
        eps = _eps(any_backend)
        max_entropy = log_scalar(float(vocab_size), any_backend)
        assert entropy <= eps * max(1.0, max_entropy)

    def test_flatten_to_hidden_3d(self, any_backend):
        """Should extract last token from [batch, seq, hidden] tensor."""
        projector = LayerEntropyProjector(backend=any_backend)
        # Create 3D tensor
        hidden = any_backend.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # batch 0: 2 tokens
        ])

        flattened = projector._flatten_to_hidden(hidden)

        # Should be last token of batch 0
        expected = [4.0, 5.0, 6.0]
        assert flattened.shape == (3,)
        eps = _eps(any_backend)
        flattened_list = any_backend.tolist(flattened)
        assert all(
            abs(flattened_list[i] - expected[i]) <= eps * max(1.0, abs(expected[i]))
            for i in range(3)
        )

    def test_flatten_to_hidden_2d(self, any_backend):
        """Should extract last token from [seq, hidden] tensor."""
        projector = LayerEntropyProjector(backend=any_backend)
        hidden = any_backend.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        flattened = projector._flatten_to_hidden(hidden)

        expected = [4.0, 5.0, 6.0]
        assert flattened.shape == (3,)
        eps = _eps(any_backend)
        flattened_list = any_backend.tolist(flattened)
        assert all(
            abs(flattened_list[i] - expected[i]) <= eps * max(1.0, abs(expected[i]))
            for i in range(3)
        )

    def test_flatten_to_hidden_1d(self, any_backend):
        """Should return 1D tensor as-is."""
        projector = LayerEntropyProjector(backend=any_backend)
        hidden = any_backend.array([1.0, 2.0, 3.0])

        flattened = projector._flatten_to_hidden(hidden)

        expected = [1.0, 2.0, 3.0]
        assert flattened.shape == (3,)
        eps = _eps(any_backend)
        flattened_list = any_backend.tolist(flattened)
        assert all(
            abs(flattened_list[i] - expected[i]) <= eps * max(1.0, abs(expected[i]))
            for i in range(3)
        )

    def test_compute_layer_entropy_requires_unembedding(self, any_backend):
        """Should raise error if unembedding matrix not set."""
        projector = LayerEntropyProjector(backend=any_backend)
        hidden_state = any_backend.ones((64,))

        with pytest.raises(ValueError, match="Unembedding matrix not set"):
            projector.compute_layer_entropy(hidden_state)

    def test_entropy_varies_with_hidden_state(self, any_backend):
        """Different hidden states should produce different entropy values."""
        vocab_size = 100
        hidden_dim = 32

        # Random unembedding matrix
        any_backend.random_seed(42)
        unembedding = any_backend.random_normal((vocab_size, hidden_dim))
        projector = LayerEntropyProjector(backend=any_backend)
        projector._unembedding_matrix = unembedding
        projector._vocab_size = vocab_size
        projector._hidden_dim = hidden_dim

        # Two different hidden states
        hidden1 = any_backend.random_normal((hidden_dim,))
        hidden2 = any_backend.random_normal((hidden_dim,)) * 10.0  # Very different

        entropy1, _ = projector.compute_layer_entropy(hidden1)
        entropy2, _ = projector.compute_layer_entropy(hidden2)

        # Entropy values should differ
        assert entropy1 != entropy2

    def test_no_numpy_in_entropy_computation(self, any_backend):
        """Core entropy computation should use Backend protocol, not numpy."""
        vocab_size = 100
        hidden_dim = 32

        any_backend.random_seed(42)
        unembedding = any_backend.random_normal((vocab_size, hidden_dim))
        projector = LayerEntropyProjector(backend=any_backend)
        projector._unembedding_matrix = unembedding
        projector._vocab_size = vocab_size
        projector._hidden_dim = hidden_dim

        hidden_state = any_backend.random_normal((hidden_dim,))

        # This should not raise any numpy-related errors
        entropy, variance = projector.compute_layer_entropy(hidden_state)

        assert isinstance(entropy, float)
        assert isinstance(variance, float)
        eps = _eps(any_backend)
        max_entropy = log_scalar(float(vocab_size), any_backend)
        assert entropy >= -eps
        assert entropy <= max_entropy + eps * max(1.0, max_entropy)


class TestLayerEntropyProjectorIntegration:
    """Integration tests for LayerEntropyProjector with mock model."""

    @pytest.fixture
    def mock_model(self, any_backend):
        """Create a minimal mock model for testing."""

        class MockEmbedTokens:
            def __init__(self):
                self.weight = any_backend.random_normal((1000, 64))  # vocab=1000, hidden=64

        class MockLMHead:
            def __init__(self):
                self.weight = any_backend.random_normal((1000, 64))  # vocab=1000, hidden=64

        class MockLayer:
            def __init__(self, idx):
                self._idx = idx

            def __call__(self, x):
                # Simple passthrough with slight modification
                return x + any_backend.array(self._idx * 0.01)

        class MockBaseModel:
            def __init__(self):
                self.embed_tokens = MockEmbedTokens()
                self.layers = [MockLayer(i) for i in range(4)]

            def __call__(self, input_ids):
                x = self.embed_tokens.weight[input_ids.squeeze()]
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockModel:
            def __init__(self):
                self.model = MockBaseModel()
                self.lm_head = MockLMHead()

        return MockModel()

    def test_set_unembedding_matrix_lm_head(self, any_backend, mock_model):
        """Should extract unembedding from lm_head.weight."""
        projector = LayerEntropyProjector(backend=any_backend)

        source = projector.set_unembedding_matrix(mock_model)

        assert source == "lm_head"
        assert projector._vocab_size == 1000
        assert projector._hidden_dim == 64
        assert projector._unembedding_matrix is not None

    def test_set_unembedding_matrix_embed_tokens(self, any_backend):
        """Should fall back to embed_tokens if no lm_head."""

        class MockModelNoLMHead:
            class MockInner:
                class MockEmbed:
                    weight = any_backend.random_normal((500, 32))

                embed_tokens = MockEmbed()
                layers = []

            model = MockInner()

        mock_model = MockModelNoLMHead()
        projector = LayerEntropyProjector(backend=any_backend)

        source = projector.set_unembedding_matrix(mock_model)

        assert source == "embed_tokens_transposed"
        assert projector._vocab_size == 500
        assert projector._hidden_dim == 32
