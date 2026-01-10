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
Tests for Affine Bridge module.

Tests cover:
- AffineBridgeResult dataclass
- AffineBridge class (train, transform, evaluate)
- VocabConstrainedProjection class
- HybridBridge class
- Generalization behavior
- Edge cases
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.affine_bridge import (
    AffineBridge,
    AffineBridgeResult,
    HybridBridge,
    VocabConstrainedProjection,
    VocabConstrainedResult,
)


def _eps(backend, *values: float) -> float:
    """Get machine epsilon for tolerance calculations."""
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# AffineBridgeResult Tests
# =============================================================================


class TestAffineBridgeResult:
    """Tests for AffineBridgeResult dataclass."""

    def _make_result(self, **kwargs) -> AffineBridgeResult:
        """Create result with default values."""
        defaults = {
            "W": [[1.0, 0.0], [0.0, 1.0]],
            "b": [0.0, 0.0],
            "train_mse": 0.001,
            "train_cosine": 0.95,
            "test_cosine": 0.92,
            "generalization_gap": 0.03,
            "source_dim": 2,
            "target_dim": 2,
            "n_train_samples": 100,
            "n_test_samples": 20,
            "regularization": 0.01,
        }
        defaults.update(kwargs)
        return AffineBridgeResult(**defaults)

    def test_all_fields_accessible(self) -> None:
        """Result should have all required fields."""
        result = self._make_result()
        assert result.W is not None
        assert result.b is not None
        assert isinstance(result.train_mse, float)
        assert isinstance(result.train_cosine, float)
        assert isinstance(result.source_dim, int)
        assert isinstance(result.target_dim, int)

    def test_summary_includes_metrics(self) -> None:
        """Summary should include key metrics."""
        result = self._make_result()
        summary = result.summary
        assert "Train MSE" in summary
        assert "Train cosine" in summary
        assert "Test cosine" in summary
        assert "Generalization gap" in summary

    def test_summary_handles_none_test(self) -> None:
        """Summary should handle None test metrics."""
        result = self._make_result(test_cosine=None, generalization_gap=None, n_test_samples=None)
        summary = result.summary
        assert "N/A" in summary


# =============================================================================
# AffineBridge Tests
# =============================================================================


class TestAffineBridge:
    """Tests for AffineBridge class."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def bridge(self, backend):
        """Create AffineBridge instance."""
        return AffineBridge(backend)

    def test_train_identity_mapping(self, backend, bridge) -> None:
        """Training on identity mapping should learn near-identity transform."""
        # Create paired data where Y = X (identity)
        X = backend.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0],
        ])
        Y = X  # Identity mapping

        result = bridge.train(X, Y)

        # Should learn near-identity with high cosine
        assert result.train_cosine > 0.99
        assert result.train_mse < 0.01

    def test_train_with_translation(self, backend, bridge) -> None:
        """Training should learn translation (bias)."""
        X = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ])
        # Y = X + [1, 2] (constant translation)
        Y = X + backend.array([1.0, 2.0])

        result = bridge.train(X, Y)

        # Should achieve very high cosine (direction preserved)
        assert result.train_cosine > 0.95
        # Bias should be close to [1, 2]
        assert abs(result.b[0] - 1.0) < 0.5
        assert abs(result.b[1] - 2.0) < 0.5

    def test_train_with_scaling(self, backend, bridge) -> None:
        """Training should learn scaling."""
        X = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ])
        # Y = 2 * X (uniform scaling)
        Y = 2.0 * X

        result = bridge.train(X, Y)

        # High cosine (direction preserved)
        assert result.train_cosine > 0.99

    def test_train_with_test_set(self, backend, bridge) -> None:
        """Training with test set should compute generalization gap."""
        X_train = backend.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        Y_train = X_train * 1.5

        X_test = backend.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ])
        Y_test = X_test * 1.5

        result = bridge.train(X_train, Y_train, X_test, Y_test)

        assert result.test_cosine is not None
        assert result.generalization_gap is not None
        assert result.n_test_samples == 2

    def test_transform_requires_training(self, backend, bridge) -> None:
        """Transform should raise error if not trained."""
        X = backend.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="Must call train"):
            bridge.transform(X)

    def test_transform_applies_learned_weights(self, backend, bridge) -> None:
        """Transform should apply learned W and b."""
        X = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        Y = X + backend.array([1.0, 2.0])

        bridge.train(X, Y)

        # Transform new point
        X_new = backend.array([[0.5, 0.5]])
        Y_pred = bridge.transform(X_new)

        backend.eval(Y_pred)
        y_list = backend.tolist(Y_pred)[0]

        # Should be close to [0.5 + 1, 0.5 + 2] = [1.5, 2.5]
        assert abs(y_list[0] - 1.5) < 0.5
        assert abs(y_list[1] - 2.5) < 0.5

    def test_load_weights(self, backend, bridge) -> None:
        """Loading weights should enable transform without training."""
        W = backend.array([[2.0, 0.0], [0.0, 2.0]])
        b = backend.array([1.0, 1.0])

        bridge.load_weights(W, b)

        X = backend.array([[1.0, 1.0]])
        Y = bridge.transform(X)

        backend.eval(Y)
        y_list = backend.tolist(Y)[0]

        # Y = X @ W + b = [1, 1] @ [[2, 0], [0, 2]] + [1, 1] = [2, 2] + [1, 1] = [3, 3]
        assert abs(y_list[0] - 3.0) < 0.01
        assert abs(y_list[1] - 3.0) < 0.01


# =============================================================================
# VocabConstrainedProjection Tests
# =============================================================================


class TestVocabConstrainedProjection:
    """Tests for VocabConstrainedProjection class."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def proj(self, backend):
        """Create VocabConstrainedProjection instance."""
        return VocabConstrainedProjection(backend)

    def test_project_requires_vocabulary(self, backend, proj) -> None:
        """Project should raise error if vocabulary not set."""
        X = backend.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="Must call set_vocabulary"):
            proj.project(X)

    def test_project_nearest_token(self, backend, proj) -> None:
        """Project should find nearest vocabulary token."""
        # Simple vocabulary: 3 tokens in 2D
        vocab = backend.array([
            [1.0, 0.0],   # Token 0: points right
            [0.0, 1.0],   # Token 1: points up
            [-1.0, 0.0],  # Token 2: points left
        ])
        proj.set_vocabulary(vocab)

        # Query point close to token 0
        X = backend.array([[0.9, 0.1]])
        result = proj.project(X, temperature=0.1)  # Low temp = hard assignment

        assert result.nearest_token_ids[0] == 0

    def test_project_soft_mixture(self, backend, proj) -> None:
        """High temperature should produce soft mixture."""
        vocab = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        proj.set_vocabulary(vocab)

        # Query point equidistant from both
        X = backend.array([[0.7071, 0.7071]])  # 45 degrees
        result = proj.project(X, temperature=2.0)  # High temp = soft

        # Attention should be relatively balanced
        attn = result.attention_weights[0]
        assert abs(attn[0] - attn[1]) < 0.3  # Not too different

    def test_aligned_is_vocab_mixture(self, backend, proj) -> None:
        """Aligned output should be weighted sum of vocabulary."""
        vocab = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        proj.set_vocabulary(vocab)

        X = backend.array([[1.0, 0.0]])  # Exactly token 0
        result = proj.project(X, temperature=0.1)

        # Aligned should be close to token 0
        aligned = result.aligned[0]
        assert abs(aligned[0] - 1.0) < 0.2
        assert abs(aligned[1]) < 0.2


# =============================================================================
# HybridBridge Tests
# =============================================================================


class TestHybridBridge:
    """Tests for HybridBridge class."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def hybrid(self, backend):
        """Create HybridBridge instance."""
        return HybridBridge(backend)

    def test_train_and_transform(self, backend, hybrid) -> None:
        """Hybrid should train affine then apply vocab constraint."""
        # Training data
        X_train = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ])
        Y_train = X_train * 2.0  # Scale by 2

        # Vocabulary
        vocab = backend.array([
            [2.0, 0.0],   # Scaled token 0
            [0.0, 2.0],   # Scaled token 1
            [-2.0, 0.0],  # Scaled token 2
        ])

        result = hybrid.train(X_train, Y_train, vocab)

        # Affine should learn scaling
        assert result.train_cosine > 0.9

        # Transform should project onto vocab
        X_new = backend.array([[1.0, 0.0]])  # Should map to [2, 0]
        vocab_result = hybrid.transform(X_new, temperature=0.1)

        # Should be nearest to token 0 ([2, 0])
        assert vocab_result.nearest_token_ids[0] == 0

    def test_load_weights_and_vocab_separately(self, backend, hybrid) -> None:
        """Should be able to load affine weights and vocab separately."""
        W = backend.array([[1.0, 0.0], [0.0, 1.0]])
        b = backend.array([0.0, 0.0])
        vocab = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        hybrid.load_affine_weights(W, b)
        hybrid.set_vocabulary(vocab)

        X = backend.array([[1.0, 0.0]])
        result = hybrid.transform(X)

        assert result.nearest_token_ids[0] == 0


# =============================================================================
# VocabConstrainedResult Tests
# =============================================================================


class TestVocabConstrainedResult:
    """Tests for VocabConstrainedResult dataclass."""

    def test_summary(self) -> None:
        """Summary should include key info."""
        result = VocabConstrainedResult(
            aligned=[[1.0, 0.0], [0.0, 1.0]],
            attention_weights=[[0.9, 0.1], [0.1, 0.9]],
            nearest_token_ids=[0, 1],
            temperature=1.0,
        )
        summary = result.summary
        assert "Samples: 2" in summary
        assert "Temperature: 1.0" in summary


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    def test_single_sample_training(self, backend) -> None:
        """Should handle single sample (degenerate case)."""
        bridge = AffineBridge(backend)
        X = backend.array([[1.0, 0.0]])
        Y = backend.array([[2.0, 0.0]])

        # Should not crash, though result may not be meaningful
        result = bridge.train(X, Y)
        assert result is not None

    def test_high_dimensional(self, backend) -> None:
        """Should handle high-dimensional embeddings."""
        bridge = AffineBridge(backend)

        # 64-dimensional embeddings, 20 samples
        X = backend.random_normal((20, 64))
        Y = backend.random_normal((20, 64))

        result = bridge.train(X, Y)
        assert result.source_dim == 64
        assert result.target_dim == 64

    def test_large_vocabulary(self, backend) -> None:
        """Should handle large vocabulary."""
        proj = VocabConstrainedProjection(backend)

        # 1000-token vocabulary, 128-dim
        vocab = backend.random_normal((1000, 128))
        proj.set_vocabulary(vocab)

        X = backend.random_normal((5, 128))
        result = proj.project(X)

        assert len(result.nearest_token_ids) == 5
        assert all(0 <= tid < 1000 for tid in result.nearest_token_ids)

    def test_zero_input(self, backend) -> None:
        """Should handle zero input vectors."""
        proj = VocabConstrainedProjection(backend)
        vocab = backend.array([[1.0, 0.0], [0.0, 1.0]])
        proj.set_vocabulary(vocab)

        X = backend.zeros((1, 2))
        result = proj.project(X)

        # Should not crash, output should be valid
        assert len(result.aligned) == 1
