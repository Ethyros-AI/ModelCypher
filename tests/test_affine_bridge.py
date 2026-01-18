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

import math
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    regularization_epsilon,
)
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
        backend.random_seed(42)
        X = backend.random_normal((30, 3))
        backend.eval(X)
        Y = X  # Identity mapping

        result = bridge.train(X, Y)

        # MSE is the direct measure of transformation quality for identity mapping.
        # Geodesic cosine is unstable when pred ≈ Y (duplicate points in interleaved set).
        scale_arr = backend.mean(X * X)
        backend.eval(scale_arr)
        scale = float(backend.to_scalar(scale_arr))
        tol = regularization_epsilon(backend, X) * max(scale, 1.0)
        assert result.train_mse < tol, f"MSE should be near-zero for identity, got {result.train_mse}"
        # W should be near-identity (diagonal close to 1, off-diagonal close to 0)
        diag_tol = math.sqrt(
            max(result.train_mse, _eps(backend, result.train_mse))
        ) * math.sqrt(float(result.source_dim))
        for i in range(3):
            assert abs(result.W[i][i] - 1.0) < diag_tol, f"W[{i}][{i}] should be ~1.0"

    def test_train_with_translation(self, backend, bridge) -> None:
        """Training should learn translation (bias)."""
        # Use enough samples for reliable ridge regression
        backend.random_seed(42)
        X = backend.random_normal((30, 2))
        backend.eval(X)
        # Y = X + [1, 2] (constant translation)
        translation = backend.array([1.0, 2.0])
        Y = X + translation
        backend.eval(Y)

        result = bridge.train(X, Y)

        # MSE should be low for pure translation
        mean_Y = backend.mean(Y, axis=0, keepdims=True)
        baseline_arr = backend.mean((Y - mean_Y) * (Y - mean_Y))
        backend.eval(baseline_arr)
        baseline = float(backend.to_scalar(baseline_arr))
        tol = regularization_epsilon(backend, Y)
        assert result.train_mse <= baseline * (1.0 + tol), (
            f"MSE should improve on baseline, got {result.train_mse} vs baseline {baseline}"
        )
        # Bias should be close to [1, 2]
        b_tol = math.sqrt(
            max(result.train_mse, _eps(backend, result.train_mse))
        ) * math.sqrt(float(result.target_dim))
        assert abs(result.b[0] - 1.0) < b_tol, f"b[0] should be ~1.0, got {result.b[0]}"
        assert abs(result.b[1] - 2.0) < b_tol, f"b[1] should be ~2.0, got {result.b[1]}"

    def test_train_with_scaling(self, backend, bridge) -> None:
        """Training should learn scaling."""
        # Use enough samples for reliable learning
        backend.random_seed(42)
        X = backend.random_normal((30, 2))
        backend.eval(X)
        # Y = 2 * X (uniform scaling)
        Y = 2.0 * X
        backend.eval(Y)

        result = bridge.train(X, Y)

        # MSE should be low for learned scaling
        mean_Y = backend.mean(Y, axis=0, keepdims=True)
        baseline_arr = backend.mean((Y - mean_Y) * (Y - mean_Y))
        backend.eval(baseline_arr)
        baseline = float(backend.to_scalar(baseline_arr))
        tol = regularization_epsilon(backend, Y)
        assert result.train_mse <= baseline * (1.0 + tol), (
            f"MSE should improve on baseline, got {result.train_mse} vs baseline {baseline}"
        )
        # W should be close to 2*I
        w_tol = math.sqrt(
            max(result.train_mse, _eps(backend, result.train_mse))
        ) * math.sqrt(float(result.source_dim))
        assert abs(result.W[0][0] - 2.0) < w_tol, f"W[0][0] should be ~2.0, got {result.W[0][0]}"
        assert abs(result.W[1][1] - 2.0) < w_tol, f"W[1][1] should be ~2.0, got {result.W[1][1]}"

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
        tol = regularization_epsilon(backend, Y_pred) * max(1.0, max(abs(v) for v in y_list))
        assert abs(y_list[0] - 1.5) <= tol
        assert abs(y_list[1] - 2.5) <= tol

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
        tol = regularization_epsilon(backend, Y) * max(1.0, max(abs(v) for v in y_list))
        assert abs(y_list[0] - 3.0) <= tol
        assert abs(y_list[1] - 3.0) <= tol


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
        result = proj.project(X)  # Temperature auto-derived

        assert result.nearest_token_ids[0] == 0

    def test_project_auto_derives_temperature(self, backend, proj) -> None:
        """Temperature should be auto-derived from similarity distribution."""
        vocab = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        proj.set_vocabulary(vocab)

        # Query point equidistant from both
        X = backend.array([[0.7071, 0.7071]])  # 45 degrees
        result = proj.project(X)  # Temperature auto-derived

        # Temperature should be auto-derived (positive value)
        assert result.temperature_used > 0
        # Attention weights should sum to 1
        attn = result.attention_weights[0]
        tol = regularization_epsilon(backend, backend.array(attn))
        assert abs(sum(attn) - 1.0) <= tol

    def test_aligned_is_vocab_mixture(self, backend, proj) -> None:
        """Aligned output should be weighted sum of vocabulary."""
        vocab = backend.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        proj.set_vocabulary(vocab)

        X = backend.array([[1.0, 0.0]])  # Exactly token 0
        result = proj.project(X)  # Temperature auto-derived

        # Aligned should point in direction of token 0
        aligned = result.aligned[0]
        # Check that aligned points more toward token 0 than token 1
        assert aligned[0] > aligned[1]


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
        # Training data - use enough samples for reliable learning
        backend.random_seed(42)
        X_train = backend.random_normal((20, 2))
        backend.eval(X_train)
        Y_train = X_train * 2.0  # Scale by 2
        backend.eval(Y_train)

        # Vocabulary
        vocab = backend.array([
            [2.0, 0.0],   # Scaled token 0
            [0.0, 2.0],   # Scaled token 1
            [-2.0, 0.0],  # Scaled token 2
        ])

        result = hybrid.train(X_train, Y_train, vocab)

        # Affine should learn scaling - check via MSE
        mean_abs = backend.mean(backend.abs(Y_train))
        backend.eval(mean_abs)
        scale = float(backend.to_scalar(mean_abs))
        tol = regularization_epsilon(backend, Y_train) * max(1.0, scale)
        assert result.train_mse <= tol, f"MSE should be within precision, got {result.train_mse}"

        # Transform should project onto vocab (temperature auto-derived)
        X_new = backend.array([[1.0, 0.0]])  # Should map to [2, 0]
        vocab_result = hybrid.transform(X_new)

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
            temperature_used=1.0,  # Auto-derived temperature
        )
        summary = result.summary
        assert "Samples: 2" in summary
        assert "Temperature" in summary


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
