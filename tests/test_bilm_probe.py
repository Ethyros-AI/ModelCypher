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

"""Tests for bilm_probe.py - Bidirectional LM probes for token classification.

Tests cover:
- BiLMRepresentations dataclass
- BiLMProbeWeights dataclass
- BiLMProbeResult dataclass
- PredictionResult dataclass
- BiLMProbeTrainer.build_representations()
- BiLMProbeTrainer.train()
- BiLMProbeTrainer.predict()
- BiLMProbeTrainer.predict_from_activations()
- Edge cases: empty input, unbalanced classes
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.bilm_probe import (
    BiLMProbeResult,
    BiLMProbeTrainer,
    BiLMProbeWeights,
    BiLMRepresentations,
    PredictionResult,
)

# =============================================================================
# Dataclass Tests
# =============================================================================


class TestBiLMRepresentations:
    """Tests for BiLMRepresentations dataclass."""

    def test_fields_stored_correctly(self):
        """BiLMRepresentations stores all fields."""
        backend = get_default_backend()
        forward = backend.array([[1.0, 2.0], [3.0, 4.0]])
        backward = backend.array([[5.0, 6.0], [7.0, 8.0]])
        combined = backend.concatenate([forward, backward], axis=1)
        labels = backend.array([0.0, 1.0])

        reps = BiLMRepresentations(
            forward=forward,
            backward=backward,
            combined=combined,
            labels=labels,
        )

        backend.eval(reps.combined)
        assert int(reps.combined.shape[0]) == 2
        assert int(reps.combined.shape[1]) == 4  # 2 + 2


class TestBiLMProbeWeights:
    """Tests for BiLMProbeWeights dataclass."""

    def test_fields_stored_correctly(self):
        """BiLMProbeWeights stores all fields."""
        backend = get_default_backend()
        weights = backend.array([0.1, 0.2, 0.3, 0.4])

        probe_weights = BiLMProbeWeights(
            weights=weights,
            bias=0.5,
            threshold=0.5,
            hidden_dim=2,
        )

        assert probe_weights.bias == 0.5
        assert probe_weights.threshold == 0.5
        assert probe_weights.hidden_dim == 2


class TestBiLMProbeResult:
    """Tests for BiLMProbeResult dataclass."""

    def test_fields_stored_correctly(self):
        """BiLMProbeResult stores all fields."""
        backend = get_default_backend()
        weights = BiLMProbeWeights(
            weights=backend.zeros((4,)),
            bias=0.0,
            threshold=0.5,
            hidden_dim=2,
        )

        result = BiLMProbeResult(
            weights=weights,
            train_accuracy=0.9,
            train_precision=0.85,
            train_recall=0.88,
            train_f1=0.865,
            val_accuracy=0.85,
            val_precision=0.80,
            val_recall=0.82,
            val_f1=0.81,
            n_train=80,
            n_val=20,
        )

        assert result.train_accuracy == 0.9
        assert result.val_accuracy == 0.85
        assert result.n_train == 80
        assert result.n_val == 20


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_fields_stored_correctly(self):
        """PredictionResult stores all fields."""
        backend = get_default_backend()
        predictions = backend.array([1, 0, 1], dtype="int32")
        probabilities = backend.array([0.9, 0.3, 0.7])

        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
        )

        backend.eval(result.predictions, result.probabilities)
        assert int(result.predictions.shape[0]) == 3
        assert int(result.probabilities.shape[0]) == 3


# =============================================================================
# BiLMProbeTrainer Tests
# =============================================================================


class TestBiLMProbeTrainerInit:
    """Tests for BiLMProbeTrainer initialization."""

    def test_default_backend(self):
        """BiLMProbeTrainer uses default backend if none provided."""
        trainer = BiLMProbeTrainer()
        assert trainer.backend is not None

    def test_custom_backend(self):
        """BiLMProbeTrainer accepts custom backend."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)
        assert trainer.backend is backend


class TestBuildRepresentations:
    """Tests for BiLMProbeTrainer.build_representations()."""

    def test_concatenates_forward_backward(self):
        """build_representations correctly concatenates forward and backward."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        forward = backend.array([[1.0, 2.0], [3.0, 4.0]])
        backward = backend.array([[5.0, 6.0], [7.0, 8.0]])
        labels = backend.array([0.0, 1.0])

        reps = trainer.build_representations(forward, backward, labels)

        backend.eval(reps.combined)
        assert int(reps.combined.shape[0]) == 2
        assert int(reps.combined.shape[1]) == 4

        # Check concatenation order
        combined_list = [float(x) for x in backend.tolist(reps.combined[0])]
        assert combined_list == [1.0, 2.0, 5.0, 6.0]


class TestTrain:
    """Tests for BiLMProbeTrainer.train()."""

    def test_trains_on_separable_data(self):
        """train achieves good accuracy on linearly separable data."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        # Create separable data:
        # Positive class: forward=[1,1], backward=[1,1] -> combined=[1,1,1,1]
        # Negative class: forward=[-1,-1], backward=[-1,-1] -> combined=[-1,-1,-1,-1]
        n_samples = 50
        noise_scale = 0.1

        forward_pos = backend.ones((n_samples, 2)) + backend.random_normal((n_samples, 2)) * noise_scale
        backward_pos = backend.ones((n_samples, 2)) + backend.random_normal((n_samples, 2)) * noise_scale
        forward_neg = -backend.ones((n_samples, 2)) + backend.random_normal((n_samples, 2)) * noise_scale
        backward_neg = -backend.ones((n_samples, 2)) + backend.random_normal((n_samples, 2)) * noise_scale
        backend.eval(forward_pos, backward_pos, forward_neg, backward_neg)

        forward_all = backend.concatenate([forward_pos, forward_neg], axis=0)
        backward_all = backend.concatenate([backward_pos, backward_neg], axis=0)
        labels = backend.concatenate([backend.ones((n_samples,)), backend.zeros((n_samples,))], axis=0)
        backend.eval(forward_all, backward_all, labels)

        reps = trainer.build_representations(forward_all, backward_all, labels)
        result = trainer.train(reps, val_split=0.2, max_iterations=500)

        # Should achieve high training accuracy on separable data
        assert result.train_accuracy > 0.8
        assert result.weights.hidden_dim == 2

    def test_handles_empty_input(self):
        """train handles empty input gracefully."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        forward = backend.zeros((0, 2))
        backward = backend.zeros((0, 2))
        labels = backend.zeros((0,))

        reps = trainer.build_representations(forward, backward, labels)
        result = trainer.train(reps)

        assert result.n_train == 0
        assert result.train_accuracy == 0.0

    def test_returns_validation_metrics(self):
        """train computes validation metrics when val_split > 0."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        n_samples = 40
        forward = backend.random_normal((n_samples, 4))
        backward = backend.random_normal((n_samples, 4))
        labels = backend.array([1.0 if i < n_samples // 2 else 0.0 for i in range(n_samples)])
        backend.eval(forward, backward, labels)

        reps = trainer.build_representations(forward, backward, labels)
        result = trainer.train(reps, val_split=0.25, max_iterations=100)

        assert result.val_accuracy is not None
        assert result.val_f1 is not None
        assert result.n_val > 0


class TestPredict:
    """Tests for BiLMProbeTrainer.predict()."""

    def test_predict_uses_threshold(self):
        """predict applies threshold correctly."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        # Create weights that give predictable outputs
        # For combined = [1, 1, 1, 1], weights @ combined + bias > 0.5
        weights = BiLMProbeWeights(
            weights=backend.array([0.5, 0.5, 0.5, 0.5]),
            bias=0.0,
            threshold=0.5,
            hidden_dim=2,
        )

        forward = backend.array([[1.0, 1.0], [-1.0, -1.0]])
        backward = backend.array([[1.0, 1.0], [-1.0, -1.0]])
        labels = backend.zeros((2,))

        reps = trainer.build_representations(forward, backward, labels)
        result = trainer.predict(reps, weights)

        backend.eval(result.predictions)
        preds = [int(x) for x in backend.tolist(result.predictions)]

        # First sample should be positive (high dot product)
        # Second sample should be negative (low dot product)
        assert preds[0] == 1
        assert preds[1] == 0


class TestPredictFromActivations:
    """Tests for BiLMProbeTrainer.predict_from_activations()."""

    def test_predict_from_activations_matches_predict(self):
        """predict_from_activations gives same results as predict."""
        backend = get_default_backend()
        trainer = BiLMProbeTrainer(backend=backend)

        weights = BiLMProbeWeights(
            weights=backend.array([0.25, 0.25, 0.25, 0.25]),
            bias=0.1,
            threshold=0.5,
            hidden_dim=2,
        )

        forward = backend.random_normal((10, 2))
        backward = backend.random_normal((10, 2))
        labels = backend.zeros((10,))
        backend.eval(forward, backward)

        # Via predict
        reps = trainer.build_representations(forward, backward, labels)
        result1 = trainer.predict(reps, weights)

        # Via predict_from_activations
        result2 = trainer.predict_from_activations(forward, backward, weights)

        backend.eval(result1.predictions, result2.predictions)
        preds1 = [int(x) for x in backend.tolist(result1.predictions)]
        preds2 = [int(x) for x in backend.tolist(result2.predictions)]

        assert preds1 == preds2


# =============================================================================
# BiLMProbeService Tests
# =============================================================================


class TestBiLMProbeService:
    """Tests for BiLMProbeService."""

    def test_train_basic(self):
        """train produces valid results."""
        from modelcypher.core.use_cases.bilm_probe_service import BiLMProbeService

        backend = get_default_backend()
        service = BiLMProbeService(backend)

        # Create test data
        n_pos = 20
        n_neg = 20
        hidden_dim = 4

        forward_pos = backend.ones((n_pos, hidden_dim)) + backend.random_normal((n_pos, hidden_dim)) * 0.1
        backward_pos = backend.ones((n_pos, hidden_dim)) + backend.random_normal((n_pos, hidden_dim)) * 0.1
        forward_neg = -backend.ones((n_neg, hidden_dim)) + backend.random_normal((n_neg, hidden_dim)) * 0.1
        backward_neg = -backend.ones((n_neg, hidden_dim)) + backend.random_normal((n_neg, hidden_dim)) * 0.1

        summary, result = service.train(
            forward_positive=forward_pos,
            backward_positive=backward_pos,
            forward_negative=forward_neg,
            backward_negative=backward_neg,
            val_split=0.2,
            max_iterations=200,
        )

        assert summary.n_train > 0
        assert 0.0 <= summary.train_accuracy <= 1.0
        assert result.weights is not None

    def test_predict_basic(self):
        """predict produces valid results."""
        from modelcypher.core.use_cases.bilm_probe_service import BiLMProbeService

        backend = get_default_backend()
        service = BiLMProbeService(backend)

        # Create simple weights
        weights = BiLMProbeWeights(
            weights=backend.array([0.5, 0.5, 0.5, 0.5]),
            bias=0.0,
            threshold=0.5,
            hidden_dim=2,
        )

        forward = backend.random_normal((10, 2))
        backward = backend.random_normal((10, 2))

        summary, result = service.predict(
            forward_acts=forward,
            backward_acts=backward,
            weights=weights,
        )

        assert summary.total_tokens == 10
        assert 0.0 <= summary.positive_rate <= 1.0
