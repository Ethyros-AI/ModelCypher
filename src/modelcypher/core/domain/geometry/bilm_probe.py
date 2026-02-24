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
Bidirectional LM (biLM) Probes for Token-Level Classification.

Implements biLM probes from "Shaping capabilities with token-level data
filtering" (arXiv:2601.21571v1).

Key ideas:
- Use bidirectional representations (forward + backward LM) for richer context
- Train linear probe on concatenated representations
- Classify tokens as belonging to target domain or not

The bidirectional approach captures both preceding and following context,
making it more effective for domain identification than unidirectional probes.

References:
    - "Shaping capabilities with token-level data filtering" (Anthropic, 2025)
    - Peters et al. (2018) "Deep contextualized word representations" (ELMo)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class BiLMRepresentations:
    """Bidirectional LM representations for probe training.

    Attributes
    ----------
    forward : Array
        Forward (left-to-right) representations. Shape: [n_samples, hidden_dim].
    backward : Array
        Backward (right-to-left) representations. Shape: [n_samples, hidden_dim].
    combined : Array
        Concatenated representations. Shape: [n_samples, 2*hidden_dim].
    labels : Array
        Binary labels for each sample. Shape: [n_samples].
    """

    forward: "Array"
    backward: "Array"
    combined: "Array"
    labels: "Array"


@dataclass(frozen=True)
class BiLMProbeWeights:
    """Trained biLM probe weights.

    Attributes
    ----------
    weights : Array
        Linear probe weights. Shape: [2*hidden_dim].
    bias : float
        Probe bias term.
    threshold : float
        Classification threshold (0.5 for balanced data, Bayes-optimal decision boundary).
    hidden_dim : int
        Hidden dimension of the underlying LM.
    """

    weights: "Array"
    bias: float
    threshold: float
    hidden_dim: int


@dataclass(frozen=True)
class BiLMProbeResult:
    """Result of biLM probe training.

    Attributes
    ----------
    weights : BiLMProbeWeights
        Trained probe weights.
    train_accuracy : float
        Accuracy on training set.
    train_precision : float
        Precision on training set.
    train_recall : float
        Recall on training set.
    train_f1 : float
        F1 score on training set.
    val_accuracy : float | None
        Accuracy on validation set, if validation was performed.
    val_precision : float | None
        Precision on validation set.
    val_recall : float | None
        Recall on validation set.
    val_f1 : float | None
        F1 score on validation set.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    """

    weights: BiLMProbeWeights
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float
    val_accuracy: float | None
    val_precision: float | None
    val_recall: float | None
    val_f1: float | None
    n_train: int
    n_val: int


@dataclass(frozen=True)
class PredictionResult:
    """Result of biLM probe prediction.

    Attributes
    ----------
    predictions : Array
        Binary predictions. Shape: [n_samples].
    probabilities : Array
        Predicted probabilities. Shape: [n_samples].
    """

    predictions: "Array"
    probabilities: "Array"


class BiLMProbeTrainer:
    """Trains and applies biLM probes for token classification.

    Implements a simple linear probe on concatenated bidirectional
    representations. Training uses logistic regression with gradient descent.

    Example
    -------
    >>> trainer = BiLMProbeTrainer()
    >>> representations = trainer.build_representations(
    ...     forward_acts=forward_hidden_states,
    ...     backward_acts=backward_hidden_states,
    ...     labels=token_labels,
    ... )
    >>> result = trainer.train(representations, val_split=0.1)
    >>> predictions = trainer.predict(test_representations, result.weights)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize trainer.

        Parameters
        ----------
        backend : Backend, optional
            Computation backend. If None, uses default.
        """
        self._backend = backend or get_default_backend()

    @property
    def backend(self) -> "Backend":
        """Get computation backend."""
        return self._backend

    def build_representations(
        self,
        forward_acts: "Array",
        backward_acts: "Array",
        labels: "Array",
    ) -> BiLMRepresentations:
        """Build bidirectional representations for training.

        Parameters
        ----------
        forward_acts : Array
            Forward LM hidden states. Shape: [n_samples, hidden_dim].
        backward_acts : Array
            Backward LM hidden states. Shape: [n_samples, hidden_dim].
        labels : Array
            Binary labels. Shape: [n_samples].

        Returns
        -------
        BiLMRepresentations
            Combined representations with labels.
        """
        b = self._backend

        forward = b.astype(forward_acts, "float32")
        backward = b.astype(backward_acts, "float32")
        labels_arr = b.astype(labels, "float32")
        b.eval(forward, backward, labels_arr)

        # Concatenate forward and backward
        combined = b.concatenate([forward, backward], axis=1)
        b.eval(combined)

        return BiLMRepresentations(
            forward=forward,
            backward=backward,
            combined=combined,
            labels=labels_arr,
        )

    def train(
        self,
        representations: BiLMRepresentations,
        val_split: float = 0.1,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        convergence_threshold: float | None = None,
    ) -> BiLMProbeResult:
        """Train a linear probe on biLM representations.

        Uses logistic regression with gradient descent. The convergence
        threshold is derived from machine epsilon if not provided.

        Parameters
        ----------
        representations : BiLMRepresentations
            Training representations.
        val_split : float
            Fraction of data to use for validation.
        learning_rate : float
            Learning rate for gradient descent.
        max_iterations : int
            Maximum training iterations.
        convergence_threshold : float, optional
            Stop when loss change is below this. Derived from eps if None.

        Returns
        -------
        BiLMProbeResult
            Training results with weights and metrics.
        """
        b = self._backend

        X = representations.combined
        y = representations.labels
        b.eval(X, y)

        n_samples = int(X.shape[0])
        hidden_dim_2x = int(X.shape[1])
        hidden_dim = hidden_dim_2x // 2

        if n_samples == 0:
            weights = BiLMProbeWeights(
                weights=b.zeros((hidden_dim_2x,)),
                bias=0.0,
                threshold=0.5,
                hidden_dim=hidden_dim,
            )
            return BiLMProbeResult(
                weights=weights,
                train_accuracy=0.0,
                train_precision=0.0,
                train_recall=0.0,
                train_f1=0.0,
                val_accuracy=None,
                val_precision=None,
                val_recall=None,
                val_f1=None,
                n_train=0,
                n_val=0,
            )

        # Split into train/val
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        # Random shuffle indices
        indices = b.randperm(n_samples)
        b.eval(indices)

        train_idx = b.take(indices, b.arange(n_train), axis=0)
        X_train = b.take(X, train_idx, axis=0)
        y_train = b.take(y, train_idx, axis=0)
        b.eval(X_train, y_train)

        if n_val > 0:
            val_idx = b.take(indices, b.arange(n_train, n_samples), axis=0)
            X_val = b.take(X, val_idx, axis=0)
            y_val = b.take(y, val_idx, axis=0)
            b.eval(X_val, y_val)
        else:
            X_val = None
            y_val = None

        # Derive convergence threshold from machine epsilon
        if convergence_threshold is None:
            eps = machine_epsilon(b, X_train)
            convergence_threshold = b.sqrt(b.array(eps))
            b.eval(convergence_threshold)
            convergence_threshold = float(b.to_scalar(convergence_threshold))

        # Initialize weights
        init_scale = 1.0 / (hidden_dim_2x ** 0.5)
        weights = b.random_normal(shape=(hidden_dim_2x,)) * init_scale
        bias = b.array(0.0)
        b.eval(weights, bias)

        # Training loop
        prev_loss = float("inf")
        for iteration in range(max_iterations):
            # Forward pass: logits = X @ w + b
            logits = b.matmul(X_train, b.reshape(weights, (-1, 1)))
            logits = b.squeeze(logits, axis=1) + bias
            b.eval(logits)

            # Sigmoid
            probs = self._sigmoid(logits)
            b.eval(probs)

            # Binary cross-entropy loss
            eps = regularization_epsilon(b, probs)
            probs_clamped = b.clip(probs, eps, 1.0 - eps)
            loss = -b.mean(
                y_train * b.log(probs_clamped) + (1.0 - y_train) * b.log(1.0 - probs_clamped)
            )
            b.eval(loss)
            loss_val = float(b.to_scalar(loss))

            # Check convergence
            if abs(prev_loss - loss_val) < convergence_threshold:
                break
            prev_loss = loss_val

            # Backward pass
            grad_logits = probs - y_train  # [n_train]
            grad_weights = b.matmul(b.transpose(X_train), b.reshape(grad_logits, (-1, 1)))
            grad_weights = b.squeeze(grad_weights, axis=1) / float(n_train)
            grad_bias = b.mean(grad_logits)
            b.eval(grad_weights, grad_bias)

            # Update
            weights = weights - learning_rate * grad_weights
            bias = bias - learning_rate * grad_bias
            b.eval(weights, bias)

        # Compute metrics
        train_metrics = self._compute_metrics(X_train, y_train, weights, bias)

        if X_val is not None and y_val is not None:
            val_metrics = self._compute_metrics(X_val, y_val, weights, bias)
        else:
            val_metrics = None

        bias_val = float(b.to_scalar(bias))
        probe_weights = BiLMProbeWeights(
            weights=weights,
            bias=bias_val,
            threshold=0.5,
            hidden_dim=hidden_dim,
        )

        return BiLMProbeResult(
            weights=probe_weights,
            train_accuracy=train_metrics["accuracy"],
            train_precision=train_metrics["precision"],
            train_recall=train_metrics["recall"],
            train_f1=train_metrics["f1"],
            val_accuracy=val_metrics["accuracy"] if val_metrics else None,
            val_precision=val_metrics["precision"] if val_metrics else None,
            val_recall=val_metrics["recall"] if val_metrics else None,
            val_f1=val_metrics["f1"] if val_metrics else None,
            n_train=n_train,
            n_val=n_val,
        )

    def predict(
        self,
        representations: BiLMRepresentations,
        weights: BiLMProbeWeights,
    ) -> PredictionResult:
        """Predict using trained probe.

        Parameters
        ----------
        representations : BiLMRepresentations
            Representations to classify.
        weights : BiLMProbeWeights
            Trained probe weights.

        Returns
        -------
        PredictionResult
            Predictions and probabilities.
        """
        b = self._backend

        X = representations.combined
        b.eval(X)

        logits = b.matmul(X, b.reshape(weights.weights, (-1, 1)))
        logits = b.squeeze(logits, axis=1) + weights.bias
        b.eval(logits)

        probs = self._sigmoid(logits)
        predictions = b.astype(probs >= weights.threshold, "int32")
        b.eval(probs, predictions)

        return PredictionResult(
            predictions=predictions,
            probabilities=probs,
        )

    def predict_from_activations(
        self,
        forward_acts: "Array",
        backward_acts: "Array",
        weights: BiLMProbeWeights,
    ) -> PredictionResult:
        """Predict directly from activations without pre-built representations.

        Parameters
        ----------
        forward_acts : Array
            Forward LM hidden states. Shape: [n_samples, hidden_dim].
        backward_acts : Array
            Backward LM hidden states. Shape: [n_samples, hidden_dim].
        weights : BiLMProbeWeights
            Trained probe weights.

        Returns
        -------
        PredictionResult
            Predictions and probabilities.
        """
        b = self._backend

        forward = b.astype(forward_acts, "float32")
        backward = b.astype(backward_acts, "float32")
        combined = b.concatenate([forward, backward], axis=1)
        b.eval(combined)

        logits = b.matmul(combined, b.reshape(weights.weights, (-1, 1)))
        logits = b.squeeze(logits, axis=1) + weights.bias
        b.eval(logits)

        probs = self._sigmoid(logits)
        predictions = b.astype(probs >= weights.threshold, "int32")
        b.eval(probs, predictions)

        return PredictionResult(
            predictions=predictions,
            probabilities=probs,
        )

    def _sigmoid(self, x: "Array") -> "Array":
        """Numerically stable sigmoid.

        Uses: sigmoid(x) = 1 / (1 + exp(-x))
        With clamping for numerical stability.
        """
        b = self._backend
        # Clamp x to avoid overflow in exp
        # For float32: exp(88) overflows, so clamp to [-50, 50] is safe
        x_clamped = b.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + b.exp(-x_clamped))

    def _compute_metrics(
        self,
        X: "Array",
        y: "Array",
        weights: "Array",
        bias: "Array",
    ) -> dict[str, float]:
        """Compute classification metrics."""
        b = self._backend

        logits = b.matmul(X, b.reshape(weights, (-1, 1)))
        logits = b.squeeze(logits, axis=1) + bias
        probs = self._sigmoid(logits)
        preds = b.astype(probs >= 0.5, "float32")
        b.eval(preds)

        # Convert to float for comparison
        y_float = b.astype(y, "float32")

        # True positives, false positives, false negatives
        tp = b.sum(preds * y_float)
        fp = b.sum(preds * (1.0 - y_float))
        fn = b.sum((1.0 - preds) * y_float)
        tn = b.sum((1.0 - preds) * (1.0 - y_float))
        b.eval(tp, fp, fn, tn)

        tp_val = float(b.to_scalar(tp))
        fp_val = float(b.to_scalar(fp))
        fn_val = float(b.to_scalar(fn))
        tn_val = float(b.to_scalar(tn))

        eps = division_epsilon(b, preds)

        accuracy = (tp_val + tn_val) / max(tp_val + fp_val + fn_val + tn_val, eps)
        precision = tp_val / max(tp_val + fp_val, eps)
        recall = tp_val / max(tp_val + fn_val, eps)
        f1 = 2 * precision * recall / max(precision + recall, eps)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


__all__ = [
    "BiLMRepresentations",
    "BiLMProbeWeights",
    "BiLMProbeResult",
    "PredictionResult",
    "BiLMProbeTrainer",
]
