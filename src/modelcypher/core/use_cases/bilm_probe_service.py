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
Bidirectional LM Probe Service.

Provides CLI-consumable operations for training and using biLM probes
for token-level domain classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.bilm_probe import (
    BiLMProbeResult,
    BiLMProbeTrainer,
    BiLMProbeWeights,
    PredictionResult,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary of biLM probe training run.

    Attributes
    ----------
    train_accuracy : float
        Training accuracy.
    train_f1 : float
        Training F1 score.
    val_accuracy : float | None
        Validation accuracy.
    val_f1 : float | None
        Validation F1 score.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    output_path : str | None
        Path where weights were saved.
    """

    train_accuracy: float
    train_f1: float
    val_accuracy: float | None
    val_f1: float | None
    n_train: int
    n_val: int
    output_path: str | None


@dataclass(frozen=True)
class PredictionRunResult:
    """Summary of biLM probe prediction run.

    Attributes
    ----------
    total_tokens : int
        Total tokens classified.
    positive_predictions : int
        Number of positive predictions.
    positive_rate : float
        Fraction predicted positive.
    output_path : str | None
        Path where predictions were saved.
    """

    total_tokens: int
    positive_predictions: int
    positive_rate: float
    output_path: str | None


class BiLMProbeService:
    """Service for biLM probe training and inference.

    Provides methods for:
    - Training probes on labeled token data
    - Running inference on new data
    - Loading/saving probe weights
    """

    def __init__(self, backend: "Backend") -> None:
        """Initialize service.

        Parameters
        ----------
        backend : Backend
            Computation backend.
        """
        self._backend = backend

    def train(
        self,
        forward_positive: "Array",
        backward_positive: "Array",
        forward_negative: "Array",
        backward_negative: "Array",
        val_split: float = 0.1,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        output_path: str | None = None,
    ) -> tuple[TrainingRunResult, BiLMProbeResult]:
        """Train a biLM probe on positive and negative examples.

        Parameters
        ----------
        forward_positive : Array
            Forward LM activations for positive samples. Shape: [n_pos, hidden_dim].
        backward_positive : Array
            Backward LM activations for positive samples. Shape: [n_pos, hidden_dim].
        forward_negative : Array
            Forward LM activations for negative samples. Shape: [n_neg, hidden_dim].
        backward_negative : Array
            Backward LM activations for negative samples. Shape: [n_neg, hidden_dim].
        val_split : float
            Fraction of data for validation.
        learning_rate : float
            Learning rate for training.
        max_iterations : int
            Maximum training iterations.
        output_path : str, optional
            Path to save trained weights.

        Returns
        -------
        tuple[TrainingRunResult, BiLMProbeResult]
            Summary and detailed results.
        """
        b = self._backend
        trainer = BiLMProbeTrainer(backend=b)

        # Combine positive and negative samples
        forward_pos = b.astype(forward_positive, "float32")
        backward_pos = b.astype(backward_positive, "float32")
        forward_neg = b.astype(forward_negative, "float32")
        backward_neg = b.astype(backward_negative, "float32")
        b.eval(forward_pos, backward_pos, forward_neg, backward_neg)

        n_pos = int(forward_pos.shape[0])
        n_neg = int(forward_neg.shape[0])

        forward_all = b.concatenate([forward_pos, forward_neg], axis=0)
        backward_all = b.concatenate([backward_pos, backward_neg], axis=0)
        labels = b.concatenate([b.ones((n_pos,)), b.zeros((n_neg,))], axis=0)
        b.eval(forward_all, backward_all, labels)

        # Build representations
        representations = trainer.build_representations(
            forward_acts=forward_all,
            backward_acts=backward_all,
            labels=labels,
        )

        # Train
        result = trainer.train(
            representations=representations,
            val_split=val_split,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
        )

        # Save if output path provided
        if output_path:
            self.save_weights(result.weights, output_path)

        summary = TrainingRunResult(
            train_accuracy=result.train_accuracy,
            train_f1=result.train_f1,
            val_accuracy=result.val_accuracy,
            val_f1=result.val_f1,
            n_train=result.n_train,
            n_val=result.n_val,
            output_path=output_path,
        )

        return summary, result

    def predict(
        self,
        forward_acts: "Array",
        backward_acts: "Array",
        weights: BiLMProbeWeights,
        output_path: str | None = None,
    ) -> tuple[PredictionRunResult, PredictionResult]:
        """Run probe inference on activations.

        Parameters
        ----------
        forward_acts : Array
            Forward LM activations. Shape: [n_samples, hidden_dim].
        backward_acts : Array
            Backward LM activations. Shape: [n_samples, hidden_dim].
        weights : BiLMProbeWeights
            Trained probe weights.
        output_path : str, optional
            Path to save predictions.

        Returns
        -------
        tuple[PredictionRunResult, PredictionResult]
            Summary and detailed results.
        """
        b = self._backend
        trainer = BiLMProbeTrainer(backend=b)

        result = trainer.predict_from_activations(
            forward_acts=forward_acts,
            backward_acts=backward_acts,
            weights=weights,
        )

        # Compute summary
        pred_sum = b.sum(result.predictions)
        b.eval(pred_sum)
        positive_count = int(b.to_scalar(pred_sum))
        total = int(result.predictions.shape[0])
        positive_rate = positive_count / max(total, 1)

        # Save if output path provided
        if output_path:
            self._save_predictions(result, output_path)

        summary = PredictionRunResult(
            total_tokens=total,
            positive_predictions=positive_count,
            positive_rate=positive_rate,
            output_path=output_path,
        )

        return summary, result

    def save_weights(self, weights: BiLMProbeWeights, path: str) -> None:
        """Save probe weights to JSON file.

        Parameters
        ----------
        weights : BiLMProbeWeights
            Weights to save.
        path : str
            Output path.
        """
        b = self._backend
        b.eval(weights.weights)

        data = {
            "weights": [float(x) for x in b.tolist(weights.weights)],
            "bias": weights.bias,
            "threshold": weights.threshold,
            "hidden_dim": weights.hidden_dim,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_weights(self, path: str) -> BiLMProbeWeights:
        """Load probe weights from JSON file.

        Parameters
        ----------
        path : str
            Path to weights file.

        Returns
        -------
        BiLMProbeWeights
            Loaded weights.
        """
        b = self._backend
        data = json.loads(Path(path).read_text())

        return BiLMProbeWeights(
            weights=b.array(data["weights"], dtype="float32"),
            bias=float(data["bias"]),
            threshold=float(data.get("threshold", 0.5)),
            hidden_dim=int(data["hidden_dim"]),
        )

    def _save_predictions(self, result: PredictionResult, path: str) -> None:
        """Save predictions to JSONL file."""
        b = self._backend
        b.eval(result.predictions, result.probabilities)

        preds_list = [int(x) for x in b.tolist(result.predictions)]
        probs_list = [float(x) for x in b.tolist(result.probabilities)]

        with open(path, "w") as f:
            for i, (pred, prob) in enumerate(zip(preds_list, probs_list)):
                record = {
                    "index": i,
                    "prediction": pred,
                    "probability": prob,
                }
                f.write(json.dumps(record) + "\n")

    @staticmethod
    def training_payload(result: TrainingRunResult) -> dict:
        """Convert training result to CLI payload."""
        return {
            "trainAccuracy": result.train_accuracy,
            "trainF1": result.train_f1,
            "valAccuracy": result.val_accuracy,
            "valF1": result.val_f1,
            "nTrain": result.n_train,
            "nVal": result.n_val,
            "outputPath": result.output_path,
        }

    @staticmethod
    def prediction_payload(result: PredictionRunResult) -> dict:
        """Convert prediction result to CLI payload."""
        return {
            "totalTokens": result.total_tokens,
            "positivePredictions": result.positive_predictions,
            "positiveRate": result.positive_rate,
            "outputPath": result.output_path,
        }


__all__ = [
    "BiLMProbeService",
    "TrainingRunResult",
    "PredictionRunResult",
]
