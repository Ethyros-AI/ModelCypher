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

"""SEP Probe online training for semantic entropy probes.

Based on Kossen 2024: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection"

Key insight from the research:
- Semantic entropy is encoded in LLM hidden states
- Simple linear probes achieve R² ~ 0.8 on single generations
- Online training enables adaptation to new models without offline dataset

Online Training Strategy:
1. Sample Collection: During inference, occasionally compute full semantic entropy
   (expensive) and collect hidden states as ground-truth training pairs
2. Incremental Updates: Periodically update probe weights using collected samples
3. Convergence Monitoring: Track R² improvement and stop when converged
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


logger = logging.getLogger(__name__)


class ConvergenceStatus(str, Enum):
    """Convergence status of online training."""

    WARMUP = "warmup"
    """Not enough samples yet."""

    CONVERGING = "converging"
    """R² improving, training should continue."""

    CONVERGED = "converged"
    """R² stable, training complete."""

    DIVERGING = "diverging"
    """R² decreasing, may be overfitting."""


@dataclass(frozen=True)
class SEPProbeTrainingConfiguration:
    """Configuration for online SEP probe training."""

    learning_rate: float = 0.01
    """Learning rate for weight updates."""

    min_samples_before_update: int = 50
    """Minimum samples before first weight update."""

    update_frequency: int = 100
    """Steps between weight updates."""

    batch_size: int = 32
    """Batch size for gradient computation."""

    max_buffer_size: int = 1000
    """Maximum samples to retain in buffer."""

    l2_regularization: float = 0.001
    """L2 regularization strength."""

    convergence_threshold: float = 0.001
    """Convergence threshold (stop when R² improvement < threshold)."""

    convergence_patience: int = 5
    """Patience (number of updates without improvement before convergence)."""

    @classmethod
    def default(cls) -> SEPProbeTrainingConfiguration:
        """Default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> SEPProbeTrainingConfiguration:
        """Fast configuration for quick adaptation."""
        return cls(
            learning_rate=0.05,
            min_samples_before_update=20,
            update_frequency=50,
            batch_size=16,
            max_buffer_size=500,
            l2_regularization=0.01,
            convergence_threshold=0.005,
            convergence_patience=3,
        )


@dataclass(frozen=True)
class SEPTrainingSample:
    """A training sample for online probe training."""

    hidden_state: tuple[float, ...]
    """Hidden state from target layer."""

    ground_truth_entropy: float
    """Ground-truth semantic entropy (from full computation)."""

    layer_index: int
    """Layer index this sample is from."""

    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this sample was collected."""


@dataclass(frozen=True)
class SEPTrainingMetrics:
    """Metrics from a training update."""

    current_r2: float
    """Current R² on held-out samples."""

    r2_improvement: float
    """R² improvement from previous update."""

    samples_collected: int
    """Total samples collected."""

    samples_used_in_update: int
    """Samples used in this update."""

    update_step: int
    """Training step number."""

    training_loss: float
    """Training loss (MSE)."""

    convergence_status: ConvergenceStatus
    """Convergence status."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this update occurred."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_r2": self.current_r2,
            "r2_improvement": self.r2_improvement,
            "samples_collected": self.samples_collected,
            "samples_used_in_update": self.samples_used_in_update,
            "update_step": self.update_step,
            "training_loss": self.training_loss,
            "convergence_status": self.convergence_status.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class _TrainableProbeWeights:
    """Weights being trained for a single layer."""

    weights: list[float]
    bias: float
    mean_accumulator: float
    m2_accumulator: float  # For online variance
    sample_count: int


class SEPProbeTrainingError(Exception):
    """Error during SEP probe training."""

    pass


class SEPProbeOnlineTrainer:
    """Online trainer for semantic entropy probes.

    Collects training samples during inference and periodically updates
    probe weights to adapt to new models without offline datasets.
    """

    def __init__(
        self,
        target_layers: set[int],
        hidden_dim: int,
        configuration: SEPProbeTrainingConfiguration | None = None,
        model_id: str | None = None,
    ) -> None:
        """Create an online SEP probe trainer.

        Args:
            target_layers: Set of layer indices to train probes for.
            hidden_dim: Hidden dimension of the model.
            configuration: Training configuration.
            model_id: Optional model identifier.
        """
        self._config = configuration or SEPProbeTrainingConfiguration.default()
        self._target_layers = target_layers
        self._hidden_dim = hidden_dim
        self._model_id = model_id

        # Per-layer sample buffers
        self._sample_buffers: dict[int, list[SEPTrainingSample]] = {
            layer: [] for layer in target_layers
        }

        # Per-layer probe weights
        self._trainable_weights: dict[int, _TrainableProbeWeights] = {
            layer: _TrainableProbeWeights(
                weights=self._initialize_weights(hidden_dim),
                bias=0.0,
                mean_accumulator=0.0,
                m2_accumulator=0.0,
                sample_count=0,
            )
            for layer in target_layers
        }

        # Training history
        self._training_history: list[SEPTrainingMetrics] = []
        self._updates_without_improvement = 0
        self._previous_r2 = 0.0
        self._total_samples_collected = 0
        self._update_step = 0

    @property
    def model_id(self) -> str | None:
        """Model ID being trained."""
        return self._model_id

    @property
    def has_sufficient_samples(self) -> bool:
        """Whether sufficient samples have been collected for training."""
        return self._total_samples_collected >= self._config.min_samples_before_update

    @property
    def convergence_status(self) -> ConvergenceStatus:
        """Current convergence status."""
        if not self.has_sufficient_samples:
            return ConvergenceStatus.WARMUP

        if self._updates_without_improvement >= self._config.convergence_patience:
            return ConvergenceStatus.CONVERGED

        if len(self._training_history) > 2:
            recent_r2s = [m.current_r2 for m in self._training_history[-3:]]
            if recent_r2s[0] > recent_r2s[-1] + 0.05:
                return ConvergenceStatus.DIVERGING

        return ConvergenceStatus.CONVERGING

    @property
    def training_history(self) -> list[SEPTrainingMetrics]:
        """Get training history."""
        return list(self._training_history)

    def should_update(self, step: int) -> bool:
        """Whether training should update weights based on frequency."""
        return self.has_sufficient_samples and step % self._config.update_frequency == 0

    def collect_sample(
        self,
        hidden_states: dict[int, list[float]],
        ground_truth_entropy: float,
    ) -> None:
        """Collect a training sample from hidden states and ground-truth entropy.

        Args:
            hidden_states: Dictionary of layer index to hidden state vector.
            ground_truth_entropy: Full semantic entropy value (ground truth).
        """
        for layer in self._target_layers:
            hidden_array = hidden_states.get(layer)
            if hidden_array is None or len(hidden_array) != self._hidden_dim:
                continue

            sample = SEPTrainingSample(
                hidden_state=tuple(hidden_array),
                ground_truth_entropy=ground_truth_entropy,
                layer_index=layer,
            )

            self._sample_buffers[layer].append(sample)

            # Evict old samples if necessary
            if len(self._sample_buffers[layer]) > self._config.max_buffer_size:
                self._sample_buffers[layer].pop(0)

            # Update online statistics
            self._update_online_statistics(layer, hidden_array)

        self._total_samples_collected += 1

        if self._total_samples_collected % 10 == 0:
            logger.debug("Collected %d training samples", self._total_samples_collected)

    def update_weights(self) -> SEPTrainingMetrics:
        """Update probe weights using collected samples.

        Returns:
            Training metrics from this update.

        Raises:
            SEPProbeTrainingError: If insufficient samples.
        """
        if not self.has_sufficient_samples:
            raise SEPProbeTrainingError(
                f"Insufficient training samples: need {self._config.min_samples_before_update}, "
                f"have {self._total_samples_collected}"
            )

        self._update_step += 1
        total_loss = 0.0
        total_samples_used = 0

        # Update each layer's probe
        for layer in self._target_layers:
            samples = self._sample_buffers.get(layer, [])
            probe_weights = self._trainable_weights.get(layer)

            if not samples or len(samples) < self._config.batch_size or not probe_weights:
                continue

            # Sample a batch
            batch = random.sample(samples, min(len(samples), self._config.batch_size))
            total_samples_used += len(batch)

            # Compute gradient and update
            loss = self._update_layer_weights(layer, batch, probe_weights)
            total_loss += loss

        # Compute current R² on held-out samples
        current_r2 = self._compute_validation_r2()
        improvement = current_r2 - self._previous_r2

        # Track convergence
        if improvement < self._config.convergence_threshold:
            self._updates_without_improvement += 1
        else:
            self._updates_without_improvement = 0
        self._previous_r2 = current_r2

        metrics = SEPTrainingMetrics(
            current_r2=current_r2,
            r2_improvement=improvement,
            samples_collected=self._total_samples_collected,
            samples_used_in_update=total_samples_used,
            update_step=self._update_step,
            training_loss=total_loss / max(1, len(self._target_layers)),
            convergence_status=self.convergence_status,
        )

        self._training_history.append(metrics)

        logger.info(
            "SEP probe update #%d: R²=%.3f (Δ=%+.4f), loss=%.4f",
            self._update_step,
            current_r2,
            improvement,
            metrics.training_loss,
        )

        return metrics

    def export_weights(self) -> dict:
        """Export trained weights as a dictionary.

        Returns:
            Dictionary with trained weights and metadata.
        """
        layer_weights = []

        for layer in sorted(self._target_layers):
            probe = self._trainable_weights.get(layer)
            if probe is None:
                continue

            mean, std = self._compute_normalization_stats(layer)
            r2 = self._compute_layer_r2(layer)

            layer_weights.append({
                "layer": layer,
                "weights": probe.weights,
                "bias": probe.bias,
                "validation_r2": r2,
                "train_mean": mean,
                "train_std": std,
            })

        return {
            "model_id": self._model_id or "unknown",
            "hidden_dim": self._hidden_dim,
            "target_layers": sorted(self._target_layers),
            "layer_weights": layer_weights,
            "training_samples": self._total_samples_collected,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        """Reset training state."""
        for layer in self._target_layers:
            self._sample_buffers[layer] = []
            self._trainable_weights[layer] = _TrainableProbeWeights(
                weights=self._initialize_weights(self._hidden_dim),
                bias=0.0,
                mean_accumulator=0.0,
                m2_accumulator=0.0,
                sample_count=0,
            )

        self._training_history.clear()
        self._updates_without_improvement = 0
        self._previous_r2 = 0.0
        self._total_samples_collected = 0
        self._update_step = 0

    def _initialize_weights(self, dim: int) -> list[float]:
        """Initialize weights using Xavier initialization."""
        scale = math.sqrt(2.0 / dim)
        return [random.uniform(-scale, scale) for _ in range(dim)]

    def _update_online_statistics(self, layer: int, hidden_state: list[float]) -> None:
        """Update online mean/variance statistics for normalization."""
        probe = self._trainable_weights.get(layer)
        if probe is None:
            return

        probe.sample_count += 1
        n = float(probe.sample_count)

        # Compute mean of hidden state
        sample_mean = sum(hidden_state) / len(hidden_state)

        # Welford's online algorithm for mean/variance
        delta = sample_mean - probe.mean_accumulator
        probe.mean_accumulator += delta / n
        delta2 = sample_mean - probe.mean_accumulator
        probe.m2_accumulator += delta * delta2

    def _update_layer_weights(
        self,
        layer: int,
        batch: list[SEPTrainingSample],
        weights: _TrainableProbeWeights,
    ) -> float:
        """Update layer weights using gradient descent."""
        lr = self._config.learning_rate
        l2 = self._config.l2_regularization
        dim = self._hidden_dim

        total_loss = 0.0
        grad_w = [0.0] * dim
        grad_b = 0.0

        mean, std = self._compute_normalization_stats(layer)

        for sample in batch:
            # Normalize hidden state
            normalized = [
                (sample.hidden_state[i] - mean) / max(std, 1e-6) for i in range(dim)
            ]

            # Forward pass: prediction = w^T h + b
            prediction = weights.bias
            for i in range(dim):
                prediction += weights.weights[i] * normalized[i]
            prediction = max(0.0, min(1.0, prediction))  # Clamp

            # Loss: MSE
            error = prediction - sample.ground_truth_entropy
            total_loss += error * error

            # Gradients
            for i in range(dim):
                grad_w[i] += 2 * error * normalized[i] / len(batch)
            grad_b += 2 * error / len(batch)

        # Apply gradients with L2 regularization
        for i in range(dim):
            weights.weights[i] -= lr * (grad_w[i] + l2 * weights.weights[i])
        weights.bias -= lr * grad_b

        return total_loss / len(batch)

    def _compute_normalization_stats(self, layer: int) -> tuple[float, float]:
        """Compute normalization statistics for a layer."""
        probe = self._trainable_weights.get(layer)
        if probe is None or probe.sample_count <= 1:
            return 0.0, 1.0

        mean = probe.mean_accumulator
        variance = probe.m2_accumulator / (probe.sample_count - 1)
        std = math.sqrt(max(variance, 1e-6))

        return mean, std

    def _compute_validation_r2(self) -> float:
        """Compute validation R² across all layers."""
        total_ss = 0.0
        residual_ss = 0.0
        sample_count = 0

        for layer in self._target_layers:
            samples = self._sample_buffers.get(layer, [])
            probe = self._trainable_weights.get(layer)

            if not samples or probe is None:
                continue

            # Use last 20% as validation
            validation_count = max(1, len(samples) // 5)
            validation = samples[-validation_count:]

            mean, std = self._compute_normalization_stats(layer)
            target_mean = sum(s.ground_truth_entropy for s in validation) / len(validation)

            for sample in validation:
                # Predict
                normalized = [
                    (sample.hidden_state[i] - mean) / max(std, 1e-6)
                    for i in range(self._hidden_dim)
                ]

                prediction = probe.bias
                for i in range(self._hidden_dim):
                    prediction += probe.weights[i] * normalized[i]
                prediction = max(0.0, min(1.0, prediction))

                error = prediction - sample.ground_truth_entropy
                residual_ss += error * error

                total_error = sample.ground_truth_entropy - target_mean
                total_ss += total_error * total_error

                sample_count += 1

        if total_ss <= 0:
            return 0.0
        return max(0.0, 1.0 - residual_ss / total_ss)

    def _compute_layer_r2(self, layer: int) -> float:
        """Compute R² for a single layer."""
        samples = self._sample_buffers.get(layer, [])
        probe = self._trainable_weights.get(layer)

        if len(samples) <= 10 or probe is None:
            return 0.0

        mean, std = self._compute_normalization_stats(layer)

        total_ss = 0.0
        residual_ss = 0.0
        target_mean = sum(s.ground_truth_entropy for s in samples) / len(samples)

        for sample in samples:
            normalized = [
                (sample.hidden_state[i] - mean) / max(std, 1e-6)
                for i in range(self._hidden_dim)
            ]

            prediction = probe.bias
            for i in range(self._hidden_dim):
                prediction += probe.weights[i] * normalized[i]
            prediction = max(0.0, min(1.0, prediction))

            error = prediction - sample.ground_truth_entropy
            residual_ss += error * error

            total_error = sample.ground_truth_entropy - target_mean
            total_ss += total_error * total_error

        if total_ss <= 0:
            return 0.0
        return max(0.0, 1.0 - residual_ss / total_ss)
