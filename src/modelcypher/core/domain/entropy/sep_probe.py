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
Semantic Entropy Probe (SEP) for Fast Entropy Prediction.

Ported 1:1 from the reference Swift implementation.

Linear probes on transformer hidden states predict semantic entropy
with R² ~ 0.8 while being 1000x faster than full computation (0.3ms vs 15s).

Architecture:
    For each layer l: ŜE_l = w_l^T h_l + b_l
    Final: ŜE = Fréchet mean of per-layer predictions (no weighting)

Research Basis:
    arXiv:2406.15927 - Semantic Entropy Probes

No configuration needed. All parameters derived from:
- Model architecture (layer_count, hidden_dim)
- Loaded weights (R² threshold from weight quality distribution)
- Baseline measurements (circuit breaker from entropy distribution)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Research-based constants (arXiv:2406.15927)
# =============================================================================

# Target layer fractions - most predictive for entropy estimation
# These are fixed by research, not configuration
TARGET_LAYER_FRACTIONS = [0.75, 0.78, 0.81, 0.84, 0.875]


def _target_layers_for_model(layer_count: int) -> list[int]:
    """Derive target layer indices from model layer count."""
    return [int(layer_count * f) for f in TARGET_LAYER_FRACTIONS]


# =============================================================================
# Exceptions
# =============================================================================


class SEPProbeError(Exception):
    """SEP probe errors."""

    pass


class WeightsNotLoadedError(SEPProbeError):
    """Weights not loaded."""

    pass


class IncompatibleWeightsError(SEPProbeError):
    """Weight dimensions mismatch."""

    pass


class LayerNotFoundError(SEPProbeError):
    """Probe weights for layer not found."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LayerProbeWeights:
    """Trained weights for a single layer probe."""

    layer: int
    weights: list[float]
    bias: float
    validation_r2: float
    train_mean: float = 0.0
    train_std: float = 1.0


@dataclass
class ProbeWeightsBundle:
    """Container for all probe weights."""

    model_id: str
    layer_count: int
    hidden_dim: int
    layer_weights: list[LayerProbeWeights]
    training_samples: int
    trained_at: datetime


@dataclass
class PredictionResult:
    """Result from SEP probe prediction."""

    predicted_entropy: float
    layer_predictions: dict[int, float]
    should_trip_circuit_breaker: bool
    latency_ms: float


# =============================================================================
# SEP Probe
# =============================================================================


class SEPProbe:
    """
    Semantic Entropy Probe using linear projection on hidden states.

    Fast (0.3ms) prediction of semantic entropy from layer activations.

    All parameters are derived from data:
    - layer_count, hidden_dim: From model architecture
    - Target layers: Research-based fractions (75-87.5%)
    - R² threshold: Derived from loaded weight quality distribution
    - Circuit breaker: Derived from baseline entropy distribution

    Usage:
        probe = SEPProbe(layer_count=32, hidden_dim=4096)
        await probe.load_weights(path)

        # During inference with hidden states:
        result = probe.predict(hidden_states)
        if result.should_trip_circuit_breaker:
            # Handle high entropy
    """

    def __init__(
        self,
        layer_count: int,
        hidden_dim: int,
        backend: "Backend | None" = None,
    ) -> None:
        """Create SEP probe for a specific model architecture.

        Args:
            layer_count: Number of transformer layers in the model.
            hidden_dim: Hidden dimension of the model.
            backend: Optional compute backend.
        """
        self._layer_count = layer_count
        self._hidden_dim = hidden_dim
        self._target_layers = _target_layers_for_model(layer_count)
        self._backend = backend or get_default_backend()

        # Weights storage
        self._probe_weights: dict[int, LayerProbeWeights] = {}
        self._cached_weight_arrays: "dict[int, Array]" = {}
        self._is_ready: bool = False
        self._trained_model_id: str | None = None

        # Thresholds derived from data
        self._min_r2_threshold: float | None = None
        self._circuit_breaker_threshold: float | None = None

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        return self._layer_count

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension."""
        return self._hidden_dim

    @property
    def target_layers(self) -> list[int]:
        """Target layer indices for probing."""
        return self._target_layers

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def trained_model_id(self) -> str | None:
        return self._trained_model_id

    def set_circuit_breaker_from_baseline(self, baseline_entropies: list[float]) -> None:
        """Set circuit breaker threshold from baseline entropy distribution.

        Uses mean + 2*std as threshold (97.7th percentile).

        Args:
            baseline_entropies: Entropy values from baseline model runs.
        """
        if not baseline_entropies:
            return

        n = len(baseline_entropies)
        mean = sum(baseline_entropies) / n
        variance = sum((x - mean) ** 2 for x in baseline_entropies) / n
        std = math.sqrt(variance)

        self._circuit_breaker_threshold = mean + 2 * std

    async def load_weights(self, path: Path) -> None:
        """Load trained probe weights from file.

        R² threshold is derived from the quality distribution of loaded weights:
        Uses mean - 2*std to filter out low-quality probes.

        Args:
            path: Path to weights file.
        """
        data = json.loads(path.read_text())

        loaded_hidden_dim = data.get("hidden_dim", 4096)
        if loaded_hidden_dim != self._hidden_dim:
            raise IncompatibleWeightsError(
                f"Incompatible: expected hidden_dim={self._hidden_dim}, found {loaded_hidden_dim}"
            )

        # Collect all R² values to derive threshold
        all_r2_values: list[float] = []
        layer_weights_data = data.get("layer_weights", [])
        for lw in layer_weights_data:
            r2 = lw.get("validation_r2", 0.0)
            all_r2_values.append(r2)

        # Derive R² threshold from quality distribution: mean - 2*std
        if all_r2_values:
            n = len(all_r2_values)
            mean_r2 = sum(all_r2_values) / n
            variance_r2 = sum((x - mean_r2) ** 2 for x in all_r2_values) / n
            std_r2 = math.sqrt(variance_r2)
            self._min_r2_threshold = max(0.0, mean_r2 - 2 * std_r2)

        # Load weights that pass R² threshold and are in target layers
        for lw in layer_weights_data:
            layer = lw["layer"]
            r2 = lw.get("validation_r2", 0.0)

            passes_r2 = self._min_r2_threshold is None or r2 >= self._min_r2_threshold
            if layer in self._target_layers and passes_r2:
                weights = LayerProbeWeights(
                    layer=layer,
                    weights=lw["weights"],
                    bias=lw.get("bias", 0.0),
                    validation_r2=r2,
                    train_mean=lw.get("train_mean", 0.0),
                    train_std=lw.get("train_std", 1.0),
                )
                self._probe_weights[layer] = weights
                self._cached_weight_arrays[layer] = self._backend.array(weights.weights)

        self._trained_model_id = data.get("model_id", "unknown")
        self._is_ready = len(self._probe_weights) > 0

    def register_weights(self, weights: list[LayerProbeWeights], model_id: str) -> None:
        """Register weights directly (for testing or in-memory).

        R² threshold is derived from the quality distribution of provided weights.

        Args:
            weights: List of layer probe weights.
            model_id: Model identifier.
        """
        # Derive R² threshold from quality distribution
        all_r2_values = [lw.validation_r2 for lw in weights]
        if all_r2_values:
            n = len(all_r2_values)
            mean_r2 = sum(all_r2_values) / n
            variance_r2 = sum((x - mean_r2) ** 2 for x in all_r2_values) / n
            std_r2 = math.sqrt(variance_r2)
            self._min_r2_threshold = max(0.0, mean_r2 - 2 * std_r2)

        for lw in weights:
            passes_r2 = self._min_r2_threshold is None or lw.validation_r2 >= self._min_r2_threshold
            if passes_r2:
                self._probe_weights[lw.layer] = lw
                self._cached_weight_arrays[lw.layer] = self._backend.array(lw.weights)

        self._trained_model_id = model_id
        self._is_ready = len(self._probe_weights) > 0

    def predict(self, hidden_states: "dict[int, Array]") -> PredictionResult:
        """
        Predict semantic entropy from hidden states.

        Args:
            hidden_states: Dict mapping layer index to hidden state tensor

        Returns:
            PredictionResult with entropy estimate and layer predictions
        """
        if not self._is_ready:
            raise WeightsNotLoadedError("SEP probe weights not loaded. Call load_weights() first.")

        start = time.time()
        b = self._backend

        predictions: dict[int, float] = {}

        for layer, probe in self._probe_weights.items():
            hidden_state = hidden_states.get(layer)
            if hidden_state is None:
                continue

            weight_array = self._cached_weight_arrays.get(layer)
            if weight_array is None:
                continue

            # Extract last token if sequence
            h: "Array"
            if hidden_state.ndim > 1:
                h = hidden_state[-1]
            else:
                h = hidden_state

            # Normalize - use division_epsilon for precision-aware threshold
            div_eps = division_epsilon(b, h)
            h_normalized = (h - probe.train_mean) / max(probe.train_std, div_eps)

            # Linear projection: ŜE_l = w_l^T h_l + b_l
            projection = b.sum(weight_array * h_normalized)
            prediction = projection + probe.bias
            b.eval(prediction)

            pred_value = float(b.to_scalar(prediction))
            predictions[layer] = pred_value

        if not predictions:
            raise SEPProbeError("No matching hidden states for configured probe layers.")

        # Final prediction is the intrinsic mean (Fréchet mean) across layers.
        # For scalar outputs, this is the arithmetic mean.
        final = sum(predictions.values()) / len(predictions)

        latency_ms = (time.time() - start) * 1000

        # Only trip circuit breaker if threshold is set from baseline
        should_trip = (
            self._circuit_breaker_threshold is not None
            and final >= self._circuit_breaker_threshold
        )

        return PredictionResult(
            predicted_entropy=final,
            layer_predictions=predictions,
            should_trip_circuit_breaker=should_trip,
            latency_ms=latency_ms,
        )

    def predict_single_layer(self, layer: int, hidden_state: "Array") -> float:
        """Predict from a single layer (for debugging)."""
        probe = self._probe_weights.get(layer)
        if probe is None:
            raise LayerNotFoundError(f"No probe weights for layer {layer}")

        weight_array = self._cached_weight_arrays.get(layer)
        if weight_array is None:
            raise LayerNotFoundError(f"No cached weights for layer {layer}")

        b = self._backend
        h = hidden_state[-1] if hidden_state.ndim > 1 else hidden_state
        div_eps = division_epsilon(b, h)
        h_normalized = (h - probe.train_mean) / max(probe.train_std, div_eps)

        projection = b.sum(weight_array * h_normalized)
        prediction = projection + probe.bias
        b.eval(prediction)

        return max(0.0, min(1.0, float(b.to_scalar(prediction))))

    def probe_info(self) -> list[tuple]:
        """Return info about loaded probes: [(layer, r2), ...]"""
        return sorted(
            [(layer, w.validation_r2) for layer, w in self._probe_weights.items()],
            key=lambda x: x[0],
        )

    def reset(self) -> None:
        """Clear loaded weights and reset state."""
        self._probe_weights.clear()
        self._cached_weight_arrays.clear()
        self._trained_model_id = None
        self._is_ready = False
        self._min_r2_threshold = None
        self._circuit_breaker_threshold = None
