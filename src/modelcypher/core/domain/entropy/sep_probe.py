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
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


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


@dataclass
class SEPProbeConfig:
    """Configuration for SEP probe.

    Thresholds must be derived from baseline measurements. No arbitrary defaults.
    """

    layer_count: int = 32
    layer_fractions: list[float] = field(default_factory=lambda: [0.75, 0.78, 0.81, 0.84, 0.875])
    hidden_dim: int = 4096
    # Thresholds - MUST be derived from baseline measurements, no arbitrary defaults
    min_r2_threshold: float | None = None
    """Minimum R² for probe to be considered valid. Derive from training validation."""

    circuit_breaker_threshold: float | None = None
    """Risk threshold for tripping circuit breaker. Derive from baseline risk distribution."""

    @property
    def target_layers(self) -> list[int]:
        """Target layer indices based on fractions."""
        return [int(self.layer_count * f) for f in self.layer_fractions]

    @classmethod
    def default(cls) -> "SEPProbeConfig":
        """Create config with architecture defaults. Thresholds must be set separately."""
        return cls()


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
    config: SEPProbeConfig
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


class SEPProbe:
    """
    Semantic Entropy Probe using linear projection on hidden states.

    Fast (0.3ms) prediction of semantic entropy from layer activations.

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
        config: SEPProbeConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or SEPProbeConfig.default()
        self._backend = backend or get_default_backend()

        # Weights storage
        self._probe_weights: dict[int, LayerProbeWeights] = {}
        self._cached_weight_arrays: "dict[int, Array]" = {}
        self._is_ready: bool = False
        self._trained_model_id: str | None = None

    @classmethod
    def for_model(cls, layer_count: int, hidden_dim: int) -> "SEPProbe":
        """Create probe configured for a specific model."""
        return cls(SEPProbeConfig(layer_count=layer_count, hidden_dim=hidden_dim))

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def trained_model_id(self) -> str | None:
        return self._trained_model_id

    async def load_weights(self, path: Path):
        """Load trained probe weights from file."""
        data = json.loads(path.read_text())

        # Parse bundle
        config = SEPProbeConfig(
            layer_count=data.get("layer_count", 32),
            hidden_dim=data.get("hidden_dim", 4096),
        )

        if config.hidden_dim != self.config.hidden_dim:
            raise IncompatibleWeightsError(
                f"Incompatible: expected hiddenDim={self.config.hidden_dim}, found {config.hidden_dim}"
            )

        # Load weights
        for lw in data.get("layer_weights", []):
            layer = lw["layer"]
            r2 = lw.get("validation_r2", 0.0)

            # Filter by R² threshold if configured; if None, accept all
            passes_r2 = self.config.min_r2_threshold is None or r2 >= self.config.min_r2_threshold
            if layer in self.config.target_layers and passes_r2:
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
        """Register weights directly (for testing or in-memory)."""
        for lw in weights:
            # Filter by R² threshold if configured; if None, accept all
            passes_r2 = self.config.min_r2_threshold is None or lw.validation_r2 >= self.config.min_r2_threshold
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

            # Normalize
            h_normalized = (h - probe.train_mean) / max(probe.train_std, 1e-6)

            # Linear projection: ŜE_l = w_l^T h_l + b_l
            projection = b.sum(weight_array * h_normalized)
            prediction = projection + probe.bias
            b.eval(prediction)

            pred_np = b.to_numpy(prediction)
            pred_value = float(pred_np.item())
            predictions[layer] = pred_value
        if not predictions:
            raise SEPProbeError("No matching hidden states for configured probe layers.")

        # Final prediction is the intrinsic mean (Fréchet mean) across layers.
        # For scalar outputs, this is the arithmetic mean.
        final = sum(predictions.values()) / len(predictions)

        latency_ms = (time.time() - start) * 1000

        # Only trip circuit breaker if threshold is configured
        should_trip = (
            self.config.circuit_breaker_threshold is not None
            and final >= self.config.circuit_breaker_threshold
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
        h_normalized = (h - probe.train_mean) / max(probe.train_std, 1e-6)

        projection = b.sum(weight_array * h_normalized)
        prediction = projection + probe.bias
        b.eval(prediction)

        pred_np = b.to_numpy(prediction)
        return max(0.0, min(1.0, float(pred_np.item())))

    def probe_info(self) -> list[tuple]:
        """Return info about loaded probes: [(layer, r2), ...]"""
        return sorted(
            [(layer, w.validation_r2) for layer, w in self._probe_weights.items()],
            key=lambda x: x[0],
        )

    def reset(self):
        """Clear loaded weights and reset state."""
        self._probe_weights.clear()
        self._cached_weight_arrays.clear()
        self._trained_model_id = None
        self._is_ready = False
