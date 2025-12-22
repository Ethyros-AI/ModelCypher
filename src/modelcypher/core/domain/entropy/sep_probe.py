"""
Semantic Entropy Probe (SEP) for Fast Entropy Prediction.

Ported 1:1 from the reference Swift implementation.

Linear probes on transformer hidden states predict semantic entropy
with R² ~ 0.8 while being 1000x faster than full computation (0.3ms vs 15s).

Architecture:
    For each layer l: ŜE_l = w_l^T h_l + b_l
    Final: ŜE = Σ (R²_l * ŜE_l) / Σ R²_l (R²-weighted ensemble)

Research Basis:
    arXiv:2406.15927 - Semantic Entropy Probes
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

import mlx.core as mx


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
    """Configuration for SEP probe."""
    layer_count: int = 32
    layer_fractions: List[float] = field(default_factory=lambda: [0.75, 0.78, 0.81, 0.84, 0.875])
    hidden_dim: int = 4096
    use_ensemble: bool = True
    min_r2_threshold: float = 0.5
    circuit_breaker_threshold: float = 0.7

    @property
    def target_layers(self) -> List[int]:
        """Target layer indices based on fractions."""
        return [int(self.layer_count * f) for f in self.layer_fractions]

    @classmethod
    def default(cls) -> "SEPProbeConfig":
        return cls()


@dataclass
class LayerProbeWeights:
    """Trained weights for a single layer probe."""
    layer: int
    weights: List[float]
    bias: float
    validation_r2: float
    train_mean: float = 0.0
    train_std: float = 1.0


@dataclass
class ProbeWeightsBundle:
    """Container for all probe weights."""
    model_id: str
    config: SEPProbeConfig
    layer_weights: List[LayerProbeWeights]
    training_samples: int
    trained_at: datetime


@dataclass
class PredictionResult:
    """Result from SEP probe prediction."""
    predicted_entropy: float
    layer_predictions: Dict[int, float]
    ensemble_weights: Dict[int, float]
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

    def __init__(self, config: Optional[SEPProbeConfig] = None):
        self.config = config or SEPProbeConfig.default()

        # Weights storage
        self._probe_weights: Dict[int, LayerProbeWeights] = {}
        self._cached_weight_arrays: Dict[int, mx.array] = {}
        self._is_ready: bool = False
        self._trained_model_id: Optional[str] = None

    @classmethod
    def for_model(cls, layer_count: int, hidden_dim: int) -> "SEPProbe":
        """Create probe configured for a specific model."""
        return cls(SEPProbeConfig(layer_count=layer_count, hidden_dim=hidden_dim))

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def trained_model_id(self) -> Optional[str]:
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

            if layer in self.config.target_layers and r2 >= self.config.min_r2_threshold:
                weights = LayerProbeWeights(
                    layer=layer,
                    weights=lw["weights"],
                    bias=lw.get("bias", 0.0),
                    validation_r2=r2,
                    train_mean=lw.get("train_mean", 0.0),
                    train_std=lw.get("train_std", 1.0),
                )
                self._probe_weights[layer] = weights
                self._cached_weight_arrays[layer] = mx.array(weights.weights)

        self._trained_model_id = data.get("model_id", "unknown")
        self._is_ready = len(self._probe_weights) > 0

    def register_weights(self, weights: List[LayerProbeWeights], model_id: str):
        """Register weights directly (for testing or in-memory)."""
        for lw in weights:
            if lw.validation_r2 >= self.config.min_r2_threshold:
                self._probe_weights[lw.layer] = lw
                self._cached_weight_arrays[lw.layer] = mx.array(lw.weights)

        self._trained_model_id = model_id
        self._is_ready = len(self._probe_weights) > 0

    def predict(self, hidden_states: Dict[int, mx.array]) -> PredictionResult:
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

        predictions: Dict[int, float] = {}
        ensemble_weights: Dict[int, float] = {}
        weighted_sum: float = 0.0
        total_weight: float = 0.0

        for layer, probe in self._probe_weights.items():
            hidden_state = hidden_states.get(layer)
            if hidden_state is None:
                continue

            weight_array = self._cached_weight_arrays.get(layer)
            if weight_array is None:
                continue

            # Extract last token if sequence
            h: mx.array
            if hidden_state.ndim > 1:
                h = hidden_state[-1]
            else:
                h = hidden_state

            # Normalize
            h_normalized = (h - probe.train_mean) / max(probe.train_std, 1e-6)

            # Linear projection: ŜE_l = w_l^T h_l + b_l
            projection = mx.sum(weight_array * h_normalized)
            prediction = projection + probe.bias
            mx.eval(prediction)

            # Clamp to [0, 1]
            pred_value = max(0.0, min(1.0, float(prediction.item())))
            predictions[layer] = pred_value

            # R²-weighted ensemble
            weight = probe.validation_r2
            ensemble_weights[layer] = weight
            weighted_sum += pred_value * weight
            total_weight += weight

        # Final ensemble prediction
        final = weighted_sum / total_weight if total_weight > 0 else 0.0

        latency_ms = (time.time() - start) * 1000

        return PredictionResult(
            predicted_entropy=final,
            layer_predictions=predictions,
            ensemble_weights=ensemble_weights,
            should_trip_circuit_breaker=final >= self.config.circuit_breaker_threshold,
            latency_ms=latency_ms,
        )

    def predict_single_layer(self, layer: int, hidden_state: mx.array) -> float:
        """Predict from a single layer (for debugging)."""
        probe = self._probe_weights.get(layer)
        if probe is None:
            raise LayerNotFoundError(f"No probe weights for layer {layer}")

        weight_array = self._cached_weight_arrays.get(layer)
        if weight_array is None:
            raise LayerNotFoundError(f"No cached weights for layer {layer}")

        h = hidden_state[-1] if hidden_state.ndim > 1 else hidden_state
        h_normalized = (h - probe.train_mean) / max(probe.train_std, 1e-6)

        projection = mx.sum(weight_array * h_normalized)
        prediction = projection + probe.bias
        mx.eval(prediction)

        return max(0.0, min(1.0, float(prediction.item())))

    def probe_info(self) -> List[tuple]:
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
