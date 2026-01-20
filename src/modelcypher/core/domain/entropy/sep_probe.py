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

Linear probes on transformer hidden states predict semantic entropy.
The caller specifies which layers to probe - layer selection comes
from upstream geometric analysis (CKA alignment, density profiling).

Architecture:
    For each layer l: ŜE_l = w_l^T h_l + b_l
    Final: ŜE = Fréchet mean of per-layer predictions

Research Basis:
    arXiv:2406.15927 - Semantic Entropy Probes
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
# Exceptions
# =============================================================================


class SEPProbeError(Exception):
    """SEP probe errors."""


class WeightsNotLoadedError(SEPProbeError):
    """Weights not loaded."""


class IncompatibleWeightsError(SEPProbeError):
    """Weight dimensions mismatch."""


class LayerNotFoundError(SEPProbeError):
    """Probe weights for layer not found."""


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
    latency_ms: float


# =============================================================================
# SEP Probe
# =============================================================================


class SEPProbe:
    """
    Semantic Entropy Probe using linear projection on hidden states.

    Layer selection is NOT this module's responsibility. The caller
    determines which layers to probe based on geometric analysis.
    This class just runs predictions on whatever layers have weights.

    Usage:
        probe = SEPProbe(hidden_dim=4096)
        await probe.load_weights(path)

        # During inference with hidden states:
        result = probe.predict(hidden_states)
    """

    def __init__(
        self,
        hidden_dim: int,
        backend: "Backend | None" = None,
    ) -> None:
        """Create SEP probe.

        Args:
            hidden_dim: Hidden dimension of the model.
            backend: Optional compute backend.
        """
        self._hidden_dim = hidden_dim
        self._backend = backend or get_default_backend()

        # Weights storage
        self._probe_weights: dict[int, LayerProbeWeights] = {}
        self._cached_weight_arrays: "dict[int, Array]" = {}
        self._is_ready: bool = False
        self._trained_model_id: str | None = None

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension."""
        return self._hidden_dim

    @property
    def is_ready(self) -> bool:
        """Whether probe weights are loaded."""
        return self._is_ready

    @property
    def trained_model_id(self) -> str | None:
        """Model ID the weights were trained on."""
        return self._trained_model_id

    @property
    def available_layers(self) -> set[int]:
        """Layers that have loaded weights."""
        return set(self._probe_weights.keys())

    async def load_weights(self, path: Path) -> None:
        """Load trained probe weights from file.

        Args:
            path: Path to weights file.
        """
        data = json.loads(path.read_text())

        loaded_hidden_dim = data.get("hidden_dim", 4096)
        if loaded_hidden_dim != self._hidden_dim:
            raise IncompatibleWeightsError(
                f"Incompatible: expected hidden_dim={self._hidden_dim}, "
                f"found {loaded_hidden_dim}"
            )

        layer_weights_data = data.get("layer_weights", [])
        for lw in layer_weights_data:
            layer = lw["layer"]
            weights = LayerProbeWeights(
                layer=layer,
                weights=lw["weights"],
                bias=lw.get("bias", 0.0),
                validation_r2=lw.get("validation_r2", 0.0),
                train_mean=lw.get("train_mean", 0.0),
                train_std=lw.get("train_std", 1.0),
            )
            self._probe_weights[layer] = weights
            self._cached_weight_arrays[layer] = self._backend.array(weights.weights)

        self._trained_model_id = data.get("model_id", "unknown")
        self._is_ready = len(self._probe_weights) > 0

    def register_weights(self, weights: list[LayerProbeWeights], model_id: str) -> None:
        """Register weights directly (for testing or in-memory).

        Args:
            weights: List of layer probe weights.
            model_id: Model identifier.
        """
        for lw in weights:
            self._probe_weights[lw.layer] = lw
            self._cached_weight_arrays[lw.layer] = self._backend.array(lw.weights)

        self._trained_model_id = model_id
        self._is_ready = len(self._probe_weights) > 0

    def predict(self, hidden_states: "dict[int, Array]") -> PredictionResult:
        """
        Predict semantic entropy from hidden states.

        Args:
            hidden_states: Dict mapping layer index to hidden state tensor.

        Returns:
            PredictionResult with entropy estimate and layer predictions.
        """
        if not self._is_ready:
            raise WeightsNotLoadedError(
                "SEP probe weights not loaded. Call load_weights() first."
            )

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
            div_eps = division_epsilon(b, h)
            h_normalized = (h - probe.train_mean) / max(probe.train_std, div_eps)

            # Linear projection: ŜE_l = w_l^T h_l + b_l
            projection = b.sum(weight_array * h_normalized)
            prediction = projection + probe.bias
            b.eval(prediction)

            pred_value = float(b.to_scalar(prediction))
            predictions[layer] = pred_value

        if not predictions:
            raise SEPProbeError(
                "No matching hidden states for loaded probe layers."
            )

        # Final prediction: Fréchet mean (arithmetic mean for scalars)
        final = sum(predictions.values()) / len(predictions)

        latency_ms = (time.time() - start) * 1000

        return PredictionResult(
            predicted_entropy=final,
            layer_predictions=predictions,
            latency_ms=latency_ms,
        )

    def predict_single_layer(self, layer: int, hidden_state: "Array") -> float:
        """Predict from a single layer."""
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

        return float(b.to_scalar(prediction))

    def probe_info(self) -> list[tuple[int, float]]:
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
