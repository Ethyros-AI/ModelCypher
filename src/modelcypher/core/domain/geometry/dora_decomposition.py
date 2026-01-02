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
DoRA Decomposition: Weight-Decomposed Adaptation Analysis.

Ported 1:1 from the reference Swift implementation.

Based on Liu et al. 2024: "DoRA: Weight-Decomposed Low-Rank Adaptation" - ICML 2024

Key concepts:
- Weights = Magnitude × Direction
- Magnitude: ||W|| (overall scale)
- Direction: W / ||W|| (normalized orientation)
- Fine-tuning primarily changes direction

Usage:
    result = DoRADecomposition.analyze_adapter(base_weights, adapted_weights)
    print(result.overall_directional_drift)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class ChangeType(str, Enum):
    """Dominant type of weight change."""

    MAGNITUDE_DOMINATED = "magnitude_dominated"
    DIRECTION_DOMINATED = "direction_dominated"
    BALANCED = "balanced"
    MINIMAL = "minimal"




@dataclass
class DoRAConfig:
    """Configuration for DoRA decomposition.

    Use with_parameters() to create with explicit values.
    """

    compute_per_layer_metrics: bool = True

    @classmethod
    def with_parameters(
        cls,
        *,
        compute_per_layer_metrics: bool = True,
    ) -> "DoRAConfig":
        """Create configuration with explicit parameters.

        Args:
            compute_per_layer_metrics: Whether to compute per-layer metrics.

        Returns:
            Configuration with specified parameters.
        """
        return cls(
            compute_per_layer_metrics=compute_per_layer_metrics,
        )


@dataclass
class MagnitudeDirectionMetrics:
    """Magnitude/direction metrics for a single layer."""

    layer_name: str
    base_magnitude: float
    current_magnitude: float
    magnitude_ratio: float
    direction_cosine: float
    directional_drift: float
    absolute_magnitude_change: float
    relative_magnitude_change: float



@dataclass
class DecompositionResult:
    """Complete DoRA decomposition result."""

    per_layer_metrics: dict[str, MagnitudeDirectionMetrics]
    overall_magnitude_change: float
    overall_directional_drift: float
    dominant_change_type: ChangeType
    magnitude_to_direction_ratio: float
    layers_with_significant_direction_change: list[str]
    layers_with_significant_magnitude_change: list[str]
    computed_at: datetime = field(default_factory=datetime.now)



class DoRADecomposition:
    """
    Decomposes weights into magnitude and direction components.

    Analyzes how fine-tuning changes weight geometry:
    - Magnitude-dominated: Amplifying/attenuating existing features
    - Direction-dominated: Learning new feature combinations
    - Balanced: Combination of both
    """

    def __init__(self, config: DoRAConfig | None = None, backend: "Backend | None" = None):
        """Initialize with configuration or defaults.

        Args:
            config: Optional DoRA configuration (use with_parameters() to create).
            backend: Optional backend for array operations.
        """
        self.config = config or DoRAConfig()
        self._backend = backend or get_default_backend()

    def decompose(
        self,
        base_weight: "Array",
        current_weight: "Array",
        layer_name: str = "",
    ) -> MagnitudeDirectionMetrics | None:
        """
        Decompose a single weight pair into magnitude and direction.

        Args:
            base_weight: Original weight tensor
            current_weight: Adapted weight tensor
            layer_name: Name of this layer

        Returns:
            MagnitudeDirectionMetrics or None if computation fails
        """
        if base_weight.shape != current_weight.shape:
            return None

        b = self._backend
        # Compute magnitudes (L2 norm)
        base_flat = b.reshape(base_weight, (-1,))
        current_flat = b.reshape(current_weight, (-1,))

        base_sq = b.sqrt(b.sum(base_flat**2))
        current_sq = b.sqrt(b.sum(current_flat**2))
        b.eval(base_sq, current_sq)

        base_mag = float(b.to_numpy(base_sq).item())
        current_mag = float(b.to_numpy(current_sq).item())

        min_norm = division_epsilon(b, base_flat)
        if base_mag < min_norm:
            return None

        magnitude_ratio = current_mag / base_mag
        absolute_change = abs(current_mag - base_mag)
        relative_change = (current_mag - base_mag) / base_mag

        # Compute directional similarity (cosine)
        dot_arr = b.sum(base_flat * current_flat)
        b.eval(dot_arr)
        dot = float(b.to_numpy(dot_arr).item())
        eps = division_epsilon(b, base_flat)
        cosine = dot / (base_mag * current_mag + eps)
        cosine = max(-1.0, min(1.0, cosine))  # Clamp

        directional_drift = 1.0 - cosine

        return MagnitudeDirectionMetrics(
            layer_name=layer_name,
            base_magnitude=base_mag,
            current_magnitude=current_mag,
            magnitude_ratio=magnitude_ratio,
            direction_cosine=cosine,
            directional_drift=directional_drift,
            absolute_magnitude_change=absolute_change,
            relative_magnitude_change=relative_change,
        )

    def analyze_adapter(
        self,
        base_weights: "dict[str, Array]",
        current_weights: "dict[str, Array]",
    ) -> DecompositionResult:
        """
        Analyze an entire adapter's weights.

        Args:
            base_weights: Dict of layer name → base weight
            current_weights: Dict of layer name → current weight

        Returns:
            DecompositionResult with analysis
        """
        backend = get_default_backend()
        per_layer: dict[str, MagnitudeDirectionMetrics] = {}
        total_mag_change = 0.0
        total_dir_drift = 0.0
        total_weight = 0.0
        sig_direction: list[str] = []
        sig_magnitude: list[str] = []

        for name, base in base_weights.items():
            current = current_weights.get(name)
            if current is None:
                continue

            metrics = self.decompose(base, current, name)
            if metrics is None:
                continue

            per_layer[name] = metrics

            # Weight by parameter count
            weight = float(base.size)
            total_mag_change += abs(metrics.magnitude_ratio - 1.0) * weight
            total_dir_drift += metrics.directional_drift * weight
            total_weight += weight

            # Track any measurable changes (no arbitrary thresholds)
            # Caller interprets significance from the raw metrics
            if metrics.directional_drift > 0:
                sig_direction.append(name)
            if metrics.magnitude_ratio > 1.0 or metrics.magnitude_ratio < 1.0:
                sig_magnitude.append(name)

        eps = division_epsilon(backend, backend.array([total_weight]))
        if total_weight < eps:
            return self._empty_result()

        overall_mag = total_mag_change / total_weight
        overall_drift = total_dir_drift / total_weight

        # Compute ratio
        eps = division_epsilon(backend, backend.array([overall_mag, overall_drift]))
        if overall_drift > eps:
            ratio = overall_mag / overall_drift
        elif overall_mag > eps:
            ratio = float("inf")
        else:
            ratio = 0.0

        # Classify
        dominant = self._classify_change_type(overall_mag, overall_drift, ratio)

        return DecompositionResult(
            per_layer_metrics=per_layer,
            overall_magnitude_change=overall_mag,
            overall_directional_drift=overall_drift,
            dominant_change_type=dominant,
            magnitude_to_direction_ratio=ratio if math.isfinite(ratio) else 0.0,
            layers_with_significant_direction_change=sorted(sig_direction),
            layers_with_significant_magnitude_change=sorted(sig_magnitude),
        )

    def _classify_change_type(
        self,
        mag_change: float,
        dir_drift: float,
        ratio: float,
    ) -> ChangeType:
        """Classify the dominant change type."""
        backend = self._backend
        eps = division_epsilon(backend, backend.array([mag_change, dir_drift]))
        if mag_change < eps and dir_drift < eps:
            return ChangeType.MINIMAL

        if ratio > 1.0 + eps:
            return ChangeType.MAGNITUDE_DOMINATED

        if ratio < 1.0 - eps:
            return ChangeType.DIRECTION_DOMINATED

        return ChangeType.BALANCED

    def _empty_result(self) -> DecompositionResult:
        return DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )


# =============================================================================
# Metric Keys for Training Progress Emission
# =============================================================================


class DoRAMetricKey:
    """Metric keys for geometry tracking."""

    MAGNITUDE_CHANGE = "geometry/dora_magnitude_change"
    DIRECTIONAL_DRIFT = "geometry/dora_directional_drift"
    MAG_DIR_RATIO = "geometry/dora_mag_dir_ratio"
    DOMINANT_TYPE = "geometry/dora_dominant_type"


def to_metrics_dict(result: DecompositionResult) -> dict[str, float]:
    """Convert decomposition result to metrics dictionary."""
    return {
        DoRAMetricKey.MAGNITUDE_CHANGE: result.overall_magnitude_change,
        DoRAMetricKey.DIRECTIONAL_DRIFT: result.overall_directional_drift,
        DoRAMetricKey.MAG_DIR_RATIO: result.magnitude_to_direction_ratio,
    }
