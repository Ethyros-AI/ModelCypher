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
    print(result.interpretation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import mlx.core as mx


class ChangeType(str, Enum):
    """Dominant type of weight change."""
    MAGNITUDE_DOMINATED = "magnitude_dominated"
    DIRECTION_DOMINATED = "direction_dominated"
    BALANCED = "balanced"
    MINIMAL = "minimal"


class ChangeInterpretation(str, Enum):
    """Interpretation of layer-level change."""
    AMPLIFICATION = "amplification"
    ATTENUATION = "attenuation"
    ROTATION = "rotation"
    MIXED = "mixed"


@dataclass
class DoRAConfig:
    """Configuration for DoRA decomposition."""
    magnitude_dominance_threshold: float = 2.0
    direction_dominance_threshold: float = 2.0
    compute_per_layer_metrics: bool = True
    minimum_norm: float = 1e-8

    @classmethod
    def default(cls) -> "DoRAConfig":
        return cls()


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

    @property
    def interpretation(self) -> ChangeInterpretation:
        """Interpret this layer's change."""
        mag_change = abs(self.magnitude_ratio - 1.0)
        if mag_change > self.directional_drift * 2:
            return ChangeInterpretation.AMPLIFICATION if self.magnitude_ratio > 1 else ChangeInterpretation.ATTENUATION
        elif self.directional_drift > mag_change * 2:
            return ChangeInterpretation.ROTATION
        else:
            return ChangeInterpretation.MIXED


@dataclass
class DecompositionResult:
    """Complete DoRA decomposition result."""
    per_layer_metrics: Dict[str, MagnitudeDirectionMetrics]
    overall_magnitude_change: float
    overall_directional_drift: float
    dominant_change_type: ChangeType
    magnitude_to_direction_ratio: float
    layers_with_significant_direction_change: List[str]
    layers_with_significant_magnitude_change: List[str]
    computed_at: datetime = field(default_factory=datetime.now)

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.dominant_change_type == ChangeType.MAGNITUDE_DOMINATED:
            sign = "+" if self.overall_magnitude_change > 0 else ""
            return f"Adapter primarily amplifies/attenuates features (magnitude {sign}{int(self.overall_magnitude_change * 100)}%)"
        elif self.dominant_change_type == ChangeType.DIRECTION_DOMINATED:
            return f"Adapter primarily rotates feature space (drift: {self.overall_directional_drift:.2f})"
        elif self.dominant_change_type == ChangeType.BALANCED:
            return "Adapter combines scaling and rotation (balanced change)"
        else:
            return "Adapter has minimal impact on weight geometry"

    @property
    def suggests_good_quality(self) -> bool:
        """Whether this suggests good adapter quality."""
        if self.dominant_change_type in (ChangeType.BALANCED, ChangeType.DIRECTION_DOMINATED):
            return self.overall_directional_drift < 0.5
        elif self.dominant_change_type == ChangeType.MAGNITUDE_DOMINATED:
            return self.overall_magnitude_change < 0.3
        return True


class DoRADecomposition:
    """
    Decomposes weights into magnitude and direction components.

    Analyzes how fine-tuning changes weight geometry:
    - Magnitude-dominated: Amplifying/attenuating existing features
    - Direction-dominated: Learning new feature combinations
    - Balanced: Combination of both
    """

    def __init__(self, config: Optional[DoRAConfig] = None):
        self.config = config or DoRAConfig.default()

    def decompose(
        self,
        base_weight: mx.array,
        current_weight: mx.array,
        layer_name: str = "",
    ) -> Optional[MagnitudeDirectionMetrics]:
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

        # Compute magnitudes (L2 norm)
        base_flat = base_weight.flatten()
        current_flat = current_weight.flatten()

        base_mag = float(mx.sqrt(mx.sum(base_flat ** 2)).item())
        current_mag = float(mx.sqrt(mx.sum(current_flat ** 2)).item())

        if base_mag < self.config.minimum_norm:
            return None

        magnitude_ratio = current_mag / base_mag
        absolute_change = abs(current_mag - base_mag)
        relative_change = (current_mag - base_mag) / base_mag

        # Compute directional similarity (cosine)
        dot = float(mx.sum(base_flat * current_flat).item())
        cosine = dot / (base_mag * current_mag + 1e-10)
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
        base_weights: Dict[str, mx.array],
        current_weights: Dict[str, mx.array],
    ) -> DecompositionResult:
        """
        Analyze an entire adapter's weights.

        Args:
            base_weights: Dict of layer name → base weight
            current_weights: Dict of layer name → current weight

        Returns:
            DecompositionResult with analysis
        """
        per_layer: Dict[str, MagnitudeDirectionMetrics] = {}
        total_mag_change = 0.0
        total_dir_drift = 0.0
        total_weight = 0.0
        sig_direction: List[str] = []
        sig_magnitude: List[str] = []

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

            # Track significant changes
            if metrics.directional_drift > 0.1:
                sig_direction.append(name)
            if metrics.magnitude_ratio > 1.2 or metrics.magnitude_ratio < 0.8:
                sig_magnitude.append(name)

        if total_weight < 1e-10:
            return self._empty_result()

        overall_mag = total_mag_change / total_weight
        overall_drift = total_dir_drift / total_weight

        # Compute ratio
        if overall_drift > self.config.minimum_norm:
            ratio = overall_mag / overall_drift
        elif overall_mag > self.config.minimum_norm:
            ratio = float('inf')
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
        if mag_change < 0.01 and dir_drift < 0.01:
            return ChangeType.MINIMAL

        if ratio > self.config.magnitude_dominance_threshold:
            return ChangeType.MAGNITUDE_DOMINATED

        if ratio < 1.0 / self.config.direction_dominance_threshold:
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


def to_metrics_dict(result: DecompositionResult) -> Dict[str, float]:
    """Convert decomposition result to metrics dictionary."""
    return {
        DoRAMetricKey.MAGNITUDE_CHANGE: result.overall_magnitude_change,
        DoRAMetricKey.DIRECTIONAL_DRIFT: result.overall_directional_drift,
        DoRAMetricKey.MAG_DIR_RATIO: result.magnitude_to_direction_ratio,
    }
