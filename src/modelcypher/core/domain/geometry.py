from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class VectorMath:
    @staticmethod
    def dot(lhs: list[float], rhs: list[float]) -> Optional[float]:
        if len(lhs) != len(rhs) or not lhs:
            return None
        return sum(float(a) * float(b) for a, b in zip(lhs, rhs))

    @staticmethod
    def l2_norm(vector: list[float]) -> Optional[float]:
        if not vector:
            return None
        sum_sq = sum(float(v) * float(v) for v in vector)
        if sum_sq <= 0:
            return None
        return sum_sq ** 0.5

    @staticmethod
    def l2_normalized(vector: list[float]) -> list[float]:
        norm = VectorMath.l2_norm(vector)
        if not norm or norm <= 0:
            return vector
        inv_norm = 1.0 / norm
        return [float(v) * inv_norm for v in vector]

    @staticmethod
    def cosine_similarity(lhs: list[float], rhs: list[float]) -> Optional[float]:
        if len(lhs) != len(rhs) or not lhs:
            return None
        dot = 0.0
        lhs_norm = 0.0
        rhs_norm = 0.0
        for a, b in zip(lhs, rhs):
            dot += float(a) * float(b)
            lhs_norm += float(a) * float(a)
            rhs_norm += float(b) * float(b)
        if lhs_norm <= 0 or rhs_norm <= 0:
            return None
        return dot / ((lhs_norm ** 0.5) * (rhs_norm ** 0.5))

    @staticmethod
    def sparse_l2_norm(vector: dict[str, float]) -> Optional[float]:
        if not vector:
            return None
        sum_sq = sum(float(v) * float(v) for v in vector.values())
        if sum_sq <= 0:
            return None
        return sum_sq ** 0.5

    @staticmethod
    def sparse_cosine_similarity(lhs: dict[str, float], rhs: dict[str, float]) -> Optional[float]:
        if not lhs or not rhs:
            return None
        lhs_norm = VectorMath.sparse_l2_norm(lhs)
        rhs_norm = VectorMath.sparse_l2_norm(rhs)
        if not lhs_norm or not rhs_norm:
            return None
        smaller, larger = (lhs, rhs) if len(lhs) <= len(rhs) else (rhs, lhs)
        dot = 0.0
        for token, weight in smaller.items():
            other = larger.get(token)
            if other is not None:
                dot += float(weight) * float(other)
        return dot / (lhs_norm * rhs_norm)


class SetMath:
    @staticmethod
    def intersection_count(lhs: set, rhs: set) -> int:
        if not lhs or not rhs:
            return 0
        smaller, larger = (lhs, rhs) if len(lhs) <= len(rhs) else (rhs, lhs)
        return sum(1 for element in smaller if element in larger)

    @staticmethod
    def jaccard_similarity(lhs: set, rhs: set) -> float:
        if not lhs or not rhs:
            return 0.0
        intersection = SetMath.intersection_count(lhs, rhs)
        union = len(lhs) + len(rhs) - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)


class ChangeInterpretation(str, Enum):
    amplification = "amplification"
    attenuation = "attenuation"
    rotation = "rotation"
    mixed = "mixed"


class ChangeType(str, Enum):
    magnitude_dominated = "magnitudeDominated"
    direction_dominated = "directionDominated"
    balanced = "balanced"
    minimal = "minimal"


@dataclass(frozen=True)
class DoRAConfiguration:
    magnitude_dominance_threshold: float = 2.0
    direction_dominance_threshold: float = 2.0
    compute_per_layer_metrics: bool = True
    minimum_norm: float = 1e-8


@dataclass(frozen=True)
class MagnitudeDirectionMetrics:
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
        mag_change = abs(self.magnitude_ratio - 1.0)
        if mag_change > self.directional_drift * 2:
            return ChangeInterpretation.amplification if self.magnitude_ratio > 1 else ChangeInterpretation.attenuation
        if self.directional_drift > mag_change * 2:
            return ChangeInterpretation.rotation
        return ChangeInterpretation.mixed


@dataclass(frozen=True)
class DecompositionResult:
    per_layer_metrics: dict[str, MagnitudeDirectionMetrics]
    overall_magnitude_change: float
    overall_directional_drift: float
    dominant_change_type: ChangeType
    magnitude_to_direction_ratio: float
    layers_with_significant_direction_change: list[str]
    layers_with_significant_magnitude_change: list[str]
    computed_at: datetime


class DoRADecomposition:
    @staticmethod
    def decompose(
        base_weight: list[float],
        current_weight: list[float],
        layer_name: str,
        configuration: DoRAConfiguration = DoRAConfiguration(),
    ) -> Optional[MagnitudeDirectionMetrics]:
        if len(base_weight) != len(current_weight) or not base_weight:
            return None
        base_norm = VectorMath.l2_norm(base_weight)
        current_norm = VectorMath.l2_norm(current_weight)
        if base_norm is None or current_norm is None:
            return None
        base_mag = float(base_norm)
        current_mag = float(current_norm)
        if base_mag <= configuration.minimum_norm:
            return None
        magnitude_ratio = current_mag / base_mag
        absolute_change = abs(current_mag - base_mag)
        relative_change = (current_mag - base_mag) / base_mag
        cosine = VectorMath.cosine_similarity(base_weight, current_weight)
        if cosine is None:
            return None
        direction_cosine = float(cosine)
        directional_drift = 1.0 - direction_cosine
        return MagnitudeDirectionMetrics(
            layer_name=layer_name,
            base_magnitude=base_mag,
            current_magnitude=current_mag,
            magnitude_ratio=magnitude_ratio,
            direction_cosine=direction_cosine,
            directional_drift=directional_drift,
            absolute_magnitude_change=absolute_change,
            relative_magnitude_change=relative_change,
        )

    @staticmethod
    def analyze_adapter(
        base_weights: dict[str, list[float]],
        current_weights: dict[str, list[float]],
        configuration: DoRAConfiguration = DoRAConfiguration(),
    ) -> DecompositionResult:
        per_layer: dict[str, MagnitudeDirectionMetrics] = {}
        total_magnitude_change = 0.0
        total_directional_drift = 0.0
        total_weight = 0.0
        significant_direction_layers: list[str] = []
        significant_magnitude_layers: list[str] = []

        for name, base_weight in base_weights.items():
            current_weight = current_weights.get(name)
            if current_weight is None:
                continue
            metrics = DoRADecomposition.decompose(
                base_weight=base_weight,
                current_weight=current_weight,
                layer_name=name,
                configuration=configuration,
            )
            if metrics is None:
                continue
            per_layer[name] = metrics
            weight = float(len(base_weight))
            total_magnitude_change += abs(metrics.magnitude_ratio - 1.0) * weight
            total_directional_drift += metrics.directional_drift * weight
            total_weight += weight
            if metrics.directional_drift > 0.1:
                significant_direction_layers.append(name)
            if metrics.magnitude_ratio > 1.2 or metrics.magnitude_ratio < 0.8:
                significant_magnitude_layers.append(name)

        if total_weight <= 0:
            return DecompositionResult(
                per_layer_metrics={},
                overall_magnitude_change=0.0,
                overall_directional_drift=0.0,
                dominant_change_type=ChangeType.minimal,
                magnitude_to_direction_ratio=0.0,
                layers_with_significant_direction_change=[],
                layers_with_significant_magnitude_change=[],
                computed_at=datetime.utcnow(),
            )

        overall_magnitude_change = total_magnitude_change / total_weight
        overall_directional_drift = total_directional_drift / total_weight
        if overall_directional_drift > configuration.minimum_norm:
            ratio = overall_magnitude_change / overall_directional_drift
        else:
            ratio = float("inf") if overall_magnitude_change > configuration.minimum_norm else 0.0

        dominant_type = DoRADecomposition.classify_change_type(
            magnitude_change=overall_magnitude_change,
            directional_drift=overall_directional_drift,
            ratio=ratio,
            configuration=configuration,
        )

        return DecompositionResult(
            per_layer_metrics=per_layer,
            overall_magnitude_change=overall_magnitude_change,
            overall_directional_drift=overall_directional_drift,
            dominant_change_type=dominant_type,
            magnitude_to_direction_ratio=ratio,
            layers_with_significant_direction_change=sorted(significant_direction_layers),
            layers_with_significant_magnitude_change=sorted(significant_magnitude_layers),
            computed_at=datetime.utcnow(),
        )

    @staticmethod
    def classify_change_type(
        magnitude_change: float,
        directional_drift: float,
        ratio: float,
        configuration: DoRAConfiguration,
    ) -> ChangeType:
        if magnitude_change < 0.01 and directional_drift < 0.01:
            return ChangeType.minimal
        if ratio > configuration.magnitude_dominance_threshold:
            return ChangeType.magnitude_dominated
        if ratio < 1.0 / configuration.direction_dominance_threshold:
            return ChangeType.direction_dominated
        return ChangeType.balanced
