from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


class Thresholds:
    strong_correlation: ClassVar[float] = 0.7
    moderate_correlation: ClassVar[float] = 0.4
    strong_weight: ClassVar[float] = 1.0
    moderate_weight: ClassVar[float] = 0.6
    weak_weight: ClassVar[float] = 0.2


@dataclass(frozen=True)
class DimensionCorrelation:
    source_dim: int
    target_dim: int
    correlation: float

    @property
    def is_strong_correlation(self) -> bool:
        return self.correlation > Thresholds.strong_correlation

    @property
    def is_moderate_correlation(self) -> bool:
        return Thresholds.moderate_correlation < self.correlation <= Thresholds.strong_correlation

    @property
    def is_weak_correlation(self) -> bool:
        return self.correlation <= Thresholds.moderate_correlation


@dataclass(frozen=True)
class LayerConfidence:
    layer: int
    strong_correlations: int
    moderate_correlations: int
    weak_correlations: int
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.strong_correlations + self.moderate_correlations + self.weak_correlations
        if total > 0:
            weighted = (
                float(self.strong_correlations) * Thresholds.strong_weight
                + float(self.moderate_correlations) * Thresholds.moderate_weight
                + float(self.weak_correlations) * Thresholds.weak_weight
            )
            value = weighted / float(total)
        else:
            value = 0.0
        object.__setattr__(self, "confidence", value)

    @property
    def total_correlations(self) -> int:
        return self.strong_correlations + self.moderate_correlations + self.weak_correlations


@dataclass(frozen=True)
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    overall_correlation: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]
