from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
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


class ProbeSpace(str, Enum):
    prelogits_hidden = "prelogits-hidden"
    output_logits = "output-logits"


output_layer_marker = (2**63 - 1) - 1


@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

    def __lt__(self, other: "ActivatedDimension") -> bool:
        return self.activation > other.activation


@dataclass(frozen=True)
class ActivationFingerprint:
    prime_id: str
    prime_text: str
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass(frozen=True)
class SparseActivationVector:
    indices: list[int]
    values: list[float]
    length: int

    def dot(self, other: "SparseActivationVector") -> float:
        count_a = min(len(self.indices), len(self.values))
        count_b = min(len(other.indices), len(other.values))
        if count_a <= 0 or count_b <= 0:
            return 0.0
        i = 0
        j = 0
        total = 0.0
        while i < count_a and j < count_b:
            idx_a = self.indices[i]
            idx_b = other.indices[j]
            if idx_a == idx_b:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif idx_a < idx_b:
                i += 1
            else:
                j += 1
        return total

    def dot_dense(self, dense: list[float]) -> float:
        count = min(len(self.indices), len(self.values))
        if count <= 0:
            return 0.0
        total = 0.0
        for i in range(count):
            idx = self.indices[i]
            if 0 <= idx < len(dense):
                total += self.values[i] * dense[idx]
        return total


@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    probe_space: ProbeSpace
    probe_capture_key: Optional[str]
    hidden_dim: int
    layer_count: int
    fingerprints: list[ActivationFingerprint]
    activation_vectors: Optional[dict[str, list[float]]] = None
    activation_sparse_vectors: Optional[dict[str, SparseActivationVector]] = None


def intersection_map_from_dict(payload: dict[str, Any]) -> IntersectionMap:
    def _get(key: str, fallback: str | None = None) -> Any:
        if key in payload:
            return payload[key]
        if fallback and fallback in payload:
            return payload[fallback]
        return None

    raw_correlations = _get("dimensionCorrelations", "dimension_correlations") or {}
    dimension_correlations: dict[int, list[DimensionCorrelation]] = {}
    for layer_key, entries in raw_correlations.items():
        try:
            layer = int(layer_key)
        except (TypeError, ValueError):
            continue
        parsed: list[DimensionCorrelation] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            source_dim = entry.get("sourceDim", entry.get("source_dim"))
            target_dim = entry.get("targetDim", entry.get("target_dim"))
            correlation = entry.get("correlation")
            if source_dim is None or target_dim is None or correlation is None:
                continue
            parsed.append(
                DimensionCorrelation(
                    source_dim=int(source_dim),
                    target_dim=int(target_dim),
                    correlation=float(correlation),
                )
            )
        if parsed:
            dimension_correlations[layer] = parsed

    raw_layer_confidences = _get("layerConfidences", "layer_confidences") or []
    layer_confidences: list[LayerConfidence] = []
    for entry in raw_layer_confidences:
        if not isinstance(entry, dict):
            continue
        layer = entry.get("layer")
        strong = entry.get("strongCorrelations", entry.get("strong_correlations"))
        moderate = entry.get("moderateCorrelations", entry.get("moderate_correlations"))
        weak = entry.get("weakCorrelations", entry.get("weak_correlations"))
        if layer is None or strong is None or moderate is None or weak is None:
            continue
        layer_confidences.append(
            LayerConfidence(
                layer=int(layer),
                strong_correlations=int(strong),
                moderate_correlations=int(moderate),
                weak_correlations=int(weak),
            )
        )

    return IntersectionMap(
        source_model=str(_get("sourceModel", "source_model") or ""),
        target_model=str(_get("targetModel", "target_model") or ""),
        dimension_correlations=dimension_correlations,
        overall_correlation=float(_get("overallCorrelation", "overall_correlation") or 0.0),
        aligned_dimension_count=int(_get("alignedDimensionCount", "aligned_dimension_count") or 0),
        total_source_dims=int(_get("totalSourceDims", "total_source_dims") or 0),
        total_target_dims=int(_get("totalTargetDims", "total_target_dims") or 0),
        layer_confidences=layer_confidences,
    )
