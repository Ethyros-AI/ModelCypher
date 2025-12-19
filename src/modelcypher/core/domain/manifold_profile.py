from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional
from uuid import UUID, uuid4


@dataclass
class ManifoldProfile:
    id: UUID
    model_id: str
    model_name: str
    regions: list["ManifoldRegion"] = field(default_factory=list)
    recent_points: list["ManifoldPoint"] = field(default_factory=list)
    total_point_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    @dataclass(frozen=True)
    class Configuration:
        max_recent_points: int = 100
        min_points_per_region: int = 5
        clustering_epsilon: float = 0.3
        compute_intrinsic_dimension: bool = True

    @dataclass(frozen=True)
    class Statistics:
        total_points: int
        region_count: int
        safe_region_count: int
        sparse_region_count: int
        boundary_region_count: int
        mean_intrinsic_dimension: Optional[float]
        recent_point_count: int

    def compute_statistics(self) -> "ManifoldProfile.Statistics":
        safe_count = sum(1 for region in self.regions if region.region_type == ManifoldRegion.RegionType.safe)
        sparse_count = sum(1 for region in self.regions if region.region_type == ManifoldRegion.RegionType.sparse)
        boundary_count = sum(1 for region in self.regions if region.region_type == ManifoldRegion.RegionType.boundary)
        dimensions = [region.intrinsic_dimension for region in self.regions if region.intrinsic_dimension is not None]
        mean_dim = sum(dimensions) / float(len(dimensions)) if dimensions else None

        return ManifoldProfile.Statistics(
            total_points=self.total_point_count,
            region_count=len(self.regions),
            safe_region_count=safe_count,
            sparse_region_count=sparse_count,
            boundary_region_count=boundary_count,
            mean_intrinsic_dimension=mean_dim,
            recent_point_count=len(self.recent_points),
        )


@dataclass(frozen=True)
class ManifoldPoint:
    id: UUID
    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    gate_count: int
    mean_gate_confidence: float
    dominant_gate_category: float
    entropy_path_correlation: float
    assessment_strength: float
    prompt_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    intervention_level: Optional[int] = None

    feature_dimension: ClassVar[int] = 8
    feature_names: ClassVar[list[str]] = [
        "meanEntropy",
        "entropyVariance",
        "firstTokenEntropy",
        "gateCount",
        "meanGateConfidence",
        "dominantGateCategory",
        "entropyPathCorrelation",
        "assessmentStrength",
    ]

    @property
    def feature_vector(self) -> list[float]:
        return [
            min(1.0, max(0.0, self.mean_entropy / 10.0)),
            min(1.0, max(0.0, self.entropy_variance / 5.0)),
            min(1.0, max(0.0, self.first_token_entropy / 10.0)),
            min(1.0, max(0.0, float(self.gate_count) / 20.0)),
            min(1.0, max(0.0, self.mean_gate_confidence)),
            min(1.0, max(0.0, self.dominant_gate_category)),
            min(1.0, max(0.0, (self.entropy_path_correlation + 1.0) / 2.0)),
            min(1.0, max(0.0, self.assessment_strength)),
        ]

    @staticmethod
    def from_measurement(
        measurement,
        prompt_hash: str,
        intervention_level: Optional[int] = None,
    ) -> "ManifoldPoint":
        if measurement.gate_details:
            mean_confidence = sum(detail.confidence for detail in measurement.gate_details) / float(
                len(measurement.gate_details)
            )
        else:
            mean_confidence = 0.0

        dominant_category = ManifoldPoint._compute_dominant_gate_category(measurement.gate_sequence)
        relationship_strength = measurement.assessment.relationship_strength
        strength_key = relationship_strength.value if hasattr(relationship_strength, "value") else str(relationship_strength)
        strength_key = strength_key.lower()
        assessment_strength = {
            "strong": 1.0,
            "moderate": 0.66,
            "weak": 0.33,
            "none": 0.0,
        }.get(strength_key, 0.0)

        return ManifoldPoint(
            id=uuid4(),
            mean_entropy=measurement.mean_entropy,
            entropy_variance=measurement.entropy_variance,
            first_token_entropy=measurement.first_token_entropy,
            gate_count=measurement.gate_count,
            mean_gate_confidence=mean_confidence,
            dominant_gate_category=dominant_category,
            entropy_path_correlation=measurement.entropy_path_correlation or 0.0,
            assessment_strength=assessment_strength,
            prompt_hash=prompt_hash,
            intervention_level=intervention_level,
        )

    @staticmethod
    def _compute_dominant_gate_category(gate_sequence: list[str]) -> float:
        if not gate_sequence:
            return 0.0
        counts: dict[str, int] = {}
        for gate in gate_sequence:
            counts[gate] = counts.get(gate, 0) + 1
        dominant = max(counts.items(), key=lambda item: item[1])[0]

        known_gates = [
            "INIT",
            "REASON",
            "BRANCH",
            "LOOP",
            "CONCLUDE",
            "RECALL",
            "COMPARE",
            "SYNTHESIZE",
            "EVALUATE",
            "OUTPUT",
        ]
        if dominant in known_gates:
            return float(known_gates.index(dominant)) / float(len(known_gates) - 1)
        return 0.5

    def distance(self, other: "ManifoldPoint") -> float:
        v1 = self.feature_vector
        v2 = other.feature_vector
        total = 0.0
        for i in range(self.feature_dimension):
            diff = v1[i] - v2[i]
            total += diff * diff
        return total**0.5


@dataclass(frozen=True)
class ManifoldRegion:
    class RegionType(str, Enum):
        safe = "safe"
        sparse = "sparse"
        boundary = "boundary"

    id: UUID
    region_type: "ManifoldRegion.RegionType"
    centroid: ManifoldPoint
    member_count: int
    member_ids: list[UUID]
    dominant_gates: list[str]
    intrinsic_dimension: Optional[float]
    radius: float
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def classify(centroid: ManifoldPoint) -> "ManifoldRegion.RegionType":
        entropy = centroid.mean_entropy
        variance = centroid.entropy_variance
        coherence = centroid.mean_gate_confidence

        low_entropy = 3.0
        high_entropy = 6.0
        low_variance = 0.5
        high_variance = 2.0
        high_coherence = 0.7
        low_coherence = 0.4

        if variance > high_variance:
            return ManifoldRegion.RegionType.boundary
        if entropy > low_entropy and entropy < high_entropy and variance > low_variance:
            return ManifoldRegion.RegionType.boundary
        if entropy > high_entropy and coherence < low_coherence:
            return ManifoldRegion.RegionType.sparse
        if entropy > high_entropy:
            return ManifoldRegion.RegionType.sparse
        if entropy < low_entropy and variance < low_variance and coherence > high_coherence:
            return ManifoldRegion.RegionType.safe
        return ManifoldRegion.RegionType.boundary


@dataclass(frozen=True)
class RegionQueryResult:
    nearest_region: Optional[ManifoldRegion]
    distance: float
    is_within_region: bool
    suggested_type: ManifoldRegion.RegionType
    confidence: float


@dataclass(frozen=True)
class InterventionSuggestion:
    level: int
    reason: str
    confidence: float
    based_on_history: bool
    similar_point_count: int

    @staticmethod
    def no_history() -> "InterventionSuggestion":
        return InterventionSuggestion(
            level=0,
            reason="No historical data available",
            confidence=0.0,
            based_on_history=False,
            similar_point_count=0,
        )
