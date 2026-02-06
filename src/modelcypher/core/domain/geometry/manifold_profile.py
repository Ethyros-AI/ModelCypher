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

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import ulp_scalar

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
    class Statistics:
        total_points: int
        region_count: int
        dense_region_count: int
        sparse_region_count: int
        transitional_region_count: int
        mean_intrinsic_dimension: float | None
        recent_point_count: int


    def compute_statistics(self) -> "ManifoldProfile.Statistics":
        dense_count = sum(
            1
            for region in self.regions
            if region.region_type == ManifoldRegion.RegionCharacter.DENSE
        )
        sparse_count = sum(
            1
            for region in self.regions
            if region.region_type == ManifoldRegion.RegionCharacter.SPARSE
        )
        transitional_count = sum(
            1
            for region in self.regions
            if region.region_type == ManifoldRegion.RegionCharacter.TRANSITIONAL
        )
        dimensions = [
            region.intrinsic_dimension
            for region in self.regions
            if region.intrinsic_dimension is not None
        ]
        mean_dim = sum(dimensions) / float(len(dimensions)) if dimensions else None

        return ManifoldProfile.Statistics(
            total_points=self.total_point_count,
            region_count=len(self.regions),
            dense_region_count=dense_count,
            sparse_region_count=sparse_count,
            transitional_region_count=transitional_count,
            mean_intrinsic_dimension=mean_dim,
            recent_point_count=len(self.recent_points),
        )


@dataclass(frozen=True)
class ManifoldPoint:
    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    gate_count: int
    mean_gate_similarity: float
    dominant_gate_category: float
    entropy_path_correlation: float
    assessment_strength: float
    prompt_hash: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    intervention_level: int | None = None

    feature_dimension: ClassVar[int] = 8
    feature_names: ClassVar[list[str]] = [
        "meanEntropy",
        "entropyVariance",
        "firstTokenEntropy",
        "gateCount",
        "meanGateSimilarity",
        "dominantGateCategory",
        "entropyPathCorrelation",
        "assessmentStrength",
    ]

    @property
    def feature_vector(self) -> list[float]:
        return [
            float(self.mean_entropy),
            float(self.entropy_variance),
            float(self.first_token_entropy),
            float(self.gate_count),
            float(self.mean_gate_similarity),
            float(self.dominant_gate_category),
            float(self.entropy_path_correlation),
            float(self.assessment_strength),
        ]

    @staticmethod
    def from_measurement(
        measurement,
        prompt_hash: str,
        intervention_level: int | None = None,
    ) -> "ManifoldPoint":
        if measurement.gate_details:
            mean_similarity = sum(detail.similarity for detail in measurement.gate_details) / float(
                len(measurement.gate_details)
            )
        else:
            mean_similarity = 0.0

        dominant_category = ManifoldPoint._compute_dominant_gate_category(measurement.gate_sequence)
        # Use raw correlation magnitude as assessment strength (0-1 range)
        correlation = measurement.assessment.correlation
        assessment_strength = abs(correlation) if correlation is not None else 0.0

        return ManifoldPoint(
            id=uuid4(),
            mean_entropy=measurement.mean_entropy,
            entropy_variance=measurement.entropy_variance,
            first_token_entropy=measurement.first_token_entropy,
            gate_count=measurement.gate_count,
            mean_gate_similarity=mean_similarity,
            dominant_gate_category=dominant_category,
            entropy_path_correlation=measurement.entropy_path_correlation or 0.0,
            assessment_strength=assessment_strength,
            prompt_hash=prompt_hash,
            intervention_level=intervention_level,
        )

    def distance(self, other: "ManifoldPoint") -> float:
        """Compute geodesic distance to another ManifoldPoint.

        For two points, the k-NN graph uses a single edge, so geodesic
        reduces to the chord distance. We use the geodesic code path
        for consistency with manifold-aware operations.

        Args:
            other: The other ManifoldPoint to measure distance to.

        Returns:
            Geodesic distance between the two points.
        """
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        backend = get_default_backend()

        # Build feature matrix for both points
        p1 = backend.array(self.feature_vector)
        p2 = backend.array(other.feature_vector)
        features = backend.stack([p1, p2], axis=0)

        # Compute geodesic distance via k-NN graph
        rg = RiemannianGeometry(backend)
        result = rg.geodesic_distances(features, k_neighbors=1)
        backend.eval(result.distances)

        row = backend.take(result.distances, backend.array([0]), axis=0)
        row = backend.squeeze(row, axis=0)
        cell = backend.take(row, backend.array([1]), axis=0)
        cell = backend.squeeze(cell)
        backend.eval(cell)
        return float(backend.to_scalar(cell))

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


@dataclass(frozen=True)
class RegionThresholds:
    """Thresholds for region classification, derived from data.

    Thresholds define boundaries between topological regions (DENSE/SPARSE/TRANSITIONAL).
    These are geometric classifications, not quality judgments.

    Use `from_data()` to derive thresholds directly from observed distributions.
    """

    low_entropy: float
    high_entropy: float
    low_variance: float
    high_variance: float
    high_coherence: float
    low_coherence: float

    @staticmethod
    def _gap_thresholds(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        sorted_vals = sorted(float(v) for v in values)
        if len(sorted_vals) == 1:
            return sorted_vals[0], sorted_vals[0]
        backend = get_default_backend()
        scale = max(max(abs(v) for v in sorted_vals), 1.0)
        eps = ulp_scalar(scale, backend)
        gaps: list[tuple[float, int]] = []
        for i in range(len(sorted_vals) - 1):
            denom = max(abs(sorted_vals[i]), eps)
            rel_gap = (sorted_vals[i + 1] - sorted_vals[i]) / denom
            gaps.append((rel_gap, i))
        gaps_sorted = sorted(gaps, key=lambda item: item[0], reverse=True)
        idxs = sorted(i for _, i in gaps_sorted[:2])
        if len(idxs) == 1:
            threshold = sorted_vals[idxs[0]]
            return threshold, threshold
        low = sorted_vals[idxs[0]]
        high = sorted_vals[idxs[1]]
        if high < low:
            low, high = high, low
        return low, high

    @classmethod
    def from_data(
        cls,
        entropies: list[float],
        variances: list[float],
        coherences: list[float],
    ) -> "RegionThresholds":
        """Derive thresholds from observed data distributions.

        Uses largest relative gaps in each distribution to split low/high regimes.
        No fixed percentiles or external parameters are required.
        """
        low_entropy, high_entropy = cls._gap_thresholds(entropies)
        low_variance, high_variance = cls._gap_thresholds(variances)
        low_coherence, high_coherence = cls._gap_thresholds(coherences)
        return cls(
            low_entropy=low_entropy,
            high_entropy=high_entropy,
            low_variance=low_variance,
            high_variance=high_variance,
            high_coherence=high_coherence,
            low_coherence=low_coherence,
        )


# =============================================================================
# NO DEFAULT THRESHOLDS
# =============================================================================
# All thresholds must be derived from data using RegionThresholds.from_data().
# There are no "standard" thresholds - they depend on the data distribution.
# =============================================================================


@dataclass(frozen=True)
class ManifoldRegion:
    class RegionCharacter(str, Enum):
        """Topological character of a manifold region.

        These describe the local geometry, not safety judgments.
        """

        DENSE = "dense"  # Well-sampled, tightly clustered
        SPARSE = "sparse"  # Poorly sampled, high entropy
        TRANSITIONAL = "transitional"  # Region boundary, high variance

    id: UUID
    region_type: "ManifoldRegion.RegionCharacter"
    centroid: ManifoldPoint
    member_count: int
    member_ids: list[UUID]
    dominant_gates: list[str]
    intrinsic_dimension: float | None
    radius: float
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Raw measurements for the region (what the geometry IS)
    @property
    def entropy(self) -> float:
        """Mean entropy of the region centroid."""
        return self.centroid.mean_entropy

    @property
    def variance(self) -> float:
        """Entropy variance of the region centroid."""
        return self.centroid.entropy_variance

    @property
    def coherence(self) -> float:
        """Gate coherence of the region centroid."""
        return self.centroid.mean_gate_similarity

    @staticmethod
    def classify(
        centroid: ManifoldPoint,
        thresholds: RegionThresholds,
    ) -> "ManifoldRegion.RegionCharacter":
        """Classify region character based on centroid measurements.

        Args:
            centroid: The centroid point to classify.
            thresholds: Classification thresholds derived from data using
                `RegionThresholds.from_data()`.

        Returns:
            Topological character (DENSE, SPARSE, or TRANSITIONAL).
        """
        entropy = centroid.mean_entropy
        variance = centroid.entropy_variance
        coherence = centroid.mean_gate_similarity

        # High variance indicates transitional region
        if variance > thresholds.high_variance:
            return ManifoldRegion.RegionCharacter.TRANSITIONAL
        if entropy > thresholds.low_entropy and entropy < thresholds.high_entropy and variance > thresholds.low_variance:
            return ManifoldRegion.RegionCharacter.TRANSITIONAL

        # High entropy with low coherence indicates sparse region
        if entropy > thresholds.high_entropy and coherence < thresholds.low_coherence:
            return ManifoldRegion.RegionCharacter.SPARSE
        if entropy > thresholds.high_entropy:
            return ManifoldRegion.RegionCharacter.SPARSE

        # Low entropy, low variance, high coherence indicates dense region
        if entropy < thresholds.low_entropy and variance < thresholds.low_variance and coherence > thresholds.high_coherence:
            return ManifoldRegion.RegionCharacter.DENSE

        return ManifoldRegion.RegionCharacter.TRANSITIONAL


@dataclass(frozen=True)
class RegionQueryResult:
    nearest_region: ManifoldRegion | None
    distance: float
    is_within_region: bool
    suggested_character: ManifoldRegion.RegionCharacter
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
