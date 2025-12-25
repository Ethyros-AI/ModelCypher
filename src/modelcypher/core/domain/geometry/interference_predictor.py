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

"""Interference prediction for model merging.

Uses ConceptVolume analysis to predict whether merging two models
will result in constructive or destructive interference.

Constructive Interference:
- Concepts reinforce each other
- Merged model gains capabilities from both sources
- Overlap is complementary, not conflicting

Destructive Interference:
- Concepts cancel or confuse each other
- Merged model loses capabilities
- Overlap creates inconsistent representations

Key Insight: By measuring volume overlap and curvature mismatch
BEFORE merging, we can predict quality without expensive post-merge
evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    pass

from .riemannian_density import (
    ConceptVolume,
    ConceptVolumeRelation,
    RiemannianDensityEstimator,
)

logger = logging.getLogger(__name__)


class InterferenceType(str, Enum):
    """Classification of interference between concept volumes."""

    CONSTRUCTIVE = "constructive"  # Concepts reinforce (good)
    NEUTRAL = "neutral"  # Minimal interaction (safe)
    PARTIAL_DESTRUCTIVE = "partial_destructive"  # Some conflict (risky)
    DESTRUCTIVE = "destructive"  # Major conflict (dangerous)
    UNKNOWN = "unknown"  # Insufficient data


class InterferenceMechanism(str, Enum):
    """Root cause of interference."""

    VOLUME_OVERLAP = "volume_overlap"  # Physical overlap in activation space
    CURVATURE_MISMATCH = "curvature_mismatch"  # Different local geometries
    SUBSPACE_CONFLICT = "subspace_conflict"  # Misaligned principal directions
    BOUNDARY_COLLISION = "boundary_collision"  # Edge effects at volume boundaries
    SEMANTIC_COLLISION = "semantic_collision"  # Same region, different meanings


@dataclass(frozen=True)
class InterferencePredictorConfig:
    """Configuration for interference prediction.

    Thresholds classify interference but don't reject merges.
    Use from_overlap_distribution() to derive thresholds from actual data.
    Equal weights let geometry speak for itself.
    """

    # Thresholds for interference classification
    constructive_bhattacharyya_min: float = 0.25
    constructive_bhattacharyya_max: float = 0.75
    destructive_overlap_threshold: float = 0.9
    neutral_overlap_max: float = 0.1

    # Curvature mismatch thresholds
    curvature_mismatch_warning: float = 0.25
    curvature_mismatch_critical: float = 0.5

    # Subspace alignment thresholds
    subspace_alignment_good: float = 0.75
    subspace_alignment_bad: float = 0.25

    # Equal weights - geometry speaks for itself
    overlap_weight: float = 0.25
    curvature_weight: float = 0.25
    alignment_weight: float = 0.25
    distance_weight: float = 0.25

    @classmethod
    def from_overlap_distribution(
        cls,
        overlap_scores: list[float],
        *,
        destructive_percentile: float = 0.90,
        constructive_min_percentile: float = 0.25,
        constructive_max_percentile: float = 0.75,
        neutral_max_percentile: float = 0.10,
    ) -> "InterferencePredictorConfig":
        """Derive thresholds from observed overlap score distribution.

        Instead of arbitrary thresholds, derives them from the actual
        distribution of Bhattacharyya coefficients or overlap scores.

        Args:
            overlap_scores: List of overlap scores from concept pairs.
            destructive_percentile: Percentile for destructive threshold.
            constructive_min_percentile: Percentile for constructive min.
            constructive_max_percentile: Percentile for constructive max.
            neutral_max_percentile: Percentile for neutral max.

        Returns:
            Configuration with distribution-derived thresholds.
        """
        if not overlap_scores:
            return cls()

        sorted_scores = sorted(overlap_scores)
        n = len(sorted_scores)

        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_scores[idx]

        return cls(
            destructive_overlap_threshold=percentile(destructive_percentile),
            constructive_bhattacharyya_min=percentile(constructive_min_percentile),
            constructive_bhattacharyya_max=percentile(constructive_max_percentile),
            neutral_overlap_max=percentile(neutral_max_percentile),
        )


@dataclass
class InterferenceResult:
    """Result of interference prediction between two concept volumes."""

    # Volumes analyzed
    volume_a_id: str
    volume_b_id: str

    # Primary classification
    interference_type: InterferenceType

    # Confidence in prediction (0-1)
    confidence: float

    # Root causes
    mechanisms: list[InterferenceMechanism]

    # Detailed scores
    overlap_score: float  # 0=no overlap, 1=complete overlap
    curvature_score: float  # 0=perfect match, 1=severe mismatch
    alignment_score: float  # 0=orthogonal, 1=aligned
    distance_score: float  # 0=identical, 1=far apart

    # Composite safety score (0=dangerous, 1=safe)
    safety_score: float

    # Recommendations
    recommended_action: str
    risk_factors: list[str]
    mitigation_strategies: list[str]

    @property
    def is_safe(self) -> bool:
        """Check if merge is considered safe."""
        return self.interference_type in (InterferenceType.CONSTRUCTIVE, InterferenceType.NEUTRAL)

    @property
    def is_risky(self) -> bool:
        """Check if merge has significant risk."""
        return self.interference_type in (
            InterferenceType.PARTIAL_DESTRUCTIVE,
            InterferenceType.DESTRUCTIVE,
        )


@dataclass
class GlobalInterferenceReport:
    """Aggregate interference analysis across all concept pairs."""

    # Per-pair results
    pair_results: dict[tuple[str, str], InterferenceResult]

    # Aggregate statistics
    total_pairs: int
    constructive_count: int
    neutral_count: int
    partial_destructive_count: int
    destructive_count: int

    # Global safety assessment
    overall_safety_score: float
    overall_recommendation: str

    # High-risk pairs
    critical_pairs: list[tuple[str, str]]

    # Concept-level risk
    concept_risk_scores: dict[str, float]

    @property
    def is_globally_safe(self) -> bool:
        """Check if overall merge is considered safe."""
        return self.destructive_count == 0 and self.overall_safety_score >= 0.5

    def get_pairs_by_type(self, interference_type: InterferenceType) -> list[tuple[str, str]]:
        """Get all pairs with specific interference type."""
        return [
            pair
            for pair, result in self.pair_results.items()
            if result.interference_type == interference_type
        ]


class InterferencePredictor:
    """Predicts interference between concept volumes for merge planning.

    This is the primary interface for pre-merge quality prediction.

    Usage:
        predictor = InterferencePredictor()
        result = predictor.predict(volume_a, volume_b)
        if result.is_risky:
            print(result.mitigation_strategies)
    """

    def __init__(self, config: InterferencePredictorConfig | None = None):
        self.config = config or InterferencePredictorConfig()
        self.density_estimator = RiemannianDensityEstimator()

    def predict(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
        relation: ConceptVolumeRelation | None = None,
    ) -> InterferenceResult:
        """Predict interference between two concept volumes.

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume
            relation: Pre-computed relation (optional, will compute if not provided)

        Returns:
            InterferenceResult with classification and recommendations
        """
        # Compute relation if not provided
        if relation is None:
            relation = self.density_estimator.compute_relation(volume_a, volume_b)

        # Compute component scores
        overlap_score = self._compute_overlap_score(relation)
        curvature_score = self._compute_curvature_score(relation)
        alignment_score = self._compute_alignment_score(relation)
        distance_score = self._compute_distance_score(relation)

        # Identify mechanisms
        mechanisms = self._identify_mechanisms(
            relation, overlap_score, curvature_score, alignment_score
        )

        # Classify interference type
        interference_type = self._classify_interference(
            overlap_score, curvature_score, alignment_score, distance_score, mechanisms
        )

        # Compute safety score
        safety_score = self._compute_safety_score(
            overlap_score, curvature_score, alignment_score, distance_score
        )

        # Compute confidence
        confidence = self._compute_confidence(relation, interference_type)

        # Generate recommendations
        recommended_action, risk_factors, mitigations = self._generate_recommendations(
            interference_type, mechanisms, safety_score
        )

        return InterferenceResult(
            volume_a_id=volume_a.concept_id,
            volume_b_id=volume_b.concept_id,
            interference_type=interference_type,
            confidence=confidence,
            mechanisms=mechanisms,
            overlap_score=overlap_score,
            curvature_score=curvature_score,
            alignment_score=alignment_score,
            distance_score=distance_score,
            safety_score=safety_score,
            recommended_action=recommended_action,
            risk_factors=risk_factors,
            mitigation_strategies=mitigations,
        )

    def predict_global(
        self,
        volumes: dict[str, ConceptVolume],
        relations: dict[tuple[str, str], ConceptVolumeRelation] | None = None,
    ) -> GlobalInterferenceReport:
        """Predict interference across all concept volume pairs.

        Args:
            volumes: Dict mapping concept_id to ConceptVolume
            relations: Pre-computed relations (optional)

        Returns:
            GlobalInterferenceReport with aggregate analysis
        """
        from .riemannian_density import compute_pairwise_relations

        # Compute relations if not provided
        if relations is None:
            relations = compute_pairwise_relations(self.density_estimator, volumes)

        # Predict for each pair
        pair_results = {}
        for (id_a, id_b), relation in relations.items():
            result = self.predict(volumes[id_a], volumes[id_b], relation)
            pair_results[(id_a, id_b)] = result

        # Aggregate statistics
        total = len(pair_results)
        type_counts = {t: 0 for t in InterferenceType}
        for result in pair_results.values():
            type_counts[result.interference_type] += 1

        # Identify critical pairs
        critical_pairs = [
            pair
            for pair, result in pair_results.items()
            if result.interference_type == InterferenceType.DESTRUCTIVE
        ]

        # Compute per-concept risk scores
        concept_risk_scores = self._compute_concept_risk_scores(pair_results, volumes)

        # Overall safety score (average of pair safety scores)
        if pair_results:
            scores = [r.safety_score for r in pair_results.values()]
            overall_safety = sum(scores) / len(scores)
        else:
            overall_safety = 1.0

        # Generate overall recommendation
        overall_recommendation = self._generate_global_recommendation(
            type_counts, overall_safety, critical_pairs
        )

        return GlobalInterferenceReport(
            pair_results=pair_results,
            total_pairs=total,
            constructive_count=type_counts[InterferenceType.CONSTRUCTIVE],
            neutral_count=type_counts[InterferenceType.NEUTRAL],
            partial_destructive_count=type_counts[InterferenceType.PARTIAL_DESTRUCTIVE],
            destructive_count=type_counts[InterferenceType.DESTRUCTIVE],
            overall_safety_score=overall_safety,
            overall_recommendation=overall_recommendation,
            critical_pairs=critical_pairs,
            concept_risk_scores=concept_risk_scores,
        )

    def _compute_overlap_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute overlap score from relation metrics."""
        bc = relation.bhattacharyya_coefficient
        oc = relation.overlap_coefficient
        jc = relation.jaccard_index

        # Equal contribution from each geometric measure
        return (bc + oc + jc) / 3.0

    def _compute_curvature_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute curvature mismatch score (higher = worse)."""
        return relation.curvature_divergence

    def _compute_alignment_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute subspace alignment score (higher = better)."""
        return relation.subspace_alignment

    def _compute_distance_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute normalized distance score (higher = farther apart)."""
        r_a = relation.volume_a.effective_radius
        r_b = relation.volume_b.effective_radius
        sum_radius = r_a + r_b

        if sum_radius < 1e-10:
            return 0.0

        # Normalize by sum of radii (touching spheres = 1.0)
        normalized_dist = relation.geodesic_centroid_distance / sum_radius
        return min(normalized_dist, 1.0)

    def _identify_mechanisms(
        self,
        relation: ConceptVolumeRelation,
        overlap_score: float,
        curvature_score: float,
        alignment_score: float,
    ) -> list[InterferenceMechanism]:
        """Identify root causes of interference."""
        mechanisms = []

        # Use config thresholds consistently
        if overlap_score > self.config.constructive_bhattacharyya_max:
            mechanisms.append(InterferenceMechanism.VOLUME_OVERLAP)

        if curvature_score > self.config.curvature_mismatch_warning:
            mechanisms.append(InterferenceMechanism.CURVATURE_MISMATCH)

        if alignment_score < self.config.subspace_alignment_bad:
            mechanisms.append(InterferenceMechanism.SUBSPACE_CONFLICT)

        # Boundary collision from Mahalanobis asymmetry
        mahal_sum = relation.mahalanobis_distance_ab + relation.mahalanobis_distance_ba
        if mahal_sum > 1e-10:
            mahal_asymmetry = abs(
                relation.mahalanobis_distance_ab - relation.mahalanobis_distance_ba
            ) / mahal_sum

            if mahal_asymmetry > 0.5 and overlap_score > self.config.constructive_bhattacharyya_min:
                mechanisms.append(InterferenceMechanism.BOUNDARY_COLLISION)

        return mechanisms

    def _classify_interference(
        self,
        overlap_score: float,
        curvature_score: float,
        alignment_score: float,
        distance_score: float,
        mechanisms: list[InterferenceMechanism],
    ) -> InterferenceType:
        """Classify interference type based on scores."""
        cfg = self.config

        # High overlap with poor alignment = destructive
        if overlap_score > cfg.destructive_overlap_threshold:
            if alignment_score < cfg.subspace_alignment_bad:
                return InterferenceType.DESTRUCTIVE
            elif curvature_score > cfg.curvature_mismatch_critical:
                return InterferenceType.DESTRUCTIVE

        # Very low overlap = neutral
        if overlap_score < cfg.neutral_overlap_max:
            return InterferenceType.NEUTRAL

        # Moderate overlap with good alignment = constructive
        if (
            cfg.constructive_bhattacharyya_min
            <= overlap_score
            <= cfg.constructive_bhattacharyya_max
        ):
            if alignment_score >= cfg.subspace_alignment_good:
                if curvature_score < cfg.curvature_mismatch_warning:
                    return InterferenceType.CONSTRUCTIVE

        # High overlap or poor alignment = partial destructive
        if overlap_score > cfg.constructive_bhattacharyya_max:
            return InterferenceType.PARTIAL_DESTRUCTIVE

        if len(mechanisms) >= 2:
            return InterferenceType.PARTIAL_DESTRUCTIVE

        return InterferenceType.NEUTRAL

    def _compute_safety_score(
        self,
        overlap_score: float,
        curvature_score: float,
        alignment_score: float,
        distance_score: float,
    ) -> float:
        """Compute composite safety score (0=dangerous, 1=safe)."""
        cfg = self.config

        # Transform scores to safety contributions
        overlap_safety = max(1 - overlap_score, distance_score)
        curvature_safety = 1 - curvature_score
        alignment_safety = alignment_score

        # Equal contribution from each geometric signal
        safety = (
            cfg.overlap_weight * overlap_safety
            + cfg.curvature_weight * curvature_safety
            + cfg.alignment_weight * alignment_safety
            + cfg.distance_weight * distance_score
        )

        return max(0.0, min(1.0, safety))

    def _compute_confidence(
        self,
        relation: ConceptVolumeRelation,
        interference_type: InterferenceType,
    ) -> float:
        """Compute confidence in the interference prediction."""
        n_a = relation.volume_a.num_samples
        n_b = relation.volume_b.num_samples
        d = relation.volume_a.dimension

        # Confidence from sample/dimension ratio (statistical sufficiency)
        min_samples = min(n_a, n_b)
        sample_ratio = min_samples / max(d, 1)

        # Curvature availability increases confidence
        has_curvature_a = relation.volume_a.local_curvature is not None
        has_curvature_b = relation.volume_b.local_curvature is not None
        curvature_factor = 1.0 if (has_curvature_a and has_curvature_b) else 0.5

        confidence = min(1.0, sample_ratio) * curvature_factor
        return max(0.0, min(1.0, confidence))

    def _generate_recommendations(
        self,
        interference_type: InterferenceType,
        mechanisms: list[InterferenceMechanism],
        safety_score: float,
    ) -> tuple[str, list[str], list[str]]:
        """Generate action recommendation and mitigation strategies."""
        risk_factors = []
        mitigations = []

        for mechanism in mechanisms:
            if mechanism == InterferenceMechanism.VOLUME_OVERLAP:
                risk_factors.append("High activation space overlap")
                mitigations.append("Reduce alpha for overlapping layers")

            elif mechanism == InterferenceMechanism.CURVATURE_MISMATCH:
                risk_factors.append("Manifold curvature mismatch")
                mitigations.append("Use curvature-corrected alpha adjustment")

            elif mechanism == InterferenceMechanism.SUBSPACE_CONFLICT:
                risk_factors.append("Misaligned principal subspaces")
                mitigations.append("Apply Procrustes alignment before merge")

            elif mechanism == InterferenceMechanism.BOUNDARY_COLLISION:
                risk_factors.append("Asymmetric boundary effects")
                mitigations.append("Apply Gaussian smoothing at boundaries")

            elif mechanism == InterferenceMechanism.SEMANTIC_COLLISION:
                risk_factors.append("Semantic meaning conflict")
                mitigations.append("Use knowledge probes to verify post-merge")

        # Primary recommendation
        if interference_type == InterferenceType.CONSTRUCTIVE:
            action = "Proceed with merge - concepts should reinforce"
        elif interference_type == InterferenceType.NEUTRAL:
            action = "Safe to merge - minimal interaction expected"
        elif interference_type == InterferenceType.PARTIAL_DESTRUCTIVE:
            action = "Proceed with caution - apply mitigations before merge"
        elif interference_type == InterferenceType.DESTRUCTIVE:
            action = "High risk - consider alternative merge strategy or skip pair"
        else:
            action = "Insufficient data for confident prediction"

        return action, risk_factors, mitigations

    def _compute_concept_risk_scores(
        self,
        pair_results: dict[tuple[str, str], InterferenceResult],
        volumes: dict[str, ConceptVolume],
    ) -> dict[str, float]:
        """Compute per-concept aggregate risk scores."""
        concept_risks: dict[str, list[float]] = {cid: [] for cid in volumes}

        for (id_a, id_b), result in pair_results.items():
            risk = 1 - result.safety_score
            concept_risks[id_a].append(risk)
            concept_risks[id_b].append(risk)

        # Average risk per concept
        return {
            cid: (sum(risks) / len(risks) if risks else 0.0) for cid, risks in concept_risks.items()
        }

    def _generate_global_recommendation(
        self,
        type_counts: dict[InterferenceType, int],
        overall_safety: float,
        critical_pairs: list[tuple[str, str]],
    ) -> str:
        """Generate overall merge recommendation."""
        if type_counts[InterferenceType.DESTRUCTIVE] > 0:
            n_critical = len(critical_pairs)
            return f"HIGH RISK: {n_critical} critical pair(s) detected. Review and mitigate before proceeding."

        n_partial = type_counts[InterferenceType.PARTIAL_DESTRUCTIVE]
        if n_partial > 0:
            return f"MODERATE RISK: {n_partial} partially destructive pairs. Apply recommended mitigations."

        if overall_safety >= self.config.subspace_alignment_good:
            return "LOW RISK: Merge should proceed smoothly with good knowledge retention."

        if overall_safety >= 0.5:
            return "ACCEPTABLE: Merge feasible with minor quality monitoring recommended."

        return "UNCERTAIN: Mixed signals - consider targeted alpha reduction for risky pairs."


def quick_interference_check(
    source_activations: dict[str, "Array"],
    target_activations: dict[str, "Array"],
) -> GlobalInterferenceReport:
    """Quick interface for interference prediction.

    Args:
        source_activations: Dict mapping concept_id to source model activations
        target_activations: Dict mapping concept_id to target model activations

    Returns:
        GlobalInterferenceReport with merge safety assessment
    """
    # Find common concepts
    common_concepts = set(source_activations.keys()) & set(target_activations.keys())

    if not common_concepts:
        logger.warning("No common concepts between source and target")
        return GlobalInterferenceReport(
            pair_results={},
            total_pairs=0,
            constructive_count=0,
            neutral_count=0,
            partial_destructive_count=0,
            destructive_count=0,
            overall_safety_score=1.0,
            overall_recommendation="No common concepts to analyze",
            critical_pairs=[],
            concept_risk_scores={},
        )

    # Estimate volumes
    estimator = RiemannianDensityEstimator()
    source_volumes = {}
    target_volumes = {}

    for concept_id in common_concepts:
        source_volumes[f"source:{concept_id}"] = estimator.estimate_concept_volume(
            f"source:{concept_id}",
            source_activations[concept_id],
        )
        target_volumes[f"target:{concept_id}"] = estimator.estimate_concept_volume(
            f"target:{concept_id}",
            target_activations[concept_id],
        )

    # Combine and predict
    all_volumes = {**source_volumes, **target_volumes}
    predictor = InterferencePredictor()

    # Only compare source:X with target:X (same concept across models)
    pair_results = {}
    for concept_id in common_concepts:
        source_key = f"source:{concept_id}"
        target_key = f"target:{concept_id}"
        result = predictor.predict(all_volumes[source_key], all_volumes[target_key])
        pair_results[(source_key, target_key)] = result

    # Build report
    total = len(pair_results)
    type_counts = {t: 0 for t in InterferenceType}
    for result in pair_results.values():
        type_counts[result.interference_type] += 1

    critical_pairs = [
        pair
        for pair, result in pair_results.items()
        if result.interference_type == InterferenceType.DESTRUCTIVE
    ]

    if pair_results:
        scores = [r.safety_score for r in pair_results.values()]
        overall_safety = sum(scores) / len(scores)
    else:
        overall_safety = 1.0

    concept_risk_scores = predictor._compute_concept_risk_scores(pair_results, all_volumes)

    overall_recommendation = predictor._generate_global_recommendation(
        type_counts, overall_safety, critical_pairs
    )

    return GlobalInterferenceReport(
        pair_results=pair_results,
        total_pairs=total,
        constructive_count=type_counts[InterferenceType.CONSTRUCTIVE],
        neutral_count=type_counts[InterferenceType.NEUTRAL],
        partial_destructive_count=type_counts[InterferenceType.PARTIAL_DESTRUCTIVE],
        destructive_count=type_counts[InterferenceType.DESTRUCTIVE],
        overall_safety_score=overall_safety,
        overall_recommendation=overall_recommendation,
        critical_pairs=critical_pairs,
        concept_risk_scores=concept_risk_scores,
    )
