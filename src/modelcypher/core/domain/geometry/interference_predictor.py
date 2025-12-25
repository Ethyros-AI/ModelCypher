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

"""Merge analysis for model merging.

Uses ConceptVolume analysis to identify what geometric transformations
are needed to align two models for merging.

Key Insight: Models are ALWAYS compatible. The high-dimensional shape
of knowledge is invariant. This module identifies WHAT transformations
to apply, not WHETHER to merge.

Transformation types:
- ALPHA_SCALING: Apply weighted blending in overlapping regions
- CURVATURE_CORRECTION: Apply curvature-corrected interpolation
- PROCRUSTES_ROTATION: Apply Procrustes rotation to align subspaces
- BOUNDARY_SMOOTHING: Apply Gaussian smoothing at volume boundaries
- SEMANTIC_VERIFICATION: Verify alignment with knowledge probes
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


class TransformationType(str, Enum):
    """Geometric transformation needed for alignment.

    Each type indicates what operation should be applied during merge.
    These are factual descriptions of the geometry, not judgments.
    """

    ALPHA_SCALING = "alpha_scaling"  # Overlapping regions need weighted blending
    CURVATURE_CORRECTION = "curvature_correction"  # Apply curvature-corrected interpolation
    PROCRUSTES_ROTATION = "procrustes_rotation"  # Align subspaces before merge
    BOUNDARY_SMOOTHING = "boundary_smoothing"  # Apply Gaussian smoothing at edges
    SEMANTIC_VERIFICATION = "semantic_verification"  # Verify with knowledge probes


# Backward compatibility - old name pointed to different concept
InterferenceMechanism = TransformationType


@dataclass(frozen=True)
class MergeAnalysisConfig:
    """Configuration for merge analysis.

    Thresholds determine when specific transformations are applied.
    These are NOT safety thresholds - they identify what transformations
    the geometry requires. Models are always compatible.
    """

    # Thresholds for triggering transformations
    # When overlap exceeds this, apply alpha scaling
    alpha_scaling_threshold: float = 0.5

    # When curvature divergence exceeds this, apply curvature correction
    curvature_correction_threshold: float = 0.25

    # When alignment is below this, apply Procrustes rotation
    procrustes_threshold: float = 0.5

    # Mahalanobis asymmetry threshold for boundary smoothing
    boundary_asymmetry_threshold: float = 0.5

    # Equal weights for composite metrics (diagnostic only)
    overlap_weight: float = 0.25
    curvature_weight: float = 0.25
    alignment_weight: float = 0.25
    distance_weight: float = 0.25

    @classmethod
    def from_overlap_distribution(
        cls,
        overlap_scores: list[float],
        *,
        alpha_percentile: float = 0.50,
        curvature_percentile: float = 0.25,
        procrustes_percentile: float = 0.50,
    ) -> "MergeAnalysisConfig":
        """Derive thresholds from observed overlap score distribution.

        Uses data-driven thresholds rather than arbitrary values.

        Args:
            overlap_scores: List of overlap scores from concept pairs.
            alpha_percentile: Percentile for alpha scaling trigger.
            curvature_percentile: Percentile for curvature correction trigger.
            procrustes_percentile: Percentile for Procrustes rotation trigger.

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
            alpha_scaling_threshold=percentile(alpha_percentile),
            curvature_correction_threshold=percentile(curvature_percentile),
            procrustes_threshold=1.0 - percentile(procrustes_percentile),
        )


# Backward compatibility alias
InterferencePredictorConfig = MergeAnalysisConfig


@dataclass
class MergeAnalysisResult:
    """Result of merge analysis between two concept volumes.

    Contains raw geometric measurements and the transformations needed
    for alignment. Models are ALWAYS compatible - this describes HOW
    to merge, not WHETHER to merge.
    """

    # Volumes analyzed
    volume_a_id: str
    volume_b_id: str

    # Transformations needed for this merge
    transformations: list[TransformationType]

    # Raw geometric measurements (for diagnostics, not gating)
    overlap_score: float  # 0=no overlap, 1=complete overlap
    curvature_divergence: float  # 0=identical curvature, 1=maximum divergence
    alignment_score: float  # 0=orthogonal, 1=aligned
    distance_score: float  # 0=identical, 1=far apart

    # Statistical confidence in measurements
    measurement_confidence: float

    # Transformation descriptions (what the math will do)
    transformation_descriptions: list[str]


# Backward compatibility alias
InterferenceResult = MergeAnalysisResult


@dataclass
class GlobalMergeAnalysisReport:
    """Aggregate merge analysis across all concept pairs.

    Summarizes what transformations are needed across the entire merge.
    Models are ALWAYS compatible - this describes the total transformation
    effort required, not safety verdicts.
    """

    # Per-pair results
    pair_results: dict[tuple[str, str], MergeAnalysisResult]

    # Aggregate statistics
    total_pairs: int

    # Transformation counts (how many pairs need each transformation)
    transformation_counts: dict[TransformationType, int]

    # Average geometric measurements (for diagnostics)
    mean_overlap: float
    mean_curvature_divergence: float
    mean_alignment: float

    # Summary of transformations needed
    transformation_summary: str

    def get_pairs_needing_transformation(
        self, transformation: TransformationType
    ) -> list[tuple[str, str]]:
        """Get all pairs that need a specific transformation."""
        return [
            pair
            for pair, result in self.pair_results.items()
            if transformation in result.transformations
        ]


# Backward compatibility alias
GlobalInterferenceReport = GlobalMergeAnalysisReport


class MergeAnalyzer:
    """Analyzes concept volumes to determine merge transformations needed.

    This is the primary interface for pre-merge analysis. It identifies
    WHAT transformations are needed, not WHETHER to merge. Models are
    ALWAYS compatible.

    Usage:
        analyzer = MergeAnalyzer()
        result = analyzer.analyze(volume_a, volume_b)
        for t in result.transformations:
            print(f"Apply: {t.value}")
    """

    def __init__(self, config: MergeAnalysisConfig | None = None):
        self.config = config or MergeAnalysisConfig()
        self.density_estimator = RiemannianDensityEstimator()

    def analyze(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
        relation: ConceptVolumeRelation | None = None,
    ) -> MergeAnalysisResult:
        """Analyze merge requirements between two concept volumes.

        Args:
            volume_a: First concept volume
            volume_b: Second concept volume
            relation: Pre-computed relation (optional, will compute if not provided)

        Returns:
            MergeAnalysisResult with transformations needed
        """
        # Compute relation if not provided
        if relation is None:
            relation = self.density_estimator.compute_relation(volume_a, volume_b)

        # Compute raw geometric measurements
        overlap_score = self._compute_overlap_score(relation)
        curvature_divergence = self._compute_curvature_divergence(relation)
        alignment_score = self._compute_alignment_score(relation)
        distance_score = self._compute_distance_score(relation)

        # Identify transformations needed based on geometry
        transformations = self._identify_transformations(
            relation, overlap_score, curvature_divergence, alignment_score
        )

        # Generate transformation descriptions
        descriptions = self._generate_transformation_descriptions(transformations)

        # Compute measurement confidence
        confidence = self._compute_measurement_confidence(relation)

        return MergeAnalysisResult(
            volume_a_id=volume_a.concept_id,
            volume_b_id=volume_b.concept_id,
            transformations=transformations,
            overlap_score=overlap_score,
            curvature_divergence=curvature_divergence,
            alignment_score=alignment_score,
            distance_score=distance_score,
            measurement_confidence=confidence,
            transformation_descriptions=descriptions,
        )

    # Backward compatibility
    def predict(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
        relation: ConceptVolumeRelation | None = None,
    ) -> MergeAnalysisResult:
        """Backward compatibility alias for analyze()."""
        return self.analyze(volume_a, volume_b, relation)

    def analyze_global(
        self,
        volumes: dict[str, ConceptVolume],
        relations: dict[tuple[str, str], ConceptVolumeRelation] | None = None,
    ) -> GlobalMergeAnalysisReport:
        """Analyze merge requirements across all concept volume pairs.

        Args:
            volumes: Dict mapping concept_id to ConceptVolume
            relations: Pre-computed relations (optional)

        Returns:
            GlobalMergeAnalysisReport with aggregate analysis
        """
        from .riemannian_density import compute_pairwise_relations

        # Compute relations if not provided
        if relations is None:
            relations = compute_pairwise_relations(self.density_estimator, volumes)

        # Analyze each pair
        pair_results = {}
        for (id_a, id_b), relation in relations.items():
            result = self.analyze(volumes[id_a], volumes[id_b], relation)
            pair_results[(id_a, id_b)] = result

        # Count transformations needed
        transformation_counts = {t: 0 for t in TransformationType}
        for result in pair_results.values():
            for t in result.transformations:
                transformation_counts[t] += 1

        # Compute mean measurements
        if pair_results:
            mean_overlap = sum(r.overlap_score for r in pair_results.values()) / len(
                pair_results
            )
            mean_curvature = sum(
                r.curvature_divergence for r in pair_results.values()
            ) / len(pair_results)
            mean_alignment = sum(r.alignment_score for r in pair_results.values()) / len(
                pair_results
            )
        else:
            mean_overlap = 0.0
            mean_curvature = 0.0
            mean_alignment = 1.0

        # Generate summary
        summary = self._generate_transformation_summary(transformation_counts, len(pair_results))

        return GlobalMergeAnalysisReport(
            pair_results=pair_results,
            total_pairs=len(pair_results),
            transformation_counts=transformation_counts,
            mean_overlap=mean_overlap,
            mean_curvature_divergence=mean_curvature,
            mean_alignment=mean_alignment,
            transformation_summary=summary,
        )

    # Backward compatibility
    def predict_global(
        self,
        volumes: dict[str, ConceptVolume],
        relations: dict[tuple[str, str], ConceptVolumeRelation] | None = None,
    ) -> GlobalMergeAnalysisReport:
        """Backward compatibility alias for analyze_global()."""
        return self.analyze_global(volumes, relations)

    def _compute_overlap_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute overlap score from relation metrics."""
        bc = relation.bhattacharyya_coefficient
        oc = relation.overlap_coefficient
        jc = relation.jaccard_index

        # Equal contribution from each geometric measure
        return (bc + oc + jc) / 3.0

    def _compute_curvature_divergence(self, relation: ConceptVolumeRelation) -> float:
        """Compute curvature divergence score."""
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

    def _identify_transformations(
        self,
        relation: ConceptVolumeRelation,
        overlap_score: float,
        curvature_divergence: float,
        alignment_score: float,
    ) -> list[TransformationType]:
        """Identify what transformations are needed based on geometry."""
        transformations = []
        cfg = self.config

        # High overlap -> need alpha scaling
        if overlap_score > cfg.alpha_scaling_threshold:
            transformations.append(TransformationType.ALPHA_SCALING)

        # Curvature divergence -> need curvature correction
        if curvature_divergence > cfg.curvature_correction_threshold:
            transformations.append(TransformationType.CURVATURE_CORRECTION)

        # Low alignment -> need Procrustes rotation
        if alignment_score < cfg.procrustes_threshold:
            transformations.append(TransformationType.PROCRUSTES_ROTATION)

        # Asymmetric Mahalanobis distances -> boundary smoothing
        mahal_sum = relation.mahalanobis_distance_ab + relation.mahalanobis_distance_ba
        if mahal_sum > 1e-10:
            mahal_asymmetry = abs(
                relation.mahalanobis_distance_ab - relation.mahalanobis_distance_ba
            ) / mahal_sum

            if mahal_asymmetry > cfg.boundary_asymmetry_threshold:
                transformations.append(TransformationType.BOUNDARY_SMOOTHING)

        return transformations

    def _generate_transformation_descriptions(
        self, transformations: list[TransformationType]
    ) -> list[str]:
        """Generate human-readable descriptions of transformations."""
        descriptions = []

        for t in transformations:
            if t == TransformationType.ALPHA_SCALING:
                descriptions.append("Apply weighted alpha scaling in overlapping regions")
            elif t == TransformationType.CURVATURE_CORRECTION:
                descriptions.append("Apply curvature-corrected interpolation")
            elif t == TransformationType.PROCRUSTES_ROTATION:
                descriptions.append("Apply Procrustes rotation to align subspaces")
            elif t == TransformationType.BOUNDARY_SMOOTHING:
                descriptions.append("Apply Gaussian smoothing at volume boundaries")
            elif t == TransformationType.SEMANTIC_VERIFICATION:
                descriptions.append("Verify semantic alignment with knowledge probes")

        if not descriptions:
            descriptions.append("Direct merge - no transformations needed")

        return descriptions

    def _compute_measurement_confidence(
        self, relation: ConceptVolumeRelation
    ) -> float:
        """Compute confidence in the geometric measurements."""
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

    def _generate_transformation_summary(
        self, counts: dict[TransformationType, int], total_pairs: int
    ) -> str:
        """Generate summary of transformations needed across all pairs."""
        if total_pairs == 0:
            return "No pairs to analyze"

        parts = []
        for t, count in counts.items():
            if count > 0:
                pct = (count / total_pairs) * 100
                parts.append(f"{t.value}: {count} pairs ({pct:.0f}%)")

        if not parts:
            return "All pairs can be merged directly without transformation"

        return "Transformations needed: " + ", ".join(parts)


# Backward compatibility alias
InterferencePredictor = MergeAnalyzer


def quick_merge_analysis(
    source_activations: dict[str, "Array"],
    target_activations: dict[str, "Array"],
) -> GlobalMergeAnalysisReport:
    """Quick interface for merge analysis.

    Args:
        source_activations: Dict mapping concept_id to source model activations
        target_activations: Dict mapping concept_id to target model activations

    Returns:
        GlobalMergeAnalysisReport describing transformations needed
    """
    # Find common concepts
    common_concepts = set(source_activations.keys()) & set(target_activations.keys())

    if not common_concepts:
        logger.warning("No common concepts between source and target")
        return GlobalMergeAnalysisReport(
            pair_results={},
            total_pairs=0,
            transformation_counts={t: 0 for t in TransformationType},
            mean_overlap=0.0,
            mean_curvature_divergence=0.0,
            mean_alignment=1.0,
            transformation_summary="No common concepts to analyze",
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

    # Analyze pairs
    analyzer = MergeAnalyzer()
    pair_results = {}
    for concept_id in common_concepts:
        source_key = f"source:{concept_id}"
        target_key = f"target:{concept_id}"
        result = analyzer.analyze(source_volumes[source_key], target_volumes[target_key])
        pair_results[(source_key, target_key)] = result

    # Aggregate
    transformation_counts = {t: 0 for t in TransformationType}
    for result in pair_results.values():
        for t in result.transformations:
            transformation_counts[t] += 1

    if pair_results:
        mean_overlap = sum(r.overlap_score for r in pair_results.values()) / len(pair_results)
        mean_curvature = sum(r.curvature_divergence for r in pair_results.values()) / len(pair_results)
        mean_alignment = sum(r.alignment_score for r in pair_results.values()) / len(pair_results)
    else:
        mean_overlap = 0.0
        mean_curvature = 0.0
        mean_alignment = 1.0

    summary = analyzer._generate_transformation_summary(transformation_counts, len(pair_results))

    return GlobalMergeAnalysisReport(
        pair_results=pair_results,
        total_pairs=len(pair_results),
        transformation_counts=transformation_counts,
        mean_overlap=mean_overlap,
        mean_curvature_divergence=mean_curvature,
        mean_alignment=mean_alignment,
        transformation_summary=summary,
    )


# Backward compatibility alias
quick_interference_check = quick_merge_analysis
