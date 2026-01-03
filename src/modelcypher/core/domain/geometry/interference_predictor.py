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

"""Merge analysis using ConceptVolume geometry.

Reports raw geometric measurements used to align models for merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

from .riemannian_density import (
    ConceptVolume,
    ConceptVolumeRelation,
    RiemannianDensityEstimator,
)

logger = logging.getLogger(__name__)


@dataclass
class MergeAnalysisResult:
    """Geometric measurements for a volume pair."""

    # Volumes analyzed
    volume_a_id: str
    volume_b_id: str

    # Raw geometric measurements (for diagnostics, not gating)
    overlap_score: float  # 0=no overlap, 1=complete overlap
    curvature_divergence: float  # 0=identical curvature, 1=maximum divergence
    alignment_score: float  # 0=orthogonal, 1=aligned
    distance_score: float  # 0=identical, 1=far apart


@dataclass
class GlobalMergeAnalysisReport:
    """Aggregate geometric measurements across all concept pairs."""

    # Per-pair results
    pair_results: dict[tuple[str, str], MergeAnalysisResult]

    # Aggregate statistics
    total_pairs: int

    # Average geometric measurements (for diagnostics)
    mean_overlap: float
    mean_curvature_divergence: float
    mean_alignment: float
    mean_distance: float


class MergeAnalyzer:
    """Analyzes concept volumes to report merge geometry."""

    def __init__(self) -> None:
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

        return MergeAnalysisResult(
            volume_a_id=volume_a.concept_id,
            volume_b_id=volume_b.concept_id,
            overlap_score=overlap_score,
            curvature_divergence=curvature_divergence,
            alignment_score=alignment_score,
            distance_score=distance_score,
        )

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
            mean_distance = sum(r.distance_score for r in pair_results.values()) / len(
                pair_results
            )
        else:
            mean_overlap = 0.0
            mean_curvature = 0.0
            mean_alignment = 1.0
            mean_distance = 0.0

        return GlobalMergeAnalysisReport(
            pair_results=pair_results,
            total_pairs=len(pair_results),
            mean_overlap=mean_overlap,
            mean_curvature_divergence=mean_curvature,
            mean_alignment=mean_alignment,
            mean_distance=mean_distance,
        )

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
        """Compute subspace alignment score (higher indicates larger alignment magnitude)."""
        return relation.subspace_alignment

    def _compute_distance_score(self, relation: ConceptVolumeRelation) -> float:
        """Compute normalized distance score (higher = farther apart)."""
        r_a = relation.volume_a.effective_radius
        r_b = relation.volume_b.effective_radius
        sum_radius = r_a + r_b

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([sum_radius]))
        if sum_radius < eps:
            return 0.0

        # Normalize by sum of radii (touching spheres = 1.0)
        normalized_dist = relation.geodesic_centroid_distance / sum_radius
        return min(normalized_dist, 1.0)

def quick_merge_analysis(
    source_activations: dict[str, "Array"],
    target_activations: dict[str, "Array"],
) -> GlobalMergeAnalysisReport:
    """Quick interface for merge analysis.

    Args:
        source_activations: Dict mapping concept_id to source model activations
        target_activations: Dict mapping concept_id to target model activations

    Returns:
        GlobalMergeAnalysisReport with raw geometric measurements
    """
    # Find common concepts
    common_concepts = set(source_activations.keys()) & set(target_activations.keys())

    if not common_concepts:
        logger.warning("No common concepts between source and target")
        return GlobalMergeAnalysisReport(
            pair_results={},
            total_pairs=0,
            mean_overlap=0.0,
            mean_curvature_divergence=0.0,
            mean_alignment=1.0,
            mean_distance=0.0,
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

    if pair_results:
        mean_overlap = sum(r.overlap_score for r in pair_results.values()) / len(pair_results)
        mean_curvature = sum(r.curvature_divergence for r in pair_results.values()) / len(pair_results)
        mean_alignment = sum(r.alignment_score for r in pair_results.values()) / len(pair_results)
        mean_distance = sum(r.distance_score for r in pair_results.values()) / len(pair_results)
    else:
        mean_overlap = 0.0
        mean_curvature = 0.0
        mean_alignment = 1.0
        mean_distance = 0.0

    return GlobalMergeAnalysisReport(
        pair_results=pair_results,
        total_pairs=len(pair_results),
        mean_overlap=mean_overlap,
        mean_curvature_divergence=mean_curvature,
        mean_alignment=mean_alignment,
        mean_distance=mean_distance,
    )
