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

"""Concept Volume Analysis Service.

Provides multi-concept activation probing using Riemannian density estimation.
For each named concept (defined by a set of prompts), collects hidden activations,
estimates a ConceptVolume, and computes pairwise relations (overlap, distance,
curvature divergence).

Example:
    from modelcypher.cli.composition import get_concept_volume_service

    service = get_concept_volume_service()
    result = service.analyze_concept_volumes(
        model, tokenizer,
        concepts={"math": ["2+2=4", "3*3=9"], "geo": ["Paris is in France"]},
        layer_idx=12,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.riemannian_density import (
    ConceptVolume,
    ConceptVolumeRelation,
    RiemannianDensityEstimator,
    batch_estimate_volumes,
    compute_pairwise_relations,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptVolumeStats:
    """Summary statistics for a single concept volume."""

    concept_id: str
    num_samples: int
    influence_type: str
    geodesic_radius: float
    effective_radius: float
    volume: float
    dimension: int
    mean_sectional_curvature: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "num_samples": self.num_samples,
            "influence_type": self.influence_type,
            "geodesic_radius": self.geodesic_radius,
            "effective_radius": self.effective_radius,
            "volume": self.volume,
            "dimension": self.dimension,
            "mean_sectional_curvature": self.mean_sectional_curvature,
        }


@dataclass(frozen=True)
class PairwiseRelationStats:
    """Summary statistics for a pairwise concept relation."""

    concept_a: str
    concept_b: str
    overlap_coefficient: float
    jaccard_index: float
    bhattacharyya_coefficient: float
    centroid_distance: float
    geodesic_centroid_distance: float
    mahalanobis_distance_ab: float
    mahalanobis_distance_ba: float
    curvature_divergence: float
    subspace_alignment: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "overlap_coefficient": self.overlap_coefficient,
            "jaccard_index": self.jaccard_index,
            "bhattacharyya_coefficient": self.bhattacharyya_coefficient,
            "centroid_distance": self.centroid_distance,
            "geodesic_centroid_distance": self.geodesic_centroid_distance,
            "mahalanobis_distance_ab": self.mahalanobis_distance_ab,
            "mahalanobis_distance_ba": self.mahalanobis_distance_ba,
            "curvature_divergence": self.curvature_divergence,
            "subspace_alignment": self.subspace_alignment,
        }


@dataclass(frozen=True)
class ConceptVolumeAnalysisResult:
    """Complete result of concept volume analysis."""

    layer_idx: int
    concept_stats: list[ConceptVolumeStats]
    pairwise_relations: list[PairwiseRelationStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "concept_stats": [s.to_dict() for s in self.concept_stats],
            "pairwise_relations": [r.to_dict() for r in self.pairwise_relations],
        }


def _volume_to_stats(vol: ConceptVolume) -> ConceptVolumeStats:
    """Extract summary stats from a ConceptVolume."""
    curvature = None
    if vol.local_curvature is not None:
        curvature = vol.local_curvature.mean_sectional
    return ConceptVolumeStats(
        concept_id=vol.concept_id,
        num_samples=vol.num_samples,
        influence_type=vol.influence_type.value,
        geodesic_radius=vol.geodesic_radius,
        effective_radius=vol.effective_radius,
        volume=vol.volume,
        dimension=vol.dimension,
        mean_sectional_curvature=curvature,
    )


def _relation_to_stats(rel: ConceptVolumeRelation) -> PairwiseRelationStats:
    """Extract summary stats from a ConceptVolumeRelation."""
    return PairwiseRelationStats(
        concept_a=rel.volume_a.concept_id,
        concept_b=rel.volume_b.concept_id,
        overlap_coefficient=rel.overlap_coefficient,
        jaccard_index=rel.jaccard_index,
        bhattacharyya_coefficient=rel.bhattacharyya_coefficient,
        centroid_distance=rel.centroid_distance,
        geodesic_centroid_distance=rel.geodesic_centroid_distance,
        mahalanobis_distance_ab=rel.mahalanobis_distance_ab,
        mahalanobis_distance_ba=rel.mahalanobis_distance_ba,
        curvature_divergence=rel.curvature_divergence,
        subspace_alignment=rel.subspace_alignment,
    )


class ConceptVolumeService:
    """Service for multi-concept activation probing via Riemannian density estimation.

    Collects hidden activations for named concept groups, estimates ConceptVolumes,
    and computes pairwise geometric relations.
    """

    def __init__(
        self,
        activation_provider: "ActivationProvider",
        backend: "Backend",
    ) -> None:
        self.activation_provider = activation_provider
        self.backend = backend

    def analyze_concept_volumes(
        self,
        model: Any,
        tokenizer: Any,
        concepts: dict[str, list[str]],
        layer_idx: int,
    ) -> ConceptVolumeAnalysisResult:
        """Analyze concept volumes at a specific layer.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            concepts: Mapping of concept_id -> list of prompt strings.
            layer_idx: Layer index to collect activations from.

        Returns:
            ConceptVolumeAnalysisResult with per-concept stats and pairwise relations.
        """
        backend = self.backend
        provider = self.activation_provider

        # Collect activations per concept
        concept_activations: dict[str, "Array"] = {}
        for concept_id, prompts in concepts.items():
            acts: list["Array"] = []
            for prompt in prompts:
                hidden = provider.collect_hidden_activations(model, tokenizer, prompt)
                if layer_idx in hidden:
                    acts.append(hidden[layer_idx])
                else:
                    logger.warning(
                        "Layer %d not found in activations for concept %s, prompt: %s",
                        layer_idx, concept_id, prompt[:50],
                    )
            if not acts:
                logger.warning("No activations collected for concept %s", concept_id)
                continue
            stacked = backend.stack(acts, axis=0)
            backend.eval(stacked)
            concept_activations[concept_id] = stacked

        if not concept_activations:
            return ConceptVolumeAnalysisResult(
                layer_idx=layer_idx,
                concept_stats=[],
                pairwise_relations=[],
            )

        # Estimate volumes
        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concept_activations)

        # Compute pairwise relations
        relations = compute_pairwise_relations(estimator, volumes)

        # Build result
        concept_stats = [_volume_to_stats(volumes[cid]) for cid in volumes]
        pairwise_stats = [_relation_to_stats(rel) for rel in relations.values()]

        return ConceptVolumeAnalysisResult(
            layer_idx=layer_idx,
            concept_stats=concept_stats,
            pairwise_relations=pairwise_stats,
        )
