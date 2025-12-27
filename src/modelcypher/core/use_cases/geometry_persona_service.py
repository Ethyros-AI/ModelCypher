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

"""
Geometry Persona and Manifold Service.

Exposes persona vector extraction, drift monitoring, and manifold profiling
as CLI/MCP-consumable operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from modelcypher.core.domain.geometry.manifold_clusterer import (
    ClusteringResult,
    ManifoldClusterer,
)
from modelcypher.core.domain.geometry.manifold_clusterer import (
    Configuration as ClustererConfig,
)
from modelcypher.core.domain.geometry.manifold_dimensionality import (
    IDEstimateSummary,
    ManifoldDimensionality,
)
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
    RegionQueryResult,
)
from modelcypher.core.domain.geometry.persona_vector_monitor import (
    STANDARD_TRAITS,
    PersonaBaseline,
    PersonaPosition,
    PersonaTraitDefinition,
    PersonaVector,
    PersonaVectorMonitor,
    TrainingDriftMetrics,
)
from modelcypher.core.domain.geometry.persona_vector_monitor import (
    Configuration as PersonaConfig,
)


@dataclass(frozen=True)
class TraitInfo:
    """Summary info for a persona trait."""

    id: str
    name: str
    description: str
    positive_prompt_count: int
    negative_prompt_count: int


class GeometryPersonaService:
    """
    Service for persona vector and manifold profile operations.
    """

    def list_traits(self) -> list[TraitInfo]:
        """List all standard persona traits."""
        return [
            TraitInfo(
                id=trait.id,
                name=trait.name,
                description=trait.description,
                positive_prompt_count=len(trait.positive_prompts),
                negative_prompt_count=len(trait.negative_prompts),
            )
            for trait in STANDARD_TRAITS
        ]

    def get_trait(self, trait_id: str) -> PersonaTraitDefinition | None:
        """Get a trait definition by ID."""
        return next((t for t in STANDARD_TRAITS if t.id == trait_id), None)

    def extract_persona_vector(
        self,
        positive_activations: list[list[float]],
        negative_activations: list[list[float]],
        trait_id: str,
        layer_index: int,
        model_id: str,
        normalize: bool = True,
        correlation_threshold: float = 0.5,
    ) -> PersonaVector | None:
        """
        Extract a persona vector for a specific trait.

        Args:
            positive_activations: Activations from trait-positive prompts
            negative_activations: Activations from trait-negative prompts
            trait_id: ID of the trait to extract
            layer_index: Layer these activations come from
            model_id: Model identifier
            normalize: Whether to normalize the direction vector
            correlation_threshold: Minimum correlation for valid extraction

        Returns:
            PersonaVector if extraction succeeds, None otherwise
        """
        trait = self.get_trait(trait_id)
        if trait is None:
            return None

        config = PersonaConfig(
            normalize_vectors=normalize,
            correlation_threshold=correlation_threshold,
        )
        return PersonaVectorMonitor.extract_vector(
            positive_activations=positive_activations,
            negative_activations=negative_activations,
            trait=trait,
            configuration=config,
            layer_index=layer_index,
            model_id=model_id,
        )

    def measure_position(
        self,
        activation: list[float],
        persona_vector: PersonaVector,
        baseline_positions: dict[str, float] | None = None,
        model_id: str = "unknown",
    ) -> PersonaPosition | None:
        """
        Measure position along a persona vector.

        Args:
            activation: Hidden state activation to measure
            persona_vector: Persona vector to project onto
            baseline_positions: Optional baseline positions for delta computation
            model_id: Model identifier for baseline

        Returns:
            PersonaPosition with projection and assessment
        """
        baseline = None
        if baseline_positions:
            baseline = PersonaBaseline(
                model_id=model_id,
                baseline_positions=baseline_positions,
                captured_at=datetime.utcnow(),
                is_pretrained_baseline=True,
            )
        return PersonaVectorMonitor.measure_position(activation, persona_vector, baseline)

    def compute_drift(
        self,
        positions: list[dict],
        step: int,
        drift_threshold: float = 0.2,
    ) -> TrainingDriftMetrics:
        """
        Compute drift metrics from position measurements.

        Args:
            positions: List of position dicts. Accepts both formats:
                - {trait_id, projection, normalized_position, ...} (full format)
                - {trait, position} (simple format - auto-computes normalized_position)
            step: Training step number
            drift_threshold: Threshold for significant drift

        Returns:
            TrainingDriftMetrics with overall drift assessment
        """
        parsed_positions = []
        for p in positions:
            # Support both "trait_id" and "trait" keys
            trait_id = p.get("trait_id") or p.get("trait", "unknown")
            trait_name = p.get("trait_name", trait_id)

            # Support both "projection" and "position" keys
            position = p.get("projection") or p.get("position")
            if position is None:
                position = [0.0]  # Default if neither is provided

            # Compute projection as sum if given a list (scalar projection on unit axis)
            projection = sum(position) if isinstance(position, list) else float(position)

            # Auto-compute normalized_position if not provided
            # normalized_position is a float in [-1, 1] representing position on persona direction
            normalized_position = p.get("normalized_position")
            if normalized_position is None:
                # Use projection magnitude normalized to unit scale
                norm = (sum(x * x for x in position) ** 0.5) if isinstance(position, list) else abs(position)
                normalized_position = projection / (norm + 1e-8) if norm > 0 else 0.0

            parsed_positions.append(
                PersonaPosition(
                    trait_id=trait_id,
                    trait_name=trait_name,
                    projection=projection,
                    normalized_position=normalized_position,
                    delta_from_baseline=p.get("delta_from_baseline"),
                    layer_index=p.get("layer_index", 0),
                )
            )
        return PersonaVectorMonitor.compute_drift_metrics(
            positions=parsed_positions,
            step=step,
            drift_threshold=drift_threshold,
        )

    def cluster_points(
        self,
        points: list[dict],
        epsilon: float = 0.3,
        min_points: int = 5,
        compute_dimension: bool = True,
    ) -> ClusteringResult:
        """
        Cluster manifold points into regions.

        Args:
            points: List of point dicts with feature values
            epsilon: DBSCAN epsilon (distance threshold)
            min_points: Minimum points per cluster
            compute_dimension: Whether to estimate intrinsic dimension

        Returns:
            ClusteringResult with regions and noise points
        """
        config = ClustererConfig(
            epsilon=epsilon,
            min_points=min_points,
            compute_intrinsic_dimension=compute_dimension,
        )
        clusterer = ManifoldClusterer(config)

        manifold_points = [
            ManifoldPoint(
                id=uuid4(),
                mean_entropy=p.get("mean_entropy", 0.0),
                entropy_variance=p.get("entropy_variance", 0.0),
                first_token_entropy=p.get("first_token_entropy", 0.0),
                gate_count=p.get("gate_count", 0),
                mean_gate_confidence=p.get("mean_gate_confidence", 0.0),
                dominant_gate_category=p.get("dominant_gate_category", 0.0),
                entropy_path_correlation=p.get("entropy_path_correlation", 0.0),
                assessment_strength=p.get("assessment_strength", 0.0),
                prompt_hash=p.get("prompt_hash", ""),
            )
            for p in points
        ]

        return clusterer.cluster(manifold_points)

    def estimate_dimension(
        self,
        points: list[list[float]],
        bootstrap_samples: int = 0,
        use_regression: bool = True,
    ) -> IDEstimateSummary:
        """
        Estimate intrinsic dimension of a point cloud.

        Args:
            points: List of feature vectors
            bootstrap_samples: Number of bootstrap resamples (0 = no bootstrap)
            use_regression: Use regression-based estimation

        Returns:
            IDEstimateSummary with dimension estimate and confidence
        """
        return ManifoldDimensionality.estimate_id(
            points=points,
            bootstrap_resamples=bootstrap_samples if bootstrap_samples > 0 else None,
            use_regression=use_regression,
        )

    def query_region(
        self,
        point: dict,
        regions: list[dict],
        epsilon: float = 0.3,
    ) -> RegionQueryResult:
        """
        Find nearest region for a point.

        Args:
            point: Point dict with feature values
            regions: List of region dicts
            epsilon: Distance threshold for confidence calculation

        Returns:
            RegionQueryResult with nearest region and classification
        """
        manifold_point = ManifoldPoint(
            id=uuid4(),
            mean_entropy=point.get("mean_entropy", 0.0),
            entropy_variance=point.get("entropy_variance", 0.0),
            first_token_entropy=point.get("first_token_entropy", 0.0),
            gate_count=point.get("gate_count", 0),
            mean_gate_confidence=point.get("mean_gate_confidence", 0.0),
            dominant_gate_category=point.get("dominant_gate_category", 0.0),
            entropy_path_correlation=point.get("entropy_path_correlation", 0.0),
            assessment_strength=point.get("assessment_strength", 0.0),
            prompt_hash=point.get("prompt_hash", ""),
        )

        manifold_regions = []
        for r in regions:
            centroid_data = r.get("centroid", {})
            centroid = ManifoldPoint(
                id=UUID(r.get("id", str(uuid4()))),
                mean_entropy=centroid_data.get("mean_entropy", 0.0),
                entropy_variance=centroid_data.get("entropy_variance", 0.0),
                first_token_entropy=centroid_data.get("first_token_entropy", 0.0),
                gate_count=centroid_data.get("gate_count", 0),
                mean_gate_confidence=centroid_data.get("mean_gate_confidence", 0.0),
                dominant_gate_category=centroid_data.get("dominant_gate_category", 0.0),
                entropy_path_correlation=centroid_data.get("entropy_path_correlation", 0.0),
                assessment_strength=centroid_data.get("assessment_strength", 0.0),
                prompt_hash=centroid_data.get("prompt_hash", "centroid"),
            )
            region_type = ManifoldRegion.RegionType(r.get("region_type", "transitional"))
            manifold_regions.append(
                ManifoldRegion(
                    id=UUID(r.get("id", str(uuid4()))),
                    region_type=region_type,
                    centroid=centroid,
                    member_count=r.get("member_count", 0),
                    member_ids=r.get("member_ids", []),
                    dominant_gates=r.get("dominant_gates", []),
                    intrinsic_dimension=r.get("intrinsic_dimension"),
                    radius=r.get("radius", 0.0),
                )
            )

        config = ClustererConfig(epsilon=epsilon)
        clusterer = ManifoldClusterer(config)
        return clusterer.find_nearest_region(manifold_point, manifold_regions)

    @staticmethod
    def traits_payload(traits: list[TraitInfo]) -> dict:
        """Convert trait list to CLI/MCP payload."""
        return {
            "traits": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "positivePromptCount": t.positive_prompt_count,
                    "negativePromptCount": t.negative_prompt_count,
                }
                for t in traits
            ],
            "count": len(traits),
        }

    @staticmethod
    def persona_vector_payload(vector: PersonaVector) -> dict:
        """Convert persona vector to CLI/MCP payload."""
        return {
            "id": vector.id,
            "name": vector.name,
            "layerIndex": vector.layer_index,
            "hiddenSize": vector.hidden_size,
            "strength": vector.strength,
            "correlationCoefficient": vector.correlation_coefficient,
            "modelId": vector.model_id,
            "computedAt": vector.computed_at.isoformat(),
            "directionNorm": sum(x * x for x in vector.direction) ** 0.5,
        }

    @staticmethod
    def drift_metrics_payload(metrics: TrainingDriftMetrics) -> dict:
        """Convert drift metrics to CLI/MCP payload."""
        return {
            "step": metrics.step,
            "overallDriftMagnitude": metrics.overall_drift_magnitude,
            "hasSignificantDrift": metrics.has_significant_drift,
            "driftingTraits": metrics.drifting_traits,
            "interpretation": metrics.interpretation,
            "positions": [
                {
                    "traitId": p.trait_id,
                    "traitName": p.trait_name,
                    "projection": p.projection,
                    "normalizedPosition": p.normalized_position,
                    "deltaFromBaseline": p.delta_from_baseline,
                    "layerIndex": p.layer_index,
                }
                for p in metrics.positions
            ],
            "timestamp": metrics.timestamp.isoformat(),
        }

    @staticmethod
    def clustering_payload(result: ClusteringResult) -> dict:
        """Convert clustering result to CLI/MCP payload."""
        return {
            "regionCount": len(result.regions),
            "noisePointCount": len(result.noise_points),
            "newClustersFormed": result.new_clusters_formed,
            "clustersMerged": result.clusters_merged,
            "pointsAssignedToExisting": result.points_assigned_to_existing,
            "regions": [
                {
                    "id": str(r.id),
                    "regionType": r.region_type.value,
                    "memberCount": r.member_count,
                    "intrinsicDimension": r.intrinsic_dimension,
                    "radius": r.radius,
                    "dominantGates": r.dominant_gates,
                }
                for r in result.regions
            ],
        }

    @staticmethod
    def dimension_payload(summary: IDEstimateSummary) -> dict:
        """Convert ID estimate to CLI/MCP payload."""
        return {
            "intrinsicDimension": summary.intrinsic_dimension,
            "ci95Lower": summary.ci95_lower,
            "ci95Upper": summary.ci95_upper,
            "sampleCount": summary.sample_count,
            "usableCount": summary.usable_count,
            "usesRegression": summary.uses_regression,
        }

    @staticmethod
    def region_query_payload(result: RegionQueryResult) -> dict:
        """Convert region query result to CLI/MCP payload."""
        return {
            "nearestRegion": {
                "id": str(result.nearest_region.id),
                "regionType": result.nearest_region.region_type.value,
                "memberCount": result.nearest_region.member_count,
            }
            if result.nearest_region
            else None,
            "distance": result.distance,
            "isWithinRegion": result.is_within_region,
            "suggestedType": result.suggested_type.value,
            "confidence": result.confidence,
        }
