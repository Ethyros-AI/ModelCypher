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

"""Knowledge density estimation for model merging.

Measures how "densely" a model represents each concept. Dense concepts are
well-learned and should not be modified during merge. Sparse concepts are
gaps in knowledge where grafting from another model adds value.

Key insight: Merge is not "blend two models" but "fill gaps in target with
source knowledge." This module identifies where those gaps are.

Density Metrics:
    - Intrinsic dimension: Lower = more compressed = denser representation
    - Activation variance: Lower = more stable = denser representation
    - Cluster tightness: Higher = concepts cluster tightly = denser

The null space of sparse concepts is larger - the model hasn't filled
that capacity yet. Dense concepts have "used up" their null space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityConfig,
    ConceptDimensionalityResult,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.probe_calibration import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeDensityConfig:
    """Configuration for knowledge density estimation.

    All thresholds are derived from the data, not hardcoded.
    """

    # Intrinsic dimension config
    dim_config: ConceptDimensionalityConfig = field(
        default_factory=ConceptDimensionalityConfig
    )

    # Whether to compute activation variance
    compute_variance: bool = True

    # Whether to compute cluster tightness
    compute_clustering: bool = True

    # Number of neighbors for clustering metric
    k_neighbors: int = 10


@dataclass(frozen=True)
class ConceptDensity:
    """Density metrics for a single concept."""

    probe_id: str
    name: str
    domain: str
    layer: int

    # Intrinsic dimension (lower = denser)
    intrinsic_dimension: float

    # Normalized density score (higher = denser, 0-1 range)
    density_score: float

    # Activation variance (lower = more stable)
    activation_variance: float | None = None

    # Mean pairwise similarity within concept (higher = tighter cluster)
    cluster_tightness: float | None = None

    # Raw dimension class
    dimension_class: str = "unknown"


@dataclass(frozen=True)
class LayerDensityProfile:
    """Density profile for a single layer."""

    layer: int
    concept_densities: list[ConceptDensity]

    # Aggregate statistics
    mean_density: float
    median_density: float
    sparse_concept_count: int
    dense_concept_count: int

    # Threshold used to classify sparse vs dense (derived from data)
    density_threshold: float


@dataclass(frozen=True)
class ModelDensityProfile:
    """Complete density profile for a model."""

    model_path: str
    layers: list[int]
    layer_profiles: dict[int, LayerDensityProfile]

    # Per-domain aggregates
    domain_densities: dict[str, float]

    # Overall model density
    overall_density: float

    # Sparse concepts (graft opportunities)
    sparse_concepts: list[ConceptDensity]

    # Dense concepts (do not touch)
    dense_concepts: list[ConceptDensity]


class KnowledgeDensityAnalyzer:
    """Analyze knowledge density across concepts in a model.

    Uses intrinsic dimension as the primary density signal:
    - Low intrinsic dimension = model has compressed the concept efficiently
    - High intrinsic dimension = model representation is sparse/incomplete

    Combined with:
    - Activation variance (stability)
    - Cluster tightness (coherence)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._dim_analyzer = ConceptDimensionalityAnalyzer(backend=self._backend)
        self._id_estimator = IntrinsicDimension(backend=self._backend)

    def analyze_layer(
        self,
        probes: list["AtlasProbeProtocol"],
        activation_provider: "ActivationProvider",
        layer: int,
        config: KnowledgeDensityConfig | None = None,
    ) -> LayerDensityProfile:
        """Analyze knowledge density for all concepts at a layer.

        Args:
            probes: Atlas probes to analyze.
            activation_provider: Provider for activation extraction.
            layer: Layer index to analyze.
            config: Analysis configuration.

        Returns:
            LayerDensityProfile with per-concept densities and aggregates.
        """
        resolved = config or KnowledgeDensityConfig()
        b = self._backend

        # Get dimensionality results from existing analyzer
        dim_report = self._dim_analyzer.analyze(
            probes=probes,
            activation_provider=activation_provider,
            layer=layer,
            config=resolved.dim_config,
        )

        concept_densities: list[ConceptDensity] = []

        # Process each analyzed concept
        for result in dim_report.results:
            # Convert intrinsic dimension to density score
            # Lower dimension = higher density
            # Normalize by ambient dimension (hidden_dim)
            density_score = self._compute_density_score(result)

            variance = None
            tightness = None

            if resolved.compute_variance or resolved.compute_clustering:
                # Get activations for this concept
                texts = self._get_support_texts(probes, result.probe_id, resolved)
                if texts:
                    try:
                        activations = activation_provider.get_activations(texts, layer)
                        if activations:
                            act_array = b.stack([b.array(a) for a in activations], axis=0)
                            b.eval(act_array)

                            if resolved.compute_variance:
                                variance = self._compute_activation_variance(act_array)

                            if resolved.compute_clustering:
                                tightness = self._compute_cluster_tightness(
                                    act_array, resolved.k_neighbors
                                )
                    except Exception as e:
                        logger.debug(
                            "Failed to compute extended metrics for %s: %s",
                            result.probe_id,
                            e,
                        )

            concept_densities.append(
                ConceptDensity(
                    probe_id=result.probe_id,
                    name=result.name,
                    domain=result.domain,
                    layer=layer,
                    intrinsic_dimension=result.intrinsic_dimension,
                    density_score=density_score,
                    activation_variance=variance,
                    cluster_tightness=tightness,
                    dimension_class=result.dimension_class,
                )
            )

        # Compute threshold from data distribution
        density_threshold = self._compute_density_threshold(concept_densities)

        # Classify sparse vs dense
        sparse = [c for c in concept_densities if c.density_score < density_threshold]
        dense = [c for c in concept_densities if c.density_score >= density_threshold]

        # Aggregate statistics
        if concept_densities:
            densities = [c.density_score for c in concept_densities]
            mean_density = sum(densities) / len(densities)
            sorted_densities = sorted(densities)
            n = len(sorted_densities)
            median_density = (
                sorted_densities[n // 2]
                if n % 2 == 1
                else (sorted_densities[n // 2 - 1] + sorted_densities[n // 2]) / 2
            )
        else:
            mean_density = 0.0
            median_density = 0.0

        return LayerDensityProfile(
            layer=layer,
            concept_densities=concept_densities,
            mean_density=mean_density,
            median_density=median_density,
            sparse_concept_count=len(sparse),
            dense_concept_count=len(dense),
            density_threshold=density_threshold,
        )

    def analyze_model(
        self,
        probes: list["AtlasProbe"],
        activation_provider: "ActivationProvider",
        layers: list[int],
        config: KnowledgeDensityConfig | None = None,
    ) -> ModelDensityProfile:
        """Analyze knowledge density across all specified layers.

        Args:
            probes: Atlas probes to analyze.
            activation_provider: Provider for activation extraction.
            layers: Layer indices to analyze.
            config: Analysis configuration.

        Returns:
            ModelDensityProfile with complete density analysis.
        """
        layer_profiles: dict[int, LayerDensityProfile] = {}

        for layer in layers:
            logger.info("Analyzing layer %d...", layer)
            profile = self.analyze_layer(probes, activation_provider, layer, config)
            layer_profiles[layer] = profile

        # Aggregate across layers
        all_concepts: list[ConceptDensity] = []
        for profile in layer_profiles.values():
            all_concepts.extend(profile.concept_densities)

        # Per-domain aggregates
        domain_densities: dict[str, list[float]] = {}
        for c in all_concepts:
            if c.domain not in domain_densities:
                domain_densities[c.domain] = []
            domain_densities[c.domain].append(c.density_score)

        domain_means = {
            domain: sum(scores) / len(scores)
            for domain, scores in domain_densities.items()
            if scores
        }

        # Overall density
        if all_concepts:
            overall = sum(c.density_score for c in all_concepts) / len(all_concepts)
        else:
            overall = 0.0

        # Global threshold for sparse/dense classification
        global_threshold = self._compute_density_threshold(all_concepts)
        sparse = [c for c in all_concepts if c.density_score < global_threshold]
        dense = [c for c in all_concepts if c.density_score >= global_threshold]

        return ModelDensityProfile(
            model_path="",  # Set by caller
            layers=layers,
            layer_profiles=layer_profiles,
            domain_densities=domain_means,
            overall_density=overall,
            sparse_concepts=sparse,
            dense_concepts=dense,
        )

    def _compute_density_score(self, result: ConceptDimensionalityResult) -> float:
        """Convert intrinsic dimension to density score.

        Lower intrinsic dimension = higher density (more compressed).
        Score is normalized to [0, 1] range.
        """
        # Use inverse relationship: density ~ 1 / intrinsic_dim
        # Clamp intrinsic dimension to avoid division issues
        dim = max(1.0, result.intrinsic_dimension)

        # Normalize assuming reasonable intrinsic dim range [1, 100]
        # Score of 1.0 at dim=1, score approaching 0 as dim grows
        score = 1.0 / (1.0 + math.log(dim))

        return min(1.0, max(0.0, score))

    def _compute_activation_variance(self, activations: "Array") -> float:
        """Compute mean variance across activation dimensions."""
        b = self._backend

        # Variance along sample axis (axis=0)
        mean_vec = b.mean(activations, axis=0)
        b.eval(mean_vec)
        diff = activations - mean_vec
        var_vec = b.mean(diff * diff, axis=0)
        b.eval(var_vec)

        # Mean variance across dimensions
        mean_var = b.mean(var_vec)
        b.eval(mean_var)

        return float(b.to_numpy(mean_var).item())

    def _compute_cluster_tightness(self, activations: "Array", k: int) -> float:
        """Compute mean k-NN similarity within concept cluster."""
        b = self._backend

        n_samples = int(activations.shape[0])
        if n_samples < k + 1:
            # Not enough samples for k-NN
            return 0.0

        # Compute pairwise cosine similarities
        norms = b.norm(activations, axis=1, keepdims=True)
        b.eval(norms)
        eps = 1e-8
        normalized = activations / b.maximum(norms, b.full(norms.shape, eps))
        b.eval(normalized)

        # Similarity matrix
        sim_matrix = b.matmul(normalized, b.transpose(normalized))
        b.eval(sim_matrix)

        # For each sample, get mean of top-k similarities (excluding self)
        sim_np = b.to_numpy(sim_matrix)

        tightness_scores = []
        for i in range(n_samples):
            sims = sim_np[i].copy()
            sims[i] = -float("inf")  # Exclude self
            top_k = sorted(sims, reverse=True)[:k]
            if top_k:
                tightness_scores.append(sum(top_k) / len(top_k))

        if tightness_scores:
            return sum(tightness_scores) / len(tightness_scores)
        return 0.0

    def _compute_density_threshold(
        self, concepts: list[ConceptDensity]
    ) -> float:
        """Compute density threshold from data distribution.

        Uses median as the natural boundary between sparse and dense.
        No arbitrary thresholds - the data tells us where the split is.
        """
        if not concepts:
            return 0.5

        scores = sorted(c.density_score for c in concepts)
        n = len(scores)

        # Use median as threshold
        if n % 2 == 1:
            return scores[n // 2]
        else:
            return (scores[n // 2 - 1] + scores[n // 2]) / 2

    def _get_support_texts(
        self,
        probes: list["AtlasProbe"],
        probe_id: str,
        config: KnowledgeDensityConfig,
    ) -> list[str]:
        """Get support texts for a probe."""
        for probe in probes:
            if probe.probe_id == probe_id:
                texts = list(probe.support_texts or [])
                if config.dim_config.include_name_description:
                    if probe.name:
                        texts.insert(0, probe.name)
                    if probe.description:
                        texts.insert(0, probe.description)
                return texts[: config.dim_config.max_total_texts]
        return []


__all__ = [
    "KnowledgeDensityConfig",
    "ConceptDensity",
    "LayerDensityProfile",
    "ModelDensityProfile",
    "KnowledgeDensityAnalyzer",
]
