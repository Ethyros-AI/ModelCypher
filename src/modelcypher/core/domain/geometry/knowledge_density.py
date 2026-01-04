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
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.vector_math import geodesic_cosine_matrix
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityResult,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.probe_calibration import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class LayerDensityProfile:
    """Density profile for a single layer."""

    layer: int
    concept_densities: list[ConceptDensity]

    # Aggregate statistics
    mean_density: float
    median_density: float


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
        b = self._backend

        # Get dimensionality results from existing analyzer
        dim_report = self._dim_analyzer.analyze(
            probes=probes,
            activation_provider=activation_provider,
            layer=layer,
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

            # Get activations for this concept
            texts = self._get_support_texts(probes, result.probe_id)
            if texts:
                try:
                    activations = activation_provider.get_activations(texts, layer)
                    if activations:
                        act_array = b.stack([b.array(a) for a in activations], axis=0)
                        b.eval(act_array)

                        variance = self._compute_activation_variance(act_array)
                        tightness = self._compute_cluster_tightness(act_array)
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
                )
            )

        # Aggregate statistics
        if concept_densities:
            densities = [c.density_score for c in concept_densities]
            mean_density = sum(densities) / len(densities)
            median_density = median(densities)
        else:
            mean_density = 0.0
            median_density = 0.0

        return LayerDensityProfile(
            layer=layer,
            concept_densities=concept_densities,
            mean_density=mean_density,
            median_density=median_density,
        )

    def analyze_model(
        self,
        probes: list[AtlasProbeProtocol],
        activation_provider: "ActivationProvider",
        layers: list[int],
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
            profile = self.analyze_layer(probes, activation_provider, layer)
            layer_profiles[layer] = profile

        # Aggregate across layers
        all_concepts: list[ConceptDensity] = []
        for profile in layer_profiles.values():
            all_concepts.extend(profile.concept_densities)

        # Per-domain aggregates
        domain_densities: dict[str, list[float]] = defaultdict(list)
        for c in all_concepts:
            domain_densities[c.domain].append(c.density_score)

        domain_means = {
            domain: sum(scores) / len(scores)
            for domain, scores in domain_densities.items()
        }

        # Overall density
        if all_concepts:
            overall = sum(c.density_score for c in all_concepts) / len(all_concepts)
        else:
            overall = 0.0

        return ModelDensityProfile(
            model_path="",  # Set by caller
            layers=layers,
            layer_profiles=layer_profiles,
            domain_densities=domain_means,
            overall_density=overall,
        )

    def _compute_density_score(self, result: ConceptDimensionalityResult) -> float:
        """Convert intrinsic dimension to density score.

        Lower intrinsic dimension = higher density (more compressed).
        """
        # Use inverse relationship: density ~ 1 / intrinsic_dim
        # Clamp intrinsic dimension to avoid division issues
        dim = max(1.0, result.intrinsic_dimension)
        return 1.0 / dim

    def _compute_activation_variance(self, activations: "Array") -> float:
        """Compute mean variance across activation dimensions."""
        b = self._backend

        # Variance along sample axis (axis=0)
        mean_vec = b.mean(activations, axis=0)
        diff = activations - mean_vec
        var_vec = b.mean(diff * diff, axis=0)

        # Mean variance across dimensions
        mean_var = b.mean(var_vec)
        b.eval(mean_var)

        return float(b.to_scalar(mean_var))

    def _compute_cluster_tightness(self, activations: "Array") -> float:
        """Compute mean pairwise similarity within concept cluster."""
        b = self._backend

        n_samples = int(activations.shape[0])
        if n_samples < 2:
            return 0.0

        # Pairwise geodesic cosine similarities
        sim_matrix = geodesic_cosine_matrix(activations, b)

        # Mean similarity to all other samples (exclude self)
        sim_sum = b.sum(sim_matrix, axis=1) - b.diag(sim_matrix)
        denom = float(n_samples - 1)
        mean_per_sample = sim_sum / denom
        mean_tightness = b.mean(mean_per_sample)
        b.eval(mean_tightness)
        return float(b.to_scalar(mean_tightness))

    def _get_support_texts(
        self,
        probes: list[AtlasProbeProtocol],
        probe_id: str,
    ) -> list[str]:
        """Get support texts for a probe."""
        for probe in probes:
            if probe.probe_id == probe_id:
                texts = list(probe.support_texts or [])
                if probe.name:
                    texts.insert(0, probe.name)
                if probe.description:
                    texts.insert(0, probe.description)
                return texts
        return []


__all__ = [
    "ConceptDensity",
    "LayerDensityProfile",
    "ModelDensityProfile",
    "KnowledgeDensityAnalyzer",
]
