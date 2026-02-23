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

Computes per-concept density metrics (intrinsic dimension, activation variance,
cluster tightness) and aggregates them at layer and model levels.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityResult,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "name": self.name,
            "domain": self.domain,
            "layer": self.layer,
            "intrinsic_dimension": self.intrinsic_dimension,
            "density_score": self.density_score,
            "activation_variance": self.activation_variance,
            "cluster_tightness": self.cluster_tightness,
        }


@dataclass(frozen=True)
class LayerDensityProfile:
    """Density profile for a single layer."""

    layer: int
    concept_densities: list[ConceptDensity]

    # Aggregate statistics
    mean_density: float
    median_density: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "concept_densities": [c.to_dict() for c in self.concept_densities],
            "mean_density": self.mean_density,
            "median_density": self.median_density,
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "layers": list(self.layers),
            "layer_profiles": {
                str(layer): profile.to_dict()
                for layer, profile in self.layer_profiles.items()
            },
            "domain_densities": dict(self.domain_densities),
            "overall_density": self.overall_density,
        }


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
        # Clamp using machine precision (no arbitrary floor).
        b = self._backend
        dim_val = float(result.intrinsic_dimension)
        eps = float(machine_epsilon(b, b.array([dim_val])))
        dim = max(dim_val, eps)
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


@dataclass(frozen=True)
class PointCloudDensityResult:
    """Result of k-NN point cloud density comparison.

    Contains per-point density values for source and target, and the
    density difference that drives knowledge transfer.
    """

    # Per-point densities (one value per activation point)
    source_densities: "Array"  # [n_points] - density at each source point
    target_densities: "Array"  # [n_points] - density at each target point

    # Density difference: source - target (positive = source denser = transfer opportunity)
    density_diff: "Array"  # [n_points]

    # Aggregate statistics
    mean_source_density: float
    mean_target_density: float
    mean_density_diff: float
    positive_diff_count: int  # Points where source is denser
    negative_diff_count: int  # Points where target is denser
    distance_metrics: dict[str, float]

    def to_dict(self, backend: "Backend") -> dict[str, Any]:
        backend.eval(self.source_densities, self.target_densities, self.density_diff)
        return {
            "source_densities": backend.tolist(self.source_densities),
            "target_densities": backend.tolist(self.target_densities),
            "density_diff": backend.tolist(self.density_diff),
            "mean_source_density": self.mean_source_density,
            "mean_target_density": self.mean_target_density,
            "mean_density_diff": self.mean_density_diff,
            "positive_diff_count": self.positive_diff_count,
            "negative_diff_count": self.negative_diff_count,
            "distance_metrics": dict(self.distance_metrics),
        }


def compute_knn_point_cloud_density(
    source_activations: "Array",
    target_activations: "Array",
    k: int | None = None,
    backend: "Backend | None" = None,
) -> PointCloudDensityResult:
    """Compute k-NN based local density for point cloud comparison.

    For each point in the activation manifold:
    1. Find k nearest neighbors
    2. Compute local density = 1 / (mean distance to k neighbors)
    3. Compare densities between source and target

    The density difference tells us where source has denser representation
    (more knowledge) and where target has gaps to fill.

    Args:
        source_activations: [n_points, dim] activations from source model
        target_activations: [n_points, dim] activations from target model
        k: Number of neighbors. If None, derived from connectivity of the k-NN graph.
        backend: Compute backend.

    Returns:
        PointCloudDensityResult with per-point densities and comparison.
    """
    from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon, machine_epsilon, sqrt_scalar
    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
    from modelcypher.core.domain.geometry.riemannian_validation import derive_k_neighbors

    b = backend or get_default_backend()

    source = b.array(source_activations)
    target = b.array(target_activations)
    b.eval(source, target)

    n_source = int(source.shape[0])
    n_target = int(target.shape[0])

    if n_source < 2 or n_target < 2:
        # Degenerate case - return uniform densities
        zeros_src = b.zeros((n_source,))
        zeros_tgt = b.zeros((n_target,))
        b.eval(zeros_src, zeros_tgt)
        return PointCloudDensityResult(
            source_densities=zeros_src,
            target_densities=zeros_tgt,
            density_diff=zeros_src,
            mean_source_density=0.0,
            mean_target_density=0.0,
            mean_density_diff=0.0,
            positive_diff_count=0,
            negative_diff_count=0,
            distance_metrics={},
        )

    # Derive k from intrinsic connectivity if not specified
    if k is None:
        k_source = derive_k_neighbors(source, b)
        k_target = derive_k_neighbors(target, b)
        k = max(k_source, k_target, 1)

    # Ensure k doesn't exceed available points
    k = min(k, n_source - 1, n_target - 1)

    # Distance selection is geometry-driven (no user modes).
    rg = RiemannianGeometry(b)
    eps = float(division_epsilon(b, source))
    precision = sqrt_scalar(machine_epsilon(b, target), b)

    # Get target dimension
    target_dim = int(target.shape[1])

    # =======================================================================
    # DISTANCE SELECTION: ALWAYS GEODESIC
    # =======================================================================
    # Neural network activation manifolds are empirically curved (Bernardi
    # et al., Nature Neuroscience 2025; arXiv:2506.12187 "Characterizing
    # Neural Manifolds"). Geodesic distance via k-NN graph is the correct
    # metric for curved Riemannian manifolds (see AGENTS.md §"Geometry Type
    # Matters"). Euclidean distance is only correct for flat spaces.
    #
    # Previously, this code ran curvature estimation and then set
    # use_geodesic=True regardless of the result. The curvature estimation
    # was removed because: (1) it was O(n × d³), (2) the MLX SVD can crash
    # on ill-conditioned high-dim matrices, and (3) the result never changed
    # the distance selection. Curvature analysis is still available for
    # structural diagnostics via SectionalCurvatureEstimator, but it does
    # not belong in the density hot path.
    # =======================================================================
    use_geodesic = True
    anisotropy_mean = 0.0
    anisotropy_max = 0.0

    distance_metrics = {
        "anisotropy_mean": anisotropy_mean,
        "anisotropy_max": anisotropy_max,
        "precision_floor": precision,
    }

    logger.info(
        "DENSITY GEOMETRY: anisotropy_mean=%.3e, anisotropy_max=%.3e, "
        "precision=%.3e, use_geodesic=%s",
        anisotropy_mean,
        anisotropy_max,
        precision,
        use_geodesic,
    )

    def _distance_matrix(points: "Array") -> "Array":
        if use_geodesic:
            geo_result = rg.geodesic_distances(points, k_neighbors=k)
            return geo_result.distances
        # Chord (Euclidean) distances when curvature is numerically flat.
        return rg._chord_distance_matrix(points, use_cache=False)

    # Compute distance matrices
    source_dist_matrix = _distance_matrix(source)
    target_dist_matrix = _distance_matrix(target)

    # Sort each row to get k nearest distances (exclude self = distance 0)
    # All operations below are lazy until final batch eval
    source_sorted = b.sort(source_dist_matrix, axis=1)
    target_sorted = b.sort(target_dist_matrix, axis=1)

    # Skip first column (self-distance = 0), take next k columns
    source_k_dists = source_sorted[:, 1:k+1]
    target_k_dists = target_sorted[:, 1:k+1]

    source_mean_k_dist = b.mean(source_k_dists, axis=1)
    target_mean_k_dist = b.mean(target_k_dists, axis=1)

    source_densities = 1.0 / (source_mean_k_dist + eps)
    target_densities = 1.0 / (target_mean_k_dist + eps)

    # Normalize densities to [0, 1] range for comparison
    max_src = b.max(source_densities)
    max_tgt = b.max(target_densities)

    # Single eval for all density computations
    b.eval(source_densities, target_densities, max_src, max_tgt)

    # Use global max to ensure same scale
    global_max = max(float(b.to_scalar(max_src)), float(b.to_scalar(max_tgt)), eps)

    source_densities_norm = source_densities / global_max
    target_densities_norm = target_densities / global_max

    # Density difference: positive where source is denser (transfer opportunity)
    n_compare = min(n_source, n_target)
    src_compare = source_densities_norm[:n_compare]
    tgt_compare = target_densities_norm[:n_compare]
    density_diff = src_compare - tgt_compare

    # Compute all statistics lazily
    mean_src_arr = b.mean(source_densities_norm)
    mean_tgt_arr = b.mean(target_densities_norm)
    mean_diff_arr = b.mean(density_diff)

    # Count positive/negative differences
    positive_mask = density_diff > eps
    negative_mask = density_diff < -eps

    # Single batch eval for all downstream computations
    b.eval(
        source_densities_norm, target_densities_norm, density_diff,
        mean_src_arr, mean_tgt_arr, mean_diff_arr,
        positive_mask, negative_mask
    )

    # Extract scalars after batch eval
    mean_src = float(b.to_scalar(mean_src_arr))
    mean_tgt = float(b.to_scalar(mean_tgt_arr))
    mean_diff = float(b.to_scalar(mean_diff_arr))

    from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

    count_dtype = precision_dtype(b, reference=density_diff)
    positive_count = int(b.to_scalar(b.sum(b.astype(positive_mask, count_dtype))))
    negative_count = int(b.to_scalar(b.sum(b.astype(negative_mask, count_dtype))))

    return PointCloudDensityResult(
        source_densities=source_densities_norm,
        target_densities=target_densities_norm,
        density_diff=density_diff,
        mean_source_density=mean_src,
        mean_target_density=mean_tgt,
        mean_density_diff=mean_diff,
        positive_diff_count=positive_count,
        negative_diff_count=negative_count,
        distance_metrics=distance_metrics,
    )


def compute_density_weights(
    source_densities: "Array",
    target_densities: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Convert source/target densities to transfer weights.

    Uses the direct density ratio:
        weight = source / (source + target)

    This is purely data-derived (no heuristics) and yields weights in [0, 1].

    Args:
        source_densities: Per-point source densities.
        target_densities: Per-point target densities.
        backend: Compute backend.

    Returns:
        Per-point transfer weights in [0, 1].
    """
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    b = backend or get_default_backend()
    src = b.array(source_densities)
    tgt = b.array(target_densities)
    b.eval(src, tgt)

    # Handle mismatched lengths by truncating to minimum
    # This can happen in cross-architecture merges where source and target
    # activations have different sample counts
    n_src = int(src.shape[0])
    n_tgt = int(tgt.shape[0])
    if n_src != n_tgt:
        n_compare = min(n_src, n_tgt)
        src = src[:n_compare]
        tgt = tgt[:n_compare]
        b.eval(src, tgt)
        logger.debug(
            "DENSITY: Truncated densities to %d samples (src=%d, tgt=%d)",
            n_compare, n_src, n_tgt
        )

    eps = float(division_epsilon(b, src))
    total = src + tgt
    denom = b.maximum(total, b.full(b.shape(total), eps))
    weights = src / denom
    weights = b.clip(weights, 0.0, 1.0)
    b.eval(weights)

    return weights


__all__ = [
    "ConceptDensity",
    "LayerDensityProfile",
    "ModelDensityProfile",
    "KnowledgeDensityAnalyzer",
    "PointCloudDensityResult",
    "compute_knn_point_cloud_density",
    "compute_density_weights",
]
