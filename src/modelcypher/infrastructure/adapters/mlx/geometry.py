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


from typing import Any

import mlx.core as mx

from modelcypher.core.domain.geometry.generalized_procrustes import Config as GPAConfig
from modelcypher.core.domain.geometry.generalized_procrustes import FrechetMeanConfig
from modelcypher.core.domain.geometry.generalized_procrustes import GeneralizedProcrustes
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)
from modelcypher.core.domain.geometry.manifold_clusterer import (
    ManifoldClusterer as MLXManifoldClusterer,
)
from modelcypher.core.domain.geometry.model_fingerprints_projection import (
    ModelFingerprintsProjection,
)
from modelcypher.core.domain.geometry.permutation_aligner import Config as MLXAlignConfig
from modelcypher.core.domain.geometry.permutation_aligner import (
    PermutationAligner as MLXPermutationAligner,
)
from modelcypher.core.domain.geometry.compositional_probes import CompositionalProbes
from modelcypher.core.domain.geometry.types import (
    AlignmentConfig,
    BatchMergerResult,
    ClusteringConfiguration,
    ClusteringResult,
    CompositionAnalysis,
    CompositionProbe,
    ConsistencyResult,
    IntrinsicDimensionResult,
    ManifoldPoint,
    MergerConfig,
    MergerResult,
    ModelFingerprints,
    PermutationAlignmentResult,
    ProcrustesConfig,
    ProcrustesResult,
    ProjectionMethod,
    ProjectionResult,
    RebasinResult,
    RefusalConfig,
    RefusalDirection,
    RefusalDistanceMetrics,
)
from modelcypher.infrastructure.adapters.mlx.merger import TransportGuidedMerger
from modelcypher.ports.async_geometry import GeometryPort


class MLXGeometryAdapter(GeometryPort):
    async def align_permutations(
        self, source_weight: Any, target_weight: Any, anchors: Any | None, config: AlignmentConfig
    ) -> PermutationAlignmentResult:
        # Convert config
        mlx_conf = MLXAlignConfig(
            min_match_threshold=config.min_match_threshold,
            use_anchor_grounding=config.use_anchor_grounding,
            top_k=config.top_k,
        )

        # Ensure mx arrays?
        # The core implementation expects mx arrays.
        # The inputs here are Any, but usually vectors.
        # We can try/except wrap.

        res = MLXPermutationAligner.align(source_weight, target_weight, anchors, mlx_conf)
        # res is Domain AlignmentResult
        return PermutationAlignmentResult(
            permutation=res.permutation,
            signs=res.signs,
            match_quality=res.match_quality,
            match_confidences=res.match_confidences,
            sign_flip_count=res.sign_flip_count,
            is_sparse_permutation=res.is_sparse_permutation,
            assignment_indices=res.assignment_indices,
        )

    async def align_via_anchor_projection(
        self, source_weight: Any, target_weight: Any, anchors: Any, config: AlignmentConfig
    ) -> PermutationAlignmentResult:
        mlx_conf = MLXAlignConfig(
            min_match_threshold=config.min_match_threshold,
            use_anchor_grounding=config.use_anchor_grounding,
            top_k=config.top_k,
        )

        res = MLXPermutationAligner.align_via_anchor_projection(
            source_weight, target_weight, anchors, mlx_conf
        )

        return PermutationAlignmentResult(
            permutation=res.permutation,
            signs=res.signs,
            match_quality=res.match_quality,
            match_confidences=res.match_confidences,
            sign_flip_count=res.sign_flip_count,
            is_sparse_permutation=res.is_sparse_permutation,
            assignment_indices=res.assignment_indices,
        )

    async def rebasin_mlp(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        anchors: Any,
        config: AlignmentConfig,
    ) -> RebasinResult:
        mlx_conf = MLXAlignConfig(
            min_match_threshold=config.min_match_threshold,
            use_anchor_grounding=config.use_anchor_grounding,
            top_k=config.top_k,
        )

        aligned, quality, count = MLXPermutationAligner.rebasin_mlp_only(
            source_weights, target_weights, anchors, mlx_conf
        )

        return RebasinResult(aligned_weights=aligned, quality=quality, sign_flip_count=count)

    async def compute_refusal_direction(
        self,
        harmful_activations: Any,
        harmless_activations: Any,
        config: RefusalConfig,
        metadata: dict[str, Any],
    ) -> RefusalDirection | None:
        # Helper to ensure mlx
        try:
            h_acts = mx.array(harmful_activations)
            hl_acts = mx.array(harmless_activations)
        except Exception:
            return None

        if h_acts.shape[0] == 0 or hl_acts.shape[0] == 0:
            return None

        # Mean diff
        h_mean = mx.mean(h_acts, axis=0)
        hl_mean = mx.mean(hl_acts, axis=0)

        diff = h_mean - hl_mean
        norm = mx.linalg.norm(diff).item()

        if norm < config.activation_difference_threshold:
            return None

        direction = diff / norm if config.normalize_direction else diff

        # Explained Variance (Simplified)
        # For now just stubbing variance calculation to match swift logic roughly or keep it simple.
        # Swift logic: ratio of between-class var to total var.

        # Project
        h_proj = h_acts @ direction
        hl_proj = hl_acts @ direction

        mean_h = mx.mean(h_proj)
        mean_hl = mx.mean(hl_proj)

        between = (mean_h - mean_hl) ** 2

        var_h = mx.var(h_proj)
        var_hl = mx.var(hl_proj)
        within = (var_h + var_hl) / 2  # simplified pooling

        total = between + within
        explained = (between / total).item() if total.item() > 0 else 0.0

        return RefusalDirection(
            direction=direction,
            layer_index=metadata.get("layer_index", 0),
            hidden_size=direction.shape[0],
            strength=norm,
            explained_variance=explained,
            model_id=metadata.get("model_id", "unknown"),
        )

    async def measure_refusal_distance(
        self,
        activation: Any,
        direction: RefusalDirection,
        token_index: int,
        previous_projection: float | None = None,
    ) -> RefusalDistanceMetrics:
        vec = mx.array(activation)
        ref_dir = mx.array(direction.direction)

        # Dot product (projection magnitude)
        proj = mx.tensordot(vec, ref_dir, axes=1).item()

        # Cosine Distance
        # 1 - cos_sim
        vec_norm = mx.linalg.norm(vec).item()
        dir_norm = mx.linalg.norm(ref_dir).item()  # Should be 1 if normalized

        cos_sim = 0.0
        if vec_norm > 0 and dir_norm > 0:
            cos_sim = proj / (vec_norm * dir_norm)

        dist = 1.0 - cos_sim

        # Assessment
        assessment = "neutral"
        if proj > 0.5:
            assessment = "likely"
        elif proj > 0.2:
            assessment = "possible"
        elif proj < -0.2:
            assessment = "unlikely"

        is_approaching = False
        if previous_projection is not None:
            is_approaching = proj > previous_projection
        elif proj > 0:
            is_approaching = True

        return RefusalDistanceMetrics(
            distance_to_refusal=dist,
            projection_magnitude=proj,
            is_approaching=is_approaching,
            layer_index=direction.layer_index,
            token_index=token_index,
            assessment=assessment,
        )

    async def merge_models_transport(
        self,
        source_weights: Any,
        target_weights: Any,
        source_activations: Any,
        target_activations: Any,
        config: MergerConfig,
    ) -> MergerResult | BatchMergerResult:
        return await TransportGuidedMerger.merge_models(
            source_weights, target_weights, source_activations, target_activations, config
        )

    async def cluster_manifold(
        self, points: list[ManifoldPoint], config: ClusteringConfiguration
    ) -> ClusteringResult:
        """Cluster manifold points using DBSCAN-style algorithm.

        Note: The MLX implementation uses Euclidean distance; the metric
        parameter from config is accepted for interface compatibility but
        not applied.
        """
        mlx_config = MLXManifoldClusterer.Configuration(
            epsilon=config.epsilon, min_points=config.min_samples
        )
        clusterer = MLXManifoldClusterer(mlx_config)
        return clusterer.cluster(points)

    async def estimate_intrinsic_dimension(
        self, points: list[Any], method: str = "mle"
    ) -> IntrinsicDimensionResult:
        pts = points
        if not isinstance(points, mx.array):
            try:
                pts = mx.array(points)
            except Exception:
                pass

        estimate = IntrinsicDimension.compute_two_nn(pts)

        return IntrinsicDimensionResult(estimated_dimension=estimate, method=method, details={})

    async def project_fingerprints(
        self,
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: set[int] | None = None,
        seed: int = 42,
    ) -> ProjectionResult:
        return ModelFingerprintsProjection.project_2d(
            fingerprints, method, max_features, layers, seed
        )

    async def align_procrustes(
        self, activations: list[list[list[float]]], config: ProcrustesConfig
    ) -> ProcrustesResult | None:
        # Build Fréchet mean config if enabled
        frechet_config = None
        if config.use_frechet_mean:
            frechet_config = FrechetMeanConfig(enabled=True)

        gpa_config = GPAConfig(
            max_iterations=config.max_iterations,
            convergence_threshold=config.convergence_threshold,
            allow_reflections=config.allow_reflections,
            min_models=config.min_models,
            allow_scaling=config.allow_scaling,
            frechet_mean=frechet_config,
        )

        res = GeneralizedProcrustes.align(activations, gpa_config)
        if not res:
            return None

        return ProcrustesResult(
            consensus=res.consensus,
            rotations=res.rotations,
            scales=res.scales,
            residuals=res.residuals,
            converged=res.converged,
            iterations=res.iterations,
            alignment_error=res.alignment_error,
            per_model_errors=res.per_model_errors,
            consensus_variance_ratio=res.consensus_variance_ratio,
            sample_count=res.sample_count,
            dimension=res.dimension,
            model_count=res.model_count,
        )

    async def analyze_composition(
        self, composition_embedding: Any, component_embeddings: Any, probe: CompositionProbe
    ) -> CompositionAnalysis:
        comp = (
            composition_embedding
            if isinstance(composition_embedding, mx.array)
            else mx.array(composition_embedding)
        )
        comps = (
            component_embeddings
            if isinstance(component_embeddings, mx.array)
            else mx.array(component_embeddings)
        )

        return CompositionalProbes.analyze_composition(comp, comps, probe)

    async def check_consistency(
        self, analyses_a: list[CompositionAnalysis], analyses_b: list[CompositionAnalysis]
    ) -> ConsistencyResult:
        return CompositionalProbes.check_consistency(analyses_a, analyses_b)
