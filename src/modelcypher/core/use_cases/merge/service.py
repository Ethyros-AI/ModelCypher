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

"""End-to-end merge pipeline service.

Orchestrates: Pre-merge analysis → Execute merge → Post-merge validation

All stages return raw geometric measurements. No interpretation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.core.use_cases.merge.models import UnifiedMergeResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreMergeAnalysis:
    """Pre-merge geometric analysis results.

    All values are raw measurements. No interpretation strings.
    """

    source_model: str
    target_model: str
    timestamp: str

    # Domain-level analysis
    domains_analyzed: list[str]
    domain_results: dict[str, dict[str, Any]]

    # Global metrics
    mean_overlap: float
    mean_alignment: float
    mean_curvature_divergence: float
    mean_distance: float


@dataclass(frozen=True)
class PostMergeValidation:
    """Post-merge validation results.

    All values are raw measurements from the merged model.
    """

    merged_model: str
    timestamp: str

    # Geometry from merge result
    geometry_metrics: dict[str, Any]

    # Transplant details
    layers_transplanted: int
    weights_transplanted: int
    mean_preserved_fraction: float
    mean_cka_after: float


@dataclass(frozen=True)
class PipelineResult:
    """Complete pipeline result.

    Contains raw measurements from all stages. No recommendations.
    """

    pipeline_id: str
    timestamp: str

    # Source/target
    source_model: str
    target_model: str
    output_dir: str

    # Stage results
    pre_merge: PreMergeAnalysis
    merge_result: dict[str, Any]
    post_merge: PostMergeValidation

    # Timing
    pre_merge_duration_s: float = 0.0
    merge_duration_s: float = 0.0
    validation_duration_s: float = 0.0


class MergePipelineService:
    """Orchestrates the full merge pipeline.

    Pipeline stages:
    1. Pre-merge analysis: Interference prediction, entropy profiling
    2. Execute merge: Unified geometric merge
    3. Post-merge validation: Extract geometry metrics
    """

    def __init__(
        self,
    ):
        """Initialize the pipeline service.
        """

    def run(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
        transplant_domains: list[str],
        *,
        skip_pre_analysis: bool = False,
        use_cpu_weights: bool = False,
    ) -> PipelineResult:
        """Run the complete merge pipeline.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            output_dir: Output directory for merged model
            transplant_domains: Domains to transplant (e.g., ["mathematical", "logical"])
            skip_pre_analysis: Skip pre-merge interference analysis
            use_cpu_weights: Load weights on CPU to reduce GPU memory usage
        Returns:
            PipelineResult with all stage results
        """
        import time

        pipeline_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        logger.info("Starting merge pipeline %s", pipeline_id)

        # Stage 1: Pre-merge analysis
        pre_start = time.time()
        if skip_pre_analysis:
            pre_merge = PreMergeAnalysis(
                source_model=source_path,
                target_model=target_path,
                timestamp=datetime.utcnow().isoformat(),
                domains_analyzed=[],
                domain_results={},
                mean_overlap=0.0,
                mean_alignment=0.0,
                mean_curvature_divergence=0.0,
                mean_distance=0.0,
            )
        else:
            pre_merge = self._run_pre_merge_analysis(
                source_path, target_path, transplant_domains
            )
        pre_duration = time.time() - pre_start
        logger.info("Pre-merge analysis completed in %.2fs", pre_duration)

        # Stage 2: Execute merge
        merge_start = time.time()
        merge_result = self._execute_merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            transplant_domains=transplant_domains,
            use_cpu_weights=use_cpu_weights,
        )
        merge_duration = time.time() - merge_start
        logger.info("Merge completed in %.2fs", merge_duration)

        # Stage 3: Post-merge validation (extract metrics from merge result)
        val_start = time.time()
        post_merge = self._extract_post_merge_validation(merge_result, output_dir)
        val_duration = time.time() - val_start

        return PipelineResult(
            pipeline_id=pipeline_id,
            timestamp=datetime.utcnow().isoformat(),
            source_model=source_path,
            target_model=target_path,
            output_dir=output_dir,
            pre_merge=pre_merge,
            merge_result=self._merge_result_to_dict(merge_result),
            post_merge=post_merge,
            pre_merge_duration_s=pre_duration,
            merge_duration_s=merge_duration,
            validation_duration_s=val_duration,
        )

    def _run_pre_merge_analysis(
        self,
        source_path: str,
        target_path: str,
        domains: list[str],
    ) -> PreMergeAnalysis:
        """Run pre-merge interference analysis."""
        from modelcypher.cli.composition import get_domain_geometry_waypoint_service
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.domains import (
            AtlasDomain,
            resolve_domains,
        )
        from modelcypher.core.domain.geometry.interference_predictor import (
            MergeAnalyzer,
        )
        from modelcypher.core.domain.geometry.riemannian_density import (
            RiemannianDensityEstimator,
        )

        backend = get_default_backend()
        waypoint_service = get_domain_geometry_waypoint_service()
        density_estimator = RiemannianDensityEstimator()
        predictor = MergeAnalyzer()

        # Map domain strings to AtlasDomain enums using the canonical resolver
        domain_list = resolve_domains(domains)

        if not domain_list:
            # Fall back to all domains
            domain_list = list(AtlasDomain)

        # Collect activations - load models once, not per-domain
        source_activations: dict[str, dict[str, Any]] = {}
        target_activations: dict[str, dict[str, Any]] = {}

        from modelcypher.adapters.mlx_model_loader import MLXModelLoader

        model_loader = MLXModelLoader()
        source_model, source_tokenizer = model_loader.load_model_for_training(source_path)
        target_model, target_tokenizer = model_loader.load_model_for_training(target_path)

        for domain in domain_list:
            try:
                source_acts = self._extract_domain_activations_cached(
                    source_model, source_tokenizer, domain, -1, waypoint_service
                )
                target_acts = self._extract_domain_activations_cached(
                    target_model, target_tokenizer, domain, -1, waypoint_service
                )
                source_activations[domain.value] = source_acts
                target_activations[domain.value] = target_acts
            except Exception as e:
                logger.warning("Failed to extract %s: %s", domain.value, e)

        # Analyze interference per domain
        domain_results: dict[str, dict] = {}

        for domain_name, source_acts in source_activations.items():
            target_acts = target_activations.get(domain_name, {})
            if not source_acts or not target_acts:
                continue

            source_volumes = {}
            target_volumes = {}
            common_concepts = set(source_acts.keys()) & set(target_acts.keys())

            for concept_id in common_concepts:
                src_arr = source_acts[concept_id]
                tgt_arr = target_acts[concept_id]

                if src_arr.ndim == 1:
                    src_arr = src_arr.reshape(1, -1)
                if tgt_arr.ndim == 1:
                    tgt_arr = tgt_arr.reshape(1, -1)

                source_volumes[concept_id] = density_estimator.estimate_concept_volume(
                    f"source:{concept_id}", src_arr, store_raw_activations=True
                )
                target_volumes[concept_id] = density_estimator.estimate_concept_volume(
                    f"target:{concept_id}", tgt_arr, store_raw_activations=True
                )

            domain_analysis = {
                "concepts_analyzed": len(common_concepts),
                "overlap_scores": [],
                "alignment_scores": [],
                "curvature_scores": [],
                "distance_scores": [],
            }

            for concept_id in common_concepts:
                result = predictor.analyze(
                    source_volumes[concept_id], target_volumes[concept_id]
                )
                domain_analysis["overlap_scores"].append(result.overlap_score)
                domain_analysis["alignment_scores"].append(result.alignment_score)
                domain_analysis["curvature_scores"].append(result.curvature_divergence)
                domain_analysis["distance_scores"].append(result.distance_score)

            if domain_analysis["overlap_scores"]:
                # Batch array creation and mean computation - single eval for all
                overlap_arr = backend.array(domain_analysis["overlap_scores"])
                align_arr = backend.array(domain_analysis["alignment_scores"])
                curvature_arr = backend.array(domain_analysis["curvature_scores"])
                distance_arr = backend.array(domain_analysis["distance_scores"])
                mean_overlap = backend.mean(overlap_arr)
                mean_align = backend.mean(align_arr)
                mean_curv = backend.mean(curvature_arr)
                mean_dist = backend.mean(distance_arr)
                # Single eval for all 4 means
                backend.eval(mean_overlap, mean_align, mean_curv, mean_dist)
                domain_analysis["mean_overlap"] = float(backend.to_scalar(mean_overlap))
                domain_analysis["mean_alignment"] = float(backend.to_scalar(mean_align))
                domain_analysis["mean_curvature_divergence"] = float(backend.to_scalar(mean_curv))
                domain_analysis["mean_distance"] = float(backend.to_scalar(mean_dist))
            else:
                domain_analysis["mean_overlap"] = 0.0
                domain_analysis["mean_alignment"] = 1.0
                domain_analysis["mean_curvature_divergence"] = 0.0
                domain_analysis["mean_distance"] = 0.0

            del domain_analysis["overlap_scores"]
            del domain_analysis["alignment_scores"]
            del domain_analysis["curvature_scores"]
            del domain_analysis["distance_scores"]
            domain_results[domain_name] = domain_analysis

        # Compute global metrics
        all_overlap_scores = []
        all_alignment_scores = []
        all_curvature_scores = []
        all_distance_scores = []

        for dr in domain_results.values():
            all_overlap_scores.append(dr["mean_overlap"])
            all_alignment_scores.append(dr["mean_alignment"])
            all_curvature_scores.append(dr["mean_curvature_divergence"])
            all_distance_scores.append(dr["mean_distance"])

        if all_overlap_scores:
            # Batch array creation and eval
            overlap_global = backend.mean(backend.array(all_overlap_scores))
            align_global = backend.mean(backend.array(all_alignment_scores))
            curv_global = backend.mean(backend.array(all_curvature_scores))
            dist_global = backend.mean(backend.array(all_distance_scores))
            backend.eval(overlap_global, align_global, curv_global, dist_global)
            mean_overlap = float(backend.to_scalar(overlap_global))
            mean_alignment = float(backend.to_scalar(align_global))
            mean_curvature = float(backend.to_scalar(curv_global))
            mean_distance = float(backend.to_scalar(dist_global))
        else:
            mean_overlap = 0.0
            mean_alignment = 1.0
            mean_curvature = 0.0
            mean_distance = 0.0

        return PreMergeAnalysis(
            source_model=source_path,
            target_model=target_path,
            timestamp=datetime.utcnow().isoformat(),
            domains_analyzed=[d.value for d in domain_list],
            domain_results=domain_results,
            mean_overlap=mean_overlap,
            mean_alignment=mean_alignment,
            mean_curvature_divergence=mean_curvature,
            mean_distance=mean_distance,
        )

    def _extract_domain_activations_cached(
        self,
        model: Any,
        tokenizer: Any,
        domain: "AtlasDomain",
        layer: int,
        waypoint_service: "DomainGeometryWaypointService",
    ) -> dict[str, Any]:
        """Extract activations for a domain using pre-loaded model.

        This avoids repeated disk I/O by reusing a loaded model instance.
        """
        waypoints = waypoint_service.extract(
            model, tokenizer, domain=domain, layer_idx=layer
        )
        return {wp.concept_id: wp.activations for wp in waypoints}

    def _extract_domain_activations(
        self,
        model_path: str,
        domain: "AtlasDomain",
        layer: int,
        waypoint_service: "DomainGeometryWaypointService",
    ) -> dict[str, Any]:
        """Extract activations for a specific domain (loads model from disk).

        Note: For multiple domains, use _extract_domain_activations_cached with
        a pre-loaded model to avoid repeated disk I/O.
        """
        from modelcypher.adapters.mlx_model_loader import MLXModelLoader

        model_loader = MLXModelLoader()
        model, tokenizer = model_loader.load_model_for_training(model_path)
        return self._extract_domain_activations_cached(
            model, tokenizer, domain, layer, waypoint_service
        )

    def _execute_merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
        transplant_domains: list[str],
        use_cpu_weights: bool,
    ) -> "UnifiedMergeResult":
        """Execute the geometric merge."""
        from modelcypher.cli.composition import get_geometric_merger

        merger = get_geometric_merger()
        return merger.merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            transplant_domains=transplant_domains,
            use_cpu_weights=use_cpu_weights,
        )

    def _extract_post_merge_validation(
        self,
        merge_result: "UnifiedMergeResult",
        output_dir: str,
    ) -> PostMergeValidation:
        """Extract validation metrics from merge result."""
        transplant_metrics = merge_result.transplant_metrics
        geometry_metrics = merge_result.geometry_metrics

        return PostMergeValidation(
            merged_model=output_dir,
            timestamp=datetime.utcnow().isoformat(),
            geometry_metrics=geometry_metrics,
            layers_transplanted=transplant_metrics.get("layers_transplanted", 0),
            weights_transplanted=transplant_metrics.get("weights_transplanted", 0),
            mean_preserved_fraction=transplant_metrics.get("mean_preserved_fraction", 0.0),
            mean_cka_after=geometry_metrics.get("mean_cka_after", 0.0),
        )

    def _merge_result_to_dict(self, merge_result: "UnifiedMergeResult") -> dict[str, Any]:
        """Convert merge result to dictionary."""
        return {
            "output_path": merge_result.output_path,
            "layer_count": merge_result.layer_count,
            "weight_count": merge_result.weight_count,
            "mean_preserved_fraction": merge_result.mean_preserved_fraction,
            "mean_procrustes_error": merge_result.mean_procrustes_error,
            "geometry_metrics": merge_result.geometry_metrics,
            "transplant_metrics": merge_result.transplant_metrics,
        }


__all__ = [
    "PreMergeAnalysis",
    "PostMergeValidation",
    "PipelineResult",
    "MergePipelineService",
]
