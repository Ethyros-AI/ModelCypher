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

Orchestrates: Pre-merge analysis → Execute merge → Post-merge validation → Verification

All stages return raw geometric measurements. No interpretation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
    transformation_counts: dict[str, int]
    total_transformations_needed: int

    # Layer predictions (for later verification)
    layer_predictions: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class PostMergeValidation:
    """Post-merge validation results.

    All values are raw measurements from the merged model.
    """

    merged_model: str
    timestamp: str

    # Geometry from merge result
    mean_confidence: float
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

    # Verification (if predictions were made)
    verification: dict[str, Any] | None = None

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
    4. Verification: Compare predictions to actuals (if enabled)
    """

    def __init__(
        self,
        verification_registry_path: str | Path | None = None,
    ):
        """Initialize the pipeline service.

        Args:
            verification_registry_path: Path to store prediction/verification history.
                If None, uses in-memory only.
        """
        self.verification_registry_path = verification_registry_path

    def run(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
        transplant_domains: list[str],
        *,
        skip_pre_analysis: bool = False,
        verify_predictions: bool = True,
    ) -> PipelineResult:
        """Run the complete merge pipeline.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            output_dir: Output directory for merged model
            transplant_domains: Domains to transplant (e.g., ["mathematical", "logical"])
            skip_pre_analysis: Skip pre-merge interference analysis
            verify_predictions: Enable prediction verification

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
                transformation_counts={},
                total_transformations_needed=0,
            )
        else:
            pre_merge = self._run_pre_merge_analysis(
                source_path, target_path, transplant_domains
            )
        pre_duration = time.time() - pre_start
        logger.info("Pre-merge analysis completed in %.2fs", pre_duration)

        # Store prediction for later verification
        merge_id = None
        if verify_predictions and not skip_pre_analysis:
            merge_id = self._store_prediction(pre_merge)

        # Stage 2: Execute merge
        merge_start = time.time()
        merge_result = self._execute_merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            transplant_domains=transplant_domains,
        )
        merge_duration = time.time() - merge_start
        logger.info("Merge completed in %.2fs", merge_duration)

        # Stage 3: Post-merge validation (extract metrics from merge result)
        val_start = time.time()
        post_merge = self._extract_post_merge_validation(merge_result, output_dir)
        val_duration = time.time() - val_start

        # Stage 4: Verification (compare predictions to actuals)
        verification = None
        if verify_predictions and merge_id:
            verification = self._verify_predictions(merge_id, merge_result)

        return PipelineResult(
            pipeline_id=pipeline_id,
            timestamp=datetime.utcnow().isoformat(),
            source_model=source_path,
            target_model=target_path,
            output_dir=output_dir,
            pre_merge=pre_merge,
            merge_result=self._merge_result_to_dict(merge_result),
            post_merge=post_merge,
            verification=verification,
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
            TransformationType,
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

        # Collect activations
        source_activations: dict[str, dict[str, Any]] = {}
        target_activations: dict[str, dict[str, Any]] = {}

        for domain in domain_list:
            try:
                source_acts = self._extract_domain_activations(
                    source_path, domain, -1, waypoint_service
                )
                target_acts = self._extract_domain_activations(
                    target_path, domain, -1, waypoint_service
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
                "transformation_counts": {t.value: 0 for t in TransformationType},
                "overlap_scores": [],
                "alignment_scores": [],
            }

            for concept_id in common_concepts:
                result = predictor.analyze(
                    source_volumes[concept_id], target_volumes[concept_id]
                )
                for t in result.transformations:
                    domain_analysis["transformation_counts"][t.value] += 1
                domain_analysis["overlap_scores"].append(result.overlap_score)
                domain_analysis["alignment_scores"].append(result.alignment_score)

            if domain_analysis["overlap_scores"]:
                overlap_arr = backend.array(domain_analysis["overlap_scores"])
                align_arr = backend.array(domain_analysis["alignment_scores"])
                domain_analysis["mean_overlap"] = float(backend.mean(overlap_arr))
                domain_analysis["mean_alignment"] = float(backend.mean(align_arr))
            else:
                domain_analysis["mean_overlap"] = 0.0
                domain_analysis["mean_alignment"] = 1.0

            del domain_analysis["overlap_scores"]
            del domain_analysis["alignment_scores"]
            domain_results[domain_name] = domain_analysis

        # Compute global metrics
        global_transformation_counts: dict[str, int] = {}
        all_overlap_scores = []
        all_alignment_scores = []
        total_transformations = 0

        for dr in domain_results.values():
            all_overlap_scores.append(dr["mean_overlap"])
            all_alignment_scores.append(dr["mean_alignment"])
            for ttype, count in dr.get("transformation_counts", {}).items():
                global_transformation_counts[ttype] = (
                    global_transformation_counts.get(ttype, 0) + count
                )
                total_transformations += count

        if all_overlap_scores:
            mean_overlap = float(backend.mean(backend.array(all_overlap_scores)))
            mean_alignment = float(backend.mean(backend.array(all_alignment_scores)))
        else:
            mean_overlap = 0.0
            mean_alignment = 1.0

        return PreMergeAnalysis(
            source_model=source_path,
            target_model=target_path,
            timestamp=datetime.utcnow().isoformat(),
            domains_analyzed=[d.value for d in domain_list],
            domain_results=domain_results,
            mean_overlap=mean_overlap,
            mean_alignment=mean_alignment,
            transformation_counts=global_transformation_counts,
            total_transformations_needed=total_transformations,
        )

    def _extract_domain_activations(
        self,
        model_path: str,
        domain: "AtlasDomain",
        layer: int,
        waypoint_service: "DomainGeometryWaypointService",
    ) -> dict[str, Any]:
        """Extract activations for a specific domain."""
        from modelcypher.adapters.mlx_model_loader import MLXModelLoader

        model_loader = MLXModelLoader()
        model, tokenizer = model_loader.load_model_for_training(model_path)
        waypoints = waypoint_service.extract(
            model, tokenizer, domain=domain, layer_idx=layer
        )

        return {wp.concept_id: wp.activations for wp in waypoints}

    def _execute_merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
        transplant_domains: list[str],
    ) -> "UnifiedMergeResult":
        """Execute the geometric merge."""
        from modelcypher.cli.composition import get_geometric_merger

        merger = get_geometric_merger()
        return merger.merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            transplant_domains=transplant_domains,
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
            mean_confidence=merge_result.mean_confidence,
            geometry_metrics=geometry_metrics,
            layers_transplanted=transplant_metrics.get("layers_transplanted", 0),
            weights_transplanted=transplant_metrics.get("weights_transplanted", 0),
            mean_preserved_fraction=transplant_metrics.get("mean_preserved_fraction", 0.0),
            mean_cka_after=geometry_metrics.get("mean_cka_after", 0.0),
        )

    def _store_prediction(self, pre_merge: PreMergeAnalysis) -> str | None:
        """Store prediction for later verification."""
        try:
            from modelcypher.core.use_cases.interference_verification_service import (
                InterferenceVerificationService,
            )

            service = InterferenceVerificationService(
                registry_path=self.verification_registry_path
            )

            prediction = service.create_prediction_from_analysis(
                source_model=pre_merge.source_model,
                target_model=pre_merge.target_model,
                layer_predictions=pre_merge.layer_predictions,
                transformation_counts=pre_merge.transformation_counts,
                config_thresholds={},
            )

            return prediction.merge_id
        except Exception as e:
            logger.warning("Failed to store prediction: %s", e)
            return None

    def _verify_predictions(
        self,
        merge_id: str,
        merge_result: "UnifiedMergeResult",
    ) -> dict[str, Any] | None:
        """Verify predictions against actual merge outcome."""
        try:
            from modelcypher.core.use_cases.interference_verification_service import (
                InterferenceVerificationService,
            )

            service = InterferenceVerificationService(
                registry_path=self.verification_registry_path
            )

            result = service.verify_merge_result(merge_id, merge_result)
            if result:
                return {
                    "merge_id": result.merge_id,
                    "overlap_delta": result.overlap_delta,
                    "curvature_delta": result.curvature_delta,
                    "alignment_delta": result.alignment_delta,
                    "transformation_accuracy": result.transformation_accuracy,
                    "mean_absolute_error": result.mean_absolute_error,
                }
            return None
        except Exception as e:
            logger.warning("Failed to verify predictions: %s", e)
            return None

    def _merge_result_to_dict(self, merge_result: "UnifiedMergeResult") -> dict[str, Any]:
        """Convert merge result to dictionary."""
        return {
            "output_path": merge_result.output_path,
            "layer_count": merge_result.layer_count,
            "weight_count": merge_result.weight_count,
            "mean_confidence": merge_result.mean_confidence,
            "vocab_aligned": merge_result.vocab_aligned,
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
