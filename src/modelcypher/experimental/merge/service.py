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

Orchestrates: Execute merge → Post-merge validation

All stages return raw geometric measurements. No interpretation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.inference import InferenceEngine
    from modelcypher.ports.model_loader import ModelLoaderPort

    from .merger import UnifiedGeometricMerger
    from .models import UnifiedMergeResult

logger = logging.getLogger(__name__)


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
    merge_result: dict[str, Any]
    post_merge: PostMergeValidation

    # Timing
    merge_duration_s: float = 0.0
    validation_duration_s: float = 0.0


class MergePipelineService:
    """Orchestrates the full merge pipeline.

    Pipeline stages:
    1. Execute merge: Unified geometric merge
    2. Post-merge validation: Extract geometry metrics
    """

    def __init__(
        self,
        geometric_merger: "UnifiedGeometricMerger",
        model_loader: "ModelLoaderPort",
        inference_engine: "InferenceEngine | None" = None,
    ):
        """Initialize the pipeline service with dependencies.

        Args:
            geometric_merger: Merger for executing geometric merges.
            model_loader: Port for loading models.
            inference_engine: Optional inference engine for validation.
        """
        self._geometric_merger = geometric_merger
        self._model_loader = model_loader
        self._inference_engine = inference_engine

    def run(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
    ) -> PipelineResult:
        """Run the complete merge pipeline.

        Args:
            source_path: Path to source model
            target_path: Path to target model
            output_dir: Output directory for merged model
        Returns:
            PipelineResult with all stage results
        """
        import time

        pipeline_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        logger.info("Starting merge pipeline %s", pipeline_id)

        # Load models once for entire pipeline
        logger.info("Loading models...")
        source_model, source_tokenizer = self._model_loader.load_model_for_training(source_path)
        target_model, target_tokenizer = self._model_loader.load_model_for_training(target_path)
        logger.info("Models loaded successfully")

        # Stage 1: Execute merge
        merge_start = time.time()
        merge_result = self._execute_merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        merge_duration = time.time() - merge_start
        logger.info("Merge completed in %.2fs", merge_duration)

        # Stage 2: Post-merge validation (extract metrics from merge result)
        val_start = time.time()
        post_merge = self._extract_post_merge_validation(merge_result, output_dir)
        val_duration = time.time() - val_start

        return PipelineResult(
            pipeline_id=pipeline_id,
            timestamp=datetime.utcnow().isoformat(),
            source_model=source_path,
            target_model=target_path,
            output_dir=output_dir,
            merge_result=self._merge_result_to_dict(merge_result),
            post_merge=post_merge,
            merge_duration_s=merge_duration,
            validation_duration_s=val_duration,
        )

    def _execute_merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str,
        source_model: Any | None = None,
        target_model: Any | None = None,
        source_tokenizer: Any | None = None,
        target_tokenizer: Any | None = None,
    ) -> "UnifiedMergeResult":
        """Execute the geometric merge."""
        return self._geometric_merger.merge(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            inference_engine=self._inference_engine,
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
        result = {
            "output_path": merge_result.output_path,
            "layer_count": merge_result.layer_count,
            "weight_count": merge_result.weight_count,
            "mean_preserved_fraction": merge_result.mean_preserved_fraction,
            "mean_procrustes_error": merge_result.mean_procrustes_error,
            "merge_strategy": merge_result.merge_strategy,
            "probe_metrics": merge_result.probe_metrics,
            "permute_metrics": merge_result.permute_metrics,
            "geometry_metrics": merge_result.geometry_metrics,
            "transplant_metrics": merge_result.transplant_metrics,
            "density_metrics": merge_result.density_metrics,
            "validation_metrics": merge_result.validation_metrics,
            "post_merge_density": merge_result.post_merge_density,
            "refusal_preserved": merge_result.refusal_preserved,
        }

        # Add fingerprint comparison if available
        fp = merge_result.fingerprint_comparison
        if fp is not None:
            result["fingerprints"] = {
                "source_gram_hash": fp.source_gram_hash,
                "target_gram_hash": fp.target_gram_hash,
                "source_condition_number": fp.source_condition_number,
                "target_condition_number": fp.target_condition_number,
                "source_effective_dim": fp.source_effective_dim,
                "target_effective_dim": fp.target_effective_dim,
                "condition_number_ratio": fp.condition_number_ratio,
                "effective_dim_delta": fp.effective_dim_delta,
            }

        return result


__all__ = [
    "PostMergeValidation",
    "PipelineResult",
    "MergePipelineService",
]
