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

"""Service for verifying interference predictions against actual merge outcomes.

Implements the closed loop: Predict → Merge → Verify → Learn.

All measurements are raw geometric signals - no interpretation or thresholds.
Calibration is empirical, derived from actual prediction error distributions.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.prediction_registry import (
    CalibrationStats,
    MergePrediction,
    MergeVerification,
    PredictionRegistry,
    VerificationResult,
)

if TYPE_CHECKING:
    from modelcypher.core.use_cases.unified_merge.models import UnifiedMergeResult
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class InterferenceVerificationService:
    """Service for prediction verification and calibration.

    Usage:
        1. Before merge: create_prediction_from_analysis()
        2. After merge: verify_merge_result()
        3. Periodically: get_calibration_stats()
    """

    def __init__(
        self,
        registry_path: str | Path | None = None,
        registry: PredictionRegistry | None = None,
    ):
        """Initialize the verification service.

        Args:
            registry_path: Path to load/save registry. If None, uses in-memory only.
            registry: Existing registry instance. If None, creates new or loads from path.
        """
        self.registry_path = Path(registry_path) if registry_path else None

        if registry is not None:
            self.registry = registry
        elif self.registry_path and self.registry_path.exists():
            self.registry = PredictionRegistry.load(self.registry_path)
        else:
            self.registry = PredictionRegistry()

    def create_prediction_from_analysis(
        self,
        source_model: str,
        target_model: str,
        layer_predictions: dict[int, dict[str, Any]],
        transformation_counts: dict[str, int],
        config_thresholds: dict[str, float],
    ) -> MergePrediction:
        """Create a prediction from pre-merge analysis.

        Args:
            source_model: Source model path/name
            target_model: Target model path/name
            layer_predictions: Per-layer prediction dicts with overlap_score, etc.
            transformation_counts: Count of each TransformationType needed
            config_thresholds: Thresholds used in analysis

        Returns:
            MergePrediction stored in registry
        """
        merge_id = str(uuid.uuid4())[:8]

        # Compute aggregate predictions
        if layer_predictions:
            overlaps = [p.get("overlap_score", 0.0) for p in layer_predictions.values()]
            curvatures = [p.get("curvature_divergence", 0.0) for p in layer_predictions.values()]
            alignments = [p.get("alignment_score", 0.0) for p in layer_predictions.values()]

            mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
            mean_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
            mean_alignment = sum(alignments) / len(alignments) if alignments else 0.0
        else:
            mean_overlap = 0.0
            mean_curvature = 0.0
            mean_alignment = 0.0

        prediction = MergePrediction(
            merge_id=merge_id,
            source_model=source_model,
            target_model=target_model,
            timestamp=datetime.utcnow().isoformat(),
            layer_predictions=layer_predictions,
            predicted_mean_overlap=mean_overlap,
            predicted_mean_curvature_divergence=mean_curvature,
            predicted_mean_alignment=mean_alignment,
            predicted_transformation_counts=transformation_counts,
            config_thresholds=config_thresholds,
        )

        self.registry.store_prediction(prediction)
        self._save_if_needed()

        return prediction

    def create_prediction_from_transplant_metrics(
        self,
        source_model: str,
        target_model: str,
        transplant_metrics: dict[str, Any],
    ) -> MergePrediction:
        """Create a prediction from transplant stage metrics.

        The transplant stage computes interference analysis as part of the merge.
        This extracts the predictions for later verification.

        Args:
            source_model: Source model path
            target_model: Target model path
            transplant_metrics: Metrics dict from stage_3_transplant

        Returns:
            MergePrediction stored in registry
        """
        merge_id = str(uuid.uuid4())[:8]

        # Extract per-layer predictions from transplant metrics
        layer_predictions = {}
        transform_reqs = transplant_metrics.get("transform_requirements_by_layer", {})
        for layer_idx, requirements in transform_reqs.items():
            layer_predictions[int(layer_idx)] = {
                "transformations": requirements,
                "overlap_score": 0.0,  # Not directly available
                "curvature_divergence": 0.0,
                "alignment_score": 0.0,
            }

        # Count transformations
        transformation_counts = transplant_metrics.get("transform_requirements_counts", {})

        # Get thresholds if available
        config_thresholds = transplant_metrics.get("interference_thresholds", {})

        prediction = MergePrediction(
            merge_id=merge_id,
            source_model=source_model,
            target_model=target_model,
            timestamp=datetime.utcnow().isoformat(),
            layer_predictions=layer_predictions,
            predicted_mean_overlap=0.0,
            predicted_mean_curvature_divergence=0.0,
            predicted_mean_alignment=0.0,
            predicted_transformation_counts=transformation_counts,
            config_thresholds=config_thresholds,
        )

        self.registry.store_prediction(prediction)
        self._save_if_needed()

        return prediction

    def verify_merge_result(
        self,
        merge_id: str,
        merge_result: "UnifiedMergeResult",
    ) -> VerificationResult | None:
        """Verify a merge result against its prediction.

        Args:
            merge_id: ID of the prediction to verify
            merge_result: Actual merge result

        Returns:
            VerificationResult if prediction exists, None otherwise
        """
        if merge_id not in self.registry.predictions:
            logger.warning("No prediction found for merge_id %s", merge_id)
            return None

        # Extract actuals from merge result
        geometry_metrics = merge_result.geometry_metrics
        transplant_metrics = merge_result.transplant_metrics

        # Per-layer actuals
        layer_actuals = {}
        preserved_fractions = transplant_metrics.get("preserved_fractions", [])
        layers_transplanted = transplant_metrics.get("layers_transplanted", 0)

        # Map preserved fractions to layer indices
        # This assumes sequential layers - may need refinement
        for i, pf in enumerate(preserved_fractions):
            layer_actuals[i] = {"preserved_fraction": pf}

        # Actual transformation counts
        actual_transformation_counts = transplant_metrics.get(
            "transform_requirements_counts", {}
        )

        verification = MergeVerification(
            merge_id=merge_id,
            timestamp=datetime.utcnow().isoformat(),
            actual_mean_confidence=merge_result.mean_confidence,
            actual_preserved_fraction=geometry_metrics.get("mean_preserved_fraction", 0.0),
            actual_cka_after=geometry_metrics.get("mean_cka_after", 0.0),
            actual_safety_verdict=merge_result.safety_verdict,
            layer_actuals=layer_actuals,
            actual_transformation_counts=actual_transformation_counts,
        )

        self.registry.store_verification(verification)
        self._save_if_needed()

        return self.registry.results.get(merge_id)

    def verify_from_metrics(
        self,
        merge_id: str,
        geometry_metrics: dict[str, Any],
        transplant_metrics: dict[str, Any],
        safety_verdict: str,
    ) -> VerificationResult | None:
        """Verify using raw metrics instead of UnifiedMergeResult.

        Useful when only metrics are available (e.g., from JSON output).

        Args:
            merge_id: ID of the prediction to verify
            geometry_metrics: Geometry metrics dict
            transplant_metrics: Transplant metrics dict
            safety_verdict: Safety verdict string

        Returns:
            VerificationResult if prediction exists, None otherwise
        """
        if merge_id not in self.registry.predictions:
            logger.warning("No prediction found for merge_id %s", merge_id)
            return None

        # Per-layer actuals
        layer_actuals = {}
        preserved_fractions = transplant_metrics.get("preserved_fractions", [])
        for i, pf in enumerate(preserved_fractions):
            layer_actuals[i] = {"preserved_fraction": pf}

        verification = MergeVerification(
            merge_id=merge_id,
            timestamp=datetime.utcnow().isoformat(),
            actual_mean_confidence=geometry_metrics.get("mean_preserved_fraction", 0.0),
            actual_preserved_fraction=geometry_metrics.get("mean_preserved_fraction", 0.0),
            actual_cka_after=geometry_metrics.get("mean_cka_after", 0.0),
            actual_safety_verdict=safety_verdict,
            layer_actuals=layer_actuals,
            actual_transformation_counts=transplant_metrics.get(
                "transform_requirements_counts", {}
            ),
        )

        self.registry.store_verification(verification)
        self._save_if_needed()

        return self.registry.results.get(merge_id)

    def get_calibration_stats(self) -> CalibrationStats:
        """Get calibration statistics from verification history.

        Returns empirical distribution of prediction errors.
        """
        return self.registry.compute_calibration_stats()

    def get_verification_result(self, merge_id: str) -> VerificationResult | None:
        """Get verification result for a specific merge."""
        return self.registry.results.get(merge_id)

    def get_prediction(self, merge_id: str) -> MergePrediction | None:
        """Get prediction for a specific merge."""
        return self.registry.predictions.get(merge_id)

    def list_pending_verifications(self) -> list[str]:
        """List merge IDs that have predictions but no verification."""
        return [
            merge_id
            for merge_id in self.registry.predictions
            if merge_id not in self.registry.verifications
        ]

    def export_calibration_report(self, output_path: str | Path) -> dict[str, Any]:
        """Export calibration report to JSON.

        Args:
            output_path: Path to write report

        Returns:
            Report dict (also written to file)
        """
        stats = self.get_calibration_stats()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "calibration": asdict(stats),
            "summary": {
                "n_verifications": stats.n_verifications,
                "mean_absolute_error": stats.mean_absolute_error,
                "transformation_accuracy_rates": stats.transformation_accuracy_rates,
            },
        }

        import json
        output_path.write_text(json.dumps(report, indent=2))
        logger.info("Exported calibration report to %s", output_path)

        return report

    def _save_if_needed(self) -> None:
        """Save registry if path is configured."""
        if self.registry_path:
            self.registry.save(self.registry_path)


__all__ = [
    "InterferenceVerificationService",
]
