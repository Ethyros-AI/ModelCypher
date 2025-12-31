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

"""Registry for storing merge predictions and verification outcomes.

Enables the closed loop: Predict → Merge → Verify → Learn.

All stored values are raw measurements - no interpretations or thresholds.
Calibration statistics are computed from the empirical distribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergePrediction:
    """Prediction made before a merge operation.

    Contains geometric predictions that can be verified after merge.
    """

    merge_id: str
    source_model: str
    target_model: str
    timestamp: str

    # Per-layer predictions
    # Keys are layer indices, values are prediction dicts
    layer_predictions: dict[int, dict[str, Any]]

    # Aggregate predictions
    predicted_mean_overlap: float
    predicted_mean_curvature_divergence: float
    predicted_mean_alignment: float
    predicted_transformation_counts: dict[str, int]

    # Configuration used
    config_thresholds: dict[str, float]


@dataclass(frozen=True)
class MergeVerification:
    """Actual outcomes measured after a merge operation.

    Contains measurements that can be compared to predictions.
    """

    merge_id: str
    timestamp: str

    # Actual merge outcomes
    actual_mean_confidence: float
    actual_preserved_fraction: float
    actual_cka_after: float
    actual_safety_verdict: str

    # Per-layer actuals (for layer-level comparison)
    layer_actuals: dict[int, dict[str, float]]

    # Actual transformation counts from merge
    actual_transformation_counts: dict[str, int]


@dataclass
class VerificationResult:
    """Comparison between prediction and actual outcome.

    Contains raw error measurements - no interpretation.
    """

    merge_id: str
    timestamp: str

    # Prediction vs actual deltas (actual - predicted)
    # Positive = underestimated, Negative = overestimated
    overlap_delta: float
    curvature_delta: float
    alignment_delta: float

    # Transformation prediction accuracy
    # For each transformation type: was it correctly predicted?
    transformation_accuracy: dict[str, bool]

    # Overall prediction error (mean absolute error across signals)
    mean_absolute_error: float

    # Layer-level errors
    layer_errors: dict[int, float]


@dataclass
class CalibrationStats:
    """Calibration statistics derived from verification history.

    All values are empirical measurements from the verification history.
    No magic thresholds - just the distribution of prediction errors.
    """

    # Sample size
    n_verifications: int

    # Error distribution (for bias correction)
    mean_overlap_error: float
    std_overlap_error: float
    mean_curvature_error: float
    std_curvature_error: float
    mean_alignment_error: float
    std_alignment_error: float

    # Per-transformation accuracy rates
    transformation_accuracy_rates: dict[str, float]

    # Overall calibration
    mean_absolute_error: float
    median_absolute_error: float

    # 90th percentile error (for confidence bounds)
    error_90th_percentile: float


@dataclass
class PredictionRegistry:
    """Registry for storing and verifying merge predictions.

    Enables closed-loop learning: predictions are stored before merge,
    outcomes are recorded after, and calibration improves over time.
    """

    predictions: dict[str, MergePrediction] = field(default_factory=dict)
    verifications: dict[str, MergeVerification] = field(default_factory=dict)
    results: dict[str, VerificationResult] = field(default_factory=dict)

    def store_prediction(self, prediction: MergePrediction) -> None:
        """Store a prediction before merge."""
        self.predictions[prediction.merge_id] = prediction
        logger.info(
            "Stored prediction for merge %s: overlap=%.3f, curvature=%.3f, alignment=%.3f",
            prediction.merge_id,
            prediction.predicted_mean_overlap,
            prediction.predicted_mean_curvature_divergence,
            prediction.predicted_mean_alignment,
        )

    def store_verification(self, verification: MergeVerification) -> None:
        """Store actual outcomes after merge."""
        self.verifications[verification.merge_id] = verification

        # If we have the prediction, compute comparison
        if verification.merge_id in self.predictions:
            result = self._compare(
                self.predictions[verification.merge_id], verification
            )
            self.results[verification.merge_id] = result
            logger.info(
                "Verified merge %s: MAE=%.4f",
                verification.merge_id,
                result.mean_absolute_error,
            )

    def _compare(
        self, prediction: MergePrediction, verification: MergeVerification
    ) -> VerificationResult:
        """Compare prediction to actual outcome."""
        # Compute deltas
        # For overlap/curvature, we compare to preserved_fraction as proxy
        # since preserved_fraction reflects how much geometry was maintained
        overlap_delta = verification.actual_preserved_fraction - prediction.predicted_mean_overlap
        curvature_delta = 0.0  # Need curvature measurement in verification
        alignment_delta = verification.actual_cka_after - prediction.predicted_mean_alignment

        # Compute transformation accuracy
        transformation_accuracy = {}
        for t_name, predicted_count in prediction.predicted_transformation_counts.items():
            actual_count = verification.actual_transformation_counts.get(t_name, 0)
            # Consider accurate if both agree on whether transformation was needed
            predicted_needed = predicted_count > 0
            actual_needed = actual_count > 0
            transformation_accuracy[t_name] = predicted_needed == actual_needed

        # Compute mean absolute error
        errors = [abs(overlap_delta), abs(curvature_delta), abs(alignment_delta)]
        mean_absolute_error = sum(errors) / len(errors) if errors else 0.0

        # Compute layer-level errors
        layer_errors = {}
        for layer_idx, pred_layer in prediction.layer_predictions.items():
            if layer_idx in verification.layer_actuals:
                actual_layer = verification.layer_actuals[layer_idx]
                # Compare predicted overlap to actual preserved_fraction
                pred_overlap = pred_layer.get("overlap_score", 0.0)
                actual_pres = actual_layer.get("preserved_fraction", 0.0)
                layer_errors[layer_idx] = abs(actual_pres - pred_overlap)

        return VerificationResult(
            merge_id=verification.merge_id,
            timestamp=datetime.utcnow().isoformat(),
            overlap_delta=overlap_delta,
            curvature_delta=curvature_delta,
            alignment_delta=alignment_delta,
            transformation_accuracy=transformation_accuracy,
            mean_absolute_error=mean_absolute_error,
            layer_errors=layer_errors,
        )

    def compute_calibration_stats(self) -> CalibrationStats:
        """Compute calibration statistics from verification history.

        Returns empirical distribution of prediction errors.
        """
        if not self.results:
            return CalibrationStats(
                n_verifications=0,
                mean_overlap_error=0.0,
                std_overlap_error=0.0,
                mean_curvature_error=0.0,
                std_curvature_error=0.0,
                mean_alignment_error=0.0,
                std_alignment_error=0.0,
                transformation_accuracy_rates={},
                mean_absolute_error=0.0,
                median_absolute_error=0.0,
                error_90th_percentile=0.0,
            )

        import math

        results = list(self.results.values())
        n = len(results)

        # Compute error statistics
        overlap_errors = [r.overlap_delta for r in results]
        curvature_errors = [r.curvature_delta for r in results]
        alignment_errors = [r.alignment_delta for r in results]
        maes = [r.mean_absolute_error for r in results]

        def mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def std(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            m = mean(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

        def median(vals: list[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

        def percentile(vals: list[float], p: float) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            idx = int(p * (len(s) - 1))
            return s[idx]

        # Compute per-transformation accuracy rates
        transformation_accuracy_rates = {}
        all_transformations = set()
        for r in results:
            all_transformations.update(r.transformation_accuracy.keys())
        for t_name in all_transformations:
            correct = sum(
                1 for r in results if r.transformation_accuracy.get(t_name, False)
            )
            transformation_accuracy_rates[t_name] = correct / n

        return CalibrationStats(
            n_verifications=n,
            mean_overlap_error=mean(overlap_errors),
            std_overlap_error=std(overlap_errors),
            mean_curvature_error=mean(curvature_errors),
            std_curvature_error=std(curvature_errors),
            mean_alignment_error=mean(alignment_errors),
            std_alignment_error=std(alignment_errors),
            transformation_accuracy_rates=transformation_accuracy_rates,
            mean_absolute_error=mean(maes),
            median_absolute_error=median(maes),
            error_90th_percentile=percentile(maes, 0.9),
        )

    def save(self, path: str | Path) -> None:
        """Save registry to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "predictions": {k: asdict(v) for k, v in self.predictions.items()},
            "verifications": {k: asdict(v) for k, v in self.verifications.items()},
            "results": {k: asdict(v) for k, v in self.results.items()},
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved registry to %s (%d predictions, %d verifications)", path, len(self.predictions), len(self.verifications))

    @classmethod
    def load(cls, path: str | Path) -> "PredictionRegistry":
        """Load registry from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text())

        registry = cls()
        for k, v in data.get("predictions", {}).items():
            registry.predictions[k] = MergePrediction(**v)
        for k, v in data.get("verifications", {}).items():
            registry.verifications[k] = MergeVerification(**v)
        for k, v in data.get("results", {}).items():
            registry.results[k] = VerificationResult(**v)

        logger.info(
            "Loaded registry from %s (%d predictions, %d verifications)",
            path,
            len(registry.predictions),
            len(registry.verifications),
        )
        return registry


__all__ = [
    "MergePrediction",
    "MergeVerification",
    "VerificationResult",
    "CalibrationStats",
    "PredictionRegistry",
]
