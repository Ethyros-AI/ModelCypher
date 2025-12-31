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

"""Tests for interference prediction verification.

Verifies the closed loop: Predict → Merge → Verify → Learn.
"""

import json
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.domain.geometry.prediction_registry import (
    CalibrationStats,
    MergePrediction,
    MergeVerification,
    PredictionRegistry,
    VerificationResult,
)
from modelcypher.core.use_cases.interference_verification_service import (
    InterferenceVerificationService,
)


class TestMergePrediction:
    """Tests for MergePrediction dataclass."""

    def test_create_prediction(self):
        """Can create a prediction with all fields."""
        pred = MergePrediction(
            merge_id="test123",
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp="2025-01-01T00:00:00",
            layer_predictions={0: {"overlap_score": 0.7}},
            predicted_mean_overlap=0.7,
            predicted_mean_curvature_divergence=0.2,
            predicted_mean_alignment=0.85,
            predicted_transformation_counts={"alpha_scaling": 3},
            config_thresholds={"alpha_scaling_threshold": 0.5},
        )

        assert pred.merge_id == "test123"
        assert pred.predicted_mean_overlap == 0.7
        assert pred.predicted_transformation_counts["alpha_scaling"] == 3

    def test_prediction_is_frozen(self):
        """Predictions should be immutable."""
        pred = MergePrediction(
            merge_id="test",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={},
            predicted_mean_overlap=0.5,
            predicted_mean_curvature_divergence=0.3,
            predicted_mean_alignment=0.8,
            predicted_transformation_counts={},
            config_thresholds={},
        )

        with pytest.raises(Exception):
            pred.merge_id = "modified"


class TestMergeVerification:
    """Tests for MergeVerification dataclass."""

    def test_create_verification(self):
        """Can create a verification with all fields."""
        verif = MergeVerification(
            merge_id="test123",
            timestamp="2025-01-01T00:00:00",
            actual_mean_confidence=0.85,
            actual_preserved_fraction=0.78,
            actual_cka_after=0.92,
            actual_safety_verdict="healthy",
            layer_actuals={0: {"preserved_fraction": 0.8}},
            actual_transformation_counts={"alpha_scaling": 2},
        )

        assert verif.merge_id == "test123"
        assert verif.actual_preserved_fraction == 0.78
        assert verif.actual_safety_verdict == "healthy"


class TestPredictionRegistry:
    """Tests for PredictionRegistry."""

    def test_store_and_retrieve_prediction(self):
        """Can store and retrieve a prediction."""
        registry = PredictionRegistry()
        pred = MergePrediction(
            merge_id="abc",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={},
            predicted_mean_overlap=0.6,
            predicted_mean_curvature_divergence=0.2,
            predicted_mean_alignment=0.9,
            predicted_transformation_counts={},
            config_thresholds={},
        )

        registry.store_prediction(pred)

        assert "abc" in registry.predictions
        assert registry.predictions["abc"].predicted_mean_overlap == 0.6

    def test_store_verification_creates_result(self):
        """Storing verification with matching prediction creates result."""
        registry = PredictionRegistry()

        pred = MergePrediction(
            merge_id="xyz",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={},
            predicted_mean_overlap=0.7,
            predicted_mean_curvature_divergence=0.3,
            predicted_mean_alignment=0.8,
            predicted_transformation_counts={"alpha_scaling": 3},
            config_thresholds={},
        )
        registry.store_prediction(pred)

        verif = MergeVerification(
            merge_id="xyz",
            timestamp="now",
            actual_mean_confidence=0.75,
            actual_preserved_fraction=0.65,
            actual_cka_after=0.85,
            actual_safety_verdict="healthy",
            layer_actuals={},
            actual_transformation_counts={"alpha_scaling": 2},
        )
        registry.store_verification(verif)

        assert "xyz" in registry.results
        result = registry.results["xyz"]
        assert result.merge_id == "xyz"

    def test_verification_without_prediction(self):
        """Verification without prediction doesn't create result."""
        registry = PredictionRegistry()

        verif = MergeVerification(
            merge_id="orphan",
            timestamp="now",
            actual_mean_confidence=0.8,
            actual_preserved_fraction=0.75,
            actual_cka_after=0.9,
            actual_safety_verdict="healthy",
            layer_actuals={},
            actual_transformation_counts={},
        )
        registry.store_verification(verif)

        assert "orphan" not in registry.results
        assert "orphan" in registry.verifications

    def test_save_and_load(self, tmp_path):
        """Registry can be saved and loaded."""
        registry = PredictionRegistry()
        pred = MergePrediction(
            merge_id="persist",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={0: {"overlap": 0.5}},
            predicted_mean_overlap=0.5,
            predicted_mean_curvature_divergence=0.2,
            predicted_mean_alignment=0.8,
            predicted_transformation_counts={},
            config_thresholds={},
        )
        registry.store_prediction(pred)

        save_path = tmp_path / "registry.json"
        registry.save(save_path)

        loaded = PredictionRegistry.load(save_path)
        assert "persist" in loaded.predictions
        assert loaded.predictions["persist"].predicted_mean_overlap == 0.5

    def test_calibration_stats_empty_registry(self):
        """Empty registry returns zero calibration stats."""
        registry = PredictionRegistry()
        stats = registry.compute_calibration_stats()

        assert stats.n_verifications == 0
        assert stats.mean_absolute_error == 0.0

    def test_calibration_stats_with_data(self):
        """Calibration stats computed from verification history."""
        registry = PredictionRegistry()

        # Add several prediction/verification pairs
        for i, (overlap, actual) in enumerate([(0.7, 0.65), (0.5, 0.55), (0.8, 0.75)]):
            pred = MergePrediction(
                merge_id=f"merge{i}",
                source_model="src",
                target_model="tgt",
                timestamp="now",
                layer_predictions={},
                predicted_mean_overlap=overlap,
                predicted_mean_curvature_divergence=0.2,
                predicted_mean_alignment=0.8,
                predicted_transformation_counts={},
                config_thresholds={},
            )
            registry.store_prediction(pred)

            verif = MergeVerification(
                merge_id=f"merge{i}",
                timestamp="now",
                actual_mean_confidence=actual,
                actual_preserved_fraction=actual,
                actual_cka_after=0.85,
                actual_safety_verdict="healthy",
                layer_actuals={},
                actual_transformation_counts={},
            )
            registry.store_verification(verif)

        stats = registry.compute_calibration_stats()

        assert stats.n_verifications == 3
        assert stats.mean_absolute_error > 0  # Should have some error


class TestVerificationResult:
    """Tests for VerificationResult computation."""

    def test_error_computation(self):
        """Errors are computed correctly."""
        registry = PredictionRegistry()

        pred = MergePrediction(
            merge_id="err",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={},
            predicted_mean_overlap=0.8,
            predicted_mean_curvature_divergence=0.0,
            predicted_mean_alignment=0.9,
            predicted_transformation_counts={"alpha_scaling": 2},
            config_thresholds={},
        )
        registry.store_prediction(pred)

        verif = MergeVerification(
            merge_id="err",
            timestamp="now",
            actual_mean_confidence=0.7,
            actual_preserved_fraction=0.7,
            actual_cka_after=0.85,
            actual_safety_verdict="healthy",
            layer_actuals={},
            actual_transformation_counts={"alpha_scaling": 2},
        )
        registry.store_verification(verif)

        result = registry.results["err"]

        # overlap_delta = actual - predicted = 0.7 - 0.8 = -0.1
        assert result.overlap_delta == pytest.approx(-0.1, rel=0.01)
        # alignment_delta = actual_cka - predicted_alignment = 0.85 - 0.9 = -0.05
        assert result.alignment_delta == pytest.approx(-0.05, rel=0.01)

    def test_transformation_accuracy(self):
        """Transformation prediction accuracy computed correctly."""
        registry = PredictionRegistry()

        pred = MergePrediction(
            merge_id="trans",
            source_model="src",
            target_model="tgt",
            timestamp="now",
            layer_predictions={},
            predicted_mean_overlap=0.5,
            predicted_mean_curvature_divergence=0.0,
            predicted_mean_alignment=0.8,
            predicted_transformation_counts={
                "alpha_scaling": 3,
                "procrustes_rotation": 0,
            },
            config_thresholds={},
        )
        registry.store_prediction(pred)

        verif = MergeVerification(
            merge_id="trans",
            timestamp="now",
            actual_mean_confidence=0.5,
            actual_preserved_fraction=0.5,
            actual_cka_after=0.8,
            actual_safety_verdict="healthy",
            layer_actuals={},
            actual_transformation_counts={
                "alpha_scaling": 2,
                "procrustes_rotation": 1,
            },
        )
        registry.store_verification(verif)

        result = registry.results["trans"]

        # alpha_scaling: predicted needed (3>0), actual needed (2>0) → correct
        assert result.transformation_accuracy["alpha_scaling"] is True
        # procrustes: predicted not needed (0), actual needed (1>0) → incorrect
        assert result.transformation_accuracy["procrustes_rotation"] is False


class TestInterferenceVerificationService:
    """Tests for the verification service."""

    def test_create_prediction_from_analysis(self):
        """Can create prediction from analysis data."""
        service = InterferenceVerificationService()

        pred = service.create_prediction_from_analysis(
            source_model="/path/source",
            target_model="/path/target",
            layer_predictions={
                0: {"overlap_score": 0.7, "curvature_divergence": 0.2},
                1: {"overlap_score": 0.6, "curvature_divergence": 0.3},
            },
            transformation_counts={"alpha_scaling": 2},
            config_thresholds={"alpha_scaling_threshold": 0.5},
        )

        assert pred.merge_id is not None
        assert pred.predicted_mean_overlap == pytest.approx(0.65, rel=0.01)
        assert len(service.registry.predictions) == 1

    def test_verify_from_metrics(self):
        """Can verify using metrics dict."""
        service = InterferenceVerificationService()

        # Create prediction
        pred = service.create_prediction_from_analysis(
            source_model="src",
            target_model="tgt",
            layer_predictions={0: {"overlap_score": 0.7}},
            transformation_counts={},
            config_thresholds={},
        )

        # Verify with metrics
        result = service.verify_from_metrics(
            merge_id=pred.merge_id,
            geometry_metrics={
                "mean_preserved_fraction": 0.65,
                "mean_cka_after": 0.85,
            },
            transplant_metrics={
                "preserved_fractions": [0.65],
                "transform_requirements_counts": {},
            },
            safety_verdict="healthy",
        )

        assert result is not None
        assert result.merge_id == pred.merge_id

    def test_list_pending_verifications(self):
        """Can list predictions awaiting verification."""
        service = InterferenceVerificationService()

        # Create two predictions
        pred1 = service.create_prediction_from_analysis(
            source_model="src1",
            target_model="tgt1",
            layer_predictions={},
            transformation_counts={},
            config_thresholds={},
        )
        pred2 = service.create_prediction_from_analysis(
            source_model="src2",
            target_model="tgt2",
            layer_predictions={},
            transformation_counts={},
            config_thresholds={},
        )

        # Verify only first
        service.verify_from_metrics(
            merge_id=pred1.merge_id,
            geometry_metrics={"mean_preserved_fraction": 0.5},
            transplant_metrics={},
            safety_verdict="healthy",
        )

        pending = service.list_pending_verifications()
        assert pred1.merge_id not in pending
        assert pred2.merge_id in pending

    def test_calibration_stats(self):
        """Can get calibration statistics."""
        service = InterferenceVerificationService()

        # Create and verify several predictions
        for i in range(5):
            pred = service.create_prediction_from_analysis(
                source_model=f"src{i}",
                target_model=f"tgt{i}",
                layer_predictions={0: {"overlap_score": 0.7 + i * 0.05}},
                transformation_counts={},
                config_thresholds={},
            )
            service.verify_from_metrics(
                merge_id=pred.merge_id,
                geometry_metrics={"mean_preserved_fraction": 0.65 + i * 0.05},
                transplant_metrics={},
                safety_verdict="healthy",
            )

        stats = service.get_calibration_stats()
        assert stats.n_verifications == 5

    def test_persistence(self, tmp_path):
        """Service persists data to file."""
        registry_path = tmp_path / "registry.json"
        service = InterferenceVerificationService(registry_path=registry_path)

        pred = service.create_prediction_from_analysis(
            source_model="src",
            target_model="tgt",
            layer_predictions={},
            transformation_counts={},
            config_thresholds={},
        )

        # File should exist now
        assert registry_path.exists()

        # Load in new service instance
        service2 = InterferenceVerificationService(registry_path=registry_path)
        assert pred.merge_id in service2.registry.predictions

    def test_export_calibration_report(self, tmp_path):
        """Can export calibration report to JSON."""
        service = InterferenceVerificationService()

        # Add some data
        for i in range(3):
            pred = service.create_prediction_from_analysis(
                source_model=f"src{i}",
                target_model=f"tgt{i}",
                layer_predictions={},
                transformation_counts={},
                config_thresholds={},
            )
            service.verify_from_metrics(
                merge_id=pred.merge_id,
                geometry_metrics={"mean_preserved_fraction": 0.7},
                transplant_metrics={},
                safety_verdict="healthy",
            )

        report_path = tmp_path / "calibration.json"
        report = service.export_calibration_report(report_path)

        assert report_path.exists()
        assert report["summary"]["n_verifications"] == 3

        # Verify JSON is valid
        with open(report_path) as f:
            loaded = json.load(f)
        assert loaded["summary"]["n_verifications"] == 3


class TestCalibrationStats:
    """Tests for calibration statistics."""

    def test_accuracy_rates(self):
        """Transformation accuracy rates computed correctly."""
        registry = PredictionRegistry()

        # Add 4 merges with varying prediction accuracy
        for i, (pred_alpha, actual_alpha) in enumerate([(3, 2), (0, 0), (1, 0), (2, 3)]):
            pred = MergePrediction(
                merge_id=f"m{i}",
                source_model="src",
                target_model="tgt",
                timestamp="now",
                layer_predictions={},
                predicted_mean_overlap=0.5,
                predicted_mean_curvature_divergence=0.0,
                predicted_mean_alignment=0.8,
                predicted_transformation_counts={"alpha_scaling": pred_alpha},
                config_thresholds={},
            )
            registry.store_prediction(pred)

            verif = MergeVerification(
                merge_id=f"m{i}",
                timestamp="now",
                actual_mean_confidence=0.5,
                actual_preserved_fraction=0.5,
                actual_cka_after=0.8,
                actual_safety_verdict="healthy",
                layer_actuals={},
                actual_transformation_counts={"alpha_scaling": actual_alpha},
            )
            registry.store_verification(verif)

        stats = registry.compute_calibration_stats()

        # i=0: pred=3>0, actual=2>0 → correct (both needed)
        # i=1: pred=0, actual=0 → correct (neither needed)
        # i=2: pred=1>0, actual=0 → incorrect (predicted needed, wasn't)
        # i=3: pred=2>0, actual=3>0 → correct (both needed)
        # 3 correct / 4 total = 0.75
        assert stats.transformation_accuracy_rates["alpha_scaling"] == 0.75

    def test_error_percentile(self):
        """Error percentiles computed correctly."""
        registry = PredictionRegistry()

        # Add 10 merges with known errors
        for i in range(10):
            pred = MergePrediction(
                merge_id=f"p{i}",
                source_model="src",
                target_model="tgt",
                timestamp="now",
                layer_predictions={},
                predicted_mean_overlap=0.5,
                predicted_mean_curvature_divergence=0.0,
                predicted_mean_alignment=0.8,
                predicted_transformation_counts={},
                config_thresholds={},
            )
            registry.store_prediction(pred)

            # Create varying errors
            verif = MergeVerification(
                merge_id=f"p{i}",
                timestamp="now",
                actual_mean_confidence=0.5 + (i * 0.02),
                actual_preserved_fraction=0.5 + (i * 0.02),
                actual_cka_after=0.8,
                actual_safety_verdict="healthy",
                layer_actuals={},
                actual_transformation_counts={},
            )
            registry.store_verification(verif)

        stats = registry.compute_calibration_stats()

        assert stats.n_verifications == 10
        assert stats.error_90th_percentile >= stats.median_absolute_error
