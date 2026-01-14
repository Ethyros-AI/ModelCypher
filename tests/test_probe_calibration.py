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

"""Tests for probe_calibration.py - Probe calibration via CKA measurement.

Tests cover:
- ProbeCalibrationResult, CalibrationReport dataclasses
- ProbeCalibrator.compute_cka() method
- ProbeCalibrator.calibrate_probe() method
- ProbeCalibrator.generate_calibration_report() method
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.probe_calibration import (
    CalibrationReport,
    ProbeCalibrationResult,
    ProbeCalibrator,
)


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestProbeCalibrationResult:
    """Tests for ProbeCalibrationResult dataclass."""

    def test_fields_stored(self):
        """ProbeCalibrationResult stores all fields."""
        result = ProbeCalibrationResult(
            probe_id="test_probe",
            measured_cka=0.95,
            cka_std=0.02,
            n_model_pairs=5,
            min_cka=0.90,
            max_cka=0.99,
        )
        assert result.probe_id == "test_probe"
        assert result.measured_cka == 0.95
        assert result.n_model_pairs == 5


class TestCalibrationReport:
    """Tests for CalibrationReport dataclass."""

    def test_fields_stored(self):
        """CalibrationReport stores all fields."""
        probe_result = ProbeCalibrationResult("p1", 0.9, 0.01, 3, 0.88, 0.92)
        report = CalibrationReport(
            per_probe_results={"p1": probe_result},
            model_pairs_used=[("A", "B")],
            mean_cka=0.9,
        )
        assert "p1" in report.per_probe_results
        assert report.mean_cka == 0.9


# =============================================================================
# ProbeCalibrator.compute_cka Tests
# =============================================================================


class TestProbeCalibrator:
    """Tests for ProbeCalibrator class."""

    def test_compute_cka_identical(self):
        """compute_cka with identical activations returns ~1.0."""
        import math
        calibrator = ProbeCalibrator()
        activations = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]

        cka = calibrator.compute_cka(activations, activations)

        # Threshold derived from machine epsilon: 1.0 - sqrt(eps)
        # For float64, eps ~= 2.2e-16, sqrt(eps) ~= 1.5e-8
        eps = 2.220446049250313e-16  # sys.float_info.epsilon
        numerical_identity_threshold = 1.0 - math.sqrt(eps)
        assert cka >= numerical_identity_threshold, (
            f"CKA of identical data ({cka}) below numerical identity threshold "
            f"({numerical_identity_threshold})"
        )

    def test_compute_cka_range(self):
        """compute_cka returns value in [0, 1]."""
        calibrator = ProbeCalibrator()
        activations_a = [[1.0, 0.0], [0.0, 1.0]]
        activations_b = [[0.5, 0.5], [0.3, 0.7]]
        
        cka = calibrator.compute_cka(activations_a, activations_b)
        
        assert 0.0 <= cka <= 1.0

    def test_calibrate_probe_single_model(self):
        """calibrate_probe with single model returns result."""
        calibrator = ProbeCalibrator()
        activations = [("model_A", [[1.0, 0.0], [0.0, 1.0]])]
        
        result = calibrator.calibrate_probe(
            probe_id="test",
            probe_texts=["text1", "text2"],
            model_activations=activations,
        )
        
        assert result.probe_id == "test"
        assert result.n_model_pairs == 0  # Need 2+ models for pairs

    def test_calibrate_probe_multiple_models(self):
        """calibrate_probe with multiple models computes CKA."""
        calibrator = ProbeCalibrator()
        model_activations = [
            ("model_A", [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            ("model_B", [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        ]
        
        result = calibrator.calibrate_probe(
            probe_id="multi_probe",
            probe_texts=["t1", "t2", "t3"],
            model_activations=model_activations,
        )
        
        assert result.probe_id == "multi_probe"
        assert result.n_model_pairs == 1  # A-B pair

    def test_generate_calibration_report(self):
        """generate_calibration_report creates valid report."""
        calibrator = ProbeCalibrator()
        results = [
            ProbeCalibrationResult("p1", 0.9, 0.01, 1, 0.9, 0.9),
            ProbeCalibrationResult("p2", 0.8, 0.02, 1, 0.8, 0.8),
        ]
        model_pairs = [("A", "B")]
        
        report = calibrator.generate_calibration_report(results, model_pairs)
        
        assert isinstance(report, CalibrationReport)
        assert "p1" in report.per_probe_results
        assert "p2" in report.per_probe_results
