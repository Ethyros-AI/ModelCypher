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

"""Tests for thermo calibrator (thermodynamic parameter calibration)."""

from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    BehavioralOutcome,
    LinguisticModifier,
)
from modelcypher.core.domain.thermo.thermo_calibrator import (
    CalibrationProgress,
    ThermoCalibrator,
    get_default_calibration_probes,
)


class TestCalibrationProgress:
    """Tests for CalibrationProgress dataclass."""

    def test_defaults(self):
        progress = CalibrationProgress()
        assert progress.baseline_measured == 0
        assert progress.modifier_measured == {}
        assert progress.outcomes_observed == {}
        assert isinstance(progress.started_at, datetime)

    def test_update_outcome(self):
        progress = CalibrationProgress()
        progress.update_outcome(BehavioralOutcome.REFUSED)
        assert progress.outcomes_observed["refused"] == 1

        progress.update_outcome(BehavioralOutcome.REFUSED)
        assert progress.outcomes_observed["refused"] == 2

        progress.update_outcome(BehavioralOutcome.SOLVED)
        assert progress.outcomes_observed["solved"] == 1

    def test_update_modifier(self):
        progress = CalibrationProgress()
        progress.update_modifier(LinguisticModifier.POLITE)
        assert progress.modifier_measured["polite"] == 1

        progress.update_modifier(LinguisticModifier.POLITE)
        assert progress.modifier_measured["polite"] == 2

        progress.update_modifier(LinguisticModifier.DIRECT)
        assert progress.modifier_measured["direct"] == 1

    def test_all_outcomes_tracked(self):
        progress = CalibrationProgress()
        for outcome in BehavioralOutcome:
            progress.update_outcome(outcome)
        assert len(progress.outcomes_observed) == len(BehavioralOutcome)


class TestThermoCalibrator:
    """Tests for ThermoCalibrator class."""

    def test_init_with_model_path(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        assert calibrator.model_path == model_path
        assert calibrator.adapter_path is None
        assert calibrator.model_id == "model"

    def test_init_with_adapter_path(self, tmp_path):
        model_path = tmp_path / "model"
        adapter_path = tmp_path / "adapter"
        model_path.mkdir()
        adapter_path.mkdir()
        calibrator = ThermoCalibrator(model_path, adapter_path)
        assert calibrator.adapter_path == adapter_path

    def test_model_id_from_path(self, tmp_path):
        model_path = tmp_path / "my-cool-model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        assert calibrator.model_id == "my-cool-model"

    def test_calibrate_empty_probes_raises(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        with pytest.raises(ValueError, match="empty probe corpus"):
            calibrator.calibrate([])

    def test_calibrate_thresholds_insufficient_samples(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        # With fewer baseline samples than required
        result = calibrator._calibrate_thresholds([1.0, 2.0, 3.0])
        # Should return None due to insufficient samples (default requires 50)
        assert result is None

    def test_calibrate_modifier_profile_empty(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        result = calibrator._calibrate_modifier_profile([], temperature=0.7)
        assert result is None

    def test_calibrate_basin_topology_insufficient(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        calibrator = ThermoCalibrator(model_path)
        # With fewer than 10 total outcomes
        result = calibrator._calibrate_basin_topology({"refused": 2, "solved": 3})
        assert result is None


class TestGetDefaultCalibrationProbes:
    """Tests for get_default_calibration_probes function."""

    def test_returns_list(self):
        probes = get_default_calibration_probes()
        assert isinstance(probes, list)

    def test_not_empty(self):
        probes = get_default_calibration_probes()
        assert len(probes) > 0

    def test_all_strings(self):
        probes = get_default_calibration_probes()
        for probe in probes:
            assert isinstance(probe, str)

    def test_contains_solution_territory(self):
        probes = get_default_calibration_probes()
        # Check for basic math question
        assert any("2 + 2" in p for p in probes)

    def test_contains_moderate_territory(self):
        probes = get_default_calibration_probes()
        # Check for explanation prompts
        assert any("gravity" in p.lower() for p in probes)

    def test_contains_edge_territory(self):
        probes = get_default_calibration_probes()
        # Check for security-related prompts
        assert any("security" in p.lower() for p in probes)

    def test_contains_refusal_territory(self):
        probes = get_default_calibration_probes()
        # Check for prompts designed to test safety
        assert any("malware" in p.lower() for p in probes)

    def test_contains_creative_territory(self):
        probes = get_default_calibration_probes()
        # Check for creative prompts
        assert any("haiku" in p.lower() for p in probes)

    def test_probe_count_reasonable(self):
        probes = get_default_calibration_probes()
        # Should have a reasonable number of probes
        assert 20 <= len(probes) <= 100
