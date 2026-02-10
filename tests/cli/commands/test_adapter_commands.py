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

"""Tests for adapter CLI commands and baseline artifact loading."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app
from modelcypher.core.use_cases import lora_baseline_artifact as baseline_artifacts

runner = CliRunner()


class TestAdapterCommandHelp:
    """Basic adapter command help coverage."""

    def test_adapter_help(self):
        """`mc adapter --help` lists adapter commands."""
        result = runner.invoke(app, ["adapter", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout

    def test_adapter_analyze_help(self):
        """`mc adapter analyze --help` includes baseline artifact option."""
        result = runner.invoke(app, ["adapter", "analyze", "--help"])
        assert result.exit_code == 0
        assert "--baseline-artifact" in result.stdout

    def test_adapter_calibrate_baseline_help(self):
        """`mc adapter calibrate-baseline --help` includes calibration options."""
        result = runner.invoke(app, ["adapter", "calibrate-baseline", "--help"])
        assert result.exit_code == 0
        assert "--four-condition-results" in result.stdout
        assert "--output-artifact" in result.stdout
        assert "--format" in result.stdout


class TestAdapterBaselineArtifact:
    """Baseline artifact loading for adapter metrics."""

    def test_load_reference_baseline_from_artifact(self, tmp_path):
        """Loads measured baseline scalars from artifact JSON."""
        artifact = tmp_path / "summary.json"
        artifact.write_text(
            json.dumps(
                {
                    "experiment_date": "2026-02-05",
                    "findings": {
                        "synthetic_random_baseline": {
                            "amplification_cv": 0.26,
                            "weyl_utilization": 0.054,
                            "source": "Exp 2 four-condition synthetic test",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        baseline = baseline_artifacts.load_reference_baseline(str(artifact))
        assert baseline is not None
        assert baseline["amplification_cv"] == pytest.approx(0.26)
        assert baseline["weyl_utilization"] == pytest.approx(0.054)
        assert baseline["type"] == "synthetic_random_baseline"
        assert baseline["experiment_date"] == "2026-02-05"

    def test_load_reference_baseline_rejects_non_positive_scalars(self, tmp_path):
        """Rejects invalid baseline values that would break ratios."""
        artifact = tmp_path / "summary.json"
        artifact.write_text(
            json.dumps(
                {
                    "findings": {
                        "synthetic_random_baseline": {
                            "amplification_cv": 0.26,
                            "weyl_utilization": 0.0,
                            "source": "invalid",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="must be > 0"):
            baseline_artifacts.load_reference_baseline(str(artifact))

    def test_resolve_baseline_artifact_path_requires_existing_file(self, tmp_path):
        """Explicit artifact path must exist."""
        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError, match="Baseline artifact not found"):
            baseline_artifacts.resolve_baseline_artifact_path(str(missing))

    def test_calibrate_reference_baseline_from_four_condition_results(self, tmp_path):
        """Calibrates baseline artifact from pure_random measurements."""
        four_condition = tmp_path / "raw_measurements.json"
        four_condition.write_text(
            json.dumps(
                {
                    "conditions": {
                        "pure_random": [
                            {
                                "adapter_id": "r1",
                                "mean_amplification_cv": 0.25,
                                "mean_weyl_utilization": 0.05,
                            },
                            {
                                "adapter_id": "r2",
                                "mean_amplification_cv": 0.27,
                                "mean_weyl_utilization": 0.06,
                            },
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        output_artifact = tmp_path / "summary.json"
        payload = baseline_artifacts.calibrate_reference_baseline(
            four_condition_results=str(four_condition),
            output_artifact=str(output_artifact),
            source_label="test calibration",
        )

        assert output_artifact.exists()
        baseline = payload["reference_baseline"]
        assert baseline["amplification_cv"] == pytest.approx(0.26)
        assert baseline["weyl_utilization"] == pytest.approx(0.055)
        assert baseline["sample_count"] == 2
        assert baseline["source"] == "test calibration"
