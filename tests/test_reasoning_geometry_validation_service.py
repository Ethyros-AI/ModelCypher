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

"""Tests for reasoning geometry validation service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelcypher.core.use_cases.reasoning_geometry_validation_service import (
    ReasoningGeometryValidationRequest,
    run_reasoning_geometry_validation,
)


def test_validation_service_builds_expected_command(tmp_path, monkeypatch):
    script_path = tmp_path / "reasoning_geometry_validation.py"
    script_path.write_text("print('ok')\n")

    captured: dict[str, list[str]] = {}

    def fake_run(command, check, capture_output, text):
        captured["command"] = command
        assert check is False
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "modelcypher.core.use_cases.reasoning_geometry_validation_service._validation_script_path",
        lambda: script_path,
    )
    monkeypatch.setattr(
        "modelcypher.core.use_cases.reasoning_geometry_validation_service.subprocess.run",
        fake_run,
    )

    request = ReasoningGeometryValidationRequest(
        models=("LFM2-350M",),
        benchmarks=("arithmetic",),
        samples=20,
        max_tokens=64,
        seed=7,
        batch_size=8,
        output_dir=tmp_path / "results",
    )
    result = run_reasoning_geometry_validation(request)

    command = captured["command"]
    assert "--models" in command
    assert "LFM2-350M" in command
    assert "--benchmark" in command
    assert "arithmetic" in command
    assert result.output_dir == (tmp_path / "results").resolve()
    assert result.report_path.name == "VALIDATION_REPORT.md"
    assert result.results_path.name == "per_model_results.json"


def test_validation_service_raises_on_nonzero_exit(tmp_path, monkeypatch):
    script_path = tmp_path / "reasoning_geometry_validation.py"
    script_path.write_text("print('ok')\n")

    def fake_run(command, check, capture_output, text):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(
        "modelcypher.core.use_cases.reasoning_geometry_validation_service._validation_script_path",
        lambda: script_path,
    )
    monkeypatch.setattr(
        "modelcypher.core.use_cases.reasoning_geometry_validation_service.subprocess.run",
        fake_run,
    )

    request = ReasoningGeometryValidationRequest(output_dir=tmp_path / "results")
    with pytest.raises(RuntimeError, match="Reasoning geometry validation failed"):
        run_reasoning_geometry_validation(request)

