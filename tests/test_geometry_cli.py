from __future__ import annotations

import json

from typer.testing import CliRunner

from modelcypher.cli.app import app


runner = CliRunner()


def test_geometry_validate_cli():
    result = runner.invoke(app, ["geometry", "validate", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["_schema"] == "tc.geometry.validation.v1"
    assert "gromovWasserstein" in payload


def test_geometry_path_detect_cli():
    result = runner.invoke(
        app,
        ["geometry", "path", "detect", "def sum(a, b): return a + b", "--output", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["modelID"] == "input-text"
    assert "detectedGates" in payload


def test_geometry_path_compare_cli():
    result = runner.invoke(
        app,
        [
            "geometry",
            "path",
            "compare",
            "--text-a",
            "def f(x): return x + 1",
            "--text-b",
            "f = lambda x: x + 1",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["modelA"] == "text-a"
    assert "rawDistance" in payload
