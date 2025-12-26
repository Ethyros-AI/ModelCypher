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

"""Integration tests for CLI commands.

Tests Phase 2 CLI commands: safety, entropy, and agent.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


# === Safety Commands ===


def test_safety_adapter_probe_basic(tmp_path):
    """Test safety adapter-probe command with a mock adapter directory."""
    adapter_dir = tmp_path / "test-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "safety",
            "adapter-probe",
            "--adapter",
            str(adapter_dir),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "adapterPath" in data
    assert "layerCount" in data
    assert "isSafe" in data


def test_safety_adapter_probe_missing_adapter():
    """Test safety adapter-probe with non-existent adapter."""
    result = runner.invoke(
        app,
        [
            "safety",
            "adapter-probe",
            "--adapter",
            "/nonexistent/path",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1


# === Entropy Commands ===


def test_entropy_analyze_basic():
    """Test entropy analyze command with sample data."""
    # Requires at least 5 samples for calibration-based analysis
    samples = "[[3.5, 0.2], [3.6, 0.15], [3.4, 0.18], [3.7, 0.22], [3.55, 0.19]]"

    result = runner.invoke(
        app,
        [
            "entropy",
            "analyze",
            samples,
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "trendSlope" in data  # Actual geometric measurement
    assert "entropyMean" in data
    assert "sampleCount" in data
    assert data["sampleCount"] == 5


def test_entropy_analyze_invalid_samples():
    """Test entropy analyze with invalid input."""
    result = runner.invoke(
        app,
        [
            "entropy",
            "analyze",
            "not valid json",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1


def test_entropy_detect_distress_nominal():
    """Test entropy detect-distress with normal samples."""
    samples = "[[2.5, 0.1], [2.6, 0.12], [2.4, 0.09]]"

    result = runner.invoke(
        app,
        [
            "entropy",
            "detect-distress",
            samples,
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0


def test_entropy_verify_baseline():
    """Test entropy verify-baseline command."""
    result = runner.invoke(
        app,
        [
            "entropy",
            "verify-baseline",
            "--mean",
            "0.1",
            "--std-dev",
            "0.05",
            "--max",
            "0.3",
            "--min",
            "0.0",
            "--observed",
            "[0.08, 0.12, 0.09, 0.11]",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "passed" in data or "verdict" in data


def test_entropy_window_basic():
    """Test entropy window sliding analysis."""
    samples = "[[3.0, 0.2], [3.1, 0.21], [3.2, 0.19], [2.9, 0.18]]"

    result = runner.invoke(
        app,
        [
            "entropy",
            "window",
            samples,
            "--size",
            "10",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "currentEntropy" in data  # Actual geometric measurement
    assert "sampleCount" in data
    assert "movingAverage" in data


def test_entropy_window_with_circuit_breaker():
    """Test entropy window with high entropy triggering circuit breaker."""
    # High entropy samples that should trip the circuit breaker
    samples = "[[5.0, 1.0], [5.2, 1.1], [5.5, 1.2], [5.8, 1.3], [6.0, 1.5]]"

    result = runner.invoke(
        app,
        [
            "entropy",
            "window",
            samples,
            "--size",
            "5",
            "--circuit-threshold",
            "5.0",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "shouldTripCircuitBreaker" in data


def test_entropy_conversation_track(tmp_path):
    """Test entropy conversation-track command."""
    session_file = tmp_path / "session.json"
    session_data = {
        "turns": [
            {"token_count": 100, "avg_delta": 0.1, "anomaly_count": 0},
            {"token_count": 150, "avg_delta": 0.12, "anomaly_count": 0},
            {"token_count": 80, "avg_delta": 0.08, "anomaly_count": 0},
        ]
    }
    session_file.write_text(json.dumps(session_data), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "entropy",
            "conversation-track",
            "--session",
            str(session_file),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "turnCount" in data
    assert "oscillationAmplitude" in data  # Actual geometric measurement
    assert "manipulationComponents" in data  # Raw signal components
    assert data["turnCount"] == 3


def test_entropy_conversation_track_missing_file():
    """Test entropy conversation-track with missing file."""
    result = runner.invoke(
        app,
        [
            "entropy",
            "conversation-track",
            "--session",
            "/nonexistent/session.json",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1


def test_entropy_dual_path_nominal():
    """Test entropy dual-path with normal divergence."""
    samples = '[{"base": [3.5, 0.2], "adapter": [3.6, 0.22]}]'

    result = runner.invoke(
        app,
        [
            "entropy",
            "dual-path",
            samples,
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "assessment" in data
    assert "averageDelta" in data
    assert data["assessment"] == "nominal"


def test_entropy_dual_path_suspicious():
    """Test entropy dual-path with suspicious divergence."""
    # High base entropy + low adapter entropy = suspicious pattern
    samples = '[{"base": [5.0, 1.0], "adapter": [1.5, 0.1]}]'

    result = runner.invoke(
        app,
        [
            "entropy",
            "dual-path",
            samples,
            "--anomaly-threshold",
            "0.5",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "anomalyCount" in data
    assert data["anomalyCount"] >= 1


# === Agent Commands ===


def test_agent_trace_import(tmp_path):
    """Test agent trace-import command."""
    trace_file = tmp_path / "traces.json"
    trace_data = {
        "spans": [
            {
                "trace_id": "abc123",
                "span_id": "span1",
                "name": "llm.call",
                "start_time_unix_nano": 1700000000000000000,
                "end_time_unix_nano": 1700000001000000000,
                "attributes": [
                    {"key": "llm.model", "value": {"stringValue": "test-model"}},
                ],
            }
        ]
    }
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "agent",
            "trace-import",
            "--file",
            str(trace_file),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "filePath" in data
    assert "tracesImported" in data


def test_agent_trace_import_missing_file():
    """Test agent trace-import with missing file."""
    result = runner.invoke(
        app,
        [
            "agent",
            "trace-import",
            "--file",
            "/nonexistent/traces.json",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1


def test_agent_trace_analyze(tmp_path):
    """Test agent trace-analyze command."""
    trace_file = tmp_path / "traces.json"
    trace_data = {
        "spans": [
            {
                "trace_id": "abc123",
                "span_id": "span1",
                "name": "llm.call",
                "start_time_unix_nano": 1700000000000000000,
                "end_time_unix_nano": 1700000001000000000,
                "attributes": [
                    {"key": "llm.model", "value": {"stringValue": "test-model"}},
                ],
            }
        ]
    }
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "agent",
            "trace-analyze",
            "--file",
            str(trace_file),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "traceCount" in data
    assert "totalSpans" in data
    assert "kinds" in data
    assert "statuses" in data


def test_agent_validate_action_valid():
    """Test agent validate-action with valid response action."""
    action = '{"kind": "response", "content": "Hello, I can help you with that."}'

    result = runner.invoke(
        app,
        [
            "agent",
            "validate-action",
            action,
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "valid" in data
    assert "kind" in data
    assert data["kind"] == "response"


def test_agent_validate_action_tool_call():
    """Test agent validate-action with tool call action."""
    action = '{"kind": "tool_call", "tool": "search", "input": {"query": "test"}}'

    result = runner.invoke(
        app,
        [
            "agent",
            "validate-action",
            action,
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "valid" in data
    assert "kind" in data


def test_agent_validate_action_invalid_json():
    """Test agent validate-action with invalid JSON."""
    result = runner.invoke(
        app,
        [
            "agent",
            "validate-action",
            "not valid json",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 1


# === Text Output Format Tests ===


def test_safety_adapter_probe_text_output(tmp_path):
    """Test safety adapter-probe with text output."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "safety",
            "adapter-probe",
            "--adapter",
            str(adapter_dir),
            "--output",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert "ADAPTER SAFETY PROBE" in result.stdout


def test_entropy_window_text_output():
    """Test entropy window with text output."""
    samples = "[[3.0, 0.2], [3.1, 0.21]]"

    result = runner.invoke(
        app,
        [
            "entropy",
            "window",
            samples,
            "--output",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert "ENTROPY WINDOW ANALYSIS" in result.stdout


def test_entropy_dual_path_text_output():
    """Test entropy dual-path with text output."""
    samples = '[{"base": [3.5, 0.2], "adapter": [3.6, 0.22]}]'

    result = runner.invoke(
        app,
        [
            "entropy",
            "dual-path",
            samples,
            "--output",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert "DUAL-PATH ENTROPY ANALYSIS" in result.stdout


def test_agent_validate_action_text_output():
    """Test agent validate-action with text output."""
    action = '{"kind": "response", "content": "Hello"}'

    result = runner.invoke(
        app,
        [
            "agent",
            "validate-action",
            action,
            "--output",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert "ACTION VALIDATION RESULT" in result.stdout


# === Help Tests ===


def test_safety_help():
    """Test safety command help."""
    result = runner.invoke(app, ["safety", "--help"])
    assert result.exit_code == 0
    assert "adapter-probe" in result.stdout


def test_entropy_help():
    """Test entropy command help."""
    result = runner.invoke(app, ["entropy", "--help"])
    assert result.exit_code == 0
    assert "analyze" in result.stdout
    assert "window" in result.stdout


def test_agent_help():
    """Test agent command help."""
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "trace-import" in result.stdout
    assert "trace-analyze" in result.stdout
    assert "validate-action" in result.stdout
