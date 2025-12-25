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

"""MCP safety and entropy tool tests.

Tests for safety-related tools:
- mc_safety_circuit_breaker
- mc_safety_persona_drift
- mc_safety_redteam_scan
- mc_safety_behavioral_probe
- mc_safety_adapter_probe
- mc_safety_dataset_scan
- mc_safety_lint_identity

Tests for entropy-related tools:
- mc_entropy_analyze
- mc_entropy_detect_distress
- mc_entropy_verify_baseline
- mc_entropy_window
- mc_entropy_conversation_track
- mc_entropy_dual_path
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from modelcypher.core.domain._backend import get_default_backend

DEFAULT_TIMEOUT_SECONDS = 15


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_env(tmp_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = _repo_root()
    python_path = os.pathsep.join(
        path for path in [str(repo_root / "src"), env.get("PYTHONPATH")] if path
    )
    env["PYTHONPATH"] = python_path
    env["MODELCYPHER_HOME"] = str(tmp_home)
    env["MC_MCP_PROFILE"] = "full"
    return env


def _extract_structured(result: types.CallToolResult) -> dict:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for content in result.content:
        if isinstance(content, types.TextContent):
            return json.loads(content.text)
    raise AssertionError("No structured content returned from tool call")


async def _await_with_timeout(coro, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    return await asyncio.wait_for(coro, timeout=timeout)


def _run_mcp(env: dict[str, str], runner):
    async def _run():
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "modelcypher.mcp.server"],
            env=env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await _await_with_timeout(session.initialize())
                return await runner(session)

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def mcp_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_home = tmp_path_factory.mktemp("mcp_safety_home")
    return _build_env(tmp_home)


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a sample JSONL dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("datasets")
    dataset_path = tmp_dir / "sample.jsonl"
    samples = [
        {"text": "Hello, how are you?"},
        {"text": "What is machine learning?"},
        {"text": "The quick brown fox jumps over the lazy dog."},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(s) for s in samples),
        encoding="utf-8",
    )
    return dataset_path


@pytest.fixture(scope="module")
def sample_adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a sample adapter directory for testing."""
    from safetensors.numpy import save_file

    backend = get_default_backend()
    tmp_dir = tmp_path_factory.mktemp("adapters")
    adapter_dir = tmp_dir / "test-adapter"
    adapter_dir.mkdir()

    ones_arr = backend.ones((4, 8), dtype="float32")
    backend.eval(ones_arr)
    weights = {"layer.lora_A": backend.to_numpy(ones_arr)}
    save_file(weights, adapter_dir / "adapter_model.safetensors")

    config = {"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj", "v_proj"]}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    return adapter_dir


# =============================================================================
# Safety Circuit Breaker Tests
# =============================================================================


class TestSafetyCircuitBreakerTool:
    """Tests for mc_safety_circuit_breaker tool."""

    def test_circuit_breaker_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_circuit_breaker",
                    arguments={
                        "adapterName": "test-adapter",
                        "adapterDescription": "A test adapter for safety testing",
                        "skillTags": ["general", "reasoning"],
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.circuit_breaker.v1"
        assert "nextActions" in payload


# =============================================================================
# Safety Persona Drift Tests
# =============================================================================


class TestSafetyPersonaDriftTool:
    """Tests for mc_safety_persona_drift tool."""

    def test_persona_drift_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        baseline = {
            "helpfulness": 0.9,
            "harmlessness": 0.95,
            "honesty": 0.85,
        }
        current_behavior = [
            "I'd be happy to help you with that question.",
            "Let me explain how this works.",
            "That's an interesting perspective.",
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_persona_drift",
                    arguments={
                        "baselinePersona": baseline,
                        "currentBehavior": current_behavior,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.persona_drift.v1"
        assert "nextActions" in payload


# =============================================================================
# Safety Redteam Scan Tests
# =============================================================================


class TestSafetyRedteamScanTool:
    """Tests for mc_safety_redteam_scan tool."""

    def test_redteam_scan_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_redteam_scan",
                    arguments={
                        "name": "suspicious-adapter",
                        "description": "An adapter that might do bad things",
                        "skillTags": ["jailbreak", "uncensored"],
                        "creator": "unknown",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.redteam_scan.v1"
        assert "nextActions" in payload


# =============================================================================
# Safety Behavioral Probe Tests
# =============================================================================


class TestSafetyBehavioralProbeTool:
    """Tests for mc_safety_behavioral_probe tool."""

    def test_behavioral_probe_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_behavioral_probe",
                    arguments={
                        "name": "test-adapter",
                        "tier": "quick",
                        "description": "Test adapter for behavioral probing",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.behavioral_probe.v1"
        assert "nextActions" in payload

    @pytest.mark.parametrize("tier", ["quick", "standard", "full"])
    def test_behavioral_probe_tiers(self, mcp_env: dict[str, str], tier: str) -> None:
        """Tool should accept different safety tiers."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_behavioral_probe",
                    arguments={"name": "test-adapter", "tier": tier},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.behavioral_probe.v1"


# =============================================================================
# Safety Adapter Probe Tests
# =============================================================================


class TestSafetyAdapterProbeTool:
    """Tests for mc_safety_adapter_probe tool."""

    def test_adapter_probe_schema(self, mcp_env: dict[str, str], sample_adapter: Path) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_adapter_probe",
                    arguments={"adapterPath": str(sample_adapter)},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.adapter_probe.v1"
        assert "layerCount" in payload
        # Raw measurements - no arbitrary "isSafe" classification
        assert "maxL2Norm" in payload
        assert "meanL2Norm" in payload
        assert "suspectLayerFraction" in payload
        assert "nextActions" in payload


# =============================================================================
# Safety Dataset Scan Tests
# =============================================================================


class TestSafetyDatasetScanTool:
    """Tests for mc_safety_dataset_scan tool."""

    def test_dataset_scan_schema(self, mcp_env: dict[str, str], sample_dataset: Path) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_dataset_scan",
                    arguments={"datasetPath": str(sample_dataset)},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.dataset_scan.v1"
        assert "samplesScanned" in payload
        assert "passed" in payload
        assert "nextActions" in payload


# =============================================================================
# Safety Lint Identity Tests
# =============================================================================


class TestSafetyLintIdentityTool:
    """Tests for mc_safety_lint_identity tool."""

    def test_lint_identity_schema(self, mcp_env: dict[str, str], sample_dataset: Path) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_safety_lint_identity",
                    arguments={"datasetPath": str(sample_dataset)},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.safety.lint_identity.v1"
        assert "samplesChecked" in payload
        assert "passed" in payload
        assert "nextActions" in payload


# =============================================================================
# Entropy Analyze Tests
# =============================================================================


class TestEntropyAnalyzeTool:
    """Tests for mc_entropy_analyze tool."""

    def test_entropy_analyze_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        samples = [[1.5, 0.3], [1.6, 0.35], [1.7, 0.4], [1.8, 0.45]]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_analyze",
                    arguments={"samples": samples},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.analyze.v1"
        assert "nextActions" in payload


# =============================================================================
# Entropy Detect Distress Tests
# =============================================================================


class TestEntropyDetectDistressTool:
    """Tests for mc_entropy_detect_distress tool."""

    def test_detect_distress_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        # High entropy samples to simulate distress
        samples = [[3.5, 1.2], [3.8, 1.5], [4.0, 1.8], [4.2, 2.0]]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_detect_distress",
                    arguments={"samples": samples},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.detect_distress.v1"
        assert "nextActions" in payload


# =============================================================================
# Entropy Verify Baseline Tests
# =============================================================================


class TestEntropyVerifyBaselineTool:
    """Tests for mc_entropy_verify_baseline tool."""

    def test_verify_baseline_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        observed = [0.1, 0.15, 0.2, -0.05, 0.12]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_verify_baseline",
                    arguments={
                        "declaredMean": 0.1,
                        "declaredStdDev": 0.1,
                        "declaredMax": 0.3,
                        "declaredMin": -0.1,
                        "observedDeltas": observed,
                        "baseModelId": "test-model",
                        "adapterPath": "/path/to/adapter",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.verify_baseline.v1"
        assert "nextActions" in payload


# =============================================================================
# Entropy Window Tests
# =============================================================================


class TestEntropyWindowTool:
    """Tests for mc_entropy_window tool."""

    def test_entropy_window_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        samples = [[1.5, 0.3], [1.6, 0.35], [1.7, 0.4], [1.8, 0.45], [1.9, 0.5]]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_window",
                    arguments={
                        "samples": samples,
                        "windowSize": 3,
                        "highThreshold": 3.0,
                        "circuitThreshold": 4.0,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.window.v1"
        assert "samplesProcessed" in payload
        assert "circuitBreakerTripped" in payload
        assert "nextActions" in payload

    def test_entropy_window_circuit_breaker_trips(self, mcp_env: dict[str, str]) -> None:
        """Circuit breaker should trip on very high entropy."""
        # Very high entropy samples
        samples = [[5.0, 2.0], [5.5, 2.5], [6.0, 3.0]]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_window",
                    arguments={
                        "samples": samples,
                        "windowSize": 3,
                        "highThreshold": 3.0,
                        "circuitThreshold": 4.0,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["circuitBreakerTripped"] is True


# =============================================================================
# Entropy Conversation Track Tests
# =============================================================================


class TestEntropyConversationTrackTool:
    """Tests for mc_entropy_conversation_track tool."""

    def test_conversation_track_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        turns = [
            {"role": "user", "entropy": 1.5, "variance": 0.3},
            {"role": "assistant", "entropy": 1.6, "variance": 0.35},
            {"role": "user", "entropy": 1.7, "variance": 0.4},
            {"role": "assistant", "entropy": 1.8, "variance": 0.45},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_conversation_track",
                    arguments={
                        "turns": turns,
                        "oscillationThreshold": 0.8,
                        "driftThreshold": 1.5,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.conversation_track.v1"
        assert "turnsProcessed" in payload
        # Raw measurements - no arbitrary "oscillationDetected" classification
        assert "oscillationAmplitude" in payload
        assert "oscillationFrequency" in payload
        assert "cumulativeDrift" in payload
        assert "nextActions" in payload


# =============================================================================
# Entropy Dual Path Tests
# =============================================================================


class TestEntropyDualPathTool:
    """Tests for mc_entropy_dual_path tool."""

    def test_dual_path_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        samples = [
            {"base": [1.5, 0.3], "adapter": [1.6, 0.35]},
            {"base": [1.6, 0.35], "adapter": [1.7, 0.4]},
            {"base": [1.7, 0.4], "adapter": [1.8, 0.45]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_dual_path",
                    arguments={
                        "samples": samples,
                        "anomalyThreshold": 0.6,
                        "deltaThreshold": 1.0,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.dual_path.v1"
        assert "samplesProcessed" in payload
        assert "anomalyCount" in payload
        # Raw measurements - no arbitrary "verdict" classification
        assert "anomalyRate" in payload
        assert "deltaThreshold" in payload
        assert "nextActions" in payload

    def test_dual_path_detects_anomalies(self, mcp_env: dict[str, str]) -> None:
        """Large entropy delta should be flagged as anomaly."""
        samples = [
            {"base": [1.0, 0.2], "adapter": [5.0, 2.0]},  # Large delta
            {"base": [1.1, 0.25], "adapter": [5.1, 2.1]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_dual_path",
                    arguments={
                        "samples": samples,
                        "deltaThreshold": 1.0,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        # Raw measurements - check anomalyCount directly, not a boolean "hasAnomalies"
        assert payload["anomalyCount"] >= 1
        assert payload["anomalyRate"] > 0


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestSafetyEntropyInvariants:
    """Tests for mathematical invariants in safety/entropy tools."""

    def test_entropy_window_samples_processed_matches_input(self, mcp_env: dict[str, str]) -> None:
        """Samples processed should match input length."""
        samples = [[1.0, 0.1]] * 10

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_window",
                    arguments={"samples": samples, "windowSize": 5},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["samplesProcessed"] == len(samples)

    def test_dual_path_anomaly_rate_bounded(self, mcp_env: dict[str, str]) -> None:
        """Anomaly rate should be in [0, 1]."""
        samples = [
            {"base": [1.5, 0.3], "adapter": [1.6, 0.35]},
            {"base": [1.6, 0.35], "adapter": [4.0, 1.5]},  # Anomaly
            {"base": [1.7, 0.4], "adapter": [1.8, 0.45]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_dual_path",
                    arguments={"samples": samples, "deltaThreshold": 1.0},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert 0.0 <= payload["anomalyRate"] <= 1.0

    def test_conversation_track_turns_processed_matches_input(
        self, mcp_env: dict[str, str]
    ) -> None:
        """Turns processed should match input length."""
        turns = [
            {"role": "user", "entropy": 1.5, "variance": 0.3},
            {"role": "assistant", "entropy": 1.6, "variance": 0.35},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_conversation_track",
                    arguments={"turns": turns},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["turnsProcessed"] == len(turns)
