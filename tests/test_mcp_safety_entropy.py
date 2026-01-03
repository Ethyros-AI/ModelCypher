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
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

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
    env["MC_ALLOW_STUB_INFERENCE"] = "1"
    env["MC_ALLOW_STUB_EMBEDDINGS"] = "1"
    return env


def _extract_structured(result: types.CallToolResult) -> dict:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for content in result.content:
        if isinstance(content, types.TextContent):
            return json.loads(content.text)
    raise AssertionError("No structured content returned from tool call")


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


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
def sample_adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a sample adapter directory for testing."""
    backend = get_default_backend()
    tmp_dir = tmp_path_factory.mktemp("adapters")
    adapter_dir = tmp_dir / "test-adapter"
    adapter_dir.mkdir()

    ones_arr = backend.ones((4, 8), dtype="float32")
    backend.eval(ones_arr)
    weights = {"layer.lora_A": ones_arr}
    backend.save_safetensors(str(adapter_dir / "adapter_model.safetensors"), weights)

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
        assert "outlierLayerFraction" in payload


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


# =============================================================================
# Entropy Verify Baseline Tests
# =============================================================================


class TestEntropyVerifyBaselineTool:
    """Tests for mc_entropy_verify_baseline tool."""

    def test_verify_baseline_schema(self, mcp_env: dict[str, str], tmp_path: Path) -> None:
        """Tool should return properly structured response."""
        observed = [0.1, 0.15, 0.2, -0.05, 0.12]
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "modelId": "test-model",
                    "statistics": {
                        "mean": 0.1,
                        "stdDev": 0.1,
                        "min": -0.1,
                        "max": 0.3,
                    },
                    "sampleCount": 5,
                }
            )
        )

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_verify_baseline",
                    arguments={
                        "baselinePath": str(baseline_path),
                        "observedDeltas": observed,
                        "adapterPath": "/path/to/adapter",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.verify_baseline.v1"


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
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.window.v1"
        assert "samplesProcessed" in payload


# =============================================================================
# Entropy Conversation Track Tests
# =============================================================================


class TestEntropyConversationTrackTool:
    """Tests for mc_entropy_conversation_track tool."""

    def test_conversation_track_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        turns = [
            {
                "tokenCount": 50,
                "avgDelta": 0.1,
                "maxAnomalyScore": 0.2,
                "anomalyCount": 0,
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "tokenCount": 60,
                "avgDelta": 0.12,
                "maxAnomalyScore": 0.18,
                "anomalyCount": 1,
                "timestamp": "2025-01-01T00:01:00Z",
            },
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_conversation_track",
                    arguments={
                        "turns": turns,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.conversation_track.v1"
        assert "turnsProcessed" in payload
        # Raw measurements - no arbitrary classification
        assert "meanDelta" in payload
        assert "stdDelta" in payload
        assert "oscillationAmplitude" in payload
        assert "oscillationFrequency" in payload
        assert "cumulativeDrift" in payload


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
                    arguments={"samples": samples},  # No threshold - raw measurements
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.entropy.dual_path.v1"
        assert "samplesProcessed" in payload
        # Raw statistics derived from the data itself
        assert "meanDelta" in payload
        assert "medianDelta" in payload
        assert "minDelta" in payload
        assert "maxDelta" in payload
        # All samples with measurements - no filtering
        assert "samples" in payload
        assert len(payload["samples"]) == len(samples)

    def test_dual_path_returns_all_samples(self, mcp_env: dict[str, str]) -> None:
        """Tool returns measurements for ALL samples - the geometry speaks."""
        samples = [
            {"base": [1.0, 0.2], "adapter": [5.0, 2.0]},  # Large delta
            {"base": [1.1, 0.25], "adapter": [5.1, 2.1]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_dual_path",
                    arguments={"samples": samples},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        # All samples returned with their delta measurements
        assert len(payload["samples"]) == 2
        # Each sample has raw measurements
        for s in payload["samples"]:
            assert "deltaEntropy" in s
            assert "deltaVariance" in s
            assert "combinedDelta" in s


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
                    arguments={"samples": samples},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["samplesProcessed"] == len(samples)

    def test_dual_path_statistics_bounded(self, mcp_env: dict[str, str]) -> None:
        """Delta statistics should be non-negative."""
        samples = [
            {"base": [1.5, 0.3], "adapter": [1.6, 0.35]},
            {"base": [1.6, 0.35], "adapter": [4.0, 1.5]},  # Large delta
            {"base": [1.7, 0.4], "adapter": [1.8, 0.45]},
        ]

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_entropy_dual_path",
                    arguments={"samples": samples},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        # All delta statistics are non-negative (absolute values)
        eps = _eps(
            payload["minDelta"],
            payload["meanDelta"],
            payload["medianDelta"],
            payload["maxDelta"],
        )
        assert payload["minDelta"] >= -eps
        assert payload["meanDelta"] >= -eps
        assert payload["medianDelta"] >= -eps
        assert payload["maxDelta"] >= -eps
        # min <= median <= max
        assert payload["minDelta"] <= payload["medianDelta"] + eps
        assert payload["medianDelta"] <= payload["maxDelta"] + eps

    def test_conversation_track_turns_processed_matches_input(
        self, mcp_env: dict[str, str]
    ) -> None:
        """Turns processed should match input length."""
        turns = [
            {
                "tokenCount": 40,
                "avgDelta": 0.05,
                "maxAnomalyScore": 0.1,
                "anomalyCount": 0,
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "tokenCount": 45,
                "avgDelta": 0.07,
                "maxAnomalyScore": 0.12,
                "anomalyCount": 1,
                "timestamp": "2025-01-01T00:01:00Z",
            },
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
