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

"""MCP merge entropy validation tool tests.

Tests for merge entropy tools:
- mc_merge_entropy_profile
- mc_merge_entropy_guide
- mc_merge_entropy_validate
- mc_model_validate_knowledge

Set MC_TEST_MODEL_PATH environment variable to a local model path for full testing.
Tests requiring models are skipped if no model is available.
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


def _find_test_model() -> Path | None:
    """Find a model for testing. Returns None if no model available."""
    # Check explicit env var first
    if env_path := os.environ.get("MC_TEST_MODEL_PATH"):
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    # Check MODELCYPHER_HOME/models
    if mc_home := os.environ.get("MODELCYPHER_HOME"):
        models_dir = Path(mc_home) / "models"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    return model_dir

    return None


# Module-level model discovery
_TEST_MODEL = _find_test_model()
requires_model = pytest.mark.skipif(
    _TEST_MODEL is None,
    reason="No test model available (set MC_TEST_MODEL_PATH)",
)


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
    tmp_home = tmp_path_factory.mktemp("mcp_merge_home")
    return _build_env(tmp_home)


@pytest.fixture(scope="module")
def test_model_path() -> str:
    """Return the test model path as a string."""
    assert _TEST_MODEL is not None
    return str(_TEST_MODEL)


# =============================================================================
# Merge Entropy Profile Tests
# =============================================================================


@requires_model
class TestMergeEntropyProfileTool:
    """Tests for mc_merge_entropy_profile tool."""

    def test_entropy_profile_schema(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={
                        "model": test_model_path,
                        "numLayers": 32,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.merge.entropy.profile.v1"
        assert "modelName" in payload
        assert "meanEntropy" in payload
        assert "dominantPhase" in payload
        assert "entropyVariance" in payload
        assert "criticalLayerCount" in payload

    def test_entropy_profile_phase_valid(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Dominant phase should be a valid phase value."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": 16},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        valid_phases = {"ordered", "critical", "disordered"}
        assert payload["dominantPhase"] in valid_phases

    @pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
    def test_entropy_profile_various_layer_counts(
        self, mcp_env: dict[str, str], test_model_path: str, num_layers: int
    ) -> None:
        """Tool should handle different layer counts."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": num_layers},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.merge.entropy.profile.v1"


# =============================================================================
# Merge Entropy Guide Tests
# =============================================================================


@requires_model
class TestMergeEntropyGuideTool:
    """Tests for mc_merge_entropy_guide tool."""

    def test_entropy_guide_schema(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Tool should return properly structured response."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_guide",
                    arguments={
                        "source": test_model_path,
                        "target": test_model_path,
                        "numLayers": 32,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.merge.entropy.guide.v1"
        assert "alphaAdjustments" in payload
        assert "smoothingSigmas" in payload
        assert "alphaStats" in payload
        assert "sigmaStats" in payload

    def test_entropy_guide_alpha_in_bounds(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Global alpha adjustment should be in reasonable bounds."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_guide",
                    arguments={
                        "source": test_model_path,
                        "target": test_model_path,
                        "numLayers": 32,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        # Alpha adjustments should be reasonable values
        for value in payload.get("alphaAdjustments", {}).values():
            assert 0.0 < value <= 2.0


# =============================================================================
# Merge Entropy Validate Tests
# =============================================================================


class TestMergeEntropyValidateTool:
    """Tests for mc_merge_entropy_validate tool."""

    def test_entropy_validate_schema(self, mcp_env: dict[str, str]) -> None:
        """Tool should return properly structured response."""
        source_entropies = {f"layer.{i}": 1.5 + 0.1 * i for i in range(8)}
        target_entropies = {f"layer.{i}": 1.6 + 0.1 * i for i in range(8)}
        merged_entropies = {f"layer.{i}": 1.55 + 0.1 * i for i in range(8)}

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": source_entropies,
                        "targetEntropies": target_entropies,
                        "mergedEntropies": merged_entropies,
                        "sourceModel": "source",
                        "targetModel": "target",
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.merge.entropy.validate.v1"
        assert "knowledgeRetention" in payload
        assert "meanEntropyRatio" in payload
        assert "maxEntropyRatio" in payload
        assert "entropyRatioStd" in payload

    def test_entropy_validate_ratio_ordering(self, mcp_env: dict[str, str]) -> None:
        """Max entropy ratio should be >= mean entropy ratio."""
        source_entropies = {"layer.0": 1.5, "layer.1": 1.6}
        target_entropies = {"layer.0": 1.6, "layer.1": 1.7}
        merged_entropies = {"layer.0": 1.55, "layer.1": 1.65}

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": source_entropies,
                        "targetEntropies": target_entropies,
                        "mergedEntropies": merged_entropies,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["maxEntropyRatio"] >= payload["meanEntropyRatio"]

    def test_entropy_validate_retention_bounded(self, mcp_env: dict[str, str]) -> None:
        """Knowledge retention should be in [0, 1]."""
        source_entropies = {"layer.0": 1.5}
        target_entropies = {"layer.0": 1.6}
        merged_entropies = {"layer.0": 1.55}

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": source_entropies,
                        "targetEntropies": target_entropies,
                        "mergedEntropies": merged_entropies,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert 0.0 <= payload["knowledgeRetention"] <= 1.0


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestMergeEntropyInvariants:
    """Tests for mathematical invariants in merge entropy tools."""

    @requires_model
    def test_profile_entropy_non_negative(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Mean entropy should be non-negative."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": 32},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["meanEntropy"] >= 0.0

    @requires_model
    def test_profile_variance_non_negative(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Entropy variance should be non-negative."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": 32},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["entropyVariance"] >= 0.0

    @requires_model
    def test_profile_critical_layer_count_bounded(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Critical layer count should be bounded by total layers."""
        num_layers = 32

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": num_layers},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert 0 <= payload["criticalLayerCount"] <= num_layers

    @pytest.mark.parametrize("seed", range(3))
    def test_validate_layers_count_matches_input(self, mcp_env: dict[str, str], seed: int) -> None:
        """Total layers validated should match input."""
        backend = get_default_backend()
        backend.random_seed(seed)
        num_layers = int(backend.to_numpy(backend.random_randint(5, 20, (1,)))[0])

        source_entropies = {f"layer.{i}": float(backend.to_numpy(backend.random_uniform(1.0, 3.0, (1,)))[0]) for i in range(num_layers)}
        target_entropies = {f"layer.{i}": float(backend.to_numpy(backend.random_uniform(1.0, 3.0, (1,)))[0]) for i in range(num_layers)}
        merged_entropies = {f"layer.{i}": float(backend.to_numpy(backend.random_uniform(1.0, 3.0, (1,)))[0]) for i in range(num_layers)}

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": source_entropies,
                        "targetEntropies": target_entropies,
                        "mergedEntropies": merged_entropies,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["totalLayersValidated"] == num_layers

    @requires_model
    def test_guide_recommendations_valid_alpha_adjust(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Layer alpha adjustments should be reasonable values."""

        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_guide",
                    arguments={
                        "source": test_model_path,
                        "target": test_model_path,
                        "numLayers": 32,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        for value in payload.get("alphaAdjustments", {}).values():
            assert 0.0 < value <= 2.0
        for value in payload.get("smoothingSigmas", {}).values():
            assert value > 0


# =============================================================================
# Integration-Style Tests
# =============================================================================


@requires_model
class TestMergeWorkflowIntegration:
    """Tests for merge workflow integration across tools."""

    def test_profile_then_guide_workflow(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Workflow: profile → guide should work."""

        async def runner(session: ClientSession):
            # Step 1: Profile source
            profile_result = await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path, "numLayers": 16},
                )
            )

            # Step 2: Get guide for merge
            guide_result = await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_guide",
                    arguments={
                        "source": test_model_path,
                        "target": test_model_path,
                        "numLayers": 16,
                    },
                )
            )

            return profile_result, guide_result

        profile_result, guide_result = _run_mcp(mcp_env, runner)
        profile_payload = _extract_structured(profile_result)
        guide_payload = _extract_structured(guide_result)

        # Both should succeed
        assert profile_payload["_schema"] == "mc.merge.entropy.profile.v1"
        assert guide_payload["_schema"] == "mc.merge.entropy.guide.v1"

        assert guide_payload["layerCount"] >= 0
        assert isinstance(guide_payload.get("alphaAdjustments", {}), dict)

    def test_guide_then_validate_workflow(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        """Workflow: guide → merge (simulated) → validate should work."""

        async def runner(session: ClientSession):
            # Step 1: Get guide
            guide_result = await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_guide",
                    arguments={
                        "source": test_model_path,
                        "target": test_model_path,
                        "numLayers": 8,
                    },
                )
            )

            # Step 2: Simulate merge (just create fake entropies)
            source_entropies = {f"layer.{i}": 1.5 + 0.1 * i for i in range(8)}
            target_entropies = {f"layer.{i}": 1.6 + 0.1 * i for i in range(8)}
            merged_entropies = {f"layer.{i}": 1.55 + 0.1 * i for i in range(8)}

            # Step 3: Validate merge
            validate_result = await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": source_entropies,
                        "targetEntropies": target_entropies,
                        "mergedEntropies": merged_entropies,
                    },
                )
            )

            return guide_result, validate_result

        guide_result, validate_result = _run_mcp(mcp_env, runner)
        guide_payload = _extract_structured(guide_result)
        validate_payload = _extract_structured(validate_result)

        # Both should succeed
        assert guide_payload["_schema"] == "mc.merge.entropy.guide.v1"
        assert validate_payload["_schema"] == "mc.merge.entropy.validate.v1"

        # Validate should report on the merge
        assert validate_payload["totalLayersValidated"] == 8
