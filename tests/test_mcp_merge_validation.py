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

"""MCP merge entropy validation tool tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

DEFAULT_TIMEOUT_SECONDS = 15


def _find_test_model() -> Path | None:
    """Find a model for testing. Returns None if no model available."""
    if env_path := os.environ.get("MC_TEST_MODEL_PATH"):
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    if mc_home := os.environ.get("MODELCYPHER_HOME"):
        models_dir = Path(mc_home) / "models"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    return model_dir

    return None


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
    assert _TEST_MODEL is not None
    return str(_TEST_MODEL)


@requires_model
class TestMergeEntropyProfileTool:
    """Tests for mc_merge_entropy_profile tool."""

    def test_entropy_profile_schema(self, mcp_env: dict[str, str], test_model_path: str) -> None:
        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_profile",
                    arguments={"model": test_model_path},
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.merge.entropy.profile.v1"
        assert "modelName" in payload
        assert "meanEntropy" in payload
        assert "entropyVariance" in payload
        assert "layerCount" in payload
        assert "topEntropyLayers" in payload


class TestMergeEntropyValidateTool:
    """Tests for mc_merge_entropy_validate tool."""

    def test_entropy_validate_schema(self, mcp_env: dict[str, str]) -> None:
        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_merge_entropy_validate",
                    arguments={
                        "sourceEntropies": {"layers.0": 2.0, "layers.1": 2.5},
                        "targetEntropies": {"layers.0": 2.2, "layers.1": 2.4},
                        "mergedEntropies": {"layers.0": 2.1, "layers.1": 2.45},
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
        assert payload["totalLayersValidated"] == 2
        assert isinstance(payload.get("topEntropyRatioLayers", []), list)


@requires_model
class TestKnowledgeValidationTool:
    """Tests for mc_model_validate_knowledge tool."""

    def test_knowledge_validation_schema(
        self, mcp_env: dict[str, str], test_model_path: str
    ) -> None:
        async def runner(session: ClientSession):
            return await _await_with_timeout(
                session.call_tool(
                    "mc_model_validate_knowledge",
                    arguments={
                        "sourceModel": test_model_path,
                        "mergedModel": test_model_path,
                    },
                )
            )

        result = _run_mcp(mcp_env, runner)
        payload = _extract_structured(result)

        assert payload["_schema"] == "mc.model.validate_knowledge.v1"
        assert "overallRetention" in payload
        assert "perDomain" in payload
