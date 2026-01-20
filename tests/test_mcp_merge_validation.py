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

from tests.fixtures.models import ensure_model

DEFAULT_TIMEOUT_SECONDS = 15

pytestmark = [pytest.mark.mlx, pytest.mark.real_model]

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
    return str(ensure_model())


@pytest.fixture(scope="module")
def mcp_payloads(
    mcp_env: dict[str, str],
    test_model_path: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    tmp_root = tmp_path_factory.mktemp("mcp_merge_validation")

    async def runner(session: ClientSession):
        entropy_profile = await _await_with_timeout(
            session.call_tool(
                "mc_merge_entropy_profile",
                arguments={"model": test_model_path},
            )
        )
        entropy_validate = await _await_with_timeout(
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
        knowledge_validate = await _await_with_timeout(
            session.call_tool(
                "mc_model_validate_knowledge",
                arguments={
                    "sourceModel": test_model_path,
                    "mergedModel": test_model_path,
                },
            )
        )
        return {
            "mc_merge_entropy_profile": entropy_profile,
            "mc_merge_entropy_validate": entropy_validate,
            "mc_model_validate_knowledge": knowledge_validate,
        }

    results = _run_mcp(mcp_env, runner)
    return {
        "mc_merge_entropy_profile": _extract_structured(results["mc_merge_entropy_profile"]),
        "mc_merge_entropy_validate": _extract_structured(results["mc_merge_entropy_validate"]),
        "mc_model_validate_knowledge": _extract_structured(results["mc_model_validate_knowledge"]),
    }


class TestMergeEntropyProfileTool:
    """Tests for mc_merge_entropy_profile tool."""

    def test_entropy_profile_schema(self, mcp_payloads: dict[str, object]) -> None:
        payload = mcp_payloads["mc_merge_entropy_profile"]

        assert payload["_schema"] == "mc.merge.entropy.profile.v1"
        assert "modelName" in payload
        assert "meanEntropy" in payload
        assert "entropyVariance" in payload
        assert "layerCount" in payload
        assert "topEntropyLayers" in payload


class TestMergeEntropyValidateTool:
    """Tests for mc_merge_entropy_validate tool."""

    def test_entropy_validate_schema(self, mcp_payloads: dict[str, object]) -> None:
        payload = mcp_payloads["mc_merge_entropy_validate"]

        assert payload["_schema"] == "mc.merge.entropy.validate.v1"
        assert "knowledgeRetention" in payload
        assert "meanEntropyRatio" in payload
        assert "maxEntropyRatio" in payload
        assert "entropyRatioStd" in payload
        assert payload["totalLayersValidated"] == 2
        assert isinstance(payload.get("topEntropyRatioLayers", []), list)


class TestKnowledgeValidationTool:
    """Tests for mc_model_validate_knowledge tool."""

    def test_knowledge_validation_schema(self, mcp_payloads: dict[str, object]) -> None:
        payload = mcp_payloads["mc_model_validate_knowledge"]
        assert payload["_schema"] == "mc.model.validate_knowledge.v1"
        assert "overallRetention" in payload
        assert "perDomain" in payload
