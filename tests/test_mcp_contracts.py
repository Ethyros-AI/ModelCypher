from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import AnyUrl

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


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
    env["TC_MCP_PROFILE"] = "training"
    return env


def _extract_structured(result: types.CallToolResult) -> dict:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for content in result.content:
        if isinstance(content, types.TextContent):
            return json.loads(content.text)
    raise AssertionError("No structured content returned from tool call")


@pytest.fixture(scope="module")
async def mcp_session(tmp_path_factory: pytest.TempPathFactory):
    tmp_home = tmp_path_factory.mktemp("mcp_home")
    env = _build_env(tmp_home)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "modelcypher.mcp.server"],
        env=env,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_tool_list_includes_core_tools(mcp_session: ClientSession):
    tool_list = await mcp_session.list_tools()
    names = {tool.name for tool in tool_list.tools}
    assert "tc_inventory" in names
    assert "tc_system_status" in names
    assert "tc_model_list" in names
    assert "tc_dataset_validate" in names
    assert "tc_job_list" in names


async def test_tc_inventory_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_inventory", arguments={})
    payload = _extract_structured(result)
    assert "models" in payload
    assert "datasets" in payload
    assert "checkpoints" in payload
    assert "jobs" in payload
    assert "workspace" in payload
    assert "mlxVersion" in payload
    assert "policies" in payload


async def test_tc_system_status_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_system_status", arguments={})
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.system.status.v1"
    assert "machineName" in payload
    assert "unifiedMemoryGB" in payload
    assert "readinessScore" in payload
    assert "scoreBreakdown" in payload
    assert "blockers" in payload
    assert "nextActions" in payload


async def test_tc_model_list_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_model_list", arguments={})
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.model.list.v1"
    assert "models" in payload
    assert "count" in payload
    assert "nextActions" in payload


async def test_tc_dataset_validate_schema(mcp_session: ClientSession, tmp_path: Path):
    dataset_path = tmp_path / "sample.jsonl"
    dataset_path.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8")
    result = await mcp_session.call_tool(
        "tc_dataset_validate", arguments={"path": str(dataset_path)}
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.dataset.validate.v1"
    assert payload["path"] == str(dataset_path.resolve())
    assert "exampleCount" in payload
    assert "tokenStats" in payload
    assert "warnings" in payload
    assert "errors" in payload
    assert "nextActions" in payload


async def test_tc_job_list_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_job_list", arguments={})
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.job.list.v1"
    assert "jobs" in payload
    assert "count" in payload
    assert payload["count"] == len(payload["jobs"])


async def test_tc_system_resource(mcp_session: ClientSession):
    resource = await mcp_session.read_resource(AnyUrl("tc://system"))
    assert resource.contents
    content = resource.contents[0]
    assert isinstance(content, types.TextResourceContents)
    payload = json.loads(content.text)
    assert payload["_schema"] == "tc.system.status.v1"
