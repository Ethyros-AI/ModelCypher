from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import numpy as np
from pydantic import AnyUrl

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.geometric_training_metrics import GeometryMetricKey
from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import TrainingStatus


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


def _seed_geometry_job(tmp_home: Path, job_id: str) -> None:
    previous_home = os.environ.get("MODELCYPHER_HOME")
    os.environ["MODELCYPHER_HOME"] = str(tmp_home)
    try:
        store = FileSystemStore()
        metrics = {
            GeometryMetricKey.top_eigenvalue: 0.4,
            GeometryMetricKey.gradient_snr: 5.2,
            GeometryMetricKey.param_divergence: 0.03,
            GeometryMetricKey.circuit_breaker_severity: 0.2,
            GeometryMetricKey.circuit_breaker_tripped: 0.0,
            GeometryMetricKey.refusal_distance: 0.45,
            GeometryMetricKey.refusal_approaching: 0.0,
            GeometryMetricKey.persona_overall_drift: 0.28,
            GeometryMetricKey.persona_delta("curiosity"): 0.32,
            GeometryMetricKey.layer_grad_norm("layer1"): 0.5,
            GeometryMetricKey.layer_grad_fraction("layer1"): 0.12,
        }
        metrics_history = [
            {"step": 1, "metrics": {GeometryMetricKey.gradient_snr: 2.4}},
            {"step": 2, "metrics": {GeometryMetricKey.gradient_snr: 3.1}},
        ]
        now = datetime.now(timezone.utc)
        job = TrainingJob(
            job_id=job_id,
            status=TrainingStatus.running,
            model_id="test-model",
            dataset_path="/tmp/dataset.jsonl",
            created_at=now,
            updated_at=now,
            current_step=12,
            total_steps=100,
            current_epoch=1,
            total_epochs=3,
            loss=1.234,
            learning_rate=1e-5,
            config=None,
            checkpoints=None,
            loss_history=None,
            metrics=metrics,
            metrics_history=metrics_history,
        )
        store.save_job(job)
    finally:
        if previous_home is None:
            os.environ.pop("MODELCYPHER_HOME", None)
        else:
            os.environ["MODELCYPHER_HOME"] = previous_home


def _seed_adapter_files(tmp_path: Path) -> tuple[Path, Path]:
    base_path = tmp_path / "base.npz"
    checkpoint_path = tmp_path / "adapter.npz"
    base_weight = np.arange(12, dtype=np.float32).reshape(4, 3)
    lora_a = np.arange(6, dtype=np.float32).reshape(2, 3)
    lora_b = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.savez(base_path, layer=base_weight)
    np.savez(checkpoint_path, **{"layer.lora_A": lora_a, "layer.lora_B": lora_b})
    return checkpoint_path, base_path


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
    _seed_geometry_job(tmp_home, "job-geometry-1")
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
    assert "tc_settings_snapshot" in names
    assert "tc_system_status" in names
    assert "tc_model_list" in names
    assert "tc_dataset_validate" in names
    assert "tc_job_list" in names
    assert "tc_geometry_validate" in names
    assert "tc_geometry_training_status" in names
    assert "tc_geometry_training_history" in names
    assert "tc_safety_circuit_breaker" in names
    assert "tc_safety_persona_drift" in names
    assert "tc_geometry_dare_sparsity" in names
    assert "tc_geometry_dora_decomposition" in names


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


async def test_tc_settings_snapshot_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_settings_snapshot", arguments={})
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.settings.snapshot.v1"
    assert "idleTrainingEnabled" in payload
    assert "idleTrainingMinIdleSeconds" in payload
    assert "idleTrainingMaxThermalState" in payload
    assert "maxMemoryUsagePercent" in payload
    assert "autoSaveCheckpoints" in payload
    assert "platformLoggingOptIn" in payload


async def test_tc_geometry_validate_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool("tc_geometry_validate", arguments={})
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.geometry.validation.v1"
    assert "gromovWasserstein" in payload
    assert "traversalCoherence" in payload
    assert "pathSignature" in payload


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


async def test_tc_geometry_training_status_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool(
        "tc_geometry_training_status", arguments={"jobId": "job-geometry-1", "format": "summary"}
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.geometry.training_status.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["flatnessScore"] is not None
    assert payload["gradientSNR"] is not None


async def test_tc_geometry_training_history_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool(
        "tc_geometry_training_history", arguments={"jobId": "job-geometry-1"}
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.geometry.training_history.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["sampleCount"] >= 1


async def test_tc_safety_circuit_breaker_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool(
        "tc_safety_circuit_breaker", arguments={"jobId": "job-geometry-1"}
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.safety.circuit_breaker.v1"
    assert "severity" in payload
    assert "state" in payload
    assert "inputs" in payload


async def test_tc_safety_persona_drift_schema(mcp_session: ClientSession):
    result = await mcp_session.call_tool(
        "tc_safety_persona_drift", arguments={"jobId": "job-geometry-1"}
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.safety.persona_drift.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["overallDriftMagnitude"] >= 0.0


async def test_tc_geometry_dare_sparsity_schema(mcp_session: ClientSession, tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)
    result = await mcp_session.call_tool(
        "tc_geometry_dare_sparsity",
        arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.geometry.dare_sparsity.v1"
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "effectiveSparsity" in payload
    assert "qualityAssessment" in payload


async def test_tc_geometry_dora_decomposition_schema(mcp_session: ClientSession, tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)
    result = await mcp_session.call_tool(
        "tc_geometry_dora_decomposition",
        arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
    )
    payload = _extract_structured(result)
    assert payload["_schema"] == "tc.geometry.dora_decomposition.v1"
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "overallMagnitudeChange" in payload
    assert "overallDirectionalDrift" in payload
