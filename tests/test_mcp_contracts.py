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

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
import numpy as np
from pydantic import AnyUrl

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.training.geometric_training_metrics import GeometryMetricKey
from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import TrainingStatus


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
    tmp_home = tmp_path_factory.mktemp("mcp_home")
    _seed_geometry_job(tmp_home, "job-geometry-1")
    return _build_env(tmp_home)


def test_tool_list_includes_core_tools(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.list_tools())

    tool_list = _run_mcp(mcp_env, runner)
    names = {tool.name for tool in tool_list.tools}
    assert "mc_inventory" in names
    assert "mc_settings_snapshot" in names
    assert "mc_system_status" in names
    assert "mc_model_list" in names
    assert "mc_dataset_validate" in names
    assert "mc_job_list" in names
    assert "mc_geometry_validate" in names
    assert "mc_geometry_training_status" in names
    assert "mc_geometry_training_history" in names
    assert "mc_safety_circuit_breaker" in names
    assert "mc_safety_persona_drift" in names
    assert "mc_geometry_dare_sparsity" in names
    assert "mc_geometry_dora_decomposition" in names


def test_mc_inventory_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_inventory", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert "models" in payload
    assert "datasets" in payload
    assert "checkpoints" in payload
    assert "jobs" in payload
    assert "workspace" in payload
    assert "mlxVersion" in payload
    assert "policies" in payload


def test_mc_system_status_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_system_status", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.system.status.v1"
    assert "machineName" in payload
    assert "unifiedMemoryGB" in payload
    assert "readinessScore" in payload
    assert "scoreBreakdown" in payload
    assert "blockers" in payload
    assert "nextActions" in payload


def test_mc_settings_snapshot_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_settings_snapshot", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.settings.snapshot.v1"
    assert "idleTrainingEnabled" in payload
    assert "idleTrainingMinIdleSeconds" in payload
    assert "idleTrainingMaxThermalState" in payload
    assert "maxMemoryUsagePercent" in payload
    assert "autoSaveCheckpoints" in payload
    assert "platformLoggingOptIn" in payload


def test_mc_geometry_validate_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_geometry_validate", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.validation.v1"
    assert "gromovWasserstein" in payload
    assert "traversalCoherence" in payload
    assert "pathSignature" in payload


def test_mc_model_list_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_model_list", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.model.list.v1"
    assert "models" in payload
    assert "count" in payload
    assert "nextActions" in payload


def test_mc_dataset_validate_schema(mcp_env: dict[str, str], tmp_path: Path):
    dataset_path = tmp_path / "sample.jsonl"
    dataset_path.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8")

    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool("mc_dataset_validate", arguments={"path": str(dataset_path)})
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.dataset.validate.v1"
    assert payload["path"] == str(dataset_path.resolve())
    assert "exampleCount" in payload
    assert "tokenStats" in payload
    assert "warnings" in payload
    assert "errors" in payload
    assert "nextActions" in payload


def test_mc_job_list_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.call_tool("mc_job_list", arguments={}))

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.job.list.v1"
    assert "jobs" in payload
    assert "count" in payload
    assert payload["count"] == len(payload["jobs"])


def test_mc_system_resource(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(session.read_resource(AnyUrl("mc://system")))

    resource = _run_mcp(mcp_env, runner)
    assert resource.contents
    content = resource.contents[0]
    assert isinstance(content, types.TextResourceContents)
    payload = json.loads(content.text)
    assert payload["_schema"] == "mc.system.status.v1"


def test_mc_geometry_training_status_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_geometry_training_status",
                arguments={"jobId": "job-geometry-1", "format": "summary"},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.training_status.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["flatnessScore"] is not None
    assert payload["gradientSNR"] is not None


def test_mc_geometry_training_history_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool("mc_geometry_training_history", arguments={"jobId": "job-geometry-1"})
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.training_history.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["sampleCount"] >= 1


def test_mc_safety_circuit_breaker_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool("mc_safety_circuit_breaker", arguments={"jobId": "job-geometry-1"})
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.safety.circuit_breaker.v1"
    assert "severity" in payload
    assert "state" in payload
    assert "signals" in payload
    assert "thresholds" in payload
    assert "recommendedAction" in payload


def test_mc_safety_persona_drift_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool("mc_safety_persona_drift", arguments={"jobId": "job-geometry-1"})
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.safety.persona_drift.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["overallDriftMagnitude"] >= 0.0
    assert "traitDrifts" in payload
    assert "refusalDirectionCorrelation" in payload


def test_mc_geometry_dare_sparsity_schema(mcp_env: dict[str, str], tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)

    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_geometry_dare_sparsity",
                arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.dare_sparsity.v1"
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "effectiveSparsity" in payload
    assert "qualityAssessment" in payload
    assert "interpretation" in payload
    assert "layerRanking" in payload


def test_mc_geometry_dora_decomposition_schema(mcp_env: dict[str, str], tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)

    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_geometry_dora_decomposition",
                arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.dora_decomposition.v1"
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "magnitudeChangeRatio" in payload
    assert "directionalDrift" in payload
    assert "learningTypeConfidence" in payload
    assert "perLayerDecomposition" in payload


def test_mc_geometry_path_detect_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_geometry_path_detect",
                arguments={"text": "Hello from ModelCypher."},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.path.detect.v1"
    assert "modelID" in payload
    assert "promptID" in payload
    assert "detectedGates" in payload
    assert "meanConfidence" in payload


def test_mc_geometry_path_compare_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_geometry_path_compare",
                arguments={"textA": "Alpha path", "textB": "Beta path"},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.geometry.path.compare.v1"
    assert "modelA" in payload
    assert "modelB" in payload
    assert "normalizedDistance" in payload


def test_mc_thermo_analyze_schema(mcp_env: dict[str, str]):
    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_thermo_analyze",
                arguments={"jobId": "job-geometry-1"},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.thermo.analyze.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert "entropy" in payload
    assert "temperature" in payload


def test_mc_ensemble_list_schema(mcp_env: dict[str, str], tmp_path: Path):
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()

    async def runner(session: ClientSession):
        created = await _await_with_timeout(
            session.call_tool(
                "mc_ensemble_create",
                arguments={"models": [str(model_a), str(model_b)]},
            )
        )
        listed = await _await_with_timeout(session.call_tool("mc_ensemble_list", arguments={"limit": 10}))
        created_payload = _extract_structured(created)
        ensemble_id = created_payload["ensembleId"]
        deleted = await _await_with_timeout(
            session.call_tool("mc_ensemble_delete", arguments={"ensembleId": ensemble_id})
        )
        return listed, deleted

    list_result, delete_result = _run_mcp(mcp_env, runner)
    list_payload = _extract_structured(list_result)
    delete_payload = _extract_structured(delete_result)
    assert list_payload["_schema"] == "mc.ensemble.list.v1"
    assert "ensembles" in list_payload
    assert "count" in list_payload
    assert delete_payload["_schema"] == "mc.ensemble.delete.v1"


def test_mc_adapter_inspect_schema(mcp_env: dict[str, str], tmp_path: Path):
    import numpy as np
    from safetensors.numpy import save_file

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weights = {"layer.lora_A": np.ones((2, 3), dtype=np.float32)}
    save_file(weights, adapter_dir / "adapter_model.safetensors")
    (adapter_dir / "adapter_config.json").write_text(
        '{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"]}', encoding="utf-8"
    )

    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_adapter_inspect",
                arguments={"adapterPath": str(adapter_dir)},
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.adapter.inspect.v1"
    assert payload["rank"] == 4
    assert payload["alpha"] == 8.0
    assert "layerAnalysis" in payload


def test_mc_doc_convert_schema(mcp_env: dict[str, str], tmp_path: Path):
    input_path = tmp_path / "notes.txt"
    input_path.write_text("Hello docs", encoding="utf-8")
    output_path = tmp_path / "dataset.jsonl"

    async def runner(session: ClientSession):
        return await _await_with_timeout(
            session.call_tool(
                "mc_doc_convert",
                arguments={
                    "inputs": [str(input_path)],
                    "outputPath": str(output_path),
                },
            )
        )

    result = _run_mcp(mcp_env, runner)
    payload = _extract_structured(result)
    assert payload["_schema"] == "mc.doc.convert.v1"
    assert payload["status"] == "completed"
    assert payload["outputPath"] == str(output_path)


def test_mc_rag_build_and_list_schema(mcp_env: dict[str, str], tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("RAG content", encoding="utf-8")

    async def runner(session: ClientSession):
        build = await _await_with_timeout(
            session.call_tool(
                "mc_rag_build",
                arguments={
                    "indexName": "demo-index",
                    "paths": [str(doc_path)],
                    "modelPath": str(model_path),
                },
            )
        )
        listed = await _await_with_timeout(session.call_tool("mc_rag_list", arguments={}))
        deleted = await _await_with_timeout(
            session.call_tool("mc_rag_delete", arguments={"indexName": "demo-index"})
        )
        return build, listed, deleted

    build_result, list_result, delete_result = _run_mcp(mcp_env, runner)
    build_payload = _extract_structured(build_result)
    list_payload = _extract_structured(list_result)
    delete_payload = _extract_structured(delete_result)

    assert build_payload["_schema"] == "mc.rag.build.v1"
    assert build_payload["indexName"] == "demo-index"
    assert list_payload["_schema"] == "mc.rag.list.v1"
    assert list_payload["count"] >= 1
    assert delete_payload["_schema"] == "mc.rag.delete.v1"
    assert delete_payload["deleted"] == "demo-index"
