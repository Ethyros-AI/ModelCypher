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
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import TrainingStatus
from modelcypher.core.domain.training.geometric_training_metrics import GeometryMetricKey

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
        from modelcypher.utils.paths import get_jobs_dir
        checkpoint_dir = get_jobs_dir() / job_id / "checkpoints" / "checkpoint-0001"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    finally:
        if previous_home is None:
            os.environ.pop("MODELCYPHER_HOME", None)
        else:
            os.environ["MODELCYPHER_HOME"] = previous_home


def _seed_adapter_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create test adapter files in safetensors format."""
    backend = get_default_backend()
    # LoRA shapes: A is [in, rank], B is [rank, out]
    # delta = A @ B = [in, out]
    # Base weight must match delta shape [in, out]
    lora_a = backend.arange(6, dtype="float32").reshape(3, 2)  # [3, 2] - in=3, rank=2
    lora_b = backend.arange(8, dtype="float32").reshape(2, 4)  # [2, 4] - rank=2, out=4
    # delta = [3, 2] @ [2, 4] = [3, 4], so base weight must be [3, 4]
    base_weight = backend.arange(12, dtype="float32").reshape(3, 4)
    backend.eval(base_weight)
    backend.eval(lora_a)
    backend.eval(lora_b)

    # Create base model directory with safetensors
    # Base weight key must match the adapter layer name (without lora_a/lora_b suffix)
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    base_path = base_dir / "model.safetensors"
    backend.save_safetensors(
        str(base_path),
        {"layers.0.self_attn.q_proj.weight": base_weight},
    )

    # Create adapter directory with safetensors
    # Note: weight keys must end with lora_a or lora_b (lowercase) for detection
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    checkpoint_path = adapter_dir / "adapters.safetensors"
    backend.save_safetensors(
        str(checkpoint_path),
        {
            "layers.0.self_attn.q_proj.lora_a": lora_a,
            "layers.0.self_attn.q_proj.lora_b": lora_b,
        },
    )

    return adapter_dir, base_dir


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

@pytest.fixture(scope="module")
def mcp_payloads(
    mcp_env: dict[str, str], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, object]:
    tmp_root = tmp_path_factory.mktemp("mcp_contracts")
    checkpoint_path, base_path = _seed_adapter_files(tmp_root)

    adapter_dir = tmp_root / "adapter_inspect"
    adapter_dir.mkdir()
    backend = get_default_backend()
    ones_arr = backend.ones((2, 3), dtype="float32")
    backend.eval(ones_arr)
    weights = {"layer.lora_A": ones_arr}
    backend.save_safetensors(str(adapter_dir / "adapter_model.safetensors"), weights)
    (adapter_dir / "adapter_config.json").write_text(
        '{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"]}', encoding="utf-8"
    )

    async def runner(session: ClientSession):
        tool_list = await _await_with_timeout(session.list_tools())
        inventory = await _await_with_timeout(session.call_tool("mc_inventory", arguments={}))
        system_status = await _await_with_timeout(
            session.call_tool("mc_system_status", arguments={})
        )
        settings_snapshot = await _await_with_timeout(
            session.call_tool("mc_settings_snapshot", arguments={})
        )
        geometry_validate = await _await_with_timeout(
            session.call_tool("mc_geometry_validate", arguments={})
        )
        model_list = await _await_with_timeout(session.call_tool("mc_model_list", arguments={}))
        system_resource = await _await_with_timeout(session.read_resource(AnyUrl("mc://system")))
        geometry_training_status = await _await_with_timeout(
            session.call_tool(
                "mc_geometry_training_status",
                arguments={"jobId": "job-geometry-1", "format": "summary"},
            )
        )
        geometry_training_history = await _await_with_timeout(
            session.call_tool("mc_geometry_training_history", arguments={"jobId": "job-geometry-1"})
        )
        safety_circuit_breaker = await _await_with_timeout(
            session.call_tool(
                "mc_safety_circuit_breaker",
                arguments={
                    "adapterName": "test-adapter",
                    "adapterDescription": "Test adapter for circuit breaker",
                    "skillTags": ["general"],
                    "entropyDelta": [0.1, 0.2, 0.15],
                },
            )
        )
        safety_persona_drift = await _await_with_timeout(
            session.call_tool(
                "mc_safety_persona_drift",
                arguments={
                    "baselinePersona": {"helpful": 0.9, "harmless": 0.95, "honest": 0.85},
                    "currentBehavior": ["responds helpfully", "provides accurate information"],
                },
            )
        )
        geometry_dare_sparsity = await _await_with_timeout(
            session.call_tool(
                "mc_geometry_dare_sparsity",
                arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
            )
        )
        geometry_dora_decomposition = await _await_with_timeout(
            session.call_tool(
                "mc_geometry_dora_decomposition",
                arguments={"checkpointPath": str(checkpoint_path), "basePath": str(base_path)},
            )
        )
        geometry_path_detect = await _await_with_timeout(
            session.call_tool(
                "mc_geometry_path_detect",
                arguments={"text": "Hello from ModelCypher."},
            ),
            timeout=30,
        )
        geometry_path_compare = await _await_with_timeout(
            session.call_tool(
                "mc_geometry_path_compare",
                arguments={"textA": "Alpha path", "textB": "Beta path"},
            ),
            timeout=30,
        )
        thermo_analyze = await _await_with_timeout(
            session.call_tool("mc_thermo_analyze", arguments={"jobId": "job-geometry-1"})
        )
        adapter_inspect = await _await_with_timeout(
            session.call_tool(
                "mc_adapter_inspect",
                arguments={"adapterPath": str(adapter_dir)},
            )
        )
        return {
            "tool_list": tool_list,
            "mc_inventory": inventory,
            "mc_system_status": system_status,
            "mc_settings_snapshot": settings_snapshot,
            "mc_geometry_validate": geometry_validate,
            "mc_model_list": model_list,
            "mc_system_resource": system_resource,
            "mc_geometry_training_status": geometry_training_status,
            "mc_geometry_training_history": geometry_training_history,
            "mc_safety_circuit_breaker": safety_circuit_breaker,
            "mc_safety_persona_drift": safety_persona_drift,
            "mc_geometry_dare_sparsity": geometry_dare_sparsity,
            "mc_geometry_dora_decomposition": geometry_dora_decomposition,
            "mc_geometry_path_detect": geometry_path_detect,
            "mc_geometry_path_compare": geometry_path_compare,
            "mc_thermo_analyze": thermo_analyze,
            "mc_adapter_inspect": adapter_inspect,
        }

    results = _run_mcp(mcp_env, runner)
    system_resource = results["mc_system_resource"]
    payloads = {
        "tool_list": results["tool_list"],
        "mc_inventory": _extract_structured(results["mc_inventory"]),
        "mc_system_status": _extract_structured(results["mc_system_status"]),
        "mc_settings_snapshot": _extract_structured(results["mc_settings_snapshot"]),
        "mc_geometry_validate": _extract_structured(results["mc_geometry_validate"]),
        "mc_model_list": _extract_structured(results["mc_model_list"]),
        "mc_geometry_training_status": _extract_structured(results["mc_geometry_training_status"]),
        "mc_geometry_training_history": _extract_structured(results["mc_geometry_training_history"]),
        "mc_safety_circuit_breaker": _extract_structured(results["mc_safety_circuit_breaker"]),
        "mc_safety_persona_drift": _extract_structured(results["mc_safety_persona_drift"]),
        "mc_geometry_dare_sparsity": _extract_structured(results["mc_geometry_dare_sparsity"]),
        "mc_geometry_dora_decomposition": _extract_structured(
            results["mc_geometry_dora_decomposition"]
        ),
        "mc_geometry_path_detect": _extract_structured(results["mc_geometry_path_detect"]),
        "mc_geometry_path_compare": _extract_structured(results["mc_geometry_path_compare"]),
        "mc_thermo_analyze": _extract_structured(results["mc_thermo_analyze"]),
        "mc_adapter_inspect": _extract_structured(results["mc_adapter_inspect"]),
        "mc_system_resource": json.loads(system_resource.contents[0].text),
        "paths": {
            "checkpoint": str(checkpoint_path),
            "base": str(base_path),
            "adapter": str(adapter_dir),
        },
    }
    return payloads


def test_tool_list_includes_core_tools(mcp_payloads: dict[str, object]):
    tool_list = mcp_payloads["tool_list"]
    names = {tool.name for tool in tool_list.tools}
    assert "mc_inventory" in names
    assert "mc_settings_snapshot" in names
    assert "mc_system_status" in names
    assert "mc_model_list" in names
    assert "mc_geometry_validate" in names
    assert "mc_geometry_training_status" in names
    assert "mc_geometry_training_history" in names
    assert "mc_safety_circuit_breaker" in names
    assert "mc_safety_persona_drift" in names
    assert "mc_geometry_dare_sparsity" in names
    assert "mc_geometry_dora_decomposition" in names


def test_mc_inventory_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_inventory"]
    assert "models" in payload
    assert "checkpoints" in payload
    assert "jobs" in payload
    assert "workspace" in payload
    assert "mlxVersion" in payload
    assert "cudaVersion" in payload
    assert "jaxVersion" in payload
    assert "policies" in payload


def test_mc_system_status_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_system_status"]
    assert payload["_schema"] == "mc.system.status.v1"
    assert "machineName" in payload
    assert "unifiedMemoryGB" in payload
    assert "mlxVersion" in payload
    assert "cudaVersion" in payload
    assert "jaxVersion" in payload
    assert "preferredBackend" in payload
    assert "cudaAvailable" in payload
    assert "jaxAvailable" in payload
    assert "readinessScore" in payload
    assert "scoreBreakdown" in payload
    assert "blockers" in payload


def test_mc_settings_snapshot_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_settings_snapshot"]
    assert payload["_schema"] == "mc.settings.snapshot.v1"
    assert "idleTrainingEnabled" in payload
    assert "idleTrainingMinIdleSeconds" in payload
    assert "idleTrainingMaxThermalState" in payload
    assert "maxMemoryUsagePercent" in payload
    assert "autoSaveCheckpoints" in payload
    assert "platformLoggingOptIn" in payload


def test_mc_geometry_validate_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_validate"]
    assert payload["_schema"] == "mc.geometry.validation.v1"
    assert "gromovWasserstein" in payload
    assert "traversalCoherence" in payload
    assert "pathSignature" in payload


def test_mc_model_list_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_model_list"]
    assert payload["_schema"] == "mc.model.list.v1"
    assert "models" in payload
    assert "count" in payload


def test_mc_system_resource(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_system_resource"]
    assert payload["_schema"] == "mc.system.status.v1"


def test_mc_geometry_training_status_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_training_status"]
    assert payload["_schema"] == "mc.geometry.training_status.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert payload["flatnessScore"] is not None
    assert payload["gradientSNR"] is not None


def test_mc_geometry_training_history_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_training_history"]
    assert payload["_schema"] == "mc.geometry.training_history.v1"
    assert payload["jobId"] == "job-geometry-1"
    histories = [
        payload.get("flatnessHistory"),
        payload.get("snrHistory"),
        payload.get("parameterDivergenceHistory"),
    ]
    lengths = [len(items) for items in histories if items]
    expected = max(lengths) if lengths else 0
    assert payload["sampleCount"] == expected


def test_mc_safety_circuit_breaker_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_safety_circuit_breaker"]
    assert payload["_schema"] == "mc.safety.circuit_breaker.v1"
    assert payload["adapterName"] == "test-adapter"
    assert "threatIndicatorCount" in payload
    assert "maxMeanDistance" in payload
    assert "entropyStats" in payload
    assert "indicators" in payload


def test_mc_safety_persona_drift_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_safety_persona_drift"]
    assert payload["_schema"] == "mc.safety.persona_drift.v1"
    # Raw measurements for ALL traits - no threshold filtering
    trait_scores = payload["traitScores"]
    total_drift = sum(value["drift"] for value in trait_scores.values())
    trait_count = payload["traitCount"]
    expected_mean = total_drift / trait_count if trait_count > 0 else 0.0
    assert payload["meanDrift"] == expected_mean
    assert "traitCount" in payload
    assert "traitScores" in payload


def test_mc_geometry_dare_sparsity_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_dare_sparsity"]
    paths = mcp_payloads["paths"]
    assert payload["_schema"] == "mc.geometry.dare_sparsity.v1"
    assert payload["checkpointPath"] == paths["checkpoint"]
    assert "effectiveSparsity" in payload
    assert "layerRanking" in payload


def test_mc_geometry_dora_decomposition_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_dora_decomposition"]
    paths = mcp_payloads["paths"]
    assert payload["_schema"] == "mc.geometry.dora_decomposition.v1"
    assert payload["checkpointPath"] == paths["checkpoint"]
    assert "magnitudeChangeRatio" in payload
    assert "directionalDrift" in payload
    assert "magnitudeToDirectionRatio" in payload
    assert "perLayerDecomposition" in payload


def test_mc_geometry_path_detect_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_path_detect"]
    assert payload["_schema"] == "mc.geometry.path.detect.v1"
    assert "modelID" in payload
    assert "promptID" in payload
    assert "detectedGates" in payload
    assert "meanSimilarity" in payload


def test_mc_geometry_path_compare_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_geometry_path_compare"]
    assert payload["_schema"] == "mc.geometry.path.compare.v1"
    assert "modelA" in payload
    assert "modelB" in payload
    assert "normalizedDistance" in payload


def test_mc_thermo_analyze_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_thermo_analyze"]
    assert payload["_schema"] == "mc.thermo.analyze.v1"
    assert payload["jobId"] == "job-geometry-1"
    assert "entropy" in payload
    assert "temperature" in payload


def test_mc_adapter_inspect_schema(mcp_payloads: dict[str, object]):
    payload = mcp_payloads["mc_adapter_inspect"]
    assert payload["_schema"] == "mc.adapter.inspect.v1"
    assert payload["rank"] == 4
    assert payload["alpha"] == 8.0
    assert "layerAnalysis" in payload
