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

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.cli.app import app
from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import TrainingStatus
from modelcypher.core.domain.training.geometric_training_metrics import GeometryMetricKey

runner = CliRunner()


def _seed_geometry_job(tmp_home: Path, job_id: str) -> None:
    previous_home = os.environ.get("MODELCYPHER_HOME")
    os.environ["MODELCYPHER_HOME"] = str(tmp_home)
    try:
        store = FileSystemStore()
        metrics = {
            GeometryMetricKey.top_eigenvalue: 0.6,
            GeometryMetricKey.gradient_snr: 4.8,
            GeometryMetricKey.circuit_breaker_severity: 0.1,
            GeometryMetricKey.circuit_breaker_tripped: 0.0,
            GeometryMetricKey.persona_overall_drift: 0.22,
            GeometryMetricKey.persona_delta("directness"): 0.27,
        }
        metrics_history = [{"step": 1, "metrics": {GeometryMetricKey.gradient_snr: 3.2}}]
        now = datetime.now(timezone.utc)
        job = TrainingJob(
            job_id=job_id,
            status=TrainingStatus.running,
            model_id="test-model",
            dataset_path="/tmp/dataset.jsonl",
            created_at=now,
            updated_at=now,
            current_step=8,
            total_steps=80,
            current_epoch=1,
            total_epochs=2,
            loss=1.0,
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


def test_geometry_validate_cli():
    result = runner.invoke(app, ["geometry", "validate", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["_schema"] == "mc.geometry.validation.v1"
    assert "gromovWasserstein" in payload


def test_geometry_path_detect_cli():
    result = runner.invoke(
        app,
        ["geometry", "path", "detect", "def sum(a, b): return a + b", "--output", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["modelID"] == "input-text"
    assert "detectedGates" in payload


def test_geometry_path_compare_cli():
    result = runner.invoke(
        app,
        [
            "geometry",
            "path",
            "compare",
            "--text-a",
            "def f(x): return x + 1",
            "--text-b",
            "f = lambda x: x + 1",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["modelA"] == "text-a"
    assert "rawDistance" in payload


def test_geometry_training_status_cli(tmp_path: Path):
    tmp_home = tmp_path / "home"
    _seed_geometry_job(tmp_home, "job-geometry-1")
    result = runner.invoke(
        app,
        [
            "geometry",
            "training",
            "status",
            "--job",
            "job-geometry-1",
            "--format",
            "summary",
            "--output",
            "json",
        ],
        env={"MODELCYPHER_HOME": str(tmp_home)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["jobId"] == "job-geometry-1"
    assert payload["gradientSNR"] is not None


def test_geometry_training_history_cli(tmp_path: Path):
    tmp_home = tmp_path / "home"
    _seed_geometry_job(tmp_home, "job-geometry-1")
    result = runner.invoke(
        app,
        ["geometry", "training", "history", "--job", "job-geometry-1", "--output", "json"],
        env={"MODELCYPHER_HOME": str(tmp_home)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["jobId"] == "job-geometry-1"
    assert payload["sampleCount"] >= 1


def test_geometry_safety_circuit_breaker_cli(tmp_path: Path):
    tmp_home = tmp_path / "home"
    _seed_geometry_job(tmp_home, "job-geometry-1")
    result = runner.invoke(
        app,
        ["geometry", "safety", "circuit-breaker", "--job", "job-geometry-1", "--output", "json"],
        env={"MODELCYPHER_HOME": str(tmp_home)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "severity" in payload
    assert "state" in payload


def test_geometry_safety_persona_cli(tmp_path: Path):
    tmp_home = tmp_path / "home"
    _seed_geometry_job(tmp_home, "job-geometry-1")
    result = runner.invoke(
        app,
        ["geometry", "safety", "persona", "--job", "job-geometry-1", "--output", "json"],
        env={"MODELCYPHER_HOME": str(tmp_home)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["jobId"] == "job-geometry-1"
    assert "overallDriftMagnitude" in payload


def test_geometry_adapter_sparsity_cli(tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "geometry",
            "adapter",
            "sparsity",
            "--checkpoint",
            str(checkpoint_path),
            "--base",
            str(base_path),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "effectiveSparsity" in payload


def test_geometry_adapter_decomposition_cli(tmp_path: Path):
    checkpoint_path, base_path = _seed_adapter_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "geometry",
            "adapter",
            "decomposition",
            "--checkpoint",
            str(checkpoint_path),
            "--base",
            str(base_path),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["checkpointPath"] == str(checkpoint_path)
    assert "directionalDrift" in payload
