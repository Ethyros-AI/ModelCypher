# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import os
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest

from modelcypher.core.domain.runtime_status import RuntimeMemoryStatus, RuntimeOwner
from modelcypher.core.use_cases.runtime_coordinator import RuntimeCoordinator


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "canonical_train_workflow_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "canonical_train_workflow_smoke_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_extract_validated_next_actions_reads_matching_top_level_and_result() -> None:
    actions = [
        {
            "name": "evaluate",
            "reason": "Measure the saved adapter on held-out loss.",
            "command": "mc train evaluate -m /model -a /adapter -d /eval.jsonl",
        },
        {
            "name": "compare",
            "reason": "Compare the adapter against the base model.",
            "command": "mc train compare -m /model --adapter-a /adapter -d /eval.jsonl",
        },
    ]
    envelope = {
        "command": "mc train run",
        "next_actions": actions,
        "result": {
            "next_actions": list(actions),
        },
    }

    parsed = SCRIPT._extract_validated_next_actions(envelope)

    assert parsed == actions


def test_extract_validated_next_actions_rejects_mismatch() -> None:
    envelope = {
        "command": "mc train run",
        "next_actions": [
            {
                "name": "evaluate",
                "reason": "Measure held-out loss.",
                "command": "mc train evaluate -m /model -a /adapter -d /eval.jsonl",
            },
        ],
        "result": {
            "next_actions": [
                {
                    "name": "compare",
                    "reason": "Compare against the base model.",
                    "command": "mc train compare -m /model --adapter-a /adapter -d /eval.jsonl",
                },
            ],
        },
    }

    with pytest.raises(ValueError, match="Top-level next_actions do not match"):
        SCRIPT._extract_validated_next_actions(envelope)


def test_command_to_argv_preserves_generated_command_tokens_verbatim() -> None:
    command = "mc train compare -m /model --adapter-a /adapter -d /eval.jsonl"

    argv = SCRIPT._command_to_argv(command)

    assert argv[:2] == ["poetry", "run"]
    assert argv[2:-1] == shlex.split(command)
    assert argv[-1] == "--json"


def test_snapshot_runtime_status_reads_owner_phase_and_memory(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator(base_path=tmp_path)
    with coordinator.session(
        owner=RuntimeOwner.TRAINING,
        job_id="train-1",
        phase="starting",
        details={"pid": os.getpid()},
    ):
        coordinator.update(
            phase="complete",
            eta_seconds=0.0,
            throughput_tokens_per_second=42.0,
            memory=RuntimeMemoryStatus(
                active_gpu_memory_gb=3.5,
                peak_gpu_memory_gb=5.25,
            ),
        )

        snapshot = SCRIPT._snapshot_runtime_status(RuntimeCoordinator(base_path=tmp_path))

    assert snapshot is not None
    assert snapshot["owner"] == "training"
    assert snapshot["phase"] == "complete"
    assert snapshot["memory"]["active_gpu_memory_gb"] == 3.5
    assert snapshot["memory"]["peak_gpu_memory_gb"] == 5.25


def test_parse_busy_failure_requires_active_runtime() -> None:
    payload = {
        "error": {
            "code": "MC-2022",
            "title": "Runtime busy",
            "active_runtime": {
                "owner": "export",
                "phase": "export",
                "job_id": "export-1",
            },
        },
        "exitCode": 1,
    }

    parsed = SCRIPT._parse_busy_failure(SCRIPT.json.dumps(payload))

    assert parsed["error"]["active_runtime"]["owner"] == "export"


def test_parse_busy_failure_rejects_missing_active_runtime() -> None:
    payload = {
        "error": {
            "code": "MC-2022",
            "title": "Runtime busy",
        },
        "exitCode": 1,
    }

    with pytest.raises(ValueError, match="active_runtime"):
        SCRIPT._parse_busy_failure(SCRIPT.json.dumps(payload))
