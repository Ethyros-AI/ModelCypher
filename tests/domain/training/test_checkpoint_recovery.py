# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from modelcypher.core.domain.training.checkpoint_models import CheckpointErrorKind
from modelcypher.core.domain.training.checkpoint_recovery import CheckpointRecovery
from modelcypher.core.domain.training.checkpoint_validation import CheckpointValidation
from modelcypher.core.domain.training.exceptions import CheckpointError


def _write_checkpoint_json(directory: Path, step: int, name: str | None = None) -> Path:
    payload = {
        "version": 2,
        "step": step,
        "total_steps": 100,
        "timestamp": datetime.now().isoformat(),
        "checksum": f"checksum-{step}",
        "weights_file": f"checkpoint-{step}.safetensors",
        "loss_history": [1.0, 0.5],
    }
    file_name = name or f"checkpoint-{step}.json"
    path = directory / file_name
    path.write_text(json.dumps(payload))
    return path


async def test_mark_active_and_inactive_lifecycle(tmp_path) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-markers")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    await recovery.mark_training_active(output_dir)
    marker = recovery._get_crash_marker_path(output_dir)
    assert marker.exists()

    marker_payload = json.loads(marker.read_text())
    assert marker_payload["output_dir"] == str(output_dir)
    assert "started" in marker_payload

    await recovery.mark_training_inactive(output_dir)
    assert not marker.exists()


async def test_update_progress_marker_writes_current_progress(tmp_path) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-progress")
    output_dir = tmp_path / "run-progress"
    output_dir.mkdir()

    await recovery.update_progress_marker(step=7, total_steps=10, output_dir=output_dir)

    marker = recovery._get_progress_marker_path(output_dir)
    payload = json.loads(marker.read_text())
    assert payload["step"] == 7
    assert payload["total"] == 10
    assert "timestamp" in payload


async def test_recover_without_crash_marker_returns_none_and_records_seen(tmp_path) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-seen")
    output_dir = tmp_path / "normal-run"
    output_dir.mkdir()

    result = await recovery.recover_from_crash_if_needed(output_dir)

    assert result is None
    assert recovery._get_crash_marker_path(output_dir).name in recovery._seen_markers


async def test_recover_raises_and_cleans_marker_when_no_checkpoints_dir(tmp_path) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-missing")
    output_dir = tmp_path / "missing-checkpoints"
    output_dir.mkdir()
    await recovery.mark_training_active(output_dir)

    marker = recovery._get_crash_marker_path(output_dir)
    assert marker.exists()

    with pytest.raises(CheckpointError) as exc:
        await recovery.recover_from_crash_if_needed(output_dir)

    assert exc.value.args[0] == CheckpointErrorKind.NO_VALID_CHECKPOINTS
    assert "No checkpoints directory found" in exc.value.args[1]
    assert not marker.exists()


async def test_recover_selects_latest_valid_checkpoint(tmp_path, monkeypatch) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-valid")
    output_dir = tmp_path / "recoverable"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    _write_checkpoint_json(checkpoints_dir, step=10)
    _write_checkpoint_json(checkpoints_dir, step=20)
    _write_checkpoint_json(checkpoints_dir, step=30)
    await recovery.mark_training_active(output_dir)

    async def _fake_validate(metadata, _directory: Path) -> bool:
        return metadata.step == 20

    monkeypatch.setattr(CheckpointValidation, "validate_checkpoint_async", _fake_validate)

    info = await recovery.recover_from_crash_if_needed(output_dir)

    assert info is not None
    assert info.checkpoint.step == 20
    assert info.checkpoints_dir == checkpoints_dir
    assert info.output_dir == output_dir
    assert not recovery._get_crash_marker_path(output_dir).exists()


async def test_list_checkpoints_skips_best_and_malformed(tmp_path) -> None:
    recovery = CheckpointRecovery(temp_dir=tmp_path / "tmp-list")
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    _write_checkpoint_json(checkpoints_dir, step=1, name="checkpoint-1.json")
    _write_checkpoint_json(checkpoints_dir, step=999, name="checkpoint-best.json")
    (checkpoints_dir / "checkpoint-bad.json").write_text("{not-valid-json")

    checkpoints = await recovery._list_checkpoints(checkpoints_dir)

    assert [c.step for c in checkpoints] == [1]

