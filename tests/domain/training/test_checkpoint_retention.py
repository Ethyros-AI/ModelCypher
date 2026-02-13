# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from modelcypher.core.domain.training.checkpoint_models import CheckpointMetadataV2
from modelcypher.core.domain.training.checkpoint_retention import CheckpointRetention


def _write_checkpoint(directory, step: int) -> None:
    """Write a minimal checkpoint JSON and weights file."""
    meta = CheckpointMetadataV2(
        version=2, step=step, total_steps=1000,
        timestamp=datetime.now(), checksum=f"hash_{step}",
        weights_file=f"checkpoint-{step}.safetensors",
    )
    meta_path = directory / f"checkpoint-{step}.json"
    meta_path.write_text(json.dumps(meta.to_dict()))
    weights_path = directory / f"checkpoint-{step}.safetensors"
    weights_path.write_bytes(b"weights")


class TestMaxCheckpointsNone:
    def test_no_pruning(self, tmp_path):
        """max_checkpoints=None → all checkpoints retained."""
        retention = CheckpointRetention(max_checkpoints=None)
        for step in range(5):
            _write_checkpoint(tmp_path, step)

        deleted = asyncio.run(retention.prune_old_checkpoints(tmp_path))
        assert deleted == 0

        # All 5 checkpoint JSONs should exist
        remaining = list(tmp_path.glob("checkpoint-*.json"))
        assert len(remaining) == 5


class TestMaxCheckpointsTwo:
    def test_keeps_two_newest(self, tmp_path):
        """max_checkpoints=2, write 5 → keeps steps 3 and 4."""
        retention = CheckpointRetention(max_checkpoints=2)
        for step in range(5):
            _write_checkpoint(tmp_path, step)

        deleted = asyncio.run(retention.prune_old_checkpoints(tmp_path))
        assert deleted == 3

        remaining = asyncio.run(retention.list_checkpoints(tmp_path))
        remaining_steps = sorted(c.step for c in remaining)
        assert remaining_steps == [3, 4]


class TestConfirmPrune:
    def test_callback_fires(self, tmp_path):
        """confirm_prune=True → callback called, no files deleted."""
        callback_calls = []

        def on_prune(dir_str, keep, pending):
            callback_calls.append((dir_str, keep, pending))

        retention = CheckpointRetention(
            max_checkpoints=2,
            confirm_prune=True,
            on_prune_requested=on_prune,
        )
        for step in range(5):
            _write_checkpoint(tmp_path, step)

        deleted = asyncio.run(retention.prune_old_checkpoints(tmp_path))
        assert deleted == 0
        assert len(callback_calls) == 1
        assert callback_calls[0][1] == 2  # keep=2
        assert callback_calls[0][2] == 3  # pending=3


class TestListCheckpoints:
    def test_parses_json(self, tmp_path):
        """Checkpoint JSONs are parsed and returned."""
        _write_checkpoint(tmp_path, 100)
        _write_checkpoint(tmp_path, 200)
        _write_checkpoint(tmp_path, 300)

        retention = CheckpointRetention()
        checkpoints = asyncio.run(retention.list_checkpoints(tmp_path))
        assert len(checkpoints) == 3

    def test_skips_best(self, tmp_path):
        """checkpoint-best.json is excluded from listing."""
        _write_checkpoint(tmp_path, 100)
        best_path = tmp_path / "checkpoint-best.json"
        best_path.write_text("{}")

        retention = CheckpointRetention()
        checkpoints = asyncio.run(retention.list_checkpoints(tmp_path))
        assert len(checkpoints) == 1

    def test_skips_malformed(self, tmp_path):
        """Invalid JSON file is skipped."""
        _write_checkpoint(tmp_path, 100)
        bad_path = tmp_path / "checkpoint-999.json"
        bad_path.write_text("not valid json")

        retention = CheckpointRetention()
        checkpoints = asyncio.run(retention.list_checkpoints(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0].step == 100
