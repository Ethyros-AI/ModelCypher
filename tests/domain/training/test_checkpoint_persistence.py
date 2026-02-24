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

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointMetadataV2,
    OptimizerStateMetadata,
)
from modelcypher.core.domain.training.checkpoint_persistence import CheckpointPersistence
from modelcypher.core.domain.training.exceptions import CheckpointError


class TestEstimateCheckpointSize:
    def test_known_params(self):
        p = CheckpointPersistence()
        # 1M params * 4 bytes + 1000 metadata
        size = p.estimate_checkpoint_size(1_000_000, 4, 1000)
        assert size == 4_001_000

    def test_zero_metadata(self):
        p = CheckpointPersistence()
        size = p.estimate_checkpoint_size(100, 2)
        assert size == 200


class TestAtomicWrite:
    def test_success_roundtrip(self, tmp_path):
        """Write + read back → content matches."""
        p = CheckpointPersistence()
        dest = tmp_path / "test.bin"
        content = b"hello world"

        p.atomic_write(content, dest)

        assert dest.read_bytes() == content

    def test_temp_cleanup_on_error(self, tmp_path, monkeypatch):
        """If rename fails, temp file is cleaned up."""
        p = CheckpointPersistence()
        dest = tmp_path / "test.bin"

        # Make rename fail by patching Path.rename
        type(dest).rename

        def failing_rename(self, target):
            raise OSError("forced rename failure")

        monkeypatch.setattr(type(dest), "rename", failing_rename)

        with pytest.raises(CheckpointError):
            p.atomic_write(b"data", dest)

        # No temp file should remain
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0


class TestAtomicWriteJson:
    def test_roundtrip(self, tmp_path):
        """Dict → file → parse → original dict."""
        p = CheckpointPersistence()
        dest = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        p.atomic_write_json(data, dest)

        with open(dest) as f:
            loaded = json.load(f)
        assert loaded == data


class TestSyncToDisk:
    def test_existing_file(self, tmp_path):
        """sync_to_disk on real file → no error."""
        p = CheckpointPersistence()
        f = tmp_path / "test.bin"
        f.write_bytes(b"data")

        p.sync_to_disk(f)  # Should not raise

    def test_missing_file(self, tmp_path):
        """sync_to_disk on non-existent file → CheckpointError."""
        p = CheckpointPersistence()
        with pytest.raises(CheckpointError):
            p.sync_to_disk(tmp_path / "nonexistent.bin")


class TestDeleteCheckpoint:
    def test_removes_files(self, tmp_path):
        """Create files, delete checkpoint → files gone."""
        p = CheckpointPersistence()

        # Create checkpoint files
        weights = tmp_path / "checkpoint-100.safetensors"
        metadata = tmp_path / "checkpoint-100.json"
        weights.write_bytes(b"weights")
        metadata.write_text("{}")

        ckpt = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(), checksum="abc",
            weights_file="checkpoint-100.safetensors",
        )

        p.delete_checkpoint(ckpt, tmp_path)

        assert not weights.exists()
        assert not metadata.exists()

    def test_removes_optimizer_state(self, tmp_path):
        """Optimizer state file also deleted."""
        p = CheckpointPersistence()

        weights = tmp_path / "checkpoint-100.safetensors"
        opt_file = tmp_path / "optimizer-100.safetensors"
        metadata_file = tmp_path / "checkpoint-100.json"
        weights.write_bytes(b"w")
        opt_file.write_bytes(b"o")
        metadata_file.write_text("{}")

        ckpt = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(), checksum="abc",
            weights_file="checkpoint-100.safetensors",
            optimizer_state=OptimizerStateMetadata(
                type_name="AdamW",
                state_file="optimizer-100.safetensors",
                checksum="opt_hash",
            ),
        )

        p.delete_checkpoint(ckpt, tmp_path)

        assert not weights.exists()
        assert not opt_file.exists()
