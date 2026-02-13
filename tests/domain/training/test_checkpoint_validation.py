# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointMetadataV2,
    OptimizerStateMetadata,
)
from modelcypher.core.domain.training.checkpoint_validation import CheckpointValidation


class TestCalculateChecksum:
    def test_known_content(self, tmp_path):
        """Known bytes → correct SHA256."""
        content = b"hello world"
        f = tmp_path / "test.bin"
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = CheckpointValidation.calculate_checksum(f)
        assert result == expected

    def test_empty_file(self, tmp_path):
        """Empty file → valid SHA256 hash."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        result = CheckpointValidation.calculate_checksum(f)
        assert result == expected

    def test_missing_file_raises(self, tmp_path):
        """Non-existent file → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CheckpointValidation.calculate_checksum(tmp_path / "nonexistent")


class TestValidateCheckpoint:
    def test_valid_checkpoint(self, tmp_path):
        """Matching checksum → True."""
        content = b"model weights data"
        weights_file = tmp_path / "checkpoint-100.safetensors"
        weights_file.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()

        meta = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(), checksum=checksum,
            weights_file="checkpoint-100.safetensors",
        )

        assert CheckpointValidation.validate_checkpoint(meta, tmp_path) is True

    def test_checksum_mismatch(self, tmp_path):
        """Wrong checksum → False."""
        weights_file = tmp_path / "checkpoint-100.safetensors"
        weights_file.write_bytes(b"model weights data")

        meta = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(), checksum="wrong_checksum",
            weights_file="checkpoint-100.safetensors",
        )

        assert CheckpointValidation.validate_checkpoint(meta, tmp_path) is False

    def test_missing_weights_file(self, tmp_path):
        """Non-existent weights file → False."""
        meta = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(), checksum="abc",
            weights_file="nonexistent.safetensors",
        )

        assert CheckpointValidation.validate_checkpoint(meta, tmp_path) is False

    def test_validates_optimizer_state(self, tmp_path):
        """Valid weights + valid optimizer → True."""
        w_content = b"weights"
        o_content = b"optimizer"
        (tmp_path / "w.safetensors").write_bytes(w_content)
        (tmp_path / "o.safetensors").write_bytes(o_content)

        meta = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(),
            checksum=hashlib.sha256(w_content).hexdigest(),
            weights_file="w.safetensors",
            optimizer_state=OptimizerStateMetadata(
                type_name="AdamW",
                state_file="o.safetensors",
                checksum=hashlib.sha256(o_content).hexdigest(),
            ),
        )

        assert CheckpointValidation.validate_checkpoint(meta, tmp_path) is True

    def test_optimizer_checksum_mismatch(self, tmp_path):
        """Valid weights but wrong optimizer checksum → False."""
        w_content = b"weights"
        (tmp_path / "w.safetensors").write_bytes(w_content)
        (tmp_path / "o.safetensors").write_bytes(b"optimizer")

        meta = CheckpointMetadataV2(
            version=2, step=100, total_steps=1000,
            timestamp=datetime.now(),
            checksum=hashlib.sha256(w_content).hexdigest(),
            weights_file="w.safetensors",
            optimizer_state=OptimizerStateMetadata(
                type_name="AdamW",
                state_file="o.safetensors",
                checksum="wrong",
            ),
        )

        assert CheckpointValidation.validate_checkpoint(meta, tmp_path) is False
