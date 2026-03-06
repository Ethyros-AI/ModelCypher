# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from datetime import datetime

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
    FineTunedModelMetadata,
    ModelArchitectureSpec,
    OptimizerStateMetadata,
)


class TestOptimizerStateMetadata:
    def test_roundtrip(self):
        original = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer-1000.safetensors",
            checksum="abc123",
            scalar_hyperparameters={"lr": 1e-4, "weight_decay": 0.01},
            vector_hyperparameters={"betas": [0.9, 0.999]},
        )
        d = original.to_dict()
        restored = OptimizerStateMetadata.from_dict(d)

        assert restored.type_name == original.type_name
        assert restored.state_file == original.state_file
        assert restored.checksum == original.checksum
        assert restored.scalar_hyperparameters == original.scalar_hyperparameters
        assert restored.vector_hyperparameters == original.vector_hyperparameters

    def test_from_dict_defaults(self):
        with pytest.raises(ValueError, match="missing required fields"):
            OptimizerStateMetadata.from_dict({})


class TestFineTunedModelMetadata:
    def test_roundtrip_with_optional_fields(self):
        original = FineTunedModelMetadata(
            base_model_id="meta-llama/Llama-3-8B",
            tokenizer_strategy="default",
            lora_config={"rank": 8},
            quantization_config={"bits": 4},
            hyperparameters={"lr": 1e-4},
        )
        d = original.to_dict()
        restored = FineTunedModelMetadata.from_dict(d)

        assert restored.base_model_id == original.base_model_id
        assert restored.lora_config == original.lora_config
        assert restored.quantization_config == original.quantization_config
        assert restored.hyperparameters == original.hyperparameters

    def test_none_fields_excluded_from_dict(self):
        meta = FineTunedModelMetadata(
            base_model_id="test",
            tokenizer_strategy="default",
        )
        d = meta.to_dict()
        assert "lora_config" not in d
        assert "quantization_config" not in d
        assert "hyperparameters" not in d


class TestModelArchitectureSpec:
    def test_roundtrip(self):
        original = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        d = original.to_dict()
        restored = ModelArchitectureSpec.from_dict(d)

        assert restored.model_type == original.model_type
        assert restored.vocabulary_size == original.vocabulary_size
        assert restored.hidden_size == original.hidden_size
        assert restored.num_layers == original.num_layers
        assert restored.num_heads == original.num_heads

    def test_defaults(self):
        with pytest.raises(ValueError, match="missing required fields"):
            ModelArchitectureSpec.from_dict({})


class TestCheckpointMetadataV2:
    def test_roundtrip(self):
        now = datetime(2026, 1, 15, 12, 0, 0)
        original = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=now,
            checksum="sha256abc",
            weights_file="checkpoint-1000.safetensors",
            loss_history=[1.0, 0.5, 0.3],
        )
        d = original.to_dict()
        restored = CheckpointMetadataV2.from_dict(d)

        assert restored.version == 2
        assert restored.step == 1000
        assert restored.total_steps == 5000
        assert restored.checksum == "sha256abc"
        assert restored.weights_file == "checkpoint-1000.safetensors"
        assert restored.loss_history == [1.0, 0.5, 0.3]

    def test_timestamp_iso_string(self):
        """from_dict with ISO string timestamp parses correctly."""
        d = {
            "version": 2,
            "step": 100,
            "total_steps": 1000,
            "timestamp": "2026-01-15T12:00:00",
            "checksum": "abc",
            "weights_file": "w.safetensors",
            "loss_history": [],
        }
        restored = CheckpointMetadataV2.from_dict(d)
        assert restored.timestamp == datetime(2026, 1, 15, 12, 0, 0)

    def test_timestamp_missing_rejected(self):
        """Missing timestamp is a malformed checkpoint payload."""
        d = {
            "version": 2,
            "step": 0,
            "total_steps": 0,
            "checksum": "",
            "weights_file": "",
            "loss_history": [],
        }
        with pytest.raises(ValueError, match="missing required fields: timestamp"):
            CheckpointMetadataV2.from_dict(d)

    def test_nested_metadata_roundtrip(self):
        """Full metadata with nested model_config and optimizer_state."""
        original = CheckpointMetadataV2(
            version=2,
            step=500,
            total_steps=1000,
            timestamp=datetime(2026, 2, 1),
            checksum="xyz",
            weights_file="checkpoint-500.safetensors",
            model_config=ModelArchitectureSpec(
                model_type="llama", vocabulary_size=32000,
                hidden_size=4096, num_layers=32, num_heads=32,
            ),
            optimizer_state=OptimizerStateMetadata(
                type_name="AdamW", state_file="opt-500.safetensors", checksum="opt_hash",
            ),
        )
        d = original.to_dict()
        restored = CheckpointMetadataV2.from_dict(d)

        assert restored.model_config is not None
        assert restored.model_config.model_type == "llama"
        assert restored.optimizer_state is not None
        assert restored.optimizer_state.type_name == "AdamW"


class TestCheckpointErrorKind:
    def test_error_kinds_are_distinct(self):
        """All error kinds have distinct values — prevents accidental duplicates."""
        values = [member.value for member in CheckpointErrorKind]
        assert len(values) == len(set(values))

    def test_error_kind_usable_in_exception(self):
        """CheckpointError can carry an error kind for programmatic handling."""
        from modelcypher.core.domain.training.exceptions import CheckpointError

        err = CheckpointError(CheckpointErrorKind.WRITE_FAILED, "disk full")
        assert CheckpointErrorKind.WRITE_FAILED in err.args
        assert "disk full" in err.args
