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
from pathlib import Path

import pytest

from modelcypher.experimental.merge.exceptions import MergeError
from modelcypher.experimental.merge.lora_adapter_merger import LoRAAdapterMerger


class _FakeBackend:
    def load_safetensors(self, _path: str) -> dict[str, object]:
        return {"model.layers.0.self_attn.q_proj.weight": [1.0]}

    def array(self, value: object) -> object:
        return value


def _write_adapter(
    directory: Path,
    *,
    metadata: dict[str, str] | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": "base-model",
        "r": 4,
        "lora_alpha": 8,
    }
    if metadata is not None:
        config["metadata"] = metadata
    (directory / "adapter_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (directory / "adapter_model.safetensors").write_bytes(b"stub")


def test_load_adapter_rejects_missing_provenance_metadata(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, metadata=None)

    with pytest.raises(MergeError, match="missing provenance metadata"):
        LoRAAdapterMerger._load_adapter(adapter_dir, backend=_FakeBackend())


def test_load_adapter_rejects_capability_transfer_false(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(
        adapter_dir,
        metadata={
            "training_objective": "centered_outcome",
            "capability_transfer": "false",
        },
    )

    with pytest.raises(MergeError, match="capability_transfer=false"):
        LoRAAdapterMerger._load_adapter(adapter_dir, backend=_FakeBackend())


def test_load_adapter_rejects_invalid_capability_transfer_literal(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(
        adapter_dir,
        metadata={
            "training_objective": "ce",
            "capability_transfer": "maybe",
        },
    )

    with pytest.raises(MergeError, match="invalid capability_transfer"):
        LoRAAdapterMerger._load_adapter(adapter_dir, backend=_FakeBackend())


def test_load_adapter_accepts_explicit_capability_transfer_true(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(
        adapter_dir,
        metadata={
            "training_objective": "ce",
            "capability_transfer": "true",
        },
    )

    payload = LoRAAdapterMerger._load_adapter(adapter_dir, backend=_FakeBackend())

    assert payload.capability_transfer is True
    assert payload.training_objective == "ce"
