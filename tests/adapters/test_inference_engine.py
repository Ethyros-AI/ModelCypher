# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelcypher.adapters.inference_engine import (
    GEOMETRIC_ADAPTER_WEIGHT_CANDIDATES,
    InferenceEngine,
)


class _StubModel:
    def __init__(self) -> None:
        self._params = {
            "target.weight": "base",
            "untouched.weight": "stay",
        }
        self.updated_params: dict[str, Any] | None = None

    def parameters(self) -> dict[str, Any]:
        return dict(self._params)

    def update(self, params: dict[str, Any]) -> None:
        self.updated_params = dict(params)


class _StubBackend:
    def __init__(self) -> None:
        self.model = _StubModel()
        self.loaded_model_paths: list[tuple[str, str | None]] = []
        self.loaded_safetensors: list[str] = []
        self.loaded_binary: list[str] = []

    def load_model(
        self,
        model_path: str,
        adapter_path: str | None = None,
    ) -> tuple[_StubModel, dict[str, str]]:
        self.loaded_model_paths.append((model_path, adapter_path))
        return self.model, {"tokenizer": "stub"}

    def load_safetensors(self, path: str) -> dict[str, Any]:
        self.loaded_safetensors.append(path)
        return {
            "target.weight": f"safetensors::{Path(path).name}",
            "ignored.weight": "ignore-me",
        }

    def load_binary_weights(self, path: str) -> dict[str, Any]:
        self.loaded_binary.append(path)
        return {
            "target.weight": f"binary::{Path(path).name}",
            "ignored.weight": "ignore-me",
        }

    def tree_flatten(self, params: dict[str, Any]) -> list[tuple[str, Any]]:
        return list(params.items())


def _write_adapter_dir(adapter_path: Path, filename: str) -> None:
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"type": "geometric_lora"}),
        encoding="utf-8",
    )
    (adapter_path / filename).write_bytes(b"stub")


@pytest.mark.parametrize(
    ("filename", "expected_loader", "expected_value"),
    [
        ("adapters.safetensors", "safetensors", "safetensors::adapters.safetensors"),
        ("adapter_model.safetensors", "safetensors", "safetensors::adapter_model.safetensors"),
        ("adapter.safetensors", "safetensors", "safetensors::adapter.safetensors"),
        ("lora_weights.safetensors", "safetensors", "safetensors::lora_weights.safetensors"),
        ("adapters.bin", "binary", "binary::adapters.bin"),
        ("adapter_model.bin", "binary", "binary::adapter_model.bin"),
        ("adapter_model.pt", "binary", "binary::adapter_model.pt"),
    ],
)
def test_load_geometric_lora_adapter_supports_repo_weight_filenames(
    tmp_path: Path,
    filename: str,
    expected_loader: str,
    expected_value: str,
) -> None:
    backend = _StubBackend()
    engine = InferenceEngine(backend=backend, base_path=tmp_path)
    model_path = tmp_path / "base_model"
    adapter_path = tmp_path / "adapter"
    _write_adapter_dir(adapter_path, filename)

    model, tokenizer = engine._load_geometric_lora_adapter(model_path, adapter_path)

    assert model is backend.model
    assert tokenizer == {"tokenizer": "stub"}
    assert backend.loaded_model_paths == [(str(model_path), None)]
    assert backend.model.updated_params is not None
    assert backend.model.updated_params["target.weight"] == expected_value
    assert backend.model.updated_params["untouched.weight"] == "stay"

    weights_path = str((adapter_path / filename).resolve())
    if expected_loader == "safetensors":
        assert backend.loaded_safetensors == [weights_path]
        assert backend.loaded_binary == []
    else:
        assert backend.loaded_binary == [weights_path]
        assert backend.loaded_safetensors == []


def test_load_geometric_lora_adapter_raises_when_no_supported_weights_exist(tmp_path: Path) -> None:
    backend = _StubBackend()
    engine = InferenceEngine(backend=backend, base_path=tmp_path)
    model_path = tmp_path / "base_model"
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"type": "geometric_lora"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        engine._load_geometric_lora_adapter(model_path, adapter_path)

    message = str(excinfo.value)
    for candidate in GEOMETRIC_ADAPTER_WEIGHT_CANDIDATES:
        assert candidate in message
