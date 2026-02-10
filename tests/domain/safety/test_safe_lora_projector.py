# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import modelcypher.core.domain.safety.safe_lora_projector as projector_mod
from modelcypher.core.domain.safety.safe_lora_projector import (
    SafeLoRAProjectionResult,
    SafeLoRAProjectionStatus,
    SafeLoRAProjector,
)


class _FakeBackend:
    def __init__(self, projection_weights: dict[str, np.ndarray], adapter_weights: dict[str, np.ndarray]) -> None:
        self._projection_weights = projection_weights
        self._adapter_weights = adapter_weights
        self.saved: dict[str, object] = {}

    def load_safetensors(self, path: str):
        if path.endswith("projection.safetensors"):
            return self._projection_weights
        return self._adapter_weights

    def array(self, value):
        return np.array(value)

    def matmul(self, left, right):
        return np.matmul(left, right)

    def eval(self, *_args) -> None:
        return None

    def astype(self, arr, dtype: str):
        return arr.astype(dtype)

    def save_safetensors(self, path: str, tensors: dict[str, np.ndarray]):
        self.saved = {"path": path, "tensors": tensors}


class _BackendProxy:
    """Wrap a real backend while stubbing load/save safetensors calls."""

    def __init__(self, backend, projection_weights: dict, adapter_weights: dict) -> None:
        self._backend = backend
        self._projection_weights = projection_weights
        self._adapter_weights = adapter_weights
        self.saved: dict[str, object] = {}

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def load_safetensors(self, path: str):
        if path.endswith("projection.safetensors"):
            return self._projection_weights
        return self._adapter_weights

    def save_safetensors(self, path: str, tensors: dict):
        self.saved = {"path": path, "tensors": tensors}


def test_projection_result_helpers() -> None:
    applied = SafeLoRAProjectionResult.applied(details="ok")
    skipped = SafeLoRAProjectionResult.skipped(warnings=("warn",), details="later")
    unavailable = SafeLoRAProjectionResult.unavailable("missing")

    assert applied.status == SafeLoRAProjectionStatus.APPLIED
    assert applied.was_applied is True
    assert applied.is_available is True

    assert skipped.status == SafeLoRAProjectionStatus.SKIPPED
    assert skipped.was_applied is False
    assert skipped.is_available is True
    assert skipped.warnings == ("warn",)

    assert unavailable.status == SafeLoRAProjectionStatus.UNAVAILABLE
    assert unavailable.was_applied is False
    assert unavailable.is_available is False
    assert unavailable.warnings == ("missing",)


async def test_project_unavailable_without_projection_matrix(tmp_path: Path) -> None:
    projector = SafeLoRAProjector(resources_path=tmp_path)

    result = await projector.project("community/Llama-3.2 3B", tmp_path / "adapter")

    assert result.status == SafeLoRAProjectionStatus.UNAVAILABLE
    assert result.is_available is False
    assert result.warnings
    assert "community/Llama-3.2 3B" in result.warnings[0]


async def test_project_applies_projection_and_saves(monkeypatch, tmp_path: Path) -> None:
    projector = SafeLoRAProjector(resources_path=tmp_path)
    base_model_id = "community/Llama-3.2 3B"
    sanitized = projector._sanitize(base_model_id)

    projection_dir = tmp_path / "safety" / "projections" / sanitized
    projection_dir.mkdir(parents=True)
    projection_file = projection_dir / "projection.safetensors"
    projection_file.write_bytes(b"projection")

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_file.write_bytes(b"adapter")

    P = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    Wb = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    Wa = np.array([[1.0, 1.0]], dtype=np.float32)

    backend = _FakeBackend(
        projection_weights={"layers.0.proj": P},
        adapter_weights={
            "layers.0.lora_b.weight": Wb,
            "layers.0.lora_a.weight": Wa,
        },
    )
    monkeypatch.setattr(projector_mod, "get_default_backend", lambda: backend)

    result = await projector.project(base_model_id=base_model_id, adapter_path=adapter_path)

    assert result.status == SafeLoRAProjectionStatus.APPLIED
    assert result.was_applied is True
    assert result.details is not None
    assert "Projected 1 weight matrices" in result.details

    assert backend.saved["path"] == str(adapter_file)
    saved_tensors = backend.saved["tensors"]
    assert np.allclose(saved_tensors["layers.0.lora_b.weight"], Wb - np.matmul(P, Wb))
    assert np.array_equal(saved_tensors["layers.0.lora_a.weight"], Wa)


async def test_project_applies_projection_with_any_backend(
    monkeypatch, any_backend, tmp_path: Path
) -> None:
    projector = SafeLoRAProjector(resources_path=tmp_path)
    base_model_id = "community/Llama-3.2 3B"
    sanitized = projector._sanitize(base_model_id)

    projection_dir = tmp_path / "safety" / "projections" / sanitized
    projection_dir.mkdir(parents=True)
    projection_file = projection_dir / "projection.safetensors"
    projection_file.write_bytes(b"projection")

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_file.write_bytes(b"adapter")

    b = any_backend
    P = b.array([[0.5, 0.0], [0.0, 0.5]])
    Wb = b.array([[2.0, 0.0], [0.0, 2.0]])
    Wa = b.array([[1.0, 1.0]])
    expected = Wb - b.matmul(P, Wb)
    b.eval(expected)

    backend = _BackendProxy(
        backend=b,
        projection_weights={"layers.0.proj": P},
        adapter_weights={
            "layers.0.lora_b.weight": Wb,
            "layers.0.lora_a.weight": Wa,
        },
    )
    monkeypatch.setattr(projector_mod, "get_default_backend", lambda: backend)

    result = await projector.project(base_model_id=base_model_id, adapter_path=adapter_path)

    assert result.status == SafeLoRAProjectionStatus.APPLIED
    assert backend.saved["path"] == str(adapter_file)
    saved_tensors = backend.saved["tensors"]
    projected_list = b.tolist(saved_tensors["layers.0.lora_b.weight"])
    expected_list = b.tolist(expected)
    for row_projected, row_expected in zip(projected_list, expected_list):
        assert row_projected == pytest.approx(row_expected, abs=1e-6)

    lora_a_saved = b.tolist(saved_tensors["layers.0.lora_a.weight"])
    lora_a_expected = b.tolist(Wa)
    for row_saved, row_expected in zip(lora_a_saved, lora_a_expected):
        assert row_saved == pytest.approx(row_expected, abs=1e-6)


async def test_apply_projection_error_and_fallback_paths(monkeypatch, tmp_path: Path) -> None:
    projector = SafeLoRAProjector(resources_path=tmp_path)
    projection_path = tmp_path / "projection.safetensors"
    projection_path.write_bytes(b"projection")

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    backend_empty_projection = _FakeBackend(projection_weights={}, adapter_weights={})
    monkeypatch.setattr(projector_mod, "get_default_backend", lambda: backend_empty_projection)
    with pytest.raises(ValueError, match="Empty projection file"):
        await projector._apply_projection(adapter_path=adapter_path, projection_path=projection_path)

    backend_no_adapter = _FakeBackend(
        projection_weights={"layers.0.proj": np.eye(2, dtype=np.float32)},
        adapter_weights={},
    )
    monkeypatch.setattr(projector_mod, "get_default_backend", lambda: backend_no_adapter)
    with pytest.raises(ValueError, match="No adapter weights found"):
        await projector._apply_projection(adapter_path=adapter_path, projection_path=projection_path)

    (adapter_path / "adapter_model.safetensors").write_bytes(b"adapter")

    backend_no_match = _FakeBackend(
        projection_weights={"layers.1.proj": np.eye(2, dtype=np.float32)},
        adapter_weights={"layers.0.lora_b.weight": np.eye(2, dtype=np.float32)},
    )
    monkeypatch.setattr(projector_mod, "get_default_backend", lambda: backend_no_match)

    projected_count = await projector._apply_projection(
        adapter_path=adapter_path,
        projection_path=projection_path,
    )

    assert projected_count == 0
    assert backend_no_match.saved == {}


def test_projection_key_lookup_find_file_and_sanitize(tmp_path: Path) -> None:
    projector = SafeLoRAProjector(resources_path=tmp_path)

    weights = {
        "layers.5.proj": np.eye(2, dtype=np.float32),
        "layer_7_proj": np.eye(2, dtype=np.float32),
        "projection.9": np.eye(2, dtype=np.float32),
    }

    assert projector._get_projection_key("layers.5.lora_b.weight", weights) == "layers.5.proj"
    assert projector._get_projection_key("layers.7.lora_b.weight", weights) == "layer_7_proj"
    assert projector._get_projection_key("layers.9.lora_b.weight", weights) == "projection.9"
    assert projector._get_projection_key("bad_key", weights) is None

    subdir = "safety/projections/community_Llama-3.2_3B"
    projection_dir = tmp_path / subdir
    projection_dir.mkdir(parents=True)
    projection_file = projection_dir / "projection.safetensors"
    projection_file.write_bytes(b"projection")

    assert projector._find_projection_file(subdir) == projection_file
    assert projector._sanitize("community/Llama:3.2 3B") == "community_Llama_3.2_3B"
