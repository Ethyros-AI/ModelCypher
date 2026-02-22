# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.use_cases.star_training_service import StarTrainingService


class _FakeBackend:
    def __init__(self) -> None:
        self._weights: dict[str, dict[str, np.ndarray]] = {}

    def load_safetensors(self, path: str):
        return self._weights[path]

    def save_safetensors(self, path: str, tensors, metadata=None):
        del metadata
        self._weights[path] = tensors
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    def concatenate(self, arrays, axis: int):
        return np.concatenate(arrays, axis=axis)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _prepare_adapter_dir(
    adapter_dir: Path,
    *,
    with_manifest: bool,
    base_hash: str = "hash",
    sigma_k_by_module: dict[str, float] | None = None,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        adapter_dir / "adapter_config.json",
        {
            "num_layers": 1,
            "target_modules": ["self_attn.q_proj"],
        },
    )
    if with_manifest:
        sigma_payload = (
            dict(sigma_k_by_module)
            if sigma_k_by_module is not None
            else {"model.layers.0.self_attn.q_proj.weight": 2.0}
        )
        _write_json(
            adapter_dir / "geometry_manifest.json",
            {
                "base_model_hash": base_hash,
                "sigma_k_by_module": sigma_payload,
            },
        )
    (adapter_dir / "adapters.safetensors").touch()


def test_compose_requires_geometry_manifest(tmp_path: Path):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta = tmp_path / "delta"
    out = tmp_path / "out"

    _prepare_adapter_dir(prior, with_manifest=False)
    _prepare_adapter_dir(delta, with_manifest=True)

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._compose_adapters(
            prior_adapter=prior,
            delta_adapter=delta,
            output_adapter=out,
        )

    assert excinfo.value.failure_class == "insufficient_adapter_geometry"


def test_compose_requires_adapter_config(tmp_path: Path):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta = tmp_path / "delta"
    out = tmp_path / "out"

    prior.mkdir(parents=True, exist_ok=True)
    _prepare_adapter_dir(delta, with_manifest=True)
    _write_json(
        prior / "geometry_manifest.json",
        {
            "base_model_hash": "hash",
            "sigma_k_by_module": {
                "model.layers.0.self_attn.q_proj.weight": 2.0,
            },
        },
    )
    (prior / "adapters.safetensors").touch()

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._compose_adapters(
            prior_adapter=prior,
            delta_adapter=delta,
            output_adapter=out,
        )

    assert excinfo.value.failure_class == "insufficient_adapter_geometry"


def test_compose_derives_scale_from_geometry_manifest(tmp_path: Path, monkeypatch):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta = tmp_path / "delta"
    out = tmp_path / "out"

    _prepare_adapter_dir(prior, with_manifest=True, base_hash="same_hash")
    _prepare_adapter_dir(delta, with_manifest=True, base_hash="same_hash")

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }

    # ratio = ||BA|| / sigma_k = 0.5 -> max scale = sigma_k / ||BA|| = 2.0
    monkeypatch.setattr(
        "modelcypher.core.use_cases.star_training_service.compute_budget_ratios",
        lambda *_args, **_kwargs: [0.5],
    )

    service._compose_adapters(
        prior_adapter=prior,
        delta_adapter=delta,
        output_adapter=out,
    )

    with (out / "adapter_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    assert config["lora_parameters"]["scale"] == pytest.approx(2.0)
    assert config["lora_parameters"]["scale"] != pytest.approx(1.0)
    assert "scale_derivation" in config
    assert config["scale_derivation"]["method"] == "min_module_sigma_k_over_spectral_norm"
    with (out / "geometry_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["base_model_hash"] == "same_hash"
    assert manifest["sigma_k_by_module"]["model.layers.0.self_attn.q_proj.weight"] == pytest.approx(2.0)


def test_compose_rejects_base_model_hash_mismatch(tmp_path: Path):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta = tmp_path / "delta"
    out = tmp_path / "out"

    _prepare_adapter_dir(prior, with_manifest=True, base_hash="hash_a")
    _prepare_adapter_dir(delta, with_manifest=True, base_hash="hash_b")

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._compose_adapters(
            prior_adapter=prior,
            delta_adapter=delta,
            output_adapter=out,
        )
    assert excinfo.value.failure_class == "insufficient_adapter_geometry"
    diagnostics = excinfo.value.diagnostics or {}
    assert diagnostics["prior_base_model_hash"] == "hash_a"
    assert diagnostics["delta_base_model_hash"] == "hash_b"


def test_compose_fails_when_sigma_k_missing_for_layer(tmp_path: Path):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta = tmp_path / "delta"
    out = tmp_path / "out"

    _prepare_adapter_dir(
        prior,
        with_manifest=True,
        base_hash="same_hash",
        sigma_k_by_module={"model.layers.0.self_attn.k_proj.weight": 2.0},
    )
    _prepare_adapter_dir(
        delta,
        with_manifest=True,
        base_hash="same_hash",
        sigma_k_by_module={"model.layers.0.self_attn.k_proj.weight": 2.0},
    )

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._compose_adapters(
            prior_adapter=prior,
            delta_adapter=delta,
            output_adapter=out,
        )
    assert excinfo.value.failure_class == "insufficient_adapter_geometry"
    diagnostics = excinfo.value.diagnostics or {}
    assert diagnostics["layer"] == "model.layers.0.self_attn.q_proj.weight"


def test_composed_manifest_enables_next_round_composition(tmp_path: Path, monkeypatch):
    backend = _FakeBackend()
    service = StarTrainingService(
        backend=backend,
        dataset_training_service=object(),
        training_adapter=object(),
    )

    prior = tmp_path / "prior"
    delta_1 = tmp_path / "delta_1"
    delta_2 = tmp_path / "delta_2"
    composed_1 = tmp_path / "composed_1"
    composed_2 = tmp_path / "composed_2"

    _prepare_adapter_dir(prior, with_manifest=True, base_hash="same_hash")
    _prepare_adapter_dir(delta_1, with_manifest=True, base_hash="same_hash")
    _prepare_adapter_dir(delta_2, with_manifest=True, base_hash="same_hash")

    key_a = "model.layers.0.self_attn.q_proj.lora_a"
    key_b = "model.layers.0.self_attn.q_proj.lora_b"
    backend._weights[str(prior / "adapters.safetensors")] = {
        key_a: np.array([[1.0], [0.0]], dtype=np.float32),
        key_b: np.array([[1.0, 0.0]], dtype=np.float32),
    }
    backend._weights[str(delta_1 / "adapters.safetensors")] = {
        key_a: np.array([[0.0], [1.0]], dtype=np.float32),
        key_b: np.array([[0.0, 1.0]], dtype=np.float32),
    }
    backend._weights[str(delta_2 / "adapters.safetensors")] = {
        key_a: np.array([[0.5], [0.5]], dtype=np.float32),
        key_b: np.array([[0.5, 0.5]], dtype=np.float32),
    }

    monkeypatch.setattr(
        "modelcypher.core.use_cases.star_training_service.compute_budget_ratios",
        lambda *_args, **_kwargs: [0.5],
    )

    service._compose_adapters(
        prior_adapter=prior,
        delta_adapter=delta_1,
        output_adapter=composed_1,
    )
    assert (composed_1 / "geometry_manifest.json").exists()

    service._compose_adapters(
        prior_adapter=composed_1,
        delta_adapter=delta_2,
        output_adapter=composed_2,
    )
    assert (composed_2 / "geometry_manifest.json").exists()
