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
from types import SimpleNamespace

import pytest

import modelcypher.core.domain.lora_memory_store as lora_memory_store
from modelcypher.core.domain.geometry.cayley_lora import cayley_transform_full
from modelcypher.core.domain.lora_memory_store import LoRAMemoryStore


def _get_backend_or_skip(backend_name: str):
    from modelcypher.backends import get_backend

    try:
        return get_backend(backend_name)
    except Exception as exc:
        pytest.skip(f"{backend_name} backend unavailable: {exc}")


def _make_store(monkeypatch, tmp_path: Path, backend_name: str) -> tuple[LoRAMemoryStore, object]:
    backend = _get_backend_or_skip(backend_name)
    model_dir = tmp_path / f"base-model-{backend_name}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "weights.safetensors").write_bytes(b"dummy")

    monkeypatch.setattr(lora_memory_store, "LORA_MEMORY_BASE_DIR", tmp_path / "memory")
    monkeypatch.setattr(lora_memory_store, "get_default_backend", lambda: backend)

    store = LoRAMemoryStore(
        agent_id=f"agent-{backend_name}",
        base_model_path=model_dir,
        backend=backend,
    )
    return store, backend


def _scalar(backend, array) -> float:
    backend.eval(array)
    return float(backend.to_scalar(array))


def _norm(backend, array) -> float:
    return _scalar(backend, backend.norm(array))


def _configure_single_event_layer(
    store: LoRAMemoryStore,
    backend,
    *,
    confidence: float,
    heat: float,
) -> tuple[tuple[int, str], object, object]:
    key = (0, "mlp.up_proj")
    hidden = backend.array([1.0, -0.5], dtype="float32")
    delta = backend.array([[0.2, -0.1], [0.3, 0.4]], dtype="float32")
    base_weight = backend.array([[2.0, 0.0], [0.0, 0.2]], dtype="float32")
    backend.eval(hidden, delta, base_weight)

    store.register_base_weight(0, "mlp.up_proj", base_weight)
    accepted = store.accumulate(
        hidden_state=hidden,
        delta=delta,
        layer_id=0,
        weight_name="mlp.up_proj",
        confidence=confidence,
        heat=heat,
    )
    assert accepted
    layer = store._get_or_init_lora(0, "mlp.up_proj", store._event_buffer[key][0])
    return key, hidden, delta


def _event_mse(store: LoRAMemoryStore, hidden, delta) -> float:
    b = store._backend
    key = (0, "mlp.up_proj")
    layer = store._lora_layers[key]
    A_eff, B_eff = cayley_transform_full(layer.A_tilde, layer.B_tilde, b)
    S = layer.get_S()
    B_scaled = 2.0 * (b.reshape(S, (-1, 1)) * B_eff)
    h = hidden if len(hidden.shape) > 1 else b.reshape(hidden, (1, -1))
    proj = b.matmul(A_eff, b.transpose(h))
    lora_out = b.matmul(b.transpose(B_scaled), proj)
    target = b.matmul(delta, b.transpose(h))
    diff = lora_out - target
    mse = b.mean(diff * diff)
    b.eval(mse)
    return float(b.to_scalar(mse))


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_heat_auto_computed_from_norms(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    hidden = backend.array([2.0, -1.0, 0.5], dtype="float32")
    delta = backend.array([[0.4, 0.2, -0.1], [0.1, 0.3, 0.2]], dtype="float32")
    backend.eval(hidden, delta)

    assert store.accumulate(hidden, delta, layer_id=1, weight_name="mlp.up_proj")
    _, _, _, stored_heat = store._event_buffer[(1, "mlp.up_proj")][0]

    expected = _norm(backend, delta) / _norm(backend, hidden)
    assert stored_heat == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_heat_explicit_override(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    hidden = backend.array([1.0, 2.0], dtype="float32")
    delta = backend.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
    backend.eval(hidden, delta)

    assert store.accumulate(hidden, delta, layer_id=2, weight_name="mlp.up_proj", heat=0.5)
    _, _, _, stored_heat = store._event_buffer[(2, "mlp.up_proj")][0]
    assert stored_heat == 0.5


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_heat_zero_hidden_state_uses_floor(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    hidden = backend.zeros((4,), dtype="float32")
    delta = backend.array([[0.2, 0.1, -0.3, 0.4]], dtype="float32")
    backend.eval(hidden, delta)

    assert store.accumulate(hidden, delta, layer_id=3, weight_name="mlp.up_proj")
    _, _, _, stored_heat = store._event_buffer[(3, "mlp.up_proj")][0]

    expected = _norm(backend, delta) / store.sqrt_eps()
    assert stored_heat == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert stored_heat < float("inf")


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_spectral_confidence_fresh_adapter(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    key, hidden, delta = _configure_single_event_layer(store, backend, confidence=1.0, heat=1.0)
    layer = store._lora_layers[key]

    # Fresh adapter headroom: force max(S)=0 so budget ratio is zero.
    layer.S_raw = backend.zeros_like(layer.S_raw)
    backend.eval(layer.S_raw)

    sigma_k = 2.0 * layer.scale_bound
    sigma_max = store._layer_sigma_max[key]
    condition_ratio = sigma_k / sigma_max if sigma_max > 0 else 1.0
    expected_loss = condition_ratio * _event_mse(store, hidden, delta)

    result = store.train_step(batch_size=1, learning_rate=0.0)
    assert result.loss == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_spectral_confidence_near_exhaustion(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    key, _, _ = _configure_single_event_layer(store, backend, confidence=1.0, heat=1.0)
    layer = store._lora_layers[key]

    layer.S_raw = backend.ones_like(layer.S_raw) * layer.scale_bound
    backend.eval(layer.S_raw)

    result = store.train_step(batch_size=1, learning_rate=0.0)
    assert result.loss <= store.sqrt_eps()


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_spectral_confidence_well_conditioned(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    key = (5, "mlp.up_proj")
    layer = SimpleNamespace(scale_bound=0.5)
    S = backend.array([0.2], dtype="float32")
    backend.eval(S)

    sigma_k = 2.0 * layer.scale_bound
    store._layer_sigma_max[key] = sigma_k
    confidence = store._spectral_confidence(
        layer_id=key[0],
        weight_name=key[1],
        layer=layer,
        S=S,
    )
    expected = max(0.0, 1.0 - (0.2 / 0.5))
    assert confidence == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_spectral_confidence_ill_conditioned(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    key = (6, "mlp.up_proj")
    layer = SimpleNamespace(scale_bound=0.5)
    S = backend.array([0.2], dtype="float32")
    backend.eval(S)

    sigma_k = 2.0 * layer.scale_bound
    store._layer_sigma_max[key] = 10.0 * sigma_k
    confidence = store._spectral_confidence(
        layer_id=key[0],
        weight_name=key[1],
        layer=layer,
        S=S,
    )
    expected = max(0.0, 1.0 - (0.2 / 0.5)) * (sigma_k / (10.0 * sigma_k))
    assert confidence == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("backend_name", ["mlx"])
def test_effective_confidence_multiplicative(monkeypatch, tmp_path, backend_name) -> None:
    store, backend = _make_store(monkeypatch, tmp_path, backend_name)
    key, hidden, delta = _configure_single_event_layer(store, backend, confidence=0.5, heat=1.0)
    layer = store._lora_layers[key]

    layer.S_raw = backend.zeros_like(layer.S_raw)
    backend.eval(layer.S_raw)
    S = layer.get_S()
    spectral_confidence = store._spectral_confidence(
        layer_id=key[0],
        weight_name=key[1],
        layer=layer,
        S=S,
    )
    expected_loss = 0.5 * spectral_confidence * _event_mse(store, hidden, delta)

    result = store.train_step(batch_size=1, learning_rate=0.0)
    assert result.loss == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
