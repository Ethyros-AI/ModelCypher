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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelcypher.core.domain.lora_memory_store as lora_memory_store
from modelcypher.core.domain.lora_memory_store import (
    HeatSignal,
    LoRAMemoryEvent,
    LoRAMemoryMetadata,
    LoRAMemoryStore,
    MemoryEventSource,
)


def _make_store(monkeypatch, tmp_path: Path, any_backend) -> LoRAMemoryStore:
    model_dir = tmp_path / "base-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "weights.safetensors").write_bytes(b"dummy")

    monkeypatch.setattr(lora_memory_store, "LORA_MEMORY_BASE_DIR", tmp_path / "memory")

    return LoRAMemoryStore(
        agent_id="agent-test",
        base_model_path=model_dir,
        backend=any_backend,
    )


def test_heat_signal_event_and_metadata_roundtrip() -> None:
    heat = HeatSignal(
        timestamp=123,
        surprise_percentile=0.9,
        preserved_fraction=0.8,
        entropy_normalized=0.6,
        entropy_derivative_abs=0.1,
        heat=0.432,
        eigenscore=1.2,
        capacity_fraction=0.3,
    )
    assert HeatSignal.from_dict(heat.to_dict()) == heat

    event = LoRAMemoryEvent(
        timestamp=123,
        hidden_state_hash="h",
        delta_hash="d",
        layer_id=1,
        weight_name="mlp.up_proj",
        source=MemoryEventSource.EXTERNAL,
        confidence=0.7,
        heat=0.2,
    )
    assert LoRAMemoryEvent.from_dict(event.to_dict()) == event

    metadata = LoRAMemoryMetadata(agent_id="a", base_model_path="/tmp/model")
    roundtrip = LoRAMemoryMetadata.from_dict(metadata.to_dict())
    assert roundtrip.agent_id == "a"
    assert roundtrip.store_version == lora_memory_store.LORA_MEMORY_VERSION
    assert roundtrip.target_modules == ["q_proj", "v_proj", "up_proj"]
    assert roundtrip.created_at
    assert roundtrip.updated_at


def test_compute_array_hash_is_stable_for_identical_arrays(any_backend) -> None:
    b = any_backend
    a = b.array([1.0, 2.0, 3.0])
    same = b.array([1.0, 2.0, 3.0])
    different = b.array([1.0, 2.0, 4.0])

    hash_a = lora_memory_store._compute_array_hash(a, b)
    hash_same = lora_memory_store._compute_array_hash(same, b)
    hash_diff = lora_memory_store._compute_array_hash(different, b)

    assert hash_a == hash_same
    assert hash_a != hash_diff


def test_store_accumulate_dedup_and_known_region(monkeypatch, tmp_path, any_backend) -> None:
    b = any_backend
    store = _make_store(monkeypatch, tmp_path, b)

    hidden = b.array([1.0, 2.0])
    delta = b.array([[0.1, 0.2], [0.3, 0.4]])

    assert store.buffer_size == 0
    assert store.derive_optimizer_config() is None
    assert store.derive_critical_batch_size()[0] == 0
    assert store.compute_spectral_budget_ratios() == []

    accepted = store.accumulate(hidden, delta, layer_id=0, weight_name="mlp.up_proj", heat=0.5)
    assert accepted is True
    assert store.buffer_size == 1
    assert store.metadata.event_count == 1
    assert store.metadata.buffer_size == 1

    h_hash = lora_memory_store._compute_array_hash(hidden, b)
    d_hash = lora_memory_store._compute_array_hash(delta, b)
    store.metadata.learned_region_hashes.add(f"{h_hash}_{d_hash}")

    rejected = store.accumulate(hidden, delta, layer_id=0, weight_name="mlp.up_proj")
    assert rejected is False
    assert store.is_known_region(hidden) is True

    stats = store.get_stats()
    assert stats["buffer_size"] == 1
    assert stats["event_count"] == 1


def test_store_save_load_and_reset(monkeypatch, tmp_path, any_backend) -> None:
    b = any_backend
    store = _make_store(monkeypatch, tmp_path, b)
    store.accumulate(
        b.array([1.0, 2.0]),
        b.array([[0.1, 0.2], [0.3, 0.4]]),
        layer_id=1,
        weight_name="mlp.up_proj",
        confidence=0.9,
    )

    saved_dir = store.save()
    assert saved_dir.exists()
    assert (saved_dir / lora_memory_store.METADATA_FILE).exists()
    assert (saved_dir / lora_memory_store.EVENTS_FILE).exists()

    assert store.load() is True

    # Create dummy LoRA file so reset exercises cleanup path.
    (saved_dir / lora_memory_store.LORA_WEIGHTS_FILE).write_bytes(b"dummy")
    store.reset_lora()
    assert store.buffer_size == 0
    assert not (saved_dir / lora_memory_store.EVENTS_FILE).exists()
    assert not (saved_dir / lora_memory_store.LORA_WEIGHTS_FILE).exists()


@dataclass
class _FakeNBLayer:
    delta: object
    scale_bound: float = 0.25

    def get_effective_delta(self):
        return self.delta


def test_merge_to_base_empty_and_success(monkeypatch, tmp_path, any_backend) -> None:
    b = any_backend
    store = _make_store(monkeypatch, tmp_path, b)

    empty_result = store.merge_to_base(model=SimpleNamespace(model=SimpleNamespace(layers=[])))
    assert empty_result.success is False
    assert "No NB-LoRA layers" in (empty_result.error or "")

    key = (0, "mlp.up_proj")
    delta = b.array([[1.0, 0.0], [0.0, 1.0]])
    store._lora_layers[key] = _FakeNBLayer(delta=delta)
    hidden = b.array([0.1, 0.2])
    store._event_buffer[key] = [(hidden, delta, 1.0, 0.0)]

    weight_holder = SimpleNamespace(weight=b.zeros((2, 2)))
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(up_proj=weight_holder))]
        )
    )
    tracker = SimpleNamespace(
        get_layer_projector=lambda layer_id: SimpleNamespace(project=lambda arr: 0.5 * arr)
    )

    merged = store.merge_to_base(model=model, null_space_tracker=tracker)

    assert merged.success is True
    assert merged.layers_merged == 1
    assert merged.preserved_fraction == pytest.approx(0.5, rel=1e-5, abs=1e-5)
    merged_weight = b.tolist(weight_holder.weight)
    assert merged_weight[0] == pytest.approx([0.5, 0.0], abs=1e-6)
    assert merged_weight[1] == pytest.approx([0.0, 0.5], abs=1e-6)
    assert len(store.metadata.learned_region_hashes) >= 1
    assert any(store._history_dir.glob("merged_*.json"))


def test_critical_batch_and_budget_ratio_paths(monkeypatch, tmp_path, any_backend) -> None:
    b = any_backend
    store = _make_store(monkeypatch, tmp_path, b)
    key = (2, "mlp.up_proj")
    hidden = b.array([1.0, 0.0])
    delta = b.array([[1.0, 0.0], [0.0, 1.0]])
    store._event_buffer[key] = [(hidden, delta, 1.0, 0.2)] * 3
    store._metadata.buffer_size = 3

    fake_layer = SimpleNamespace(
        A_tilde=b.array([[1.0, 0.0]]),
        B_tilde=b.array([[1.0, 0.0]]),
        get_S=lambda: b.array([1.0]),
        scale_bound=0.3,
    )
    monkeypatch.setattr(store, "_get_or_init_lora", lambda *args, **kwargs: fake_layer)
    monkeypatch.setattr(
        lora_memory_store,
        "cayley_transform_full",
        lambda A_tilde, B_tilde, _backend: (A_tilde, B_tilde),
    )
    monkeypatch.setattr(
        lora_memory_store.GradientSmoothnessEstimator,
        "gradient_quality",
        staticmethod(lambda **_kwargs: SimpleNamespace(variance=0.2, snr=0.5)),
    )

    critical, measurements = store.derive_critical_batch_size()
    assert critical == 2
    assert measurements["layer_count"] == 1
    assert "layer_2.mlp.up_proj.weight" in measurements["layers"]

    store._lora_layers[key] = fake_layer
    monkeypatch.setattr(
        lora_memory_store,
        "compute_budget_ratios",
        lambda products, _backend: [float(len(products))],
    )
    ratios = store.compute_spectral_budget_ratios()
    assert ratios == [1.0]


def test_load_handles_failure(monkeypatch, tmp_path, any_backend) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    monkeypatch.setattr(store, "_load_or_create_metadata", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert store.load() is False
