from __future__ import annotations

import json
from pathlib import Path

import pytest

import modelcypher.core.domain.profile as profile_mod
from modelcypher.core.domain.profile import (
    GeometricProfile,
    GeometricProfileStore,
    LayerGeometricProfile,
    ProfileActivations,
    ProfileConvergenceMetrics,
    compute_weights_hash,
    load_activations,
    save_activations,
)


class _BackendLoadProxy:
    def __init__(self, backend, tensors):
        self._backend = backend
        self._tensors = tensors

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def load_safetensors(self, _path: str):
        return self._tensors


def test_profile_activations_complete_and_compute_hash(any_backend, tmp_path) -> None:
    b = any_backend
    acts = ProfileActivations()
    assert acts.is_complete() is False

    acts.hidden[0] = b.array([[1.0, 0.0]])
    acts.intermediate[0] = b.array([[0.0, 1.0]])
    acts.gate[0] = b.array([[0.5, 0.5]])
    acts.embedding = b.array([[0.1, 0.2]])
    assert acts.is_complete() is True

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"name":"x"}', encoding="utf-8")
    (model_dir / "a.safetensors").write_bytes(b"A" * 32)
    (model_dir / "b.safetensors").write_bytes(b"B" * 32)

    h1 = compute_weights_hash(model_dir, max_bytes=16)
    (model_dir / "config.json").write_text('{"name":"y"}', encoding="utf-8")
    h2 = compute_weights_hash(model_dir, max_bytes=16)
    assert h1 != h2


def test_profile_metrics_and_layer_roundtrip() -> None:
    metrics = ProfileConvergenceMetrics(
        final_rank={0: 3},
        trajectory_rank={0: 4},
        ceiling_achieved={0: False},
        cka_delta=0.01,
        probes_processed=10,
        probes_failed=1,
        augmentation_rounds=2,
        augmentation_probes=8,
        batches_to_saturation={0: 5},
        total_batches=7,
        all_layers_saturated=False,
        domains_covered=["math"],
    )
    metrics_roundtrip = ProfileConvergenceMetrics.from_dict(metrics.to_dict())
    assert metrics_roundtrip.final_rank == {0: 3}
    assert metrics_roundtrip.domains_covered == ["math"]

    layer = LayerGeometricProfile(
        layer_idx=1,
        activation_rank=7,
        trajectory_rank=8,
        structural_capacity=10,
        hidden_dim=16,
        saturated=True,
        domains_sampled=["logical"],
    )
    assert layer.is_probe_limited is True
    assert layer.unused_capacity == 3

    layer_roundtrip = LayerGeometricProfile.from_dict(layer.to_dict())
    assert layer_roundtrip.layer_idx == 1
    assert layer_roundtrip.domains_sampled == ["logical"]


def test_geometric_profile_validation_roundtrip_and_file_io(monkeypatch, tmp_path) -> None:
    layer = LayerGeometricProfile(layer_idx=0, activation_rank=3, trajectory_rank=3)
    convergence = ProfileConvergenceMetrics(probes_processed=4)
    profile = GeometricProfile(
        model_path="/tmp/model",
        weights_hash="hash-ok",
        layer_profiles={0: layer},
        convergence=convergence,
        has_activations=True,
        has_hidden=True,
        has_intermediate=True,
        has_gate=True,
        has_embedding=True,
    )

    assert profile.created_at
    assert profile.is_complete_for_merge() is True

    monkeypatch.setattr(profile_mod, "compute_weights_hash", lambda _path: "hash-ok")
    assert profile.is_valid_for("/tmp/model") is True

    monkeypatch.setattr(profile_mod, "compute_weights_hash", lambda _path: "hash-bad")
    assert profile.is_valid_for("/tmp/model") is False

    profile.profile_version = "bad-version"
    assert profile.is_valid_for("/tmp/model") is False
    profile.profile_version = profile_mod.PROFILE_VERSION

    payload = profile.to_dict()
    restored = GeometricProfile.from_dict(payload)
    assert restored.get_layer_profile(0) is not None

    profile_dir = tmp_path / "profile"
    metadata_path = profile.save(profile_dir)
    assert metadata_path.exists()

    # Load when activations file exists.
    (profile_dir / profile.activations_file).write_bytes(b"x")
    loaded = GeometricProfile.load(profile_dir)
    assert loaded.has_activations is True

    # Load when activations file is missing.
    (profile_dir / profile.activations_file).unlink()
    loaded_missing = GeometricProfile.load(profile_dir)
    assert loaded_missing.has_activations is False


def test_profile_store_save_load_exists_invalidate_and_fallback(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(profile_mod, "compute_weights_hash", lambda _path: "stable-hash")

    profile = GeometricProfile(model_path=str(model_dir), weights_hash="stable-hash")
    store = GeometricProfileStore(central_dir=tmp_path / "central")

    sidecar_dir = store.save(profile, model_dir, prefer_sidecar=True)
    assert sidecar_dir.name == profile_mod.PROFILE_DIR_NAME
    assert store.exists(model_dir) is True

    loaded_from_cache = store.load(model_dir)
    assert loaded_from_cache is not None

    store.invalidate(model_dir)
    assert str(model_dir.resolve()) not in store._cache

    # Fallback path: sidecar save raises PermissionError, central cache used.
    original_save = GeometricProfile.save

    def _save_with_permission_error(self, path):
        if Path(path).name == profile_mod.PROFILE_DIR_NAME:
            raise PermissionError("read only")
        return original_save(self, path)

    monkeypatch.setattr(GeometricProfile, "save", _save_with_permission_error)
    fallback_store = GeometricProfileStore(central_dir=tmp_path / "central_fallback")
    fallback_profile = GeometricProfile(model_path=str(model_dir), weights_hash="stable-hash")

    central_dir = fallback_store.save(fallback_profile, model_dir, prefer_sidecar=True)
    assert str(central_dir).startswith(str((tmp_path / "central_fallback").resolve()))


def test_profile_store_load_sidecar_errors_and_central_fallback(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model2"
    model_dir.mkdir()

    monkeypatch.setattr(profile_mod, "compute_weights_hash", lambda _path: "hash2")

    store = GeometricProfileStore(central_dir=tmp_path / "central2")

    # Create corrupt sidecar profile to trigger sidecar exception path.
    sidecar_dir = store.profile_dir_for_model(model_dir)
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / profile_mod.PROFILE_METADATA_FILE).write_text("{bad json", encoding="utf-8")

    # Create valid central profile as fallback.
    central_dir = store.central_profile_dir(model_dir)
    central_dir.mkdir(parents=True)
    good_profile = GeometricProfile(model_path=str(model_dir), weights_hash="hash2")
    good_profile.save(central_dir)

    loaded = store.load(model_dir)
    assert loaded is not None
    assert loaded.weights_hash == "hash2"


def test_save_and_load_activations_all_types_and_legacy(any_backend, tmp_path) -> None:
    b = any_backend
    profile_dir = tmp_path / "acts"

    hidden = {
        0: [b.array([1.0, 0.0]), b.array([0.0, 1.0])],
        1: b.array([[0.2, 0.3]]),
    }
    intermediate = {0: b.array([[0.4, 0.5]])}
    gate = {0: b.array([[0.6, 0.7]])}
    k = {0: b.array([[0.8, 0.9]])}
    mean_pooled = {0: b.array([[0.1, 0.1]])}
    embedding = [b.array([0.9, 0.1]), b.array([0.8, 0.2])]

    activations_path = save_activations(
        activations=hidden,
        embedding_activations=embedding,
        profile_dir=profile_dir,
        backend=b,
        intermediate_activations=intermediate,
        gate_activations=gate,
        k_activations=k,
        mean_pooled_activations=mean_pooled,
    )

    assert activations_path.exists()

    loaded = load_activations(profile_dir, b)
    assert set(loaded.hidden.keys()) == {0, 1}
    assert set(loaded.intermediate.keys()) == {0}
    assert set(loaded.gate.keys()) == {0}
    assert set(loaded.k.keys()) == {0}
    assert set(loaded.mean_pooled.keys()) == {0}
    assert loaded.embedding is not None

    with pytest.raises(FileNotFoundError):
        load_activations(tmp_path / "missing", b)

    # Legacy key parsing branch: layer_{idx} -> hidden_{idx}.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / profile_mod.PROFILE_ACTIVATIONS_FILE).write_bytes(b"placeholder")

    proxy = _BackendLoadProxy(
        b,
        {
            "layer_3": b.array([[1.0, 2.0]]),
            "embedding": b.array([[0.1, 0.2]]),
        },
    )
    legacy_loaded = load_activations(legacy_dir, proxy)
    assert 3 in legacy_loaded.hidden
    assert legacy_loaded.embedding is not None
