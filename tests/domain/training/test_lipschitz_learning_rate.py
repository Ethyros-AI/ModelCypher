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
import modelcypher.core.domain.training.hessian_estimator as hessian_mod
from modelcypher.core.domain.lora_memory_store import LoRAMemoryStore
from modelcypher.core.domain.training.geometric_optimizer import (
    derive_optimizer_geometry_config,
)


def _make_store(monkeypatch, tmp_path: Path, any_backend) -> LoRAMemoryStore:
    model_dir = tmp_path / "base-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "weights.safetensors").write_bytes(b"dummy")

    monkeypatch.setattr(lora_memory_store, "LORA_MEMORY_BASE_DIR", tmp_path / "memory")
    monkeypatch.setattr(lora_memory_store, "get_default_backend", lambda: any_backend)
    monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)

    return LoRAMemoryStore(
        agent_id="agent-lipschitz",
        base_model_path=model_dir,
        backend=any_backend,
    )


def _configure_quadratic_setup(
    store: LoRAMemoryStore,
    any_backend,
    *,
    delta_value: float,
) -> None:
    key = (0, "mlp.up_proj")
    hidden = any_backend.array([1.0], dtype="float32")
    delta = any_backend.array([[delta_value]], dtype="float32")
    any_backend.eval(hidden, delta)
    store._event_buffer[key] = [(hidden, delta, 1.0, 0.0)]
    store._metadata.buffer_size = 1

    # Use non-saturated S so spectral confidence stays positive.
    layer = SimpleNamespace(
        A_tilde=any_backend.array([[1.0]], dtype="float32"),
        B_tilde=any_backend.array([[0.0]], dtype="float32"),
        S_raw=any_backend.array([0.25], dtype="float32"),
        scale_bound=0.5,
    )
    any_backend.eval(layer.A_tilde, layer.B_tilde, layer.S_raw)
    store._lora_layers[key] = layer


def _quadratic_cayley_forward(A_tilde, B_tilde, backend):
    return A_tilde, backend.ones_like(B_tilde), {}


def _quadratic_cayley_pullback(grad_A_eff, grad_B_eff, cache):
    del cache
    return grad_A_eff, grad_B_eff


def test_measure_lipschitz_on_quadratic_loss(monkeypatch, tmp_path, any_backend) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    _configure_quadratic_setup(store, any_backend, delta_value=0.0)
    monkeypatch.setattr(
        store,
        "_cayley_transform_full_unregularized",
        lambda A_tilde, B_tilde: _quadratic_cayley_forward(A_tilde, B_tilde, any_backend),
    )
    monkeypatch.setattr(
        store,
        "_cayley_pullback_full_unregularized",
        _quadratic_cayley_pullback,
    )

    measured = store.measure_lipschitz_constant()
    assert measured is not None
    assert measured > 0.0
    tolerance = hessian_mod.Config.moderate().power_iterations * store.sqrt_eps()
    assert store.derive_learning_rate() == pytest.approx(1.0 / measured, abs=tolerance)


def test_lipschitz_fallback_when_no_lora_layers(monkeypatch, tmp_path, any_backend) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    hidden = any_backend.array([1.0, 0.0], dtype="float32")
    delta = any_backend.array([[2.0, 0.0], [0.0, 0.5]], dtype="float32")
    store.accumulate(hidden, delta, layer_id=1, weight_name="mlp.up_proj")

    assert store.measure_lipschitz_constant() is None

    weight_views = store._event_weight_views()
    expected_proxy_lr = derive_optimizer_geometry_config(weight_views, any_backend).base_lr
    tolerance = hessian_mod.Config.moderate().power_iterations * store.sqrt_eps()
    assert store.derive_learning_rate() == pytest.approx(expected_proxy_lr, abs=tolerance)


def test_lipschitz_fallback_on_empty_buffer(monkeypatch, tmp_path, any_backend) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    store._lora_layers[(0, "mlp.up_proj")] = SimpleNamespace(
        A_tilde=any_backend.array([[1.0]], dtype="float32"),
        B_tilde=any_backend.array([[1.0]], dtype="float32"),
        S_raw=any_backend.array([0.1], dtype="float32"),
        scale_bound=0.5,
    )

    assert store.measure_lipschitz_constant() is None
    assert store.derive_learning_rate() == pytest.approx(store.sqrt_eps(), abs=store.sqrt_eps())


def test_lipschitz_produces_smaller_lr_than_spectral_proxy(
    monkeypatch, tmp_path, any_backend
) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    _configure_quadratic_setup(store, any_backend, delta_value=0.25)
    monkeypatch.setattr(
        store,
        "_cayley_transform_full_unregularized",
        lambda A_tilde, B_tilde: _quadratic_cayley_forward(A_tilde, B_tilde, any_backend),
    )
    monkeypatch.setattr(
        store,
        "_cayley_pullback_full_unregularized",
        _quadratic_cayley_pullback,
    )

    measured_lr = store.derive_learning_rate()
    proxy_lr = derive_optimizer_geometry_config(store._event_weight_views(), any_backend).base_lr

    tolerance = hessian_mod.Config.moderate().power_iterations * store.sqrt_eps()
    assert measured_lr <= proxy_lr + tolerance


def test_derive_learning_rate_uses_lipschitz_when_available(
    monkeypatch, tmp_path, any_backend
) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    known_lipschitz = 8.0
    monkeypatch.setattr(store, "measure_lipschitz_constant", lambda: known_lipschitz)

    assert store.derive_learning_rate() == pytest.approx(1.0 / known_lipschitz, abs=store.sqrt_eps())


def test_cayley_pullback_matches_finite_difference(
    monkeypatch, tmp_path, any_backend
) -> None:
    store = _make_store(monkeypatch, tmp_path, any_backend)
    b = any_backend

    A_tilde = b.array([[0.12, -0.07]], dtype="float32")
    B_tilde = b.array([[0.03, 0.09]], dtype="float32")
    grad_A_eff = b.array([[0.5, -0.2]], dtype="float32")
    grad_B_eff = b.array([[0.4, 0.1]], dtype="float32")

    A_eff, B_eff, cache = store._cayley_transform_full_unregularized(A_tilde, B_tilde)
    grad_A_tilde, grad_B_tilde = store._cayley_pullback_full_unregularized(
        grad_A_eff,
        grad_B_eff,
        cache,
    )
    b.eval(A_eff, B_eff, grad_A_tilde, grad_B_tilde)

    def scalar_fn(A_mat, B_mat) -> float:
        A_cur, B_cur, _ = store._cayley_transform_full_unregularized(A_mat, B_mat)
        value = b.sum(A_cur * grad_A_eff) + b.sum(B_cur * grad_B_eff)
        b.eval(value)
        return float(b.to_scalar(value))

    eps = store.sqrt_eps()
    A_vals = b.tolist(A_tilde)
    B_vals = b.tolist(B_tilde)
    param_count = float(len(A_vals) * len(A_vals[0]) + len(B_vals) * len(B_vals[0]))
    tolerance = param_count * eps

    analytic_A = b.tolist(grad_A_tilde)
    analytic_B = b.tolist(grad_B_tilde)

    for i in range(len(A_vals)):
        for j in range(len(A_vals[0])):
            A_plus_vals = [row[:] for row in A_vals]
            A_minus_vals = [row[:] for row in A_vals]
            A_plus_vals[i][j] += eps
            A_minus_vals[i][j] -= eps
            fd = (
                scalar_fn(b.array(A_plus_vals, dtype="float32"), B_tilde)
                - scalar_fn(b.array(A_minus_vals, dtype="float32"), B_tilde)
            ) / (2.0 * eps)
            assert analytic_A[i][j] == pytest.approx(fd, abs=tolerance)

    for i in range(len(B_vals)):
        for j in range(len(B_vals[0])):
            B_plus_vals = [row[:] for row in B_vals]
            B_minus_vals = [row[:] for row in B_vals]
            B_plus_vals[i][j] += eps
            B_minus_vals[i][j] -= eps
            fd = (
                scalar_fn(A_tilde, b.array(B_plus_vals, dtype="float32"))
                - scalar_fn(A_tilde, b.array(B_minus_vals, dtype="float32"))
            ) / (2.0 * eps)
            assert analytic_B[i][j] == pytest.approx(fd, abs=tolerance)
