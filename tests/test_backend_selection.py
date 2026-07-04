"""Backend-selection regression tests."""

from __future__ import annotations

import modelcypher.backends as backends


def test_mlx_disable_selects_jax_fallback_on_macos(monkeypatch) -> None:
    monkeypatch.delenv("MC_BACKEND", raising=False)
    monkeypatch.delenv("MODELCYPHER_BACKEND", raising=False)
    monkeypatch.setenv("MC_DISABLE_MLX", "1")
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(backends, "_try_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_try_jax_available", lambda: True)

    assert backends.detect_default_backend_type() == "jax"


def test_jax_put_along_axis_returns_updated_copy(jax_backend) -> None:
    array = jax_backend.zeros((2, 3))
    indices = jax_backend.array([[0, 2], [1, 0]])
    values = jax_backend.array([[1.0, 2.0], [3.0, 4.0]])

    result = jax_backend.put_along_axis(array, indices, values, axis=1)

    assert jax_backend.tolist(result) == [[1.0, 0.0, 2.0], [4.0, 3.0, 0.0]]
