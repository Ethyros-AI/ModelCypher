# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

import modelcypher.core.domain.training.geometric_optimizer as optimizer_mod
from modelcypher.core.domain.training.geometric_lora import LayerGeometry
from modelcypher.core.domain.training.geometric_optimizer import (
    OptimizerGeometryConfig,
    compute_decay_scale,
    compute_geometric_epsilon,
    compute_lr_statistics,
    derive_layer_optimizer_config,
    derive_optimizer_geometry_config,
)


def _geometry(key: str, sigma_max: float, sigma_k: float, spectral_gap: float) -> LayerGeometry:
    return LayerGeometry(
        layer_key=key,
        shape=(2, 2),
        sigma_max=sigma_max,
        sigma_k=sigma_k,
        effective_rank=2,
        full_rank=2,
        decay_ratio=sigma_max / max(sigma_k, 1e-9),
        tail_dims=1,
        shannon_effective_rank=1.4,
        spectral_gap=spectral_gap,
    )


def test_epsilon_decay_and_layer_config(any_backend) -> None:
    b = any_backend

    epsilon = compute_geometric_epsilon(sigma_max=4.0, sigma_k=0.1, backend=b)
    expected_machine = math.sqrt(b.finfo().eps) * (4.0 ** 2)
    assert epsilon == pytest.approx(max(0.1**2, expected_machine))

    assert compute_decay_scale(sigma_max=0.0, sigma_k=1.0) == 1.0
    assert compute_decay_scale(sigma_max=4.0, sigma_k=1.0) == pytest.approx(0.25)

    geom = _geometry("layer.0", sigma_max=2.0, sigma_k=0.5, spectral_gap=0.2)
    cfg = derive_layer_optimizer_config(geometry=geom, max_sigma=4.0, backend=b)

    assert cfg.layer_key == "layer.0"
    assert cfg.lr_scale == pytest.approx(2.0)
    assert cfg.decay_scale == pytest.approx(0.25)
    assert cfg.spectral_gap == pytest.approx(0.2)
    assert cfg.epsilon > 0.0


def test_optimizer_config_roundtrip_and_invalid_type() -> None:
    config = OptimizerGeometryConfig(
        base_lr=0.1,
        max_sigma=10.0,
        layer_configs={
            "layer.1": optimizer_mod.LayerOptimizerConfig(
                layer_key="layer.1",
                sigma_max=10.0,
                sigma_k=2.0,
                lr_scale=1.0,
                epsilon=0.01,
                decay_scale=0.2,
                spectral_gap=0.5,
            )
        },
    )

    encoded = config.to_dict()
    decoded = OptimizerGeometryConfig.from_dict(encoded)

    assert decoded.base_lr == pytest.approx(0.1)
    assert decoded.max_sigma == pytest.approx(10.0)
    assert decoded.n_layers == 1
    assert decoded.layer_configs["layer.1"].spectral_gap == pytest.approx(0.5)

    with pytest.raises(ValueError, match="Invalid optimizer config type"):
        OptimizerGeometryConfig.from_dict({"type": "wrong"})


def test_derive_optimizer_geometry_config_and_statistics(monkeypatch, any_backend) -> None:
    b = any_backend

    geoms = {
        "layer.good1": _geometry("layer.good1", sigma_max=4.0, sigma_k=1.0, spectral_gap=0.4),
        "layer.good2": _geometry("layer.good2", sigma_max=2.0, sigma_k=0.5, spectral_gap=0.2),
    }

    def _fake_compute_layer_geometry(_weight, key: str, _backend):
        if key == "layer.fail":
            raise RuntimeError("boom")
        return geoms[key]

    monkeypatch.setattr(optimizer_mod, "compute_layer_geometry", _fake_compute_layer_geometry)

    weights = {
        "layer.good1": b.array([[1.0, 0.0], [0.0, 1.0]]),
        "layer.skip": b.array([1.0, 2.0, 3.0]),  # non-2D should be skipped
        "layer.fail": b.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer.good2": b.array([[0.2, 0.1], [0.5, 0.3]]),
    }

    config = derive_optimizer_geometry_config(weights=weights, backend=b)

    assert config.max_sigma == pytest.approx(4.0)
    assert config.base_lr == pytest.approx(0.25)
    assert set(config.layer_configs.keys()) == {"layer.good1", "layer.good2"}
    assert config.layer_configs["layer.good2"].lr_scale == pytest.approx(2.0)

    stats = compute_lr_statistics({"a": 0.1, "b": 0.3, "c": 0.5}, backend=b)
    assert stats["lr_mean"] == pytest.approx(0.3)
    assert stats["lr_min"] == pytest.approx(0.1)
    assert stats["lr_max"] == pytest.approx(0.5)
    assert stats["lr_std"] > 0.0

    assert compute_lr_statistics({}, backend=b) == {}


def test_derive_optimizer_geometry_config_raises_when_nothing_valid(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(
        optimizer_mod,
        "compute_layer_geometry",
        lambda _w, _k, _b: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    with pytest.raises(ValueError, match="No valid weight matrices"):
        derive_optimizer_geometry_config(
            weights={"layer.fail": b.array([[1.0, 0.0], [0.0, 1.0]])},
            backend=b,
        )


def test_derive_optimizer_geometry_config_reuses_precomputed_geometries(monkeypatch, any_backend) -> None:
    b = any_backend

    geoms = {
        "layer.good1": _geometry("layer.good1", sigma_max=3.0, sigma_k=1.0, spectral_gap=0.3),
        "layer.good2": _geometry("layer.good2", sigma_max=1.5, sigma_k=0.5, spectral_gap=0.1),
    }

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("compute_layer_geometry should not run when geometries are provided")

    monkeypatch.setattr(optimizer_mod, "compute_layer_geometry", _should_not_run)

    config = derive_optimizer_geometry_config(
        weights={},
        backend=b,
        geometries=geoms,
    )

    assert config.max_sigma == pytest.approx(3.0)
    assert config.base_lr == pytest.approx(1.0 / 3.0)
    assert set(config.layer_configs.keys()) == {"layer.good1", "layer.good2"}
