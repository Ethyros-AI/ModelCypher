# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modelcypher.core.domain.geometry.manifold_boundary as manifold_boundary


def test_measure_coherence_magnitude_entropy_and_curvature(any_backend) -> None:
    b = any_backend
    activation = b.array([1.0, 0.0])
    direction = b.array([1.0, 0.0])

    magnitude = manifold_boundary.measure_coherence(
        activation=activation,
        direction=direction,
        radius=0.5,
        forward_fn=lambda x: 2.0 * x,
        backend=b,
        metric="magnitude_stability",
    )
    assert isinstance(magnitude, manifold_boundary.ManifoldCoherenceResult)
    assert magnitude.coherence == pytest.approx(0.5, rel=1e-5, abs=1e-5)
    assert magnitude.curvature == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert magnitude.output_scale == pytest.approx(2.0, rel=1e-5, abs=1e-5)
    assert magnitude.is_coherent is False

    entropy = manifold_boundary.measure_coherence(
        activation=activation,
        direction=direction,
        radius=0.1,
        forward_fn=lambda _x: b.array([0.5, 0.5]),
        backend=b,
        metric="entropy",
    )
    assert entropy.coherence == pytest.approx(0.0, abs=1e-6)
    assert entropy.is_coherent is False

    curvature = manifold_boundary.measure_coherence(
        activation=activation,
        direction=direction,
        radius=0.1,
        forward_fn=lambda x: x * x,
        backend=b,
        metric="curvature",
    )
    assert curvature.curvature == pytest.approx(2.0, rel=1e-2, abs=1e-2)
    assert curvature.coherence == pytest.approx(1.0 / 3.0, rel=1e-2, abs=1e-2)
    assert curvature.is_coherent is False


def test_find_boundary_radius_bounded_unbounded_and_low_baseline(any_backend) -> None:
    b = any_backend

    unbounded = manifold_boundary.find_boundary_radius(
        activation=b.array([1.0, 0.0]),
        direction=b.array([1.0, 0.0]),
        forward_fn=lambda x: x,
        backend=b,
        max_radius=2.0,
        tolerance=0.01,
    )
    assert unbounded.is_bounded is False
    assert unbounded.boundary_radius == pytest.approx(2.0, abs=1e-6)

    bounded = manifold_boundary.find_boundary_radius(
        activation=b.array([1.0]),
        direction=b.array([1.0]),
        forward_fn=lambda x: x * x,
        backend=b,
        max_radius=6.0,
        tolerance=0.01,
    )
    assert bounded.is_bounded is True
    assert 2.5 < bounded.boundary_radius < 3.5

    low_baseline = manifold_boundary.find_boundary_radius(
        activation=b.array([1.0]),
        direction=b.array([1.0]),
        forward_fn=lambda x: b.exp(10.0 * x),
        backend=b,
        max_radius=2.0,
        tolerance=0.01,
    )
    assert low_baseline.is_bounded is True
    assert low_baseline.boundary_radius == pytest.approx(0.0, abs=1e-8)
    assert low_baseline.max_radius_tested == pytest.approx(0.001, abs=1e-8)


def test_detect_manifold_boundary_all_unbounded(any_backend) -> None:
    b = any_backend
    activations = b.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])

    result = manifold_boundary.detect_manifold_boundary(
        activations=activations,
        forward_fn=lambda x: x,
        backend=b,
        n_directions=4,
        max_radius=1.5,
        seed=7,
    )

    assert len(result.boundary_radii) == 2
    assert tuple(result.directions.shape) == (2, 2)
    assert result.n_bounded == 0
    assert result.mean_radius == pytest.approx(1.5, abs=1e-3)
    assert result.min_radius == pytest.approx(1.5, abs=1e-3)
    assert result.max_radius == pytest.approx(1.5, abs=1e-3)
    assert 0.0 <= result.utilized_volume_fraction <= 1.0


def test_create_layer_forward_fn_for_mlp_and_full_model(monkeypatch, any_backend) -> None:
    b = any_backend

    class _Layer:
        def __call__(self, h):
            return (h + 2.0, None)

    class _MLP:
        def __call__(self, h):
            return h + 1.0

    fake_arch = SimpleNamespace(
        layers=[_Layer(), _Layer()],
        layer_accessor=lambda _idx: SimpleNamespace(mlp=_MLP()),
        norm=lambda h: h * 2.0,
        output_projection=lambda h: h,
    )
    monkeypatch.setattr(
        "modelcypher.ports.model_architecture_factory.get_model_architecture",
        lambda _model, config=None: fake_arch,
    )

    mlp_forward = manifold_boundary.create_layer_forward_fn(
        model=object(),
        layer_idx=0,
        config={"num_hidden_layers": 2},
        backend=b,
        mode="mlp",
    )
    mlp_out = mlp_forward(b.array([1.0, 2.0]))
    assert b.tolist(mlp_out) == pytest.approx([2.0, 3.0], abs=1e-6)

    full_forward = manifold_boundary.create_layer_forward_fn(
        model=object(),
        layer_idx=0,
        config={"num_hidden_layers": 2},
        backend=b,
        mode="full_model",
    )
    probs = full_forward(b.array([1.0, 2.0]))
    probs_list = b.tolist(probs)
    assert len(probs_list) == 2
    assert sum(probs_list) == pytest.approx(1.0, abs=1e-5)


def test_detect_layer_boundary_delegates_to_helper(monkeypatch, any_backend) -> None:
    b = any_backend
    expected = manifold_boundary.ManifoldBoundaryResult(
        centroid=b.array([0.0]),
        directions=b.array([[1.0]]),
        boundary_radii=[1.0],
        mean_radius=1.0,
        min_radius=1.0,
        max_radius=1.0,
        utilized_volume_fraction=1.0,
        n_bounded=1,
    )

    monkeypatch.setattr(manifold_boundary, "create_layer_forward_fn", lambda *args, **kwargs: (lambda x: x))
    monkeypatch.setattr(manifold_boundary, "detect_manifold_boundary", lambda **kwargs: expected)

    result = manifold_boundary.detect_layer_boundary(
        model=object(),
        layer_idx=0,
        activations=b.array([[1.0]]),
        config={"num_hidden_layers": 1},
        backend=b,
        n_directions=1,
        max_radius=1.0,
    )

    assert result is expected


def test_compute_boundary_radii_from_weights_handles_missing_and_simple_case(any_backend) -> None:
    b = any_backend

    empty = manifold_boundary.compute_boundary_radii_from_weights(
        target_activations={0: b.array([[1.0, 0.0]])},
        target_weights={"some.other.weight": b.array([[1.0, 0.0]])},
        backend=b,
        n_directions=2,
    )
    assert empty == {}

    radii = manifold_boundary.compute_boundary_radii_from_weights(
        target_activations={0: b.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])},
        target_weights={"model.layers.0.mlp.up_proj.weight": b.array([[1.0, 0.0], [0.0, 1.0]])},
        backend=b,
        n_directions=2,
        seed=123,
    )
    assert 0 in radii
    assert radii[0] > 0.0
