# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.geometric_context import (
    GeometricContext,
    compute_exit_convergence,
    compute_hidden_norm,
)
from modelcypher.core.domain.training.loop_preservation import (
    LoopPreservationConfig,
    compute_entropy_delta,
    compute_spectral_entropy,
    derive_loop_config_from_geometry,
    find_highway_layer_from_intrinsic_dims,
    loop_preservation_loss,
    select_layers_to_sample,
)


def test_geometric_context_from_metrics_and_format() -> None:
    context = GeometricContext.from_metrics(
        h_highway_entropy=0.3,
        h_exit_entropy=0.7,
        base_delta_entropy=0.2,
        layer_norms=[1.0, 3.0, 2.0],
        highway_layer=2,
        n_layers=8,
        exit_mean_norm=10.0,
        exit_dev_norm=2.0,
    )

    assert context.loop_persistence == pytest.approx(0.2)
    assert context.expansion_ratio == pytest.approx(1.5)
    assert context.highway_depth == pytest.approx(0.25)
    assert context.exit_convergence == pytest.approx(5.0)
    assert context.has_reasoning_loops is True
    assert context.to_dict()["loop_persistence"] == pytest.approx(0.2)

    formatted = context.format()
    assert formatted.startswith("[GEOMETRY]")
    assert "reasoning_loops: yes" in formatted
    assert formatted.endswith("\n\n")


def test_geometric_context_empty_defaults() -> None:
    empty = GeometricContext.empty()

    assert empty.loop_persistence == 0.0
    assert empty.expansion_ratio == 1.0
    assert empty.highway_depth == 0.0
    assert empty.exit_convergence == 1.0
    assert empty.has_reasoning_loops is False


def test_compute_hidden_norm_for_2d_and_3d(any_backend) -> None:
    b = any_backend
    hidden_2d = b.array([[3.0, 4.0], [0.0, 5.0]])
    hidden_3d = b.reshape(hidden_2d, (1, 2, 2))

    assert compute_hidden_norm(hidden_2d, b) == pytest.approx(5.0)
    assert compute_hidden_norm(hidden_3d, b) == pytest.approx(5.0)

    empty = b.array([]).reshape((0, 2))
    assert compute_hidden_norm(empty, b) == 0.0


def test_compute_exit_convergence(any_backend) -> None:
    b = any_backend
    hidden = b.array([[3.0, 4.0], [5.0, 12.0]])  # Norms 5 and 13
    mean_norm, dev_norm = compute_exit_convergence(hidden, b)

    assert mean_norm == pytest.approx(9.0)
    assert dev_norm == pytest.approx(4.0)

    empty = b.array([]).reshape((0, 2))
    empty_mean, empty_dev = compute_exit_convergence(empty, b)
    assert empty_mean == 0.0
    assert empty_dev == 1.0


def test_find_highway_layer_from_intrinsic_dims() -> None:
    dims = [9.0, 8.0, 7.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    highway = find_highway_layer_from_intrinsic_dims(dims, n_layers=12)
    assert highway == 3

    assert find_highway_layer_from_intrinsic_dims([], n_layers=0) == 0
    assert find_highway_layer_from_intrinsic_dims([float("inf")], n_layers=1) == 0


def test_compute_entropy_delta_and_loop_loss() -> None:
    delta = compute_entropy_delta([0.2, 0.3], [0.6, 0.8])
    assert delta == pytest.approx(0.45)
    assert compute_entropy_delta([], [1.0]) == 0.0

    config = LoopPreservationConfig(
        highway_layer=3,  # Not present in trajectory; nearest layer should be used.
        base_delta_entropy=0.4,
        lambda_scale=2.0,
    )
    loss, current_delta = loop_preservation_loss(
        current_trajectory={2: 0.5, 5: 0.3},
        config=config,
    )

    assert current_delta == pytest.approx(-0.2)
    assert loss == pytest.approx(1.2)


def test_derive_loop_config_and_layer_sampling() -> None:
    config = derive_loop_config_from_geometry(
        highway_layer=6,
        base_delta_entropy=0.15,
        sigma_max=4.0,
    )

    assert config.highway_layer == 6
    assert config.base_delta_entropy == pytest.approx(0.15)
    assert config.lambda_scale == pytest.approx(0.25)
    assert config.to_dict()["lambda_scale"] == pytest.approx(0.25)

    assert select_layers_to_sample(0) == []
    assert select_layers_to_sample(9) == [3, 4, 8]


def test_compute_spectral_entropy(any_backend) -> None:
    b = any_backend

    hidden = b.array([[1.0, 0.0], [0.0, 1.0]])
    entropy = compute_spectral_entropy(hidden, b)
    assert entropy > 0.0

    degenerate = b.array([[1.0]])
    assert compute_spectral_entropy(degenerate, b) == 0.0

