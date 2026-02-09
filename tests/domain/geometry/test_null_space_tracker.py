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

from modelcypher.core.domain.geometry.null_space_tracker import NullSpaceTracker


def test_null_space_tracker_add_and_state(any_backend) -> None:
    b = any_backend
    tracker = NullSpaceTracker(n_layers=2, hidden_dim=2, backend=b)

    with pytest.raises(ValueError):
        tracker.add_activation(layer_id=2, activation=b.array([1.0, 0.0]))

    # Layer 0: constant activations (low-rank).
    tracker.add_activation(0, b.array([1.0, 1.0]))
    tracker.add_activation(0, b.array([1.0, 1.0]))
    tracker.add_activation(0, b.array([1.0, 1.0]))

    # Layer 1: varying activations.
    tracker.add_all_layers(
        {
            1: b.array([1.0, 0.0]),
        }
    )
    tracker.add_activation(1, b.array([0.0, 1.0]))
    tracker.add_activation(1, b.array([1.0, 1.0]))

    assert tracker.total_samples == 6
    assert tracker.should_update() is True

    tracker.update_all_layers()
    layer_0 = tracker.get_layer_state(0)
    layer_1 = tracker.get_layer_state(1)
    model_state = tracker.get_model_state()

    assert layer_0.layer_id == 0
    assert layer_0.hidden_dim == 2
    assert layer_0.n_samples == 3
    assert 0.0 <= layer_0.capacity_fraction <= 1.0
    assert layer_1.n_samples == 3

    assert model_state.layer_id == -1
    assert model_state.hidden_dim == 2
    assert model_state.n_samples >= 0
    assert 0.0 <= model_state.capacity_fraction <= 1.0

    as_dict = layer_0.as_dict()
    assert as_dict["layer_id"] == 0
    assert as_dict["hidden_dim"] == 2


def test_null_space_tracker_basis_projector_weights_and_projection(any_backend) -> None:
    b = any_backend
    tracker = NullSpaceTracker(n_layers=1, hidden_dim=2, backend=b)

    # Constant activations -> zero variance -> full null space available.
    constant = b.array([1.0, 1.0])
    tracker.add_activation(0, constant)
    tracker.add_activation(0, constant)
    tracker.add_activation(0, constant)
    tracker.update_layer(0)

    basis = tracker.get_null_basis(0)
    projector = tracker.get_null_projector(0)
    weights = tracker.get_variance_weights(0)

    assert basis is not None
    assert tuple(basis.shape)[1] == 2
    assert projector is not None
    assert tuple(projector.shape) == (2, 2)
    assert weights is not None
    assert b.tolist(weights) == pytest.approx([1.0, 1.0])

    delta_vec = b.array([2.0, -3.0])
    delta_mat = b.array([[1.0, 2.0], [3.0, 4.0]])

    weighted_vec = tracker.project_to_null_space(0, delta_vec, use_variance_weighting=True)
    weighted_mat = tracker.project_to_null_space(0, delta_mat, use_variance_weighting=True)
    hard_vec = tracker.project_to_null_space(0, delta_vec, use_variance_weighting=False)
    hard_mat = tracker.project_to_null_space(0, delta_mat, use_variance_weighting=False)

    assert weighted_vec is not None
    assert weighted_mat is not None
    assert hard_vec is not None
    assert hard_mat is not None

    assert b.tolist(weighted_vec) == pytest.approx([2.0, -3.0])
    assert b.tolist(weighted_mat)[0] == pytest.approx([1.0, 2.0])
    assert tuple(hard_vec.shape) == (2,)
    assert tuple(hard_mat.shape) == (2, 2)


def test_null_space_tracker_reset_operations(any_backend) -> None:
    b = any_backend
    tracker = NullSpaceTracker(n_layers=2, hidden_dim=2, backend=b)
    tracker.add_activation(0, b.array([1.0, 0.0]))
    tracker.add_activation(0, b.array([0.0, 1.0]))
    tracker.add_activation(0, b.array([1.0, 1.0]))
    tracker.update_layer(0)
    assert tracker.total_samples == 3

    tracker.reset_layer(0)
    assert tracker.get_layer_buffer(0).current_size == 0

    tracker.add_activation(1, b.array([1.0, 1.0]))
    assert tracker.total_samples == 4

    tracker.reset()
    assert tracker.total_samples == 0
    assert tracker.get_layer_buffer(0).current_size == 0
    assert tracker.get_layer_buffer(1).current_size == 0

