# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for per-layer activation divergence routing signals."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.inference.divergence_router import LayerDivergenceComputer


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def test_compute_layer_divergence_identical_vectors(any_backend) -> None:
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array([1.0, 2.0, 3.0], dtype="float32")
    adapted = backend.array([1.0, 2.0, 3.0], dtype="float32")

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=2,
        adapter_id="adapter_a",
    )

    tol = division_epsilon(backend, base)
    assert abs(measurement.kl_divergence) <= tol
    assert abs(measurement.activation_norm_ratio - 1.0) <= tol
    assert abs(measurement.cosine_similarity - 1.0) <= tol


def test_compute_layer_divergence_sequence_shapes(any_backend) -> None:
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array(
        [
            [0.5, 1.0, 1.5],
            [1.0, 2.0, 4.0],
        ],
        dtype="float32",
    )
    adapted = backend.array(
        [
            [0.5, 1.0, 1.5],
            [2.0, 4.0, 8.0],
        ],
        dtype="float32",
    )

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=4,
        adapter_id="adapter_scaled",
    )

    tol = division_epsilon(backend, base)
    assert abs(measurement.activation_norm_ratio - 2.0) <= tol
    assert abs(measurement.cosine_similarity - 1.0) <= tol
    assert measurement.kl_divergence >= 0.0


def test_compute_layer_divergence_shape_mismatch_raises(any_backend) -> None:
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array([1.0, 2.0, 3.0], dtype="float32")
    adapted = backend.array([1.0, 2.0], dtype="float32")

    with pytest.raises(ValueError, match="Activation shape mismatch"):
        computer.compute_layer_divergence(
            base_activation=base,
            adapted_activation=adapted,
            layer_index=0,
            adapter_id="adapter_bad",
        )


def test_compute_routing_snapshot_across_adapters(any_backend) -> None:
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array([1.0, 2.0, 3.0], dtype="float32")
    adapted_activations = {
        "adapter_b": backend.array([2.0, 1.0, 0.5], dtype="float32"),
        "adapter_a": backend.array([1.1, 2.0, 2.9], dtype="float32"),
    }

    snapshot = computer.compute_routing_snapshot(
        base_activation=base,
        adapted_activations=adapted_activations,
        layer_index=7,
    )

    assert snapshot.layer_index == 7
    assert len(snapshot.measurements) == 2
    assert tuple(measurement.adapter_id for measurement in snapshot.measurements) == (
        "adapter_a",
        "adapter_b",
    )
    assert snapshot.base_activation_norm > 0.0


def test_compute_layer_divergence_handles_zero_base_norm(any_backend) -> None:
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.zeros((4,), dtype="float32")
    adapted = backend.ones((4,), dtype="float32")

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=1,
        adapter_id="adapter_nonzero",
    )

    assert _is_finite(measurement.kl_divergence)
    assert _is_finite(measurement.activation_norm_ratio)
    assert _is_finite(measurement.cosine_similarity)

