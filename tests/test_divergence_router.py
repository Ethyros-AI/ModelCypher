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


def test_cosine_zero_when_base_norm_zero(any_backend) -> None:
    """Zero base vector → cosine_denom ≤ eps → cosine_similarity must be 0.0, not NaN."""
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.zeros((4,), dtype="float32")
    adapted = backend.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=0,
        adapter_id="a",
    )

    assert measurement.cosine_similarity == 0.0, (
        f"Expected cosine=0.0 for zero base, got {measurement.cosine_similarity}"
    )


def test_cosine_zero_for_orthogonal_vectors(any_backend) -> None:
    """Orthogonal vectors → cosine = 0. Validates the dot-product branch."""
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array([1.0, 0.0, 0.0], dtype="float32")
    adapted = backend.array([0.0, 1.0, 0.0], dtype="float32")

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=0,
        adapter_id="a",
    )

    tol = division_epsilon(backend, base)
    assert abs(measurement.cosine_similarity) <= tol, (
        f"Expected cosine≈0 for orthogonal vectors, got {measurement.cosine_similarity}"
    )


def test_cosine_minus_one_for_anti_parallel_vectors(any_backend) -> None:
    """Anti-parallel vectors (v, -v) → cosine = -1.0. Validates clamp to [-1, 1]."""
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    base = backend.array([1.0, 0.0, 0.0], dtype="float32")
    adapted = backend.array([-1.0, 0.0, 0.0], dtype="float32")

    measurement = computer.compute_layer_divergence(
        base_activation=base,
        adapted_activation=adapted,
        layer_index=0,
        adapter_id="a",
    )

    tol = division_epsilon(backend, base)
    assert abs(measurement.cosine_similarity - (-1.0)) <= tol, (
        f"Expected cosine=-1.0 for anti-parallel, got {measurement.cosine_similarity}"
    )
    # Also verifies clamping: value must not go below -1
    assert measurement.cosine_similarity >= -1.0


def test_last_token_vector_3d_extracts_last_token(any_backend) -> None:
    """[batch=1, seq=3, hidden=4] → result == activation[0, -1] with shape [4]."""
    import numpy as np
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    arr = backend.array(data)
    result = computer._last_token_vector(arr)
    backend.eval(result)

    assert tuple(int(d) for d in result.shape) == (4,)
    # data[0, -1] = [8, 9, 10, 11]
    first_val = float(backend.to_scalar(result[0]))
    assert abs(first_val - 8.0) < 1e-5, f"Expected 8.0, got {first_val}"


def test_last_token_vector_1d_returns_as_is(any_backend) -> None:
    """[hidden=4] → returned unchanged."""
    import numpy as np
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    arr = backend.array(data)
    result = computer._last_token_vector(arr)
    backend.eval(result)

    assert tuple(int(d) for d in result.shape) == (4,)
    assert abs(float(backend.to_scalar(result[0])) - 1.0) < 1e-5


def test_last_token_vector_4d_raises(any_backend) -> None:
    """ndim=4 → ValueError (unsupported rank)."""
    import numpy as np
    backend = any_backend
    computer = LayerDivergenceComputer(backend)

    arr = backend.array(np.zeros((2, 3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="ndim=4"):
        computer._last_token_vector(arr)

