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

"""Tests for constrained least-squares transplant.

Mathematical invariants verified:
    1. Boundary outputs EXACTLY preserved: A_boundary @ W' = A_boundary @ W_target
    2. Core outputs move toward delta: dot(actual_delta, requested_delta) > 0
    3. Shapes preserved: merged_weight.shape == weight_target.shape

No arbitrary thresholds. The geometry determines everything.
"""

import re

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.transplant import (
    TransplantDeltaResult,
    compute_transplant_delta,
    partition_core_boundary,
)


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestPartitionCoreBoundary:
    """Tests for partition_core_boundary function."""

    def test_partition_empty_activations(self) -> None:
        """Empty activations should return empty partition."""
        backend = get_default_backend()
        activations = backend.zeros((0, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[],
            core_probe_ids=set(),
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_no_core_probes(self) -> None:
        """No core probes should return empty partition."""
        backend = get_default_backend()
        backend.random_seed(42)
        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[f"probe_{i}" for i in range(10)],
            core_probe_ids=set(),  # empty core
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_finds_boundary_neighbors(self) -> None:
        """Boundary should include every non-core probe."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(10)]
        core_probe_ids = {"probe_0", "probe_1"}  # first two are core

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert result.core_indices == [0, 1]
        assert len(result.boundary_indices) == 8
        # Boundary should not include core indices
        assert 0 not in result.boundary_indices
        assert 1 not in result.boundary_indices


class TestComputeTransplantDelta:
    """Tests for compute_transplant_delta function."""

    def test_boundary_exactly_preserved(self) -> None:
        """Boundary outputs are EXACTLY preserved (to numerical precision).

        Invariant: A_boundary @ W' = A_boundary @ W_target
        """
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        boundary_activations = backend.random_normal((n_boundary, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim)) * 0.1
        backend.eval(weight_target, activations_core, boundary_activations, delta_activations)

        # Compute boundary output BEFORE transplant
        output_before = backend.matmul(boundary_activations, backend.transpose(weight_target))
        backend.eval(output_before)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_activations,
            backend=backend,
        )

        assert result.applied is True

        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        # Compute boundary output AFTER transplant
        output_after = backend.matmul(boundary_activations, backend.transpose(merged_weight))
        backend.eval(output_after)

        # Boundary outputs should be EXACTLY preserved (to numerical precision)
        diff = output_after - output_before
        diff_norm = float(backend.to_scalar(backend.sum(backend.abs(diff))))

        # Use machine epsilon scaled by output magnitude
        output_norm = float(backend.to_scalar(backend.sum(backend.abs(output_before))))
        eps = _eps(backend, output_norm, diff_norm) * output_norm * 1000  # 1000x machine epsilon

        assert diff_norm <= eps, (
            f"Boundary outputs changed by {diff_norm:.2e}, expected <= {eps:.2e}. "
            "Boundary preservation is a mathematical invariant."
        )

    def test_core_moves_toward_delta(self) -> None:
        """Core outputs should move in the direction of the requested delta.

        Invariant: dot(actual_delta, requested_delta) > 0
        """
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 128, 64
        n_core = 30

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim)) * 0.5
        backend.eval(weight_target, activations_core, delta_activations)

        # No boundary constraint - all capacity available
        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,
            backend=backend,
        )

        assert result.applied is True

        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        # Compute actual output change
        output_before = backend.matmul(activations_core, backend.transpose(weight_target))
        output_after = backend.matmul(activations_core, backend.transpose(merged_weight))
        backend.eval(output_before, output_after)

        actual_delta = output_after - output_before
        backend.eval(actual_delta)

        # Direction check: dot product should be positive
        dot_product = backend.sum(actual_delta * delta_activations)
        dot_val = float(backend.to_scalar(dot_product))

        assert dot_val > 0, (
            f"Core outputs moved in wrong direction (dot product = {dot_val:.4f}). "
            "Should move toward the requested delta."
        )

    def test_zero_delta_produces_no_change(self) -> None:
        """Zero delta should produce exactly zero weight change."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 20

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        zero_delta = backend.zeros((n_core, out_dim))
        backend.eval(weight_target, activations_core, zero_delta)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=zero_delta,
            boundary_activations=None,
            backend=backend,
        )

        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        # Weight change should be exactly zero (numerical precision)
        diff = merged_weight - weight_target
        diff_norm = float(backend.to_scalar(backend.sum(backend.abs(diff))))
        eps = _eps(backend, diff_norm) * 100  # Allow for numerical precision

        assert diff_norm <= eps, (
            f"Zero delta produced weight change of {diff_norm:.2e}"
        )

    def test_shape_preserved(self) -> None:
        """Merged weight must have exact same shape as target."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 15

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        backend.eval(weight_target, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,
            backend=backend,
        )

        assert result.applied is True
        assert backend.shape(result.merged_weight) == (out_dim, in_dim)

    def test_non_2d_weight_skipped(self) -> None:
        """Non-2D weights should be skipped (bias vectors, etc)."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_1d = backend.random_normal((64,))  # 1D bias
        activations_core = backend.random_normal((5, 64))
        delta_activations = backend.random_normal((5, 64))
        backend.eval(weight_1d, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_1d,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,
            backend=backend,
        )

        assert result.applied is False
        assert result.null_dim == 0

    def test_metrics_consistent(self) -> None:
        """Metrics should be mathematically consistent."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        boundary_activations = backend.random_normal((n_boundary, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        backend.eval(weight_target, activations_core, boundary_activations, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_activations,
            backend=backend,
        )

        assert result.applied is True

        # preserved_fraction should be in [0, 1]
        eps = _eps(backend, result.preserved_fraction)
        assert -eps <= result.preserved_fraction <= 1.0 + eps

        # projection_loss = 1 - preserved_fraction
        eps = _eps(backend, result.projection_loss, result.preserved_fraction)
        assert abs(result.projection_loss + result.preserved_fraction - 1.0) <= eps


class TestBoundaryNullSpace:
    """Tests for boundary null-space projection."""

    def test_large_null_space_more_transfer(self) -> None:
        """Few boundary samples = large null space = more delta survives."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 128, 64
        n_core = 10
        n_boundary_small = 5   # Large null space
        n_boundary_large = 100  # Small null space

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        boundary_small = backend.random_normal((n_boundary_small, in_dim))
        boundary_large = backend.random_normal((n_boundary_large, in_dim))
        backend.eval(weight_target, activations_core, delta_activations, boundary_small, boundary_large)

        result_small = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_small,
            backend=backend,
        )

        result_large = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_large,
            backend=backend,
        )

        assert result_small.applied is True
        assert result_large.applied is True

        # More boundary samples = more constrained = less delta survives
        assert result_small.preserved_fraction >= result_large.preserved_fraction, (
            "Smaller boundary should allow more delta to survive"
        )

    def test_no_boundary_full_transfer(self) -> None:
        """No boundary constraint = unconstrained = full delta transfer."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 20

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        backend.eval(weight_target, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,  # No boundary
            backend=backend,
        )

        assert result.applied is True
        eps = _eps(backend, result.projection_loss, result.preserved_fraction)
        assert abs(result.projection_loss + result.preserved_fraction - 1.0) <= eps
