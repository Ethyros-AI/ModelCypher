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

"""Tests for transplant geometry - additive null-space merging.

Key geometric principles verified:
    1. Merged weight = target + delta (additive, not replacement)
    2. Merged weight is closer to target than source (null-space addition)
    3. Shape is preserved exactly

Note: CKA = 1.0 only on SHARED manifold. Random test data has no shared
geometry, so CKA thresholds are meaningless. We test structural properties
instead of correlation metrics.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.transplant import (
    TransplantDeltaResult,
    compute_transplant_delta,
)


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestAdditiveMerging:
    """Tests for additive merging (not replacement)."""

    def test_merged_closer_to_target(self) -> None:
        """Merged weight should be closer to target than any arbitrary matrix.

        We're adding delta to target in its null-space, so result resembles target
        more than a random matrix would.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 15

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim)) * 0.1
        backend.eval(weight_target, activations_core, delta_activations)

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

        # Closed-form expectation: boundary_activations=None -> N = I
        # delta_W_unc = pinv(A_core) @ delta_A; merged = weight_target + delta_W_unc.T
        delta_W_unc = backend.matmul(backend.pinv(activations_core), delta_activations)
        expected_merged = weight_target + backend.transpose(delta_W_unc)
        backend.eval(delta_W_unc, expected_merged)

        diff = backend.abs(merged_weight - expected_merged)
        backend.eval(diff)
        max_diff = float(backend.to_scalar(backend.max(diff)))
        eps = machine_epsilon(backend, expected_merged)
        assert max_diff <= eps

    def test_shape_exactly_preserved(self) -> None:
        """Merged weight must have exact same shape as target."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 10

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

    def test_additive_not_replacement(self) -> None:
        """Verify that merge is additive (target + delta), not replacement.

        If we pass zero delta, merged should equal target exactly.
        If we pass non-zero delta, merged should differ from target by delta.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 10

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        backend.eval(weight_target, activations_core)

        # Test 1: Zero delta should produce exact target
        zero_delta = backend.zeros((n_core, out_dim))
        backend.eval(zero_delta)

        result_zero = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=zero_delta,
            boundary_activations=None,
            backend=backend,
        )

        merged_zero = backend.array(result_zero.merged_weight)
        diff_zero = backend.subtract(merged_zero, weight_target)
        diff_zero_norm = float(backend.to_scalar(backend.norm(diff_zero)))

        eps = _eps(backend, diff_zero_norm) * 100
        assert diff_zero_norm <= eps, (
            f"Zero delta should produce target exactly, got diff {diff_zero_norm:.2e}"
        )

        # Test 2: Non-zero delta should produce different result
        nonzero_delta = backend.random_normal((n_core, out_dim))
        backend.eval(nonzero_delta)

        result_nonzero = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=nonzero_delta,
            boundary_activations=None,
            backend=backend,
        )

        merged_nonzero = backend.array(result_nonzero.merged_weight)
        diff_nonzero = backend.subtract(merged_nonzero, weight_target)
        diff_nonzero_norm = float(backend.to_scalar(backend.norm(diff_nonzero)))

        assert diff_nonzero_norm > eps, (
            f"Non-zero delta should change weights, but diff is only {diff_nonzero_norm:.2e}"
        )


class TestProjectionProperties:
    """Tests for null-space projection mathematical properties."""

    def test_projection_reduces_or_preserves_norm(self) -> None:
        """Null-space projection can only reduce or preserve delta norm.

        preserved_fraction <= 1.0 always (can't create energy from nothing)
        """
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 20

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

        # preserved_fraction <= 1.0 (projection can only reduce norm)
        eps = _eps(backend, result.preserved_fraction)
        assert result.preserved_fraction <= 1.0 + eps, (
            f"preserved_fraction {result.preserved_fraction} > 1.0 "
            "violates projection norm reduction property"
        )

        # projection_loss >= 0 (can't have negative loss)
        eps = _eps(backend, result.projection_loss)
        assert result.projection_loss >= -eps, (
            f"projection_loss {result.projection_loss} < 0 is invalid"
        )

    def test_metrics_sum_to_one(self) -> None:
        """preserved_fraction + projection_loss = 1.0 exactly."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 15, 10

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

        total = result.preserved_fraction + result.projection_loss
        eps = _eps(backend, total, 1.0)
        assert abs(total - 1.0) <= eps, (
            f"preserved_fraction ({result.preserved_fraction}) + "
            f"projection_loss ({result.projection_loss}) = {total} != 1.0"
        )
