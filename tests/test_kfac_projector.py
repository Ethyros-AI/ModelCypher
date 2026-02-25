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

"""Tests for K-FAC null-space projector."""

from __future__ import annotations

from modelcypher.core.domain.geometry.kfac_diagnostic import compute_kfac_diagnostic
from modelcypher.core.domain.geometry.kfac_projector import (
    compute_kfac_factors,
    project_kfac,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    svd_rank_threshold,
)


class TestKFACProjector:
    """K-FAC projector invariants."""

    def test_projector_idempotent(self, any_backend) -> None:
        """Projecting twice should match projecting once."""
        b = any_backend
        b.random_seed(101)

        n_probes = 40
        in_dim = 6
        out_dim = 5

        activations = b.random_normal((n_probes, in_dim))
        output_gradients = b.random_normal((n_probes, out_dim))
        delta = b.random_normal((out_dim, in_dim))
        b.eval(activations, output_gradients, delta)

        factors = compute_kfac_factors(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )

        p1 = project_kfac(delta, factors, backend=b)
        p2 = project_kfac(p1, factors, backend=b)

        diff = p2 - p1
        diff_norm_arr = b.sqrt(b.sum(diff * diff))
        ref_norm_arr = b.sqrt(b.sum(p1 * p1))
        b.eval(diff_norm_arr, ref_norm_arr)

        diff_norm = float(b.to_scalar(diff_norm_arr))
        ref_norm = float(b.to_scalar(ref_norm_arr))
        tol = machine_epsilon(b, p1) * max(1.0, ref_norm) * max(in_dim, out_dim)
        assert diff_norm <= tol

    def test_activation_null_direction_is_preserved(self, any_backend) -> None:
        """Directions in activation null space must survive K-FAC projection."""
        b = any_backend
        b.random_seed(202)

        n_probes = 48
        in_dim = 6
        out_dim = 4
        rank_a = 3

        # Build activations with known null columns (last in_dim-rank_a columns).
        latent = b.random_normal((n_probes, rank_a))
        basis = b.concatenate(
            [b.eye(rank_a), b.zeros((rank_a, in_dim - rank_a))],
            axis=1,
        )
        activations = b.matmul(latent, basis)
        output_gradients = b.random_normal((n_probes, out_dim))
        b.eval(activations, output_gradients)

        factors = compute_kfac_factors(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )

        # Delta uses only activation-null input directions (columns rank_a:).
        dense_null = b.random_normal((out_dim, in_dim - rank_a))
        delta = b.concatenate([b.zeros((out_dim, rank_a)), dense_null], axis=1)
        b.eval(delta)

        projected = project_kfac(delta, factors, backend=b)
        diff = projected - delta
        diff_norm_arr = b.sqrt(b.sum(diff * diff))
        delta_norm_arr = b.sqrt(b.sum(delta * delta))
        b.eval(diff_norm_arr, delta_norm_arr)

        diff_norm = float(b.to_scalar(diff_norm_arr))
        delta_norm = float(b.to_scalar(delta_norm_arr))
        tol = machine_epsilon(b, projected) * max(1.0, delta_norm) * max(in_dim, out_dim)
        assert diff_norm <= tol

    def test_threshold_matches_precision_formula(self, any_backend) -> None:
        """K-FAC threshold must follow max_dim * eps * max_kron_eigenvalue."""
        b = any_backend
        b.random_seed(303)

        n_probes = 24
        in_dim = 5
        out_dim = 3

        activations = b.random_normal((n_probes, in_dim))
        output_gradients = b.random_normal((n_probes, out_dim))
        b.eval(activations, output_gradients)

        diagnostic = compute_kfac_diagnostic(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )
        factors = compute_kfac_factors(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )

        scale = svd_rank_threshold(
            b,
            diagnostic.activation_eigenvalues,
            max(in_dim, out_dim),
        )
        eps = machine_epsilon(b, diagnostic.activation_eigenvalues)
        expected_threshold = max(diagnostic.max_kron_eigenvalue, eps) * scale

        tol = machine_epsilon(
            b,
            b.array([factors.kron_threshold, expected_threshold]),
        ) * max(1.0, abs(expected_threshold))
        assert abs(factors.kron_threshold - expected_threshold) <= tol

