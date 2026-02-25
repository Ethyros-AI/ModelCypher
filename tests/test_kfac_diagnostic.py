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

"""Tests for K-FAC diagnostics and activation-vs-behavior null-space geometry."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.kfac_diagnostic import (
    compute_kfac_diagnostic,
    count_activation_null_subset_violations,
    estimate_output_gradients_from_weight_gradients,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)


class TestKFACDiagnostic:
    """K-FAC null-space diagnostic tests."""

    def test_gain_ratio_matches_rank_theory(self, any_backend) -> None:
        """Synthetic low-rank factors should match exact rank arithmetic.

        If rank(A)=r_a and rank(S)=r_s:
            rank(S ⊗ A) = r_s * r_a
            null_rank_kfac = out_dim * in_dim - r_s * r_a
            null_rank_activation = out_dim * (in_dim - r_a)
        """
        b = any_backend
        b.random_seed(7)

        n_probes = 32
        in_dim = 4
        out_dim = 3
        rank_a = 2
        rank_s = 2

        latent_a = b.random_normal((n_probes, rank_a))
        basis_a = b.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        activations = b.matmul(latent_a, basis_a)  # [N, 4], rank 2

        latent_s = b.random_normal((n_probes, rank_s))
        basis_s = b.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        output_gradients = b.matmul(latent_s, basis_s)  # [N, 3], rank 2
        b.eval(activations, output_gradients)

        result = compute_kfac_diagnostic(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )

        expected_activation_null = out_dim * (in_dim - rank_a)  # 3 * 2 = 6
        expected_kfac_null = (out_dim * in_dim) - (rank_a * rank_s)  # 12 - 4 = 8
        expected_gain = expected_kfac_null / expected_activation_null

        assert result.activation_rank == rank_a
        assert result.activation_null_rank_weight == expected_activation_null
        assert result.kfac_null_rank == expected_kfac_null

        eps = machine_epsilon(
            b,
            b.array([result.kfac_gain_ratio, expected_gain]),
        )
        assert abs(result.kfac_gain_ratio - expected_gain) <= eps * max(1.0, expected_gain)

    @pytest.mark.parametrize(
        ("rank_a", "rank_s"),
        [
            (1, 1),
            (2, 2),
            (3, 1),
        ],
    )
    def test_activation_null_is_subset_of_kfac_null(
        self,
        any_backend,
        rank_a: int,
        rank_s: int,
    ) -> None:
        """Verify Null(K_cap) ⊆ Null(G_cap) on synthetic low-rank factors."""
        b = any_backend
        b.random_seed(17 + rank_a + rank_s)

        n_probes = 48
        in_dim = 6
        out_dim = 5

        rank_a = min(rank_a, in_dim - 1)
        rank_s = min(rank_s, out_dim)

        latent_a = b.random_normal((n_probes, rank_a))
        eye_a = b.eye(rank_a)
        tail_a = b.zeros((rank_a, in_dim - rank_a))
        basis_a = b.concatenate([eye_a, tail_a], axis=1)
        activations = b.matmul(latent_a, basis_a)

        latent_s = b.random_normal((n_probes, rank_s))
        eye_s = b.eye(rank_s)
        tail_s = b.zeros((rank_s, out_dim - rank_s))
        basis_s = b.concatenate([eye_s, tail_s], axis=1)
        output_gradients = b.matmul(latent_s, basis_s)
        b.eval(activations, output_gradients)

        result = compute_kfac_diagnostic(
            input_activations=activations,
            output_gradients=output_gradients,
            backend=b,
        )
        violations = count_activation_null_subset_violations(result, backend=b)
        assert violations == 0

    def test_reconstructs_rank1_output_gradients_from_weight_gradients(self, any_backend) -> None:
        """Rank-1 gradients dW = s a^T should recover s exactly (up to precision)."""
        b = any_backend
        b.random_seed(23)

        n_probes = 20
        in_dim = 7
        out_dim = 4

        activations = b.random_normal((n_probes, in_dim))
        true_output_grads = b.random_normal((n_probes, out_dim))
        b.eval(activations, true_output_grads)

        # Construct per-probe gradients exactly as rank-1 outer products:
        # dW_i = s_i @ a_i^T
        grads_3d = (
            b.reshape(true_output_grads, (n_probes, out_dim, 1))
            * b.reshape(activations, (n_probes, 1, in_dim))
        )
        grads_flat = b.reshape(grads_3d, (n_probes, out_dim * in_dim))
        b.eval(grads_flat)

        recovered = estimate_output_gradients_from_weight_gradients(
            per_probe_weight_gradients=grads_flat,
            input_activations=activations,
            weight_shape=(out_dim, in_dim),
            backend=b,
        )

        diff = recovered - true_output_grads
        diff_norm_arr = b.sqrt(b.sum(diff * diff))
        ref_norm_arr = b.sqrt(b.sum(true_output_grads * true_output_grads))
        b.eval(diff_norm_arr, ref_norm_arr)

        diff_norm = float(b.to_scalar(diff_norm_arr))
        ref_norm = float(b.to_scalar(ref_norm_arr))
        tol = sqrt_scalar(machine_epsilon(b, recovered), b) * max(1.0, ref_norm)
        assert diff_norm <= tol

