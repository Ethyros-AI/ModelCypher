# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

"""Tests for K-FAC curvature alignment diagnostic."""

from __future__ import annotations

from modelcypher.core.domain.geometry.kfac_projector import compute_kfac_factors
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.training.kfac_curvature_monitor import (
    aggregate_epoch_curvature,
    compute_curvature_alignment,
)


class TestKFACCurvatureMonitor:
    """K-FAC curvature alignment diagnostic invariants."""

    def test_fractions_sum_to_one_or_less(self, any_backend) -> None:
        """Top-10% + remaining fractions should be consistent with total."""
        b = any_backend
        b.random_seed(401)

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

        result = compute_curvature_alignment(
            delta_weight=delta,
            factors=factors,
            layer_name="test",
            backend=b,
        )

        # Top-10% <= Top-25% (nested subsets)
        assert result.top_10pct_fraction <= result.top_25pct_fraction + 1e-6
        # All fractions are non-negative
        assert result.top_10pct_fraction >= 0.0
        assert result.top_25pct_fraction >= 0.0
        assert result.null_fraction >= 0.0
        # Top-25% fraction should not exceed 1.0
        assert result.top_25pct_fraction <= 1.0 + 1e-6
        # n_total is correct
        assert result.n_total_directions == in_dim * out_dim

    def test_null_delta_has_full_null_fraction(self, any_backend) -> None:
        """A delta entirely in the activation null space should have high null fraction."""
        b = any_backend
        b.random_seed(402)

        n_probes = 48
        in_dim = 6
        out_dim = 4
        rank_a = 3

        # Build activations with known null columns (last in_dim-rank_a)
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

        # Delta only in activation-null input columns
        dense_null = b.random_normal((out_dim, in_dim - rank_a))
        delta = b.concatenate([b.zeros((out_dim, rank_a)), dense_null], axis=1)
        b.eval(delta)

        result = compute_curvature_alignment(
            delta_weight=delta,
            factors=factors,
            layer_name="null_test",
            backend=b,
        )

        # Since Null(K_cap) ⊆ Null(G_cap), a delta in the activation null space
        # must also be in the K-FAC null space
        eps = machine_epsilon(b, delta)
        assert result.null_fraction > 1.0 - eps * in_dim * out_dim

    def test_aggregate_report(self, any_backend) -> None:
        """Epoch aggregation should compute correct median/max."""
        b = any_backend
        b.random_seed(403)

        n_probes = 30
        in_dim = 4
        out_dim = 3

        results = []
        for i in range(3):
            activations = b.random_normal((n_probes, in_dim))
            output_gradients = b.random_normal((n_probes, out_dim))
            delta = b.random_normal((out_dim, in_dim))
            b.eval(activations, output_gradients, delta)

            factors = compute_kfac_factors(
                input_activations=activations,
                output_gradients=output_gradients,
                backend=b,
            )
            result = compute_curvature_alignment(
                delta_weight=delta,
                factors=factors,
                layer_name=f"layer_{i}",
                backend=b,
            )
            results.append(result)

        report = aggregate_epoch_curvature(epoch=1, per_layer_results=results)
        assert report.n_layers == 3
        assert report.max_top_10pct_fraction >= report.median_top_10pct_fraction
        assert report.median_top_10pct_fraction >= 0.0
        assert report.median_null_fraction >= 0.0
