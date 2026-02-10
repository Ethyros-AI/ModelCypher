# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for LoRA isometry composite scores.

Validates that composite scores (IS, Defect, AWP, SBF) behave correctly
on synthetic weight matrices with known geometric properties.
"""

from __future__ import annotations

import math

import pytest


class TestIsometryComposites:
    """Test composite score computation."""

    def test_perfect_isometry_score_near_one(self, any_backend):
        """Identity perturbation (ΔW=0) -> IS ~ 1.0, defect ~ 0."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
        )

        W = any_backend.random_normal((64, 32), dtype="float32")
        any_backend.eval(W)
        # Zero delta
        delta = any_backend.multiply(W, 0.0)
        any_backend.eval(delta)
        W_mod = any_backend.add(W, delta)
        any_backend.eval(W_mod)

        iso = compute_isometry_metrics(W, W_mod, any_backend)
        weyl = compute_weyl_utilization(W, delta, any_backend)
        composites = compute_isometry_composites(iso, weyl)

        assert composites.isometry_score > 0.95, (
            f"Expected IS near 1.0 for identity, got {composites.isometry_score}"
        )
        assert composites.isometry_defect < 0.1, (
            f"Expected defect near 0 for identity, got {composites.isometry_defect}"
        )
        assert composites.alignment_weighted_perturbation < 0.01

    def test_random_lora_lower_score(self, any_backend):
        """Random ΔW should have IS < isometric IS."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
        )

        any_backend.random_seed(42)
        W = any_backend.random_normal((64, 32), dtype="float32")
        any_backend.eval(W)

        # Identity case
        delta_zero = any_backend.multiply(W, 0.0)
        any_backend.eval(delta_zero)
        iso_id = compute_isometry_metrics(W, W, any_backend)
        weyl_id = compute_weyl_utilization(W, delta_zero, any_backend)
        c_id = compute_isometry_composites(iso_id, weyl_id)

        # Random perturbation
        A = any_backend.random_normal((4, 32), dtype="float32")
        B = any_backend.random_normal((64, 4), dtype="float32")
        delta = any_backend.matmul(B, A)
        any_backend.eval(delta)
        W_mod = any_backend.add(W, delta)
        any_backend.eval(W_mod)

        iso_rand = compute_isometry_metrics(W, W_mod, any_backend)
        weyl_rand = compute_weyl_utilization(W, delta, any_backend)
        c_rand = compute_isometry_composites(iso_rand, weyl_rand)

        assert c_rand.isometry_score < c_id.isometry_score, (
            f"Random IS ({c_rand.isometry_score}) should be < identity IS ({c_id.isometry_score})"
        )
        assert c_rand.isometry_defect > c_id.isometry_defect

    def test_orthogonal_lora_low_awp(self, any_backend):
        """Null-space ΔW should have AWP near zero."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
            create_synthetic_orthogonal_lora,
        )

        lora = create_synthetic_orthogonal_lora(
            m=64, n=32, rank=4, scale=0.1, backend=any_backend,
        )
        delta = any_backend.subtract(lora.weight_modified, lora.weight_original)
        any_backend.eval(delta)

        iso = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, any_backend,
        )
        weyl = compute_weyl_utilization(lora.weight_original, delta, any_backend)
        composites = compute_isometry_composites(iso, weyl)

        # Orthogonal LoRA has low η, so AWP = η × RFD should be small
        assert composites.alignment_weighted_perturbation < 0.05, (
            f"Expected low AWP for orthogonal LoRA, got {composites.alignment_weighted_perturbation}"
        )

    def test_sbf_none_without_sigma_k(self, any_backend):
        """SBF is None when sigma_k not provided."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
        )

        W = any_backend.random_normal((32, 16), dtype="float32")
        delta = any_backend.random_normal((32, 16), dtype="float32")
        delta = any_backend.multiply(delta, 0.01)
        W_mod = any_backend.add(W, delta)
        any_backend.eval(W, delta, W_mod)

        iso = compute_isometry_metrics(W, W_mod, any_backend)
        weyl = compute_weyl_utilization(W, delta, any_backend)
        composites = compute_isometry_composites(iso, weyl, sigma_k=None)

        assert composites.spectral_budget_fraction is None

    def test_sbf_scales_with_perturbation(self, any_backend):
        """Larger ΔW -> larger SBF."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
        )

        any_backend.random_seed(123)
        W = any_backend.random_normal((64, 32), dtype="float32")
        any_backend.eval(W)

        # Get sigma_k from W
        _, S, _ = any_backend.svd(W)
        any_backend.eval(S)
        s_list = any_backend.tolist(S)
        sigma_k = float(s_list[-1])  # Smallest SV

        # Small perturbation
        delta_small = any_backend.random_normal((64, 32), dtype="float32")
        delta_small = any_backend.multiply(delta_small, 0.01)
        any_backend.eval(delta_small)
        W_mod_small = any_backend.add(W, delta_small)
        any_backend.eval(W_mod_small)

        iso_s = compute_isometry_metrics(W, W_mod_small, any_backend)
        weyl_s = compute_weyl_utilization(W, delta_small, any_backend)
        c_s = compute_isometry_composites(iso_s, weyl_s, sigma_k=sigma_k)

        # Large perturbation (same direction, scaled up)
        delta_large = any_backend.multiply(delta_small, 10.0)
        any_backend.eval(delta_large)
        W_mod_large = any_backend.add(W, delta_large)
        any_backend.eval(W_mod_large)

        iso_l = compute_isometry_metrics(W, W_mod_large, any_backend)
        weyl_l = compute_weyl_utilization(W, delta_large, any_backend)
        c_l = compute_isometry_composites(iso_l, weyl_l, sigma_k=sigma_k)

        assert c_l.spectral_budget_fraction > c_s.spectral_budget_fraction, (
            f"Large SBF ({c_l.spectral_budget_fraction}) should exceed "
            f"small SBF ({c_s.spectral_budget_fraction})"
        )

    def test_composites_monotonic_with_scale(self, any_backend):
        """Scaling ΔW up -> defect increases, IS decreases."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_composites,
            compute_isometry_metrics,
            compute_weyl_utilization,
        )

        any_backend.random_seed(99)
        W = any_backend.random_normal((64, 32), dtype="float32")
        any_backend.eval(W)

        base_delta = any_backend.random_normal((64, 32), dtype="float32")
        any_backend.eval(base_delta)

        prev_is = 2.0  # Start above max
        prev_defect = -1.0  # Start below min

        for scale in [0.01, 0.1, 0.5]:
            delta = any_backend.multiply(base_delta, scale)
            any_backend.eval(delta)
            W_mod = any_backend.add(W, delta)
            any_backend.eval(W_mod)

            iso = compute_isometry_metrics(W, W_mod, any_backend)
            weyl = compute_weyl_utilization(W, delta, any_backend)
            c = compute_isometry_composites(iso, weyl)

            assert c.isometry_score <= prev_is + 1e-6, (
                f"IS should decrease with scale: {c.isometry_score} > {prev_is} at scale={scale}"
            )
            assert c.isometry_defect >= prev_defect - 1e-6, (
                f"Defect should increase with scale: {c.isometry_defect} < {prev_defect} at scale={scale}"
            )
            prev_is = c.isometry_score
            prev_defect = c.isometry_defect

    def test_multiplicative_catches_single_bad_factor(self, any_backend):
        """IS is low when any ONE input metric is bad."""
        from modelcypher.experimental.lora_isometry import (
            IsometryMetrics,
            IsometryCompositeScores,
            WeylMetrics,
            compute_isometry_composites,
        )

        # Construct "good" metrics manually
        good_iso = IsometryMetrics(
            spectral_preservation_ratio=0.99,
            spectral_angle=1.0,
            condition_ratio=1.01,
            grassmann_distance=0.02,
            relative_frobenius_deviation=0.01,
            rank_original=32,
            rank_modified=32,
            isometry_ratio=0.98,
        )
        good_weyl = WeylMetrics(
            weyl_utilization=0.01,
            delta_spectral_norm=0.1,
            max_singular_value_change=0.001,
            max_change_index=0,
            delta_effective_rank=4,
            projection_depth=16,
        )
        good_c = compute_isometry_composites(good_iso, good_weyl)
        assert good_c.isometry_score > 0.9, f"Good case IS={good_c.isometry_score}"

        # Bad SPR (spectral collapse)
        bad_spr_iso = IsometryMetrics(
            spectral_preservation_ratio=0.1,
            spectral_angle=1.0,
            condition_ratio=1.01,
            grassmann_distance=0.02,
            relative_frobenius_deviation=0.01,
            rank_original=32,
            rank_modified=32,
            isometry_ratio=0.1,
        )
        bad_spr_c = compute_isometry_composites(bad_spr_iso, good_weyl)
        assert bad_spr_c.isometry_score < 0.2, (
            f"Bad SPR should tank IS, got {bad_spr_c.isometry_score}"
        )

        # Bad Weyl utilization (aligned perturbation)
        bad_weyl = WeylMetrics(
            weyl_utilization=0.95,
            delta_spectral_norm=0.1,
            max_singular_value_change=0.095,
            max_change_index=0,
            delta_effective_rank=4,
            projection_depth=16,
        )
        bad_weyl_c = compute_isometry_composites(good_iso, bad_weyl)
        assert bad_weyl_c.isometry_score < 0.1, (
            f"Bad η should tank IS, got {bad_weyl_c.isometry_score}"
        )

        # Bad Grassmann distance (subspace rotated)
        bad_grass_iso = IsometryMetrics(
            spectral_preservation_ratio=0.99,
            spectral_angle=80.0,
            condition_ratio=1.01,
            grassmann_distance=math.pi / 2 - 0.01,
            relative_frobenius_deviation=0.01,
            rank_original=32,
            rank_modified=32,
            isometry_ratio=0.1,
        )
        bad_grass_c = compute_isometry_composites(bad_grass_iso, good_weyl)
        assert bad_grass_c.isometry_score < 0.05, (
            f"Bad Grassmann should tank IS, got {bad_grass_c.isometry_score}"
        )
