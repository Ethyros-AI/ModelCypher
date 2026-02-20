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

from modelcypher.core.domain.training.spectral_budget import (
    compute_budget_ratios,
    compute_initialization_vectors,
    compute_projected_residuals,
    is_budget_exhausted,
)


class TestComputeBudgetRatios:
    """Tests for compute_budget_ratios() — spectral budget per LoRA layer."""

    def test_empty_products_returns_empty(self, any_backend):
        """Empty lora_products → empty ratios list."""
        ratios = compute_budget_ratios([], any_backend)
        assert ratios == []

    def test_known_ratio_identity(self, any_backend):
        """scale=1, A=I, B=I → product spectral norm = 1.0, ratio = 1/sigma_k."""
        b = any_backend
        rank = 4
        A = b.eye(rank)
        B = b.eye(rank)
        b.eval(A, B)

        sigma_k = 2.0
        ratios = compute_budget_ratios([(1.0, A, B, sigma_k)], b)

        assert len(ratios) == 1
        assert ratios[0] == pytest.approx(0.5, abs=1e-5)

    def test_sigma_k_zero_skipped(self, any_backend):
        """sigma_k <= 0 entries are silently skipped."""
        b = any_backend
        A = b.eye(2)
        B = b.eye(2)
        b.eval(A, B)

        ratios = compute_budget_ratios(
            [(1.0, A, B, 0.0), (1.0, A, B, -1.0)], b
        )
        assert ratios == []

    def test_scale_factor_applied(self, any_backend):
        """scale=2 doubles the spectral norm → doubles ratio."""
        b = any_backend
        A = b.eye(3)
        B = b.eye(3)
        b.eval(A, B)

        sigma_k = 1.0
        ratios_s1 = compute_budget_ratios([(1.0, A, B, sigma_k)], b)
        ratios_s2 = compute_budget_ratios([(2.0, A, B, sigma_k)], b)

        assert ratios_s2[0] == pytest.approx(2.0 * ratios_s1[0], abs=1e-5)


class TestIsBudgetExhausted:
    """Tests for is_budget_exhausted() — pure Python exhaustion check."""

    def test_empty_ratios(self):
        """Empty ratios → (False, 0.0)."""
        exhausted, median = is_budget_exhausted([], 0.9)
        assert exhausted is False
        assert median == 0.0

    def test_scalar_threshold_below(self):
        """Median < threshold → not exhausted."""
        exhausted, median = is_budget_exhausted([0.1, 0.2, 0.3], 0.5)
        assert exhausted is False
        assert median == pytest.approx(0.2)

    def test_scalar_threshold_above(self):
        """Median > threshold → exhausted."""
        exhausted, median = is_budget_exhausted([0.6, 0.7, 0.8], 0.5)
        assert exhausted is True
        assert median == pytest.approx(0.7)

    def test_weyl_crossing_breached(self):
        """Per-layer ratio > gap/(2*sigma_k) → exhausted."""
        ratios = [0.5, 0.1]
        spectral_gaps = [0.2, 0.4]
        sigma_ks = [1.0, 1.0]
        # Layer 0: crossing = 0.2/(2*1.0) = 0.1 → 0.5 > 0.1 → crossed
        exhausted, _ = is_budget_exhausted(
            ratios, threshold=10.0,
            spectral_gaps=spectral_gaps, sigma_ks=sigma_ks,
        )
        assert exhausted is True

    def test_weyl_crossing_safe(self):
        """All ratios below per-layer Weyl threshold → not exhausted."""
        ratios = [0.01, 0.02]
        spectral_gaps = [0.5, 0.5]
        sigma_ks = [1.0, 1.0]
        # crossing = 0.5/(2*1.0) = 0.25 → both below
        exhausted, _ = is_budget_exhausted(
            ratios, threshold=10.0,
            spectral_gaps=spectral_gaps, sigma_ks=sigma_ks,
        )
        assert exhausted is False


class TestComputeInitializationVectors:
    """Tests for compute_initialization_vectors() — extract u_k, v_k."""

    def test_shapes_correct(self, any_backend):
        """u_k is [out, 1], v_k is [in, 1]."""
        b = any_backend
        W = b.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # [3, 2]
        b.eval(W)

        u_k, v_k = compute_initialization_vectors(W, structural_rank=1, backend=b)

        assert u_k.shape == (3, 1)
        assert v_k.shape == (2, 1)

    def test_first_sv_direction(self, any_backend):
        """For diagonal W, u_1 should align with first standard basis."""
        b = any_backend
        # W = diag(5, 3, 1) — first SV direction is e_1
        W = b.array([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
        b.eval(W)

        u_k, v_k = compute_initialization_vectors(W, structural_rank=1, backend=b)

        # u_1 should be approximately [1, 0, 0] (up to sign)
        u_vals = [abs(float(b.to_scalar(u_k[i, 0]))) for i in range(3)]
        assert u_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert u_vals[1] == pytest.approx(0.0, abs=1e-5)

    def test_second_sv_direction(self, any_backend):
        """For diagonal W, u_2 should align with second standard basis."""
        b = any_backend
        W = b.array([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
        b.eval(W)

        u_k, v_k = compute_initialization_vectors(W, structural_rank=2, backend=b)

        u_vals = [abs(float(b.to_scalar(u_k[i, 0]))) for i in range(3)]
        assert u_vals[1] == pytest.approx(1.0, abs=1e-5)
        assert u_vals[0] == pytest.approx(0.0, abs=1e-5)


class TestComputeProjectedResiduals:
    """Tests for compute_projected_residuals() — tighter Weyl diagnostic."""

    def test_aligned_delta_gives_max_residual(self, any_backend):
        """Delta aligned with u_k, v_k: projected residual == spectral norm."""
        b = any_backend
        # W = diag(5, 3, 1), so u_1=[1,0,0], v_1=[1,0,0]
        # delta = scale * u_1 @ v_1^T = [[scale, 0, 0], [0,0,0], [0,0,0]]
        # lora_a [in=3, r=1], lora_b [r=1, out=3]
        # product = lora_a @ lora_b must give [[1,0,0],[0,0,0],[0,0,0]]^T
        # = [[1,0,0],[0,0,0],[0,0,0]] (since it's [in,out]=[3,3])
        # For the monitoring convention: delta_weight = (product)^T = [out,in]
        # We want delta_weight = u_1 @ v_1^T = [[1],[0],[0]] @ [[1,0,0]]
        # So product^T = [[1,0,0],[0,0,0],[0,0,0]] => product = same (symmetric)
        # lora_a = [[1],[0],[0]] [3,1], lora_b = [[1,0,0]] [1,3]
        lora_a = b.array([[1.0], [0.0], [0.0]])  # [3, 1]
        lora_b = b.array([[1.0, 0.0, 0.0]])       # [1, 3]
        b.eval(lora_a, lora_b)

        u_k = b.array([[1.0], [0.0], [0.0]])
        v_k = b.array([[1.0], [0.0], [0.0]])
        b.eval(u_k, v_k)

        scale = 2.0
        sigma_k = 5.0
        products = [(scale, lora_a, lora_b, sigma_k)]
        residuals = compute_projected_residuals(products, [u_k], [v_k], b)

        assert len(residuals) == 1
        # Projected residual = |scale * (lora_b @ u_k)^T @ (lora_a^T @ v_k)|
        # = |2 * [1] * [1]| = 2.0
        assert residuals[0] == pytest.approx(2.0, abs=1e-5)

        # Spectral norm of delta is also 2.0, so residual == spectral norm
        budget = compute_budget_ratios(products, b)
        assert residuals[0] == pytest.approx(budget[0] * sigma_k, abs=1e-4)

    def test_orthogonal_delta_gives_zero(self, any_backend):
        """Delta orthogonal to u_k, v_k: projected residual ~= 0."""
        b = any_backend
        # u_k = [1,0,0], v_k = [1,0,0]
        # delta acts in direction [0,1,0] x [0,1,0] — orthogonal
        lora_a = b.array([[0.0], [1.0], [0.0]])  # [3, 1]
        lora_b = b.array([[0.0, 1.0, 0.0]])       # [1, 3]
        b.eval(lora_a, lora_b)

        u_k = b.array([[1.0], [0.0], [0.0]])
        v_k = b.array([[1.0], [0.0], [0.0]])
        b.eval(u_k, v_k)

        products = [(1.0, lora_a, lora_b, 5.0)]
        residuals = compute_projected_residuals(products, [u_k], [v_k], b)

        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self, any_backend):
        """Manual computation of projected residual for a known case."""
        b = any_backend
        # lora_a = [[1, 0], [0, 1], [0, 0]]  [3, 2]
        # lora_b = [[1, 0, 0], [0, 1, 0]]     [2, 3]
        # product = lora_a @ lora_b = [[1,0,0],[0,1,0],[0,0,0]]  [3, 3]
        # product^T (delta in W-space) = [[1,0,0],[0,1,0],[0,0,0]]
        # u_k = [1/sqrt(2), 1/sqrt(2), 0], v_k = [1/sqrt(2), 1/sqrt(2), 0]
        # u_k^T @ delta @ v_k = [1/√2, 1/√2, 0] @ [[1,0,0],[0,1,0],[0,0,0]] @ [1/√2, 1/√2, 0]^T
        # = [1/√2, 1/√2, 0] @ [1/√2, 1/√2, 0]^T = 0.5 + 0.5 = 1.0
        import math
        s = 1.0 / math.sqrt(2.0)

        lora_a = b.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lora_b = b.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        u_k = b.array([[s], [s], [0.0]])
        v_k = b.array([[s], [s], [0.0]])
        b.eval(lora_a, lora_b, u_k, v_k)

        scale = 1.0
        products = [(scale, lora_a, lora_b, 1.0)]
        residuals = compute_projected_residuals(products, [u_k], [v_k], b)

        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(1.0, abs=1e-5)

    def test_empty_inputs(self, any_backend):
        """Empty lora_products → empty residuals."""
        residuals = compute_projected_residuals([], [], [], any_backend)
        assert residuals == []

    def test_mismatched_lengths_skips(self, any_backend):
        """More products than vectors → excess products skipped."""
        b = any_backend
        A = b.eye(2)
        B = b.eye(2)
        b.eval(A, B)

        products = [(1.0, A, B, 1.0), (1.0, A, B, 1.0)]
        u_ks = [b.array([[1.0], [0.0]])]
        v_ks = [b.array([[1.0], [0.0]])]

        residuals = compute_projected_residuals(products, u_ks, v_ks, b)
        # Only first product has matching vectors
        assert len(residuals) == 1
