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
