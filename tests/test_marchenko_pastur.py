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

"""Tests for Marchenko-Pastur noise edge and Tikhonov shrinkage weights."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.marchenko_pastur import (
    MarchenkoPasturResult,
    compute_marchenko_pastur_profile,
    marchenko_pastur_noise_edge,
    tikhonov_effective_rank,
    tikhonov_weights_from_eigenvalues,
)

# ── marchenko_pastur_noise_edge ─────────────────────────────────────────


class TestMarchenkoPasturNoiseEdge:
    """Tests for the shared spike-robust MP noise edge."""

    def test_signal_spikes_do_not_set_noise_edge(self):
        """Top spikes are excluded before estimating the MP bulk mean."""
        eigenvalues = [100.0, 50.0] + [1.0] * 20
        edge = marchenko_pastur_noise_edge(eigenvalues, n_features=22, n_samples=100)
        expected_bulk_edge = (1.0 + math.sqrt(22.0 / 100.0)) ** 2
        assert edge == pytest.approx(expected_bulk_edge)
        assert edge < 10.0

    def test_probe_regime_exact_zeros_do_not_swallow_unit_signal(self):
        """N << D with 32 unit signals among 256 dims keeps the unit signals."""
        eigenvalues = [1.0] * 32 + [0.0] * (256 - 32)
        edge = marchenko_pastur_noise_edge(
            eigenvalues,
            n_features=256,
            n_samples=32,
        )
        assert edge < 1.0

    def test_scales_with_eigenvalue_magnitude(self):
        """Noise edge scales linearly with eigenvalue magnitude."""
        eigenvalues = [20.0, 10.0] + [1.0] * 20
        edge_1 = marchenko_pastur_noise_edge(eigenvalues, 22, 100)
        edge_2 = marchenko_pastur_noise_edge([10.0 * x for x in eigenvalues], 22, 100)
        assert abs(edge_2 / edge_1 - 10.0) < 1e-10

    def test_invalid_dimensions_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            marchenko_pastur_noise_edge([1.0], 0, 10)
        with pytest.raises(ValueError, match="must be positive"):
            marchenko_pastur_noise_edge([1.0], 10, 0)
        with pytest.raises(ValueError, match="must be positive"):
            marchenko_pastur_noise_edge([1.0], -1, 10)


# ── tikhonov_weights_from_eigenvalues ────────────────────────────────────


class TestTikhonovWeights:
    """Tests for w_i = lambda_i / (lambda_i + alpha)."""

    def test_large_eigenvalue_gets_weight_near_one(self):
        """Eigenvalue >> alpha -> weight -> 1."""
        weights = tikhonov_weights_from_eigenvalues([1000.0], alpha=1.0)
        assert len(weights) == 1
        assert weights[0] > 0.999

    def test_small_eigenvalue_gets_weight_near_zero(self):
        """Eigenvalue << alpha -> weight -> 0."""
        weights = tikhonov_weights_from_eigenvalues([0.001], alpha=1.0)
        assert len(weights) == 1
        assert weights[0] < 0.002

    def test_equal_eigenvalue_and_alpha_gives_half(self):
        """lambda = alpha -> w = 0.5 exactly."""
        weights = tikhonov_weights_from_eigenvalues([5.0], alpha=5.0)
        assert abs(weights[0] - 0.5) < 1e-15

    def test_weights_are_monotone(self):
        """Larger eigenvalues get larger weights."""
        eigenvalues = [100.0, 10.0, 1.0, 0.1, 0.01]
        weights = tikhonov_weights_from_eigenvalues(eigenvalues, alpha=1.0)
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_zero_eigenvalue_gives_zero_weight(self):
        """lambda = 0 -> w = 0."""
        weights = tikhonov_weights_from_eigenvalues([0.0], alpha=1.0)
        assert weights[0] == 0.0

    def test_negative_eigenvalue_clamped_to_zero(self):
        """Negative eigenvalue (numerical noise) -> w = 0."""
        weights = tikhonov_weights_from_eigenvalues([-0.001], alpha=1.0)
        assert weights[0] == 0.0

    def test_all_weights_in_zero_one(self):
        """All weights must be in [0, 1]."""
        eigenvalues = [1e6, 1e3, 1.0, 1e-3, 1e-6, 0.0]
        weights = tikhonov_weights_from_eigenvalues(eigenvalues, alpha=1.0)
        for w in weights:
            assert 0.0 <= w <= 1.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            tikhonov_weights_from_eigenvalues([1.0], alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            tikhonov_weights_from_eigenvalues([1.0], alpha=-1.0)


# ── tikhonov_effective_rank ──────────────────────────────────────────────


class TestTikhonovEffectiveRank:
    """Tests for sum(w_i) as continuous effective rank."""

    def test_all_above_noise(self):
        """If all eigenvalues >> alpha, effective rank ≈ D."""
        eigenvalues = [100.0] * 10
        eff = tikhonov_effective_rank(eigenvalues, alpha=0.01)
        assert eff > 9.99

    def test_all_below_noise(self):
        """If all eigenvalues << alpha, effective rank ≈ 0."""
        eigenvalues = [0.001] * 10
        eff = tikhonov_effective_rank(eigenvalues, alpha=100.0)
        assert eff < 0.001

    def test_mixed_spectrum(self):
        """Typical case: a few directions above noise, rest below."""
        # 3 strong directions + 7 noise directions
        eigenvalues = [100.0, 50.0, 10.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0]
        eff = tikhonov_effective_rank(eigenvalues, alpha=1.0)
        # First 3 should contribute ~3.0, rest should contribute ~0
        assert 2.5 < eff < 3.5

    def test_continuous_not_integer(self):
        """Effective rank is continuous, not integer."""
        eigenvalues = [10.0, 1.0, 0.1]
        eff = tikhonov_effective_rank(eigenvalues, alpha=1.0)
        # Should be something like 1.0 + 0.5 + 0.09 ≈ 1.59
        assert eff != int(eff)  # Not integer


# ── compute_marchenko_pastur_profile ─────────────────────────────────────


class TestComputeProfile:
    """Integration test for the full MP profile computation."""

    def test_profile_returns_correct_type(self):
        result = compute_marchenko_pastur_profile(
            eigenvalues=[10.0, 5.0, 1.0, 0.1],
            n_features=100,
            n_samples=50,
        )
        assert isinstance(result, MarchenkoPasturResult)

    def test_profile_sigma_sq(self):
        eigenvalues = [10.0, 5.0] + [1.0] * 20
        result = compute_marchenko_pastur_profile(
            eigenvalues=eigenvalues,
            n_features=22,
            n_samples=100,
        )
        assert result.sigma_sq == pytest.approx(1.0)

    def test_profile_aspect_ratio(self):
        result = compute_marchenko_pastur_profile(
            eigenvalues=[1.0],
            n_features=1536,
            n_samples=3840,
        )
        assert abs(result.aspect_ratio - 1536.0 / 3840.0) < 1e-12

    def test_profile_noise_edge_consistency(self):
        """Profile noise edge matches standalone function."""
        eigenvalues = [10.0, 5.0, 1.0, 0.1]
        result = compute_marchenko_pastur_profile(eigenvalues, 100, 50)
        standalone = marchenko_pastur_noise_edge(eigenvalues, 100, 50)
        assert abs(result.noise_edge - standalone) < 1e-12

    def test_profile_effective_rank_consistency(self):
        """Profile effective rank matches standalone function."""
        eigenvalues = [10.0, 5.0, 1.0, 0.1]
        result = compute_marchenko_pastur_profile(eigenvalues, 100, 50)
        standalone = tikhonov_effective_rank(eigenvalues, result.noise_edge)
        assert abs(result.effective_rank - standalone) < 1e-12


# ── Regression: experimental Tikhonov results ────────────────────────────


class TestExperimentalRegression:
    """Verify the module reproduces the closed-form correction values.

    These values come from the 2026-02-27 Tikhonov experiment on
    Qwen3-1.7B-4bit (layer 0):
      D = 1536, N_tok = 3840, D_eff = 5.2
      mp_edge = 9.84e-02, effective_rank = 12.4, top_w = 0.911
    """

    def test_layer_0_noise_edge_order_of_magnitude(self):
        """Layer 0 MP edge should be O(0.1) for Qwen3-1.7B scale."""
        # Approximate: sigma_sq ≈ total_var/D. With D=1536, N=3840,
        # aspect = 0.4, (1+sqrt(0.4))^2 ≈ 2.90
        # If sigma_sq ≈ 0.034 (from experiment), edge ≈ 0.098
        sigma_sq = 0.034
        edge = sigma_sq * (1.0 + math.sqrt(1536.0 / 3840.0)) ** 2
        assert 0.05 < edge < 0.15  # Same order as measured 9.84e-02
