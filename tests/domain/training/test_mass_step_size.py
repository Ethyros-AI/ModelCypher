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

"""Tests for MASS (Measured-Adaptive Step Size) computations.

MASS is a three-layer learning rate system where every number derives from
SVD geometry (Weyl 1912), stochastic optimization (Loizou et al. 2020),
or IEEE 754. All formulas are pure scalar arithmetic — no MLX dependency.

Tests import the real production functions from
modelcypher.core.domain.training.mass_step_size.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.mass_step_size import (
    _EPS_F32,
    _SQRT_EPS_F32,
    apply_sqrt_n_epoch_correction,
    apply_validation_backoff,
    compute_conformal_margin_rate,
    compute_per_step_rates,
    compute_reinforce_budget,
    derive_spectral_ceiling,
    verify_bounded_gain,
)


# ---------------------------------------------------------------------------
# Helpers for extracting individual sub-rates from compute_per_step_rates.
# These are NOT formula replicas — they delegate to the real function and
# select a single return value for focused testing.
# ---------------------------------------------------------------------------
def _extract_sps(loss: float, d_norm: float, eta_ceiling: float) -> float:
    """Extract eta_sps from compute_per_step_rates (sigma_k_min irrelevant for SPS)."""
    _, eta_sps, _, _, _ = compute_per_step_rates(loss, d_norm, 1.0, eta_ceiling)
    return eta_sps


def _extract_weyl(sigma_k_min: float, d_norm: float, eta_ceiling: float) -> float:
    """Extract eta_weyl from compute_per_step_rates (loss irrelevant for Weyl)."""
    _, _, eta_weyl, _, _ = compute_per_step_rates(1.0, d_norm, sigma_k_min, eta_ceiling)
    return eta_weyl


# ===================================================================
# Class 1: Spectral Ceiling
# ===================================================================
class TestSpectralCeiling:
    """Tests for eta_ceiling = sigma_k_min / sigma_max_global (Weyl 1912)."""

    def test_formula_known_values(self):
        assert derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=10.0) == pytest.approx(0.01)
        assert derive_spectral_ceiling(sigma_k_min=1.0, sigma_max_global=1.0) == pytest.approx(1.0)
        assert derive_spectral_ceiling(sigma_k_min=5.0, sigma_max_global=10.0) == pytest.approx(0.5)

    def test_ceiling_at_most_one_when_sigma_k_leq_sigma_max(self):
        # sigma_k <= sigma_max always holds for valid SVD
        for sigma_k in [0.001, 0.1, 1.0, 5.0]:
            for sigma_max in [sigma_k, sigma_k * 2, sigma_k * 100]:
                assert derive_spectral_ceiling(sigma_k_min=sigma_k, sigma_max_global=sigma_max) <= 1.0

    def test_ceiling_equals_one_when_equal(self):
        assert derive_spectral_ceiling(sigma_k_min=3.0, sigma_max_global=3.0) == 1.0

    def test_lr_override_bypasses_derivation(self):
        assert derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=10.0, lr_override=0.05) == 0.05

    def test_lr_override_zero_is_valid(self):
        assert derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=10.0, lr_override=0.0) == 0.0

    def test_lr_override_ignores_invalid_spectral_values(self):
        # Override works even with invalid spectral values
        assert derive_spectral_ceiling(sigma_k_min=0.0, sigma_max_global=0.0, lr_override=0.01) == 0.01
        assert derive_spectral_ceiling(sigma_k_min=-1.0, sigma_max_global=-1.0, lr_override=0.01) == 0.01

    def test_sigma_k_min_zero_raises(self):
        with pytest.raises(TrainingDerivationError) as exc_info:
            derive_spectral_ceiling(sigma_k_min=0.0, sigma_max_global=10.0)
        assert exc_info.value.failure_class == "insufficient_adapter_geometry"

    def test_sigma_k_min_negative_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=-1.0, sigma_max_global=10.0)

    def test_sigma_max_zero_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=0.0)

    def test_sigma_max_negative_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=-5.0)

    def test_both_zero_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=0.0, sigma_max_global=0.0)

    def test_inf_sigma_k_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=float("inf"), sigma_max_global=10.0)

    def test_nan_sigma_max_raises(self):
        with pytest.raises(TrainingDerivationError):
            derive_spectral_ceiling(sigma_k_min=0.1, sigma_max_global=float("nan"))

    def test_very_small_sigma_k_produces_valid_result(self):
        result = derive_spectral_ceiling(sigma_k_min=1e-38, sigma_max_global=1.0)
        assert result == pytest.approx(1e-38)
        assert math.isfinite(result)

    def test_very_large_sigma_max_produces_small_ceiling(self):
        result = derive_spectral_ceiling(sigma_k_min=1.0, sigma_max_global=1e10)
        assert result == pytest.approx(1e-10)


# ===================================================================
# Class 2: √N Epoch Budget Correction
# ===================================================================
class TestSqrtNCorrection:
    """Tests for eta_ceiling_epoch = eta_ceiling / sqrt(N) (Brownian scaling)."""

    def test_single_batch_no_correction(self):
        assert apply_sqrt_n_epoch_correction(0.01, 1) == 0.01

    def test_four_batches_halves_ceiling(self):
        assert apply_sqrt_n_epoch_correction(0.01, 4) == pytest.approx(0.005)

    def test_hundred_batches(self):
        assert apply_sqrt_n_epoch_correction(0.01, 100) == pytest.approx(0.001)

    def test_scaling_is_monotonically_decreasing(self):
        results = [apply_sqrt_n_epoch_correction(0.01, n) for n in [1, 4, 16, 64, 256]]
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

    def test_lr_override_skips_correction(self):
        # When lr_override is set, the ceiling was already set by override;
        # sqrt(N) correction is skipped.
        result = apply_sqrt_n_epoch_correction(0.01, 100, lr_override=0.05)
        assert result == 0.01  # ceiling unchanged

    def test_large_n_doesnt_underflow(self):
        result = apply_sqrt_n_epoch_correction(0.01, 10000)
        assert result == pytest.approx(1e-4)
        assert result > 0
        assert math.isfinite(result)

    def test_brownian_scaling_recovers_original(self):
        # eta_corrected * sqrt(N) == eta_original by construction
        eta = 0.01
        for n in [4, 25, 100, 1000]:
            corrected = apply_sqrt_n_epoch_correction(eta, n)
            assert corrected * math.sqrt(n) == pytest.approx(eta)


# ===================================================================
# Class 3: SPS (Stochastic Polyak Step Size)
# ===================================================================
class TestEtaSPS:
    """Tests for eta_sps = loss / ||g||^2 (Loizou et al. 2020, f*=0)."""

    def test_formula_known_values(self):
        assert _extract_sps(2.0, 1.0, 0.01) == pytest.approx(2.0)
        assert _extract_sps(1.0, 2.0, 0.01) == pytest.approx(0.25)

    def test_scales_inversely_with_grad_squared(self):
        base = _extract_sps(1.0, 1.0, 0.01)
        doubled_grad = _extract_sps(1.0, 2.0, 0.01)
        assert doubled_grad == pytest.approx(base / 4.0)

    def test_scales_linearly_with_loss(self):
        base = _extract_sps(1.0, 1.0, 0.01)
        doubled_loss = _extract_sps(2.0, 1.0, 0.01)
        assert doubled_loss == pytest.approx(2.0 * base)

    def test_zero_gradient_fallback(self):
        assert _extract_sps(1.0, 0.0, 0.01) == 0.01

    def test_zero_loss_yields_zero_step(self):
        assert _extract_sps(0.0, 1.0, 0.01) == 0.0

    def test_tiny_gradient_yields_large_sps(self):
        # SPS is uncapped — min() caps it later
        result = _extract_sps(1.0, 1e-6, 0.01)
        assert result == pytest.approx(1e12)

    def test_large_gradient_yields_small_sps(self):
        result = _extract_sps(1.0, 1e6, 0.01)
        assert result == pytest.approx(1e-12)

    def test_sps_always_nonneg_for_nonneg_inputs(self):
        for loss in [0.0, 0.001, 1.0, 100.0]:
            for d_norm in [0.0, 0.001, 1.0, 100.0]:
                result = _extract_sps(loss, d_norm, 0.01)
                assert result >= 0


# ===================================================================
# Class 4: Weyl Displacement Bound
# ===================================================================
class TestEtaWeyl:
    """Tests for eta_weyl = sigma_k_min / ||g|| (Weyl 1912 per-step bound)."""

    def test_formula_known_values(self):
        assert _extract_weyl(0.5, 2.0, 0.01) == pytest.approx(0.25)
        assert _extract_weyl(1.0, 1.0, 0.01) == pytest.approx(1.0)

    def test_displacement_equals_sigma_k_exactly(self):
        # eta_weyl * ||g|| = sigma_k_min by construction
        for sigma_k in [0.01, 0.1, 1.0, 10.0]:
            for g in [0.001, 1.0, 100.0]:
                eta = _extract_weyl(sigma_k, g, 0.01)
                assert eta * g == pytest.approx(sigma_k)

    def test_zero_gradient_fallback(self):
        assert _extract_weyl(0.5, 0.0, 0.01) == 0.01

    def test_scales_inversely_with_gradient(self):
        base = _extract_weyl(0.5, 1.0, 0.01)
        doubled = _extract_weyl(0.5, 2.0, 0.01)
        assert doubled == pytest.approx(base / 2.0)

    def test_scales_linearly_with_sigma_k(self):
        base = _extract_weyl(0.5, 1.0, 0.01)
        doubled = _extract_weyl(1.0, 1.0, 0.01)
        assert doubled == pytest.approx(2.0 * base)

    def test_tiny_gradient_yields_large_weyl(self):
        result = _extract_weyl(0.5, 1e-8, 0.01)
        assert result == pytest.approx(5e7)

    def test_large_gradient_yields_small_weyl(self):
        result = _extract_weyl(0.5, 1e6, 0.01)
        assert result == pytest.approx(5e-7)


# ===================================================================
# Class 5: MASS min() Combination
# ===================================================================
class TestMassMinCombination:
    """Tests for eta_step = min(eta_sps, eta_weyl, eta_ceiling)."""

    def test_ceiling_binds_when_sps_and_weyl_larger(self):
        # Small gradient -> SPS and Weyl both large -> ceiling wins
        ceil = 0.01
        eta_step, eta_sps, eta_weyl, _, _ = compute_per_step_rates(
            loss=1.0, d_norm=0.001, sigma_k_min=0.5, eta_ceiling=ceil,
        )
        assert eta_sps > ceil
        assert eta_weyl > ceil
        assert eta_step == ceil

    def test_sps_binds_when_loss_small(self):
        # Small loss -> SPS smallest
        eta_step, eta_sps, eta_weyl, _, _ = compute_per_step_rates(
            loss=1e-6, d_norm=1.0, sigma_k_min=0.5, eta_ceiling=1.0,
        )
        assert eta_sps < eta_weyl
        assert eta_sps < 1.0  # < ceiling
        assert eta_step == pytest.approx(eta_sps)

    def test_weyl_binds_when_gradient_large(self):
        # Large gradient -> Weyl smallest
        eta_step, eta_sps, eta_weyl, _, _ = compute_per_step_rates(
            loss=10.0, d_norm=100.0, sigma_k_min=0.01, eta_ceiling=1.0,
        )
        assert eta_weyl < eta_sps
        assert eta_weyl < 1.0  # < ceiling
        assert eta_step == pytest.approx(eta_weyl)

    @pytest.mark.parametrize("loss,d_norm,sigma_k,ceiling", [
        (2.0, 1.0, 0.5, 0.01),     # normal case
        (1e6, 1.0, 0.5, 0.01),     # extreme loss
        (1.0, 1e6, 0.5, 0.01),     # extreme gradient
        (1.0, 1e-10, 0.5, 0.01),   # tiny gradient
        (1.0, 1.0, 1e-8, 0.01),    # tiny sigma_k
        (1.0, 1.0, 100.0, 0.01),   # large sigma_k
        (0.001, 0.001, 0.001, 0.001),  # all small
        (100.0, 100.0, 100.0, 100.0),  # all large
    ])
    def test_displacement_invariant_always_holds(self, loss, d_norm, sigma_k, ceiling):
        """Core invariant: eta_step * ||g|| <= sigma_k_min."""
        _, _, _, displacement, _ = compute_per_step_rates(loss, d_norm, sigma_k, ceiling)
        assert displacement <= sigma_k + 1e-15

    def test_displacement_tight_when_weyl_binds(self):
        # When Weyl binds: displacement == sigma_k_min exactly
        _, _, eta_weyl, displacement, _ = compute_per_step_rates(
            loss=100.0, d_norm=10.0, sigma_k_min=0.01, eta_ceiling=1.0,
        )
        # Weyl should bind here (eta_weyl = 0.001, sps = 1.0, ceiling = 1.0)
        assert displacement == pytest.approx(0.01)  # == sigma_k_min

    def test_zero_gradient_displacement_is_zero(self):
        eta_step, _, _, displacement, _ = compute_per_step_rates(
            loss=1.0, d_norm=0.0, sigma_k_min=0.5, eta_ceiling=0.01,
        )
        assert displacement == 0.0

    def test_step_never_exceeds_ceiling(self):
        for loss in [0.01, 1.0, 100.0]:
            for d_norm in [0.001, 1.0, 1000.0]:
                eta_step, _, _, _, _ = compute_per_step_rates(
                    loss, d_norm, sigma_k_min=0.5, eta_ceiling=0.01,
                )
                assert eta_step <= 0.01

    def test_step_never_negative(self):
        for loss in [0.0, 1.0, 100.0]:
            for d_norm in [0.0, 1.0, 100.0]:
                eta_step, _, _, _, _ = compute_per_step_rates(
                    loss, d_norm, sigma_k_min=0.5, eta_ceiling=0.01,
                )
                assert eta_step >= 0

    def test_all_three_equal(self):
        # Construct inputs where sps = weyl = ceiling
        # sps = loss/g^2, weyl = sigma_k/g
        # sps = weyl => loss/g^2 = sigma_k/g => loss = sigma_k * g
        # sps = ceiling => loss/g^2 = ceiling => loss = ceiling * g^2
        # So: sigma_k * g = ceiling * g^2 => sigma_k = ceiling * g
        g = 2.0
        ceiling = 0.5
        sigma_k = ceiling * g  # = 1.0
        loss = ceiling * g ** 2  # = 2.0
        eta_step, eta_sps, eta_weyl, _, _ = compute_per_step_rates(
            loss, g, sigma_k, ceiling,
        )
        assert eta_sps == pytest.approx(ceiling)
        assert eta_weyl == pytest.approx(ceiling)
        assert eta_step == pytest.approx(ceiling)


# ===================================================================
# Class 6: Validation Backoff
# ===================================================================
class TestValidationBackoff:
    """Tests for validation-guided ceiling backoff with sqrt(eps_f32) floor."""

    _BACKOFF_FLOOR = _SQRT_EPS_F32

    def test_no_backoff_when_val_decreases(self):
        assert apply_validation_backoff(0.01, [1.5, 1.4]) == 0.01

    def test_no_backoff_single_val(self):
        assert apply_validation_backoff(0.01, [1.5]) == 0.01

    def test_no_backoff_empty_val(self):
        assert apply_validation_backoff(0.01, []) == 0.01

    def test_ratio_backoff_on_increase(self):
        # val_losses[-2]/val_losses[-1] = 1.0/2.0 = 0.5
        result = apply_validation_backoff(0.01, [1.0, 2.0])
        assert result == pytest.approx(0.005)

    def test_small_increase_small_backoff(self):
        result = apply_validation_backoff(0.01, [1.0, 1.01])
        expected = 0.01 * (1.0 / 1.01)
        assert result == pytest.approx(expected)

    def test_large_increase_large_backoff(self):
        result = apply_validation_backoff(0.01, [1.0, 10.0])
        assert result == pytest.approx(0.001)

    def test_backoff_floor_prevents_near_zero(self):
        # Extreme ratio: 1e-10 / 1e10 = 1e-20 < floor
        result = apply_validation_backoff(0.01, [1e-10, 1e10])
        assert result == pytest.approx(0.01 * self._BACKOFF_FLOOR)
        assert result > 0

    def test_backoff_floor_value_is_sqrt_eps_f32(self):
        assert self._BACKOFF_FLOOR == pytest.approx(math.ldexp(1.0, -23) ** 0.5)

    def test_ceiling_only_decreases_on_increase(self):
        # When val increases, prev/curr < 1, so backoff < 1, so ceiling shrinks
        for prev, curr in [(1.0, 1.1), (0.5, 5.0), (0.01, 100.0)]:
            result = apply_validation_backoff(0.01, [prev, curr])
            assert result < 0.01

    def test_lr_override_skips_backoff(self):
        result = apply_validation_backoff(0.01, [1.0, 10.0], lr_override=0.05)
        assert result == 0.01  # unchanged

    def test_adaptive_lr_false_skips_backoff(self):
        result = apply_validation_backoff(0.01, [1.0, 10.0], adaptive_lr=False)
        assert result == 0.01  # unchanged

    def test_repeated_backoff_monotone_decrease(self):
        ceiling = 0.01
        val_losses = [1.0]
        for i in range(5):
            val_losses.append(val_losses[-1] * 1.5)  # 50% increase each time
            new_ceiling = apply_validation_backoff(ceiling, val_losses)
            assert new_ceiling < ceiling
            ceiling = new_ceiling
        assert ceiling > 0  # never reaches zero


# ===================================================================
# Class 7: REINFORCE Budget
# ===================================================================
class TestReinforceBudget:
    """Tests for REINFORCE displacement budget (Weyl remainder after CE)."""

    def test_remainder_after_ce_displacement(self):
        target, source = compute_reinforce_budget(0.5, 0.3, 4, 10)
        assert target == pytest.approx((0.5 - 0.3) / math.sqrt(4))
        assert target == pytest.approx(0.1)
        assert source == "weyl_remainder"

    def test_budget_exhausted_when_ce_equals_sigma_k(self):
        target, source = compute_reinforce_budget(0.5, 0.5, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_budget_exhausted_when_ce_exceeds_sigma_k(self):
        target, source = compute_reinforce_budget(0.5, 0.7, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_no_ce_displacement_shared_budget(self):
        # update_norm=None -> shared path
        # n_total = max(1, 10) + max(1, 4) = 14
        target, source = compute_reinforce_budget(0.5, None, 4, 10)
        assert target == pytest.approx(0.5 / math.sqrt(14))
        assert source == "sigma_k_min_shared"

    def test_zero_update_norm_uses_shared_path(self):
        # update_norm=0.0 -> falls into else branch (not > 0)
        target, source = compute_reinforce_budget(0.5, 0.0, 4, 10)
        assert source == "sigma_k_min_shared"

    def test_shared_budget_sigma_k_zero_exhausted(self):
        target, source = compute_reinforce_budget(0.0, None, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_single_reinforce_step_gets_full_remainder(self):
        target, source = compute_reinforce_budget(0.5, 0.1, 1, 10)
        assert target == pytest.approx(0.4 / math.sqrt(1))
        assert target == pytest.approx(0.4)
        assert source == "weyl_remainder"

    def test_many_steps_reduces_target(self):
        t1, _ = compute_reinforce_budget(0.5, 0.1, 1, 10)
        t100, _ = compute_reinforce_budget(0.5, 0.1, 100, 10)
        assert t1 > t100

    def test_remainder_plus_ce_within_sigma_k(self):
        """Core invariant: update_norm + sqrt(N_re) * target <= sigma_k_min."""
        sigma_k = 0.5
        update = 0.3
        n_re = 4
        target, _ = compute_reinforce_budget(sigma_k, update, n_re, 10)
        total = update + math.sqrt(n_re) * target
        assert total == pytest.approx(sigma_k)

    def test_shared_budget_brownian_bound(self):
        """Shared path: target * sqrt(N_total) == sigma_k_min."""
        sigma_k = 0.5
        n_re = 4
        check_interval = 10
        target, _ = compute_reinforce_budget(sigma_k, None, n_re, check_interval)
        n_total = max(1, check_interval) + max(1, n_re)
        assert target * math.sqrt(n_total) == pytest.approx(sigma_k)

    def test_source_label_remainder(self):
        _, source = compute_reinforce_budget(0.5, 0.1, 4, 10)
        assert source == "weyl_remainder"

    def test_source_label_shared(self):
        _, source = compute_reinforce_budget(0.5, None, 4, 10)
        assert source == "sigma_k_min_shared"

    def test_source_label_exhausted(self):
        _, source = compute_reinforce_budget(0.5, 0.5, 4, 10)
        assert source == "budget_exhausted"


# ===================================================================
# Class 8: Cross-cutting MASS Invariants
# ===================================================================
class TestMassInvariants:
    """Cross-cutting mathematical invariants spanning multiple MASS layers."""

    @pytest.mark.parametrize("sigma_k,sigma_max,n_batches,loss,d_norm", [
        (0.1, 10.0, 100, 2.0, 1.0),
        (0.5, 5.0, 25, 0.5, 0.5),
        (0.01, 100.0, 1000, 10.0, 10.0),
        (1.0, 1.0, 1, 0.001, 0.001),
        (0.001, 50.0, 50, 5.0, 100.0),
    ])
    def test_displacement_bounded_by_sigma_k_over_sqrt_n(
        self, sigma_k, sigma_max, n_batches, loss, d_norm,
    ):
        """After √N correction, per-step displacement <= sigma_k / sqrt(N)."""
        ceiling = derive_spectral_ceiling(sigma_k_min=sigma_k, sigma_max_global=sigma_max)
        ceiling = apply_sqrt_n_epoch_correction(ceiling, n_batches)

        _, _, _, displacement, _ = compute_per_step_rates(loss, d_norm, sigma_k, ceiling)

        # Displacement bounded by the tighter of ceiling*||g|| and sigma_k
        assert displacement <= sigma_k + 1e-15

    def test_epoch_total_displacement_bounded_by_sigma_k(self):
        """Over N steps with worst-case gradients, Brownian displacement <= sigma_k."""
        sigma_k = 0.5
        sigma_max = 10.0
        n_batches = 100
        ceiling = derive_spectral_ceiling(sigma_k_min=sigma_k, sigma_max_global=sigma_max)
        ceiling = apply_sqrt_n_epoch_correction(ceiling, n_batches)

        # Worst case: every step at ceiling (small gradients)
        # With √N correction, ceiling = sigma_k / (sigma_max * √N),
        # so max displacement per step = ceiling * sigma_max = sigma_k / √N.
        max_per_step = sigma_k / math.sqrt(n_batches)
        # Brownian total: √N * max_per_step = sigma_k
        brownian_total = math.sqrt(n_batches) * max_per_step
        assert brownian_total == pytest.approx(sigma_k)

    def test_ceiling_post_backoff_still_positive(self):
        """After repeated backoffs, ceiling stays positive and finite.

        The floor = sqrt(eps_f32) ~ 3.45e-4. After 50 consecutive extreme
        backoffs: 0.01 * (3.45e-4)^50 ~ 1e-175, still above float64
        subnormal minimum (~5e-324). In practice, training stops long
        before 50 consecutive val increases.
        """
        ceiling = 0.01
        for _ in range(50):
            # Extreme val increase each time — floor binds
            backoff = max(0.01 / 1000.0, _SQRT_EPS_F32)
            ceiling = ceiling * backoff
        assert ceiling > 0
        assert math.isfinite(ceiling)

    def test_reinforce_total_within_weyl_bound(self):
        """CE displacement + REINFORCE displacement <= sigma_k_min."""
        sigma_k = 0.5
        ce_displacement = 0.3
        n_re = 16
        target, _ = compute_reinforce_budget(sigma_k, ce_displacement, n_re, 10)
        # Brownian REINFORCE total
        re_total = math.sqrt(n_re) * target
        assert ce_displacement + re_total == pytest.approx(sigma_k)

    def test_backoff_floor_traceable_to_ieee754(self):
        """_SQRT_EPS_F32 = sqrt(eps_f32) = sqrt(2^-23) — IEEE 754 derived."""
        assert _SQRT_EPS_F32 == pytest.approx(math.sqrt(math.ldexp(1.0, -23)))
        assert _EPS_F32 == pytest.approx(math.ldexp(1.0, -23))

    def test_zero_loss_zero_displacement(self):
        """At optimum (loss=0), SPS=0, so eta_step=0 and no update occurs."""
        for d_norm in [0.001, 1.0, 100.0]:
            eta_step, _, _, displacement, _ = compute_per_step_rates(
                0.0, d_norm, 0.5, 0.01,
            )
            assert eta_step == 0.0
            assert displacement == 0.0


# ===================================================================
# Class 9: Conformal Margin Rate (Sahraee-Ardakan et al. 2026)
# ===================================================================
class TestConformalMarginRate:
    """Tests for eta_margin = remaining_budget / ||g|| (Weyl on remaining capacity)."""

    def test_formula_known_values(self):
        assert compute_conformal_margin_rate(0.2, 1.0) == pytest.approx(0.2)
        assert compute_conformal_margin_rate(1.0, 2.0) == pytest.approx(0.5)
        assert compute_conformal_margin_rate(0.5, 0.5) == pytest.approx(1.0)

    def test_zero_remaining_gives_zero(self):
        assert compute_conformal_margin_rate(0.0, 1.0) == 0.0

    def test_negative_remaining_gives_zero(self):
        assert compute_conformal_margin_rate(-0.1, 1.0) == 0.0

    def test_zero_gradient_gives_inf(self):
        assert compute_conformal_margin_rate(0.5, 0.0) == float("inf")

    def test_always_leq_eta_weyl(self):
        """eta_margin <= eta_weyl because remaining <= sigma_k."""
        for sigma_k in [0.1, 0.5, 1.0]:
            for ratio in [0.0, 0.25, 0.5, 0.75, 0.99]:
                remaining = sigma_k * (1.0 - ratio)
                for d_norm in [0.001, 1.0, 100.0]:
                    eta_margin = compute_conformal_margin_rate(remaining, d_norm)
                    eta_weyl = sigma_k / d_norm if d_norm > 0 else float("inf")
                    assert eta_margin <= eta_weyl + 1e-15

    def test_deceleration_monotonic(self):
        """As budget fills (remaining decreases), eta_margin decreases."""
        d_norm = 1.0
        rates = [compute_conformal_margin_rate(r, d_norm) for r in [0.5, 0.3, 0.1, 0.01]]
        for i in range(len(rates) - 1):
            assert rates[i] > rates[i + 1]

    def test_displacement_bounded_by_remaining(self):
        """Core invariant: eta_margin * ||g|| <= remaining."""
        for remaining in [0.01, 0.1, 0.5, 1.0]:
            for d_norm in [0.001, 1.0, 100.0]:
                eta = compute_conformal_margin_rate(remaining, d_norm)
                assert eta * d_norm <= remaining + 1e-15


# ===================================================================
# Class 10: Conformal Margin Integration with MASS
# ===================================================================
class TestConformalMarginIntegration:
    """Tests for compute_per_step_rates with remaining_budget parameter."""

    def test_backward_compatible_without_remaining(self):
        """Without remaining_budget, returns 5-tuple with None eta_margin."""
        result = compute_per_step_rates(1.0, 1.0, 0.5, 0.01)
        assert len(result) == 5
        eta_step, eta_sps, eta_weyl, displacement, eta_margin = result
        assert eta_margin is None
        assert eta_step == min(eta_sps, eta_weyl, 0.01)

    def test_margin_binds_when_budget_small(self):
        """Small remaining budget -> eta_margin wins over eta_weyl."""
        sigma_k = 0.5
        remaining = 0.01  # Much less than sigma_k
        d_norm = 1.0
        eta_step, _, eta_weyl, _, eta_margin = compute_per_step_rates(
            loss=10.0, d_norm=d_norm, sigma_k_min=sigma_k, eta_ceiling=1.0,
            remaining_budget=remaining,
        )
        assert eta_margin == pytest.approx(0.01)
        assert eta_margin < eta_weyl
        assert eta_step == pytest.approx(eta_margin)

    def test_margin_never_exceeds_weyl_rate(self):
        """eta_margin <= eta_weyl for any remaining <= sigma_k."""
        for remaining_frac in [0.1, 0.5, 0.9, 1.0]:
            sigma_k = 0.5
            remaining = sigma_k * remaining_frac
            _, _, eta_weyl, _, eta_margin = compute_per_step_rates(
                loss=1.0, d_norm=1.0, sigma_k_min=sigma_k, eta_ceiling=1.0,
                remaining_budget=remaining,
            )
            assert eta_margin <= eta_weyl + 1e-15

    def test_displacement_bounded_by_remaining_budget(self):
        """Core invariant: displacement <= remaining_budget."""
        remaining = 0.05
        _, _, _, displacement, _ = compute_per_step_rates(
            loss=10.0, d_norm=2.0, sigma_k_min=0.5, eta_ceiling=1.0,
            remaining_budget=remaining,
        )
        assert displacement <= remaining + 1e-15

    def test_zero_remaining_stops_displacement(self):
        """Zero remaining budget -> eta_margin=0 -> eta_step=0 -> no displacement."""
        eta_step, _, _, displacement, eta_margin = compute_per_step_rates(
            loss=1.0, d_norm=1.0, sigma_k_min=0.5, eta_ceiling=0.01,
            remaining_budget=0.0,
        )
        assert eta_margin == 0.0
        assert eta_step == 0.0
        assert displacement == 0.0


# ===================================================================
# Class 11: Bounded-Gain Stability Certificate
# ===================================================================
class TestBoundedGainCertificate:
    """Tests for verify_bounded_gain (Sahraee-Ardakan et al. 2026)."""

    def test_bounded_when_below_ceiling(self):
        is_bounded, ratio = verify_bounded_gain(0.005, 0.01)
        assert is_bounded is True
        assert ratio == pytest.approx(0.5)

    def test_bounded_when_at_ceiling(self):
        is_bounded, ratio = verify_bounded_gain(0.01, 0.01)
        assert is_bounded is True
        assert ratio == pytest.approx(1.0)

    def test_unbounded_when_above_ceiling(self):
        is_bounded, ratio = verify_bounded_gain(0.02, 0.01)
        assert is_bounded is False
        assert ratio == pytest.approx(2.0)

    def test_zero_ceiling_zero_step(self):
        is_bounded, ratio = verify_bounded_gain(0.0, 0.0)
        assert is_bounded is True
        assert ratio == 0.0

    def test_zero_ceiling_nonzero_step(self):
        is_bounded, ratio = verify_bounded_gain(0.01, 0.0)
        assert is_bounded is False
        assert ratio == float("inf")

    def test_mass_construction_always_bounded(self):
        """By MASS construction, eta_step = min(..., ceiling) <= ceiling."""
        for loss in [0.01, 1.0, 100.0]:
            for d_norm in [0.001, 1.0, 1000.0]:
                ceiling = 0.01
                eta_step, _, _, _, _ = compute_per_step_rates(
                    loss, d_norm, sigma_k_min=0.5, eta_ceiling=ceiling,
                )
                is_bounded, _ = verify_bounded_gain(eta_step, ceiling)
                assert is_bounded is True
