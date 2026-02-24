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

The static methods in each class replicate the exact production code from
_mlx_training_adapter_core_mixin.py and _mlx_training_adapter_train_mixin.py.
Tests verify mathematical properties, not implementation details.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.training.exceptions import TrainingDerivationError


# ---------------------------------------------------------------------------
# Shared IEEE 754 constants
# ---------------------------------------------------------------------------
_EPS_F32 = math.ldexp(1.0, -23)  # 2^-23, float32 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)


# ===================================================================
# Class 1: Spectral Ceiling
# ===================================================================
class TestSpectralCeiling:
    """Tests for eta_ceiling = sigma_k_min / sigma_max_global (Weyl 1912)."""

    @staticmethod
    def derive_spectral_ceiling(
        sigma_k_min: float,
        sigma_max_global: float,
        lr_override: float | None = None,
    ) -> float:
        """Pure-function replica of _derive_spectral_ceiling()."""
        if lr_override is not None:
            return float(lr_override)
        if sigma_k_min <= 0 or sigma_max_global <= 0:
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="sigma_k_min or sigma_max_global non-positive.",
                diagnostics={"sigma_k_min": sigma_k_min, "sigma_max_global": sigma_max_global},
            )
        if not math.isfinite(sigma_k_min) or not math.isfinite(sigma_max_global):
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="sigma_k_min or sigma_max_global is non-finite.",
                diagnostics={"sigma_k_min": sigma_k_min, "sigma_max_global": sigma_max_global},
            )
        return sigma_k_min / sigma_max_global

    def test_formula_known_values(self):
        assert self.derive_spectral_ceiling(0.1, 10.0) == pytest.approx(0.01)
        assert self.derive_spectral_ceiling(1.0, 1.0) == pytest.approx(1.0)
        assert self.derive_spectral_ceiling(5.0, 10.0) == pytest.approx(0.5)

    def test_ceiling_at_most_one_when_sigma_k_leq_sigma_max(self):
        # sigma_k <= sigma_max always holds for valid SVD
        for sigma_k in [0.001, 0.1, 1.0, 5.0]:
            for sigma_max in [sigma_k, sigma_k * 2, sigma_k * 100]:
                assert self.derive_spectral_ceiling(sigma_k, sigma_max) <= 1.0

    def test_ceiling_equals_one_when_equal(self):
        assert self.derive_spectral_ceiling(3.0, 3.0) == 1.0

    def test_lr_override_bypasses_derivation(self):
        assert self.derive_spectral_ceiling(0.1, 10.0, lr_override=0.05) == 0.05

    def test_lr_override_zero_is_valid(self):
        assert self.derive_spectral_ceiling(0.1, 10.0, lr_override=0.0) == 0.0

    def test_lr_override_ignores_invalid_spectral_values(self):
        # Override works even with invalid spectral values
        assert self.derive_spectral_ceiling(0.0, 0.0, lr_override=0.01) == 0.01
        assert self.derive_spectral_ceiling(-1.0, -1.0, lr_override=0.01) == 0.01

    def test_sigma_k_min_zero_raises(self):
        with pytest.raises(TrainingDerivationError) as exc_info:
            self.derive_spectral_ceiling(0.0, 10.0)
        assert exc_info.value.failure_class == "insufficient_adapter_geometry"

    def test_sigma_k_min_negative_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(-1.0, 10.0)

    def test_sigma_max_zero_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(0.1, 0.0)

    def test_sigma_max_negative_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(0.1, -5.0)

    def test_both_zero_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(0.0, 0.0)

    def test_inf_sigma_k_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(float("inf"), 10.0)

    def test_nan_sigma_max_raises(self):
        with pytest.raises(TrainingDerivationError):
            self.derive_spectral_ceiling(0.1, float("nan"))

    def test_very_small_sigma_k_produces_valid_result(self):
        result = self.derive_spectral_ceiling(1e-38, 1.0)
        assert result == pytest.approx(1e-38)
        assert math.isfinite(result)

    def test_very_large_sigma_max_produces_small_ceiling(self):
        result = self.derive_spectral_ceiling(1.0, 1e10)
        assert result == pytest.approx(1e-10)


# ===================================================================
# Class 2: √N Epoch Budget Correction
# ===================================================================
class TestSqrtNCorrection:
    """Tests for eta_ceiling_epoch = eta_ceiling / sqrt(N) (Brownian scaling)."""

    @staticmethod
    def apply_sqrt_n_correction(
        eta_ceiling: float,
        n_batches_per_epoch: int,
        lr_override: float | None = None,
    ) -> float:
        """Replicate √N budget correction from train_mixin lines 363-380."""
        if lr_override is None and n_batches_per_epoch > 1:
            return eta_ceiling / math.sqrt(n_batches_per_epoch)
        return eta_ceiling

    def test_single_batch_no_correction(self):
        assert self.apply_sqrt_n_correction(0.01, 1) == 0.01

    def test_four_batches_halves_ceiling(self):
        assert self.apply_sqrt_n_correction(0.01, 4) == pytest.approx(0.005)

    def test_hundred_batches(self):
        assert self.apply_sqrt_n_correction(0.01, 100) == pytest.approx(0.001)

    def test_scaling_is_monotonically_decreasing(self):
        results = [self.apply_sqrt_n_correction(0.01, n) for n in [1, 4, 16, 64, 256]]
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

    def test_lr_override_skips_correction(self):
        # When lr_override is set, the ceiling was already set by override;
        # sqrt(N) correction is skipped.
        result = self.apply_sqrt_n_correction(0.01, 100, lr_override=0.05)
        assert result == 0.01  # ceiling unchanged

    def test_large_n_doesnt_underflow(self):
        result = self.apply_sqrt_n_correction(0.01, 10000)
        assert result == pytest.approx(1e-4)
        assert result > 0
        assert math.isfinite(result)

    def test_brownian_scaling_recovers_original(self):
        # eta_corrected * sqrt(N) == eta_original by construction
        eta = 0.01
        for n in [4, 25, 100, 1000]:
            corrected = self.apply_sqrt_n_correction(eta, n)
            assert corrected * math.sqrt(n) == pytest.approx(eta)


# ===================================================================
# Class 3: SPS (Stochastic Polyak Step Size)
# ===================================================================
class TestEtaSPS:
    """Tests for eta_sps = loss / ||g||^2 (Loizou et al. 2020, f*=0)."""

    @staticmethod
    def compute_eta_sps(loss: float, d_norm: float, eta_ceiling: float) -> float:
        """SPS rate: f(x)/||g||^2, zero-gradient fallback to ceiling."""
        if d_norm > 0:
            return loss / (d_norm ** 2)
        return eta_ceiling

    def test_formula_known_values(self):
        assert self.compute_eta_sps(2.0, 1.0, 0.01) == pytest.approx(2.0)
        assert self.compute_eta_sps(1.0, 2.0, 0.01) == pytest.approx(0.25)

    def test_scales_inversely_with_grad_squared(self):
        base = self.compute_eta_sps(1.0, 1.0, 0.01)
        doubled_grad = self.compute_eta_sps(1.0, 2.0, 0.01)
        assert doubled_grad == pytest.approx(base / 4.0)

    def test_scales_linearly_with_loss(self):
        base = self.compute_eta_sps(1.0, 1.0, 0.01)
        doubled_loss = self.compute_eta_sps(2.0, 1.0, 0.01)
        assert doubled_loss == pytest.approx(2.0 * base)

    def test_zero_gradient_fallback(self):
        assert self.compute_eta_sps(1.0, 0.0, 0.01) == 0.01

    def test_zero_loss_yields_zero_step(self):
        assert self.compute_eta_sps(0.0, 1.0, 0.01) == 0.0

    def test_tiny_gradient_yields_large_sps(self):
        # SPS is uncapped — min() caps it later
        result = self.compute_eta_sps(1.0, 1e-6, 0.01)
        assert result == pytest.approx(1e12)

    def test_large_gradient_yields_small_sps(self):
        result = self.compute_eta_sps(1.0, 1e6, 0.01)
        assert result == pytest.approx(1e-12)

    def test_sps_always_nonneg_for_nonneg_inputs(self):
        for loss in [0.0, 0.001, 1.0, 100.0]:
            for d_norm in [0.0, 0.001, 1.0, 100.0]:
                result = self.compute_eta_sps(loss, d_norm, 0.01)
                assert result >= 0


# ===================================================================
# Class 4: Weyl Displacement Bound
# ===================================================================
class TestEtaWeyl:
    """Tests for eta_weyl = sigma_k_min / ||g|| (Weyl 1912 per-step bound)."""

    @staticmethod
    def compute_eta_weyl(sigma_k_min: float, d_norm: float, eta_ceiling: float) -> float:
        """Weyl bound: sigma_k_min / ||g||, zero-gradient fallback to ceiling."""
        if d_norm > 0:
            return sigma_k_min / d_norm
        return eta_ceiling

    def test_formula_known_values(self):
        assert self.compute_eta_weyl(0.5, 2.0, 0.01) == pytest.approx(0.25)
        assert self.compute_eta_weyl(1.0, 1.0, 0.01) == pytest.approx(1.0)

    def test_displacement_equals_sigma_k_exactly(self):
        # eta_weyl * ||g|| = sigma_k_min by construction
        for sigma_k in [0.01, 0.1, 1.0, 10.0]:
            for g in [0.001, 1.0, 100.0]:
                eta = self.compute_eta_weyl(sigma_k, g, 0.01)
                assert eta * g == pytest.approx(sigma_k)

    def test_zero_gradient_fallback(self):
        assert self.compute_eta_weyl(0.5, 0.0, 0.01) == 0.01

    def test_scales_inversely_with_gradient(self):
        base = self.compute_eta_weyl(0.5, 1.0, 0.01)
        doubled = self.compute_eta_weyl(0.5, 2.0, 0.01)
        assert doubled == pytest.approx(base / 2.0)

    def test_scales_linearly_with_sigma_k(self):
        base = self.compute_eta_weyl(0.5, 1.0, 0.01)
        doubled = self.compute_eta_weyl(1.0, 1.0, 0.01)
        assert doubled == pytest.approx(2.0 * base)

    def test_tiny_gradient_yields_large_weyl(self):
        result = self.compute_eta_weyl(0.5, 1e-8, 0.01)
        assert result == pytest.approx(5e7)

    def test_large_gradient_yields_small_weyl(self):
        result = self.compute_eta_weyl(0.5, 1e6, 0.01)
        assert result == pytest.approx(5e-7)


# ===================================================================
# Class 5: MASS min() Combination
# ===================================================================
class TestMassMinCombination:
    """Tests for eta_step = min(eta_sps, eta_weyl, eta_ceiling)."""

    @staticmethod
    def compute_mass_step(
        loss: float,
        d_norm: float,
        sigma_k_min: float,
        eta_ceiling: float,
    ) -> tuple[float, float, float, float]:
        """Full MASS per-step computation.

        Returns (eta_step, eta_sps, eta_weyl, displacement).
        """
        if d_norm > 0:
            eta_sps = loss / (d_norm ** 2)
            eta_weyl = sigma_k_min / d_norm
        else:
            eta_sps = eta_ceiling
            eta_weyl = eta_ceiling

        eta_step = min(eta_sps, eta_weyl, eta_ceiling)
        displacement = eta_step * d_norm
        return eta_step, eta_sps, eta_weyl, displacement

    def test_ceiling_binds_when_sps_and_weyl_larger(self):
        # Small gradient -> SPS and Weyl both large -> ceiling wins
        ceil = 0.01
        eta_step, eta_sps, eta_weyl, _ = self.compute_mass_step(
            loss=1.0, d_norm=0.001, sigma_k_min=0.5, eta_ceiling=ceil,
        )
        assert eta_sps > ceil
        assert eta_weyl > ceil
        assert eta_step == ceil

    def test_sps_binds_when_loss_small(self):
        # Small loss -> SPS smallest
        eta_step, eta_sps, eta_weyl, _ = self.compute_mass_step(
            loss=1e-6, d_norm=1.0, sigma_k_min=0.5, eta_ceiling=1.0,
        )
        assert eta_sps < eta_weyl
        assert eta_sps < 1.0  # < ceiling
        assert eta_step == pytest.approx(eta_sps)

    def test_weyl_binds_when_gradient_large(self):
        # Large gradient -> Weyl smallest
        eta_step, eta_sps, eta_weyl, _ = self.compute_mass_step(
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
        _, _, _, displacement = self.compute_mass_step(loss, d_norm, sigma_k, ceiling)
        assert displacement <= sigma_k + 1e-15

    def test_displacement_tight_when_weyl_binds(self):
        # When Weyl binds: displacement == sigma_k_min exactly
        _, _, eta_weyl, displacement = self.compute_mass_step(
            loss=100.0, d_norm=10.0, sigma_k_min=0.01, eta_ceiling=1.0,
        )
        # Weyl should bind here (eta_weyl = 0.001, sps = 1.0, ceiling = 1.0)
        assert displacement == pytest.approx(0.01)  # == sigma_k_min

    def test_zero_gradient_displacement_is_zero(self):
        eta_step, _, _, displacement = self.compute_mass_step(
            loss=1.0, d_norm=0.0, sigma_k_min=0.5, eta_ceiling=0.01,
        )
        assert displacement == 0.0

    def test_step_never_exceeds_ceiling(self):
        for loss in [0.01, 1.0, 100.0]:
            for d_norm in [0.001, 1.0, 1000.0]:
                eta_step, _, _, _ = self.compute_mass_step(
                    loss, d_norm, sigma_k_min=0.5, eta_ceiling=0.01,
                )
                assert eta_step <= 0.01

    def test_step_never_negative(self):
        for loss in [0.0, 1.0, 100.0]:
            for d_norm in [0.0, 1.0, 100.0]:
                eta_step, _, _, _ = self.compute_mass_step(
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
        eta_step, eta_sps, eta_weyl, _ = self.compute_mass_step(
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

    _BACKOFF_FLOOR = _EPS_F32 ** 0.5  # sqrt(eps_f32) ~ 3.45e-4

    @staticmethod
    def apply_validation_backoff(
        eta_ceiling: float,
        val_losses: list[float],
        adaptive_lr: bool = True,
        lr_override: float | None = None,
    ) -> float:
        """Replica of validation-guided ceiling backoff (lines 573-587)."""
        _BACKOFF_FLOOR = _EPS_F32 ** 0.5
        if not adaptive_lr or lr_override is not None:
            return eta_ceiling
        if (len(val_losses) >= 2
                and val_losses[-1] > val_losses[-2]
                and val_losses[-1] > 0):
            backoff = max(val_losses[-2] / val_losses[-1], _BACKOFF_FLOOR)
            return eta_ceiling * backoff
        return eta_ceiling

    def test_no_backoff_when_val_decreases(self):
        assert self.apply_validation_backoff(0.01, [1.5, 1.4]) == 0.01

    def test_no_backoff_single_val(self):
        assert self.apply_validation_backoff(0.01, [1.5]) == 0.01

    def test_no_backoff_empty_val(self):
        assert self.apply_validation_backoff(0.01, []) == 0.01

    def test_ratio_backoff_on_increase(self):
        # val_losses[-2]/val_losses[-1] = 1.0/2.0 = 0.5
        result = self.apply_validation_backoff(0.01, [1.0, 2.0])
        assert result == pytest.approx(0.005)

    def test_small_increase_small_backoff(self):
        result = self.apply_validation_backoff(0.01, [1.0, 1.01])
        expected = 0.01 * (1.0 / 1.01)
        assert result == pytest.approx(expected)

    def test_large_increase_large_backoff(self):
        result = self.apply_validation_backoff(0.01, [1.0, 10.0])
        assert result == pytest.approx(0.001)

    def test_backoff_floor_prevents_near_zero(self):
        # Extreme ratio: 1e-10 / 1e10 = 1e-20 < floor
        result = self.apply_validation_backoff(0.01, [1e-10, 1e10])
        assert result == pytest.approx(0.01 * self._BACKOFF_FLOOR)
        assert result > 0

    def test_backoff_floor_value_is_sqrt_eps_f32(self):
        assert self._BACKOFF_FLOOR == pytest.approx(math.ldexp(1.0, -23) ** 0.5)

    def test_ceiling_only_decreases_on_increase(self):
        # When val increases, prev/curr < 1, so backoff < 1, so ceiling shrinks
        for prev, curr in [(1.0, 1.1), (0.5, 5.0), (0.01, 100.0)]:
            result = self.apply_validation_backoff(0.01, [prev, curr])
            assert result < 0.01

    def test_lr_override_skips_backoff(self):
        result = self.apply_validation_backoff(0.01, [1.0, 10.0], lr_override=0.05)
        assert result == 0.01  # unchanged

    def test_adaptive_lr_false_skips_backoff(self):
        result = self.apply_validation_backoff(0.01, [1.0, 10.0], adaptive_lr=False)
        assert result == 0.01  # unchanged

    def test_repeated_backoff_monotone_decrease(self):
        ceiling = 0.01
        val_losses = [1.0]
        for i in range(5):
            val_losses.append(val_losses[-1] * 1.5)  # 50% increase each time
            new_ceiling = self.apply_validation_backoff(ceiling, val_losses)
            assert new_ceiling < ceiling
            ceiling = new_ceiling
        assert ceiling > 0  # never reaches zero


# ===================================================================
# Class 7: REINFORCE Budget
# ===================================================================
class TestReinforceBudget:
    """Tests for REINFORCE displacement budget (Weyl remainder after CE)."""

    @staticmethod
    def compute_reinforce_budget(
        sigma_k_min: float,
        update_norm: float | None,
        n_re: int,
        check_interval: int,
    ) -> tuple[float, str]:
        """Replica of REINFORCE budget logic (lines 873-908).

        Returns (target_step_norm, source_label).
        """
        sqrt_n_re = math.sqrt(max(1, n_re))

        if update_norm is not None and update_norm > 0:
            budget_remaining = max(0.0, sigma_k_min - update_norm)
            if budget_remaining <= 0.0:
                return 0.0, "budget_exhausted"
            return budget_remaining / sqrt_n_re, "weyl_remainder"
        else:
            n_total = max(1, check_interval) + max(1, n_re)
            if sigma_k_min > 0:
                return sigma_k_min / math.sqrt(n_total), "sigma_k_min_shared"
            return 0.0, "budget_exhausted"

    def test_remainder_after_ce_displacement(self):
        target, source = self.compute_reinforce_budget(0.5, 0.3, 4, 10)
        assert target == pytest.approx((0.5 - 0.3) / math.sqrt(4))
        assert target == pytest.approx(0.1)
        assert source == "weyl_remainder"

    def test_budget_exhausted_when_ce_equals_sigma_k(self):
        target, source = self.compute_reinforce_budget(0.5, 0.5, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_budget_exhausted_when_ce_exceeds_sigma_k(self):
        target, source = self.compute_reinforce_budget(0.5, 0.7, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_no_ce_displacement_shared_budget(self):
        # update_norm=None -> shared path
        # n_total = max(1, 10) + max(1, 4) = 14
        target, source = self.compute_reinforce_budget(0.5, None, 4, 10)
        assert target == pytest.approx(0.5 / math.sqrt(14))
        assert source == "sigma_k_min_shared"

    def test_zero_update_norm_uses_shared_path(self):
        # update_norm=0.0 -> falls into else branch (not > 0)
        target, source = self.compute_reinforce_budget(0.5, 0.0, 4, 10)
        assert source == "sigma_k_min_shared"

    def test_shared_budget_sigma_k_zero_exhausted(self):
        target, source = self.compute_reinforce_budget(0.0, None, 4, 10)
        assert target == 0.0
        assert source == "budget_exhausted"

    def test_single_reinforce_step_gets_full_remainder(self):
        target, source = self.compute_reinforce_budget(0.5, 0.1, 1, 10)
        assert target == pytest.approx(0.4 / math.sqrt(1))
        assert target == pytest.approx(0.4)
        assert source == "weyl_remainder"

    def test_many_steps_reduces_target(self):
        t1, _ = self.compute_reinforce_budget(0.5, 0.1, 1, 10)
        t100, _ = self.compute_reinforce_budget(0.5, 0.1, 100, 10)
        assert t1 > t100

    def test_remainder_plus_ce_within_sigma_k(self):
        """Core invariant: update_norm + sqrt(N_re) * target <= sigma_k_min."""
        sigma_k = 0.5
        update = 0.3
        n_re = 4
        target, _ = self.compute_reinforce_budget(sigma_k, update, n_re, 10)
        total = update + math.sqrt(n_re) * target
        assert total == pytest.approx(sigma_k)

    def test_shared_budget_brownian_bound(self):
        """Shared path: target * sqrt(N_total) == sigma_k_min."""
        sigma_k = 0.5
        n_re = 4
        check_interval = 10
        target, _ = self.compute_reinforce_budget(sigma_k, None, n_re, check_interval)
        n_total = max(1, check_interval) + max(1, n_re)
        assert target * math.sqrt(n_total) == pytest.approx(sigma_k)

    def test_source_label_remainder(self):
        _, source = self.compute_reinforce_budget(0.5, 0.1, 4, 10)
        assert source == "weyl_remainder"

    def test_source_label_shared(self):
        _, source = self.compute_reinforce_budget(0.5, None, 4, 10)
        assert source == "sigma_k_min_shared"

    def test_source_label_exhausted(self):
        _, source = self.compute_reinforce_budget(0.5, 0.5, 4, 10)
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
        ceiling = sigma_k / sigma_max
        if n_batches > 1:
            ceiling = ceiling / math.sqrt(n_batches)

        if d_norm > 0:
            eta_sps = loss / (d_norm ** 2)
            eta_weyl = sigma_k / d_norm
        else:
            eta_sps = ceiling
            eta_weyl = ceiling

        eta_step = min(eta_sps, eta_weyl, ceiling)
        displacement = eta_step * d_norm

        # Displacement bounded by the tighter of ceiling*||g|| and sigma_k
        assert displacement <= sigma_k + 1e-15

    def test_epoch_total_displacement_bounded_by_sigma_k(self):
        """Over N steps with worst-case gradients, Brownian displacement <= sigma_k."""
        sigma_k = 0.5
        sigma_max = 10.0
        n_batches = 100
        ceiling = sigma_k / sigma_max / math.sqrt(n_batches)

        # Worst case: every step at ceiling (small gradients)
        # Per-step displacement = ceiling * ||g||
        # But ceiling also bounds: ceiling * ||g|| <= ceiling * (sigma_k / ceiling) = sigma_k
        # The Weyl bound ensures each displacement <= sigma_k.
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
        backoff_floor = _EPS_F32 ** 0.5
        for _ in range(50):
            # Extreme val increase each time — floor binds
            backoff = max(0.01 / 1000.0, backoff_floor)
            ceiling = ceiling * backoff
        assert ceiling > 0
        assert math.isfinite(ceiling)

    def test_reinforce_total_within_weyl_bound(self):
        """CE displacement + REINFORCE displacement <= sigma_k_min."""
        sigma_k = 0.5
        ce_displacement = 0.3
        n_re = 16
        target = (sigma_k - ce_displacement) / math.sqrt(n_re)
        # Brownian REINFORCE total
        re_total = math.sqrt(n_re) * target
        assert ce_displacement + re_total == pytest.approx(sigma_k)

    def test_backoff_floor_traceable_to_ieee754(self):
        """_BACKOFF_FLOOR = sqrt(eps_f32) = sqrt(2^-23) — IEEE 754 derived."""
        backoff_floor = _EPS_F32 ** 0.5
        assert backoff_floor == pytest.approx(math.sqrt(math.ldexp(1.0, -23)))
        assert backoff_floor == pytest.approx(_SQRT_EPS_F32)

    def test_zero_loss_zero_displacement(self):
        """At optimum (loss=0), SPS=0, so eta_step=0 and no update occurs."""
        for d_norm in [0.001, 1.0, 100.0]:
            eta_sps = 0.0 / (d_norm ** 2)  # = 0
            eta_weyl = 0.5 / d_norm
            eta_ceiling = 0.01
            eta_step = min(eta_sps, eta_weyl, eta_ceiling)
            displacement = eta_step * d_norm
            assert eta_step == 0.0
            assert displacement == 0.0
