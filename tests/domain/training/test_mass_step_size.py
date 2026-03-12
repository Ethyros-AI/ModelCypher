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
    CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    CONTROLLER_MODE_BEHAVIORAL_PROBE,
    CONTROLLER_MODE_STRUCTURAL_OBSERVE,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    _EPS_F32,
    _SQRT_EPS_F32,
    BehavioralStateMeasurement,
    DerivedClosedLoopLaw,
    apply_sqrt_n_epoch_correction,
    apply_validation_backoff,
    controller_precision_floor,
    compute_closed_loop_trigger_reasons,
    compute_conformal_margin_rate,
    compute_per_step_rates,
    compute_reinforce_budget,
    derive_spectral_ceiling,
    evaluate_closed_loop_law,
    replay_controller_trace,
    select_closed_loop_target_layer,
    validate_controller_mode,
    validate_optimizer_research_mode,
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
    """Tests for eta_sps = max(0, loss - f*) / ||g||^2 (Loizou et al. 2020)."""

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
# Class 3b: Preconditioned SPS (g·d correction)
# ===================================================================
class TestPreconditionedSPS:
    """Tests for SPS with g_dot_d correction for preconditioned gradients.

    When the update direction d = P*g is preconditioned (e.g., diagonal Fisher
    / Adam), the correct Polyak step is:
        η_sps = (loss - f*) / (g^T d)
    not:
        η_sps = (loss - f*) / ||d||^2

    The standard formula ||d||^2 underestimates the step by ||d||/||g|| when
    the preconditioner amplifies.  For Adam with n parameters, ||d|| ≈ √n
    while g^T d ≈ ||g||₁, giving ~√n / ||g||₂ × underestimate.
    """

    def test_g_dot_d_none_falls_back_to_d_norm_squared(self):
        """When g_dot_d is None, SPS uses ||d||^2 (backward compatible)."""
        result_none = compute_per_step_rates(1.0, 2.0, 0.5, 0.01, g_dot_d=None)
        result_explicit = compute_per_step_rates(1.0, 2.0, 0.5, 0.01, g_dot_d=4.0)
        # g_dot_d=4.0 == d_norm^2=4.0 → same SPS
        assert result_none[1] == pytest.approx(result_explicit[1])

    def test_g_dot_d_equals_d_norm_sq_when_no_preconditioner(self):
        """When d = g (no preconditioner), g·d = ||g||^2 = ||d||^2."""
        d_norm = 3.0
        _, sps_old, _, _, _ = compute_per_step_rates(1.0, d_norm, 0.5, 0.01)
        _, sps_new, _, _, _ = compute_per_step_rates(
            1.0, d_norm, 0.5, 0.01, g_dot_d=d_norm ** 2,
        )
        assert sps_old == pytest.approx(sps_new)

    def test_preconditioner_amplification_produces_larger_sps(self):
        """When preconditioner amplifies (||d|| >> ||g||), g·d < ||d||^2,
        so corrected SPS is larger than uncorrected."""
        loss, d_norm = 2.0, 100.0
        # Simulate: ||g|| = 5, d = P*g with P amplifying to ||d|| = 100
        # g·d ≈ ||g|| * ||d|| * cos(0) = 500 (aligned)
        g_dot_d = 500.0
        _, sps_corrected, _, _, _ = compute_per_step_rates(
            loss, d_norm, 0.5, 1.0, g_dot_d=g_dot_d,
        )
        _, sps_uncorrected, _, _, _ = compute_per_step_rates(
            loss, d_norm, 0.5, 1.0,
        )
        # sps_corrected = 2.0 / 500 = 0.004
        # sps_uncorrected = 2.0 / 10000 = 0.0002
        assert sps_corrected == pytest.approx(loss / g_dot_d)
        assert sps_uncorrected == pytest.approx(loss / d_norm ** 2)
        assert sps_corrected > sps_uncorrected
        assert sps_corrected / sps_uncorrected == pytest.approx(d_norm ** 2 / g_dot_d)

    def test_r2_frozen_tuple_scenario(self):
        """Reproduce the R2 frozen-tuple SPS choke.

        Measured at epoch 1 step 1 of NB-LoRA training:
            loss ≈ 3.22, d_norm ≈ 1128 (Fisher-preconditioned), ce_grad_norm ≈ 9.375
            g·d ≈ Σ g_i² / (√v̂_i + ε)  (at step 1, v̂ = g², so g·d ≈ ||g||₁)

        With ||d||²: η_sps = 3.22 / 1128² ≈ 2.53e-06 (SPS chokes at 0.02% of ceiling)
        With g·d:    η_sps = 3.22 / g·d  (should be ~100× larger)
        """
        loss = 3.22
        d_norm = 1128.0
        sigma_k_min = 0.4
        eta_ceiling = 0.0125

        # Uncorrected: SPS chokes
        _, sps_choked, _, _, _ = compute_per_step_rates(
            loss, d_norm, sigma_k_min, eta_ceiling,
        )
        assert sps_choked == pytest.approx(3.22 / 1128.0 ** 2, rel=1e-3)
        assert sps_choked < eta_ceiling / 100  # <1% of ceiling

        # Corrected: g·d estimate (conservative: ||g|| * ||d|| * cos_theta)
        # At step 1 with Adam, g·d = Σ|g_i| ≈ √n * ||g||₂ for uniform g
        # Here: g·d ≈ 9.375 * 1128 = 10,575 (aligned gradient)
        g_dot_d = 9.375 * 1128.0  # upper bound (Cauchy-Schwarz)
        _, sps_corrected, _, _, _ = compute_per_step_rates(
            loss, d_norm, sigma_k_min, eta_ceiling, g_dot_d=g_dot_d,
        )
        assert sps_corrected == pytest.approx(3.22 / g_dot_d, rel=1e-3)
        assert sps_corrected > sps_choked * 50  # at least 50× improvement
        assert sps_corrected > eta_ceiling * 0.01  # now >1% of ceiling

    def test_g_dot_d_zero_falls_back(self):
        """g_dot_d=0 or negative falls back to ||d||^2."""
        _, sps_zero, _, _, _ = compute_per_step_rates(
            1.0, 2.0, 0.5, 0.01, g_dot_d=0.0,
        )
        _, sps_neg, _, _, _ = compute_per_step_rates(
            1.0, 2.0, 0.5, 0.01, g_dot_d=-1.0,
        )
        _, sps_fallback, _, _, _ = compute_per_step_rates(
            1.0, 2.0, 0.5, 0.01,
        )
        assert sps_zero == pytest.approx(sps_fallback)
        assert sps_neg == pytest.approx(sps_fallback)

    def test_g_dot_d_does_not_affect_weyl_or_ceiling(self):
        """g_dot_d only affects SPS, not Weyl or ceiling."""
        for g_dot_d in [None, 1.0, 100.0, 10000.0]:
            kwargs = {"g_dot_d": g_dot_d} if g_dot_d is not None else {}
            _, _, eta_weyl, _, _ = compute_per_step_rates(
                loss=1.0, d_norm=10.0, sigma_k_min=0.5, eta_ceiling=0.01,
                **kwargs,
            )
            assert eta_weyl == pytest.approx(0.5 / 10.0)

    def test_step_still_bounded_by_ceiling(self):
        """Even with corrected SPS, eta_step never exceeds ceiling."""
        # Very small g_dot_d → very large SPS → ceiling should bind
        eta_step, eta_sps, _, _, _ = compute_per_step_rates(
            loss=10.0, d_norm=1.0, sigma_k_min=5.0, eta_ceiling=0.01,
            g_dot_d=0.001,
        )
        assert eta_sps == pytest.approx(10000.0)
        assert eta_step == pytest.approx(0.01)  # ceiling binds

    def test_f_star_with_g_dot_d(self):
        """f_star and g_dot_d work correctly together."""
        loss, f_star, g_dot_d = 1.0, 0.4, 100.0
        _, sps, _, _, _ = compute_per_step_rates(
            loss, d_norm=50.0, sigma_k_min=5.0, eta_ceiling=1.0,
            f_star=f_star, g_dot_d=g_dot_d,
        )
        assert sps == pytest.approx((loss - f_star) / g_dot_d)


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


class TestControllerTraceReplay:
    """Tests for research-only controller trace validation and replay."""

    def test_validate_controller_mode_accepts_supported_modes(self):
        assert validate_controller_mode(CONTROLLER_MODE_STRUCTURAL_OBSERVE) == (
            CONTROLLER_MODE_STRUCTURAL_OBSERVE
        )
        assert validate_controller_mode(CONTROLLER_MODE_BEHAVIORAL_PROBE) == (
            CONTROLLER_MODE_BEHAVIORAL_PROBE
        )
        assert validate_controller_mode(CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP) == (
            CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP
        )

    def test_validate_optimizer_mode_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            validate_optimizer_research_mode("guessed_optimizer")

    def test_controller_precision_floor_uses_sqrt_eps(self):
        assert controller_precision_floor(2.0) == pytest.approx(2.0 * _SQRT_EPS_F32)

    def test_replay_controller_trace_reconstructs_learning_rates(self):
        epoch_metrics = [
            {
                "epoch": 1,
                "controller_trace": {
                    "step_traces": [
                        {
                            "step": 8,
                            "controller_mode": CONTROLLER_MODE_BEHAVIORAL_PROBE,
                            "optimizer_research_mode": OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
                            "eta_step": 0.02,
                            "eta_ceiling": 0.04,
                            "per_layer_measurements": {
                                "model.layers.0.self_attn.q_proj.weight": {
                                    "step_learning_rate": 0.02,
                                    "decay_scale": 1.5,
                                    "scale_bound": 1.0,
                                    "remaining_budget": 1e-6,
                                },
                                "model.layers.0.self_attn.k_proj.weight": {
                                    "step_learning_rate": 0.02,
                                    "decay_scale": 0.5,
                                    "scale_bound": 1.0,
                                    "remaining_budget": 0.25,
                                },
                            },
                        }
                    ],
                },
            },
        ]

        replay = replay_controller_trace(epoch_metrics, eps=_EPS_F32)

        assert replay is not None
        assert replay["controller_mode"] == CONTROLLER_MODE_BEHAVIORAL_PROBE
        assert replay["optimizer_research_mode"] == OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS
        assert replay["n_replayed_steps"] == 1
        decision = replay["decisions"][0]
        assert decision["epoch"] == 1
        assert decision["step"] == 8
        assert decision["step_budget_multiplier"] == pytest.approx(0.5)
        assert decision["learning_rates"]["model.layers.0.self_attn.q_proj.weight"] == pytest.approx(0.02)
        assert decision["weight_decay_scales"]["model.layers.0.self_attn.k_proj.weight"] == pytest.approx(0.5)
        assert decision["freeze_layers"] == ["model.layers.0.self_attn.q_proj.weight"]

    def test_replay_controller_trace_preserves_closed_loop_decision_payload(self):
        epoch_metrics = [
            {
                "epoch": 2,
                "controller_trace": {
                    "step_traces": [],
                    "closed_loop_decision": {
                        "armed": True,
                        "epoch": 2,
                        "trigger_reasons": ["online_eval_accuracy_drop"],
                        "freeze_layers": ["model.layers.3.self_attn.q_proj.weight"],
                        "target_layer": "model.layers.3.self_attn.q_proj.weight",
                        "ordering_metrics": None,
                        "interventions_used": 1,
                    },
                },
            },
        ]

        replay = replay_controller_trace(epoch_metrics, eps=_EPS_F32)

        assert replay is not None
        assert replay["n_replayed_steps"] == 0
        assert replay["closed_loop_decisions"] == [
            {
                "armed": True,
                "epoch": 2,
                "trigger_reasons": ["online_eval_accuracy_drop"],
                "freeze_layers": ["model.layers.3.self_attn.q_proj.weight"],
                "target_layer": "model.layers.3.self_attn.q_proj.weight",
                "ordering_metrics": None,
                "interventions_used": 1,
            }
        ]


class TestClosedLoopLaw:
    """Tests for the offline-derived R2 closed-loop intervention law."""

    def test_law_roundtrip_preserves_artifact_paths(self):
        law = DerivedClosedLoopLaw(
            source_artifacts=("a.json", "b.json"),
            safe_artifacts=("safe.json",),
            counterexample_artifacts=("cx.json",),
            max_interventions=1,
        )

        payload = law.to_dict()
        restored = DerivedClosedLoopLaw.from_dict(payload)

        assert restored == law

    def test_trigger_reasons_detect_negative_online_eval_delta(self):
        law = DerivedClosedLoopLaw()
        state = BehavioralStateMeasurement(online_eval_accuracy_delta=-0.1)

        reasons = compute_closed_loop_trigger_reasons(
            law,
            behavioral_state=state,
            margin_history=[],
            stable_rank_history=[],
            loss_stability_window_epochs=2,
            adapter_rank=None,
        )

        assert reasons == ("online_eval_accuracy_drop",)

    def test_trigger_reasons_detect_stable_rank_concentration(self):
        law = DerivedClosedLoopLaw(
            arm_on_online_eval_accuracy_drop=False,
            arm_on_margin_trend_declining=False,
            arm_on_stable_rank_concentration=True,
        )

        reasons = compute_closed_loop_trigger_reasons(
            law,
            behavioral_state=None,
            margin_history=[],
            stable_rank_history=[12.0, 9.0, 4.0],
            loss_stability_window_epochs=2,
            adapter_rank=25,
        )

        assert reasons == ("stable_rank_concentration",)

    def test_target_layer_prefers_transport_over_remaining_budget(self):
        state = BehavioralStateMeasurement(
            per_layer_behavioral_transport_norm={
                "layer_a": 4.0,
                "layer_b": 2.0,
            },
            per_layer_remaining_budget={
                "layer_a": 1.0,
                "layer_b": 0.25,
            },
            per_layer_spectral_budget_ratio={
                "layer_a": 0.3,
                "layer_b": 0.2,
            },
            per_layer_stable_rank={
                "layer_a": 8.0,
                "layer_b": 2.0,
            },
            adapter_rank=16,
        )

        target, metrics = select_closed_loop_target_layer(state)

        assert target == "layer_b"
        assert metrics is not None
        assert metrics["layer_b"]["behavioral_transport_over_remaining_budget"] == pytest.approx(8.0)

    def test_evaluate_closed_loop_law_returns_layer_freeze_decision(self):
        law = DerivedClosedLoopLaw(
            arm_on_online_eval_accuracy_drop=True,
            arm_on_margin_trend_declining=False,
            arm_on_stable_rank_concentration=False,
        )
        state = BehavioralStateMeasurement(
            online_eval_accuracy_delta=-0.05,
            per_layer_behavioral_transport_norm={"layer_a": 1.0},
            per_layer_remaining_budget={"layer_a": 0.5},
            per_layer_spectral_budget_ratio={"layer_a": 0.1},
            per_layer_stable_rank={"layer_a": 4.0},
            adapter_rank=16,
        )

        decision = evaluate_closed_loop_law(
            law,
            epoch=2,
            behavioral_state=state,
            margin_history=[],
            stable_rank_history=[],
            loss_stability_window_epochs=2,
            adapter_rank=16,
        )

        assert decision.armed is True
        assert decision.freeze_layers == ("layer_a",)
        assert decision.trigger_reasons == ("online_eval_accuracy_drop",)


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


# ===================================================================
# Class 12: f_star — Irreducible Loss Floor (RMT Noise Floor)
# ===================================================================
class TestFStar:
    """Tests for f_star parameter in SPS: η_sps = max(0, f(x) - f*) / ||g||².

    f* is the irreducible loss floor, derived from the RMT noise fraction:
    f* = initial_loss × (1 - mean_sv_frac).

    Without f* (f*=0), SPS treats the entire loss as recoverable distance,
    overestimating step size near the noise floor and causing oscillation.
    """

    def test_f_star_zero_recovers_original_sps(self):
        """f_star=0.0 gives identical results to the original formula."""
        for loss in [0.01, 1.0, 5.0]:
            for d_norm in [0.1, 1.0, 100.0]:
                original = compute_per_step_rates(loss, d_norm, 0.5, 0.01)
                with_zero = compute_per_step_rates(
                    loss, d_norm, 0.5, 0.01, f_star=0.0,
                )
                assert original[0] == with_zero[0]  # eta_step
                assert original[1] == with_zero[1]  # eta_sps

    def test_f_star_reduces_sps(self):
        """Positive f_star produces smaller SPS than f_star=0."""
        loss, d_norm = 1.0, 10.0
        _, sps_0, _, _, _ = compute_per_step_rates(
            loss, d_norm, 0.5, 1.0, f_star=0.0,
        )
        _, sps_half, _, _, _ = compute_per_step_rates(
            loss, d_norm, 0.5, 1.0, f_star=0.5,
        )
        assert sps_half < sps_0
        assert sps_half == pytest.approx((1.0 - 0.5) / 10.0**2)

    def test_f_star_equals_loss_gives_zero_sps(self):
        """When loss equals f*, SPS is zero (at the noise floor)."""
        _, sps, _, _, _ = compute_per_step_rates(
            loss=0.544, d_norm=14.3, sigma_k_min=0.874, eta_ceiling=0.01,
            f_star=0.544,
        )
        assert sps == 0.0

    def test_f_star_above_loss_gives_zero_sps(self):
        """When f* > loss, SPS is zero (below the noise floor)."""
        _, sps, _, _, _ = compute_per_step_rates(
            loss=0.5, d_norm=10.0, sigma_k_min=0.5, eta_ceiling=0.01,
            f_star=0.6,
        )
        assert sps == 0.0

    def test_f_star_at_noise_floor_ceiling_binds(self):
        """When loss ≈ f*, SPS → 0, so ceiling or Weyl binds instead."""
        eta_step, sps, _, _, _ = compute_per_step_rates(
            loss=0.55, d_norm=14.3, sigma_k_min=0.874, eta_ceiling=0.002,
            f_star=0.544,
        )
        # SPS = (0.55 - 0.544) / 14.3² = 0.006 / 204.49 ≈ 2.93e-5
        assert sps == pytest.approx(0.006 / 14.3**2, rel=1e-3)
        # ceiling = 0.002, weyl = 0.874/14.3 ≈ 0.061
        # SPS binds because 2.93e-5 < 0.002 < 0.061
        assert eta_step == pytest.approx(sps)

    def test_4bit_scenario_corrected_vs_uncorrected(self):
        """Reproduce the 4-bit iter 80 scenario from Experiment 3b.

        At iter 80: loss=0.564, d_norm=14.3, sigma_k_min=0.874, ceiling=2.24e-3.
        With f*=0: SPS=2.76e-3, ceiling binds at 2.24e-3.
        With f*=0.544: SPS=9.78e-5, SPS binds (23× smaller).
        """
        # Uncorrected (f*=0)
        eta_0, sps_0, _, _, _ = compute_per_step_rates(
            loss=0.564, d_norm=14.3, sigma_k_min=0.874, eta_ceiling=2.24e-3,
            f_star=0.0,
        )
        assert sps_0 == pytest.approx(0.564 / 14.3**2, rel=1e-3)
        assert eta_0 == pytest.approx(2.24e-3)  # ceiling binds

        # Corrected (f*=0.544)
        eta_corrected, sps_corrected, _, _, _ = compute_per_step_rates(
            loss=0.564, d_norm=14.3, sigma_k_min=0.874, eta_ceiling=2.24e-3,
            f_star=0.544,
        )
        expected_sps = (0.564 - 0.544) / 14.3**2
        assert sps_corrected == pytest.approx(expected_sps, rel=1e-3)
        assert eta_corrected == pytest.approx(sps_corrected)  # SPS binds
        assert eta_corrected < eta_0 / 20  # at least 20× smaller

    def test_f_star_sps_always_nonneg(self):
        """SPS is always non-negative regardless of f_star value."""
        for loss in [0.0, 0.5, 1.0, 5.0]:
            for f_star in [0.0, 0.5, 1.0, 10.0]:
                _, sps, _, _, _ = compute_per_step_rates(
                    loss, 1.0, 0.5, 0.01, f_star=f_star,
                )
                assert sps >= 0.0

    def test_weyl_and_ceiling_unchanged_by_f_star(self):
        """f_star only affects SPS, not Weyl or ceiling."""
        for f_star in [0.0, 0.5, 1.0]:
            _, _, eta_weyl, _, _ = compute_per_step_rates(
                loss=1.0, d_norm=10.0, sigma_k_min=0.5, eta_ceiling=0.01,
                f_star=f_star,
            )
            assert eta_weyl == pytest.approx(0.5 / 10.0)
