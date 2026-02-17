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

from modelcypher.core.domain.training.geometric_early_stopping import (
    _SQRT_EPS,
    StoppingCertificate,
    check_grad_norm_stable,
    check_loss_stable,
    check_stopping_certificate,
    check_val_loss_converged,
)


class TestCheckLossStable:
    """Tests for check_loss_stable() — data-derived early stopping."""

    def test_constant_loss_is_stable(self):
        """20 identical losses → converged."""
        losses = [(i, 1.0, 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is True
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_diverging_loss_not_stable(self):
        """Recent mean >> earlier mean → not converged."""
        losses = [(i, 1.0 if i < 10 else 5.0, 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is False

    def test_auto_window_derivation(self):
        """10 losses → auto window = 5, splits into two windows of 5."""
        losses = [(i, 1.0, 100.0) for i in range(10)]
        is_stable, threshold = check_loss_stable(losses)

        # 10 // 2 = 5, both windows all 1.0 → stable
        assert is_stable is True

    def test_insufficient_history(self):
        """3 losses with window=5 → insufficient (need 2*5=10)."""
        losses = [(i, 1.0, 100.0) for i in range(3)]
        is_stable, threshold = check_loss_stable(losses, window=5)

        assert is_stable is False
        assert threshold == 0.0

    def test_window_less_than_two(self):
        """window=1 → early return (False, 0.0)."""
        losses = [(i, 1.0, 100.0) for i in range(10)]
        is_stable, threshold = check_loss_stable(losses, window=1)

        assert is_stable is False
        assert threshold == 0.0

    def test_numeric_floor_triggers(self):
        """Low-variance data → threshold clamped to _SQRT_EPS."""
        # All losses within 1e-8 of 1.0 → SE_diff ~ 0, clamped to _SQRT_EPS
        losses = [(i, 1.0 + 1e-10 * (i % 2), 100.0) for i in range(20)]
        is_stable, threshold = check_loss_stable(losses)

        assert is_stable is True
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_high_variance_data_driven_threshold(self):
        """Noisy losses → threshold > _SQRT_EPS."""
        # Alternating pattern creates measurable variance
        losses = [(i, 1.0 + 0.1 * (-1) ** i, 100.0) for i in range(20)]
        _, threshold = check_loss_stable(losses)

        assert threshold > _SQRT_EPS


class TestCheckValLossConverged:
    """Tests for check_val_loss_converged() — validation-based early stopping."""

    def test_constant_val_loss_is_stable(self):
        """Flat validation loss → val_stable."""
        val_losses = [1.0] * 10
        should_stop, reason, threshold = check_val_loss_converged(val_losses)

        assert should_stop is True
        assert reason == "val_stable"
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_increasing_val_loss_is_overfitting(self):
        """Rising validation loss → val_increasing (overfitting detected)."""
        # 3 epochs at 1.0, then 3 epochs at 2.0 — clear increase
        val_losses = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is True
        assert reason == "val_increasing"

    def test_decreasing_val_loss_continues(self):
        """Decreasing validation loss → don't stop, still improving."""
        # 3 epochs at 2.0, then 3 epochs at 1.0 — still improving
        val_losses = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is False
        assert reason == ""

    def test_insufficient_history(self):
        """Less than 2*window entries → don't stop."""
        val_losses = [1.0, 1.0, 1.0]  # 3 < 2*3=6
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is False
        assert reason == ""

    def test_window_too_small(self):
        """window < 2 → early return."""
        val_losses = [1.0] * 10
        should_stop, reason, threshold = check_val_loss_converged(val_losses, window=1)

        assert should_stop is False
        assert threshold == 0.0

    def test_gradual_overfitting_detected(self):
        """Slowly increasing val loss (realistic overfitting pattern)."""
        # Epochs 0-2: val loss around 1.5 (improving)
        # Epochs 3-5: val loss around 1.8 (degrading)
        val_losses = [1.5, 1.45, 1.52, 1.75, 1.80, 1.82]
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is True
        assert reason == "val_increasing"

    def test_noisy_but_flat_is_stable(self):
        """Noisy but centered val loss → stable if within SE."""
        # Both windows oscillate around 1.0 with same variance
        val_losses = [0.9, 1.1, 0.95, 1.05, 0.92, 1.08]
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is True
        assert reason == "val_stable"

    def test_classic_overfitting_curve(self):
        """U-shaped val loss: decrease then increase."""
        # Improving: 3.0 → 2.0 → 1.5 (still improving, don't stop)
        # Then: 1.5 → 1.6 → 1.8 → 2.0 (overfitting, stop)
        val_losses = [3.0, 2.0, 1.5, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]
        should_stop, reason, _ = check_val_loss_converged(val_losses)

        assert should_stop is True
        assert reason == "val_increasing"


class TestCheckStoppingCertificate:
    """Tests for the geometric stopping certificate."""

    def test_all_conditions_met_stops(self):
        """When all four conditions hold, certificate says stop."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,      # Below _SQRT_EPS (~3.45e-4)
            alignment=1e-8,
            curvature=1.0,
            val_ci_half_width=1e-3,
            mean_token_entropy=3.5,
            repetition_rate=0.1,
        )
        assert isinstance(cert, StoppingCertificate)
        assert cert.stationarity_met is True
        assert cert.improvement_bound_met is True
        assert cert.worst_group_met is True
        assert cert.no_drift is True
        assert cert.all_conditions_met is True

    def test_large_gradient_blocks(self):
        """Stationarity not met: gradient norm above floor."""
        cert = check_stopping_certificate(
            precond_grad_norm=1.0,       # Way above _SQRT_EPS
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
        )
        assert cert.stationarity_met is False
        assert cert.all_conditions_met is False

    def test_large_improvement_blocks(self):
        """Improvement bound not met: step can still improve val."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=1.0,
            curvature=0.1,
            val_ci_half_width=1e-6,
        )
        # delta_max = 1.0^2 / (2*0.1) = 5.0, way above CI of 1e-6
        assert cert.delta_max_val == pytest.approx(5.0)
        assert cert.improvement_bound_met is False
        assert cert.all_conditions_met is False

    def test_negative_alignment_satisfies(self):
        """Negative alignment = step worsens val, delta_max = 0."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=-0.5,
            curvature=1.0,
            val_ci_half_width=1e-3,
        )
        assert cert.delta_max_val == 0.0
        assert cert.improvement_bound_met is True

    def test_zero_curvature_satisfies(self):
        """Zero curvature with positive alignment gives no finite improvement bound."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=1.0,
            curvature=0.0,
            val_ci_half_width=1e-3,
        )
        assert cert.delta_max_val == float("inf")
        assert cert.improvement_bound_met is False

    def test_negative_curvature_satisfies(self):
        """Negative curvature with positive alignment gives no finite improvement bound."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=1.0,
            curvature=-5.0,
            val_ci_half_width=1e-3,
        )
        assert cert.delta_max_val == float("inf")
        assert cert.improvement_bound_met is False

    def test_non_positive_curvature_with_non_descent_alignment(self):
        """If alignment <= 0, improvement bound is zero even when curvature <= 0."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=-1.0,
            curvature=-5.0,
            val_ci_half_width=1e-3,
        )
        assert cert.delta_max_val == 0.0
        assert cert.improvement_bound_met is True

    def test_entropy_collapse_detects_drift(self):
        """Entropy below floor = mechanism drift."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
            mean_token_entropy=1e-10,
            repetition_rate=0.0,
        )
        assert cert.entropy_collapsed is True
        assert cert.no_drift is False
        assert cert.all_conditions_met is False

    def test_repetition_spike_detects_drift(self):
        """Repetition near 1.0 = mechanism drift."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
            mean_token_entropy=3.0,
            repetition_rate=0.9999,
        )
        assert cert.repetition_spiked is True
        assert cert.no_drift is False
        assert cert.all_conditions_met is False

    def test_none_probes_not_drift(self):
        """None entropy/repetition = probes unavailable, not drift."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
            mean_token_entropy=None,
            repetition_rate=None,
        )
        assert cert.entropy_collapsed is False
        assert cert.repetition_spiked is False
        assert cert.no_drift is True

    def test_worst_group_blocks(self):
        """One batch with large delta_max_i blocks."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
            per_batch_alignments=[0.0, 0.0, 5.0],
            per_batch_curvatures=[1.0, 1.0, 0.1],
            per_batch_ci_half_widths=[1e-3, 1e-3, 1e-3],
        )
        # Worst batch: 5.0^2 / (2*0.1) = 125, way above CI of 1e-3
        assert cert.delta_max_worst == pytest.approx(125.0)
        assert cert.worst_group_met is False
        assert cert.all_conditions_met is False

    def test_worst_group_satisfied_when_all_small(self):
        """All per-batch improvements below CI → worst-group met."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
            per_batch_alignments=[1e-6, 1e-6, 1e-6],
            per_batch_curvatures=[1.0, 1.0, 1.0],
            per_batch_ci_half_widths=[1e-3, 1e-3, 1e-3],
        )
        assert cert.worst_group_met is True

    def test_no_val_data_vacuously_satisfied(self):
        """val_ci_half_width = 0 → conditions 2 & 3 vacuously satisfied."""
        cert = check_stopping_certificate(
            precond_grad_norm=1e-5,
            alignment=1.0,
            curvature=0.5,
            val_ci_half_width=0.0,
        )
        # delta_max = 1.0, but CI=0 → vacuously true
        assert cert.improvement_bound_met is True
        assert cert.worst_group_met is True

    def test_to_dict(self):
        """StoppingCertificate.to_dict() returns all fields."""
        cert = check_stopping_certificate(precond_grad_norm=1e-5)
        d = cert.to_dict()
        assert "precond_grad_norm" in d
        assert "all_conditions_met" in d
        assert isinstance(d, dict)

    def test_frozen_dataclass(self):
        """StoppingCertificate is immutable."""
        cert = check_stopping_certificate(precond_grad_norm=1e-5)
        with pytest.raises(AttributeError):
            cert.precond_grad_norm = 999.0  # type: ignore[misc]

    def test_stochastic_stationarity_with_stable_history(self):
        """Gradient norm history that has converged → stationarity_met = True."""
        # 6 epochs oscillating tightly around 2.3 — both windows see same mean
        history = [2.31, 2.29, 2.32, 2.30, 2.28, 2.31]
        cert = check_stopping_certificate(
            precond_grad_norm=2.31,
            grad_norm_history=history,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
        )
        assert cert.stationarity_met is True
        assert cert.stationarity_floor > 0.0

    def test_stochastic_stationarity_includes_current_norm(self):
        """A current spike must break stationarity even with stable history."""
        history = [2.31, 2.29, 2.32, 2.30, 2.28, 2.31]
        cert = check_stopping_certificate(
            precond_grad_norm=9.0,
            grad_norm_history=history,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
        )
        assert cert.stationarity_met is False

    def test_stochastic_stationarity_with_decreasing_history(self):
        """Gradient norm still decreasing → stationarity_met = False."""
        # Norm is clearly still dropping
        history = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
        cert = check_stopping_certificate(
            precond_grad_norm=2.5,
            grad_norm_history=history,
            alignment=0.0,
            curvature=1.0,
            val_ci_half_width=1e-3,
        )
        assert cert.stationarity_met is False

    def test_stochastic_stationarity_all_conditions_met(self):
        """All four conditions met with stochastic stationarity → stop."""
        # Norm stabilized tightly around 2.37
        history = [2.36, 2.38, 2.37, 2.37, 2.36, 2.38]
        cert = check_stopping_certificate(
            precond_grad_norm=2.38,
            grad_norm_history=history,
            alignment=1e-8,
            curvature=1.0,
            val_ci_half_width=1e-3,
            mean_token_entropy=3.5,
            repetition_rate=0.1,
        )
        assert cert.stationarity_met is True
        assert cert.improvement_bound_met is True
        assert cert.no_drift is True
        assert cert.all_conditions_met is True

    def test_stochastic_stationarity_short_history_falls_back(self):
        """Too-short gradient norm history → falls back to deterministic check."""
        # Only 3 entries, need at least 2*3=6 for window=3
        history = [5.0, 4.0, 3.0]
        cert = check_stopping_certificate(
            precond_grad_norm=3.0,  # Above _SQRT_EPS → stationarity False
            grad_norm_history=history,
        )
        # Falls back to precond_grad_norm < numeric_floor → False
        assert cert.stationarity_met is False

        cert2 = check_stopping_certificate(
            precond_grad_norm=1e-5,  # Below _SQRT_EPS → stationarity True
            grad_norm_history=history,
        )
        assert cert2.stationarity_met is True


class TestCheckGradNormStable:
    """Tests for check_grad_norm_stable() — stochastic stationarity."""

    def test_constant_norm_is_stable(self):
        """Flat gradient norm → converged."""
        norms = [2.3] * 10
        is_stable, threshold = check_grad_norm_stable(norms)

        assert is_stable is True
        assert threshold == pytest.approx(_SQRT_EPS)

    def test_decreasing_norm_not_stable(self):
        """Steadily decreasing norm → not converged."""
        norms = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
        is_stable, threshold = check_grad_norm_stable(norms)

        assert is_stable is False

    def test_oscillating_norm_is_stable(self):
        """Norm oscillating around a value → converged."""
        norms = [2.3, 2.35, 2.28, 2.32, 2.30, 2.34]
        is_stable, threshold = check_grad_norm_stable(norms)

        assert is_stable is True

    def test_insufficient_history(self):
        """Not enough points → not stable."""
        norms = [2.3, 2.4, 2.35]  # 3 < 2*3=6
        is_stable, threshold = check_grad_norm_stable(norms)

        assert is_stable is False
        assert threshold == 0.0

    def test_window_too_small(self):
        """window < 2 → early return."""
        norms = [2.3] * 10
        is_stable, threshold = check_grad_norm_stable(norms, window=1)

        assert is_stable is False
        assert threshold == 0.0

    def test_step_change_not_stable(self):
        """Norm drops between windows → not converged."""
        # First window: ~4.0, second window: ~2.0
        norms = [4.0, 4.1, 3.9, 2.0, 2.1, 1.9]
        is_stable, threshold = check_grad_norm_stable(norms)

        assert is_stable is False
