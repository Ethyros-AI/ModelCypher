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

"""MASS (Measured-Adaptive Step Size) learning rate derivation.

Three-layer system where every number derives from:
- Weyl 1912 (spectral displacement bound)
- Loizou et al. 2020 (stochastic Polyak step size)
- Sahraee-Ardakan, Delbracio & Milanfar 2026 (conformal margin deceleration)
- IEEE 754 (numerical stability floors)

Zero framework dependencies. All formulas are pure Python scalar arithmetic.
Framework-specific optimizer updates (mx.array, torch.tensor, etc.) stay
in the backend adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from modelcypher.core.domain.training.exceptions import TrainingDerivationError

# IEEE 754 float32 derived constants
_EPS_F32 = math.ldexp(1.0, -23)  # 2^-23, float32 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)  # ~3.45e-4, used as backoff floor

CONTROLLER_MODE_STRUCTURAL_OBSERVE = "mass_structural_observe"
CONTROLLER_MODE_BEHAVIORAL_PROBE = "mass_behavioral_probe"
CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP = "mass_behavioral_closed_loop"

OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS = "cayley_stiefel_mass"
OPTIMIZER_MODE_ADAMW_MATCHED_TRACE = "adamw_matched_trace"

VALID_CONTROLLER_MODES = frozenset(
    {
        CONTROLLER_MODE_STRUCTURAL_OBSERVE,
        CONTROLLER_MODE_BEHAVIORAL_PROBE,
        CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    },
)
VALID_OPTIMIZER_RESEARCH_MODES = frozenset(
    {
        OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
        OPTIMIZER_MODE_ADAMW_MATCHED_TRACE,
    },
)


def _serialize_mapping(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
    if mapping is None:
        return None
    return {
        str(key): (
            value.to_dict() if hasattr(value, "to_dict") else value
        )
        for key, value in mapping.items()
    }


@dataclass(frozen=True)
class ControllerLayerMeasurement:
    """Raw per-layer measurements for controller derivation."""

    parameter_update_norm: float | None = None
    behavioral_transport_norm: float | None = None
    spectral_budget_ratio: float | None = None
    remaining_budget: float | None = None
    total_step_fraction: float | None = None
    decay_scale: float | None = None
    scale_bound: float | None = None
    step_learning_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class BehavioralStateMeasurement:
    """Expensive retained-probe measurements collected at eval cadence."""

    per_layer_behavioral_transport_norm: dict[str, float] | None = None
    per_layer_spectral_budget_ratio: dict[str, float] | None = None
    per_layer_remaining_budget: dict[str, float] | None = None
    margin_mean_delta: float | None = None
    margin_n_flipped_sign: int | None = None
    margin_n_near_zero_baseline: int | None = None
    margin_n_near_zero_current: int | None = None
    online_eval_accuracy_delta: float | None = None
    online_eval_n_lost: int | None = None
    online_eval_n_gained: int | None = None
    cka_blindness_ratio: float | None = None
    cka_blindness_worst_layer: int | None = None
    null_accessibility: dict[str, dict[str, float | int]] | None = None
    null_observability: dict[str, dict[str, float | int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_layer_behavioral_transport_norm": self.per_layer_behavioral_transport_norm,
            "per_layer_spectral_budget_ratio": self.per_layer_spectral_budget_ratio,
            "per_layer_remaining_budget": self.per_layer_remaining_budget,
            "margin_mean_delta": self.margin_mean_delta,
            "margin_n_flipped_sign": self.margin_n_flipped_sign,
            "margin_n_near_zero_baseline": self.margin_n_near_zero_baseline,
            "margin_n_near_zero_current": self.margin_n_near_zero_current,
            "online_eval_accuracy_delta": self.online_eval_accuracy_delta,
            "online_eval_n_lost": self.online_eval_n_lost,
            "online_eval_n_gained": self.online_eval_n_gained,
            "cka_blindness_ratio": self.cka_blindness_ratio,
            "cka_blindness_worst_layer": self.cka_blindness_worst_layer,
            "null_accessibility": self.null_accessibility,
            "null_observability": self.null_observability,
        }


@dataclass(frozen=True)
class ControllerStepTrace:
    """Per-step controller state emitted by the training adapter."""

    step: int
    controller_mode: str
    optimizer_research_mode: str
    objective_components: list[str]
    ce_grad_norm: float | None = None
    auxiliary_grad_norms: dict[str, float] | None = None
    total_effective_step_norm: float | None = None
    eta_ceiling: float | None = None
    eta_sps: float | None = None
    eta_weyl: float | None = None
    eta_margin: float | None = None
    eta_step: float | None = None
    effective_gain_ratio: float | None = None
    optimizer_first_moment_norm: float | None = None
    optimizer_second_moment_norm: float | None = None
    optimizer_preconditioner_scale: float | None = None
    per_layer_measurements: dict[str, ControllerLayerMeasurement] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "controller_mode": self.controller_mode,
            "optimizer_research_mode": self.optimizer_research_mode,
            "objective_components": list(self.objective_components),
            "ce_grad_norm": self.ce_grad_norm,
            "auxiliary_grad_norms": self.auxiliary_grad_norms,
            "total_effective_step_norm": self.total_effective_step_norm,
            "eta_ceiling": self.eta_ceiling,
            "eta_sps": self.eta_sps,
            "eta_weyl": self.eta_weyl,
            "eta_margin": self.eta_margin,
            "eta_step": self.eta_step,
            "effective_gain_ratio": self.effective_gain_ratio,
            "optimizer_first_moment_norm": self.optimizer_first_moment_norm,
            "optimizer_second_moment_norm": self.optimizer_second_moment_norm,
            "optimizer_preconditioner_scale": self.optimizer_preconditioner_scale,
            "per_layer_measurements": _serialize_mapping(self.per_layer_measurements),
        }


@dataclass(frozen=True)
class ControllerReplayDecision:
    """Deterministic replay of a controller decision from saved telemetry."""

    epoch: int
    step: int
    controller_mode: str
    optimizer_research_mode: str
    learning_rates: dict[str, float]
    step_budget_multiplier: float | None
    weight_decay_scales: dict[str, float] | None
    freeze_layers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "controller_mode": self.controller_mode,
            "optimizer_research_mode": self.optimizer_research_mode,
            "learning_rates": self.learning_rates,
            "step_budget_multiplier": self.step_budget_multiplier,
            "weight_decay_scales": self.weight_decay_scales,
            "freeze_layers": list(self.freeze_layers),
        }


def validate_controller_mode(controller_mode: str) -> str:
    """Validate a research controller mode name."""
    if controller_mode not in VALID_CONTROLLER_MODES:
        raise ValueError(
            f"Unsupported controller_mode={controller_mode!r}; "
            f"expected one of {sorted(VALID_CONTROLLER_MODES)}"
        )
    return controller_mode


def validate_optimizer_research_mode(optimizer_research_mode: str) -> str:
    """Validate a research optimizer comparison mode name."""
    if optimizer_research_mode not in VALID_OPTIMIZER_RESEARCH_MODES:
        raise ValueError(
            f"Unsupported optimizer_research_mode={optimizer_research_mode!r}; "
            f"expected one of {sorted(VALID_OPTIMIZER_RESEARCH_MODES)}"
        )
    return optimizer_research_mode


def controller_precision_floor(
    scale_bound: float,
    *,
    eps: float = _EPS_F32,
) -> float:
    """Precision-derived remaining-budget floor for replayed freeze decisions."""
    if scale_bound <= 0.0 or not math.isfinite(scale_bound):
        return 0.0
    return scale_bound * math.sqrt(max(eps, 0.0))


def replay_controller_trace(
    epoch_metrics: Sequence[dict[str, Any]] | None,
    *,
    eps: float = _EPS_F32,
) -> dict[str, Any] | None:
    """Replay controller decisions exactly from saved telemetry.

    The current MASS controller is global, so every active layer shares the
    measured step learning rate. Replay exists to prove that the saved raw
    trace is sufficient to reconstruct the emitted control law without
    re-running training.
    """
    if not epoch_metrics:
        return None

    decisions: list[dict[str, Any]] = []
    controller_mode: str | None = None
    optimizer_mode: str | None = None
    replay_floor_multiplier = math.sqrt(max(eps, 0.0))

    for epoch_metric in epoch_metrics:
        if not isinstance(epoch_metric, dict):
            continue
        epoch_idx = int(epoch_metric.get("epoch", 0))
        controller_trace = epoch_metric.get("controller_trace")
        if not isinstance(controller_trace, dict):
            continue
        for step_trace in controller_trace.get("step_traces", []) or []:
            if not isinstance(step_trace, dict):
                continue
            controller_mode = str(
                step_trace.get("controller_mode", controller_mode or CONTROLLER_MODE_STRUCTURAL_OBSERVE)
            )
            optimizer_mode = str(
                step_trace.get(
                    "optimizer_research_mode",
                    optimizer_mode or OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
                )
            )
            eta_step = step_trace.get("eta_step")
            eta_ceiling = step_trace.get("eta_ceiling")
            per_layer_raw = step_trace.get("per_layer_measurements") or {}
            learning_rates: dict[str, float] = {}
            weight_decay_scales: dict[str, float] = {}
            freeze_layers: list[str] = []
            for layer_key, layer_raw in dict(per_layer_raw).items():
                if not isinstance(layer_raw, dict):
                    continue
                layer_lr = layer_raw.get("step_learning_rate")
                if layer_lr is None:
                    layer_lr = eta_step
                if layer_lr is not None:
                    learning_rates[str(layer_key)] = float(layer_lr)
                decay_scale = layer_raw.get("decay_scale")
                if decay_scale is not None:
                    weight_decay_scales[str(layer_key)] = float(decay_scale)
                remaining_budget = layer_raw.get("remaining_budget")
                scale_bound = layer_raw.get("scale_bound")
                if (
                    remaining_budget is not None
                    and scale_bound is not None
                    and float(remaining_budget) <= controller_precision_floor(float(scale_bound), eps=eps)
                ):
                    freeze_layers.append(str(layer_key))

            budget_multiplier: float | None = None
            if eta_step is not None and eta_ceiling not in (None, 0.0):
                budget_multiplier = float(eta_step) / float(eta_ceiling)

            decision = ControllerReplayDecision(
                epoch=epoch_idx,
                step=int(step_trace.get("step", 0)),
                controller_mode=controller_mode,
                optimizer_research_mode=optimizer_mode,
                learning_rates=learning_rates,
                step_budget_multiplier=budget_multiplier,
                weight_decay_scales=weight_decay_scales or None,
                freeze_layers=sorted(freeze_layers),
            )
            decisions.append(decision.to_dict())

    if not decisions:
        return None

    return {
        "controller_mode": controller_mode,
        "optimizer_research_mode": optimizer_mode,
        "precision_floor_multiplier": replay_floor_multiplier,
        "n_replayed_steps": len(decisions),
        "decisions": decisions,
    }


def derive_spectral_ceiling(
    *,
    sigma_k_min: float,
    sigma_max_global: float,
) -> float:
    """Derive static learning rate ceiling from adapter geometry (Weyl 1912).

    eta_ceiling = sigma_k_min / sigma_max_global

    Where sigma_k_min is the minimum structural-rank singular value across
    all adapted layers (smallest spectral gap) and sigma_max_global is the
    largest singular value (maximum gradient amplification).

    After computing n_batches_per_epoch, the caller applies the sqrt(N) epoch
    budget correction via :func:`apply_sqrt_n_epoch_correction`.
    """
    if sigma_k_min <= 0 or sigma_max_global <= 0:
        raise TrainingDerivationError(
            failure_class="insufficient_adapter_geometry",
            detail=(
                "Spectral ceiling derivation failed: sigma_k_min or sigma_max_global "
                "non-positive. Check that adapted layers have valid SVD geometry."
            ),
            diagnostics={
                "sigma_k_min": sigma_k_min,
                "sigma_max_global": sigma_max_global,
            },
        )

    if not math.isfinite(sigma_k_min) or not math.isfinite(sigma_max_global):
        raise TrainingDerivationError(
            failure_class="insufficient_adapter_geometry",
            detail="sigma_k_min or sigma_max_global is non-finite.",
            diagnostics={
                "sigma_k_min": sigma_k_min,
                "sigma_max_global": sigma_max_global,
            },
        )

    return sigma_k_min / sigma_max_global


def apply_sqrt_n_epoch_correction(
    eta_ceiling: float,
    n_batches_per_epoch: int,
) -> float:
    """Apply sqrt(N) Brownian scaling correction for epoch budget.

    Over N steps per epoch, accumulated displacement scales as sqrt(N) * eta * ||d||
    (random walk). Dividing the per-step ceiling by sqrt(N) keeps the epoch
    total within sigma_k_min.
    """
    if n_batches_per_epoch > 1:
        return eta_ceiling / math.sqrt(n_batches_per_epoch)
    return eta_ceiling


def compute_conformal_margin_rate(
    remaining_budget: float,
    d_norm: float,
) -> float:
    """Conformal margin rate: eta_margin = remaining_budget / ||g||.

    Ensures a single gradient step cannot exceed the remaining spectral
    budget, creating smooth deceleration as the adapter approaches capacity.

    Derivation (Weyl 1912, applied to remaining capacity):
        remaining = sigma_k - ||DeltaW||_2
        One gradient step at rate eta displaces at most eta * ||g||.
        To stay within remaining: eta * ||g|| <= remaining
        Therefore: eta <= remaining / ||g||

    Analogous to the conformal metric cancellation in Sahraee-Ardakan,
    Delbracio & Milanfar (2026, arXiv:2602.18428): the effective gain
    lambda(t) -> 0 as t -> 0 near the data manifold, converting an
    infinitely deep potential well into a stable attractor.

    Properties:
        - Always <= eta_weyl (tighter, since remaining <= sigma_k)
        - -> 0 as budget fills (conformal deceleration)
        - eta_margin * d_norm <= remaining (displacement bounded by margin)
    """
    if remaining_budget <= 0.0:
        return 0.0
    if d_norm <= 0.0:
        return float("inf")
    return remaining_budget / d_norm


def compute_per_step_rates(
    loss: float,
    d_norm: float,
    sigma_k_min: float,
    eta_ceiling: float,
    remaining_budget: float | None = None,
    f_star: float = 0.0,
) -> tuple[float, float, float, float, float | None]:
    """Compute SPS, Weyl, optional conformal margin, and combined eta_step.

    - SPS (Loizou et al. 2020): eta_sps = max(0, f(x) - f*) / ||g||^2
    - Weyl displacement bound: eta_weyl = sigma_k_min / ||g||
    - Conformal margin (Sahraee-Ardakan et al. 2026): eta_margin = remaining / ||g||
    - Combined: eta_step = min(eta_sps, eta_weyl, [eta_margin,] eta_ceiling)

    Args:
        f_star: Irreducible loss floor. For MSE distillation, derived from RMT:
            f_star = initial_loss × (1 - mean_sv_frac), where sv_frac is the
            mean signal_variance_fraction from RMT analysis of E_q. The MP
            noise floor bounds the loss achievable by any low-rank corrector.
            Default 0.0 preserves original SPS behavior.

    Returns (eta_step, eta_sps, eta_weyl, displacement, eta_margin).
    eta_margin is None when remaining_budget is None.
    """
    if d_norm > 0:
        eta_sps = max(0.0, loss - f_star) / (d_norm ** 2)
        eta_weyl = sigma_k_min / d_norm
    else:
        eta_sps = eta_ceiling
        eta_weyl = eta_ceiling

    candidates = [eta_sps, eta_weyl, eta_ceiling]
    eta_margin: float | None = None

    if remaining_budget is not None:
        eta_margin = compute_conformal_margin_rate(remaining_budget, d_norm)
        candidates.append(eta_margin)

    eta_step = min(candidates)
    displacement = eta_step * d_norm
    return eta_step, eta_sps, eta_weyl, displacement, eta_margin


def apply_validation_backoff(
    eta_ceiling: float,
    val_losses: list[float],
    *,
    adaptive_lr: bool = True,
) -> float:
    """Apply validation-guided ceiling backoff with sqrt(eps_f32) floor.

    When validation loss increases, reduce the ceiling by the ratio
    prev_loss / curr_loss, floored at sqrt(eps_f32) ~ 3.45e-4 to prevent
    underflow to zero.
    """
    if not adaptive_lr:
        return eta_ceiling
    if (len(val_losses) >= 2
            and val_losses[-1] > val_losses[-2]
            and val_losses[-1] > 0):
        backoff = max(val_losses[-2] / val_losses[-1], _SQRT_EPS_F32)
        return eta_ceiling * backoff
    return eta_ceiling


def verify_bounded_gain(
    max_eta_step: float,
    eta_ceiling: float,
) -> tuple[bool, float]:
    """Verify bounded-gain stability certificate.

    Bounded gain implies structural stability: if the effective gain
    (eta_step / eta_ceiling) stays <= 1.0 across all training steps,
    the trajectory is structurally stable.

    Cayley-Stiefel + MASS + scale clamping provides bounded gain by
    construction: eta_step = min(eta_sps, eta_weyl, eta_ceiling) <= eta_ceiling.
    This function verifies that construction was upheld.

    Reference: Sahraee-Ardakan, Delbracio & Milanfar (2026, arXiv:2602.18428),
    Theorem on velocity parameterization stability (nu(t) = 1 => bounded gain).

    Returns (is_bounded, gain_ratio).
    """
    if eta_ceiling <= 0.0:
        return max_eta_step <= 0.0, float("inf") if max_eta_step > 0 else 0.0
    gain_ratio = max_eta_step / eta_ceiling
    return gain_ratio <= 1.0, gain_ratio


def compute_reinforce_budget(
    sigma_k_min: float,
    update_norm: float | None,
    n_reinforce: int,
    check_interval: int,
) -> tuple[float, str]:
    """Compute REINFORCE displacement budget (Weyl remainder after CE).

    Total epoch displacement must stay within sigma_k_min. CE consumed
    update_norm of the budget. REINFORCE gets the remainder, distributed
    across sqrt(N_re) steps (Brownian scaling).

    Returns (target_step_norm, budget_source_label).
    """
    sqrt_n_re = math.sqrt(max(1, n_reinforce))

    if update_norm is not None and update_norm > 0:
        budget_remaining = max(0.0, sigma_k_min - update_norm)
        if budget_remaining <= 0.0:
            return 0.0, "budget_exhausted"
        return budget_remaining / sqrt_n_re, "weyl_remainder"
    else:
        n_total = max(1, check_interval) + max(1, n_reinforce)
        if sigma_k_min > 0:
            return sigma_k_min / math.sqrt(n_total), "sigma_k_min_shared"
        return 0.0, "budget_exhausted"
