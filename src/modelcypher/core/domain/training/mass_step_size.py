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

from modelcypher.core.domain.training.exceptions import TrainingDerivationError

# IEEE 754 float32 derived constants
_EPS_F32 = math.ldexp(1.0, -23)  # 2^-23, float32 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)  # ~3.45e-4, used as backoff floor


def derive_spectral_ceiling(
    *,
    sigma_k_min: float,
    sigma_max_global: float,
    lr_override: float | None = None,
) -> float:
    """Derive static learning rate ceiling from adapter geometry (Weyl 1912).

    eta_ceiling = sigma_k_min / sigma_max_global

    Where sigma_k_min is the minimum structural-rank singular value across
    all adapted layers (smallest spectral gap) and sigma_max_global is the
    largest singular value (maximum gradient amplification).

    After computing n_batches_per_epoch, the caller applies the sqrt(N) epoch
    budget correction via :func:`apply_sqrt_n_epoch_correction`.
    """
    if lr_override is not None:
        return float(lr_override)

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
    *,
    lr_override: float | None = None,
) -> float:
    """Apply sqrt(N) Brownian scaling correction for epoch budget.

    Over N steps per epoch, accumulated displacement scales as sqrt(N) * eta * ||d||
    (random walk). Dividing the per-step ceiling by sqrt(N) keeps the epoch
    total within sigma_k_min.
    """
    if lr_override is None and n_batches_per_epoch > 1:
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
    lr_override: float | None = None,
) -> float:
    """Apply validation-guided ceiling backoff with sqrt(eps_f32) floor.

    When validation loss increases, reduce the ceiling by the ratio
    prev_loss / curr_loss, floored at sqrt(eps_f32) ~ 3.45e-4 to prevent
    underflow to zero.
    """
    if not adaptive_lr or lr_override is not None:
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
