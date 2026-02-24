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


def compute_per_step_rates(
    loss: float,
    d_norm: float,
    sigma_k_min: float,
    eta_ceiling: float,
) -> tuple[float, float, float, float]:
    """Compute SPS, Weyl, and combined eta_step with displacement.

    - SPS (Loizou et al. 2020): eta_sps = f(x) / ||g||^2, f* = 0
    - Weyl displacement bound: eta_weyl = sigma_k_min / ||g||
    - Combined: eta_step = min(eta_sps, eta_weyl, eta_ceiling)

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
