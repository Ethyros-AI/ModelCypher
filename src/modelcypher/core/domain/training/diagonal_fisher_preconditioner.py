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

"""Diagonal Fisher preconditioner for Cayley-Stiefel training.

Maintains a per-parameter running EMA of squared gradients (v_t), which
converges to the diagonal of the empirical Fisher information matrix
(Hwang et al. 2024, "FAdam"). The preconditioned gradient g/√v̂ provides
curvature-aware steps within the MASS framework.

This is NOT momentum. Momentum carries state across retraction boundaries
and would violate the Cayley-Stiefel MASS step-size bound. Preconditioning
scales the gradient in the unconstrained parameter space BEFORE Cayley
retraction, which is valid because the retraction maps unconstrained
parameters to orthonormal factors regardless of the gradient's scale.

Mathematical background:
    Adam's update: θ -= η × m_t / (√v_t + ε)
    v_t = β₂ v_{t-1} + (1-β₂) g_t²  (EMA of squared gradients)

    Hwang et al. (2024) prove: v_t → diag(F_empirical) as t → ∞,
    where F is the Fisher information matrix. So Adam is implicitly
    doing natural gradient descent with a diagonal Fisher approximation.

    Our preconditioner extracts this curvature-awareness without momentum:
    d_t = g_t / (√v̂_t + ε)   (curvature-preconditioned gradient direction)

    MASS then bounds: η_step = min(η_sps, η_weyl, η_ceiling) using ||d_t||.

β₂ derivation:
    For EMA estimation error < √ε_f32 after N steps:
    (1-β₂)² × N > ε_f32 → β₂ < 1 - √(ε_f32/N).
    For N≥119 steps, β₂=0.999 satisfies this bound.

ε derivation:
    √ε_f32 ≈ 3.45e-4 (IEEE 754 float32 machine epsilon).
    Same numerical significance floor used throughout ModelCypher.

Zero framework dependencies. Operates on dicts of arrays via Backend protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# IEEE 754 float32 derived constants
_EPS_F32 = math.ldexp(1.0, -23)  # 2^-23, float32 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)  # ~3.45e-4, numerical significance floor

# β₂ for EMA: estimation error < √ε_f32 after N≥119 steps.
# Derivation: (1-β₂)² × N > ε_f32 → β₂ < 1 - √(ε_f32/119) ≈ 0.9997.
# β₂=0.999 is within this bound.
_DEFAULT_BETA2 = 0.999


@dataclass
class DiagonalFisherState:
    """Running state for diagonal Fisher EMA estimator.

    Attributes:
        v: Per-parameter running EMA of squared gradients.
            v[k] ≈ E[g_k²] → diag(F)_k as step_count → ∞.
        step_count: Number of update steps applied.
        beta2: EMA decay rate for squared gradients.
    """

    v: dict[str, "Array"] = field(default_factory=dict)
    step_count: int = 0
    beta2: float = _DEFAULT_BETA2


def init_fisher_state(
    trainable_params: dict[str, "Array"],
    backend: "Backend",
    beta2: float = _DEFAULT_BETA2,
) -> DiagonalFisherState:
    """Initialize diagonal Fisher state with zeros.

    Args:
        trainable_params: Dict of parameter name → array (flattened tree).
        backend: Compute backend.
        beta2: EMA decay rate. Default 0.999 (derived from IEEE 754).

    Returns:
        DiagonalFisherState with v initialized to zeros matching param shapes.
    """
    v: dict[str, "Array"] = {}
    for key, param in trainable_params.items():
        v[key] = backend.zeros_like(param)
        backend.eval(v[key])
    return DiagonalFisherState(v=v, step_count=0, beta2=beta2)


def update_fisher_state(
    state: DiagonalFisherState,
    grad_flat: dict[str, "Array"],
    backend: "Backend",
) -> DiagonalFisherState:
    """Update the diagonal Fisher EMA with new gradients.

    v[k] = β₂ × v[k] + (1-β₂) × g[k]²

    Args:
        state: Current Fisher state.
        grad_flat: Flattened gradient dict (same keys as state.v).
        backend: Compute backend.

    Returns:
        Updated DiagonalFisherState (mutates v in-place for efficiency,
        but returns the state for clarity).
    """
    b = backend
    beta2 = state.beta2
    one_minus_beta2 = 1.0 - beta2

    for key, g in grad_flat.items():
        if key not in state.v:
            # New parameter (shouldn't happen in normal flow)
            state.v[key] = b.zeros_like(g)
            b.eval(state.v[key])

        # v[k] = β₂ × v[k] + (1-β₂) × g[k]²
        state.v[key] = beta2 * state.v[key] + one_minus_beta2 * (g * g)
        b.eval(state.v[key])

    state.step_count += 1
    return state


def precondition_gradient(
    grad_flat: dict[str, "Array"],
    state: DiagonalFisherState,
    backend: "Backend",
) -> dict[str, "Array"]:
    """Precondition gradient by inverse sqrt of bias-corrected Fisher EMA.

    d[k] = g[k] / (√v̂[k] + ε)

    Where v̂ = v / (1 - β₂^t) is the bias-corrected second moment estimate
    (Kingma & Ba 2015). The bias correction compensates for the zero
    initialization of v, which causes underestimation in early steps.

    Args:
        grad_flat: Flattened gradient dict.
        state: Current Fisher state (must have step_count >= 1).
        backend: Compute backend.

    Returns:
        Dict of preconditioned gradient arrays (same keys as grad_flat).
    """
    b = backend

    # Bias correction factor: 1 / (1 - β₂^t)
    # At step 1: 1/(1-0.999) = 1000 (large correction)
    # At step 1000: 1/(1-0.999^1000) ≈ 1.58 (negligible)
    t = max(state.step_count, 1)
    bias_correction = 1.0 / (1.0 - state.beta2 ** t)

    preconditioned: dict[str, "Array"] = {}
    for key, g in grad_flat.items():
        v = state.v.get(key)
        if v is None:
            # No Fisher history for this parameter — pass through raw gradient
            preconditioned[key] = g
            continue

        # v̂ = v × bias_correction
        v_hat = v * bias_correction

        # d = g / (√v̂ + ε)
        # ε = √ε_f32 — same numerical floor used throughout ModelCypher
        d = g / (b.sqrt(v_hat) + _SQRT_EPS_F32)
        b.eval(d)
        preconditioned[key] = d

    return preconditioned


__all__ = [
    "DiagonalFisherState",
    "_DEFAULT_BETA2",
    "_SQRT_EPS_F32",
    "init_fisher_state",
    "precondition_gradient",
    "update_fisher_state",
]
