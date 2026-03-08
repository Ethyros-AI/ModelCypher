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

Maintains per-parameter running EMAs of gradients (m_t, first moment) and
squared gradients (v_t, second moment). v_t converges to the diagonal of
the empirical Fisher information matrix (Hwang et al. 2024, "FAdam").
The preconditioned update direction m̂/√v̂ provides curvature-aware steps
with direction smoothing within the MASS framework.

Mathematical background:
    Adam's update: θ -= η × m̂_t / (√v̂_t + ε)
    m_t = β₁ m_{t-1} + (1-β₁) g_t  (EMA of gradients — direction smoothing)
    v_t = β₂ v_{t-1} + (1-β₂) g_t²  (EMA of squared gradients — curvature)

    Hwang et al. (2024) prove: v_t → diag(F_empirical) as t → ∞,
    where F is the Fisher information matrix. So Adam is implicitly
    doing natural gradient descent with a diagonal Fisher approximation.

    d_t = m̂_t / (√v̂_t + ε)   (direction-smoothed, curvature-preconditioned)

    MASS then bounds: η_step = min(η_sps, η_weyl, η_ceiling) using ||d_t||.

First moment safety in Cayley-Stiefel:
    A_tilde, B_tilde are unconstrained Euclidean parameters. The Cayley
    transform maps them to semi-orthogonal A, B at each forward pass.
    Momentum in the Euclidean parametrization space is valid because the
    update θ -= η × m̂/(√v̂+ε) modifies A_tilde, B_tilde, and Cayley then
    maps to orthonormal A, B (spectral bound by construction). MASS bounds
    the step η using ||d_t|| = ||m̂/(√v̂+ε)||.

β₁ derivation:
    Effective window W = 1/(1-β₁). Must not exceed the direction
    decorrelation time — gradient directions from >1 epoch ago are stale.
    Two derived bounds (half-epoch ∩ precision ceiling):
      Bound 1 (direction decorrelation): β₁ = 1 - 2/T_epoch
      Bound 2 (EMA precision): β₁ < 1 - √(ε_f32/T_epoch)
    Derived dynamically from dataset size and batch size.

β₂ derivation:
    For EMA estimation error < √ε_f32 after N steps:
    (1-β₂)² × N > ε_f32 → β₂ < 1 - √(ε_f32/N).
    For N≥119 steps, β₂=0.999 satisfies this bound.

β₁ precision ceiling (same form as β₂):
    (1-β₁)² × T_epoch > ε_f32 → β₁ < 1 - √(ε_f32/T_epoch).
    For T=100: ceiling ≈ 0.99997. For T=10: ceiling ≈ 0.99989.
    Always far above the half-epoch bound, so serves only as a
    backstop against pathological epoch sizes.

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
        m: Per-parameter running EMA of gradients (first moment).
            m[k] ≈ E[g_k] — direction smoothing.
        step_count: Number of update steps applied.
        beta2: EMA decay rate for squared gradients.
        beta1: EMA decay rate for gradients (first moment).
            0 = no momentum (backward compatible). Derived from
            dataset size via ``derive_beta1()``.
    """

    v: dict[str, "Array"] = field(default_factory=dict)
    m: dict[str, "Array"] = field(default_factory=dict)
    step_count: int = 0
    beta2: float = _DEFAULT_BETA2
    beta1: float = 0.0


def derive_beta1(n_batches_per_epoch: int) -> float:
    """β₁ from two derived bounds: half-epoch window ∩ precision ceiling.

    Bound 1 (direction decorrelation): β₁ = 1 - 2/T_epoch
        Effective window W = 1/(1-β₁) = T/2. Gradient directions older
        than half an epoch are stale.

    Bound 2 (EMA precision): β₁ < 1 - √(ε_f32/T_epoch)
        Same form as β₂ derivation. Ensures EMA estimation error
        exceeds machine epsilon within one epoch.

    The precision ceiling is always far above the half-epoch bound for
    realistic training (e.g., T=100: ceiling≈0.99997 vs half-epoch=0.98),
    so the half-epoch derivation is the binding constraint. The precision
    ceiling serves as a backstop — no literal cap needed.

    Examples:
        50 batches/epoch → β₁ = 0.96
        100 batches/epoch → β₁ = 0.98
        200 batches/epoch → β₁ = 0.99
        10 batches/epoch → β₁ = 0.80
        ≤2 batches/epoch → β₁ = 0.0 (no smoothing)
    """
    if n_batches_per_epoch <= 2:
        return 0.0
    half_epoch = 1.0 - 2.0 / n_batches_per_epoch
    precision_ceiling = 1.0 - math.sqrt(_EPS_F32 / n_batches_per_epoch)
    return max(0.0, min(half_epoch, precision_ceiling))


def init_fisher_state(
    trainable_params: dict[str, "Array"],
    backend: "Backend",
    beta2: float = _DEFAULT_BETA2,
    n_batches_per_epoch: int = 0,
) -> DiagonalFisherState:
    """Initialize diagonal Fisher state with zeros.

    Args:
        trainable_params: Dict of parameter name → array (flattened tree).
        backend: Compute backend.
        beta2: EMA decay rate for second moment. Default 0.999 (IEEE 754).
        n_batches_per_epoch: Batches per epoch for β₁ derivation.
            0 = no first moment (backward compatible).

    Returns:
        DiagonalFisherState with v and m initialized to zeros.
    """
    beta1 = derive_beta1(n_batches_per_epoch) if n_batches_per_epoch > 0 else 0.0
    v: dict[str, "Array"] = {}
    m: dict[str, "Array"] = {}
    for key, param in trainable_params.items():
        v[key] = backend.zeros_like(param)
        m[key] = backend.zeros_like(param)
        backend.eval(v[key], m[key])
    return DiagonalFisherState(v=v, m=m, step_count=0, beta2=beta2, beta1=beta1)


def update_fisher_state(
    state: DiagonalFisherState,
    grad_flat: dict[str, "Array"],
    backend: "Backend",
) -> DiagonalFisherState:
    """Update the diagonal Fisher EMAs with new gradients.

    v[k] = β₂ × v[k] + (1-β₂) × g[k]²   (second moment / curvature)
    m[k] = β₁ × m[k] + (1-β₁) × g[k]     (first moment / direction)

    Args:
        state: Current Fisher state.
        grad_flat: Flattened gradient dict (same keys as state.v).
        backend: Compute backend.

    Returns:
        Updated DiagonalFisherState (mutates v, m in-place for efficiency,
        but returns the state for clarity).
    """
    b = backend
    beta2 = state.beta2
    one_minus_beta2 = 1.0 - beta2
    beta1 = state.beta1
    one_minus_beta1 = 1.0 - beta1
    use_first_moment = beta1 > 0.0

    for key, g in grad_flat.items():
        if key not in state.v:
            # New parameter (shouldn't happen in normal flow)
            state.v[key] = b.zeros_like(g)
            state.m[key] = b.zeros_like(g)
            b.eval(state.v[key], state.m[key])

        # v[k] = β₂ × v[k] + (1-β₂) × g[k]²
        state.v[key] = beta2 * state.v[key] + one_minus_beta2 * (g * g)

        # m[k] = β₁ × m[k] + (1-β₁) × g[k]
        if use_first_moment:
            state.m[key] = beta1 * state.m[key] + one_minus_beta1 * g
            b.eval(state.v[key], state.m[key])
        else:
            b.eval(state.v[key])

    state.step_count += 1
    return state


def precondition_gradient(
    grad_flat: dict[str, "Array"],
    state: DiagonalFisherState,
    backend: "Backend",
) -> dict[str, "Array"]:
    """Precondition gradient using bias-corrected first and second moments.

    When β₁ > 0: d[k] = m̂[k] / (√v̂[k] + ε)  (Adam-equivalent direction)
    When β₁ = 0: d[k] = g[k] / (√v̂[k] + ε)  (backward compatible)

    Where m̂ = m / (1 - β₁^t) and v̂ = v / (1 - β₂^t) are bias-corrected
    moment estimates (Kingma & Ba 2015).

    Args:
        grad_flat: Flattened gradient dict.
        state: Current Fisher state (must have step_count >= 1).
        backend: Compute backend.

    Returns:
        Dict of preconditioned gradient arrays (same keys as grad_flat).
    """
    b = backend

    t = max(state.step_count, 1)

    # Second moment bias correction: 1 / (1 - β₂^t)
    bc2 = 1.0 / (1.0 - state.beta2 ** t)

    # First moment bias correction: 1 / (1 - β₁^t)
    use_first_moment = state.beta1 > 0.0
    bc1 = 1.0 / (1.0 - state.beta1 ** t) if use_first_moment else 1.0

    preconditioned: dict[str, "Array"] = {}
    for key, g in grad_flat.items():
        v = state.v.get(key)
        if v is None:
            # No Fisher history for this parameter — pass through raw gradient
            preconditioned[key] = g
            continue

        # v̂ = v × bc2
        v_hat = v * bc2

        # Numerator: m̂ (direction-smoothed) or raw g (no smoothing)
        if use_first_moment:
            m = state.m.get(key)
            numerator = m * bc1 if m is not None else g
        else:
            numerator = g

        # d = numerator / (√v̂ + ε)
        d = numerator / (b.sqrt(v_hat) + _SQRT_EPS_F32)
        b.eval(d)
        preconditioned[key] = d

    return preconditioned


__all__ = [
    "DiagonalFisherState",
    "_DEFAULT_BETA2",
    "_SQRT_EPS_F32",
    "derive_beta1",
    "init_fisher_state",
    "precondition_gradient",
    "update_fisher_state",
]
