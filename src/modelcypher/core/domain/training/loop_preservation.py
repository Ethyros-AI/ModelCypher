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

"""Loop preservation loss for reasoning training.

Penalizes spectral entropy collapse during training. The key insight from
our research is that:
- Δβ₁ > 0 (loops grow) ↔ spectral entropy increases toward exit
- Δβ₁ < 0 (loops collapse) ↔ spectral entropy drops before exit

Since β₁ (ripser) is O(n³) and too slow for training, we use spectral
entropy trajectory as a differentiable proxy for topological complexity.

All parameters are derived from model geometry - no arbitrary constants:
- highway_layer: argmin(intrinsic_dimension) across layers
- lambda_scale: 1 / σ_max (inverse of largest singular value)
- base_delta_entropy: H_exit - H_highway (base model's trajectory)

The loss penalizes when training makes the entropy trajectory WORSE
than the base model. If base model has ΔH = +0.5 (healthy: entropy grows)
and adapter produces ΔH = -0.2 (collapse), the loss is:
    L = λ * max(0, 0.5 - (-0.2)) = λ * 0.7

This preserves the topological loop structure that correlates with
reasoning capability.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific model inference code lives in adapters/training/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)
from modelcypher.core.domain.geometry.stable_extremum import (
    StableExtremumResult,
    find_stable_inflection,
    find_stable_minimum,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LoopPreservationConfig:
    """All parameters derived from base model geometry.

    Attributes:
        highway_layer: Layer index where intrinsic dimension is minimum.
            This is the "highway entry" - the compression bottleneck.
        base_delta_entropy: Base model's H_exit - H_highway.
            Positive = healthy (entropy grows after highway).
            Negative = collapsing (entropy drops after highway).
        lambda_scale: 1 / σ_max of base model.
            Natural scale factor for the loss weight.
    """

    highway_layer: int
    base_delta_entropy: float
    lambda_scale: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/saving."""
        return {
            "highway_layer": self.highway_layer,
            "base_delta_entropy": self.base_delta_entropy,
            "lambda_scale": self.lambda_scale,
        }


def find_highway_layer_from_intrinsic_dims(
    layer_intrinsic_dims: list[float],
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[int | None, "StableExtremumResult | None"]:
    """Find highway entry via bootstrap-stable ID minimum.

    The highway is where the model compresses representations before
    the final reasoning/output layers. It's characterized by a dip
    in intrinsic dimension — information is concentrated.

    Tests whether the argmin of per-layer intrinsic dimension is stable
    under bootstrap resampling — no skip ranges, no fallback constants.

    If the minimum is not stable (appears in <50% of resamples), returns
    None — the caller should mark the result INCONCLUSIVE rather than
    silently using a fallback.

    Args:
        layer_intrinsic_dims: Per-layer intrinsic dimension measurements.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (highway_layer, stability_result).
        - highway_layer: Index of the stable minimum, or None if unstable.
        - stability_result: Full StableExtremumResult, or None if too few layers.
    """
    # Filter out inf values — these are measurement failures, not layer properties
    valid_dims = [d if d != float("inf") else float("nan") for d in layer_intrinsic_dims]
    # Replace NaN with max finite value + 1 (so they never win argmin)
    finite_vals = [d for d in valid_dims if d == d]  # filter NaN
    if not finite_vals:
        return None, None

    fill_value = max(finite_vals) + 1.0
    clean_dims = [d if d == d else fill_value for d in valid_dims]

    if len(clean_dims) < 2:
        return 0 if clean_dims else None, None

    result = find_stable_minimum(clean_dims, n_bootstrap=n_bootstrap, seed=seed)

    logger.info(
        "Highway geometric: layer=%d, ID=%.2f, frequency=%.2f, stable=%s, "
        "ci_range=%s",
        result.index,
        result.value,
        result.frequency,
        result.is_stable,
        result.ci_range,
    )

    if not result.is_stable:
        logger.info("Highway minimum unstable — INCONCLUSIVE")
        return None, result

    return result.index, result


def compute_entropy_delta(
    highway_entropies: list[float],
    exit_entropies: list[float],
) -> float:
    """Compute entropy trajectory from pre-computed entropies.

    This is a pure function. The actual model inference to compute
    these entropies lives in adapters.

    Args:
        highway_entropies: Spectral entropies at highway layer per sample.
        exit_entropies: Spectral entropies at exit layer per sample.

    Returns:
        ΔH = mean(H_exit) - mean(H_highway).
        Positive means entropy grows after highway (healthy reasoning).
        Negative means entropy collapses after highway (degraded).
    """
    if not highway_entropies or not exit_entropies:
        return 0.0

    mean_highway = sum(highway_entropies) / len(highway_entropies)
    mean_exit = sum(exit_entropies) / len(exit_entropies)

    delta = mean_exit - mean_highway

    logger.info(
        "Entropy trajectory: H_highway=%.4f, H_exit=%.4f, ΔH=%+.4f",
        mean_highway,
        mean_exit,
        delta,
    )

    return delta


def loop_preservation_loss(
    current_trajectory: dict[int, float],
    config: LoopPreservationConfig,
) -> tuple[float, float]:
    """Compute loop preservation loss.

    Loss = λ * max(0, base_delta - current_delta)

    This penalizes when the current model's entropy trajectory is
    WORSE than the base model's. "Worse" means:
    - Base has positive ΔH (healthy) but current has lower or negative ΔH
    - The gap between base and current represents loop collapse

    Args:
        current_trajectory: Dict mapping layer_idx -> spectral_entropy.
            Must contain at least highway_layer and the exit layer.
        config: Loop preservation configuration from base model.

    Returns:
        Tuple of (loss_value, current_delta_entropy).
        loss_value is already scaled by λ.
    """
    if not current_trajectory:
        return 0.0, 0.0

    # Get current delta entropy
    layers = sorted(current_trajectory.keys())
    if len(layers) < 2:
        return 0.0, 0.0

    highway_layer = config.highway_layer
    exit_layer = max(layers)

    # Find closest layer to highway if exact not available
    if highway_layer not in current_trajectory:
        highway_layer = min(layers, key=lambda x: abs(x - config.highway_layer))

    h_highway = current_trajectory.get(highway_layer, 0.0)
    h_exit = current_trajectory.get(exit_layer, 0.0)

    current_delta = h_exit - h_highway

    # Loss: penalize when current is worse than base
    # If base_delta = 0.5 and current_delta = -0.2, loss = 0.7
    # If base_delta = 0.5 and current_delta = 0.6, loss = 0 (better)
    gap = config.base_delta_entropy - current_delta
    loss = config.lambda_scale * max(0.0, gap)

    return loss, current_delta


def derive_loop_config_from_geometry(
    highway_layer: int,
    base_delta_entropy: float,
    sigma_max: float,
    weight_shape: tuple[int, int] = (1, 1),
) -> LoopPreservationConfig:
    """Derive complete loop preservation config from pre-computed geometry.

    This is a pure function. The actual model analysis to compute
    highway_layer and base_delta_entropy lives in adapters.

    Args:
        highway_layer: Pre-computed highway layer index.
        base_delta_entropy: Pre-computed entropy delta (H_exit - H_highway).
        sigma_max: Largest singular value from layer geometry.
        weight_shape: (m, n) dimensions of the weight matrix whose SVD
            produced sigma_max.  Used to derive the SVD noise floor
            (Demmel & Kahan 1990: absolute error ≤ sqrt(max(m,n)) * eps).

    Returns:
        LoopPreservationConfig with all parameters derived from geometry.
    """
    import math as _math

    _eps = _math.ldexp(1.0, -23)  # IEEE 754 float32 machine epsilon
    _max_dim = max(weight_shape[0], weight_shape[1])
    # SVD noise floor: singular values below this are indistinguishable
    # from zero (Demmel & Kahan 1990, Golub & Van Loan §8.6.1).
    _svd_floor = _math.sqrt(_max_dim) * _eps
    # Lambda scale = 1 / σ_max (natural spectral scale)
    lambda_scale = 1.0 / max(sigma_max, _svd_floor)

    config = LoopPreservationConfig(
        highway_layer=highway_layer,
        base_delta_entropy=base_delta_entropy,
        lambda_scale=lambda_scale,
    )

    logger.info("Loop preservation config: %s", config.to_dict())

    return config


def select_layers_to_sample(
    layer_intrinsic_dims: list[float],
    entropy_trajectory: list[float] | None = None,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> list[int]:
    """Select layers from measured trajectory events.

    Select layers from measured trajectory events:
    - Highway: bootstrap-stable ID minimum
    - Middle: entropy re-expansion inflection (derivative sign change)
    - Exit: last layer (always)

    When no stable highway or inflection is found, returns only the
    exit layer. The caller handles partial sampling.

    Args:
        layer_intrinsic_dims: Per-layer intrinsic dimension.
        entropy_trajectory: Per-layer spectral entropy (optional).
        n_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        List of layer indices to sample.
    """
    n = len(layer_intrinsic_dims)
    if n == 0:
        return []

    exit_layer = n - 1
    layers: set[int] = {exit_layer}

    # Highway from stable ID minimum
    highway, _ = find_highway_layer_from_intrinsic_dims(
        layer_intrinsic_dims, n_bootstrap=n_bootstrap, seed=seed
    )
    if highway is not None:
        layers.add(highway)

    # Middle from entropy inflection (if trajectory provided)
    if entropy_trajectory and len(entropy_trajectory) >= 3:
        inflection = find_stable_inflection(
            entropy_trajectory, n_bootstrap=n_bootstrap, seed=seed
        )
        if inflection.is_stable and inflection.index != exit_layer:
            layers.add(inflection.index)

    return sorted(layers)


def compute_spectral_entropy(
    hidden: "Array",
    backend: "Backend",
) -> float:
    """Compute spectral entropy from hidden states using Backend protocol.

    Spectral entropy = Shannon entropy of normalized singular values.
    H = -Σ p_i log(p_i) where p_i = σ_i² / Σσ²

    Higher entropy = more uniform singular values = richer representation.
    Lower entropy = concentrated singular values = compressed representation.

    Args:
        hidden: Hidden states tensor [batch, seq, hidden_dim] or [n, hidden_dim].
        backend: Backend for tensor operations.

    Returns:
        Spectral entropy value.
    """
    b = backend

    # Flatten to [n_samples, hidden_dim]
    if len(hidden.shape) == 3:
        hidden = b.reshape(hidden, (-1, int(hidden.shape[-1])))

    b.eval(hidden)

    n_samples, hidden_dim = int(hidden.shape[0]), int(hidden.shape[1])
    if n_samples < 2 or hidden_dim < 2:
        return 0.0

    try:
        S = b.svd(hidden, compute_uv=False)
        b.eval(S)
    except Exception:
        return 0.0

    n_svs = int(S.shape[0])
    if n_svs == 0:
        return 0.0

    # Compute spectral entropy
    S_sq = S * S
    total = b.sum(S_sq)
    b.eval(total)
    total_val = float(b.to_scalar(total))

    # Energy of a zero matrix perturbed by float32 roundoff:
    # each of n*d entries has magnitude ≤ eps, so total squared energy
    # ≤ n*d * eps^2 (Higham 2002, §3.1).  Below this, the singular values
    # are pure roundoff and entropy is undefined.
    import math as _math

    _eps = _math.ldexp(1.0, -23)  # IEEE 754 float32
    _energy_floor = float(n_svs) * _eps * _eps
    if total_val < _energy_floor:
        return 0.0

    # Normalize to probabilities
    p = S_sq / total_val
    b.eval(p)

    # Compute entropy: -Σ p_i log(p_i)
    # Only sum over significant energy fractions
    sqrt_eps = division_epsilon(b, S)
    entropy = 0.0

    # Iterate through singular values to compute entropy
    for i in range(n_svs):
        p_i = float(b.to_scalar(p[i]))
        if p_i > sqrt_eps:
            entropy -= p_i * log_scalar(p_i, b)

    return entropy


__all__ = [
    "LoopPreservationConfig",
    "find_highway_layer_from_intrinsic_dims",
    "compute_entropy_delta",
    "loop_preservation_loss",
    "compute_spectral_entropy",
    "derive_loop_config_from_geometry",
    "select_layers_to_sample",
    "StableExtremumResult",
]
