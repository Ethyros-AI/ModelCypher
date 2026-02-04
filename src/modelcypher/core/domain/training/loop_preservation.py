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
    n_layers: int,
) -> int:
    """Find highway entry point from pre-computed intrinsic dimension profile.

    The highway is where the model compresses representations before
    the final reasoning/output layers. It's characterized by a dip
    in intrinsic dimension - information is concentrated.

    This is a pure function that operates on pre-computed intrinsic dimensions.
    The actual model inference to compute these dims lives in adapters.

    Args:
        layer_intrinsic_dims: List of intrinsic dimension per layer.
        n_layers: Total number of layers.

    Returns:
        Layer index where intrinsic dimension is minimum (highway entry).
    """
    if n_layers == 0 or not layer_intrinsic_dims:
        return 0

    # Skip early layers (embedding artifacts) and late layers (exit processing)
    # These fractions are derived from typical transformer architecture:
    # - First ~1/6 of layers: embedding projection, not semantic compression
    # - Last ~1/6 of layers: output projection, not highway
    skip_early = max(1, n_layers // 6)
    skip_late = max(1, n_layers // 6)
    search_start = skip_early
    search_end = n_layers - skip_late

    if search_start >= search_end:
        # Model too small, use middle layer
        return n_layers // 2

    # Find layer with minimum intrinsic dimension in the search range
    # (skip early/late layers which have artifacts)
    search_dims = [
        (i, d) for i, d in enumerate(layer_intrinsic_dims)
        if search_start <= i < search_end and d != float("inf")
    ]

    if not search_dims:
        # Fallback: search all layers
        search_dims = [
            (i, d) for i, d in enumerate(layer_intrinsic_dims)
            if d != float("inf")
        ]

    if not search_dims:
        # Ultimate fallback: use 1/3 of layers
        return n_layers // 3

    highway, min_dim = min(search_dims, key=lambda x: x[1])

    logger.info(
        "Highway detected at layer %d (ID=%.2f, search_range=[%d,%d), n_layers=%d)",
        highway,
        min_dim,
        search_start,
        search_end,
        n_layers,
    )

    return highway


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
) -> LoopPreservationConfig:
    """Derive complete loop preservation config from pre-computed geometry.

    This is a pure function. The actual model analysis to compute
    highway_layer and base_delta_entropy lives in adapters.

    Args:
        highway_layer: Pre-computed highway layer index.
        base_delta_entropy: Pre-computed entropy delta (H_exit - H_highway).
        sigma_max: Largest singular value from layer geometry.

    Returns:
        LoopPreservationConfig with all parameters derived from geometry.
    """
    # Lambda scale = 1 / σ_max (natural spectral scale)
    lambda_scale = 1.0 / max(sigma_max, 1e-8)

    config = LoopPreservationConfig(
        highway_layer=highway_layer,
        base_delta_entropy=base_delta_entropy,
        lambda_scale=lambda_scale,
    )

    logger.info("Loop preservation config: %s", config.to_dict())

    return config


def select_layers_to_sample(n_layers: int) -> list[int]:
    """Select which layers to sample for entropy trajectory.

    Default: sample at highway (1/3), middle (1/2), exit (last).

    Args:
        n_layers: Total number of layers.

    Returns:
        List of layer indices to sample.
    """
    if n_layers == 0:
        return []

    highway = n_layers // 3
    middle = n_layers // 2
    exit_layer = n_layers - 1

    return sorted(set([highway, middle, exit_layer]))


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

    # Compute singular values via Backend SVD
    try:
        _, S, _ = b.svd(hidden, compute_uv=False)
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

    if total_val < 1e-10:
        return 0.0

    # Normalize to probabilities
    p = S_sq / total_val
    b.eval(p)

    # Compute entropy: -Σ p_i log(p_i)
    # Only sum over significant probabilities
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
]
