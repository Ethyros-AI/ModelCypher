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

"""
Alpha Smoothing for Model Merging.

Applies Gaussian smoothing across layer alphas to prevent sharp transitions
that cause "tearing" effects in merged models. Adjacent layers should have
similar blending ratios for smooth gradient flow.

Reference:
- Prevents layer-to-layer discontinuities in the merged representation

Mathematical Foundation
-----------------------
For each layer l, compute smoothed alpha:

    smoothed_alpha[l] = sum(w[i] * raw_alpha[l+i]) / sum(w[i])

where weights follow a Gaussian:
    w[i] = exp(-i^2 / (2 * sigma^2))

for i in [-window, +window].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class AlphaSmoothingConfig:
    """Configuration for Gaussian alpha smoothing.

    Use with_parameters() to create with explicit values.
    """

    # Number of layers on each side to consider
    smoothing_window: int = 2

    # Gaussian standard deviation (controls smoothing strength)
    # Higher sigma = more smoothing
    sigma: float = 1.0

    # Minimum alpha value (prevents complete source override)
    alpha_min: float = 0.1

    # Maximum alpha value (prevents complete target override)
    alpha_max: float = 0.9

    @classmethod
    def with_parameters(
        cls,
        *,
        smoothing_window: int = 2,
        sigma: float = 1.0,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
    ) -> "AlphaSmoothingConfig":
        """Create configuration with explicit parameters.

        Args:
            smoothing_window: Number of layers on each side (total = 2*window + 1).
            sigma: Gaussian standard deviation (higher = more smoothing).
            alpha_min: Minimum alpha value.
            alpha_max: Maximum alpha value.

        Returns:
            Configuration with specified parameters.
        """
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if alpha_min < 0 or alpha_min > 1:
            raise ValueError(f"alpha_min must be in [0, 1], got {alpha_min}")
        if alpha_max < 0 or alpha_max > 1:
            raise ValueError(f"alpha_max must be in [0, 1], got {alpha_max}")
        if alpha_min >= alpha_max:
            raise ValueError(f"alpha_min ({alpha_min}) must be < alpha_max ({alpha_max})")
        return cls(
            smoothing_window=smoothing_window,
            sigma=sigma,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )


def compute_gaussian_weights(window: int, sigma: float) -> list[float]:
    """
    Compute Gaussian weights for a symmetric window.

    Args:
        window: Number of layers on each side (total window = 2*window + 1)
        sigma: Standard deviation of the Gaussian

    Returns:
        List of weights for offsets [-window, ..., 0, ..., +window]
    """
    weights = []
    for offset in range(-window, window + 1):
        # Gaussian: exp(-x^2 / (2 * sigma^2))
        weight = math.exp(-(offset * offset) / (2 * sigma * sigma))
        weights.append(weight)
    return weights


def gaussian_smooth_alpha_profile(
    raw_alphas: dict[int, float],
    config: AlphaSmoothingConfig,
) -> dict[int, float]:
    """
    Apply Gaussian smoothing across layer alphas.

    Args:
        raw_alphas: Mapping from layer index to raw alpha value
        config: Smoothing configuration (use with_parameters() to create)

    Returns:
        Smoothed alpha profile with same layer indices
    """

    if not raw_alphas:
        return {}

    # Sort layers for consistent iteration
    sorted_layers = sorted(raw_alphas.keys())

    # Compute Gaussian weights
    weights = compute_gaussian_weights(config.smoothing_window, config.sigma)
    sum(weights)

    smoothed_alphas: dict[int, float] = {}

    for layer in sorted_layers:
        weighted_sum = 0.0
        total_weight = 0.0

        # Gather contributions from neighboring layers
        for offset_idx, offset in enumerate(
            range(-config.smoothing_window, config.smoothing_window + 1)
        ):
            neighbor_layer = layer + offset

            if neighbor_layer in raw_alphas:
                neighbor_alpha = raw_alphas[neighbor_layer]
                weight = weights[offset_idx]
                weighted_sum += neighbor_alpha * weight
                total_weight += weight

        # Compute smoothed alpha
        if total_weight > 0:
            smoothed = weighted_sum / total_weight
        else:
            smoothed = raw_alphas[layer]  # Fallback

        # Clamp to valid range
        smoothed = max(config.alpha_min, min(config.alpha_max, smoothed))
        smoothed_alphas[layer] = smoothed

    return smoothed_alphas


def smooth_alpha_vectors(
    raw_vectors: dict[int, "Array"],
    config: AlphaSmoothingConfig,
    backend: "Backend | None" = None,
) -> dict[int, "Array"]:
    """
    Apply Gaussian smoothing to per-dimension alpha vectors.

    Unlike gaussian_smooth_alpha_profile which smooths scalar alphas,
    this smooths full alpha vectors, maintaining per-dimension structure
    while reducing layer-to-layer discontinuities.

    Args:
        raw_vectors: Mapping from layer index to alpha vector [hidden_dim]
        config: Smoothing configuration (use with_parameters() to create)
        backend: Optional backend for array operations

    Returns:
        Smoothed alpha vectors
    """

    if not raw_vectors:
        return {}

    b = backend or get_default_backend()

    sorted_layers = sorted(raw_vectors.keys())
    first_vec = next(iter(raw_vectors.values()))
    hidden_dim = first_vec.shape[0] if hasattr(first_vec, "shape") else len(first_vec)

    weights = compute_gaussian_weights(config.smoothing_window, config.sigma)

    smoothed_vectors: dict[int, "Array"] = {}

    for layer in sorted_layers:
        # Initialize accumulator
        weighted_sum = b.zeros((hidden_dim,))
        total_weight = 0.0

        for offset_idx, offset in enumerate(
            range(-config.smoothing_window, config.smoothing_window + 1)
        ):
            neighbor_layer = layer + offset

            if neighbor_layer in raw_vectors:
                neighbor_vector = raw_vectors[neighbor_layer]
                weight = weights[offset_idx]
                weighted_sum = weighted_sum + neighbor_vector * weight
                total_weight += weight

        if total_weight > 0:
            smoothed = weighted_sum / total_weight
        else:
            smoothed = raw_vectors[layer]

        # Clamp to valid range
        smoothed = b.clip(smoothed, config.alpha_min, config.alpha_max)
        smoothed_vectors[layer] = smoothed

    return smoothed_vectors


def interpolate_missing_layers(
    alphas: dict[int, float],
    all_layers: list[int],
) -> dict[int, float]:
    """
    Interpolate alpha values for layers not in the original mapping.

    Uses linear interpolation between known layers.

    Args:
        alphas: Known alpha values for some layers
        all_layers: All layer indices that need alpha values

    Returns:
        Complete alpha mapping for all layers
    """
    if not alphas:
        return {layer: 0.5 for layer in all_layers}

    known_layers = sorted(alphas.keys())
    result: dict[int, float] = dict(alphas)

    for layer in all_layers:
        if layer in result:
            continue

        # Find bounding known layers
        lower = None
        upper = None

        for known in known_layers:
            if known < layer:
                lower = known
            elif known > layer and upper is None:
                upper = known
                break

        if lower is None and upper is None:
            result[layer] = 0.5  # Default
        elif lower is None:
            result[layer] = alphas[upper]  # type: ignore
        elif upper is None:
            result[layer] = alphas[lower]
        else:
            # Linear interpolation
            t = (layer - lower) / (upper - lower)
            result[layer] = alphas[lower] * (1 - t) + alphas[upper] * t

    return result
