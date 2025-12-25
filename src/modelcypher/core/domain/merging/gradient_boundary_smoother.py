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
Gradient Boundary Smoothing for Model Merging.

Detects and smooths gradient discontinuities at merge boundaries to prevent
"tearing" effects in the merged model's loss landscape.

Integrates with:
- GradientSmoothnessEstimator: Per-layer SNR computation
- HessianEstimator: Curvature analysis
- LinguisticThermodynamics: Entropy trajectory detection
- UnifiedManifoldMerger: Alpha smoothing integration
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array, Backend

if TYPE_CHECKING:
    pass  # Any type-only imports go here

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GradientBoundaryConfig:
    """Configuration for gradient boundary smoothing.

    Attributes
    ----------
    epsilon : float
        Small constant for numerical stability.
    use_hessian_penalty : bool
        Whether to apply Hessian-based curvature penalty.
    """

    epsilon: float = 1e-10
    use_hessian_penalty: bool = True


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LayerGradientStats:
    """Gradient statistics for a single layer."""

    layer: int
    snr: float
    """Signal-to-noise ratio of gradients."""

    variance: float
    """Gradient variance across samples."""

    mean_norm: float
    """Mean gradient norm."""

    sample_count: int
    """Number of samples used to compute stats."""

    @property
    def is_noisy(self) -> bool:
        """Whether this layer has noisy gradients (SNR < 1)."""
        return self.snr < 1.0

    @property
    def is_stable(self) -> bool:
        """Whether this layer has stable gradients (SNR > 2)."""
        return self.snr > 2.0


@dataclass
class GradientBoundaryProfile:
    """Gradient continuity analysis across layers.

    Identifies discontinuity points where gradient quality changes
    sharply between adjacent layers.
    """

    snr_by_layer: dict[int, float]
    """Signal-to-noise ratio for each layer."""

    delta_snr_by_boundary: dict[int, float]
    """SNR change at each layer boundary: delta[i] = SNR[i+1] - SNR[i]."""

    discontinuity_layers: list[int]
    """Layers where |delta_snr| > threshold."""

    recommended_smoothing: dict[int, float]
    """Recommended sigma multiplier for each layer."""

    config: GradientBoundaryConfig
    """Configuration used for this analysis."""

    @property
    def mean_snr(self) -> float:
        """Mean SNR across all layers."""
        if not self.snr_by_layer:
            return 0.0
        return sum(self.snr_by_layer.values()) / len(self.snr_by_layer)

    @property
    def snr_variance(self) -> float:
        """Variance of SNR across layers."""
        if len(self.snr_by_layer) < 2:
            return 0.0
        mean = self.mean_snr
        return sum((s - mean) ** 2 for s in self.snr_by_layer.values()) / len(self.snr_by_layer)

    @property
    def has_discontinuities(self) -> bool:
        """Whether any discontinuities were detected."""
        return len(self.discontinuity_layers) > 0

    @property
    def discontinuity_fraction(self) -> float:
        """Fraction of layer boundaries that are discontinuous."""
        if not self.delta_snr_by_boundary:
            return 0.0
        return len(self.discontinuity_layers) / len(self.delta_snr_by_boundary)

    def summary(self) -> dict[str, any]:
        """Get summary dict for reporting."""
        return {
            "num_layers": len(self.snr_by_layer),
            "mean_snr": round(self.mean_snr, 4),
            "snr_variance": round(self.snr_variance, 4),
            "num_discontinuities": len(self.discontinuity_layers),
            "discontinuity_fraction": round(self.discontinuity_fraction, 4),
            "discontinuity_layers": self.discontinuity_layers,
            "max_delta_snr": round(max(abs(d) for d in self.delta_snr_by_boundary.values()), 4)
            if self.delta_snr_by_boundary
            else 0.0,
        }


# =============================================================================
# Boundary Analysis
# =============================================================================


def compute_layer_gradient_stats(
    per_sample_gradients: list[dict[str, Array]],
    backend: Backend | None = None,
) -> dict[int, LayerGradientStats]:
    """Compute per-layer gradient statistics from sample gradients.

    Args:
        per_sample_gradients: List of dicts mapping param_name to gradient array,
            one dict per sample.
        backend: Compute backend (defaults to MLX).

    Returns:
        Dict mapping layer index to LayerGradientStats.
    """
    import re

    _backend = backend or get_default_backend()

    # Group gradients by layer
    layer_gradients: dict[int, list[Array]] = {}

    for sample_grads in per_sample_gradients:
        for param_name, grad in sample_grads.items():
            # Extract layer index from param name
            match = re.search(r"layers?\.(\d+)", param_name)
            if not match:
                continue
            layer = int(match.group(1))

            if layer not in layer_gradients:
                layer_gradients[layer] = []
            layer_gradients[layer].append(_backend.reshape(grad, (-1,)))

    # Compute statistics per layer
    stats: dict[int, LayerGradientStats] = {}

    for layer, gradients in layer_gradients.items():
        if not gradients:
            continue

        # Stack and compute stats
        stacked = _backend.stack(gradients, axis=0)  # [num_samples, num_params]

        # Mean gradient across samples
        mean_grad = _backend.mean(stacked, axis=0)

        # Variance: E[(g - E[g])^2]
        centered = stacked - mean_grad
        variance = float(_backend.to_numpy(_backend.mean(_backend.sum(centered**2, axis=1))))

        # Mean norm: E[||g||]
        norms = _backend.sqrt(_backend.sum(stacked**2, axis=1))
        mean_norm = float(_backend.to_numpy(_backend.mean(norms)))

        # SNR: ||E[g]||^2 / variance
        mean_grad_norm_sq = float(_backend.to_numpy(_backend.sum(mean_grad**2)))
        snr = mean_grad_norm_sq / (variance + 1e-10)

        stats[layer] = LayerGradientStats(
            layer=layer,
            snr=snr,
            variance=variance,
            mean_norm=mean_norm,
            sample_count=len(gradients),
        )

    return stats


def compute_boundary_profile(
    per_sample_gradients: list[dict[str, Array]],
    config: GradientBoundaryConfig | None = None,
    backend: Backend | None = None,
) -> GradientBoundaryProfile:
    """Analyze gradient continuity across layer boundaries.

    Parameters
    ----------
    per_sample_gradients : list of dict
        List of dicts mapping param_name to gradient array, one dict per sample.
    config : GradientBoundaryConfig, optional
        Configuration for boundary analysis.
    backend : Backend, optional
        Compute backend (defaults to MLX).

    Returns
    -------
    GradientBoundaryProfile
        Profile containing SNR analysis and discontinuity detection results.
    """
    cfg = config or GradientBoundaryConfig()

    layer_stats = compute_layer_gradient_stats(per_sample_gradients, backend=backend)

    if not layer_stats:
        return GradientBoundaryProfile(
            snr_by_layer={},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={},
            config=cfg,
        )

    snr_by_layer = {layer: stats.snr for layer, stats in layer_stats.items()}
    sorted_layers = sorted(snr_by_layer.keys())

    # Compute delta SNR at each boundary
    delta_snr: dict[int, float] = {}
    for i in range(len(sorted_layers) - 1):
        layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]
        delta_snr[layer] = snr_by_layer[next_layer] - snr_by_layer[layer]

    # Discontinuity threshold: median absolute deviation of deltas
    # Points beyond 2 * MAD are statistical outliers
    if delta_snr:
        abs_deltas = [abs(d) for d in delta_snr.values()]
        median_delta = sorted(abs_deltas)[len(abs_deltas) // 2]
        mad = sorted([abs(d - median_delta) for d in abs_deltas])[len(abs_deltas) // 2]
        threshold = median_delta + 2 * mad + cfg.epsilon
    else:
        threshold = cfg.epsilon

    discontinuities = [layer for layer, delta in delta_snr.items() if abs(delta) > threshold]

    # Smoothing: inversely proportional to SNR (geometry determines amount)
    median_snr = sorted(snr_by_layer.values())[len(snr_by_layer) // 2] if snr_by_layer else 1.0
    recommended: dict[int, float] = {}
    for layer in sorted_layers:
        snr = snr_by_layer[layer]
        # Smoothing = median_snr / snr (noisier layers get more smoothing)
        recommended[layer] = median_snr / max(snr, cfg.epsilon)

    return GradientBoundaryProfile(
        snr_by_layer=snr_by_layer,
        delta_snr_by_boundary=delta_snr,
        discontinuity_layers=discontinuities,
        recommended_smoothing=recommended,
        config=cfg,
    )


# =============================================================================
# Alpha Smoothing Integration
# =============================================================================


def apply_adaptive_smoothing(
    alpha_by_layer: dict[int, float],
    boundary_profile: GradientBoundaryProfile,
) -> dict[int, float]:
    """Apply Gaussian smoothing with sigma derived from SNR.

    Parameters
    ----------
    alpha_by_layer : dict
        Mapping from layer index to alpha value.
    boundary_profile : GradientBoundaryProfile
        Profile containing recommended smoothing parameters.

    Returns
    -------
    dict
        Smoothed alpha values by layer.
    """
    if not alpha_by_layer:
        return {}

    sorted_layers = sorted(alpha_by_layer.keys())
    smoothed: dict[int, float] = {}

    for layer in sorted_layers:
        raw_alpha = alpha_by_layer.get(layer, 0.5)
        sigma = boundary_profile.recommended_smoothing.get(layer, 1.0)

        # Gaussian smoothing over all layers (not arbitrary window)
        weighted_sum = 0.0
        weight_total = 0.0

        for other_layer in sorted_layers:
            offset = abs(other_layer - layer)
            weight = math.exp(-(offset**2) / (2 * sigma**2 + 1e-10))
            weighted_sum += weight * alpha_by_layer[other_layer]
            weight_total += weight

        if weight_total > 0:
            smoothed[layer] = weighted_sum / weight_total
        else:
            smoothed[layer] = raw_alpha

    return smoothed


def compute_gradient_adjusted_alpha(
    alpha_by_layer: dict[int, float],
    boundary_profile: GradientBoundaryProfile,
) -> dict[int, float]:
    """Adjust alpha based on SNR relative to median.

    Parameters
    ----------
    alpha_by_layer : dict
        Mapping from layer index to alpha value.
    boundary_profile : GradientBoundaryProfile
        Profile containing SNR statistics.

    Returns
    -------
    dict
        Adjusted alpha values by layer.

    Notes
    -----
    Alpha adjustment = (median_snr - layer_snr) / (median_snr + layer_snr)
    This maps SNR to [-1, 1] adjustment centered on median.
    High SNR -> negative adjustment (trust source)
    Low SNR -> positive adjustment (trust target)
    """
    snr_values = list(boundary_profile.snr_by_layer.values())
    if not snr_values:
        return dict(alpha_by_layer)

    median_snr = sorted(snr_values)[len(snr_values) // 2]
    adjusted: dict[int, float] = {}

    for layer, alpha in alpha_by_layer.items():
        snr = boundary_profile.snr_by_layer.get(layer, median_snr)
        # Normalized difference from median: range [-1, 1]
        adjustment = (median_snr - snr) / (median_snr + snr + 1e-10)
        adjusted[layer] = max(0.0, min(1.0, alpha + adjustment))

    return adjusted


# =============================================================================
# Unified Integration
# =============================================================================


def smooth_merge_boundaries(
    alpha_by_layer: dict[int, float],
    per_sample_gradients: list[dict[str, Array]] | None = None,
    config: GradientBoundaryConfig | None = None,
    backend: Backend | None = None,
) -> tuple[dict[int, float], GradientBoundaryProfile | None]:
    """Apply gradient boundary smoothing.

    Parameters
    ----------
    alpha_by_layer : dict
        Mapping from layer index to alpha value.
    per_sample_gradients : list of dict, optional
        List of gradient dicts for boundary analysis.
    config : GradientBoundaryConfig, optional
        Configuration for smoothing.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    dict
        Smoothed alpha values by layer.
    GradientBoundaryProfile or None
        Boundary profile if gradients provided, None otherwise.
    """
    cfg = config or GradientBoundaryConfig()

    if per_sample_gradients:
        boundary_profile = compute_boundary_profile(per_sample_gradients, cfg, backend=backend)
        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, boundary_profile)
        smoothed = apply_adaptive_smoothing(adjusted, boundary_profile)
        return smoothed, boundary_profile
    else:
        # No gradients: return input unchanged
        return dict(alpha_by_layer), None
