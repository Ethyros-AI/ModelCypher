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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
import math

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
    """Configuration for gradient boundary smoothing."""

    snr_discontinuity_threshold: float = 0.5
    """SNR difference threshold to flag a boundary as discontinuous."""

    base_smoothing_sigma: float = 1.0
    """Base sigma for Gaussian smoothing."""

    max_smoothing_multiplier: float = 3.0
    """Maximum multiplier for sigma at discontinuity points."""

    min_snr_for_smoothing: float = 0.1
    """Minimum SNR below which we increase smoothing."""

    use_hessian_penalty: bool = True
    """Whether to incorporate Hessian curvature in boundary detection."""

    hessian_threshold: float = 10.0
    """Maximum eigenvalue threshold for curvature penalty."""

    smoothing_window: int = 3
    """Number of layers to consider for smoothing window."""

    adaptive_smoothing: bool = True
    """Whether to adapt smoothing strength based on local SNR."""


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

    snr_by_layer: Dict[int, float]
    """Signal-to-noise ratio for each layer."""

    delta_snr_by_boundary: Dict[int, float]
    """SNR change at each layer boundary: delta[i] = SNR[i+1] - SNR[i]."""

    discontinuity_layers: List[int]
    """Layers where |delta_snr| > threshold."""

    recommended_smoothing: Dict[int, float]
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
        return sum((s - mean) ** 2 for s in self.snr_by_layer.values()) / len(
            self.snr_by_layer
        )

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

    def summary(self) -> Dict[str, any]:
        """Get summary dict for reporting."""
        return {
            "num_layers": len(self.snr_by_layer),
            "mean_snr": round(self.mean_snr, 4),
            "snr_variance": round(self.snr_variance, 4),
            "num_discontinuities": len(self.discontinuity_layers),
            "discontinuity_fraction": round(self.discontinuity_fraction, 4),
            "discontinuity_layers": self.discontinuity_layers,
            "max_delta_snr": round(
                max(abs(d) for d in self.delta_snr_by_boundary.values()), 4
            )
            if self.delta_snr_by_boundary
            else 0.0,
        }


# =============================================================================
# Boundary Analysis
# =============================================================================


def compute_layer_gradient_stats(
    per_sample_gradients: List[Dict[str, Array]],
    backend: Backend | None = None,
) -> Dict[int, LayerGradientStats]:
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
    layer_gradients: Dict[int, List[Array]] = {}

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
    stats: Dict[int, LayerGradientStats] = {}

    for layer, gradients in layer_gradients.items():
        if not gradients:
            continue

        # Stack and compute stats
        stacked = _backend.stack(gradients, axis=0)  # [num_samples, num_params]

        # Mean gradient across samples
        mean_grad = _backend.mean(stacked, axis=0)

        # Variance: E[(g - E[g])^2]
        centered = stacked - mean_grad
        variance = float(_backend.to_numpy(_backend.mean(_backend.sum(centered ** 2, axis=1))))

        # Mean norm: E[||g||]
        norms = _backend.sqrt(_backend.sum(stacked ** 2, axis=1))
        mean_norm = float(_backend.to_numpy(_backend.mean(norms)))

        # SNR: ||E[g]||^2 / variance
        mean_grad_norm_sq = float(_backend.to_numpy(_backend.sum(mean_grad ** 2)))
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
    per_sample_gradients: List[Dict[str, Array]],
    config: Optional[GradientBoundaryConfig] = None,
    backend: Backend | None = None,
) -> GradientBoundaryProfile:
    """Analyze gradient continuity across layer boundaries.

    Args:
        per_sample_gradients: List of gradient dicts per sample.
        config: Analysis configuration.
        backend: Compute backend (defaults to MLX).

    Returns:
        GradientBoundaryProfile with discontinuity detection.
    """
    cfg = config or GradientBoundaryConfig()

    # Compute per-layer stats
    layer_stats = compute_layer_gradient_stats(per_sample_gradients, backend=backend)

    if not layer_stats:
        return GradientBoundaryProfile(
            snr_by_layer={},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={},
            config=cfg,
        )

    # Extract SNR by layer
    snr_by_layer = {layer: stats.snr for layer, stats in layer_stats.items()}

    # Compute delta SNR at each boundary
    sorted_layers = sorted(snr_by_layer.keys())
    delta_snr: Dict[int, float] = {}
    discontinuities: List[int] = []

    for i in range(len(sorted_layers) - 1):
        layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]

        delta = snr_by_layer[next_layer] - snr_by_layer[layer]
        delta_snr[layer] = delta

        if abs(delta) > cfg.snr_discontinuity_threshold:
            discontinuities.append(layer)

    # Compute recommended smoothing
    recommended: Dict[int, float] = {}

    for layer in sorted_layers:
        snr = snr_by_layer[layer]

        # Base smoothing
        multiplier = 1.0

        if cfg.adaptive_smoothing:
            # Increase smoothing for low SNR layers
            if snr < cfg.min_snr_for_smoothing:
                # More smoothing for noisier layers
                multiplier = min(cfg.max_smoothing_multiplier, 1.0 / (snr + 0.1))
            elif layer in discontinuities:
                # Moderate increase at discontinuity boundaries
                multiplier = 1.5

        recommended[layer] = multiplier

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
    alpha_by_layer: Dict[int, float],
    boundary_profile: GradientBoundaryProfile,
    base_alpha: float = 0.5,
    min_alpha: float = 0.1,
    max_alpha: float = 0.95,
) -> Dict[int, float]:
    """Apply adaptive Gaussian smoothing to alpha profile.

    Increases smoothing at gradient discontinuity boundaries to
    prevent tearing effects in merged model.

    Args:
        alpha_by_layer: Raw alpha values per layer.
        boundary_profile: Gradient boundary analysis.
        base_alpha: Default alpha for missing layers.
        min_alpha: Minimum allowed alpha.
        max_alpha: Maximum allowed alpha.

    Returns:
        Smoothed alpha values per layer.
    """
    if not alpha_by_layer:
        return {}

    cfg = boundary_profile.config
    sorted_layers = sorted(alpha_by_layer.keys())
    smoothed: Dict[int, float] = {}

    for layer in sorted_layers:
        raw_alpha = alpha_by_layer.get(layer, base_alpha)

        # Get recommended smoothing multiplier for this layer
        multiplier = boundary_profile.recommended_smoothing.get(layer, 1.0)

        # Compute effective sigma
        sigma = cfg.base_smoothing_sigma * multiplier

        # Apply Gaussian-weighted smoothing
        if sigma > 0 and cfg.smoothing_window > 0:
            weighted_sum = 0.0
            weight_total = 0.0

            for offset in range(-cfg.smoothing_window, cfg.smoothing_window + 1):
                neighbor_layer = layer + offset
                if neighbor_layer in alpha_by_layer:
                    # Gaussian weight
                    weight = math.exp(-(offset ** 2) / (2 * sigma ** 2))
                    weighted_sum += weight * alpha_by_layer[neighbor_layer]
                    weight_total += weight

            if weight_total > 0:
                raw_alpha = weighted_sum / weight_total

        # Clamp to bounds
        smoothed[layer] = max(min_alpha, min(max_alpha, raw_alpha))

    return smoothed


def compute_gradient_adjusted_alpha(
    alpha_by_layer: Dict[int, float],
    boundary_profile: GradientBoundaryProfile,
    adjustment_strength: float = 0.3,
) -> Dict[int, float]:
    """Adjust alpha based on gradient quality at each layer.

    Layers with low SNR (noisy gradients) get alpha pushed toward
    conservative values (trust target more).

    Args:
        alpha_by_layer: Base alpha values.
        boundary_profile: Gradient analysis.
        adjustment_strength: How much to adjust based on SNR.

    Returns:
        Adjusted alpha values.
    """
    adjusted: Dict[int, float] = {}

    for layer, alpha in alpha_by_layer.items():
        snr = boundary_profile.snr_by_layer.get(layer, 1.0)

        # Map SNR to adjustment factor
        # High SNR (>2): trust source more (lower alpha toward 0)
        # Low SNR (<0.5): trust target more (higher alpha toward 1)
        if snr > 2.0:
            # Good gradients, can trust source
            factor = -adjustment_strength * min(1.0, (snr - 2.0) / 3.0)
        elif snr < 0.5:
            # Noisy gradients, be conservative
            factor = adjustment_strength * min(1.0, (0.5 - snr) / 0.5)
        else:
            factor = 0.0

        adjusted[layer] = alpha + factor

    return adjusted


# =============================================================================
# Unified Integration
# =============================================================================


def smooth_merge_boundaries(
    alpha_by_layer: Dict[int, float],
    per_sample_gradients: Optional[List[Dict[str, Array]]] = None,
    config: Optional[GradientBoundaryConfig] = None,
    base_alpha: float = 0.5,
    min_alpha: float = 0.1,
    max_alpha: float = 0.95,
    backend: Backend | None = None,
) -> Tuple[Dict[int, float], Optional[GradientBoundaryProfile]]:
    """Apply full gradient boundary smoothing to merge alpha profile.

    This is the main entry point for integrating gradient smoothing
    into the merge pipeline.

    Args:
        alpha_by_layer: Initial alpha values per layer.
        per_sample_gradients: Gradient samples for boundary analysis (optional).
        config: Smoothing configuration.
        base_alpha: Default alpha for missing layers.
        min_alpha: Minimum allowed alpha.
        max_alpha: Maximum allowed alpha.
        backend: Compute backend (defaults to MLX).

    Returns:
        Tuple of (smoothed_alpha_by_layer, boundary_profile).
        boundary_profile is None if no gradients provided.
    """
    cfg = config or GradientBoundaryConfig()

    if per_sample_gradients:
        # Full gradient-aware smoothing
        boundary_profile = compute_boundary_profile(per_sample_gradients, cfg, backend=backend)

        # First adjust alpha based on gradient quality
        adjusted = compute_gradient_adjusted_alpha(
            alpha_by_layer, boundary_profile, adjustment_strength=0.2
        )

        # Then apply adaptive Gaussian smoothing
        smoothed = apply_adaptive_smoothing(
            adjusted, boundary_profile, base_alpha, min_alpha, max_alpha
        )

        return smoothed, boundary_profile
    else:
        # Fallback: simple Gaussian smoothing without gradient info
        # Create a dummy profile with uniform smoothing
        dummy_profile = GradientBoundaryProfile(
            snr_by_layer={layer: 1.0 for layer in alpha_by_layer},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={layer: 1.0 for layer in alpha_by_layer},
            config=cfg,
        )

        smoothed = apply_adaptive_smoothing(
            alpha_by_layer, dummy_profile, base_alpha, min_alpha, max_alpha
        )

        return smoothed, None
