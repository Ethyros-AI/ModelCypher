"""
Unified Manifold Merger: Geometry-Aware Model Merging with Adaptive Blending.

Ported from the reference Swift implementation (core features only).

Key Features:
1. Adaptive Alpha Profile - Per-layer blending weights based on confidence
2. Gaussian Smoothing - Prevents "tearing" from sharp alpha transitions
3. Spectral Penalty - Reduces trust in poorly-conditioned rotations
4. Dimension Blending - Per-dimension weights from intersection maps

Mathematical Foundation:
- confidence(l) = intersection map confidence for layer l
- α(l) = blending factor = f(confidence(l), procrustes_error(l))
- W_merged = α * W_target + (1-α) * project(W_source, Ω_out, Ω_in)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from .rotational_merger import MergeOptions, RotationalModelMerger

logger = logging.getLogger("modelcypher.merging.unified_manifold_merger")


# =============================================================================
# Configuration
# =============================================================================


class BlendMode(str, Enum):
    """Module-level blending behavior."""
    SOFT = "soft"             # Standard alpha blending
    HARD_SWAP = "hard_swap"   # Binary choice (source or target)
    SKIP = "skip"             # Don't merge this module


@dataclass
class ModuleBlendPolicy:
    """
    Policy that maps module kinds to blending strategies.
    
    Allows different treatment for attention vs MLP modules based on
    empirical observations about which benefit from blending vs swapping.
    """
    soft_blend_kinds: set = field(default_factory=lambda: {
        "q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"
    })
    hard_swap_kinds: set = field(default_factory=lambda: {"v_proj"})
    skip_kinds: set = field(default_factory=lambda: {"o_proj"})
    soft_blend_max_error: float = 1.3
    hard_swap_advantage_threshold: float = 0.05


@dataclass
class UnifiedMergeConfig:
    """Configuration for unified manifold merging."""
    
    # --- Core Parameters ---
    alignment_rank: int = 32
    base_alpha: float = 0.5  # Fallback when confidence unavailable
    
    # --- Confidence Thresholds ---
    # Minimum confidence to apply permutation alignment
    permutation_confidence_threshold: float = 0.6
    # Minimum confidence to apply full Procrustes rotation
    rotation_confidence_threshold: float = 0.4
    
    # --- Adaptive Alpha Smoothing ---
    use_adaptive_alpha_smoothing: bool = True
    adaptive_alpha_smoothing_window: int = 2
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    
    # --- Spectral Penalty ---
    # Strength of spectral penalty applied to alpha (0 = disabled, 1 = full)
    spectral_penalty_strength: float = 0.5
    spectral_epsilon: float = 1e-6
    
    # --- Dimension Blending ---
    # Use per-dimension blending based on correlation strength
    use_dimension_blending: bool = True
    dimension_blend_threshold: float = 0.3
    
    # --- Module Policy ---
    use_module_blend_policy: bool = True
    module_blend_policy: ModuleBlendPolicy = field(default_factory=ModuleBlendPolicy)
    
    @classmethod
    def conservative(cls) -> "UnifiedMergeConfig":
        """Preset for conservative merging (prefer target in uncertain regions)."""
        return cls(
            base_alpha=0.7,
            permutation_confidence_threshold=0.7,
            rotation_confidence_threshold=0.5,
            spectral_penalty_strength=0.5,
        )
    
    @classmethod
    def aggressive(cls) -> "UnifiedMergeConfig":
        """Preset for aggressive merging (trust source more)."""
        return cls(
            alignment_rank=48,
            base_alpha=0.3,
            permutation_confidence_threshold=0.4,
            rotation_confidence_threshold=0.3,
            spectral_penalty_strength=0.3,
        )


# =============================================================================
# Layer Alpha Profile
# =============================================================================


@dataclass
class LayerAlphaProfile:
    """
    Per-layer alpha profile with smoothing to prevent "tearing".
    
    Instead of computing alpha independently for each layer, this profile:
    1. Computes raw alpha based on confidence and Procrustes error
    2. Applies Gaussian smoothing across adjacent layers
    3. Clamps values to prevent extreme blending
    """
    alpha_by_layer: Dict[int, float]
    smoothing_window: int
    base_alpha: float
    used_procrustes_error: bool
    
    @property
    def mean_alpha(self) -> float:
        """Average alpha across all layers."""
        values = list(self.alpha_by_layer.values())
        if not values:
            return self.base_alpha
        return sum(values) / len(values)
    
    @property
    def alpha_variance(self) -> float:
        """Alpha variance (smoothing effectiveness indicator)."""
        values = list(self.alpha_by_layer.values())
        if len(values) <= 1:
            return 0.0
        mean = self.mean_alpha
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def alpha(self, layer: int) -> float:
        """Gets alpha for a specific layer, falling back to base alpha."""
        return self.alpha_by_layer.get(layer, self.base_alpha)


def compute_adaptive_alpha_profile(
    layer_confidences: Dict[int, float],
    base_alpha: float = 0.5,
    smoothing_window: int = 2,
    procrustes_error_by_layer: Optional[Dict[int, float]] = None,
    min_alpha: float = 0.1,
    max_alpha: float = 0.95,
) -> LayerAlphaProfile:
    """
    Computes an adaptive alpha profile with Gaussian smoothing.
    
    Alpha Formula (per layer):
        raw_alpha = 1.0 - (confidence * 0.7)  # High confidence → trust source
        
        If Procrustes error available:
            error_adjustment = clamp(procrustes_error * 0.5, 0, 0.3)
            raw_alpha += error_adjustment  # Higher error → trust target more
    
    Smoothing: Gaussian weighted average across adjacent layers prevents
    sharp alpha transitions that cause tearing artifacts.
    
    Args:
        layer_confidences: Per-layer confidence values (from intersection map)
        base_alpha: Fallback alpha for layers without data
        smoothing_window: Number of layers on each side for smoothing
        procrustes_error_by_layer: Optional per-layer Procrustes error
        min_alpha: Minimum allowed alpha
        max_alpha: Maximum allowed alpha
    
    Returns:
        Smoothed per-layer alpha profile
    """
    if not layer_confidences:
        return LayerAlphaProfile(
            alpha_by_layer={},
            smoothing_window=smoothing_window,
            base_alpha=base_alpha,
            used_procrustes_error=procrustes_error_by_layer is not None,
        )
    
    # Step 1: Compute raw alpha for each layer
    raw_alphas: Dict[int, float] = {}
    
    for layer, confidence in layer_confidences.items():
        # Base formula: high confidence → lower alpha → trust source more
        alpha = 1.0 - (confidence * 0.7)
        
        # Incorporate Procrustes error if available
        if procrustes_error_by_layer and layer in procrustes_error_by_layer:
            error = procrustes_error_by_layer[layer]
            # Higher error → increase alpha → trust target more
            error_adjustment = min(0.3, error * 0.5)
            alpha += error_adjustment
        
        # Clamp before smoothing
        raw_alphas[layer] = max(min_alpha, min(max_alpha, alpha))
    
    if smoothing_window <= 0:
        return LayerAlphaProfile(
            alpha_by_layer=raw_alphas,
            smoothing_window=smoothing_window,
            base_alpha=base_alpha,
            used_procrustes_error=procrustes_error_by_layer is not None,
        )
    
    # Step 2: Apply Gaussian smoothing
    sorted_layers = sorted(raw_alphas.keys())
    smoothed_alphas: Dict[int, float] = {}
    
    # Precompute Gaussian weights
    sigma = smoothing_window / 2.0
    gaussian_weights = []
    for offset in range(-smoothing_window, smoothing_window + 1):
        weight = math.exp(-(offset * offset) / (2 * sigma * sigma))
        gaussian_weights.append(weight)
    weight_sum = sum(gaussian_weights)
    gaussian_weights = [w / weight_sum for w in gaussian_weights]
    
    for layer in sorted_layers:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for offset_idx, offset in enumerate(range(-smoothing_window, smoothing_window + 1)):
            neighbor_layer = layer + offset
            if neighbor_layer in raw_alphas:
                weight = gaussian_weights[offset_idx]
                weighted_sum += raw_alphas[neighbor_layer] * weight
                total_weight += weight
        
        # Normalize and clamp
        fallback_alpha = raw_alphas.get(layer, base_alpha)
        smoothed_alpha = weighted_sum / total_weight if total_weight > 0 else fallback_alpha
        smoothed_alphas[layer] = max(min_alpha, min(max_alpha, smoothed_alpha))
    
    logger.debug(
        f"Alpha profile: {len(smoothed_alphas)} layers, "
        f"mean={sum(smoothed_alphas.values())/len(smoothed_alphas):.2f}, "
        f"window={smoothing_window}"
    )
    
    return LayerAlphaProfile(
        alpha_by_layer=smoothed_alphas,
        smoothing_window=smoothing_window,
        base_alpha=base_alpha,
        used_procrustes_error=procrustes_error_by_layer is not None,
    )


# =============================================================================
# Spectral Penalty
# =============================================================================


def compute_spectral_penalty(
    weight: mx.array,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes spectral penalty based on condition number.
    
    High condition number indicates near-singular matrices that are
    unreliable for rotation. Returns value in [0, 1] where:
    - 0 = well-conditioned (low condition number, trustworthy)
    - 1 = ill-conditioned (high condition number, untrustworthy)
    
    Args:
        weight: Weight matrix to analyze
        epsilon: Numerical stability epsilon
    
    Returns:
        Penalty value in [0, 1]
    """
    if weight.ndim != 2:
        return 0.0
    
    try:
        # Compute singular values
        s = mx.linalg.svd(weight.astype(mx.float32), compute_uv=False)
        mx.eval(s)
        s_list = s.tolist()
        
        if not s_list:
            return 0.0
        
        s_max = max(s_list)
        s_min = min(abs(x) for x in s_list if abs(x) > epsilon)
        
        if s_min < epsilon:
            return 1.0  # Ill-conditioned
        
        condition_number = s_max / s_min
        
        # Map condition number to penalty:
        # - < 10: penalty ≈ 0
        # - 10-100: penalty scales linearly
        # - > 100: penalty ≈ 1
        normalized = (math.log10(max(condition_number, 1.0)) - 1.0) / 1.0
        return max(0.0, min(1.0, normalized))
        
    except Exception:
        return 0.5  # Default to moderate penalty on error


def apply_spectral_penalty_to_alpha(
    alpha: float,
    source_weight: mx.array,
    target_weight: mx.array,
    strength: float = 0.5,
) -> float:
    """
    Adjusts alpha based on spectral properties of weights.
    
    If source has high condition number (ill-conditioned), increase alpha
    to trust target more. If target has high condition number, decrease
    alpha to trust source more.
    
    Args:
        alpha: Base alpha value
        source_weight: Source model weight
        target_weight: Target model weight
        strength: Strength of penalty effect (0 = disabled, 1 = full)
    
    Returns:
        Adjusted alpha value
    """
    if strength <= 0:
        return alpha
    
    source_penalty = compute_spectral_penalty(source_weight)
    target_penalty = compute_spectral_penalty(target_weight)
    
    # If source is ill-conditioned, push toward target (increase alpha)
    # If target is ill-conditioned, push toward source (decrease alpha)
    penalty_diff = source_penalty - target_penalty
    adjustment = penalty_diff * strength * 0.3
    
    adjusted_alpha = max(0.1, min(0.95, alpha + adjustment))
    return adjusted_alpha


# =============================================================================
# Dimension Blending Weights
# =============================================================================


@dataclass
class DimensionBlendingWeights:
    """
    Per-dimension blending weights based on intersection correlations.
    
    Instead of a single scalar alpha, this uses per-dimension weights
    derived from how well dimensions correlate between source and target.
    """
    weights: mx.array  # [hidden_dim] or [out_dim, in_dim]
    threshold: float
    mean_weight: float
    covered_fraction: float  # Fraction of dimensions with high correlation


def compute_dimension_blending_weights(
    source_activations: mx.array,
    target_activations: mx.array,
    threshold: float = 0.3,
    fallback_weight: float = 0.5,
) -> DimensionBlendingWeights:
    """
    Computes per-dimension blending weights from activation correlations.
    
    High correlation → trust source more (lower weight)
    Low correlation → trust target more (higher weight)
    
    Args:
        source_activations: Source model activations [samples, hidden_dim]
        target_activations: Target model activations [samples, hidden_dim]
        threshold: Correlation threshold for "high confidence"
        fallback_weight: Weight for dimensions below threshold
    
    Returns:
        DimensionBlendingWeights with per-dimension values
    """
    if source_activations.shape != target_activations.shape:
        raise ValueError("Activation shapes must match")
    
    hidden_dim = source_activations.shape[-1]
    
    # Normalize
    source_norm = source_activations - mx.mean(source_activations, axis=0, keepdims=True)
    target_norm = target_activations - mx.mean(target_activations, axis=0, keepdims=True)
    
    source_std = mx.sqrt(mx.sum(source_norm ** 2, axis=0) + 1e-8)
    target_std = mx.sqrt(mx.sum(target_norm ** 2, axis=0) + 1e-8)
    
    # Per-dimension correlation
    correlations = mx.sum(source_norm * target_norm, axis=0) / (source_std * target_std)
    mx.eval(correlations)
    
    corr_list = correlations.tolist()
    
    # Convert correlation to weight:
    # High correlation (>threshold) → trust source → lower alpha
    # Low correlation (<threshold) → trust target → higher alpha
    weights = []
    high_conf_count = 0
    
    for corr in corr_list:
        abs_corr = abs(corr)
        if abs_corr >= threshold:
            # High correlation: weight = 1 - abs_corr (low = trust source)
            weight = max(0.1, 1.0 - abs_corr)
            high_conf_count += 1
        else:
            weight = fallback_weight
        weights.append(weight)
    
    weights_array = mx.array(weights).astype(mx.float32)
    mean_weight = sum(weights) / len(weights)
    covered_fraction = high_conf_count / hidden_dim
    
    return DimensionBlendingWeights(
        weights=weights_array,
        threshold=threshold,
        mean_weight=mean_weight,
        covered_fraction=covered_fraction,
    )


# =============================================================================
# Unified Manifold Merger
# =============================================================================


@dataclass
class UnifiedMergeResult:
    """Result of unified manifold merging."""
    merged_weights: Dict[str, mx.array]
    alpha_profile: LayerAlphaProfile
    layers_merged: int
    mean_alpha: float
    spectral_penalty_applied: bool
    dimension_blending_applied: bool


class UnifiedManifoldMerger:
    """
    Unified Manifold Merger combining geometric alignment methods.
    
    Pipeline Stages:
    1. PROBE - Run semantic primes through models → Intersection map
    2. PERMUTE - Use intersection to guide Re-Basin on MLP neurons
    3. ROTATE - Apply Procrustes on strongly-correlated dimensions
    4. BLEND - Use intersection confidence as per-layer adaptive alpha
    5. PROPAGATE - Carry alignment forward geometrically (zipper)
    
    This implementation provides the core blending logic. The full
    probing and intersection mapping should be done externally.
    """
    
    def __init__(self, config: UnifiedMergeConfig = None):
        self.config = config or UnifiedMergeConfig()
        self._rotational_merger = RotationalModelMerger(
            MergeOptions(
                alignment_rank=self.config.alignment_rank,
                alpha=self.config.base_alpha,
            )
        )
    
    def merge_with_confidence(
        self,
        source_weights: Dict[str, mx.array],
        target_weights: Dict[str, mx.array],
        layer_confidences: Dict[int, float],
        procrustes_errors: Optional[Dict[int, float]] = None,
    ) -> UnifiedMergeResult:
        """
        Merge weights using confidence-adaptive blending.
        
        Args:
            source_weights: Source model weights by key
            target_weights: Target model weights by key
            layer_confidences: Per-layer confidence from intersection map
            procrustes_errors: Optional per-layer Procrustes errors
        
        Returns:
            UnifiedMergeResult with merged weights and diagnostics
        """
        # Compute adaptive alpha profile
        alpha_profile = compute_adaptive_alpha_profile(
            layer_confidences=layer_confidences,
            base_alpha=self.config.base_alpha,
            smoothing_window=self.config.adaptive_alpha_smoothing_window,
            procrustes_error_by_layer=procrustes_errors,
            min_alpha=self.config.min_alpha,
            max_alpha=self.config.max_alpha,
        )
        
        merged_weights: Dict[str, mx.array] = {}
        spectral_applied = False
        dimension_blending_applied = False
        
        for key in source_weights:
            if key not in target_weights:
                continue
            
            source_w = source_weights[key]
            target_w = target_weights[key]
            
            # Extract layer index from key
            layer = self._extract_layer_index(key)
            alpha = alpha_profile.alpha(layer)
            
            # Apply spectral penalty if enabled
            if self.config.spectral_penalty_strength > 0 and source_w.ndim == 2:
                alpha = apply_spectral_penalty_to_alpha(
                    alpha=alpha,
                    source_weight=source_w,
                    target_weight=target_w,
                    strength=self.config.spectral_penalty_strength,
                )
                spectral_applied = True
            
            # Simple linear blend (dimension blending would be applied per-dim)
            merged = alpha * target_w + (1.0 - alpha) * source_w
            mx.eval(merged)
            merged_weights[key] = merged
        
        return UnifiedMergeResult(
            merged_weights=merged_weights,
            alpha_profile=alpha_profile,
            layers_merged=len(merged_weights),
            mean_alpha=alpha_profile.mean_alpha,
            spectral_penalty_applied=spectral_applied,
            dimension_blending_applied=dimension_blending_applied,
        )
    
    def _extract_layer_index(self, key: str) -> int:
        """Extract layer index from weight key like 'model.layers.5.mlp.up_proj.weight'."""
        import re
        match = re.search(r'layers\.(\d+)', key)
        if match:
            return int(match.group(1))
        return 0
    
    def compute_blend_mode(self, key: str, procrustes_error: float) -> BlendMode:
        """
        Determines blend mode for a module based on policy.
        
        Args:
            key: Weight key
            procrustes_error: Procrustes alignment error
        
        Returns:
            BlendMode for this module
        """
        if not self.config.use_module_blend_policy:
            return BlendMode.SOFT
        
        policy = self.config.module_blend_policy
        
        # Check module kind
        for kind in policy.skip_kinds:
            if kind in key:
                return BlendMode.SKIP
        
        for kind in policy.hard_swap_kinds:
            if kind in key:
                # Hard swap if error exceeds threshold
                if procrustes_error > policy.soft_blend_max_error:
                    return BlendMode.HARD_SWAP
        
        return BlendMode.SOFT
