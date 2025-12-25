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
Fisher Information-Weighted Blending for Model Merging.

Uses Fisher information to weight parameter importance during merging.
Higher Fisher weight = more confident dimension = trust that model more.

Fisher information approximates how much the loss changes when a parameter changes:
    F_i = E[(d log p(x|θ) / dθ_i)^2]

In practice, we estimate this from gradient statistics during fine-tuning:
    F_i ≈ 1 / (Var(gradients_i) + epsilon)

Reference:
- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
- Matena & Raffel (2022) "Merging Models with Fisher-Weighted Averaging"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class FisherEstimationMethod(str, Enum):
    """Method for estimating Fisher information."""

    GRADIENT_VARIANCE = "gradient_variance"  # Inverse of gradient variance
    DIAGONAL_HESSIAN = "diagonal_hessian"  # Diagonal Hessian approximation
    EMPIRICAL = "empirical"  # Empirical Fisher from gradients
    IDENTITY = "identity"  # Uniform weights (baseline)


class FisherNormalization(str, Enum):
    """How to normalize Fisher weights."""

    NONE = "none"  # Raw Fisher values
    LAYER = "layer"  # Normalize within each layer
    GLOBAL = "global"  # Normalize across all parameters
    SOFTMAX = "softmax"  # Apply softmax normalization


@dataclass(frozen=True)
class FisherBlendingConfig:
    """Configuration for Fisher-weighted blending."""

    estimation_method: FisherEstimationMethod = FisherEstimationMethod.GRADIENT_VARIANCE
    normalization: FisherNormalization = FisherNormalization.LAYER
    epsilon: float = 1e-6  # Numerical stability
    strength: float = 0.5  # How much to weight by Fisher (0=ignore, 1=full)
    temperature: float = 1.0  # Temperature for softmax normalization
    min_fisher: float = 1e-8  # Minimum Fisher value (prevents division by zero)
    max_fisher: float = 1e8  # Maximum Fisher value (prevents overflow)
    source_bias: float = 0.0  # Bias toward source model (-1 to 1)
    clip_alpha: bool = True  # Whether to clip resulting alpha to [0, 1]


@dataclass
class FisherWeights:
    """Fisher information weights for a model's parameters."""

    weights_by_key: "dict[str, Array]"
    estimation_method: FisherEstimationMethod
    total_parameters: int
    mean_fisher: float
    std_fisher: float

    @classmethod
    def from_gradients(
        cls,
        gradient_history: "dict[str, list[Array]]",
        config: FisherBlendingConfig | None = None,
        backend: "Backend | None" = None,
    ) -> "FisherWeights":
        """
        Estimate Fisher weights from gradient history.

        Args:
            gradient_history: Dict mapping parameter keys to lists of gradients
                              from multiple training steps
            config: Optional configuration
            backend: Optional backend

        Returns:
            FisherWeights with estimated Fisher information
        """
        if config is None:
            config = FisherBlendingConfig()

        b = backend or get_default_backend()
        weights_by_key: "dict[str, Array]" = {}
        all_fisher_values: list[float] = []
        total_params = 0

        for key, gradients in gradient_history.items():
            if not gradients:
                continue

            # Stack gradients and compute variance
            stacked = b.stack(gradients, axis=0)
            variance = b.var(stacked, axis=0)
            b.eval(variance)

            # Fisher = 1 / (variance + epsilon)
            fisher = 1.0 / (variance + config.epsilon)

            # Clip to prevent numerical issues
            fisher = b.clip(fisher, config.min_fisher, config.max_fisher)
            b.eval(fisher)

            weights_by_key[key] = fisher
            total_params += fisher.size

            # Track statistics
            mean_arr = b.mean(fisher)
            b.eval(mean_arr)
            mean_val = float(b.to_numpy(mean_arr).item())
            all_fisher_values.append(mean_val)

        # Compute global statistics
        if all_fisher_values:
            mean_fisher = sum(all_fisher_values) / len(all_fisher_values)
            variance = sum((x - mean_fisher) ** 2 for x in all_fisher_values) / len(
                all_fisher_values
            )
            std_fisher = math.sqrt(variance)
        else:
            mean_fisher = 0.0
            std_fisher = 0.0

        return cls(
            weights_by_key=weights_by_key,
            estimation_method=config.estimation_method,
            total_parameters=total_params,
            mean_fisher=mean_fisher,
            std_fisher=std_fisher,
        )

    @classmethod
    def uniform(
        cls,
        keys: list[str],
        shapes: dict[str, tuple[int, ...]],
        backend: "Backend | None" = None,
    ) -> "FisherWeights":
        """Create uniform Fisher weights (baseline)."""
        b = backend or get_default_backend()
        weights_by_key = {key: b.ones(shapes[key]) for key in keys if key in shapes}
        total_params = sum(w.size for w in weights_by_key.values())

        return cls(
            weights_by_key=weights_by_key,
            estimation_method=FisherEstimationMethod.IDENTITY,
            total_parameters=total_params,
            mean_fisher=1.0,
            std_fisher=0.0,
        )


@dataclass
class FisherBlendingResult:
    """Result of Fisher-weighted blending."""

    merged_weights: "dict[str, Array]"
    effective_alphas: "dict[str, Array]"  # Per-parameter effective blending weights
    mean_effective_alpha: float
    fisher_applied: bool
    parameters_blended: int


def normalize_fisher_weights(
    fisher: "Array",
    method: FisherNormalization,
    temperature: float = 1.0,
    global_stats: tuple[float, float] | None = None,
    backend: "Backend | None" = None,
) -> "Array":
    """
    Normalize Fisher weights according to the specified method.

    Args:
        fisher: Raw Fisher weights
        method: Normalization method
        temperature: Temperature for softmax
        global_stats: (mean, std) for global normalization
        backend: Optional backend

    Returns:
        Normalized Fisher weights
    """
    b = backend or get_default_backend()

    if method == FisherNormalization.NONE:
        return fisher

    elif method == FisherNormalization.LAYER:
        # Normalize to [0, 1] within the layer
        f_min = b.min(fisher)
        f_max = b.max(fisher)
        b.eval(f_min, f_max)

        f_min_val = float(b.to_numpy(f_min).item())
        f_max_val = float(b.to_numpy(f_max).item())
        if (f_max_val - f_min_val) < 1e-10:
            return b.ones_like(fisher)

        return (fisher - f_min) / (f_max - f_min + 1e-10)

    elif method == FisherNormalization.GLOBAL:
        if global_stats is None:
            mean_arr = b.mean(fisher)
            std_arr = b.std(fisher)
            b.eval(mean_arr, std_arr)
            mean = float(b.to_numpy(mean_arr).item())
            std = float(b.to_numpy(std_arr).item())
        else:
            mean, std = global_stats

        if std < 1e-10:
            return b.ones_like(fisher)

        # Z-score normalization, then sigmoid to [0, 1]
        z = (fisher - mean) / (std + 1e-10)
        return 1.0 / (1.0 + b.exp(-z))

    elif method == FisherNormalization.SOFTMAX:
        # Apply softmax with temperature
        fisher_flat = b.reshape(fisher, (-1,))
        scaled = fisher_flat / temperature
        # Numerical stability: subtract max
        max_scaled = b.max(scaled)
        scaled = scaled - max_scaled
        exp_scaled = b.exp(scaled)
        softmax = exp_scaled / (b.sum(exp_scaled) + 1e-10)
        # Scale up to preserve relative magnitudes
        return b.reshape(softmax * fisher_flat.size, fisher.shape)

    return fisher


def apply_fisher_blending(
    source_weight: "Array",
    target_weight: "Array",
    base_alpha: float,
    source_fisher: "Array | None" = None,
    target_fisher: "Array | None" = None,
    config: FisherBlendingConfig | None = None,
    backend: "Backend | None" = None,
) -> "tuple[Array, Array]":
    """
    Apply Fisher-weighted blending between source and target weights.

    The effective alpha for each parameter dimension is:
        alpha_eff = base_alpha * target_importance / (source_importance + target_importance)

    Where importance is derived from Fisher information.

    Args:
        source_weight: Source model weight tensor
        target_weight: Target model weight tensor
        base_alpha: Base blending factor (0 = all source, 1 = all target)
        source_fisher: Fisher weights for source (optional)
        target_fisher: Fisher weights for target (optional)
        config: Blending configuration
        backend: Optional backend

    Returns:
        Tuple of (merged_weight, effective_alpha_per_dim)
    """
    if config is None:
        config = FisherBlendingConfig()

    b = backend or get_default_backend()

    # If no Fisher info, fall back to standard blending
    if source_fisher is None and target_fisher is None:
        merged = base_alpha * target_weight + (1.0 - base_alpha) * source_weight
        alpha_effective = b.full(source_weight.shape, base_alpha)
        return merged, alpha_effective

    # Use uniform weights if only one is provided
    if source_fisher is None:
        source_fisher = b.ones_like(source_weight)
    if target_fisher is None:
        target_fisher = b.ones_like(target_weight)

    # Ensure shapes match
    if source_fisher.shape != source_weight.shape:
        source_fisher = b.broadcast_to(source_fisher, source_weight.shape)
    if target_fisher.shape != target_weight.shape:
        target_fisher = b.broadcast_to(target_fisher, target_weight.shape)

    # Normalize Fisher weights
    source_fisher_norm = normalize_fisher_weights(
        source_fisher, config.normalization, config.temperature, backend=b
    )
    target_fisher_norm = normalize_fisher_weights(
        target_fisher, config.normalization, config.temperature, backend=b
    )

    # Compute importance-weighted alpha
    # Higher source Fisher = trust source more = lower effective alpha
    # Higher target Fisher = trust target more = higher effective alpha
    total_fisher = source_fisher_norm + target_fisher_norm + config.epsilon

    # Relative importance of target
    target_importance = target_fisher_norm / total_fisher

    # Apply source bias if configured
    if config.source_bias != 0:
        target_importance = target_importance - config.source_bias * 0.5
        target_importance = b.clip(target_importance, 0.0, 1.0)

    # Compute effective alpha: blend of base_alpha and Fisher-derived importance
    # strength=0 -> use base_alpha only
    # strength=1 -> use pure Fisher weighting
    alpha_effective = (
        (1.0 - config.strength) * base_alpha
        + config.strength
        * target_importance
        * base_alpha
        * 2.0  # Scale by 2 since importance is 0-1
    )

    if config.clip_alpha:
        alpha_effective = b.clip(alpha_effective, 0.0, 1.0)

    b.eval(alpha_effective)

    # Apply per-dimension blending
    merged = alpha_effective * target_weight + (1.0 - alpha_effective) * source_weight
    b.eval(merged)

    return merged, alpha_effective


def fisher_weighted_merge(
    source_weights: "dict[str, Array]",
    target_weights: "dict[str, Array]",
    source_fisher: FisherWeights,
    target_fisher: FisherWeights,
    base_alpha: float = 0.5,
    config: FisherBlendingConfig | None = None,
    backend: "Backend | None" = None,
) -> FisherBlendingResult:
    """
    Merge two models using Fisher-weighted averaging.

    Args:
        source_weights: Source model weights by key
        target_weights: Target model weights by key
        source_fisher: Fisher information for source model
        target_fisher: Fisher information for target model
        base_alpha: Base blending factor
        config: Blending configuration
        backend: Optional backend

    Returns:
        FisherBlendingResult with merged weights and diagnostics
    """
    if config is None:
        config = FisherBlendingConfig()

    b = backend or get_default_backend()
    merged_weights: "dict[str, Array]" = {}
    effective_alphas: "dict[str, Array]" = {}
    all_alphas: list[float] = []
    params_blended = 0

    for key in source_weights:
        if key not in target_weights:
            continue

        src_w = source_weights[key]
        tgt_w = target_weights[key]
        src_f = source_fisher.weights_by_key.get(key)
        tgt_f = target_fisher.weights_by_key.get(key)

        merged, alpha_eff = apply_fisher_blending(
            source_weight=src_w,
            target_weight=tgt_w,
            base_alpha=base_alpha,
            source_fisher=src_f,
            target_fisher=tgt_f,
            config=config,
            backend=b,
        )

        merged_weights[key] = merged
        effective_alphas[key] = alpha_eff
        mean_arr = b.mean(alpha_eff)
        b.eval(mean_arr)
        all_alphas.append(float(b.to_numpy(mean_arr).item()))
        params_blended += 1

    mean_alpha = sum(all_alphas) / len(all_alphas) if all_alphas else base_alpha
    fisher_applied = config.strength > 0 and (
        bool(source_fisher.weights_by_key) or bool(target_fisher.weights_by_key)
    )

    return FisherBlendingResult(
        merged_weights=merged_weights,
        effective_alphas=effective_alphas,
        mean_effective_alpha=mean_alpha,
        fisher_applied=fisher_applied,
        parameters_blended=params_blended,
    )


# =============================================================================
# Fisher Estimation Utilities
# =============================================================================


def estimate_fisher_from_loss_landscape(
    weights: "dict[str, Array]",
    loss_fn,  # Callable[[dict[str, Array]], float]
    num_samples: int = 100,
    perturbation_scale: float = 0.01,
    config: FisherBlendingConfig | None = None,
    backend: "Backend | None" = None,
) -> FisherWeights:
    """
    Estimate Fisher information by sampling the loss landscape.

    This is an approximation that doesn't require access to gradients.
    It perturbs each parameter and measures loss sensitivity.

    Args:
        weights: Model weights
        loss_fn: Function that computes loss given weights
        num_samples: Number of perturbation samples
        perturbation_scale: Scale of random perturbations
        config: Configuration
        backend: Optional backend

    Returns:
        Estimated Fisher weights
    """
    if config is None:
        config = FisherBlendingConfig()

    b = backend or get_default_backend()
    fisher_by_key: "dict[str, Array]" = {}
    all_fisher: list[float] = []
    total_params = 0

    for key, w in weights.items():
        # Sample perturbations and measure loss changes
        loss_deltas: "list[Array]" = []

        for _ in range(num_samples):
            # Generate random perturbation
            perturbation = b.random_normal(w.shape) * perturbation_scale

            # Compute loss with perturbation
            perturbed_weights = {**weights, key: w + perturbation}
            loss_perturbed = loss_fn(perturbed_weights)

            # Compute base loss
            loss_base = loss_fn(weights)

            # Loss sensitivity
            delta = abs(loss_perturbed - loss_base)
            loss_deltas.append(b.full(w.shape, delta))

        # Average sensitivity = proxy for Fisher
        stacked = b.stack(loss_deltas, axis=0)
        mean_sensitivity = b.mean(stacked, axis=0)

        # Fisher ~ sensitivity (higher sensitivity = more important)
        fisher = mean_sensitivity / (perturbation_scale**2 + config.epsilon)
        fisher = b.clip(fisher, config.min_fisher, config.max_fisher)
        b.eval(fisher)

        fisher_by_key[key] = fisher
        total_params += fisher.size
        mean_arr = b.mean(fisher)
        b.eval(mean_arr)
        all_fisher.append(float(b.to_numpy(mean_arr).item()))

    mean_fisher = sum(all_fisher) / len(all_fisher) if all_fisher else 0.0
    variance = (
        sum((x - mean_fisher) ** 2 for x in all_fisher) / len(all_fisher) if all_fisher else 0.0
    )
    std_fisher = math.sqrt(variance)

    return FisherWeights(
        weights_by_key=fisher_by_key,
        estimation_method=FisherEstimationMethod.EMPIRICAL,
        total_parameters=total_params,
        mean_fisher=mean_fisher,
        std_fisher=std_fisher,
    )


def combine_fisher_weights(
    fisher_list: list[FisherWeights],
    combination_method: str = "mean",
    backend: "Backend | None" = None,
) -> FisherWeights:
    """
    Combine multiple Fisher weight estimates.

    Useful when you have Fisher estimates from multiple data sources.

    Args:
        fisher_list: List of Fisher weight estimates
        combination_method: "mean", "max", or "harmonic"
        backend: Optional backend

    Returns:
        Combined Fisher weights
    """
    if not fisher_list:
        raise ValueError("Need at least one FisherWeights to combine")

    if len(fisher_list) == 1:
        return fisher_list[0]

    b = backend or get_default_backend()

    # Collect all keys
    all_keys = set()
    for fw in fisher_list:
        all_keys.update(fw.weights_by_key.keys())

    combined: "dict[str, Array]" = {}

    for key in all_keys:
        weights_for_key = [fw.weights_by_key[key] for fw in fisher_list if key in fw.weights_by_key]

        if not weights_for_key:
            continue

        stacked = b.stack(weights_for_key, axis=0)

        if combination_method == "mean":
            combined[key] = b.mean(stacked, axis=0)
        elif combination_method == "max":
            combined[key] = b.max(stacked, axis=0)
        elif combination_method == "harmonic":
            # Harmonic mean
            reciprocal = 1.0 / (stacked + 1e-10)
            combined[key] = len(weights_for_key) / b.sum(reciprocal, axis=0)
        else:
            combined[key] = b.mean(stacked, axis=0)

        b.eval(combined[key])

    total_params = sum(w.size for w in combined.values())
    all_means = []
    for w in combined.values():
        mean_arr = b.mean(w)
        b.eval(mean_arr)
        all_means.append(float(b.to_numpy(mean_arr).item()))
    mean_fisher = sum(all_means) / len(all_means) if all_means else 0.0

    return FisherWeights(
        weights_by_key=combined,
        estimation_method=fisher_list[0].estimation_method,
        total_parameters=total_params,
        mean_fisher=mean_fisher,
        std_fisher=0.0,  # Not computed for combined
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_fisher_blend(
    source: "dict[str, Array]",
    target: "dict[str, Array]",
    alpha: float = 0.5,
    strength: float = 0.5,
    backend: "Backend | None" = None,
) -> "dict[str, Array]":
    """
    Quick Fisher-weighted blend using uniform Fisher estimates.

    Use this when you don't have gradient history but want
    Fisher-inspired blending behavior.

    Args:
        source: Source weights
        target: Target weights
        alpha: Base blending factor
        strength: Fisher influence strength
        backend: Optional backend

    Returns:
        Merged weights
    """
    b = backend or get_default_backend()
    config = FisherBlendingConfig(strength=strength)

    # Create uniform Fisher weights
    keys = list(set(source.keys()) & set(target.keys()))
    shapes = {k: source[k].shape for k in keys}

    source_fisher = FisherWeights.uniform(keys, shapes, backend=b)
    target_fisher = FisherWeights.uniform(keys, shapes, backend=b)

    result = fisher_weighted_merge(
        source_weights=source,
        target_weights=target,
        source_fisher=source_fisher,
        target_fisher=target_fisher,
        base_alpha=alpha,
        config=config,
        backend=b,
    )

    return result.merged_weights
