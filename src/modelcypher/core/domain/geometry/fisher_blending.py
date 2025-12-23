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
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx


class FisherEstimationMethod(str, Enum):
    """Method for estimating Fisher information."""
    GRADIENT_VARIANCE = "gradient_variance"  # Inverse of gradient variance
    DIAGONAL_HESSIAN = "diagonal_hessian"    # Diagonal Hessian approximation
    EMPIRICAL = "empirical"                   # Empirical Fisher from gradients
    IDENTITY = "identity"                     # Uniform weights (baseline)


class FisherNormalization(str, Enum):
    """How to normalize Fisher weights."""
    NONE = "none"           # Raw Fisher values
    LAYER = "layer"         # Normalize within each layer
    GLOBAL = "global"       # Normalize across all parameters
    SOFTMAX = "softmax"     # Apply softmax normalization


@dataclass(frozen=True)
class FisherBlendingConfig:
    """Configuration for Fisher-weighted blending."""
    estimation_method: FisherEstimationMethod = FisherEstimationMethod.GRADIENT_VARIANCE
    normalization: FisherNormalization = FisherNormalization.LAYER
    epsilon: float = 1e-6          # Numerical stability
    strength: float = 0.5          # How much to weight by Fisher (0=ignore, 1=full)
    temperature: float = 1.0       # Temperature for softmax normalization
    min_fisher: float = 1e-8       # Minimum Fisher value (prevents division by zero)
    max_fisher: float = 1e8        # Maximum Fisher value (prevents overflow)
    source_bias: float = 0.0       # Bias toward source model (-1 to 1)
    clip_alpha: bool = True        # Whether to clip resulting alpha to [0, 1]


@dataclass
class FisherWeights:
    """Fisher information weights for a model's parameters."""
    weights_by_key: Dict[str, mx.array]
    estimation_method: FisherEstimationMethod
    total_parameters: int
    mean_fisher: float
    std_fisher: float

    @classmethod
    def from_gradients(
        cls,
        gradient_history: Dict[str, List[mx.array]],
        config: Optional[FisherBlendingConfig] = None,
    ) -> "FisherWeights":
        """
        Estimate Fisher weights from gradient history.

        Args:
            gradient_history: Dict mapping parameter keys to lists of gradients
                              from multiple training steps
            config: Optional configuration

        Returns:
            FisherWeights with estimated Fisher information
        """
        if config is None:
            config = FisherBlendingConfig()

        weights_by_key: Dict[str, mx.array] = {}
        all_fisher_values: List[float] = []
        total_params = 0

        for key, gradients in gradient_history.items():
            if not gradients:
                continue

            # Stack gradients and compute variance
            stacked = mx.stack(gradients, axis=0)
            variance = mx.var(stacked, axis=0)
            mx.eval(variance)

            # Fisher = 1 / (variance + epsilon)
            fisher = 1.0 / (variance + config.epsilon)

            # Clip to prevent numerical issues
            fisher = mx.clip(fisher, config.min_fisher, config.max_fisher)
            mx.eval(fisher)

            weights_by_key[key] = fisher
            total_params += fisher.size

            # Track statistics
            mean_val = float(mx.mean(fisher))
            all_fisher_values.append(mean_val)

        # Compute global statistics
        if all_fisher_values:
            mean_fisher = sum(all_fisher_values) / len(all_fisher_values)
            variance = sum((x - mean_fisher) ** 2 for x in all_fisher_values) / len(all_fisher_values)
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
    def uniform(cls, keys: List[str], shapes: Dict[str, Tuple[int, ...]]) -> "FisherWeights":
        """Create uniform Fisher weights (baseline)."""
        weights_by_key = {
            key: mx.ones(shapes[key]) for key in keys if key in shapes
        }
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
    merged_weights: Dict[str, mx.array]
    effective_alphas: Dict[str, mx.array]  # Per-parameter effective blending weights
    mean_effective_alpha: float
    fisher_applied: bool
    parameters_blended: int


def normalize_fisher_weights(
    fisher: mx.array,
    method: FisherNormalization,
    temperature: float = 1.0,
    global_stats: Optional[Tuple[float, float]] = None,
) -> mx.array:
    """
    Normalize Fisher weights according to the specified method.

    Args:
        fisher: Raw Fisher weights
        method: Normalization method
        temperature: Temperature for softmax
        global_stats: (mean, std) for global normalization

    Returns:
        Normalized Fisher weights
    """
    if method == FisherNormalization.NONE:
        return fisher

    elif method == FisherNormalization.LAYER:
        # Normalize to [0, 1] within the layer
        f_min = mx.min(fisher)
        f_max = mx.max(fisher)
        mx.eval(f_min, f_max)

        if float(f_max - f_min) < 1e-10:
            return mx.ones_like(fisher)

        return (fisher - f_min) / (f_max - f_min + 1e-10)

    elif method == FisherNormalization.GLOBAL:
        if global_stats is None:
            mean = float(mx.mean(fisher))
            std = float(mx.std(fisher))
        else:
            mean, std = global_stats

        if std < 1e-10:
            return mx.ones_like(fisher)

        # Z-score normalization, then sigmoid to [0, 1]
        z = (fisher - mean) / (std + 1e-10)
        return 1.0 / (1.0 + mx.exp(-z))

    elif method == FisherNormalization.SOFTMAX:
        # Apply softmax with temperature
        fisher_flat = fisher.reshape(-1)
        scaled = fisher_flat / temperature
        # Numerical stability: subtract max
        scaled = scaled - mx.max(scaled)
        exp_scaled = mx.exp(scaled)
        softmax = exp_scaled / (mx.sum(exp_scaled) + 1e-10)
        # Scale up to preserve relative magnitudes
        return (softmax * fisher_flat.size).reshape(fisher.shape)

    return fisher


def apply_fisher_blending(
    source_weight: mx.array,
    target_weight: mx.array,
    base_alpha: float,
    source_fisher: Optional[mx.array] = None,
    target_fisher: Optional[mx.array] = None,
    config: Optional[FisherBlendingConfig] = None,
) -> Tuple[mx.array, mx.array]:
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

    Returns:
        Tuple of (merged_weight, effective_alpha_per_dim)
    """
    if config is None:
        config = FisherBlendingConfig()

    # If no Fisher info, fall back to standard blending
    if source_fisher is None and target_fisher is None:
        merged = base_alpha * target_weight + (1.0 - base_alpha) * source_weight
        alpha_effective = mx.full(source_weight.shape, base_alpha)
        return merged, alpha_effective

    # Use uniform weights if only one is provided
    if source_fisher is None:
        source_fisher = mx.ones_like(source_weight)
    if target_fisher is None:
        target_fisher = mx.ones_like(target_weight)

    # Ensure shapes match
    if source_fisher.shape != source_weight.shape:
        source_fisher = mx.broadcast_to(source_fisher, source_weight.shape)
    if target_fisher.shape != target_weight.shape:
        target_fisher = mx.broadcast_to(target_fisher, target_weight.shape)

    # Normalize Fisher weights
    source_fisher_norm = normalize_fisher_weights(
        source_fisher, config.normalization, config.temperature
    )
    target_fisher_norm = normalize_fisher_weights(
        target_fisher, config.normalization, config.temperature
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
        target_importance = mx.clip(target_importance, 0.0, 1.0)

    # Compute effective alpha: blend of base_alpha and Fisher-derived importance
    # strength=0 -> use base_alpha only
    # strength=1 -> use pure Fisher weighting
    alpha_effective = (
        (1.0 - config.strength) * base_alpha +
        config.strength * target_importance * base_alpha * 2.0  # Scale by 2 since importance is 0-1
    )

    if config.clip_alpha:
        alpha_effective = mx.clip(alpha_effective, 0.0, 1.0)

    mx.eval(alpha_effective)

    # Apply per-dimension blending
    merged = alpha_effective * target_weight + (1.0 - alpha_effective) * source_weight
    mx.eval(merged)

    return merged, alpha_effective


def fisher_weighted_merge(
    source_weights: Dict[str, mx.array],
    target_weights: Dict[str, mx.array],
    source_fisher: FisherWeights,
    target_fisher: FisherWeights,
    base_alpha: float = 0.5,
    config: Optional[FisherBlendingConfig] = None,
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

    Returns:
        FisherBlendingResult with merged weights and diagnostics
    """
    if config is None:
        config = FisherBlendingConfig()

    merged_weights: Dict[str, mx.array] = {}
    effective_alphas: Dict[str, mx.array] = {}
    all_alphas: List[float] = []
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
        )

        merged_weights[key] = merged
        effective_alphas[key] = alpha_eff
        all_alphas.append(float(mx.mean(alpha_eff)))
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
    weights: Dict[str, mx.array],
    loss_fn,  # Callable[[Dict[str, mx.array]], float]
    num_samples: int = 100,
    perturbation_scale: float = 0.01,
    config: Optional[FisherBlendingConfig] = None,
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

    Returns:
        Estimated Fisher weights
    """
    if config is None:
        config = FisherBlendingConfig()

    fisher_by_key: Dict[str, mx.array] = {}
    all_fisher: List[float] = []
    total_params = 0

    for key, w in weights.items():
        # Sample perturbations and measure loss changes
        loss_deltas: List[mx.array] = []

        for _ in range(num_samples):
            # Generate random perturbation
            perturbation = mx.random.normal(w.shape) * perturbation_scale

            # Compute loss with perturbation
            perturbed_weights = {**weights, key: w + perturbation}
            loss_perturbed = loss_fn(perturbed_weights)

            # Compute base loss
            loss_base = loss_fn(weights)

            # Loss sensitivity
            delta = abs(loss_perturbed - loss_base)
            loss_deltas.append(mx.full(w.shape, delta))

        # Average sensitivity = proxy for Fisher
        stacked = mx.stack(loss_deltas, axis=0)
        mean_sensitivity = mx.mean(stacked, axis=0)

        # Fisher ~ sensitivity (higher sensitivity = more important)
        fisher = mean_sensitivity / (perturbation_scale ** 2 + config.epsilon)
        fisher = mx.clip(fisher, config.min_fisher, config.max_fisher)
        mx.eval(fisher)

        fisher_by_key[key] = fisher
        total_params += fisher.size
        all_fisher.append(float(mx.mean(fisher)))

    mean_fisher = sum(all_fisher) / len(all_fisher) if all_fisher else 0.0
    variance = sum((x - mean_fisher) ** 2 for x in all_fisher) / len(all_fisher) if all_fisher else 0.0
    std_fisher = math.sqrt(variance)

    return FisherWeights(
        weights_by_key=fisher_by_key,
        estimation_method=FisherEstimationMethod.EMPIRICAL,
        total_parameters=total_params,
        mean_fisher=mean_fisher,
        std_fisher=std_fisher,
    )


def combine_fisher_weights(
    fisher_list: List[FisherWeights],
    combination_method: str = "mean",
) -> FisherWeights:
    """
    Combine multiple Fisher weight estimates.

    Useful when you have Fisher estimates from multiple data sources.

    Args:
        fisher_list: List of Fisher weight estimates
        combination_method: "mean", "max", or "harmonic"

    Returns:
        Combined Fisher weights
    """
    if not fisher_list:
        raise ValueError("Need at least one FisherWeights to combine")

    if len(fisher_list) == 1:
        return fisher_list[0]

    # Collect all keys
    all_keys = set()
    for fw in fisher_list:
        all_keys.update(fw.weights_by_key.keys())

    combined: Dict[str, mx.array] = {}

    for key in all_keys:
        weights_for_key = [
            fw.weights_by_key[key] for fw in fisher_list
            if key in fw.weights_by_key
        ]

        if not weights_for_key:
            continue

        stacked = mx.stack(weights_for_key, axis=0)

        if combination_method == "mean":
            combined[key] = mx.mean(stacked, axis=0)
        elif combination_method == "max":
            combined[key] = mx.max(stacked, axis=0)
        elif combination_method == "harmonic":
            # Harmonic mean
            reciprocal = 1.0 / (stacked + 1e-10)
            combined[key] = len(weights_for_key) / mx.sum(reciprocal, axis=0)
        else:
            combined[key] = mx.mean(stacked, axis=0)

        mx.eval(combined[key])

    total_params = sum(w.size for w in combined.values())
    all_means = [float(mx.mean(w)) for w in combined.values()]
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
    source: Dict[str, mx.array],
    target: Dict[str, mx.array],
    alpha: float = 0.5,
    strength: float = 0.5,
) -> Dict[str, mx.array]:
    """
    Quick Fisher-weighted blend using uniform Fisher estimates.

    Use this when you don't have gradient history but want
    Fisher-inspired blending behavior.

    Args:
        source: Source weights
        target: Target weights
        alpha: Base blending factor
        strength: Fisher influence strength

    Returns:
        Merged weights
    """
    config = FisherBlendingConfig(strength=strength)

    # Create uniform Fisher weights
    keys = list(set(source.keys()) & set(target.keys()))
    shapes = {k: source[k].shape for k in keys}

    source_fisher = FisherWeights.uniform(keys, shapes)
    target_fisher = FisherWeights.uniform(keys, shapes)

    result = fisher_weighted_merge(
        source_weights=source,
        target_weights=target,
        source_fisher=source_fisher,
        target_fisher=target_fisher,
        base_alpha=alpha,
        config=config,
    )

    return result.merged_weights
