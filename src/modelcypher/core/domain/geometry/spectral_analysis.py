"""
Spectral Analysis for Model Merging.

Computes spectral metrics (condition numbers, singular value ratios) to assess
weight matrix compatibility between source and target models. When spectral
mismatch is high (indicating incompatible representations), the merge should
trust the target more to maintain stability.

Reference:
- TrainingCypher/UnifiedManifoldMerger.swift (lines 1347-1373)
- Prevents merging from creating ill-conditioned weight matrices

Mathematical Foundation
-----------------------
For weight matrices W_source and W_target:

1. Spectral ratio = σ_max(W_source) / σ_max(W_target)
   - σ_max = largest singular value
   - Ratio near 1.0 = good alignment

2. Spectral confidence = min(ratio, 1/ratio)
   - Symmetric: both 2.0 and 0.5 give same confidence
   - Range: [0, 1], higher = better

3. Spectral penalty:
   - penalty = (1 - spectral_confidence) * strength
   - alpha_adjusted = alpha + (1 - alpha) * penalty
   - Effect: Low confidence → increase alpha → trust target more
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralMetrics:
    """Spectral metrics for a weight matrix pair."""

    # Condition number of the target weight matrix
    condition_number: float

    # Ratio of max singular values: source / target
    spectral_ratio: float

    # Symmetric confidence: min(ratio, 1/ratio) in [0, 1]
    spectral_confidence: float

    # Max singular value of source
    source_spectral_norm: float

    # Max singular value of target
    target_spectral_norm: float

    # Frobenius norm of the difference
    delta_frobenius: float

    @property
    def is_ill_conditioned(self) -> bool:
        """Check if target matrix is ill-conditioned (condition > 100)."""
        return self.condition_number > 100.0

    @property
    def has_high_mismatch(self) -> bool:
        """Check if spectral mismatch is high (confidence < 0.5)."""
        return self.spectral_confidence < 0.5


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for spectral analysis."""

    # Strength of spectral penalty [0, 1]
    # Higher = more aggressive penalty for mismatch
    penalty_strength: float = 0.5

    # Epsilon for numerical stability
    epsilon: float = 1e-6

    # Maximum condition number before clamping
    max_condition_number: float = 1e6

    # Whether to use full SVD (slower but more accurate) or just top-k
    use_full_svd: bool = False

    # Number of singular values to compute if not full
    top_k: int = 10

    @classmethod
    def default(cls) -> SpectralConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> SpectralConfig:
        """Conservative: less aggressive penalty."""
        return cls(penalty_strength=0.3)

    @classmethod
    def aggressive(cls) -> SpectralConfig:
        """Aggressive: strong penalty for mismatch."""
        return cls(penalty_strength=0.7)


def compute_spectral_metrics(
    source_weight: np.ndarray,
    target_weight: np.ndarray,
    config: Optional[SpectralConfig] = None,
) -> SpectralMetrics:
    """
    Compute spectral metrics for a weight matrix pair.

    Args:
        source_weight: Source model weight matrix [out_dim, in_dim] or [dim]
        target_weight: Target model weight matrix (same shape)
        config: Spectral analysis configuration

    Returns:
        SpectralMetrics with condition number, spectral ratio, and confidence
    """
    if config is None:
        config = SpectralConfig.default()

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, use vector norms instead of singular values
        source_norm = float(np.linalg.norm(source_weight))
        target_norm = float(np.linalg.norm(target_weight))
        delta_norm = float(np.linalg.norm(source_weight - target_weight))

        if target_norm < config.epsilon:
            target_norm = config.epsilon

        ratio = source_norm / target_norm
        confidence = min(ratio, 1.0 / max(ratio, config.epsilon))

        return SpectralMetrics(
            condition_number=1.0,  # 1D vectors don't have condition numbers
            spectral_ratio=ratio,
            spectral_confidence=confidence,
            source_spectral_norm=source_norm,
            target_spectral_norm=target_norm,
            delta_frobenius=delta_norm,
        )

    # 2D weight matrices
    source_np = np.asarray(source_weight, dtype=np.float32)
    target_np = np.asarray(target_weight, dtype=np.float32)

    # Compute singular values
    if config.use_full_svd:
        source_s = np.linalg.svd(source_np, compute_uv=False)
        target_s = np.linalg.svd(target_np, compute_uv=False)
    else:
        # Approximate top-k singular values using power iteration or truncated SVD
        # For speed, just compute full SVD but only use top values
        source_s = np.linalg.svd(source_np, compute_uv=False)[:config.top_k]
        target_s = np.linalg.svd(target_np, compute_uv=False)[:config.top_k]

    # Spectral norms (max singular value)
    source_spectral = float(source_s[0]) if len(source_s) > 0 else config.epsilon
    target_spectral = float(target_s[0]) if len(target_s) > 0 else config.epsilon

    # Condition number of target
    target_min_s = float(target_s[-1]) if len(target_s) > 0 else config.epsilon
    condition_number = target_spectral / max(target_min_s, config.epsilon)
    condition_number = min(condition_number, config.max_condition_number)

    # Spectral ratio
    spectral_ratio = source_spectral / max(target_spectral, config.epsilon)

    # Spectral confidence (symmetric)
    if spectral_ratio > 0:
        spectral_confidence = min(spectral_ratio, 1.0 / spectral_ratio)
    else:
        spectral_confidence = 0.0

    # Delta Frobenius norm
    delta_frobenius = float(np.linalg.norm(source_np - target_np))

    return SpectralMetrics(
        condition_number=condition_number,
        spectral_ratio=spectral_ratio,
        spectral_confidence=spectral_confidence,
        source_spectral_norm=source_spectral,
        target_spectral_norm=target_spectral,
        delta_frobenius=delta_frobenius,
    )


def apply_spectral_penalty(
    alpha: float,
    spectral_confidence: float,
    penalty_strength: float = 0.5,
) -> float:
    """
    Apply spectral penalty to alpha.

    When spectral confidence is low (mismatch is high), increase alpha
    to trust target more and maintain stability.

    Args:
        alpha: Base alpha value [0, 1]
        spectral_confidence: Spectral confidence [0, 1]
        penalty_strength: How aggressively to penalize [0, 1]

    Returns:
        Adjusted alpha with spectral penalty applied
    """
    # Clamp inputs
    alpha = max(0.0, min(1.0, alpha))
    spectral_confidence = max(0.0, min(1.0, spectral_confidence))
    penalty_strength = max(0.0, min(1.0, penalty_strength))

    if penalty_strength <= 0:
        return alpha

    # Penalty = (1 - confidence) * strength
    # Low confidence → high penalty → alpha increases toward 1 (trust target)
    penalty = (1.0 - spectral_confidence) * penalty_strength

    # Apply penalty: alpha moves toward 1.0
    adjusted = alpha + (1.0 - alpha) * penalty

    return max(0.0, min(1.0, adjusted))


def compute_spectral_alpha_adjustments(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    base_alphas: dict[str, float],
    config: Optional[SpectralConfig] = None,
) -> tuple[dict[str, float], dict[str, SpectralMetrics]]:
    """
    Compute spectral-adjusted alphas for all weight matrices.

    Args:
        source_weights: Source model weights by name
        target_weights: Target model weights by name
        base_alphas: Base alpha per weight (before spectral adjustment)
        config: Spectral analysis configuration

    Returns:
        Tuple of (adjusted_alphas, spectral_metrics)
    """
    if config is None:
        config = SpectralConfig.default()

    adjusted_alphas: dict[str, float] = {}
    metrics: dict[str, SpectralMetrics] = {}

    for name, base_alpha in base_alphas.items():
        if name not in source_weights or name not in target_weights:
            adjusted_alphas[name] = base_alpha
            continue

        source_w = source_weights[name]
        target_w = target_weights[name]

        # Skip if shapes don't match
        if source_w.shape != target_w.shape:
            logger.warning(
                "Shape mismatch for %s: %s vs %s",
                name,
                source_w.shape,
                target_w.shape,
            )
            adjusted_alphas[name] = base_alpha
            continue

        # Compute spectral metrics
        spectral = compute_spectral_metrics(source_w, target_w, config)
        metrics[name] = spectral

        # Apply penalty
        adjusted = apply_spectral_penalty(
            base_alpha,
            spectral.spectral_confidence,
            config.penalty_strength,
        )
        adjusted_alphas[name] = adjusted

        if spectral.is_ill_conditioned:
            logger.debug(
                "%s: ill-conditioned (cond=%.1f), alpha %.3f → %.3f",
                name,
                spectral.condition_number,
                base_alpha,
                adjusted,
            )

    return adjusted_alphas, metrics


def spectral_summary(metrics: dict[str, SpectralMetrics]) -> dict:
    """
    Summarize spectral metrics across all weight matrices.

    Args:
        metrics: Per-weight spectral metrics

    Returns:
        Summary statistics
    """
    if not metrics:
        return {
            "total_weights": 0,
            "ill_conditioned_count": 0,
            "high_mismatch_count": 0,
            "mean_confidence": 0.0,
            "mean_condition_number": 0.0,
        }

    confidences = [m.spectral_confidence for m in metrics.values()]
    conditions = [m.condition_number for m in metrics.values()]

    return {
        "total_weights": len(metrics),
        "ill_conditioned_count": sum(1 for m in metrics.values() if m.is_ill_conditioned),
        "high_mismatch_count": sum(1 for m in metrics.values() if m.has_high_mismatch),
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "mean_condition_number": float(np.mean(conditions)),
        "max_condition_number": float(np.max(conditions)),
    }
