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
Spectral Analysis for Model Merging.

Computes spectral metrics (condition numbers, singular value ratios) to assess
transformation effort between source and target models. When spectral
mismatch is high (indicating different representation scales), the merge should
trust the target more to maintain stability.

Reference:
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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

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


def _to_float(val: Any) -> float:
    """Convert any scalar (including MLX) to Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def compute_spectral_metrics(
    source_weight: "Array",
    target_weight: "Array",
    config: SpectralConfig | None = None,
    backend: "Backend | None" = None,
) -> SpectralMetrics:
    """
    Compute spectral metrics for a weight matrix pair.

    Args:
        source_weight: Source model weight matrix [out_dim, in_dim] or [dim]
        target_weight: Target model weight matrix (same shape)
        config: Spectral analysis configuration
        backend: Optional Backend for GPU-accelerated SVD.

    Returns:
        SpectralMetrics with condition number, spectral ratio, and confidence
    """
    if config is None:
        config = SpectralConfig.default()

    b = backend or get_default_backend()

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, use vector norms instead of singular values
        source_norm = _to_float(b.norm(source_weight))
        target_norm = _to_float(b.norm(target_weight))
        delta_norm = _to_float(b.norm(source_weight - target_weight))

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

    # 2D weight matrices - use backend for GPU acceleration
    source_arr = b.array(source_weight) if not hasattr(source_weight, "shape") else source_weight
    target_arr = b.array(target_weight) if not hasattr(target_weight, "shape") else target_weight

    # SVD - compute only singular values (not U or Vt) to avoid 92GB allocation
    # For (vocab_size, hidden_dim) matrices, full U would be vocab_size^2
    source_s = b.svd(source_arr, compute_uv=False)
    target_s = b.svd(target_arr, compute_uv=False)

    # Limit to top_k if not using full SVD
    if not config.use_full_svd:
        source_s = source_s[: config.top_k]
        target_s = target_s[: config.top_k]

    # Evaluate and extract values
    b.eval(source_s, target_s)

    # Extract values
    source_s_np = b.to_numpy(source_s)
    target_s_np = b.to_numpy(target_s)

    source_spectral = float(source_s_np[0]) if len(source_s_np) > 0 else config.epsilon
    target_spectral = float(target_s_np[0]) if len(target_s_np) > 0 else config.epsilon
    target_min_s = float(target_s_np[-1]) if len(target_s_np) > 0 else config.epsilon

    # Delta Frobenius norm
    delta_arr = b.norm(source_arr - target_arr)
    b.eval(delta_arr)
    delta_frobenius = _to_float(delta_arr)

    # Condition number of target
    condition_number = target_spectral / max(target_min_s, config.epsilon)
    condition_number = min(condition_number, config.max_condition_number)

    # Spectral ratio
    spectral_ratio = source_spectral / max(target_spectral, config.epsilon)

    # Spectral confidence (symmetric)
    if spectral_ratio > 0:
        spectral_confidence = min(spectral_ratio, 1.0 / spectral_ratio)
    else:
        spectral_confidence = 0.0

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
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    base_alphas: dict[str, float],
    config: SpectralConfig | None = None,
    backend: "Backend | None" = None,
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
        spectral = compute_spectral_metrics(source_w, target_w, config, backend=backend)
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
        "mean_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "mean_condition_number": sum(conditions) / len(conditions),
        "max_condition_number": max(conditions),
    }
