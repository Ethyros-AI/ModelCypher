# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometry-derived residual connection scaling.

Standard residual: output = x + f(x) with α=1
When σ_max(f(x))/σ_max(x) varies across layers, gradient flow becomes uneven.

Geometric solution: α = σ_max(x) / σ_max(f(x))
This normalizes so ||α × f(x)|| ≈ ||x||, making residual contributions comparable.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific hooks and model wrappers live in adapters/training/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def spectral_norm_power_iteration(
    x: "Array",
    backend: "Backend",
    n_iters: int = 3,
) -> float:
    """Fast spectral norm via power iteration using Backend protocol.

    For activation tensors [batch, seq, hidden], computes spectral norm
    of the [seq, hidden] matrix (treating batch as independent).

    Args:
        x: Input tensor (any shape, flattened to 2D if needed).
        backend: Backend for tensor operations.
        n_iters: Power iterations (default 3 for speed).

    Returns:
        Spectral norm (largest singular value).
    """
    b = backend
    sqrt_eps = division_epsilon(b, x)

    # Handle different tensor shapes
    ndim = len(x.shape)
    if ndim == 1:
        norm_sq = b.sum(x * x)
        b.eval(norm_sq)
        return sqrt_scalar(float(b.to_scalar(norm_sq)), b)
    elif ndim == 2:
        M = x
    elif ndim == 3:
        # [batch, seq, hidden] -> take first batch, [seq, hidden]
        M = x[0]
    else:
        # Flatten to 2D
        M = b.reshape(x, (-1, int(x.shape[-1])))

    m, n = int(M.shape[0]), int(M.shape[1])

    # Handle degenerate cases
    if m == 0 or n == 0:
        return 0.0

    # Initialize v as unit vector
    ones = b.ones((n,))
    sqrt_n = sqrt_scalar(float(n), b)
    v = ones / sqrt_n
    b.eval(v)

    for _ in range(n_iters):
        # u = M @ v
        u = b.matmul(M, v)
        u_norm_sq = b.sum(u * u)
        b.eval(u_norm_sq)
        u_norm = sqrt_scalar(float(b.to_scalar(u_norm_sq)), b)

        if u_norm < sqrt_eps:
            return 0.0

        u = u / u_norm

        # v = M^T @ u
        M_T = b.transpose(M)
        v = b.matmul(M_T, u)
        v_norm_sq = b.sum(v * v)
        b.eval(v_norm_sq)
        v_norm = sqrt_scalar(float(b.to_scalar(v_norm_sq)), b)

        if v_norm < sqrt_eps:
            return 0.0

        v = v / v_norm
        b.eval(v)

    # Spectral norm = ||M @ v||
    Mv = b.matmul(M, v)
    spectral_sq = b.sum(Mv * Mv)
    b.eval(spectral_sq)

    return sqrt_scalar(float(b.to_scalar(spectral_sq)), b)


@dataclass
class ResidualScaleStats:
    """Statistics for residual scaling at a single layer."""

    layer_idx: int
    input_spectral: float
    residual_spectral: float
    alpha: float  # Computed scale factor

    @property
    def is_valid(self) -> bool:
        """Whether the scale factor is in a reasonable range."""
        return 0.1 < self.alpha < 10.0


@dataclass
class ResidualScalingState:
    """Global state tracking for residual scaling."""

    layer_stats: list[ResidualScaleStats] = field(default_factory=list)
    step: int = 0

    def add_layer_stats(self, stats: ResidualScaleStats) -> None:
        """Add stats for a layer."""
        self.layer_stats.append(stats)

    def reset_step(self) -> None:
        """Reset for new forward pass."""
        self.layer_stats = []
        self.step += 1

    def get_alpha_summary(self) -> dict:
        """Get summary statistics for alpha values.

        Uses pure Python math (no numpy).
        """
        if not self.layer_stats:
            return {}

        alphas = [s.alpha for s in self.layer_stats]
        n = len(alphas)

        # Mean
        alpha_mean = sum(alphas) / n

        # Min/Max
        alpha_min = min(alphas)
        alpha_max = max(alphas)

        # Standard deviation: sqrt(E[(X - μ)²])
        variance = sum((a - alpha_mean) ** 2 for a in alphas) / n
        alpha_std = variance ** 0.5

        return {
            "mean": alpha_mean,
            "min": alpha_min,
            "max": alpha_max,
            "std": alpha_std,
            "n_invalid": sum(1 for s in self.layer_stats if not s.is_valid),
        }


def compute_residual_scale(
    input_spectral: float,
    residual_spectral: float,
    layer_idx: int,
    min_alpha: float = 0.1,
    max_alpha: float = 10.0,
    sqrt_eps: float = 1e-6,
) -> tuple[float, ResidualScaleStats]:
    """Compute the residual scale factor (pure function).

    Args:
        input_spectral: Spectral norm of input x.
        residual_spectral: Spectral norm of residual f(x).
        layer_idx: Layer index for logging.
        min_alpha: Minimum allowed scale (clamp floor).
        max_alpha: Maximum allowed scale (clamp ceiling).
        sqrt_eps: Numerical stability threshold.

    Returns:
        Tuple of (alpha, stats) where alpha is the scale factor.
    """
    # Compute alpha = σ_max(x) / σ_max(f(x))
    if residual_spectral > sqrt_eps:
        alpha = input_spectral / residual_spectral
    else:
        # Residual is near-zero, don't scale
        alpha = 1.0

    # Clamp to valid range
    alpha = max(min_alpha, min(alpha, max_alpha))

    stats = ResidualScaleStats(
        layer_idx=layer_idx,
        input_spectral=input_spectral,
        residual_spectral=residual_spectral,
        alpha=alpha,
    )

    return alpha, stats


@dataclass
class ResidualScalingConfig:
    """Configuration for residual scaling.

    Contains all parameters needed to configure residual scaling
    without any framework-specific dependencies.
    """

    min_alpha: float = 0.1
    max_alpha: float = 10.0
    enabled: bool = True

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResidualScalingConfig":
        """Deserialize from checkpoint."""
        return cls(
            min_alpha=data.get("min_alpha", 0.1),
            max_alpha=data.get("max_alpha", 10.0),
            enabled=data.get("enabled", True),
        )


def log_residual_stats(state: ResidualScalingState, prefix: str = "") -> None:
    """Log alpha distribution statistics."""
    summary = state.get_alpha_summary()
    if summary:
        logger.info(
            "%sResidual scaling: α=%.3f (range: %.3f-%.3f, std=%.3f, invalid=%d)",
            prefix,
            summary["mean"],
            summary["min"],
            summary["max"],
            summary["std"],
            summary["n_invalid"],
        )


__all__ = [
    "ResidualScaleStats",
    "ResidualScalingState",
    "ResidualScalingConfig",
    "spectral_norm_power_iteration",
    "compute_residual_scale",
    "log_residual_stats",
]
