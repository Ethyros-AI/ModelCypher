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

Implementation is hook-based (non-invasive) - no model modifications required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Machine epsilon for numerical stability
SQRT_EPS = np.sqrt(np.finfo(np.float32).eps)


def _spectral_norm_fast(x: mx.array, n_iters: int = 3) -> float:
    """Fast spectral norm via power iteration.

    For activation tensors [batch, seq, hidden], computes spectral norm
    of the [seq, hidden] matrix (treating batch as independent).

    Args:
        x: Input tensor (any shape, flattened to 2D if needed)
        n_iters: Power iterations (default 3 for speed)

    Returns:
        Spectral norm (largest singular value)
    """
    # Handle different tensor shapes
    if x.ndim == 1:
        return float(mx.sqrt(mx.sum(x * x)))
    elif x.ndim == 2:
        M = x
    elif x.ndim == 3:
        # [batch, seq, hidden] -> take first batch, [seq, hidden]
        M = x[0]
    else:
        # Flatten to 2D
        M = mx.reshape(x, (-1, x.shape[-1]))

    m, n = int(M.shape[0]), int(M.shape[1])

    # Handle degenerate cases
    if m == 0 or n == 0:
        return 0.0

    # Initialize v as unit vector
    v = mx.ones((n,)) / mx.sqrt(mx.array(float(n)))
    mx.eval(v)

    for _ in range(n_iters):
        # u = M @ v
        u = M @ v
        u_norm = mx.sqrt(mx.sum(u * u))
        mx.eval(u_norm)

        if float(u_norm) < SQRT_EPS:
            return 0.0

        u = u / u_norm

        # v = M^T @ u
        v = M.T @ u
        v_norm = mx.sqrt(mx.sum(v * v))
        mx.eval(v_norm)

        if float(v_norm) < SQRT_EPS:
            return 0.0

        v = v / v_norm
        mx.eval(v)

    # Spectral norm = ||M @ v||
    Mv = M @ v
    spectral = mx.sqrt(mx.sum(Mv * Mv))
    mx.eval(spectral)

    return float(spectral)


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
        """Get summary statistics for alpha values."""
        if not self.layer_stats:
            return {}

        alphas = [s.alpha for s in self.layer_stats]
        return {
            "mean": float(np.mean(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
            "std": float(np.std(alphas)),
            "n_invalid": sum(1 for s in self.layer_stats if not s.is_valid),
        }


class ResidualScalingHook:
    """Hook for geometry-derived residual connection scaling.

    Computes α = σ_max(x) / σ_max(f(x)) for each residual connection.
    This normalizes residual contributions so ||α × f(x)|| ≈ ||x||.

    Usage (training):
        hook = ResidualScalingHook()
        # Apply to model's transformer blocks
        hook.apply_to_model(model)

        # During training, hooks automatically scale residuals
        for batch in data:
            output = model(batch)
            hook.log_stats()  # Optional: see alpha distribution

        # Remove hooks when done
        hook.remove_from_model()

    Usage (inference only):
        hook = ResidualScalingHook(inference_only=True)
        hook.apply_to_model(model)
        output = model(input)
    """

    def __init__(
        self,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        enabled: bool = True,
    ):
        """Initialize residual scaling hook.

        Args:
            min_alpha: Minimum allowed scale (clamp floor). Default 0.1.
            max_alpha: Maximum allowed scale (clamp ceiling). Default 10.0.
            enabled: Whether scaling is active. Default True.
        """
        self._min_alpha = min_alpha
        self._max_alpha = max_alpha
        self._enabled = enabled
        self._state = ResidualScalingState()
        self._hooks: list[tuple] = []  # Track applied hooks for removal

    @property
    def enabled(self) -> bool:
        """Whether scaling is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable scaling."""
        self._enabled = value

    @property
    def state(self) -> ResidualScalingState:
        """Get current state with layer statistics."""
        return self._state

    def compute_residual_scale(
        self,
        x: mx.array,
        f_x: mx.array,
        layer_idx: int,
    ) -> tuple[float, ResidualScaleStats]:
        """Compute the residual scale factor.

        Args:
            x: Input to residual block [batch, seq, hidden]
            f_x: Residual contribution (output - input) [batch, seq, hidden]
            layer_idx: Layer index for logging

        Returns:
            Tuple of (alpha, stats) where alpha is the scale factor.
        """
        # Compute spectral norms
        input_spectral = _spectral_norm_fast(x)
        residual_spectral = _spectral_norm_fast(f_x)

        # Compute alpha = σ_max(x) / σ_max(f(x))
        if residual_spectral > SQRT_EPS:
            alpha = input_spectral / residual_spectral
        else:
            # Residual is near-zero, don't scale
            alpha = 1.0

        # Clamp to valid range
        alpha = max(self._min_alpha, min(alpha, self._max_alpha))

        stats = ResidualScaleStats(
            layer_idx=layer_idx,
            input_spectral=input_spectral,
            residual_spectral=residual_spectral,
            alpha=alpha,
        )

        return alpha, stats

    def scale_residual(
        self,
        x: mx.array,
        output: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """Apply residual scaling to layer output.

        Assumes output = x + f(x) and transforms to x + α × f(x).

        Args:
            x: Input to residual block
            output: Output from residual block (x + f(x))
            layer_idx: Layer index

        Returns:
            Scaled output: x + α × f(x)
        """
        if not self._enabled:
            return output

        # Extract residual contribution
        f_x = output - x

        # Compute scale factor
        alpha, stats = self.compute_residual_scale(x, f_x, layer_idx)
        self._state.add_layer_stats(stats)

        # Apply scaling: x + α × f(x)
        scaled_output = x + alpha * f_x
        mx.eval(scaled_output)

        return scaled_output

    def reset_for_forward(self) -> None:
        """Reset state for new forward pass. Call before each forward."""
        self._state.reset_step()

    def log_stats(self, prefix: str = "") -> None:
        """Log alpha distribution statistics."""
        summary = self._state.get_alpha_summary()
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


def apply_residual_scaling_to_output(
    hook: ResidualScalingHook,
    layer_idx: int,
    input_tensor: mx.array,
    output_tensor: mx.array,
) -> mx.array:
    """Apply residual scaling as a post-processing step.

    This is the non-hook approach - call manually after each layer.

    Args:
        hook: The ResidualScalingHook instance
        layer_idx: Layer index
        input_tensor: Input to the layer
        output_tensor: Output from the layer

    Returns:
        Scaled output tensor
    """
    return hook.scale_residual(input_tensor, output_tensor, layer_idx)


def create_residual_wrapper(
    hook: ResidualScalingHook,
    layer_idx: int,
    original_call: Callable,
) -> Callable:
    """Create a wrapper function that applies residual scaling.

    Use this to wrap transformer block __call__ methods.

    Args:
        hook: The ResidualScalingHook instance
        layer_idx: Layer index
        original_call: The original __call__ method

    Returns:
        Wrapped callable that applies residual scaling
    """

    def wrapped_call(self, x, *args, **kwargs):
        # Reset state at start of forward pass (layer 0)
        if layer_idx == 0:
            hook.reset_for_forward()

        # Get original output
        output = original_call(self, x, *args, **kwargs)

        # Apply residual scaling
        if hook.enabled:
            output = hook.scale_residual(x, output, layer_idx)

        return output

    return wrapped_call


__all__ = [
    "ResidualScaleStats",
    "ResidualScalingState",
    "ResidualScalingHook",
    "apply_residual_scaling_to_output",
    "create_residual_wrapper",
]
