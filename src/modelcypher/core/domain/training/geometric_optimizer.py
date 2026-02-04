# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometry-derived optimizer with Barzilai-Borwein adaptation.

Replaces Adam/AdamW with pure geometry - no magic hyperparameters:
- Base LR: 1 / max(σ_max) across all layers (first step)
- Per-layer LR: Barzilai-Borwein derived, bounded by spectral structure
- Epsilon: max(σ_k², √ε_mach × σ_max²)
- Weight decay: condition-aware scaling
- No momentum

All parameters derived from spectral structure of weight matrices.
The LoRA spectral scale bound proved this works - adapters were 600-2700x
over the geometric limit with standard configs.

Barzilai-Borwein (BB) Method:
- After first step, LR is derived from gradient history:
    α_k = (s·s) / (s·y)  where s = θ_k - θ_{k-1}, y = g_k - g_{k-1}
- This approximates inverse Hessian along gradient direction
- Bounded by spectral structure: [σ_k/σ_max, 1/σ_max]
- Zero hyperparameters, superlinear convergence for quadratics

Reference: Barzilai & Borwein (1988), "Two-Point Step Size Gradient Methods"
https://epubs.siam.org/doi/10.1137/S1052623494266365
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .geometric_lora import compute_layer_geometry, LayerGeometry

logger = logging.getLogger(__name__)

# Machine epsilon for float32 (used in geometry derivation)
EPS_F32 = np.finfo(np.float32).eps
SQRT_EPS_F32 = np.sqrt(EPS_F32)


@dataclass
class LayerGeometricConfig:
    """Per-layer optimizer config derived from spectral structure."""

    layer_key: str
    sigma_max: float  # Largest singular value
    sigma_k: float  # Smallest significant SV
    lr_scale: float  # Relative LR scale (≥ 1.0)
    epsilon: float  # Geometric epsilon for this layer
    decay_scale: float  # Condition-aware weight decay scale (0, 1]


def _compute_geometric_epsilon(sigma_max: float, sigma_k: float) -> float:
    """Compute geometry-derived epsilon for numerical stability.

    Formula: ε = max(σ_k², √ε_mach × σ_max²)

    This is the natural scale for numerical precision in this layer's geometry.
    """
    noise_floor = sigma_k ** 2
    machine_floor = SQRT_EPS_F32 * sigma_max ** 2
    return max(noise_floor, machine_floor)


def _compute_decay_scale(sigma_max: float, sigma_k: float) -> float:
    """Compute condition-aware weight decay scale.

    Formula: scale = σ_k / σ_max (condition ratio)

    Poorly-conditioned layers (high κ = σ_max/σ_k) get less decay because
    their small singular values are already near the noise floor.
    """
    if sigma_max < 1e-10:
        return 1.0
    return sigma_k / sigma_max


def analyze_model_for_optimizer(model) -> dict[str, LayerGeometricConfig]:
    """Analyze model geometry for optimizer configuration.

    Computes spectral structure of all 2D weight matrices and derives
    per-layer optimizer parameters.

    Returns:
        Dict mapping layer key -> LayerGeometricConfig
    """
    configs = {}
    max_sigma = 0.0

    # First pass: compute geometry for all weight matrices
    geometries: dict[str, LayerGeometry] = {}

    # Flatten parameters to get all weights
    flat_params = tree_flatten(model.parameters())

    n_analyzed = 0
    for key, param in flat_params:
        if not isinstance(param, mx.array):
            continue

        # Only analyze 2D weight matrices (skip biases, norms, etc.)
        if param.ndim != 2:
            continue

        # Skip very small matrices (embeddings, classifiers handled separately)
        if min(param.shape) < 4:
            continue

        try:
            geom = compute_layer_geometry(param, key)
            geometries[key] = geom
            max_sigma = max(max_sigma, geom.sigma_max)
            n_analyzed += 1
            if n_analyzed % 50 == 0:
                logger.info("Analyzed %d weight matrices...", n_analyzed)
            logger.debug(
                "Layer %s: σ_max=%.4f, σ_k=%.6f, κ=%.1f",
                key, geom.sigma_max, geom.sigma_k, geom.decay_ratio
            )
        except Exception as e:
            logger.warning("Failed to analyze layer %s: %s", key, e)
            continue

    if max_sigma < 1e-10:
        raise ValueError("No valid weight matrices found for geometric optimization")

    logger.info(
        "Analyzed %d weight matrices, max σ_max=%.4f",
        len(geometries), max_sigma
    )

    # Second pass: compute per-layer configs relative to max
    for key, geom in geometries.items():
        # LR scale: smaller σ_max → larger effective LR
        # This means effective_lr = base_lr × lr_scale = (1/max_σ) × (max_σ/σ_max_i) = 1/σ_max_i
        lr_scale = max_sigma / geom.sigma_max if geom.sigma_max > 1e-10 else 1.0

        epsilon = _compute_geometric_epsilon(geom.sigma_max, geom.sigma_k)
        decay_scale = _compute_decay_scale(geom.sigma_max, geom.sigma_k)

        configs[key] = LayerGeometricConfig(
            layer_key=key,
            sigma_max=geom.sigma_max,
            sigma_k=geom.sigma_k,
            lr_scale=lr_scale,
            epsilon=epsilon,
            decay_scale=decay_scale,
        )

    return configs


class GeometricOptimizer:
    """MLX optimizer with geometry-derived per-layer scaling and Barzilai-Borwein adaptation.

    No momentum. No magic hyperparameters. Gradient descent with learning rate
    derived from spectral structure and local curvature.

    First step:
        LR = 1 / σ_max_i (spectral bound, no gradient history yet)

    Subsequent steps:
        LR = (s·s) / (s·y) bounded to [σ_k/σ_max, 1/σ_max]
        where s = θ_k - θ_{k-1}, y = g_k - g_{k-1}

    This is the optimal step size for quadratic loss surfaces (quasi-Newton),
    bounded by spectral structure to ensure stability.

    Compatible with MLX optimizer interface for use with engine_mlx.py.
    """

    def __init__(
        self,
        base_decay: float = 0.0,
        gradient_clip_mode: str = "none",
        global_clip_value: float = 1.0,
    ):
        """Initialize geometric optimizer.

        Args:
            base_decay: Base weight decay (will be scaled per-layer by condition).
                       Set to 0.0 for pure SGD without regularization.
            gradient_clip_mode: One of "none", "global", "spectral".
                - "none": No gradient clipping (default, relies on BB bounds)
                - "global": Clip all gradients at global_clip_value (industry standard)
                - "spectral": Clip each layer at its σ_max (geometry-derived)
            global_clip_value: Clip threshold for "global" mode (default 1.0).
        """
        self.base_decay = base_decay
        self.gradient_clip_mode = gradient_clip_mode
        self.global_clip_value = global_clip_value
        self.base_lr: float | None = None
        self._max_sigma: float = 0.0
        self.layer_configs: dict[str, LayerGeometricConfig] = {}
        self._state: dict = {}
        self._initialized = False

        # Barzilai-Borwein state tracking
        self._prev_params: dict[str, mx.array] = {}
        self._prev_grads: dict[str, mx.array] = {}
        self._step_count: int = 0
        self._per_layer_lr: dict[str, float] = {}  # Track effective LR per layer

        # BB stability tracking for adaptive warmup
        self._sdy_history: list[float] = []  # s·y values for stability tracking
        self._gradient_norms: dict[str, list[float]] = {}  # For logging

    def init_from_model(self, model) -> None:
        """Compute spectral structure and derive all parameters from geometry.

        This must be called before training. It:
        1. Analyzes spectral structure of all weight matrices
        2. Derives base_lr = 1 / max(σ_max)
        3. Computes per-layer scaling factors

        Args:
            model: The MLX model to optimize
        """
        logger.info("Analyzing model geometry for optimizer initialization...")

        self.layer_configs = analyze_model_for_optimizer(model)

        # Derive base learning rate from geometry
        max_sigma = max(cfg.sigma_max for cfg in self.layer_configs.values())
        self._max_sigma = max_sigma
        self.base_lr = 1.0 / max_sigma

        # Log configuration summary
        lr_scales = [cfg.lr_scale for cfg in self.layer_configs.values()]
        decay_scales = [cfg.decay_scale for cfg in self.layer_configs.values()]

        logger.info(
            "Geometric optimizer initialized: base_lr=%.6f (from max σ=%.4f)",
            self.base_lr, max_sigma
        )
        logger.info(
            "  LR scales: min=%.2f, max=%.2f, mean=%.2f",
            min(lr_scales), max(lr_scales), np.mean(lr_scales)
        )
        if self.base_decay > 0:
            logger.info(
                "  Decay scales: min=%.4f, max=%.4f, mean=%.4f",
                min(decay_scales), max(decay_scales), np.mean(decay_scales)
            )
        logger.info("  Configured %d layers with geometry-derived parameters", len(self.layer_configs))

        self._initialized = True

    def _clip_gradients(self, flat_grads: list) -> list:
        """Apply gradient clipping based on configured mode.

        Args:
            flat_grads: List of (key, gradient) tuples

        Returns:
            List of (key, clipped_gradient) tuples
        """
        if self.gradient_clip_mode == "none":
            return flat_grads

        clipped = []

        if self.gradient_clip_mode == "global":
            # Compute global gradient norm
            total_norm_sq = 0.0
            for key, grad in flat_grads:
                if grad is not None:
                    total_norm_sq += float(mx.sum(grad * grad))
            global_norm = np.sqrt(total_norm_sq)

            # Clip if necessary
            if global_norm > self.global_clip_value:
                scale = self.global_clip_value / (global_norm + 1e-10)
                for key, grad in flat_grads:
                    if grad is not None:
                        clipped.append((key, grad * scale))
                    else:
                        clipped.append((key, grad))
                logger.debug(
                    "Global gradient clipping: norm=%.4f → %.4f",
                    global_norm, self.global_clip_value
                )
            else:
                clipped = flat_grads

        elif self.gradient_clip_mode == "spectral":
            # Per-layer clipping at σ_max
            for key, grad in flat_grads:
                if grad is None:
                    clipped.append((key, grad))
                    continue

                config = self.layer_configs.get(key)
                if config is not None:
                    grad_norm = float(mx.sqrt(mx.sum(grad * grad)))

                    # Track gradient norms for logging
                    if key not in self._gradient_norms:
                        self._gradient_norms[key] = []
                    self._gradient_norms[key].append(grad_norm)

                    # Clip at σ_max
                    if grad_norm > config.sigma_max:
                        scale = config.sigma_max / (grad_norm + 1e-10)
                        clipped.append((key, grad * scale))
                        logger.debug(
                            "Spectral gradient clipping %s: norm=%.4f → σ_max=%.4f",
                            key, grad_norm, config.sigma_max
                        )
                    else:
                        clipped.append((key, grad))
                else:
                    clipped.append((key, grad))
        else:
            raise ValueError(f"Unknown gradient_clip_mode: {self.gradient_clip_mode}")

        return clipped

    def update(self, model: nn.Module, gradients: dict) -> nn.Module:
        """Apply geometry-scaled gradient update with Barzilai-Borwein adaptation.

        First step:
            effective_lr = base_lr × lr_scale = 1/σ_max_i
            w = w - effective_lr × grad - decay × w

        Subsequent steps:
            α_k = (s·s) / (s·y) bounded to [σ_k/σ_max, 1/σ_max]
            w = w - α_k × grad - decay × w

        For layers without geometry (biases, norms):
            w = w - base_lr × grad

        Args:
            model: The MLX model to update
            gradients: Gradient dict from nn.value_and_grad

        Returns:
            The updated model
        """
        if not self._initialized or self.base_lr is None:
            raise RuntimeError(
                "GeometricOptimizer not initialized. Call init_from_model() first."
            )

        # Flatten gradients and parameters
        flat_grads = tree_flatten(gradients)
        flat_params = tree_flatten(model.parameters())

        # Apply gradient clipping
        flat_grads = self._clip_gradients(flat_grads)

        # Build param dict for lookup (gradients may be subset of params for LoRA)
        param_dict = {key: param for key, param in flat_params}

        # Build update dict
        updates = []
        self._per_layer_lr = {}

        for key, grad in flat_grads:
            param = param_dict[key]

            if grad is None:
                updates.append((key, param))
                continue

            config = self.layer_configs.get(key)

            if config is not None:
                # Determine learning rate
                lr = self._compute_layer_lr(key, param, grad, config)
                self._per_layer_lr[key] = lr

                effective_decay = self.base_decay * config.decay_scale

                # SGD update: w = w - lr*grad - decay*w
                new_param = param - lr * grad
                if effective_decay > 0:
                    new_param = new_param - effective_decay * param
            else:
                # Non-weight params (biases, norms): use base_lr only
                lr = self.base_lr
                self._per_layer_lr[key] = lr
                new_param = param - lr * grad

            updates.append((key, new_param))

        # Store current params/grads for next BB computation (only trainable params)
        self._prev_params = {key: param_dict[key] for key, grad in flat_grads if grad is not None}
        self._prev_grads = {key: grad for key, grad in flat_grads if grad is not None}

        self._step_count += 1

        # Unflatten and apply updates
        new_params = tree_unflatten(updates)
        model.update(new_params)

        return model

    def _compute_layer_lr(
        self, key: str, param: mx.array, grad: mx.array, config: LayerGeometricConfig
    ) -> float:
        """Compute learning rate for a layer using spectral or BB method.

        Args:
            key: Layer key
            param: Current parameter value
            grad: Current gradient
            config: Layer's geometric config

        Returns:
            Learning rate for this update
        """
        # First step: use spectral LR (no gradient history yet)
        if self._step_count == 0 or key not in self._prev_grads:
            return self.base_lr * config.lr_scale

        # BB adaptation
        prev_param = self._prev_params.get(key)
        prev_grad = self._prev_grads.get(key)

        if prev_param is None or prev_grad is None:
            return self.base_lr * config.lr_scale

        # s = θ_k - θ_{k-1} (parameter difference)
        # y = g_k - g_{k-1} (gradient difference)
        s = param - prev_param
        y = grad - prev_grad

        # Flatten for dot products
        s_flat = s.flatten()
        y_flat = y.flatten()

        # s·y (curvature measure)
        s_dot_y = mx.sum(s_flat * y_flat)

        # Need to evaluate to get float value
        s_dot_y_val = float(s_dot_y)

        # BB undefined if s·y ≈ 0 (no curvature information)
        if abs(s_dot_y_val) < config.epsilon:
            return self.base_lr * config.lr_scale

        # Track s·y for stability monitoring
        self._sdy_history.append(s_dot_y_val)
        if len(self._sdy_history) > 100:
            self._sdy_history = self._sdy_history[-100:]

        # BB1: α = (s·s) / (s·y)
        s_dot_s = float(mx.sum(s_flat * s_flat))
        bb_lr = s_dot_s / s_dot_y_val

        # Spectral bounds: [σ_k/σ_max, 1/σ_max]
        # These bounds ensure:
        # - min_lr: don't step smaller than the noise floor ratio
        # - max_lr: don't step larger than inverse spectral norm
        min_lr = config.sigma_k / config.sigma_max
        max_lr = 1.0 / config.sigma_max

        # Clamp BB LR to spectral bounds
        lr = max(min_lr, min(bb_lr, max_lr))

        return lr

    def get_lr_stats(self) -> dict:
        """Return per-layer LR statistics for logging.

        Returns:
            Dict with lr_mean, lr_min, lr_max, lr_std, and step_count.
            Empty dict if no LRs recorded yet.
        """
        if not self._per_layer_lr:
            return {}

        lrs = list(self._per_layer_lr.values())
        return {
            "lr_mean": float(np.mean(lrs)),
            "lr_min": float(np.min(lrs)),
            "lr_max": float(np.max(lrs)),
            "lr_std": float(np.std(lrs)),
            "step_count": self._step_count,
        }

    def get_bb_stability(self) -> float:
        """Return variance of s·y values for adaptive warmup.

        Returns:
            Variance of recent s·y values, or inf if insufficient history.
            Low variance indicates stable curvature estimates.
        """
        if len(self._sdy_history) < 10:
            return float('inf')
        return float(np.var(self._sdy_history[-10:]))

    def is_bb_stable(self, threshold: float = 1e-4) -> bool:
        """Check if BB curvature estimates have stabilized.

        Args:
            threshold: Maximum variance for stability (relative to mean).

        Returns:
            True if BB estimates are stable (variance below threshold).
        """
        if len(self._sdy_history) < 10:
            return False
        recent = self._sdy_history[-10:]
        mean_sdy = np.mean(np.abs(recent))
        if mean_sdy < 1e-10:
            return False
        relative_var = np.var(recent) / (mean_sdy ** 2)
        return relative_var < threshold

    def get_gradient_stats(self) -> dict:
        """Return gradient norm statistics per layer for analysis.

        Returns:
            Dict mapping layer_key -> {mean, max, clip_ratio}.
        """
        stats = {}
        for key, norms in self._gradient_norms.items():
            if not norms:
                continue
            config = self.layer_configs.get(key)
            sigma_max = config.sigma_max if config else 1.0
            stats[key] = {
                "mean_norm": float(np.mean(norms)),
                "max_norm": float(np.max(norms)),
                "sigma_max": sigma_max,
                "clip_ratio": float(np.mean([1 if n > sigma_max else 0 for n in norms])),
            }
        return stats

    @property
    def state(self) -> dict:
        """Return optimizer state for checkpointing.

        Includes BB state (step count) for proper resumption.
        Does not include prev_params/prev_grads as those are large
        and BB will adapt from the next step anyway.
        """
        return {
            "type": "geometric",
            "base_lr": self.base_lr,
            "base_decay": self.base_decay,
            "max_sigma": self._max_sigma,
            "step_count": self._step_count,
            "layer_configs": {
                key: {
                    "sigma_max": cfg.sigma_max,
                    "sigma_k": cfg.sigma_k,
                    "lr_scale": cfg.lr_scale,
                    "epsilon": cfg.epsilon,
                    "decay_scale": cfg.decay_scale,
                }
                for key, cfg in self.layer_configs.items()
            },
        }

    def load_state(self, state: dict) -> None:
        """Restore optimizer state from checkpoint.

        Args:
            state: State dict from a previous checkpoint
        """
        if state.get("type") != "geometric":
            raise ValueError(f"Invalid optimizer state type: {state.get('type')}")

        self.base_lr = state["base_lr"]
        self.base_decay = state.get("base_decay", 0.0)
        self._max_sigma = state.get("max_sigma", 0.0)
        self._step_count = state.get("step_count", 0)

        # Reset BB state - will rebuild from next step
        self._prev_params = {}
        self._prev_grads = {}
        self._per_layer_lr = {}

        self.layer_configs = {}
        for key, cfg_dict in state.get("layer_configs", {}).items():
            self.layer_configs[key] = LayerGeometricConfig(
                layer_key=key,
                sigma_max=cfg_dict["sigma_max"],
                sigma_k=cfg_dict["sigma_k"],
                lr_scale=cfg_dict["lr_scale"],
                epsilon=cfg_dict["epsilon"],
                decay_scale=cfg_dict["decay_scale"],
            )

        self._initialized = True
        logger.info(
            "Loaded geometric optimizer state: base_lr=%.6f, step=%d, %d layer configs",
            self.base_lr, self._step_count, len(self.layer_configs)
        )

    @property
    def learning_rate(self) -> float:
        """Return current learning rate for compatibility with engine_mlx.py warmup."""
        return self.base_lr if self.base_lr is not None else 0.0

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate (used by warmup logic in engine_mlx.py).

        Note: This overrides the geometry-derived base_lr. During warmup,
        engine_mlx.py scales the LR linearly from 0 to base_lr. The per-layer
        scales remain unchanged.
        """
        self.base_lr = value


__all__ = [
    "LayerGeometricConfig",
    "GeometricOptimizer",
    "analyze_model_for_optimizer",
]
