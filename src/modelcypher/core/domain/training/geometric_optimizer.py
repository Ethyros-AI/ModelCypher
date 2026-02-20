# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometry-derived optimizer configuration.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific optimizers live behind backend-specific adapters.

Provides per-layer spectral geometry for the ScaledGD + measured Lipschitz
training pipeline:
- Epsilon: max(σ_k², √ε_mach × σ_max²) — regularization for ScaledGD inverse
- Weight decay: σ_k/σ_max (condition-aware scaling)
- sigma_max, sigma_k per layer — for spectral budget monitoring

Learning rate derivation:
  The base learning rate η = 1/L comes from measuring the Lipschitz constant
  L = λ_max(Hessian) via power iteration (Nesterov 2004). ScaledGD
  preconditioning (Tong, Ma, Chi — JMLR 2021) handles per-layer adaptation
  by projecting each LoRA factor's gradient through the pseudoinverse of
  the other factor. This replaces per-layer LR heuristics (σ_k/σ_max, etc.)
  with a mathematically optimal approach.

  The lr_scale and base_lr fields are retained for serialization compatibility
  but are vestigial when using the ScaledGD + measured η pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.training.geometric_lora import (
    LayerGeometry,
    compute_layer_geometry,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerOptimizerConfig:
    """Per-layer optimizer config derived from spectral structure.

    All values are geometry-derived:
    - sigma_max: Largest singular value
    - sigma_k: Smallest significant SV (noise floor)
    - lr_scale: Vestigial (retained for serialization). ScaledGD + measured η
                replaces per-layer LR scaling.
    - epsilon: Numerical stability threshold (used as ScaledGD regularization)
    - decay_scale: Condition-aware weight decay multiplier
    """

    layer_key: str
    sigma_max: float
    sigma_k: float
    lr_scale: float
    epsilon: float
    decay_scale: float
    spectral_gap: float  # σ_{k-1} - σ_k (Weyl crossing threshold)


@dataclass
class OptimizerGeometryConfig:
    """Global optimizer configuration derived from model geometry.

    Contains all information needed to configure a geometric optimizer
    without any framework-specific dependencies.
    """

    base_lr: float
    max_sigma: float
    layer_configs: dict[str, LayerOptimizerConfig]

    @property
    def n_layers(self) -> int:
        """Number of configured layers."""
        return len(self.layer_configs)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "type": "geometric",
            "base_lr": self.base_lr,
            "max_sigma": self.max_sigma,
            "layer_configs": {
                key: {
                    "sigma_max": cfg.sigma_max,
                    "sigma_k": cfg.sigma_k,
                    "lr_scale": cfg.lr_scale,
                    "epsilon": cfg.epsilon,
                    "decay_scale": cfg.decay_scale,
                    "spectral_gap": cfg.spectral_gap,
                }
                for key, cfg in self.layer_configs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizerGeometryConfig":
        """Deserialize from checkpoint."""
        if data.get("type") != "geometric":
            raise ValueError(f"Invalid optimizer config type: {data.get('type')}")

        layer_configs = {}
        for key, cfg_dict in data.get("layer_configs", {}).items():
            layer_configs[key] = LayerOptimizerConfig(
                layer_key=key,
                sigma_max=cfg_dict["sigma_max"],
                sigma_k=cfg_dict["sigma_k"],
                lr_scale=cfg_dict["lr_scale"],
                epsilon=cfg_dict["epsilon"],
                decay_scale=cfg_dict["decay_scale"],
                spectral_gap=cfg_dict.get("spectral_gap", 0.0),
            )

        return cls(
            base_lr=data["base_lr"],
            max_sigma=data.get("max_sigma", 0.0),
            layer_configs=layer_configs,
        )


def compute_geometric_epsilon(
    sigma_max: float,
    sigma_k: float,
    backend: "Backend",
) -> float:
    """Compute geometry-derived epsilon for numerical stability.

    Formula: ε = max(σ_k², √ε_mach × σ_max²)

    This is the natural scale for numerical precision in this layer's geometry.

    Args:
        sigma_max: Largest singular value.
        sigma_k: Smallest structural-boundary singular value.
        backend: Backend for scalar operations.

    Returns:
        Epsilon value for numerical stability.
    """
    # Get machine epsilon from backend (use default dtype)
    eps = backend.finfo().eps
    sqrt_eps = sqrt_scalar(eps, backend)

    noise_floor = sigma_k ** 2
    machine_floor = sqrt_eps * sigma_max ** 2
    return max(noise_floor, machine_floor)


def compute_decay_scale(sigma_max: float, sigma_k: float) -> float:
    """Compute condition-aware weight decay scale.

    Formula: scale = σ_k / σ_max (condition ratio)

    Poorly-conditioned layers (high κ = σ_max/σ_k) get less decay because
    their small singular values are already near the noise floor.

    Args:
        sigma_max: Largest singular value.
        sigma_k: Smallest structural-boundary singular value.

    Returns:
        Decay scale factor in (0, 1].
    """
    if sigma_max <= 0.0:
        return 1.0
    return sigma_k / sigma_max


def derive_layer_optimizer_config(
    geometry: LayerGeometry,
    max_sigma: float,
    backend: "Backend",
) -> LayerOptimizerConfig:
    """Derive optimizer config for a single layer from its geometry.

    Args:
        geometry: Pre-computed spectral geometry for this layer.
        max_sigma: Maximum sigma_max across all layers (for relative scaling).
        backend: Backend for scalar operations.

    Returns:
        LayerOptimizerConfig with all geometry-derived parameters.
    """
    # LR scale: vestigial field (ScaledGD + measured η replaces per-layer LR).
    # Kept for serialization compatibility.
    lr_scale = max_sigma / geometry.sigma_max if geometry.sigma_max > 0.0 else 1.0

    epsilon = compute_geometric_epsilon(geometry.sigma_max, geometry.sigma_k, backend)
    decay_scale = compute_decay_scale(geometry.sigma_max, geometry.sigma_k)

    return LayerOptimizerConfig(
        layer_key=geometry.layer_key,
        sigma_max=geometry.sigma_max,
        sigma_k=geometry.sigma_k,
        lr_scale=lr_scale,
        epsilon=epsilon,
        decay_scale=decay_scale,
        spectral_gap=geometry.spectral_gap,
    )


def derive_optimizer_geometry_config(
    weights: dict[str, "Array"],
    backend: "Backend",
    geometries: dict[str, LayerGeometry] | None = None,
) -> OptimizerGeometryConfig:
    """Derive complete optimizer configuration from weight matrices.

    Analyzes spectral structure of all 2D weight matrices and derives
    per-layer optimizer parameters. SVD is well-defined for any matrix;
    degenerate 1×N or N×1 matrices will correctly compute as
    non-targetable (tail_dims = 0) in compute_layer_geometry().

    Args:
        weights: Dict mapping layer_key -> weight array.
        backend: Backend for tensor operations.

    Returns:
        OptimizerGeometryConfig with all geometry-derived parameters.
    """
    if geometries is None:
        geometries = {}
        max_sigma = 0.0

        # First pass: compute geometry for all weight matrices
        n_analyzed = 0
        for key, weight in weights.items():
            # Only analyze 2D weight matrices (skip biases, norms, etc.)
            if len(weight.shape) != 2:
                continue

            try:
                geom = compute_layer_geometry(weight, key, backend)
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
    else:
        # Reuse caller-provided SVD geometry to avoid recomputing all matrices.
        max_sigma = max((geom.sigma_max for geom in geometries.values()), default=0.0)
        logger.info(
            "Reusing precomputed geometry for optimizer config (%d layers)",
            len(geometries),
        )

    if max_sigma <= 0.0:
        raise ValueError("No valid weight matrices found for geometric optimization")

    logger.info(
        "Analyzed %d weight matrices, max σ_max=%.4f",
        len(geometries), max_sigma
    )

    # Second pass: compute per-layer configs relative to max
    layer_configs: dict[str, LayerOptimizerConfig] = {}
    for key, geom in geometries.items():
        layer_configs[key] = derive_layer_optimizer_config(geom, max_sigma, backend)

    # Base learning rate from geometry
    base_lr = 1.0 / max_sigma

    # Log configuration summary
    lr_scales = [cfg.lr_scale for cfg in layer_configs.values()]
    decay_scales = [cfg.decay_scale for cfg in layer_configs.values()]

    logger.info(
        "Geometric optimizer config: base_lr=%.6f (from max σ=%.4f)",
        base_lr, max_sigma
    )
    logger.info(
        "  LR scales: min=%.2f, max=%.2f",
        min(lr_scales), max(lr_scales)
    )
    logger.info(
        "  Decay scales: min=%.4f, max=%.4f",
        min(decay_scales), max(decay_scales)
    )
    logger.info("  Configured %d layers with geometry-derived parameters", len(layer_configs))

    return OptimizerGeometryConfig(
        base_lr=base_lr,
        max_sigma=max_sigma,
        layer_configs=layer_configs,
    )


def compute_lr_statistics(per_layer_lrs: dict[str, float], backend: "Backend") -> dict:
    """Compute statistics on per-layer learning rates.

    Args:
        per_layer_lrs: Dict mapping layer_key -> learning rate.
        backend: Backend for scalar operations.

    Returns:
        Dict with lr_mean, lr_min, lr_max, lr_std.
    """
    if not per_layer_lrs:
        return {}

    lrs = list(per_layer_lrs.values())
    n = len(lrs)

    # Compute statistics using basic math (no numpy dependency)
    lr_sum = sum(lrs)
    lr_mean = lr_sum / n
    lr_min = min(lrs)
    lr_max = max(lrs)

    # Variance: E[(X - μ)²]
    variance = sum((lr - lr_mean) ** 2 for lr in lrs) / n
    lr_std = sqrt_scalar(variance, backend)

    return {
        "lr_mean": lr_mean,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "lr_std": lr_std,
    }


__all__ = [
    "LayerOptimizerConfig",
    "OptimizerGeometryConfig",
    "compute_geometric_epsilon",
    "compute_decay_scale",
    "derive_layer_optimizer_config",
    "derive_optimizer_geometry_config",
    "compute_lr_statistics",
]
