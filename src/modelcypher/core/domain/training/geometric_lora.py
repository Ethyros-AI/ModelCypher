# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometry-derived LoRA implementation.

All parameters are derived from the spectral structure of base weights:
- Target modules: where decay_ratio < threshold
- Scale: σ_k(W) per layer (smallest significant singular value)
- Rank: bounded by tail_dims = full_rank - rank_90

No hyperparameters. The geometry IS the configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Numerical precision threshold (derived from float32 machine epsilon)
SQRT_EPS = np.sqrt(np.finfo(np.float32).eps)


@dataclass
class LayerGeometry:
    """Spectral geometry of a weight matrix."""

    layer_key: str
    shape: tuple[int, int]
    sigma_max: float
    sigma_k: float  # Smallest significant SV
    effective_rank: int
    full_rank: int
    decay_ratio: float  # σ_max / σ_k
    rank_90: int  # SVs needed for 90% energy
    tail_dims: int  # full_rank - rank_90 (max LoRA rank)

    @property
    def is_targetable(self) -> bool:
        """Whether this layer is safe to target (decay < 100×)."""
        return self.decay_ratio < 100.0


def compute_layer_geometry(W: mx.array, layer_key: str) -> LayerGeometry:
    """Compute spectral geometry of a weight matrix."""
    # Convert to numpy for SVD
    W_f32 = W.astype(mx.float32)
    mx.eval(W_f32)
    W_np = np.array(W_f32.tolist(), dtype=np.float32)

    # Full SVD
    _, S, _ = np.linalg.svd(W_np, full_matrices=False)

    sigma_max = float(S[0])
    sigma_min = float(S[-1])

    # Noise threshold from numerical precision
    threshold = SQRT_EPS * sigma_max

    # Effective rank (SVs above noise floor)
    significant = S > threshold
    effective_rank = int(np.sum(significant))
    sigma_k = float(S[significant][-1]) if np.any(significant) else float(S[-1])

    # Energy distribution
    total_energy = np.sum(S**2)
    cumsum = np.cumsum(S**2) / total_energy
    rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)

    full_rank = min(W_np.shape)
    tail_dims = full_rank - rank_90

    decay_ratio = sigma_max / sigma_k if sigma_k > 0 else float('inf')

    return LayerGeometry(
        layer_key=layer_key,
        shape=W_np.shape,
        sigma_max=sigma_max,
        sigma_k=sigma_k,
        effective_rank=effective_rank,
        full_rank=full_rank,
        decay_ratio=decay_ratio,
        rank_90=rank_90,
        tail_dims=tail_dims,
    )


def analyze_model_geometry(model) -> dict[str, LayerGeometry]:
    """Analyze spectral geometry of all targetable layers in a model."""
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])

    geometries = {}

    for layer_idx, layer in enumerate(layers):
        # Check for attention projections
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue

            layer_key = f"model.layers.{layer_idx}.self_attn.{proj_name}"
            geometry = compute_layer_geometry(proj.weight, layer_key)
            geometries[layer_key] = geometry

            logger.debug(
                "%s: decay=%.1f×, σ_k=%.4f, tail=%d, targetable=%s",
                layer_key, geometry.decay_ratio, geometry.sigma_k,
                geometry.tail_dims, geometry.is_targetable
            )

    return geometries


def select_target_modules(geometries: dict[str, LayerGeometry]) -> list[str]:
    """Select modules to target based on geometry (decay_ratio < 100)."""
    return [key for key, geom in geometries.items() if geom.is_targetable]


def compute_geometric_rank(geometries: dict[str, LayerGeometry], target_modules: list[str]) -> int:
    """Compute global LoRA rank from geometry (minimum tail_dims across targets).

    DEPRECATED: Use compute_per_layer_ranks() for curvature-adaptive ranks.
    """
    tail_dims = [geometries[key].tail_dims for key in target_modules if key in geometries]
    if not tail_dims:
        raise ValueError("No target modules found")
    return min(tail_dims)


def compute_per_layer_ranks(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
    min_rank: int = 4,
    max_rank: int = 64,
) -> dict[str, int]:
    """Compute per-layer LoRA ranks based on curvature (spectral decay).

    Layers with low decay_ratio (uniform singular values) are doing
    distributed, high-curvature computation → need higher LoRA rank.

    Layers with high decay_ratio (spikey, highway-like) are doing
    focused, low-curvature computation → need lower LoRA rank.

    Formula: rank_i = base_rank × sqrt(mean_decay / decay_i)

    This allocates more capacity to layers that need it, while respecting
    the spectral structure of each layer's weight matrix.

    Args:
        geometries: Pre-computed layer geometries
        target_modules: Which modules to compute ranks for
        min_rank: Minimum rank (numerical stability)
        max_rank: Maximum rank (memory constraint)

    Returns:
        Dict of layer_key -> rank
    """
    if not target_modules:
        raise ValueError("No target modules provided")

    # Get decay ratios for target modules
    decays = {}
    tail_dims_map = {}
    for key in target_modules:
        if key not in geometries:
            continue
        geom = geometries[key]
        # Clamp decay_ratio to avoid division issues
        decays[key] = max(geom.decay_ratio, 1.0)
        tail_dims_map[key] = geom.tail_dims

    if not decays:
        raise ValueError("No geometries found for target modules")

    # Compute geometric mean of decay ratios (more robust than arithmetic mean)
    log_decays = [np.log(d) for d in decays.values()]
    mean_log_decay = np.mean(log_decays)
    mean_decay = np.exp(mean_log_decay)

    # Base rank is the minimum tail_dims (conservative starting point)
    base_rank = min(tail_dims_map.values())

    # Compute per-layer ranks
    per_layer_ranks = {}
    for key, decay in decays.items():
        # Curvature factor: higher for low-decay (complex) layers
        # sqrt dampens the effect to avoid extreme values
        curvature_factor = np.sqrt(mean_decay / decay)

        # Scale base rank by curvature factor
        raw_rank = base_rank * curvature_factor

        # Clamp to bounds, respecting layer's tail_dims
        layer_max = min(max_rank, tail_dims_map[key])
        rank = int(np.clip(raw_rank, min_rank, layer_max))

        per_layer_ranks[key] = rank

        logger.debug(
            "%s: decay=%.1f, factor=%.2f, rank=%d (tail=%d)",
            key, decay, curvature_factor, rank, tail_dims_map[key]
        )

    return per_layer_ranks


class GeometricLoRALinear(nn.Module):
    """Linear layer with geometry-normalized LoRA.

    The LoRA delta is:
        delta = σ_k * (B @ A) / ||B @ A||_spectral

    Where σ_k is the smallest significant singular value of the base weight.
    This guarantees the perturbation respects the spectral structure.

    Initialization: ||B @ A||_spectral = σ_k at step 0
    Uses FULL geometric budget from step 0, derived from base weight spectral structure.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        sigma_k: float,
        rank: int,
    ):
        super().__init__()

        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        self.base_weight = base_layer.weight
        self.base_bias = getattr(base_layer, "bias", None)
        self.sigma_k = sigma_k
        self.rank = rank

        # Spectral-normalized initialization (geometry-derived)
        # Initialize so ||B @ A||_spectral = σ_k at step 0
        # Each matrix gets ||·||_spectral = sqrt(σ_k)
        sqrt_sigma_k = np.sqrt(sigma_k)

        # Initialize A: [rank, in_features]
        A_init = mx.random.normal(shape=(rank, in_features))
        A_spectral = self._spectral_norm(A_init)
        self.lora_a = A_init * (sqrt_sigma_k / (float(A_spectral) + SQRT_EPS))

        # Initialize B: [out_features, rank]
        B_init = mx.random.normal(shape=(out_features, rank))
        B_spectral = self._spectral_norm(B_init)
        self.lora_b = B_init * (sqrt_sigma_k / (float(B_spectral) + SQRT_EPS))

        mx.eval(self.lora_a, self.lora_b)

        logger.debug(
            "Spectral init: σ_k=%.4f, ||A||=%.4f, ||B||=%.4f, target=%.4f",
            sigma_k, float(self._spectral_norm(self.lora_a)),
            float(self._spectral_norm(self.lora_b)), sqrt_sigma_k
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Base computation
        out = x @ self.base_weight.T
        if self.base_bias is not None:
            out = out + self.base_bias

        # LoRA delta: B @ A gives [out_features, in_features]
        delta = self.lora_b @ self.lora_a

        # Spectral normalization: scale by σ_k / ||B @ A||_spectral
        # This ensures the perturbation respects the spectral structure of the base weight
        spectral_norm = self._spectral_norm(delta)

        # Normalize and scale by σ_k (add epsilon for numerical stability)
        delta_normalized = delta / (spectral_norm + 1e-8)
        lora_out = x @ (self.sigma_k * delta_normalized).T

        out = out + lora_out
        return out

    def _spectral_norm(self, M: mx.array, n_iters: int = 3) -> mx.array:
        """Power iteration for spectral norm.

        Uses deterministic initialization and avoids Python if-statements
        to ensure gradients flow properly through the computation.
        """
        # Initialize with deterministic vector (sum of columns)
        # This is more stable than random init for gradient computation
        v = mx.ones((M.shape[1],)) / mx.sqrt(mx.array(M.shape[1], dtype=M.dtype))

        for _ in range(n_iters):
            u = M @ v
            # Use maximum with small epsilon to avoid division by zero
            # (preserves gradient flow unlike Python if-statement)
            u_norm = mx.maximum(mx.linalg.norm(u), mx.array(1e-8))
            u = u / u_norm

            v = M.T @ u
            v_norm = mx.maximum(mx.linalg.norm(v), mx.array(1e-8))
            v = v / v_norm

        # Spectral norm is ||M @ v||
        return mx.linalg.norm(M @ v)

    def lora_parameters(self) -> Iterator[tuple[str, mx.array]]:
        """Yield only the LoRA parameters (for training)."""
        yield "lora_a", self.lora_a
        yield "lora_b", self.lora_b


def apply_geometric_lora(
    model,
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
    rank: int | dict[str, int],
) -> dict[str, GeometricLoRALinear]:
    """Apply geometric LoRA to target modules.

    Args:
        model: The model to modify
        geometries: Pre-computed layer geometries
        target_modules: Which modules to target
        rank: Either a global LoRA rank (int) or per-layer ranks (dict).
              Use compute_per_layer_ranks() for curvature-adaptive ranks.

    Returns:
        Dict of layer_key -> GeometricLoRALinear for the modified layers
    """
    base_model = getattr(model, "model", model)
    layers = base_model.layers

    # Normalize rank to per-layer dict
    if isinstance(rank, int):
        per_layer_ranks = {key: rank for key in target_modules}
    else:
        per_layer_ranks = rank

    lora_layers = {}

    for layer_key in target_modules:
        if layer_key not in geometries:
            logger.warning("No geometry for %s, skipping", layer_key)
            continue

        if layer_key not in per_layer_ranks:
            logger.warning("No rank for %s, skipping", layer_key)
            continue

        geom = geometries[layer_key]
        layer_rank = per_layer_ranks[layer_key]

        # Parse layer key: model.layers.{idx}.self_attn.{proj}
        parts = layer_key.split(".")
        layer_idx = int(parts[2])
        proj_name = parts[4]

        layer = layers[layer_idx]
        attn = layer.self_attn
        base_linear = getattr(attn, proj_name)

        # Create geometric LoRA layer with per-layer rank
        lora_layer = GeometricLoRALinear(
            base_layer=base_linear,
            sigma_k=geom.sigma_k,
            rank=layer_rank,
        )

        # Replace in model
        setattr(attn, proj_name, lora_layer)
        lora_layers[layer_key] = lora_layer

        logger.info(
            "Applied geometric LoRA to %s: rank=%d, σ_k=%.4f, decay=%.1f×",
            layer_key, layer_rank, geom.sigma_k, geom.decay_ratio
        )

    return lora_layers


def get_lora_parameters(lora_layers: dict[str, GeometricLoRALinear]) -> dict[str, mx.array]:
    """Get all LoRA parameters for training."""
    params = {}
    for layer_key, lora_layer in lora_layers.items():
        for name, param in lora_layer.lora_parameters():
            params[f"{layer_key}.{name}"] = param
    return params


__all__ = [
    "LayerGeometry",
    "compute_layer_geometry",
    "analyze_model_geometry",
    "select_target_modules",
    "compute_geometric_rank",
    "compute_per_layer_ranks",
    "GeometricLoRALinear",
    "apply_geometric_lora",
    "get_lora_parameters",
]
