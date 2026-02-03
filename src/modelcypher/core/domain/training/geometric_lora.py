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
    """Compute LoRA rank from geometry (minimum tail_dims across targets)."""
    tail_dims = [geometries[key].tail_dims for key in target_modules if key in geometries]
    if not tail_dims:
        raise ValueError("No target modules found")
    return min(tail_dims)


class GeometricLoRALinear(nn.Module):
    """Linear layer with geometry-normalized LoRA.

    The LoRA delta is:
        delta = σ_k * (B @ A) / ||B @ A||_spectral

    Where σ_k is the smallest significant singular value of the base weight.
    This guarantees the perturbation respects the spectral structure.
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

        # Initialize LoRA matrices
        # A: [rank, in_features] - initialized small random
        # B: [out_features, rank] - initialized to zero (standard LoRA init)
        scale = 0.01
        self.lora_a = mx.random.normal(shape=(rank, in_features)) * scale
        self.lora_b = mx.zeros((out_features, rank))

    def __call__(self, x: mx.array) -> mx.array:
        # Base computation
        out = x @ self.base_weight.T
        if self.base_bias is not None:
            out = out + self.base_bias

        # LoRA delta with spectral normalization
        delta = self.lora_b @ self.lora_a  # [out_features, in_features]

        # Compute spectral norm (largest singular value)
        # For efficiency, use power iteration approximation
        spectral_norm = self._spectral_norm(delta)

        # Normalize and scale by σ_k
        if spectral_norm > 1e-8:
            delta_normalized = delta / spectral_norm
            lora_out = x @ (self.sigma_k * delta_normalized).T
            out = out + lora_out

        return out

    def _spectral_norm(self, M: mx.array, n_iters: int = 3) -> mx.array:
        """Power iteration for spectral norm."""
        # Initialize with deterministic vector (sum of columns)
        # This is more stable than random init for gradient computation
        v = mx.ones((M.shape[1],)) / mx.sqrt(mx.array(M.shape[1], dtype=M.dtype))

        for _ in range(n_iters):
            u = M @ v
            u_norm = mx.linalg.norm(u)
            if u_norm > 1e-8:
                u = u / u_norm

            v = M.T @ u
            v_norm = mx.linalg.norm(v)
            if v_norm > 1e-8:
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
    rank: int,
) -> dict[str, GeometricLoRALinear]:
    """Apply geometric LoRA to target modules.

    Args:
        model: The model to modify
        geometries: Pre-computed layer geometries
        target_modules: Which modules to target
        rank: LoRA rank (should be from compute_geometric_rank)

    Returns:
        Dict of layer_key -> GeometricLoRALinear for the modified layers
    """
    base_model = getattr(model, "model", model)
    layers = base_model.layers

    lora_layers = {}

    for layer_key in target_modules:
        if layer_key not in geometries:
            logger.warning("No geometry for %s, skipping", layer_key)
            continue

        geom = geometries[layer_key]

        # Parse layer key: model.layers.{idx}.self_attn.{proj}
        parts = layer_key.split(".")
        layer_idx = int(parts[2])
        proj_name = parts[4]

        layer = layers[layer_idx]
        attn = layer.self_attn
        base_linear = getattr(attn, proj_name)

        # Create geometric LoRA layer
        lora_layer = GeometricLoRALinear(
            base_layer=base_linear,
            sigma_k=geom.sigma_k,
            rank=rank,
        )

        # Replace in model
        setattr(attn, proj_name, lora_layer)
        lora_layers[layer_key] = lora_layer

        logger.info(
            "Applied geometric LoRA to %s: rank=%d, σ_k=%.4f",
            layer_key, rank, geom.sigma_k
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
    "GeometricLoRALinear",
    "apply_geometric_lora",
    "get_lora_parameters",
]
