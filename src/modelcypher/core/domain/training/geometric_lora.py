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

"""Geometry-derived LoRA configuration and analysis.

All parameters are derived from the spectral structure of base weights:
- Target modules: where effective_rank > 0 (at least one SV above noise floor)
- Scale: σ_k(W) per layer (smallest significant singular value)
- Rank: bounded by tail_dims = full_rank - rank_90

No hyperparameters. The geometry IS the configuration.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific LoRA implementations live in adapters/training/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    log_scalar,
    sqrt_scalar,
)
from modelcypher.ports.training import LoRALayerConfig

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

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
        """Whether this layer is safe to target for LoRA.

        A layer is targetable if it has effective_rank > 0, meaning at least
        one singular value is above the dtype-derived noise floor (sqrt(eps) × σ_max).

        This is equivalent to requiring decay_ratio < 1/sqrt(eps) ≈ 2900 for float32.
        The bound is derived from the definition of significant singular values:
        σ_k > sqrt(eps) × σ_max implies σ_max/σ_k < 1/sqrt(eps).

        No magic numbers—the geometry determines targetability.
        """
        return self.effective_rank > 0


def compute_layer_geometry(
    weight: "Array",
    layer_key: str,
    backend: "Backend",
) -> LayerGeometry:
    """Compute spectral geometry of a weight matrix using Backend protocol.

    Args:
        weight: Weight matrix [out_features, in_features].
        layer_key: Identifier for this layer.
        backend: Backend for tensor operations.

    Returns:
        LayerGeometry with all spectral information.
    """
    b = backend

    # Ensure float32 for SVD stability
    W = b.astype(weight, "float32")
    b.eval(W)

    shape = (int(W.shape[0]), int(W.shape[1]))
    full_rank = min(shape)

    # Full SVD
    U, S, Vt = b.svd(W, compute_uv=True)
    b.eval(S)

    n_svs = int(S.shape[0])
    if n_svs == 0:
        return LayerGeometry(
            layer_key=layer_key,
            shape=shape,
            sigma_max=0.0,
            sigma_k=0.0,
            effective_rank=0,
            full_rank=full_rank,
            decay_ratio=float("inf"),
            rank_90=0,
            tail_dims=full_rank,
        )

    # Extract singular values
    sigma_max = float(b.to_scalar(S[0]))
    sigma_min = float(b.to_scalar(S[n_svs - 1]))

    # Noise threshold from numerical precision: sqrt(eps) * sigma_max
    sqrt_eps = division_epsilon(b, S)
    threshold = sqrt_eps * sigma_max

    # Effective rank (SVs above noise floor)
    significant_mask = S > threshold
    significant_count = b.sum(b.astype(significant_mask, "int32"))
    b.eval(significant_count)
    effective_rank = int(b.to_scalar(significant_count))

    # Find sigma_k (smallest significant SV)
    if effective_rank > 0:
        # Get the effective_rank-1 index (0-indexed)
        idx = min(effective_rank - 1, n_svs - 1)
        sigma_k = float(b.to_scalar(S[idx]))
    else:
        sigma_k = sigma_min

    # Ensure sigma_k is positive
    if sigma_k <= 0:
        sigma_k = sigma_min if sigma_min > 0 else sqrt_eps

    # Energy distribution for rank_90
    S_sq = S * S
    total_energy = b.sum(S_sq)
    b.eval(total_energy)
    total_energy_val = float(b.to_scalar(total_energy))

    if total_energy_val > 0:
        cumsum = b.cumsum(S_sq)
        b.eval(cumsum)

        # Find rank_90 (SVs needed for 90% energy)
        threshold_energy = 0.90 * total_energy_val
        above_threshold = cumsum >= threshold_energy
        above_count = b.sum(b.astype(above_threshold, "int32"))
        b.eval(above_count)

        # rank_90 is where cumsum first exceeds 90%
        # If k values are above threshold, rank_90 = n_svs - k + 1
        n_above = int(b.to_scalar(above_count))
        rank_90 = n_svs - n_above + 1 if n_above > 0 else n_svs
    else:
        rank_90 = 1

    rank_90 = max(1, min(rank_90, full_rank))
    tail_dims = full_rank - rank_90

    decay_ratio = sigma_max / sigma_k if sigma_k > 0 else float("inf")

    return LayerGeometry(
        layer_key=layer_key,
        shape=shape,
        sigma_max=sigma_max,
        sigma_k=sigma_k,
        effective_rank=effective_rank,
        full_rank=full_rank,
        decay_ratio=decay_ratio,
        rank_90=rank_90,
        tail_dims=tail_dims,
    )


def analyze_weight_geometries(
    weights: dict[str, "Array"],
    backend: "Backend",
) -> dict[str, LayerGeometry]:
    """Analyze spectral geometry of all weight matrices.

    Args:
        weights: Dict mapping layer_key -> weight array.
        backend: Backend for tensor operations.

    Returns:
        Dict mapping layer_key -> LayerGeometry.
    """
    geometries = {}

    for layer_key, weight in weights.items():
        try:
            geometry = compute_layer_geometry(weight, layer_key, backend)
            geometries[layer_key] = geometry

            logger.debug(
                "%s: decay=%.1f×, σ_k=%.4f, tail=%d, targetable=%s",
                layer_key,
                geometry.decay_ratio,
                geometry.sigma_k,
                geometry.tail_dims,
                geometry.is_targetable,
            )
        except Exception as e:
            logger.warning("Failed to analyze layer %s: %s", layer_key, e)
            continue

    return geometries


def select_target_modules(
    geometries: dict[str, LayerGeometry],
) -> list[str]:
    """Select modules to target based on geometry.

    Returns layers with effective_rank > 0 (at least one singular value
    above the noise floor). No arbitrary thresholds.

    Args:
        geometries: Pre-computed layer geometries.

    Returns:
        List of layer keys that are safe to target.
    """
    return [key for key, geom in geometries.items() if geom.is_targetable]


def compute_geometric_rank(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
) -> int:
    """Compute global LoRA rank from geometry (minimum tail_dims across targets).

    DEPRECATED: Use compute_per_layer_ranks() for curvature-adaptive ranks.

    Args:
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to consider.

    Returns:
        Global LoRA rank (minimum tail_dims).
    """
    tail_dims = [
        geometries[key].tail_dims for key in target_modules if key in geometries
    ]
    if not tail_dims:
        raise ValueError("No target modules found")
    return min(tail_dims)


def compute_per_layer_ranks(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
    backend: "Backend",
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
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to compute ranks for.
        backend: Backend for scalar operations.
        min_rank: Minimum rank (numerical stability).
        max_rank: Maximum rank (memory constraint).

    Returns:
        Dict of layer_key -> rank.
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
    log_sum = 0.0
    for d in decays.values():
        log_sum += log_scalar(d, backend)
    mean_log_decay = log_sum / len(decays)
    mean_decay = exp_scalar(mean_log_decay, backend)

    # Base rank is the minimum tail_dims (conservative starting point)
    base_rank = min(tail_dims_map.values())

    # Compute per-layer ranks
    per_layer_ranks = {}
    for key, decay in decays.items():
        # Curvature factor: higher for low-decay (complex) layers
        # sqrt dampens the effect to avoid extreme values
        curvature_factor = sqrt_scalar(mean_decay / decay, backend)

        # Scale base rank by curvature factor
        raw_rank = base_rank * curvature_factor

        # Clamp to bounds, respecting layer's tail_dims
        layer_max = min(max_rank, tail_dims_map[key])
        rank = max(min_rank, min(int(raw_rank), layer_max))

        per_layer_ranks[key] = rank

        logger.debug(
            "%s: decay=%.1f, factor=%.2f, rank=%d (tail=%d)",
            key,
            decay,
            curvature_factor,
            rank,
            tail_dims_map[key],
        )

    return per_layer_ranks


def derive_lora_configs(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
    backend: "Backend",
    adaptive_rank: bool = True,
    min_rank: int = 4,
    max_rank: int = 64,
) -> list[LoRALayerConfig]:
    """Derive LoRA configurations from layer geometries.

    Args:
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to target.
        backend: Backend for scalar operations.
        adaptive_rank: If True, use per-layer adaptive ranks.
        min_rank: Minimum rank.
        max_rank: Maximum rank.

    Returns:
        List of LoRALayerConfig for each target module.
    """
    if adaptive_rank:
        per_layer_ranks = compute_per_layer_ranks(
            geometries, target_modules, backend, min_rank, max_rank
        )
    else:
        global_rank = compute_geometric_rank(geometries, target_modules)
        per_layer_ranks = {key: global_rank for key in target_modules}

    configs = []
    for key in target_modules:
        if key not in geometries:
            continue

        geom = geometries[key]
        rank = per_layer_ranks.get(key, min_rank)

        configs.append(
            LoRALayerConfig(
                layer_key=key,
                rank=rank,
                sigma_k=geom.sigma_k,
                in_features=geom.shape[1],
                out_features=geom.shape[0],
            )
        )

    return configs


__all__ = [
    "LayerGeometry",
    "analyze_weight_geometries",
    "compute_geometric_rank",
    "compute_layer_geometry",
    "compute_per_layer_ranks",
    "derive_lora_configs",
    "select_target_modules",
]
