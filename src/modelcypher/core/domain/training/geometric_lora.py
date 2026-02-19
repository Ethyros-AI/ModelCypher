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
- Target modules: where tail_dims > 0 (non-zero null-space capacity)
- sigma_k: SV at the edge of the informationally significant subspace
  (derived from Shannon effective rank, not numerical precision)
- Rank: bounded by tail_dims = full_rank - floor(shannon_eff_rank)

No hyperparameters. The geometry IS the configuration.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific LoRA implementations live in adapters/training/.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    safe_log_epsilon,
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
    sigma_k: float  # SV at Shannon eff rank boundary (structural, not precision)
    effective_rank: int  # Precision-based: count(S > sqrt(eps) * sigma_max)
    full_rank: int
    decay_ratio: float  # σ_max / σ_k
    tail_dims: int  # full_rank - floor(shannon_eff_rank) (structural null-space)
    shannon_effective_rank: float  # exp(H(σ²)) - continuous spectral utilization
    spectral_gap: float  # σ_{k-1} - σ_k at structural rank boundary (Weyl crossing threshold)

    @property
    def is_targetable(self) -> bool:
        """Whether this layer is safe to target for LoRA.

        A layer is targetable if it has non-zero null-space capacity:
        tail_dims > 0 after precision-derived rank estimation.
        """
        return self.tail_dims > 0


def compute_layer_geometry(
    weight: "Array",
    layer_key: str,
    backend: "Backend",
) -> LayerGeometry:
    """Compute spectral geometry of a weight matrix using Backend protocol.

    sigma_k and tail_dims are derived from Shannon effective rank (spectral
    entropy), which measures how many dimensions carry meaningful information.
    This is a structural measure, not a numerical precision threshold.

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
            tail_dims=full_rank,
            shannon_effective_rank=0.0,
            spectral_gap=0.0,
        )

    # Extract singular values
    sigma_max = float(b.to_scalar(S[0]))
    sigma_min = float(b.to_scalar(S[n_svs - 1]))

    # Precision-based effective rank (SVs above numerical noise floor)
    sqrt_eps = division_epsilon(b, S)
    threshold = sqrt_eps * sigma_max
    significant_mask = S > threshold
    significant_count = b.sum(b.astype(significant_mask, "int32"))
    b.eval(significant_count)
    effective_rank = int(b.to_scalar(significant_count))

    # Shannon effective rank: exp(H(σ²)) where H is spectral entropy
    # Continuous measure of spectral utilization (Roy & Vetterli 2007)
    # This is the STRUCTURAL rank — how many dimensions carry significant energy
    eigvals = S * S  # σ² = eigenvalues of W^T W
    sum_eig = b.sum(eigvals)
    b.eval(sum_eig)
    sum_eig_val = float(b.to_scalar(sum_eig))
    eps = division_epsilon(b, eigvals)

    if sum_eig_val > eps:
        p = eigvals / sum_eig
        log_eps = safe_log_epsilon(b, eigvals)
        eps_arr = b.full(p.shape, log_eps, dtype=p.dtype)
        p_safe = b.where(p > log_eps, p, eps_arr)
        entropy = b.sum(-p * b.log(p_safe))
        shannon_eff_rank = b.exp(entropy)
        b.eval(shannon_eff_rank)
        shannon_effective_rank = float(b.to_scalar(shannon_eff_rank))
    else:
        shannon_effective_rank = 0.0

    # sigma_k and tail_dims derive from Shannon effective rank (structural),
    # not the precision-based effective_rank. The precision threshold
    # sqrt(eps) * sigma_max only tells us what float32 can distinguish from zero.
    # Shannon eff rank tells us how many dimensions carry meaningful information.
    structural_rank = max(1, min(math.floor(shannon_effective_rank), n_svs - 1))

    # sigma_k = SV at the edge of the informationally significant subspace
    sigma_k = float(b.to_scalar(S[structural_rank - 1]))

    # Ensure sigma_k is positive
    if sigma_k <= 0:
        sigma_k = sigma_min if sigma_min > 0 else sqrt_eps

    # Spectral gap at the structural rank boundary (Weyl crossing threshold)
    # gap_k = σ_{k-1} - σ_k measures how far apart adjacent singular values are
    # at the edge of the informationally significant subspace.
    # Weyl (1912): perturbation crossing occurs when ||E||_2 > gap_k / 2.
    if structural_rank >= 2:
        sigma_k_prev = float(b.to_scalar(S[structural_rank - 2]))
        spectral_gap = sigma_k_prev - sigma_k
    else:
        spectral_gap = sigma_k  # Only one significant SV; gap = sigma_k itself

    # Null-space capacity: dimensions beyond the structural rank
    tail_dims = max(0, full_rank - structural_rank)

    decay_ratio = sigma_max / sigma_k if sigma_k > 0 else float("inf")

    return LayerGeometry(
        layer_key=layer_key,
        shape=shape,
        sigma_max=sigma_max,
        sigma_k=sigma_k,
        effective_rank=effective_rank,
        full_rank=full_rank,
        decay_ratio=decay_ratio,
        tail_dims=tail_dims,
        shannon_effective_rank=shannon_effective_rank,
        spectral_gap=spectral_gap,
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

    Returns layers with non-zero null-space capacity (tail_dims > 0).
    No arbitrary thresholds.

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
) -> dict[str, int]:
    """Compute per-layer LoRA ranks from null-space capacity.

    Rank is derived directly from precision-limited capacity:
    rank_i = tail_dims_i (full_rank - effective_rank).

    Args:
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to compute ranks for.

    Returns:
        Dict of layer_key -> rank.
    """
    if not target_modules:
        raise ValueError("No target modules provided")

    per_layer_ranks: dict[str, int] = {}
    for key in target_modules:
        geom = geometries.get(key)
        if geom is None:
            continue
        per_layer_ranks[key] = max(0, int(geom.tail_dims))

    if not per_layer_ranks:
        raise ValueError("No geometries found for target modules")

    return per_layer_ranks


def compute_coupled_ranks(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
) -> dict[str, int]:
    """Compute per-layer LoRA ranks with cross-projection coupling.

    In multi-head attention, queries can only learn to look for features
    that keys can distinguish. Excess query rank with no corresponding
    key rank creates noise in the attention pattern.

    Coupling rule per attention layer:
        rank(q_proj) = min(tail_dims_q, tail_dims_k)
        rank(k_proj) = tail_dims_k  (unchanged)
        rank(v_proj) = tail_dims_v  (unchanged)
        MLP projections: unchanged (tail_dims)

    Keys expected in format: model.layers.{idx}.self_attn.{proj}.weight

    Args:
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to compute ranks for.

    Returns:
        Dict of layer_key -> rank (with q_proj capped by k_proj capacity).
    """
    if not target_modules:
        raise ValueError("No target modules provided")

    # Start with uncoupled ranks
    per_layer_ranks = compute_per_layer_ranks(geometries, target_modules)

    # Group attention projections by layer index
    layer_attn_keys: dict[int, dict[str, str]] = {}
    for key in target_modules:
        if ".self_attn." not in key:
            continue
        parts = key.split(".")
        try:
            layer_idx = int(parts[2])
        except (IndexError, ValueError):
            logger.warning(
                "Cannot parse layer index from attention key '%s' "
                "(expected model.layers.{idx}.self_attn.{proj}.weight); "
                "rank coupling skipped for this key",
                key,
            )
            continue
        for part in parts:
            if part in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layer_attn_keys.setdefault(layer_idx, {})[part] = key
                break

    # Apply coupling: cap q_proj rank at k_proj tail_dims
    n_coupled = 0
    for layer_idx, proj_keys in sorted(layer_attn_keys.items()):
        q_key = proj_keys.get("q_proj")
        k_key = proj_keys.get("k_proj")

        if q_key is None or k_key is None:
            continue
        if q_key not in per_layer_ranks or k_key not in per_layer_ranks:
            continue

        q_rank = per_layer_ranks[q_key]
        k_rank = per_layer_ranks[k_key]

        if q_rank > k_rank:
            logger.info(
                "Rank coupling layer %d: q_proj %d -> %d (capped by k_proj tail_dims)",
                layer_idx, q_rank, k_rank,
            )
            per_layer_ranks[q_key] = k_rank
            n_coupled += 1

    if n_coupled > 0:
        logger.info(
            "Cross-projection rank coupling: %d layers capped",
            n_coupled,
        )

    return per_layer_ranks


def compute_geometric_dropout(geometry: LayerGeometry, rank: int) -> float:
    """Derive per-layer dropout rate from weight spectral geometry.

    Dropout acts as a low-rank regularizer (Cavazza et al., AISTATS 2018).
    The rate is the product of two spectral ratios — no arbitrary constants:

        dropout = redundancy × adapter_fraction

    Where:
        - redundancy = 1 - shannon_eff_rank / full_rank
          (how concentrated the spectrum is; 0 = flat, 1 = single SV)
        - adapter_fraction = rank / full_rank
          (how much of the weight's space the LoRA adapter occupies)

    Both factors are ratios of measured geometric quantities.

    Geometric interpretation: with this dropout, the expected active LoRA
    dimensions per training step are rank × (1 - redundancy × rank/full_rank).
    When the spectrum is flat (redundancy ≈ 0), all LoRA dims stay active.
    When the spectrum is steep and rank is large relative to full_rank,
    more dims are dropped — the adapter's effective training dimensionality
    reflects the weight's spectral utilization.

    For NB-LoRA (Cayley transform), dropout = 0.0 because the spectral bound
    is a strictly tighter constraint than dropout's nuclear norm regularization.

    Args:
        geometry: Pre-computed spectral geometry for this layer.
        rank: LoRA rank assigned to this layer.

    Returns:
        Dropout rate (product of two spectral ratios).

    Reference:
        Cavazza, J. et al. (2018). Dropout as a Low-Rank Regularizer. AISTATS.
        Roy, O. & Vetterli, M. (2007). Effective rank definition.
    """
    if geometry.full_rank == 0 or rank <= 1:
        return 0.0

    # Spectral redundancy: how concentrated the energy is (not uniformly spread)
    utilization = geometry.shannon_effective_rank / geometry.full_rank
    utilization = max(0.0, min(1.0, utilization))
    redundancy = 1.0 - utilization

    # Adapter fraction: how much of the total space LoRA occupies
    adapter_fraction = rank / geometry.full_rank

    # Product of two spectral ratios — both purely geometric
    dropout = redundancy * adapter_fraction

    # Mathematical constraint: at least 1 rank dimension must survive
    p_max_from_rank = 1.0 - 1.0 / rank
    dropout = min(dropout, p_max_from_rank)

    return round(dropout, 4)


def derive_lora_configs(
    geometries: dict[str, LayerGeometry],
    target_modules: list[str],
    adaptive_rank: bool = True,
) -> list[LoRALayerConfig]:
    """Derive LoRA configurations from layer geometries.

    Args:
        geometries: Pre-computed layer geometries.
        target_modules: Which modules to target.
        adaptive_rank: If True, use per-layer ranks from null-space capacity.

    Returns:
        List of LoRALayerConfig for each target module.
    """
    if adaptive_rank:
        per_layer_ranks = compute_per_layer_ranks(geometries, target_modules)
    else:
        global_rank = compute_geometric_rank(geometries, target_modules)
        per_layer_ranks = {key: global_rank for key in target_modules}

    configs = []
    for key in target_modules:
        if key not in geometries:
            continue

        geom = geometries[key]
        rank = per_layer_ranks.get(key, 0)

        dropout = compute_geometric_dropout(geom, rank) if rank > 0 else 0.0

        configs.append(
            LoRALayerConfig(
                layer_key=key,
                rank=rank,
                sigma_k=geom.sigma_k,
                in_features=geom.shape[1],
                out_features=geom.shape[0],
                dropout=dropout,
            )
        )

    return configs


__all__ = [
    "LayerGeometry",
    "analyze_weight_geometries",
    "compute_coupled_ranks",
    "compute_geometric_dropout",
    "compute_geometric_rank",
    "compute_layer_geometry",
    "compute_per_layer_ranks",
    "derive_lora_configs",
    "select_target_modules",
]
