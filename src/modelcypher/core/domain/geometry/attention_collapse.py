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

"""Attention collapse detection via per-head SVD of attention matrices.

Detects rank-1 collapse (lazy heads/layers) using IEEE 754 derived thresholds.
No tuned parameters.

Metrics:
    rank1_ratio: σ₂(A) / σ₁(A) — numerically rank-1 when < √ε_dtype
    column_mass: ||A_{·,j}||₂² / ||A||_F² per column — threshold-free
    gradient_suppression: σ₂ / √(2T) — Theorem H.1 bound on ∂L/∂W_Q, ∂L/∂W_K
    effective_rank: exp(Shannon entropy of normalized σ²) — spectral diversity

References:
    Sanyal et al. "When Attention Sinks Emerge in Generative Transformers" (TMLR 2025)
    Theorem H.1: if σ₂(A) ≤ ε√(2T), then ||∂L/∂W_Q||_F = O(ε)
    Roy & Vetterli (2007): Shannon effective rank = exp(H(σ²/Σσ²))
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# IEEE 754 machine epsilon constants.
# These are the exact values for each dtype's mantissa precision.
_EPS_F32 = math.ldexp(1.0, -23)  # float32: 23-bit mantissa
_EPS_F16 = math.ldexp(1.0, -10)  # float16: 10-bit mantissa
_EPS_BF16 = math.ldexp(1.0, -7)  # bfloat16: 7-bit mantissa

_SQRT_EPS_F32 = math.sqrt(_EPS_F32)
_SQRT_EPS_F16 = math.sqrt(_EPS_F16)
_SQRT_EPS_BF16 = math.sqrt(_EPS_BF16)

_DTYPE_SQRT_EPS: dict[str, float] = {
    "float32": _SQRT_EPS_F32,
    "float16": _SQRT_EPS_F16,
    "bfloat16": _SQRT_EPS_BF16,
}


@dataclass(frozen=True)
class AttentionCollapseResult:
    """Per-head attention collapse metrics."""

    singular_values: list[float]
    rank1_ratio: float  # σ₂ / σ₁ (0 = perfect rank-1)
    is_rank1: bool  # rank1_ratio < √ε_dtype
    column_mass: list[float]  # ||A_{·,j}||₂² / ||A||_F² per column
    gradient_suppression: float  # σ₂ / √(2T) — Theorem H.1 bound
    effective_rank: float  # exp(Shannon entropy of normalized σ²)

    def to_dict(self) -> dict[str, object]:
        return {
            "singularValues": self.singular_values,
            "rank1Ratio": self.rank1_ratio,
            "isRank1": self.is_rank1,
            "columnMass": self.column_mass,
            "gradientSuppression": self.gradient_suppression,
            "effectiveRank": self.effective_rank,
        }


@dataclass(frozen=True)
class LayerCollapseResult:
    """Per-layer collapse summary across all heads."""

    layer_idx: int
    is_collapsed: bool  # ALL heads are rank-1
    collapsed_head_count: int
    active_head_count: int
    max_effective_rank: float  # max effective rank across heads
    mean_gradient_suppression: float

    def to_dict(self) -> dict[str, object]:
        return {
            "layerIdx": self.layer_idx,
            "isCollapsed": self.is_collapsed,
            "collapsedHeadCount": self.collapsed_head_count,
            "activeHeadCount": self.active_head_count,
            "maxEffectiveRank": self.max_effective_rank,
            "meanGradientSuppression": self.mean_gradient_suppression,
        }


@dataclass(frozen=True)
class CollapseProfile:
    """Full model collapse profile."""

    total_layers: int
    attention_layers: int
    collapsed_layer_count: int
    collapse_onset_layer: int | None  # first layer where ANY head is collapsed
    layer_results: list[LayerCollapseResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "totalLayers": self.total_layers,
            "attentionLayers": self.attention_layers,
            "collapsedLayerCount": self.collapsed_layer_count,
            "collapseOnsetLayer": self.collapse_onset_layer,
            "layerResults": [lr.to_dict() for lr in self.layer_results],
        }


def compute_attention_collapse(
    attention_matrix: list[list[float]],
    dtype: str = "float32",
) -> AttentionCollapseResult:
    """Compute collapse metrics for a single attention head matrix.

    Args:
        attention_matrix: T×T post-softmax attention weights (list of lists).
        dtype: Model dtype for IEEE 754 threshold derivation.

    Returns:
        AttentionCollapseResult with SVD-based metrics.
    """
    import numpy as np

    A = np.array(attention_matrix, dtype=np.float64)
    T = A.shape[0]

    # SVD (compute_uv=False for singular values only)
    sv = np.linalg.svd(A, compute_uv=False)
    sv_list = [float(s) for s in sv]

    # Rank-1 ratio: σ₂ / σ₁
    sigma1 = sv_list[0] if sv_list else 1.0
    sigma2 = sv_list[1] if len(sv_list) > 1 else 0.0
    rank1_ratio = sigma2 / sigma1 if sigma1 > 0 else 0.0

    # Rank-1 threshold from IEEE 754 machine epsilon for the model dtype
    sqrt_eps = _DTYPE_SQRT_EPS.get(dtype, _SQRT_EPS_F32)
    is_rank1 = rank1_ratio < sqrt_eps

    # Column mass: ||A_{·,j}||₂² / ||A||_F² per column
    col_norms_sq = np.sum(A ** 2, axis=0)
    frob_sq = float(np.sum(col_norms_sq))
    column_mass = (col_norms_sq / frob_sq).tolist() if frob_sq > 0 else [0.0] * T

    # Gradient suppression (Theorem H.1, Sanyal et al.):
    # If σ₂(A) ≤ ε√(2T), then ||∂L/∂W_Q||_F = O(ε).
    # We report σ₂ / √(2T) as the suppression factor.
    gradient_suppression = sigma2 / math.sqrt(2.0 * T)

    # Effective rank: exp(Shannon entropy of normalized σ²)
    # Roy & Vetterli (2007)
    sv_sq = sv ** 2
    sv_sq_sum = float(np.sum(sv_sq))
    if sv_sq_sum > 0:
        p = sv_sq / sv_sq_sum
        p = p[p > 0]  # filter zeros for log
        entropy = float(-np.sum(p * np.log(p)))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    return AttentionCollapseResult(
        singular_values=sv_list,
        rank1_ratio=rank1_ratio,
        is_rank1=is_rank1,
        column_mass=column_mass,
        gradient_suppression=gradient_suppression,
        effective_rank=effective_rank,
    )


def summarize_layer_collapse(
    head_results: list[AttentionCollapseResult],
    layer_idx: int = 0,
) -> LayerCollapseResult:
    """Summarize collapse across all heads in a layer.

    Args:
        head_results: Per-head collapse results.
        layer_idx: The layer index (caller provides).

    Returns:
        LayerCollapseResult summarizing the layer.
    """
    collapsed = sum(1 for h in head_results if h.is_rank1)
    active = len(head_results) - collapsed
    max_eff_rank = max((h.effective_rank for h in head_results), default=0.0)
    mean_grad_supp = (
        sum(h.gradient_suppression for h in head_results) / len(head_results)
        if head_results
        else 0.0
    )

    return LayerCollapseResult(
        layer_idx=layer_idx,
        is_collapsed=(collapsed == len(head_results) and len(head_results) > 0),
        collapsed_head_count=collapsed,
        active_head_count=active,
        max_effective_rank=max_eff_rank,
        mean_gradient_suppression=mean_grad_supp,
    )


def compute_collapse_profile(
    layer_results: list[LayerCollapseResult],
) -> CollapseProfile:
    """Compute full model collapse profile.

    Args:
        layer_results: Per-layer collapse results (attention layers only).

    Returns:
        CollapseProfile summarizing the model.
    """
    collapsed_count = sum(1 for lr in layer_results if lr.is_collapsed)
    onset = None
    for lr in layer_results:
        if lr.collapsed_head_count > 0:
            onset = lr.layer_idx
            break

    return CollapseProfile(
        total_layers=len(layer_results),
        attention_layers=len(layer_results),
        collapsed_layer_count=collapsed_count,
        collapse_onset_layer=onset,
        layer_results=layer_results,
    )
