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

"""Experimental: LoRA Isometry Ratio metrics.

This module implements candidate metrics for measuring geometric preservation
in LoRA adapters. These are EXPERIMENTAL and pending validation before
integration into the main codebase.

See: docs/research/lora_isometry_derivation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import svd_rank_threshold

if TYPE_CHECKING:
    from modelcypher.backends.backend import Backend
    from modelcypher.backends.array import Array


@dataclass(frozen=True)
class IsometryMetrics:
    """Isometry metrics for a single weight matrix."""

    # Spectral Preservation Ratio: how much of original spectrum is preserved
    spectral_preservation_ratio: float

    # Subspace Overlap: how much LoRA acts in base weight's subspace
    subspace_overlap: float

    # Combined Isometry Ratio: SPR × SubspaceOverlap
    isometry_ratio: float

    # Grassmann distance between subspaces (radians)
    grassmann_distance: float

    # Relative Frobenius deviation: ||ΔW||_F / ||W||_F
    relative_frobenius_deviation: float

    # Effective ranks
    rank_original: int
    rank_modified: int


def compute_isometry_metrics(
    weight_original: "Array",
    weight_modified: "Array",
    backend: "Backend | None" = None,
) -> IsometryMetrics:
    """Compute all isometry metrics for a weight matrix before/after LoRA.

    Args:
        weight_original: Original weight matrix W [out_features, in_features]
        weight_modified: Modified weight matrix W' = W + ΔW
        backend: Compute backend (uses default if None)

    Returns:
        IsometryMetrics with all computed values
    """
    if backend is None:
        backend = get_default_backend()

    # Compute ΔW
    delta_w = backend.subtract(weight_modified, weight_original)

    # SVD of original and modified
    U_orig, S_orig, Vt_orig = backend.svd(weight_original)
    U_mod, S_mod, Vt_mod = backend.svd(weight_modified)
    backend.eval(U_orig, S_orig, Vt_orig, U_mod, S_mod, Vt_mod)

    # Determine effective ranks
    max_dim = max(int(d) for d in backend.shape(weight_original))
    threshold_orig = svd_rank_threshold(backend, S_orig, max_dim)
    threshold_mod = svd_rank_threshold(backend, S_mod, max_dim)

    s_orig_list = backend.tolist(S_orig)
    s_mod_list = backend.tolist(S_mod)

    rank_orig = sum(1 for s in s_orig_list if float(s) > threshold_orig)
    rank_mod = sum(1 for s in s_mod_list if float(s) > threshold_mod)

    # 1. Spectral Preservation Ratio
    spr = _compute_spectral_preservation_ratio(s_orig_list, s_mod_list, rank_orig)

    # 2. Subspace Overlap
    subspace_overlap = _compute_subspace_overlap(
        U_orig, delta_w, rank_orig, backend
    )

    # 3. Isometry Ratio (combined metric)
    isometry_ratio = spr * subspace_overlap

    # 4. Grassmann Distance
    grassmann_dist = _compute_grassmann_distance(
        U_orig, U_mod, rank_orig, rank_mod, backend
    )

    # 5. Relative Frobenius Deviation
    rfd = _compute_relative_frobenius_deviation(weight_original, delta_w, backend)

    return IsometryMetrics(
        spectral_preservation_ratio=spr,
        subspace_overlap=subspace_overlap,
        isometry_ratio=isometry_ratio,
        grassmann_distance=grassmann_dist,
        relative_frobenius_deviation=rfd,
        rank_original=rank_orig,
        rank_modified=rank_mod,
    )


def _compute_spectral_preservation_ratio(
    s_orig: list[float],
    s_mod: list[float],
    k: int,
) -> float:
    """Compute SPR = sum_i min(σ'_i, σ_i) / sum_i σ_i for top-k."""
    if k == 0:
        return 1.0  # No spectrum to preserve

    # Truncate to k
    s_orig_k = [float(s) for s in s_orig[:k]]
    s_mod_k = [float(s) for s in s_mod[:k]]

    # Pad if modified has fewer singular values
    while len(s_mod_k) < k:
        s_mod_k.append(0.0)

    sum_orig = sum(s_orig_k)
    if sum_orig <= 0:
        return 1.0  # Empty original

    sum_preserved = sum(min(a, b) for a, b in zip(s_orig_k, s_mod_k))
    return sum_preserved / sum_orig


def _compute_subspace_overlap(
    U_orig: "Array",
    delta_w: "Array",
    k: int,
    backend: "Backend",
) -> float:
    """Compute ||U_W^T ΔW||_F / ||ΔW||_F."""
    # Frobenius norm of ΔW
    delta_norm = float(backend.to_scalar(backend.norm(delta_w)))
    if delta_norm <= 0:
        return 1.0  # No change, perfect "overlap"

    # Project ΔW onto U_orig's column space
    # U_k = U_orig[:, :k] (first k columns)
    if k > 0:
        # Use slicing instead of take for MLX compatibility
        U_k = U_orig[:, :k]
    else:
        # If no rank, return 0 overlap
        return 0.0

    # Projection: U_k^T @ ΔW
    projection = backend.matmul(backend.transpose(U_k), delta_w)
    backend.eval(projection)

    proj_norm = float(backend.to_scalar(backend.norm(projection)))
    return proj_norm / delta_norm


def _compute_grassmann_distance(
    U_orig: "Array",
    U_mod: "Array",
    k_orig: int,
    k_mod: int,
    backend: "Backend",
) -> float:
    """Compute Grassmann angle between subspaces."""
    import math

    k = min(k_orig, k_mod)
    if k == 0:
        return math.pi / 2  # Maximum distance

    # U_k for both (use slicing for MLX compatibility)
    U_orig_k = U_orig[:, :k]
    U_mod_k = U_mod[:, :k]

    # U_orig^T @ U_mod
    overlap_matrix = backend.matmul(backend.transpose(U_orig_k), U_mod_k)
    backend.eval(overlap_matrix)

    # SVD to get principal angles
    _, sigmas, _ = backend.svd(overlap_matrix)
    backend.eval(sigmas)

    sigma_list = backend.tolist(sigmas)
    if not sigma_list:
        return math.pi / 2

    # Grassmann distance = arccos of minimum singular value
    sigma_min = min(float(s) for s in sigma_list)
    # Clamp to valid arccos range
    sigma_min = max(-1.0, min(1.0, sigma_min))

    return math.acos(sigma_min)


def _compute_relative_frobenius_deviation(
    weight_original: "Array",
    delta_w: "Array",
    backend: "Backend",
) -> float:
    """Compute ||ΔW||_F / ||W||_F."""
    orig_norm = float(backend.to_scalar(backend.norm(weight_original)))
    if orig_norm <= 0:
        return float("inf")

    delta_norm = float(backend.to_scalar(backend.norm(delta_w)))
    return delta_norm / orig_norm


# =============================================================================
# Synthetic Ground Truth for Validation
# =============================================================================


@dataclass(frozen=True)
class SyntheticLoRA:
    """Synthetic LoRA with known isometry properties."""

    name: str
    weight_original: "Array"
    weight_modified: "Array"
    expected_spr: float  # Expected SPR (approximately)
    expected_overlap: float  # Expected subspace overlap


def create_synthetic_isometric_lora(
    m: int = 64,
    n: int = 32,
    rank: int = 4,
    scale: float = 0.1,
    backend: "Backend | None" = None,
) -> SyntheticLoRA:
    """Create a LoRA that scales the weight (perfectly isometric-ish).

    This LoRA does ΔW = α * W, which preserves all subspaces.
    """
    if backend is None:
        backend = get_default_backend()

    # Random base weight
    W = backend.random_normal((m, n), dtype="float32")
    backend.eval(W)

    # ΔW = scale * W (this preserves subspace completely)
    delta = backend.multiply(W, scale)
    W_mod = backend.add(W, delta)
    backend.eval(W_mod)

    return SyntheticLoRA(
        name="isometric_scale",
        weight_original=W,
        weight_modified=W_mod,
        expected_spr=1.0,  # Spectrum scaled uniformly
        expected_overlap=1.0,  # Acts entirely in original subspace
    )


def create_synthetic_random_lora(
    m: int = 64,
    n: int = 32,
    rank: int = 4,
    scale: float = 0.1,
    backend: "Backend | None" = None,
) -> SyntheticLoRA:
    """Create a random LoRA (low isometry expected)."""
    if backend is None:
        backend = get_default_backend()

    # Random base weight
    W = backend.random_normal((m, n), dtype="float32")
    backend.eval(W)

    # Random low-rank ΔW
    A = backend.random_normal((rank, n), dtype="float32")
    B = backend.random_normal((m, rank), dtype="float32")
    delta = backend.matmul(B, A)
    delta = backend.multiply(delta, scale)
    W_mod = backend.add(W, delta)
    backend.eval(W_mod)

    return SyntheticLoRA(
        name="random_lora",
        weight_original=W,
        weight_modified=W_mod,
        expected_spr=0.9,  # Mostly preserved (small perturbation)
        expected_overlap=0.5,  # Random direction, partial overlap
    )


def create_synthetic_orthogonal_lora(
    m: int = 64,
    n: int = 32,
    rank: int = 4,
    scale: float = 0.5,
    backend: "Backend | None" = None,
) -> SyntheticLoRA:
    """Create a LoRA that acts in null space of W (zero overlap expected)."""
    if backend is None:
        backend = get_default_backend()

    # Low-rank base weight
    W_A = backend.random_normal((m, rank), dtype="float32")
    W_B = backend.random_normal((rank, n), dtype="float32")
    W = backend.matmul(W_A, W_B)
    backend.eval(W)

    # Get null space of W via SVD
    U, S, Vt = backend.svd(W)
    backend.eval(U, S, Vt)

    # Use bottom singular vectors (null space) for LoRA
    # These directions are orthogonal to W's column space
    # Use slicing for MLX compatibility
    null_start = rank
    null_end = min(m, rank + 4)
    U_null = U[:, null_start:null_end]
    n_null_dims = null_end - null_start
    delta_basis = backend.random_normal((n_null_dims, n), dtype="float32")
    delta = backend.matmul(U_null, delta_basis)
    delta = backend.multiply(delta, scale)
    W_mod = backend.add(W, delta)
    backend.eval(W_mod)

    return SyntheticLoRA(
        name="orthogonal_lora",
        weight_original=W,
        weight_modified=W_mod,
        expected_spr=0.95,  # Original spectrum mostly preserved
        expected_overlap=0.0,  # Acts in null space
    )


__all__ = [
    "IsometryMetrics",
    "compute_isometry_metrics",
    "SyntheticLoRA",
    "create_synthetic_isometric_lora",
    "create_synthetic_random_lora",
    "create_synthetic_orthogonal_lora",
]
