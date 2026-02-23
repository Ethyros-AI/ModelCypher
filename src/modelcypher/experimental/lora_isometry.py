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
from modelcypher.util.math_utils import compute_coefficient_of_variation
import math

if TYPE_CHECKING:
    from modelcypher.backends.backend import Backend
    from modelcypher.backends.array import Array


@dataclass(frozen=True)
class IsometryMetrics:
    """Isometry metrics for a single weight matrix."""

    # Spectral Preservation Ratio: how much of original spectrum is preserved
    spectral_preservation_ratio: float

    # Spectral Angle: rotation of top-k singular vectors (degrees)
    # 0° = perfect alignment, 90° = orthogonal (maximum deviation)
    spectral_angle: float

    # Condition number ratio: κ(W') / κ(W)
    # 1.0 = preserved, <1 = regularized (better), >1 = worse conditioning
    condition_ratio: float

    # Grassmann distance between subspaces (radians)
    grassmann_distance: float

    # Relative Frobenius deviation: ||ΔW||_F / ||W||_F
    relative_frobenius_deviation: float

    # Effective ranks
    rank_original: int
    rank_modified: int

    # Legacy: Combined Isometry Ratio (deprecated, kept for compatibility)
    isometry_ratio: float


@dataclass(frozen=True)
class WeylMetrics:
    """Deterministic LoRA geometry metrics based on Weyl perturbation theory.

    The key insight: Weyl's inequality states that for a perturbation ΔW,
    |σᵢ(W+ΔW) - σᵢ(W)| ≤ ‖ΔW‖₂

    The Weyl Utilization η measures what fraction of this bound is actually used:
        η = max|Δσᵢ| / ‖ΔW‖₂

    Values:
        - η ≈ 1 (100%): ΔW is aligned with W's principal structure (BAD)
        - η ≈ 0 (0%): ΔW is orthogonal to W's principal structure (GOOD)

    A well-trained LoRA has η < 1% (acts in unused dimensions).
    """

    # Weyl Utilization: η = max|Δσᵢ| / ‖ΔW‖₂
    # Range [0, 1]. Lower is better.
    weyl_utilization: float

    # Spectral norm of the LoRA perturbation
    delta_spectral_norm: float

    # Maximum singular value change (observed)
    max_singular_value_change: float

    # Which singular value had the largest change
    max_change_index: int

    # Effective rank of ΔW (should match LoRA rank)
    delta_effective_rank: int

    # r_eff: dimensions of W needed to capture 50% of ΔW's projection
    # High r_eff = diffuse (good), low r_eff = concentrated (bad)
    projection_depth: int

    def max_scale_for_bound(self, epsilon: float) -> float:
        """Compute maximum scale α to keep max|Δσ| < ε.

        Formula: α ≤ ε / (η · ‖ΔW‖₂)

        Args:
            epsilon: Maximum allowed singular value change

        Returns:
            Maximum safe scale factor
        """
        denominator = self.weyl_utilization * self.delta_spectral_norm
        if denominator <= 0:
            return float("inf")
        return epsilon / denominator


@dataclass(frozen=True)
class SpectralSelectivity:
    """Metric measuring how selectively ΔW interacts with W's singular vectors.

    Trained LoRAs exhibit "spiky" interaction profiles (amplifying some
    directions, suppressing others), while random LoRAs are more uniform.

    The primary metric is the Coefficient of Variation (CV) of the amplification
    ratios across W's top singular vectors.
    """

    # Coefficient of Variation of ||ΔW v_k|| norms
    # High CV = High Selectivity (Trained)
    # Low CV = Low Selectivity (Random)
    amplification_cv: float

    # Max amplification ratio (max_k ||ΔW v_k|| / mean)
    max_amplification: float

    # Min amplification ratio (min_k ||ΔW v_k|| / mean)
    min_amplification: float

    # Index of the vector with max amplification (semantic hot-spot)
    max_amplification_index: int


@dataclass(frozen=True)
class SpectralSelectivityProfile:
    """Raw amplification profile for spectral selectivity analysis.

    Provides per-direction amplification norms and ratios so downstream
    analysis can build full histograms without summary aggregation.
    """

    # Raw ||ΔW v_k|| values for each singular direction k
    amplification_norms: list[float]

    # Per-k ratios: ||ΔW v_k|| / mean(||ΔW v||)
    amplification_ratios: list[float]

    # Mean amplification used for ratio normalization
    mean_amplification: float


def compute_spectral_selectivity_profile(
    weight_original: "Array",
    delta_w: "Array",
    k: int | None = None,
    backend: "Backend | None" = None,
) -> SpectralSelectivityProfile:
    """Return per-direction amplification norms and ratios.

    Args:
        weight_original: Base weight W
        delta_w: LoRA perturbation ΔW
        k: Number of singular vectors to check (None = all)
        backend: Compute backend

    Returns:
        SpectralSelectivityProfile with raw norms and ratios.
    """
    if backend is None:
        backend = get_default_backend()

    # Get W's right singular vectors (V)
    _, _, Vt = backend.svd(weight_original)

    total_dirs = int(Vt.shape[0])
    if k is None or k <= 0 or k > total_dirs:
        k = total_dirs

    # Rows of Vt are v_k directions
    Vt_k = Vt[:k, :]

    # delta_w shape (m, n), Vt_k.T shape (n, k) -> (m, k)
    out = backend.matmul(delta_w, backend.transpose(Vt_k))
    out_sq = backend.multiply(out, out)
    norms_sq = backend.sum(out_sq, axis=0)  # (k,)
    norms = backend.sqrt(norms_sq)

    backend.eval(norms)
    norm_values = [float(x) for x in backend.tolist(norms)]

    if not norm_values:
        return SpectralSelectivityProfile(
            amplification_norms=[],
            amplification_ratios=[],
            mean_amplification=0.0,
        )

    mean_val = sum(norm_values) / len(norm_values)
    if mean_val > 0:
        ratios = [v / mean_val for v in norm_values]
    else:
        ratios = [0.0 for _ in norm_values]

    return SpectralSelectivityProfile(
        amplification_norms=norm_values,
        amplification_ratios=ratios,
        mean_amplification=mean_val,
    )


def compute_spectral_selectivity(
    weight_original: "Array",
    delta_w: "Array",
    k: int = 64,
    backend: "Backend | None" = None,
) -> SpectralSelectivity:
    """Compute spectral selectivity metrics.

    Args:
        weight_original: Base weight W
        delta_w: LoRA perturbation ΔW
        k: Number of singular vectors to check (default 64)
        backend: Compute backend

    Returns:
        SpectralSelectivity metrics
    """
    if backend is None:
        backend = get_default_backend()

    # Get W's right singular vectors (V)
    # W = U S V^T -> Right singular vectors are rows of Vt (or cols of V)
    # MLX svd returns U, S, Vt
    _, _, Vt = backend.svd(weight_original)
    
    # We want rows of Vt (which are the columns of V, i.e., input directions)
    # Take top k rows
    Vt_k = Vt[:k, :] # Shape (k, n)

    # Compute amplification: ||ΔW v_k|| for each k
    # v_k are rows of Vt_k
    # We can batch this: out = ΔW @ Vt_k.T -> Shape (m, k)
    # Then independent norms of columns of out
    
    # delta_w shape (m, n), Vt_k.T shape (n, k) -> (m, k)
    out = backend.matmul(delta_w, backend.transpose(Vt_k))
    
    # Compute column norms (axis 0)
    # MLX norm reduces, so careful with axes
    # Actually simpler: out^2 then sum cols then sqrt
    out_sq = backend.multiply(out, out)
    norms_sq = backend.sum(out_sq, axis=0) # Shape (k,)
    norms = backend.sqrt(norms_sq)
    
    backend.eval(norms)
    norm_values = [float(x) for x in backend.tolist(norms)]
    
    cv = compute_coefficient_of_variation(norm_values)
    mean_val = sum(norm_values) / len(norm_values) if norm_values else 1.0
    max_val = max(norm_values) if norm_values else 0.0
    min_val = min(norm_values) if norm_values else 0.0
    max_idx = norm_values.index(max_val) if norm_values else 0

    return SpectralSelectivity(
        amplification_cv=cv,
        max_amplification=max_val / mean_val if mean_val > 0 else 0.0,
        min_amplification=min_val / mean_val if mean_val > 0 else 0.0,
        max_amplification_index=max_idx,
    )
    

def compute_weyl_utilization(
    weight_original: "Array",
    delta_w: "Array",
    backend: "Backend | None" = None,
) -> WeylMetrics:
    """Compute Weyl utilization metrics for a LoRA perturbation.

    This is the DETERMINISTIC measure of how much ΔW interferes with
    W's principal structure.

    Args:
        weight_original: Original weight matrix W
        delta_w: LoRA perturbation ΔW = A @ B
        backend: Compute backend

    Returns:
        WeylMetrics with all geometric measurements
    """
    if backend is None:
        backend = get_default_backend()

    # SVD of delta_w
    _, S_delta, _ = backend.svd(delta_w)
    backend.eval(S_delta)
    s_delta = backend.tolist(S_delta)
    delta_spectral_norm = float(s_delta[0]) if s_delta else 0.0

    # Effective rank of delta (SVD rank threshold: max(m,n) × ε_mach)
    max_dim = max(int(d) for d in backend.shape(delta_w))
    delta_threshold = svd_rank_threshold(backend, S_delta, max_dim)
    delta_eff_rank = sum(1 for s in s_delta if float(s) > delta_threshold)

    # SVD of original and modified
    U_orig, S_orig, _ = backend.svd(weight_original)
    backend.eval(U_orig, S_orig)
    s_orig = [float(s) for s in backend.tolist(S_orig)]

    weight_modified = backend.add(weight_original, delta_w)
    _, S_mod, _ = backend.svd(weight_modified)
    backend.eval(S_mod)
    s_mod = [float(s) for s in backend.tolist(S_mod)]

    # Find max singular value change
    n = min(len(s_orig), len(s_mod))
    changes = [(i, abs(s_mod[i] - s_orig[i])) for i in range(n)]
    max_idx, max_change = max(changes, key=lambda x: x[1])

    # Weyl utilization
    weyl_util = max_change / delta_spectral_norm if delta_spectral_norm > 0 else 0.0

    # Projection depth: smallest k where cumulative projection energy ≥ 50% of ||ΔW||²_F
    # Per-direction energy: ||u_i^T ΔW||² (each row of U_orig^T @ ΔW)
    # 50% is the median of the energy distribution — a definition, not a tuned threshold.
    delta_frob_sq = float(backend.to_scalar(backend.norm(delta_w))) ** 2
    projection_depth = 0
    if delta_frob_sq > 0:
        proj_full = backend.matmul(backend.transpose(U_orig), delta_w)
        proj_sq = backend.multiply(proj_full, proj_full)
        per_dir_energy = backend.sum(proj_sq, axis=1)
        backend.eval(per_dir_energy)
        energies = [float(e) for e in backend.tolist(per_dir_energy)]
        cumulative = 0.0
        for k, e in enumerate(energies, 1):
            cumulative += e
            if cumulative / delta_frob_sq >= 0.5:
                projection_depth = k
                break

    return WeylMetrics(
        weyl_utilization=weyl_util,
        delta_spectral_norm=delta_spectral_norm,
        max_singular_value_change=max_change,
        max_change_index=max_idx,
        delta_effective_rank=delta_eff_rank,
        projection_depth=projection_depth,
    )


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

    # SVD of original and modified (eval separately to avoid LAPACK crash
    # when MLX batches two large SVDs concurrently)
    U_orig, S_orig, Vt_orig = backend.svd(weight_original)
    backend.eval(U_orig, S_orig, Vt_orig)
    U_mod, S_mod, Vt_mod = backend.svd(weight_modified)
    backend.eval(U_mod, S_mod, Vt_mod)

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

    # 2. Spectral Angle (rotation of top-k singular vectors)
    spectral_angle = _compute_spectral_angle(
        U_orig, U_mod, rank_orig, rank_mod, backend
    )

    # 3. Condition Number Ratio
    condition_ratio = _compute_condition_ratio(
        s_orig_list, s_mod_list, threshold_orig, threshold_mod
    )

    # 4. Grassmann Distance
    grassmann_dist = _compute_grassmann_distance(
        U_orig, U_mod, rank_orig, rank_mod, backend
    )

    # 5. Relative Frobenius Deviation
    rfd = _compute_relative_frobenius_deviation(weight_original, delta_w, backend)

    # Legacy isometry ratio (deprecated)
    isometry_ratio = spr * (1.0 - spectral_angle / 90.0)

    return IsometryMetrics(
        spectral_preservation_ratio=spr,
        spectral_angle=spectral_angle,
        condition_ratio=condition_ratio,
        grassmann_distance=grassmann_dist,
        relative_frobenius_deviation=rfd,
        rank_original=rank_orig,
        rank_modified=rank_mod,
        isometry_ratio=isometry_ratio,
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


def _compute_spectral_angle(
    U_orig: "Array",
    U_mod: "Array",
    k_orig: int,
    k_mod: int,
    backend: "Backend",
) -> float:
    """Compute angle between top-k singular vectors (degrees).

    k = min(k_orig, k_mod): the SVD-derived effective ranks determine
    how many directions to compare. No arbitrary cap.

    Returns 0° for identical, 90° for orthogonal.
    """
    import math

    k = min(k_orig, k_mod)
    if k == 0:
        return 90.0  # Maximum deviation

    # Get top-k singular vectors
    U_orig_k = U_orig[:, :k]
    U_mod_k = U_mod[:, :k]

    # Compute overlap matrix
    overlap_matrix = backend.matmul(backend.transpose(U_orig_k), U_mod_k)
    backend.eval(overlap_matrix)

    # SVD to get principal angles
    _, sigmas, _ = backend.svd(overlap_matrix)
    backend.eval(sigmas)

    sigma_list = backend.tolist(sigmas)
    if not sigma_list:
        return 90.0

    # Spectral angle = arccos of MINIMUM singular value (worst alignment)
    sigma_min = min(float(s) for s in sigma_list)
    sigma_min = max(-1.0, min(1.0, sigma_min))

    return math.acos(sigma_min) * 180.0 / math.pi


def _compute_condition_ratio(
    s_orig: list[float],
    s_mod: list[float],
    threshold_orig: float,
    threshold_mod: float,
) -> float:
    """Compute condition number ratio κ(W') / κ(W).

    Thresholds are dtype-derived SVD rank thresholds (max(m,n) × ε_mach).
    After filtering by threshold, the smallest SV is guaranteed > 0,
    so no division guard is needed.

    Returns 1.0 for preserved, <1 for regularized (better), >1 for worse.
    """
    s_orig_vals = [float(s) for s in s_orig if float(s) > threshold_orig]
    s_mod_vals = [float(s) for s in s_mod if float(s) > threshold_mod]

    if not s_orig_vals or not s_mod_vals:
        return 1.0

    kappa_orig = s_orig_vals[0] / s_orig_vals[-1]
    kappa_mod = s_mod_vals[0] / s_mod_vals[-1]

    return kappa_mod / kappa_orig


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
    """Compute Grassmann angle between subspaces.

    k = min(k_orig, k_mod): the SVD-derived effective ranks determine
    the subspace dimension. No arbitrary cap.
    """
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


@dataclass(frozen=True)
class IsometryCompositeScores:
    """Composite scores derived from perturbation theory.

    These combine multiple isometry/Weyl metrics to capture joint geometric
    structure that no single metric reveals alone.

    Attributes:
        isometry_score: IS = SPR × (1 - η) × cos(d_G). Range [0, 1].
            Multiplicative: one bad factor tanks the score.
        isometry_defect: L2 distance from perfect-isometry point.
            d = √((1-SPR)² + (θ/90)² + (ln κ_ratio)² + η²)
        alignment_weighted_perturbation: AWP = η × RFD.
            Perturbation magnitude weighted by structural alignment.
        spectral_budget_fraction: SBF = η × (‖ΔW‖₂ / σ_k).
            Fraction of smallest significant SV consumed. None if σ_k unavailable.
    """

    isometry_score: float
    isometry_defect: float
    alignment_weighted_perturbation: float
    spectral_budget_fraction: float | None


def compute_isometry_composites(
    iso: IsometryMetrics,
    weyl: WeylMetrics,
    sigma_k: float | None = None,
) -> IsometryCompositeScores:
    """Compute composite isometry scores from individual metrics.

    Each composite is derived from perturbation theory, not heuristics:
    - IS: multiplicative quality score (one bad factor tanks it)
    - Defect: L2 norm in defect space (worst single deviation dominates)
    - AWP: alignment-weighted perturbation magnitude
    - SBF: fraction of spectral budget consumed (requires sigma_k)

    Args:
        iso: IsometryMetrics for the layer.
        weyl: WeylMetrics for the layer.
        sigma_k: Smallest significant singular value of W (optional).

    Returns:
        IsometryCompositeScores with all four composites.
    """
    spr = iso.spectral_preservation_ratio
    eta = weyl.weyl_utilization
    d_g = iso.grassmann_distance
    theta = iso.spectral_angle
    kappa = iso.condition_ratio
    rfd = iso.relative_frobenius_deviation

    # A. Isometry Score: IS = SPR × (1 - η) × cos(d_G)
    isometry_score = spr * (1.0 - eta) * math.cos(d_g)
    isometry_score = max(0.0, min(1.0, isometry_score))

    # B. Isometry Defect: L2 in defect space
    ln_kappa = math.log(max(kappa, 1e-30))
    isometry_defect = math.sqrt(
        (1.0 - spr) ** 2
        + (theta / 90.0) ** 2
        + ln_kappa ** 2
        + eta ** 2
    )

    # C. Alignment-Weighted Perturbation: AWP = η × RFD
    awp = eta * rfd

    # D. Spectral Budget Fraction: SBF = η × (‖ΔW‖₂ / σ_k)
    sbf: float | None = None
    if sigma_k is not None and sigma_k > 0:
        sbf = eta * (weyl.delta_spectral_norm / sigma_k)

    return IsometryCompositeScores(
        isometry_score=isometry_score,
        isometry_defect=isometry_defect,
        alignment_weighted_perturbation=awp,
        spectral_budget_fraction=sbf,
    )


__all__ = [
    "IsometryMetrics",
    "IsometryCompositeScores",
    "compute_isometry_metrics",
    "compute_isometry_composites",
    "SyntheticLoRA",
    "create_synthetic_isometric_lora",
    "create_synthetic_random_lora",
    "create_synthetic_orthogonal_lora",
]
