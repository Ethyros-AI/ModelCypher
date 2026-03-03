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

"""Principled RBF sigma calibration via constraint satisfaction.

Replaces the gap-based sigma heuristic with a constraint-satisfaction approach.
Finds the sigma interval where ALL layers have non-degenerate Gram matrices
(S₂ bounded away from both 0 and log₂(N)), then picks the geometric midpoint.

Key invariant: S₂(K_l(σ)) is strictly monotonically decreasing in σ for any
fixed set of distinct points. This guarantees binary search correctness.

See docs/research/sigma_calibration_design.md for the full derivation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class CalibrationResult:
    """Result of sigma calibration.

    Attributes:
        sigma_star: Calibrated sigma (geometric midpoint of feasible interval),
            or None if the model is intrinsically multi-scale.
        feasible_lower: Lower bound of the feasible sigma interval.
        feasible_upper: Upper bound of the feasible sigma interval.
        is_multi_scale: True if no single sigma satisfies non-degeneracy for
            all layers. This is a real finding, not an error.
        per_layer_entropy: S₂ per layer at sigma_star (or empty if multi-scale).
        per_layer_ci: Bootstrap CIs for S₂ at sigma_star, or None if skipped.
    """

    sigma_star: float | None
    feasible_lower: float | None
    feasible_upper: float | None
    is_multi_scale: bool
    per_layer_entropy: list[float]
    per_layer_ci: list[tuple[float, float]] | None


def _entropy_at_sigma(
    sq_dist: "Array", sigma: float, backend: "Backend"
) -> float:
    """Compute Rényi-2 entropy S₂(K(σ)) from pre-computed squared distances.

    Builds the RBF Gram matrix at the given sigma, then computes S₂.
    Reuses existing infrastructure from cka.py and renyi_mi.py.

    Args:
        sq_dist: [N, N] squared geodesic distance matrix.
        sigma: RBF bandwidth parameter.
        backend: Backend for tensor operations.

    Returns:
        S₂ in bits (log base 2).
    """
    from modelcypher.core.domain.geometry.cka import _rbf_gram_from_sq_distances
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    gram = _rbf_gram_from_sq_distances(sq_dist, sigma, backend)
    return compute_renyi_entropy_alpha2(gram, backend)


def _bootstrap_entropy_ci(
    sq_dist: "Array",
    sigma: float,
    n_probes: int,
    backend: "Backend",
    n_bootstrap: int,
    alpha: float,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for S₂ at given sigma, resampling probe indices.

    Resamples probe indices with replacement, extracts sub-Gram matrix,
    computes S₂ on the bootstrap sample. Repeats n_bootstrap times.
    Returns percentile CI at confidence 1-alpha.

    Args:
        sq_dist: [N, N] squared geodesic distance matrix.
        sigma: RBF bandwidth.
        n_probes: Number of probes (N).
        backend: Backend for tensor operations.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (e.g., 0.01 for 99% CI).
        rng_seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the bootstrap CI.
    """
    from modelcypher.core.domain.geometry.cka import _rbf_gram_from_sq_distances
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    rng = random.Random(rng_seed)

    # Build full Gram once
    gram = _rbf_gram_from_sq_distances(sq_dist, sigma, backend)
    backend.eval(gram)

    s2_samples = []
    for _ in range(n_bootstrap):
        # Resample probe indices with replacement
        indices = [rng.randint(0, n_probes - 1) for _ in range(n_probes)]
        idx_arr = backend.array(indices)

        # Extract sub-Gram: K_b[i,j] = K[indices[i], indices[j]]
        sub_gram = backend.take(backend.take(gram, idx_arr, axis=0), idx_arr, axis=1)
        backend.eval(sub_gram)

        s2 = compute_renyi_entropy_alpha2(sub_gram, backend)
        s2_samples.append(s2)

    s2_samples.sort()

    # Percentile CI
    lo_idx = max(0, int(math.floor(n_bootstrap * alpha / 2)) - 1)
    hi_idx = min(n_bootstrap - 1, int(math.ceil(n_bootstrap * (1 - alpha / 2))) - 1)

    return s2_samples[lo_idx], s2_samples[hi_idx]


def _find_boundary_sigma(
    sq_dist: "Array",
    target_s2: float,
    sigma_lo: float,
    sigma_hi: float,
    backend: "Backend",
    max_iters: int,
) -> float:
    """Binary search for the sigma where S₂(K(σ)) = target_s2.

    S₂ is monotonically decreasing in sigma:
    - At sigma_lo (small): S₂ ≈ log₂(N) (high)
    - At sigma_hi (large): S₂ ≈ 0 (low)

    We search for the sigma where S₂ crosses the target.

    Args:
        sq_dist: [N, N] squared geodesic distance matrix.
        target_s2: Target entropy value.
        sigma_lo: Lower bound of search (S₂ > target here).
        sigma_hi: Upper bound of search (S₂ < target here).
        backend: Backend for tensor operations.
        max_iters: Maximum binary search iterations.

    Returns:
        Sigma where S₂ ≈ target_s2.
    """
    for _ in range(max_iters):
        # Geometric midpoint in log space
        sigma_mid = math.exp((math.log(sigma_lo) + math.log(sigma_hi)) / 2)
        s2_mid = _entropy_at_sigma(sq_dist, sigma_mid, backend)

        if s2_mid > target_s2:
            # S₂ too high → need larger sigma (to decrease S₂)
            sigma_lo = sigma_mid
        else:
            # S₂ too low → need smaller sigma (to increase S₂)
            sigma_hi = sigma_mid

    return math.exp((math.log(sigma_lo) + math.log(sigma_hi)) / 2)


def compute_calibrated_sigma(
    sq_dist_matrices: list["Array"],
    n_probes: int,
    backend: "Backend",
    alpha: float = 0.01,
) -> CalibrationResult:
    """Find the calibrated sigma where all layers are non-degenerate.

    Algorithm:
    1. For each layer, binary search for the sigma boundaries where
       S₂ transitions from non-degenerate to degenerate.
    2. Take the intersection of all layers' feasible intervals.
    3. If empty: report multi-scale. If non-empty: geometric midpoint.
    4. Bootstrap verification at sigma_star.

    Non-degeneracy constraints (both must hold for all layers):
        S₂(K_l(σ)) > √ε_mach              (not collapsed)
        log₂(N) - S₂(K_l(σ)) > √ε_mach   (not saturated)

    Args:
        sq_dist_matrices: List of [N, N] squared geodesic distance matrices,
            one per layer.
        n_probes: Number of probes (N).
        backend: Backend for tensor operations.
        alpha: Confidence level for bootstrap CIs (default 0.01).

    Returns:
        CalibrationResult with sigma_star and diagnostics.
    """
    if not sq_dist_matrices:
        return CalibrationResult(
            sigma_star=None,
            feasible_lower=None,
            feasible_upper=None,
            is_multi_scale=True,
            per_layer_entropy=[],
            per_layer_ci=None,
        )

    # Derived constants
    eps = division_epsilon(backend, sq_dist_matrices[0])
    sqrt_eps = eps  # division_epsilon already returns sqrt(machine_eps)
    log2_n = math.log2(n_probes)

    # Non-degeneracy thresholds
    s2_lower_threshold = sqrt_eps  # S₂ must exceed this (not collapsed)
    s2_upper_threshold = log2_n - sqrt_eps  # S₂ must be below this (not saturated)

    if s2_lower_threshold >= s2_upper_threshold:
        # N too small for meaningful calibration
        return CalibrationResult(
            sigma_star=None,
            feasible_lower=None,
            feasible_upper=None,
            is_multi_scale=True,
            per_layer_entropy=[],
            per_layer_ci=None,
        )

    # Derive search bounds from data
    # Find global min and max positive squared distances
    d_sq_min = float("inf")
    d_sq_max = 0.0
    for sq_dist in sq_dist_matrices:
        flat = backend.reshape(sq_dist, (-1,))
        backend.eval(flat)

        # Filter to positive values (skip diagonal zeros)
        mask = flat > eps * eps  # compare against eps² since these are squared
        pos_vals = backend.where(mask, flat, backend.full(flat.shape, float("inf")))
        neg_vals = backend.where(mask, flat, backend.full(flat.shape, 0.0))

        min_val = backend.min(pos_vals)
        max_val = backend.max(neg_vals)
        backend.eval(min_val, max_val)

        d_sq_min = min(d_sq_min, float(backend.to_scalar(min_val)))
        d_sq_max = max(d_sq_max, float(backend.to_scalar(max_val)))

    if d_sq_min <= 0 or d_sq_max <= 0 or d_sq_min == float("inf"):
        return CalibrationResult(
            sigma_star=None,
            feasible_lower=None,
            feasible_upper=None,
            is_multi_scale=True,
            per_layer_entropy=[],
            per_layer_ci=None,
        )

    # Search bounds: sigma_lo where K ≈ I, sigma_hi where K ≈ 11^T
    # sigma_lo = d_min * sqrt(eps): exp(-d_min²/(2·(d_min·√ε)²)) = exp(-1/(2ε)) ≈ 0
    # sigma_hi = d_max / sqrt(2·eps): exp(-d_max²/(2·(d_max/√(2ε))²)) = exp(-ε) ≈ 1
    d_min = math.sqrt(d_sq_min)
    d_max = math.sqrt(d_sq_max)
    sigma_search_lo = d_min * sqrt_eps
    sigma_search_hi = d_max / math.sqrt(2.0 * sqrt_eps)

    if sigma_search_lo <= 0 or sigma_search_hi <= sigma_search_lo:
        return CalibrationResult(
            sigma_star=None,
            feasible_lower=None,
            feasible_upper=None,
            is_multi_scale=True,
            per_layer_entropy=[],
            per_layer_ci=None,
        )

    # Binary search iterations: ceil(log₂((ln σ_hi - ln σ_lo) / ε))
    log_range = math.log(sigma_search_hi) - math.log(sigma_search_lo)
    max_iters = max(20, int(math.ceil(math.log2(log_range / (eps * eps)))))
    # Cap at reasonable maximum to prevent runaway
    max_iters = min(max_iters, 100)

    # For each layer, find the feasible sigma interval
    # sigma_l_lo: where S₂ drops to s2_upper_threshold (entering feasible from below)
    # sigma_l_hi: where S₂ drops to s2_lower_threshold (leaving feasible)
    global_feasible_lo = 0.0  # will take max over layers
    global_feasible_hi = float("inf")  # will take min over layers

    for sq_dist in sq_dist_matrices:
        # Verify the search bounds bracket the thresholds
        s2_at_lo = _entropy_at_sigma(sq_dist, sigma_search_lo, backend)
        s2_at_hi = _entropy_at_sigma(sq_dist, sigma_search_hi, backend)

        # At sigma_lo: S₂ should be high (near log₂(N))
        # At sigma_hi: S₂ should be low (near 0)
        # If not, the search bounds don't bracket → this layer can't be calibrated
        if s2_at_lo <= s2_upper_threshold:
            # Even at smallest sigma, S₂ is already below upper threshold
            # This means constraint 2 can't be satisfied → no feasible sigma
            # (but constraint 1 might still work, layer is always "non-saturated")
            # Actually this means the upper boundary is at or below sigma_search_lo
            # Just set this layer's lo boundary very small
            layer_lo = sigma_search_lo
        else:
            # Binary search for sigma where S₂ = s2_upper_threshold
            layer_lo = _find_boundary_sigma(
                sq_dist, s2_upper_threshold, sigma_search_lo, sigma_search_hi,
                backend, max_iters,
            )

        if s2_at_hi >= s2_lower_threshold:
            # Even at largest sigma, S₂ is still above lower threshold
            # This layer is never collapsed → feasible up to sigma_search_hi
            layer_hi = sigma_search_hi
        else:
            # Binary search for sigma where S₂ = s2_lower_threshold
            layer_hi = _find_boundary_sigma(
                sq_dist, s2_lower_threshold, sigma_search_lo, sigma_search_hi,
                backend, max_iters,
            )

        global_feasible_lo = max(global_feasible_lo, layer_lo)
        global_feasible_hi = min(global_feasible_hi, layer_hi)

    # Check if feasible interval is non-empty
    if global_feasible_lo >= global_feasible_hi:
        return CalibrationResult(
            sigma_star=None,
            feasible_lower=global_feasible_lo,
            feasible_upper=global_feasible_hi,
            is_multi_scale=True,
            per_layer_entropy=[],
            per_layer_ci=None,
        )

    # Geometric midpoint
    sigma_star = math.exp(
        (math.log(global_feasible_lo) + math.log(global_feasible_hi)) / 2
    )

    # Compute per-layer entropy at sigma_star
    per_layer_entropy = [
        _entropy_at_sigma(sq_dist, sigma_star, backend)
        for sq_dist in sq_dist_matrices
    ]

    # Bootstrap verification at sigma_star
    n_bootstrap = math.ceil(2.0 / alpha)
    per_layer_ci = []
    for sq_dist in sq_dist_matrices:
        ci = _bootstrap_entropy_ci(
            sq_dist, sigma_star, n_probes, backend,
            n_bootstrap=n_bootstrap, alpha=alpha,
        )
        per_layer_ci.append(ci)

    return CalibrationResult(
        sigma_star=sigma_star,
        feasible_lower=global_feasible_lo,
        feasible_upper=global_feasible_hi,
        is_multi_scale=False,
        per_layer_entropy=per_layer_entropy,
        per_layer_ci=per_layer_ci,
    )
