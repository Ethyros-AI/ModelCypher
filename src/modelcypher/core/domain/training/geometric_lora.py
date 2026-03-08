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
    svd_rank_threshold,
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
    effective_rank: int  # Precision-based: count(S > max(m,n)*eps*sigma_max)
    full_rank: int
    decay_ratio: float  # σ_max / σ_k
    tail_dims: int  # full_rank - floor(shannon_eff_rank) (structural null-space)
    shannon_effective_rank: float  # exp(H(σ²)) - continuous spectral utilization
    spectral_gap: float  # σ_{k-1} - σ_k at structural rank boundary (Weyl crossing threshold)

    @property
    def is_targetable(self) -> bool:
        """Whether this layer is safe to target for LoRA.

        A layer is targetable if it has non-zero null-space capacity:
        tail_dims > 0 after Shannon structural rank estimation.
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

    # Singular values only (U, Vt not needed for geometry analysis).
    # compute_uv=False is significantly faster, especially on CPU where
    # MLX forces SVD execution for large 8B+ weight matrices.
    S = b.svd(W, compute_uv=False)
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

    # Precision-based effective rank (LAPACK/MATLAB convention):
    # significant if σ_i > max(m,n) * eps(dtype) * σ_max
    rank_eps = svd_rank_threshold(b, S, max(shape))
    threshold = rank_eps * sigma_max
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
    # max(m,n)*eps*sigma_max only tells us what the dtype can distinguish from zero.
    # Shannon eff rank tells us how many dimensions carry meaningful information.
    structural_rank = max(1, min(math.floor(shannon_effective_rank), n_svs - 1))

    # sigma_k = SV at the edge of the informationally significant subspace
    sigma_k = float(b.to_scalar(S[structural_rank - 1]))

    # Ensure sigma_k is positive
    if sigma_k <= 0:
        div_eps = division_epsilon(b, S)
        sigma_k = sigma_min if sigma_min > 0 else max(div_eps, threshold)

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


def _randomized_singular_values(
    W: "Array",
    target_rank: int,
    backend: "Backend",
    power_iters: int = 2,
    seed: int | None = None,
) -> "Array":
    """Randomized truncated SVD returning top-k singular values.

    Uses Halko et al. (2011), Algorithm 5.1 to compute the top
    ``target_rank`` singular values of ``W`` via random projection +
    subspace iteration, without forming the full decomposition.

    Cost: O(m * n * target_rank * (2 * power_iters + 1)), which is
    O(mnk) when target_rank << min(m, n) — vastly cheaper than
    the O(mn * min(m,n)) full SVD.

    Args:
        W: Weight matrix [m, n] in float32.
        target_rank: Number of singular values to extract.
        backend: Backend protocol for tensor operations.
        power_iters: Subspace power iterations. Error decays as
            (σ_{k+1}/σ_k)^{2q+1} (Halko et al. 2011, Algorithm 4.3).
        seed: RNG seed. None uses current backend state.

    Returns:
        Array of top ``target_rank`` singular values in descending order.

    References:
        Halko, N., Martinsson, P.G. & Tropp, J.A. (2011). Finding
        Structure with Randomness. SIAM Review, 53(2), 217-288.
    """
    b = backend
    m, n = int(W.shape[0]), int(W.shape[1])

    if seed is not None:
        b.random_seed(seed)

    # Step 1: Random Gaussian test matrix Ω ∈ R^{n × target_rank}
    omega = b.random_normal((n, target_rank))
    omega = b.astype(omega, "float32")
    b.eval(omega)

    # Step 2: Sample matrix Y = W @ Ω ∈ R^{m × target_rank}
    Y = b.matmul(W, omega)
    b.eval(Y)

    # Step 3: Subspace power iteration for spectral gap amplification
    for _ in range(power_iters):
        Y = b.matmul(b.transpose(W), Y)  # W^T @ Y: [n, target_rank]
        b.eval(Y)
        Y = b.matmul(W, Y)               # W @ (W^T @ Y): [m, target_rank]
        b.eval(Y)

    # Step 4: Orthonormal basis via thin SVD of sample matrix
    Q, _s_y, _vt_y = b.svd(Y, compute_uv=True)
    b.eval(Q)

    # Step 5: Project into low-rank subspace: B = Q^T @ W ∈ R^{target_rank × n}
    B = b.matmul(b.transpose(Q), W)
    b.eval(B)

    # Step 6: SVD of the small matrix B — gives approximate top singular values
    S_approx = b.svd(B, compute_uv=False)
    b.eval(S_approx)

    return S_approx


def compute_layer_geometry_randomized(
    weight: "Array",
    layer_key: str,
    backend: "Backend",
    oversampling: int = 10,
    max_iters: int = 3,
    power_iters: int = 2,
    seed: int | None = None,
) -> LayerGeometry:
    """Compute spectral geometry via randomized SVD with tail-energy validation.

    Avoids full SVD (O(mn·min(m,n))) by using randomized truncated SVD
    (Halko et al. 2011) to capture the top-k singular values, then
    validates that the uncaptured tail energy is negligible.

    The algorithm:
    1. Compute ||W||_F^2 in O(mn) — this is the total spectral energy.
    2. Start with k = min(initial_k, full_rank/2) singular values via
       randomized SVD.
    3. Check tail energy: tail = ||W||_F^2 - sum(top_k σ_i^2).
       If tail / ||W||_F^2 < sqrt(eps), the spectrum is captured.
    4. If not, double k and retry (converges in 1-2 iterations for
       typical transformer weights where Shannon eff rank << full_rank).
    5. Compute Shannon effective rank from the captured spectrum,
       accounting for the tail energy as a correction term.

    Falls back to full SVD when the matrix is small enough that
    randomized SVD has no advantage (target >= min(m, n)).

    Args:
        weight: Weight matrix [out_features, in_features].
        layer_key: Identifier for this layer.
        backend: Backend for tensor operations.
        oversampling: Extra columns beyond target rank for Halko capture.
            p >= 5 gives high-probability accuracy for float32
            (Halko et al. 2011, §10.3).
        max_iters: Maximum doubling iterations for tail convergence.
        power_iters: Subspace power iterations per randomized SVD call.
        seed: RNG seed for reproducibility.

    Returns:
        LayerGeometry with all spectral information.
    """
    b = backend

    W = b.astype(weight, "float32")
    b.eval(W)

    shape = (int(W.shape[0]), int(W.shape[1]))
    full_rank = min(shape)

    # Total spectral energy: ||W||_F^2 = trace(W^T W) = sum(σ_i^2)
    # Computable in O(mn) without any decomposition.
    frob_sq = b.sum(W * W)
    b.eval(frob_sq)
    frob_sq_val = float(b.to_scalar(frob_sq))

    if frob_sq_val <= 0:
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

    eps_mach = float(b.finfo().eps)
    tail_threshold = math.sqrt(eps_mach)  # tail energy fraction below which we stop

    # Initial target: min(256, full_rank // 2) — generous for most transformer layers
    initial_k = min(256, max(16, full_rank // 2))

    S = None
    for iteration in range(max_iters):
        target = min(initial_k, full_rank)

        # If target covers the full rank, just do full SVD
        if target + oversampling >= full_rank:
            S = b.svd(W, compute_uv=False)
            b.eval(S)
            break

        S = _randomized_singular_values(
            W, target + oversampling, b,
            power_iters=power_iters,
            seed=(seed + iteration) if seed is not None else None,
        )

        # Check tail energy convergence
        captured_energy = b.sum(S * S)
        b.eval(captured_energy)
        captured_val = float(b.to_scalar(captured_energy))
        tail_fraction = (frob_sq_val - captured_val) / frob_sq_val

        if tail_fraction < tail_threshold:
            logger.debug(
                "%s: randomized SVD converged at k=%d (tail=%.2e, iter=%d)",
                layer_key, target, tail_fraction, iteration,
            )
            break

        # Double target and retry
        initial_k = min(initial_k * 2, full_rank)
        logger.debug(
            "%s: tail energy %.2e > threshold %.2e, doubling to k=%d",
            layer_key, tail_fraction, tail_threshold, initial_k,
        )
    else:
        logger.warning(
            "%s: randomized SVD did not converge after %d iterations "
            "(tail=%.2e); falling back to full SVD",
            layer_key, max_iters, tail_fraction,
        )
        S = b.svd(W, compute_uv=False)
        b.eval(S)

    # From here, compute geometry identically to compute_layer_geometry()
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

    sigma_max = float(b.to_scalar(S[0]))

    # Precision-based effective rank (LAPACK/MATLAB convention)
    rank_eps = svd_rank_threshold(b, S, max(shape))
    threshold = rank_eps * sigma_max
    significant_mask = S > threshold
    significant_count = b.sum(b.astype(significant_mask, "int32"))
    b.eval(significant_count)
    effective_rank = int(b.to_scalar(significant_count))

    # Shannon effective rank from captured spectrum.
    # For randomized SVD, the captured singular values approximate the top-k.
    # The tail energy (uncaptured dimensions) contributes negligibly to
    # entropy because Shannon entropy is dominated by the large singular
    # values. When tail_fraction < sqrt(eps), the entropy error is bounded
    # by O(tail_fraction * log(full_rank / n_svs)), which is negligible.
    eigvals = S * S
    sum_eig = b.sum(eigvals)
    b.eval(sum_eig)
    sum_eig_val = float(b.to_scalar(sum_eig))

    # Use Frobenius norm as the normalizer (captures total energy including
    # any uncaptured tail). This ensures the probability distribution sums
    # correctly even with a truncated spectrum.
    eps = division_epsilon(b, eigvals)

    if frob_sq_val > eps:
        frob_sq_arr = b.full((1,), frob_sq_val, dtype=eigvals.dtype)
        b.eval(frob_sq_arr)
        p = eigvals / frob_sq_arr
        log_eps = safe_log_epsilon(b, eigvals)
        eps_arr = b.full(p.shape, log_eps, dtype=p.dtype)
        p_safe = b.where(p > log_eps, p, eps_arr)
        entropy = b.sum(-p * b.log(p_safe))
        # Tail correction: the uncaptured dimensions each contribute
        # at most (frob_sq - captured) / (full_rank - n_svs) energy.
        # Their entropy contribution is bounded but can be accounted for.
        tail_energy = frob_sq_val - sum_eig_val
        n_tail = full_rank - n_svs
        if n_tail > 0 and tail_energy > eps:
            # Upper bound: tail dims have uniform energy = tail_energy / n_tail
            p_tail = tail_energy / (n_tail * frob_sq_val)
            if p_tail > 0:
                tail_entropy = -n_tail * p_tail * math.log(p_tail)
                entropy = entropy + b.full((1,), tail_entropy, dtype=entropy.dtype)
        b.eval(entropy)
        shannon_eff_rank = b.exp(entropy)
        b.eval(shannon_eff_rank)
        shannon_effective_rank = float(b.to_scalar(shannon_eff_rank))
    else:
        shannon_effective_rank = 0.0

    structural_rank = max(1, min(math.floor(shannon_effective_rank), n_svs - 1))

    sigma_k = float(b.to_scalar(S[structural_rank - 1]))

    if sigma_k <= 0:
        sigma_min = float(b.to_scalar(S[n_svs - 1]))
        div_eps = division_epsilon(b, S)
        sigma_k = sigma_min if sigma_min > 0 else max(div_eps, threshold)

    if structural_rank >= 2:
        sigma_k_prev = float(b.to_scalar(S[structural_rank - 2]))
        spectral_gap = sigma_k_prev - sigma_k
    else:
        spectral_gap = sigma_k

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
    total = len(weights)
    progress_interval = max(1, total // 5)
    analyzed = 0

    for layer_key, weight in weights.items():
        try:
            geometry = compute_layer_geometry(weight, layer_key, backend)
            geometries[layer_key] = geometry
            analyzed += 1

            logger.debug(
                "%s: decay=%.1f×, σ_k=%.4f, tail=%d, targetable=%s",
                layer_key,
                geometry.decay_ratio,
                geometry.sigma_k,
                geometry.tail_dims,
                geometry.is_targetable,
            )
            if analyzed % progress_interval == 0 or analyzed == total:
                logger.info("Analyzed %d/%d weight matrices...", analyzed, total)
        except Exception as e:
            logger.warning("Failed to analyze layer %s: %s", layer_key, e)
            continue

    return geometries


def select_target_modules(
    geometries: dict[str, LayerGeometry],
    include_zero_tail: bool = False,
) -> list[str]:
    """Select modules to target based on geometry.

    By default, returns layers with non-zero null-space capacity (tail_dims > 0).

    When ``include_zero_tail=True``, also includes layers with ``tail_dims == 0``
    but positive ``spectral_gap`` — these are full-rank layers that can support
    minimal (rank-1) adaptation bounded by half the spectral gap (Weyl 1912
    crossing threshold).

    Args:
        geometries: Pre-computed layer geometries.
        include_zero_tail: If True, also target full-rank layers with
            positive spectral gap.

    Returns:
        List of layer keys that are safe to target.
    """
    targets = [key for key, geom in geometries.items() if geom.is_targetable]
    if include_zero_tail:
        for key, geom in geometries.items():
            if key not in targets and geom.tail_dims == 0 and geom.spectral_gap > 0:
                targets.append(key)
    return targets


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

    Rank is derived directly from structural null-space capacity:
    rank_i = tail_dims_i = full_rank - floor(shannon_effective_rank_i).

    For layers with ``tail_dims == 0`` (full-rank, included via
    ``include_zero_tail=True``): rank = 1 (minimum adaptation that can
    learn without being underdetermined).

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
        if geom.tail_dims > 0:
            per_layer_ranks[key] = int(geom.tail_dims)
        elif geom.spectral_gap > 0:
            # Full-rank layer with positive spectral gap: rank-1 adaptation
            per_layer_ranks[key] = 1
        # else: skip (neither null-space nor adaptable gap)

    if not per_layer_ranks:
        raise ValueError("No geometries found for target modules")

    return per_layer_ranks


def apply_data_rank_ceiling(
    per_layer_ranks: dict[str, int],
    n_samples: int,
) -> dict[str, int]:
    """Cap per-layer ranks by the finite data rank ceiling.

    Any activation matrix built from ``n_samples`` observations has rank at most
    ``n_samples``. Adapting more directions than the data can identify is
    underdetermined and inflates trainable parameters without added signal.

    Args:
        per_layer_ranks: Proposed per-layer ranks (from tail_dims).
        n_samples: Number of training samples used to fit the adapter.

    Returns:
        New dict with ranks capped at ``n_samples``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    capped: dict[str, int] = {}
    for key, rank in per_layer_ranks.items():
        if rank <= 0:
            capped[key] = 0
            continue
        capped[key] = min(int(rank), int(n_samples))
    return capped


def compute_per_layer_signal_ranks(
    base_activations: dict[int, list["Array"]],
    backend: "Backend",
) -> dict[int, "SignalRankResult"]:
    """Compute intrinsic signal rank per layer via RMT Marchenko-Pastur separation.

    Uses singular value decomposition on stacked per-layer activations to
    identify the number of eigenvalues above the MP bulk edge — the intrinsic
    signal dimensionality.  This is typically 10–50, far below tail_dims.

    Args:
        base_activations: Per-layer lists of mean-pooled hidden activations
            (dict[layer_idx, list[Array[hidden_dim]]]).
        backend: Compute backend (for SVD and array ops).

    Returns:
        dict mapping layer index to SignalRankResult.  Layers with < 2 probes
        are skipped.
    """
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        SignalRankResult,
        compute_signal_rank_from_singular_values,
    )

    results: dict[int, SignalRankResult] = {}
    for layer_idx, acts_list in sorted(base_activations.items()):
        if len(acts_list) < 2:
            continue
        # Stack [n_samples, hidden_dim] and center
        A = backend.stack(acts_list)
        col_mean = backend.mean(A, axis=0)
        A_centered = A - col_mean
        backend.eval(A_centered)

        n_samples, hidden_dim = int(A_centered.shape[0]), int(A_centered.shape[1])

        # SVD on centered activation matrix (cast to float32 — SVD requires float32+)
        A_f32 = backend.astype(A_centered, "float32")
        backend.eval(A_f32)
        S = backend.svd(A_f32, compute_uv=False)
        backend.eval(S)

        result = compute_signal_rank_from_singular_values(
            S, n_samples, hidden_dim, backend, center_correction=True,
        )
        results[layer_idx] = result

        logger.info(
            "RMT signal rank layer %d: signal_rank=%d, noise_rank=%d, "
            "mp_upper_edge=%.6f, signal_var=%.1f%%",
            layer_idx,
            result.signal_rank,
            result.noise_rank,
            result.mp_upper_edge,
            100.0 * result.signal_variance_fraction,
        )

    return results


def apply_signal_rank_ceiling(
    per_module_ranks: dict[str, int],
    signal_rank_results: dict[int, "SignalRankResult"],
) -> dict[str, int]:
    """Cap per-module LoRA ranks at the RMT signal rank for each layer.

    The signal rank measures how many directions actually carry information
    in the training data's activation space.  Adapting beyond this wastes
    parameters on noise dimensions.

    Modules with rank=0 stay 0 (not targetable).  Modules in layers without
    a signal-rank measurement keep their original rank.

    Args:
        per_module_ranks: Per-module proposed ranks (from coupling / data ceiling).
        signal_rank_results: Per-layer SignalRankResult from
            ``compute_per_layer_signal_ranks``.

    Returns:
        New dict with ranks capped at ``max(1, signal_rank)`` per layer.
    """
    capped: dict[str, int] = {}
    for key, rank in per_module_ranks.items():
        if rank <= 0:
            capped[key] = 0
            continue
        # Parse layer index: model.layers.{idx}.self_attn.{proj}.weight
        parts = key.split(".")
        try:
            layer_idx = int(parts[2])
        except (IndexError, ValueError):
            capped[key] = rank
            continue
        sr = signal_rank_results.get(layer_idx)
        if sr is None:
            capped[key] = rank
            continue
        capped[key] = min(rank, max(1, sr.signal_rank))
    return capped


def estimate_nb_lora_parameter_count(
    geometries: dict[str, LayerGeometry],
    per_layer_ranks: dict[str, int],
) -> int:
    """Estimate NB-LoRA trainable parameters from geometry and ranks.

    NB-LoRA trainables per layer:
      - A_tilde: [rank, in_features]
      - B_tilde: [rank, out_features]
      - S_raw:   [rank]
    Total per layer = rank * (in_features + out_features + 1)

    Args:
        geometries: Layer geometries keyed by layer name.
        per_layer_ranks: Rank assignment keyed by layer name.

    Returns:
        Total trainable parameter count across layers present in both dicts.
    """
    total = 0
    for key, rank in per_layer_ranks.items():
        if rank <= 0:
            continue
        geom = geometries.get(key)
        if geom is None:
            continue
        in_features = int(geom.shape[1])
        out_features = int(geom.shape[0])
        total += int(rank) * (in_features + out_features + 1)
    return int(total)


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
    "apply_data_rank_ceiling",
    "apply_signal_rank_ceiling",
    "LayerGeometry",
    "analyze_weight_geometries",
    "compute_coupled_ranks",
    "compute_geometric_dropout",
    "compute_geometric_rank",
    "compute_layer_geometry",
    "compute_layer_geometry_randomized",
    "compute_per_layer_ranks",
    "compute_per_layer_signal_ranks",
    "derive_lora_configs",
    "estimate_nb_lora_parameter_count",
    "select_target_modules",
]
