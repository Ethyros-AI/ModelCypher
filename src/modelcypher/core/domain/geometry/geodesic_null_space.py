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

"""Geodesic null-space filtering for model merging.

Implements null-space filtering using geodesic distances on a k-NN graph and
Random Matrix Theory (Marchenko-Pastur distribution) for signal/noise separation.
All computation stays on the backend.

Key innovation: Instead of arbitrary variance heuristics for null-space detection,
uses RMT-derived thresholds based on the eigenvalue spectrum of the activation
covariance matrix. Eigenvalues above the Marchenko-Pastur bulk edge are true
signal (occupied, protect); eigenvalues in the bulk are noise (available for
transfer).

References:
    - Tenenbaum et al. (2000) "Isomap" - geodesic via graph
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - Marchenko & Pastur (1967) "Distribution of eigenvalues for random matrices"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_svd,
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
    svd_auto_rank,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
)
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    compute_rmt_null_space_weights,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()


def _as_2d(matrix: "Array", backend: "Backend") -> "Array":
    arr = backend.array(matrix) if not hasattr(matrix, "shape") else matrix
    shape = backend.shape(arr)
    if len(shape) != 2 or int(shape[0]) < 1:
        return backend.reshape(arr, (1, -1))
    return arr


def _geodesic_frobenius_from_row_norms(
    row_norms: "Array",
    backend: "Backend",
) -> float:
    sum_sq = backend.sum(row_norms * row_norms)
    backend.eval(sum_sq)
    return float(
        backend.to_scalar(
            backend.sqrt(sum_sq + regularization_epsilon(backend, row_norms))
        )
    )


def _geodesic_frobenius_norm(
    matrix: "Array",
    backend: "Backend",
) -> float:
    arr = _as_2d(matrix, backend)
    energy = backend.sum(arr * arr)
    backend.eval(energy)
    if float(backend.to_scalar(energy)) <= regularization_epsilon(backend, arr):
        return 0.0
    norms = geodesic_norms(arr, backend, use_cache=False)
    backend.eval(norms)
    return _geodesic_frobenius_from_row_norms(norms, backend)


def _geodesic_frobenius_norms_pair(
    original: "Array",
    filtered: "Array",
    backend: "Backend",
) -> tuple[float, float]:
    original_arr = _as_2d(original, backend)
    filtered_arr = _as_2d(filtered, backend)
    original_energy = backend.sum(original_arr * original_arr)
    filtered_energy = backend.sum(filtered_arr * filtered_arr)
    backend.eval(original_energy, filtered_energy)
    original_zero = (
        float(backend.to_scalar(original_energy))
        <= regularization_epsilon(backend, original_arr)
    )
    filtered_zero = (
        float(backend.to_scalar(filtered_energy))
        <= regularization_epsilon(backend, filtered_arr)
    )
    if original_zero and filtered_zero:
        return 0.0, 0.0
    combined = backend.concatenate([original_arr, filtered_arr], axis=0)
    norms = geodesic_norms(combined, backend, use_cache=False)
    backend.eval(norms)
    original_rows = int(backend.shape(original_arr)[0])
    original_norms = norms[:original_rows]
    filtered_norms = norms[original_rows:]
    original_norm = 0.0 if original_zero else _geodesic_frobenius_from_row_norms(
        original_norms, backend
    )
    filtered_norm = 0.0 if filtered_zero else _geodesic_frobenius_from_row_norms(
        filtered_norms, backend
    )
    return (
        original_norm,
        filtered_norm,
    )


@dataclass
class GeodesicNullSpaceResult:
    """Result of geodesic null-space filtering."""

    # The filtered delta (projected to geodesic-orthogonal directions)
    filtered_delta: Any

    # Original delta (for comparison)
    original_delta: Any

    # Dimension of geodesic-orthogonal space
    orthogonal_dim: int

    # Fraction of delta that was removed (interference component)
    projection_loss: float

    # Fraction of delta preserved (safe component)
    preserved_fraction: float

    # Geodesic norm of original delta
    original_norm: float

    # Geodesic norm of filtered delta
    filtered_norm: float

    # Whether filtering was applied
    filtering_applied: bool

    # k-NN connectivity used
    k_neighbors: int

    # Mean geodesic distance (manifold scale)
    mean_geodesic_distance: float

    # Per-dimension occupancy implied by the filtered delta (0-1, data-derived)
    delta_weights: Any | None = None


@dataclass(frozen=True)
class GeodesicNullSpaceBasis:
    """Reusable geodesic null-space basis for repeated projections."""

    Q: Any
    orthogonal_dim: int
    k_neighbors: int
    mean_geodesic_distance: float
    regularization: float


class GeodesicNullSpaceFilter:
    """
    GPU-accelerated null-space filter using geodesic geometry.

    Instead of computing the exact linear null space via SVD (CPU-only on MLX),
    this filter finds geodesic-orthogonal directions using only GPU operations.

    All operations are GPU-accelerated:
    - Pairwise distances: matmul
    - k-NN graph: argsort
    - Geodesic distances: vectorized Floyd-Warshall
    - Projection: Gram matrix operations

    Uses Random Matrix Theory (Marchenko-Pastur distribution) to separate
    true signal from noise, providing principled, data-derived thresholds
    instead of arbitrary variance heuristics.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._riemannian = RiemannianGeometry(self._backend)

    def prepare_basis(
        self,
        prior_activations: Any,
        k_neighbors: int | None = None,
    ) -> GeodesicNullSpaceBasis:
        """Precompute geodesic null-space basis for reuse across deltas."""
        return self._compute_basis(prior_activations, k_neighbors=k_neighbors)

    def filter_delta(
        self,
        weight_delta: Any,
        prior_activations: Any,
        k_neighbors: int | None = None,
        occupancy_weights: Any | None = None,
        basis: GeodesicNullSpaceBasis | None = None,
        source_activations: Any | None = None,
    ) -> GeodesicNullSpaceResult:
        """
        Filter weight delta using RMT-based null-space projection.

        Uses Random Matrix Theory (Marchenko-Pastur distribution) to separate
        true signal from noise, providing principled thresholds for null-space
        detection.

        If source_activations is provided:
            transfer = target_noise × source_signal
            Transfer where target has room AND source has knowledge.

        If source_activations is not provided:
            transfer = target_noise
            Transfer where target has unused capacity.

        Algorithm:
            1. Build k-NN graph from activations (GPU: matmul + argsort)
            2. Compute geodesic distances (GPU: Floyd-Warshall)
            3. Use RMT to compute per-dimension signal/noise separation
            4. Scale delta by transfer weights per dimension

        Args:
            weight_delta: Weight update to filter. Shape: [out, in] or [d].
            prior_activations: Target activation matrix. Shape: [n_samples, d].
            k_neighbors: k for k-NN graph. If None, auto-derived.
            occupancy_weights: Optional prior occupancy weights to combine with RMT.
            basis: Optional precomputed geodesic basis for reuse.
            source_activations: Optional source activations (already aligned).
                If provided, enables density-aware transfer.

        Returns:
            GeodesicNullSpaceResult with filtered delta and diagnostics.
        """
        backend = self._backend
        weight_delta = backend.array(weight_delta)
        prior_activations = backend.array(prior_activations)
        backend.eval(weight_delta, prior_activations)

        original_shape = weight_delta.shape
        d = int(prior_activations.shape[1])

        transpose_back = False
        delta_matrix = None
        if len(original_shape) == 1:
            if int(original_shape[0]) == d:
                delta_matrix = backend.reshape(weight_delta, (1, d))
        elif len(original_shape) == 2:
            if int(original_shape[1]) == d:
                delta_matrix = weight_delta
            elif int(original_shape[0]) == d:
                delta_matrix = backend.transpose(weight_delta)
                transpose_back = True

        # Handle dimension mismatch
        if delta_matrix is None:
            delta_flat = backend.reshape(weight_delta, (-1,))
            backend.eval(delta_flat)
            delta_dim = int(delta_flat.shape[0])
            logger.warning(
                f"Dimension mismatch: delta has {delta_dim} elements, "
                f"activations have {d} features. Returning original delta."
            )
            norm_arr = geodesic_norms(backend.reshape(delta_flat, (1, -1)), backend)
            backend.eval(norm_arr)
            return GeodesicNullSpaceResult(
                filtered_delta=weight_delta,
                original_delta=weight_delta,
                orthogonal_dim=0,
                projection_loss=0.0,
                preserved_fraction=1.0,
                original_norm=float(backend.to_scalar(norm_arr[0])),
                filtered_norm=float(backend.to_scalar(norm_arr[0])),
                filtering_applied=False,
                k_neighbors=0,
                mean_geodesic_distance=0.0,
            )

        delta_flat = backend.reshape(delta_matrix, (-1,))
        backend.eval(delta_flat)

        basis = basis or self._compute_basis(
            prior_activations,
            k_neighbors=k_neighbors,
            source_activations=source_activations,
        )

        # Step 4: Compute tangent space projection matrix (GPU)
        # The tangent vectors span the "occupied" directions on the manifold.
        # We project delta onto the orthogonal complement (geodesic null space).
        #
        # Projection onto column space of T: P_T = T @ pinv(T)
        # Projection onto orthogonal complement: P_null = I - P_T
        #
        # Uses native b.pinv() for EXACT pseudo-inverse computation on GPU.

        # =====================================================================
        # VARIANCE-WEIGHTED PROJECTION (replaces binary Q @ Q^T)
        # =====================================================================
        # Q now contains variance_weights [d, 1] where:
        #   weight[i] = normalized variance of dimension i
        #   High weight = dense direction = project out more
        #   Low weight = sparse direction = keep more delta
        #
        # Formula: delta_safe = delta * (1 - variance_weights)
        # This keeps delta in sparse directions, removes in dense directions.
        # =====================================================================
        reg = basis.regularization
        Q = basis.Q  # Contains variance_weights [d, 1]

        delta_dtype = backend.dtype(delta_matrix)
        proj_dtype = precision_dtype(backend, reference=delta_matrix)
        delta_proj = delta_matrix
        if str(proj_dtype) != str(delta_dtype):
            delta_proj = backend.astype(delta_matrix, proj_dtype)
            backend.eval(delta_proj)
        if str(backend.dtype(Q)) != str(proj_dtype):
            Q = backend.astype(Q, proj_dtype)
            backend.eval(Q)

        # Extract variance weights and compute keep weights
        combined_weights = backend.squeeze(Q)  # [d]
        if occupancy_weights is not None:
            occ = occupancy_weights
            if not hasattr(occ, "shape"):
                occ = backend.array(occ)
            occ = backend.astype(occ, proj_dtype)
            if int(backend.shape(occ)[0]) == d:
                occ = backend.clip(occ, 0.0, 1.0)
                combined_weights = 1.0 - (1.0 - combined_weights) * (1.0 - occ)
                backend.eval(combined_weights)
            else:
                logger.warning(
                    "Occupancy weights dim mismatch: expected %d, got %d - ignoring",
                    d,
                    int(backend.shape(occ)[0]),
                )
        keep_weights = 1.0 - combined_weights  # [d] - high in sparse, low in dense
        backend.eval(keep_weights)

        # Log occupancy stats
        mean_keep = float(backend.to_scalar(backend.mean(keep_weights)))
        if occupancy_weights is not None:
            mean_occ = float(backend.to_scalar(backend.mean(occ)))
            logger.debug(
                "NULL-SPACE: dim=%d, mean_keep=%.3f, prior_occupancy=%.3f",
                d, mean_keep, mean_occ
            )
        else:
            logger.debug("NULL-SPACE: dim=%d, mean_keep=%.3f (no prior occupancy)", d, mean_keep)

        # Apply variance-weighted projection
        # delta_safe = delta * keep_weights (per-dimension scaling)
        n_rows = int(delta_proj.shape[0])
        keep_weights_row = backend.reshape(keep_weights, (1, d))
        delta_safe = delta_proj * keep_weights_row
        backend.eval(delta_safe)

        if str(proj_dtype) != str(delta_dtype):
            delta_safe = backend.astype(delta_safe, delta_dtype)
            backend.eval(delta_safe)

        # Delta occupancy (per-dim magnitude of applied delta)
        delta_weights = None
        delta_reg = regularization_epsilon(backend, delta_safe)
        delta_magnitude = backend.mean(backend.abs(delta_safe), axis=0)
        max_delta = backend.max(delta_magnitude)
        backend.eval(delta_magnitude, max_delta)
        max_delta_val = float(backend.to_scalar(max_delta))
        if max_delta_val > delta_reg:
            delta_weights = delta_magnitude / max_delta_val
        else:
            delta_weights = backend.zeros((d,))
        backend.eval(delta_weights)

        # Compute geodesic Frobenius-like norms on a shared graph
        original_norm, filtered_norm = _geodesic_frobenius_norms_pair(
            delta_proj, delta_safe, backend
        )
        filtering_applied = abs(original_norm - filtered_norm) > reg
        orthogonal_dim = basis.orthogonal_dim

        if original_norm > 0:
            preserved_fraction = filtered_norm / original_norm
            projection_loss = 1.0 - preserved_fraction
        else:
            preserved_fraction = 1.0
            projection_loss = 0.0

        # Reshape back to original
        filtered_delta = delta_safe
        if transpose_back:
            filtered_delta = backend.transpose(filtered_delta)
        if len(original_shape) == 1:
            filtered_delta = backend.reshape(filtered_delta, original_shape)
        backend.eval(filtered_delta)

        return GeodesicNullSpaceResult(
            filtered_delta=filtered_delta,
            original_delta=weight_delta,
            orthogonal_dim=orthogonal_dim,
            projection_loss=projection_loss,
            preserved_fraction=preserved_fraction,
            original_norm=original_norm,
            filtered_norm=filtered_norm,
            filtering_applied=filtering_applied,
            k_neighbors=basis.k_neighbors,
            mean_geodesic_distance=basis.mean_geodesic_distance,
            delta_weights=delta_weights,
        )

    def _compute_basis(
        self,
        prior_activations: Any,
        k_neighbors: int | None = None,
        source_activations: Any | None = None,
    ) -> GeodesicNullSpaceBasis:
        """Compute a reusable geodesic null-space basis for projections.

        Uses Random Matrix Theory (Marchenko-Pastur distribution) to separate
        true signal from noise, providing principled thresholds instead of
        arbitrary variance heuristics.

        Args:
            prior_activations: Target model activations defining the manifold.
            k_neighbors: k-NN connectivity for geodesic computation.
            source_activations: Optional source model activations (already aligned).
                If provided, transfer weights are computed as:
                    transfer = target_noise × source_signal
                This transfers knowledge where target has room AND source has knowledge.

                If not provided, uses target-only RMT filtering:
                    keep = target_noise_fraction (available dimensions)
        """
        backend = self._backend
        # Only wrap if not already a backend array - avoid breaking id()-based cache
        if not hasattr(prior_activations, 'shape'):
            prior_activations = backend.array(prior_activations)
        backend.eval(prior_activations)

        # Cache only if no source activations (source changes the basis computation)
        cache_enabled = source_activations is None
        cache_key = _cache.make_basis_key(prior_activations, backend, k_neighbors)
        if cache_enabled:
            cached = _cache.get_basis(cache_key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        n_samples = int(prior_activations.shape[0])
        d = int(prior_activations.shape[1])

        geo_result = self._riemannian.geodesic_distances(
            prior_activations, k_neighbors=k_neighbors
        )
        actual_k = geo_result.k_neighbors
        geo_distances = geo_result.distances
        backend.eval(geo_distances)

        # Compute mean geodesic distance for scale reference
        n = n_samples
        eye_mask = backend.eye(n)
        finite_mask = backend.isfinite(geo_distances)
        non_diag_mask = eye_mask < 0.5  # Off-diagonal
        valid_mask = finite_mask * non_diag_mask
        count_dtype = precision_dtype(backend, reference=geo_distances)
        valid_count = backend.sum(backend.astype(valid_mask, count_dtype))
        backend.eval(valid_count)
        valid_count_val = int(backend.to_scalar(valid_count))

        if valid_count_val > 0:
            masked_geo = backend.where(
                valid_mask, geo_distances, backend.zeros_like(geo_distances)
            )
            mean_geo_arr = backend.sum(masked_geo) / valid_count_val
            backend.eval(mean_geo_arr)
            mean_geo = float(backend.to_scalar(mean_geo_arr))
        else:
            mean_geo = 0.0

        # Compute Fréchet mean and tangent space (GPU)
        frechet_result = self._riemannian.frechet_mean(
            prior_activations,
            k_neighbors=actual_k,
            geo_result=geo_result,
        )
        frechet_mean = frechet_result.mean
        backend.eval(frechet_mean)

        # Log map: tangent vectors from mean to each activation
        tangent_vectors = self._riemannian.log_map(
            prior_activations, frechet_mean, geo_result=geo_result
        )
        backend.eval(tangent_vectors)

        reg = regularization_epsilon(backend, prior_activations)

        # =================================================================
        # RMT-BASED NULL-SPACE DETECTION
        # =================================================================
        # Uses Random Matrix Theory (Marchenko-Pastur distribution) to
        # separate true signal from noise in a principled, data-derived way.
        #
        # For TARGET activations:
        #   - Eigenvalues above MP bulk edge = SIGNAL (occupied, protect)
        #   - Eigenvalues in MP bulk = NOISE (unused, available for transfer)
        #   - target_noise_fraction = how much of each dimension is noise
        #
        # For SOURCE activations (if available):
        #   - Eigenvalues above MP bulk edge = SIGNAL (knowledge to transfer)
        #   - source_signal_fraction = how much knowledge each dimension has
        #
        # Transfer weight = target_noise × source_signal
        #   - Transfer where target has room AND source has knowledge
        #   - If no source: transfer weight = target_noise
        # =================================================================

        # Compute target noise fraction using RMT
        target_keep_weights, target_mp = compute_rmt_null_space_weights(
            prior_activations, backend=backend
        )
        backend.eval(target_keep_weights)

        if source_activations is not None:
            # Compute source signal fraction using RMT
            if not hasattr(source_activations, "shape"):
                source_activations = backend.array(source_activations)
            backend.eval(source_activations)

            source_keep_weights, source_mp = compute_rmt_null_space_weights(
                source_activations, backend=backend
            )
            backend.eval(source_keep_weights)

            # Source signal = 1 - source_noise (dimensions where source has knowledge)
            source_signal = 1.0 - source_keep_weights
            backend.eval(source_signal)

            # Transfer = target_noise × source_signal
            # High where target has room AND source has knowledge
            keep_weights = target_keep_weights * source_signal
            backend.eval(keep_weights)

            logger.debug(
                "RMT transfer: target_noise_mean=%.3f, source_signal_mean=%.3f, keep_mean=%.3f",
                float(backend.to_scalar(backend.mean(target_keep_weights))),
                float(backend.to_scalar(backend.mean(source_signal))),
                float(backend.to_scalar(backend.mean(keep_weights))),
            )
        else:
            # Target-only: transfer where target has room (noise dimensions)
            keep_weights = target_keep_weights

            logger.debug(
                "RMT transfer (target-only): noise_mean=%.3f",
                float(backend.to_scalar(backend.mean(keep_weights))),
            )

        combined_weights = 1.0 - keep_weights  # For compatibility with filter_delta
        backend.eval(combined_weights)

        Q = backend.reshape(combined_weights, (d, 1))  # [d, 1] for compatibility
        backend.eval(Q)

        # Orthogonal dimension: sum of keep_weights (effective null space size)
        # Higher = more room for delta, lower = more protected
        effective_null = backend.sum(keep_weights)
        backend.eval(effective_null)
        orthogonal_dim = int(float(backend.to_scalar(effective_null)))

        basis = GeodesicNullSpaceBasis(
            Q=Q,
            orthogonal_dim=orthogonal_dim,
            k_neighbors=actual_k,
            mean_geodesic_distance=mean_geo,
            regularization=reg,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if cache_enabled:
            _cache.set_basis(cache_key, basis, elapsed_ms)
            if k_neighbors is None:
                explicit_key = _cache.make_basis_key(prior_activations, backend, actual_k)
                _cache.set_basis(explicit_key, basis, elapsed_ms)
        return basis


def filter_delta_svd(
    weight_delta: Any,
    backend: "Backend | None" = None,
    energy_threshold: float | None = None,
) -> GeodesicNullSpaceResult:
    """SVD-based delta filtering with low-rank truncation.

    Parameters
    ----------
    weight_delta : Array
        Weight difference (source_aligned - target). Shape: [out, in] or [d].
    backend : Backend, optional
        Backend for tensor operations.
    energy_threshold : float, optional
        Fraction of total energy (Frobenius norm squared) to preserve.
        If None, uses numerical rank from precision-derived SVD cutoff.

    Returns
    -------
    GeodesicNullSpaceResult
        Result with filtered delta and diagnostics.

    References
    ----------
    - Yu et al. (2025). "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
    - Zhang et al. (2025). "STF: Superpose Task-specific Features"
    """
    b = backend or get_default_backend()
    delta = b.array(weight_delta)
    delta = b.astype(delta, precision_dtype(b, reference=delta))
    b.eval(delta)

    original_shape = delta.shape
    original_ndim = len(original_shape)

    # Flatten to 2D if needed (1D vectors treated as [1, d])
    if original_ndim == 1:
        delta_2d = b.reshape(delta, (1, int(original_shape[0])))
    else:
        delta_2d = delta
    b.eval(delta_2d)

    m, n = int(delta_2d.shape[0]), int(delta_2d.shape[1])

    reg = regularization_epsilon(b, delta_2d)
    # Check for zero or near-zero delta (geodesic Frobenius-like norm)
    delta_norm = _geodesic_frobenius_norm(delta_2d, b)
    if delta_norm < reg:
        # Delta is effectively zero - return unchanged
        return GeodesicNullSpaceResult(
            filtered_delta=weight_delta,
            original_delta=weight_delta,
            orthogonal_dim=0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            original_norm=0.0,
            filtered_norm=0.0,
            filtering_applied=False,
            k_neighbors=0,
            mean_geodesic_distance=0.0,
        )

    # Compute SVD: delta = U @ diag(S) @ Vt
    U, S, Vt = geodesic_svd(b, delta_2d)
    b.eval(U, S, Vt)

    n_sv = int(S.shape[0])
    if n_sv == 0:
        return GeodesicNullSpaceResult(
            filtered_delta=weight_delta,
            original_delta=weight_delta,
            orthogonal_dim=0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            original_norm=delta_norm,
            filtered_norm=delta_norm,
            filtering_applied=False,
            k_neighbors=0,
            mean_geodesic_distance=0.0,
        )

    # Auto-determine rank by energy threshold (or numeric rank if None)
    k = svd_auto_rank(S, b, energy_threshold, max_dim=max(m, n))
    k = max(0, min(k, n_sv))

    if k == 0:
        zero_2d = b.zeros_like(delta_2d)
        if original_ndim == 1:
            delta_filtered = b.reshape(zero_2d, original_shape)
        else:
            delta_filtered = zero_2d
        b.eval(delta_filtered)

        if delta_norm > 0:
            preserved_fraction = 0.0
        else:
            preserved_fraction = 1.0

        return GeodesicNullSpaceResult(
            filtered_delta=delta_filtered,
            original_delta=weight_delta,
            orthogonal_dim=0,
            projection_loss=1.0 - preserved_fraction,
            preserved_fraction=preserved_fraction,
            original_norm=delta_norm,
            filtered_norm=0.0,
            filtering_applied=True,
            k_neighbors=0,
            mean_geodesic_distance=0.0,
        )

    # Truncate to top-k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    b.eval(U_k, S_k, Vt_k)

    # Reconstruct low-rank delta: delta_k = U_k @ diag(S_k) @ Vt_k
    # U_k: [m, k], S_k: [k], Vt_k: [k, n]
    S_diag = b.reshape(S_k, (k, 1))  # [k, 1] for broadcasting
    scaled_Vt = S_diag * Vt_k  # [k, n] - each row scaled by singular value
    delta_filtered_2d = b.matmul(U_k, scaled_Vt)  # [m, n]
    b.eval(delta_filtered_2d)

    # Reshape back to original shape
    if original_ndim == 1:
        delta_filtered = b.reshape(delta_filtered_2d, original_shape)
    else:
        delta_filtered = delta_filtered_2d
    b.eval(delta_filtered)

    # Compute preserved energy fraction (SVD spectrum)
    S_sq = S * S
    S_k_sq = S_k * S_k
    total_energy = b.sum(S_sq)
    kept_energy = b.sum(S_k_sq)
    b.eval(total_energy, kept_energy)

    total_energy_val = float(b.to_scalar(total_energy))
    kept_energy_val = float(b.to_scalar(kept_energy))

    if total_energy_val > reg:
        energy_preserved = kept_energy_val / total_energy_val
    else:
        energy_preserved = 1.0

    # Compute geodesic Frobenius-like norms for reporting (shared graph)
    original_norm, filtered_norm = _geodesic_frobenius_norms_pair(
        delta_2d, delta_filtered_2d, b
    )

    if original_norm > 0:
        preserved_fraction = filtered_norm / original_norm
    else:
        preserved_fraction = 1.0

    projection_loss = 1.0 - preserved_fraction

    logger.info(
        "SVD FILTER: rank=%d/%d (%.1f%% energy), geo_preserved=%.3f",
        k, n_sv, 100.0 * energy_preserved, preserved_fraction
    )

    return GeodesicNullSpaceResult(
        filtered_delta=delta_filtered,
        original_delta=weight_delta,
        orthogonal_dim=k,  # Using k as "effective dimension"
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        original_norm=original_norm,
        filtered_norm=filtered_norm,
        filtering_applied=True,
        k_neighbors=0,  # Not applicable for SVD method
        mean_geodesic_distance=0.0,  # Not applicable for SVD method
    )


def filter_merge_delta_geodesic(
    source_weights: Any,
    target_weights: Any,
    prior_activations: Any,
    k_neighbors: int | None = None,
    source_activations: Any | None = None,
) -> tuple[Any, GeodesicNullSpaceResult]:
    """
    Compute and filter merge delta using geodesic null-space projection.

    This is the geodesic equivalent of filter_merge_delta_to_null_space().
    Uses only GPU-accelerated operations (no SVD, no pinv).

    NO ALPHA. NO BLENDING. This is geometric ADDITION in geodesic space.

    Formula:
        delta = source - target
        safe_delta = geodesic_null_projection(delta, prior_activations)
        merged = target + safe_delta

    Two modes depending on whether source_activations is provided:

    MODE 1 (legacy): Target-only filtering
        Finds "room" in target based on low variance + low magnitude.

    MODE 2 (density-aware): Source+target relative density
        Transfers where source is DENSE and target is SPARSE.
        Implements the "fog cloud overlay" principle.

    Args:
        source_weights: Source model weights.
        target_weights: Target model weights.
        prior_activations: Target activations defining the manifold structure.
        k_neighbors: k for k-NN graph (auto-derived if None).
        source_activations: Optional source activations (already aligned).
            If provided, enables density-aware transfer mode.

    Returns:
        Tuple of (merged_weights, filter_result).
    """
    backend = get_default_backend()

    # Compute delta
    source_weights = backend.array(source_weights)
    target_weights = backend.array(target_weights)
    backend.eval(source_weights, target_weights)
    delta = source_weights - target_weights
    backend.eval(delta)

    # Filter using geodesic null-space projection (GPU-only)
    geo_filter = GeodesicNullSpaceFilter(backend)
    result = geo_filter.filter_delta(
        delta,
        prior_activations,
        k_neighbors=k_neighbors,
        source_activations=source_activations,
    )

    # Merge = target + projected_delta (NO ALPHA)
    merged = target_weights + result.filtered_delta
    backend.eval(merged)

    return merged, result


# Orthogonal null-space projection for boundary preservation.


@dataclass
class NullSpaceProjectionResult:
    """Result of orthogonal null-space projection."""

    # The projected delta (A @ delta.T ~= 0 within numerical precision)
    projected_delta: Any

    # Original delta before projection
    original_delta: Any

    # Geodesic Frobenius-like norm of original delta
    original_norm: float

    # Geodesic Frobenius-like norm of projected delta
    projected_norm: float

    # Fraction of delta preserved (projected_norm / original_norm)
    preserved_fraction: float

    # Maximum boundary violation: max(|A @ delta.T|)
    # Should be ~0 (within numerical precision)
    boundary_violation: float

    # Rank of the activation matrix (effective constraints)
    activation_rank: int

    # Condition number of A @ A.T (numerical stability indicator)
    condition_number: float


def project_to_null_space(
    delta: Any,
    boundary_activations: Any,
    backend: "Backend | None" = None,
    verify: bool = True,
) -> NullSpaceProjectionResult:
    """Project delta into the orthogonal null-space of boundary activations.

    Uses a pseudoinverse-based projection and can optionally report
    the boundary violation magnitude.
    """
    b = backend or get_default_backend()

    # Ensure arrays and precision dtype
    delta = b.array(delta)
    A = b.array(boundary_activations)
    compute_dtype = precision_dtype(b, reference=delta)
    if hasattr(A, "dtype"):
        try:
            if b.finfo(A.dtype).eps < b.finfo(compute_dtype).eps:
                compute_dtype = A.dtype
        except Exception:
            pass
    delta = b.astype(delta, compute_dtype)
    A = b.astype(A, compute_dtype)
    b.eval(delta, A)

    # Validate dimensions
    delta_shape = b.shape(delta)
    A_shape = b.shape(A)

    if len(delta_shape) != 2:
        raise ValueError(f"delta must be 2D [out, in], got shape {delta_shape}")
    if len(A_shape) != 2:
        raise ValueError(f"activations must be 2D [n, d], got shape {A_shape}")

    out_dim, in_dim = int(delta_shape[0]), int(delta_shape[1])
    n_samples, d = int(A_shape[0]), int(A_shape[1])

    if in_dim != d:
        raise ValueError(
            f"Dimension mismatch: delta has in_dim={in_dim}, "
            f"activations have d={d}. These must match."
        )

    # Compute original norm (geodesic Frobenius-like)
    original_norm = _geodesic_frobenius_norm(delta, b)

    # Handle zero delta
    eps = regularization_epsilon(b, delta)
    if original_norm < eps:
        return NullSpaceProjectionResult(
            projected_delta=delta,
            original_delta=delta,
            original_norm=0.0,
            projected_norm=0.0,
            preserved_fraction=1.0,
            boundary_violation=0.0,
            activation_rank=0,
            condition_number=1.0,
        )

    # Step 1: Compute G = A @ A.T [n, n] - small Gram matrix
    G = b.matmul(A, b.transpose(A))  # [n, n]
    b.eval(G)

    # Step 2: Compute pinv(G) using stable solver
    # G is symmetric positive semi-definite
    # Add small regularization for numerical stability
    reg = regularization_epsilon(b, G)
    G_reg = G + reg * b.eye(n_samples)
    b.eval(G_reg)

    # Compute condition number for diagnostics
    # Using eigenvalues of G
    try:
        eigvals = b.eigvalsh(G)
        b.eval(eigvals)
        eigvals_pos = b.maximum(eigvals, reg)
        max_eig = float(b.to_scalar(b.max(eigvals_pos)))
        min_eig = float(b.to_scalar(b.min(eigvals_pos)))
        condition_number = max_eig / min_eig if min_eig > 0 else float("inf")
        rank_mask = b.astype(eigvals > reg * 10, precision_dtype(b, reference=eigvals))
        activation_rank = int(b.to_scalar(b.sum(rank_mask)))
    except Exception:
        condition_number = float("inf")
        activation_rank = min(n_samples, d)

    # Step 3: Compute G_inv = pinv(G)
    # G_inv @ G ≈ I
    # Using pinv directly - it handles rank-deficiency correctly
    G_inv = b.pinv(G_reg)
    b.eval(G_inv)

    # Step 4: Compute delta_safe = delta - (delta @ A.T) @ G_inv @ A
    # This avoids forming the d×d projection matrix

    # delta @ A.T: [out, in] @ [in, n] = [out, n]
    delta_AT = b.matmul(delta, b.transpose(A))
    b.eval(delta_AT)

    # (delta @ A.T) @ G_inv: [out, n] @ [n, n] = [out, n]
    temp = b.matmul(delta_AT, G_inv)
    b.eval(temp)

    # temp @ A: [out, n] @ [n, in] = [out, in]
    correction = b.matmul(temp, A)
    b.eval(correction)

    # delta_safe = delta - correction
    delta_safe = delta - correction
    b.eval(delta_safe)

    # Compute projected norm (geodesic Frobenius-like)
    projected_norm = _geodesic_frobenius_norm(delta_safe, b)

    preserved_fraction = projected_norm / original_norm if original_norm > 0 else 1.0

    # Verify boundary condition if requested
    boundary_violation = 0.0
    if verify:
        # Compute A @ delta_safe.T - should be ≈ 0
        # A @ delta_safe.T: [n, in] @ [in, out] = [n, out]
        violation_matrix = b.matmul(A, b.transpose(delta_safe))
        b.eval(violation_matrix)
        max_violation = b.max(b.abs(violation_matrix))
        b.eval(max_violation)
        boundary_violation = float(b.to_scalar(max_violation))

        max_abs_A = b.max(b.abs(A))
        max_abs_delta = b.max(b.abs(delta_safe))
        b.eval(max_abs_A, max_abs_delta)
        scale = float(b.to_scalar(max_abs_A)) * float(b.to_scalar(max_abs_delta))
        scale = max(scale, 1.0)
        violation_tol = scale * machine_epsilon(b, delta_safe)

        if boundary_violation > violation_tol:
            logger.warning(
                "NULL-SPACE: Boundary violation %.2e exceeds tolerance. "
                "Check activation rank (%d) and condition number (%.2e).",
                boundary_violation,
                activation_rank,
                condition_number,
            )

    logger.info(
        "NULL-SPACE PROJECTION: preserved=%.1f%%, violation=%.2e, rank=%d/%d, cond=%.2e",
        preserved_fraction * 100,
        boundary_violation,
        activation_rank,
        n_samples,
        condition_number,
    )

    return NullSpaceProjectionResult(
        projected_delta=delta_safe,
        original_delta=delta,
        original_norm=original_norm,
        projected_norm=projected_norm,
        preserved_fraction=preserved_fraction,
        boundary_violation=boundary_violation,
        activation_rank=activation_rank,
        condition_number=condition_number,
    )


def filter_merge_delta_null_space(
    source_weights: Any,
    target_weights: Any,
    boundary_activations: Any,
    backend: "Backend | None" = None,
) -> tuple[Any, NullSpaceProjectionResult]:
    """Merge weights using orthogonal null-space projection.

    Parameters
    ----------
    source_weights : Array
        Source weights (already aligned to target space). Shape: [out, in].
    target_weights : Array
        Target weights to preserve boundary behavior. Shape: [out, in].
    boundary_activations : Array
        Target activations defining the boundary. Shape: [n_samples, in].

    Returns
    -------
    tuple[Array, NullSpaceProjectionResult]
        (merged_weights, projection_result)
    """
    b = backend or get_default_backend()

    source_weights = b.array(source_weights)
    target_weights = b.array(target_weights)
    b.eval(source_weights, target_weights)

    # Compute delta
    delta = source_weights - target_weights
    b.eval(delta)

    # Project to true null-space
    result = project_to_null_space(delta, boundary_activations, backend=b)

    # Merge = target + projected_delta
    merged = target_weights + result.projected_delta
    b.eval(merged)

    return merged, result


@dataclass
class TSVMergeResult:
    """Result of Task Singular Vector merging for N-to-1 merge.

    References:
        CVPR 2025 "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
    """

    # The merged delta (sum of orthogonalized task deltas)
    merged_delta: Any

    # Original deltas before orthogonalization
    original_deltas: list[Any]

    # Number of source tasks merged
    n_tasks: int

    # Rank used for each task (after truncation)
    task_ranks: list[int]

    # Energy preserved for each task (fraction of singular value energy kept)
    task_energy_preserved: list[float]

    # Interference reduction: ||original_sum - orthogonalized_sum|| / ||original_sum||
    # Lower is better - indicates less destructive interference
    interference_reduction: float

    # Geodesic norm of merged delta
    merged_norm: float


def filter_deltas_tsv(
    weight_deltas: list[Any],
    backend: "Backend | None" = None,
    energy_threshold: float | None = None,
    max_rank: int | None = None,
) -> TSVMergeResult:
    """Merge multiple weight deltas using Task Singular Vector orthogonalization.

    Parameters
    ----------
    weight_deltas : list[Array]
        List of N weight deltas from different source models.
        Each delta should be 2D [out, in] with matching shapes.
    backend : Backend, optional
        Compute backend. Uses default if not provided.
    energy_threshold : float, optional
        Fraction of singular value energy to preserve per task.
        If None, uses numerical rank from precision-derived SVD cutoff.
    max_rank : int, optional
        Maximum rank per task. If None, determined by rank selection only.

    Returns
    -------
    TSVMergeResult
        Result containing merged delta and diagnostics.

    References
    ----------
    - CVPR 2025 "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
    - Zhang et al. (2025) "STF: Superpose Task-specific Features"
    """
    b = backend or get_default_backend()
    n_tasks = len(weight_deltas)

    if n_tasks == 0:
        raise ValueError("weight_deltas must contain at least one delta")

    if n_tasks == 1:
        # Single delta: just apply standard SVD filtering
        svd_result = filter_delta_svd(
            weight_deltas[0],
            backend=b,
            energy_threshold=energy_threshold,
        )
        return TSVMergeResult(
            merged_delta=svd_result.filtered_delta,
            original_deltas=weight_deltas,
            n_tasks=1,
            task_ranks=[svd_result.orthogonal_dim],
            task_energy_preserved=[svd_result.preserved_fraction],
            interference_reduction=0.0,
            merged_norm=svd_result.filtered_norm,
        )

    # Ensure all deltas are arrays with consistent dtype
    compute_dtype = precision_dtype(b, reference=b.array(weight_deltas[0]))
    deltas = [b.astype(b.array(d), compute_dtype) for d in weight_deltas]
    for d in deltas:
        b.eval(d)

    # Validate shapes match
    ref_shape = b.shape(deltas[0])
    for i, d in enumerate(deltas):
        if b.shape(d) != ref_shape:
            raise ValueError(
                f"Shape mismatch: delta[0] has shape {ref_shape}, "
                f"delta[{i}] has shape {b.shape(d)}"
            )

    m, n = int(ref_shape[0]), int(ref_shape[1])
    min_dim = min(m, n)
    max_dim = max(m, n)
    reg = regularization_epsilon(b, deltas[0])

    # Step 1: Compute SVD for each delta and determine ranks
    task_svds: list[tuple[Any, Any, Any]] = []
    task_ranks: list[int] = []
    task_energy_preserved: list[float] = []

    for i, delta in enumerate(deltas):
        U, S, Vt = geodesic_svd(b, delta)
        b.eval(U, S, Vt)

        n_sv = int(b.shape(S)[0])
        if n_sv == 0:
            # Zero delta
            task_svds.append((None, None, None))
            task_ranks.append(0)
            task_energy_preserved.append(1.0)
            continue

        # Determine rank by energy threshold
        k = svd_auto_rank(S, b, energy_threshold, max_dim=max_dim)
        if max_rank is not None:
            k = min(k, max_rank)
        k = max(0, min(k, n_sv))

        if k == 0:
            task_svds.append((None, None, None))
            task_ranks.append(0)
            task_energy_preserved.append(0.0)
            continue

        # Compute energy preserved
        S_sq = S * S
        total_energy = b.sum(S_sq)
        kept_energy = b.sum(S_sq[:k])
        b.eval(total_energy, kept_energy)
        energy_frac = float(b.to_scalar(kept_energy)) / max(
            float(b.to_scalar(total_energy)), reg
        )

        task_svds.append((U[:, :k], S[:k], Vt[:k, :]))
        task_ranks.append(k)
        task_energy_preserved.append(energy_frac)

    # Step 2: Stack all U matrices column-wise
    # M = [U_1 | U_2 | ... | U_n]  where each U_i is [m, k_i]
    U_blocks = [svd[0] for svd in task_svds if svd[0] is not None]
    if len(U_blocks) == 0:
        # All deltas are zero
        zero_delta = b.zeros(ref_shape)
        b.eval(zero_delta)
        return TSVMergeResult(
            merged_delta=zero_delta,
            original_deltas=weight_deltas,
            n_tasks=n_tasks,
            task_ranks=task_ranks,
            task_energy_preserved=task_energy_preserved,
            interference_reduction=0.0,
            merged_norm=0.0,
        )

    M = b.concatenate(U_blocks, axis=1)  # [m, sum(k_i)]
    b.eval(M)

    total_cols = int(b.shape(M)[1])

    # Step 3: Procrustes orthogonalization via SVD
    # Q = U_M @ V_M^T where M = U_M @ S_M @ V_M^T
    # This finds the closest orthonormal matrix to M
    U_M, S_M, Vt_M = geodesic_svd(b, M)
    b.eval(U_M, S_M, Vt_M)

    # Q = U_M @ Vt_M (closest orthonormal to M)
    # But we only take as many columns as we had in M
    n_cols = min(int(b.shape(U_M)[1]), total_cols)
    Q = b.matmul(U_M[:, :n_cols], Vt_M[:n_cols, :n_cols])
    b.eval(Q)

    # Step 4: Partition Q back into per-task blocks
    # Q_i = Q[:, start_i:end_i] where ranges correspond to original U_i blocks
    Q_blocks: list[Any | None] = []
    col_idx = 0
    valid_task_idx = 0
    for i in range(n_tasks):
        if task_svds[i][0] is None:
            Q_blocks.append(None)
        else:
            k_i = task_ranks[i]
            if col_idx + k_i <= n_cols:
                Q_i = Q[:, col_idx : col_idx + k_i]
                Q_blocks.append(Q_i)
                b.eval(Q_i)
            else:
                # Edge case: not enough orthogonalized columns
                Q_blocks.append(None)
            col_idx += k_i
            valid_task_idx += 1

    # Step 5: Reconstruct orthogonalized deltas: delta_i' = Q_i @ diag(S_i) @ V_i^T
    ortho_deltas: list[Any] = []
    for i in range(n_tasks):
        if Q_blocks[i] is None or task_svds[i][0] is None:
            # Zero delta or insufficient rank
            ortho_deltas.append(b.zeros(ref_shape))
            continue

        Q_i = Q_blocks[i]
        S_i = task_svds[i][1]
        Vt_i = task_svds[i][2]

        k_i = task_ranks[i]
        # Q_i: [m, k_i], S_i: [k_i], Vt_i: [k_i, n]
        S_diag = b.reshape(S_i, (k_i, 1))  # [k_i, 1] for broadcasting
        scaled_Vt = S_diag * Vt_i  # [k_i, n]
        delta_ortho = b.matmul(Q_i, scaled_Vt)  # [m, n]
        b.eval(delta_ortho)
        ortho_deltas.append(delta_ortho)

    for d in ortho_deltas:
        b.eval(d)

    # Step 6: Merge = sum of orthogonalized deltas
    merged_delta = ortho_deltas[0]
    for d in ortho_deltas[1:]:
        merged_delta = merged_delta + d
    b.eval(merged_delta)

    # Compute interference reduction metric
    # Compare naive sum vs orthogonalized sum
    naive_sum = deltas[0]
    for d in deltas[1:]:
        naive_sum = naive_sum + d
    b.eval(naive_sum)

    naive_norm = _geodesic_frobenius_norm(naive_sum, b)
    merged_norm = _geodesic_frobenius_norm(merged_delta, b)

    # Interference reduction: how much the orthogonalization changed the result
    # Lower difference suggests less destructive interference in original
    if naive_norm > reg:
        diff = naive_sum - merged_delta
        b.eval(diff)
        diff_norm = _geodesic_frobenius_norm(diff, b)
        interference_reduction = diff_norm / naive_norm
    else:
        interference_reduction = 0.0

    logger.info(
        "TSV MERGE: %d tasks, ranks=%s, interference_reduction=%.3f",
        n_tasks,
        task_ranks,
        interference_reduction,
    )

    return TSVMergeResult(
        merged_delta=merged_delta,
        original_deltas=weight_deltas,
        n_tasks=n_tasks,
        task_ranks=task_ranks,
        task_energy_preserved=task_energy_preserved,
        interference_reduction=interference_reduction,
        merged_norm=merged_norm,
    )


def merge_weights_tsv(
    source_weights_list: list[Any],
    target_weights: Any,
    backend: "Backend | None" = None,
    energy_threshold: float | None = None,
    max_rank: int | None = None,
) -> tuple[Any, TSVMergeResult]:
    """Merge multiple source weights into target using TSV orthogonalization.

    This is the high-level API for N-to-1 model merging with reduced interference.

    Parameters
    ----------
    source_weights_list : list[Array]
        List of N source model weights (each already aligned to target space).
    target_weights : Array
        Target weights to add knowledge to.
    backend : Backend, optional
        Compute backend.
    energy_threshold : float, optional
        Fraction of singular value energy to preserve per task.
        If None, uses numerical rank from precision-derived SVD cutoff.
    max_rank : int, optional
        Maximum rank per task.

    Returns
    -------
    tuple[Array, TSVMergeResult]
        (merged_weights, tsv_result)
    """
    b = backend or get_default_backend()

    target = b.array(target_weights)
    b.eval(target)

    # Compute deltas: source_i - target
    deltas = []
    for source in source_weights_list:
        src = b.array(source)
        b.eval(src)
        delta = src - target
        b.eval(delta)
        deltas.append(delta)

    # Apply TSV merging
    result = filter_deltas_tsv(
        deltas,
        backend=b,
        energy_threshold=energy_threshold,
        max_rank=max_rank,
    )

    # Merge = target + orthogonalized_merged_delta
    merged = target + result.merged_delta
    b.eval(merged)

    return merged, result


__all__ = [
    "GeodesicNullSpaceBasis",
    "GeodesicNullSpaceResult",
    "GeodesicNullSpaceFilter",
    "filter_delta_svd",  # SVD-based low-rank filtering
    "filter_merge_delta_geodesic",  # Legacy variance-based filtering (HEURISTIC)
    # Orthogonal null-space projection
    "NullSpaceProjectionResult",
    "project_to_null_space",
    "filter_merge_delta_null_space",
    # Task Singular Vector merging (N-to-1)
    "TSVMergeResult",
    "filter_deltas_tsv",
    "merge_weights_tsv",
]
