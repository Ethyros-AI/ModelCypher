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

"""
Geodesic Null-Space Filtering for interference-free model merging.

This module provides GPU-accelerated geodesic null-space filtering for model merging.
Flat-space linear algebra (SVD, pinv, eigendecomposition) is mathematically wrong for
high-dimensional manifolds (8kD+) - it only works up to 3D. This module uses geodesic
geometry to find directions orthogonal to the manifold structure accurately.

Key insight: In high-dimensional spaces with curvature, the "null space" concept
from linear algebra (flat space) should be replaced with "geodesic-orthogonal
directions" - directions that don't disturb the manifold's geodesic structure.

Mathematical Foundation:
    On a Riemannian manifold (M, g), we want to find directions orthogonal to
    the local tangent space structure. Instead of:

        null(A) = {v : Av = 0}  (linear, flat space)

    We compute:

        geodesic_orthogonal(X) = {v : ∇_v d(x, y) = 0 for all x, y ∈ neighbors}

    In practice, we approximate this by:
    1. Building k-NN graph (captures local manifold structure)
    2. Computing geodesic distances (exact on discrete manifold)
    3. Finding directions that preserve geodesic structure

GPU Acceleration:
    All operations use only GPU-friendly primitives:
    - matmul: Pairwise distance computation
    - argsort: k-NN construction
    - vectorized min: Floyd-Warshall shortest paths
    - Gram matrix operations: Tangent space projection

    All linear algebra stays on the backend (no CPU fallbacks).

Usage:
    filter = GeodesicNullSpaceFilter(backend)
    result = filter.filter_delta(weight_delta, prior_activations)
    safe_delta = result.filtered_delta

References:
    - Tenenbaum et al. (2000) "Isomap" - geodesic via graph
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - ModelCypher CLAUDE.md: "Geodesic is Correct"
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
    regularization_epsilon,
    svd_auto_rank,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()


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

    The key insight is that on curved manifolds, "null space" should mean
    "directions that don't disturb the manifold structure" - not just
    "directions orthogonal in flat space."

    All operations are GPU-accelerated:
    - Pairwise distances: matmul
    - k-NN graph: argsort
    - Geodesic distances: vectorized Floyd-Warshall
    - Projection: Gram matrix operations
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
        source_sample_weights: Any | None = None,
        target_sample_weights: Any | None = None,
    ) -> GeodesicNullSpaceResult:
        """
        Filter weight delta using unified saliency projection.

        Two modes depending on whether source_activations is provided:

        MODE 1 (legacy): Target-only filtering
            Combines variance and magnitude signals from target:
            keep = (1 - target_variance) * (1 - target_magnitude)

        MODE 2 (density-aware): Source+target relative density
            Transfers knowledge where source is dense AND target is sparse:
            transfer = source_confidence * (1 - target_confidence) * (1 - magnitude)

            - Source dense, target sparse → transfer ≈ 1 (source's unique knowledge)
            - Source sparse, target dense → transfer ≈ 0 (protect target)
            - Both dense → transfer ≈ 0 (target already knows it)
            - Both sparse → transfer ≈ 0 (neither knows, ignore)

        Algorithm:
            1. Build k-NN graph from activations (GPU: matmul + argsort)
            2. Compute geodesic distances (GPU: Floyd-Warshall)
            3. Compute per-dimension variance for source AND target
            4. Scale delta by transfer weights per dimension

        Args:
            weight_delta: Weight update to filter. Shape: [out, in] or [d].
            prior_activations: Target activation matrix. Shape: [n_samples, d].
            k_neighbors: k for k-NN graph. If None, auto-derived.
            basis: Optional precomputed geodesic basis for reuse.
            source_activations: Optional source activations (already aligned).
                If provided, enables density-aware transfer mode.

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
            source_sample_weights=source_sample_weights,
            target_sample_weights=target_sample_weights,
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
        proj_dtype = delta_dtype
        if str(delta_dtype) in ("float16", "bfloat16"):
            proj_dtype = "float32"
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

        # Compute metrics (single geodesic pass for original/filtered norms)
        norm_inputs = backend.concatenate(
            [
                backend.reshape(delta_flat, (1, -1)),
                backend.reshape(delta_safe, (1, -1)),
            ],
            axis=0,
        )
        norms_arr = geodesic_norms(norm_inputs, backend)
        backend.eval(norms_arr)

        original_norm = float(backend.to_scalar(norms_arr[0]))
        filtered_norm = float(backend.to_scalar(norms_arr[1]))
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
        source_sample_weights: Any | None = None,
        target_sample_weights: Any | None = None,
    ) -> GeodesicNullSpaceBasis:
        """Compute a reusable geodesic null-space basis for projections.

        Args:
            prior_activations: Target model activations defining the manifold.
            k_neighbors: k-NN connectivity for geodesic computation.
            source_activations: Optional source model activations (already aligned).
                If provided, transfer weights are computed as:
                    transfer = source_confidence * (1 - target_confidence)
                This transfers knowledge where source is dense AND target is sparse.

                If not provided (legacy mode), uses target-only filtering:
                    keep = (1 - target_variance) * (1 - target_magnitude)
        """
        backend = self._backend
        # Only wrap if not already a backend array - avoid breaking id()-based cache
        if not hasattr(prior_activations, 'shape'):
            prior_activations = backend.array(prior_activations)
        backend.eval(prior_activations)

        cache_enabled = source_sample_weights is None and target_sample_weights is None
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
        valid_count = backend.sum(backend.astype(valid_mask, "float32"))
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

        # =====================================================================
        # UNIFIED SALIENCY PROJECTION (Variance + Magnitude)
        # =====================================================================
        # The shape is invariant. All models encode the same manifold.
        # What differs is DENSITY and SENSITIVITY across dimensions.
        #
        # Binary null-space projection (Q @ Q^T) is WRONG because:
        # - With many samples, Q spans all dimensions
        # - orthogonal_dim = 0, so delta_safe = delta - delta = 0
        #
        # Unified saliency projection is CORRECT because it combines:
        #
        # 1. VARIANCE (occupancy): How spread are concepts in this direction?
        #    High variance = densely populated = protect
        #    Low variance = sparse = room for new concepts
        #
        # 2. MAGNITUDE (sensitivity, from AIM arXiv:2502.02421):
        #    High activation magnitude = perturbations cause large errors = protect
        #    Low magnitude = insensitive to changes = safe to modify
        #
        # Combined formula (probabilistic AND for "safe to modify"):
        #   keep[i] = (1 - variance_norm[i]) * (1 - magnitude_norm[i])
        #   delta_safe = delta * keep
        #
        # This means: keep delta only where BOTH variance AND magnitude are low.
        # Either high variance OR high magnitude triggers protection.
        # No hyperparameters - pure geometry.
        # =====================================================================

        # Compute per-dimension statistics from activations.
        A = backend.transpose(tangent_vectors)  # [d, n_samples]
        if str(backend.dtype(A)) in ("float16", "bfloat16"):
            A = backend.astype(A, "float32")
        reg = regularization_epsilon(backend, A)

        def _weighted_variance(
            activations: "Any",
            sample_weights: "Any | None",
        ) -> "Any":
            acts = backend.astype(activations, "float32")
            if sample_weights is None:
                mean = backend.mean(acts, axis=0)
                centered = acts - backend.reshape(mean, (1, d))
                var = backend.mean(centered * centered, axis=0)
                backend.eval(var)
                return var

            weights = sample_weights
            if not hasattr(weights, "shape"):
                weights = backend.array(weights)
            weights = backend.astype(weights, "float32")
            weights = backend.reshape(weights, (-1, 1))
            backend.eval(weights)

            weight_sum = backend.sum(weights)
            backend.eval(weight_sum)
            weight_sum_val = float(backend.to_scalar(weight_sum))
            if weight_sum_val <= reg:
                var = backend.zeros((d,))
                backend.eval(var)
                return var

            weighted_sum = backend.sum(weights * acts, axis=0)
            mean = weighted_sum / weight_sum
            centered = acts - backend.reshape(mean, (1, d))
            var = backend.sum(weights * centered * centered, axis=0) / weight_sum
            backend.eval(var)
            return var

        def _weighted_mean_abs(
            activations: "Any",
            sample_weights: "Any | None",
        ) -> "Any":
            acts = backend.astype(activations, "float32")
            if sample_weights is None:
                mean_abs = backend.mean(backend.abs(acts), axis=0)
                backend.eval(mean_abs)
                return mean_abs

            weights = sample_weights
            if not hasattr(weights, "shape"):
                weights = backend.array(weights)
            weights = backend.astype(weights, "float32")
            weights = backend.reshape(weights, (-1, 1))
            backend.eval(weights)

            weight_sum = backend.sum(weights)
            backend.eval(weight_sum)
            weight_sum_val = float(backend.to_scalar(weight_sum))
            if weight_sum_val <= reg:
                mean_abs = backend.zeros((d,))
                backend.eval(mean_abs)
                return mean_abs

            weighted_sum = backend.sum(weights * backend.abs(acts), axis=0)
            mean_abs = weighted_sum / weight_sum
            backend.eval(mean_abs)
            return mean_abs

        def _coerce_sample_weights(
            sample_weights: "Any | None",
            expected_len: int,
            label: str,
        ) -> "Any | None":
            if sample_weights is None:
                return None
            weights = sample_weights
            if not hasattr(weights, "shape"):
                weights = backend.array(weights)
            weights = backend.reshape(weights, (-1,))
            backend.eval(weights)
            if int(backend.shape(weights)[0]) != expected_len:
                logger.warning(
                    "DENSITY: %s sample weights len %d != %d; ignoring",
                    label,
                    int(backend.shape(weights)[0]),
                    expected_len,
                )
                return None
            return weights

        # =================================================================
        # COMBINED SALIENCY: Unified projection weights
        # =================================================================
        # Two modes depending on whether source activations are available:
        #
        # MODE 1 (legacy, source_activations=None):
        #   keep_weights = (1 - target_variance) * (1 - target_magnitude)
        #   This finds "room" in target but ignores source knowledge density.
        #
        # MODE 2 (density-aware, source_activations provided):
        #   transfer_weights = source_confidence * (1 - target_confidence)
        #   This transfers knowledge where source is DENSE and target is SPARSE.
        #
        #   - Source dense, target sparse → transfer ≈ 1 (source's unique knowledge)
        #   - Source sparse, target dense → transfer ≈ 0 (protect target)
        #   - Both dense → transfer ≈ 0 (target already knows it)
        #   - Both sparse → transfer ≈ 0 (neither knows, ignore)
        #
        # Mode 2 implements the "fog cloud overlay" principle:
        # Transfer density from source where target has gaps, don't dilute.
        # =================================================================

        if source_activations is not None:
            # MODE 2: Density-aware transfer using source vs target variance.
            # If sample weights are provided and aligned, use weighted variance.
            if not hasattr(source_activations, "shape"):
                source_activations = backend.array(source_activations)
            backend.eval(source_activations)

            # Compute per-dimension variance for each activation set
            source_weights = _coerce_sample_weights(
                source_sample_weights,
                int(backend.shape(source_activations)[0]),
                "source",
            )
            target_weights = _coerce_sample_weights(
                target_sample_weights,
                n_samples,
                "target",
            )
            source_variance = _weighted_variance(source_activations, source_weights)
            target_variance = _weighted_variance(prior_activations, target_weights)

            src_max = backend.max(source_variance)
            tgt_max = backend.max(target_variance)
            backend.eval(src_max, tgt_max)
            src_max_val = float(backend.to_scalar(src_max))
            tgt_max_val = float(backend.to_scalar(tgt_max))

            source_confidence = (
                source_variance / src_max_val if src_max_val > reg else backend.zeros((d,))
            )
            target_confidence = (
                target_variance / tgt_max_val if tgt_max_val > reg else backend.zeros((d,))
            )
            backend.eval(source_confidence, target_confidence)

            # Transfer where source is denser than target (no tunable thresholds).
            diff = source_confidence - target_confidence
            diff_pos = backend.maximum(diff, backend.zeros_like(diff))
            denom = backend.maximum(
                source_confidence + target_confidence,
                backend.full(backend.shape(diff), reg),
            )
            keep_weights = diff_pos / denom
            backend.eval(keep_weights)

            logger.debug(
                "DENSITY-AWARE transfer: source_conf_mean=%.3f, target_conf_mean=%.3f, keep_mean=%.3f",
                float(backend.to_scalar(backend.mean(source_confidence))),
                float(backend.to_scalar(backend.mean(target_confidence))),
                float(backend.to_scalar(backend.mean(keep_weights))),
            )
        else:
            # MODE 1 (legacy): Target-only filtering (variance + magnitude).
            target_weights = _coerce_sample_weights(
                target_sample_weights,
                n_samples,
                "target",
            )
            variance_per_dim = _weighted_variance(prior_activations, target_weights)
            magnitude_per_dim = _weighted_mean_abs(prior_activations, target_weights)

            max_var = backend.max(variance_per_dim)
            max_mag = backend.max(magnitude_per_dim)
            backend.eval(max_var, max_mag)
            max_var_val = float(backend.to_scalar(max_var))
            max_mag_val = float(backend.to_scalar(max_mag))

            if max_var_val > reg:
                variance_weights = variance_per_dim / max_var_val
            else:
                variance_weights = backend.zeros((d,))
            backend.eval(variance_weights)

            if max_mag_val > reg:
                magnitude_weights = magnitude_per_dim / max_mag_val
            else:
                magnitude_weights = backend.zeros((d,))
            backend.eval(magnitude_weights)

            keep_weights = (1.0 - variance_weights) * (1.0 - magnitude_weights)
            backend.eval(keep_weights)

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
    energy_threshold: float = 0.99,
) -> GeodesicNullSpaceResult:
    """SVD-based delta filtering with low-rank truncation.

    This is a closed-form approach that replaces variance-based filtering.
    Task matrices (deltas) are inherently low-rank - research shows only ~3%
    of singular components capture 98.5% of task-specific information.

    Algorithm:
        1. Decompose delta: U, S, Vt = SVD(delta)
        2. Auto-rank: k = first index where cumsum(S^2) >= 0.99 * total
        3. Truncate: delta_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        4. Return: low-rank filtered delta

    This separates signal (top-k singular components) from noise (remaining)
    without arbitrary variance/magnitude thresholds.

    Parameters
    ----------
    weight_delta : Array
        Weight difference (source_aligned - target). Shape: [out, in] or [d].
    backend : Backend, optional
        Backend for tensor operations.
    energy_threshold : float
        Fraction of total energy (Frobenius norm squared) to preserve.
        Default 0.99 captures nearly all task information.

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
    delta = b.astype(b.array(weight_delta), "float32")
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

    # Check for zero or near-zero delta
    delta_norm_sq = b.sum(delta_2d * delta_2d)
    b.eval(delta_norm_sq)
    delta_norm_sq_val = float(b.to_scalar(delta_norm_sq))

    reg = regularization_epsilon(b, delta_2d)
    if delta_norm_sq_val < reg:
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
            original_norm=float(b.to_scalar(b.sqrt(delta_norm_sq))),
            filtered_norm=float(b.to_scalar(b.sqrt(delta_norm_sq))),
            filtering_applied=False,
            k_neighbors=0,
            mean_geodesic_distance=0.0,
        )

    # Auto-determine rank by energy threshold
    k = svd_auto_rank(S, b, energy_threshold)
    k = max(1, min(k, n_sv))  # Ensure k is valid

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

    # Compute preserved fraction (ratio of energy)
    S_sq = S * S
    S_k_sq = S_k * S_k
    total_energy = b.sum(S_sq)
    kept_energy = b.sum(S_k_sq)
    b.eval(total_energy, kept_energy)

    total_energy_val = float(b.to_scalar(total_energy))
    kept_energy_val = float(b.to_scalar(kept_energy))

    if total_energy_val > reg:
        preserved_fraction = kept_energy_val / total_energy_val
    else:
        preserved_fraction = 1.0

    # Compute norms for reporting
    original_norm = float(b.to_scalar(b.sqrt(delta_norm_sq)))
    filtered_norm_sq = b.sum(delta_filtered * delta_filtered)
    b.eval(filtered_norm_sq)
    filtered_norm = float(b.to_scalar(b.sqrt(filtered_norm_sq)))

    projection_loss = 1.0 - preserved_fraction

    logger.info(
        "SVD FILTER: rank=%d/%d (%.1f%% energy), preserved_fraction=%.3f",
        k, n_sv, 100.0 * preserved_fraction, preserved_fraction
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


# =============================================================================
# TRUE ORTHOGONAL NULL-SPACE PROJECTION
# =============================================================================
# This is the mathematically correct implementation that guarantees:
#     A @ delta_safe.T = 0  (boundary activations preserved)
#
# The heuristic variance-weighted approach above does NOT guarantee this.
# Use this function for precise boundary preservation.
# =============================================================================


@dataclass
class NullSpaceProjectionResult:
    """Result of true orthogonal null-space projection."""

    # The projected delta (guaranteed: A @ delta.T = 0)
    projected_delta: Any

    # Original delta before projection
    original_delta: Any

    # Frobenius norm of original delta
    original_norm: float

    # Frobenius norm of projected delta
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
    """Project delta into the true orthogonal null-space of boundary activations.

    MATHEMATICAL GUARANTEE:
    =======================
    Given boundary activations A [n_samples, d], this function computes:

        P_null = I - A.T @ pinv(A @ A.T) @ A

    And returns:

        delta_safe = delta @ P_null

    This GUARANTEES:

        A @ delta_safe.T = 0

    Meaning the boundary behavior is EXACTLY preserved (within numerical precision).

    EFFICIENT COMPUTATION:
    ======================
    We avoid materializing the d×d projection matrix by computing:

        delta_safe = delta - (delta @ A.T) @ pinv(A @ A.T) @ A

    This is efficient when n_samples << d (typical: 45-2048 probes, 576-4096 dims).

    Parameters
    ----------
    delta : Array
        Weight delta to project. Shape: [out_dim, in_dim].
        in_dim must match boundary_activations feature dimension.
    boundary_activations : Array
        Target activations defining the boundary. Shape: [n_samples, d].
        These are the activations we want to preserve exactly.
    backend : Backend, optional
        Computation backend. Defaults to MLX.
    verify : bool
        If True, compute and return boundary violation metric.

    Returns
    -------
    NullSpaceProjectionResult
        Contains projected_delta with boundary_violation ≈ 0.

    Mathematical Derivation
    -----------------------
    For weight matrix W: [out, in] and activations A: [n, in]:
        output = A @ W.T  [n, out]

    We want: A @ (W_target + delta_safe).T = A @ W_target.T
    Therefore: A @ delta_safe.T = 0
    So delta_safe.T must be in null(A) = {v : A @ v = 0}

    The projection onto null(A) is:
        P_null = I - A.T @ (A @ A.T)^{-1} @ A  (when invertible)
        P_null = I - A.T @ pinv(A @ A.T) @ A   (general)

    For delta [out, in]:
        delta_safe = delta @ P_null
                   = delta - delta @ A.T @ pinv(A @ A.T) @ A
    """
    b = backend or get_default_backend()

    # Ensure arrays and float32 precision
    delta = b.astype(b.array(delta), "float32")
    A = b.astype(b.array(boundary_activations), "float32")
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

    # Compute original norm
    original_norm_sq = b.sum(delta * delta)
    b.eval(original_norm_sq)
    original_norm = float(b.to_scalar(b.sqrt(original_norm_sq)))

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
        activation_rank = int(b.to_scalar(b.sum(b.cast(eigvals > reg * 10, "float32"))))
    except Exception:
        condition_number = float("inf")
        activation_rank = min(n_samples, d)

    # Step 3: Compute G_inv = pinv(G) via Cholesky or direct solve
    # G_inv @ G ≈ I
    try:
        # Try Cholesky for positive definite
        L = b.cholesky(G_reg)
        b.eval(L)
        # G_inv = (L.T)^{-1} @ L^{-1}
        # We'll compute G_inv @ x by solving L @ L.T @ y = x
        # For now, use pinv directly
        G_inv = b.pinv(G_reg)
    except Exception:
        # Fall back to pinv
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

    # Compute projected norm
    projected_norm_sq = b.sum(delta_safe * delta_safe)
    b.eval(projected_norm_sq)
    projected_norm = float(b.to_scalar(b.sqrt(projected_norm_sq)))

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

        if boundary_violation > 1e-4:
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
    """Merge weights using TRUE orthogonal null-space projection.

    This is the mathematically correct merge formula:

        delta = source_aligned - target
        delta_safe = project_to_null_space(delta, boundary_activations)
        merged = target + delta_safe

    GUARANTEE:
        boundary_activations @ merged.T = boundary_activations @ target.T

    The target's behavior on boundary activations is EXACTLY preserved.

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


__all__ = [
    "GeodesicNullSpaceBasis",
    "GeodesicNullSpaceResult",
    "GeodesicNullSpaceFilter",
    "filter_delta_svd",  # SVD-based low-rank filtering
    "filter_merge_delta_geodesic",  # Legacy variance-based filtering (HEURISTIC)
    # TRUE ORTHOGONAL NULL-SPACE PROJECTION (mathematically correct)
    "NullSpaceProjectionResult",
    "project_to_null_space",
    "filter_merge_delta_null_space",
]
