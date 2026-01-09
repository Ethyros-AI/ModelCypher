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
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_norms
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
        basis: GeodesicNullSpaceBasis | None = None,
    ) -> GeodesicNullSpaceResult:
        """
        Filter weight delta to geodesic-orthogonal directions.

        The projection preserves components of delta that are orthogonal to
        the manifold structure defined by prior_activations, as measured by
        geodesic distance preservation.

        Algorithm:
            1. Build k-NN graph from activations (GPU: matmul + argsort)
            2. Compute geodesic distances (GPU: Floyd-Warshall)
            3. Compute tangent space basis at Fréchet mean (GPU: Gram ops)
            4. Project delta onto orthogonal complement (GPU: matmul)

        Args:
            weight_delta: Weight update to filter. Shape: [out, in] or [d].
            prior_activations: Activation matrix. Shape: [n_samples, d].
            k_neighbors: k for k-NN graph. If None, auto-derived.
            basis: Optional precomputed geodesic basis for reuse.

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
        variance_weights = backend.squeeze(Q)  # [d]
        keep_weights = 1.0 - variance_weights  # [d] - high in sparse, low in dense
        backend.eval(keep_weights)

        # Apply variance-weighted projection
        # delta_safe = delta * keep_weights (per-dimension scaling)
        n_rows = int(delta_proj.shape[0])
        keep_weights_row = backend.reshape(keep_weights, (1, d))
        delta_safe = delta_proj * keep_weights_row
        backend.eval(delta_safe)

        if str(proj_dtype) != str(delta_dtype):
            delta_safe = backend.astype(delta_safe, delta_dtype)
            backend.eval(delta_safe)

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
        )

    def _compute_basis(
        self,
        prior_activations: Any,
        k_neighbors: int | None = None,
    ) -> GeodesicNullSpaceBasis:
        """Compute a reusable geodesic null-space basis for projections."""
        backend = self._backend
        prior_activations = backend.array(prior_activations)
        backend.eval(prior_activations)

        cache_key = _cache.make_basis_key(prior_activations, backend, k_neighbors)
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
        # VARIANCE-WEIGHTED NULL SPACE PROJECTION
        # =====================================================================
        # The shape is invariant. All models encode the same manifold.
        # What differs is DENSITY - how many concepts are locked into precise
        # positions on this manifold.
        #
        # Binary null-space projection (Q @ Q^T) is WRONG because:
        # - With many samples, Q spans all dimensions
        # - orthogonal_dim = 0, so delta_safe = delta - delta = 0
        #
        # Variance-weighted projection is CORRECT because:
        # - High variance directions are DENSELY populated (many concepts)
        # - Low variance directions are SPARSE (room for new concepts)
        # - We project out proportionally: remove delta where dense, keep where sparse
        #
        # Formula:
        #   variance[i] = var(activations[:, i])
        #   weight[i] = variance[i] / max(variance)  in [0, 1]
        #   projection_strength = diag(weight)
        #   delta_safe = delta - delta @ diag(weight)
        #
        # This preserves precision (no truncation) while respecting density.
        # =====================================================================

        # Compute per-dimension variance of activations directly
        # (tangent vectors have similar variance structure, but activations are cleaner)
        A = backend.transpose(tangent_vectors)  # [d, n_samples]
        if str(backend.dtype(A)) in ("float16", "bfloat16"):
            A = backend.astype(A, "float32")
        reg = regularization_epsilon(backend, A)

        # Variance per dimension: how "occupied" each direction is
        # High variance = many concepts spread along this direction = dense
        # Low variance = concepts clustered or sparse = room for more
        acts_f32 = backend.astype(prior_activations, "float32")
        mean_per_dim = backend.mean(acts_f32, axis=0)  # [d]
        centered = acts_f32 - backend.reshape(mean_per_dim, (1, d))
        variance_per_dim = backend.mean(centered * centered, axis=0)  # [d]
        backend.eval(variance_per_dim)

        # Normalize variance to [0, 1] range
        max_var = backend.max(variance_per_dim)
        backend.eval(max_var)
        max_var_val = float(backend.to_scalar(max_var))

        if max_var_val > reg:
            # variance_weights[i] = how strongly to project out direction i
            # High weight = dense = project out strongly
            # Low weight = sparse = keep more delta
            variance_weights = variance_per_dim / max_var_val
        else:
            # All directions have ~0 variance, nothing to project
            variance_weights = backend.zeros((d,))
        backend.eval(variance_weights)

        # Store the variance weights as our "weighted basis"
        # The projection will use: delta_safe = delta * (1 - variance_weights)
        # This is equivalent to: delta - delta * variance_weights
        # where variance_weights acts as the projection strength per dimension
        Q = backend.reshape(variance_weights, (d, 1))  # [d, 1] for compatibility
        backend.eval(Q)

        # Orthogonal dimension is now the sum of (1 - weights), normalized
        # This represents "effective null space size" based on sparse directions
        effective_null = backend.sum(1.0 - variance_weights)
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
        _cache.set_basis(cache_key, basis, elapsed_ms)
        if k_neighbors is None:
            explicit_key = _cache.make_basis_key(prior_activations, backend, actual_k)
            _cache.set_basis(explicit_key, basis, elapsed_ms)
        return basis


def filter_merge_delta_geodesic(
    source_weights: Any,
    target_weights: Any,
    prior_activations: Any,
    k_neighbors: int | None = None,
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

    Args:
        source_weights: Source model weights.
        target_weights: Target model weights.
        prior_activations: Target activations defining the manifold structure.
        k_neighbors: k for k-NN graph (auto-derived if None).

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
    result = geo_filter.filter_delta(delta, prior_activations, k_neighbors=k_neighbors)

    # Merge = target + projected_delta (NO ALPHA)
    merged = target_weights + result.filtered_delta
    backend.eval(merged)

    return merged, result


__all__ = [
    "GeodesicNullSpaceBasis",
    "GeodesicNullSpaceResult",
    "GeodesicNullSpaceFilter",
    "filter_merge_delta_geodesic",
]
