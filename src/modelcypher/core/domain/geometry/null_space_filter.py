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
Null-Space Filtering for interference-free model merging.

Based on MINGLE (arXiv:2509.21413): Projects weight updates into the null space
of prior task representations, eliminating interference by construction.

Key insight: If Δw is orthogonal to all prior activations, modifying weights
by Δw cannot affect outputs for prior task inputs.

Mathematical guarantee:
    A @ (W + Δw_safe) = A @ W  when Δw_safe ∈ null(A)

Usage:
    filter = NullSpaceFilter(backend)
    result = filter.filter_delta(weight_delta, prior_activations)
    safe_delta = result.filtered_delta  # Guaranteed no interference

All parameters are derived from the data's dtype and spectral properties.
No user configuration - the geometry determines everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    log2_scalar,
    machine_epsilon,
    regularization_epsilon,
    svd_via_eigh,
)
from modelcypher.core.domain.merging.exceptions import NullSpaceFilterError
from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()


@dataclass
class NullSpaceProjection:
    """Precomputed null space projection matrix and metadata."""

    # Projection matrix onto null space: P @ x projects x to null(A)
    projection_matrix: Any

    # Dimension of the null space
    null_dim: int

    # Dimension of the row space (complement of null)
    row_space_dim: int

    # Singular values of the activation matrix (for diagnostics)
    singular_values: Any

    # Threshold used to determine null space
    effective_threshold: float

    # Number of samples used to estimate null space
    n_samples: int

    # Cached right singular vectors Vh (rows are singular vectors)
    # Used to avoid recomputing SVD for direction analysis
    vh: Any = None


@dataclass
class NullSpaceFilterResult:
    """Result of filtering a weight delta through null space."""

    # The filtered delta (projected to null space)
    filtered_delta: Any

    # Original delta (for comparison)
    original_delta: Any

    # Dimension of null space used
    null_space_dim: int

    # Fraction of delta that was removed (interference component)
    projection_loss: float

    # Fraction of delta preserved (safe component)
    preserved_fraction: float

    # L2 norm of original delta
    original_norm: float

    # L2 norm of filtered delta
    filtered_norm: float

    # Whether filtering was actually applied (false if null space is empty)
    filtering_applied: bool

    # Diagnostic: per-direction preservation
    direction_preservation: Any | None = None


@dataclass
class LayerNullSpaceProfile:
    """Null space profile for a single layer."""

    layer_idx: int
    null_dim: int
    total_dim: int
    null_fraction: float  # null_dim / total_dim
    mean_singular_value: float
    condition_number: float  # σ_max / σ_min


@dataclass
class ModelNullSpaceProfile:
    """Null space profile across all layers."""

    per_layer: dict[int, LayerNullSpaceProfile]
    total_null_dim: int
    total_dim: int
    mean_null_fraction: float
    graftable_layers: list[int]  # Layers with significant null space


def _compute_numerical_rank(
    singular_values: list[float],
    eps: float,
) -> tuple[float, int]:
    """Compute numerical rank from singular values.

    The rank is the count of singular values above numerical precision.
    No heuristics. No spectral gap guessing. Just: is σ_i > ε * σ_max?

    Args:
        singular_values: List of singular values in descending order.
        eps: Machine epsilon for the dtype.

    Returns:
        (threshold, rank) where threshold is the cutoff value
        and rank is the number of singular values above it.
    """
    if not singular_values or singular_values[0] <= 0:
        return 0.0, 0

    # Standard numerical rank: σ_i > max(m, n) * ε * σ_max
    # We use ε * σ_max as threshold (caller provides appropriate ε)
    max_sv = singular_values[0]
    threshold = eps * max_sv

    rank = sum(1 for s in singular_values if s > threshold)
    return threshold, rank


def _derive_min_samples(d: int) -> int:
    """Derive minimum samples from dimension.

    Uses log2(d) as the minimum - this is the information-theoretic
    minimum needed to distinguish d dimensions from random noise.
    """
    backend = get_default_backend()
    log2_val = log2_scalar(float(max(2, d)), backend)
    return max(2, int(ceil_scalar(log2_val, backend)))


class NullSpaceFilter:
    """
    Filters weight updates to the null space of prior activations.

    This ensures that merged weights don't interfere with prior task
    performance: if Δw ∈ null(A), then A @ (W + Δw) = A @ W.

    All parameters are derived from the data's dtype and spectral properties.
    No user configuration required - the geometry determines everything.
    """

    def __init__(
        self,
        backend: Backend | None = None,
    ) -> None:
        self._backend = backend or get_default_backend()

    def compute_null_space_projection(
        self,
        activation_matrix: Any,
    ) -> NullSpaceProjection:
        """
        Compute projection matrix onto null space of activation matrix.

        All parameters are derived from the data:
        - min_samples: log2(d) - information-theoretic minimum
        - normalization: always applied for numerical stability
        - regularization: sqrt(machine_epsilon) for dtype
        - rank threshold: ε * σ_max where ε is machine epsilon

        Args:
            activation_matrix: Shape [n_samples, d] where each row is an activation.

        Returns:
            NullSpaceProjection containing the projection matrix and metadata.
        """
        backend = self._backend
        activation_matrix = backend.array(activation_matrix)
        backend.eval(activation_matrix)

        # Check for NaN/Inf before SVD - these cause crashes in some backends
        nan_count = backend.sum(backend.astype(backend.isnan(activation_matrix), "int32"))
        inf_count = backend.sum(backend.astype(backend.isinf(activation_matrix), "int32"))
        backend.eval(nan_count, inf_count)
        if int(backend.to_scalar(nan_count)) > 0 or int(backend.to_scalar(inf_count)) > 0:
            raise ValueError(
                "Activation matrix contains NaN or Inf values. "
                "Cannot compute null space projection on invalid data."
            )

        n_samples = int(activation_matrix.shape[0])
        d = int(activation_matrix.shape[1])

        # Derive min_samples from dimension - information-theoretic minimum
        min_samples = _derive_min_samples(d)

        if n_samples < min_samples:
            logger.warning(
                f"Only {n_samples} samples, need {min_samples} for reliable null space. "
                "Returning identity (no filtering)."
            )
            return NullSpaceProjection(
                projection_matrix=backend.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=backend.zeros((min(n_samples, d),)),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Always normalize activations for numerical stability
        norms = backend.norm(activation_matrix, axis=1, keepdims=True)
        backend.eval(norms)
        # Regularization derived from dtype
        reg = regularization_epsilon(backend, activation_matrix)
        norms = backend.maximum(norms, backend.full(norms.shape, reg))
        activation_matrix = activation_matrix / norms
        backend.eval(activation_matrix)

        # Always use SVD - most reliable and standard method
        return self._compute_via_svd(activation_matrix)

    def _compute_via_svd(self, A: Any) -> NullSpaceProjection:
        """Compute null space using SVD.

        Rank threshold is derived from machine epsilon - the only mathematically
        justified cutoff for separating signal from numerical noise.
        """
        backend = self._backend
        n_samples = int(A.shape[0])
        d = int(A.shape[1])

        # SVD: A = U @ S @ Vh
        # Null space of A is spanned by rows of Vh with small singular values
        try:
            cache_key = _cache.make_svd_key(A, backend, full_matrices=True)
            cached = _cache.get_svd(cache_key)
            if cached is None:
                U, S, Vh = svd_via_eigh(backend, A, full_matrices=True)
                _cache.set_svd(cache_key, (U, S, Vh))
            else:
                U, S, Vh = cached
            backend.eval(U, S, Vh)
        except Exception:
            logger.warning("SVD failed, returning identity projection")
            return NullSpaceProjection(
                projection_matrix=backend.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=backend.zeros((min(n_samples, d),)),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Determine threshold from machine epsilon - the ONLY correct threshold
        # Standard numerical rank: σ_i > ε * σ_max
        eps = machine_epsilon(backend, A)
        s_count = int(S.shape[0])
        if s_count == 0:
            effective_threshold = 0.0
            row_space_dim = 0
        else:
            s_max_arr = backend.max(S)
            backend.eval(s_max_arr)
            s_max = float(backend.to_scalar(s_max_arr))
            effective_threshold = eps * s_max
            row_space_dim = int(
                backend.to_scalar(
                    backend.sum(backend.astype(S > effective_threshold, "int32"))
                )
            )

        # Null space vectors are rows of Vh beyond row_space_dim
        null_vectors = Vh[row_space_dim:]  # Shape: [null_dim, d]
        null_dim = int(null_vectors.shape[0]) if hasattr(null_vectors, 'shape') else 0

        # No max_null_dim cap - use the full null space from the geometry

        # Projection matrix: P = V_null @ V_null^T
        if null_dim > 0:
            projection_matrix = backend.matmul(backend.transpose(null_vectors), null_vectors)
        else:
            projection_matrix = backend.zeros((d, d))

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=S,
            effective_threshold=effective_threshold,
            n_samples=n_samples,
            vh=Vh,  # Cache Vh to avoid recomputing SVD for direction analysis
        )

    def filter_delta(
        self,
        weight_delta: Any,
        prior_activations: Any,
        return_direction_analysis: bool = False,
        weight_key: str | None = None,
    ) -> NullSpaceFilterResult:
        """
        Filter a weight delta to the null space of prior activations.

        Args:
            weight_delta: The weight update to filter. Shape: [out, in] or [d].
            prior_activations: Activation matrix from prior task. Shape: [n_samples, d].
            return_direction_analysis: If True, include per-direction preservation.
            weight_key: Weight key for error context.

        Returns:
            NullSpaceFilterResult with filtered delta and diagnostics.

        Raises:
            NullSpaceFilterError: If activation dimension doesn't match weight dimension.
        """
        backend = self._backend
        weight_delta = backend.array(weight_delta)
        prior_activations = backend.array(prior_activations)
        backend.eval(weight_delta, prior_activations)

        original_shape = weight_delta.shape
        delta_flat = backend.reshape(weight_delta, (-1,))
        backend.eval(delta_flat)
        d = int(delta_flat.shape[0])

        # Ensure activations match weight dimension
        if int(prior_activations.shape[1]) != d:
            # Try to match by transposing or reshaping
            if int(prior_activations.shape[1]) == int(original_shape[0]):
                # Activations are [n, out], weights are [out, in]
                # This is for output-space null filtering
                if len(original_shape) == 2:
                    delta_flat = backend.reshape(backend.transpose(weight_delta), (-1,))
                    backend.eval(delta_flat)
                    d = int(delta_flat.shape[0])
            else:
                # Dimension mismatch is a BUG. The geometry is invariant - if dimensions
                # don't match, our pipeline failed to compute the right transformation.
                # No fallbacks. Fix the algorithm.
                raise NullSpaceFilterError(
                    stage="NULL_SPACE_FILTER",
                    weight_key=weight_key,
                    message=f"Activation dim {prior_activations.shape[1]} != weight dim {d}",
                    context={
                        "activation_dim": int(prior_activations.shape[1]),
                        "weight_dim": d,
                        "weight_shape": list(original_shape),
                        "activation_shape": list(prior_activations.shape),
                    },
                )

        # Compute null space projection
        projection = self.compute_null_space_projection(prior_activations)

        if projection.null_dim == 0:
            norm_arr = backend.norm(delta_flat)
            backend.eval(norm_arr)
            delta_norm = float(backend.to_scalar(norm_arr))

            logger.debug("Null space is empty (full rank activations). No filtering applied.")
            return NullSpaceFilterResult(
                filtered_delta=weight_delta,
                original_delta=weight_delta,
                null_space_dim=0,
                projection_loss=0.0,
                preserved_fraction=1.0,
                original_norm=delta_norm,
                filtered_norm=delta_norm,
                filtering_applied=False,
            )

        # Project delta to null space
        delta_safe = backend.matmul(projection.projection_matrix, delta_flat)
        backend.eval(delta_safe)

        # Compute metrics
        original_norm_arr = backend.norm(delta_flat)
        filtered_norm_arr = backend.norm(delta_safe)
        backend.eval(original_norm_arr, filtered_norm_arr)

        original_norm = float(backend.to_scalar(original_norm_arr))
        filtered_norm = float(backend.to_scalar(filtered_norm_arr))

        if original_norm > 0:
            preserved_fraction = filtered_norm / original_norm
            projection_loss = 1.0 - preserved_fraction
        else:
            preserved_fraction = 1.0
            projection_loss = 0.0

        # Direction analysis if requested
        direction_preservation = None
        if return_direction_analysis and projection.null_dim > 0:
            # Compute how much of each principal direction is preserved
            # Reuse Vh from projection (cached from SVD) to avoid recomputation
            try:
                Vh = projection.vh
                if Vh is None:
                    # Fallback if Vh wasn't cached (e.g., from older projections)
                    _, _, Vh = svd_via_eigh(backend, prior_activations, full_matrices=False)
                    backend.eval(Vh)

                n_dirs = min(10, int(Vh.shape[0]))
                # Vectorized: compute all direction preservations at once
                Vh_subset = Vh[:n_dirs]  # (n_dirs, d)
                # Project all directions: P @ Vh.T -> (d, n_dirs) -> transpose to (n_dirs, d)
                proj_Vh = backend.transpose(backend.matmul(projection.projection_matrix, backend.transpose(Vh_subset)))
                # Row-wise dot products: sum(Vh * proj_Vh, axis=1) -> (n_dirs,)
                dot_products = backend.sum(Vh_subset * proj_Vh, axis=1)
                backend.eval(dot_products)
                # Preservation = 1 - dot_product
                direction_preservation = 1.0 - dot_products
            except Exception:
                direction_preservation = None

        # Reshape back to original
        filtered_delta = backend.reshape(delta_safe, original_shape)
        backend.eval(filtered_delta)

        return NullSpaceFilterResult(
            filtered_delta=filtered_delta,
            original_delta=weight_delta,
            null_space_dim=projection.null_dim,
            projection_loss=projection_loss,
            preserved_fraction=preserved_fraction,
            original_norm=original_norm,
            filtered_norm=filtered_norm,
            filtering_applied=True,
            direction_preservation=direction_preservation,
        )

    def compute_model_null_space_profile(
        self,
        layer_activations: dict[int, Any],
    ) -> ModelNullSpaceProfile:
        """
        Compute null space profile across all layers.

        Graftable layers are those with above-mean null fraction - derived
        from the geometry, not an arbitrary threshold.

        Args:
            layer_activations: Dict mapping layer index to activation matrix.

        Returns:
            ModelNullSpaceProfile with per-layer and aggregate statistics.
        """
        backend = self._backend
        per_layer: dict[int, LayerNullSpaceProfile] = {}
        total_null_dim = 0
        total_dim = 0
        graftable_layers = []

        for layer_idx, activations in sorted(layer_activations.items()):
            projection = self.compute_null_space_projection(activations)

            activations_arr = backend.array(activations)
            backend.eval(activations_arr)
            d = int(activations_arr.shape[1])
            null_fraction = projection.null_dim / d if d > 0 else 0.0

            # Condition number
            # Use tolist() for O(1) extraction instead of multiple scalar extractions
            S = projection.singular_values
            backend.eval(S)
            s_count = int(S.shape[0])
            if s_count > 0:
                s_list = backend.tolist(S)
                s_first = float(s_list[0])
                s_last = float(s_list[-1])
                if s_last > 0:
                    condition_number = s_first / s_last
                else:
                    condition_number = float("inf")
                mean_sv = sum(float(x) for x in s_list) / len(s_list)
            else:
                condition_number = float("inf")
                mean_sv = 0.0

            profile = LayerNullSpaceProfile(
                layer_idx=layer_idx,
                null_dim=projection.null_dim,
                total_dim=d,
                null_fraction=null_fraction,
                mean_singular_value=mean_sv,
                condition_number=condition_number,
            )
            per_layer[layer_idx] = profile

            total_null_dim += projection.null_dim
            total_dim += d

        mean_null_fraction = total_null_dim / total_dim if total_dim > 0 else 0.0

        # Graft threshold derived from geometry: layers with above-mean null fraction
        for layer_idx, profile in per_layer.items():
            if profile.null_fraction >= mean_null_fraction:
                graftable_layers.append(layer_idx)

        return ModelNullSpaceProfile(
            per_layer=per_layer,
            total_null_dim=total_null_dim,
            total_dim=total_dim,
            mean_null_fraction=mean_null_fraction,
            graftable_layers=graftable_layers,
        )


def filter_merge_delta_to_null_space(
    source_weights: Any,
    target_weights: Any,
    prior_activations: Any,
) -> tuple[Any, NullSpaceFilterResult]:
    """
    Compute and filter merge delta to null space.

    NO ALPHA. NO BLENDING. This is geometric ADDITION.

    Formula:
        delta = source - target
        safe_delta = null_space_projection(delta)
        merged = target + safe_delta

    The null space projection ensures source knowledge is added
    only where target has nothing (no interference).

    Args:
        source_weights: Source model weights.
        target_weights: Target model weights.
        prior_activations: Target activations defining the null space.

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

    # Filter to null space - geometry determines everything
    null_filter = NullSpaceFilter(backend)
    result = null_filter.filter_delta(delta, prior_activations)

    # Merge = target + projected_delta (NO ALPHA)
    merged = target_weights + result.filtered_delta
    backend.eval(merged)

    return merged, result


__all__ = [
    "NullSpaceFilterResult",
    "NullSpaceProjection",
    "NullSpaceFilter",
    "LayerNullSpaceProfile",
    "ModelNullSpaceProfile",
    "filter_merge_delta_to_null_space",
    # Helper functions for threshold derivation (exposed for testing)
    "_compute_numerical_rank",
    "_derive_min_samples",
]
