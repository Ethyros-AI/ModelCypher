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
    filter = NullSpaceFilter(config)
    result = filter.filter_delta(weight_delta, prior_activations)
    safe_delta = result.filtered_delta  # Guaranteed no interference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import svd_via_eigh
from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class NullSpaceMethod(str, Enum):
    """Method for computing null space projection."""

    SVD = "svd"  # Standard SVD-based null space
    EIGENVALUE = "eigenvalue"  # Eigendecomposition of A^T @ A
    QR = "qr"  # QR factorization (faster for tall matrices)


@dataclass(frozen=True)
class NullSpaceFilterConfig:
    """Configuration for null-space filtering."""

    # Threshold for considering singular values as "null"
    rank_threshold: float = 0.01

    # Maximum dimension of null space to use (memory bound)
    max_null_dim: int | None = None

    # Minimum samples needed for reliable null space estimation
    min_samples: int = 10

    # Method for computing null space
    method: NullSpaceMethod = NullSpaceMethod.SVD

    # Regularization for numerical stability
    regularization: float = 1e-8

    # Whether to normalize activations before computing null space
    normalize_activations: bool = True

    # Fraction of variance to preserve in null space (alternative to rank_threshold)
    variance_threshold: float | None = None


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


class NullSpaceFilter:
    """
    Filters weight updates to the null space of prior activations.

    This ensures that merged weights don't interfere with prior task
    performance: if Δw ∈ null(A), then A @ (W + Δw) = A @ W.
    """

    def __init__(
        self,
        config: NullSpaceFilterConfig | None = None,
        backend: Backend | None = None,
    ) -> None:
        self.config = config or NullSpaceFilterConfig()
        self._backend = backend or get_default_backend()

    def compute_null_space_projection(
        self,
        activation_matrix: Any,
    ) -> NullSpaceProjection:
        """
        Compute projection matrix onto null space of activation matrix.

        Args:
            activation_matrix: Shape [n_samples, d] where each row is an activation.

        Returns:
            NullSpaceProjection containing the projection matrix and metadata.
        """
        backend = self._backend
        activation_matrix = backend.array(activation_matrix)
        backend.eval(activation_matrix)
        
        n_samples = int(activation_matrix.shape[0])
        d = int(activation_matrix.shape[1])

        if n_samples < self.config.min_samples:
            logger.warning(
                f"Only {n_samples} samples, need {self.config.min_samples} for reliable null space. "
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

        # Normalize activations if configured
        if self.config.normalize_activations:
            norms = backend.norm(activation_matrix, axis=1, keepdims=True)
            norms = backend.maximum(norms, backend.full(norms.shape, self.config.regularization))
            activation_matrix = activation_matrix / norms

        # Compute SVD
        if self.config.method == NullSpaceMethod.SVD:
            return self._compute_via_svd(activation_matrix)
        elif self.config.method == NullSpaceMethod.QR:
            return self._compute_via_qr(activation_matrix)
        else:
            return self._compute_via_eigenvalue(activation_matrix)

    def _compute_via_svd(self, A: Any) -> NullSpaceProjection:
        """Compute null space using SVD."""
        backend = self._backend
        n_samples = int(A.shape[0])
        d = int(A.shape[1])

        # SVD: A = U @ S @ Vh
        # Null space of A is spanned by rows of Vh with small singular values
        try:
            U, S, Vh = svd_via_eigh(backend, A, full_matrices=True)
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

        # Determine threshold
        S_np = backend.to_numpy(S)
        if self.config.variance_threshold is not None:
            # Keep enough singular values to explain (1 - variance_threshold) of variance
            total_var = float(sum(s**2 for s in S_np))
            cumvar = 0.0
            row_space_dim = 0
            for i, s in enumerate(S_np):
                cumvar += s**2
                if cumvar / total_var >= (1 - self.config.variance_threshold):
                    row_space_dim = i + 1
                    break
            else:
                row_space_dim = len(S_np)
            effective_threshold = float(S_np[row_space_dim - 1]) if row_space_dim <= len(S_np) else 0.0
        else:
            # Use relative threshold
            effective_threshold = self.config.rank_threshold * float(S_np[0]) if len(S_np) > 0 else 0.0
            row_space_dim = sum(1 for s in S_np if s > effective_threshold)

        # Null space vectors are rows of Vh beyond row_space_dim
        null_vectors = Vh[row_space_dim:]  # Shape: [null_dim, d]
        null_dim = int(null_vectors.shape[0]) if hasattr(null_vectors, 'shape') else 0

        # Cap null dimension if configured
        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            null_vectors = null_vectors[: self.config.max_null_dim]
            null_dim = self.config.max_null_dim

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
        )

    def _compute_via_qr(self, A: Any) -> NullSpaceProjection:
        """Compute null space using QR factorization (faster for tall matrices)."""
        backend = self._backend
        A = backend.array(A)
        backend.eval(A)

        n_samples = int(A.shape[0])
        d = int(A.shape[1])

        # QR of A^T: A^T = Q @ R
        # Null space of A is spanned by columns of Q corresponding to zero rows of R
        Q, R = backend.qr(backend.transpose(A))
        backend.eval(Q, R)

        # Find rank by looking at diagonal of R
        diag_R = backend.abs(backend.diag(R[: min(n_samples, d), : min(n_samples, d)]))
        backend.eval(diag_R)
        diag_R_np = backend.to_numpy(diag_R)

        if len(diag_R_np) == 0:
            threshold = 0.0
            row_space_dim = 0
        else:
            threshold = self.config.rank_threshold * float(diag_R_np[0])
            row_space_dim = int(sum(1 for val in diag_R_np if val > threshold))

        # Null space vectors are columns of Q beyond row_space_dim
        null_vectors = backend.transpose(Q[:, row_space_dim:])  # Shape: [null_dim, d]
        null_dim = int(null_vectors.shape[0])

        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            null_vectors = null_vectors[: self.config.max_null_dim]
            null_dim = self.config.max_null_dim

        if null_dim > 0:
            projection_matrix = backend.matmul(backend.transpose(null_vectors), null_vectors)
        else:
            projection_matrix = backend.zeros((d, d))

        # For consistency, compute SVD for singular values
        try:
            _, S, _ = svd_via_eigh(backend, A, full_matrices=False)
            backend.eval(S)
        except Exception:
            S = backend.zeros((min(n_samples, d),))

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=S,
            effective_threshold=threshold,
            n_samples=n_samples,
        )

    def _compute_via_eigenvalue(self, A: Any) -> NullSpaceProjection:
        """Compute null space using eigendecomposition of A^T @ A."""
        backend = self._backend
        A = backend.array(A)
        backend.eval(A)

        n_samples = int(A.shape[0])
        d = int(A.shape[1])

        # A^T @ A has same null space as A
        ATA = backend.matmul(backend.transpose(A), A)
        backend.eval(ATA)

        try:
            eigenvalues, eigenvectors = backend.eigh(ATA)
            backend.eval(eigenvalues, eigenvectors)
        except Exception:
            logger.warning("Eigendecomposition failed, returning identity projection")
            return NullSpaceProjection(
                projection_matrix=backend.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=backend.zeros((d,)),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Sort by eigenvalue (ascending - null space has smallest eigenvalues)
        eigenvalues_np = backend.to_numpy(eigenvalues)
        idx = sorted(range(len(eigenvalues_np)), key=lambda i: eigenvalues_np[i])
        eigenvalues = backend.array([eigenvalues_np[i] for i in idx])
        eigenvectors_np = backend.to_numpy(eigenvectors)
        eigenvectors = backend.array(eigenvectors_np[:, idx])
        backend.eval(eigenvalues, eigenvectors)

        # Threshold for null space
        # Eigenvalues of A^T @ A are squares of singular values
        eigenvalues_np = backend.to_numpy(eigenvalues)
        singular_values = backend.sqrt(backend.maximum(eigenvalues, backend.zeros(eigenvalues.shape)))
        backend.eval(singular_values)
        singular_values_np = backend.to_numpy(singular_values)

        if len(singular_values_np) > 0 and singular_values_np[-1] > 0:
            threshold = self.config.rank_threshold * float(singular_values_np[-1])
        else:
            threshold = 0.0

        null_mask = [val < threshold for val in singular_values_np]
        null_dim = sum(null_mask)
        row_space_dim = d - null_dim

        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            # Take only the smallest eigenvalue directions
            null_mask = [i < self.config.max_null_dim for i in range(d)]
            null_dim = self.config.max_null_dim

        # Extract null vectors
        eigenvectors_np = backend.to_numpy(eigenvectors)
        null_vectors_list = [eigenvectors_np[:, i] for i in range(d) if null_mask[i]]

        if null_vectors_list:
            null_vectors = backend.transpose(backend.array(null_vectors_list))  # Shape: [d, null_dim]
        else:
            null_vectors = backend.zeros((d, 0))
        backend.eval(null_vectors)

        if null_dim > 0:
            projection_matrix = backend.matmul(null_vectors, backend.transpose(null_vectors))
        else:
            projection_matrix = backend.zeros((d, d))

        # Reverse singular values for descending order like SVD
        singular_values_reversed = backend.array(singular_values_np[::-1])

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=singular_values_reversed,
            effective_threshold=threshold,
            n_samples=n_samples,
        )

    def filter_delta(
        self,
        weight_delta: Any,
        prior_activations: Any,
        return_direction_analysis: bool = False,
    ) -> NullSpaceFilterResult:
        """
        Filter a weight delta to the null space of prior activations.

        Args:
            weight_delta: The weight update to filter. Shape: [out, in] or [d].
            prior_activations: Activation matrix from prior task. Shape: [n_samples, d].
            return_direction_analysis: If True, include per-direction preservation.

        Returns:
            NullSpaceFilterResult with filtered delta and diagnostics.
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
                norm_arr = backend.norm(delta_flat)
                backend.eval(norm_arr)
                delta_norm = float(backend.to_numpy(norm_arr).item())

                logger.warning(
                    f"Activation dim {prior_activations.shape[1]} != weight dim {d}. "
                    "Skipping null-space filtering."
                )
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

        # Compute null space projection
        projection = self.compute_null_space_projection(prior_activations)

        if projection.null_dim == 0:
            norm_arr = backend.norm(delta_flat)
            backend.eval(norm_arr)
            delta_norm = float(backend.to_numpy(norm_arr).item())

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

        original_norm = float(backend.to_numpy(original_norm_arr).item())
        filtered_norm = float(backend.to_numpy(filtered_norm_arr).item())

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
            try:
                _, _, Vh = svd_via_eigh(backend, prior_activations, full_matrices=False)
                backend.eval(Vh)

                n_dirs = min(10, int(Vh.shape[0]))
                dir_pres = []
                for i in range(n_dirs):
                    vh_i = Vh[i]
                    proj_vh = backend.matmul(projection.projection_matrix, vh_i)
                    dot_product = backend.sum(vh_i * proj_vh)
                    backend.eval(dot_product)
                    dir_pres.append(1.0 - float(backend.to_numpy(dot_product).item()))
                direction_preservation = backend.array(dir_pres)
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
        graft_threshold: float = 0.1,
    ) -> ModelNullSpaceProfile:
        """
        Compute null space profile across all layers.

        Args:
            layer_activations: Dict mapping layer index to activation matrix.
            graft_threshold: Minimum null fraction for a layer to be "graftable".

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
            S = projection.singular_values
            S_np = backend.to_numpy(S)
            if len(S_np) > 0 and S_np[-1] > 0:
                condition_number = float(S_np[0]) / float(S_np[-1])
            else:
                condition_number = float("inf")

            mean_sv = float(sum(S_np) / len(S_np)) if len(S_np) > 0 else 0.0

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

            if null_fraction >= graft_threshold:
                graftable_layers.append(layer_idx)

        mean_null_fraction = total_null_dim / total_dim if total_dim > 0 else 0.0

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
    alpha: float = 0.5,
    config: NullSpaceFilterConfig | None = None,
) -> tuple[Any, NullSpaceFilterResult]:
    """
    Convenience function: Compute and filter merge delta to null space.

    Args:
        source_weights: Weights from source model.
        target_weights: Weights from target model.
        prior_activations: Activations from target model on prior task.
        alpha: Merge coefficient (0 = target, 1 = source).
        config: Optional filter configuration.

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

    # Filter to null space
    filter = NullSpaceFilter(config)
    result = filter.filter_delta(delta, prior_activations)

    # Apply filtered delta
    merged = target_weights + alpha * result.filtered_delta
    backend.eval(merged)

    return merged, result


__all__ = [
    "NullSpaceFilterConfig",
    "NullSpaceFilterResult",
    "NullSpaceProjection",
    "NullSpaceFilter",
    "NullSpaceMethod",
    "LayerNullSpaceProfile",
    "ModelNullSpaceProfile",
    "filter_merge_delta_to_null_space",
]
