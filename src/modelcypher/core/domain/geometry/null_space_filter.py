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
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NullSpaceMethod(str, Enum):
    """Method for computing null space projection."""
    SVD = "svd"              # Standard SVD-based null space
    EIGENVALUE = "eigenvalue"  # Eigendecomposition of A^T @ A
    QR = "qr"                # QR factorization (faster for tall matrices)


@dataclass(frozen=True)
class NullSpaceFilterConfig:
    """Configuration for null-space filtering."""

    # Threshold for considering singular values as "null"
    rank_threshold: float = 0.01

    # Maximum dimension of null space to use (memory bound)
    max_null_dim: Optional[int] = None

    # Minimum samples needed for reliable null space estimation
    min_samples: int = 10

    # Method for computing null space
    method: NullSpaceMethod = NullSpaceMethod.SVD

    # Regularization for numerical stability
    regularization: float = 1e-8

    # Whether to normalize activations before computing null space
    normalize_activations: bool = True

    # Fraction of variance to preserve in null space (alternative to rank_threshold)
    variance_threshold: Optional[float] = None


@dataclass
class NullSpaceProjection:
    """Precomputed null space projection matrix and metadata."""

    # Projection matrix onto null space: P @ x projects x to null(A)
    projection_matrix: np.ndarray

    # Dimension of the null space
    null_dim: int

    # Dimension of the row space (complement of null)
    row_space_dim: int

    # Singular values of the activation matrix (for diagnostics)
    singular_values: np.ndarray

    # Threshold used to determine null space
    effective_threshold: float

    # Number of samples used to estimate null space
    n_samples: int


@dataclass
class NullSpaceFilterResult:
    """Result of filtering a weight delta through null space."""

    # The filtered delta (projected to null space)
    filtered_delta: np.ndarray

    # Original delta (for comparison)
    original_delta: np.ndarray

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
    direction_preservation: Optional[np.ndarray] = None


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

    def __init__(self, config: Optional[NullSpaceFilterConfig] = None) -> None:
        self.config = config or NullSpaceFilterConfig()

    def compute_null_space_projection(
        self,
        activation_matrix: np.ndarray,
    ) -> NullSpaceProjection:
        """
        Compute projection matrix onto null space of activation matrix.

        Args:
            activation_matrix: Shape [n_samples, d] where each row is an activation.

        Returns:
            NullSpaceProjection containing the projection matrix and metadata.
        """
        n_samples, d = activation_matrix.shape

        if n_samples < self.config.min_samples:
            logger.warning(
                f"Only {n_samples} samples, need {self.config.min_samples} for reliable null space. "
                "Returning identity (no filtering)."
            )
            return NullSpaceProjection(
                projection_matrix=np.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=np.zeros(min(n_samples, d)),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Normalize activations if configured
        if self.config.normalize_activations:
            norms = np.linalg.norm(activation_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, self.config.regularization)
            activation_matrix = activation_matrix / norms

        # Compute SVD
        if self.config.method == NullSpaceMethod.SVD:
            return self._compute_via_svd(activation_matrix)
        elif self.config.method == NullSpaceMethod.QR:
            return self._compute_via_qr(activation_matrix)
        else:
            return self._compute_via_eigenvalue(activation_matrix)

    def _compute_via_svd(self, A: np.ndarray) -> NullSpaceProjection:
        """Compute null space using SVD."""
        n_samples, d = A.shape

        # SVD: A = U @ S @ Vh
        # Null space of A is spanned by rows of Vh with small singular values
        try:
            U, S, Vh = np.linalg.svd(A, full_matrices=True)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed, returning identity projection")
            return NullSpaceProjection(
                projection_matrix=np.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=np.zeros(min(n_samples, d)),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Determine threshold
        if self.config.variance_threshold is not None:
            # Keep enough singular values to explain (1 - variance_threshold) of variance
            total_var = np.sum(S ** 2)
            cumvar = np.cumsum(S ** 2) / total_var
            row_space_dim = np.searchsorted(cumvar, 1 - self.config.variance_threshold) + 1
            effective_threshold = S[row_space_dim - 1] if row_space_dim <= len(S) else 0.0
        else:
            # Use relative threshold
            effective_threshold = self.config.rank_threshold * S[0] if len(S) > 0 else 0.0
            row_space_dim = np.sum(S > effective_threshold)

        # Null space vectors are rows of Vh beyond row_space_dim
        null_vectors = Vh[row_space_dim:]  # Shape: [null_dim, d]
        null_dim = null_vectors.shape[0]

        # Cap null dimension if configured
        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            null_vectors = null_vectors[:self.config.max_null_dim]
            null_dim = self.config.max_null_dim

        # Projection matrix: P = V_null @ V_null^T
        if null_dim > 0:
            projection_matrix = null_vectors.T @ null_vectors
        else:
            projection_matrix = np.zeros((d, d))

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=S,
            effective_threshold=effective_threshold,
            n_samples=n_samples,
        )

    def _compute_via_qr(self, A: np.ndarray) -> NullSpaceProjection:
        """Compute null space using QR factorization (faster for tall matrices)."""
        n_samples, d = A.shape

        # QR of A^T: A^T = Q @ R
        # Null space of A is spanned by columns of Q corresponding to zero rows of R
        Q, R = np.linalg.qr(A.T, mode='complete')

        # Find rank by looking at diagonal of R
        diag_R = np.abs(np.diag(R[:min(n_samples, d), :min(n_samples, d)]))
        if len(diag_R) == 0:
            threshold = 0.0
            row_space_dim = 0
        else:
            threshold = self.config.rank_threshold * diag_R[0]
            row_space_dim = np.sum(diag_R > threshold)

        # Null space vectors are columns of Q beyond row_space_dim
        null_vectors = Q[:, row_space_dim:].T  # Shape: [null_dim, d]
        null_dim = null_vectors.shape[0]

        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            null_vectors = null_vectors[:self.config.max_null_dim]
            null_dim = self.config.max_null_dim

        if null_dim > 0:
            projection_matrix = null_vectors.T @ null_vectors
        else:
            projection_matrix = np.zeros((d, d))

        # For consistency, compute SVD for singular values
        try:
            S = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            S = np.zeros(min(n_samples, d))

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=S,
            effective_threshold=threshold,
            n_samples=n_samples,
        )

    def _compute_via_eigenvalue(self, A: np.ndarray) -> NullSpaceProjection:
        """Compute null space using eigendecomposition of A^T @ A."""
        n_samples, d = A.shape

        # A^T @ A has same null space as A
        ATA = A.T @ A

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(ATA)
        except np.linalg.LinAlgError:
            logger.warning("Eigendecomposition failed, returning identity projection")
            return NullSpaceProjection(
                projection_matrix=np.eye(d),
                null_dim=d,
                row_space_dim=0,
                singular_values=np.zeros(d),
                effective_threshold=0.0,
                n_samples=n_samples,
            )

        # Sort by eigenvalue (ascending - null space has smallest eigenvalues)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Threshold for null space
        # Eigenvalues of A^T @ A are squares of singular values
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))
        if len(singular_values) > 0 and singular_values[-1] > 0:
            threshold = self.config.rank_threshold * singular_values[-1]
        else:
            threshold = 0.0

        null_mask = singular_values < threshold
        null_dim = np.sum(null_mask)
        row_space_dim = d - null_dim

        if self.config.max_null_dim is not None and null_dim > self.config.max_null_dim:
            # Take only the smallest eigenvalue directions
            null_mask = np.zeros(d, dtype=bool)
            null_mask[:self.config.max_null_dim] = True
            null_dim = self.config.max_null_dim

        null_vectors = eigenvectors[:, null_mask].T  # Shape: [null_dim, d]

        if null_dim > 0:
            projection_matrix = null_vectors.T @ null_vectors
        else:
            projection_matrix = np.zeros((d, d))

        return NullSpaceProjection(
            projection_matrix=projection_matrix,
            null_dim=null_dim,
            row_space_dim=row_space_dim,
            singular_values=singular_values[::-1],  # Descending order like SVD
            effective_threshold=threshold,
            n_samples=n_samples,
        )

    def filter_delta(
        self,
        weight_delta: np.ndarray,
        prior_activations: np.ndarray,
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
        original_shape = weight_delta.shape
        delta_flat = weight_delta.flatten()
        d = delta_flat.shape[0]

        # Ensure activations match weight dimension
        if prior_activations.shape[1] != d:
            # Try to match by transposing or reshaping
            if prior_activations.shape[1] == original_shape[0]:
                # Activations are [n, out], weights are [out, in]
                # This is for output-space null filtering
                prior_activations = prior_activations
                delta_flat = weight_delta.T.flatten() if len(original_shape) == 2 else delta_flat
                d = delta_flat.shape[0]
            else:
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
                    original_norm=float(np.linalg.norm(delta_flat)),
                    filtered_norm=float(np.linalg.norm(delta_flat)),
                    filtering_applied=False,
                )

        # Compute null space projection
        projection = self.compute_null_space_projection(prior_activations)

        if projection.null_dim == 0:
            logger.debug("Null space is empty (full rank activations). No filtering applied.")
            return NullSpaceFilterResult(
                filtered_delta=weight_delta,
                original_delta=weight_delta,
                null_space_dim=0,
                projection_loss=0.0,
                preserved_fraction=1.0,
                original_norm=float(np.linalg.norm(delta_flat)),
                filtered_norm=float(np.linalg.norm(delta_flat)),
                filtering_applied=False,
            )

        # Project delta to null space
        delta_safe = projection.projection_matrix @ delta_flat

        # Compute metrics
        original_norm = np.linalg.norm(delta_flat)
        filtered_norm = np.linalg.norm(delta_safe)

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
                _, _, Vh = np.linalg.svd(prior_activations, full_matrices=False)
                direction_preservation = np.array([
                    1.0 - np.dot(Vh[i], projection.projection_matrix @ Vh[i])
                    for i in range(min(10, Vh.shape[0]))
                ])
            except np.linalg.LinAlgError:
                direction_preservation = None

        # Reshape back to original
        filtered_delta = delta_safe.reshape(original_shape)

        return NullSpaceFilterResult(
            filtered_delta=filtered_delta,
            original_delta=weight_delta,
            null_space_dim=projection.null_dim,
            projection_loss=projection_loss,
            preserved_fraction=preserved_fraction,
            original_norm=float(original_norm),
            filtered_norm=float(filtered_norm),
            filtering_applied=True,
            direction_preservation=direction_preservation,
        )

    def compute_model_null_space_profile(
        self,
        layer_activations: dict[int, np.ndarray],
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
        per_layer: dict[int, LayerNullSpaceProfile] = {}
        total_null_dim = 0
        total_dim = 0
        graftable_layers = []

        for layer_idx, activations in sorted(layer_activations.items()):
            projection = self.compute_null_space_projection(activations)

            d = activations.shape[1]
            null_fraction = projection.null_dim / d if d > 0 else 0.0

            # Condition number
            S = projection.singular_values
            if len(S) > 0 and S[-1] > 0:
                condition_number = S[0] / S[-1]
            else:
                condition_number = float('inf')

            profile = LayerNullSpaceProfile(
                layer_idx=layer_idx,
                null_dim=projection.null_dim,
                total_dim=d,
                null_fraction=null_fraction,
                mean_singular_value=float(np.mean(S)) if len(S) > 0 else 0.0,
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
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    prior_activations: np.ndarray,
    alpha: float = 0.5,
    config: Optional[NullSpaceFilterConfig] = None,
) -> tuple[np.ndarray, NullSpaceFilterResult]:
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
    # Compute delta
    delta = source_weights - target_weights

    # Filter to null space
    filter = NullSpaceFilter(config)
    result = filter.filter_delta(delta, prior_activations)

    # Apply filtered delta
    merged = target_weights + alpha * result.filtered_delta

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
