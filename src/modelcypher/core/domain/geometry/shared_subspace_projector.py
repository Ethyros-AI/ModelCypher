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
Shared Subspace Projector - CCA/SVCCA alignment for cross-model representation matching.

Discovers the shared geometric subspace between two models' representations,
enabling dimension-agnostic alignment for model merging. When models encode
the same knowledge in different coordinate systems, this module finds the
common subspace where their representations align.

Mathematical Foundation:
========================

Canonical Correlation Analysis (CCA) finds pairs of linear projections that
maximize correlation between two sets of variables. Given centered activations
X_s [n, d_s] and X_t [n, d_t]:

    1. Covariance whitening: Transform each space to have identity covariance
       C_ss^{-1/2} @ X_s and C_tt^{-1/2} @ X_t

    2. Cross-covariance SVD: Decompose the whitened cross-covariance
       SVD(C_ss^{-1/2} @ C_st @ C_tt^{-1/2}) = U @ S @ V^T

    3. Canonical correlations: Singular values S are the canonical correlations
       representing alignment strength in each shared dimension

    4. Projection matrices: Final projections combine PCA reduction, whitening,
       and SVD components to map both representations to the shared subspace

SVCCA (Singular Vector CCA) adds a PCA preprocessing step to reduce each
representation to its high-variance subspace before CCA. This improves
numerical stability when d >> n and filters noise dimensions.

Key Concepts:
=============

PcaMode enum:
    - auto: Choose based on n vs d (Gram-space when d > n)
    - svd: Direct SVD on activations (stable for n >= d)
    - gram: Eigendecomposition of Gram matrix (efficient for d >> n)

Spectral gap detection:
    When variance_threshold is None, component count is determined by finding
    the largest relative drop in the sorted variance spectrum. This is
    geometry-derived, not an arbitrary cutoff.

Properties:
===========

- Dimension-agnostic: Works across ANY dimensions via Gram matrices
- Linear alignment is closed-form; geodesic CKA reports overlap for aligned layers
- No interpolation: Projects to shared subspace, doesn't blend representations
- Numerically stable: Uses regularization derived from dtype, not heuristics

Usage:
    from modelcypher.core.domain.geometry.shared_subspace_projector import (
        SharedSubspaceProjector,
    )

    result = SharedSubspaceProjector.discover(
        source_crm, target_crm, layer=12
    )
    if result is not None:
        # result.source_projection @ source_activations -> shared space
        # result.target_projection @ target_activations -> shared space
        print(f"Shared dimension: {result.shared_dimension}")
        print(f"Top correlation: {result.alignment_strengths[0]:.4f}")

References:
    - Raghu, M., et al. (2017). "SVCCA: Singular Vector Canonical Correlation
      Analysis for Deep Learning Dynamics and Interpretability."
      arXiv:1706.05806. https://arxiv.org/abs/1706.05806
    - Morcos, A., et al. (2018). "Insights on representational similarity in
      neural networks with canonical correlation."
      NeurIPS 2018. https://arxiv.org/abs/1806.05759
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.geometry_fingerprint import GeometricFingerprint
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    power_iteration_eigh,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_cosine_between_sets,
    geodesic_cosine_matrix,
    geodesic_norms,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# unified_atlas imported lazily in validate_crm_uses_atlas to avoid circular imports

_cache = ComputationCache.shared()


class AlignmentMethod(str, Enum):
    cca = "cca"


class PcaMode(str, Enum):
    auto = "auto"
    svd = "svd"
    gram = "gram"


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - variance_threshold: Uses spectral gap detection
# - pca_variance_threshold: Uses spectral gap detection
# - max_shared_dimension: min(source_dim, target_dim)
# - cca_regularization: sqrt(machine_epsilon)
# - min_samples: sqrt(n_features)
# - min_canonical_correlation: sqrt(machine_epsilon)
# - alignment_method: Always CCA (mathematically correct method)
# - pca_mode: Always auto (chooses based on data shape)
# - All anchors: Used with uniform weights
# =============================================================================


def validate_crm_uses_atlas(
    crm: ConceptResponseMatrix,
    atlas_probe_ids: set[str] | None = None,
) -> dict:
    """Measure ConceptResponseMatrix coverage of unified atlas probes.

    The unified atlas provides N probes across all sources for cross-domain
    triangulation. CRM data built from atlas probes enables more robust
    dimension-agnostic alignment.

    Args:
        crm: The ConceptResponseMatrix to validate

    Returns:
        Dict with raw coverage measurements (no validity judgment):
        - atlas_probe_count: Number of atlas probes
        - crm_concept_count: Number of CRM concepts
        - overlap_count: Number of matching IDs
        - coverage: Fraction of atlas IDs present in CRM
        - has_all_atlas: Whether CRM contains all atlas probes
        - uses_atlas_subset: Whether CRM uses only atlas probes
    """
    if atlas_probe_ids is None:
        atlas_probe_ids = {probe.probe_id for probe in get_atlas_probes()}
    atlas_ids = set(atlas_probe_ids)
    crm_ids = set(crm.concept_ids) if hasattr(crm, "concept_ids") else set()

    overlap = len(atlas_ids & crm_ids)
    coverage = overlap / len(atlas_ids) if atlas_ids else 0.0

    return {
        "atlas_probe_count": len(atlas_ids),
        "crm_concept_count": len(crm_ids),
        "overlap_count": overlap,
        "coverage": coverage,
        "has_all_atlas": atlas_ids.issubset(crm_ids),
        "uses_atlas_subset": crm_ids.issubset(atlas_ids),
    }


def _array_to_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)


def _array_to_2d_list(backend: "Backend", array: "Array") -> list[list[float]]:
    """Convert 2D array to nested Python list using native tolist() - O(1) vs O(n*m)."""
    return backend.tolist(array)


@dataclass(frozen=True)
class H3ValidationMetrics:
    shared_dimension: int
    top_canonical_correlation: float
    alignment_error: float
    shared_variance_ratio: float


@dataclass(frozen=True)
class Result:
    shared_dimension: int
    source_dimension: int
    target_dimension: int
    source_projection: list[list[float]]
    target_projection: list[list[float]]
    alignment_strengths: list[float]
    alignment_error: float
    shared_variance_ratio: float
    sample_count: int
    method: AlignmentMethod

    @property
    def has_shared_structure(self) -> bool:
        """Check if any shared structure was found (shared_dimension > 0)."""
        return self.shared_dimension > 0

    @property
    def h3_metrics(self) -> H3ValidationMetrics:
        return H3ValidationMetrics(
            shared_dimension=self.shared_dimension,
            top_canonical_correlation=self.alignment_strengths[0]
            if self.alignment_strengths
            else 0.0,
            alignment_error=self.alignment_error,
            shared_variance_ratio=self.shared_variance_ratio,
        )


class SharedSubspaceProjector:
    @staticmethod
    def discover(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        layer: int,
        target_layer: int | None = None,
    ) -> Result | None:
        """Discover shared subspace between source and target CRMs.

        All parameters are derived from data - no configuration needed.
        Always uses CCA (mathematically correct method) with all anchors
        equally weighted.
        """
        source_layer = int(layer)
        target_layer = source_layer if target_layer is None else int(target_layer)
        matrices = SharedSubspaceProjector._extract_activation_matrices(
            source_crm,
            target_crm,
            source_layer,
            target_layer,
        )
        if matrices is None:
            return None
        source_matrix, target_matrix = matrices

        n = len(source_matrix)
        d_source = len(source_matrix[0])
        d_target = len(target_matrix[0])

        # Derive min_samples from sqrt(n_features)
        backend = get_default_backend()
        min_samples = max(3, int(sqrt_scalar(float(max(d_source, d_target)), backend)))
        if len(source_matrix) != len(target_matrix) or n < min_samples:
            return None

        # Always use CCA - the mathematically correct method for finding shared subspace
        return SharedSubspaceProjector._discover_with_cca(
            source_matrix, target_matrix, n, d_source, d_target
        )

    @staticmethod
    def _discover_with_cca(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        n: int,
        d_source: int,
        d_target: int,
        backend: "Backend | None" = None,
    ) -> Result | None:
        """Discover shared subspace using CCA.

        All parameters derived from data - no configuration needed.
        """
        b = backend or get_default_backend()

        # Convert to backend arrays
        source_array = b.array(source_activations)
        target_array = b.array(target_activations)
        b.eval(source_array, target_array)

        if source_array.shape[0] != target_array.shape[0]:
            return None

        # No weights - all anchors contribute equally
        source_centered, _ = SharedSubspaceProjector._center_array(source_array, None, backend=b)
        target_centered, _ = SharedSubspaceProjector._center_array(target_array, None, backend=b)

        # Derive max_shared_dimension from min(source_dim, target_dim)
        max_shared_dim = min(int(source_centered.shape[1]), int(target_centered.shape[1]))

        # SVCCA: reduce to high-variance subspaces before CCA to avoid ill-conditioned covariance.
        max_components_source = min(
            max_shared_dim,
            int(source_centered.shape[0]),
            int(source_centered.shape[1]),
        )
        max_components_target = min(
            max_shared_dim,
            int(target_centered.shape[0]),
            int(target_centered.shape[1]),
        )
        # All thresholds derived from data (spectral gap detection when None)
        # Always use auto PCA mode (chooses based on data shape)
        source_reduced, source_components, source_variances = SharedSubspaceProjector._pca_reduce(
            source_centered, None, max_components_source, PcaMode.auto, backend=b
        )
        target_reduced, target_components, target_variances = SharedSubspaceProjector._pca_reduce(
            target_centered, None, max_components_target, PcaMode.auto, backend=b
        )
        if source_reduced is None or target_reduced is None:
            return None
        if source_reduced.shape[0] != target_reduced.shape[0]:
            return None

        sample_count = int(source_reduced.shape[0])

        # Covariance matrices: C = X^T @ X / n
        source_reduced_t = b.transpose(source_reduced)
        target_reduced_t = b.transpose(target_reduced)
        cxx = b.matmul(source_reduced_t, source_reduced) / float(sample_count)
        cyy = b.matmul(target_reduced_t, target_reduced) / float(sample_count)
        cxy = b.matmul(source_reduced_t, target_reduced) / float(sample_count)
        b.eval(cxx, cyy, cxy)

        # Derive cca_regularization from sqrt(machine_epsilon)
        from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
        cca_reg = regularization_epsilon(b, cxx)
        cxx = SharedSubspaceProjector._regularize_covariance(cxx, cca_reg, backend=b)
        cyy = SharedSubspaceProjector._regularize_covariance(cyy, cca_reg, backend=b)
        inv_sqrt_x, x_eigenvalues = SharedSubspaceProjector._whiten_covariance(cxx, backend=b)
        inv_sqrt_y, y_eigenvalues = SharedSubspaceProjector._whiten_covariance(cyy, backend=b)
        if inv_sqrt_x is None or inv_sqrt_y is None:
            return None

        # Cross-covariance in whitened space: inv_sqrt_x @ cxy @ inv_sqrt_y
        cross_cov = b.matmul(b.matmul(inv_sqrt_x, cxy), inv_sqrt_y)
        b.eval(cross_cov)

        # SVD of cross-covariance
        u, singular_values, v_t = _cache.get_or_compute_svd(cross_cov, b, full_matrices=False)
        b.eval(u, singular_values, v_t)

        # Clip singular values to [0, 1] for canonical correlations
        canonical_arr = b.clip(singular_values, 0.0, 1.0)
        b.eval(canonical_arr)
        canonical_sq = canonical_arr * canonical_arr
        b.eval(canonical_sq)

        # Filter by min_canonical_correlation derived from data
        # Use sqrt(machine_epsilon) as minimum meaningful correlation
        eps = float(machine_epsilon(b, canonical_arr))
        min_corr_val = eps ** 0.5
        min_corr = b.full(canonical_arr.shape, min_corr_val)
        valid_mask = canonical_arr >= min_corr
        valid_count_arr = b.sum(b.astype(valid_mask, "int32"))
        b.eval(valid_count_arr)
        valid_count = int(b.to_scalar(valid_count_arr))
        if valid_count == 0:
            return None

        valid_variances = b.where(valid_mask, canonical_sq, b.zeros_like(canonical_sq))
        b.eval(valid_variances)

        # Determine shared dimension using spectral gap detection (data-derived)
        valid_count = SharedSubspaceProjector._select_component_count(
            valid_variances,
            None,  # Always use spectral gap detection
            backend=b,
        )

        # shared_dim is the actual dimension we'll use (indices into u, v_t, etc.)
        shared_dim = min(valid_count, max_shared_dim)
        if shared_dim <= 0:
            return None

        # Compute variance ratio for the selected dimensions
        selected_variance_arr = b.sum(valid_variances[:shared_dim])
        total_variance_arr = b.sum(valid_variances)
        b.eval(selected_variance_arr, total_variance_arr)
        selected_variance = float(b.to_scalar(selected_variance_arr))
        total_variance = float(b.to_scalar(total_variance_arr))
        shared_variance_ratio = selected_variance / total_variance if total_variance > 0 else 0.0

        # Truncate to shared_dim
        u_truncated = u[:, :shared_dim]
        v_t_truncated = v_t[:shared_dim, :]
        v_truncated = b.transpose(v_t_truncated)
        b.eval(u_truncated, v_truncated)

        # Projection matrices
        source_projection = b.matmul(source_components, b.matmul(inv_sqrt_x, u_truncated))
        target_projection = b.matmul(target_components, b.matmul(inv_sqrt_y, v_truncated))
        b.eval(source_projection, target_projection)

        # Project data to shared space
        source_projected = b.matmul(source_reduced, b.matmul(inv_sqrt_x, u_truncated))
        target_projected = b.matmul(target_reduced, b.matmul(inv_sqrt_y, v_truncated))
        b.eval(source_projected, target_projected)

        # Compute alignment error using geodesic norms
        diff = source_projected - target_projected
        diff_flat = b.reshape(diff, (1, -1))
        target_flat = b.reshape(target_projected, (1, -1))
        diff_norm_arr = geodesic_norms(diff_flat, b)
        target_norm_arr = geodesic_norms(target_flat, b)
        b.eval(diff_norm_arr, target_norm_arr)
        diff_norm = float(b.to_scalar(diff_norm_arr[0]))
        target_norm = float(b.to_scalar(target_norm_arr[0]))
        alignment_error = (diff_norm / target_norm) if target_norm > 0 else 0.0

        # Convert projections to lists
        source_proj_list = _array_to_2d_list(b, source_projection)
        target_proj_list = _array_to_2d_list(b, target_projection)

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d_source,
            target_dimension=d_target,
            source_projection=source_proj_list,
            target_projection=target_proj_list,
            alignment_strengths=_array_to_list(b, canonical_arr)[:shared_dim],
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            sample_count=sample_count,
            method=AlignmentMethod.cca,
        )

    @staticmethod
    def _extract_activation_matrix(
        crm: ConceptResponseMatrix,
        layer: int,
    ) -> list[list[float]] | None:
        layer_acts = crm.activations.get(layer)
        if layer_acts is None:
            return None
        sorted_anchors = sorted(layer_acts.keys())
        if not sorted_anchors:
            return None
        matrix: list[list[float]] = []
        for anchor_id in sorted_anchors:
            activation = layer_acts.get(anchor_id)
            if activation is not None:
                matrix.append(activation.activation)
        return matrix if matrix else None

    @staticmethod
    def _extract_activation_matrices(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_layer: int,
        target_layer: int,
    ) -> tuple[list[list[float]], list[list[float]]] | None:
        """Extract activation matrices from CRMs.

        Uses all common anchors with uniform weights - no configuration.
        """
        source_layer_acts = source_crm.activations.get(source_layer)
        target_layer_acts = target_crm.activations.get(target_layer)
        if source_layer_acts is None or target_layer_acts is None:
            return None

        # Use all common anchors (no filtering by prefix)
        source_ids = set(source_crm.anchor_metadata.anchor_ids)
        target_ids = set(target_crm.anchor_metadata.anchor_ids)
        anchor_ids = sorted(source_ids.intersection(target_ids))
        if not anchor_ids:
            return None

        source_matrix: list[list[float]] = []
        target_matrix: list[list[float]] = []

        for anchor_id in anchor_ids:
            source_activation = source_layer_acts.get(anchor_id)
            target_activation = target_layer_acts.get(anchor_id)
            if source_activation is None or target_activation is None:
                continue
            source_matrix.append(source_activation.activation)
            target_matrix.append(target_activation.activation)

        if not source_matrix or not target_matrix:
            return None

        return source_matrix, target_matrix

    @staticmethod
    def _center_array(
        array: "Array",
        weights: "Array | None",
        backend: "Backend | None" = None,
    ) -> tuple["Array", "Array"]:
        b = backend or get_default_backend()

        if array.size == 0:
            return array, b.zeros((int(array.shape[1]),))

        if weights is None or weights.shape[0] != array.shape[0]:
            mean = b.mean(array, axis=0)
            b.eval(mean)
            centered = array - mean
            b.eval(centered)
            return centered, mean

        # Weighted mean: sum(array * weights[:, None], axis=0)
        weights_col = b.reshape(weights, (-1, 1))
        weighted_array = array * weights_col
        mean = b.sum(weighted_array, axis=0)
        b.eval(mean)

        centered = array - mean
        sqrt_weights = b.sqrt(weights_col)
        weighted = centered * sqrt_weights
        b.eval(weighted)
        return weighted, mean

    @staticmethod
    def _pca_reduce(
        matrix: "Array",
        variance_threshold: float | None,
        max_components: int,
        mode: PcaMode,
        backend: "Backend | None" = None,
    ) -> tuple["Array | None", "Array | None", "Array | None"]:
        b = backend or get_default_backend()

        if matrix.size == 0:
            return None, None, None
        n, d = int(matrix.shape[0]), int(matrix.shape[1])
        if max_components <= 0:
            return None, None, None
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            if normalized == "gram":
                mode = PcaMode.gram
            elif normalized == "svd":
                mode = PcaMode.svd
            else:
                mode = PcaMode.auto
        if mode == PcaMode.auto:
            # Gram-space PCA avoids forming d x d covariances when n << d.
            mode = PcaMode.gram if d > n else PcaMode.svd

        if mode == PcaMode.gram:
            # Gram matrix: matrix @ matrix.T
            matrix_t = b.transpose(matrix)
            gram = b.matmul(matrix, matrix_t)
            b.eval(gram)

            # GPU-only power iteration eigendecomposition - iterates until convergence
            n_eig_full = int(gram.shape[0])
            eigenvalues, eigenvectors = power_iteration_eigh(b, gram, k=n_eig_full)
            b.eval(eigenvalues, eigenvectors)

            # Sort in descending order (eigh returns ascending)
            # Use backend operations for reversal instead of NumPy
            n_eig = int(eigenvalues.shape[0])
            reverse_order = b.arange(n_eig - 1, -1, -1)
            b.eval(reverse_order)
            eigenvectors_reordered = b.take(eigenvectors, reverse_order, axis=1)
            eigenvalues_sorted = b.take(eigenvalues, reverse_order, axis=0)
            b.eval(eigenvectors_reordered, eigenvalues_sorted)

            # Singular values from eigenvalues
            eigenvalues_clamped = b.maximum(eigenvalues_sorted, b.zeros(eigenvalues_sorted.shape))
            singular_values = b.sqrt(eigenvalues_clamped)
            b.eval(singular_values)

            # Components: matrix.T @ (eigenvectors / singular_values)
            # Handle floor for division
            sv_eps = division_epsilon(b, singular_values)
            denom = b.maximum(
                singular_values,
                b.full(singular_values.shape, sv_eps),
            )
            b.eval(denom)
            eigenvectors_scaled = eigenvectors_reordered / denom
            components = b.matmul(matrix_t, eigenvectors_scaled)
            b.eval(components)
        else:
            # Direct SVD
            _, singular_values, v_t = _cache.get_or_compute_svd(matrix, b, full_matrices=False)
            b.eval(singular_values, v_t)
            components = b.transpose(v_t)
            b.eval(components)

        variances = singular_values * singular_values
        b.eval(variances)

        # Select number of components based on variance threshold
        k = SharedSubspaceProjector._select_component_count(
            variances, variance_threshold, backend=b
        )
        k = min(k, max_components, int(components.shape[1]))
        if k <= 0:
            return None, None, None

        # Truncate to k components
        reduced = b.matmul(matrix, components[:, :k])
        b.eval(reduced)

        return (
            reduced,
            components[:, :k],
            variances[:k],
        )

    @staticmethod
    def _spectral_gap_rank_list(variances: list[float]) -> int:
        """Find natural signal/noise boundary from max relative drop in variances.

        This is geometry-derived, not an arbitrary threshold.
        """
        if not variances:
            return 0
        # Filter positive values and sort descending
        positive = sorted([v for v in variances if v > 0], reverse=True)
        if len(positive) < 2:
            return len(positive)

        # Numerical floor derived from Python float dtype (float64)
        # Uses sqrt(machine_eps) consistent with division_epsilon pattern
        eps = sqrt_scalar(sys.float_info.epsilon, get_default_backend())
        max_gap = 0.0
        gap_index = 1  # Keep at least 1 component

        for i in range(len(positive) - 1):
            if positive[i] < eps:
                break
            relative_drop = (positive[i] - positive[i + 1]) / positive[i]
            if relative_drop > max_gap:
                max_gap = relative_drop
                gap_index = i + 1

        return gap_index

    @staticmethod
    def _select_component_count_list(variances: list[float], threshold: float | None) -> int:
        """Select component count from Python list of variances.

        If threshold is None, uses spectral gap detection.
        """
        if not variances:
            return 0

        # Use spectral gap detection when no threshold provided
        if threshold is None:
            return SharedSubspaceProjector._spectral_gap_rank_list(variances)

        total = sum(variances)
        if total <= 0.0:
            return 0
        cumulative = 0.0
        for idx, value in enumerate(variances):
            cumulative += value
            if cumulative / total >= threshold:
                return idx + 1
        return len(variances)

    @staticmethod
    def _select_component_count(
        variances: "Array",
        threshold: float | None,
        backend: "Backend | None" = None,
    ) -> int:
        """Select component count from backend array of variances.

        If threshold is None, uses spectral gap detection.
        """
        b = backend or get_default_backend()
        if variances.size == 0:
            return 0

        variances_flat = b.reshape(variances, (-1,))
        mask = variances_flat > 0
        pos_count_arr = b.sum(b.astype(mask, "int32"))
        b.eval(pos_count_arr)
        pos_count = int(b.to_scalar(pos_count_arr))
        if pos_count == 0:
            return 0

        neg_inf = b.full(variances_flat.shape, float("-inf"))
        filtered = b.where(mask, variances_flat, neg_inf)
        sorted_vals = b.sort(filtered)
        n_vals = int(variances_flat.shape[0])
        reverse_order = b.arange(n_vals - 1, -1, -1)
        b.eval(sorted_vals, reverse_order)
        desc = b.take(sorted_vals, reverse_order, axis=0)
        vals = desc[:pos_count]
        b.eval(vals)

        # Use spectral gap detection when no threshold provided
        if threshold is None:
            if pos_count < 2:
                return pos_count
            prev = vals[:-1]
            next_vals = vals[1:]
            eps = division_epsilon(b, variances_flat)
            denom = b.where(prev > eps, prev, b.full(prev.shape, float("inf")))
            rel_drop = (prev - next_vals) / denom
            rel_drop = b.where(prev > eps, rel_drop, b.full(prev.shape, float("-inf")))
            b.eval(rel_drop)
            max_drop_arr = b.max(rel_drop)
            max_drop_idx = b.argmax(rel_drop)
            b.eval(max_drop_arr, max_drop_idx)
            max_drop = float(b.to_scalar(max_drop_arr))
            if max_drop <= 0.0:
                return pos_count
            gap_index = int(b.to_scalar(max_drop_idx)) + 1
            return gap_index

        total_arr = b.sum(vals)
        b.eval(total_arr)
        total = float(b.to_scalar(total_arr))
        if total <= 0.0:
            return 0
        cumsum = b.cumsum(vals)
        ratio = cumsum / total
        meets = ratio >= threshold
        meets_count_arr = b.sum(b.astype(meets, "int32"))
        meets_idx = b.argmax(meets)
        b.eval(meets_count_arr, meets_idx)
        meets_count = int(b.to_scalar(meets_count_arr))
        if meets_count == 0:
            return pos_count
        return int(b.to_scalar(meets_idx)) + 1

    @staticmethod
    def _regularize_covariance(cov: "Array", epsilon: float, backend: "Backend | None" = None) -> "Array":
        if epsilon <= 0:
            return cov
        b = backend or get_default_backend()
        dim = int(cov.shape[0])
        eye = b.eye(dim)
        regularized = cov + (epsilon * eye)
        b.eval(regularized)
        return regularized

    @staticmethod
    def _whiten_covariance(
        cov: "Array",
        backend: "Backend | None" = None,
    ) -> tuple["Array | None", "Array | None"]:
        b = backend or get_default_backend()

        if cov.size == 0:
            return None, None

        # GPU-only power iteration eigendecomposition - iterates until convergence
        n_cov = int(cov.shape[0])
        eigenvalues, eigenvectors = power_iteration_eigh(b, cov, k=n_cov)
        b.eval(eigenvalues, eigenvectors)

        # Floor eigenvalues
        eig_eps = division_epsilon(b, eigenvalues)
        floor = b.full(eigenvalues.shape, eig_eps)
        eigenvalues_floored = b.maximum(eigenvalues, floor)
        b.eval(eigenvalues_floored)

        # Compute inverse sqrt diagonal: diag(1 / sqrt(eigenvalues))
        inv_sqrt_diag = 1.0 / b.sqrt(eigenvalues_floored)
        b.eval(inv_sqrt_diag)

        # inv_sqrt = eigenvectors @ diag(inv_sqrt_diag) @ eigenvectors.T
        diag_matrix = b.diag(inv_sqrt_diag)
        eigenvectors_t = b.transpose(eigenvectors)
        inv_sqrt = b.matmul(b.matmul(eigenvectors, diag_matrix), eigenvectors_t)
        b.eval(inv_sqrt)

        return inv_sqrt, eigenvalues_floored

    @staticmethod
    def _compute_covariance(x: list[list[float]], y: list[list[float]], n: int) -> list[float]:
        if n <= 0:
            return []
        d_x = len(x[0])
        d_y = len(y[0])
        b = get_default_backend()
        x_arr = b.array(x)
        y_arr = b.array(y)
        b.eval(x_arr, y_arr)

        x_feat = b.transpose(x_arr)  # [d_x, n]
        y_feat = b.transpose(y_arr)  # [d_y, n]

        cos_arr = geodesic_cosine_between_sets(x_feat, y_feat, b)
        x_norms = geodesic_norms(x_feat, b)
        y_norms = geodesic_norms(y_feat, b)
        b.eval(cos_arr, x_norms, y_norms)

        x_norm_col = b.reshape(x_norms, (-1, 1))
        y_norm_row = b.reshape(y_norms, (1, -1))
        cov_arr = cos_arr * x_norm_col * y_norm_row
        cov_arr = cov_arr * (1.0 / float(n))
        b.eval(cov_arr)

        flat = b.reshape(cov_arr, (-1,))
        return [float(v) for v in b.tolist(flat)]

    @staticmethod
    def _compute_covariance_flat(
        x: list[float],
        y: list[float],
        n: int,
        d_x: int,
        d_y: int,
    ) -> list[float]:
        if n <= 0:
            return []
        b = get_default_backend()
        x_arr = b.reshape(b.array(x), (n, d_x))
        y_arr = b.reshape(b.array(y), (n, d_y))
        b.eval(x_arr, y_arr)

        x_feat = b.transpose(x_arr)
        y_feat = b.transpose(y_arr)

        cos_arr = geodesic_cosine_between_sets(x_feat, y_feat, b)
        x_norms = geodesic_norms(x_feat, b)
        y_norms = geodesic_norms(y_feat, b)
        b.eval(cos_arr, x_norms, y_norms)

        x_norm_col = b.reshape(x_norms, (-1, 1))
        y_norm_row = b.reshape(y_norms, (1, -1))
        cov_arr = cos_arr * x_norm_col * y_norm_row
        cov_arr = cov_arr * (1.0 / float(n))
        b.eval(cov_arr)

        flat = b.reshape(cov_arr, (-1,))
        return [float(v) for v in b.tolist(flat)]

    @staticmethod
    def _compute_gram_matrix(x: list[list[float]], n: int, d: int) -> list[float]:
        if n <= 0 or d <= 0:
            return []
        b = get_default_backend()
        x_arr = b.array(x)
        cos_arr = geodesic_cosine_matrix(x_arr, b)
        norms = geodesic_norms(x_arr, b)
        b.eval(cos_arr, norms)

        norm_col = b.reshape(norms, (-1, 1))
        norm_row = b.reshape(norms, (1, -1))
        gram_arr = cos_arr * norm_col * norm_row
        b.eval(gram_arr)

        flat = b.reshape(gram_arr, (-1,))
        return [float(v) for v in b.tolist(flat)]

    @staticmethod
    def _compute_whitening_transform(
        cov: list[float],
        dim: int,
    ) -> tuple[list[float], list[float]] | None:
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(cov, dim)
        if eigenvalues is None:
            return None
        eigen_float = [float(val) for val in eigenvalues]
        # Use machine epsilon as threshold for meaningful eigenvalues
        eps = sys.float_info.epsilon
        min_eigen = min([val for val in eigen_float if val > eps], default=eps)

        inv_sqrt = [0.0 for _ in range(dim * dim)]
        for i in range(dim):
            diag_val = cov[i * dim + i]
            inv_sqrt[i * dim + i] = 1.0 / sqrt_scalar(max(diag_val, min_eigen), get_default_backend())
        return inv_sqrt, eigen_float
