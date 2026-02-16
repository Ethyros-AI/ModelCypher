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

"""Entanglement spectrum: CCA-based coupling measurement between activation matrices.

Measures the degree of shared structure between two activation matrices using
Canonical Correlation Analysis. Returns raw measurements only (no interpretation).

The entanglement spectrum measures:
- Shared variance between two representations (canonical correlations)
- Effective dimensionality of the shared subspace
- Distribution of correlation strengths

Entropy measures the uniformity of correlations:
- Low entropy = one dominant correlation (simple dependence)
- High entropy = uniform correlations (complex entanglement)

References:
    - Hotelling, H. (1936). "Relations Between Two Sets of Variates."
      Biometrika, 28(3/4), 321-377.
    - Raghu, M., et al. (2017). "SVCCA: Singular Vector Canonical Correlation
      Analysis for Deep Learning Dynamics and Interpretability."
      arXiv:1706.05806.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
    power_iteration_eigh,
    regularization_epsilon,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class EntanglementSpectrumResult:
    """Raw entanglement spectrum measurements between two activation matrices.

    All fields are raw measurements with no interpretation strings.
    """

    canonical_correlations: list[float]
    """Canonical correlations sigma_1 >= sigma_2 >= ... >= sigma_k, clipped to [0, 1]."""

    entanglement_entropy: float
    """Shannon entropy of normalized correlation spectrum: H = -sum(p_i * log(p_i))."""

    effective_rank_shannon: float
    """Shannon effective rank: exp(H). Measures effective number of shared dimensions."""

    effective_rank_renyi: float
    """Renyi effective rank (participation ratio): (sum sigma)^2 / sum(sigma^2)."""

    correlation_count: int
    """Number of canonical correlations: min(d_source, d_target)."""

    sample_count: int
    """Number of samples used for CCA."""

    source_dimension: int
    """Feature dimension of source activations."""

    target_dimension: int
    """Feature dimension of target activations."""

    condition_number: float
    """Condition number of whitening transformation. High values indicate instability."""


class EntanglementSpectrum:
    """Compute entanglement spectrum between two activation matrices via CCA."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize with optional backend.

        Args:
            backend: Backend for tensor operations. If None, uses default.
        """
        self._backend = backend or get_default_backend()

    def compute(
        self,
        source: "Array",
        target: "Array",
    ) -> EntanglementSpectrumResult:
        """Compute entanglement spectrum via Canonical Correlation Analysis.

        Args:
            source: Activation matrix A [n_samples, d_source].
            target: Activation matrix B [n_samples, d_target].
                    Must have same n_samples as source.

        Returns:
            EntanglementSpectrumResult with raw measurements.

        Raises:
            ValueError: If inputs have incompatible shapes.
        """
        b = self._backend

        # Ensure arrays
        source_arr = b.array(source) if not hasattr(source, "shape") else source
        target_arr = b.array(target) if not hasattr(target, "shape") else target
        b.eval(source_arr, target_arr)

        # Validate shapes
        if len(source_arr.shape) != 2 or len(target_arr.shape) != 2:
            raise ValueError(
                f"Expected 2D arrays, got shapes {source_arr.shape} and {target_arr.shape}"
            )

        n_source = int(source_arr.shape[0])
        n_target = int(target_arr.shape[0])
        if n_source != n_target:
            raise ValueError(
                f"Sample counts must match: source has {n_source}, target has {n_target}"
            )

        n_samples = n_source
        d_source = int(source_arr.shape[1])
        d_target = int(target_arr.shape[1])

        # Handle degenerate cases
        if n_samples == 0 or d_source == 0 or d_target == 0:
            return EntanglementSpectrumResult(
                canonical_correlations=[],
                entanglement_entropy=0.0,
                effective_rank_shannon=0.0,
                effective_rank_renyi=0.0,
                correlation_count=0,
                sample_count=n_samples,
                source_dimension=d_source,
                target_dimension=d_target,
                condition_number=0.0,
            )

        # Minimum samples for stable CCA: n >= sqrt(max_dim)
        # Following SharedSubspaceProjector pattern - prevents meaningless outputs
        max_dim = max(d_source, d_target)
        min_samples = int(max_dim**0.5) + 1
        if n_samples < min_samples:
            return EntanglementSpectrumResult(
                canonical_correlations=[],
                entanglement_entropy=0.0,
                effective_rank_shannon=0.0,
                effective_rank_renyi=0.0,
                correlation_count=0,
                sample_count=n_samples,
                source_dimension=d_source,
                target_dimension=d_target,
                condition_number=float("inf"),  # Signal insufficient samples
            )

        # Center both matrices
        source_centered, _ = self._center_array(source_arr)
        target_centered, _ = self._center_array(target_arr)

        # Compute canonical correlations
        correlations, condition_number = self._compute_canonical_correlations(
            source_centered, target_centered, n_samples
        )

        if correlations is None:
            return EntanglementSpectrumResult(
                canonical_correlations=[],
                entanglement_entropy=0.0,
                effective_rank_shannon=0.0,
                effective_rank_renyi=0.0,
                correlation_count=0,
                sample_count=n_samples,
                source_dimension=d_source,
                target_dimension=d_target,
                condition_number=condition_number,
            )

        # Compute entropy and effective ranks
        entropy, shannon_rank, renyi_rank = self._compute_entropy_metrics(correlations)

        # Convert correlations to list
        correlation_list = [float(v) for v in b.tolist(correlations)]
        correlation_count = len(correlation_list)

        return EntanglementSpectrumResult(
            canonical_correlations=correlation_list,
            entanglement_entropy=entropy,
            effective_rank_shannon=shannon_rank,
            effective_rank_renyi=renyi_rank,
            correlation_count=correlation_count,
            sample_count=n_samples,
            source_dimension=d_source,
            target_dimension=d_target,
            condition_number=condition_number,
        )

    def _center_array(self, array: "Array") -> tuple["Array", "Array"]:
        """Center array by subtracting mean along sample axis.

        Args:
            array: Input array [n_samples, features].

        Returns:
            Tuple of (centered_array, mean).
        """
        b = self._backend
        mean = b.mean(array, axis=0)
        b.eval(mean)
        centered = array - mean
        b.eval(centered)
        return centered, mean

    def _compute_canonical_correlations(
        self,
        source_centered: "Array",
        target_centered: "Array",
        n_samples: int,
    ) -> tuple["Array | None", float]:
        """Compute canonical correlations via SVD of whitened cross-covariance.

        Follows the pattern from shared_subspace_projector.py:
        1. Compute covariances Cxx, Cyy, Cxy
        2. Regularize with dtype-derived epsilon
        3. Whiten via eigendecomposition
        4. SVD of whitened cross-covariance
        5. Clip singular values to [0, 1]

        Args:
            source_centered: Centered source activations [n, d_source].
            target_centered: Centered target activations [n, d_target].
            n_samples: Number of samples.

        Returns:
            Tuple of (canonical_correlations, condition_number).
            correlations is None if computation fails.
        """
        b = self._backend

        # Covariances: C = X^T @ X / n
        source_t = b.transpose(source_centered)  # [d_source, n]
        target_t = b.transpose(target_centered)  # [d_target, n]

        scale = 1.0 / float(n_samples)
        cxx = b.matmul(source_t, source_centered) * scale  # [d_source, d_source]
        cyy = b.matmul(target_t, target_centered) * scale  # [d_target, d_target]
        cxy = b.matmul(source_t, target_centered) * scale  # [d_source, d_target]
        b.eval(cxx, cyy, cxy)

        # Regularize covariances with dtype-derived epsilon
        reg_eps = regularization_epsilon(b, cxx)
        cxx = self._regularize_covariance(cxx, reg_eps)
        cyy = self._regularize_covariance(cyy, reg_eps)

        # Whiten covariances
        inv_sqrt_x, x_eigenvalues, cond_x = self._whiten_covariance(cxx)
        inv_sqrt_y, y_eigenvalues, cond_y = self._whiten_covariance(cyy)

        if inv_sqrt_x is None or inv_sqrt_y is None:
            return None, max(cond_x, cond_y)

        condition_number = max(cond_x, cond_y)

        # Cross-covariance in whitened space
        cross_cov = b.matmul(b.matmul(inv_sqrt_x, cxy), inv_sqrt_y)
        b.eval(cross_cov)

        # SVD of cross-covariance (use geodesic_svd for backend compatibility)
        _, singular_values, _ = geodesic_svd(b, cross_cov)
        b.eval(singular_values)

        # Clip to [0, 1] for canonical correlations
        correlations = b.clip(singular_values, 0.0, 1.0)
        b.eval(correlations)

        # Filter by minimum meaningful correlation: sqrt(machine_epsilon)
        eps = float(machine_epsilon(b, correlations))
        min_corr = eps**0.5
        min_corr_arr = b.full(correlations.shape, min_corr)
        valid_mask = correlations >= min_corr_arr
        valid_count_arr = b.sum(b.astype(valid_mask, "int32"))
        b.eval(valid_count_arr)
        valid_count = int(b.to_scalar(valid_count_arr))

        if valid_count == 0:
            return None, condition_number

        # Keep only valid correlations (sorted descending from SVD)
        correlations = correlations[:valid_count]
        b.eval(correlations)

        return correlations, condition_number

    def _regularize_covariance(self, cov: "Array", epsilon: float) -> "Array":
        """Add regularization to covariance diagonal.

        Args:
            cov: Covariance matrix [d, d].
            epsilon: Regularization strength.

        Returns:
            Regularized covariance.
        """
        if epsilon <= 0:
            return cov

        b = self._backend
        dim = int(cov.shape[0])
        eye = b.eye(dim)
        regularized = cov + (epsilon * eye)
        b.eval(regularized)
        return regularized

    def _whiten_covariance(
        self,
        cov: "Array",
    ) -> tuple["Array | None", "Array | None", float]:
        """Compute whitening transformation via eigendecomposition.

        Args:
            cov: Covariance matrix [d, d].

        Returns:
            Tuple of (inv_sqrt_cov, eigenvalues, condition_number).
            inv_sqrt_cov is None if whitening fails.
        """
        b = self._backend

        if cov.size == 0:
            return None, None, 0.0

        # Eigendecomposition
        n_cov = int(cov.shape[0])
        eigenvalues, eigenvectors = power_iteration_eigh(b, cov, k=n_cov)
        b.eval(eigenvalues, eigenvectors)

        # Compute condition number
        div_eps = division_epsilon(b, eigenvalues)
        max_eig_arr = b.max(eigenvalues)
        min_eig_arr = b.min(eigenvalues)
        b.eval(max_eig_arr, min_eig_arr)
        max_eig = float(b.to_scalar(max_eig_arr))
        min_eig = float(b.to_scalar(min_eig_arr))

        if min_eig > div_eps:
            condition_number = max_eig / min_eig
        else:
            condition_number = float("inf")

        # Floor eigenvalues for numerical stability
        floor = b.full(eigenvalues.shape, div_eps)
        eigenvalues_floored = b.maximum(eigenvalues, floor)
        b.eval(eigenvalues_floored)

        # Compute inverse sqrt: V @ diag(1/sqrt(lambda)) @ V^T
        inv_sqrt_diag = 1.0 / b.sqrt(eigenvalues_floored)
        b.eval(inv_sqrt_diag)

        diag_matrix = b.diag(inv_sqrt_diag)
        eigenvectors_t = b.transpose(eigenvectors)
        inv_sqrt = b.matmul(b.matmul(eigenvectors, diag_matrix), eigenvectors_t)
        b.eval(inv_sqrt)

        return inv_sqrt, eigenvalues_floored, condition_number

    def _compute_entropy_metrics(
        self,
        correlations: "Array",
    ) -> tuple[float, float, float]:
        """Compute entanglement entropy and effective ranks from correlations.

        Follows patterns from spectral_signature.py and effective_rank.py:
        - Shannon entropy: H = -sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma)
        - Shannon effective rank: exp(H)
        - Renyi effective rank: (sum sigma)^2 / sum(sigma^2)

        Args:
            correlations: Canonical correlations array.

        Returns:
            Tuple of (entropy, shannon_rank, renyi_rank).
        """
        b = self._backend

        # Sum for normalization
        total = b.sum(correlations)
        total_sq = b.sum(correlations * correlations)
        b.eval(total, total_sq)

        total_val = float(b.to_scalar(total))
        total_sq_val = float(b.to_scalar(total_sq))

        div_eps = division_epsilon(b, correlations)

        # Handle degenerate case
        if total_val <= div_eps:
            return 0.0, 0.0, 0.0

        # Normalize to simplex weights
        probs = correlations / total_val
        b.eval(probs)

        # Shannon entropy with safe log
        log_eps = safe_log_epsilon(b, probs)
        log_eps_arr = b.full(probs.shape, log_eps)
        probs_safe = b.where(probs > log_eps, probs, log_eps_arr)
        log_probs = b.log(probs_safe)
        entropy_terms = -probs * log_probs
        # Zero out terms where p was effectively zero
        entropy_terms = b.where(probs > log_eps, entropy_terms, b.zeros_like(entropy_terms))
        entropy_arr = b.sum(entropy_terms)
        b.eval(entropy_arr)
        entropy = float(b.to_scalar(entropy_arr))

        # Shannon effective rank: exp(H)
        shannon_rank_arr = b.exp(entropy_arr)
        b.eval(shannon_rank_arr)
        shannon_rank = float(b.to_scalar(shannon_rank_arr))

        # Renyi effective rank (participation ratio): (sum)^2 / sum(sq)
        if total_sq_val > div_eps:
            renyi_rank = (total_val * total_val) / total_sq_val
        else:
            renyi_rank = 0.0

        return entropy, shannon_rank, renyi_rank


def compute_entanglement_spectrum(
    source: "Array",
    target: "Array",
    backend: "Backend | None" = None,
) -> EntanglementSpectrumResult:
    """Convenience function for computing entanglement spectrum.

    Args:
        source: Source activation matrix [n_samples, d_source].
        target: Target activation matrix [n_samples, d_target].
        backend: Optional backend. If None, uses default.

    Returns:
        EntanglementSpectrumResult with raw measurements.
    """
    es = EntanglementSpectrum(backend)
    return es.compute(source, target)
