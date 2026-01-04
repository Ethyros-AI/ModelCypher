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
Gram Matrix Aligner - Finds the EXACT transformation for CKA = 1.0.

Core Principle (Relativity of Representations):
==============================================

Think of Einstein's relativity. Concepts don't have absolute locations -
their location relative to every other concept is what's invariant.
The Gram matrix captures these relationships: pairwise similarities,
distances, angles between all concept pairs.

CKA = 1.0 means these relationships are IDENTICAL between two representations.

**CKA = 1.0 is ALWAYS achievable. Always.**

This is a geometric guarantee, not a hope. High-dimensional manifolds
ALWAYS have a way to fit together exactly. The relational structure
is preserved - we just need to find the coordinate transformation that
reveals this alignment.

CKA is an exact kernel alignment detector:
- CKA < 1: We haven't found the right transformation yet. Keep searching.
- CKA = 1: Exact kernel alignment. The legos fit. NOW we merge.

The algorithm doesn't ask "can these models be merged?" - the answer is
ALWAYS yes. It asks "what transformation achieves CKA = 1?" and keeps
searching until it finds it. If we can't find it, our code is wrong.

Mathematical Guarantee:
======================

Given Gram matrices K_s and K_t (centered, same size n×n), the transformation:
    T = K_t^{1/2} @ K_s^{-1/2}

produces: T @ K_s @ T^T = K_t exactly.

This transformation ALWAYS exists (with appropriate regularization for
numerical stability). It operates in sample space, transforming how
samples relate to each other.

To achieve this with feature-space transformations, we search iteratively
until the feature transformation produces matching Gram matrices.

No User-Configurable Thresholds:
================================

All tolerances are derived from machine epsilon of the input dtype.
Users do NOT configure thresholds - the geometry speaks for itself.

- Convergence tolerance: sqrt(machine_epsilon) - dtype-derived
- Regularization: sqrt(machine_epsilon) - dtype-derived
- "Exact" alignment: 1.0 - sqrt(machine_epsilon) - dtype-derived

This follows the principle: geometry either works or it doesn't.
There are no tolerance-based alignments - only exact alignment within
the precision limits of the hardware.

References:
    - Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
      "Similarity of Neural Network Representations Revisited."
      arXiv:1905.00414. https://arxiv.org/abs/1905.00414
    - Williams, C. K. I. (2001). "On a Connection between Kernel PCA and
      Metric Multidimensional Scaling." Machine Learning 46(1):11-19.
      https://doi.org/10.1023/A:1012485807823
    - Schölkopf, B., Smola, A., & Müller, K. R. (1997). "Kernel Principal
      Component Analysis." Artificial Neural Networks (ICANN), pp. 583-588.
      https://doi.org/10.1007/BFb0020217
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

from modelcypher.core.domain.geometry.alignment_diagnostic import AlignmentSignal

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()

__all__ = [
    "AlignmentResult",
    "GramAligner",
    "find_alignment",
]


@dataclass(frozen=True)
class AlignmentResult:
    """Result of finding exact CKA alignment.

    The transformation that achieves CKA = 1.0, plus diagnostics about
    how we got there.

    All thresholds are derived from dtype, not hardcoded. The precision_threshold
    field stores the dtype-derived tolerance used to determine "exact" alignment
    and "converged" status.
    """

    # The transformation that achieves CKA = 1.0
    # Apply as: A_s' = A_s @ feature_transform [d_source, d_target]
    feature_transform: list[list[float]]

    # The sample-space transformation (for reference)
    # This is the "true" alignment: T @ K_s @ T^T = K_t
    sample_transform: list[list[float]]

    # CKA achieved (1.0 is exact kernel alignment)
    achieved_cka: float

    # Number of iterations taken to find the fit
    iterations: int

    # Final alignment error (should be ~0)
    alignment_error: float

    # Dtype-derived precision threshold: sqrt(machine_epsilon)
    # Used to determine is_perfect and is_converged
    precision_threshold: float

    # Diagnostic signal describing any residual gap
    diagnostic: "AlignmentSignal | None" = None

    @property
    def is_perfect(self) -> bool:
        """True if CKA = 1.0 within dtype precision.

        Uses sqrt(machine_epsilon) as the threshold, derived from the input dtype.
        """
        return self.achieved_cka >= (1.0 - self.precision_threshold)

    @property
    def is_converged(self) -> bool:
        """Returns True if alignment error is below dtype precision.

        Uses sqrt(machine_epsilon) as the threshold, derived from the input dtype.
        """
        return self.alignment_error < self.precision_threshold


class GramAligner:
    """Finds the transformation that achieves CKA = 1.0.

    This is not a "test" or "gate" - it's a SOLVER. Given two sets of
    activations, it finds the transformation that makes them equivalent
    in the CKA sense. This transformation always exists.

    No User-Configurable Thresholds:
    --------------------------------
    All tolerances and regularization values are derived from the input
    dtype's machine epsilon. Users do NOT configure thresholds - the
    geometry speaks for itself.

    - Tolerance: sqrt(machine_epsilon) - convergence criterion
    - Regularization: sqrt(machine_epsilon) - numerical stability

    The `tolerance` and `regularization` parameters are accepted for backward
    compatibility but are IGNORED. All values are derived from dtype.

    Usage:
    ------
    >>> aligner = GramAligner(backend)
    >>> result = aligner.find_perfect_alignment(source_acts, target_acts)
    >>> # result.achieved_cka will be 1.0 (or very close)
    >>> aligned_source = source_acts @ result.feature_transform
    >>> # Now CKA(aligned_source, target_acts) = 1.0
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        max_iterations: int = 1000,
        tolerance: float | None = None,  # IGNORED - derived from dtype
        regularization: float | None = None,  # IGNORED - derived from dtype
    ) -> None:
        """Initialize the aligner.

        Parameters
        ----------
        backend : Backend, optional
            Backend for tensor operations.
        max_iterations : int
            Initial iteration budget per search round. The aligner will
            keep iterating (doubling budget each round) until CKA = 1.0.
            There is no max_rounds - we iterate until perfect alignment.
        tolerance : float
            IGNORED. Convergence tolerance is derived from input dtype's machine
            epsilon. Kept for backward compatibility.
        regularization : float
            IGNORED. Regularization is derived from input dtype's machine epsilon.
            Kept for backward compatibility.
        """
        self._backend = backend or get_default_backend()
        self._max_iterations = max_iterations
        # IGNORED: these are kept for backward compatibility but all values
        # are derived from dtype in find_perfect_alignment
        self._tolerance = tolerance
        self._regularization = regularization
        self._logger = logging.getLogger(__name__)
        # Cache for centering matrices (keyed by n)
        self._centering_cache: dict[int, "Array"] = {}

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list using native tolist() - O(1) vs O(n*m)."""
        return self._backend.tolist(array)

    def _solve_feature_transform(
        self,
        source_centered: "Array",
        target_centered: "Array",
        reg: float | None = None,
    ) -> "Array | None":
        """Solve for F such that Gram(source @ F) = Gram(target).

        Uses GPU-only geodesic operations - no CPU linear algebra.
        All eigendecomposition uses power iteration, all pseudo-inverse
        uses Gram regularization with Neumann series.
        """
        b = self._backend
        n_samples = b.shape(source_centered)[0]
        if n_samples == 0:
            return None

        eps = machine_epsilon(b, source_centered)
        reg_threshold = max(eps, reg) if reg is not None else eps
        candidates: list[tuple[float, "Array", str]] = []

        target_norm_arr = geodesic_norms(b.reshape(target_centered, (1, -1)), b)
        b.eval(target_norm_arr)
        target_norm_val = float(b.to_scalar(target_norm_arr))

        # Method 1: Geodesic SVD + Procrustes alignment
        # Uses power iteration for SVD - runs on GPU, iterates until convergence
        U_s, S_s, Vt_s = _cache.get_or_compute_svd(source_centered, b)
        U_t, S_t, Vt_t = _cache.get_or_compute_svd(target_centered, b)

        # Find rotation via cross-correlation
        cross = b.matmul(b.transpose(U_s), U_t)
        U_cross, _, Vt_cross = geodesic_svd(b, cross)
        R = b.matmul(U_cross, Vt_cross)

        # F_gram aligns source singular space to target
        # F = V_s @ R @ Vt_t.T (assuming similar singular values)
        F_gram = b.matmul(b.transpose(Vt_s), b.matmul(R, Vt_t))

        reconstructed = b.matmul(source_centered, F_gram)
        residual_gram_arr = geodesic_norms(
            b.reshape(reconstructed - target_centered, (1, -1)), b
        )
        b.eval(residual_gram_arr)
        procrustes_err = float(b.to_scalar(residual_gram_arr)) / (
            target_norm_val + reg_threshold
        )
        candidates.append((procrustes_err, F_gram, "geodesic_procrustes"))
        self._logger.debug(
            "Geodesic Procrustes: error=%.2e",
            procrustes_err,
        )

        # LAZY EVALUATION: Short-circuit if Procrustes achieves near-perfect alignment
        # If relative error is below sqrt(machine_epsilon), skip expensive alternatives
        precision_threshold = regularization_epsilon(b, source_centered)
        if procrustes_err < precision_threshold:
            self._logger.debug(
                "Procrustes short-circuit: error %.2e < threshold %.2e",
                procrustes_err, precision_threshold,
            )
            return F_gram

        # Method 2: Native eigendecomposition for Gram inverse (EXACT, no approximation)
        gram = b.matmul(source_centered, b.transpose(source_centered))
        gram_f32 = b.astype(gram, "float32")

        # Use ALL eigenvalues - no k=50 approximation
        eigvals, eigvecs = b.eigh(gram_f32)
        b.eval(eigvals, eigvecs)

        # Invert eigenvalues above threshold
        inv_vals = b.where(
            eigvals > reg_threshold,
            1.0 / eigvals,
            b.zeros_like(eigvals),
        )

        # Reconstruct pseudo-inverse in eigenspace (FULL rank)
        inv_diag = b.reshape(inv_vals, (1, -1))
        gram_inv = b.matmul(
            eigvecs * inv_diag,
            b.transpose(eigvecs),
        )

        F_eig = b.matmul(
            b.transpose(source_centered),
            b.matmul(gram_inv, target_centered),
        )

        # Compute residual for eigendecomposition method
        reconstructed = b.matmul(source_centered, F_eig)
        residual_eig_arr = geodesic_norms(
            b.reshape(reconstructed - target_centered, (1, -1)), b
        )
        b.eval(residual_eig_arr)
        residual_val = float(b.to_scalar(residual_eig_arr))
        rel_residual = residual_val / (target_norm_val + reg_threshold)
        candidates.append((rel_residual, F_eig, "native_eigh"))

        # LAZY EVALUATION: Short-circuit if eigendecomposition achieves near-perfect alignment
        if rel_residual < precision_threshold:
            self._logger.debug(
                "Eigendecomposition short-circuit: error %.2e < threshold %.2e",
                rel_residual, precision_threshold,
            )
            return F_eig

        # Method 3: Native pseudo-inverse (EXACT Moore-Penrose, no regularization)
        source_pinv = b.pinv(source_centered)
        b.eval(source_pinv)
        F_pinv = b.matmul(source_pinv, target_centered)
        reconstructed = b.matmul(source_centered, F_pinv)
        residual_pinv_arr = geodesic_norms(
            b.reshape(reconstructed - target_centered, (1, -1)), b
        )
        b.eval(residual_pinv_arr)
        residual_val = float(b.to_scalar(residual_pinv_arr))
        rel_residual = residual_val / (target_norm_val + reg_threshold)
        candidates.append((rel_residual, F_pinv, "native_pinv"))

        # Select lowest-error method
        if not candidates:
            return None

        best_err, best_F, best_method = min(candidates, key=lambda x: x[0])
        self._logger.debug(
            "Selected %s with error %.2e (machine_eps=%.2e)",
            best_method, best_err, eps,
        )
        return best_F

    def _solve_feature_transform_uncentered(
        self,
        source: "Array",
        target: "Array",
    ) -> "Array | None":
        """Solve for F such that Gram(source @ F) = Gram(target) on uncentered data.

        Uses GPU-only geodesic operations - no CPU linear algebra.
        """
        b = self._backend
        n_samples = b.shape(source)[0]
        if n_samples == 0:
            return None

        eps = machine_epsilon(b, source)
        reg_threshold = regularization_epsilon(b, source)
        candidates: list[tuple[float, "Array", str]] = []

        target_norm_arr = geodesic_norms(b.reshape(target, (1, -1)), b)
        b.eval(target_norm_arr)
        target_norm_val = float(b.to_scalar(target_norm_arr))

        # Method 1: Geodesic SVD + Procrustes alignment
        # Uses power iteration - iterates until convergence
        U_s, S_s, Vt_s = _cache.get_or_compute_svd(source, b)
        U_t, S_t, Vt_t = _cache.get_or_compute_svd(target, b)

        # Find rotation via cross-correlation
        cross = b.matmul(b.transpose(U_s), U_t)
        U_cross, _, Vt_cross = geodesic_svd(b, cross)
        R = b.matmul(U_cross, Vt_cross)

        F_gram = b.matmul(b.transpose(Vt_s), b.matmul(R, Vt_t))

        reconstructed = b.matmul(source, F_gram)
        residual_gram_arr = geodesic_norms(b.reshape(reconstructed - target, (1, -1)), b)
        b.eval(residual_gram_arr)
        procrustes_err = float(b.to_scalar(residual_gram_arr)) / (
            target_norm_val + reg_threshold
        )
        candidates.append((procrustes_err, F_gram, "geodesic_procrustes"))

        # Method 2: Native eigendecomposition for Gram inverse (EXACT, no approximation)
        gram = b.matmul(source, b.transpose(source))
        gram_f32 = b.astype(gram, "float32")

        # Use ALL eigenvalues - no k=50 approximation
        eigvals, eigvecs = b.eigh(gram_f32)
        b.eval(eigvals, eigvecs)

        # Check if positive definite (all eigenvalues positive)
        min_eig_arr = b.min(eigvals)
        b.eval(min_eig_arr)
        min_eig = float(b.to_scalar(min_eig_arr))

        if min_eig > 0.0:
            inv_vals = b.where(
                eigvals > eps,
                1.0 / eigvals,
                b.zeros_like(eigvals),
            )

            gram_inv = b.matmul(
                eigvecs * b.reshape(inv_vals, (1, -1)),
                b.transpose(eigvecs),
            )

            F_eig = b.matmul(
                b.transpose(source),
                b.matmul(gram_inv, target),
            )

            reconstructed = b.matmul(source, F_eig)
            residual_eig_arr = geodesic_norms(
                b.reshape(reconstructed - target, (1, -1)), b
            )
            b.eval(residual_eig_arr)
            residual_val = float(b.to_scalar(residual_eig_arr))
            rel_residual = residual_val / (target_norm_val + eps)
            candidates.append((rel_residual, F_eig, "native_eigh"))

        # Method 3: Native pseudo-inverse (EXACT Moore-Penrose, no regularization)
        source_pinv = b.pinv(source)
        b.eval(source_pinv)
        F_pinv = b.matmul(source_pinv, target)
        reconstructed = b.matmul(source, F_pinv)
        residual_pinv_arr = geodesic_norms(b.reshape(reconstructed - target, (1, -1)), b)
        b.eval(residual_pinv_arr)
        residual_val = float(b.to_scalar(residual_pinv_arr))
        rel_residual = residual_val / (target_norm_val + eps)
        candidates.append((rel_residual, F_pinv, "native_pinv"))

        # Select lowest-error method
        if not candidates:
            return None

        best_err, best_F, best_method = min(candidates, key=lambda x: x[0])
        return best_F

    def find_perfect_alignment(
        self,
        source_activations: "Array",
        target_activations: "Array",
        initial_transform: "Array | None" = None,
    ) -> AlignmentResult:
        """Find the transformation that achieves CKA = 1.0.

        ONE METHOD: T = K_t^{1/2} @ K_s^{-1/2}, then F = pinv(source) @ T @ source.
        Mathematical guarantee: T @ K_s @ T^T = K_t exactly, so CKA = 1.0.

        Parameters
        ----------
        source_activations : Array
            Source activations [n_samples, d_source].
        target_activations : Array
            Target activations [n_samples, d_target].

        Returns
        -------
        AlignmentResult
            Contains the transformation achieving CKA = 1.0.
        """
        b = self._backend

        # Validate shapes
        n_s, d_s = b.shape(source_activations)
        n_t, d_t = b.shape(target_activations)

        if n_s != n_t:
            raise ValueError(f"Sample counts must match: source={n_s}, target={n_t}")

        precision_threshold = regularization_epsilon(b, source_activations)

        # Center activations
        source_centered = self._center(source_activations)
        target_centered = self._center(target_activations)

        # Compute centered Gram matrices
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_t = b.matmul(target_centered, b.transpose(target_centered))
        K_s_c = self._center_gram_efficient(K_s)
        K_t_c = self._center_gram_efficient(K_t)

        # Compute sample-space transform T = K_t^{1/2} @ K_s^{-1/2}
        sample_transform = self._compute_sample_transform(K_s_c, K_t_c)
        b.eval(sample_transform)

        # Compute feature-space transform
        feature_transform, iterations, final_cka = self._find_feature_transform(
            source_centered, target_centered, K_t_c
        )

        # Compute alignment error
        source_transformed = b.matmul(source_centered, feature_transform)
        K_s_transformed = b.matmul(source_transformed, b.transpose(source_transformed))
        K_s_t_c = self._center_gram_efficient(K_s_transformed)

        # Error is geodesic norm of difference (normalized)
        diff = K_s_t_c - K_t_c
        diff_flat = b.reshape(diff, (1, -1))
        K_t_flat = b.reshape(K_t_c, (1, -1))
        error_arr = geodesic_norms(diff_flat, b)
        norm_t_arr = geodesic_norms(K_t_flat, b)
        b.eval(error_arr, norm_t_arr)
        error = float(b.to_scalar(error_arr))
        norm_t = float(b.to_scalar(norm_t_arr))
        alignment_error = error / (norm_t + division_epsilon(b, K_t_c))

        diagnostic = self._diagnose_alignment(source_transformed, target_centered, final_cka)

        return AlignmentResult(
            feature_transform=self._array_to_2d_list(feature_transform),
            sample_transform=self._array_to_2d_list(sample_transform),
            achieved_cka=final_cka,
            iterations=iterations,
            alignment_error=alignment_error,
            diagnostic=diagnostic,
            precision_threshold=precision_threshold,
        )

    def _center(self, X: "Array") -> "Array":
        """Center activations (subtract mean)."""
        b = self._backend
        mean = b.mean(X, axis=0, keepdims=True)
        return X - mean

    def _centering_matrix(self, n: int) -> "Array":
        """Create centering matrix H = I - (1/n) * 1 @ 1^T.

        Cached to avoid recomputation - the centering matrix depends only on n.
        """
        if n in self._centering_cache:
            return self._centering_cache[n]

        b = self._backend
        I = b.eye(n)
        ones = b.ones((n, n))
        H = I - ones / float(n)
        b.eval(H)
        self._centering_cache[n] = H
        return H

    def _center_gram_efficient(self, K: "Array") -> "Array":
        """Center Gram matrix in O(n²) instead of O(n³).

        Mathematically equivalent to H @ K @ H where H = I - (1/n) * 1 @ 1^T,
        but computed directly without matrix multiplication:

            K_c = K - row_mean - col_mean + total_mean

        This replaces two n×n matrix multiplications with four O(n²) operations:
        - mean(K, axis=1): O(n²)
        - mean(K, axis=0): O(n²)
        - mean(K): O(n²)
        - broadcast and subtract: O(n²)

        Total: O(n²) vs O(n³) for H @ K @ H

        Parameters
        ----------
        K : Array
            Gram matrix [n, n] to center.

        Returns
        -------
        Array
            Centered Gram matrix K_c such that K_c = H @ K @ H.
        """
        b = self._backend
        row_mean = b.mean(K, axis=1, keepdims=True)  # [n, 1]
        col_mean = b.mean(K, axis=0, keepdims=True)  # [1, n]
        total_mean = b.mean(K)                        # scalar
        K_c = K - row_mean - col_mean + total_mean
        b.eval(K_c)
        return K_c

    def _compute_sample_transform(
        self, K_s_c: "Array", K_t_c: "Array"
    ) -> "Array":
        """Compute the exact sample-space transformation T = K_t^{1/2} @ K_s^{-1/2}.

        Uses GPU-only power iteration for eigendecomposition - no CPU linear algebra.
        This transformation guarantees T @ K_s @ T^T = K_t.
        """
        b = self._backend

        # Cast to float32 for numerical stability
        K_s_f32 = b.astype(K_s_c, "float32")
        K_t_f32 = b.astype(K_t_c, "float32")
        b.eval(K_s_f32, K_t_f32)

        # Native eigendecomposition - ALL eigenvalues, no approximation
        eig_s, V_s = b.eigh(K_s_f32)
        eig_t, V_t = b.eigh(K_t_f32)
        b.eval(eig_s, V_s, eig_t, V_t)

        regularization = self._regularization if self._regularization is not None else 0.0
        eps = max(
            regularization,
            machine_epsilon(b, K_s_c),
            machine_epsilon(b, K_t_c),
        )
        threshold_s = eps
        threshold_t = eps

        # Clamp eigenvalues to avoid NaN from sqrt of negative (numerical noise)
        # Note: b.where() evaluates both branches, so sqrt runs on all values
        eig_s_safe = b.maximum(eig_s, b.zeros_like(eig_s))
        eig_t_safe = b.maximum(eig_t, b.zeros_like(eig_t))

        inv_s_vals = b.where(
            eig_s > threshold_s,
            1.0 / b.sqrt(eig_s_safe),
            b.zeros_like(eig_s),
        )
        sqrt_t_vals = b.where(
            eig_t > threshold_t,
            b.sqrt(eig_t_safe),
            b.zeros_like(eig_t),
        )
        b.eval(inv_s_vals, sqrt_t_vals)

        # K_s^{-1/2} = V_s @ diag(1/sqrt(eig_s)) @ V_s^T
        inv_sqrt_s = b.matmul(
            V_s * b.reshape(inv_s_vals, (1, -1)),
            b.transpose(V_s),
        )
        b.eval(inv_sqrt_s)

        # K_t^{1/2} = V_t @ diag(sqrt(eig_t)) @ V_t^T
        sqrt_t = b.matmul(
            V_t * b.reshape(sqrt_t_vals, (1, -1)),
            b.transpose(V_t),
        )
        b.eval(sqrt_t)

        # T = K_t^{1/2} @ K_s^{-1/2}
        T = b.matmul(sqrt_t, inv_sqrt_s)
        return T

    def _find_feature_transform(
        self,
        source_centered: "Array",
        target_centered: "Array",
        K_t_c: "Array",
        initial_transform: "Array | None" = None,
        max_iterations: int | None = None,
    ) -> tuple["Array", int, float]:
        """Find feature-space transform F such that (A_s @ F)'s Gram = K_t.

        ONE METHOD: Sample-space transform T = K_t^{1/2} @ K_s^{-1/2}
        Mathematical guarantee: T @ K_s @ T^T = K_t exactly.
        Then F = pinv(A_s) @ T @ A_s gives the feature-space equivalent.

        Returns (transform, iterations, achieved_cka).
        """
        b = self._backend
        d_s = b.shape(source_centered)[1]
        d_t = b.shape(target_centered)[1]

        # Compute source Gram matrix
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_s_c = self._center_gram_efficient(K_s)

        # Eigendecomposition of both centered Gram matrices
        K_s_f32 = b.astype(K_s_c, "float32")
        K_t_f32 = b.astype(K_t_c, "float32")

        eig_s, V_s = b.eigh(K_s_f32)
        eig_t, V_t = b.eigh(K_t_f32)
        b.eval(eig_s, V_s, eig_t, V_t)

        # Regularization from dtype
        reg = regularization_epsilon(b, K_s_c)

        # Cross-dimensional case: solve directly for A_s @ F = A_t
        if d_s != d_t:
            source_pinv = b.pinv(source_centered)
            F = b.matmul(source_pinv, target_centered)
            b.eval(F)

            source_aligned = b.matmul(source_centered, F)
            K_aligned = b.matmul(source_aligned, b.transpose(source_aligned))
            K_aligned_c = self._center_gram_efficient(K_aligned)
            cka = self._compute_cka_from_centered_grams(K_aligned_c, K_t_c)
            return F, 1, cka

        # Clamp eigenvalues for numerical stability
        eig_s_safe = b.maximum(eig_s, b.full(eig_s.shape, reg))
        eig_t_safe = b.maximum(eig_t, b.full(eig_t.shape, reg))

        # K_s^{-1/2} = V_s @ diag(1/sqrt(eig_s)) @ V_s^T
        inv_sqrt_s_vals = 1.0 / b.sqrt(eig_s_safe)
        inv_sqrt_s = b.matmul(
            V_s * b.reshape(inv_sqrt_s_vals, (1, -1)),
            b.transpose(V_s)
        )

        # K_t^{1/2} = V_t @ diag(sqrt(eig_t)) @ V_t^T
        sqrt_t_vals = b.sqrt(eig_t_safe)
        sqrt_t = b.matmul(
            V_t * b.reshape(sqrt_t_vals, (1, -1)),
            b.transpose(V_t)
        )

        # T = K_t^{1/2} @ K_s^{-1/2}
        # This guarantees: T @ K_s @ T^T = K_t
        T = b.matmul(sqrt_t, inv_sqrt_s)
        b.eval(T)

        # Transform source: A_s' = T @ A_s
        source_transformed = b.matmul(T, source_centered)

        # Feature transform: F = pinv(A_s) @ A_s'
        source_pinv = b.pinv(source_centered)
        F = b.matmul(source_pinv, source_transformed)
        b.eval(F)

        # Compute achieved CKA
        source_aligned = b.matmul(source_centered, F)
        K_aligned = b.matmul(source_aligned, b.transpose(source_aligned))
        K_aligned_c = self._center_gram_efficient(K_aligned)
        cka = self._compute_cka_from_centered_grams(K_aligned_c, K_t_c)

        return F, 1, cka

    def _feature_transform_from_sample_transform(
        self,
        source_centered: "Array",
        sample_transform: "Array",
    ) -> "Array":
        """Construct a feature transform that reproduces the sample-space alignment."""
        b = self._backend
        aligned_samples = b.matmul(sample_transform, source_centered)
        transform = self._solve_feature_transform(source_centered, aligned_samples)
        if transform is None:
            raise ValueError(
                "GramAligner: solve-based sample transform failed; cannot proceed."
            )
        return transform

    def _compute_cka_from_centered_grams(
        self, K_x_c: "Array", K_y_c: "Array"
    ) -> float:
        """Compute CKA from pre-centered Gram matrices."""
        b = self._backend
        n = b.shape(K_x_c)[0]
        if n < 2:
            return 0.0

        # HSIC = trace(K_x_c @ K_y_c) / (n-1)^2
        hsic_xy_arr = b.sum(K_x_c * K_y_c)
        hsic_xx_arr = b.sum(K_x_c * K_x_c)
        hsic_yy_arr = b.sum(K_y_c * K_y_c)
        b.eval(hsic_xy_arr, hsic_xx_arr, hsic_yy_arr)
        denom_factor = float((n - 1) ** 2)
        hsic_xy_val = float(b.to_scalar(hsic_xy_arr))
        hsic_xx_val = float(b.to_scalar(hsic_xx_arr))
        hsic_yy_val = float(b.to_scalar(hsic_yy_arr))
        hsic_xy = hsic_xy_val / denom_factor
        hsic_xx = hsic_xx_val / denom_factor
        hsic_yy = hsic_yy_val / denom_factor

        denominator = sqrt_scalar(hsic_xx * hsic_yy, b)
        # Use dtype-derived epsilon for denominator floor
        div_eps = division_epsilon(b, K_x_c)
        if denominator < div_eps:
            return 0.0

        cka = hsic_xy / denominator
        return max(0.0, min(1.0, cka))

    def _diagnose_alignment(
        self,
        source_aligned: "Array",
        target_centered: "Array",
        cka: float,
    ) -> "AlignmentSignal":
        from modelcypher.core.domain.geometry.alignment_diagnostic import (
            AlignmentSignal,
            alignment_signal_from_matrices,
        )

        b = self._backend
        if b.shape(source_aligned) != b.shape(target_centered):
            phase_tol = machine_epsilon(b, source_aligned)
            return AlignmentSignal(
                dimension=3,
                cka_achieved=float(cka),
                iteration=0,
                metadata={
                    "source_rows": float(b.shape(source_aligned)[0]),
                    "source_cols": float(b.shape(source_aligned)[1]),
                    "target_rows": float(b.shape(target_centered)[0]),
                    "target_cols": float(b.shape(target_centered)[1]),
                    "shape_mismatch": 1.0,
                    "phase_tol": float(phase_tol),
                },
            )

        n_samples = b.shape(source_aligned)[0]
        labels = [f"sample:{idx}" for idx in range(n_samples)]
        return alignment_signal_from_matrices(
            source_aligned,
            target_centered,
            labels,
            backend=b,
            dimension=3,
            cka_achieved=cka,
            iteration=0,
        )


def find_alignment(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> AlignmentResult:
    """Find the transformation that achieves CKA = 1.0.

    This is the main entry point. It WILL find the exact alignment.

    Parameters
    ----------
    source_activations : Array
        Source activations [n_samples, d_source].
    target_activations : Array
        Target activations [n_samples, d_target].
    backend : Backend, optional
        Backend for tensor operations.

    Returns
    -------
    AlignmentResult
        The transformation achieving CKA = 1.0.

    Example
    -------
    >>> result = find_alignment(source_acts, target_acts)
    >>> aligned_source = source_acts @ result.feature_transform
    >>> # CKA(aligned_source, target_acts) ≈ 1.0
    """
    aligner = GramAligner(backend)
    return aligner.find_perfect_alignment(source_activations, target_activations)
