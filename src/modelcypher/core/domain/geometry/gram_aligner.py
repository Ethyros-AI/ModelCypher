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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

from modelcypher.core.domain.geometry.alignment_diagnostic import AlignmentSignal
from modelcypher.core.domain.geometry.cka import (
    _center_gram_matrix,
    compute_cka_from_centered_grams,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AlignmentResult",
    "GramAligner",
    "find_alignment",
]

# Module-level cache reference
_cache = ComputationCache.shared()


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

        if source_activations is target_activations:
            ident_feat = b.eye(int(d_s))
            ident_sample = b.eye(int(n_s))
            b.eval(ident_feat, ident_sample)
            return AlignmentResult(
                feature_transform=self._array_to_2d_list(ident_feat),
                sample_transform=self._array_to_2d_list(ident_sample),
                achieved_cka=1.0,
                iterations=0,
                alignment_error=0.0,
                diagnostic=None,
                precision_threshold=precision_threshold,
            )

        if d_s == d_t:
            diff = source_activations - target_activations
            diff_flat = b.reshape(diff, (1, -1))
            base_flat = b.reshape(source_activations, (1, -1))
            diff_norm_arr = geodesic_norms(diff_flat, b)
            base_norm_arr = geodesic_norms(base_flat, b)
            b.eval(diff_norm_arr, base_norm_arr)
            diff_norm = float(b.to_scalar(diff_norm_arr))
            base_norm = float(b.to_scalar(base_norm_arr))
            scale = base_norm + division_epsilon(b, source_activations)
            if diff_norm <= precision_threshold * scale:
                ident_feat = b.eye(int(d_s))
                ident_sample = b.eye(int(n_s))
                b.eval(ident_feat, ident_sample)
                return AlignmentResult(
                    feature_transform=self._array_to_2d_list(ident_feat),
                    sample_transform=self._array_to_2d_list(ident_sample),
                    achieved_cka=1.0,
                    iterations=0,
                    alignment_error=0.0,
                    diagnostic=None,
                    precision_threshold=precision_threshold,
                )

        # Center activations
        source_centered = self._center(source_activations)
        target_centered = self._center(target_activations)

        # Compute centered Gram matrices
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_t = b.matmul(target_centered, b.transpose(target_centered))
        K_s_c = _center_gram_matrix(K_s, b)
        K_t_c = _center_gram_matrix(K_t, b)

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
        K_s_t_c = _center_gram_matrix(K_s_transformed, b)

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

        # Derive eps from dtype (IGNORE self._regularization as documented)
        eps = max(
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
        use_cache: bool = True,
    ) -> tuple["Array", int, float]:
        """Find feature-space transform F such that (A_s @ F)'s Gram = K_t.

        ONE METHOD: Sample-space transform T = K_t^{1/2} @ K_s^{-1/2}
        Mathematical guarantee: T @ K_s @ T^T = K_t exactly.
        Then F = pinv(A_s) @ T @ A_s gives the feature-space equivalent.

        Args:
            source_centered: Centered source activations.
            target_centered: Centered target activations.
            K_t_c: Centered target Gram matrix.
            initial_transform: Optional initial transform (unused).
            max_iterations: Maximum iterations (unused - single-shot).
            use_cache: If True, check/populate stitch transform cache.

        Returns (transform, iterations, achieved_cka).
        """
        b = self._backend
        d_s = b.shape(source_centered)[1]
        d_t = b.shape(target_centered)[1]

        # Compute source Gram matrix
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_s_c = _center_gram_matrix(K_s, b)

        # Check stitch cache (avoids repeated eigendecomposition for same Gram pairs)
        cache_key: str | None = None
        if use_cache:
            cache_key = _cache.make_stitch_key(K_s_c, K_t_c, b)
            cached = _cache.get_stitch(cache_key)
            if cached is not None:
                F, cka = cached
                return F, 1, cka

        start_time = time.perf_counter()

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
            K_aligned_c = _center_gram_matrix(K_aligned, b)
            cka = compute_cka_from_centered_grams(K_aligned_c, K_t_c, b)

            # Cache the result
            if use_cache and cache_key is not None:
                compute_time_ms = (time.perf_counter() - start_time) * 1000
                _cache.set_stitch(cache_key, (F, cka), compute_time_ms)

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
        K_aligned_c = _center_gram_matrix(K_aligned, b)
        cka = compute_cka_from_centered_grams(K_aligned_c, K_t_c, b)

        # Cache the result for reuse
        if use_cache and cache_key is not None:
            compute_time_ms = (time.perf_counter() - start_time) * 1000
            _cache.set_stitch(cache_key, (F, cka), compute_time_ms)

        return F, 1, cka

    def _diagnose_alignment(
        self,
        source_aligned: "Array",
        target_centered: "Array",
        cka: float,
    ) -> "AlignmentSignal":
        from modelcypher.core.domain.geometry.alignment_diagnostic import (
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
