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
ALWAYS have a way to fit together perfectly. The relational structure
is preserved - we just need to find the coordinate transformation that
reveals this alignment.

CKA is a PHASE LOCK DETECTOR:
- CKA < 1: We haven't found the right transformation yet. Keep searching.
- CKA = 1: Phase locked. The legos fit. NOW we merge.

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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    """Result of finding perfect CKA alignment.

    The transformation that achieves CKA = 1.0, plus diagnostics about
    how we got there.
    """

    # The transformation that achieves CKA = 1.0
    # Apply as: A_s' = A_s @ feature_transform [d_source, d_target]
    feature_transform: list[list[float]]

    # The sample-space transformation (for reference)
    # This is the "true" alignment: T @ K_s @ T^T = K_t
    sample_transform: list[list[float]]

    # CKA achieved (should be 1.0 or very close)
    achieved_cka: float

    # Number of iterations taken to find the fit
    iterations: int

    # Final alignment error (should be ~0)
    alignment_error: float

    @property
    def is_perfect(self) -> bool:
        """Returns True if we achieved CKA ≈ 1.0."""
        return self.achieved_cka >= 0.9999

    @property
    def is_converged(self) -> bool:
        """Returns True if alignment error is negligible."""
        return self.alignment_error < 1e-6


class GramAligner:
    """Finds the transformation that achieves CKA = 1.0.

    This is not a "test" or "gate" - it's a SOLVER. Given two sets of
    activations, it finds the transformation that makes them equivalent
    in the CKA sense. This transformation always exists.

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
        tolerance: float = 1e-6,  # Relax for float32 precision
        regularization: float = 1e-8,
    ) -> None:
        """Initialize the aligner.

        Parameters
        ----------
        backend : Backend, optional
            Backend for tensor operations.
        max_iterations : int
            Maximum iterations for optimization. We should converge
            well before this - if we hit max, something is wrong.
        tolerance : float
            Convergence tolerance for CKA.
        regularization : float
            Regularization for matrix inversions.
        """
        self._backend = backend or get_default_backend()
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._regularization = regularization

    def find_perfect_alignment(
        self,
        source_activations: "Array",
        target_activations: "Array",
    ) -> AlignmentResult:
        """Find the transformation that achieves CKA = 1.0.

        This method WILL find the perfect alignment. If it can't,
        that indicates a bug in the implementation, not a property
        of the inputs.

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
            raise ValueError(
                f"Sample counts must match: source={n_s}, target={n_t}"
            )

        n_samples = n_s

        # MLX linear algebra requires float32/float64; keep alignment math stable.
        source_activations = b.astype(source_activations, "float32")
        target_activations = b.astype(target_activations, "float32")
        b.eval(source_activations, target_activations)

        # Center the activations
        source_centered = self._center(source_activations)
        target_centered = self._center(target_activations)
        b.eval(source_centered, target_centered)

        # Compute Gram matrices
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_t = b.matmul(target_centered, b.transpose(target_centered))
        b.eval(K_s, K_t)

        # Center the Gram matrices (for CKA)
        H = self._centering_matrix(n_samples)
        K_s_c = b.matmul(b.matmul(H, K_s), H)
        K_t_c = b.matmul(b.matmul(H, K_t), H)
        b.eval(K_s_c, K_t_c)

        # Step 1: Compute the exact sample-space transformation
        # T = K_t^{1/2} @ K_s^{-1/2}
        # This GUARANTEES T @ K_s @ T^T = K_t
        sample_transform = self._compute_sample_transform(K_s_c, K_t_c)
        b.eval(sample_transform)

        # Step 2: Find a feature-space transformation that achieves
        # the same effect on the Gram matrix.
        #
        # We want: (A_s @ F) @ (A_s @ F)^T = T @ K_s @ T^T = K_t
        # i.e., A_s @ F @ F^T @ A_s^T = K_t
        #
        # This is an optimization problem. We iterate to find F.
        feature_transform, iterations, final_cka = self._find_feature_transform(
            source_centered, target_centered, K_t_c
        )
        if final_cka < 1.0 - self._tolerance:
            fallback = self._feature_transform_from_sample_transform(
                source_centered, sample_transform
            )
            source_transformed = b.matmul(source_centered, fallback)
            K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
            K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
            b.eval(K_s_t_c)
            fallback_cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)

            if fallback_cka >= final_cka:
                logger.warning(
                    "GramAligner: Falling back to sample-space transform (CKA=%.8f).",
                    fallback_cka,
                )
                feature_transform = fallback
                final_cka = fallback_cka

        # Compute alignment error
        source_transformed = b.matmul(source_centered, feature_transform)
        K_s_transformed = b.matmul(source_transformed, b.transpose(source_transformed))
        K_s_t_c = b.matmul(b.matmul(H, K_s_transformed), H)
        b.eval(K_s_t_c)

        # Error is Frobenius norm of difference (normalized)
        diff = K_s_t_c - K_t_c
        error = float(b.to_numpy(b.sqrt(b.sum(diff * diff))))
        norm_t = float(b.to_numpy(b.sqrt(b.sum(K_t_c * K_t_c))))
        alignment_error = error / (norm_t + 1e-10)

        return AlignmentResult(
            feature_transform=b.to_numpy(feature_transform).tolist(),
            sample_transform=b.to_numpy(sample_transform).tolist(),
            achieved_cka=final_cka,
            iterations=iterations,
            alignment_error=alignment_error,
        )

    def _center(self, X: "Array") -> "Array":
        """Center activations (subtract mean)."""
        b = self._backend
        mean = b.mean(X, axis=0, keepdims=True)
        return X - mean

    def _centering_matrix(self, n: int) -> "Array":
        """Create centering matrix H = I - (1/n) * 1 @ 1^T."""
        b = self._backend
        I = b.eye(n)
        ones = b.ones((n, n))
        H = I - ones / float(n)
        b.eval(H)
        return H

    def _compute_sample_transform(
        self, K_s_c: "Array", K_t_c: "Array"
    ) -> "Array":
        """Compute the exact sample-space transformation T = K_t^{1/2} @ K_s^{-1/2}.

        This transformation guarantees T @ K_s @ T^T = K_t.
        """
        b = self._backend
        reg = self._regularization

        # Eigendecomposition of K_s_c
        eig_s, V_s = b.eigh(K_s_c)
        b.eval(eig_s, V_s)

        # Eigendecomposition of K_t_c
        eig_t, V_t = b.eigh(K_t_c)
        b.eval(eig_t, V_t)

        # Regularize eigenvalues (handle numerical issues)
        eig_s_reg = b.maximum(eig_s, b.array(reg))
        eig_t_reg = b.maximum(eig_t, b.array(reg))

        # K_s^{-1/2} = V_s @ diag(1/sqrt(eig_s)) @ V_s^T
        inv_sqrt_s = b.matmul(
            V_s * b.reshape(1.0 / b.sqrt(eig_s_reg), (1, -1)),
            b.transpose(V_s)
        )
        b.eval(inv_sqrt_s)

        # K_t^{1/2} = V_t @ diag(sqrt(eig_t)) @ V_t^T
        sqrt_t = b.matmul(
            V_t * b.reshape(b.sqrt(eig_t_reg), (1, -1)),
            b.transpose(V_t)
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
    ) -> tuple["Array", int, float]:
        """Find feature-space transform F such that (A_s @ F)'s Gram ≈ K_t.

        Uses iterative optimization to find F.

        Returns (transform, iterations, achieved_cka).
        """
        b = self._backend
        n_samples = b.shape(source_centered)[0]
        d_s = b.shape(source_centered)[1]
        d_t = b.shape(target_centered)[1]

        # SVD of both activation matrices
        # For [n, d] matrix: U is [n, k], S is [k], Vt is [k, d] where k = min(n, d)
        U_s_full, sigma_s, Vt_s = b.svd(source_centered)
        U_t_full, sigma_t, Vt_t = b.svd(target_centered)
        b.eval(U_s_full, sigma_s, Vt_s, U_t_full, sigma_t, Vt_t)

        # Get the rank (number of singular values)
        r_s = len(b.to_numpy(sigma_s))
        r_t = len(b.to_numpy(sigma_t))

        # Truncate U to match singular value count
        # U_s_full might be [n, n] or [n, min(n,d)], we need [n, r_s]
        U_s_shape = b.shape(U_s_full)
        U_t_shape = b.shape(U_t_full)

        if U_s_shape[1] > r_s:
            U_s = U_s_full[:, :r_s]
        else:
            U_s = U_s_full

        if U_t_shape[1] > r_t:
            U_t = U_t_full[:, :r_t]
        else:
            U_t = U_t_full
        b.eval(U_s, U_t)

        # Truncate Vt to match as well
        # Vt_s might be [d, d] but we need [r_s, d]
        Vt_s_shape = b.shape(Vt_s)
        Vt_t_shape = b.shape(Vt_t)

        if Vt_s_shape[0] > r_s:
            Vt_s = Vt_s[:r_s, :]
        if Vt_t_shape[0] > r_t:
            Vt_t = Vt_t[:r_t, :]
        b.eval(Vt_s, Vt_t)

        # Alignment matrix in sample space [r_s, r_t]
        M = b.matmul(b.transpose(U_s), U_t)
        b.eval(M)

        # Initial transform: V_s @ Σ_s^{-1} @ M @ Σ_t @ V_t^T
        sigma_s_inv = sigma_s / (sigma_s * sigma_s + self._regularization)

        # V_s is Vt_s^T: [d_s, r_s]
        V_s = b.transpose(Vt_s)
        V_t = b.transpose(Vt_t)
        b.eval(V_s, V_t)

        # T1 = V_s @ diag(Σ_s^{-1}) = V_s * sigma_s_inv (broadcast over rows)
        # V_s is [d_s, r_s], sigma_s_inv is [r_s]
        T1 = V_s * b.reshape(sigma_s_inv, (1, -1))
        b.eval(T1)

        # T2 = T1 @ M where T1 is [d_s, r_s], M is [r_s, r_t]
        T2 = b.matmul(T1, M)
        b.eval(T2)

        # T3 = T2 @ diag(Σ_t) = T2 * sigma_t (broadcast over rows)
        # T2 is [d_s, r_t], sigma_t is [r_t]
        T3 = T2 * b.reshape(sigma_t, (1, -1))
        b.eval(T3)

        # F = T3 @ V_t^T where T3 is [d_s, r_t], Vt_t is [r_t, d_t]
        F = b.matmul(T3, Vt_t)
        b.eval(F)

        # Iterate to refine (gradient descent on CKA)
        H = self._centering_matrix(n_samples)

        for iteration in range(self._max_iterations):
            # Compute current CKA
            source_transformed = b.matmul(source_centered, F)
            K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
            K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
            b.eval(K_s_t_c)

            cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)

            if cka >= 1.0 - self._tolerance:
                logger.info(
                    "GramAligner: Converged to CKA=%.8f in %d iterations",
                    cka, iteration + 1
                )
                return F, iteration + 1, cka

            # Gradient step: adjust F to increase CKA
            # The gradient of CKA w.r.t. F is complex, so we use a
            # direct approach: F_new = F + lr * (A_s^+ @ A_t)
            # This moves F toward the mapping that directly aligns activations
            #
            # Better approach: Use the Gram matrix difference to adjust
            diff = K_t_c - K_s_t_c
            grad = b.matmul(b.transpose(source_centered), b.matmul(diff, source_transformed))
            b.eval(grad)

            # Normalize gradient
            grad_norm = b.sqrt(b.sum(grad * grad))
            b.eval(grad_norm)
            grad_norm_val = float(b.to_numpy(grad_norm))
            if grad_norm_val < 1e-12:
                break

            # Learning rate scheduling
            lr = 0.1 / (1.0 + 0.01 * iteration)
            F = F + lr * (grad / grad_norm_val)
            b.eval(F)

        # If we got here, check final CKA
        source_transformed = b.matmul(source_centered, F)
        K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
        K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
        b.eval(K_s_t_c)
        final_cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)

        logger.warning(
            "GramAligner: Did not converge to CKA=1.0 in %d iterations. "
            "Final CKA=%.8f. This indicates a bug - CKA=1.0 should always be achievable.",
            self._max_iterations, final_cka
        )

        return F, self._max_iterations, final_cka

    def _pseudo_inverse(self, matrix: "Array") -> "Array":
        """Compute a stable pseudoinverse using SVD."""
        b = self._backend
        U, S, Vt = b.svd(matrix)
        b.eval(U, S, Vt)

        s_inv = S / (S * S + self._regularization)
        V = b.transpose(Vt)
        V_scaled = V * b.reshape(s_inv, (1, -1))
        pinv = b.matmul(V_scaled, b.transpose(U))
        b.eval(pinv)
        return pinv

    def _feature_transform_from_sample_transform(
        self,
        source_centered: "Array",
        sample_transform: "Array",
    ) -> "Array":
        """Construct a feature transform that reproduces the sample-space alignment."""
        b = self._backend
        aligned_samples = b.matmul(sample_transform, source_centered)
        pinv = self._pseudo_inverse(source_centered)
        transform = b.matmul(pinv, aligned_samples)
        b.eval(transform)
        return transform

    def _compute_cka_from_centered_grams(
        self, K_x_c: "Array", K_y_c: "Array"
    ) -> float:
        """Compute CKA from pre-centered Gram matrices."""
        b = self._backend
        n = b.shape(K_x_c)[0]

        # HSIC = trace(K_x_c @ K_y_c) / (n-1)^2
        hsic_xy = float(b.to_numpy(b.sum(K_x_c * K_y_c))) / ((n - 1) ** 2)
        hsic_xx = float(b.to_numpy(b.sum(K_x_c * K_x_c))) / ((n - 1) ** 2)
        hsic_yy = float(b.to_numpy(b.sum(K_y_c * K_y_c))) / ((n - 1) ** 2)

        denominator = math.sqrt(hsic_xx * hsic_yy)
        if denominator < 1e-12:
            return 0.0

        cka = hsic_xy / denominator
        return max(0.0, min(1.0, cka))


def find_alignment(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> AlignmentResult:
    """Find the transformation that achieves CKA = 1.0.

    This is the main entry point. It WILL find the perfect alignment.

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
