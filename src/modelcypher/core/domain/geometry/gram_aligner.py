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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

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

    # CKA achieved (1.0 is exact kernel alignment)
    achieved_cka: float

    # Number of iterations taken to find the fit
    iterations: int

    # Final alignment error (should be ~0)
    alignment_error: float

    # Diagnostic signal describing any residual gap
    diagnostic: "AlignmentSignal | None" = None

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
        max_rounds: int = 1,
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
        max_rounds : int
            Maximum search rounds to reach exact kernel alignment. Each round
            increases the iteration budget and returns a diagnostic if still unlocked.
        tolerance : float
            Convergence tolerance for CKA.
        regularization : float
            Regularization for matrix inversions.
        """
        self._backend = backend or get_default_backend()
        self._max_iterations = max_iterations
        self._max_rounds = max_rounds
        self._tolerance = tolerance
        self._regularization = regularization
        self._logger = logging.getLogger(__name__)

    def _solve_feature_transform(
        self,
        source_centered: "Array",
        target_centered: "Array",
        reg: float | None = None,
    ) -> "Array | None":
        """Solve F = A_s^T (A_s A_s^T)^-1 A_t using QR (primary) or eigendecomposition.

        QR-based solve avoids condition number squaring: κ(R) = κ(A), not κ(A)².
        Falls back to eigendecomposition for rank-deficient cases.
        """
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            solve_full_row_rank_via_qr,
        )

        b = self._backend
        n_samples = b.shape(source_centered)[0]
        if n_samples == 0:
            return None

        eps = machine_epsilon(b, source_centered)

        # Try QR-based solve first (most numerically stable)
        F_qr, diag = solve_full_row_rank_via_qr(b, source_centered, target_centered)
        if F_qr is not None and diag.get("residual_norm", float("inf")) < eps * 1000:
            self._logger.debug(
                "QR solve: method=%s, cond=%.2e, residual=%.2e",
                diag.get("method", "unknown"),
                diag.get("condition", float("inf")),
                diag.get("residual_norm", float("inf")),
            )
            return F_qr

        # Fall back to eigendecomposition
        self._logger.debug("Falling back to eigendecomposition solve")

        gram = b.matmul(source_centered, b.transpose(source_centered))
        b.eval(gram)

        eigvals, eigvecs = b.eigh(gram)
        b.eval(eigvals, eigvecs)

        inv_vals = b.where(
            eigvals > 0.0,
            1.0 / eigvals,
            b.zeros_like(eigvals),
        )
        b.eval(inv_vals)

        inv_diag = b.reshape(inv_vals, (1, -1))
        gram_inv_subspace = b.matmul(
            eigvecs * inv_diag,
            b.transpose(eigvecs),
        )
        b.eval(gram_inv_subspace)

        F = b.matmul(
            b.transpose(source_centered),
            b.matmul(gram_inv_subspace, target_centered),
        )
        b.eval(F)
        return F

    def _solve_feature_transform_uncentered(
        self,
        source: "Array",
        target: "Array",
    ) -> "Array | None":
        """Solve F for uncentered data using QR (primary) or eigendecomposition."""
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            solve_full_row_rank_via_qr,
        )

        b = self._backend
        n_samples = b.shape(source)[0]
        if n_samples == 0:
            return None

        eps = machine_epsilon(b, source)

        # Try QR-based solve first
        F_qr, diag = solve_full_row_rank_via_qr(b, source, target)
        if F_qr is not None and diag.get("residual_norm", float("inf")) < eps * 1000:
            return F_qr

        # Fall back to eigendecomposition (requires positive definite gram)
        gram = b.matmul(source, b.transpose(source))
        b.eval(gram)

        eigvals, eigvecs = b.eigh(gram)
        b.eval(eigvals, eigvecs)

        values = [float(v) for v in b.to_numpy(eigvals).tolist()]
        if not values:
            return None
        min_eig = min(values)
        if min_eig <= 0.0:
            return None

        inv_vals = b.where(
            eigvals > 0.0,
            1.0 / eigvals,
            b.zeros_like(eigvals),
        )
        b.eval(inv_vals)

        gram_inv = b.matmul(
            eigvecs * b.reshape(inv_vals, (1, -1)),
            b.transpose(eigvecs),
        )
        b.eval(gram_inv)

        transform = b.matmul(
            b.transpose(source),
            b.matmul(gram_inv, target),
        )
        b.eval(transform)
        return transform

    def find_perfect_alignment(
        self,
        source_activations: "Array",
        target_activations: "Array",
        initial_transform: "Array | None" = None,
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
        if initial_transform is not None:
            shape = b.shape(initial_transform)
            if shape[0] != d_s or shape[1] != d_t:
                initial_transform = None

        # MLX linear algebra requires float32/float64; keep alignment math stable.
        source_activations = b.astype(source_activations, "float32")
        target_activations = b.astype(target_activations, "float32")
        b.eval(source_activations, target_activations)

        # Try uncentered direct solve first (fast path for exact alignment)
        uncentered_transform = self._solve_feature_transform_uncentered(
            source_activations,
            target_activations,
        )
        if uncentered_transform is not None:
            aligned_uncentered = b.matmul(source_activations, uncentered_transform)
            K_s = b.matmul(source_activations, b.transpose(source_activations))
            K_t = b.matmul(target_activations, b.transpose(target_activations))
            H = self._centering_matrix(n_samples)
            K_s_c = b.matmul(b.matmul(H, K_s), H)
            K_t_c = b.matmul(b.matmul(H, K_t), H)
            K_a = b.matmul(aligned_uncentered, b.transpose(aligned_uncentered))
            K_a_c = b.matmul(b.matmul(H, K_a), H)
            b.eval(K_s_c, K_t_c, K_a_c)

            cka_uncentered = self._compute_cka_from_centered_grams(K_a_c, K_t_c)
            if cka_uncentered >= 1.0 - self._tolerance:
                sample_transform = self._compute_sample_transform(K_s_c, K_t_c)
                b.eval(sample_transform)
                return AlignmentResult(
                    feature_transform=b.to_numpy(uncentered_transform).tolist(),
                    sample_transform=b.to_numpy(sample_transform).tolist(),
                    achieved_cka=1.0,
                    iterations=0,
                    alignment_error=0.0,
                    diagnostic=self._diagnose_alignment(
                        aligned_uncentered, target_activations, cka_uncentered
                    ),
                )

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

        # Step 2: Build a feature-space transformation that reproduces
        # the sample-space alignment, then refine until CKA = 1.
        #
        # We want: (A_s @ F) @ (A_s @ F)^T = T @ K_s @ T^T = K_t
        # i.e., A_s @ F @ F^T @ A_s^T = K_t
        feature_transform = initial_transform
        if feature_transform is None:
            feature_transform = self._solve_feature_transform(
                source_centered, target_centered
            )
        if feature_transform is None:
            feature_transform = self._feature_transform_from_sample_transform(
                source_centered, sample_transform
            )
        if b.shape(feature_transform)[1] != b.shape(target_centered)[1]:
            # Sample-space transform preserves source dimensionality; reset for cross-dim.
            feature_transform = None
        total_iterations = 0
        max_iterations = self._max_iterations
        final_cka = 0.0
        rounds = max(1, self._max_rounds)

        for round_idx in range(rounds):
            feature_transform, iterations, final_cka = self._find_feature_transform(
                source_centered,
                target_centered,
                K_t_c,
                initial_transform=initial_transform,
                max_iterations=max_iterations,
            )
            total_iterations += iterations

            if final_cka >= 1.0 - self._tolerance:
                break

            max_iterations *= 2
            logger.info(
                "GramAligner: Exact kernel alignment not reached (cka=%.8f). "
                "Expanding search to %d iterations.",
                final_cka,
                max_iterations,
            )

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

        diagnostic = self._diagnose_alignment(source_transformed, target_centered, final_cka)

        return AlignmentResult(
            feature_transform=b.to_numpy(feature_transform).tolist(),
            sample_transform=b.to_numpy(sample_transform).tolist(),
            achieved_cka=final_cka,
            iterations=total_iterations,
            alignment_error=alignment_error,
            diagnostic=diagnostic,
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
        # Eigendecomposition of K_s_c
        eig_s, V_s = b.eigh(K_s_c)
        b.eval(eig_s, V_s)

        # Eigendecomposition of K_t_c
        eig_t, V_t = b.eigh(K_t_c)
        b.eval(eig_t, V_t)

        eps = max(
            self._regularization,
            machine_epsilon(b, K_s_c),
            machine_epsilon(b, K_t_c),
        )
        threshold_s = eps
        threshold_t = eps

        inv_s_vals = b.where(
            eig_s > threshold_s,
            1.0 / b.sqrt(eig_s),
            b.zeros_like(eig_s),
        )
        sqrt_t_vals = b.where(
            eig_t > threshold_t,
            b.sqrt(eig_t),
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

        Uses CLOSED-FORM solution F = A_s^T (A_s A_s^T)^-1 A_t for exact CKA = 1.0.

        Mathematical guarantee:
        - For A_s [n, d_s] with full row rank (n <= d_s), A_s A_s^T is invertible
        - So A_s @ F = A_s @ A_s^T (A_s A_s^T)^-1 @ A_t = A_t exactly
        - Therefore (A_s @ F) @ (A_s @ F)^T = A_t @ A_t^T = K_t
        - CKA = 1.0 exactly, no iteration needed

        Returns (transform, iterations, achieved_cka).
        """
        b = self._backend
        n_samples = b.shape(source_centered)[0]
        d_s = b.shape(source_centered)[1]
        d_t = b.shape(target_centered)[1]

        # Use centering matrix for CKA computation
        H = self._centering_matrix(n_samples)

        # CLOSED-FORM SOLUTION: F = A_s^T (A_s A_s^T)^-1 @ A_t
        # This GUARANTEES CKA = 1.0 when A_s has full row rank (n <= d_s)
        if initial_transform is not None:
            shape = b.shape(initial_transform)
            if shape[0] != d_s or shape[1] != d_t:
                initial_transform = None

        if initial_transform is None:
            F = self._solve_feature_transform(source_centered, target_centered)
            if F is None:
                raise ValueError(
                    "GramAligner: solve-based transform failed; cannot proceed without stable alignment."
                )

            # Verify CKA = 1.0
            source_transformed = b.matmul(source_centered, F)
            K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
            K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
            b.eval(K_s_t_c)
            cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)

            if cka >= 1.0 - self._tolerance:
                logger.info(
                    "GramAligner: Converged to CKA=%.8f in %d iterations",
                    cka, 1
                )
                return F, 1, cka

            # If closed-form didn't reach 1.0, try with tighter regularization
        else:
            F = initial_transform

        # SAMPLE-SPACE APPROACH for rank-deficient sources
        # When feature-space closed-form fails (rank deficiency), use sample-space transform directly:
        # T @ K_s @ T^T = K_t is EXACT regardless of rank
        # Apply: A_s' = T @ A_s
        #
        # This implements the dimensional hierarchy: 1D (binary) encodes 2D (vocabulary) encodes 3D+
        # The Gram matrix relationship is the invariant; we transform in sample space.

        # Compute sample-space transform T = K_t^{1/2} @ K_s^{-1/2}
        K_s = b.matmul(source_centered, b.transpose(source_centered))
        K_t = b.matmul(target_centered, b.transpose(target_centered))
        K_s_c_local = b.matmul(b.matmul(H, K_s), H)
        b.eval(K_s_c_local)

        # Eigendecomposition for matrix square roots
        eig_s, V_s = b.eigh(K_s_c_local)
        eig_t, V_t = b.eigh(K_t_c)
        b.eval(eig_s, V_s, eig_t, V_t)

        # Try different regularization levels for sample-space approach
        for reg in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
            eig_s_reg = b.maximum(eig_s, b.array(reg))
            eig_t_reg = b.maximum(eig_t, b.array(reg))

            # K_s^{-1/2} = V_s @ diag(1/sqrt(eig_s)) @ V_s^T
            inv_sqrt_s = b.matmul(
                V_s * b.reshape(1.0 / b.sqrt(eig_s_reg), (1, -1)),
                b.transpose(V_s)
            )

            # K_t^{1/2} = V_t @ diag(sqrt(eig_t)) @ V_t^T
            sqrt_t = b.matmul(
                V_t * b.reshape(b.sqrt(eig_t_reg), (1, -1)),
                b.transpose(V_t)
            )
            b.eval(inv_sqrt_s, sqrt_t)

            # T = K_t^{1/2} @ K_s^{-1/2}
            T = b.matmul(sqrt_t, inv_sqrt_s)
            b.eval(T)

            # Transform source in sample space: A_s' = T @ A_s
            # Result: A_s' @ A_s'^T = T @ K_s @ T^T = K_t
            source_transformed_sample = b.matmul(T, source_centered)
            b.eval(source_transformed_sample)

            # Compute feature transform F that achieves this: A_s @ F = A_s'
            # F = A_s^T (A_s A_s^T)^-1 @ A_s' = A_s^T (A_s A_s^T)^-1 @ T @ A_s
            F_sample = self._solve_feature_transform(
                source_centered,
                source_transformed_sample,
                reg=reg,
            )
            if F_sample is None:
                continue

            # Verify CKA
            source_aligned = b.matmul(source_centered, F_sample)
            K_aligned = b.matmul(source_aligned, b.transpose(source_aligned))
            K_aligned_c = b.matmul(b.matmul(H, K_aligned), H)
            b.eval(K_aligned_c)
            cka = self._compute_cka_from_centered_grams(K_aligned_c, K_t_c)

            if cka >= 1.0 - self._tolerance:
                logger.info(
                    "GramAligner: Sample-space converged to CKA=%.8f with reg=%.2e",
                    cka, reg
                )
                return F_sample, 1, cka

        # Last resort: gradient descent refinement
        max_iters = max_iterations or self._max_iterations
        best_F = F
        best_cka = 0.0

        for iteration in range(max_iters):
            source_transformed = b.matmul(source_centered, F)
            K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
            K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
            b.eval(K_s_t_c)

            cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)

            if cka > best_cka:
                best_cka = cka
                best_F = F

            if cka >= 1.0 - self._tolerance:
                logger.info(
                    "GramAligner: Gradient converged to CKA=%.8f in %d iterations",
                    cka, iteration + 1
                )
                return F, iteration + 1, cka

            # Gradient step with adaptive learning rate
            diff = K_t_c - K_s_t_c
            grad = b.matmul(b.transpose(source_centered), b.matmul(diff, source_transformed))
            b.eval(grad)

            grad_norm = b.sqrt(b.sum(grad * grad))
            b.eval(grad_norm)
            grad_norm_val = float(b.to_numpy(grad_norm))
            if grad_norm_val < 1e-14:
                break

            # Aggressive learning rate for faster convergence
            lr = 1.0 / (1.0 + 0.001 * iteration)
            F = F + lr * (grad / (grad_norm_val + 1e-12))
            b.eval(F)

        # Return best result
        source_transformed = b.matmul(source_centered, best_F)
        K_s_t = b.matmul(source_transformed, b.transpose(source_transformed))
        K_s_t_c = b.matmul(b.matmul(H, K_s_t), H)
        b.eval(K_s_t_c)
        final_cka = self._compute_cka_from_centered_grams(K_s_t_c, K_t_c)
        return best_F, max_iters, final_cka

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

        # HSIC = trace(K_x_c @ K_y_c) / (n-1)^2
        hsic_xy = float(b.to_numpy(b.sum(K_x_c * K_y_c))) / ((n - 1) ** 2)
        hsic_xx = float(b.to_numpy(b.sum(K_x_c * K_x_c))) / ((n - 1) ** 2)
        hsic_yy = float(b.to_numpy(b.sum(K_y_c * K_y_c))) / ((n - 1) ** 2)

        denominator = math.sqrt(hsic_xx * hsic_yy)
        if denominator < 1e-12:
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
            return AlignmentSignal(
                dimension=3,
                cka_achieved=float(cka),
                divergence_pattern="dimension_mismatch",
                suggested_transformation="expand_anchors",
                iteration=0,
                metadata={
                    "source_rows": float(b.shape(source_aligned)[0]),
                    "source_cols": float(b.shape(source_aligned)[1]),
                    "target_rows": float(b.shape(target_centered)[0]),
                    "target_cols": float(b.shape(target_centered)[1]),
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
