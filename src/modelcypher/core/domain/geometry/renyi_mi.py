# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Matrix-based Renyi alpha=2 mutual information from kernel matrices.

Computes resolution-dependent mutual information between point clouds using
the Hadamard product of RBF Gram matrices. Every formula is derived from
algebra — no arbitrary constants, no heuristics.

Mathematical foundations (see docs/research/information_bridge_derivation.md):

1. Shannon MI is +infinity for deterministic continuous maps (Goldfeld et al. 2019).
   Kernel bandwidth sigma creates a finite measurement resolution.

2. The Hadamard product K_X * K_Y defines a valid PSD product kernel (Schur 1911).
   For geodesic RBF kernels (used in ModelCypher), this is NOT the RBF kernel on
   the concatenated space (Pythagorean decomposition fails for geodesic distances),
   but the product kernel is still valid for MI computation. (Section 3 of derivation.)

3. The Gaussian RBF kernel is infinitely divisible, and the Hadamard product of
   infinitely divisible kernels is infinitely divisible, satisfying the requirements
   for Giraldo et al.'s (2014) matrix-based Renyi entropy axioms.

References:
    Giraldo, Rao, Principe (2014). "Measures of entropy from data using
        infinitely divisible kernels." IEEE Trans. Info Theory. arXiv:1211.2459.
    Yu, Giraldo, Jenssen, Principe (2019). "Multivariate Extension of Matrix-based
        Renyi's alpha-order Entropy Functional." arXiv:1808.07912.
    Schur (1911). Hadamard product of PSD matrices is PSD (Theorem VII).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def compute_renyi_entropy_alpha2(gram: "Array", backend: "Backend") -> float:
    """Compute matrix-based Renyi alpha=2 entropy from a kernel Gram matrix.

    S_2(A) = -log_2(tr(A^2)) = -log_2(||A||_F^2)

    where A = K / tr(K) is the normalized kernel matrix.

    Derivation of bounds (Section 4.4 of information_bridge_derivation.md):
        0 <= S_2 <= log_2(N)
        S_2 = 0 iff rank(K) = 1 (all points identical at resolution sigma)
        S_2 = log_2(N) iff K = I (all points maximally distinguishable)

    Args:
        gram: [N, N] positive semi-definite kernel Gram matrix from an
              infinitely divisible kernel (RBF satisfies this).
        backend: Backend for tensor operations.

    Returns:
        Renyi alpha=2 entropy in bits (log base 2).
    """
    # Normalize: A = K / tr(K)
    trace_k = backend.trace(gram)
    backend.eval(trace_k)
    trace_val = float(backend.to_scalar(trace_k))

    eps = division_epsilon(backend, gram)
    if trace_val < eps:
        return 0.0

    a = gram / trace_k

    # tr(A^2) = ||A||_F^2 = sum of all squared elements
    a_sq = a * a  # elementwise square
    frob_sq = backend.sum(a_sq)
    backend.eval(frob_sq)
    frob_sq_val = float(backend.to_scalar(frob_sq))

    # Guard: frob_sq must be in (0, 1] for valid entropy
    if frob_sq_val <= 0.0:
        return 0.0

    # S_2 = -log_2(||A||_F^2)
    return -math.log2(frob_sq_val)


def compute_renyi_joint_entropy_alpha2(
    gram_x: "Array", gram_y: "Array", backend: "Backend"
) -> float:
    """Compute joint Renyi alpha=2 entropy via Hadamard product.

    S_2(A_XY) = -log_2(||A_XY||_F^2)

    where A_XY = (K_X * K_Y) / tr(K_X * K_Y) and * is the Hadamard
    (elementwise) product.

    The Hadamard product K_X * K_Y defines the product kernel (tensor product
    kernel in Shawe-Taylor & Cristianini Ch. 3). Guaranteed PSD by Schur's
    theorem (1911). For geodesic RBF kernels, this is NOT the RBF kernel on
    the concatenated space, but it is a valid PSD kernel from infinitely
    divisible components, satisfying the Giraldo axioms for MI computation.

    Args:
        gram_x: [N, N] kernel Gram matrix for X.
        gram_y: [N, N] kernel Gram matrix for Y. Must have same N.
        backend: Backend for tensor operations.

    Returns:
        Joint Renyi alpha=2 entropy in bits.
    """
    # Hadamard product = product kernel Gram matrix (PSD by Schur)
    hadamard = gram_x * gram_y

    # Normalize: A_XY = hadamard / tr(hadamard)
    trace_h = backend.trace(hadamard)
    backend.eval(trace_h)
    trace_val = float(backend.to_scalar(trace_h))

    eps = division_epsilon(backend, gram_x)
    if trace_val < eps:
        return 0.0

    a_xy = hadamard / trace_h

    # ||A_XY||_F^2
    a_xy_sq = a_xy * a_xy
    frob_sq = backend.sum(a_xy_sq)
    backend.eval(frob_sq)
    frob_sq_val = float(backend.to_scalar(frob_sq))

    if frob_sq_val <= 0.0:
        return 0.0

    return -math.log2(frob_sq_val)


def compute_renyi_mi_alpha2(
    gram_x: "Array", gram_y: "Array", backend: "Backend"
) -> float:
    """Compute matrix-based Renyi alpha=2 mutual information.

    I_2(X; Y) = S_2(A_X) + S_2(A_Y) - S_2(A_XY)

    Equivalently:
        I_2 = log_2(||A_XY||_F^2 / (||A_X||_F^2 * ||A_Y||_F^2))

    Properties (derived, not assumed):
        I_2 >= 0     (Giraldo et al. 2014, Theorem 3: subadditivity)
        I_2 = 0      iff X independent of Y (for characteristic kernels)
        I_2(X;Y) = I_2(Y;X)  (Hadamard product is commutative)

    Args:
        gram_x: [N, N] kernel Gram matrix for X.
        gram_y: [N, N] kernel Gram matrix for Y.
        backend: Backend for tensor operations.

    Returns:
        Renyi alpha=2 MI in bits. Non-negative by construction.
    """
    s2_x = compute_renyi_entropy_alpha2(gram_x, backend)
    s2_y = compute_renyi_entropy_alpha2(gram_y, backend)
    s2_xy = compute_renyi_joint_entropy_alpha2(gram_x, gram_y, backend)

    mi = s2_x + s2_y - s2_xy

    # Clamp to zero: finite-precision arithmetic can produce tiny negatives
    return max(0.0, mi)
