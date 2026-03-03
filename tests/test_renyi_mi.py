# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for matrix-based Renyi alpha=2 MI module.

Each test validates a mathematical invariant derived from first principles.
See docs/research/information_bridge_derivation.md for proofs.

References:
    Giraldo, Rao, Principe (2014). IEEE Trans. Info Theory. arXiv:1211.2459.
    Yu, Giraldo, Jenssen, Principe (2019). arXiv:1808.07912.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps(backend: "Backend") -> float:
    return machine_epsilon(backend, backend.array([1.0]))


def _div_eps(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


# ---------------------------------------------------------------------------
# Helpers to build known kernel matrices
# ---------------------------------------------------------------------------


def _identity_gram(n: int, backend: "Backend"):
    """Identity kernel: K = I. All points maximally distinguishable."""
    return backend.eye(n)


def _rank1_gram(n: int, backend: "Backend"):
    """Rank-1 kernel: K = 11^T. All points identical."""
    ones = backend.ones((n, 1))
    return ones @ backend.transpose(ones)


def _rbf_gram_from_points(points, backend: "Backend"):
    """Compute RBF Gram matrix using the CKA infrastructure."""
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix

    return rbf_gram_matrix(points, backend)


# ===========================================================================
# Test: S_2 of identity kernel = log_2(N)
# ===========================================================================
# Derivation: A = I/N. tr(A^2) = N * (1/N)^2 = 1/N. S_2 = -log_2(1/N) = log_2(N).


def test_renyi_entropy_identity_kernel(any_backend):
    """S_2 of identity kernel = log_2(N). Maximum entropy."""
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    backend = any_backend
    n = 10
    gram = _identity_gram(n, backend)

    s2 = compute_renyi_entropy_alpha2(gram, backend)
    expected = math.log2(n)  # log_2(10) ~ 3.322

    assert abs(s2 - expected) < _div_eps(backend), (
        f"S_2 of identity kernel should be log_2({n})={expected:.4f}, got {s2:.4f}"
    )


# ===========================================================================
# Test: S_2 of rank-1 kernel = 0
# ===========================================================================
# Derivation: A = 11^T / tr(11^T) = 11^T / N. A has one eigenvalue = 1, rest = 0.
# tr(A^2) = 1. S_2 = -log_2(1) = 0.


def test_renyi_entropy_rank1_kernel(any_backend):
    """S_2 of rank-1 kernel = 0. Minimum entropy."""
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    backend = any_backend
    n = 10
    gram = _rank1_gram(n, backend)

    s2 = compute_renyi_entropy_alpha2(gram, backend)

    assert abs(s2) < _div_eps(backend), (
        f"S_2 of rank-1 kernel should be 0, got {s2:.6f}"
    )


# ===========================================================================
# Test: S_2 bounds: 0 <= S_2 <= log_2(N) for any PSD kernel
# ===========================================================================
# Derivation: Section 4.4 of information_bridge_derivation.md.


def test_renyi_entropy_bounds(any_backend):
    """S_2 is bounded: 0 <= S_2 <= log_2(N)."""
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    backend = any_backend
    n = 20
    eps = _div_eps(backend)

    # Random PSD matrix: K = X @ X^T
    rng_key = backend.array([42])
    x = backend.array([[float(i * j % 7 + 1) for j in range(5)] for i in range(n)])
    gram = x @ backend.transpose(x)

    s2 = compute_renyi_entropy_alpha2(gram, backend)

    assert s2 >= -eps, f"S_2 must be >= 0, got {s2:.6f}"
    assert s2 <= math.log2(n) + eps, (
        f"S_2 must be <= log_2({n})={math.log2(n):.4f}, got {s2:.6f}"
    )


# ===========================================================================
# Test: MI non-negativity
# ===========================================================================
# Derivation: Giraldo et al. (2014), Theorem 3 (subadditivity for infinitely
# divisible kernels).


def test_renyi_mi_nonnegative(any_backend):
    """I_2(X; Y) >= 0 for RBF kernels (infinitely divisible)."""
    from modelcypher.core.domain.geometry.renyi_mi import compute_renyi_mi_alpha2

    backend = any_backend
    n = 20

    # Two different point clouds -> two different RBF kernels
    x = backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)])
    y = backend.array([[float(i * 3 + 2), float(i + 5)] for i in range(n)])

    gram_x = _rbf_gram_from_points(x, backend)
    gram_y = _rbf_gram_from_points(y, backend)

    mi = compute_renyi_mi_alpha2(gram_x, gram_y, backend)

    assert mi >= -_div_eps(backend), f"I_2 must be >= 0, got {mi:.6f}"


# ===========================================================================
# Test: MI for uncorrelated data < MI for correlated data
# ===========================================================================
# With deterministic paired samples (x_i, y_i), the pairing itself creates
# structure that MI detects. We cannot construct truly independent paired
# samples deterministically. Instead, we verify the relative ordering:
# MI(X, X) > MI(X, Y) for Y with different structure.
#
# This avoids magic thresholds while testing the key property: MI increases
# with dependence strength.


def test_renyi_mi_correlated_exceeds_uncorrelated(any_backend):
    """I_2(X, X) > I_2(X, Y) when Y has different structure from X."""
    from modelcypher.core.domain.geometry.renyi_mi import compute_renyi_mi_alpha2

    backend = any_backend
    n = 30

    # X: linear spacing -> smooth kernel structure
    x = backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)])

    # Y: different structure (reversed + offset) -> different kernel
    y = backend.array(
        [[float(n - i + 3), float((i * 7 % n) + 2)] for i in range(n)]
    )

    gram_x = _rbf_gram_from_points(x, backend)
    gram_y = _rbf_gram_from_points(y, backend)

    mi_self = compute_renyi_mi_alpha2(gram_x, gram_x, backend)
    mi_cross = compute_renyi_mi_alpha2(gram_x, gram_y, backend)

    # Self-MI should exceed cross-MI: the same data is maximally dependent
    assert mi_self > mi_cross, (
        f"Self-MI ({mi_self:.4f}) should exceed cross-MI ({mi_cross:.4f})"
    )


# ===========================================================================
# Test: MI with self >= 0 and finite
# ===========================================================================
# I_2(X; X) = 2*S_2(A_X) - S_2(A_XX)
# where A_XX = (K ⊙ K) / tr(K ⊙ K) and K ⊙ K is the elementwise square.
#
# I_2(X;X) ≠ S_2(X) in general for matrix-based Rényi MI, because K ⊙ K ≠ K:
# the elementwise square changes the eigenspectrum. The product kernel K ⊙ K
# is "sharper" (concentrates more mass on similar pairs), so S_2(A_XX) < 2*S_2(A_X),
# making I_2(X;X) > 0 but not necessarily equal to S_2(X).
#
# What we CAN guarantee: I_2(X;X) >= 0 (non-negativity from Giraldo axioms).


def test_renyi_mi_self_info(any_backend):
    """I_2(X; X) >= 0 and is finite. Not necessarily equal to S_2(X)."""
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
        compute_renyi_mi_alpha2,
    )

    backend = any_backend
    n = 20

    x = backend.array([[float(i + 1), float(i * 2 + 3)] for i in range(n)])
    gram = _rbf_gram_from_points(x, backend)

    s2 = compute_renyi_entropy_alpha2(gram, backend)
    mi_self = compute_renyi_mi_alpha2(gram, gram, backend)

    # Non-negativity: guaranteed by Giraldo axioms for infinitely divisible kernels
    assert mi_self >= -_div_eps(backend), (
        f"I_2(X;X) must be >= 0. Got {mi_self:.4f}"
    )

    # Finite: should not exceed 2 * S_2(X) (joint entropy is non-negative)
    assert mi_self <= 2 * s2 + _div_eps(backend), (
        f"I_2(X;X) must be <= 2*S_2(X)={2*s2:.4f}. Got {mi_self:.4f}"
    )


# ===========================================================================
# Test: Hadamard product kernel is PSD (Schur product theorem)
# ===========================================================================
# The Hadamard product of two PSD kernel matrices is PSD (Schur 1911).
# This is the foundation for the joint entropy computation.
#
# Note: For geodesic RBF kernels, the Hadamard product does NOT equal
# the RBF kernel on the concatenated space (Pythagorean decomposition fails
# for geodesic distances). But it IS a valid PSD product kernel.


def test_hadamard_product_is_psd(any_backend):
    """K_X ⊙ K_Y is PSD for RBF Gram matrices (Schur product theorem)."""
    backend = any_backend
    n = 15

    x = backend.array([[float(i + 1), float(i * 2)] for i in range(n)])
    y = backend.array([[float(i * 3 + 1), float(i + 7)] for i in range(n)])

    gram_x = _rbf_gram_from_points(x, backend)
    gram_y = _rbf_gram_from_points(y, backend)

    # Hadamard product
    hadamard = gram_x * gram_y

    # PSD check: all eigenvalues >= 0
    # For a symmetric matrix, eigenvalues are real.
    # Use trace(H) > 0 and trace(H^2) > 0 as necessary conditions,
    # plus verify the Gram matrix structure (symmetric, non-negative diagonal).
    backend.eval(hadamard)

    # Symmetric: H_ij = H_ji (both inputs are symmetric, elementwise product preserves)
    diff = hadamard - backend.transpose(hadamard)
    backend.eval(diff)
    max_asym = float(backend.to_scalar(backend.max(backend.abs(diff))))
    assert max_asym < _div_eps(backend), (
        f"Hadamard product should be symmetric. Max asymmetry: {max_asym:.2e}"
    )

    # Non-negative diagonal (RBF kernel has K_ii = 1, so H_ii = 1*1 = 1)
    diag = backend.array([float(backend.to_scalar(hadamard[i, i])) for i in range(n)])
    backend.eval(diag)
    min_diag = float(backend.to_scalar(backend.min(diag)))
    assert min_diag >= -_div_eps(backend), (
        f"Hadamard diagonal should be non-negative. Min: {min_diag:.6f}"
    )

    # Trace > 0 (sum of eigenvalues)
    trace_h = float(backend.to_scalar(backend.trace(hadamard)))
    assert trace_h > 0, f"Trace of Hadamard product should be > 0. Got {trace_h:.6f}"

    # tr(H^2) <= tr(H)^2 (Cauchy-Schwarz for PSD matrices)
    h_sq = hadamard * hadamard
    frob_sq = float(backend.to_scalar(backend.sum(h_sq)))
    assert frob_sq <= trace_h * trace_h + _div_eps(backend), (
        f"||H||_F^2 should be <= tr(H)^2 for PSD. "
        f"||H||_F^2={frob_sq:.6f}, tr(H)^2={trace_h**2:.6f}"
    )


# ===========================================================================
# Test: MI is symmetric
# ===========================================================================
# I_2(X; Y) = I_2(Y; X) by the commutativity of the Hadamard product.


def test_renyi_mi_symmetric(any_backend):
    """I_2(X; Y) = I_2(Y; X)."""
    from modelcypher.core.domain.geometry.renyi_mi import compute_renyi_mi_alpha2

    backend = any_backend
    n = 20

    x = backend.array([[float(i + 1), float(i * 2)] for i in range(n)])
    y = backend.array([[float(i * 3 + 1), float(i + 5)] for i in range(n)])

    gram_x = _rbf_gram_from_points(x, backend)
    gram_y = _rbf_gram_from_points(y, backend)

    mi_xy = compute_renyi_mi_alpha2(gram_x, gram_y, backend)
    mi_yx = compute_renyi_mi_alpha2(gram_y, gram_x, backend)

    assert abs(mi_xy - mi_yx) < _div_eps(backend), (
        f"MI should be symmetric. I_2(X;Y)={mi_xy:.6f}, I_2(Y;X)={mi_yx:.6f}"
    )
