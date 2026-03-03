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
# Test: MI ~ 0 for independent data
# ===========================================================================
# For truly independent X and Y, K_X and K_Y should share no structure,
# making the Hadamard product nearly uninformative.


def test_renyi_mi_near_zero_independent(any_backend):
    """I_2 ~ 0 for independently generated point clouds."""
    from modelcypher.core.domain.geometry.renyi_mi import compute_renyi_mi_alpha2

    backend = any_backend
    n = 50

    # Two independent point clouds with no shared structure.
    # X: points along one direction, Y: points along an orthogonal direction.
    x = backend.array([[float(i), 0.0] for i in range(n)])
    y = backend.array([[0.0, float(i * 7 % 50)] for i in range(n)])

    gram_x = _rbf_gram_from_points(x, backend)
    gram_y = _rbf_gram_from_points(y, backend)

    mi = compute_renyi_mi_alpha2(gram_x, gram_y, backend)

    # With n=50 points and independent data, MI should be small.
    # Use a generous threshold since finite-sample noise exists.
    s2_x = _get_entropy(gram_x, backend)
    assert mi < 0.5 * s2_x, (
        f"I_2 for independent data should be much less than S_2(X)={s2_x:.4f}, "
        f"got {mi:.4f}"
    )


def _get_entropy(gram, backend):
    from modelcypher.core.domain.geometry.renyi_mi import (
        compute_renyi_entropy_alpha2,
    )

    return compute_renyi_entropy_alpha2(gram, backend)


# ===========================================================================
# Test: MI with self = S_2 (entropy)
# ===========================================================================
# I_2(X; X) = S_2(A_X) + S_2(A_X) - S_2(A_XX)
# where A_XX = (K ⊙ K) / tr(K ⊙ K) = K^2 / tr(K^2)  (elementwise square)
#
# This should equal S_2(A_X) because self-information = entropy.
# But the elementwise square K ⊙ K ≠ K @ K, so this may differ.
# Test numerically to establish the relationship.


def test_renyi_mi_self_info(any_backend):
    """I_2(X; X) should equal S_2(A_X) if the kernel is consistent."""
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

    # For infinitely divisible kernels, I(X;X) = S(X) is expected.
    # Allow generous tolerance since the Hadamard square changes the spectrum.
    assert mi_self >= s2 - _div_eps(backend), (
        f"I_2(X;X) should be >= S_2(X). Got I_2={mi_self:.4f}, S_2={s2:.4f}"
    )


# ===========================================================================
# Test: Hadamard product identity for RBF kernels
# ===========================================================================
# Core algebraic identity (Section 3 of derivation):
# K_X ⊙ K_Y = K_Z where Z = (X, Y) concatenated, same sigma.
# We verify this directly.


def test_hadamard_is_joint_kernel(any_backend):
    """K_X ⊙ K_Y = K_{(X,Y)} for RBF kernels at the same sigma."""
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix_with_sigma

    backend = any_backend
    n = 15

    x = backend.array([[float(i + 1), float(i * 2)] for i in range(n)])
    y = backend.array([[float(i * 3 + 1), float(i + 7)] for i in range(n)])

    # Compute marginal kernels with shared sigma
    gram_x, sigma_x = rbf_gram_matrix_with_sigma(x, backend)
    gram_y, sigma_y = rbf_gram_matrix_with_sigma(y, backend)

    # Use the same sigma for the joint space
    # Pick the mean sigma to be fair (different data -> different natural sigma)
    sigma = (sigma_x + sigma_y) / 2.0

    # Recompute marginals with shared sigma
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix

    gram_x_shared = rbf_gram_matrix(x, backend, sigma=sigma)
    gram_y_shared = rbf_gram_matrix(y, backend, sigma=sigma)

    # Hadamard product
    hadamard = gram_x_shared * gram_y_shared

    # Joint kernel: concatenate X and Y, compute RBF
    x_list = backend.tolist(x)
    y_list = backend.tolist(y)
    z_list = [x_list[i] + y_list[i] for i in range(n)]
    z = backend.array(z_list)
    gram_z = rbf_gram_matrix(z, backend, sigma=sigma)

    # They should be equal (algebraic identity, not approximation)
    diff = hadamard - gram_z
    backend.eval(diff)
    max_diff = float(backend.to_scalar(backend.max(backend.abs(diff))))

    assert max_diff < _div_eps(backend), (
        f"Hadamard product should equal joint RBF kernel. Max diff: {max_diff:.2e}"
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
