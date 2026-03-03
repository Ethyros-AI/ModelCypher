# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for information bridge module.

Each test validates a mathematical invariant derived from first principles.
See docs/research/information_bridge_derivation.md for proofs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _div_eps(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _rbf_gram(points, backend, sigma=None):
    """Compute RBF Gram matrix using the CKA infrastructure."""
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix

    return rbf_gram_matrix(points, backend, sigma=sigma)


# ===========================================================================
# Test: C_ex = 0 for flat manifold
# ===========================================================================
# For N points in a d-dim linear subspace (no curvature):
# - spectral_entropy ~ ln(d) (d equal singular values)
# - ID ~ d (flat d-dim manifold)
# - C_ex = ln(d) - ln(d) = 0
#
# Derivation: Section 7.2-7.3 of information_bridge_derivation.md.


def test_curvature_excess_zero_for_flat_manifold(any_backend):
    """C_ex ~ 0 for points in a flat linear subspace."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_curvature_excess,
    )

    # A flat d-dimensional manifold: spectral_entropy = ln(d), ID = d
    d = 5
    spectral_entropy_nats = math.log(d)  # ln(5) ~ 1.609
    intrinsic_dim = float(d)

    c_ex = compute_curvature_excess(spectral_entropy_nats, intrinsic_dim)

    assert abs(c_ex) < 1e-10, (
        f"C_ex should be 0 for flat manifold. Got {c_ex:.6f}"
    )


# ===========================================================================
# Test: C_ex > 0 for curved manifold
# ===========================================================================
# For a 1D curve winding through high-dim space:
# - spectral_entropy > ln(1) = 0 (variance spreads across many dimensions)
# - ID ~ 1 (locally 1D)
# - C_ex = S_spec - ln(1) = S_spec > 0


def test_curvature_excess_positive_for_curved_manifold(any_backend):
    """C_ex > 0 when spectral_entropy > ln(ID)."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_curvature_excess,
    )

    # Curved: variance spread across 10 dimensions, but locally 2D
    spectral_entropy_nats = math.log(10)  # ln(10) ~ 2.303
    intrinsic_dim = 2.0  # locally 2D

    c_ex = compute_curvature_excess(spectral_entropy_nats, intrinsic_dim)
    expected = math.log(10) - math.log(2)  # ~ 1.609

    assert c_ex > 0, f"C_ex should be > 0 for curved manifold. Got {c_ex:.6f}"
    assert abs(c_ex - expected) < 1e-10, (
        f"C_ex should be ln(10)-ln(2)={expected:.4f}, got {c_ex:.4f}"
    )


# ===========================================================================
# Test: C_ex non-negative always
# ===========================================================================
# C_ex = S_spec - ln(ID) >= 0 because eff_rank >= ID by diff. geom.
# (Section 7.2 of derivation).


def test_curvature_excess_nonnegative(any_backend):
    """C_ex >= 0 for any valid inputs."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_curvature_excess,
    )

    # Various combinations
    test_cases = [
        (math.log(10), 10.0),  # flat: C_ex = 0
        (math.log(10), 5.0),  # curved: C_ex = ln(2)
        (math.log(10), 1.0),  # very curved: C_ex = ln(10)
        (math.log(3), 3.0),  # flat: C_ex = 0
        (math.log(20), 2.5),  # curved
    ]

    for s_spec, id_val in test_cases:
        c_ex = compute_curvature_excess(s_spec, id_val)
        assert c_ex >= -1e-10, (
            f"C_ex must be >= 0. Got {c_ex:.6f} for S_spec={s_spec:.4f}, ID={id_val:.1f}"
        )


# ===========================================================================
# Test: C_ex handles edge cases
# ===========================================================================
# ID <= 0 or S_spec < 0 should not crash.


def test_curvature_excess_edge_cases(any_backend):
    """C_ex handles degenerate inputs gracefully."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_curvature_excess,
    )

    # ID near zero: return 0 (degenerate, can't take ln(0))
    c_ex = compute_curvature_excess(1.0, 0.0)
    assert c_ex == 0.0

    # Negative ID: return 0
    c_ex = compute_curvature_excess(1.0, -1.0)
    assert c_ex == 0.0


# ===========================================================================
# Test: All-pairs MI matrix is symmetric
# ===========================================================================
# MI(i,j) = MI(j,i) by commutativity of the Hadamard product.


def test_all_pairs_mi_symmetric(any_backend):
    """L×L MI matrix is symmetric."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_all_pairs_renyi_mi,
    )

    backend = any_backend
    n = 15

    # Three "layers" with different point clouds
    layers = [
        backend.array([[float(i + 1), float(i * 2 + k)] for i in range(n)])
        for k in range(3)
    ]

    grams = [_rbf_gram(layer, backend) for layer in layers]
    mi_matrix = compute_all_pairs_renyi_mi(grams, backend)

    num_layers = len(mi_matrix)
    eps = _div_eps(backend)

    for i in range(num_layers):
        for j in range(num_layers):
            assert abs(mi_matrix[i][j] - mi_matrix[j][i]) < eps, (
                f"MI matrix not symmetric at ({i},{j}): "
                f"{mi_matrix[i][j]:.6f} vs {mi_matrix[j][i]:.6f}"
            )


# ===========================================================================
# Test: Input MI trajectory is non-negative
# ===========================================================================
# I₂(X₀, X_l) >= 0 for all l (Giraldo et al. 2014, subadditivity).


def test_input_mi_trajectory_nonnegative(any_backend):
    """I₂(X₀, X_l) >= 0 for all layers."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_input_mi_trajectory,
    )

    backend = any_backend
    n = 15

    # Simulate 4 "layers" with progressively transformed data
    layers = [
        backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)]),
        backend.array([[float(i + 2), float(i * 3 + 1)] for i in range(n)]),
        backend.array([[float(i * 2 + 1), float(i + 5)] for i in range(n)]),
        backend.array([[float(n - i), float(i * 4 + 2)] for i in range(n)]),
    ]

    grams = [_rbf_gram(layer, backend) for layer in layers]
    trajectory = compute_input_mi_trajectory(grams, backend)

    eps = _div_eps(backend)
    for l, mi_val in enumerate(trajectory):
        assert mi_val >= -eps, (
            f"I₂(X₀, X_{l}) must be >= 0. Got {mi_val:.6f}"
        )


# ===========================================================================
# Test: Fixed-sigma MI trajectory for linear contraction
# ===========================================================================
# For X → AX → A²X where A contracts (singular values < 1),
# MI should decrease monotonically. This is a sanity check on the
# DPI test setup, not a proof of DPI for matrix-based Rényi MI.
#
# Note: DPI is NOT proven for matrix-based Rényi MI (Section 8.2).
# This test verifies the SETUP works correctly for the empirical test.


def test_fixed_sigma_mi_decreases_for_contraction(any_backend):
    """MI decreases under linear contraction with fixed sigma."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_fixed_sigma_mi_trajectory,
    )

    backend = any_backend
    n = 20

    # X₀: starting point cloud
    x0 = backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)])

    # Contraction matrix: scale by 0.5 in both dimensions
    # Each application reduces distances by 2x, reducing MI with original
    x0_list = [[float(i + 1), float(i * 2 + 1)] for i in range(n)]
    x1_list = [[v * 0.5 for v in row] for row in x0_list]
    x2_list = [[v * 0.25 for v in row] for row in x0_list]

    x1 = backend.array(x1_list)
    x2 = backend.array(x2_list)

    layers = [x0, x1, x2]

    # Use sigma from the input layer
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix_with_sigma

    _, sigma_0 = rbf_gram_matrix_with_sigma(x0, backend)

    trajectory = compute_fixed_sigma_mi_trajectory(layers, backend, sigma_0)

    # MI should decrease: I₂(X₀, X₀) > I₂(X₀, X₁) > I₂(X₀, X₂)
    for l in range(len(trajectory) - 1):
        assert trajectory[l] >= trajectory[l + 1] - _div_eps(backend), (
            f"MI should decrease under contraction. "
            f"I₂(X₀, X_{l})={trajectory[l]:.4f} < I₂(X₀, X_{l+1})={trajectory[l+1]:.4f}"
        )
