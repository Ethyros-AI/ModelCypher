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


# ===========================================================================
# Test: Normalized MI trajectory is non-negative
# ===========================================================================
# I₂(X₀, X_l) >= 0 for all l after L2 normalization + shared sigma.
# Same invariant as per-layer sigma (Giraldo et al. 2014, subadditivity).
# Tests with layers at wildly different scales to verify normalization works.


def test_normalized_mi_trajectory_nonnegative(any_backend):
    """I₂(X₀, X_l) >= 0 with L2 norm + shared sigma, even at different scales."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_mi_trajectory,
    )

    backend = any_backend
    n = 15

    # 4 layers at wildly different scales (1x, 10x, 100x, 1000x)
    base = [[float(i + 1), float(i * 2 + 1)] for i in range(n)]
    layers = [
        backend.array(base),
        backend.array([[v * 10.0 for v in row] for row in base]),
        backend.array([[v * 100.0 for v in row] for row in base]),
        backend.array([[v * 1000.0 for v in row] for row in base]),
    ]

    trajectory, sigma, _ = compute_normalized_mi_trajectory(layers, backend)

    assert sigma > 0.0, f"Shared sigma must be positive. Got {sigma}"

    eps = _div_eps(backend)
    for l, mi_val in enumerate(trajectory):
        assert mi_val >= -eps, (
            f"Normalized I₂(X₀, X_{l}) must be >= 0. Got {mi_val:.6f}"
        )


# ===========================================================================
# Test: Normalized MI is scale-invariant
# ===========================================================================
# L2 normalization maps X and cX to the same unit vectors.
# Therefore layers_A = [X, 10X, 100X] and layers_B = [X, X, X]
# must produce IDENTICAL MI trajectories after normalization.
# This is THE critical test: proves normalization removes the scale artifact.


def test_normalized_mi_trajectory_scale_invariant(any_backend):
    """MI trajectory is identical for [X, 10X, 100X] and [X, X, X]."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_mi_trajectory,
    )

    backend = any_backend
    n = 15

    base = [[float(i + 1), float(i * 2 + 1)] for i in range(n)]

    # layers_A: same data at different scales
    layers_a = [
        backend.array(base),
        backend.array([[v * 10.0 for v in row] for row in base]),
        backend.array([[v * 100.0 for v in row] for row in base]),
    ]

    # layers_B: same data at same scale
    layers_b = [
        backend.array(base),
        backend.array(base),
        backend.array(base),
    ]

    traj_a, _, _ = compute_normalized_mi_trajectory(layers_a, backend)
    traj_b, _, _ = compute_normalized_mi_trajectory(layers_b, backend)

    eps = _div_eps(backend)
    for l in range(len(traj_a)):
        assert abs(traj_a[l] - traj_b[l]) < eps, (
            f"Scale invariance violated at layer {l}: "
            f"scaled={traj_a[l]:.6f}, unscaled={traj_b[l]:.6f}"
        )


# ===========================================================================
# Test: Normalized MI returns positive shared sigma
# ===========================================================================
# The shared sigma derived from ALL layers' combined distance statistics
# must be positive. Also verifies trajectory and all-pairs use the same sigma.


def test_normalized_mi_shared_sigma_consistent(any_backend):
    """Shared sigma is positive and consistent between trajectory/all-pairs."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_all_pairs_mi,
        compute_normalized_mi_trajectory,
    )

    backend = any_backend
    n = 15

    layers = [
        backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)]),
        backend.array([[float(i * 3 + 2), float(i + 5)] for i in range(n)]),
        backend.array([[float(n - i), float(i * 4 + 2)] for i in range(n)]),
    ]

    _, sigma_traj, _ = compute_normalized_mi_trajectory(layers, backend)
    _, sigma_pairs, _ = compute_normalized_all_pairs_mi(layers, backend)

    assert sigma_traj > 0.0, f"Trajectory sigma must be positive. Got {sigma_traj}"
    assert sigma_pairs > 0.0, f"All-pairs sigma must be positive. Got {sigma_pairs}"

    # Same input → same sigma (deterministic computation)
    assert abs(sigma_traj - sigma_pairs) < _div_eps(backend), (
        f"Sigma should be consistent: trajectory={sigma_traj:.6f}, "
        f"all-pairs={sigma_pairs:.6f}"
    )


# ===========================================================================
# Test: Normalized all-pairs MI is symmetric
# ===========================================================================
# MI(i,j) = MI(j,i) by Hadamard commutativity, regardless of sigma regime.


def test_normalized_all_pairs_mi_symmetric(any_backend):
    """Normalized L×L MI matrix is symmetric."""
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_all_pairs_mi,
    )

    backend = any_backend
    n = 15

    # Three "layers" with different point clouds at different scales
    layers = [
        backend.array([[float(i + 1), float(i * 2 + 1)] for i in range(n)]),
        backend.array([[float(i * 3 + 2) * 10, float(i + 5) * 10] for i in range(n)]),
        backend.array([[float(n - i) * 100, float(i * 4 + 2) * 100] for i in range(n)]),
    ]

    mi_matrix, sigma, _ = compute_normalized_all_pairs_mi(layers, backend)

    assert sigma > 0.0, f"Shared sigma must be positive. Got {sigma}"

    num_layers = len(mi_matrix)
    eps = _div_eps(backend)

    for i in range(num_layers):
        for j in range(num_layers):
            assert abs(mi_matrix[i][j] - mi_matrix[j][i]) < eps, (
                f"Normalized MI matrix not symmetric at ({i},{j}): "
                f"{mi_matrix[i][j]:.6f} vs {mi_matrix[j][i]:.6f}"
            )


# ===========================================================================
# Sigma Calibration Tests
# ===========================================================================
# Tests for constraint-satisfaction sigma calibration (Regime 5).
# See docs/research/sigma_calibration_design.md for derivation.


def _make_unit_sphere_layers(backend, n_probes, n_dims, n_layers, seed=42):
    """Create synthetic layers on the unit sphere with different rotations.

    Each layer applies a progressive rotation to the base data.
    Returns L2-normalized activations and their squared geodesic distance matrices.
    """
    import random as rng

    from modelcypher.core.domain.geometry.cka import geodesic_squared_distances

    rng.seed(seed)

    # Base data: deterministic points in R^D, then normalize to unit sphere
    base = []
    for i in range(n_probes):
        row = [float(i + 1 + j * 0.3) for j in range(n_dims)]
        norm = sum(v * v for v in row) ** 0.5
        base.append([v / norm for v in row])

    layers_raw = [base]
    for l in range(1, n_layers):
        # Apply progressive rotation: mix dimensions by increasing amounts
        rotated = []
        angle = 0.3 * l  # radians
        import math as _math

        cos_a, sin_a = _math.cos(angle), _math.sin(angle)
        for row in base:
            new_row = list(row)
            # Rotate first two dimensions
            new_row[0] = row[0] * cos_a - row[1] * sin_a
            new_row[1] = row[0] * sin_a + row[1] * cos_a
            # Normalize back to unit sphere
            norm = sum(v * v for v in new_row) ** 0.5
            new_row = [v / norm for v in new_row]
            rotated.append(new_row)
        layers_raw.append(rotated)

    layer_arrays = [backend.array(layer) for layer in layers_raw]
    sq_dists = [geodesic_squared_distances(la, backend) for la in layer_arrays]
    return layer_arrays, sq_dists


# ===========================================================================
# Test: Calibration finds feasible sigma for well-behaved data
# ===========================================================================
# For layers on the unit sphere with moderate rotations, there exists a sigma
# where all layers are non-degenerate. The feasible interval is non-empty.


def test_calibration_finds_feasible_sigma(any_backend):
    """Calibration finds non-None sigma for well-behaved unit sphere data."""
    from modelcypher.core.domain.geometry.sigma_calibration import (
        compute_calibrated_sigma,
    )

    backend = any_backend
    _, sq_dists = _make_unit_sphere_layers(backend, n_probes=30, n_dims=10, n_layers=4)

    result = compute_calibrated_sigma(sq_dists, n_probes=30, backend=backend)

    assert not result.is_multi_scale, (
        "Well-behaved unit sphere data should have a feasible sigma interval"
    )
    assert result.sigma_star is not None, "sigma_star should be non-None"
    assert result.sigma_star > 0.0, f"sigma_star must be positive. Got {result.sigma_star}"
    assert result.feasible_lower is not None
    assert result.feasible_upper is not None
    assert result.feasible_lower < result.feasible_upper, (
        f"Feasible interval invalid: [{result.feasible_lower}, {result.feasible_upper}]"
    )


# ===========================================================================
# Test: Calibration detects multi-scale when layers need incompatible sigmas
# ===========================================================================
# Layer A: tight cluster (all pairwise d² ≈ 1e-6, needing σ ~ 0.001)
# Layer B: spread data (all pairwise d² ≈ 4.0, needing σ ~ 2.0)
# These require sigmas ~2000× apart — no single sigma satisfies both.
#
# Derivation: for equidistant points with d² = D, feasible σ ∈ [D/(2×2.47), D/(2×0.251)]
# when N=30. Tight: σ ∈ [0.0004, 0.004]. Spread: σ ∈ [0.81, 7.96]. Disjoint.


def test_calibration_returns_multi_scale_for_extreme_layers(any_backend):
    """Calibration reports multi-scale when layers need incompatible sigmas."""
    from modelcypher.core.domain.geometry.sigma_calibration import (
        compute_calibrated_sigma,
    )

    backend = any_backend
    n = 30

    # Tight cluster: all pairwise squared geodesic distances ≈ 1e-6
    ones = backend.array([[1.0] * n] * n)
    eye = backend.array([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
    sq_dist_tight = (ones - eye) * 1e-6

    # Spread cluster: all pairwise squared geodesic distances ≈ 4.0
    sq_dist_spread = (ones - eye) * 4.0

    result = compute_calibrated_sigma(
        [sq_dist_tight, sq_dist_spread], n_probes=n, backend=backend
    )

    assert result.is_multi_scale, (
        "Tight + spread clusters should be detected as multi-scale"
    )
    assert result.sigma_star is None, (
        "sigma_star should be None when model is multi-scale"
    )


# ===========================================================================
# Test: Calibrated Grams are non-degenerate
# ===========================================================================
# When calibration succeeds (sigma_star is not None), all layers' Gram matrices
# at sigma_star must satisfy √ε < S₂ < log₂(N) - √ε.
# This is the defining property: calibration guarantees non-degeneracy.


def test_calibrated_grams_are_nondegenerate(any_backend):
    """S₂ at sigma_star satisfies non-degeneracy constraints for all layers."""
    from modelcypher.core.domain.geometry.sigma_calibration import (
        compute_calibrated_sigma,
    )

    backend = any_backend
    n = 30
    _, sq_dists = _make_unit_sphere_layers(backend, n_probes=n, n_dims=10, n_layers=4)

    result = compute_calibrated_sigma(sq_dists, n_probes=n, backend=backend)

    assert not result.is_multi_scale, "Prerequisite: calibration must succeed"
    assert result.per_layer_entropy is not None

    # division_epsilon already returns sqrt(machine_eps), which is the threshold
    # used by compute_calibrated_sigma for non-degeneracy constraints
    sqrt_eps = _div_eps(backend)
    log2_n = math.log2(n)

    for l, s2 in enumerate(result.per_layer_entropy):
        assert s2 > sqrt_eps, (
            f"Layer {l}: S₂={s2:.6f} <= √ε={sqrt_eps:.6f} (collapsed)"
        )
        assert log2_n - s2 > sqrt_eps, (
            f"Layer {l}: log₂(N)-S₂={log2_n - s2:.6f} <= √ε={sqrt_eps:.6f} (saturated)"
        )


# ===========================================================================
# Test: Calibrated sigma is scale-invariant after L2 normalization
# ===========================================================================
# [X, 10X, 100X] and [X, X, X] produce identical unit vectors after L2 norm,
# hence identical geodesic distance matrices, hence identical calibrated sigma.


def test_calibrated_sigma_is_scale_invariant(any_backend):
    """Calibration gives identical sigma for scaled vs unscaled layers."""
    from modelcypher.core.domain.geometry.cka import geodesic_squared_distances
    from modelcypher.core.domain.geometry.numerical_stability import (
        division_epsilon,
    )
    from modelcypher.core.domain.geometry.sigma_calibration import (
        compute_calibrated_sigma,
    )

    backend = any_backend
    n = 20

    # Base data: points on unit sphere
    base_raw = [[float(i + 1 + j * 0.5) for j in range(8)] for i in range(n)]
    # Normalize to unit sphere
    base = []
    for row in base_raw:
        norm = sum(v * v for v in row) ** 0.5
        base.append([v / norm for v in row])

    base_arr = backend.array(base)

    # layers_A: same data at different scales (L2 norm maps all to same unit vectors)
    scaled_10 = base_arr * 10.0
    scaled_100 = base_arr * 100.0

    # Normalize before computing geodesics (as the pipeline does)
    def l2_normalize(arr):
        norms = backend.norm(arr, axis=1, keepdims=True)
        eps_val = division_epsilon(backend, arr)
        safe_norms = backend.maximum(norms, backend.array([[eps_val]]))
        return arr / safe_norms

    norm_a = [l2_normalize(base_arr), l2_normalize(scaled_10), l2_normalize(scaled_100)]
    norm_b = [l2_normalize(base_arr), l2_normalize(base_arr), l2_normalize(base_arr)]

    sq_dists_a = [geodesic_squared_distances(la, backend) for la in norm_a]
    sq_dists_b = [geodesic_squared_distances(lb, backend) for lb in norm_b]

    result_a = compute_calibrated_sigma(sq_dists_a, n_probes=n, backend=backend)
    result_b = compute_calibrated_sigma(sq_dists_b, n_probes=n, backend=backend)

    eps = _div_eps(backend)
    assert result_a.sigma_star is not None, "Calibration A must succeed"
    assert result_b.sigma_star is not None, "Calibration B must succeed"
    assert abs(result_a.sigma_star - result_b.sigma_star) < eps, (
        f"Scale invariance violated: σ_A={result_a.sigma_star:.6f}, "
        f"σ_B={result_b.sigma_star:.6f}"
    )


# ===========================================================================
# Test: S₂ is monotonically decreasing in sigma
# ===========================================================================
# Critical invariant for binary search correctness.
# For any layer, S₂(K(σ_small)) > S₂(K(σ_large)) when σ_small < σ_large.
# Proof: Section 2.2 of sigma_calibration_design.md.


def test_calibration_monotonicity_of_entropy(any_backend):
    """S₂ strictly decreases as sigma increases."""
    from modelcypher.core.domain.geometry.sigma_calibration import (
        _entropy_at_sigma,
    )

    backend = any_backend
    n = 30
    _, sq_dists = _make_unit_sphere_layers(backend, n_probes=n, n_dims=10, n_layers=1)
    sq_dist = sq_dists[0]

    # Test at multiple sigma pairs: smaller sigma → larger S₂
    sigmas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    entropies = [_entropy_at_sigma(sq_dist, s, backend) for s in sigmas]

    for i in range(len(sigmas) - 1):
        assert entropies[i] > entropies[i + 1], (
            f"Monotonicity violated: S₂(σ={sigmas[i]})={entropies[i]:.6f} "
            f"<= S₂(σ={sigmas[i+1]})={entropies[i+1]:.6f}"
        )
