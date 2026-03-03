# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Information bridge: connects geometric measurements to information theory.

Computes curvature excess, all-pairs Rényi MI, and MI trajectories using
pre-computed kernel matrices. Every formula traces to the derivations in
docs/research/information_bridge_derivation.md.

Sigma regimes (Section 2.3 of derivation):
    Regime 1: Per-layer sigma — each layer measured at its own scale.
        NOT comparable across layers.
    Regime 3: Fixed sigma — single sigma from input layer.
        Commensurable but saturates if sigma_0 is inappropriate for deep layers.
    Regime 4: L2-normalized + shared sigma — L2-normalize activations to unit
        hypersphere, derive one sigma from all layers' combined distances.
        Fully commensurable (same resolution, bounded distance support).

Dependencies:
    - renyi_mi.py: compute_renyi_mi_alpha2, compute_renyi_entropy_alpha2
    - cka.py: rbf_gram_matrix, rbf_gram_matrix_with_sigma,
              geodesic_squared_distances, _derive_rbf_sigma_from_values,
              _rbf_gram_from_sq_distances
    - effective_rank.py: EffectiveRank.compute() -> spectral_entropy (nats)
    - intrinsic_dimension.py: IntrinsicDimension.compute_two_nn() -> ID
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.renyi_mi import compute_renyi_mi_alpha2

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def compute_curvature_excess(
    spectral_entropy_nats: float, intrinsic_dim: float
) -> float:
    """Compute curvature excess: C_ex = S_spec - ln(ID).

    Both terms in nats for unit consistency with EffectiveRank.compute()
    which uses natural log.

    Derivation (Section 7.2-7.3 of information_bridge_derivation.md):
        C_ex >= 0 always (differential geometry).
        C_ex = 0 iff the activation manifold is locally flat.

    Args:
        spectral_entropy_nats: Shannon spectral entropy in nats (from
            EffectiveRank.compute().spectral_entropy).
        intrinsic_dim: Intrinsic dimension (from
            IntrinsicDimension.compute_two_nn().intrinsic_dimension).

    Returns:
        Curvature excess in nats. Non-negative by construction.
    """
    if intrinsic_dim <= 0.0:
        return 0.0

    c_ex = spectral_entropy_nats - math.log(intrinsic_dim)

    # Clamp to zero: finite-precision arithmetic can produce tiny negatives
    return max(0.0, c_ex)


def compute_all_pairs_renyi_mi(
    layer_grams: list["Array"], backend: "Backend"
) -> list[list[float]]:
    """Compute L x L matrix of pairwise Rényi MI between layers.

    Uses pre-computed Gram matrices (one per layer). Each entry (i, j) is
    I₂(X_i; X_j) computed via the Hadamard product kernel.

    The matrix is symmetric by the commutativity of the Hadamard product.

    Args:
        layer_grams: List of [N, N] Gram matrices, one per layer.
            Compute via rbf_gram_matrix(activations[l], backend).
        backend: Backend for tensor operations.

    Returns:
        L x L matrix of MI values in bits.
    """
    num_layers = len(layer_grams)
    mi_matrix = [[0.0] * num_layers for _ in range(num_layers)]

    for i in range(num_layers):
        for j in range(i, num_layers):
            mi = compute_renyi_mi_alpha2(layer_grams[i], layer_grams[j], backend)
            mi_matrix[i][j] = mi
            mi_matrix[j][i] = mi  # symmetric

    return mi_matrix


def compute_input_mi_trajectory(
    layer_grams: list["Array"], backend: "Backend"
) -> list[float]:
    """Compute I₂(X₀, X_l) for each layer l.

    Measures how much information the input layer shares with each
    subsequent layer, using per-layer sigma (Regime 1).

    Tests predictions P4 (highway = MI minimum) and P5 (ID tracks MI).

    Args:
        layer_grams: List of [N, N] Gram matrices, one per layer.
        backend: Backend for tensor operations.

    Returns:
        List of MI values [I₂(X₀, X₀), I₂(X₀, X₁), ..., I₂(X₀, X_{L-1})].
    """
    if not layer_grams:
        return []

    gram_0 = layer_grams[0]
    return [
        compute_renyi_mi_alpha2(gram_0, gram_l, backend)
        for gram_l in layer_grams
    ]


def compute_fixed_sigma_mi_trajectory(
    layer_activations: list["Array"], backend: "Backend", sigma: float
) -> list[float]:
    """Compute I₂(X₀, X_l) with FIXED sigma across all layers.

    All layers use the same sigma (Regime 3), making MI values at
    different depths commensurable. Required for DPI testing (P6).

    DPI is NOT proven for matrix-based Rényi MI (Section 8.2 of
    derivation). This trajectory is for empirical testing only.

    Args:
        layer_activations: List of [N, D] activation matrices per layer.
        backend: Backend for tensor operations.
        sigma: Fixed bandwidth from input layer (via
            rbf_gram_matrix_with_sigma(X_0, backend)[1]).

    Returns:
        List of MI values at fixed resolution.
    """
    from modelcypher.core.domain.geometry.cka import rbf_gram_matrix

    if not layer_activations:
        return []

    # Compute all Gram matrices with the same fixed sigma
    grams = [rbf_gram_matrix(acts, backend, sigma=sigma) for acts in layer_activations]

    gram_0 = grams[0]
    return [
        compute_renyi_mi_alpha2(gram_0, gram_l, backend)
        for gram_l in grams
    ]


def _normalize_and_build_grams(
    layer_activations: list["Array"], backend: "Backend"
) -> tuple[list["Array"], float, "CalibrationResult | None"]:
    """L2-normalize activations, calibrate sigma, build Gram matrices.

    Regime 5: projects all layers to the unit hypersphere S^{D-1}, then uses
    constraint-satisfaction calibration to find a sigma where ALL layers have
    non-degenerate Gram matrices (S₂ bounded away from both 0 and log₂(N)).

    Falls back to gap heuristic if calibration reports multi-scale.

    Args:
        layer_activations: List of [N, D] activation matrices per layer.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (list_of_gram_matrices, shared_sigma, calibration_result).
        calibration_result is None only when layer_activations is empty.
    """
    from modelcypher.core.domain.geometry.cka import (
        _derive_rbf_sigma_from_values,
        _rbf_gram_from_sq_distances,
        geodesic_squared_distances,
    )
    from modelcypher.core.domain.geometry.numerical_stability import (
        division_epsilon,
    )
    from modelcypher.core.domain.geometry.sigma_calibration import (
        compute_calibrated_sigma,
    )

    if not layer_activations:
        return [], 0.0, None

    # Step 1: L2-normalize each layer's activation rows to the unit sphere
    normalized = []
    for acts in layer_activations:
        norms = backend.norm(acts, axis=1, keepdims=True)
        eps = division_epsilon(backend, acts)
        # Clamp norms to avoid division by zero
        safe_norms = backend.maximum(norms, backend.array([[eps]]))
        normalized.append(acts / safe_norms)

    # Step 2: Compute geodesic squared distances for all normalized layers
    sq_dists = [geodesic_squared_distances(norm_acts, backend) for norm_acts in normalized]

    # Step 3: Calibrate sigma via constraint satisfaction
    n_probes = layer_activations[0].shape[0]
    cal_result = compute_calibrated_sigma(sq_dists, n_probes, backend)

    if cal_result.is_multi_scale:
        # Fallback: gap heuristic (Regime 4 behavior)
        all_values = backend.concatenate(
            [backend.reshape(sd, (-1,)) for sd in sq_dists]
        )
        shared_sigma = _derive_rbf_sigma_from_values(all_values, backend)
    else:
        shared_sigma = cal_result.sigma_star

    # Step 4: Build Gram matrices with shared sigma
    grams = [
        _rbf_gram_from_sq_distances(sd, shared_sigma, backend)
        for sd in sq_dists
    ]

    return grams, shared_sigma, cal_result


def compute_normalized_mi_trajectory(
    layer_activations: list["Array"], backend: "Backend"
) -> tuple[list[float], float, "CalibrationResult | None"]:
    """Compute I₂(X₀, X_l) with L2 normalization + calibrated sigma (Regime 5).

    Makes MI values commensurable across layers by:
    1. L2-normalizing each activation row to the unit hypersphere
    2. Computing geodesic squared distances for ALL normalized layers
    3. Calibrating sigma via constraint satisfaction (all layers non-degenerate)
    4. Building all Gram matrices with calibrated sigma

    After L2 normalization, all activations lie on S^{D-1}. Geodesic
    distances become arc lengths in [0, π], with bounded distance
    distributions comparable across layers.

    Args:
        layer_activations: List of [N, D] activation matrices per layer.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (trajectory, sigma, calibration_result) where trajectory is
        [I₂(X₀, X₀), I₂(X₀, X₁), ..., I₂(X₀, X_{L-1})],
        sigma is the shared bandwidth used, and calibration_result contains
        diagnostics (feasible interval, per-layer entropy, bootstrap CIs).
    """
    if not layer_activations:
        return [], 0.0, None

    grams, sigma, cal_result = _normalize_and_build_grams(layer_activations, backend)

    gram_0 = grams[0]
    trajectory = [
        compute_renyi_mi_alpha2(gram_0, gram_l, backend)
        for gram_l in grams
    ]

    return trajectory, sigma, cal_result


def compute_normalized_all_pairs_mi(
    layer_activations: list["Array"], backend: "Backend"
) -> tuple[list[list[float]], float]:
    """Compute L×L matrix of pairwise Rényi MI with normalized activations.

    Same Regime 4 normalization as compute_normalized_mi_trajectory:
    L2-normalize, shared sigma, commensurable MI values.

    Args:
        layer_activations: List of [N, D] activation matrices per layer.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (mi_matrix, sigma) where mi_matrix is L×L.
    """
    if not layer_activations:
        return [], 0.0

    grams, sigma = _normalize_and_build_grams(layer_activations, backend)

    num_layers = len(grams)
    mi_matrix = [[0.0] * num_layers for _ in range(num_layers)]

    for i in range(num_layers):
        for j in range(i, num_layers):
            mi = compute_renyi_mi_alpha2(grams[i], grams[j], backend)
            mi_matrix[i][j] = mi
            mi_matrix[j][i] = mi  # symmetric

    return mi_matrix, sigma
