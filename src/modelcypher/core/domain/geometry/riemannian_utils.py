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
Riemannian geometry for high-dimensional representation spaces.

This module provides the main API for Riemannian geometry operations.
It re-exports classes and functions from the implementation modules:

- RiemannianGeometry: Main class for geodesic and Fréchet mean computation
- Result types: FrechetMeanResult, GeodesicDistanceResult, CurvatureEstimate, etc.
- Validation helpers: count_nan, count_inf, all_finite, etc.
- Convenience functions: frechet_mean, geodesic_distance_matrix, etc.

Mathematical Background:
    On a Riemannian manifold (M, g), the geodesic distance d(p, q) is the
    length of the shortest path between p and q. The Fréchet mean minimizes:

        μ = argmin_{p ∈ M} Σᵢ d²(p, xᵢ)

    For discrete point clouds, the manifold is represented by a k-NN graph.
    Geodesic distance = shortest path on this graph. This is exact for the
    discrete manifold structure.

References:
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - Tenenbaum et al. (2000) "Isomap" - geodesic distance via graph
    - Sra & Hosseini (2015) "Conic Geometric Optimization on the Manifold"

Research Connections:
    Geodesic distance is the correct metric for neural representations because
    curvature is inherent in high-dimensional spaces. This aligns with the
    Platonic Representation Hypothesis (Huh et al., ICML 2024).

    See also: docs/RESEARCH-CONNECTIONS.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export from core module (main class)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_core import (
    RiemannianGeometry,
    _cache,
    _get_riemannian_geometry,
    _RG_CACHE,
)

# Re-export result types
from modelcypher.core.domain.geometry.riemannian_types import (
    CurvatureEstimate,
    DirectionalCoverage,
    FarthestPointSamplingResult,
    FrechetMeanResult,
    GeodesicDistanceResult,
)

# Re-export validation helpers
from modelcypher.core.domain.geometry.riemannian_validation import (
    all_finite,
    count_finite,
    count_inf,
    count_nan,
    count_nonfinite,
    derive_k_neighbors,
    has_inf,
    has_nan,
    safe_arithmetic_mean,
    set_matrix_element,
)



if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Convenience functions
# =============================================================================


def frechet_mean(
    points: "Array",
    weights: "Array | None" = None,
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    max_k_neighbors: int | None = None,
) -> "Array":
    """
    Compute the Fréchet mean of a point set.

    Convenience function that returns just the mean point.

    Args:
        points: Point cloud [n, d]
        weights: Optional weights [n]
        backend: Backend to use
        k_neighbors: Optional fixed k for geodesic graph connectivity
        max_k_neighbors: Optional upper bound for adaptive k retries

    Returns:
        Fréchet mean point [d]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.frechet_mean(
        points,
        weights,
        k_neighbors=k_neighbors,
        max_k_neighbors=max_k_neighbors,
    )
    return result.mean


def geodesic_distance_matrix(
    points: "Array",
    k_neighbors: int | None = None,
    backend: "Backend | None" = None,
    refine_iterations: int = 1,
) -> "Array":
    """
    Compute pairwise geodesic distances.

    Convenience function that returns just the distance matrix.

    Args:
        points: Point cloud [n, d]
        k_neighbors: Number of neighbors for graph construction
        backend: Backend to use
        refine_iterations: Number of geodesic refinement passes

    Returns:
        Geodesic distance matrix [n, n]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.geodesic_distances(
        points, k_neighbors, refine_iterations=refine_iterations
    )
    return result.distances


def farthest_point_sampling(
    points: "Array",
    n_samples: int,
    backend: "Backend | None" = None,
) -> list[int]:
    """
    Select points via geodesic farthest point sampling.

    Convenience function that returns just the selected indices.

    Args:
        points: Point cloud [n, d]
        n_samples: Number of points to select
        backend: Backend to use

    Returns:
        List of selected point indices
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.farthest_point_sampling(points, n_samples)
    return result.selected_indices


def find_sparse_direction(
    point_idx: int,
    points: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """
    Find the most under-sampled direction at a point.

    Convenience function that returns just the sparse direction vector.

    Args:
        point_idx: Index of the center point
        points: Point cloud [n, d]
        backend: Backend to use

    Returns:
        Unit vector in the most sparse direction [d]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.directional_coverage(point_idx, points)
    return result.sparse_direction


# =============================================================================
# Geodesic Vector Operations
# =============================================================================
# These replace the Euclidean operations from the deleted vector_math.py.
# ALL distance computations use k-NN geodesic (correct in high dimensions).


def _geodesic_origin_distances(
    points: "Array",
    backend: "Backend",
    use_cache: bool = True,
    *,
    connect_origin_to_all: bool = True,
) -> tuple["GeodesicDistanceResult", "Array"]:
    """Compute geodesic distances among points plus distances from origin.

    The origin is attached as an out-of-sample query so it does NOT alter
    the k-NN graph among the existing points.
    """
    rg = _get_riemannian_geometry(backend)
    geo_result = rg.geodesic_distances(points, use_cache=use_cache)
    d = int(backend.shape(points)[1])
    origin = backend.zeros((d,), dtype=getattr(points, "dtype", None))
    k_neighbors = int(backend.shape(points)[0]) if connect_origin_to_all else None
    d0 = rg._geodesic_distances_from_query(
        points,
        origin,
        geo_result=geo_result,
        k_neighbors=k_neighbors,
    )
    backend.eval(d0)
    return geo_result, d0


def _chord_cosine_matrix(
    vectors: "Array",
    backend: "Backend",
) -> "Array":
    """Compute cosine similarity matrix using chord (Euclidean) norms."""
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    vectors_arr = backend.array(vectors) if not hasattr(vectors, "shape") else vectors
    shape = backend.shape(vectors_arr)
    if len(shape) != 2:
        raise ValueError("chord_cosine_matrix requires [n, d] vectors")
    n, _ = shape
    if n == 0:
        return backend.array([])

    norms = backend.sqrt(backend.sum(vectors_arr * vectors_arr, axis=1, keepdims=True))
    denom = backend.matmul(norms, backend.transpose(norms))
    eps = division_epsilon(backend, vectors_arr)
    denom_safe = backend.maximum(denom, backend.full(backend.shape(denom), eps))
    dot = backend.matmul(vectors_arr, backend.transpose(vectors_arr))
    cos_matrix = dot / denom_safe
    cos_matrix = backend.clip(cos_matrix, -1.0, 1.0)
    valid = denom > eps
    cos_matrix = backend.where(valid, cos_matrix, backend.zeros_like(cos_matrix))
    backend.eval(cos_matrix)
    return cos_matrix


def geodesic_norms(
    vectors: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> "Array":
    """Compute geodesic norms (distance from origin) for each row.

    This computes the TRUE geodesic distance from the origin to each point,
    using k-NN graph shortest paths. In high dimensions (4D+), Euclidean
    distance is systematically wrong due to curvature.

    Args:
        vectors: Matrix [n, d] where each row is a point
        backend: Backend for tensor operations

    Returns:
        Array [n] of geodesic distances from origin
    """
    vectors_arr = backend.array(vectors) if not hasattr(vectors, "shape") else vectors
    shape = backend.shape(vectors_arr)
    if len(shape) != 2:
        raise ValueError("geodesic_norms requires [n, d] vectors")
    n, d = shape
    if n == 0:
        return backend.array([])

    rg = _get_riemannian_geometry(backend)
    d = int(backend.shape(vectors_arr)[1])
    origin = backend.zeros((d,), dtype=getattr(vectors_arr, "dtype", None))
    norms = rg._geodesic_distances_from_query(
        vectors_arr,
        origin,
        geo_result=None,
        use_single_source=True,
        use_cache=use_cache,
    )
    backend.eval(norms)
    return norms


def geodesic_cosine_matrix(
    vectors: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> "Array":
    """Compute geodesic cosine similarities between all pairs.

    Uses the geodesic law of cosines:
        cos(θ_ij) = (d²(0,i) + d²(0,j) - d²(i,j)) / (2 × d(0,i) × d(0,j))

    All distances are true geodesic (k-NN graph shortest paths).

    Args:
        vectors: Matrix [n, d] where each row is a point
        backend: Backend for tensor operations

    Returns:
        Cosine similarity matrix [n, n]
    """
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    vectors_arr = backend.array(vectors) if not hasattr(vectors, "shape") else vectors
    shape = backend.shape(vectors_arr)
    if len(shape) != 2:
        raise ValueError("geodesic_cosine_matrix requires [n, d] vectors")
    n, d = shape
    if n == 0:
        return backend.array([])
    if n < 3:
        raise ValueError(
            f"geodesic_cosine_matrix requires n >= 3 points to infer manifold structure, got n={n}. "
            "Use chord_cosine_matrix explicitly if chord distance is acceptable."
        )

    # Compute geodesic distances among points; attach origin as a query only.
    geo_result, d0 = _geodesic_origin_distances(
        vectors_arr,
        backend,
        use_cache=use_cache,
        connect_origin_to_all=True,
    )
    D_pairs = geo_result.distances  # [n, n] pairwise distances

    # =========================================================================
    # STABLE GEODESIC COSINE VIA HALF-ANGLE FORMULA
    # =========================================================================
    # The standard law of cosines: cos(θ) = (a² + b² - c²) / (2ab)
    # suffers from catastrophic cancellation when θ ≈ 0 (c ≈ 0) or θ ≈ π
    #
    # STABLE FORMULA using half-angle identity:
    #   sin²(θ/2) = (b + c - a)(a + c - b) / (4ab)
    #   cos(θ) = 1 - 2*sin²(θ/2)
    #
    # This avoids subtracting nearly-equal squared quantities by using
    # products of smaller terms. Numerically stable for all angles.
    #
    # Reference: Kahan, W. (2014) "How Futile are Mindless Assessments of
    # Roundoff in Floating-Point Computation?" - Section on triangle area
    # =========================================================================
    d0_row = backend.reshape(d0, (1, n))  # b
    d0_col = backend.reshape(d0, (n, 1))  # a

    eps = division_epsilon(backend, d0)
    denom = 2.0 * d0_col * d0_row  # 2ab
    safe_denom = backend.maximum(denom, backend.full(backend.shape(denom), eps))

    # Half-angle terms: (b + c - a) and (a + c - b)
    # where a = d0_col, b = d0_row, c = D_pairs
    term1 = d0_row + D_pairs - d0_col  # (b + c - a)
    term2 = d0_col + D_pairs - d0_row  # (a + c - b)
    backend.eval(term1, term2)

    # Clamp negative terms (can happen due to floating point, indicates θ ≈ 0 or θ ≈ π)
    # When term1 or term2 < 0, the triangle inequality is violated (numerical noise)
    term1_safe = backend.maximum(term1, backend.zeros_like(term1))
    term2_safe = backend.maximum(term2, backend.zeros_like(term2))

    # sin²(θ/2) = term1 * term2 / (4ab) = term1 * term2 / (2 * denom)
    sin_sq_half = (term1_safe * term2_safe) / (2.0 * safe_denom)
    # Clamp to [0, 1] for sin²
    sin_sq_half = backend.clip(sin_sq_half, 0.0, 1.0)

    # cos(θ) = 1 - 2*sin²(θ/2)
    cos_matrix = 1.0 - 2.0 * sin_sq_half
    # Final clamp to [-1, 1] for numerical safety
    cos_matrix = backend.clip(cos_matrix, -1.0, 1.0)

    # Zero out entries where either point is at origin
    valid = backend.minimum(d0_col > eps, d0_row > eps)
    cos_matrix = backend.where(valid, cos_matrix, backend.zeros_like(cos_matrix))

    backend.eval(cos_matrix)
    return cos_matrix


def geodesic_cosine_batch(
    anchor: "Array",
    vectors: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> "Array":
    """Compute geodesic cosine similarities between anchor and each row.

    Args:
        anchor: Single point [d]
        vectors: Matrix [n, d]
        backend: Backend for tensor operations

    Returns:
        Array [n] of cosine similarities
    """
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    anchor_arr = backend.array(anchor) if not hasattr(anchor, "shape") else anchor
    vectors_arr = backend.array(vectors) if not hasattr(vectors, "shape") else vectors

    shape_anchor = backend.shape(anchor_arr)
    shape_vectors = backend.shape(vectors_arr)

    if len(shape_anchor) != 1:
        anchor_arr = backend.reshape(anchor_arr, (-1,))
        shape_anchor = backend.shape(anchor_arr)
    if len(shape_vectors) != 2:
        raise ValueError("geodesic_cosine_batch requires [n, d] vectors")
    if shape_vectors[0] == 0:
        return backend.array([])
    if shape_anchor[0] != shape_vectors[1]:
        raise ValueError("Anchor and vectors must share feature dimension")

    n, d = shape_vectors
    if n < 3:
        raise ValueError(
            f"geodesic_cosine_from_anchor requires n >= 3 points to infer manifold structure, got n={n}. "
            "Use chord cosine explicitly if chord distance is acceptable."
        )

    rg = _get_riemannian_geometry(backend)
    k_neighbors = derive_k_neighbors(vectors_arr, backend) if n > 1 else None

    # Distances from origin to each vector (single-source path).
    origin = backend.zeros((d,), dtype=getattr(vectors_arr, "dtype", None))
    d0_vectors = rg._geodesic_distances_from_query(
        vectors_arr,
        origin,
        geo_result=None,
        k_neighbors=k_neighbors,
        use_single_source=True,
        use_cache=use_cache,
    )

    # Distances from anchor to each vector (single-source path).
    d_anchor_vectors = rg._geodesic_distances_from_query(
        vectors_arr,
        anchor_arr,
        geo_result=None,
        k_neighbors=k_neighbors,
        use_single_source=True,
        use_cache=use_cache,
    )

    # Distance from origin to anchor via the shared manifold:
    # shortest path between two query nodes attached to the same graph.
    d0_anchor = backend.min(d0_vectors + d_anchor_vectors)

    # Stable geodesic cosine via half-angle formula (see geodesic_cosine_matrix)
    # a = d0_anchor, b = d0_vectors, c = d_anchor_vectors
    eps = division_epsilon(backend, d0_vectors)
    denom = 2.0 * d0_anchor * d0_vectors
    safe_denom = backend.maximum(denom, backend.full(backend.shape(denom), eps))

    # Half-angle terms
    term1 = d0_vectors + d_anchor_vectors - d0_anchor  # (b + c - a)
    term2 = d0_anchor + d_anchor_vectors - d0_vectors  # (a + c - b)
    term1_safe = backend.maximum(term1, backend.zeros_like(term1))
    term2_safe = backend.maximum(term2, backend.zeros_like(term2))

    sin_sq_half = (term1_safe * term2_safe) / (2.0 * safe_denom)
    sin_sq_half = backend.clip(sin_sq_half, 0.0, 1.0)
    cos_vals = 1.0 - 2.0 * sin_sq_half
    cos_vals = backend.clip(cos_vals, -1.0, 1.0)

    # Zero out if anchor or vector at origin
    valid = backend.minimum(d0_anchor > eps, d0_vectors > eps)
    cos_vals = backend.where(valid, cos_vals, backend.zeros_like(cos_vals))

    backend.eval(cos_vals)
    return cos_vals


def geodesic_cosine_between_sets(
    a: "Array",
    b: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> "Array":
    """Compute geodesic cosine similarities between two sets of vectors.

    Args:
        a: First matrix [m, d]
        b: Second matrix [n, d]
        backend: Backend for tensor operations

    Returns:
        Cosine similarity matrix [m, n]
    """
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    a_arr = backend.array(a) if not hasattr(a, "shape") else a
    b_arr = backend.array(b) if not hasattr(b, "shape") else b

    shape_a = backend.shape(a_arr)
    shape_b = backend.shape(b_arr)

    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ValueError("geodesic_cosine_between_sets requires [m, d] and [n, d]")
    if shape_a[0] == 0 or shape_b[0] == 0:
        return backend.array([])
    if shape_a[1] != shape_b[1]:
        raise ValueError("Inputs must share feature dimension")

    m, d = shape_a
    n = shape_b[0]

    if m + n < 3:
        raise ValueError(
            f"geodesic_cosine_between_sets requires at least 3 total points, got {m + n}"
        )

    # Stack: set_a, set_b (origin attached as query only)
    points = backend.concatenate([a_arr, b_arr], axis=0)
    backend.eval(points)

    geo_result, d0 = _geodesic_origin_distances(
        points,
        backend,
        use_cache=use_cache,
        connect_origin_to_all=True,
    )
    D = geo_result.distances  # [m+n, m+n]

    # Indices: 0..m-1=set_a, m..m+n-1=set_b
    d0_a = d0[:m]  # [m] distances from origin to set_a
    d0_b = d0[m:]  # [n] distances from origin to set_b
    D_ab = D[:m, m:]  # [m, n] pairwise distances between sets

    # Stable geodesic cosine via half-angle formula (see geodesic_cosine_matrix)
    # a = d0_a_col, b = d0_b_row, c = D_ab
    eps = division_epsilon(backend, D_ab)
    d0_a_col = backend.reshape(d0_a, (m, 1))
    d0_b_row = backend.reshape(d0_b, (1, n))

    denom = 2.0 * d0_a_col * d0_b_row
    safe_denom = backend.maximum(denom, backend.full(backend.shape(denom), eps))

    # Half-angle terms
    term1 = d0_b_row + D_ab - d0_a_col  # (b + c - a)
    term2 = d0_a_col + D_ab - d0_b_row  # (a + c - b)
    term1_safe = backend.maximum(term1, backend.zeros_like(term1))
    term2_safe = backend.maximum(term2, backend.zeros_like(term2))

    sin_sq_half = (term1_safe * term2_safe) / (2.0 * safe_denom)
    sin_sq_half = backend.clip(sin_sq_half, 0.0, 1.0)
    cos_matrix = 1.0 - 2.0 * sin_sq_half
    cos_matrix = backend.clip(cos_matrix, -1.0, 1.0)

    valid = backend.minimum(d0_a_col > eps, d0_b_row > eps)
    cos_matrix = backend.where(valid, cos_matrix, backend.zeros_like(cos_matrix))

    backend.eval(cos_matrix)
    return cos_matrix


def geodesic_pairwise_metrics(
    a: "Array",
    b: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> tuple["Array", "Array"]:
    """Compute geodesic cosines and distances for paired vectors.

    Given matrices A and B of shape [n, d], returns arrays where
    result[i] compares A[i] with B[i].

    Args:
        a: First matrix [n, d]
        b: Second matrix [n, d] (must have same shape as a)
        backend: Backend for tensor operations

    Returns:
        Tuple of (cosine_similarities [n], distances [n])
    """
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    a_arr = backend.array(a) if not hasattr(a, "shape") else a
    b_arr = backend.array(b) if not hasattr(b, "shape") else b

    shape_a = backend.shape(a_arr)
    shape_b = backend.shape(b_arr)

    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ValueError("geodesic_pairwise_metrics requires [n, d] inputs")
    if shape_a[0] == 0:
        return backend.array([]), backend.array([])
    if shape_a != shape_b:
        raise ValueError("Inputs must share shape for paired metrics")

    n, d = shape_a
    if n < 3:
        raise ValueError(
            f"geodesic_pairwise_metrics requires n >= 3 points to infer manifold structure, got n={n}. "
            "Use chord distance explicitly if chord metrics are acceptable."
        )

    # Interleave a and b for efficient geodesic computation
    interleaved = backend.reshape(
        backend.stack([a_arr, b_arr], axis=1),
        (2 * n, d)
    )
    backend.eval(interleaved)

    geo_result, d0_all = _geodesic_origin_distances(
        interleaved,
        backend,
        use_cache=use_cache,
        connect_origin_to_all=True,
    )
    D = geo_result.distances  # [2n, 2n]

    # Indices 0, 2, 4, ... = a[0], a[1], a[2], ...
    # Indices 1, 3, 5, ... = b[0], b[1], b[2], ...
    a_indices = backend.arange(0, 2*n, 2)  # [0, 2, 4, ...]
    b_indices = backend.arange(1, 2*n, 2)  # [1, 3, 5, ...]

    d0_a = backend.take(d0_all, a_indices, axis=0)  # [n]
    d0_b = backend.take(d0_all, b_indices, axis=0)  # [n]

    # Extract pairwise distances between matched pairs D[a_i, b_i] without CPU loops
    size = int(D.shape[0])
    flat_D = backend.reshape(D, (-1,))
    linear_idx = a_indices * size + b_indices
    distances = backend.take(flat_D, linear_idx, axis=0)

    # Stable geodesic cosine via half-angle formula (see geodesic_cosine_matrix)
    # a = d0_a, b = d0_b, c = distances
    eps = division_epsilon(backend, d0_a)
    denom = 2.0 * d0_a * d0_b
    safe_denom = backend.maximum(denom, backend.full(backend.shape(denom), eps))

    # Half-angle terms
    term1 = d0_b + distances - d0_a  # (b + c - a)
    term2 = d0_a + distances - d0_b  # (a + c - b)
    term1_safe = backend.maximum(term1, backend.zeros_like(term1))
    term2_safe = backend.maximum(term2, backend.zeros_like(term2))

    sin_sq_half = (term1_safe * term2_safe) / (2.0 * safe_denom)
    sin_sq_half = backend.clip(sin_sq_half, 0.0, 1.0)
    cos_vals = 1.0 - 2.0 * sin_sq_half
    cos_vals = backend.clip(cos_vals, -1.0, 1.0)

    valid = backend.minimum(d0_a > eps, d0_b > eps)
    cos_vals = backend.where(valid, cos_vals, backend.zeros_like(cos_vals))

    backend.eval(cos_vals, distances)
    return cos_vals, distances


def geodesic_paired_distances(
    a: "Array",
    b: "Array",
    backend: "Backend",
    use_cache: bool = True,
) -> "Array":
    """Compute geodesic distances between paired vectors.

    Given matrices A and B of shape [n, d], returns an array of n geodesic
    distances where result[i] = geodesic_distance(A[i], B[i]).

    Args:
        a: First matrix [n, d]
        b: Second matrix [n, d] (must have same shape as a)
        backend: Backend for tensor operations

    Returns:
        Array of shape [n] with geodesic distance for each pair
    """
    _, distances = geodesic_pairwise_metrics(a, b, backend, use_cache=use_cache)
    return distances


def geodesic_cosine_sparse(
    a: dict[int, float],
    b: dict[int, float],
    backend: "Backend",
) -> float:
    """Compute geodesic cosine similarity between sparse vectors.

    Converts sparse vectors to dense and computes geodesic cosine.

    Args:
        a: Sparse vector as {index: value}
        b: Sparse vector as {index: value}
        backend: Backend for tensor operations

    Returns:
        Geodesic cosine similarity

    Raises:
        ValueError: If vectors are empty
    """
    if not a or not b:
        raise ValueError("Cannot compute cosine similarity of empty sparse vectors")

    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        raise ValueError("Cannot compute cosine similarity of empty sparse vectors")

    from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

    dtype = precision_dtype(backend)
    vec_a = backend.array([float(a.get(key, 0.0)) for key in keys], dtype=dtype)
    vec_b = backend.array([float(b.get(key, 0.0)) for key in keys], dtype=dtype)
    backend.eval(vec_a, vec_b)

    # Stack as 2 vectors and compute cosine
    vectors = backend.stack([vec_a, vec_b], axis=0)
    cos_matrix = geodesic_cosine_matrix(vectors, backend)
    backend.eval(cos_matrix)

    return float(backend.to_scalar(cos_matrix[0, 1]))


__all__ = [
    # Main class
    "RiemannianGeometry",
    # Result types
    "FrechetMeanResult",
    "GeodesicDistanceResult",
    "CurvatureEstimate",
    "DirectionalCoverage",
    "FarthestPointSamplingResult",
    # Validation helpers
    "count_nan",
    "count_inf",
    "count_finite",
    "count_nonfinite",
    "has_nan",
    "has_inf",
    "all_finite",
    "derive_k_neighbors",
    "safe_arithmetic_mean",
    "set_matrix_element",
    # Convenience functions
    "frechet_mean",
    "geodesic_distance_matrix",
    "farthest_point_sampling",
    "find_sparse_direction",
    # Geodesic vector operations (correct in high dimensions)
    "geodesic_norms",
    "geodesic_cosine_matrix",
    "geodesic_cosine_batch",
    "geodesic_cosine_between_sets",
    "geodesic_pairwise_metrics",
    "geodesic_paired_distances",
    "geodesic_cosine_sparse",
    # Private (for internal use)
    "_get_riemannian_geometry",
    "_RG_CACHE",
    "_cache",
]
