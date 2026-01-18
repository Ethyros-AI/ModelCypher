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

"""Unified spectral embedding for geodesic distance and spectral signature.

Computes Laplacian eigenvectors from a k-NN graph built on chord distances.
The resulting embedding supports approximate geodesic distances and spectral
signature outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.mst import minimum_k_from_mst
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    infinity_threshold,
    machine_epsilon,
    power_iteration_eigh,
    precision_dtype,
    sqrt_scalar,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralEmbeddingResult:
    """Result of unified spectral embedding computation.

    The embedding provides both geodesic distances (via Euclidean distance
    in the embedded space) and spectral signature (via eigenvalues).

    Attributes:
        embedding: The spectral embedding [n, k]. Euclidean distance in this
            space approximates geodesic distance on the manifold.
        eigenvalues: Laplacian eigenvalues used in embedding [k], sorted ascending.
            These are the non-zero eigenvalues used to build the embedding.
        eigenvectors: Raw Laplacian eigenvectors used in embedding [n, k].
        all_eigenvalues: All Laplacian eigenvalues [n], sorted ascending.
            Includes zero eigenvalues for spectral signature computation.
        k_used: Number of eigenvectors retained (may be less than n).
        k_neighbors: Number of neighbors used in k-NN graph.
        component_count: Number of connected components (zero eigenvalues).
        kernel_bandwidth: Heat kernel bandwidth σ used for edge weights.
        adjacency: The k-NN adjacency matrix [n, n] with edge weights.
        inf_value: Sentinel value used for non-edges.
        compute_time_ms: Time spent computing the embedding.
    """

    embedding: "Array"
    eigenvalues: "Array"
    eigenvectors: "Array"
    all_eigenvalues: "Array"
    k_used: int
    k_neighbors: int
    component_count: int
    kernel_bandwidth: float
    adjacency: "Array"
    inf_value: float
    compute_time_ms: float = 0.0


def compute_spectral_embedding(
    points: "Array",
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    k_eigenvectors: int | None = None,
) -> SpectralEmbeddingResult:
    """Compute spectral embedding for geodesic distance and spectral signature.

    This is the unified computation that produces both:
    1. Geodesic distances: d(i,j) = ||embedding[i] - embedding[j]||₂
    2. Spectral signature: eigenvalues for heat trace, entropy, etc.

    The embedding uses Varadhan's formula: eigenvectors of the graph Laplacian
    define an isometric embedding where geodesic distance becomes Euclidean.

    Args:
        points: Point cloud [n, d].
        backend: Backend for tensor operations. Uses default if None.
        k_neighbors: Number of neighbors for k-NN graph. If None, finds
            minimum k that makes the graph connected.
        k_eigenvectors: Number of eigenvectors to use. If None, uses
            spectral gap detection to determine automatically.

    Returns:
        SpectralEmbeddingResult with embedding, eigenvalues, and metadata.
    """
    start_time = time.perf_counter()
    backend = backend or get_default_backend()

    points_arr = backend.array(points)
    backend.eval(points_arr)

    n = int(points_arr.shape[0])

    # Edge cases
    if n == 0:
        return _empty_embedding(backend, 0, 0.0)

    if n == 1:
        return _single_point_embedding(backend, points_arr, 0.0)

    # Compute chord (Euclidean) distance matrix for Laplacian construction.
    chord_dist = _chord_distance_matrix(points_arr, backend)
    backend.eval(chord_dist)

    # Handle degenerate case: all points identical (all distances near zero)
    max_dist_arr = backend.max(chord_dist)
    backend.eval(max_dist_arr)
    max_dist = float(backend.to_scalar(max_dist_arr))
    if max_dist < tiny_value(backend, chord_dist):
        # All points at same location - return degenerate embedding
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SpectralEmbeddingResult(
            embedding=backend.zeros((n, 1)),
            eigenvalues=backend.zeros((1,)),
            eigenvectors=backend.ones((n, 1)) / float(n) ** 0.5,
            all_eigenvalues=backend.zeros((n,)),
            k_used=1,
            k_neighbors=k_neighbors if k_neighbors is not None else 0,
            component_count=1,
            kernel_bandwidth=0.0,
            adjacency=backend.zeros((n, n)),
            inf_value=float("inf"),
            compute_time_ms=elapsed_ms,
        )

    # Find minimum k for connectivity using MST if not specified
    if k_neighbors is None:
        k_neighbors, mst_result = minimum_k_from_mst(chord_dist, backend)
        component_count_from_mst = mst_result.component_count
    else:
        component_count_from_mst = None  # Will be determined from eigenvalues

    # Clamp k to valid range
    k_neighbors = max(1, min(k_neighbors, n - 1))

    # Build k-NN adjacency from chord distances with proper tie handling
    # Include ALL points at distance <= k-th nearest (handles duplicate points)
    inf_val = infinity_threshold(backend, chord_dist)
    eps = machine_epsilon(backend, chord_dist)
    self_mask = backend.eye(n) > 0.0
    dist_no_self = backend.where(self_mask, inf_val, chord_dist)

    # Find the k-th smallest distance for each row
    kth = max(0, min(k_neighbors - 1, n - 1))
    partitioned = backend.argpartition(dist_no_self, kth, axis=1)
    kth_indices = partitioned[:, kth : kth + 1]  # [n, 1]
    kth_distances = backend.take_along_axis(dist_no_self, kth_indices, axis=1)  # [n, 1]
    backend.eval(kth_distances)

    # Include all points with distance <= k-th distance (with small epsilon for ties)
    kth_dist_broadcast = backend.broadcast_to(kth_distances, (n, n))
    within_knn = dist_no_self <= (kth_dist_broadcast + eps)
    adj = backend.where(within_knn, chord_dist, inf_val)
    adj = adj * (1.0 - backend.eye(n))  # Zero diagonal

    # Symmetrize: edge exists if either direction qualifies
    adj_t = backend.transpose(adj)
    adj = backend.minimum(adj, adj_t)
    backend.eval(adj)

    # Compute kernel bandwidth from median POSITIVE neighbor distance
    # (Exclude zero distances from duplicate points and non-edges)
    inf_thresh = infinity_threshold(backend, adj)
    edge_dists = backend.where(adj < inf_thresh, adj, inf_val)
    kernel_bandwidth = _median_positive(edge_dists, backend)

    # If all neighbor distances are zero (duplicates), use fraction of max distance
    # This ensures bandwidth reflects the actual data scale
    if kernel_bandwidth <= 0.0:
        eps = machine_epsilon(backend, chord_dist)
        kernel_bandwidth = max(max_dist * eps, tiny_value(backend, chord_dist))

    # Build normalized graph Laplacian with heat kernel weights from CHORD distances
    inf_thresh = infinity_threshold(backend, adj)
    weights_dtype = precision_dtype(backend, reference=chord_dist)
    weights_arr = backend.zeros((n, n), dtype=weights_dtype)

    edge_mask = backend.where(
        adj < inf_thresh,
        backend.ones_like(adj),
        backend.zeros_like(adj),
    )
    edge_count_arr = backend.sum(backend.astype(edge_mask, "int32"))
    backend.eval(edge_count_arr)
    edge_count = int(backend.to_scalar(edge_count_arr))

    if edge_count > n:  # More than just diagonal
        sigma_sq = kernel_bandwidth * kernel_bandwidth * 2.0
        # Use CHORD distances for heat kernel weights - this is the key fix
        weights_arr = backend.exp(-(chord_dist * chord_dist) / sigma_sq)
        weights_arr = weights_arr * edge_mask
        weights_arr = weights_arr * (1.0 - backend.eye(n))
        weights_arr = backend.astype(weights_arr, weights_dtype)

    backend.eval(weights_arr)

    # Compute degree and normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    degree = backend.sum(weights_arr, axis=1)
    backend.eval(degree)

    eps = division_epsilon(backend, degree)
    safe_degree = backend.maximum(degree, eps)
    inv_sqrt = 1.0 / backend.sqrt(safe_degree)
    d_inv_sqrt = backend.diag(inv_sqrt)
    laplacian = backend.eye(n) - backend.matmul(
        d_inv_sqrt, backend.matmul(weights_arr, d_inv_sqrt)
    )
    backend.eval(laplacian)

    # Eigendecomposition (GPU-native)
    eigvals, eigvecs = power_iteration_eigh(backend, laplacian, k=n)
    backend.eval(eigvals, eigvecs)

    # Sort eigenvalues ascending (smallest first - we want the smallest non-zero)
    # power_iteration_eigh returns descending, so reverse
    n_eig = int(eigvals.shape[-1])
    rev_idx = backend.arange(n_eig - 1, -1, -1)
    eigvals = backend.take(eigvals, rev_idx, axis=-1)
    eigvecs_transposed = backend.transpose(eigvecs)
    eigvecs_rev = backend.take(eigvecs_transposed, rev_idx, axis=0)
    eigvecs = backend.transpose(eigvecs_rev)
    backend.eval(eigvals, eigvecs)

    # Count zero eigenvalues (connected components)
    eps = machine_epsilon(backend, eigvals)
    max_eig = float(backend.to_scalar(eigvals[-1]))
    zero_thresh = max(max_eig, eps) * sqrt_scalar(eps, backend)
    zero_mask = backend.abs(eigvals) < zero_thresh
    zero_count_arr = backend.sum(backend.astype(zero_mask, "int32"))
    backend.eval(zero_count_arr)
    component_count = max(1, int(backend.to_scalar(zero_count_arr)))

    # Store all eigenvalues for spectral signature computation
    all_eigenvalues = eigvals

    # Determine k_eigenvectors if not specified
    # Default: use ALL non-zero eigenvectors for best geodesic accuracy
    # (spectral gap was too aggressive, causing poor geodesic approximation)
    if k_eigenvectors is None:
        k_eigenvectors = n  # Use all eigenvectors

    k_eigenvectors = max(component_count, min(k_eigenvectors, n))

    # Build spectral embedding: Φ = eigenvectors / sqrt(eigenvalues)
    # Skip zero eigenvalues (constant eigenvector for each component)
    embedding, used_eigvals, used_eigvecs, k_actual = _build_embedding(
        eigvals, eigvecs, k_eigenvectors, component_count, backend
    )
    backend.eval(embedding)

    # Collapse identical points: eigenvectors for repeated eigenvalues can give
    # different values to identical points. Assign the same embedding to any
    # points with zero chord distance.
    embedding = _collapse_identical_points(embedding, chord_dist, backend)
    backend.eval(embedding)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SpectralEmbeddingResult(
        embedding=embedding,
        eigenvalues=used_eigvals,
        eigenvectors=used_eigvecs,
        all_eigenvalues=all_eigenvalues,
        k_used=k_actual,
        k_neighbors=k_neighbors,
        component_count=component_count,
        kernel_bandwidth=float(kernel_bandwidth),
        adjacency=adj,
        inf_value=inf_val,
        compute_time_ms=elapsed_ms,
    )


def geodesic_distances_from_embedding(
    embedding: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Compute pairwise geodesic distances from spectral embedding.

    In the spectral embedding space, geodesic distance equals Euclidean distance:
        d_geodesic(i, j) = ||embedding[i] - embedding[j]||₂

    Args:
        embedding: Spectral embedding [n, k] from compute_spectral_embedding.
        backend: Backend for tensor operations.

    Returns:
        Pairwise geodesic distance matrix [n, n].
    """
    backend = backend or get_default_backend()
    n = int(embedding.shape[0])

    if n == 0:
        return backend.zeros((0, 0))

    if n == 1:
        return backend.zeros((1, 1))

    # Compute pairwise Euclidean distances in embedding space
    # ||a - b||² = ||a||² + ||b||² - 2<a,b>
    norms_sq = backend.sum(embedding * embedding, axis=1)
    norms_col = backend.reshape(norms_sq, (n, 1))
    norms_row = backend.reshape(norms_sq, (1, n))
    cross = backend.matmul(embedding, backend.transpose(embedding))

    dist_sq = norms_col + norms_row - 2.0 * cross

    # Numerical cleanup
    dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
    distances = backend.sqrt(dist_sq)

    # Zero diagonal
    eye = backend.eye(n)
    distances = distances * (1.0 - eye)
    backend.eval(distances)

    return distances


def _determine_k_from_gap(
    eigenvalues: "Array",
    backend: "Backend",
    component_count: int,
) -> int:
    """Determine number of eigenvectors from spectral gap.

    Uses magnitude gap detection to find where eigenvalues jump,
    indicating the intrinsic dimension of the manifold.
    """
    eig_list = backend.tolist(eigenvalues)
    n = len(eig_list)

    if n <= component_count + 1:
        return n

    # Skip zero eigenvalues
    positive = [val for val in eig_list[component_count:] if val > 0]
    if not positive:
        return component_count + 1

    # Find the spectral gap
    threshold = find_magnitude_gap_threshold(positive, backend=backend)

    # Count eigenvalues below the gap
    k = component_count
    for val in positive:
        if val <= threshold:
            k += 1
        else:
            break

    # Reasonable bounds
    return max(component_count + 1, min(k, n))


def _build_embedding(
    eigenvalues: "Array",
    eigenvectors: "Array",
    k_eigenvectors: int,
    component_count: int,
    backend: "Backend",
) -> tuple["Array", "Array", "Array", int]:
    """Build the spectral embedding from eigenvectors.

    The embedding is: Φ = eigenvectors[:, c:k] / sqrt(eigenvalues[c:k])
    where c = component_count (skip zero eigenvalues).
    """
    n = int(eigenvectors.shape[0])

    # Skip zero eigenvalues (one per component)
    start_idx = component_count
    end_idx = min(k_eigenvectors, n)

    if start_idx >= end_idx:
        # No non-zero eigenvalues - degenerate case
        return (
            backend.zeros((n, 1)),
            backend.zeros((1,)),
            backend.zeros((n, 1)),
            1,
        )

    # Extract non-zero eigenvalues and their eigenvectors
    indices = backend.arange(start_idx, end_idx)
    used_eigvals = backend.take(eigenvalues, indices, axis=-1)
    used_eigvecs = eigenvectors[:, start_idx:end_idx]
    backend.eval(used_eigvals, used_eigvecs)

    k_actual = end_idx - start_idx

    # Build embedding: Φ = V / sqrt(λ)
    eps = division_epsilon(backend, used_eigvals)
    safe_eigvals = backend.maximum(used_eigvals, eps)
    scale = 1.0 / backend.sqrt(safe_eigvals)
    scale_row = backend.reshape(scale, (1, k_actual))

    embedding = used_eigvecs * scale_row
    backend.eval(embedding)

    return embedding, used_eigvals, used_eigvecs, k_actual


def _empty_embedding(backend: "Backend", k_neighbors: int, elapsed_ms: float) -> SpectralEmbeddingResult:
    """Create result for empty point set."""
    return SpectralEmbeddingResult(
        embedding=backend.zeros((0, 0)),
        eigenvalues=backend.zeros((0,)),
        eigenvectors=backend.zeros((0, 0)),
        all_eigenvalues=backend.zeros((0,)),
        k_used=0,
        k_neighbors=k_neighbors,
        component_count=0,
        kernel_bandwidth=0.0,
        adjacency=backend.zeros((0, 0)),
        inf_value=float("inf"),
        compute_time_ms=elapsed_ms,
    )


def _single_point_embedding(
    backend: "Backend",
    points: "Array",
    elapsed_ms: float,
) -> SpectralEmbeddingResult:
    """Create result for single point."""
    return SpectralEmbeddingResult(
        embedding=backend.zeros((1, 1)),
        eigenvalues=backend.zeros((1,)),
        eigenvectors=backend.ones((1, 1)),
        all_eigenvalues=backend.zeros((1,)),
        k_used=1,
        k_neighbors=0,
        component_count=1,
        kernel_bandwidth=0.0,
        adjacency=backend.zeros((1, 1)),
        inf_value=float("inf"),
        compute_time_ms=elapsed_ms,
    )


def _median_flattened(values: "Array", backend: "Backend") -> float:
    """Compute median of flattened array."""
    flat = backend.reshape(values, (-1,))
    count = int(flat.shape[0])
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2 == 1:
        part = backend.argpartition(flat, mid)
        prefix = backend.take(part, backend.arange(mid + 1), axis=0)
        mid_val = backend.max(backend.take(flat, prefix, axis=0))
        backend.eval(mid_val)
        return float(backend.to_scalar(mid_val))
    low_part = backend.argpartition(flat, mid - 1)
    low_prefix = backend.take(low_part, backend.arange(mid), axis=0)
    low_val = backend.max(backend.take(flat, low_prefix, axis=0))
    high_part = backend.argpartition(flat, mid)
    high_prefix = backend.take(high_part, backend.arange(mid + 1), axis=0)
    high_val = backend.max(backend.take(flat, high_prefix, axis=0))
    low_val = backend.squeeze(low_val)
    high_val = backend.squeeze(high_val)
    backend.eval(low_val, high_val)
    return 0.5 * (float(backend.to_scalar(low_val)) + float(backend.to_scalar(high_val)))


def _collapse_identical_points(
    embedding: "Array",
    chord_dist: "Array",
    backend: "Backend",
) -> "Array":
    """Collapse identical points to have the same embedding.

    When points have zero chord distance (are geometrically identical),
    eigenvectors for repeated eigenvalues may give them different values.
    This function assigns the mean embedding to all points in each
    equivalence class.
    """
    n = int(embedding.shape[0])
    if n <= 1:
        return embedding

    eps = machine_epsilon(backend, chord_dist)

    # Find equivalence classes via Union-Find on chord_dist = 0
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build equivalence classes from zero-distance pairs
    backend.eval(chord_dist)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = float(backend.to_scalar(chord_dist[i, j]))
            if d_ij <= eps:
                union(i, j)

    # Group points by their equivalence class
    classes: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in classes:
            classes[root] = []
        classes[root].append(i)

    # Check if any class has more than one point
    has_duplicates = any(len(members) > 1 for members in classes.values())
    if not has_duplicates:
        return embedding

    # Collapse each class to its mean embedding
    result = embedding
    for members in classes.values():
        if len(members) <= 1:
            continue

        # Compute mean embedding for this class
        member_indices = backend.array(members, dtype="int32")
        member_embeddings = backend.take(embedding, member_indices, axis=0)
        mean_embedding = backend.mean(member_embeddings, axis=0)
        mean_embedding = backend.reshape(mean_embedding, (1, -1))
        backend.eval(mean_embedding)

        # Assign mean to all members
        for idx in members:
            # Use scatter update pattern
            result = _replace_row(result, idx, mean_embedding, backend)

    backend.eval(result)
    return result


def _replace_row(
    matrix: "Array",
    row_idx: int,
    new_row: "Array",
    backend: "Backend",
) -> "Array":
    """Replace a single row in a matrix."""
    n = int(matrix.shape[0])
    k = int(matrix.shape[1])

    # Create mask for the row to replace
    row_mask = backend.arange(n) == row_idx
    row_mask = backend.reshape(row_mask, (n, 1))
    row_mask = backend.broadcast_to(row_mask, (n, k))

    # Broadcast new_row to full matrix shape
    new_row_broadcast = backend.broadcast_to(new_row, (n, k))

    # Select new_row where mask is True, else keep original
    result = backend.where(row_mask, new_row_broadcast, matrix)
    return result


def _median_positive(values: "Array", backend: "Backend") -> float:
    """Compute median of positive values only (excluding zeros).

    Returns 0.0 if no positive values exist.
    """
    flat = backend.reshape(values, (-1,))
    backend.eval(flat)

    # Filter to positive values
    eps = machine_epsilon(backend, flat)
    positive_mask = flat > eps
    positive_count_arr = backend.sum(backend.astype(positive_mask, "int32"))
    backend.eval(positive_count_arr)
    count = int(backend.to_scalar(positive_count_arr))

    if count == 0:
        return 0.0

    # Replace non-positive with inf, then find median via sort
    inf_val = infinity_threshold(backend, flat) * 2.0
    masked = backend.where(positive_mask, flat, inf_val)
    sorted_vals = backend.sort(masked)
    backend.eval(sorted_vals)

    # Median of the positive values (first `count` elements after sort)
    mid = count // 2
    if count % 2 == 1:
        median_arr = sorted_vals[mid]
    else:
        median_arr = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    backend.eval(median_arr)

    return float(backend.to_scalar(median_arr))


def _chord_distance_matrix(points: "Array", backend: "Backend") -> "Array":
    """Compute pairwise Euclidean (chord) distance matrix.

    Args:
        points: Point cloud [n, d].
        backend: Backend for tensor operations.

    Returns:
        Pairwise distance matrix [n, n].
    """
    n = int(points.shape[0])
    if n == 0:
        return backend.zeros((0, 0))

    # ||a - b||² = ||a||² + ||b||² - 2<a,b>
    norms_sq = backend.sum(points * points, axis=1)
    norms_col = backend.reshape(norms_sq, (n, 1))
    norms_row = backend.reshape(norms_sq, (1, n))
    cross = backend.matmul(points, backend.transpose(points))

    dist_sq = norms_col + norms_row - 2.0 * cross

    # Numerical cleanup
    dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
    distances = backend.sqrt(dist_sq)

    # Zero diagonal
    distances = distances * (1.0 - backend.eye(n))
    backend.eval(distances)

    return distances
