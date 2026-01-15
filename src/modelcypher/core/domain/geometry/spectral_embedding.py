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

The key insight (Varadhan's formula): Laplacian eigenvectors define a spectral
embedding where geodesic distance becomes Euclidean distance:

    Φ: x → (φ₁(x)/√λ₁, φ₂(x)/√λ₂, ..., φₖ(x)/√λₖ)
    d_geodesic(x, y) = ‖Φ(x) - Φ(y)‖₂

One eigendecomposition produces both:
- Geodesic distances (as Euclidean distance in embedding space)
- Spectral signature (eigenvalues for heat trace, entropy, etc.)

This eliminates the redundant Floyd-Warshall computation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    infinity_threshold,
    machine_epsilon,
    power_iteration_eigh,
    precision_dtype,
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

    # Build k-NN adjacency using geodesic distances
    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    rg = RiemannianGeometry(backend)

    # Get geodesic result to reuse k-NN structure
    # This uses Floyd-Warshall for the initial geodesic computation,
    # but we only do it once, then use the spectral embedding for distances
    geo_result = rg.geodesic_distances(points_arr, k_neighbors=k_neighbors)
    geodesic_dist = geo_result.distances
    adj = geo_result.adjacency
    inf_val = geo_result.inf_value
    k_neighbors = geo_result.k_neighbors
    backend.eval(geodesic_dist, adj)

    # Build neighbor indices from adjacency
    self_mask = backend.eye(n) > 0.0
    dist_no_self = backend.where(self_mask, inf_val, geodesic_dist)
    kth = max(0, min(k_neighbors - 1, n - 1))
    neighbor_indices = backend.argpartition(dist_no_self, kth, axis=1)[:, :k_neighbors]
    backend.eval(neighbor_indices)

    # Compute kernel bandwidth from median neighbor distance
    neighbor_dists = backend.take(geodesic_dist, neighbor_indices, axis=1)
    backend.eval(neighbor_dists)
    kernel_bandwidth = _median_flattened(neighbor_dists, backend)

    bandwidth_floor = tiny_value(backend, geodesic_dist)
    if kernel_bandwidth <= bandwidth_floor:
        kernel_bandwidth = bandwidth_floor

    # Build normalized graph Laplacian with heat kernel weights
    inf_thresh = infinity_threshold(backend, adj)
    weights_dtype = precision_dtype(backend, reference=geodesic_dist)
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
        weights_arr = backend.exp(-(geodesic_dist * geodesic_dist) / sigma_sq)
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
    zero_thresh = machine_epsilon(backend, eigvals) * 10.0
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
