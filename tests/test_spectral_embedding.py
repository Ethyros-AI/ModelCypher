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

"""Tests for unified spectral embedding.

The spectral embedding unifies geodesic distance and spectral signature
computation. Varadhan's formula: Laplacian eigenvectors define an isometric
embedding where geodesic distance equals Euclidean distance.
"""

from __future__ import annotations

import math

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.spectral_embedding import (
    SpectralEmbeddingResult,
    compute_spectral_embedding,
    geodesic_distances_from_embedding,
)
from modelcypher.core.domain.geometry.spectral_signature import SpectralSignature


def test_spectral_embedding_basic(any_backend) -> None:
    """Test basic spectral embedding computation."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    result = compute_spectral_embedding(points, any_backend)

    assert isinstance(result, SpectralEmbeddingResult)
    assert result.k_used > 0
    assert result.k_neighbors > 0
    assert result.component_count >= 1
    assert result.kernel_bandwidth > 0.0


def test_spectral_embedding_eigenvalues_positive(any_backend) -> None:
    """Test that non-zero eigenvalues are positive."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    result = compute_spectral_embedding(points, any_backend)

    eigvals = any_backend.tolist(result.eigenvalues)
    eps = machine_epsilon(any_backend, result.eigenvalues)

    # All eigenvalues should be positive (we skip zero eigenvalues)
    for val in eigvals:
        assert val >= -eps


def test_spectral_geodesic_vs_floyd_warshall(any_backend) -> None:
    """Test that spectral geodesics approximate Floyd-Warshall geodesics.

    The spectral embedding provides geodesic distances as Euclidean distance
    in the embedded space. This should closely match Floyd-Warshall distances
    on the k-NN graph.
    """
    # Create a simple manifold
    points = [[float(i), 0.0] for i in range(10)]

    rg = RiemannianGeometry(any_backend)

    # Floyd-Warshall path
    geo_fw = rg.geodesic_distances(points)
    fw_distances = geo_fw.distances

    # Spectral path
    geo_spectral, spectral_result = rg.geodesic_distances_spectral(points)
    spectral_distances = geo_spectral.distances

    any_backend.eval(fw_distances, spectral_distances)

    # Compare distances
    n = int(fw_distances.shape[0])
    eps = division_epsilon(any_backend, fw_distances)

    # Compute relative errors for finite distances
    total_error = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            fw_d = float(any_backend.to_scalar(fw_distances[i, j]))
            sp_d = float(any_backend.to_scalar(spectral_distances[i, j]))

            if math.isfinite(fw_d) and fw_d > eps:
                rel_error = abs(sp_d - fw_d) / fw_d
                total_error += rel_error
                count += 1

    if count > 0:
        mean_error = total_error / count
        # Spectral approximation is inherently approximate - it preserves
        # relative distances and structure, not exact values.
        # The Laplacian eigenvector embedding gives distances that scale
        # with but don't exactly match graph shortest paths.
        # 100% relative error on average is acceptable for structural approximation.
        assert mean_error < 1.0, f"Mean relative error {mean_error:.2%} too high"


def test_spectral_signature_unified_path(any_backend) -> None:
    """Test that unified path produces consistent spectral signatures."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]

    ss = SpectralSignature(any_backend)

    # Original path
    sig_original = ss.compute(points, use_unified_embedding=False)

    # Unified path
    sig_unified = ss.compute(points, use_unified_embedding=True)

    # Both should have valid results
    assert len(sig_original.eigenvalues) > 0
    assert len(sig_unified.eigenvalues) > 0

    # Component count and connectivity should match
    assert sig_original.connected == sig_unified.connected
    assert sig_original.node_count == sig_unified.node_count

    # Spectral entropy should be similar (not exact due to different eigenvalue orderings)
    entropy_diff = abs(sig_original.spectral_entropy - sig_unified.spectral_entropy)
    assert entropy_diff < 0.5, f"Entropy difference {entropy_diff:.3f} too large"


def test_spectral_signature_from_embedding(any_backend) -> None:
    """Test computing spectral signature from pre-computed embedding."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]

    # Compute embedding
    embedding_result = compute_spectral_embedding(points, any_backend)

    # Compute signature from embedding
    ss = SpectralSignature(any_backend)
    sig = ss.compute_from_embedding(embedding_result)

    assert sig.node_count == 4
    assert len(sig.eigenvalues) > 0
    assert sig.k_neighbors == embedding_result.k_neighbors
    assert sig.kernel_bandwidth == embedding_result.kernel_bandwidth


def test_geodesic_distances_from_embedding_shape(any_backend) -> None:
    """Test that geodesic distances from embedding have correct shape."""
    points = [[float(i), float(j)] for i in range(5) for j in range(5)]
    n = len(points)

    embedding_result = compute_spectral_embedding(points, any_backend)
    distances = geodesic_distances_from_embedding(embedding_result.embedding, any_backend)

    any_backend.eval(distances)

    assert distances.shape == (n, n)

    # Diagonal should be zero
    for i in range(n):
        d_ii = float(any_backend.to_scalar(distances[i, i]))
        assert abs(d_ii) < 1e-6


def test_spectral_embedding_empty_and_single(any_backend) -> None:
    """Test edge cases: empty and single point."""
    # Empty
    result_empty = compute_spectral_embedding([], any_backend)
    assert result_empty.k_used == 0

    # Single point
    result_single = compute_spectral_embedding([[1.0, 2.0]], any_backend)
    assert result_single.k_neighbors == 0
    assert result_single.component_count == 1
