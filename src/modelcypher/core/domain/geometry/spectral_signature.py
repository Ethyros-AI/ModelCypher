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

"""Graph spectral signatures for point cloud manifolds.

Builds a k-NN graph Laplacian from pairwise distances and reports raw spectral
metrics (eigenvalues, heat trace, entropy) without interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    find_magnitude_gap_threshold,
    infinity_threshold,
    power_iteration_eigh,
    precision_dtype,
    regularization_epsilon,
    safe_log_epsilon,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class SpectralSignatureResult:
    """Raw spectral signature metrics."""

    eigenvalues: list[float]
    heat_trace: list[float]
    heat_times: list[float]
    spectral_entropy: float
    algebraic_connectivity: float
    component_count: int
    node_count: int
    edge_count: int
    k_neighbors: int
    kernel_bandwidth: float
    normalized_laplacian: bool
    connected: bool


class SpectralSignature:
    """Compute graph spectral signatures for point clouds."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_from_embedding(
        self,
        embedding_result: "Any",
    ) -> SpectralSignatureResult:
        """Compute spectral signature from a pre-computed spectral embedding.

        This is the efficient path when you already have a SpectralEmbeddingResult
        (e.g., from geodesic_distances_spectral). The eigenvalues are reused
        directly, avoiding redundant eigendecomposition.

        Args:
            embedding_result: SpectralEmbeddingResult from compute_spectral_embedding
                or geodesic_distances_spectral.

        Returns:
            SpectralSignatureResult with spectral metrics.
        """
        backend = self._backend

        # Extract data from embedding result
        eigenvalues = embedding_result.eigenvalues
        adjacency = embedding_result.adjacency
        k_neighbors = embedding_result.k_neighbors
        kernel_bandwidth = embedding_result.kernel_bandwidth
        component_count = embedding_result.component_count

        backend.eval(eigenvalues, adjacency)

        n = int(adjacency.shape[0])
        if n == 0:
            return SpectralSignatureResult(
                eigenvalues=[],
                heat_trace=[],
                heat_times=[],
                spectral_entropy=0.0,
                algebraic_connectivity=0.0,
                component_count=0,
                node_count=0,
                edge_count=0,
                k_neighbors=0,
                kernel_bandwidth=0.0,
                normalized_laplacian=True,
                connected=True,
            )

        # Count edges
        inf_thresh = infinity_threshold(backend, adjacency)
        edge_mask = adjacency < inf_thresh
        edge_count_total_arr = backend.sum(backend.astype(edge_mask, "int32"))
        backend.eval(edge_count_total_arr)
        edge_count_total = int(backend.to_scalar(edge_count_total_arr))
        edge_count = max(0, (edge_count_total - n) // 2)

        # Use all_eigenvalues which includes zero eigenvalues
        full_eigvals = embedding_result.all_eigenvalues
        backend.eval(full_eigvals)
        eig_list = [float(x) for x in backend.tolist(full_eigvals)]

        # Derive heat trace times from eigenvalue spectrum
        heat_times = _derive_heat_times_from_spectrum(eig_list, backend)

        spectral_entropy = _spectral_entropy(
            backend, full_eigvals, regularization_epsilon(backend, full_eigvals)
        )
        algebraic_connectivity = eig_list[component_count] if len(eig_list) > component_count else 0.0
        connected = component_count == 1

        heat_trace = _heat_trace(backend, full_eigvals, heat_times)

        return SpectralSignatureResult(
            eigenvalues=eig_list,
            heat_trace=heat_trace,
            heat_times=heat_times,
            spectral_entropy=spectral_entropy,
            algebraic_connectivity=algebraic_connectivity,
            component_count=component_count,
            node_count=n,
            edge_count=edge_count,
            k_neighbors=k_neighbors,
            kernel_bandwidth=float(kernel_bandwidth),
            normalized_laplacian=True,
            connected=connected,
        )

    def compute(
        self,
        points: list[list[float]],
        use_unified_embedding: bool = False,
    ) -> SpectralSignatureResult:
        """Compute spectral signature for a point cloud.

        All parameters are derived from the geometry of the data:
        - k_neighbors: derived from the k-distance elbow
        - kernel_bandwidth: derived from median neighbor distance
        - heat_trace_times: derived from magnitude gaps in the spectrum
        - normalized_laplacian: canonical for graph Laplacians

        Args:
            points: Point cloud as list of lists.
            use_unified_embedding: If True, use the unified spectral embedding
                which shares computation with geodesic distances.
        """
        backend = self._backend
        points_arr = backend.array(points)
        backend.eval(points_arr)

        # Unified path: use spectral embedding (shares eigendecomposition)
        if use_unified_embedding:
            from modelcypher.core.domain.geometry.spectral_embedding import (
                compute_spectral_embedding,
            )
            embedding_result = compute_spectral_embedding(points_arr, backend)
            return self.compute_from_embedding(embedding_result)

        n = int(points_arr.shape[0]) if len(points_arr.shape) > 0 else 0

        # Edge cases: no spectrum to derive times from
        if n == 0:
            return SpectralSignatureResult(
                eigenvalues=[],
                heat_trace=[],
                heat_times=[],
                spectral_entropy=0.0,
                algebraic_connectivity=0.0,
                component_count=0,
                node_count=0,
                edge_count=0,
                k_neighbors=0,
                kernel_bandwidth=0.0,
                normalized_laplacian=True,
                connected=True,
            )

        if n == 1:
            return SpectralSignatureResult(
                eigenvalues=[0.0],
                heat_trace=[],
                heat_times=[],
                spectral_entropy=0.0,
                algebraic_connectivity=0.0,
                component_count=1,
                node_count=1,
                edge_count=0,
                k_neighbors=0,
                kernel_bandwidth=0.0,
                normalized_laplacian=True,
                connected=True,
            )

        # Build a k-NN graph using geodesic distances on the manifold.
        adjacency, geodesic_dist, inf_value, k_neighbors, neighbor_indices = (
            self._build_knn_adjacency(points_arr, None)
        )
        backend.eval(adjacency, geodesic_dist, neighbor_indices)

        # Use dtype-derived threshold (not arbitrary 0.9)
        inf_thresh = infinity_threshold(backend, adjacency)
        edge_mask = adjacency < inf_thresh
        edge_count_total_arr = backend.sum(backend.astype(edge_mask, "int32"))
        backend.eval(edge_count_total_arr)
        edge_count_total = int(backend.to_scalar(edge_count_total_arr))
        edge_count = max(0, (edge_count_total - n) // 2)

        neighbor_dists = backend.take(geodesic_dist, neighbor_indices, axis=1)
        backend.eval(neighbor_dists)

        # kernel_bandwidth derived from median neighbor distance
        kernel_bandwidth = compute_median(neighbor_dists, backend)

        bandwidth_floor = tiny_value(backend, geodesic_dist)
        if kernel_bandwidth <= bandwidth_floor:
            kernel_bandwidth = bandwidth_floor

        weights_dtype = precision_dtype(backend, reference=geodesic_dist)
        weights_arr = backend.zeros((n, n), dtype=weights_dtype)
        if edge_count > 0:
            sigma_sq = kernel_bandwidth * kernel_bandwidth * 2.0
            edge_mask = backend.where(
                adjacency < inf_thresh,
                backend.ones_like(adjacency),
                backend.zeros_like(adjacency),
            )
            weights_arr = backend.exp(-(geodesic_dist * geodesic_dist) / sigma_sq)
            weights_arr = weights_arr * edge_mask
            weights_arr = weights_arr * (1.0 - backend.eye(n))
            weights_arr = backend.astype(weights_arr, weights_dtype)

        backend.eval(weights_arr)
        degree = backend.sum(weights_arr, axis=1)
        backend.eval(degree)

        laplacian = self._build_laplacian(weights_arr, degree, normalized=True)
        backend.eval(laplacian)

        # Geodesic eigendecomposition (GPU-only)
        n_lap = int(laplacian.shape[0])
        eigvals, _ = power_iteration_eigh(backend, laplacian, k=n_lap)
        backend.eval(eigvals)
        eig_sorted = backend.sort(eigvals)
        backend.eval(eig_sorted)
        eig_list = [float(x) for x in backend.tolist(eig_sorted)]

        # Derive heat trace times from eigenvalue spectrum
        heat_times = _derive_heat_times_from_spectrum(eig_list, backend)

        spectral_entropy = _spectral_entropy(backend, eigvals, regularization_epsilon(backend, eigvals))
        algebraic_connectivity = eig_list[1] if len(eig_list) > 1 else 0.0
        neighbor_indices_list = backend.tolist(neighbor_indices)
        component_count = _count_components_from_neighbors(neighbor_indices_list, n)
        connected = component_count == 1

        heat_trace = _heat_trace(backend, eigvals, heat_times)

        return SpectralSignatureResult(
            eigenvalues=eig_list,
            heat_trace=heat_trace,
            heat_times=heat_times,
            spectral_entropy=spectral_entropy,
            algebraic_connectivity=algebraic_connectivity,
            component_count=component_count,
            node_count=n,
            edge_count=edge_count,
            k_neighbors=k_neighbors,
            kernel_bandwidth=float(kernel_bandwidth),
            normalized_laplacian=True,
            connected=connected,
        )

    def _build_laplacian(
        self,
        weights: "Array",
        degree: "Array",
        *,
        normalized: bool,
    ) -> "Array":
        backend = self._backend
        n = int(weights.shape[0])
        if normalized:
            eps = division_epsilon(backend, degree)
            safe_degree = backend.maximum(degree, eps)
            inv_sqrt = 1.0 / backend.sqrt(safe_degree)
            d_inv_sqrt = backend.diag(inv_sqrt)
            return backend.eye(n) - backend.matmul(d_inv_sqrt, backend.matmul(weights, d_inv_sqrt))
        diag = backend.diag(degree)
        return diag - weights

    def _build_knn_adjacency(
        self,
        points: "Array",
        k_neighbors: int | None,
        mutual_knn: bool = False,
    ) -> tuple["Array", "Array", float, int, "Array"]:
        """Build k-NN adjacency matrix from point cloud.

        Args:
            points: Point cloud array [n, d].
            k_neighbors: Number of neighbors (auto-detected if None).
            mutual_knn: If True, use mutual k-NN (edge only if both i→j and j→i).
                This is stricter and reduces shortcut edges in sparse manifolds.

        Returns:
            Tuple of (adjacency, geodesic_dist, inf_value, k_neighbors, neighbor_indices).
        """
        backend = self._backend
        n = int(points.shape[0])
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k_neighbors)
        geodesic_dist = geo_result.distances
        backend.eval(geodesic_dist)
        inf_val = geo_result.inf_value
        k_neighbors = geo_result.k_neighbors

        # Use geodesic distances for neighbor selection.
        self_mask = backend.eye(n) > 0.0
        dist_no_self = backend.where(self_mask, inf_val, geodesic_dist)
        kth = max(0, min(k_neighbors - 1, n - 1))
        neighbor_indices = backend.argpartition(dist_no_self, kth, axis=1)[:, :k_neighbors]
        backend.eval(neighbor_indices)

        adj = backend.full((n, n), inf_val)
        adj = backend.where(self_mask, backend.zeros_like(adj), adj)

        edge_eps = float(division_epsilon(backend, geodesic_dist))
        edge_eps_arr = backend.full(geodesic_dist.shape, edge_eps)
        weights = backend.maximum(geodesic_dist, edge_eps_arr)

        # Vectorized adjacency construction using put_along_axis
        # Gather weights at neighbor positions: [n, k]
        neighbor_weights = backend.take_along_axis(weights, neighbor_indices, axis=1)
        # Put neighbor weights into adjacency matrix at correct positions
        adj = backend.put_along_axis(adj, neighbor_indices, neighbor_weights, axis=1)

        # Symmetrization
        inf_thresh = infinity_threshold(backend, adj)
        adj_t = backend.transpose(adj)
        if mutual_knn:
            # Mutual k-NN: edge exists only if both i→j AND j→i
            # This reduces shortcut edges in sparse/holed manifolds
            mutual_mask = (adj < inf_thresh) & (adj_t < inf_thresh)
            adj = backend.where(mutual_mask, backend.minimum(adj, adj_t), inf_val)
        else:
            # Union k-NN: edge exists if i→j OR j→i (standard)
            adj = backend.minimum(adj, adj_t)

        return adj, geodesic_dist, inf_val, k_neighbors, neighbor_indices


def _derive_heat_times_from_spectrum(
    eigenvalues: list[float], backend: "Backend"
) -> list[float]:
    """Derive heat trace times from the eigenvalue spectrum.

    Uses 1/λ for significant eigenvalues, selecting the cutoff via
    magnitude-gap detection (data-derived, no fixed time grid).
    """
    if not eigenvalues:
        return []

    eig_arr = backend.array(eigenvalues or [0.0])
    eps = division_epsilon(backend, eig_arr)

    positive = [val for val in eigenvalues if val > eps]
    if not positive:
        return []

    positive_sorted = sorted(positive)
    threshold = find_magnitude_gap_threshold(positive_sorted, backend=backend)
    significant = [val for val in positive_sorted if val > threshold]
    if not significant:
        significant = positive_sorted

    times = [1.0 / val for val in significant]
    return sorted(times)


def _spectral_entropy(backend: "Backend", eigenvalues: "Array", eps: float) -> float:
    total = backend.sum(eigenvalues)
    backend.eval(total)
    total_val = float(backend.to_scalar(total))
    if total_val <= eps:
        return 0.0
    probs = eigenvalues / total
    # Use safe_log_epsilon for log clamping (smaller than division epsilon)
    log_eps = safe_log_epsilon(backend, probs)
    log_probs = backend.where(
        probs > log_eps, backend.log(probs), backend.zeros_like(probs)
    )
    entropy_arr = -backend.sum(probs * log_probs)
    backend.eval(entropy_arr)
    return float(backend.to_scalar(entropy_arr))


def _heat_trace(backend: "Backend", eigenvalues: "Array", times: list[float]) -> list[float]:
    if not times:
        return []
    times_arr = backend.array(times)
    eig_row = backend.reshape(eigenvalues, (1, -1))
    times_col = backend.reshape(times_arr, (-1, 1))
    heat = backend.sum(backend.exp(-times_col * eig_row), axis=1)
    backend.eval(heat)
    return [float(x) for x in backend.tolist(heat)]


def _count_components_from_neighbors(neighbors: list[list[int]], n: int) -> int:
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = [False] * n
    components = 0
    for i in range(n):
        if visited[i]:
            continue
        components += 1
        stack = [i]
        visited[i] = True
        while stack:
            node = stack.pop()
            for j in adjacency[node]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
    return components


# =============================================================================
# Heat Kernel Signature (HKS)
# =============================================================================


@dataclass(frozen=True)
class HeatKernelSignatureResult:
    """Result of Heat Kernel Signature computation.

    The Heat Kernel Signature is a coordinate-invariant shape descriptor
    that captures multi-scale geometry at each point. It's derived from
    the eigendecomposition of the Laplace-Beltrami operator.

    HKS(x, t) = Σᵢ exp(-λᵢt) φᵢ(x)²

    where λᵢ are eigenvalues and φᵢ are eigenvectors of the Laplacian.

    Different time scales capture different structural features:
    - Small t: local geometry (fine details)
    - Large t: global geometry (overall shape)

    The HKS is invariant to isometric transformations, making it ideal
    for cross-model comparison where coordinate systems differ.

    References:
        - Sun, J., Ovsjanikov, M., Guibas, L. (2009). "A Concise and
          Provably Informative Multi-Scale Signature Based on Heat Diffusion."
          Computer Graphics Forum (SGP).

    Attributes:
        signatures: HKS values at each point and time [n_points, n_times].
        times: Time scales used for computation [n_times].
        eigenvalues: Laplacian eigenvalues used [k].
        t_min: Minimum time scale (derived from λ_max).
        t_max: Maximum time scale (derived from λ_2).
    """

    signatures: "Array"
    times: "Array"
    eigenvalues: "Array"
    t_min: float
    t_max: float


class HeatKernelSignature:
    """Compute Heat Kernel Signatures for point cloud manifolds.

    HKS provides a coordinate-invariant descriptor for each point that
    captures multi-scale geometric information. Two points with similar
    HKS profiles are geometrically similar across all scales.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(
        self,
        points: "Array",
        n_times: int = 10,
        k_eigenvalues: int | None = None,
    ) -> HeatKernelSignatureResult:
        """Compute Heat Kernel Signature for all points.

        The time range is automatically derived from the eigenvalue spectrum:
        - t_min = 4 * log(10) / λ_max (captures local structure)
        - t_max = 4 * log(10) / λ_2 (captures global structure)

        Times are logarithmically spaced in this range.

        Args:
            points: Point cloud [n, d].
            n_times: Number of time scales (default 10).
            k_eigenvalues: Number of eigenvalues to use (default: all).

        Returns:
            HeatKernelSignatureResult with per-point signatures.
        """
        b = self._backend

        points = b.array(points) if not hasattr(points, "shape") else points
        b.eval(points)

        n = int(b.shape(points)[0])
        if n < 2:
            empty_sig = b.zeros((n, n_times)) if n > 0 else b.zeros((0, n_times))
            empty_times = b.zeros((n_times,))
            empty_eigs = b.zeros((0,))
            return HeatKernelSignatureResult(
                signatures=empty_sig,
                times=empty_times,
                eigenvalues=empty_eigs,
                t_min=0.0,
                t_max=0.0,
            )

        # Use spectral embedding to get eigendecomposition
        from modelcypher.core.domain.geometry.spectral_embedding import (
            compute_spectral_embedding,
        )

        embedding_result = compute_spectral_embedding(points, b)

        # Extract eigenvalues and eigenvectors
        # Use eigenvalues that match the eigenvectors (embedding.eigenvalues)
        # all_eigenvalues has all n eigenvalues, but eigenvectors is [n, k_used]
        eigenvalues = embedding_result.eigenvalues  # [k_used], matches eigenvectors
        eigenvectors = embedding_result.eigenvectors  # [n, k_used]
        b.eval(eigenvalues, eigenvectors)

        # Limit to k_eigenvalues if specified
        k = int(eigenvalues.shape[0])
        if k_eigenvalues is not None:
            k = min(k, k_eigenvalues)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
            b.eval(eigenvalues, eigenvectors)

        if k == 0:
            empty_sig = b.zeros((n, n_times))
            empty_times = b.zeros((n_times,))
            return HeatKernelSignatureResult(
                signatures=empty_sig,
                times=empty_times,
                eigenvalues=eigenvalues,
                t_min=0.0,
                t_max=0.0,
            )

        # Derive time range from spectrum
        # t_min from λ_max: captures local geometry
        # t_max from λ_2 (smallest non-zero): captures global geometry
        eps = division_epsilon(b, eigenvalues)

        # Find λ_max (largest eigenvalue)
        lambda_max_arr = b.max(eigenvalues)
        b.eval(lambda_max_arr)
        lambda_max = float(b.to_scalar(lambda_max_arr))

        # Find λ_2 (smallest positive eigenvalue - algebraic connectivity)
        positive_mask = eigenvalues > eps
        masked_eigs = b.where(
            positive_mask,
            eigenvalues,
            b.full(eigenvalues.shape, float("inf")),
        )
        lambda_2_arr = b.min(masked_eigs)
        b.eval(lambda_2_arr)
        lambda_2 = float(b.to_scalar(lambda_2_arr))

        if lambda_max <= eps or lambda_2 == float("inf"):
            # Degenerate spectrum
            empty_sig = b.zeros((n, n_times))
            empty_times = b.zeros((n_times,))
            return HeatKernelSignatureResult(
                signatures=empty_sig,
                times=empty_times,
                eigenvalues=eigenvalues,
                t_min=0.0,
                t_max=0.0,
            )

        # Time range: 4 * log(10) / λ ensures exp(-λt) decays appropriately
        # The factor 4 × log(10) ≈ 9.21 means at t_min, the fastest mode
        # (λ_max) has decayed by exp(-4×log(10)) = 10^{-4}, and at t_max,
        # the Fiedler mode (λ_2) has decayed by the same factor.
        # This spans 4 decades of diffusion time.
        # Reference: Sun et al. (2009) "A Concise and Provably Informative
        # Multi-Scale Signature Based on Heat Diffusion", Section 4.1
        import math

        log_10_val = math.log(10.0)
        scale_factor = 4.0 * log_10_val  # 4 decades of diffusion (Sun et al. 2009, §4.1)

        t_min = scale_factor / lambda_max
        t_max = scale_factor / lambda_2

        # Ensure t_min < t_max. When λ_2 ≈ λ_max (degenerate spectrum), the
        # range collapses. We extend by one decade (10×) — the same unit used
        # in the scale_factor derivation above (each factor of log(10) in
        # scale_factor = one decade of decay). One additional decade ensures
        # at least a minimal diffusion time range for the heat kernel.
        if t_min >= t_max:
            t_max = t_min * 10.0

        # Log-spaced times
        t_min_log = b.log(b.array([t_min]))
        t_max_log = b.log(b.array([t_max]))
        b.eval(t_min_log, t_max_log)

        time_indices = b.arange(0, n_times)
        times = b.exp(
            t_min_log + (t_max_log - t_min_log) * time_indices / max(n_times - 1, 1)
        )
        times = b.reshape(times, (-1,))
        b.eval(times)

        # Compute HKS: HKS(x, t) = Σᵢ exp(-λᵢt) φᵢ(x)²
        # shapes: eigenvalues [k], eigenvectors [n, k], times [n_times]

        # φᵢ(x)² for each point and eigenfunction: [n, k]
        phi_squared = eigenvectors * eigenvectors
        b.eval(phi_squared)

        # exp(-λᵢt) for each eigenvalue and time: [n_times, k]
        eig_row = b.reshape(eigenvalues, (1, -1))  # [1, k]
        times_col = b.reshape(times, (-1, 1))  # [n_times, 1]
        exp_decay = b.exp(-times_col * eig_row)  # [n_times, k]
        b.eval(exp_decay)

        # HKS: sum over eigenvalues
        # [n, k] @ [k, n_times] = [n, n_times]
        signatures = b.matmul(phi_squared, b.transpose(exp_decay))
        b.eval(signatures)

        return HeatKernelSignatureResult(
            signatures=signatures,
            times=times,
            eigenvalues=eigenvalues,
            t_min=t_min,
            t_max=t_max,
        )

    def compute_from_embedding(
        self,
        embedding_result: "Any",
        n_times: int = 10,
        k_eigenvalues: int | None = None,
    ) -> HeatKernelSignatureResult:
        """Compute HKS from a pre-computed spectral embedding.

        Efficient path when you already have eigenvalues and eigenvectors.

        Args:
            embedding_result: SpectralEmbeddingResult from spectral embedding.
            n_times: Number of time scales.
            k_eigenvalues: Number of eigenvalues to use.

        Returns:
            HeatKernelSignatureResult.
        """
        b = self._backend

        # Use eigenvalues that match eigenvectors dimension
        eigenvalues = embedding_result.eigenvalues  # [k_used]
        eigenvectors = embedding_result.eigenvectors  # [n, k_used]
        b.eval(eigenvalues, eigenvectors)

        n = int(eigenvectors.shape[0])
        k = int(eigenvalues.shape[0])

        if k_eigenvalues is not None:
            k = min(k, k_eigenvalues)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
            b.eval(eigenvalues, eigenvectors)

        if k == 0 or n == 0:
            empty_sig = b.zeros((n, n_times))
            empty_times = b.zeros((n_times,))
            return HeatKernelSignatureResult(
                signatures=empty_sig,
                times=empty_times,
                eigenvalues=eigenvalues,
                t_min=0.0,
                t_max=0.0,
            )

        eps = division_epsilon(b, eigenvalues)

        lambda_max_arr = b.max(eigenvalues)
        b.eval(lambda_max_arr)
        lambda_max = float(b.to_scalar(lambda_max_arr))

        positive_mask = eigenvalues > eps
        masked_eigs = b.where(
            positive_mask,
            eigenvalues,
            b.full(eigenvalues.shape, float("inf")),
        )
        lambda_2_arr = b.min(masked_eigs)
        b.eval(lambda_2_arr)
        lambda_2 = float(b.to_scalar(lambda_2_arr))

        if lambda_max <= eps or lambda_2 == float("inf"):
            empty_sig = b.zeros((n, n_times))
            empty_times = b.zeros((n_times,))
            return HeatKernelSignatureResult(
                signatures=empty_sig,
                times=empty_times,
                eigenvalues=eigenvalues,
                t_min=0.0,
                t_max=0.0,
            )

        log_10_val = 2.302585
        scale_factor = 4.0 * log_10_val

        t_min = scale_factor / lambda_max
        t_max = scale_factor / lambda_2

        if t_min >= t_max:
            t_max = t_min * 10.0

        t_min_log = b.log(b.array([t_min]))
        t_max_log = b.log(b.array([t_max]))
        b.eval(t_min_log, t_max_log)

        time_indices = b.arange(0, n_times)
        times = b.exp(
            t_min_log + (t_max_log - t_min_log) * time_indices / max(n_times - 1, 1)
        )
        times = b.reshape(times, (-1,))
        b.eval(times)

        phi_squared = eigenvectors * eigenvectors
        b.eval(phi_squared)

        eig_row = b.reshape(eigenvalues, (1, -1))
        times_col = b.reshape(times, (-1, 1))
        exp_decay = b.exp(-times_col * eig_row)
        b.eval(exp_decay)

        signatures = b.matmul(phi_squared, b.transpose(exp_decay))
        b.eval(signatures)

        return HeatKernelSignatureResult(
            signatures=signatures,
            times=times,
            eigenvalues=eigenvalues,
            t_min=t_min,
            t_max=t_max,
        )


def compute_heat_kernel_signature(
    points: "Array",
    n_times: int = 10,
    backend: "Backend | None" = None,
) -> HeatKernelSignatureResult:
    """Compute Heat Kernel Signature for a point cloud (convenience function).

    Args:
        points: Point cloud [n, d].
        n_times: Number of time scales.
        backend: Backend for tensor operations.

    Returns:
        HeatKernelSignatureResult with per-point signatures.
    """
    hks = HeatKernelSignature(backend)
    return hks.compute(points, n_times)


def compare_hks_profiles(
    hks_a: HeatKernelSignatureResult,
    hks_b: HeatKernelSignatureResult,
    backend: "Backend | None" = None,
) -> float:
    """Compare two HKS profiles for shape similarity.

    Computes the Frobenius norm of the difference between HKS signature
    matrices, normalized by the combined norm.

    A value near 0 indicates similar shapes; near 1 indicates different shapes.

    Args:
        hks_a: First HKS result.
        hks_b: Second HKS result.
        backend: Backend for tensor operations.

    Returns:
        Normalized difference in [0, 1].
    """
    b = backend or get_default_backend()

    sig_a = hks_a.signatures
    sig_b = hks_b.signatures
    b.eval(sig_a, sig_b)

    # Pad if different sizes
    n_a, t_a = int(sig_a.shape[0]), int(sig_a.shape[1])
    n_b, t_b = int(sig_b.shape[0]), int(sig_b.shape[1])

    # For comparison, use mean HKS profile (shape descriptor)
    mean_a = b.mean(sig_a, axis=0) if n_a > 0 else b.zeros((t_a,))
    mean_b = b.mean(sig_b, axis=0) if n_b > 0 else b.zeros((t_b,))
    b.eval(mean_a, mean_b)

    # Align time dimensions by truncation
    t_min = min(t_a, t_b)
    if t_min == 0:
        return 1.0  # No comparison possible

    mean_a = mean_a[:t_min]
    mean_b = mean_b[:t_min]
    b.eval(mean_a, mean_b)

    diff = mean_a - mean_b
    diff_norm = b.sqrt(b.sum(diff * diff))
    combined_norm = b.sqrt(b.sum(mean_a * mean_a) + b.sum(mean_b * mean_b))
    b.eval(diff_norm, combined_norm)

    eps = division_epsilon(b, diff_norm)
    combined_val = float(b.to_scalar(combined_norm))
    if combined_val <= eps:
        return 0.0

    similarity = float(b.to_scalar(diff_norm)) / combined_val
    return min(1.0, similarity)
