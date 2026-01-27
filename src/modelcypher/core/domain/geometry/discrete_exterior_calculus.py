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

"""Discrete Exterior Calculus (DEC) for manifold analysis.

Implements differential forms on simplicial complexes for:
- Geodesic computation via heat kernel (Varadhan's formula)
- Hodge decomposition (gradient + curl + harmonic)
- Laplacian-Beltrami operator on discrete manifolds

ALL parameters derived from geometry: k from Berry-Sauer connectivity,
time t from √eps × mean_edge², regularization from √eps.

References:
    - Hirani, A. N. (2003). "Discrete Exterior Calculus." PhD Thesis, Caltech.
    - Crane, K., de Goes, F., Desbrun, M., & Schroder, P. (2013).
      "Digital Geometry Processing with Discrete Exterior Calculus."
      ACM SIGGRAPH 2013 Courses.
    - Varadhan, S. R. S. (1967). "On the behavior of the fundamental solution
      of the heat equation with variable coefficients."
      Communications on Pure and Applied Mathematics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SimplicialComplex:
    """A simplicial complex built from point cloud data.

    Attributes:
        vertices: Point positions [n_vertices, d]
        edges: Edge indices [n_edges, 2]
        triangles: Triangle indices [n_triangles, 3]
        edge_weights: Weights for each edge (geodesic distances)
    """
    vertices: NDArray[np.float32]
    edges: NDArray[np.int32]
    triangles: NDArray[np.int32]
    edge_weights: NDArray[np.float32]

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


@dataclass
class HodgeDecomposition:
    """Result of Hodge decomposition of a 1-form.

    Any 1-form ω decomposes as:
    ω = d(α) + ★⁻¹d★(β) + harmonic

    where:
    - d(α) is the gradient component (exact form)
    - ★⁻¹d★(β) is the co-gradient/curl component (co-exact form)
    - harmonic is the harmonic component
    """
    gradient_component: NDArray[np.float32]
    curl_component: NDArray[np.float32]
    harmonic_component: NDArray[np.float32]
    gradient_norm: float
    curl_norm: float
    harmonic_norm: float

    @property
    def gradient_fraction(self) -> float:
        total = self.gradient_norm + self.curl_norm + self.harmonic_norm
        return self.gradient_norm / total if total > 1e-10 else 0.0

    @property
    def curl_fraction(self) -> float:
        total = self.gradient_norm + self.curl_norm + self.harmonic_norm
        return self.curl_norm / total if total > 1e-10 else 0.0

    @property
    def harmonic_fraction(self) -> float:
        total = self.gradient_norm + self.curl_norm + self.harmonic_norm
        return self.harmonic_norm / total if total > 1e-10 else 0.0


@dataclass
class DECGeodesicResult:
    """Result of DEC geodesic computation."""
    distances: NDArray[np.float32]
    laplacian_eigenvalues: NDArray[np.float32]
    spectral_gap: float
    is_positive_semidefinite: bool
    heat_time: float
    mean_edge_length: float


class DiscreteExteriorCalculus:
    """Discrete Exterior Calculus operators on a simplicial complex.

    Provides:
    - Boundary operators d₀, d₁
    - Hodge star operators ★₀, ★₁
    - Laplacian-Beltrami operator L
    - Heat kernel for geodesic distances
    - Hodge decomposition
    """

    def __init__(self, sqrt_eps: Optional[float] = None):
        """Initialize DEC with precision parameter.

        Args:
            sqrt_eps: Square root of machine epsilon. If None, derived from float32.
        """
        self.sqrt_eps = sqrt_eps or np.sqrt(np.finfo(np.float32).eps)

    def build_simplicial_complex(
        self,
        points: NDArray[np.float32],
        k_neighbors: Optional[int] = None,
    ) -> SimplicialComplex:
        """Build simplicial complex from point cloud via k-NN graph.

        Args:
            points: Point cloud [n, d]
            k_neighbors: Number of neighbors. If None, uses Berry-Sauer: 2*log(n)

        Returns:
            SimplicialComplex with vertices, edges, triangles
        """
        n = len(points)

        # Berry-Sauer connectivity: k = 2 * log(n)
        if k_neighbors is None:
            k_neighbors = max(5, int(2 * np.log(n)))

        logger.debug(f"Building simplicial complex: n={n}, k={k_neighbors}")

        # Compute pairwise distances
        dists = cdist(points, points)

        # Build k-NN graph
        edges = []
        edge_weights = []

        for i in range(n):
            # Get k nearest neighbors (excluding self)
            neighbor_idx = np.argsort(dists[i])[1:k_neighbors+1]
            for j in neighbor_idx:
                if i < j:  # Avoid duplicates
                    edges.append([i, j])
                    edge_weights.append(dists[i, j])

        edges = np.array(edges, dtype=np.int32)
        edge_weights = np.array(edge_weights, dtype=np.float32)

        # Build edge lookup for triangle detection
        edge_set = {(min(e[0], e[1]), max(e[0], e[1])) for e in edges}

        # Build adjacency for each vertex
        adj = {i: set() for i in range(n)}
        for e in edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])

        # Find triangles: three vertices where all pairs are connected
        triangles = []
        for i in range(n):
            neighbors_i = adj[i]
            for j in neighbors_i:
                if j > i:
                    # Check for common neighbors forming triangles
                    common = neighbors_i & adj[j]
                    for k in common:
                        if k > j:
                            # i < j < k forms a triangle
                            triangles.append([i, j, k])

        triangles = np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)

        logger.debug(f"Complex: {n} vertices, {len(edges)} edges, {len(triangles)} triangles")

        return SimplicialComplex(
            vertices=points,
            edges=edges,
            triangles=triangles,
            edge_weights=edge_weights,
        )

    def build_boundary_d0(self, complex: SimplicialComplex) -> csr_matrix:
        """Build d₀: 0-forms → 1-forms (gradient on vertices to edges).

        d₀[e, v] = +1 if v is head of edge e, -1 if tail, 0 otherwise.
        """
        n_edges = complex.n_edges
        n_vertices = complex.n_vertices

        d0 = lil_matrix((n_edges, n_vertices), dtype=np.float32)

        for e_idx, (i, j) in enumerate(complex.edges):
            d0[e_idx, i] = -1.0
            d0[e_idx, j] = +1.0

        return d0.tocsr()

    def build_boundary_d1(self, complex: SimplicialComplex) -> csr_matrix:
        """Build d₁: 1-forms → 2-forms (curl on edges to triangles).

        d₁[t, e] = +1/-1 based on orientation.
        """
        n_triangles = complex.n_triangles
        n_edges = complex.n_edges

        if n_triangles == 0:
            return csr_matrix((0, n_edges), dtype=np.float32)

        # Build edge index map
        edge_map = {}
        for e_idx, (i, j) in enumerate(complex.edges):
            edge_map[(min(i, j), max(i, j))] = e_idx

        d1 = lil_matrix((n_triangles, n_edges), dtype=np.float32)

        for t_idx, (i, j, k) in enumerate(complex.triangles):
            # Edges of triangle: (i,j), (j,k), (i,k)
            # Orientation: counterclockwise
            edges_in_tri = [
                (min(i, j), max(i, j), +1 if i < j else -1),
                (min(j, k), max(j, k), +1 if j < k else -1),
                (min(i, k), max(i, k), -1 if i < k else +1),  # Goes k→i
            ]

            for (a, b, sign) in edges_in_tri:
                if (a, b) in edge_map:
                    d1[t_idx, edge_map[(a, b)]] = sign

        return d1.tocsr()

    def build_hodge_star_0(self, complex: SimplicialComplex) -> NDArray[np.float32]:
        """Build ★₀: Hodge star on 0-forms (dual volumes/Voronoi cells).

        ★₀[v] = sum of half-edge lengths adjacent to v
        """
        n_vertices = complex.n_vertices
        star_0 = np.zeros(n_vertices, dtype=np.float32)

        for e_idx, (i, j) in enumerate(complex.edges):
            half_length = complex.edge_weights[e_idx] / 2
            star_0[i] += half_length
            star_0[j] += half_length

        # Regularize to avoid division by zero
        star_0 = np.maximum(star_0, self.sqrt_eps)

        return star_0

    def build_hodge_star_1(self, complex: SimplicialComplex) -> NDArray[np.float32]:
        """Build ★₁: Hodge star on 1-forms (cotan weights).

        For each edge, ★₁[e] = (cot(α) + cot(β)) / 2
        where α, β are angles opposite edge e in adjacent triangles.

        Falls back to 1/edge_length for edges not in triangles.
        """
        n_edges = complex.n_edges
        star_1 = np.ones(n_edges, dtype=np.float32)

        if complex.n_triangles == 0:
            # No triangles: use inverse edge length
            star_1 = 1.0 / np.maximum(complex.edge_weights, self.sqrt_eps)
            return star_1

        # Build edge to triangle adjacency
        edge_map = {}
        for e_idx, (i, j) in enumerate(complex.edges):
            edge_map[(min(i, j), max(i, j))] = e_idx

        # For each edge, accumulate cotan weights from adjacent triangles
        edge_cotans = np.zeros(n_edges, dtype=np.float32)
        edge_counts = np.zeros(n_edges, dtype=np.int32)

        for t_idx, (i, j, k) in enumerate(complex.triangles):
            # Get vertex positions
            pi = complex.vertices[i]
            pj = complex.vertices[j]
            pk = complex.vertices[k]

            # Compute angles at each vertex
            def angle_at(p_center, p_a, p_b):
                v1 = p_a - p_center
                v2 = p_b - p_center
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + self.sqrt_eps)
                cos_angle = np.clip(cos_angle, -1, 1)
                return np.arccos(cos_angle)

            angle_i = angle_at(pi, pj, pk)  # Angle at i, opposite edge j-k
            angle_j = angle_at(pj, pi, pk)  # Angle at j, opposite edge i-k
            angle_k = angle_at(pk, pi, pj)  # Angle at k, opposite edge i-j

            # Cotan weights
            cot_i = 1.0 / np.tan(angle_i + self.sqrt_eps)
            cot_j = 1.0 / np.tan(angle_j + self.sqrt_eps)
            cot_k = 1.0 / np.tan(angle_k + self.sqrt_eps)

            # Assign to edges
            # Edge (j,k) opposite angle_i
            e_jk = edge_map.get((min(j, k), max(j, k)))
            if e_jk is not None:
                edge_cotans[e_jk] += cot_i
                edge_counts[e_jk] += 1

            # Edge (i,k) opposite angle_j
            e_ik = edge_map.get((min(i, k), max(i, k)))
            if e_ik is not None:
                edge_cotans[e_ik] += cot_j
                edge_counts[e_ik] += 1

            # Edge (i,j) opposite angle_k
            e_ij = edge_map.get((min(i, j), max(i, j)))
            if e_ij is not None:
                edge_cotans[e_ij] += cot_k
                edge_counts[e_ij] += 1

        # Average cotans
        for e_idx in range(n_edges):
            if edge_counts[e_idx] > 0:
                star_1[e_idx] = edge_cotans[e_idx] / (2 * edge_counts[e_idx])
            else:
                # Edge not in any triangle: use inverse length
                star_1[e_idx] = 1.0 / max(complex.edge_weights[e_idx], self.sqrt_eps)

        # Ensure positive
        star_1 = np.maximum(star_1, self.sqrt_eps)

        return star_1

    def build_laplacian(self, complex: SimplicialComplex) -> csr_matrix:
        """Build Laplacian-Beltrami operator: L = ★₀⁻¹ d₀ᵀ ★₁ d₀.

        Returns:
            Sparse Laplacian matrix [n_vertices, n_vertices]
        """
        d0 = self.build_boundary_d0(complex)
        star_0 = self.build_hodge_star_0(complex)
        star_1 = self.build_hodge_star_1(complex)

        # L = diag(1/star_0) @ d0.T @ diag(star_1) @ d0
        # Build as: inv_star_0 @ d0.T @ star_1_diag @ d0

        star_1_diag = csr_matrix((star_1, (range(len(star_1)), range(len(star_1)))))
        inv_star_0 = 1.0 / star_0

        L = d0.T @ star_1_diag @ d0

        # Apply inverse star_0 as row scaling
        L = csr_matrix(L.multiply(inv_star_0[:, np.newaxis]))

        return L

    def compute_geodesic_distances(
        self,
        complex: SimplicialComplex,
        heat_time: Optional[float] = None,
    ) -> DECGeodesicResult:
        """Compute geodesic distances via heat kernel (Varadhan's formula).

        d_geo² ≈ -4t × log(heat_kernel) as t → 0

        Args:
            complex: Simplicial complex
            heat_time: Heat diffusion time. If None, uses √eps × mean_edge²

        Returns:
            DECGeodesicResult with distance matrix and diagnostics
        """
        n = complex.n_vertices
        mean_edge = float(np.mean(complex.edge_weights))

        # Heat time: derived from precision and edge length
        if heat_time is None:
            heat_time = self.sqrt_eps * (mean_edge ** 2)

        logger.debug(f"Computing DEC geodesics: n={n}, t={heat_time:.4e}")

        # Build Laplacian
        L = self.build_laplacian(complex)

        # Compute Laplacian eigenvalues for diagnostics
        n_eigs = min(20, n - 2)
        try:
            eigenvalues, _ = eigsh(L, k=n_eigs, which='SM')
            eigenvalues = np.sort(np.real(eigenvalues))
            spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            is_positive_semidefinite = np.all(eigenvalues >= -self.sqrt_eps)
        except Exception as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            eigenvalues = np.array([0.0])
            spectral_gap = 0.0
            is_positive_semidefinite = False

        # Compute heat kernel: H(t) = exp(-t × L)
        # For geodesics, we need H(t) for each source point
        distances = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            # Delta function at source point
            delta_i = np.zeros(n, dtype=np.float32)
            delta_i[i] = 1.0

            # Heat diffusion: u(t) = exp(-t*L) @ delta_i
            # Use Krylov subspace approximation
            try:
                u_t = expm_multiply(-heat_time * L, delta_i)
            except Exception as e:
                logger.warning(f"Heat kernel computation failed for point {i}: {e}")
                continue

            # Varadhan's formula: d² ≈ -4t × log(u)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_u = np.log(np.maximum(u_t, self.sqrt_eps))
                d_sq = -4 * heat_time * log_u
                d_sq = np.maximum(d_sq, 0)  # Ensure non-negative
                distances[i] = np.sqrt(d_sq)

        # Symmetrize
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)

        return DECGeodesicResult(
            distances=distances,
            laplacian_eigenvalues=eigenvalues,
            spectral_gap=spectral_gap,
            is_positive_semidefinite=is_positive_semidefinite,
            heat_time=heat_time,
            mean_edge_length=mean_edge,
        )

    def hodge_decomposition(
        self,
        complex: SimplicialComplex,
        one_form: NDArray[np.float32],
    ) -> HodgeDecomposition:
        """Decompose a 1-form into gradient, curl, and harmonic components.

        ω = d(α) + ★⁻¹d★(β) + harmonic

        Args:
            complex: Simplicial complex
            one_form: Values on edges [n_edges]

        Returns:
            HodgeDecomposition with component norms and vectors
        """
        n_edges = complex.n_edges
        n_vertices = complex.n_vertices

        d0 = self.build_boundary_d0(complex)
        star_0 = self.build_hodge_star_0(complex)
        star_1 = self.build_hodge_star_1(complex)

        # Gradient component: d(α) where α minimizes ||ω - d(α)||²
        # α = (d0.T @ star_1 @ d0)^(-1) @ d0.T @ star_1 @ ω
        L = d0.T @ csr_matrix((star_1, (range(len(star_1)), range(len(star_1))))) @ d0

        try:
            # Solve L @ alpha = d0.T @ diag(star_1) @ omega
            rhs = d0.T @ (star_1 * one_form)
            # Use pseudo-inverse for potentially singular L
            L_dense = L.toarray()
            alpha = np.linalg.lstsq(L_dense + self.sqrt_eps * np.eye(n_vertices), rhs, rcond=None)[0]
            gradient_component = (d0 @ alpha).astype(np.float32)
        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}")
            gradient_component = np.zeros(n_edges, dtype=np.float32)

        # Curl component: ★⁻¹d★(β) where β on triangles
        # For simplicity, compute as ω - gradient - harmonic
        # In practice with few triangles, curl is often negligible

        if complex.n_triangles > 0:
            d1 = self.build_boundary_d1(complex)

            try:
                # β on triangles minimizes ||ω - grad - ★⁻¹d1.T @ β||²
                # This is more complex; approximate by residual
                star_1_inv = 1.0 / star_1
                residual = one_form - gradient_component

                # Project residual onto image of d1.T
                if d1.shape[0] > 0:
                    d1_star_inv = csr_matrix((star_1_inv, (range(len(star_1_inv)), range(len(star_1_inv))))) @ d1.T
                    M = d1 @ d1_star_inv
                    if M.shape[0] > 0 and M.shape[1] > 0:
                        M_dense = M.toarray()
                        beta = np.linalg.lstsq(M_dense + self.sqrt_eps * np.eye(M.shape[0]),
                                               d1 @ (star_1_inv * residual), rcond=None)[0]
                        curl_component = (d1_star_inv @ beta).astype(np.float32)
                    else:
                        curl_component = np.zeros(n_edges, dtype=np.float32)
                else:
                    curl_component = np.zeros(n_edges, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Curl computation failed: {e}")
                curl_component = np.zeros(n_edges, dtype=np.float32)
        else:
            curl_component = np.zeros(n_edges, dtype=np.float32)

        # Harmonic component: what's left
        harmonic_component = one_form - gradient_component - curl_component

        # Compute norms
        gradient_norm = float(np.linalg.norm(gradient_component))
        curl_norm = float(np.linalg.norm(curl_component))
        harmonic_norm = float(np.linalg.norm(harmonic_component))

        return HodgeDecomposition(
            gradient_component=gradient_component,
            curl_component=curl_component,
            harmonic_component=harmonic_component,
            gradient_norm=gradient_norm,
            curl_norm=curl_norm,
            harmonic_norm=harmonic_norm,
        )
