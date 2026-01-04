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
Topological fingerprinting for architecture-invariant model comparison.

Mathematical Foundation:
    Persistent homology studies topological features (connected components,
    loops, voids) across multiple scales. Given a point cloud, we construct
    a Vietoris-Rips filtration by connecting points within increasing distance
    thresholds and track when features appear (birth) and disappear (death).

Key Concepts:
    - Betti Numbers: β₀ = connected components, β₁ = loops, β₂ = voids
    - Persistence Diagram: (birth, death) pairs for each topological feature
    - Bottleneck Distance: Maximum matching cost between diagrams
    - Wasserstein Distance: Total matching cost between diagrams

Algorithm Overview:
    1. Compute pairwise distances between points
    2. Build Vietoris-Rips filtration (edges sorted by distance)
    3. Track 0-dimensional persistence via Union-Find with elder rule
    4. Track 1-dimensional persistence via cycle/triangle detection
    5. Compute summary statistics (Betti numbers, persistence entropy)

Complexity:
    O(n² log n) for edge sorting, O(n³) for cycle detection.
    Practical for n < 5000 points.

References:
    - Edelsbrunner & Harer (2010) "Computational Topology: An Introduction"
    - Carlsson (2009) "Topology and Data"

Research Connections:
    This implementation enables analysis similar to the Blue Brain Project's
    discovery of 11-dimensional structures in neural circuits (Reimann et al.,
    Frontiers in Computational Neuroscience 2017). They found that neurons form
    high-dimensional simplicial complexes (cliques) with topological cavities
    that appear during stimulus processing and collapse afterward—"building
    then razing towers of multi-dimensional blocks."

    Our hypothesis: LLMs may implement similar transient high-dimensional
    structures during inference. Betti numbers (β₀, β₁, β₂) provide
    architecture-invariant fingerprints for cross-model comparison.

    See also: docs/RESEARCH-CONNECTIONS.md

See also: docs/geometry/topological_fingerprints.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    infinity_threshold,
    is_finite,
    log_scalar,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class PersistencePoint:
    """
    A topological feature with birth and death times.
    """

    birth: float
    death: float
    dimension: int

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass
class PersistenceDiagram:
    """
    Persistence diagram: collection of persistence points.
    """

    points: list[PersistencePoint]

    @property
    def count_by_dimension(self) -> dict[int, int]:
        counts = {}
        for p in self.points:
            counts[p.dimension] = counts.get(p.dimension, 0) + 1
        return counts

    def betti_numbers(self, persistence_threshold: float | None = None) -> dict[int, int]:
        betti: dict[int, int] = {}
        for p in self.points:
            if persistence_threshold is None or p.persistence >= persistence_threshold:
                betti[p.dimension] = betti.get(p.dimension, 0) + 1
        return betti


@dataclass
class TopologySummary:
    component_count: int
    cycle_count: int
    average_persistence: float
    max_persistence: float
    persistence_entropy: float


@dataclass
class Fingerprint:
    diagram: PersistenceDiagram
    betti_numbers: dict[int, int]
    summary: TopologySummary


@dataclass
class ComparisonResult:
    """Result of topological comparison between two manifolds.

    Note: Different topologies do NOT mean models are incompatible.
    Models are always compatible - this measures structural similarity.
    Raw metrics are returned; interpretation is left to callers.
    """

    bottleneck_distance: float
    wasserstein_distance: float
    betti_difference: int
    similarity_score: float
    betti_numbers_match: bool  # True if Betti numbers are identical


class TopologicalFingerprint:
    """
    Topological fingerprinting for architecture-invariant model comparison.
    Computes persistent homology (Betti numbers) via Vietoris-Rips filtration.
    """

    @staticmethod
    def compute(
        points: list[list[float]],
    ) -> Fingerprint:
        if len(points) < 2:
            return Fingerprint(
                diagram=PersistenceDiagram([]),
                betti_numbers={},
                summary=TopologySummary(len(points), 0, 0.0, 0.0, 0.0),
            )

        backend = get_default_backend()
        distances = TopologicalFingerprint._compute_pairwise_distances(points, backend=backend)

        finite_mask = backend.isfinite(distances)
        neg_inf = -float(backend.finfo().max)
        pos_inf = float(backend.finfo().max)
        finite_vals = backend.where(finite_mask, distances, backend.full(distances.shape, neg_inf))
        max_dist_arr = backend.max(finite_vals)
        backend.eval(max_dist_arr)
        max_dist_val = float(backend.to_scalar(max_dist_arr))

        if not is_finite(max_dist_val, backend):
            max_dist_val = 0.0

        max_dist = max_dist_val

        nonzero_mask = (distances > 0) & finite_mask
        min_candidates = backend.where(
            nonzero_mask, distances, backend.full(distances.shape, pos_inf)
        )
        min_dist_arr = backend.min(min_candidates)
        backend.eval(min_dist_arr)
        min_dist_val = float(backend.to_scalar(min_dist_arr))
        min_dist = min_dist_val if is_finite(min_dist_val, backend) else 0.0

        # Convert to Python list for pure-Python filtration algorithm
        # This ensures PersistencePoints store Python floats, not MLX scalars
        backend.eval(distances)
        distances_list = backend.tolist(distances)

        diagram = TopologicalFingerprint._vietoris_rips_filtration(
            distances=distances_list,
            min_filtration=min_dist,
            max_filtration=max_dist,
        )

        # Filter only numerical noise (dtype-derived)
        eps = machine_epsilon(backend, distances)
        threshold = max_dist * eps
        betti = diagram.betti_numbers(persistence_threshold=threshold)

        significant_points = [p for p in diagram.points if p.persistence > threshold]
        persistences = [p.persistence for p in significant_points]

        summary = TopologySummary(
            component_count=betti.get(0, 1),
            cycle_count=betti.get(1, 0),
            average_persistence=sum(persistences) / len(persistences) if persistences else 0.0,
            max_persistence=max(persistences) if persistences else 0.0,
            persistence_entropy=TopologicalFingerprint._compute_entropy(persistences),
        )

        return Fingerprint(diagram, betti, summary)

    @staticmethod
    def compare(fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> ComparisonResult:
        """Compare two topological fingerprints.

        Returns raw geometric metrics. Interpretation is left to the caller
        who understands their specific context - no arbitrary classification
        thresholds imposed here.

        Similarity score is derived geometrically:
        - Exponential decay based on bottleneck/scale ratio
        - Exponential decay based on wasserstein/scale ratio
        - Harmonic penalty for Betti number differences
        """
        bottleneck_ab = TopologicalFingerprint._bottleneck_distance(
            fingerprint_a.diagram, fingerprint_b.diagram
        )
        bottleneck_ba = TopologicalFingerprint._bottleneck_distance(
            fingerprint_b.diagram, fingerprint_a.diagram
        )
        bottleneck = max(bottleneck_ab, bottleneck_ba)
        wasserstein = TopologicalFingerprint._wasserstein_distance(
            fingerprint_a.diagram, fingerprint_b.diagram
        )

        betti_diff = 0
        all_dims = set(fingerprint_a.betti_numbers.keys()) | set(fingerprint_b.betti_numbers.keys())
        for dim in all_dims:
            a = fingerprint_a.betti_numbers.get(dim, 0)
            b = fingerprint_b.betti_numbers.get(dim, 0)
            betti_diff += abs(a - b)

        # Scale is derived from the data - max persistence in either fingerprint
        backend = get_default_backend()
        scale_eps = division_epsilon(
            backend,
            backend.array(
                [fingerprint_a.summary.max_persistence, fingerprint_b.summary.max_persistence]
            ),
        )
        scale = max(
            fingerprint_a.summary.max_persistence,
            fingerprint_b.summary.max_persistence,
            scale_eps,
        )

        # Similarity score: purely geometric derivation
        # exp(-x) naturally maps distances to [0, 1] similarities
        # Product of independent factors (Betti, bottleneck, Wasserstein)
        _b = get_default_backend()
        score = (
            exp_scalar(-bottleneck / scale, _b)
            * exp_scalar(-wasserstein / scale, _b)
            * (1.0 / (1 + betti_diff))
        )

        # Betti match: exact Betti numbers means topologically equivalent structure
        # No arbitrary thresholds - Betti numbers are discrete invariants
        betti_match = betti_diff == 0

        return ComparisonResult(
            bottleneck_distance=bottleneck,
            wasserstein_distance=wasserstein,
            betti_difference=betti_diff,
            similarity_score=score,
            betti_numbers_match=betti_match,
        )

    @staticmethod
    def _compute_pairwise_distances(
        points: list[list[float]], backend: "Backend | None" = None
    ) -> "Array":
        """Compute pairwise geodesic distances for persistent homology.

        Uses geodesic distances to capture true topological structure
        of the curved manifold.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
            derive_k_neighbors,
        )

        n = len(points)
        if n == 0:
            b = backend or get_default_backend()
            return b.zeros((0, 0))

        b = backend or get_default_backend()
        pts = b.astype(b.array(points), "float32")  # Ensure float for numerical operations

        # Use geodesic distances for correct topology on curved manifolds
        k_neighbors = derive_k_neighbors(pts, b)
        geo_dist = geodesic_distance_matrix(pts, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        return geo_dist

    @staticmethod
    def _vietoris_rips_filtration(
        distances: list[list[float]],
        min_filtration: float,
        max_filtration: float,
    ) -> PersistenceDiagram:
        """
        Compute persistent homology via Vietoris-Rips filtration.

        Algorithm for 0-dimensional persistence (connected components):
            1. Initialize each point as its own component (birth = 0)
            2. Sort all edges by distance
            3. Process edges in order using Union-Find:
               - If edge connects different components, merge them
               - Apply "elder rule": older component survives
               - Record (birth, death=current_distance) for dying component
            4. Surviving components persist to max_filtration

        Algorithm for 1-dimensional persistence (cycles):
            1. Process edges again, now tracking cycle formation
            2. When edge (i,j) connects already-connected vertices:
               - A cycle is born at this edge's distance
            3. Cycle dies when a triangle fills it:
               - Find vertex k where (i,k) and (j,k) edges exist
               - Death = max(dist(i,j), dist(i,k), dist(j,k))

        The "elder rule" ensures topological consistency: when two
        components merge, the older one (born first) survives, and
        the younger one dies at the current filtration value.

        Args:
            distances: Pairwise distance matrix (n x n)
            min_filtration: Minimum scale (typically min nonzero distance)
            max_filtration: Maximum scale (typically max distance)

        Returns:
            PersistenceDiagram with (birth, death, dimension) points
        """
        n = len(distances)
        persistence_points = []

        # 0-dim persistence (connected components)
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y, component_birth):
            px, py = find(x), find(y)
            if px == py:
                return None

            birth_x = component_birth[px]
            birth_y = component_birth[py]

            # Elder rule: older component receives younger one
            # If births equal, use rank
            survivor, dying = (px, py)
            if birth_x < birth_y:
                survivor, dying = px, py
            elif birth_y < birth_x:
                survivor, dying = py, px
            else:
                if rank[px] < rank[py]:
                    survivor, dying = py, px
                elif rank[px] > rank[py]:
                    survivor, dying = px, py
                else:
                    survivor, dying = px, py
                    rank[px] += 1

            parent[dying] = survivor
            return dying

        component_birth = [0.0] * n  # All points born at 0 in Rips

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i][j]))
        edges.sort(key=lambda x: x[2])

        # Process edges
        for i, j, dist in edges:
            if dist > max_filtration:
                break

            dying = union(i, j, component_birth)
            if dying is not None:
                birth = component_birth[dying]
                death = dist
                if death > birth:
                    persistence_points.append(PersistencePoint(birth, death, 0))

        # Surviving components
        for i in range(n):
            if find(i) == i:
                persistence_points.append(PersistencePoint(0.0, max_filtration, 0))

        # 1-dim persistence (cycles) - Triangle Approximation
        # KNOWN LIMITATION: This is NOT full Rips complex persistence.
        # Full persistence would require:
        # 1. Building full Vietoris-Rips complex at each filtration value
        # 2. Computing homology groups via boundary matrix reduction
        # 3. Tracking basis representatives for exact birth/death times
        #
        # This approximation:
        # - Detects cycles when edge connects already-connected vertices
        # - Death time = min triangle fill (min(max(d_ik, d_jk)) over k)
        # - Misses higher-order cycles (>3 vertices) and some edge cases
        #
        # For neural network manifolds, this captures dominant topological
        # features (small cycles) while avoiding O(n^3) Rips computation.
        if n >= 3:

            # Reset DSU for cycle detection pass
            parent = list(range(n))
            rank = [0] * n

            possible_cycles = []

            for i, j, dist in edges:
                if dist > max_filtration:
                    break

                px, py = find(i), find(j)
                if px == py:
                    # Cycle candidates
                    death_time = max_filtration
                    # Check for "filling triangles" (common neighbor k)
                    # Death is min(max(dist(i,k), dist(j,k))) over all k
                    # Actually death is when the cycle is "filled".
                    # A cycle formed by (i,j) with existing path is filled when a triangle (i,j,k) appears
                    # such that (i,k) and (j,k) exist.
                    # This approximation finds the "tightest" triangle filling this edge.

                    found_filling = False
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        dik = distances[i][k]
                        djk = distances[j][k]
                        triangle_fill = max(dist, max(dik, djk))
                        if triangle_fill < death_time:
                            death_time = triangle_fill
                            found_filling = True

                    if found_filling and death_time > dist:
                        possible_cycles.append(PersistencePoint(dist, death_time, 1))
                else:
                    union(i, j, component_birth)

            # Filter/Keep top cycles
            persistence_points.extend(possible_cycles[:20])

        return PersistenceDiagram(persistence_points)

    @staticmethod
    def _bottleneck_distance(diag_a: PersistenceDiagram, diag_b: PersistenceDiagram) -> float:
        """
        Compute bottleneck distance using optimal bipartite matching.

        Bottleneck distance = max cost in optimal matching.
        Uses Hungarian algorithm for optimal assignment, same as Wasserstein.
        """
        points_a = diag_a.points
        points_b = diag_b.points

        dims = set([p.dimension for p in points_a] + [p.dimension for p in points_b])
        max_dist = 0.0

        for dim in dims:
            pa = [p for p in points_a if p.dimension == dim]
            pb = [p for p in points_b if p.dimension == dim]

            n_a, n_b = len(pa), len(pb)

            if n_a == 0 and n_b == 0:
                continue

            # Handle case where one set is empty
            if n_a == 0:
                for b in pb:
                    max_dist = max(max_dist, (b.death - b.birth) / 2.0)
                continue
            if n_b == 0:
                for a in pa:
                    max_dist = max(max_dist, (a.death - a.birth) / 2.0)
                continue

            # Build augmented cost matrix for Hungarian algorithm
            # Same structure as Wasserstein but we track max instead of sum
            n = n_a + n_b
            cost = [[float("inf")] * n for _ in range(n)]

            # Fill matching costs between pa and pb (L-infinity for bottleneck)
            for i, a in enumerate(pa):
                for j, b in enumerate(pb):
                    cost[i][j] = max(abs(a.birth - b.birth), abs(a.death - b.death))
                # Diagonal cost for point a
                diag_cost_a = (a.death - a.birth) / 2.0
                for j in range(n_b, n_b + n_a):
                    if j - n_b == i:
                        cost[i][j] = diag_cost_a
                    else:
                        cost[i][j] = float("inf")

            # Fill diagonal costs for points in pb
            for i, b in enumerate(pb):
                diag_cost_b = (b.death - b.birth) / 2.0
                row = n_a + i
                for j in range(n_b):
                    if j == i:
                        cost[row][j] = diag_cost_b
                    else:
                        cost[row][j] = float("inf")
                # Zeros for dummy-to-dummy matching
                for j in range(n_b, n):
                    cost[row][j] = 0.0

            # Use Hungarian algorithm for optimal matching
            matching = TopologicalFingerprint._hungarian_algorithm(cost)

            # Bottleneck = max cost in matching (excluding dummy-dummy pairs)
            for i, j in enumerate(matching):
                if i < n_a or j < n_b:  # Real matches only
                    if cost[i][j] < float("inf"):
                        max_dist = max(max_dist, cost[i][j])

        return max_dist

    @staticmethod
    def _wasserstein_distance(diag_a: PersistenceDiagram, diag_b: PersistenceDiagram) -> float:
        """
        Compute symmetric Wasserstein-1 distance between persistence diagrams.

        Uses Hungarian algorithm for optimal bipartite matching to ensure
        the metric is symmetric: d(A, B) = d(B, A).

        Each point can either be matched to a point in the other diagram
        (cost = L1 distance between birth/death pairs) or matched to the
        diagonal (cost = persistence/2 for each unmatched point).
        """
        total_dist = 0.0
        count = 0

        dims = set([p.dimension for p in diag_a.points] + [p.dimension for p in diag_b.points])

        for dim in dims:
            pa = [p for p in diag_a.points if p.dimension == dim]
            pb = [p for p in diag_b.points if p.dimension == dim]

            n_a, n_b = len(pa), len(pb)

            if n_a == 0 and n_b == 0:
                continue

            # Handle case where one set is empty
            if n_a == 0:
                for b in pb:
                    total_dist += b.death - b.birth
                    count += 1
                continue
            if n_b == 0:
                for a in pa:
                    total_dist += a.death - a.birth
                    count += 1
                continue

            # Build augmented cost matrix for Hungarian algorithm
            # Size: (n_a + n_b) x (n_a + n_b)
            # First n_b columns: matching to points in pb
            # Last n_a columns: matching to diagonal (for points in pa)
            # First n_a rows: points from pa
            # Last n_b rows: points from pb (matched to diagonal)
            n = n_a + n_b
            cost = [[float("inf")] * n for _ in range(n)]

            # Fill matching costs between pa and pb
            for i, a in enumerate(pa):
                for j, b in enumerate(pb):
                    cost[i][j] = abs(a.birth - b.birth) + abs(a.death - b.death)
                # Diagonal cost for point a (matched to its projection on diagonal)
                diag_cost_a = a.death - a.birth
                for j in range(n_b, n_b + n_a):
                    if j - n_b == i:
                        cost[i][j] = diag_cost_a
                    else:
                        cost[i][j] = float("inf")

            # Fill diagonal costs for points in pb
            for i, b in enumerate(pb):
                diag_cost_b = b.death - b.birth
                row = n_a + i
                for j in range(n_b):
                    if j == i:
                        cost[row][j] = diag_cost_b
                    else:
                        cost[row][j] = float("inf")
                # Zeros for dummy-to-dummy matching
                for j in range(n_b, n):
                    cost[row][j] = 0.0

            # Use Hungarian algorithm to find optimal matching
            matching = TopologicalFingerprint._hungarian_algorithm(cost)

            # Sum up the costs from the matching
            for i, j in enumerate(matching):
                if cost[i][j] < float("inf"):
                    total_dist += cost[i][j]
                    if i < n_a or j < n_b:  # Only count real matches
                        count += 1

        return total_dist / count if count > 0 else 0.0

    @staticmethod
    def _hungarian_algorithm(cost_matrix: list[list[float]]) -> list[int]:
        """
        Hungarian algorithm for minimum cost bipartite matching.

        Returns a list where result[i] = j means row i is matched to column j.
        This implementation handles square matrices.
        """
        n = len(cost_matrix)
        if n == 0:
            return []

        # Initialize labels
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)  # p[j] = row matched to column j
        way = [0] * (n + 1)

        INF = float("inf")

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)

            while p[j0] != 0:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0

                for j in range(1, n + 1):
                    if not used[j]:
                        # Get cost, handling infinity
                        c = cost_matrix[i0 - 1][j - 1] if i0 <= n and j <= n else INF
                        cur = c - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                # Prevent infinite loop on impossible matching
                if delta == INF:
                    break

                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1

            # Reconstruct path
            while j0 != 0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        # Build result: result[i] = column matched to row i
        result = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0 and p[j] <= n:
                result[p[j] - 1] = j - 1

        return result

    @staticmethod
    def _compute_entropy(values: list[float]) -> float:
        total = sum(values)
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array(values))
        if total <= eps:
            return 0.0
        _b = get_default_backend()
        entropy = 0.0
        for v in values:
            p = v / total
            if p > 0:
                entropy -= p * log_scalar(p + eps, _b)
        return entropy


class BackendTopologicalFingerprint:
    """GPU-accelerated topological fingerprinting using the Backend protocol.

    Provides the same functionality as TopologicalFingerprint but uses
    Backend tensor operations for GPU acceleration where beneficial.

    Key optimizations:
    - Vectorized cost matrix building for bottleneck/Wasserstein distances
    - Backend-accelerated entropy computation
    - Vectorized edge extraction from distance matrix
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a Backend instance.

        Args:
            backend: Backend instance (MLXBackend, JAXBackend, etc.)
        """
        self.backend = backend

    def compute(
        self,
        points: list[list[float]],
    ) -> Fingerprint:
        """Compute topological fingerprint using Backend acceleration.

        Args:
            points: Point cloud as list of vectors.

        Returns:
            Fingerprint with persistence diagram and summary statistics.
        """
        if len(points) < 2:
            return Fingerprint(
                diagram=PersistenceDiagram([]),
                betti_numbers={},
                summary=TopologySummary(len(points), 0, 0.0, 0.0, 0.0),
            )

        b = self.backend
        distances = self._compute_pairwise_distances(points)
        b.eval(distances)

        finite_mask = b.isfinite(distances)
        neg_inf = -float(b.finfo().max)
        pos_inf = float(b.finfo().max)
        finite_vals = b.where(finite_mask, distances, b.full(distances.shape, neg_inf))
        max_val = b.max(finite_vals)
        b.eval(max_val)
        max_dist_val = float(b.to_scalar(max_val))
        if not is_finite(max_dist_val, b):
            max_dist_val = 0.0
        max_dist = max_dist_val

        nonzero_mask = (distances > 0) & finite_mask
        min_candidates = b.where(nonzero_mask, distances, b.full(distances.shape, pos_inf))
        min_val = b.min(min_candidates)
        b.eval(min_val)
        min_dist_val = float(b.to_scalar(min_val))
        min_dist = min_dist_val if is_finite(min_dist_val, b) else 0.0

        diagram = self._vietoris_rips_filtration(
            distances=distances,
            min_filtration=min_dist,
            max_filtration=max_dist,
        )

        # Filter only numerical noise (dtype-derived)
        threshold = max_dist * machine_epsilon(b, distances)
        betti = diagram.betti_numbers(persistence_threshold=threshold)

        significant_points = [p for p in diagram.points if p.persistence > threshold]
        persistences = [p.persistence for p in significant_points]

        summary = TopologySummary(
            component_count=betti.get(0, 1),
            cycle_count=betti.get(1, 0),
            average_persistence=sum(persistences) / len(persistences) if persistences else 0.0,
            max_persistence=max(persistences) if persistences else 0.0,
            persistence_entropy=self._compute_entropy(persistences),
        )

        return Fingerprint(diagram, betti, summary)

    def compare(
        self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint
    ) -> ComparisonResult:
        """Compare two topological fingerprints using Backend acceleration.

        Args:
            fingerprint_a: First fingerprint.
            fingerprint_b: Second fingerprint.

        Returns:
            ComparisonResult with distance metrics and similarity score.
        """
        bottleneck_ab = self._bottleneck_distance(fingerprint_a.diagram, fingerprint_b.diagram)
        bottleneck_ba = self._bottleneck_distance(fingerprint_b.diagram, fingerprint_a.diagram)
        bottleneck = max(bottleneck_ab, bottleneck_ba)
        wasserstein = self._wasserstein_distance(fingerprint_a.diagram, fingerprint_b.diagram)

        betti_diff = 0
        all_dims = set(fingerprint_a.betti_numbers.keys()) | set(fingerprint_b.betti_numbers.keys())
        for dim in all_dims:
            a = fingerprint_a.betti_numbers.get(dim, 0)
            b_val = fingerprint_b.betti_numbers.get(dim, 0)
            betti_diff += abs(a - b_val)

        backend = get_default_backend()
        scale_eps = division_epsilon(
            backend,
            backend.array(
                [fingerprint_a.summary.max_persistence, fingerprint_b.summary.max_persistence]
            ),
        )
        scale = max(
            fingerprint_a.summary.max_persistence,
            fingerprint_b.summary.max_persistence,
            scale_eps,
        )

        score = (
            exp_scalar(-bottleneck / scale, backend)
            * exp_scalar(-wasserstein / scale, backend)
            * (1.0 / (1 + betti_diff))
        )

        betti_match = betti_diff == 0

        # Note: interpretation removed per No Vibes rule - return raw measurements only
        return ComparisonResult(
            bottleneck_distance=bottleneck,
            wasserstein_distance=wasserstein,
            betti_difference=betti_diff,
            similarity_score=score,
            betti_numbers_match=betti_match,
        )

    def _compute_pairwise_distances(self, points: list[list[float]]) -> "Array":
        """Compute pairwise geodesic distances using Backend."""
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
            derive_k_neighbors,
        )

        n = len(points)
        if n == 0:
            return self.backend.zeros((0, 0))

        b = self.backend
        pts = b.array(points)

        k_neighbors = derive_k_neighbors(pts, b)
        geo_dist = geodesic_distance_matrix(pts, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        return geo_dist

    def _vietoris_rips_filtration(
        self,
        distances: "Array",
        min_filtration: float,
        max_filtration: float,
    ) -> PersistenceDiagram:
        """Compute persistent homology via Vietoris-Rips filtration.

        Uses Backend for vectorized edge extraction and GPU-native
        spanning-forest construction (no host-side Union-Find).
        """
        b = self.backend
        n = int(distances.shape[0])
        persistence_points: list[PersistencePoint] = []

        # Vectorized edge extraction using Backend
        dist_arr = distances
        b.eval(dist_arr)

        # Flatten and sort edges on backend to avoid Python-side sorting.
        flat_dist = b.reshape(dist_arr, (-1,))
        index = b.arange(n)
        row_idx = b.reshape(index, (n, 1))
        col_idx = b.reshape(index, (1, n))
        mask = col_idx > row_idx
        flat_mask = b.reshape(mask, (-1,))
        inf_val = float(b.finfo().max)
        masked_dist = b.where(flat_mask, flat_dist, b.full(flat_dist.shape, inf_val))
        sorted_idx = b.argsort(masked_dist)
        sorted_dist = b.take(masked_dist, sorted_idx, axis=0)
        b.eval(sorted_idx, sorted_dist)

        # Use dtype-derived threshold (not arbitrary 0.9)
        inf_thresh = infinity_threshold(b, sorted_dist)

        # Only pull the edge prefix we actually need (<= max_filtration and finite).
        max_mask = sorted_dist <= max_filtration
        finite_mask = sorted_dist < inf_thresh
        valid_mask = b.astype(max_mask, "int32") * b.astype(finite_mask, "int32")
        edge_count_arr = b.sum(valid_mask)
        b.eval(edge_count_arr)
        edge_count = int(b.to_scalar(edge_count_arr))

        mst_edges: set[int] = set()
        component_roots: list[int] = []

        # Build thresholded distance matrix for spanning forest
        thresh_mask = (dist_arr <= max_filtration) * (dist_arr < inf_thresh)
        dist_thresh = b.where(thresh_mask, dist_arr, b.full(dist_arr.shape, inf_val))
        eye = b.eye(n)
        dist_thresh = dist_thresh + eye * inf_val
        b.eval(dist_thresh)

        index = b.arange(n, dtype="int32")
        ones_int = b.ones((n,), dtype="int32")
        inf_vec = b.full((n,), float("inf"))

        selected = b.zeros((n,), dtype="int32")
        min_dist = inf_vec
        min_parent = b.full((n,), -1, dtype="int32")

        while True:
            unselected_mask = selected == 0
            unselected_count = b.sum(b.astype(unselected_mask, "int32"))
            b.eval(unselected_count)
            if int(b.to_scalar(unselected_count)) == 0:
                break

            idx_candidates = b.where(
                unselected_mask, index, b.full(index.shape, n, dtype="int32")
            )
            root_idx_arr = b.argmin(idx_candidates)
            b.eval(root_idx_arr)
            root_idx = int(b.to_scalar(root_idx_arr))
            component_roots.append(root_idx)

            selected = b.where(index == root_idx, ones_int, selected)

            root_row = b.take(dist_thresh, b.array([root_idx], dtype="int32"), axis=0)
            root_row = b.reshape(root_row, (n,))
            unselected_mask = selected == 0
            min_dist = b.where(unselected_mask, root_row, inf_vec)
            min_parent = b.where(
                unselected_mask,
                b.full(min_parent.shape, root_idx, dtype="int32"),
                min_parent,
            )

            while True:
                masked = b.where(selected == 0, min_dist, inf_vec)
                min_val_arr = b.min(masked)
                b.eval(min_val_arr)
                min_val = float(b.to_scalar(min_val_arr))
                if not is_finite(min_val, b) or min_val > max_filtration:
                    break

                idx_arr = b.argmin(masked)
                b.eval(idx_arr)
                idx = int(b.to_scalar(idx_arr))
                parent_arr = b.take(min_parent, b.array([idx], dtype="int32"), axis=0)
                b.eval(parent_arr)
                parent_idx = int(b.to_scalar(parent_arr))
                if parent_idx < 0:
                    break

                persistence_points.append(PersistencePoint(0.0, min_val, 0))
                edge_id = (
                    idx * n + parent_idx if idx < parent_idx else parent_idx * n + idx
                )
                mst_edges.add(edge_id)

                selected = b.where(index == idx, ones_int, selected)

                row = b.take(dist_thresh, b.array([idx], dtype="int32"), axis=0)
                row = b.reshape(row, (n,))
                better = b.astype(row < min_dist, "int32") * b.astype(
                    selected == 0, "int32"
                )
                min_dist = b.where(better > 0, row, min_dist)
                min_parent = b.where(
                    better > 0,
                    b.full(min_parent.shape, idx, dtype="int32"),
                    min_parent,
                )

        for _ in component_roots:
            persistence_points.append(PersistencePoint(0.0, max_filtration, 0))

        # 1-dim persistence (cycles) with Backend-accelerated triangle detection
        if n >= 3 and edge_count > 0:
            possible_cycles: list[PersistencePoint] = []
            base_mask = b.ones((n,))
            zero_vec = b.full((n,), 0.0)
            inf_vec = b.full((n,), inf_val)

            for edge_idx in range(edge_count):
                idx_arr = b.take(sorted_idx, b.array([edge_idx], dtype="int32"), axis=0)
                dist_arr_edge = b.take(sorted_dist, b.array([edge_idx], dtype="int32"), axis=0)
                b.eval(idx_arr, dist_arr_edge)

                flat_idx_int = int(b.to_scalar(idx_arr))
                if flat_idx_int in mst_edges:
                    continue
                dist_val = float(b.to_scalar(dist_arr_edge))
                if dist_val > max_filtration:
                    break

                i = flat_idx_int // n
                j = flat_idx_int % n

                # Vectorized triangle fill detection using Backend
                row_i = dist_arr[i, :]
                row_j = dist_arr[j, :]
                triangle_fills = b.maximum(
                    b.maximum(row_i, row_j), b.full((n,), dist_val)
                )
                # Mask self-edges to inf
                mask = b.where(index == i, zero_vec, base_mask)
                mask = b.where(index == j, zero_vec, mask)
                masked_fills = b.where(mask > 0, triangle_fills, inf_vec)
                min_fill_arr = b.min(masked_fills)
                b.eval(min_fill_arr)
                min_fill = float(b.to_scalar(min_fill_arr))
                if min_fill < max_filtration and min_fill > dist_val:
                    possible_cycles.append(PersistencePoint(dist_val, min_fill, 1))
                    if len(possible_cycles) >= 20:
                        break

            persistence_points.extend(possible_cycles)

        return PersistenceDiagram(persistence_points)

    @staticmethod
    def _hungarian_backend(cost_matrix: "Array", backend: "Backend") -> "Array":
        """Hungarian algorithm using Backend ops for cost updates."""
        b = backend
        cost = b.array(cost_matrix, dtype="float32")
        b.eval(cost)

        n = int(cost.shape[0])
        if n == 0:
            return b.zeros((0,), dtype="int32")

        size = n + 1
        u = b.zeros((size,), dtype="float32")
        v = b.zeros((size,), dtype="float32")
        p = b.zeros((size,), dtype="int32")
        way = b.zeros((n,), dtype="int32")

        col_idx = b.arange(1, n + 1, dtype="int32")
        row_idx = b.arange(size, dtype="int32")
        row_idx_matrix = b.reshape(row_idx, (1, size))

        inf_vec = b.full((n,), float("inf"), dtype="float32")
        zeros = b.zeros((n,), dtype="float32")
        ones = b.ones((n,), dtype="float32")

        for i in range(1, n + 1):
            p = b.where(row_idx == 0, b.full(p.shape, i, dtype="int32"), p)
            j0 = 0
            minv = inf_vec
            used = zeros
            way = b.zeros((n,), dtype="int32")

            while True:
                if j0 > 0:
                    used = b.where(col_idx == j0, ones, used)

                p_j0 = b.take(p, b.array([j0], dtype="int32"), axis=0)
                b.eval(p_j0)
                i0 = int(b.to_scalar(p_j0))
                if i0 == 0:
                    break

                row = b.take(cost, b.array([i0 - 1], dtype="int32"), axis=0)
                row = b.reshape(row, (n,))
                u_i0 = b.take(u, b.array([i0], dtype="int32"), axis=0)
                v_cols = b.take(v, col_idx, axis=0)
                cur = row - u_i0 - v_cols

                unused_mask = used == 0
                update = unused_mask * (cur < minv)
                minv = b.where(update, cur, minv)
                way = b.where(update, b.full(way.shape, j0, dtype="int32"), way)

                masked_minv = b.where(unused_mask, minv, inf_vec)
                delta = b.min(masked_minv)
                b.eval(delta)
                delta_val = float(b.to_scalar(delta))
                if not is_finite(delta_val, b):
                    break

                j1_idx = b.argmin(masked_minv)
                b.eval(j1_idx)
                j1 = int(b.to_scalar(j1_idx)) + 1

                p_cols = b.take(p, col_idx, axis=0)
                p_cols_row = b.reshape(p_cols, (n, 1))
                used_col = b.reshape(used, (n, 1))
                row_match = b.astype(p_cols_row == row_idx_matrix, "float32")
                counts = b.sum(row_match * used_col, axis=0)

                p0 = b.take(p, b.array([0], dtype="int32"), axis=0)
                p0_mask = row_idx == p0
                p0_one = b.astype(p0_mask, "float32")
                u = u + (counts + p0_one) * delta

                v0 = b.take(v, b.array([0], dtype="int32"), axis=0)
                v0 = v0 - delta
                v_cols = v_cols - used * delta
                v = b.concatenate([v0, v_cols], axis=0)

                minv = b.where(unused_mask, minv - delta, minv)
                j0 = j1

            while True:
                if j0 == 0:
                    break
                way_j0 = b.take(way, b.array([j0 - 1], dtype="int32"), axis=0)
                b.eval(way_j0)
                j1 = int(b.to_scalar(way_j0))
                p_j1 = b.take(p, b.array([j1], dtype="int32"), axis=0)
                b.eval(p_j1)
                p_val = int(b.to_scalar(p_j1))
                p = b.where(row_idx == j0, b.full(p.shape, p_val, dtype="int32"), p)
                j0 = j1

        cols = b.take(p, col_idx, axis=0)
        rows = cols - 1
        row_idx_small = b.arange(n, dtype="int32")
        rows_matrix = b.reshape(rows, (n, 1))
        row_idx_matrix = b.reshape(row_idx_small, (1, n))
        one_hot = b.astype(rows_matrix == row_idx_matrix, "float32")
        col_vals = b.reshape(col_idx - 1, (n, 1))
        result = b.sum(one_hot * col_vals, axis=0)
        return b.astype(result, "int32")

    def _bottleneck_distance(
        self, diag_a: PersistenceDiagram, diag_b: PersistenceDiagram
    ) -> float:
        """Compute bottleneck distance with Backend-accelerated cost matrix."""
        points_a = diag_a.points
        points_b = diag_b.points

        dims = set([p.dimension for p in points_a] + [p.dimension for p in points_b])
        max_dist = 0.0

        b = self.backend

        for dim in dims:
            pa = [p for p in points_a if p.dimension == dim]
            pb = [p for p in points_b if p.dimension == dim]

            n_a, n_b = len(pa), len(pb)

            if n_a == 0 and n_b == 0:
                continue

            if n_a == 0:
                for pt in pb:
                    max_dist = max(max_dist, (pt.death - pt.birth) / 2.0)
                continue
            if n_b == 0:
                for pt in pa:
                    max_dist = max(max_dist, (pt.death - pt.birth) / 2.0)
                continue

            # Vectorized cost matrix building
            n = n_a + n_b
            births_a = b.array([p.birth for p in pa])
            deaths_a = b.array([p.death for p in pa])
            births_b = b.array([p.birth for p in pb])
            deaths_b = b.array([p.death for p in pb])
            b.eval(births_a, deaths_a, births_b, deaths_b)

            # Compute L-infinity costs: max(|birth_a - birth_b|, |death_a - death_b|)
            # Shape: [n_a, n_b]
            birth_diff = b.abs(b.reshape(births_a, (-1, 1)) - b.reshape(births_b, (1, -1)))
            death_diff = b.abs(b.reshape(deaths_a, (-1, 1)) - b.reshape(deaths_b, (1, -1)))
            match_costs = b.maximum(birth_diff, death_diff)
            b.eval(match_costs)

            # Build full cost matrix on backend, then convert once for Hungarian algorithm.
            inf_val = float(b.finfo().max)
            diag_cost_a = b.array([(p.death - p.birth) / 2.0 for p in pa])
            diag_cost_b = b.array([(p.death - p.birth) / 2.0 for p in pb])
            eye_a = b.eye(n_a)
            eye_b = b.eye(n_b)
            diag_block_a = b.where(eye_a > 0, b.diag(diag_cost_a), b.full((n_a, n_a), inf_val))
            diag_block_b = b.where(eye_b > 0, b.diag(diag_cost_b), b.full((n_b, n_b), inf_val))
            top_block = b.concatenate([match_costs, diag_block_a], axis=1)
            bottom_block = b.concatenate([diag_block_b, b.zeros((n_b, n_a))], axis=1)
            cost_matrix_arr = b.concatenate([top_block, bottom_block], axis=0)
            b.eval(cost_matrix_arr)
            matching = self._hungarian_backend(cost_matrix_arr, b)
            row_idx = b.arange(n, dtype="int32")
            flat = b.reshape(cost_matrix_arr, (-1,))
            flat_idx = b.astype(row_idx * n + matching, "int32")
            matched = b.take(flat, flat_idx, axis=0)
            real_mask = (row_idx < n_a) | (matching < n_b)
            finite_mask = b.isfinite(matched)
            neg_inf = -float(b.finfo().max)
            mask = b.astype(real_mask, "int32") * b.astype(finite_mask, "int32")
            masked = b.where(mask > 0, matched, b.full(matched.shape, neg_inf))
            max_arr = b.max(masked)
            b.eval(max_arr)
            max_val = float(b.to_scalar(max_arr))
            if is_finite(max_val, b):
                max_dist = max(max_dist, max_val)

        return max_dist

    def _wasserstein_distance(
        self, diag_a: PersistenceDiagram, diag_b: PersistenceDiagram
    ) -> float:
        """Compute Wasserstein distance with Backend-accelerated cost matrix."""
        total_dist = 0.0
        count = 0

        dims = set([p.dimension for p in diag_a.points] + [p.dimension for p in diag_b.points])
        b = self.backend

        for dim in dims:
            pa = [p for p in diag_a.points if p.dimension == dim]
            pb = [p for p in diag_b.points if p.dimension == dim]

            n_a, n_b = len(pa), len(pb)

            if n_a == 0 and n_b == 0:
                continue

            if n_a == 0:
                for pt in pb:
                    total_dist += pt.death - pt.birth
                    count += 1
                continue
            if n_b == 0:
                for pt in pa:
                    total_dist += pt.death - pt.birth
                    count += 1
                continue

            # Vectorized cost matrix building
            n = n_a + n_b
            births_a = b.array([p.birth for p in pa])
            deaths_a = b.array([p.death for p in pa])
            births_b = b.array([p.birth for p in pb])
            deaths_b = b.array([p.death for p in pb])
            b.eval(births_a, deaths_a, births_b, deaths_b)

            # Compute L1 costs: |birth_a - birth_b| + |death_a - death_b|
            birth_diff = b.abs(b.reshape(births_a, (-1, 1)) - b.reshape(births_b, (1, -1)))
            death_diff = b.abs(b.reshape(deaths_a, (-1, 1)) - b.reshape(deaths_b, (1, -1)))
            match_costs = birth_diff + death_diff
            b.eval(match_costs)

            # Build full cost matrix on backend, then convert once for Hungarian algorithm.
            inf_val = float(b.finfo().max)
            diag_cost_a = b.array([p.death - p.birth for p in pa])
            diag_cost_b = b.array([p.death - p.birth for p in pb])
            eye_a = b.eye(n_a)
            eye_b = b.eye(n_b)
            diag_block_a = b.where(eye_a > 0, b.diag(diag_cost_a), b.full((n_a, n_a), inf_val))
            diag_block_b = b.where(eye_b > 0, b.diag(diag_cost_b), b.full((n_b, n_b), inf_val))
            top_block = b.concatenate([match_costs, diag_block_a], axis=1)
            bottom_block = b.concatenate([diag_block_b, b.zeros((n_b, n_a))], axis=1)
            cost_matrix_arr = b.concatenate([top_block, bottom_block], axis=0)
            b.eval(cost_matrix_arr)
            matching = self._hungarian_backend(cost_matrix_arr, b)
            row_idx = b.arange(n, dtype="int32")
            flat = b.reshape(cost_matrix_arr, (-1,))
            flat_idx = b.astype(row_idx * n + matching, "int32")
            matched = b.take(flat, flat_idx, axis=0)
            real_mask = (row_idx < n_a) | (matching < n_b)
            finite_mask = b.isfinite(matched)
            mask = b.astype(real_mask, "int32") * b.astype(finite_mask, "int32")
            masked = b.where(mask > 0, matched, b.zeros_like(matched))
            sum_arr = b.sum(masked)
            count_arr = b.sum(mask)
            b.eval(sum_arr, count_arr)
            total_dist += float(b.to_scalar(sum_arr))
            count += int(b.to_scalar(count_arr))

        return total_dist / count if count > 0 else 0.0

    def _compute_entropy(self, values: list[float]) -> float:
        """Compute persistence entropy using Backend."""
        if not values:
            return 0.0

        b = self.backend
        arr = b.array(values)
        total = b.sum(arr)
        b.eval(total)
        total_val = float(b.to_scalar(total))

        eps = division_epsilon(b, arr)

        if total_val <= eps:
            return 0.0

        probs = arr / total
        # Compute -sum(p * log(p)), avoiding log(0)
        # Add small epsilon to avoid log(0)
        log_probs = b.log(probs + eps)
        entropy = -b.sum(probs * log_probs)
        b.eval(entropy)

        return float(b.to_scalar(entropy))


def get_topological_fingerprint(
    backend: "Backend | None" = None,
) -> type[TopologicalFingerprint] | BackendTopologicalFingerprint:
    """Get the default topological fingerprint implementation.

    Args:
        backend: Optional Backend instance. If provided, returns
                 BackendTopologicalFingerprint for GPU acceleration.

    Returns:
        TopologicalFingerprint class or BackendTopologicalFingerprint instance.

    Example:
        >>> from modelcypher.core.domain._backend import get_default_backend
        >>> backend = get_default_backend()
        >>> tf = get_topological_fingerprint(backend)
        >>> fingerprint = tf.compute([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    """
    if backend is not None:
        return BackendTopologicalFingerprint(backend)
    return TopologicalFingerprint
