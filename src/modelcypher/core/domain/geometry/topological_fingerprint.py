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


@dataclass(frozen=True)
class TopologyConfig:
    """Configuration for topological fingerprinting.

    ONLY contains numerical stability parameters.
    No arbitrary classification thresholds - the geometry speaks for itself.

    Similarity interpretation is left to the caller who understands their context.
    """

    # Numerical stability threshold
    epsilon: float | None = None



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
        max_dimension: int = 1,
        max_filtration: float | None = None,
        num_steps: int = 50,
        config: TopologyConfig | None = None,
    ) -> Fingerprint:
        if config is None:
            config = TopologyConfig()

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

        if max_filtration is None:
            max_dist = max_dist_val
        else:
            max_dist = max_filtration

        nonzero_mask = (distances > 0) & finite_mask
        min_candidates = backend.where(
            nonzero_mask, distances, backend.full(distances.shape, pos_inf)
        )
        min_dist_arr = backend.min(min_candidates)
        backend.eval(min_dist_arr)
        min_dist_val = float(backend.to_scalar(min_dist_arr))
        min_dist = min_dist_val if is_finite(min_dist_val, backend) else 0.0

        diagram = TopologicalFingerprint._vietoris_rips_filtration(
            distances=distances,
            min_filtration=min_dist,
            max_filtration=max_dist,
            num_steps=num_steps,
            max_dimension=max_dimension,
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
        bottleneck = TopologicalFingerprint._bottleneck_distance(
            fingerprint_a.diagram, fingerprint_b.diagram
        )
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
        )

        n = len(points)
        if n == 0:
            b = backend or get_default_backend()
            return b.zeros((0, 0))

        b = backend or get_default_backend()
        pts = b.array(points)

        # Use geodesic distances for correct topology on curved manifolds
        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(pts, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        return geo_dist

    @staticmethod
    def _vietoris_rips_filtration(
        distances: list[list[float]],
        min_filtration: float,
        max_filtration: float,
        num_steps: int,  # unused but kept for interface parity
        max_dimension: int,
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
            num_steps: Unused (kept for interface compatibility)
            max_dimension: Maximum homology dimension (0 or 1)

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
        if max_dimension >= 1:

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
        max_dimension: int = 1,
        max_filtration: float | None = None,
        num_steps: int = 50,
        config: TopologyConfig | None = None,
    ) -> Fingerprint:
        """Compute topological fingerprint using Backend acceleration.

        Args:
            points: Point cloud as list of vectors.
            max_dimension: Maximum homology dimension (0 or 1).
            max_filtration: Maximum filtration value.
            num_steps: Number of filtration steps (unused, for interface parity).
            config: Topology configuration.

        Returns:
            Fingerprint with persistence diagram and summary statistics.
        """
        if config is None:
            config = TopologyConfig()

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
        max_dist = max_filtration if max_filtration is not None else max_dist_val

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
            num_steps=num_steps,
            max_dimension=max_dimension,
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
        bottleneck = self._bottleneck_distance(fingerprint_a.diagram, fingerprint_b.diagram)
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
            exp_scalar(-bottleneck / scale, b)
            * exp_scalar(-wasserstein / scale, b)
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
        )

        n = len(points)
        if n == 0:
            return self.backend.zeros((0, 0))

        b = self.backend
        pts = b.array(points)

        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(pts, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        return geo_dist

    def _vietoris_rips_filtration(
        self,
        distances: "Array",
        min_filtration: float,
        max_filtration: float,
        num_steps: int,
        max_dimension: int,
    ) -> PersistenceDiagram:
        """Compute persistent homology via Vietoris-Rips filtration.

        Uses Backend for vectorized edge extraction, but Union-Find
        remains sequential as it's inherently order-dependent.
        """
        b = self.backend
        n = int(distances.shape[0])
        persistence_points: list[PersistencePoint] = []

        # Vectorized edge extraction using Backend
        dist_arr = distances
        b.eval(dist_arr)

        # Get upper triangular indices and values
        # Build edges list with a single device→CPU transfer.
        edges: list[tuple[int, int, float]] = []
        dist_list = b.tolist(dist_arr)
        for i in range(n):
            row = dist_list[i]
            for j in range(i + 1, n):
                edges.append((i, j, float(row[j])))
        edges.sort(key=lambda x: x[2])

        # 0-dim persistence (connected components) - Union-Find is sequential
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int, component_birth: list[float]) -> int | None:
            px, py = find(x), find(y)
            if px == py:
                return None

            birth_x = component_birth[px]
            birth_y = component_birth[py]

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

        component_birth = [0.0] * n

        for i, j, dist in edges:
            if dist > max_filtration:
                break

            dying = union(i, j, component_birth)
            if dying is not None:
                birth = component_birth[dying]
                death = dist
                if death > birth:
                    persistence_points.append(PersistencePoint(birth, death, 0))

        for i in range(n):
            if find(i) == i:
                persistence_points.append(PersistencePoint(0.0, max_filtration, 0))

        # 1-dim persistence (cycles) with Backend-accelerated triangle detection
        if max_dimension >= 1:
            parent = list(range(n))
            rank = [0] * n

            possible_cycles: list[PersistencePoint] = []
            index = b.arange(n)
            b.eval(index)

            for i, j, dist in edges:
                if dist > max_filtration:
                    break

                px, py = find(i), find(j)
                if px == py:
                    # Vectorized triangle fill detection using Backend
                    # For each k, compute max(dist, d[i,k], d[j,k])
                    row_i = dist_arr[i, :]
                    row_j = dist_arr[j, :]
                    triangle_fills = b.maximum(b.maximum(row_i, row_j), b.full((n,), dist))
                    # Mask self-edges to inf
                    inf_val = float(b.finfo().max)
                    mask = b.ones((n,))
                    mask = b.where(index == i, b.full(mask.shape, 0.0), mask)
                    mask = b.where(index == j, b.full(mask.shape, 0.0), mask)
                    masked_fills = b.where(mask > 0, triangle_fills, b.full(triangle_fills.shape, inf_val))
                    min_fill = float(b.to_scalar(b.min(masked_fills)))
                    if min_fill < max_filtration and min_fill > dist:
                        possible_cycles.append(PersistencePoint(dist, min_fill, 1))
                else:
                    union(i, j, component_birth)

            persistence_points.extend(possible_cycles[:20])

        return PersistenceDiagram(persistence_points)

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

            # Build full cost matrix
            cost = [[float("inf")] * n for _ in range(n)]
            match_costs_list = b.tolist(match_costs)
            for i in range(n_a):
                row_costs = match_costs_list[i]
                for j in range(n_b):
                    cost[i][j] = float(row_costs[j])
                diag_cost_a = (pa[i].death - pa[i].birth) / 2.0
                cost[i][n_b + i] = diag_cost_a

            for i in range(n_b):
                diag_cost_b = (pb[i].death - pb[i].birth) / 2.0
                cost[n_a + i][i] = diag_cost_b
                for j in range(n_b, n):
                    cost[n_a + i][j] = 0.0

            matching = TopologicalFingerprint._hungarian_algorithm(cost)

            for i, j in enumerate(matching):
                if i < n_a or j < n_b:
                    if cost[i][j] < float("inf"):
                        max_dist = max(max_dist, cost[i][j])

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

            # Build full cost matrix
            cost = [[float("inf")] * n for _ in range(n)]
            match_costs_list = b.tolist(match_costs)
            for i in range(n_a):
                row_costs = match_costs_list[i]
                for j in range(n_b):
                    cost[i][j] = float(row_costs[j])
                diag_cost_a = pa[i].death - pa[i].birth
                cost[i][n_b + i] = diag_cost_a

            for i in range(n_b):
                diag_cost_b = pb[i].death - pb[i].birth
                cost[n_a + i][i] = diag_cost_b
                for j in range(n_b, n):
                    cost[n_a + i][j] = 0.0

            matching = TopologicalFingerprint._hungarian_algorithm(cost)

            for i, j in enumerate(matching):
                if cost[i][j] < float("inf"):
                    total_dist += cost[i][j]
                    if i < n_a or j < n_b:
                        count += 1

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
    """Get the best available topological fingerprint implementation.

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
