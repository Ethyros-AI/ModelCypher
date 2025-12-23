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

See also: docs/geometry/topological_fingerprints.md
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import math
import mlx.core as mx

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
    points: List[PersistencePoint]
    
    @property
    def count_by_dimension(self) -> Dict[int, int]:
        counts = {}
        for p in self.points:
            counts[p.dimension] = counts.get(p.dimension, 0) + 1
        return counts
        
    def betti_numbers(self, persistence_threshold: float = 0.1) -> Dict[int, int]:
        betti = {}
        for p in self.points:
            if p.persistence >= persistence_threshold:
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
class TopologyConstants:
    persistence_threshold_ratio: float = 0.05
    bottleneck_scale_factor: float = 0.5
    similarity_high_threshold: float = 0.8
    similarity_medium_threshold: float = 0.6
    similarity_low_threshold: float = 0.3
    max_betti_diff: int = 2

@dataclass
class Fingerprint:
    diagram: PersistenceDiagram
    betti_numbers: Dict[int, int]
    summary: TopologySummary

@dataclass
class ComparisonResult:
    bottleneck_distance: float
    wasserstein_distance: float
    betti_difference: int
    similarity_score: float
    is_compatible: bool
    interpretation: str

class TopologicalFingerprint:
    """
    Topological fingerprinting for architecture-invariant model comparison.
    Computes persistent homology (Betti numbers) via Vietoris-Rips filtration.
    """
    
    @staticmethod
    def compute(
        points: List[List[float]],
        max_dimension: int = 1,
        max_filtration: Optional[float] = None,
        num_steps: int = 50
    ) -> Fingerprint:
        if len(points) < 2:
            return Fingerprint(
                diagram=PersistenceDiagram([]),
                betti_numbers={},
                summary=TopologySummary(len(points), 0, 0.0, 0.0, 0.0)
            )
            
        distances = TopologicalFingerprint._compute_pairwise_distances(points)
        
        # Determine filtration range
        all_dists = [d for row in distances for d in row]
        max_dist = max_filtration if max_filtration is not None else (max(all_dists) if all_dists else 1.0)
        min_dist = min([d for d in all_dists if d > 0], default=0.0)
        
        diagram = TopologicalFingerprint._vietoris_rips_filtration(
            distances=distances,
            min_filtration=min_dist,
            max_filtration=max_dist,
            num_steps=num_steps,
            max_dimension=max_dimension
        )
        
        # Filter noise based on max scale
        threshold = max_dist * TopologyConstants.persistence_threshold_ratio
        betti = diagram.betti_numbers(persistence_threshold=threshold)
        
        significant_points = [p for p in diagram.points if p.persistence > threshold]
        persistences = [p.persistence for p in significant_points]
        
        summary = TopologySummary(
            component_count=betti.get(0, 1),
            cycle_count=betti.get(1, 0),
            average_persistence=sum(persistences) / len(persistences) if persistences else 0.0,
            max_persistence=max(persistences) if persistences else 0.0,
            persistence_entropy=TopologicalFingerprint._compute_entropy(persistences)
        )
        
        return Fingerprint(diagram, betti, summary)

    @staticmethod
    def compare(fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> ComparisonResult:
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
            
        scale = max(fingerprint_a.summary.max_persistence, fingerprint_b.summary.max_persistence, 1e-6)
        
        # Similarity score driven by bottleneck (max deviation) and wasserstein (total cost).
        # Betti difference is a hard structural penalty.
        score = math.exp(-bottleneck / scale) * math.exp(-wasserstein / scale) * (1.0 / (1 + betti_diff))
        
        is_compatible = (betti_diff <= TopologyConstants.max_betti_diff and 
                         bottleneck < scale * TopologyConstants.bottleneck_scale_factor)
        
        if score > TopologyConstants.similarity_high_threshold and betti_diff == 0:
            interp = "Identical topological structure."
        elif score > TopologyConstants.similarity_medium_threshold:
            interp = "Similar topological structure."
        elif score > TopologyConstants.similarity_low_threshold:
            interp = "Moderate topological similarity."
        else:
            interp = "Different topological structure."
            
        return ComparisonResult(
            bottleneck_distance=bottleneck,
            wasserstein_distance=wasserstein,
            betti_difference=betti_diff,
            similarity_score=score,
            is_compatible=is_compatible,
            interpretation=interp
        )

    @staticmethod
    def _compute_pairwise_distances(points: List[List[float]]) -> List[List[float]]:
        # Using MLX for speed
        n = len(points)
        if n == 0: return []
        
        pts = mx.array(points)
        # (N, 1, D) - (1, N, D) -> (N, N, D)
        diff = pts[:, None, :] - pts[None, :, :]
        # Norm across last axis
        dists = mx.linalg.norm(diff, axis=2).tolist()
        return dists

    @staticmethod
    def _vietoris_rips_filtration(
        distances: List[List[float]],
        min_filtration: float,
        max_filtration: float,
        num_steps: int,  # unused but kept for interface parity
        max_dimension: int
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
            if px == py: return None
            
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
                if rank[px] < rank[py]: survivor, dying = py, px
                elif rank[px] > rank[py]: survivor, dying = px, py
                else:
                    survivor, dying = px, py
                    rank[px] += 1
            
            parent[dying] = survivor
            return dying

        component_birth = [0.0] * n # All points born at 0 in Rips
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i][j]))
        edges.sort(key=lambda x: x[2])
        
        # Process edges
        for i, j, dist in edges:
            if dist > max_filtration: break
            
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
                
        # 1-dim persistence (cycles) - Simplified Logic
        if max_dimension >= 1:
            # Need to detect when edges complete a cycle but don't merge components
            # And when triangles fill them in
            # This requires full Rips complex or optimized approach.
            # Swift code used a simplified approximation:
            # - Edge creates cycle if vertices already connected
            # - Cycle death = max edge in triangle filling it
            
            # Reset DSU for cycle detection pass
            parent = list(range(n))
            rank = [0] * n
            
            possible_cycles = []
            
            for i, j, dist in edges:
                if dist > max_filtration: break
                
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
                        if k == i or k == j: continue
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
        # Simplified bottleneck: max dist between matched points
        points_a = diag_a.points
        points_b = diag_b.points
        # Warning: Full bottleneck calculation requires bipartite matching (O(N^3)).
        # We use a greedy approximation similar to the Swift code
        
        # Group by dim
        dims = set([p.dimension for p in points_a] + [p.dimension for p in points_b])
        max_dist = 0.0
        
        for dim in dims:
            pa = [p for p in points_a if p.dimension == dim]
            pb = [p for p in points_b if p.dimension == dim]
            
            # Greedy match
            # This is not optimal but matches Swift's heuristic
            used_b = set()
            for a in pa:
                best_j = -1
                min_val = float('inf')
                for j, b in enumerate(pb):
                    if j in used_b: continue
                    val = max(abs(a.birth - b.birth), abs(a.death - b.death))
                    if val < min_val:
                        min_val = val
                        best_j = j
                
                diag_cost = (a.death - a.birth) / 2.0
                if best_j != -1 and min_val <= diag_cost:
                    used_b.add(best_j)
                    max_dist = max(max_dist, min_val)
                else:
                    max_dist = max(max_dist, diag_cost)
            
            for j, b in enumerate(pb):
                if j not in used_b:
                    diag_cost = (b.death - b.birth) / 2.0
                    max_dist = max(max_dist, diag_cost)
                    
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
                    total_dist += (b.death - b.birth)
                    count += 1
                continue
            if n_b == 0:
                for a in pa:
                    total_dist += (a.death - a.birth)
                    count += 1
                continue

            # Build augmented cost matrix for Hungarian algorithm
            # Size: (n_a + n_b) x (n_a + n_b)
            # First n_b columns: matching to points in pb
            # Last n_a columns: matching to diagonal (for points in pa)
            # First n_a rows: points from pa
            # Last n_b rows: points from pb (matched to diagonal)
            n = n_a + n_b
            cost = [[float('inf')] * n for _ in range(n)]

            # Fill matching costs between pa and pb
            for i, a in enumerate(pa):
                for j, b in enumerate(pb):
                    cost[i][j] = abs(a.birth - b.birth) + abs(a.death - b.death)
                # Diagonal cost for point a (matched to its projection on diagonal)
                diag_cost_a = (a.death - a.birth)
                for j in range(n_b, n_b + n_a):
                    if j - n_b == i:
                        cost[i][j] = diag_cost_a
                    else:
                        cost[i][j] = float('inf')

            # Fill diagonal costs for points in pb
            for i, b in enumerate(pb):
                diag_cost_b = (b.death - b.birth)
                row = n_a + i
                for j in range(n_b):
                    if j == i:
                        cost[row][j] = diag_cost_b
                    else:
                        cost[row][j] = float('inf')
                # Zeros for dummy-to-dummy matching
                for j in range(n_b, n):
                    cost[row][j] = 0.0

            # Use Hungarian algorithm to find optimal matching
            matching = TopologicalFingerprint._hungarian_algorithm(cost)

            # Sum up the costs from the matching
            for i, j in enumerate(matching):
                if cost[i][j] < float('inf'):
                    total_dist += cost[i][j]
                    if i < n_a or j < n_b:  # Only count real matches
                        count += 1

        return total_dist / count if count > 0 else 0.0

    @staticmethod
    def _hungarian_algorithm(cost_matrix: List[List[float]]) -> List[int]:
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

        INF = float('inf')

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
    def _compute_entropy(values: List[float]) -> float:
        total = sum(values)
        if total <= 1e-9: return 0.0
        entropy = 0.0
        for v in values:
            p = v / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
