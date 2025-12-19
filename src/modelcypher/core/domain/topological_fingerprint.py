from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PersistencePoint:
    birth: float
    death: float
    dimension: int

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True)
class PersistenceDiagram:
    points: list[PersistencePoint]

    @property
    def count_by_dimension(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for point in self.points:
            counts[point.dimension] = counts.get(point.dimension, 0) + 1
        return counts

    def betti_numbers(self, persistence_threshold: float = 0.1) -> dict[int, int]:
        betti: dict[int, int] = {}
        for point in self.points:
            if point.persistence >= persistence_threshold:
                betti[point.dimension] = betti.get(point.dimension, 0) + 1
        return betti

    @property
    def total_persistence(self) -> dict[int, float]:
        totals: dict[int, float] = {}
        for point in self.points:
            totals[point.dimension] = totals.get(point.dimension, 0.0) + point.persistence
        return totals


@dataclass(frozen=True)
class TopologySummary:
    component_count: int
    cycle_count: int
    average_persistence: float
    max_persistence: float
    persistence_entropy: float


@dataclass(frozen=True)
class Fingerprint:
    diagram: PersistenceDiagram
    betti_numbers: dict[int, int]
    summary: TopologySummary


@dataclass(frozen=True)
class ComparisonResult:
    bottleneck_distance: float
    wasserstein_distance: float
    betti_difference: int
    similarity_score: float
    is_compatible: bool
    interpretation: str


class TopologicalFingerprint:
    @staticmethod
    def compute(
        points: list[list[float]],
        max_dimension: int = 1,
        max_filtration: float | None = None,
        num_steps: int = 50,
    ) -> Fingerprint:
        if len(points) < 2:
            return Fingerprint(
                diagram=PersistenceDiagram(points=[]),
                betti_numbers={},
                summary=TopologySummary(
                    component_count=len(points),
                    cycle_count=0,
                    average_persistence=0.0,
                    max_persistence=0.0,
                    persistence_entropy=0.0,
                ),
            )

        distances = TopologicalFingerprint._compute_pairwise_distances(points)
        all_dists = [value for row in distances for value in row]
        max_dist = max_filtration if max_filtration is not None else (max(all_dists) if all_dists else 1.0)
        min_dist = min([val for val in all_dists if val > 0], default=0.0)

        diagram = TopologicalFingerprint._vietoris_rips_filtration(
            distances=distances,
            min_filtration=min_dist,
            max_filtration=max_dist,
            num_steps=num_steps,
            max_dimension=max_dimension,
        )

        betti_numbers = diagram.betti_numbers(persistence_threshold=max_dist * 0.05)
        significant = [point for point in diagram.points if point.persistence > max_dist * 0.05]
        persistences = [point.persistence for point in significant]

        summary = TopologySummary(
            component_count=betti_numbers.get(0, 1),
            cycle_count=betti_numbers.get(1, 0),
            average_persistence=(sum(persistences) / float(len(persistences))) if persistences else 0.0,
            max_persistence=max(persistences) if persistences else 0.0,
            persistence_entropy=TopologicalFingerprint._compute_entropy(persistences),
        )

        return Fingerprint(
            diagram=diagram,
            betti_numbers=betti_numbers,
            summary=summary,
        )

    @staticmethod
    def compare(
        fingerprint_a: Fingerprint,
        fingerprint_b: Fingerprint,
    ) -> ComparisonResult:
        bottleneck = TopologicalFingerprint._bottleneck_distance(
            diagram_a=fingerprint_a.diagram,
            diagram_b=fingerprint_b.diagram,
        )
        wasserstein = TopologicalFingerprint._wasserstein_distance(
            diagram_a=fingerprint_a.diagram,
            diagram_b=fingerprint_b.diagram,
        )

        betti_diff = 0
        all_dims = set(fingerprint_a.betti_numbers.keys()).union(set(fingerprint_b.betti_numbers.keys()))
        for dim in all_dims:
            betti_diff += abs(
                fingerprint_a.betti_numbers.get(dim, 0) - fingerprint_b.betti_numbers.get(dim, 0)
            )

        scale = max(fingerprint_a.summary.max_persistence, fingerprint_b.summary.max_persistence, 1e-6)
        score = math.exp(-bottleneck / scale) * math.exp(-wasserstein / scale) * (1.0 / float(1 + betti_diff))
        is_compatible = betti_diff <= 2 and bottleneck < scale * 0.5

        if score > 0.8 and betti_diff == 0:
            interpretation = (
                "Identical topological structure. Models share the same concept organization at a fundamental level."
            )
        elif score > 0.6:
            interpretation = (
                "Similar topological structure. Minor differences in concept clustering or relationships."
            )
        elif score > 0.3:
            interpretation = (
                "Moderate topological similarity. Models organize concepts differently in some areas."
            )
        else:
            interpretation = (
                "Different topological structure. Models have fundamentally different concept organization."
            )

        return ComparisonResult(
            bottleneck_distance=bottleneck,
            wasserstein_distance=wasserstein,
            betti_difference=betti_diff,
            similarity_score=score,
            is_compatible=is_compatible,
            interpretation=interpretation,
        )

    @staticmethod
    def _compute_pairwise_distances(points: list[list[float]]) -> list[list[float]]:
        n = len(points)
        distances = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sum_sq = 0.0
                d = min(len(points[i]), len(points[j]))
                for k in range(d):
                    diff = points[i][k] - points[j][k]
                    sum_sq += diff * diff
                dist = math.sqrt(sum_sq)
                distances[i][j] = dist
                distances[j][i] = dist
        return distances

    @staticmethod
    def _vietoris_rips_filtration(
        distances: list[list[float]],
        min_filtration: float,
        max_filtration: float,
        num_steps: int,
        max_dimension: int,
    ) -> PersistenceDiagram:
        _ = min_filtration
        _ = num_steps
        n = len(distances)
        persistence_points: list[PersistencePoint] = []

        parent = list(range(n))
        rank = [0 for _ in range(n)]

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int, component_birth: list[float]) -> int | None:
            px = find(x)
            py = find(y)
            if px == py:
                return None
            birth_x = component_birth[px]
            birth_y = component_birth[py]

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

        component_birth = [0.0 for _ in range(n)]

        edges: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i][j]))
        edges.sort(key=lambda item: item[2])

        for i, j, dist in edges:
            if dist > max_filtration:
                continue
            dying_root = union(i, j, component_birth)
            if dying_root is None:
                continue
            birth = component_birth[dying_root]
            death = dist
            if death > birth:
                persistence_points.append(PersistencePoint(birth=birth, death=death, dimension=0))

        for i in range(n):
            if find(i) == i:
                persistence_points.append(PersistencePoint(birth=0.0, death=max_filtration, dimension=0))

        if max_dimension >= 1:
            cycle_edges: list[tuple[float, float]] = []
            parent = list(range(n))
            rank = [0 for _ in range(n)]

            for i, j, dist in edges:
                if dist > max_filtration:
                    continue
                px = find(i)
                py = find(j)
                if px == py:
                    cycle_birth = dist
                    cycle_death = max_filtration
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        dik = distances[i][k]
                        djk = distances[j][k]
                        triangle_dist = max(dist, max(dik, djk))
                        if triangle_dist < cycle_death:
                            cycle_death = triangle_dist
                    if cycle_death > cycle_birth:
                        cycle_edges.append((cycle_birth, cycle_death))
                else:
                    union(i, j, component_birth)

            for birth, death in cycle_edges[:10]:
                persistence_points.append(PersistencePoint(birth=birth, death=death, dimension=1))

        return PersistenceDiagram(points=persistence_points)

    @staticmethod
    def _bottleneck_distance(
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
    ) -> float:
        points_a: dict[int, list[tuple[float, float]]] = {}
        points_b: dict[int, list[tuple[float, float]]] = {}

        for point in diagram_a.points:
            points_a.setdefault(point.dimension, []).append((point.birth, point.death))
        for point in diagram_b.points:
            points_b.setdefault(point.dimension, []).append((point.birth, point.death))

        max_bottleneck = 0.0
        all_dims = set(points_a.keys()).union(points_b.keys())
        for dim in all_dims:
            p_a = points_a.get(dim, [])
            p_b = points_b.get(dim, [])
            used_b: set[int] = set()
            for a in p_a:
                min_dist = float("inf")
                best_j: int | None = None
                for j, b in enumerate(p_b):
                    if j in used_b:
                        continue
                    dist = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
                diag_dist = (a[1] - a[0]) / 2.0
                if min_dist <= diag_dist and best_j is not None:
                    used_b.add(best_j)
                    max_bottleneck = max(max_bottleneck, min_dist)
                else:
                    max_bottleneck = max(max_bottleneck, diag_dist)

            for j, b in enumerate(p_b):
                if j in used_b:
                    continue
                diag_dist = (b[1] - b[0]) / 2.0
                max_bottleneck = max(max_bottleneck, diag_dist)

        return max_bottleneck

    @staticmethod
    def _wasserstein_distance(
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
    ) -> float:
        total_dist = 0.0
        count = 0

        points_a: dict[int, list[tuple[float, float]]] = {}
        points_b: dict[int, list[tuple[float, float]]] = {}

        for point in diagram_a.points:
            points_a.setdefault(point.dimension, []).append((point.birth, point.death))
        for point in diagram_b.points:
            points_b.setdefault(point.dimension, []).append((point.birth, point.death))

        all_dims = set(points_a.keys()).union(points_b.keys())
        for dim in all_dims:
            p_a = points_a.get(dim, [])
            p_b = points_b.get(dim, [])
            used_b: set[int] = set()
            for a in p_a:
                best_j = -1
                min_dist = float("inf")
                for j, b in enumerate(p_b):
                    if j in used_b:
                        continue
                    dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
                diag_dist = a[1] - a[0]
                if best_j >= 0 and min_dist < diag_dist:
                    used_b.add(best_j)
                    total_dist += min_dist
                else:
                    total_dist += diag_dist
                count += 1

            for j, b in enumerate(p_b):
                if j in used_b:
                    continue
                total_dist += b[1] - b[0]
                count += 1

        return total_dist / float(count) if count > 0 else 0.0

    @staticmethod
    def _compute_entropy(values: list[float]) -> float:
        if not values:
            return 0.0
        total = sum(values)
        if total <= 0:
            return 0.0
        entropy = 0.0
        for value in values:
            p = value / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
