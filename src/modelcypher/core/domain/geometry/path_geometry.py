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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from modelcypher.core.domain.geometry.vector_math import VectorMath


@dataclass(frozen=True)
class PathNode:
    gate_id: str
    token_index: int
    entropy: float
    embedding: list[float] | None = None


@dataclass(frozen=True)
class PathSignature:
    model_id: str
    prompt_id: str
    nodes: list[PathNode]
    id: UUID = field(default_factory=uuid4)

    @property
    def gate_sequence(self) -> list[str]:
        return [node.gate_id for node in self.nodes]


class AlignmentOp(str, Enum):
    match = "match"
    insert = "insert"
    delete = "delete"
    substitute = "substitute"


@dataclass(frozen=True)
class AlignmentStep:
    op: AlignmentOp
    node_a: PathNode | None
    node_b: PathNode | None
    cost: float


@dataclass(frozen=True)
class PathComparison:
    total_distance: float
    normalized_distance: float
    alignment: list[AlignmentStep]


@dataclass(frozen=True)
class FrechetResult:
    distance: float
    optimal_coupling: list[tuple[int, int]]


@dataclass(frozen=True)
class DTWResult:
    total_cost: float
    normalized_cost: float
    warping_path: list[tuple[int, int]]
    compression_ratio: float


@dataclass(frozen=True)
class TruncatedSignature:
    level1: list[float]
    level2: list[list[float]]
    signed_area: float
    signature_norm: float


@dataclass(frozen=True)
class EntropyPathAnalysis:
    total_entropy: float
    mean_entropy: float
    entropy_variance: float
    max_entropy: float
    max_entropy_index: int
    mean_gradient: float
    spike_count: int
    spike_indices: list[int]
    stability_score: float


@dataclass(frozen=True)
class LocalGeometry:
    curvatures: list[float]
    mean_curvature: float
    max_curvature: float
    total_curvature: float
    torsions: list[float]
    mean_torsion: float


@dataclass(frozen=True)
class ComprehensiveComparison:
    """Result of comprehensive trajectory comparison.

    Contains objective geometric quantities only - no subjective
    categorical classifications. Downstream consumers may apply
    their own task-specific thresholds to these values.
    """

    levenshtein: PathComparison
    frechet: FrechetResult
    dtw: DTWResult
    signature_similarity: float
    overall_similarity: float


@dataclass
class SimilarityWeights:
    """
    Weights for combining distance metrics into overall similarity.

    Default: Equal weights (0.25 each).

    Information-theoretic justification: When combining metrics of
    comparable scale with no prior about which is more informative,
    equal weighting is the maximum-entropy (least-assuming) choice.

    Alternative weightings can be derived via:
    - Inverse-variance weighting if metric variances are known
    - Cross-validation on labeled trajectory pairs
    - Domain-specific knowledge about metric relevance
    """

    # Weight for Levenshtein (edit distance) similarity.
    # Captures structural/sequential similarity.
    levenshtein_weight: float = 0.25

    # Weight for Frechet distance similarity.
    # Captures worst-case geometric deviation.
    frechet_weight: float = 0.25

    # Weight for DTW (Dynamic Time Warping) similarity.
    # Captures temporal alignment quality.
    dtw_weight: float = 0.25

    # Weight for path signature similarity.
    # Captures geometric shape invariants.
    signature_weight: float = 0.25

    def __post_init__(self) -> None:
        total = (
            self.levenshtein_weight + self.frechet_weight + self.dtw_weight + self.signature_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class SignatureSimilarityWeights:
    """
    Weights for combining signature distance components.

    The path signature encodes geometric information at multiple levels:
    - Level 1: First-order increments (displacement)
    - Level 2: Second-order (curvature, enclosed area)

    Equal contribution from each signature component.
    """

    # L1 distance weight (level-1 signature component).
    l1_weight: float = 1.0

    # Signed area difference weight (level-2 antisymmetric part).
    area_weight: float = 1.0

    # Signature norm difference weight.
    norm_weight: float = 1.0


class PathGeometry:
    @staticmethod
    def compare(
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> PathComparison:
        n = len(path_a.nodes)
        m = len(path_b.nodes)

        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        ops = [[AlignmentOp.match] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = float(i) * 1.0
            ops[i][0] = AlignmentOp.delete
        for j in range(m + 1):
            dp[0][j] = float(j) * 1.0
            ops[0][j] = AlignmentOp.insert
        ops[0][0] = AlignmentOp.match

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                node_a = path_a.nodes[i - 1]
                node_b = path_b.nodes[j - 1]

                if node_a.gate_id == node_b.gate_id:
                    sub_cost = 0.0
                else:
                    vec_a = gate_embeddings.get(node_a.gate_id)
                    vec_b = gate_embeddings.get(node_b.gate_id)
                    if vec_a and vec_b:
                        sim = VectorMath.cosine_similarity(vec_a, vec_b) or 0.0
                        sub_cost = 1.0 - sim
                    else:
                        sub_cost = 1.0

                cost_match = dp[i - 1][j - 1] + sub_cost
                cost_del = dp[i - 1][j] + 1.0
                cost_ins = dp[i][j - 1] + 1.0

                if cost_match <= cost_del and cost_match <= cost_ins:
                    dp[i][j] = cost_match
                    ops[i][j] = (
                        AlignmentOp.match
                        if node_a.gate_id == node_b.gate_id
                        else AlignmentOp.substitute
                    )
                elif cost_del <= cost_ins:
                    dp[i][j] = cost_del
                    ops[i][j] = AlignmentOp.delete
                else:
                    dp[i][j] = cost_ins
                    ops[i][j] = AlignmentOp.insert

        total_cost = dp[n][m]
        max_length = float(max(n, m))
        max_step_cost = 2.0
        normalized = total_cost / (max_length * max_step_cost) if max_length > 0 else 0.0

        alignment: list[AlignmentStep] = []
        i = n
        j = m
        while i > 0 or j > 0:
            op = ops[i][j]
            if op in {AlignmentOp.match, AlignmentOp.substitute}:
                cost = dp[i][j] - dp[i - 1][j - 1]
                alignment.append(
                    AlignmentStep(
                        op=op, node_a=path_a.nodes[i - 1], node_b=path_b.nodes[j - 1], cost=cost
                    )
                )
                i -= 1
                j -= 1
            elif op == AlignmentOp.delete:
                cost = dp[i][j] - dp[i - 1][j]
                alignment.append(
                    AlignmentStep(op=op, node_a=path_a.nodes[i - 1], node_b=None, cost=cost)
                )
                i -= 1
            else:
                cost = dp[i][j] - dp[i][j - 1]
                alignment.append(
                    AlignmentStep(op=op, node_a=None, node_b=path_b.nodes[j - 1], cost=cost)
                )
                j -= 1

        alignment.reverse()
        return PathComparison(
            total_distance=total_cost, normalized_distance=normalized, alignment=alignment
        )

    @staticmethod
    def frechet_distance(
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> FrechetResult:
        n = len(path_a.nodes)
        m = len(path_b.nodes)
        if n == 0 or m == 0:
            return FrechetResult(distance=float("inf"), optimal_coupling=[])

        dp = [[float("inf")] * m for _ in range(n)]

        def dist(i: int, j: int) -> float:
            node_a = path_a.nodes[i]
            node_b = path_b.nodes[j]
            if node_a.gate_id == node_b.gate_id:
                return abs(node_a.entropy - node_b.entropy)
            vec_a = gate_embeddings.get(node_a.gate_id)
            vec_b = gate_embeddings.get(node_b.gate_id)
            if vec_a and vec_b:
                return 1.0 - (VectorMath.cosine_similarity(vec_a, vec_b) or 0.0)
            return 1.0

        dp[0][0] = dist(0, 0)
        for j in range(1, m):
            dp[0][j] = max(dp[0][j - 1], dist(0, j))
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dist(i, 0))
        for i in range(1, n):
            for j in range(1, m):
                d_val = dist(i, j)
                dp[i][j] = max(d_val, min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]))

        coupling: list[tuple[int, int]] = []
        i = n - 1
        j = m - 1
        coupling.append((i, j))
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                options = [
                    (dp[i - 1][j - 1], i - 1, j - 1),
                    (dp[i - 1][j], i - 1, j),
                    (dp[i][j - 1], i, j - 1),
                ]
                best = min(options, key=lambda item: item[0])
                i, j = best[1], best[2]
            coupling.append((i, j))

        return FrechetResult(distance=dp[n - 1][m - 1], optimal_coupling=list(reversed(coupling)))

    @staticmethod
    def dynamic_time_warping(
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
        window_size: int | None = None,
    ) -> DTWResult:
        n = len(path_a.nodes)
        m = len(path_b.nodes)
        if n == 0 or m == 0:
            return DTWResult(
                total_cost=float("inf"),
                normalized_cost=float("inf"),
                warping_path=[],
                compression_ratio=0.0,
            )

        def dist(i: int, j: int) -> float:
            node_a = path_a.nodes[i]
            node_b = path_b.nodes[j]
            if node_a.gate_id == node_b.gate_id:
                d_val = 0.0
            else:
                vec_a = gate_embeddings.get(node_a.gate_id)
                vec_b = gate_embeddings.get(node_b.gate_id)
                if vec_a and vec_b:
                    d_val = 1.0 - (VectorMath.cosine_similarity(vec_a, vec_b) or 0.0)
                else:
                    d_val = 1.0
            entropy_diff = abs(node_a.entropy - node_b.entropy) / 10.0
            return d_val + entropy_diff

        def in_window(i: int, j: int) -> bool:
            if window_size is None:
                return True
            return abs(i - j) <= window_size

        dp = [[float("inf")] * m for _ in range(n)]
        dp[0][0] = dist(0, 0)

        for j in range(1, m):
            if in_window(0, j):
                dp[0][j] = dp[0][j - 1] + dist(0, j)

        for i in range(1, n):
            if in_window(i, 0):
                dp[i][0] = dp[i - 1][0] + dist(i, 0)

        for i in range(1, n):
            for j in range(1, m):
                if not in_window(i, j):
                    continue
                d_val = dist(i, j)
                dp[i][j] = d_val + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        final_cost = dp[n - 1][m - 1]
        if not math.isfinite(final_cost):
            return DTWResult(
                total_cost=float("inf"),
                normalized_cost=float("inf"),
                warping_path=[],
                compression_ratio=0.0,
            )

        path: list[tuple[int, int]] = []
        i = n - 1
        j = m - 1
        path.append((i, j))
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                options = [
                    (dp[i - 1][j - 1], i - 1, j - 1),
                    (dp[i - 1][j], i - 1, j),
                    (dp[i][j - 1], i, j - 1),
                ]
                best = min(options, key=lambda item: item[0])
                i, j = best[1], best[2]
            path.append((i, j))

        path_length = float(len(path))
        normalized_cost = final_cost / path_length if path_length > 0 else 0.0
        ideal_length = float(max(n, m))
        compression_ratio = path_length / ideal_length if ideal_length > 0 else 0.0

        return DTWResult(
            total_cost=final_cost,
            normalized_cost=normalized_cost,
            warping_path=list(reversed(path)),
            compression_ratio=compression_ratio,
        )

    @staticmethod
    def compute_signature(
        path: PathSignature,
        gate_embeddings: dict[str, list[float]],
        projection_dim: int = 8,
    ) -> TruncatedSignature:
        if len(path.nodes) < 2:
            return TruncatedSignature(
                level1=[0.0] * projection_dim,
                level2=[[0.0] * projection_dim for _ in range(projection_dim)],
                signed_area=0.0,
                signature_norm=0.0,
            )

        coords: list[list[float]] = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append(node.entropy / 10.0)
                coords.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append(node.entropy / 10.0)
                coords.append(proj)

        level1 = [0.0] * projection_dim
        for i in range(1, len(coords)):
            for d in range(projection_dim):
                level1[d] += coords[i][d] - coords[i - 1][d]

        level2 = [[0.0] * projection_dim for _ in range(projection_dim)]
        cumulative = [0.0] * projection_dim
        for i in range(1, len(coords)):
            d_x = [coords[i][d] - coords[i - 1][d] for d in range(projection_dim)]
            for p in range(projection_dim):
                cumulative_p = cumulative[p]
                d_xp = d_x[p]
                for q in range(projection_dim):
                    level2[p][q] += cumulative_p * d_x[q] + 0.5 * d_xp * d_x[q]
            for d in range(projection_dim):
                cumulative[d] += d_x[d]

        antisymmetric_sum = 0.0
        for p in range(projection_dim):
            for q in range(p + 1, projection_dim):
                antisym = level2[p][q] - level2[q][p]
                antisymmetric_sum += antisym * antisym
        signed_area = 0.5 * math.sqrt(antisymmetric_sum)

        norm = 0.0
        for d in range(projection_dim):
            norm += level1[d] * level1[d]
        for p in range(projection_dim):
            for q in range(projection_dim):
                norm += level2[p][q] * level2[p][q]
        signature_norm = math.sqrt(norm)

        return TruncatedSignature(
            level1=level1,
            level2=level2,
            signed_area=signed_area,
            signature_norm=signature_norm,
        )

    @staticmethod
    def signature_similarity(
        sig_a: TruncatedSignature,
        sig_b: TruncatedSignature,
        weights: SignatureSimilarityWeights | None = None,
    ) -> float:
        """Compute similarity between two path signatures.

        Args:
            sig_a: First signature.
            sig_b: Second signature.
            weights: Optional weights for combining distance components.
                     Defaults to equal-ish weighting if not provided.
        """
        w = weights or SignatureSimilarityWeights()

        l1_dist = 0.0
        count = min(len(sig_a.level1), len(sig_b.level1))
        for i in range(count):
            diff = sig_a.level1[i] - sig_b.level1[i]
            l1_dist += diff * diff
        l1_dist = math.sqrt(l1_dist)

        area_diff = abs(sig_a.signed_area - sig_b.signed_area)
        norm_diff = abs(sig_a.signature_norm - sig_b.signature_norm)
        total_dist = w.l1_weight * l1_dist + w.area_weight * area_diff + w.norm_weight * norm_diff
        return 1.0 / (1.0 + total_dist)

    @staticmethod
    def analyze_entropy_path(path: PathSignature) -> EntropyPathAnalysis:
        if not path.nodes:
            return EntropyPathAnalysis(
                total_entropy=0.0,
                mean_entropy=0.0,
                entropy_variance=0.0,
                max_entropy=0.0,
                max_entropy_index=0,
                mean_gradient=0.0,
                spike_count=0,
                spike_indices=[],
                stability_score=1.0,
            )

        entropies = [node.entropy for node in path.nodes]
        n = float(len(entropies))
        total = sum(entropies)
        mean = total / n

        variance = 0.0
        max_val = entropies[0]
        max_idx = 0
        for i, val in enumerate(entropies):
            variance += (val - mean) * (val - mean)
            if val > max_val:
                max_val = val
                max_idx = i
        variance = variance / n
        std_dev = math.sqrt(variance)

        gradients = [entropies[i] - entropies[i - 1] for i in range(1, len(entropies))]
        mean_gradient = sum(gradients) / len(gradients) if gradients else 0.0

        spike_threshold = mean + 2.0 * std_dev
        spikes = [i for i, val in enumerate(entropies) if val > spike_threshold]

        variance_score = 1.0 / (1.0 + variance)
        spike_score = 1.0 / (1.0 + float(len(spikes)))
        max_score = 1.0 / (1.0 + max_val / 10.0)
        stability_score = (variance_score + spike_score + max_score) / 3.0

        return EntropyPathAnalysis(
            total_entropy=total,
            mean_entropy=mean,
            entropy_variance=variance,
            max_entropy=max_val,
            max_entropy_index=max_idx,
            mean_gradient=mean_gradient,
            spike_count=len(spikes),
            spike_indices=spikes,
            stability_score=stability_score,
        )

    @staticmethod
    def compute_local_geometry(
        path: PathSignature,
        gate_embeddings: dict[str, list[float]],
        projection_dim: int = 8,
    ) -> LocalGeometry:
        if len(path.nodes) < 3:
            return LocalGeometry(
                curvatures=[],
                mean_curvature=0.0,
                max_curvature=0.0,
                total_curvature=0.0,
                torsions=[],
                mean_torsion=0.0,
            )

        coords = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append(node.entropy / 10.0)
                coords.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append(node.entropy / 10.0)
                coords.append(proj)

        tangents: list[list[float]] = []
        for i in range(len(coords) - 1):
            t = [0.0] * projection_dim
            norm = 0.0
            for d in range(projection_dim):
                t[d] = coords[i + 1][d] - coords[i][d]
                norm += t[d] * t[d]
            norm = math.sqrt(norm)
            if norm > 0:
                t = [val / norm for val in t]
            tangents.append(t)

        curvatures: list[float] = []
        for i in range(len(tangents) - 1):
            dot = sum(tangents[i][d] * tangents[i + 1][d] for d in range(projection_dim))
            dot = max(-1.0, min(1.0, dot))
            curvatures.append(math.acos(dot))

        mean_curv = sum(curvatures) / len(curvatures) if curvatures else 0.0
        max_curv = max(curvatures) if curvatures else 0.0
        total_curv = sum(curvatures)

        torsions: list[float] = []
        if len(tangents) >= 3:
            for i in range(len(tangents) - 2):
                t1 = tangents[i]
                t2 = tangents[i + 1]
                t3 = tangents[i + 2]
                deviation = 0.0
                for d in range(projection_dim):
                    expected = (t1[d] + t3[d]) / 2.0
                    deviation += (t2[d] - expected) ** 2
                torsions.append(math.sqrt(deviation))

        mean_tors = sum(torsions) / len(torsions) if torsions else 0.0

        return LocalGeometry(
            curvatures=curvatures,
            mean_curvature=mean_curv,
            max_curvature=max_curv,
            total_curvature=total_curv,
            torsions=torsions,
            mean_torsion=mean_tors,
        )

    @staticmethod
    def comprehensive_compare(
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
        similarity_weights: SimilarityWeights | None = None,
    ) -> ComprehensiveComparison:
        """Comprehensive trajectory comparison using multiple distance metrics.

        Computes four distance metrics and combines them into an overall
        similarity score using the provided weights. Returns only objective
        geometric quantities - no subjective categorical classifications.

        Args:
            path_a: First path signature.
            path_b: Second path signature.
            gate_embeddings: Embedding vectors for gate IDs.
            similarity_weights: Optional weights for combining metrics.
                                Defaults to equal weights (maximum-entropy prior).

        Returns:
            ComprehensiveComparison with individual metrics and overall similarity.
        """
        sw = similarity_weights or SimilarityWeights()

        lev = PathGeometry.compare(path_a, path_b, gate_embeddings)
        frech = PathGeometry.frechet_distance(path_a, path_b, gate_embeddings)
        dtw = PathGeometry.dynamic_time_warping(path_a, path_b, gate_embeddings)

        sig_a = PathGeometry.compute_signature(path_a, gate_embeddings)
        sig_b = PathGeometry.compute_signature(path_b, gate_embeddings)
        sig_sim = PathGeometry.signature_similarity(sig_a, sig_b)

        lev_sim = 1.0 - lev.normalized_distance
        frech_sim = 1.0 / (1.0 + frech.distance)
        dtw_sim = 1.0 / (1.0 + dtw.normalized_cost)

        overall = (
            sw.levenshtein_weight * lev_sim
            + sw.frechet_weight * frech_sim
            + sw.dtw_weight * dtw_sim
            + sw.signature_weight * sig_sim
        )

        return ComprehensiveComparison(
            levenshtein=lev,
            frechet=frech,
            dtw=dtw,
            signature_similarity=sig_sim,
            overall_similarity=overall,
        )
