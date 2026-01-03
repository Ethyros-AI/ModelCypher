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

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    is_finite,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import VectorMath

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


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

    Contains objective geometric quantities only.
    """

    levenshtein: PathComparison
    frechet: FrechetResult
    dtw: DTWResult
    signature_similarity: float


class PathGeometry:
    @staticmethod
    def _division_epsilon() -> float:
        backend = get_default_backend()
        arr = backend.array([0.0])
        return division_epsilon(backend, arr)

    @staticmethod
    def _projection_dim(gate_embeddings: dict[str, list[float]]) -> int:
        if not gate_embeddings:
            return 1
        return max(len(vec) for vec in gate_embeddings.values()) + 1

    @staticmethod
    def _entropy_normalization(entropies: list[float]) -> tuple[float, float]:
        if not entropies:
            return 0.0, PathGeometry._division_epsilon()
        backend = get_default_backend()
        entropies_arr = backend.array(entropies)
        mean_arr = backend.mean(entropies_arr)
        diff = entropies_arr - mean_arr
        variance_arr = backend.mean(diff * diff)
        std_arr = backend.sqrt(variance_arr)
        backend.eval(mean_arr, std_arr)
        mean = float(backend.to_scalar(mean_arr))
        std = float(backend.to_scalar(std_arr))
        eps = PathGeometry._division_epsilon()
        return mean, std if std > eps else eps

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
                try:
                    return 1.0 - VectorMath.cosine_similarity(vec_a, vec_b)
                except ValueError:
                    # Zero vector: undefined similarity, maximum distance
                    return 1.0
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

        entropy_values = [node.entropy for node in path_a.nodes + path_b.nodes]
        _, entropy_scale = PathGeometry._entropy_normalization(entropy_values)

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
            entropy_diff = abs(node_a.entropy - node_b.entropy) / entropy_scale
            return d_val + entropy_diff

        dp = [[float("inf")] * m for _ in range(n)]
        dp[0][0] = dist(0, 0)

        for j in range(1, m):
            dp[0][j] = dp[0][j - 1] + dist(0, j)

        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + dist(i, 0)

        for i in range(1, n):
            for j in range(1, m):
                d_val = dist(i, j)
                dp[i][j] = d_val + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        backend = get_default_backend()
        final_cost = dp[n - 1][m - 1]
        if not is_finite(final_cost, backend):
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
    ) -> TruncatedSignature:
        projection_dim = PathGeometry._projection_dim(gate_embeddings)
        if len(path.nodes) < 2:
            return TruncatedSignature(
                level1=[0.0] * projection_dim,
                level2=[[0.0] * projection_dim for _ in range(projection_dim)],
                signed_area=0.0,
                signature_norm=0.0,
            )

        entropies = [node.entropy for node in path.nodes]
        entropy_mean, entropy_scale = PathGeometry._entropy_normalization(entropies)

        coords: list[list[float]] = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
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

        backend = get_default_backend()
        antisymmetric_sum = 0.0
        for p in range(projection_dim):
            for q in range(p + 1, projection_dim):
                antisym = level2[p][q] - level2[q][p]
                antisymmetric_sum += antisym * antisym
        signed_area = 0.5 * sqrt_scalar(antisymmetric_sum, backend)

        norm = 0.0
        for d in range(projection_dim):
            norm += level1[d] * level1[d]
        for p in range(projection_dim):
            for q in range(projection_dim):
                norm += level2[p][q] * level2[p][q]
        signature_norm = sqrt_scalar(norm, backend)

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
    ) -> float:
        """Compute similarity between two path signatures.

        Args:
            sig_a: First signature.
            sig_b: Second signature.
        """
        backend = get_default_backend()
        l1_dist = 0.0
        count = min(len(sig_a.level1), len(sig_b.level1))
        for i in range(count):
            diff = sig_a.level1[i] - sig_b.level1[i]
            l1_dist += diff * diff
        l1_dist = sqrt_scalar(l1_dist, backend)

        level2_dist = 0.0
        rows = min(len(sig_a.level2), len(sig_b.level2))
        for p in range(rows):
            row_a = sig_a.level2[p]
            row_b = sig_b.level2[p]
            cols = min(len(row_a), len(row_b))
            for q in range(cols):
                diff = row_a[q] - row_b[q]
                level2_dist += diff * diff
        level2_dist = sqrt_scalar(level2_dist, backend)

        total_dist = sqrt_scalar(l1_dist * l1_dist + level2_dist * level2_dist, backend)
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
        gradients = [entropies[i] - entropies[i - 1] for i in range(1, len(entropies))]
        mean_gradient = sum(gradients) / len(gradients) if gradients else 0.0

        return EntropyPathAnalysis(
            total_entropy=total,
            mean_entropy=mean,
            entropy_variance=variance,
            max_entropy=max_val,
            max_entropy_index=max_idx,
            mean_gradient=mean_gradient,
        )

    @staticmethod
    def compute_local_geometry(
        path: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> LocalGeometry:
        projection_dim = PathGeometry._projection_dim(gate_embeddings)
        if len(path.nodes) < 3:
            return LocalGeometry(
                curvatures=[],
                mean_curvature=0.0,
                max_curvature=0.0,
                total_curvature=0.0,
                torsions=[],
                mean_torsion=0.0,
            )

        entropies = [node.entropy for node in path.nodes]
        entropy_mean, entropy_scale = PathGeometry._entropy_normalization(entropies)
        coords = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords.append(proj)

        backend = get_default_backend()
        tangents: list[list[float]] = []
        for i in range(len(coords) - 1):
            t = [0.0] * projection_dim
            norm = 0.0
            for d in range(projection_dim):
                t[d] = coords[i + 1][d] - coords[i][d]
                norm += t[d] * t[d]
            norm = sqrt_scalar(norm, backend)
            if norm > 0:
                t = [val / norm for val in t]
            tangents.append(t)

        curvatures: list[float] = []
        for i in range(len(tangents) - 1):
            dot = sum(tangents[i][d] * tangents[i + 1][d] for d in range(projection_dim))
            dot = max(-1.0, min(1.0, dot))
            curvatures.append(acos_scalar(dot, backend))

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
                torsions.append(sqrt_scalar(deviation, backend))

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
    ) -> ComprehensiveComparison:
        """Comprehensive trajectory comparison using multiple distance metrics.

        Computes four distance metrics and reports them directly with
        signature similarity. Returns only objective geometric quantities.

        Args:
            path_a: First path signature.
            path_b: Second path signature.
            gate_embeddings: Embedding vectors for gate IDs.
        Returns:
            ComprehensiveComparison with individual metrics.
        """
        lev = PathGeometry.compare(path_a, path_b, gate_embeddings)
        frech = PathGeometry.frechet_distance(path_a, path_b, gate_embeddings)
        dtw = PathGeometry.dynamic_time_warping(path_a, path_b, gate_embeddings)

        sig_a = PathGeometry.compute_signature(path_a, gate_embeddings)
        sig_b = PathGeometry.compute_signature(path_b, gate_embeddings)
        sig_sim = PathGeometry.signature_similarity(sig_a, sig_b)

        return ComprehensiveComparison(
            levenshtein=lev,
            frechet=frech,
            dtw=dtw,
            signature_similarity=sig_sim,
        )


class BackendPathGeometry:
    """GPU-accelerated path geometry using the Backend protocol.

    This class provides the same functionality as PathGeometry but uses
    Backend tensor operations for GPU acceleration. Key optimizations:

    - Signature computation: Vectorized increments and outer products
    - Local geometry: Vectorized tangent/curvature computation
    - Entropy analysis: Backend statistics (mean, var)
    - Distance matrices: Precomputed for DP algorithms
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a Backend instance.

        Args:
            backend: Backend instance (MLXBackend, JAXBackend, etc.)
        """
        self.backend = backend
        self._finfo = backend.finfo()

    def compute_signature(
        self,
        path: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> TruncatedSignature:
        """Compute path signature using GPU-accelerated operations.

        Args:
            path: Path signature with nodes.
            gate_embeddings: Embedding vectors for gate IDs.

        Returns:
            TruncatedSignature with level1, level2, signed_area, signature_norm.
        """
        projection_dim = PathGeometry._projection_dim(gate_embeddings)
        if len(path.nodes) < 2:
            return TruncatedSignature(
                level1=[0.0] * projection_dim,
                level2=[[0.0] * projection_dim for _ in range(projection_dim)],
                signed_area=0.0,
                signature_norm=0.0,
            )

        # Build coordinate matrix [n_nodes, projection_dim]
        entropies = [node.entropy for node in path.nodes]
        entropy_mean, entropy_scale = PathGeometry._entropy_normalization(entropies)

        coords_list: list[list[float]] = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords_list.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords_list.append(proj)

        coords = self.backend.array(coords_list)  # [n, d]

        # Level 1: Sum of increments = final - initial
        # Vectorized: diff along axis 0, then sum
        n_nodes = len(coords_list)
        increments = coords[1:] - coords[:-1]  # [n-1, d]
        level1_arr = self.backend.sum(increments, axis=0)  # [d]
        self.backend.eval(level1_arr)
        level1 = self._to_list(level1_arr)

        # Level 2: Iterated integrals using Chen's identity
        # level2 = sum_i (cumulative_before_i ⊗ dx_i) + 0.5 * sum_i (dx_i ⊗ dx_i)
        cumulative = self.backend.cumsum(increments, axis=0)
        cumulative_before = cumulative - increments
        outer_sum = self.backend.matmul(self.backend.transpose(cumulative_before), increments)
        self_outer = self.backend.matmul(self.backend.transpose(increments), increments) * 0.5
        level2_arr = outer_sum + self_outer
        self.backend.eval(level2_arr)
        level2 = self._to_nested_list(level2_arr)

        # Signed area: sqrt(0.5 * sum of squared antisymmetric parts)
        antisym = level2_arr - self.backend.transpose(level2_arr)
        antisym_sum = self.backend.sum(antisym * antisym)
        antisym_scaled = antisym_sum * 0.5
        self.backend.eval(antisym_scaled)
        signed_area = 0.5 * sqrt_scalar(self._to_scalar(antisym_scaled), self.backend)

        # Signature norm: sqrt(sum of all squared components)
        level1_norm_sq = self.backend.sum(level1_arr * level1_arr)
        level2_norm_sq = self.backend.sum(level2_arr * level2_arr)
        total_norm_sq = level1_norm_sq + level2_norm_sq
        self.backend.eval(total_norm_sq)
        signature_norm = sqrt_scalar(self._to_scalar(total_norm_sq), self.backend)

        return TruncatedSignature(
            level1=level1,
            level2=level2,
            signed_area=signed_area,
            signature_norm=signature_norm,
        )

    def signature_similarity(
        self,
        sig_a: TruncatedSignature,
        sig_b: TruncatedSignature,
    ) -> float:
        """Compute similarity between two path signatures using GPU.

        Args:
            sig_a: First signature.
            sig_b: Second signature.

        Returns:
            Similarity score in [0, 1].
        """
        # Level-1 distance
        count = min(len(sig_a.level1), len(sig_b.level1))
        a_arr = self.backend.array(sig_a.level1[:count])
        b_arr = self.backend.array(sig_b.level1[:count])
        diff = a_arr - b_arr
        level1_dist_sq = self.backend.sum(diff * diff)
        self.backend.eval(level1_dist_sq)

        # Level-2 distance
        rows = min(len(sig_a.level2), len(sig_b.level2))
        cols = (
            min(len(sig_a.level2[0]), len(sig_b.level2[0]))
            if rows > 0 and sig_a.level2[0] and sig_b.level2[0]
            else 0
        )
        if rows > 0 and cols > 0:
            a2 = self.backend.array([row[:cols] for row in sig_a.level2[:rows]])
            b2 = self.backend.array([row[:cols] for row in sig_b.level2[:rows]])
            diff2 = a2 - b2
            level2_dist_sq = self.backend.sum(diff2 * diff2)
            self.backend.eval(level2_dist_sq)
        else:
            level2_dist_sq = self.backend.array(0.0)

        total_dist_sq = level1_dist_sq + level2_dist_sq
        self.backend.eval(total_dist_sq)
        total_dist = sqrt_scalar(self._to_scalar(total_dist_sq), self.backend)
        return 1.0 / (1.0 + total_dist)

    def analyze_entropy_path(self, path: PathSignature) -> EntropyPathAnalysis:
        """Analyze entropy along a path using GPU-accelerated statistics.

        Args:
            path: Path signature with nodes.

        Returns:
            EntropyPathAnalysis with statistics.
        """
        if not path.nodes:
            return EntropyPathAnalysis(
                total_entropy=0.0,
                mean_entropy=0.0,
                entropy_variance=0.0,
                max_entropy=0.0,
                max_entropy_index=0,
                mean_gradient=0.0,
            )

        entropies_list = [node.entropy for node in path.nodes]
        entropies = self.backend.array(entropies_list)
        n = len(entropies_list)

        # Basic statistics
        total = self.backend.sum(entropies)
        mean = total / float(n)
        self.backend.eval(total, mean)
        total_val = self._to_scalar(total)
        mean_val = self._to_scalar(mean)

        # Variance
        diff_from_mean = entropies - mean
        variance = self.backend.sum(diff_from_mean * diff_from_mean) / float(n)
        self.backend.eval(variance)
        variance_val = self._to_scalar(variance)
        # Max entropy and index
        max_idx_arr = self.backend.argmax(entropies)
        max_val_arr = self.backend.max(entropies)
        self.backend.eval(max_idx_arr, max_val_arr)
        max_idx = int(self._to_scalar(max_idx_arr))
        max_val = float(self._to_scalar(max_val_arr))

        # Gradients
        if n > 1:
            gradients = entropies[1:] - entropies[:-1]
            mean_gradient = self.backend.sum(gradients) / float(n - 1)
            self.backend.eval(mean_gradient)
            mean_gradient_val = self._to_scalar(mean_gradient)
        else:
            mean_gradient_val = 0.0

        return EntropyPathAnalysis(
            total_entropy=total_val,
            mean_entropy=mean_val,
            entropy_variance=variance_val,
            max_entropy=max_val,
            max_entropy_index=max_idx,
            mean_gradient=mean_gradient_val,
        )

    def compute_local_geometry(
        self,
        path: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> LocalGeometry:
        """Compute local geometry (curvature, torsion) using GPU acceleration.

        Args:
            path: Path signature with nodes.
            gate_embeddings: Embedding vectors for gate IDs.
        Returns:
            LocalGeometry with curvatures and torsions.
        """
        projection_dim = PathGeometry._projection_dim(gate_embeddings)
        if len(path.nodes) < 3:
            return LocalGeometry(
                curvatures=[],
                mean_curvature=0.0,
                max_curvature=0.0,
                total_curvature=0.0,
                torsions=[],
                mean_torsion=0.0,
            )

        # Build coordinate matrix
        entropies = [node.entropy for node in path.nodes]
        entropy_mean, entropy_scale = PathGeometry._entropy_normalization(entropies)
        coords_list: list[list[float]] = []
        for node in path.nodes:
            emb = gate_embeddings.get(node.gate_id)
            if emb:
                proj = list(emb[: projection_dim - 1])
                while len(proj) < projection_dim - 1:
                    proj.append(0.0)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords_list.append(proj)
            else:
                proj = [0.0] * (projection_dim - 1)
                proj.append((node.entropy - entropy_mean) / entropy_scale)
                coords_list.append(proj)

        coords = self.backend.array(coords_list)  # [n, d]

        # Compute tangent vectors: diff and normalize
        tangent_diff = coords[1:] - coords[:-1]  # [n-1, d]
        tangent_norms = self.backend.norm(tangent_diff, axis=1, keepdims=True)  # [n-1, 1]

        # Avoid division by zero
        safe_norms = self.backend.maximum(tangent_norms, self._finfo.eps)
        tangents = tangent_diff / safe_norms  # [n-1, d]
        self.backend.eval(tangents)

        # Curvatures: angle between consecutive tangents
        # curvature[i] = arccos(tangents[i] · tangents[i+1])
        n_tangents = len(coords_list) - 1
        if n_tangents > 1:
            dot_arr = self.backend.sum(tangents[:-1] * tangents[1:], axis=1)
            dot_clamped = self.backend.clip(dot_arr, -1.0, 1.0)
            curvatures_arr = self.backend.arccos(dot_clamped)
            mean_curv_arr = self.backend.mean(curvatures_arr)
            max_curv_arr = self.backend.max(curvatures_arr)
            total_curv_arr = self.backend.sum(curvatures_arr)
            self.backend.eval(curvatures_arr, mean_curv_arr, max_curv_arr, total_curv_arr)
            curvatures_list = [float(x) for x in self._to_list(curvatures_arr)]
            mean_curv = float(self._to_scalar(mean_curv_arr))
            max_curv = float(self._to_scalar(max_curv_arr))
            total_curv = float(self._to_scalar(total_curv_arr))
        else:
            curvatures_list = []
            mean_curv = 0.0
            max_curv = 0.0
            total_curv = 0.0

        # Torsions: deviation from linear interpolation of tangents
        if n_tangents >= 3:
            expected = (tangents[:-2] + tangents[2:]) / 2.0
            deviation = tangents[1:-1] - expected
            dev_norm_sq = self.backend.sum(deviation * deviation, axis=1)
            torsions_arr = self.backend.sqrt(dev_norm_sq)
            mean_tors_arr = self.backend.mean(torsions_arr)
            self.backend.eval(torsions_arr, mean_tors_arr)
            torsions_list = [float(x) for x in self._to_list(torsions_arr)]
            mean_tors = float(self._to_scalar(mean_tors_arr))
        else:
            torsions_list = []
            mean_tors = 0.0

        return LocalGeometry(
            curvatures=curvatures_list,
            mean_curvature=mean_curv,
            max_curvature=max_curv,
            total_curvature=total_curv,
            torsions=torsions_list,
            mean_torsion=mean_tors,
        )

    def comprehensive_compare(
        self,
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> ComprehensiveComparison:
        """Comprehensive trajectory comparison with GPU-accelerated signatures.

        Uses pure Python for DP algorithms (inherently sequential) but
        GPU-accelerated signature computation.

        Args:
            path_a: First path signature.
            path_b: Second path signature.
            gate_embeddings: Embedding vectors for gate IDs.
        Returns:
            ComprehensiveComparison with individual metrics.
        """
        # Use pure Python for DP algorithms (sequential dependencies)
        lev = PathGeometry.compare(path_a, path_b, gate_embeddings)
        frech = PathGeometry.frechet_distance(path_a, path_b, gate_embeddings)
        dtw = PathGeometry.dynamic_time_warping(path_a, path_b, gate_embeddings)

        # Use GPU for signature computation
        sig_a = self.compute_signature(path_a, gate_embeddings)
        sig_b = self.compute_signature(path_b, gate_embeddings)
        sig_sim = self.signature_similarity(sig_a, sig_b)

        return ComprehensiveComparison(
            levenshtein=lev,
            frechet=frech,
            dtw=dtw,
            signature_similarity=sig_sim,
        )

    def _outer_product(self, a: Any, b: Any) -> Any:
        """Compute outer product a[:, None] * b[None, :].

        Args:
            a: 1D array of shape [d]
            b: 1D array of shape [d]

        Returns:
            2D array of shape [d, d]
        """
        # Reshape for broadcasting
        a_col = self.backend.reshape(a, (-1, 1))  # [d, 1]
        b_row = self.backend.reshape(b, (1, -1))  # [1, d]
        return a_col * b_row  # [d, d]

    def _to_scalar(self, val: Any) -> float:
        """Convert backend scalar to Python float."""
        if hasattr(val, "shape") or hasattr(val, "item") or hasattr(val, "tolist"):
            self.backend.eval(val)
            return float(self.backend.to_scalar(val))
        try:
            return float(val)
        except TypeError:
            if isinstance(val, (list, tuple)) and val:
                return float(val[0])
            return 0.0

    def _to_list(self, arr: Any) -> list[float]:
        """Convert backend array to Python list."""
        if hasattr(arr, "shape") or hasattr(arr, "tolist"):
            return self.backend.tolist(arr)
        return list(arr)

    def _to_nested_list(self, arr: Any) -> list[list[float]]:
        """Convert 2D backend array to nested Python list."""
        if hasattr(arr, "shape") or hasattr(arr, "tolist"):
            return self.backend.tolist(arr)
        return [list(row) for row in arr]


def get_path_geometry(
    backend: "Backend | None" = None,
) -> type[PathGeometry] | BackendPathGeometry:
    """Get the best available path geometry implementation.

    Args:
        backend: Optional Backend instance. If provided, returns
                 BackendPathGeometry for GPU acceleration.

    Returns:
        PathGeometry class or BackendPathGeometry instance.

    Example:
        >>> from modelcypher.core.domain._backend import get_default_backend
        >>> backend = get_default_backend()
        >>> pg = get_path_geometry(backend)
        >>> sig = pg.compute_signature(path, embeddings)
    """
    if backend is not None:
        return BackendPathGeometry(backend)
    return PathGeometry
