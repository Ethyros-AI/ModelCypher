from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class CompositionCategory(str, Enum):
    mental_predicate = "mentalPredicate"
    action = "action"
    evaluative = "evaluative"
    temporal = "temporal"
    spatial = "spatial"
    quantified = "quantified"
    relational = "relational"


@dataclass(frozen=True)
class CompositionProbe:
    phrase: str
    components: list[str]
    category: CompositionCategory


@dataclass(frozen=True)
class CompositionAnalysis:
    probe: CompositionProbe
    barycentric_weights: list[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: list[float]

    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3


@dataclass(frozen=True)
class ConsistencyResult:
    probe_count: int
    analyses_a: list[CompositionAnalysis]
    analyses_b: list[CompositionAnalysis]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float
    is_compatible: bool
    interpretation: str


class CompositionalProbes:
    standard_probes: list[CompositionProbe] = [
        CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.mental_predicate),
        CompositionProbe("I KNOW", ["I", "KNOW"], CompositionCategory.mental_predicate),
        CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.mental_predicate),
        CompositionProbe("I FEEL", ["I", "FEEL"], CompositionCategory.mental_predicate),
        CompositionProbe("I SEE", ["I", "SEE"], CompositionCategory.mental_predicate),
        CompositionProbe("I HEAR", ["I", "HEAR"], CompositionCategory.mental_predicate),
        CompositionProbe("SOMEONE DO", ["SOMEONE", "DO"], CompositionCategory.action),
        CompositionProbe("PEOPLE DO", ["PEOPLE", "DO"], CompositionCategory.action),
        CompositionProbe("I SAY", ["I", "SAY"], CompositionCategory.action),
        CompositionProbe("GOOD THINGS", ["GOOD", "SOMETHING"], CompositionCategory.evaluative),
        CompositionProbe("BAD THINGS", ["BAD", "SOMETHING"], CompositionCategory.evaluative),
        CompositionProbe("GOOD PEOPLE", ["GOOD", "PEOPLE"], CompositionCategory.evaluative),
        CompositionProbe("BEFORE NOW", ["BEFORE", "NOW"], CompositionCategory.temporal),
        CompositionProbe("AFTER THIS", ["AFTER", "THIS"], CompositionCategory.temporal),
        CompositionProbe("A LONG TIME BEFORE", ["A_LONG_TIME", "BEFORE"], CompositionCategory.temporal),
        CompositionProbe("ABOVE HERE", ["ABOVE", "HERE"], CompositionCategory.spatial),
        CompositionProbe("FAR FROM HERE", ["FAR", "HERE"], CompositionCategory.spatial),
        CompositionProbe("NEAR THIS", ["NEAR", "THIS"], CompositionCategory.spatial),
        CompositionProbe("MUCH GOOD", ["MUCH_MANY", "GOOD"], CompositionCategory.quantified),
        CompositionProbe("MANY PEOPLE", ["MUCH_MANY", "PEOPLE"], CompositionCategory.quantified),
        CompositionProbe(
            "I WANT GOOD THINGS",
            ["I", "WANT", "GOOD", "SOMETHING"],
            CompositionCategory.mental_predicate,
        ),
        CompositionProbe(
            "SOMEONE DO BAD THINGS",
            ["SOMEONE", "DO", "BAD", "SOMETHING"],
            CompositionCategory.action,
        ),
    ]

    @staticmethod
    def analyze_composition(
        composition_embedding: list[float],
        component_embeddings: list[list[float]],
        probe: CompositionProbe,
    ) -> CompositionAnalysis:
        n = len(component_embeddings)
        d = len(composition_embedding)
        if n <= 0 or d <= 0 or any(len(comp) != d for comp in component_embeddings):
            return CompositionAnalysis(
                probe=probe,
                barycentric_weights=[],
                residual_norm=float("inf"),
                centroid_similarity=0.0,
                component_angles=[],
            )

        centroid = [0.0 for _ in range(d)]
        for comp in component_embeddings:
            for i in range(d):
                centroid[i] += comp[i]
        for i in range(d):
            centroid[i] /= float(n)

        centroid_sim = CompositionalProbes._cosine_similarity(composition_embedding, centroid)
        component_angles = [
            CompositionalProbes._cosine_similarity(composition_embedding, comp)
            for comp in component_embeddings
        ]

        weights, residual = CompositionalProbes._compute_barycentric_weights(
            target=composition_embedding,
            basis=component_embeddings,
        )

        return CompositionAnalysis(
            probe=probe,
            barycentric_weights=weights,
            residual_norm=residual,
            centroid_similarity=centroid_sim,
            component_angles=component_angles,
        )

    @staticmethod
    def check_consistency(
        analyses_a: list[CompositionAnalysis],
        analyses_b: list[CompositionAnalysis],
    ) -> ConsistencyResult:
        if len(analyses_a) != len(analyses_b) or not analyses_a:
            return ConsistencyResult(
                probe_count=0,
                analyses_a=[],
                analyses_b=[],
                barycentric_correlation=0.0,
                angular_correlation=0.0,
                consistency_score=0.0,
                is_compatible=False,
                interpretation="Insufficient data for consistency check",
            )

        all_weights_a: list[float] = []
        all_weights_b: list[float] = []
        for a, b in zip(analyses_a, analyses_b):
            if len(a.barycentric_weights) == len(b.barycentric_weights):
                all_weights_a.extend(a.barycentric_weights)
                all_weights_b.extend(b.barycentric_weights)

        all_angles_a: list[float] = []
        all_angles_b: list[float] = []
        for a, b in zip(analyses_a, analyses_b):
            if len(a.component_angles) == len(b.component_angles):
                all_angles_a.extend(a.component_angles)
                all_angles_b.extend(b.component_angles)

        bary_corr = CompositionalProbes._pearson_correlation(all_weights_a, all_weights_b)
        angular_corr = CompositionalProbes._pearson_correlation(all_angles_a, all_angles_b)

        score = 0.4 * max(0.0, bary_corr) + 0.6 * max(0.0, angular_corr)
        is_compatible = score >= 0.5 and angular_corr >= 0.4

        if score >= 0.8:
            interpretation = (
                "Excellent compositional consistency. Models combine concepts identically. "
                "Universal composition rules apply."
            )
        elif score >= 0.6:
            interpretation = (
                "Good compositional consistency. Models share similar composition patterns with minor variations."
            )
        elif score >= 0.4:
            interpretation = "Partial compositional consistency. Models differ in how they combine some concepts."
        else:
            interpretation = "Low compositional consistency. Models use fundamentally different composition rules."

        return ConsistencyResult(
            probe_count=len(analyses_a),
            analyses_a=analyses_a,
            analyses_b=analyses_b,
            barycentric_correlation=bary_corr,
            angular_correlation=angular_corr,
            consistency_score=score,
            is_compatible=is_compatible,
            interpretation=interpretation,
        )

    @staticmethod
    def analyze_all_probes(
        prime_embeddings: dict[str, list[float]],
        composition_embeddings: dict[str, list[float]],
        probes: list[CompositionProbe] | None = None,
    ) -> list[CompositionAnalysis]:
        analyses: list[CompositionAnalysis] = []
        probe_list = probes if probes is not None else CompositionalProbes.standard_probes
        for probe in probe_list:
            comp_embed = composition_embeddings.get(probe.phrase)
            if comp_embed is None:
                continue
            component_embed: list[list[float]] = []
            all_found = True
            for comp in probe.components:
                embed = prime_embeddings.get(comp)
                if embed is None:
                    all_found = False
                    break
                component_embed.append(embed)
            if not all_found or not component_embed:
                continue
            analyses.append(
                CompositionalProbes.analyze_composition(
                    composition_embedding=comp_embed,
                    component_embeddings=component_embed,
                    probe=probe,
                )
            )
        return analyses

    @staticmethod
    def _compute_barycentric_weights(
        target: list[float],
        basis: list[list[float]],
    ) -> tuple[list[float], float]:
        n = len(basis)
        d = len(target)
        if n <= 0 or d <= 0:
            return [], float("inf")

        gram = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                dot = 0.0
                for k in range(d):
                    dot += basis[i][k] * basis[j][k]
                gram[i][j] = dot
                gram[j][i] = dot

        rhs = [0.0 for _ in range(n)]
        for i in range(n):
            dot = 0.0
            for k in range(d):
                dot += basis[i][k] * target[k]
            rhs[i] = dot

        weights = [1.0 / float(n) for _ in range(n)]
        for _ in range(50):
            for i in range(n):
                total = rhs[i]
                for j in range(n):
                    if j == i:
                        continue
                    total -= gram[i][j] * weights[j]
                if abs(gram[i][i]) > 1e-10:
                    weights[i] = total / gram[i][i]

        reconstructed = [0.0 for _ in range(d)]
        for i in range(n):
            for k in range(d):
                reconstructed[k] += weights[i] * basis[i][k]

        residual_sq = 0.0
        for k in range(d):
            diff = target[k] - reconstructed[k]
            residual_sq += diff * diff

        return weights, math.sqrt(residual_sq)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        return dot / denom if denom > 1e-10 else 0.0

    @staticmethod
    def _pearson_correlation(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or len(a) < 2:
            return 0.0
        n = float(len(a))
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        sum_ab = 0.0
        sum_a2 = 0.0
        sum_b2 = 0.0
        for i in range(len(a)):
            da = a[i] - mean_a
            db = b[i] - mean_b
            sum_ab += da * db
            sum_a2 += da * da
            sum_b2 += db * db
        denom = math.sqrt(sum_a2 * sum_b2)
        if denom > 1e-10:
            return sum_ab / denom
        matches = all(abs(x - y) < 1e-6 for x, y in zip(a, b))
        return 1.0 if matches else 0.0
