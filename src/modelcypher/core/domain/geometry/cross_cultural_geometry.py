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
from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
from modelcypher.core.domain.geometry.path_geometry import (
    PathComparison,
    PathGeometry,
    PathSignature,
)


@dataclass(frozen=True)
class ComplementaryPrime:
    prime_id: str
    sharper_model: "SharperModel"
    sharpness_ratio: float


class SharperModel(str, Enum):
    model_a = "modelA"
    model_b = "modelB"


class MergeAssessment(str, Enum):
    promising = "promising"
    uncertain = "uncertain"
    incompatible = "incompatible"


@dataclass(frozen=True)
class ComparisonResult:
    gram_roughness_a: float
    gram_roughness_b: float
    merged_gram_roughness: float
    roughness_reduction: float
    complementarity_score: float
    convergent_primes: list[str]
    divergent_primes: list[str]
    complementary_primes: list[ComplementaryPrime]
    category_divergence: dict[str, float]
    merge_quality_score: float
    merge_assessment: MergeAssessment
    rationale: str
    trajectory_analysis: PathComparison | None = None


@dataclass(frozen=True)
class AlignmentAnalysis:
    cka: float
    raw_pearson: float
    alignment_gap: float
    alignment_assessment: "AlignmentAssessment"
    interpretation: str


class AlignmentAssessment(str, Enum):
    aligned = "aligned"
    centered_aligned = "centered_aligned"
    partial_overlap = "partial_overlap"
    incompatible = "incompatible"


class CrossCulturalGeometry:
    @staticmethod
    def analyze(
        gram_a: list[float],
        gram_b: list[float],
        prime_ids: list[str],
        prime_categories: dict[str, str],
    ) -> ComparisonResult | None:
        n = len(prime_ids)
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return None

        roughness_a = CrossCulturalGeometry._compute_roughness(gram_a, n)
        roughness_b = CrossCulturalGeometry._compute_roughness(gram_b, n)
        merged_gram = CrossCulturalGeometry._average_grams(gram_a, gram_b)
        merged_roughness = CrossCulturalGeometry._compute_roughness(merged_gram, n)

        avg_roughness = (roughness_a + roughness_b) / 2.0
        roughness_reduction = (
            (avg_roughness - merged_roughness) / avg_roughness if avg_roughness > 0 else 0.0
        )

        sharpness_a = CrossCulturalGeometry._compute_row_sharpness(gram_a, n)
        sharpness_b = CrossCulturalGeometry._compute_row_sharpness(gram_b, n)
        row_correlations = CrossCulturalGeometry._compute_row_correlations(gram_a, gram_b, n)

        convergent: list[str] = []
        divergent: list[str] = []
        complementary: list[ComplementaryPrime] = []

        convergence_threshold = 0.7
        divergence_threshold = 0.4
        sharpness_ratio_threshold = 1.5

        for idx, prime_id in enumerate(prime_ids):
            correlation = row_correlations[idx]
            if correlation > convergence_threshold:
                convergent.append(prime_id)
            elif correlation < divergence_threshold:
                divergent.append(prime_id)

            s_a = sharpness_a[idx]
            s_b = sharpness_b[idx]
            if s_a > 0 and s_b > 0:
                ratio = max(s_a, s_b) / min(s_a, s_b)
                if ratio > sharpness_ratio_threshold:
                    sharper = SharperModel.model_a if s_a > s_b else SharperModel.model_b
                    complementary.append(
                        ComplementaryPrime(
                            prime_id=prime_id, sharper_model=sharper, sharpness_ratio=ratio
                        )
                    )

        category_divergence = CrossCulturalGeometry._compute_category_divergence(
            row_correlations,
            prime_ids,
            prime_categories,
        )
        complementarity_score = CrossCulturalGeometry._compute_complementarity_score(
            sharpness_a,
            sharpness_b,
            row_correlations,
        )

        merge_quality, assessment, rationale = CrossCulturalGeometry._assess_merge_quality(
            roughness_reduction,
            complementarity_score,
            len(convergent),
            len(divergent),
            n,
            category_divergence,
        )

        return ComparisonResult(
            gram_roughness_a=roughness_a,
            gram_roughness_b=roughness_b,
            merged_gram_roughness=merged_roughness,
            roughness_reduction=roughness_reduction,
            complementarity_score=complementarity_score,
            convergent_primes=convergent,
            divergent_primes=divergent,
            complementary_primes=complementary,
            category_divergence=category_divergence,
            merge_quality_score=merge_quality,
            merge_assessment=assessment,
            rationale=rationale,
            trajectory_analysis=None,
        )

    @staticmethod
    def analyze_trajectories(
        path_a: PathSignature,
        path_b: PathSignature,
        gate_embeddings: dict[str, list[float]],
    ) -> PathGeometry.PathComparison:
        return PathGeometry.compare(path_a, path_b, gate_embeddings)

    @staticmethod
    def compute_cka(gram_a: list[float], gram_b: list[float], n: int) -> float:
        """Compute CKA between two flattened gram matrices.

        Delegates to the canonical implementation in cka.py.
        """
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return 0.0
        return compute_cka_from_grams(gram_a, gram_b, n)

    @staticmethod
    def analyze_alignment(
        gram_a: list[float],
        gram_b: list[float],
        n: int,
        raw_pearson: float | None = None,
    ) -> AlignmentAnalysis | None:
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return None

        cka = CrossCulturalGeometry.compute_cka(gram_a, gram_b, n)

        if raw_pearson is None:
            off_diag_a: list[float] = []
            off_diag_b: list[float] = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    off_diag_a.append(float(gram_a[i * n + j]))
                    off_diag_b.append(float(gram_b[i * n + j]))
            pearson = _pearson_correlation(off_diag_a, off_diag_b)
        else:
            pearson = raw_pearson

        gap = cka - pearson

        if cka >= 0.7:
            if pearson >= 0.6:
                assessment = AlignmentAssessment.aligned
                interpretation = (
                    f"High CKA ({cka:.2f}) and high raw Gram correlation ({pearson:.2f}) "
                    "indicate strong agreement in anchor relational geometry under both centered and "
                    "uncentered views."
                )
            else:
                assessment = AlignmentAssessment.centered_aligned
                interpretation = (
                    f"High CKA ({cka:.2f}) with low raw Gram correlation ({pearson:.2f}) indicates "
                    "agreement after centering but disagreement in raw (mean/bias) structure. "
                    "This is not evidence of a recoverable coordinate rotation; treat it as a sign "
                    "that centered relational structure matches while absolute similarity patterns differ."
                )
        elif cka >= 0.4:
            assessment = AlignmentAssessment.partial_overlap
            interpretation = (
                f"Moderate CKA ({cka:.2f}) suggests partial overlap in relational structure: "
                "some anchor relations agree, but a substantial fraction differs."
            )
        else:
            assessment = AlignmentAssessment.incompatible
            interpretation = (
                f"Low CKA ({cka:.2f}) indicates weak agreement in anchor relational structure. "
                "Simple merges are unlikely to work without task-/parameter-level alignment methods."
            )

        return AlignmentAnalysis(
            cka=cka,
            raw_pearson=pearson,
            alignment_gap=gap,
            alignment_assessment=assessment,
            interpretation=interpretation,
        )

    @staticmethod
    def _compute_roughness(gram: list[float], n: int) -> float:
        if n <= 1:
            return 0.0
        row_variances = CrossCulturalGeometry._compute_row_sharpness(gram, n)
        return sum(row_variances) / len(row_variances) if row_variances else 0.0

    @staticmethod
    def _compute_row_sharpness(gram: list[float], n: int) -> list[float]:
        sharpness: list[float] = []
        for i in range(n):
            values = [float(gram[i * n + j]) for j in range(n) if i != j]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((val - mean) ** 2 for val in values) / len(values) if values else 0.0
            sharpness.append(variance)
        return sharpness

    @staticmethod
    def _compute_row_correlations(gram_a: list[float], gram_b: list[float], n: int) -> list[float]:
        correlations: list[float] = []
        for i in range(n):
            vec_a = [float(gram_a[i * n + j]) for j in range(n) if i != j]
            vec_b = [float(gram_b[i * n + j]) for j in range(n) if i != j]
            correlations.append(_pearson_correlation(vec_a, vec_b))
        return correlations

    @staticmethod
    def _average_grams(gram_a: list[float], gram_b: list[float]) -> list[float]:
        return [(a + b) / 2.0 for a, b in zip(gram_a, gram_b)]

    @staticmethod
    def _compute_category_divergence(
        row_correlations: list[float],
        prime_ids: list[str],
        prime_categories: dict[str, str],
    ) -> dict[str, float]:
        category_correlations: dict[str, list[float]] = {}
        for idx, prime_id in enumerate(prime_ids):
            category = prime_categories.get(prime_id)
            if category is None:
                continue
            category_correlations.setdefault(category, []).append(row_correlations[idx])

        divergence: dict[str, float] = {}
        for category, correlations in category_correlations.items():
            mean_corr = sum(correlations) / len(correlations)
            divergence[category] = 1.0 - max(0.0, mean_corr)
        return divergence

    @staticmethod
    def _compute_complementarity_score(
        sharpness_a: list[float],
        sharpness_b: list[float],
        row_correlations: list[float],
    ) -> float:
        if len(sharpness_a) != len(sharpness_b) or not sharpness_a:
            return 0.0
        complementary_count = 0
        ratio_threshold = 1.3
        for i in range(len(sharpness_a)):
            s_a = sharpness_a[i]
            s_b = sharpness_b[i]
            if s_a <= 0 or s_b <= 0:
                continue
            ratio = max(s_a, s_b) / min(s_a, s_b)
            if ratio > ratio_threshold:
                complementary_count += 1
        complementary_ratio = complementary_count / len(sharpness_a)
        mean_correlation = (
            sum(row_correlations) / len(row_correlations) if row_correlations else 0.0
        )
        alignment_weight = max(0.0, min(1.0, mean_correlation))
        return complementary_ratio * (0.5 + 0.5 * alignment_weight)

    @staticmethod
    def _assess_merge_quality(
        roughness_reduction: float,
        complementarity_score: float,
        convergent_count: int,
        divergent_count: int,
        total_primes: int,
        category_divergence: dict[str, float],
    ) -> tuple[float, MergeAssessment, str]:
        score = 0.0
        score += max(0.0, roughness_reduction) * 0.3
        score += complementarity_score * 0.3
        convergent_ratio = convergent_count / total_primes if total_primes else 0.0
        score += convergent_ratio * 0.25
        divergent_ratio = divergent_count / total_primes if total_primes else 0.0
        score -= divergent_ratio * 0.15

        avg_divergence = (
            sum(category_divergence.values()) / max(1, len(category_divergence))
            if category_divergence
            else 0.0
        )
        score += (1.0 - avg_divergence) * 0.1
        score = max(0.0, min(1.0, score))

        if score > 0.6 and roughness_reduction > 0.15:
            assessment = MergeAssessment.promising
        elif score > 0.4:
            assessment = MergeAssessment.uncertain
        else:
            assessment = MergeAssessment.incompatible

        rationale = ""
        if convergent_ratio > 0.5:
            rationale += f"High prime correlation ({int(convergent_ratio * 100)}% convergent). "
        if complementarity_score > 0.5:
            rationale += "Strong complementary sharpness patterns. "

        if category_divergence:
            worst_category = max(category_divergence.items(), key=lambda item: item[1])
            best_category = min(category_divergence.items(), key=lambda item: item[1])
            if worst_category[1] > 0.3:
                rationale += (
                    f"{worst_category[0]} shows largest divergence ({worst_category[1]:.2f}). "
                )
            if best_category[1] < 0.2:
                rationale += f"{best_category[0]} strongly aligned. "

        if roughness_reduction > 0.2:
            rationale += f"Merge would reduce roughness by {int(roughness_reduction * 100)}%. "
        elif roughness_reduction < 0:
            rationale += "Warning: merge increases roughness. "

        if not rationale:
            rationale = "Moderate alignment with mixed signals."

        return score, assessment, rationale.strip()


def _pearson_correlation(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or len(lhs) != len(rhs):
        return 0.0
    mean_l = sum(lhs) / len(lhs)
    mean_r = sum(rhs) / len(rhs)
    num = 0.0
    den_l = 0.0
    den_r = 0.0
    for i in range(len(lhs)):
        diff_l = lhs[i] - mean_l
        diff_r = rhs[i] - mean_r
        num += diff_l * diff_r
        den_l += diff_l * diff_l
        den_r += diff_r * diff_r
    denom = math.sqrt(den_l) * math.sqrt(den_r)
    if denom <= 0:
        return 0.0
    return num / denom
