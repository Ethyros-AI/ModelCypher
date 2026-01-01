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

from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka_from_grams
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_pearson_correlation,
    division_epsilon,
)
from modelcypher.core.domain.geometry.path_geometry import (
    PathComparison,
    PathGeometry,
    PathSignature,
)


@dataclass(frozen=True)
class ComparisonResult:
    """Cross-cultural geometry comparison result.

    Attributes
    ----------
    gram_roughness_a : float
        Roughness of model A's Gram matrix.
    gram_roughness_b : float
        Roughness of model B's Gram matrix.
    merged_gram_roughness : float
        Roughness of merged Gram matrix.
    roughness_reduction : float
        Reduction in roughness from merging.
    row_correlations : list[float]
        Per-prime Pearson correlations between Gram rows.
    row_sharpness_a : list[float]
        Per-prime sharpness (row variance) for model A.
    row_sharpness_b : list[float]
        Per-prime sharpness (row variance) for model B.
    row_sharpness_ratio : list[float]
        Per-prime sharpness ratio max(a, b) / min(a, b).
    category_divergence : dict[str, float]
        Per-category divergence scores.
    trajectory_analysis : PathComparison or None
        Optional trajectory analysis.
    """

    gram_roughness_a: float
    gram_roughness_b: float
    merged_gram_roughness: float
    roughness_reduction: float
    row_correlations: list[float]
    row_sharpness_a: list[float]
    row_sharpness_b: list[float]
    row_sharpness_ratio: list[float]
    category_divergence: dict[str, float]
    trajectory_analysis: PathComparison | None = None


@dataclass(frozen=True)
class AlignmentAnalysis:
    """Cross-cultural alignment analysis.

    Attributes
    ----------
    cka : float
        CKA score [0, 1].
    raw_pearson : float
        Pearson correlation of off-diagonal Gram elements.
    alignment_gap : float
        Difference cka - raw_pearson. Large gap indicates centering matters.
    """

    cka: float
    raw_pearson: float
    alignment_gap: float


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
        backend = get_default_backend()
        sharpness_values = sharpness_a + sharpness_b
        eps_source = sharpness_values if sharpness_values else [0.0]
        eps = division_epsilon(backend, backend.array(eps_source))
        sharpness_ratios = []
        for s_a, s_b in zip(sharpness_a, sharpness_b):
            denom = min(s_a, s_b)
            denom = max(denom, eps)
            sharpness_ratios.append(max(s_a, s_b) / denom)

        category_divergence = CrossCulturalGeometry._compute_category_divergence(
            row_correlations,
            prime_ids,
            prime_categories,
        )
        return ComparisonResult(
            gram_roughness_a=roughness_a,
            gram_roughness_b=roughness_b,
            merged_gram_roughness=merged_roughness,
            roughness_reduction=roughness_reduction,
            row_correlations=row_correlations,
            row_sharpness_a=sharpness_a,
            row_sharpness_b=sharpness_b,
            row_sharpness_ratio=sharpness_ratios,
            category_divergence=category_divergence,
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
    def compute_cka(
        gram_a: list[float],
        gram_b: list[float],
        n: int,
        feature_dim_a: int | None = None,
        feature_dim_b: int | None = None,
    ) -> float:
        """Compute CKA between two flattened gram matrices.

        Delegates to the canonical implementation in cka.py.
        Uses feature_bias_correction when feature dimensions are provided.
        """
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return 0.0
        return compute_cka_from_grams(
            gram_a,
            gram_b,
            n,
            estimator=HSICEstimator.AUTO,
            feature_dim_a=feature_dim_a,
            feature_dim_b=feature_dim_b,
            feature_bias_correction=feature_dim_a is not None and feature_dim_b is not None,
        )

    @staticmethod
    def analyze_alignment(
        gram_a: list[float],
        gram_b: list[float],
        n: int,
        raw_pearson: float | None = None,
    ) -> AlignmentAnalysis | None:
        """Analyze alignment between two Gram matrices."""
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
            pearson = compute_pearson_correlation(off_diag_a, off_diag_b, default=0.0)
        else:
            pearson = raw_pearson

        gap = cka - pearson

        return AlignmentAnalysis(
            cka=cka,
            raw_pearson=pearson,
            alignment_gap=gap,
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
            correlations.append(compute_pearson_correlation(vec_a, vec_b, default=0.0))
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
            divergence[category] = 1.0 - mean_corr
        return divergence
