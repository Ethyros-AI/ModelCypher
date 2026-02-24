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
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)
from modelcypher.core.domain.geometry.path_geometry import (
    PathComparison,
    PathGeometry,
    PathSignature,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_pairwise_metrics


@dataclass(frozen=True)
class CrossCulturalComparisonResult:
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
        Per-prime geodesic cosine correlations between Gram rows.
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
        Geodesic cosine correlation of off-diagonal Gram elements.
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
    ) -> CrossCulturalComparisonResult | None:
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

        # Vectorized sharpness ratio computation
        s_a_arr = backend.array(sharpness_a)
        s_b_arr = backend.array(sharpness_b)
        min_arr = backend.minimum(s_a_arr, s_b_arr)
        max_arr = backend.maximum(s_a_arr, s_b_arr)
        safe_denom = backend.maximum(min_arr, backend.full(min_arr.shape, eps))
        ratio_arr = max_arr / safe_denom
        backend.eval(ratio_arr)
        sharpness_ratios = backend.tolist(ratio_arr)

        category_divergence = CrossCulturalGeometry._compute_category_divergence(
            row_correlations,
            prime_ids,
            prime_categories,
        )
        return CrossCulturalComparisonResult(
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

        Delegates to the canonical geodesic RBF implementation in cka.py.
        """
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return 0.0

        backend = get_default_backend()
        arr_a = backend.reshape(backend.array(gram_a), (n, n))
        arr_b = backend.reshape(backend.array(gram_b), (n, n))

        return compute_cka_from_grams(arr_a, arr_b, backend)

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
            # Vectorized geodesic correlation on off-diagonal elements
            backend = get_default_backend()
            gram_a_arr = backend.reshape(backend.array(gram_a), (n, n))
            gram_b_arr = backend.reshape(backend.array(gram_b), (n, n))
            mask = 1.0 - backend.eye(n)  # 1 for off-diagonal, 0 for diagonal
            off_diag_count = float(n * (n - 1))

            # Compute means of off-diagonal elements
            sum_a = backend.sum(gram_a_arr * mask)
            sum_b = backend.sum(gram_b_arr * mask)
            backend.eval(sum_a, sum_b)
            sum_a_val = float(backend.to_scalar(sum_a))
            sum_b_val = float(backend.to_scalar(sum_b))
            mean_a = sum_a_val / off_diag_count
            mean_b = sum_b_val / off_diag_count

            # Compute geodesic cosine of centered off-diagonal elements
            centered_a = (gram_a_arr - mean_a) * mask
            centered_b = (gram_b_arr - mean_b) * mask
            centered_a_flat = backend.reshape(centered_a, (1, -1))
            centered_b_flat = backend.reshape(centered_b, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(centered_a_flat, centered_b_flat, backend)
            backend.eval(cos_arr)
            pearson = float(backend.to_scalar(cos_arr[0]))
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
        if n <= 1:
            return []
        # Vectorized row sharpness: variance of off-diagonal elements per row
        backend = get_default_backend()
        gram_arr = backend.reshape(backend.array(gram), (n, n))
        mask = 1.0 - backend.eye(n)  # 1 for off-diagonal, 0 for diagonal
        off_diag_count = float(n - 1)

        # Per-row sums of off-diagonal elements
        row_sums = backend.sum(gram_arr * mask, axis=1)
        backend.eval(row_sums)
        row_means = row_sums / off_diag_count

        # Per-row variance: mean of (x - mean)² over off-diagonal
        centered = (gram_arr - backend.reshape(row_means, (n, 1))) * mask
        row_var_sums = backend.sum(centered * centered, axis=1)
        backend.eval(row_var_sums)
        row_vars = row_var_sums / off_diag_count
        backend.eval(row_vars)

        return [float(x) for x in backend.tolist(row_vars)]

    @staticmethod
    def _compute_row_correlations(gram_a: list[float], gram_b: list[float], n: int) -> list[float]:
        if n <= 1:
            return []
        # Vectorized per-row geodesic correlation of off-diagonal elements
        backend = get_default_backend()
        arr_a = backend.reshape(backend.array(gram_a), (n, n))
        arr_b = backend.reshape(backend.array(gram_b), (n, n))
        mask = 1.0 - backend.eye(n)  # 1 for off-diagonal, 0 for diagonal
        off_diag_count = float(n - 1)

        # Per-row means of off-diagonal elements
        sum_a = backend.sum(arr_a * mask, axis=1)
        sum_b = backend.sum(arr_b * mask, axis=1)
        backend.eval(sum_a, sum_b)
        mean_a = sum_a / off_diag_count
        mean_b = sum_b / off_diag_count

        # Per-row centered values (broadcast row means)
        centered_a = (arr_a - backend.reshape(mean_a, (n, 1))) * mask
        centered_b = (arr_b - backend.reshape(mean_b, (n, 1))) * mask
        cos_arr, _ = geodesic_pairwise_metrics(centered_a, centered_b, backend)
        backend.eval(cos_arr)
        return [float(x) for x in backend.tolist(cos_arr)]

    @staticmethod
    def _average_grams(gram_a: list[float], gram_b: list[float]) -> list[float]:
        backend = get_default_backend()
        arr_a = backend.array(gram_a)
        arr_b = backend.array(gram_b)
        avg = (arr_a + arr_b) / 2.0
        backend.eval(avg)
        return backend.tolist(avg)

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
