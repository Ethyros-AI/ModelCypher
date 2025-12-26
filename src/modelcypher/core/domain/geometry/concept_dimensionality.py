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

"""Concept dimensionality analysis for atlas probes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.unified_atlas import AtlasProbe
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    BootstrapConfiguration,
    GeodesicConfiguration,
    IntrinsicDimension,
    TwoNNConfiguration,
)
from modelcypher.core.domain.geometry.probe_calibration import ActivationProvider
from modelcypher.core.domain.geometry.vector_math import VectorMath

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptDimensionalityConfig:
    min_support_texts: int = 3
    max_support_texts: int = 6
    max_total_texts: int = 8
    include_name_description: bool = True
    include_probe_name: bool = True
    use_regression: bool = True
    bootstrap_resamples: int = 0
    bootstrap_seed: int = 42
    geodesic_k_neighbors: int = 10
    geodesic_distance_power: float = 2.0
    min_calibration_weight: float | None = None


@dataclass(frozen=True)
class ConceptDimensionalityResult:
    probe_id: str
    name: str
    source: str
    domain: str
    category: str
    layer: int
    support_text_count: int
    sample_count: int
    usable_count: int
    intrinsic_dimension: float
    dimension_class: str
    calibration_weight: float | None
    ci_lower: float | None
    ci_upper: float | None


@dataclass(frozen=True)
class SkippedProbe:
    probe_id: str
    name: str
    reason: str
    support_text_count: int
    calibration_weight: float | None
    activation_count: int | None = None
    invalid_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class DomainSummary:
    domain: str
    probe_count: int
    mean_dimension: float | None
    dimension_histogram: dict[str, int]


@dataclass(frozen=True)
class ConceptDimensionalityReport:
    layer: int
    total_probes: int
    analyzed_count: int
    skipped_count: int
    mean_dimension: float | None
    weighted_mean_dimension: float | None
    dimension_histogram: dict[str, int]
    domain_summaries: list[DomainSummary]
    results: list[ConceptDimensionalityResult]
    skipped: list[SkippedProbe]


@dataclass(frozen=True)
class LayerDimensionalitySummary:
    layer: int
    mean_dimension: float | None
    dimension_histogram: dict[str, int]
    domain_mean_dimensions: dict[str, float]
    domain_rank: list[str]


@dataclass(frozen=True)
class DomainRankCorrelation:
    layer_a: int
    layer_b: int
    domain_count: int
    spearman: float | None


@dataclass(frozen=True)
class ConceptDimensionalityStudyReport:
    layers: list[int]
    layer_summaries: list[LayerDimensionalitySummary]
    bottleneck_layer: int | None
    bottleneck_mean_dimension: float | None
    endpoint_mean_dimension: float | None
    collapse_ratio: float | None
    domain_rank_correlations: list[DomainRankCorrelation]
    mean_domain_rank_correlation: float | None


class ConceptDimensionalityAnalyzer:
    """Measure intrinsic dimensionality of atlas probe concept clouds."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        probes: list[AtlasProbe],
        activation_provider: ActivationProvider,
        layer: int,
        config: ConceptDimensionalityConfig | None = None,
        calibration_weights: dict[str, float] | None = None,
    ) -> ConceptDimensionalityReport:
        resolved = config or ConceptDimensionalityConfig()
        results: list[ConceptDimensionalityResult] = []
        skipped: list[SkippedProbe] = []

        for probe in probes:
            weight = calibration_weights.get(probe.probe_id) if calibration_weights else None
            if (
                resolved.min_calibration_weight is not None
                and weight is not None
                and weight < resolved.min_calibration_weight
            ):
                skipped.append(
                    SkippedProbe(
                        probe_id=probe.probe_id,
                        name=probe.name,
                        reason="calibration_below_threshold",
                        support_text_count=0,
                        calibration_weight=weight,
                    )
                )
                continue

            texts = self._build_support_texts(probe, resolved)
            if len(texts) < resolved.min_support_texts:
                skipped.append(
                    SkippedProbe(
                        probe_id=probe.probe_id,
                        name=probe.name,
                        reason="insufficient_support_texts",
                        support_text_count=len(texts),
                        calibration_weight=weight,
                    )
                )
                continue

            try:
                activations = activation_provider.get_activations(texts, layer)
            except Exception as exc:
                logger.debug("Activation provider failed for %s: %s", probe.probe_id, exc)
                skipped.append(
                    SkippedProbe(
                        probe_id=probe.probe_id,
                        name=probe.name,
                        reason="activation_provider_error",
                        support_text_count=len(texts),
                        calibration_weight=weight,
                    )
                )
                continue

            vectors, invalid_counts = self._filter_vectors(activations)
            if len(vectors) < resolved.min_support_texts:
                skipped.append(
                    SkippedProbe(
                        probe_id=probe.probe_id,
                        name=probe.name,
                        reason="insufficient_valid_vectors",
                        support_text_count=len(vectors),
                        calibration_weight=weight,
                        activation_count=len(activations),
                        invalid_counts=invalid_counts,
                    )
                )
                continue

            estimate = self._compute_intrinsic_dimension(vectors, resolved)
            if estimate is None:
                skipped.append(
                    SkippedProbe(
                        probe_id=probe.probe_id,
                        name=probe.name,
                        reason="intrinsic_dimension_failed",
                        support_text_count=len(vectors),
                        calibration_weight=weight,
                    )
                )
                continue

            dim_class = self._dimension_class(estimate.intrinsic_dimension)
            results.append(
                ConceptDimensionalityResult(
                    probe_id=probe.probe_id,
                    name=probe.name,
                    source=probe.source.value,
                    domain=probe.domain.value,
                    category=probe.category_name,
                    layer=layer,
                    support_text_count=len(texts),
                    sample_count=estimate.sample_count,
                    usable_count=estimate.usable_count,
                    intrinsic_dimension=estimate.intrinsic_dimension,
                    dimension_class=dim_class,
                    calibration_weight=weight,
                    ci_lower=estimate.ci.lower if estimate.ci else None,
                    ci_upper=estimate.ci.upper if estimate.ci else None,
                )
            )

        histogram = self._dimension_histogram(results)
        mean_dim = self._mean_dimension(results)
        weighted_mean = self._weighted_mean_dimension(results)
        domain_summaries = self._summarize_domains(results)

        return ConceptDimensionalityReport(
            layer=layer,
            total_probes=len(probes),
            analyzed_count=len(results),
            skipped_count=len(skipped),
            mean_dimension=mean_dim,
            weighted_mean_dimension=weighted_mean,
            dimension_histogram=histogram,
            domain_summaries=domain_summaries,
            results=results,
            skipped=skipped,
        )

    @staticmethod
    def _build_support_texts(
        probe: AtlasProbe,
        config: ConceptDimensionalityConfig,
    ) -> list[str]:
        texts: list[str] = []

        if config.include_name_description:
            if probe.name and probe.description:
                texts.append(f"{probe.name}: {probe.description}")
            elif probe.name:
                texts.append(probe.name)
            elif probe.description:
                texts.append(probe.description)

        support_texts = list(probe.support_texts)
        if config.max_support_texts > 0:
            support_texts = support_texts[: config.max_support_texts]
        for text in support_texts:
            if text and text not in texts:
                texts.append(text)

        if config.include_probe_name and probe.name and probe.name not in texts:
            texts.append(probe.name)

        if probe.name:
            fallback = f"The concept of {probe.name}"
            if fallback not in texts:
                texts.append(fallback)

        if config.max_total_texts > 0 and len(texts) > config.max_total_texts:
            texts = texts[: config.max_total_texts]

        return texts

    @staticmethod
    def _filter_vectors(vectors: list[list[float]]) -> tuple[list[list[float]], dict[str, int]]:
        cleaned: list[list[float]] = []
        invalid_counts = {
            "empty": 0,
            "length_mismatch": 0,
            "non_finite": 0,
        }
        expected_dim = None
        for vec in vectors:
            if not vec:
                invalid_counts["empty"] += 1
                continue
            if expected_dim is None:
                expected_dim = len(vec)
            if len(vec) != expected_dim:
                invalid_counts["length_mismatch"] += 1
                continue
            if any(not math.isfinite(float(v)) for v in vec):
                invalid_counts["non_finite"] += 1
                continue
            cleaned.append([float(v) for v in vec])
        return cleaned, invalid_counts

    def _compute_intrinsic_dimension(
        self,
        vectors: list[list[float]],
        config: ConceptDimensionalityConfig,
    ):
        if len(vectors) < 3:
            return None
        k_neighbors = min(config.geodesic_k_neighbors, len(vectors) - 1)
        k_neighbors = max(1, k_neighbors)
        two_nn = TwoNNConfiguration(
            use_regression=config.use_regression,
            geodesic=GeodesicConfiguration(
                k_neighbors=k_neighbors,
                distance_power=config.geodesic_distance_power,
            ),
        )
        bootstrap = None
        if config.bootstrap_resamples and config.bootstrap_resamples > 0:
            bootstrap = BootstrapConfiguration(
                resamples=config.bootstrap_resamples,
                confidence_level=0.95,
                seed=config.bootstrap_seed,
            )
        try:
            points = self._backend.array(vectors)
            return IntrinsicDimension(self._backend).compute(points, two_nn, bootstrap=bootstrap)
        except Exception as exc:
            logger.debug("Intrinsic dimension failed: %s", exc)
            return None

    @staticmethod
    def _dimension_class(value: float) -> str:
        if value < 1.5:
            return "1D"
        if value < 2.5:
            return "2D"
        if value < 3.5:
            return "3D"
        return "4D+"

    @staticmethod
    def _dimension_histogram(
        results: list[ConceptDimensionalityResult],
    ) -> dict[str, int]:
        histogram: dict[str, int] = {"1D": 0, "2D": 0, "3D": 0, "4D+": 0}
        for result in results:
            histogram[result.dimension_class] = histogram.get(result.dimension_class, 0) + 1
        return histogram

    @staticmethod
    def _mean_dimension(results: list[ConceptDimensionalityResult]) -> float | None:
        dims = [r.intrinsic_dimension for r in results if math.isfinite(r.intrinsic_dimension)]
        if not dims:
            return None
        return sum(dims) / float(len(dims))

    @staticmethod
    def _weighted_mean_dimension(results: list[ConceptDimensionalityResult]) -> float | None:
        weighted = [
            (r.intrinsic_dimension, r.calibration_weight)
            for r in results
            if r.calibration_weight is not None and math.isfinite(r.intrinsic_dimension)
        ]
        if not weighted:
            return None
        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0:
            return None
        return sum(dim * weight for dim, weight in weighted) / total_weight

    def _summarize_domains(
        self, results: list[ConceptDimensionalityResult]
    ) -> list[DomainSummary]:
        by_domain: dict[str, list[ConceptDimensionalityResult]] = {}
        for result in results:
            by_domain.setdefault(result.domain, []).append(result)

        summaries: list[DomainSummary] = []
        for domain, items in sorted(by_domain.items()):
            hist = self._dimension_histogram(items)
            mean_dim = self._mean_dimension(items)
            summaries.append(
                DomainSummary(
                    domain=domain,
                    probe_count=len(items),
                    mean_dimension=mean_dim,
                    dimension_histogram=hist,
                )
            )
        return summaries


class ConceptDimensionalityStudy:
    """Summarize dimensionality reports across layers."""

    @staticmethod
    def summarize(
        reports: list[ConceptDimensionalityReport],
    ) -> ConceptDimensionalityStudyReport:
        if not reports:
            return ConceptDimensionalityStudyReport(
                layers=[],
                layer_summaries=[],
                bottleneck_layer=None,
                bottleneck_mean_dimension=None,
                endpoint_mean_dimension=None,
                collapse_ratio=None,
                domain_rank_correlations=[],
                mean_domain_rank_correlation=None,
            )

        sorted_reports = sorted(reports, key=lambda r: r.layer)
        summaries: list[LayerDimensionalitySummary] = []
        for report in sorted_reports:
            domain_means = {
                summary.domain: summary.mean_dimension
                for summary in report.domain_summaries
                if summary.mean_dimension is not None
            }
            domain_rank = [
                name for name, _ in sorted(domain_means.items(), key=lambda item: item[1])
            ]
            summaries.append(
                LayerDimensionalitySummary(
                    layer=report.layer,
                    mean_dimension=report.mean_dimension,
                    dimension_histogram=report.dimension_histogram,
                    domain_mean_dimensions=domain_means,
                    domain_rank=domain_rank,
                )
            )

        bottleneck_layer = None
        bottleneck_mean = None
        for summary in summaries:
            if summary.mean_dimension is None:
                continue
            if bottleneck_mean is None or summary.mean_dimension < bottleneck_mean:
                bottleneck_mean = summary.mean_dimension
                bottleneck_layer = summary.layer

        endpoint_mean = None
        if len(summaries) >= 2:
            first = summaries[0].mean_dimension
            last = summaries[-1].mean_dimension
            if first is not None and last is not None:
                endpoint_mean = (first + last) / 2.0

        collapse_ratio = None
        if endpoint_mean is not None and bottleneck_mean is not None and endpoint_mean > 0:
            collapse_ratio = bottleneck_mean / endpoint_mean

        correlations: list[DomainRankCorrelation] = []
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                a = summaries[i]
                b = summaries[j]
                common = sorted(set(a.domain_mean_dimensions) & set(b.domain_mean_dimensions))
                if len(common) < 2:
                    correlations.append(
                        DomainRankCorrelation(
                            layer_a=a.layer,
                            layer_b=b.layer,
                            domain_count=len(common),
                            spearman=None,
                        )
                    )
                    continue
                vals_a = [a.domain_mean_dimensions[name] for name in common]
                vals_b = [b.domain_mean_dimensions[name] for name in common]
                spearman = VectorMath.spearman_correlation(vals_a, vals_b)
                correlations.append(
                    DomainRankCorrelation(
                        layer_a=a.layer,
                        layer_b=b.layer,
                        domain_count=len(common),
                        spearman=spearman,
                    )
                )

        spearman_values = [
            item.spearman for item in correlations if item.spearman is not None
        ]
        mean_spearman = None
        if spearman_values:
            mean_spearman = sum(spearman_values) / float(len(spearman_values))

        return ConceptDimensionalityStudyReport(
            layers=[summary.layer for summary in summaries],
            layer_summaries=summaries,
            bottleneck_layer=bottleneck_layer,
            bottleneck_mean_dimension=bottleneck_mean,
            endpoint_mean_dimension=endpoint_mean,
            collapse_ratio=collapse_ratio,
            domain_rank_correlations=correlations,
            mean_domain_rank_correlation=mean_spearman,
        )
