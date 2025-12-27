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

"""Domain Geometry Validator: Compare model geometry against established baselines.

This module provides baseline-relative measurements of LLM representation geometry
by comparing metrics against empirically-established baselines from known-good models.

Usage:
    validator = DomainGeometryValidator()
    results = validator.validate_model("/path/to/model", domains=["spatial", "moral"])

    for result in results:
        print(result.metrics["ollivier_ricci_mean"].z_score)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.domain_geometry_baselines import (
    BaselineMetricDelta,
    BaselineRepository,
    BaselineValidationResult,
    DomainGeometryBaseline,
    DomainGeometryBaselineExtractor,
    DomainType,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class DomainGeometryValidator:
    """Compare model geometry against established baselines.

    This validator compares a model's geometry profile against empirically-
    established baselines and returns baseline-relative measurements.
    """

    def __init__(
        self,
        baseline_dir: str | Path | None = None,
        backend: "Backend | None" = None,
    ):
        """Initialize the validator."""
        self._backend = backend or get_default_backend()
        self._repository = BaselineRepository(baseline_dir)
        self._extractor = DomainGeometryBaselineExtractor(backend=self._backend)

    @property
    def all_baselines(self) -> list[DomainGeometryBaseline]:
        """Get all available baselines."""
        return self._repository.get_all_baselines()

    def get_baselines_for_domain(self, domain: str) -> list[DomainGeometryBaseline]:
        """Get all baselines for a specific domain."""
        return self._repository.get_baselines_for_domain(domain)

    def get_baseline(
        self, domain: str, model_family: str, model_size: str
    ) -> DomainGeometryBaseline | None:
        """Get a specific baseline."""
        return self._repository.get_baseline(domain, model_family, model_size)

    def validate_model(
        self,
        model_path: str,
        domains: list[str] | None = None,
        layers: list[int] | None = None,
    ) -> list[BaselineValidationResult]:
        """Validate a model against baselines for specified domains.

        Args:
            model_path: Path to the model to validate
            domains: Domains to validate (default: all four)
            layers: Specific layers to analyze

        Returns:
            List of validation results, one per domain
        """
        domains = domains or [d.value for d in DomainType]
        results = []

        for domain in domains:
            try:
                result = self._validate_domain(model_path, domain, layers)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation failed for {domain}: {e}")
                results.append(
                    BaselineValidationResult(
                        domain=domain,
                        metrics={},
                        baseline_found=False,
                        missing_metrics=[],
                        notes=[f"Validation error: {e}"],
                        current_model=model_path,
                    )
                )

        return results

    def _validate_domain(
        self,
        model_path: str,
        domain: str,
        layers: list[int] | None,
    ) -> BaselineValidationResult:
        """Validate a single domain.

        This method:
        1. Extracts current model geometry
        2. Finds matching baseline
        3. Computes baseline-relative deltas (no hardcoded thresholds)
        """
        logger.info(f"Validating {domain} geometry for {model_path}")

        # Extract current geometry
        current = self._extractor.extract_baseline(model_path, domain, layers)

        # Find matching baseline
        baseline = self._repository.find_matching_baseline(
            domain, current.model_family, current.model_size
        )

        metrics, missing = self._build_metric_deltas(current, baseline)
        notes: list[str] = []
        baseline_model = ""
        if baseline is None:
            notes.append("No matching baseline found; returning raw measurements only.")
        else:
            baseline_model = f"{baseline.model_family}-{baseline.model_size}"

        return BaselineValidationResult(
            domain=current.domain,
            metrics=metrics,
            baseline_found=baseline is not None,
            missing_metrics=missing,
            notes=notes,
            baseline_model=baseline_model,
            current_model=current.model_path,
        )

    def _validate_against_baseline(
        self,
        current: DomainGeometryBaseline,
        baseline: DomainGeometryBaseline,
    ) -> BaselineValidationResult:
        """Baseline-relative deltas for current geometry."""
        metrics, missing = self._build_metric_deltas(current, baseline)
        return BaselineValidationResult(
            domain=current.domain,
            metrics=metrics,
            baseline_found=True,
            missing_metrics=missing,
            notes=[],
            baseline_model=f"{baseline.model_family}-{baseline.model_size}",
            current_model=current.model_path,
        )

    def _build_metric_deltas(
        self,
        current: DomainGeometryBaseline,
        baseline: DomainGeometryBaseline | None,
    ) -> tuple[dict[str, BaselineMetricDelta], list[str]]:
        metrics: dict[str, BaselineMetricDelta] = {}
        missing: list[str] = []

        baseline_ricci_values = baseline.layer_ricci_values if baseline else None
        metrics["ollivier_ricci_mean"] = self._metric_delta(
            current.ollivier_ricci_mean,
            baseline.ollivier_ricci_mean if baseline else None,
            baseline_std=baseline.ollivier_ricci_std if baseline else None,
            baseline_distribution=baseline_ricci_values,
        )
        metrics["ollivier_ricci_std"] = self._metric_delta(
            current.ollivier_ricci_std,
            baseline.ollivier_ricci_std if baseline else None,
        )
        metrics["ollivier_ricci_min"] = self._metric_delta(
            current.ollivier_ricci_min,
            baseline.ollivier_ricci_min if baseline else None,
        )
        metrics["ollivier_ricci_max"] = self._metric_delta(
            current.ollivier_ricci_max,
            baseline.ollivier_ricci_max if baseline else None,
        )

        health = current.manifold_health_distribution
        baseline_health = baseline.manifold_health_distribution if baseline else None
        metrics["healthy_layer_fraction"] = self._metric_delta(
            health.healthy,
            baseline_health.healthy if baseline_health else None,
        )
        metrics["degenerate_layer_fraction"] = self._metric_delta(
            health.degenerate,
            baseline_health.degenerate if baseline_health else None,
        )
        metrics["collapsed_layer_fraction"] = self._metric_delta(
            health.collapsed,
            baseline_health.collapsed if baseline_health else None,
        )

        metrics["intrinsic_dimension_mean"] = self._metric_delta(
            current.intrinsic_dimension_mean,
            baseline.intrinsic_dimension_mean if baseline else None,
            baseline_std=baseline.intrinsic_dimension_std if baseline else None,
        )
        metrics["intrinsic_dimension_std"] = self._metric_delta(
            current.intrinsic_dimension_std,
            baseline.intrinsic_dimension_std if baseline else None,
        )

        baseline_domain_metrics = baseline.domain_metrics if baseline else {}
        if baseline is None:
            all_domain_metrics = set(current.domain_metrics.keys())
        else:
            all_domain_metrics = set(current.domain_metrics.keys()) | set(baseline_domain_metrics.keys())
        for metric in sorted(all_domain_metrics):
            current_value = current.domain_metrics.get(metric)
            baseline_value = baseline_domain_metrics.get(metric)
            if baseline is not None:
                if metric not in baseline_domain_metrics:
                    missing.append(f"baseline:{metric}")
                if metric not in current.domain_metrics:
                    missing.append(f"current:{metric}")
            metrics[f"domain_{metric}"] = self._metric_delta(
                current_value,
                baseline_value,
            )

        return metrics, missing

    def _metric_delta(
        self,
        current: float | None,
        baseline: float | None,
        *,
        baseline_std: float | None = None,
        baseline_distribution: list[float] | None = None,
    ) -> BaselineMetricDelta:
        delta = None
        relative_delta = None
        z_score = None
        percentile = None

        if current is not None and baseline is not None:
            delta = current - baseline
            if baseline != 0:
                relative_delta = delta / abs(baseline)
            if baseline_std:
                z_score = delta / baseline_std
            if baseline_distribution:
                percentile = self._percentile_rank(baseline_distribution, current)

        return BaselineMetricDelta(
            current=current,
            baseline=baseline,
            baseline_std=baseline_std,
            delta=delta,
            relative_delta=relative_delta,
            z_score=z_score,
            percentile=percentile,
        )

    def _percentile_rank(self, values: list[float], value: float) -> float | None:
        if not values:
            return None
        if len(values) == 1:
            return 1.0
        sorted_vals = sorted(values)
        import bisect

        idx = bisect.bisect_left(sorted_vals, value)
        rank = idx / float(len(sorted_vals) - 1)
        return max(0.0, min(1.0, rank))

    def validate_baseline_sanity(
        self,
        baseline: DomainGeometryBaseline,
    ) -> tuple[bool, list[str]]:
        """Check that a baseline has usable measurement data."""
        issues: list[str] = []

        if baseline.layers_analyzed == 0:
            issues.append("No layers were analyzed")
        if not baseline.layer_ricci_values:
            issues.append("No Ricci curvature values recorded")

        return len(issues) == 0, issues


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_model_geometry(
    model_path: str,
    domains: list[str] | None = None,
    baseline_dir: str | Path | None = None,
) -> list[BaselineValidationResult]:
    """Convenience function to validate model geometry.

    Args:
        model_path: Path to the model to validate
        domains: Domains to validate (default: all)
        baseline_dir: Directory containing baselines

    Returns:
        List of validation results
    """
    validator = DomainGeometryValidator(baseline_dir=baseline_dir)
    return validator.validate_model(model_path, domains=domains)


def extract_and_save_baseline(
    model_path: str,
    domain: str,
    output_dir: str | Path | None = None,
) -> tuple[DomainGeometryBaseline, Path]:
    """Extract and save a baseline from a model.

    Args:
        model_path: Path to the reference model
        domain: Domain to extract baseline for
        output_dir: Where to save the baseline

    Returns:
        (baseline, save_path)
    """
    extractor = DomainGeometryBaselineExtractor()
    baseline = extractor.extract_baseline(model_path, domain)

    repository = BaselineRepository(output_dir)
    save_path = repository.save_baseline(baseline)

    return baseline, save_path
