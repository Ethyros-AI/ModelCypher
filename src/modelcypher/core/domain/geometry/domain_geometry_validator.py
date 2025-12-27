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

"""Domain Geometry Validator: Validate model geometry against established baselines.

This module provides validation of LLM representation geometry by comparing
measured metrics against empirically-established baselines from known-good models.

Key validation criteria (from SOTA research):
1. Ollivier-Ricci curvature should be negative (healthy hyperbolic geometry)
2. Majority of layers should be classified as "healthy"
3. Domain-specific metrics should fall within acceptable deviation of baselines
4. Intrinsic dimension should be consistent with model family

Usage:
    validator = DomainGeometryValidator()
    results = validator.validate_model("/path/to/model", domains=["spatial", "moral"])

    for result in results:
        if not result.passed:
            print(f"{result.domain} validation failed: {result.warnings}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.domain_geometry_baselines import (
    BaselineRepository,
    BaselineValidationResult,
    DomainGeometryBaseline,
    DomainGeometryBaselineExtractor,
    DomainType,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""

    # Deviation thresholds (fraction of baseline value)
    ricci_deviation_threshold: float = 0.3  # 30% deviation in Ricci curvature
    metric_deviation_threshold: float = 0.25  # 25% deviation in domain metrics
    id_deviation_threshold: float = 0.4  # 40% deviation in intrinsic dimension

    # Hard limits (absolute values)
    max_acceptable_ricci: float = 0.05  # Curvature above this is concerning
    min_healthy_layer_fraction: float = 0.4  # At least 40% healthy layers
    max_collapsed_layer_fraction: float = 0.2  # No more than 20% collapsed

    # Behavior
    require_baseline_match: bool = False  # If True, fail when no baseline found
    fallback_to_heuristics: bool = True  # Use hard limits when no baseline


class DomainGeometryValidator:
    """Validate model geometry against established baselines.

    This validator compares a model's geometry profile against empirically-
    established baselines to detect:
    - Representation collapse (positive Ricci curvature)
    - Loss of geometric structure (degenerate curvature)
    - Domain-specific degradation (poor axis alignment, etc.)
    """

    def __init__(
        self,
        baseline_dir: str | Path | None = None,
        config: ValidationConfig | None = None,
        backend: "Backend | None" = None,
    ):
        """Initialize the validator.

        Args:
            baseline_dir: Directory containing baseline JSON files
            config: Validation configuration
            backend: Compute backend (defaults to MLX/JAX)
        """
        self._backend = backend or get_default_backend()
        self._config = config or ValidationConfig()
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
                        passed=False,
                        overall_deviation=1.0,
                        deviation_scores={},
                        warnings=[f"Validation error: {e}"],
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
        3. Computes deviations
        4. Applies hard limits if no baseline
        5. Returns validation result
        """
        logger.info(f"Validating {domain} geometry for {model_path}")

        # Extract current geometry
        current = self._extractor.extract_baseline(model_path, domain, layers)

        # Find matching baseline
        baseline = self._repository.find_matching_baseline(
            domain, current.model_family, current.model_size
        )

        if baseline:
            return self._validate_against_baseline(current, baseline)
        elif self._config.require_baseline_match:
            return BaselineValidationResult(
                domain=domain,
                passed=False,
                overall_deviation=1.0,
                deviation_scores={},
                warnings=[f"No matching baseline found for {current.model_family}-{current.model_size}"],
                current_model=model_path,
            )
        elif self._config.fallback_to_heuristics:
            return self._validate_with_heuristics(current)
        else:
            return BaselineValidationResult(
                domain=domain,
                passed=True,
                overall_deviation=0.0,
                deviation_scores={},
                warnings=["No baseline available, validation skipped"],
                current_model=model_path,
            )

    def _validate_against_baseline(
        self,
        current: DomainGeometryBaseline,
        baseline: DomainGeometryBaseline,
    ) -> BaselineValidationResult:
        """Validate current geometry against a baseline.

        Computes normalized deviations for each metric and determines
        pass/fail based on thresholds.
        """
        deviations: dict[str, float] = {}
        warnings: list[str] = []
        recommendations: list[str] = []

        # 1. Ollivier-Ricci curvature deviation
        ricci_dev = self._compute_deviation(
            current.ollivier_ricci_mean,
            baseline.ollivier_ricci_mean,
            min_denominator=0.01,
        )
        deviations["ollivier_ricci_mean"] = ricci_dev

        if ricci_dev > self._config.ricci_deviation_threshold:
            warnings.append(
                f"Ricci curvature deviation {ricci_dev:.2f} exceeds threshold "
                f"({self._config.ricci_deviation_threshold})"
            )

        # Check for positive curvature (bad sign)
        if current.ollivier_ricci_mean > self._config.max_acceptable_ricci:
            warnings.append(
                f"Positive Ricci curvature ({current.ollivier_ricci_mean:.3f}) "
                "indicates potential representation collapse"
            )
            recommendations.append(
                "Consider reducing training intensity or checking for data issues"
            )

        # 2. Manifold health distribution
        health = current.manifold_health_distribution
        baseline_health = baseline.manifold_health_distribution

        healthy_dev = abs(health.healthy - baseline_health.healthy)
        deviations["healthy_layer_fraction"] = healthy_dev

        if health.healthy < self._config.min_healthy_layer_fraction:
            warnings.append(
                f"Only {health.healthy:.0%} healthy layers "
                f"(minimum: {self._config.min_healthy_layer_fraction:.0%})"
            )

        if health.collapsed > self._config.max_collapsed_layer_fraction:
            warnings.append(
                f"{health.collapsed:.0%} collapsed layers "
                f"(maximum: {self._config.max_collapsed_layer_fraction:.0%})"
            )
            recommendations.append(
                "High collapse rate may indicate over-training or architecture issues"
            )

        # 3. Intrinsic dimension deviation
        if baseline.intrinsic_dimension_mean > 0:
            id_dev = self._compute_deviation(
                current.intrinsic_dimension_mean,
                baseline.intrinsic_dimension_mean,
            )
            deviations["intrinsic_dimension"] = id_dev

            if id_dev > self._config.id_deviation_threshold:
                warnings.append(
                    f"Intrinsic dimension deviation {id_dev:.2f} exceeds threshold"
                )

        # 4. Domain-specific metrics
        for metric, value in current.domain_metrics.items():
            if metric in baseline.domain_metrics:
                baseline_value = baseline.domain_metrics[metric]
                if baseline_value != 0:
                    dev = self._compute_deviation(value, baseline_value)
                    deviations[f"domain_{metric}"] = dev

                    if dev > self._config.metric_deviation_threshold:
                        warnings.append(
                            f"Domain metric '{metric}' deviation {dev:.2f} exceeds threshold"
                        )

        # Compute overall deviation (weighted average)
        overall_deviation = self._compute_overall_deviation(deviations)

        # Determine pass/fail
        # Pass if: no critical warnings AND overall deviation acceptable
        critical_conditions = [
            current.ollivier_ricci_mean > self._config.max_acceptable_ricci,
            health.healthy < self._config.min_healthy_layer_fraction,
            health.collapsed > self._config.max_collapsed_layer_fraction,
        ]

        passed = not any(critical_conditions) and overall_deviation < 0.5

        return BaselineValidationResult(
            domain=current.domain,
            passed=passed,
            overall_deviation=overall_deviation,
            deviation_scores=deviations,
            warnings=warnings,
            recommendations=recommendations,
            baseline_model=f"{baseline.model_family}-{baseline.model_size}",
            current_model=current.model_path,
        )

    def _validate_with_heuristics(
        self,
        current: DomainGeometryBaseline,
    ) -> BaselineValidationResult:
        """Validate using hard-coded heuristics when no baseline available.

        Based on SOTA research:
        - Healthy LLMs have negative Ricci curvature
        - Most layers should be "healthy"
        - Collapsed fraction should be low
        """
        warnings: list[str] = []
        recommendations: list[str] = []
        deviations: dict[str, float] = {}

        # Check Ricci curvature
        if current.ollivier_ricci_mean > self._config.max_acceptable_ricci:
            warnings.append(
                f"Positive Ricci curvature ({current.ollivier_ricci_mean:.3f}) "
                "suggests representation collapse"
            )
            deviations["ollivier_ricci"] = current.ollivier_ricci_mean + 0.5

        # Expected range for healthy models: -0.4 to -0.1
        if current.ollivier_ricci_mean < -0.5:
            # Very negative might indicate over-dispersion
            warnings.append(
                f"Very negative Ricci curvature ({current.ollivier_ricci_mean:.3f}) "
                "may indicate excessive dispersion"
            )
        elif current.ollivier_ricci_mean > -0.05:
            # Near-flat or positive
            warnings.append(
                f"Near-flat/positive Ricci curvature ({current.ollivier_ricci_mean:.3f}) "
                "may indicate geometric structure loss"
            )

        # Check health distribution
        health = current.manifold_health_distribution

        if health.healthy < self._config.min_healthy_layer_fraction:
            warnings.append(
                f"Low healthy layer fraction ({health.healthy:.0%})"
            )
            recommendations.append(
                "Consider investigating layer-wise geometry for degradation patterns"
            )

        if health.collapsed > self._config.max_collapsed_layer_fraction:
            warnings.append(
                f"High collapsed layer fraction ({health.collapsed:.0%})"
            )
            recommendations.append(
                "Representation collapse detected - review training process"
            )

        # Heuristic-based deviation (no baseline to compare)
        # Use distance from ideal values
        ideal_ricci = -0.2  # Healthy hyperbolic
        ricci_deviation = abs(current.ollivier_ricci_mean - ideal_ricci) / 0.3
        deviations["ricci_from_ideal"] = ricci_deviation

        ideal_healthy = 0.7
        healthy_deviation = max(0, ideal_healthy - health.healthy) / 0.3
        deviations["healthy_from_ideal"] = healthy_deviation

        overall_deviation = (ricci_deviation + healthy_deviation) / 2

        # Pass if no critical issues
        passed = len([w for w in warnings if "collapse" in w.lower()]) == 0

        return BaselineValidationResult(
            domain=current.domain,
            passed=passed,
            overall_deviation=overall_deviation,
            deviation_scores=deviations,
            warnings=warnings,
            recommendations=recommendations if not passed else [],
            baseline_model="heuristic",
            current_model=current.model_path,
        )

    def _compute_deviation(
        self,
        current: float,
        baseline: float,
        min_denominator: float = 0.01,
    ) -> float:
        """Compute normalized deviation between current and baseline values.

        Returns a value where 0 = perfect match, 1 = 100% deviation.
        """
        denominator = max(abs(baseline), min_denominator)
        return abs(current - baseline) / denominator

    def _compute_overall_deviation(
        self,
        deviations: dict[str, float],
    ) -> float:
        """Compute weighted overall deviation score.

        Weights:
        - Ricci curvature: 40%
        - Health distribution: 30%
        - Domain metrics: 20%
        - Intrinsic dimension: 10%
        """
        if not deviations:
            return 0.0

        weights = {
            "ollivier_ricci_mean": 0.4,
            "healthy_layer_fraction": 0.3,
            "intrinsic_dimension": 0.1,
        }

        # Domain metrics get equal share of remaining 20%
        domain_metrics = [k for k in deviations if k.startswith("domain_")]
        if domain_metrics:
            domain_weight = 0.2 / len(domain_metrics)
            for metric in domain_metrics:
                weights[metric] = domain_weight

        total_weight = 0.0
        weighted_sum = 0.0

        for metric, dev in deviations.items():
            weight = weights.get(metric, 0.05)  # Default small weight
            weighted_sum += dev * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def validate_baseline_sanity(
        self,
        baseline: DomainGeometryBaseline,
    ) -> tuple[bool, list[str]]:
        """Check if a baseline itself appears healthy.

        Used to validate newly-extracted baselines before saving.

        Returns:
            (is_valid, list_of_issues)
        """
        issues: list[str] = []

        # Check Ricci curvature is negative (healthy)
        if baseline.ollivier_ricci_mean >= 0:
            issues.append(
                f"Non-negative Ricci curvature ({baseline.ollivier_ricci_mean:.3f}) - "
                "baseline model may be unhealthy"
            )

        # Check majority healthy layers
        if baseline.manifold_health_distribution.healthy < 0.5:
            issues.append(
                f"Less than 50% healthy layers ({baseline.manifold_health_distribution.healthy:.0%})"
            )

        # Check low collapsed fraction
        if baseline.manifold_health_distribution.collapsed > 0.3:
            issues.append(
                f"High collapsed fraction ({baseline.manifold_health_distribution.collapsed:.0%})"
            )

        # Check we have some data
        if baseline.layers_analyzed == 0:
            issues.append("No layers were analyzed")

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
