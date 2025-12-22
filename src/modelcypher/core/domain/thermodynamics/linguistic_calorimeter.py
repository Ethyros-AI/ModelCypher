"""Compatibility alias for the optimization metric calculator."""
from modelcypher.core.domain.dynamics.optimization_metric_calculator import (
    OptimizationMetricCalculator as LinguisticCalorimeter,
    OptimizationMeasurement,
    OptimizationResult,
    OptimizationMetricConfig,
)

__all__ = [
    "LinguisticCalorimeter",
    "OptimizationMeasurement",
    "OptimizationResult",
    "OptimizationMetricConfig",
]
