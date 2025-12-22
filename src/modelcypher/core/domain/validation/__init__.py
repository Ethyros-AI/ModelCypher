"""Validation domain modules.

This package contains modules for dataset validation and quality scoring,
including automatic dataset repair and quality metrics.
"""

from modelcypher.core.domain.validation.dataset_quality_scorer import (
    DatasetQualityScorer,
    QualityScore,
    ScoreRange,
)
from modelcypher.core.domain.validation.auto_fix_engine import (
    AutoFixEngine,
    AutoFixResult,
    Fix,
    FixType,
    UnfixableLine,
)

__all__ = [
    "AutoFixEngine",
    "AutoFixResult",
    "DatasetQualityScorer",
    "Fix",
    "FixType",
    "QualityScore",
    "ScoreRange",
    "UnfixableLine",
]
