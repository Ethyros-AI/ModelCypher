"""Dataset Quality Scorer.

Calculates dataset quality scores (0-100) based on validation results.

## Scoring Algorithm

Base score: 100

**Penalties:**
- Errors: -20 per error (JSON syntax, missing fields, etc.)
- Warnings: -5 per warning (few samples, short length, etc.)
- Few samples (<50): -15
- Short avg length (<50 chars): -10

**Bonuses:**
- Many samples (≥100): +10
- Good avg length (≥100 chars): +5

**Score Ranges:**
- 90-100: Production-ready dataset
- 70-89: Good quality, minor warnings
- 50-69: Usable but needs improvement
- 0-49: Critical issues, fix before training
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScoreRange(str, Enum):
    """Quality score ranges with interpretations."""

    PRODUCTION_READY = "production_ready"
    """90-100: Production-ready dataset."""

    GOOD_QUALITY = "good_quality"
    """70-89: Good quality, minor warnings."""

    NEEDS_IMPROVEMENT = "needs_improvement"
    """50-69: Usable but needs improvement."""

    CRITICAL_ISSUES = "critical_issues"
    """0-49: Critical issues, fix before training."""

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            ScoreRange.PRODUCTION_READY: "Production Ready",
            ScoreRange.GOOD_QUALITY: "Good Quality",
            ScoreRange.NEEDS_IMPROVEMENT: "Needs Improvement",
            ScoreRange.CRITICAL_ISSUES: "Critical Issues",
        }
        return names[self]

    @property
    def description(self) -> str:
        """Description of what this range means."""
        descriptions = {
            ScoreRange.PRODUCTION_READY: "Dataset is ready for production training.",
            ScoreRange.GOOD_QUALITY: "Good quality with minor warnings to review.",
            ScoreRange.NEEDS_IMPROVEMENT: "Usable but should be improved before training.",
            ScoreRange.CRITICAL_ISSUES: "Critical issues must be fixed before training.",
        }
        return descriptions[self]

    @classmethod
    def from_score(cls, score: int) -> ScoreRange:
        """Get the range for a given score."""
        if score >= 90:
            return cls.PRODUCTION_READY
        elif score >= 70:
            return cls.GOOD_QUALITY
        elif score >= 50:
            return cls.NEEDS_IMPROVEMENT
        else:
            return cls.CRITICAL_ISSUES


@dataclass(frozen=True)
class QualityScore:
    """Quality score result with breakdown."""

    score: int
    """Overall quality score (0-100)."""

    range: ScoreRange
    """Score range classification."""

    sample_count: int
    """Number of samples in dataset."""

    error_count: int
    """Number of errors detected."""

    warning_count: int
    """Number of warnings detected."""

    avg_length: int
    """Average sample length in characters."""

    breakdown: dict[str, int]
    """Breakdown of score adjustments."""

    @property
    def is_production_ready(self) -> bool:
        """Whether the dataset is production-ready."""
        return self.range == ScoreRange.PRODUCTION_READY

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Score: {self.score}/100 ({self.range.display_name}) - "
            f"{self.sample_count} samples, {self.error_count} errors, "
            f"{self.warning_count} warnings"
        )


class DatasetQualityScorer:
    """Calculates dataset quality scores based on validation results."""

    def __init__(
        self,
        error_penalty: int = 20,
        warning_penalty: int = 5,
        few_samples_threshold: int = 50,
        few_samples_penalty: int = 15,
        many_samples_threshold: int = 100,
        many_samples_bonus: int = 10,
        short_length_threshold: int = 50,
        short_length_penalty: int = 10,
        good_length_threshold: int = 100,
        good_length_bonus: int = 5,
    ):
        """Initialize scorer with configurable thresholds.

        Args:
            error_penalty: Points deducted per error.
            warning_penalty: Points deducted per warning.
            few_samples_threshold: Below this count triggers penalty.
            few_samples_penalty: Points deducted for few samples.
            many_samples_threshold: At or above this count triggers bonus.
            many_samples_bonus: Points added for many samples.
            short_length_threshold: Below this avg length triggers penalty.
            short_length_penalty: Points deducted for short avg length.
            good_length_threshold: At or above this avg length triggers bonus.
            good_length_bonus: Points added for good avg length.
        """
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty
        self.few_samples_threshold = few_samples_threshold
        self.few_samples_penalty = few_samples_penalty
        self.many_samples_threshold = many_samples_threshold
        self.many_samples_bonus = many_samples_bonus
        self.short_length_threshold = short_length_threshold
        self.short_length_penalty = short_length_penalty
        self.good_length_threshold = good_length_threshold
        self.good_length_bonus = good_length_bonus

    def calculate_score(
        self,
        sample_count: int,
        error_count: int,
        warning_count: int,
        avg_length: int,
    ) -> QualityScore:
        """Calculates quality score for a dataset validation result.

        Args:
            sample_count: Total number of samples in dataset.
            error_count: Number of validation errors detected.
            warning_count: Number of validation warnings detected.
            avg_length: Average sample length in characters.

        Returns:
            QualityScore with overall score and breakdown.
        """
        score = 100
        breakdown: dict[str, int] = {"base": 100}

        # Deduct for errors (severe)
        if error_count > 0:
            error_deduction = error_count * self.error_penalty
            score -= error_deduction
            breakdown["errors"] = -error_deduction

        # Deduct for warnings (moderate)
        if warning_count > 0:
            warning_deduction = warning_count * self.warning_penalty
            score -= warning_deduction
            breakdown["warnings"] = -warning_deduction

        # Sample count adjustments
        if sample_count >= self.many_samples_threshold:
            score += self.many_samples_bonus
            breakdown["many_samples"] = self.many_samples_bonus
        elif sample_count < self.few_samples_threshold:
            score -= self.few_samples_penalty
            breakdown["few_samples"] = -self.few_samples_penalty

        # Average length adjustments
        if avg_length >= self.good_length_threshold:
            score += self.good_length_bonus
            breakdown["good_length"] = self.good_length_bonus
        elif avg_length < self.short_length_threshold:
            score -= self.short_length_penalty
            breakdown["short_length"] = -self.short_length_penalty

        # Clamp to 0-100
        final_score = max(0, min(100, score))

        return QualityScore(
            score=final_score,
            range=ScoreRange.from_score(final_score),
            sample_count=sample_count,
            error_count=error_count,
            warning_count=warning_count,
            avg_length=avg_length,
            breakdown=breakdown,
        )

    @classmethod
    def default(cls) -> DatasetQualityScorer:
        """Create a scorer with default settings."""
        return cls()
