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

# ScoreRange enum removed - the raw score (0-100) IS the measurement.
# Classifications like "production ready" destroy information.
# A score of 89 and 90 are nearly identical, but an enum pretends they're different.


@dataclass(frozen=True)
class QualityScore:
    """Quality score result with breakdown.

    The score (0-100) IS the quality measurement. No classification.
    """

    score: int
    """Overall quality score (0-100). This IS the quality signal."""

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
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Score: {self.score}/100 - "
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
