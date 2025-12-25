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

"""Dataset safety scanner for validating training data.

Scans datasets for safety issues (PII, toxicity, prompt injection) during
validation. Uses RegexContentFilter for fast local pattern matching without
requiring external API calls.

Scanning Modes:
- Quick: Sample first N lines for rapid feedback (~1-5ms)
- Standard: Scan all samples with normal thresholds
- Full: Comprehensive scan with lower thresholds
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from modelcypher.core.domain.safety.regex_content_filter import (
    DatasetPurpose,
    RegexContentFilter,
    SafetyCategory,
    SafetyStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanConfig:
    """Configuration for dataset safety scanning."""

    max_samples: int | None = None
    """Maximum samples to scan (None = all)."""

    purpose: DatasetPurpose = DatasetPurpose.GENERAL
    """Purpose-based whitelist rules."""

    custom_whitelist: frozenset[str] = field(default_factory=frozenset)
    """Custom rule IDs to whitelist."""

    stop_on_first_block: bool = False
    """Whether to stop on first blocking issue."""

    max_findings: int = 50
    """Maximum findings to collect before stopping."""

    @classmethod
    def quick(cls) -> ScanConfig:
        """Quick scan - first 100 samples."""
        return cls(max_samples=100, max_findings=10)

    @classmethod
    def standard(cls) -> ScanConfig:
        """Standard scan - all samples."""
        return cls(max_findings=50)

    @classmethod
    def full(cls) -> ScanConfig:
        """Full scan - comprehensive with higher finding limit."""
        return cls(max_findings=100)

    def with_purpose(self, purpose: DatasetPurpose) -> ScanConfig:
        """Return a config with the specified purpose."""
        return ScanConfig(
            max_samples=self.max_samples,
            purpose=purpose,
            custom_whitelist=self.custom_whitelist,
            stop_on_first_block=self.stop_on_first_block,
            max_findings=self.max_findings,
        )


@dataclass(frozen=True)
class SafetyFinding:
    """Individual safety finding from a scan."""

    sample_index: int
    """Sample index where issue was found."""

    category: SafetyCategory | None
    """Category of the safety issue."""

    rule_id: str
    """Rule that triggered the finding."""

    reason: str
    """Human-readable description."""

    matched_text: str
    """Matched text snippet (sanitized)."""

    is_blocking: bool
    """Whether this finding should block training."""


@dataclass(frozen=True)
class ScanResult:
    """Result of a dataset safety scan."""

    samples_scanned: int
    """Total samples scanned."""

    samples_with_issues: int
    """Samples with safety issues detected."""

    findings: tuple[SafetyFinding, ...]
    """Individual findings with line context."""

    category_counts: dict[SafetyCategory, int]
    """Category breakdown of issues found."""

    has_blocking_issues: bool
    """Whether scan found issues that should block training."""

    safety_score: int
    """Overall safety score (0-100, higher = safer)."""

    @classmethod
    def clean(cls, samples_scanned: int) -> ScanResult:
        """Create an empty result for datasets with no issues."""
        return cls(
            samples_scanned=samples_scanned,
            samples_with_issues=0,
            findings=(),
            category_counts={},
            has_blocking_issues=False,
            safety_score=100,
        )


class DatasetSafetyScanner:
    """Scans datasets for safety issues during validation.

    Integrates with DatasetValidator to provide safety scanning during
    the validation pipeline. Uses RegexContentFilter for fast local pattern
    matching without requiring external API calls.
    """

    def __init__(self, filter: RegexContentFilter | None = None):
        """Create a scanner with the specified filter.

        Args:
            filter: Content filter to use. Defaults to standard regex filter.
        """
        self._filter = filter or RegexContentFilter.default()

    def scan(
        self,
        samples: list[str],
        config: ScanConfig | None = None,
    ) -> ScanResult:
        """Scan an array of text samples for safety issues.

        Args:
            samples: Text samples to scan.
            config: Scanning configuration. Defaults to standard.

        Returns:
            Scan result with findings.
        """
        config = config or ScanConfig.standard()
        samples_to_scan = samples[: config.max_samples] if config.max_samples else samples

        samples_scanned = 0
        findings: list[SafetyFinding] = []
        category_counts: dict[SafetyCategory, int] = {}
        samples_with_issues = 0
        has_blocking = False

        logger.info(
            "Starting safety scan: %d samples, purpose=%s",
            len(samples_to_scan),
            config.purpose.value,
        )

        for index, sample in enumerate(samples_to_scan):
            samples_scanned += 1
            result = self._filter.check(
                sample,
                purpose=config.purpose,
                custom_whitelist=set(config.custom_whitelist),
            )

            if result is not None:
                samples_with_issues += 1

                is_blocking = result.status == SafetyStatus.REJECTED
                if is_blocking:
                    has_blocking = True

                if result.category is not None:
                    category_counts[result.category] = category_counts.get(result.category, 0) + 1

                if len(findings) < config.max_findings:
                    sanitized_match = self._sanitize_match(result.matched_text)
                    findings.append(
                        SafetyFinding(
                            sample_index=index,
                            category=result.category,
                            rule_id=result.rule_id,
                            reason=result.reason,
                            matched_text=sanitized_match,
                            is_blocking=is_blocking,
                        )
                    )

                if config.stop_on_first_block and is_blocking:
                    logger.warning("Blocking safety issue found, stopping scan early")
                    break

        safety_score = self._calculate_safety_score(
            samples_scanned=samples_scanned,
            samples_with_issues=samples_with_issues,
            has_blocking=has_blocking,
        )

        logger.info(
            "Safety scan complete: %d/%d samples flagged, score=%d, blocking=%s",
            samples_with_issues,
            samples_scanned,
            safety_score,
            has_blocking,
        )

        return ScanResult(
            samples_scanned=samples_scanned,
            samples_with_issues=samples_with_issues,
            findings=tuple(findings),
            category_counts=category_counts,
            has_blocking_issues=has_blocking,
            safety_score=safety_score,
        )

    def scan_single(
        self,
        text: str,
        purpose: DatasetPurpose = DatasetPurpose.GENERAL,
        custom_whitelist: set[str] | None = None,
    ) -> SafetyFinding | None:
        """Scan a single sample for safety issues.

        Args:
            text: Text to scan.
            purpose: Dataset purpose for whitelist.
            custom_whitelist: Additional rule IDs to skip.

        Returns:
            Finding if issue detected, None otherwise.
        """
        result = self._filter.check(
            text,
            purpose=purpose,
            custom_whitelist=custom_whitelist or set(),
        )

        if result is None:
            return None

        return SafetyFinding(
            sample_index=0,
            category=result.category,
            rule_id=result.rule_id,
            reason=result.reason,
            matched_text=self._sanitize_match(result.matched_text),
            is_blocking=result.status == SafetyStatus.REJECTED,
        )

    def _sanitize_match(self, text: str) -> str:
        """Sanitize matched text by limiting length and redacting numbers."""
        trimmed = text[:50]
        return re.sub(r"\d{4,}", "****", trimmed)

    def _calculate_safety_score(
        self,
        samples_scanned: int,
        samples_with_issues: int,
        has_blocking: bool,
    ) -> int:
        """Calculate overall safety score."""
        if samples_scanned == 0:
            return 100

        # Base score from clean ratio
        clean_ratio = 1.0 - (samples_with_issues / samples_scanned)
        score = int(clean_ratio * 100)

        # Heavy penalty for blocking issues
        if has_blocking:
            score = min(score, 25)

        return max(0, min(100, score))
