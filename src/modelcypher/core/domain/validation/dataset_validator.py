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

"""Dataset validator base implementation.

Provides validation with caching for efficiency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from modelcypher.core.domain.validation.dataset_format_analyzer import (
    DatasetFormatAnalyzer,
)
from modelcypher.core.domain.validation.dataset_validation_models import (
    DatasetContentFormat,
    DatasetStats,
    DatasetValidationProgress,
    QuickValidationResult,
    ValidationError,
    ValidationErrorKind,
    ValidationResult,
    ValidationStatus,
    ValidationWarning,
    ValidationWarningKind,
)

logger = logging.getLogger(__name__)


# Progress callback type
ProgressCallback = Callable[[DatasetValidationProgress], None]


@dataclass
class ValidationCache:
    """Cache entry for validation results."""

    file_signature: str
    """Hash of file path + size + mtime."""

    result: ValidationResult
    """Cached validation result."""


class DatasetValidator:
    """Dataset validator with caching.

    Validates JSONL datasets for structure, format, and content.
    """

    # Default limits
    QUICK_VALIDATION_SAMPLES = 100
    MAX_LINE_LENGTH = 10_000_000  # 10MB per line

    def __init__(self):
        """Initialize validator."""
        self._format_analyzer = DatasetFormatAnalyzer()
        self._cache: dict[str, ValidationCache] = {}

    def _file_signature(self, path: Path) -> str:
        """Generate signature for cache invalidation.

        Args:
            path: File path.

        Returns:
            Hash string.
        """
        try:
            stat = path.stat()
            data = f"{path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(data.encode()).hexdigest()
        except OSError:
            return ""

    def _check_cache(self, path: Path) -> ValidationResult | None:
        """Check cache for valid result.

        Args:
            path: File path.

        Returns:
            Cached result if valid, None otherwise.
        """
        key = str(path)
        if key not in self._cache:
            return None

        entry = self._cache[key]
        current_sig = self._file_signature(path)

        if entry.file_signature == current_sig:
            logger.debug(f"Cache hit for {path}")
            return entry.result

        # Invalidate stale entry
        del self._cache[key]
        return None

    def _update_cache(self, path: Path, result: ValidationResult) -> None:
        """Update cache with result.

        Args:
            path: File path.
            result: Validation result.
        """
        key = str(path)
        sig = self._file_signature(path)
        if sig:
            self._cache[key] = ValidationCache(file_signature=sig, result=result)

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._cache.clear()

    def _iterate_samples(
        self, path: Path, limit: int | None = None
    ) -> Iterator[tuple[int, dict[str, Any], ValidationError | None]]:
        """Iterate over samples in a JSONL file.

        Args:
            path: Path to JSONL file.
            limit: Maximum samples to yield.

        Yields:
            Tuples of (line_number, parsed_sample, error_if_any).
        """
        count = 0
        line_number = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_number += 1
                    line = line.strip()

                    if not line:
                        continue

                    if len(line) > self.MAX_LINE_LENGTH:
                        yield (
                            line_number,
                            {},
                            ValidationError(
                                kind=ValidationErrorKind.IO_ERROR,
                                message=f"Line exceeds maximum length ({len(line)} > {self.MAX_LINE_LENGTH})",
                                line_number=line_number,
                            ),
                        )
                        continue

                    try:
                        sample = json.loads(line)
                        yield (line_number, sample, None)
                        count += 1

                        if limit is not None and count >= limit:
                            break

                    except json.JSONDecodeError as e:
                        yield (
                            line_number,
                            {},
                            ValidationError(
                                kind=ValidationErrorKind.NOT_VALID_JSON,
                                message=f"Invalid JSON: {e}",
                                line_number=line_number,
                            ),
                        )

        except FileNotFoundError:
            yield (
                0,
                {},
                ValidationError(
                    kind=ValidationErrorKind.FILE_NOT_FOUND,
                    message=f"File not found: {path}",
                ),
            )
        except OSError as e:
            yield (
                0,
                {},
                ValidationError(
                    kind=ValidationErrorKind.IO_ERROR,
                    message=f"I/O error: {e}",
                ),
            )

    def validate_quick(
        self,
        path: Path,
        sample_limit: int = QUICK_VALIDATION_SAMPLES,
    ) -> QuickValidationResult:
        """Quick validation of first N samples.

        Args:
            path: Path to JSONL file.
            sample_limit: Maximum samples to check.

        Returns:
            Quick validation result.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []
        samples_checked = 0
        format_votes: dict[DatasetContentFormat, int] = {}

        for line_number, sample, parse_error in self._iterate_samples(path, limit=sample_limit):
            if parse_error:
                errors.append(parse_error)
                continue

            samples_checked += 1

            # Analyze format
            result = self._format_analyzer.analyze(
                sample, line_number=line_number, sample_index=samples_checked - 1
            )

            # Vote for format
            format_votes[result.format] = format_votes.get(result.format, 0) + 1

            errors.extend(result.errors)
            warnings.extend(result.warnings)

        # Determine format
        detected_format = DatasetContentFormat.UNKNOWN
        if format_votes:
            detected_format = max(format_votes, key=lambda k: format_votes[k])

        # Determine status
        if errors:
            status = ValidationStatus.ERRORS
        elif warnings:
            status = ValidationStatus.WARNINGS
        else:
            status = ValidationStatus.VALID

        return QuickValidationResult(
            status=status,
            detected_format=detected_format,
            samples_checked=samples_checked,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def validate_full(
        self,
        path: Path,
        progress_callback: ProgressCallback | None = None,
        use_cache: bool = True,
    ) -> ValidationResult:
        """Full validation of all samples.

        Args:
            path: Path to JSONL file.
            progress_callback: Optional progress callback.
            use_cache: Whether to use cache.

        Returns:
            Full validation result.
        """
        # Check cache
        if use_cache:
            cached = self._check_cache(path)
            if cached is not None:
                return cached

        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []
        total_samples = 0
        valid_samples = 0
        format_votes: dict[DatasetContentFormat, int] = {}

        # Token estimation (rough)
        total_tokens = 0
        token_counts: list[int] = []

        # Deduplication
        seen_hashes: set[str] = set()
        unique_count = 0

        progress = DatasetValidationProgress(current_phase="validating")

        for line_number, sample, parse_error in self._iterate_samples(path):
            if parse_error:
                errors.append(parse_error)
                total_samples += 1
                continue

            total_samples += 1

            # Report progress
            if progress_callback and total_samples % 100 == 0:
                progress.samples_processed = total_samples
                progress_callback(progress)

            # Analyze format
            result = self._format_analyzer.analyze(
                sample, line_number=line_number, sample_index=total_samples - 1
            )

            format_votes[result.format] = format_votes.get(result.format, 0) + 1
            errors.extend(result.errors)
            warnings.extend(result.warnings)

            if not result.errors:
                valid_samples += 1

            # Estimate tokens (very rough: 4 chars per token)
            sample_str = json.dumps(sample)
            token_estimate = len(sample_str) // 4
            total_tokens += token_estimate
            token_counts.append(token_estimate)

            # Check for duplicates
            sample_hash = hashlib.md5(sample_str.encode()).hexdigest()
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_count += 1

        # Check for format inconsistency
        if len(format_votes) > 1:
            total_votes = sum(format_votes.values())
            majority_format = max(format_votes, key=lambda k: format_votes[k])
            majority_count = format_votes[majority_format]
            if majority_count < total_votes * 0.9:
                warnings.append(
                    ValidationWarning(
                        kind=ValidationWarningKind.INCONSISTENT_FORMAT,
                        message=f"Mixed formats detected: {dict(format_votes)}",
                    )
                )

        # Check duplicate ratio
        if total_samples > 0:
            dup_ratio = 1.0 - (unique_count / total_samples)
            if dup_ratio > 0.1:
                warnings.append(
                    ValidationWarning(
                        kind=ValidationWarningKind.HIGH_DUPLICATE_RATIO,
                        message=f"{int(dup_ratio * 100)}% duplicates detected",
                    )
                )

        # Determine format
        detected_format = DatasetContentFormat.UNKNOWN
        if format_votes:
            detected_format = max(format_votes, key=lambda k: format_votes[k])

        # Compute stats
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        avg_tokens = total_tokens / total_samples if total_samples > 0 else 0.0

        stats = DatasetStats(
            total_samples=total_samples,
            valid_samples=valid_samples,
            total_tokens=total_tokens,
            average_tokens_per_sample=avg_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            unique_samples=unique_count,
        )

        # Determine status
        if errors:
            status = ValidationStatus.ERRORS
        elif warnings:
            status = ValidationStatus.WARNINGS
        else:
            status = ValidationStatus.VALID

        result = ValidationResult(
            status=status,
            detected_format=detected_format,
            stats=stats,
            errors=tuple(errors),
            warnings=tuple(warnings),
            file_path=str(path),
        )

        # Update cache
        if use_cache:
            self._update_cache(path, result)

        return result
