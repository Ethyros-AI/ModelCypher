"""Dataset validation models.

Types for validation results, errors, warnings, and statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(str, Enum):
    """Overall validation status."""

    VALID = "valid"
    """Dataset is valid."""

    WARNINGS = "warnings"
    """Dataset has warnings but is usable."""

    ERRORS = "errors"
    """Dataset has errors and cannot be used."""


class DatasetContentFormat(str, Enum):
    """Detected content format of a dataset."""

    TEXT = "text"
    """Plain text format (single 'text' field)."""

    CHAT = "chat"
    """Chat/conversation format (messages array with role/content)."""

    TOOLS = "tools"
    """Tool-augmented chat (messages + tools array)."""

    INSTRUCTION = "instruction"
    """Instruction format (instruction + output fields)."""

    COMPLETION = "completion"
    """Completion format (prompt + completion fields)."""

    UNKNOWN = "unknown"
    """Could not determine format."""


class ValidationErrorKind(str, Enum):
    """Kind of validation error."""

    NOT_VALID_JSON = "not_valid_json"
    """Line is not valid JSON."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    """Required field is missing."""

    EMPTY_CONTENT = "empty_content"
    """Content field is empty."""

    INVALID_MESSAGES_STRUCTURE = "invalid_messages_structure"
    """Messages array has invalid structure."""

    INVALID_ROLE = "invalid_role"
    """Message has invalid role."""

    EMPTY_MESSAGE_CONTENT = "empty_message_content"
    """Message content is empty."""

    DUPLICATE_SYSTEM_MESSAGE = "duplicate_system_message"
    """Multiple system messages found."""

    MISSING_USER_MESSAGE = "missing_user_message"
    """No user message in conversation."""

    MISSING_ASSISTANT_MESSAGE = "missing_assistant_message"
    """No assistant message in conversation."""

    INVALID_MESSAGE_ORDER = "invalid_message_order"
    """Messages are not in valid order."""

    TOOL_CALL_WITHOUT_RESPONSE = "tool_call_without_response"
    """Tool call has no corresponding response."""

    ORPHAN_TOOL_RESPONSE = "orphan_tool_response"
    """Tool response without matching call."""

    FILE_NOT_FOUND = "file_not_found"
    """Dataset file not found."""

    IO_ERROR = "io_error"
    """I/O error reading file."""


@dataclass(frozen=True)
class ValidationError:
    """A validation error with location context."""

    kind: ValidationErrorKind
    """Kind of error."""

    message: str
    """Human-readable error message."""

    line_number: int | None = None
    """Line number where error occurred (1-based)."""

    field_name: str | None = None
    """Field name related to error."""

    sample_index: int | None = None
    """Sample index (0-based) if applicable."""

    def __str__(self) -> str:
        """Format error for display."""
        parts = [self.message]
        if self.line_number is not None:
            parts.append(f"line {self.line_number}")
        if self.field_name is not None:
            parts.append(f"field '{self.field_name}'")
        return " - ".join(parts)


class ValidationWarningKind(str, Enum):
    """Kind of validation warning."""

    LONG_CONTENT = "long_content"
    """Content exceeds recommended length."""

    SHORT_CONTENT = "short_content"
    """Content is very short."""

    INCONSISTENT_FORMAT = "inconsistent_format"
    """Format varies across samples."""

    MISSING_SYSTEM_PROMPT = "missing_system_prompt"
    """Chat lacks system prompt (optional but recommended)."""

    UNUSUAL_ROLE_PATTERN = "unusual_role_pattern"
    """Unusual message role pattern."""

    HIGH_DUPLICATE_RATIO = "high_duplicate_ratio"
    """Many duplicate samples detected."""

    POTENTIAL_PII = "potential_pii"
    """Potential personally identifiable information."""

    INTRINSIC_IDENTITY_INSTRUCTION = "intrinsic_identity_instruction"
    """Contains identity instructions (you are, act as)."""

    INTRINSIC_IDENTITY_ROLEPLAY = "intrinsic_identity_roleplay"
    """Contains roleplay instructions."""

    INTRINSIC_IDENTITY_PERSONA = "intrinsic_identity_persona"
    """Contains persona definition."""

    MIXED_LANGUAGES = "mixed_languages"
    """Multiple languages detected."""

    ENCODING_ISSUES = "encoding_issues"
    """Potential encoding problems."""


@dataclass(frozen=True)
class ValidationWarning:
    """A validation warning with location context."""

    kind: ValidationWarningKind
    """Kind of warning."""

    message: str
    """Human-readable warning message."""

    line_number: int | None = None
    """Line number where warning occurred (1-based)."""

    field_name: str | None = None
    """Field name related to warning."""

    sample_index: int | None = None
    """Sample index (0-based) if applicable."""

    def __str__(self) -> str:
        """Format warning for display."""
        parts = [self.message]
        if self.line_number is not None:
            parts.append(f"line {self.line_number}")
        if self.field_name is not None:
            parts.append(f"field '{self.field_name}'")
        return " - ".join(parts)


@dataclass(frozen=True)
class DatasetStats:
    """Statistics about a validated dataset."""

    total_samples: int
    """Total number of samples."""

    valid_samples: int
    """Number of valid samples."""

    total_tokens: int
    """Estimated total tokens."""

    average_tokens_per_sample: float
    """Average tokens per sample."""

    min_tokens: int
    """Minimum tokens in a sample."""

    max_tokens: int
    """Maximum tokens in a sample."""

    unique_samples: int
    """Number of unique samples (deduped count)."""

    @property
    def duplicate_count(self) -> int:
        """Number of duplicate samples."""
        return self.total_samples - self.unique_samples

    @property
    def duplicate_ratio(self) -> float:
        """Ratio of duplicates to total (0.0-1.0)."""
        if self.total_samples == 0:
            return 0.0
        return self.duplicate_count / self.total_samples


@dataclass(frozen=True)
class QuickValidationResult:
    """Result from quick validation (first N samples)."""

    status: ValidationStatus
    """Overall status."""

    detected_format: DatasetContentFormat
    """Detected content format."""

    samples_checked: int
    """Number of samples checked."""

    errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    """Validation errors found."""

    warnings: tuple[ValidationWarning, ...] = field(default_factory=tuple)
    """Validation warnings found."""

    @property
    def is_valid(self) -> bool:
        """Whether dataset passed validation."""
        return self.status != ValidationStatus.ERRORS

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)


@dataclass(frozen=True)
class ValidationResult:
    """Complete validation result."""

    status: ValidationStatus
    """Overall status."""

    detected_format: DatasetContentFormat
    """Detected content format."""

    stats: DatasetStats
    """Dataset statistics."""

    errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    """Validation errors found."""

    warnings: tuple[ValidationWarning, ...] = field(default_factory=tuple)
    """Validation warnings found."""

    file_path: str | None = None
    """Path to validated file."""

    @property
    def is_valid(self) -> bool:
        """Whether dataset passed validation."""
        return self.status != ValidationStatus.ERRORS

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validation: {self.status.value}",
            f"Format: {self.detected_format.value}",
            f"Samples: {self.stats.total_samples} ({self.stats.valid_samples} valid)",
            f"Tokens: {self.stats.total_tokens} total, {self.stats.average_tokens_per_sample:.1f} avg",
        ]
        if self.stats.duplicate_count > 0:
            lines.append(f"Duplicates: {self.stats.duplicate_count}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


@dataclass
class DatasetValidationProgress:
    """Progress during validation."""

    samples_processed: int = 0
    """Samples processed so far."""

    total_samples: int | None = None
    """Total samples if known."""

    current_phase: str = "initializing"
    """Current validation phase."""

    @property
    def progress_fraction(self) -> float | None:
        """Progress as fraction (0.0-1.0) if total known."""
        if self.total_samples is None or self.total_samples == 0:
            return None
        return min(1.0, self.samples_processed / self.total_samples)

    @property
    def progress_percent(self) -> int | None:
        """Progress as percentage if total known."""
        fraction = self.progress_fraction
        if fraction is None:
            return None
        return int(fraction * 100)


@dataclass(frozen=True)
class ValidatedTextDataset:
    """A validated text dataset ready for training."""

    file_path: str
    """Path to dataset file."""

    format: DatasetContentFormat
    """Content format."""

    stats: DatasetStats
    """Dataset statistics."""

    validation_result: ValidationResult
    """Full validation result."""

    @property
    def is_ready_for_training(self) -> bool:
        """Whether dataset can be used for training."""
        return self.validation_result.is_valid


class DatasetPreparationErrorKind(str, Enum):
    """Kind of dataset preparation error."""

    VALIDATION_FAILED = "validation_failed"
    """Dataset validation failed."""

    FORMAT_CONVERSION_FAILED = "format_conversion_failed"
    """Could not convert to target format."""

    TOKENIZATION_FAILED = "tokenization_failed"
    """Tokenization failed."""

    INSUFFICIENT_SAMPLES = "insufficient_samples"
    """Not enough valid samples."""

    IO_ERROR = "io_error"
    """I/O error during preparation."""


@dataclass(frozen=True)
class DatasetPreparationError:
    """Error during dataset preparation."""

    kind: DatasetPreparationErrorKind
    """Kind of error."""

    message: str
    """Human-readable message."""

    validation_errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    """Underlying validation errors if applicable."""
