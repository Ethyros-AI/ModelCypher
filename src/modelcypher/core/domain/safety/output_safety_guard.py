"""Output safety guard for filtering streaming model output.

Filters streaming model output for unsafe content using a sliding window
approach to catch cross-token patterns.

Architecture:
    TokenizerChatSession -> InferenceViewModel -> OutputSafetyGuard -> InferenceView
                                                       |
                                                StreamingTokenBuffer
                                                (sliding window 200 chars)

Design Decisions:
- Uses sliding window to catch cross-token patterns ("ki" + "ll" = "kill")
- Reuses existing RegexContentFilter for pattern matching
- Shows "[...]" placeholder instead of hard-stopping generation
- Developer bypass requires explicit opt-in
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain.safety.output_safety_result import (
    OutputSafetyConfiguration,
    OutputSafetyResult,
)
from modelcypher.core.domain.safety.regex_content_filter import (
    DatasetPurpose,
    RegexContentFilter,
    SafetyCategory,
)
from modelcypher.core.domain.safety.safety_audit_log import SafetyAuditLog
from modelcypher.core.domain.safety.streaming_token_buffer import (
    DEFAULT_WINDOW_SIZE,
    StreamingTokenBuffer,
)

logger = logging.getLogger(__name__)


@dataclass
class OutputSafetyGuard:
    """Guard that filters streaming model output for unsafe content.

    This guard processes tokens one at a time, maintaining a sliding window
    buffer to detect patterns that span multiple tokens. When unsafe content
    is detected, it returns a filtered result with a placeholder.

    This is a mutable class - callers must manage synchronization if used
    concurrently.
    """

    configuration: OutputSafetyConfiguration = field(
        default_factory=OutputSafetyConfiguration.default
    )
    """Configuration for filtering behavior."""

    filter: RegexContentFilter = field(default_factory=RegexContentFilter.default)
    """Content filter for pattern matching."""

    window_size: int = DEFAULT_WINDOW_SIZE
    """Buffer window size."""

    _buffer: StreamingTokenBuffer = field(init=False)
    """Sliding window buffer for cross-token detection."""

    _audit_log: SafetyAuditLog = field(init=False)
    """Audit log for privacy-preserving event logging."""

    _consecutive_violations: int = field(default=0, init=False)
    """Consecutive violations counter (for truncation logic)."""

    _total_violations: int = field(default=0, init=False)
    """Total violations in current session."""

    _is_truncated: bool = field(default=False, init=False)
    """Whether the output has been truncated."""

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._buffer = StreamingTokenBuffer(window_size=self.window_size)
        self._audit_log = SafetyAuditLog()

    def process(self, token: str) -> OutputSafetyResult:
        """Process a streaming token through the safety filter.

        Args:
            token: The token to process.

        Returns:
            Safety result indicating safe, filtered, or truncated.
        """
        # Bypass if disabled
        if not self.configuration.is_enabled:
            return OutputSafetyResult.safe(token)

        # Already truncated - don't process more
        if self._is_truncated:
            return OutputSafetyResult.truncated("Maximum violations exceeded")

        # Append to sliding window
        window = self._buffer.append(token)

        # Check for violations using the full window
        violation = self.filter.check(
            window,
            purpose=DatasetPurpose.GENERAL,
            custom_whitelist=None,
        )

        if violation is not None:
            return self._handle_violation(token, violation)

        # Reset consecutive count on safe content
        self._consecutive_violations = 0
        return OutputSafetyResult.safe(token)

    def _handle_violation(
        self,
        token: str,
        violation: "ContentFilterResult",
    ) -> OutputSafetyResult:
        """Handle a content filter violation.

        Args:
            token: The token that was being processed.
            violation: The filter result indicating the violation.

        Returns:
            Filtered or truncated result.
        """
        from modelcypher.core.domain.safety.regex_content_filter import (
            ContentFilterResult,
        )

        self._total_violations += 1
        self._consecutive_violations += 1

        category = violation.category or SafetyCategory.DANGEROUS_CODE

        # Log the event (privacy-preserving)
        if self.configuration.audit_logging_enabled:
            self._audit_log.log_filter_event(
                category=category,
                rule_id=violation.rule_id,
                context_length=self._buffer.count,
            )

        logger.debug(
            "Filter triggered: category=%s rule=%s consecutive=%d",
            category.value,
            violation.rule_id,
            self._consecutive_violations,
        )

        # Check for truncation threshold
        if self._consecutive_violations >= self.configuration.max_consecutive_violations:
            self._is_truncated = True

            if self.configuration.audit_logging_enabled:
                self._audit_log.log_truncation_event(
                    reason="Consecutive violations exceeded threshold",
                    total_violations=self._total_violations,
                )

            return OutputSafetyResult.truncated("Content safety limit reached")

        # Return filtered placeholder
        return OutputSafetyResult.filtered(
            replacement=self.configuration.filtered_placeholder,
            category=category,
            rule_id=violation.rule_id,
        )

    def reset(self) -> None:
        """Reset the guard for a new inference session."""
        # Log summary of previous session
        self._audit_log.log_session_summary()
        self._audit_log.reset()

        self._buffer.reset()
        self._consecutive_violations = 0
        self._total_violations = 0
        self._is_truncated = False

    def configure(self, configuration: OutputSafetyConfiguration) -> None:
        """Update the configuration.

        Args:
            configuration: New configuration to apply.
        """
        self.configuration = configuration

    @property
    def is_enabled(self) -> bool:
        """Whether filtering is currently enabled."""
        return self.configuration.is_enabled

    @property
    def violation_count(self) -> int:
        """Total violations in the current session."""
        return self._total_violations

    @property
    def was_truncated(self) -> bool:
        """Whether the current generation was truncated."""
        return self._is_truncated

    @property
    def buffer_window(self) -> str:
        """Current content in the sliding window buffer."""
        return self._buffer.window

    @classmethod
    def disabled(cls) -> OutputSafetyGuard:
        """Create a guard with filtering disabled (for developer bypass)."""
        return cls(configuration=OutputSafetyConfiguration.disabled())
