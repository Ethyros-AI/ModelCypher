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

"""Output safety guard for filtering streaming model output.

Filters streaming model output for unsafe content using a sliding window
approach to catch cross-token patterns.

Architecture:
    TokenizerChatSession -> InferenceViewModel -> OutputSafetyGuard -> InferenceView
                                                       |
                                                StreamingTokenBuffer
                                                (sliding window derived from filter)

Design Decisions:
- Uses sliding window to catch cross-token patterns ("ki" + "ll" = "kill")
- Reuses existing RegexContentFilter for pattern matching
- Shows "[...]" placeholder instead of hard-stopping generation
- Developer bypass requires explicit opt-in
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.safety.output_safety_result import (
    DEFAULT_FILTERED_PLACEHOLDER,
    OutputSafetyResult,
)
from modelcypher.core.domain.safety.regex_content_filter import (
    ContentFilterResult,
    DatasetPurpose,
    RegexContentFilter,
    SafetyCategory,
)
from modelcypher.core.domain.safety.safety_audit_log import SafetyAuditLog
from modelcypher.core.domain.safety.streaming_token_buffer import StreamingTokenBuffer

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

    is_enabled: bool = True
    """Whether filtering is enabled."""

    max_consecutive_violations: int = 3
    """Maximum consecutive violations before truncation."""

    filtered_placeholder: str = DEFAULT_FILTERED_PLACEHOLDER
    """Placeholder text for filtered content."""

    audit_logging_enabled: bool = True
    """Whether to log filtered content for auditing."""

    filter: RegexContentFilter = field(default_factory=RegexContentFilter.default)
    """Content filter for pattern matching."""

    window_size: int | None = None
    """Buffer window size (derived when not provided)."""

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
        if self.max_consecutive_violations < 1:
            self.max_consecutive_violations = 1
        if self.window_size is None:
            self.window_size = self._derive_window_size()
        self._buffer = StreamingTokenBuffer(window_size=self.window_size)
        self._audit_log = SafetyAuditLog()

    def _derive_window_size(self) -> int:
        """Derive buffer window size from filter rule lengths."""
        rule_lengths = [len(rule.expression.pattern) for rule in self.filter.rules]
        if rule_lengths:
            backend = get_default_backend()
            lengths_arr = backend.array(rule_lengths)
            max_len_arr = backend.max(lengths_arr)
            backend.eval(max_len_arr)
            derived = int(backend.to_scalar(max_len_arr))
        else:
            derived = len(self.filtered_placeholder)

        if derived <= 0:
            raise ValueError("Derived window_size must be positive")
        return derived

    def process(self, token: str) -> OutputSafetyResult:
        """Process a streaming token through the safety filter.

        Args:
            token: The token to process.

        Returns:
            Safety result indicating safe, filtered, or truncated.
        """
        # Bypass if disabled
        if not self.is_enabled:
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
        violation: ContentFilterResult,
    ) -> OutputSafetyResult:
        """Handle a content filter violation.

        Args:
            token: The token that was being processed.
            violation: The filter result indicating the violation.

        Returns:
            Filtered or truncated result.
        """

        self._total_violations += 1
        self._consecutive_violations += 1

        category = violation.category or SafetyCategory.DANGEROUS_CODE

        # Log the event (privacy-preserving)
        if self.audit_logging_enabled:
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
        if self._consecutive_violations >= self.max_consecutive_violations:
            self._is_truncated = True

            if self.audit_logging_enabled:
                self._audit_log.log_truncation_event(
                    reason="Consecutive violations exceeded threshold",
                    total_violations=self._total_violations,
                )

            return OutputSafetyResult.truncated("Content safety limit reached")

        # Return filtered placeholder
        return OutputSafetyResult.filtered(
            replacement=self.filtered_placeholder,
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
        return cls(is_enabled=False, audit_logging_enabled=False)
