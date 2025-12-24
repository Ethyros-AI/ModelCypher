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

"""Result types for output safety filtering.

Defines the result of safety filtering on streaming model output,
along with configuration options for filtering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.domain.safety.safety_models import SafetyCategory


class OutputSafetyResultType(str, Enum):
    """Type of output safety result."""

    SAFE = "safe"
    """Content is safe to display."""

    FILTERED = "filtered"
    """Content was filtered due to safety violation."""

    TRUNCATED = "truncated"
    """Content was truncated due to accumulated violations."""


DEFAULT_FILTERED_PLACEHOLDER = "[...]"
"""Default placeholder for filtered content."""


@dataclass(frozen=True)
class OutputSafetyResult:
    """Result of safety filtering on streaming model output."""

    result_type: OutputSafetyResultType
    """Type of result."""

    token: str | None = None
    """Original token (for safe results)."""

    replacement: str | None = None
    """Replacement text (for filtered results)."""

    category: SafetyCategory | None = None
    """Safety category that triggered filtering (for filtered results)."""

    rule_id: str | None = None
    """Identifier of the triggered rule (for filtered results)."""

    reason: str | None = None
    """Reason for truncation (for truncated results)."""

    @classmethod
    def safe(cls, token: str) -> OutputSafetyResult:
        """Create a safe result."""
        return cls(result_type=OutputSafetyResultType.SAFE, token=token)

    @classmethod
    def filtered(
        cls,
        replacement: str,
        category: SafetyCategory,
        rule_id: str,
    ) -> OutputSafetyResult:
        """Create a filtered result."""
        return cls(
            result_type=OutputSafetyResultType.FILTERED,
            replacement=replacement,
            category=category,
            rule_id=rule_id,
        )

    @classmethod
    def truncated(cls, reason: str) -> OutputSafetyResult:
        """Create a truncated result."""
        return cls(result_type=OutputSafetyResultType.TRUNCATED, reason=reason)

    @property
    def is_safe(self) -> bool:
        """Whether the original content passed safety checks."""
        return self.result_type == OutputSafetyResultType.SAFE

    @property
    def was_filtered(self) -> bool:
        """Whether this result should increment the filter count."""
        return self.result_type in (
            OutputSafetyResultType.FILTERED,
            OutputSafetyResultType.TRUNCATED,
        )

    @property
    def display_text(self) -> str:
        """The text to display to the user."""
        if self.result_type == OutputSafetyResultType.SAFE:
            return self.token or ""
        if self.result_type == OutputSafetyResultType.FILTERED:
            return self.replacement or DEFAULT_FILTERED_PLACEHOLDER
        if self.result_type == OutputSafetyResultType.TRUNCATED:
            return f"\n[Generation stopped: {self.reason or 'safety limit reached'}]"
        return ""


@dataclass(frozen=True)
class OutputSafetyConfiguration:
    """Configuration for output safety filtering."""

    is_enabled: bool = True
    """Whether filtering is enabled."""

    max_consecutive_violations: int = 3
    """Maximum consecutive violations before truncation."""

    filtered_placeholder: str = DEFAULT_FILTERED_PLACEHOLDER
    """Placeholder text for filtered content."""

    audit_logging_enabled: bool = True
    """Whether to log filtered content for auditing."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_consecutive_violations < 1:
            object.__setattr__(self, "max_consecutive_violations", 1)

    @classmethod
    def default(cls) -> OutputSafetyConfiguration:
        """Default production configuration."""
        return cls()

    @classmethod
    def disabled(cls) -> OutputSafetyConfiguration:
        """Configuration with filtering disabled (for developer bypass)."""
        return cls(is_enabled=False, audit_logging_enabled=False)
