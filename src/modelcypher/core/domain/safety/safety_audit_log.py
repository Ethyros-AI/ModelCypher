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

"""Privacy-preserving audit log for output safety filtering events.

Logs filter activations without storing the actual content that was filtered.
This enables monitoring filter effectiveness while protecting user privacy.

Privacy Design:
- No raw content is logged
- Only category, rule ID, and approximate context size are recorded
- Session IDs are ephemeral and not tied to user identity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from modelcypher.core.domain.safety.safety_models import SafetyCategory

logger = logging.getLogger(__name__)


@dataclass
class SafetyAuditLog:
    """Privacy-preserving audit log for output safety filtering events.

    Logs filter activations without storing the actual content that was filtered.
    This enables monitoring filter effectiveness while protecting user privacy.

    This is a mutable class - callers must manage synchronization if used
    concurrently.
    """

    _session_id: str = field(default_factory=lambda: uuid4().hex[:8], init=False)
    """Ephemeral session identifier for correlating events."""

    _event_count: int = field(default=0, init=False)
    """Number of filter events in this session."""

    _category_counts: dict[str, int] = field(default_factory=dict, init=False)
    """Summary of categories triggered in this session."""

    def log_filter_event(
        self,
        category: SafetyCategory,
        rule_id: str,
        context_length: int,
    ) -> None:
        """Log a filter activation event.

        Args:
            category: The safety category that triggered.
            rule_id: Identifier of the triggered rule.
            context_length: Approximate length of the context window.
        """
        self._event_count += 1
        category_key = category.value if hasattr(category, "value") else str(category)
        self._category_counts[category_key] = (
            self._category_counts.get(category_key, 0) + 1
        )

        # Log with privacy annotations - no raw content
        logger.info(
            "[SafetyFilter] session=%s event=%d category=%s rule=%s context_len=%d",
            self._session_id,
            self._event_count,
            category_key,
            rule_id,
            context_length,
        )

    def log_truncation_event(
        self,
        reason: str,
        total_violations: int,
    ) -> None:
        """Log a truncation event (generation stopped due to repeated violations).

        Args:
            reason: The reason for truncation.
            total_violations: Number of violations before truncation.
        """
        logger.warning(
            "[SafetyFilter] session=%s action=truncated reason=%s violations=%d",
            self._session_id,
            reason,
            total_violations,
        )

    def log_session_summary(self) -> None:
        """Log session summary when inference completes."""
        if self._event_count == 0:
            return

        category_breakdown = ", ".join(
            f"{cat}:{count}"
            for cat, count in sorted(
                self._category_counts.items(), key=lambda x: x[1], reverse=True
            )
        )

        logger.info(
            "[SafetyFilter] session=%s summary total=%d categories=[%s]",
            self._session_id,
            self._event_count,
            category_breakdown,
        )

    def reset(self) -> None:
        """Reset the audit log for a new inference session."""
        self._event_count = 0
        self._category_counts.clear()

    @property
    def total_events(self) -> int:
        """Total filter events in this session."""
        return self._event_count

    @property
    def category_breakdown(self) -> dict[str, int]:
        """Categories triggered and their counts."""
        return dict(self._category_counts)

    @property
    def session_id(self) -> str:
        """Ephemeral session identifier."""
        return self._session_id
