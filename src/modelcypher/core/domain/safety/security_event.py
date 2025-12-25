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

"""Structured security events for auditability.

Provides a typed event system for logging security-relevant actions
in a structured, auditable format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Type of security event."""

    ADAPTER_EVALUATED = "adapterEvaluated"
    ACTIVATION_BLOCKED = "activationBlocked"
    RISK_OVERRIDE_RECORDED = "riskOverrideRecorded"
    CANARY_TRIPPED = "canaryTripped"


class SecuritySeverity(str, Enum):
    """Severity level for security events."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class SecurityEvent:
    """Structured security event for auditability.

    Events can be emitted to a logger in JSON format for audit trails.
    Supports both adapter-focused events (via class methods) and generic
    security events (via direct construction with event_id/severity/source/message).
    """

    # Generic event fields (for direct construction)
    event_id: str | None = None
    severity: SecuritySeverity | None = None
    source: str | None = None
    message: str | None = None
    metadata: dict | None = None

    # Adapter-focused event fields (for factory methods)
    event_type: SecurityEventType | None = None
    adapter_id: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional fields depending on event type
    status: str | None = None
    tier: str | None = None
    reason: str | None = None
    probe: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Return True if severity is HIGH or CRITICAL."""
        if self.severity is None:
            return False
        return self.severity in (SecuritySeverity.HIGH, SecuritySeverity.CRITICAL)

    @classmethod
    def adapter_evaluated(
        cls,
        adapter_id: UUID,
        status: str,
        tier: str,
    ) -> SecurityEvent:
        """Create an adapter evaluation event."""
        return cls(
            event_type=SecurityEventType.ADAPTER_EVALUATED,
            adapter_id=adapter_id,
            status=status,
            tier=tier,
        )

    @classmethod
    def activation_blocked(
        cls,
        adapter_id: UUID,
        reason: str,
    ) -> SecurityEvent:
        """Create an activation blocked event."""
        return cls(
            event_type=SecurityEventType.ACTIVATION_BLOCKED,
            adapter_id=adapter_id,
            reason=reason,
        )

    @classmethod
    def risk_override_recorded(
        cls,
        adapter_id: UUID,
        reason: str,
    ) -> SecurityEvent:
        """Create a risk override recorded event."""
        return cls(
            event_type=SecurityEventType.RISK_OVERRIDE_RECORDED,
            adapter_id=adapter_id,
            reason=reason,
        )

    @classmethod
    def canary_tripped(
        cls,
        adapter_id: UUID,
        probe: str,
    ) -> SecurityEvent:
        """Create a canary tripped event."""
        return cls(
            event_type=SecurityEventType.CANARY_TRIPPED,
            adapter_id=adapter_id,
            probe=probe,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "eventType": self.event_type.value,
            "adapterId": str(self.adapter_id),
        }
        if self.status is not None:
            result["status"] = self.status
        if self.tier is not None:
            result["tier"] = self.tier
        if self.reason is not None:
            result["reason"] = self.reason
        if self.probe is not None:
            result["probe"] = self.probe
        return result

    def emit(self) -> None:
        """Emit the event to the security event logger."""
        try:
            envelope = {
                "timestamp": self.timestamp.isoformat(),
                "event": self.to_dict(),
            }
            logger.info("SecurityEvent: %s", json.dumps(envelope))
        except Exception as e:
            logger.error("SecurityEvent serialization failed: %s", e)
