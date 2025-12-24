"""Capability guard for enforcing declared resource capabilities at runtime.

Architecture:
    AgentPipeline -> CapabilityGuard.check_access() -> ToolExecutor
                            |
                       ViolationLog (audit trail)

Design Decisions:
- Thread-safe violation tracking (use locks if needed for concurrent access)
- Supports monitor-only mode for gradual rollout
- Logs violations to privacy-preserving audit trail
- Can disable adapters after repeated violations
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from uuid import UUID

from modelcypher.core.domain.safety.adapter_capability import (
    CapabilityCheckOutcome,
    CapabilityCheckResult,
    CapabilityGuardConfiguration,
    CapabilityViolation,
    EnforcementMode,
    ResourceCapability,
)

logger = logging.getLogger(__name__)


@dataclass
class AdapterCapabilityRecord:
    """Internal record of an adapter's declared capabilities."""

    adapter_id: UUID
    adapter_name: str
    declared_capabilities: frozenset[ResourceCapability]
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEventType(str, Enum):
    """Type of audit event."""

    ALLOWED = "allowed"
    VIOLATION = "violation"
    MONITOR_ONLY = "monitorOnly"


@dataclass(frozen=True)
class CapabilityAuditEvent:
    """A single audit event for capability access."""

    timestamp: datetime
    event_type: AuditEventType
    adapter_id: UUID
    capability: ResourceCapability
    resource_hash: str | None = None


class CapabilityAuditLog:
    """Privacy-preserving audit log for capability violations.

    Logs events for security monitoring without storing full resource details.
    """

    def __init__(self, max_events: int = 1000):
        """Create an audit log.

        Args:
            max_events: Maximum events to retain.
        """
        self._events: list[CapabilityAuditEvent] = []
        self._max_events = max_events

    def log_violation(self, violation: CapabilityViolation) -> None:
        """Log a capability violation.

        Args:
            violation: The violation to log.
        """
        resource_hash = None
        if violation.resource_identifier:
            resource_hash = self._hash(violation.resource_identifier)

        event = CapabilityAuditEvent(
            timestamp=violation.timestamp,
            event_type=AuditEventType.VIOLATION,
            adapter_id=violation.adapter_id,
            capability=violation.requested_capability,
            resource_hash=resource_hash,
        )
        self._events.append(event)
        self._trim_if_needed()

    def log_access_allowed(
        self,
        adapter_id: UUID,
        capability: ResourceCapability,
    ) -> None:
        """Log an allowed access.

        Args:
            adapter_id: The adapter that was allowed.
            capability: The capability that was accessed.
        """
        event = CapabilityAuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.ALLOWED,
            adapter_id=adapter_id,
            capability=capability,
        )
        self._events.append(event)
        self._trim_if_needed()

    @property
    def total_events(self) -> int:
        """Total events logged."""
        return len(self._events)

    @property
    def violation_count(self) -> int:
        """Number of violations logged."""
        return sum(1 for e in self._events if e.event_type == AuditEventType.VIOLATION)

    def reset(self) -> None:
        """Clear all events."""
        self._events.clear()

    def recent_events(self, limit: int = 50) -> list[CapabilityAuditEvent]:
        """Get recent events.

        Args:
            limit: Maximum events to return.

        Returns:
            List of recent events.
        """
        return self._events[-limit:]

    def _trim_if_needed(self) -> None:
        """Trim events if over capacity."""
        if len(self._events) > self._max_events:
            excess = len(self._events) - self._max_events
            self._events = self._events[excess:]

    @staticmethod
    def _hash(string: str) -> str:
        """Create a privacy-preserving hash of a string."""
        return hashlib.md5(string.encode()).hexdigest()[:8]


class CapabilityGuard:
    """Guard that enforces declared resource capabilities at runtime.

    Registers adapters with their declared capabilities and checks
    access requests against those declarations.
    """

    def __init__(
        self,
        configuration: CapabilityGuardConfiguration | None = None,
    ):
        """Create a capability guard.

        Args:
            configuration: Guard configuration. Defaults to enforce mode.
        """
        self._configuration = configuration or CapabilityGuardConfiguration.default()
        self._audit_log = CapabilityAuditLog()

        # State
        self._adapter_capabilities: dict[UUID, AdapterCapabilityRecord] = {}
        self._violation_history: dict[UUID, list[CapabilityViolation]] = {}
        self._disabled_adapters: set[UUID] = set()

    def register_adapter(
        self,
        adapter_id: UUID,
        name: str,
        capabilities: frozenset[ResourceCapability],
    ) -> None:
        """Register an adapter's declared capabilities.

        Call this when an adapter is loaded or activated.

        Args:
            adapter_id: Adapter's unique identifier.
            name: Adapter's display name.
            capabilities: Set of declared resource capabilities.
        """
        record = AdapterCapabilityRecord(
            adapter_id=adapter_id,
            adapter_name=name,
            declared_capabilities=capabilities,
        )
        self._adapter_capabilities[adapter_id] = record
        logger.debug(
            "Registered adapter capabilities: id=%s name=%s capabilities=%s",
            str(adapter_id)[:8],
            name,
            [c.value for c in capabilities],
        )

    def unregister_adapter(self, adapter_id: UUID) -> None:
        """Unregister an adapter.

        Call when adapter is unloaded or deactivated.

        Args:
            adapter_id: Adapter to unregister.
        """
        self._adapter_capabilities.pop(adapter_id, None)
        logger.debug("Unregistered adapter: %s", str(adapter_id)[:8])

    def check_access(
        self,
        adapter_id: UUID,
        capability: ResourceCapability,
        resource_identifier: str | None = None,
    ) -> CapabilityCheckOutcome:
        """Check if an adapter is allowed to access a resource capability.

        Args:
            adapter_id: The requesting adapter's ID.
            capability: The capability being requested.
            resource_identifier: Optional resource being accessed (for logging).

        Returns:
            Check outcome indicating allowed, denied, or monitor-only.
        """
        # Disabled guard = always allow
        if not self._configuration.is_enabled:
            return CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)

        # Always-allowed capabilities bypass checks
        if capability in self._configuration.always_allowed_capabilities:
            return CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)

        # Check if adapter is disabled
        if adapter_id in self._disabled_adapters:
            violation = self._make_violation(
                adapter_id=adapter_id,
                capability=capability,
                resource_identifier=resource_identifier,
            )
            return CapabilityCheckOutcome(
                result=CapabilityCheckResult.DENIED, violation=violation
            )

        # Get adapter record
        record = self._adapter_capabilities.get(adapter_id)
        if record is None:
            logger.warning(
                "Capability check for unregistered adapter: %s", str(adapter_id)[:8]
            )
            # Unregistered adapters get no capabilities
            violation = CapabilityViolation(
                adapter_id=adapter_id,
                adapter_name="Unknown",
                requested_capability=capability,
                declared_capabilities=frozenset(),
                resource_identifier=resource_identifier,
            )
            return self._handle_violation(violation)

        # Check if capability is declared
        if capability in record.declared_capabilities:
            return CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)

        # Violation detected
        violation = CapabilityViolation(
            adapter_id=adapter_id,
            adapter_name=record.adapter_name,
            requested_capability=capability,
            declared_capabilities=record.declared_capabilities,
            resource_identifier=resource_identifier,
        )

        return self._handle_violation(violation)

    def check_access_batch(
        self,
        adapter_id: UUID,
        capabilities: frozenset[ResourceCapability],
        resource_identifier: str | None = None,
    ) -> CapabilityCheckOutcome:
        """Check multiple capabilities, returning denied on first violation.

        Args:
            adapter_id: The requesting adapter's ID.
            capabilities: Set of capabilities being requested.
            resource_identifier: Optional resource being accessed.

        Returns:
            Check outcome for the batch.
        """
        for capability in capabilities:
            outcome = self.check_access(adapter_id, capability, resource_identifier)
            if outcome.result == CapabilityCheckResult.DENIED:
                return outcome
        return CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)

    def configure(self, configuration: CapabilityGuardConfiguration) -> None:
        """Update the guard configuration.

        Args:
            configuration: New configuration to apply.
        """
        self._configuration = configuration
        logger.info(
            "CapabilityGuard configuration updated: mode=%s",
            configuration.enforcement_mode.value,
        )

    @property
    def is_enabled(self) -> bool:
        """Whether enforcement is currently enabled."""
        return self._configuration.is_enabled

    @property
    def enforcement_mode(self) -> EnforcementMode:
        """Current enforcement mode."""
        return self._configuration.enforcement_mode

    @property
    def total_violation_count(self) -> int:
        """Total violation count across all adapters."""
        return sum(len(v) for v in self._violation_history.values())

    def violations_for(self, adapter_id: UUID) -> list[CapabilityViolation]:
        """Get violations for a specific adapter.

        Args:
            adapter_id: Adapter to query.

        Returns:
            List of violations for the adapter.
        """
        return list(self._violation_history.get(adapter_id, []))

    def all_violations(self) -> list[CapabilityViolation]:
        """Get all violations."""
        result = []
        for violations in self._violation_history.values():
            result.extend(violations)
        return result

    def is_adapter_disabled(self, adapter_id: UUID) -> bool:
        """Check if an adapter is disabled.

        Args:
            adapter_id: Adapter to check.

        Returns:
            True if disabled.
        """
        return adapter_id in self._disabled_adapters

    def capabilities_for(
        self, adapter_id: UUID
    ) -> frozenset[ResourceCapability] | None:
        """Get capabilities for a registered adapter.

        Args:
            adapter_id: Adapter to query.

        Returns:
            Declared capabilities, or None if not registered.
        """
        record = self._adapter_capabilities.get(adapter_id)
        return record.declared_capabilities if record else None

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._adapter_capabilities.clear()
        self._violation_history.clear()
        self._disabled_adapters.clear()
        self._audit_log.reset()
        logger.info("CapabilityGuard state reset")

    def reenable_adapter(self, adapter_id: UUID) -> None:
        """Re-enable a disabled adapter.

        Args:
            adapter_id: Adapter to re-enable.
        """
        self._disabled_adapters.discard(adapter_id)
        logger.info("Re-enabled adapter: %s", str(adapter_id)[:8])

    def _handle_violation(
        self, violation: CapabilityViolation
    ) -> CapabilityCheckOutcome:
        """Handle a capability violation.

        Args:
            violation: The violation to handle.

        Returns:
            Check outcome based on enforcement mode.
        """
        # Record violation
        if violation.adapter_id not in self._violation_history:
            self._violation_history[violation.adapter_id] = []
        self._violation_history[violation.adapter_id].append(violation)

        # Log to audit trail
        if self._configuration.audit_logging_enabled:
            self._audit_log.log_violation(violation)

        logger.warning(
            "Capability violation: adapter=%s requested=%s declared=%s",
            violation.adapter_name,
            violation.requested_capability.value,
            [c.value for c in violation.declared_capabilities],
        )

        # Check if adapter should be disabled
        adapter_violations = len(
            self._violation_history.get(violation.adapter_id, [])
        )
        if adapter_violations >= self._configuration.max_violations_before_disable:
            self._disabled_adapters.add(violation.adapter_id)
            logger.error(
                "Adapter disabled due to repeated violations: %s",
                violation.adapter_name,
            )

        # Return based on enforcement mode
        if self._configuration.enforcement_mode == EnforcementMode.ENFORCE:
            return CapabilityCheckOutcome(
                result=CapabilityCheckResult.DENIED, violation=violation
            )
        elif self._configuration.enforcement_mode == EnforcementMode.MONITOR:
            return CapabilityCheckOutcome(
                result=CapabilityCheckResult.MONITOR_ONLY, violation=violation
            )
        else:  # DISABLED
            return CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)

    def _make_violation(
        self,
        adapter_id: UUID,
        capability: ResourceCapability,
        resource_identifier: str | None,
    ) -> CapabilityViolation:
        """Create a violation record.

        Args:
            adapter_id: Adapter ID.
            capability: Requested capability.
            resource_identifier: Optional resource.

        Returns:
            Violation record.
        """
        record = self._adapter_capabilities.get(adapter_id)
        return CapabilityViolation(
            adapter_id=adapter_id,
            adapter_name=record.adapter_name if record else "Disabled",
            requested_capability=capability,
            declared_capabilities=record.declared_capabilities
            if record
            else frozenset(),
            resource_identifier=resource_identifier,
        )
