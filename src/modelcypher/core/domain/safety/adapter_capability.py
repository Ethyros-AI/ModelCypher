"""Adapter capability types for resource access control.

Defines resource capabilities that adapters can declare and request,
along with violation tracking and guard configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class RiskLevel(str, Enum):
    """Risk level classification for capabilities."""

    SAFE = "safe"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"

    @property
    def color(self) -> str:
        """Color for UI display."""
        return {
            RiskLevel.SAFE: "green",
            RiskLevel.MODERATE: "yellow",
            RiskLevel.ELEVATED: "orange",
            RiskLevel.HIGH: "red",
        }[self]


class ResourceCapability(str, Enum):
    """Declared resource access capabilities for LoRA adapters.

    Adapters declare their resource needs via capability signature patterns
    (e.g., "resource:file_read"). The CapabilityGuard validates runtime
    resource requests against these declarations.

    Capability Strings:
    - resource:file_read - Read files from disk
    - resource:file_write - Write files to disk
    - resource:network_http - Make HTTP requests
    - resource:code_exec - Execute code/scripts
    - resource:none - Explicit no-resource declaration

    Key Insight: LoRA adapters themselves don't execute file/network ops.
    Capabilities are enforced on *tool requests* in agent pipelines when
    adapters request external resources through tool-use interfaces.
    """

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK_HTTP = "network_http"
    CODE_EXEC = "code_exec"
    NONE = "none"

    @classmethod
    def from_capability_string(cls, s: str) -> Optional[ResourceCapability]:
        """Creates from capability signature string (e.g., 'resource:file_read')."""
        if not s.startswith("resource:"):
            return None
        value = s[9:]  # Remove "resource:" prefix
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def capability_string(self) -> str:
        """Converts to capability signature string."""
        return f"resource:{self.value}"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return {
            ResourceCapability.FILE_READ: "File Read",
            ResourceCapability.FILE_WRITE: "File Write",
            ResourceCapability.NETWORK_HTTP: "Network (HTTP)",
            ResourceCapability.CODE_EXEC: "Code Execution",
            ResourceCapability.NONE: "None",
        }[self]

    @property
    def description(self) -> str:
        """Detailed description of the capability."""
        return {
            ResourceCapability.FILE_READ: (
                "Read files from the local filesystem within allowed paths."
            ),
            ResourceCapability.FILE_WRITE: (
                "Write files to the local filesystem within allowed paths."
            ),
            ResourceCapability.NETWORK_HTTP: (
                "Make HTTP/HTTPS requests to external services."
            ),
            ResourceCapability.CODE_EXEC: (
                "Execute code or scripts in a sandboxed environment."
            ),
            ResourceCapability.NONE: (
                "No resource access required - pure text generation only."
            ),
        }[self]

    @property
    def risk_level(self) -> RiskLevel:
        """Risk level for UI display (affects badge color)."""
        return {
            ResourceCapability.NONE: RiskLevel.SAFE,
            ResourceCapability.FILE_READ: RiskLevel.MODERATE,
            ResourceCapability.NETWORK_HTTP: RiskLevel.MODERATE,
            ResourceCapability.FILE_WRITE: RiskLevel.ELEVATED,
            ResourceCapability.CODE_EXEC: RiskLevel.HIGH,
        }[self]


@dataclass(frozen=True)
class CapabilityViolation:
    """A capability access violation detected by the guard."""

    adapter_id: UUID
    """The adapter that attempted the access."""

    adapter_name: str
    """The adapter's display name."""

    requested_capability: ResourceCapability
    """Capability that was requested."""

    declared_capabilities: frozenset[ResourceCapability]
    """Capabilities the adapter declared."""

    resource_identifier: Optional[str] = None
    """Resource that was requested (e.g., file path, URL)."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the violation occurred."""

    id: UUID = field(default_factory=uuid4)
    """Unique ID for this violation."""

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        declared = (
            "none"
            if not self.declared_capabilities
            else ", ".join(c.display_name for c in sorted(self.declared_capabilities, key=lambda x: x.value))
        )
        return (
            f'Adapter "{self.adapter_name}" requested {self.requested_capability.display_name} '
            f"but only declared: {declared}"
        )


class CapabilityCheckResult(Enum):
    """Result of a capability check."""

    ALLOWED = "allowed"
    """Access allowed - capability is declared."""

    DENIED = "denied"
    """Access denied - capability not declared."""

    MONITOR_ONLY = "monitorOnly"
    """Access allowed but logged (for monitoring without enforcement)."""

    @property
    def is_allowed(self) -> bool:
        """Whether access is allowed."""
        return self in (CapabilityCheckResult.ALLOWED, CapabilityCheckResult.MONITOR_ONLY)


@dataclass(frozen=True)
class CapabilityCheckOutcome:
    """Full outcome of a capability check including violation details."""

    result: CapabilityCheckResult
    """The check result."""

    violation: Optional[CapabilityViolation] = None
    """Violation details if result is DENIED or MONITOR_ONLY."""

    @property
    def is_allowed(self) -> bool:
        """Whether access is allowed."""
        return self.result.is_allowed


class EnforcementMode(str, Enum):
    """Mode for capability enforcement."""

    ENFORCE = "enforce"
    """Block violations immediately."""

    MONITOR = "monitor"
    """Log violations but allow access (for monitoring rollout)."""

    DISABLED = "disabled"
    """No enforcement or logging (disabled)."""


@dataclass(frozen=True)
class CapabilityGuardConfiguration:
    """Configuration for capability enforcement behavior."""

    is_enabled: bool = True
    """Whether enforcement is enabled."""

    enforcement_mode: EnforcementMode = EnforcementMode.ENFORCE
    """Whether to hard-block violations or just log them."""

    max_violations_before_disable: int = 3
    """Maximum violations before the adapter is disabled."""

    audit_logging_enabled: bool = True
    """Whether to log violations to the audit log."""

    always_allowed_capabilities: frozenset[ResourceCapability] = frozenset()
    """Capabilities that are always allowed (e.g., for debugging)."""

    @classmethod
    def default(cls) -> CapabilityGuardConfiguration:
        """Default configuration with enforcement enabled."""
        return cls()

    @classmethod
    def monitoring(cls) -> CapabilityGuardConfiguration:
        """Configuration for monitoring mode (log but don't block)."""
        return cls(enforcement_mode=EnforcementMode.MONITOR)

    @classmethod
    def disabled(cls) -> CapabilityGuardConfiguration:
        """Configuration with enforcement disabled."""
        return cls(
            is_enabled=False,
            enforcement_mode=EnforcementMode.DISABLED,
            audit_logging_enabled=False,
        )
