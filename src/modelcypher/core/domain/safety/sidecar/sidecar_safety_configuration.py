"""Runtime configuration for sidecar LoRA safety monitoring.

This is intentionally file-path based (not Hub IDs) because sidecar adapters
are local, swappable artifacts managed by the adapter registry / user imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modelcypher.core.domain.safety.sidecar.session_control_state import (
    SessionControlState,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_policy import (
    SidecarSafetyPolicy,
)


@dataclass
class SidecarSafetyConfiguration:
    """Runtime configuration for sidecar LoRA safety monitoring."""

    sentinel_adapter_path: Optional[Path] = None
    """Optional sentinel observer adapter directory."""

    horror_adapter_path: Optional[Path] = None
    """Optional horror probe adapter directory (probe-only)."""

    stabilizer_adapter_path: Optional[Path] = None
    """Optional stabilizer adapter directory used for takeover."""

    policy: SidecarSafetyPolicy = field(default_factory=SidecarSafetyPolicy.default)
    """Policy thresholds for divergence-based gating."""

    session_control: Optional[SessionControlState] = None
    """Optional session control state (scenario + consent)."""

    @property
    def is_enabled(self) -> bool:
        """Whether sidecar safety monitoring is enabled."""
        return (
            self.sentinel_adapter_path is not None
            or self.horror_adapter_path is not None
            or self.stabilizer_adapter_path is not None
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sentinel_adapter_path": (
                str(self.sentinel_adapter_path) if self.sentinel_adapter_path else None
            ),
            "horror_adapter_path": (
                str(self.horror_adapter_path) if self.horror_adapter_path else None
            ),
            "stabilizer_adapter_path": (
                str(self.stabilizer_adapter_path)
                if self.stabilizer_adapter_path
                else None
            ),
            "policy": self.policy.to_dict(),
            "session_control": (
                self.session_control.to_dict() if self.session_control else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarSafetyConfiguration:
        """Create from dictionary."""
        sentinel_path = data.get("sentinel_adapter_path")
        horror_path = data.get("horror_adapter_path")
        stabilizer_path = data.get("stabilizer_adapter_path")
        policy_data = data.get("policy")
        session_data = data.get("session_control")

        return cls(
            sentinel_adapter_path=Path(sentinel_path) if sentinel_path else None,
            horror_adapter_path=Path(horror_path) if horror_path else None,
            stabilizer_adapter_path=Path(stabilizer_path) if stabilizer_path else None,
            policy=(
                SidecarSafetyPolicy.from_dict(policy_data)
                if policy_data
                else SidecarSafetyPolicy.default()
            ),
            session_control=(
                SessionControlState.from_dict(session_data) if session_data else None
            ),
        )

    @classmethod
    def disabled(cls) -> SidecarSafetyConfiguration:
        """Create a disabled configuration."""
        return cls()
