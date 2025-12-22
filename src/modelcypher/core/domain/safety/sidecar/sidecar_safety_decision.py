"""Sidecar safety decision types.

Types for sidecar divergence monitoring and intervention decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class SidecarSafetyMode(str, Enum):
    """Safety mode for sidecar monitoring."""

    NORMAL = "normal"
    """Normal operation, no concerns detected."""

    CAUTION = "caution"
    """Soft threshold crossed, elevated monitoring."""

    INTERVENTION = "intervention"
    """Hard threshold crossed, intervention required."""

    @property
    def severity_rank(self) -> int:
        """Numeric severity ranking for comparison."""
        return {
            SidecarSafetyMode.NORMAL: 0,
            SidecarSafetyMode.CAUTION: 1,
            SidecarSafetyMode.INTERVENTION: 2,
        }[self]

    def __lt__(self, other: SidecarSafetyMode) -> bool:
        """Compare modes by severity."""
        if not isinstance(other, SidecarSafetyMode):
            return NotImplemented
        return self.severity_rank < other.severity_rank

    def __le__(self, other: SidecarSafetyMode) -> bool:
        """Compare modes by severity."""
        if not isinstance(other, SidecarSafetyMode):
            return NotImplemented
        return self.severity_rank <= other.severity_rank

    def __gt__(self, other: SidecarSafetyMode) -> bool:
        """Compare modes by severity."""
        if not isinstance(other, SidecarSafetyMode):
            return NotImplemented
        return self.severity_rank > other.severity_rank

    def __ge__(self, other: SidecarSafetyMode) -> bool:
        """Compare modes by severity."""
        if not isinstance(other, SidecarSafetyMode):
            return NotImplemented
        return self.severity_rank >= other.severity_rank


class InterventionKind(str, Enum):
    """Kind of safety intervention."""

    STABILIZER_TAKEOVER = "stabilizerTakeover"
    """Stop pre-emission and request switching to the stabilizer adapter."""

    HALT = "halt"
    """Stop pre-emission and halt generation (no takeover configured/available)."""


@dataclass(frozen=True)
class SidecarSafetyIntervention:
    """Record of a safety intervention decision."""

    kind: InterventionKind
    """Type of intervention."""

    token_index: int
    """Token index at which intervention was triggered."""

    reason: str
    """Human-readable reason for intervention."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the intervention was triggered."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "kind": self.kind.value,
            "token_index": self.token_index,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarSafetyIntervention:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            kind=InterventionKind(data["kind"]),
            token_index=data["token_index"],
            reason=data["reason"],
            timestamp=timestamp,
        )


@dataclass(frozen=True)
class SidecarDivergenceSample:
    """Per-token divergence readings between active generation and probe distributions.

    Measures KL divergence from the active model distribution to various probe
    distributions (sentinel, horror) for safety monitoring.
    """

    id: UUID = field(default_factory=uuid4)
    """Unique identifier for this sample."""

    token_index: int = 0
    """Token index in the generation sequence."""

    kl_to_sentinel: Optional[float] = None
    """KL divergence to sentinel observer distribution."""

    kl_to_horror: Optional[float] = None
    """KL divergence to horror probe distribution."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this sample was recorded."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "token_index": self.token_index,
            "kl_to_sentinel": self.kl_to_sentinel,
            "kl_to_horror": self.kl_to_horror,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarDivergenceSample:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            token_index=data.get("token_index", 0),
            kl_to_sentinel=data.get("kl_to_sentinel"),
            kl_to_horror=data.get("kl_to_horror"),
            timestamp=timestamp,
        )


@dataclass(frozen=True)
class SidecarSafetyTelemetry:
    """Telemetry summary for sidecar safety monitoring."""

    total_tokens_observed: int
    """Total number of tokens observed in this session."""

    min_horror_kl: Optional[float]
    """Minimum KL divergence to horror probe (closest approach)."""

    min_sentinel_kl: Optional[float]
    """Minimum KL divergence to sentinel observer."""

    max_mode_reached: SidecarSafetyMode
    """Maximum safety mode reached during session."""

    intervention: Optional[SidecarSafetyIntervention]
    """Intervention record if one was triggered."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens_observed": self.total_tokens_observed,
            "min_horror_kl": self.min_horror_kl,
            "min_sentinel_kl": self.min_sentinel_kl,
            "max_mode_reached": self.max_mode_reached.value,
            "intervention": self.intervention.to_dict() if self.intervention else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarSafetyTelemetry:
        """Create from dictionary."""
        intervention_data = data.get("intervention")
        intervention = (
            SidecarSafetyIntervention.from_dict(intervention_data)
            if intervention_data
            else None
        )

        return cls(
            total_tokens_observed=data["total_tokens_observed"],
            min_horror_kl=data.get("min_horror_kl"),
            min_sentinel_kl=data.get("min_sentinel_kl"),
            max_mode_reached=SidecarSafetyMode(data["max_mode_reached"]),
            intervention=intervention,
        )
