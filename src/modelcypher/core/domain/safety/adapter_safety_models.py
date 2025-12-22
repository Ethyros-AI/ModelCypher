"""Adapter safety models for LoRA adapter evaluation.

Defines types for tracking adapter safety status, evaluation tiers,
risk overrides, and safety scorecards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID


class AdapterSafetyStatus(str, Enum):
    """Safety status for a LoRA adapter."""

    ALLOWED = "allowed"
    """Adapter passed the configured safety checks."""

    QUARANTINED = "quarantined"
    """Adapter requires caution; activation may be limited or gated."""

    BLOCKED = "blocked"
    """Adapter is blocked from activation."""

    PENDING = "pending"
    """Safety evaluation has not finished."""


class AdapterSafetyTier(str, Enum):
    """Evaluation tier indicating latency budget and checks performed."""

    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


class AdapterSafetyTrigger(str, Enum):
    """Trigger indicating why the evaluation was initiated."""

    IMPORT_ADAPTER = "importAdapter"
    POST_TRAINING = "postTraining"
    MERGE = "merge"
    MANUAL_RESCAN = "manualRescan"
    ACTIVATION_CHECK = "activationCheck"
    RUNTIME_CANARY = "runtimeCanary"


@dataclass(frozen=True)
class RiskOverride:
    """User-provided override for accepting safety risk."""

    adapter_id: str
    """ID of the adapter being overridden."""

    reason: str
    """User's justification for the override."""

    probe_version_at_override: str
    """Version of the safety probe when override was granted."""

    delta_score_at_override: float
    """Delta score at the time of override."""

    adapter_hash_at_override: str
    """Hash of the adapter weights when override was granted."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the override was recorded."""

    def is_valid(self, adapter_hash: str, probe_version: str) -> bool:
        """Returns True if the override is still valid.

        Override is invalidated when the hash or probe version changes.
        """
        if probe_version != self.probe_version_at_override:
            return False
        return adapter_hash == self.adapter_hash_at_override


@dataclass(frozen=True)
class AdapterSafetyScorecard:
    """Safety evaluation result for an adapter."""

    adapter_id: UUID
    """Unique identifier for the adapter."""

    status: AdapterSafetyStatus
    """Current safety status."""

    evaluation_tier: AdapterSafetyTier
    """Tier of evaluation performed."""

    evaluation_version: str
    """Version of the evaluation system."""

    adapter_hash: str
    """Hash of the adapter weights."""

    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the evaluation was performed."""

    reason: Optional[str] = None
    """Reason for the current status."""

    warnings: tuple[str, ...] = ()
    """List of warnings from the evaluation."""

    probe_version: Optional[str] = None
    """Version of the safety probe used."""

    delta_score: Optional[float] = None
    """Delta score from baseline comparison."""

    projection_status: Optional[str] = None
    """Status of safe subspace projection."""

    risk_override: Optional[RiskOverride] = None
    """Active risk override, if any."""


@dataclass(frozen=True)
class AdapterSafetyContext:
    """Context passed into safety evaluations."""

    trigger: AdapterSafetyTrigger
    """What triggered the evaluation."""

    tier: AdapterSafetyTier
    """Evaluation tier to use."""

    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the evaluation was requested."""

    probe_version_hint: Optional[str] = None
    """Suggested probe version to use."""
