"""Persisted calibration record for Geometric Alignment System (GAS) thresholds.

GAS operates on entropy geometry (content-agnostic). This calibration provides
empirically-derived sentinel thresholds per base model so the "valley / ridge"
boundary is stable across architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SentinelConfiguration:
    """Configuration for the entropy sentinel.

    Derived from calibration data, provides thresholds for detecting
    entropy anomalies in model outputs.
    """

    entropy_ceiling: float = 4.0
    """Maximum expected entropy value before triggering alerts."""

    spike_threshold: float = 1.5
    """Threshold for detecting entropy spikes (sudden increases)."""

    minimum_delta_for_signal: float = 0.3
    """Minimum entropy delta required to register a signal."""


@dataclass(frozen=True)
class GeometricAlignmentCalibration:
    """Persisted calibration record for Geometric Alignment System (GAS) thresholds.

    GAS operates on entropy geometry (content-agnostic). This calibration provides
    empirically-derived sentinel thresholds per base model so the "valley / ridge"
    boundary is stable across architectures.
    """

    base_model_id: str
    """Base model identifier this calibration applies to."""

    sample_count: int
    """Number of samples used to derive this calibration."""

    entropy_ceiling: float
    """Maximum expected entropy value before triggering alerts."""

    spike_threshold: float
    """Threshold for detecting entropy spikes (sudden increases)."""

    minimum_delta_for_signal: float
    """Minimum entropy delta required to register a signal."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this calibration was created."""

    source: str | None = None
    """Source of the calibration data (e.g., 'benchmark-v1', 'user-calibration')."""

    @property
    def sentinel_configuration(self) -> SentinelConfiguration:
        """Convert to sentinel configuration."""
        return SentinelConfiguration(
            entropy_ceiling=self.entropy_ceiling,
            spike_threshold=self.spike_threshold,
            minimum_delta_for_signal=self.minimum_delta_for_signal,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "base_model_id": self.base_model_id,
            "sample_count": self.sample_count,
            "entropy_ceiling": self.entropy_ceiling,
            "spike_threshold": self.spike_threshold,
            "minimum_delta_for_signal": self.minimum_delta_for_signal,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GeometricAlignmentCalibration:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            base_model_id=data["base_model_id"],
            sample_count=data["sample_count"],
            entropy_ceiling=data["entropy_ceiling"],
            spike_threshold=data["spike_threshold"],
            minimum_delta_for_signal=data["minimum_delta_for_signal"],
            created_at=created_at,
            source=data.get("source"),
        )

    @classmethod
    def default_for_model(cls, base_model_id: str) -> GeometricAlignmentCalibration:
        """Create default calibration for a model.

        Uses conservative default thresholds suitable for most models.
        Should be replaced with empirically-derived values when available.
        """
        return cls(
            base_model_id=base_model_id,
            sample_count=0,
            entropy_ceiling=4.0,
            spike_threshold=1.5,
            minimum_delta_for_signal=0.3,
            source="default",
        )
