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

"""Persisted calibration record for Geometric Alignment System (GAS) thresholds.

GAS operates on entropy geometry (content-agnostic). This calibration provides
empirically-derived sentinel thresholds per base model so the "valley / ridge"
boundary is stable across architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
)


@dataclass(frozen=True)
class SentinelThresholds:
    """Thresholds for the entropy sentinel.

    All thresholds must be computed from calibration data.
    No arbitrary defaults - require explicit calibration.
    """

    entropy_ceiling: float
    spike_threshold: float
    minimum_delta_for_signal: float

    @classmethod
    def from_calibration_data(
        cls,
        entropy_samples: list[float],
    ) -> SentinelThresholds:
        """Derive thresholds from measured entropy distribution."""
        if len(entropy_samples) < 2:
            raise ValueError("Calibration requires at least two entropy samples")

        backend = get_default_backend()

        sorted_samples = sorted(entropy_samples)
        entropy_eps = division_epsilon(backend, backend.array(sorted_samples))
        entropy_ceiling = find_magnitude_gap_threshold(
            sorted_samples, eps=entropy_eps, backend=backend
        )

        deltas = [
            abs(entropy_samples[i + 1] - entropy_samples[i])
            for i in range(len(entropy_samples) - 1)
        ]
        if not deltas:
            raise ValueError("Calibration requires entropy deltas")

        sorted_deltas = sorted(deltas)
        delta_eps = division_epsilon(backend, backend.array(sorted_deltas))
        gap_threshold = find_magnitude_gap_threshold(
            sorted_deltas, eps=delta_eps, backend=backend
        )
        spike_candidates = [d for d in sorted_deltas if d > gap_threshold]
        spike_threshold = spike_candidates[0] if spike_candidates else gap_threshold
        noise_band = [d for d in sorted_deltas if d <= gap_threshold]
        if noise_band:
            min_signal = find_magnitude_gap_threshold(
                noise_band, eps=delta_eps, backend=backend
            )
        else:
            min_signal = gap_threshold

        if min_signal <= 0.0:
            min_signal = delta_eps

        return cls(
            entropy_ceiling=entropy_ceiling,
            spike_threshold=spike_threshold,
            minimum_delta_for_signal=min_signal,
        )


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
    def sentinel_thresholds(self) -> SentinelThresholds:
        """Convert to sentinel thresholds."""
        return SentinelThresholds(
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
    def from_entropy_samples(
        cls,
        base_model_id: str,
        entropy_samples: list[float],
        source: str = "calibration",
    ) -> GeometricAlignmentCalibration:
        """Create calibration from measured entropy samples."""
        if not entropy_samples:
            raise ValueError("Cannot create calibration without entropy samples")

        thresholds = SentinelThresholds.from_calibration_data(entropy_samples)
        return cls(
            base_model_id=base_model_id,
            sample_count=len(entropy_samples),
            entropy_ceiling=thresholds.entropy_ceiling,
            spike_threshold=thresholds.spike_threshold,
            minimum_delta_for_signal=thresholds.minimum_delta_for_signal,
            source=source,
        )
