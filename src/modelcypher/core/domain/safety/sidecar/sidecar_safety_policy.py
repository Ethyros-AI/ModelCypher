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

"""Sidecar safety policy and thresholds.

Policy thresholds for sidecar divergence monitoring based on KL divergence.

Notes on KL interpretation:
- D_KL(p0 || pProbe) is always >= 0.
- Smaller values indicate distributions are more similar ("closer" in this probe space).
- For probe adapters that represent a catastrophic basin, a *low* KL is a proximity warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ScenarioMode,
    SessionControlState,
)


@dataclass(frozen=True)
class SidecarSafetyThresholds:
    """Computed safety thresholds for sidecar monitoring."""

    horror_hard: float
    """Hard-stop proximity threshold for the horror probe."""

    horror_soft: float
    """Soft warning threshold for the horror probe."""

    sentinel_soft: float | None
    """Soft warning threshold for the sentinel observer."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "horror_hard": self.horror_hard,
            "horror_soft": self.horror_soft,
            "sentinel_soft": self.sentinel_soft,
        }


@dataclass
class SidecarSafetyPolicy:
    """Policy thresholds for sidecar divergence monitoring.

    Defines both hard and soft thresholds for KL divergence from probe
    distributions. Hard thresholds trigger intervention; soft thresholds
    trigger caution mode. Soft thresholds can be adjusted based on
    consent grants.
    """

    horror_kl_divergence_hard: float = 0.10
    """Hard-stop proximity threshold for the horror probe.
    If D_KL(p0 || p_horror) <= this value, intervention is required."""

    horror_kl_divergence_soft: float = 0.25
    """Soft warning threshold for the horror probe (consent-adjustable)."""

    sentinel_kl_divergence_soft: float | None = None
    """Soft warning threshold for the sentinel observer (optional)."""

    relax_soft_thresholds_under_consent: bool = True
    """When true, a consent grant in scenario .horror relaxes the *soft* horror threshold.
    Hard-stop threshold is never relaxed."""

    consent_soft_threshold_multiplier: float = 0.6
    """Multiplier applied to soft thresholds when consent is active.

    Since "proximity" corresponds to *lower* KL, relaxing the soft threshold means
    decreasing the soft threshold by this multiplier (< 1.0).
    """

    @classmethod
    def default(cls) -> SidecarSafetyPolicy:
        """Create default policy."""
        return cls()

    def thresholds(
        self,
        control: SessionControlState | None = None,
        now: datetime | None = None,
    ) -> SidecarSafetyThresholds:
        """Compute effective thresholds given session control state.

        Args:
            control: Optional session control state (scenario + consent).
            now: Current time. Defaults to UTC now.

        Returns:
            Computed thresholds with consent adjustments applied.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if control is None:
            return SidecarSafetyThresholds(
                horror_hard=self.horror_kl_divergence_hard,
                horror_soft=self.horror_kl_divergence_soft,
                sentinel_soft=self.sentinel_kl_divergence_soft,
            )

        consent_active = control.is_consent_active(now)
        should_relax_soft = (
            consent_active
            and self.relax_soft_thresholds_under_consent
            and control.scenario
            in (ScenarioMode.HORROR, ScenarioMode.ROLEPLAY, ScenarioMode.FICTION)
        )

        soft_multiplier = (
            max(0.01, min(1.0, self.consent_soft_threshold_multiplier))
            if should_relax_soft
            else 1.0
        )

        return SidecarSafetyThresholds(
            horror_hard=self.horror_kl_divergence_hard,
            horror_soft=self.horror_kl_divergence_soft * soft_multiplier,
            sentinel_soft=(
                self.sentinel_kl_divergence_soft * soft_multiplier
                if self.sentinel_kl_divergence_soft is not None
                else None
            ),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "horror_kl_divergence_hard": self.horror_kl_divergence_hard,
            "horror_kl_divergence_soft": self.horror_kl_divergence_soft,
            "sentinel_kl_divergence_soft": self.sentinel_kl_divergence_soft,
            "relax_soft_thresholds_under_consent": self.relax_soft_thresholds_under_consent,
            "consent_soft_threshold_multiplier": self.consent_soft_threshold_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarSafetyPolicy:
        """Create from dictionary."""
        return cls(
            horror_kl_divergence_hard=data.get("horror_kl_divergence_hard", 0.10),
            horror_kl_divergence_soft=data.get("horror_kl_divergence_soft", 0.25),
            sentinel_kl_divergence_soft=data.get("sentinel_kl_divergence_soft"),
            relax_soft_thresholds_under_consent=data.get(
                "relax_soft_thresholds_under_consent", True
            ),
            consent_soft_threshold_multiplier=data.get(
                "consent_soft_threshold_multiplier", 0.6
            ),
        )
