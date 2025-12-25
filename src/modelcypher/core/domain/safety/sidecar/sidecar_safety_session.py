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

"""Thread-safe, per-generation sidecar safety session.

This session is designed to be updated from synchronous logit processing
and consumed from async generation loops.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ScenarioMode,
    SessionControlState,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_decision import (
    InterventionKind,
    SidecarDivergenceSample,
    SidecarSafetyIntervention,
    SidecarSafetyMode,
    SidecarSafetyTelemetry,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_policy import (
    SidecarSafetyPolicy,
)


@dataclass
class _SessionState:
    """Internal mutable state for the session."""

    tokens_observed: int = 0
    min_horror_kl: float | None = None
    min_sentinel_kl: float | None = None
    max_mode_reached: SidecarSafetyMode = SidecarSafetyMode.NORMAL
    pending_intervention: SidecarSafetyIntervention | None = None
    committed_intervention: SidecarSafetyIntervention | None = None


class SidecarSafetySession:
    """Thread-safe, per-generation sidecar safety session.

    This session tracks divergence samples during generation and determines
    when interventions are needed based on configured policy thresholds.
    """

    def __init__(
        self,
        policy: SidecarSafetyPolicy | None = None,
        stabilizer_configured: bool = False,
    ):
        """Create a new sidecar safety session.

        Args:
            policy: Safety policy thresholds. Defaults to default policy.
            stabilizer_configured: Whether a stabilizer adapter is available
                for takeover interventions.
        """
        self._policy = policy or SidecarSafetyPolicy.default()
        self._stabilizer_configured = stabilizer_configured
        self._lock = threading.Lock()
        self._state = _SessionState()

    def reset(self) -> None:
        """Reset the session state for a new generation."""
        with self._lock:
            self._state = _SessionState()

    def observe(
        self,
        sample: SidecarDivergenceSample,
        control: SessionControlState | None = None,
        now: datetime | None = None,
    ) -> SidecarSafetyMode:
        """Observe a divergence sample and return the current safety mode.

        Args:
            sample: The divergence sample to observe.
            control: Optional session control state (scenario + consent).
            now: Current time. Defaults to UTC now.

        Returns:
            The current safety mode after observing this sample.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        thresholds = self._policy.thresholds(control, now)

        with self._lock:
            self._state.tokens_observed += 1

            # Track minimum KL values (closest approach)
            if sample.kl_to_horror is not None and math.isfinite(sample.kl_to_horror):
                if self._state.min_horror_kl is None:
                    self._state.min_horror_kl = sample.kl_to_horror
                else:
                    self._state.min_horror_kl = min(self._state.min_horror_kl, sample.kl_to_horror)

            if sample.kl_to_sentinel is not None and math.isfinite(sample.kl_to_sentinel):
                if self._state.min_sentinel_kl is None:
                    self._state.min_sentinel_kl = sample.kl_to_sentinel
                else:
                    self._state.min_sentinel_kl = min(
                        self._state.min_sentinel_kl, sample.kl_to_sentinel
                    )

            mode = SidecarSafetyMode.NORMAL

            # Check horror probe thresholds
            if sample.kl_to_horror is not None and math.isfinite(sample.kl_to_horror):
                kl_horror = sample.kl_to_horror

                if kl_horror <= thresholds.horror_hard:
                    mode = SidecarSafetyMode.INTERVENTION

                    # Create intervention if not already pending/committed
                    if (
                        self._state.committed_intervention is None
                        and self._state.pending_intervention is None
                    ):
                        kind = (
                            InterventionKind.STABILIZER_TAKEOVER
                            if self._stabilizer_configured
                            else InterventionKind.HALT
                        )

                        scenario = control.scenario.value if control else ScenarioMode.DEFAULT.value
                        consent_active = control.is_consent_active(now) if control else False

                        reason = (
                            f"horror_kl<=hard (kl={kl_horror:.4f} <= "
                            f"{thresholds.horror_hard:.4f}) "
                            f"scenario={scenario} consent={consent_active}"
                        )

                        self._state.pending_intervention = SidecarSafetyIntervention(
                            kind=kind,
                            token_index=sample.token_index,
                            reason=reason,
                            timestamp=now,
                        )

                elif kl_horror <= thresholds.horror_soft:
                    mode = max(mode, SidecarSafetyMode.CAUTION)

            # Check sentinel threshold (only if not already in intervention)
            if mode != SidecarSafetyMode.INTERVENTION:
                if (
                    thresholds.sentinel_soft is not None
                    and sample.kl_to_sentinel is not None
                    and math.isfinite(sample.kl_to_sentinel)
                    and sample.kl_to_sentinel <= thresholds.sentinel_soft
                ):
                    mode = max(mode, SidecarSafetyMode.CAUTION)

            self._state.max_mode_reached = max(self._state.max_mode_reached, mode)
            return mode

    def consume_pending_intervention(self) -> SidecarSafetyIntervention | None:
        """Consume and return any pending intervention.

        Once consumed, the intervention is moved to committed state and will
        not be returned again.

        Returns:
            The pending intervention, or None if none pending.
        """
        with self._lock:
            pending = self._state.pending_intervention
            if pending is None:
                return None

            self._state.pending_intervention = None
            self._state.committed_intervention = pending
            return pending

    def telemetry_snapshot(self) -> SidecarSafetyTelemetry:
        """Get a snapshot of session telemetry.

        Returns:
            Current telemetry state.
        """
        with self._lock:
            return SidecarSafetyTelemetry(
                total_tokens_observed=self._state.tokens_observed,
                min_horror_kl=self._state.min_horror_kl,
                min_sentinel_kl=self._state.min_sentinel_kl,
                max_mode_reached=self._state.max_mode_reached,
                intervention=(
                    self._state.committed_intervention or self._state.pending_intervention
                ),
            )
