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
Thresholds are derived from baseline measurements - no arbitrary defaults.

Notes on KL interpretation:
- D_KL(p0 || pProbe) is always >= 0.
- Smaller values indicate distributions are more similar ("closer" in this probe space).
- For probe adapters that represent a catastrophic basin, a *low* KL is a proximity warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ScenarioMode,
    SessionControlState,
)


def _find_percentile_for_value(sorted_values: list[float], value: float) -> float:
    """Find the percentile rank of a value in a sorted list.

    Returns the percentile (0-100) at which the given value falls in the
    sorted distribution.
    """
    n = len(sorted_values)
    if n == 0:
        return 50.0
    # Find first index where sorted_values[i] >= value
    idx = 0
    for i, v in enumerate(sorted_values):
        if v >= value:
            idx = i
            break
    else:
        idx = n
    return 100.0 * idx / n


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

    Thresholds are derived from baseline KL measurements or from distribution
    crossing (safe vs attack). The baseline should be collected by running the
    model on safe/normal inputs and measuring KL divergence to the horror probe.

    NOTE: The percentile values and consent_soft_multiplier are deployment-specific
    POLICY choices, not geometric constants. They are not derivable from model
    geometry — they encode the operator's tradeoff between false positives and
    missed detections. Calibrate per deployment. Must be explicitly provided.
    """

    hard_percentile: float
    """Percentile for hard-stop threshold. Deployment-specific POLICY choice."""

    soft_percentile: float
    """Percentile for soft warning threshold. Deployment-specific POLICY choice."""

    baseline_kl_measurements: list[float] = field(default_factory=list)
    """Baseline KL divergence measurements from safe/normal generation.

    Used to derive thresholds via percentiles. If empty, thresholds are
    computed online from observed samples.
    """

    relax_soft_thresholds_under_consent: bool = True
    """When true, a consent grant relaxes the soft threshold (makes it stricter).

    Hard-stop threshold is NEVER relaxed.
    """

    consent_soft_multiplier: float | None = None
    """Multiplier for soft threshold when consent is active.

    Lower values make the soft threshold trigger earlier (more cautious).
    This is a POLICY choice (INCONCLUSIVE for geometric derivation) —
    calibrate based on your false-positive tolerance during consent-mode
    operation. Must be explicitly provided when consent adjustment is used.
    """

    @classmethod
    def from_kl_distributions(
        cls,
        safe_kl_samples: list[float],
        attack_kl_samples: list[float],
        n_bootstrap: int = 1000,
        seed: int | None = None,
        consent_soft_multiplier: float | None = None,
    ) -> "SidecarSafetyPolicy":
        """Create policy with thresholds derived from KL distribution crossing.

        Instead of fixed percentiles (1st/5th), finds the Bayes-optimal
        decision boundary between safe and attack KL distributions using
        empirical CDF crossing.

        Note: KL divergence is LOWER when closer to the attack basin, so
        attack_kl_samples should be LOWER than safe_kl_samples. The crossing
        finds where the distributions overlap.

        The hard threshold is set to ci_lower (most conservative — triggers
        earlier) and soft threshold to the boundary point estimate.

        Args:
            safe_kl_samples: KL divergence measurements from safe generation.
            attack_kl_samples: KL divergence measurements near attack basin.
            n_bootstrap: Bootstrap resamples for CI.
            seed: RNG seed for reproducibility.
            consent_soft_multiplier: Explicit policy multiplier (INCONCLUSIVE
                for geometric derivation — must be provided by operator).

        Returns:
            Policy with geometrically-derived baseline measurements and
            thresholds that will be computed from the crossing.

        Raises:
            ValueError: If either group has fewer than 2 samples.
        """
        from modelcypher.core.domain.geometry.distribution_crossing import (
            find_distribution_crossing,
        )

        # Attack KL is LOWER (closer to horror probe), safe KL is HIGHER.
        # Find crossing: attack_kl (group_a, lower) vs safe_kl (group_b, higher)
        result = find_distribution_crossing(
            attack_kl_samples,
            safe_kl_samples,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        # Store the crossing-derived thresholds as a synthetic baseline
        # that produces the right percentile thresholds.
        # hard = ci_lower (conservative), soft = boundary
        hard_threshold = result.ci_lower
        soft_threshold = result.boundary

        # Create a policy that stores the actual measurements and uses
        # custom percentiles that reproduce the crossing boundaries
        return cls(
            baseline_kl_measurements=sorted(safe_kl_samples + attack_kl_samples),
            hard_percentile=_find_percentile_for_value(
                sorted(safe_kl_samples + attack_kl_samples), hard_threshold
            ),
            soft_percentile=_find_percentile_for_value(
                sorted(safe_kl_samples + attack_kl_samples), soft_threshold
            ),
            consent_soft_multiplier=consent_soft_multiplier,
        )

    @classmethod
    def from_baseline(
        cls,
        baseline_measurements: list[float],
        hard_percentile: float,
        soft_percentile: float,
    ) -> SidecarSafetyPolicy:
        """Create policy with baseline measurements.

        Args:
            baseline_measurements: KL divergence measurements from safe generation
            hard_percentile: Percentile for hard threshold (deployment POLICY choice)
            soft_percentile: Percentile for soft threshold (deployment POLICY choice)

        Raises:
            ValueError: If baseline_measurements is empty
        """
        if not baseline_measurements:
            raise ValueError("baseline_measurements cannot be empty")
        return cls(
            baseline_kl_measurements=list(baseline_measurements),
            hard_percentile=hard_percentile,
            soft_percentile=soft_percentile,
        )

    def _compute_percentile(self, values: list[float], percentile: float) -> float:
        """Compute percentile value from a list of measurements."""
        if not values:
            # No data - return 0 (most conservative: any KL triggers)
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        idx = int(percentile * n / 100.0)
        idx = max(0, min(idx, n - 1))
        return sorted_values[idx]

    def thresholds(
        self,
        control: SessionControlState | None = None,
        now: datetime | None = None,
        observed_kl: list[float] | None = None,
    ) -> SidecarSafetyThresholds:
        """Compute effective thresholds given session control state.

        Thresholds are derived from baseline measurements if available,
        otherwise from observed KL values during the session.

        Args:
            control: Optional session control state (scenario + consent).
            now: Current time. Defaults to UTC now.
            observed_kl: Optional list of KL values observed during session
                         (used for online estimation if no baseline).

        Returns:
            Computed thresholds with consent adjustments applied.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Use baseline if available, otherwise use observed values
        measurements = self.baseline_kl_measurements if self.baseline_kl_measurements else (observed_kl or [])

        # Compute percentile-based thresholds
        horror_hard = self._compute_percentile(measurements, self.hard_percentile)
        horror_soft = self._compute_percentile(measurements, self.soft_percentile)

        # Ensure soft >= hard (soft is less restrictive)
        horror_soft = max(horror_soft, horror_hard)

        # Sentinel uses same percentiles if we have measurements
        sentinel_soft = horror_soft if measurements else None

        if control is None:
            return SidecarSafetyThresholds(
                horror_hard=horror_hard,
                horror_soft=horror_soft,
                sentinel_soft=sentinel_soft,
            )

        consent_active = control.is_consent_active(now)
        should_relax_soft = (
            consent_active
            and self.relax_soft_thresholds_under_consent
            and control.scenario
            in (ScenarioMode.HORROR, ScenarioMode.ROLEPLAY, ScenarioMode.FICTION)
        )

        # Consent makes soft threshold stricter (lower value = triggers earlier)
        if should_relax_soft and self.consent_soft_multiplier is None:
            raise ValueError(
                "consent_soft_multiplier must be explicitly provided when consent "
                "adjustment is active. This is a deployment-specific POLICY choice, "
                "not derivable from model geometry (INCONCLUSIVE)."
            )
        soft_multiplier: float = self.consent_soft_multiplier if should_relax_soft else 1.0  # type: ignore[assignment]

        return SidecarSafetyThresholds(
            horror_hard=horror_hard,  # Hard threshold NEVER changes
            horror_soft=horror_soft * soft_multiplier,
            sentinel_soft=(sentinel_soft * soft_multiplier if sentinel_soft is not None else None),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "baseline_kl_measurements": self.baseline_kl_measurements,
            "hard_percentile": self.hard_percentile,
            "soft_percentile": self.soft_percentile,
            "relax_soft_thresholds_under_consent": self.relax_soft_thresholds_under_consent,
            "consent_soft_multiplier": self.consent_soft_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SidecarSafetyPolicy:
        """Create from dictionary.

        Raises:
            ValueError: If hard_percentile or soft_percentile keys are missing.
        """
        if "hard_percentile" not in data or "soft_percentile" not in data:
            raise ValueError(
                "hard_percentile and soft_percentile must be explicitly provided. "
                "These are deployment-specific POLICY choices, not derivable from geometry."
            )
        return cls(
            hard_percentile=data["hard_percentile"],
            soft_percentile=data["soft_percentile"],
            baseline_kl_measurements=data.get("baseline_kl_measurements", []),
            relax_soft_thresholds_under_consent=data.get(
                "relax_soft_thresholds_under_consent", True
            ),
            consent_soft_multiplier=data.get("consent_soft_multiplier"),
        )
