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

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    gate_detection_threshold: float = 0.55
    capture_trajectory: bool = True
    max_tokens: int = 200
    temperature: float = 0.0


@dataclass(frozen=True)
class ThermoPathAssessment:
    """Assessment of thermo-path relationship.

    Caller interprets strength via strength_for_thresholds().

    Attributes
    ----------
    h3_supported : bool
        Whether measurements support hypothesis H3 (entropy-gate coupling).
    correlation : float or None
        Pearson correlation between entropy and gate count.
    spike_rate : float
        Rate of entropy spikes at gate transitions (0.0 to 1.0).
    measurement_count : int
        Number of measurements this assessment is based on.
    rationale : str
        Human-readable explanation of the assessment.
    """

    h3_supported: bool
    correlation: float | None
    spike_rate: float
    measurement_count: int
    rationale: str

    def strength_for_thresholds(
        self,
        strong_threshold: float = 0.6,
        moderate_threshold: float = 0.4,
        weak_threshold: float = 0.2,
    ) -> str:
        """Classify relationship strength using caller-provided thresholds.

        Args:
            strong_threshold: |correlation| above this is "strong"
            moderate_threshold: |correlation| above this is "moderate"
            weak_threshold: |correlation| above this is "weak"

        Returns:
            Strength label: "strong", "moderate", "weak", or "none"
        """
        if self.correlation is None:
            return "none"
        abs_corr = abs(self.correlation)
        if abs_corr > strong_threshold:
            return "strong"
        elif abs_corr > moderate_threshold:
            return "moderate"
        elif abs_corr > weak_threshold:
            return "weak"
        else:
            return "none"


@dataclass(frozen=True)
class GateDetail:
    gate_id: str
    gate_name: str
    local_entropy: float | None
    confidence: float


@dataclass(frozen=True)
class GateTransitionEntropy:
    from_gate: str
    to_gate: str
    entropy_delta: float
    is_spike: bool


@dataclass(frozen=True)
class CombinedMeasurement:
    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    entropy_trajectory: list[float] | None
    gate_sequence: list[str]
    gate_count: int
    gate_details: list[GateDetail]
    entropy_path_correlation: float | None
    gate_transition_entropies: list[GateTransitionEntropy]
    assessment: ThermoPathAssessment


@dataclass(frozen=True)
class ThermoTrajectory:
    """Container for thermo-path measurements across a response."""

    measurements: list[CombinedMeasurement]


class ThermoPathIntegration:
    def __init__(self, configuration: Configuration = Configuration()) -> None:
        self.config = configuration

    def analyze_relationship(self, measurements: list[CombinedMeasurement]) -> ThermoPathAssessment:
        if not measurements:
            return ThermoPathAssessment(
                h3_supported=False,
                correlation=None,
                spike_rate=0.0,
                measurement_count=0,
                rationale="No measurements to analyze",
            )

        entropies = [measurement.mean_entropy for measurement in measurements]
        gate_counts = [float(measurement.gate_count) for measurement in measurements]
        correlation = self._compute_pearson_correlation(entropies, gate_counts)

        total_transitions = 0
        spike_transitions = 0
        for measurement in measurements:
            total_transitions += len(measurement.gate_transition_entropies)
            spike_transitions += sum(
                1 for item in measurement.gate_transition_entropies if item.is_spike
            )
        spike_rate = (
            float(spike_transitions) / float(total_transitions) if total_transitions > 0 else 0.0
        )

        # H3 supported if correlation is moderate or strong (|r| > 0.4)
        # This is a geometric condition, not arbitrary binning
        h3_supported = correlation is not None and abs(correlation) > 0.4

        rationale = self._build_rationale(correlation, spike_rate, len(measurements))

        return ThermoPathAssessment(
            h3_supported=h3_supported,
            correlation=correlation,
            spike_rate=spike_rate,
            measurement_count=len(measurements),
            rationale=rationale,
        )

    def analyze_response(
        self,
        response_text: str,
        entropy_trajectory: list[float],
        gate_detection_result,
    ) -> CombinedMeasurement:
        if entropy_trajectory:
            mean_entropy = sum(entropy_trajectory) / float(len(entropy_trajectory))
            entropy_variance = self._compute_variance(entropy_trajectory)
            first_token_entropy = entropy_trajectory[0]
        else:
            mean_entropy = 0.0
            entropy_variance = 0.0
            first_token_entropy = 0.0

        gate_details: list[GateDetail] = []
        for gate in gate_detection_result.detected_gates:
            if entropy_trajectory:
                ratio = float(gate.character_span[0]) / float(max(1, len(response_text)))
                entropy_index = int(ratio * float(len(entropy_trajectory)))
                local_entropy = (
                    entropy_trajectory[entropy_index]
                    if entropy_index < len(entropy_trajectory)
                    else None
                )
            else:
                local_entropy = gate.local_entropy
            gate_details.append(
                GateDetail(
                    gate_id=gate.gate_id,
                    gate_name=gate.gate_name,
                    local_entropy=local_entropy,
                    confidence=gate.confidence,
                )
            )

        transitions: list[GateTransitionEntropy] = []
        for i in range(1, len(gate_details)):
            prev = gate_details[i - 1]
            curr = gate_details[i]
            if prev.local_entropy is None or curr.local_entropy is None:
                continue
            delta = curr.local_entropy - prev.local_entropy
            is_spike = abs(delta) > 0.5
            transitions.append(
                GateTransitionEntropy(
                    from_gate=prev.gate_name,
                    to_gate=curr.gate_name,
                    entropy_delta=delta,
                    is_spike=is_spike,
                )
            )

        gate_local_entropies = [
            detail.local_entropy for detail in gate_details if detail.local_entropy is not None
        ]
        gate_positions = [float(i) for i in range(len(gate_details))]
        correlation = (
            self._compute_pearson_correlation(gate_positions, gate_local_entropies)
            if len(gate_local_entropies) > 2
            else None
        )

        assessment = self._assess_single_measurement(
            correlation=correlation,
            spike_count=sum(1 for item in transitions if item.is_spike),
            gate_count=len(gate_details),
        )

        return CombinedMeasurement(
            mean_entropy=mean_entropy,
            entropy_variance=entropy_variance,
            first_token_entropy=first_token_entropy,
            entropy_trajectory=entropy_trajectory if self.config.capture_trajectory else None,
            gate_sequence=list(getattr(gate_detection_result, "gate_sequence", [])),
            gate_count=len(gate_details),
            gate_details=gate_details,
            entropy_path_correlation=correlation,
            gate_transition_entropies=transitions,
            assessment=assessment,
        )

    @staticmethod
    def _compute_pearson_correlation(x: list[float], y: list[float]) -> float | None:
        if len(x) != len(y) or len(x) <= 2:
            return None
        n = float(len(x))
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for i in range(len(x)):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            numerator += dx * dy
            denom_x += dx * dx
            denom_y += dy * dy
        denom = (denom_x**0.5) * (denom_y**0.5)
        if denom <= 0:
            return None
        return numerator / denom

    @staticmethod
    def _compute_variance(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / float(len(values))
        squared_diffs = [(value - mean) * (value - mean) for value in values]
        return sum(squared_diffs) / float(len(values) - 1)

    @staticmethod
    def _build_rationale(
        correlation: float | None,
        spike_rate: float,
        measurement_count: int,
    ) -> str:
        """Build human-readable rationale from raw measurements."""
        parts: list[str] = []
        if correlation is not None:
            parts.append(f"Entropy-gate correlation r={correlation:.3f}")
            # Interpret based on correlation magnitude
            if abs(correlation) > 0.6:
                parts.append("Strong thermo-path coupling supports H3")
            elif abs(correlation) > 0.4:
                parts.append("Moderate thermo-path coupling partially supports H3")
            elif abs(correlation) > 0.2:
                parts.append("Weak thermo-path coupling does not support H3")
            else:
                parts.append("No significant thermo-path coupling detected")
        else:
            parts.append("Insufficient data for correlation")

        parts.append(f"Entropy spike rate at transitions: {spike_rate * 100:.1f}%")
        parts.append(f"Based on {measurement_count} measurements")
        return ". ".join(parts)

    @staticmethod
    def _assess_single_measurement(
        correlation: float | None,
        spike_count: int,
        gate_count: int,
    ) -> ThermoPathAssessment:
        """Assess a single measurement's thermo-path relationship."""
        spike_rate = float(spike_count) / float(gate_count - 1) if gate_count > 1 else 0.0

        # H3 supported if moderate or strong correlation
        h3_supported = correlation is not None and abs(correlation) > 0.4

        correlation_text = f"{correlation:.2f}" if correlation is not None else "N/A"
        rationale = f"Single measurement: r={correlation_text}, spike_rate={spike_rate * 100:.1f}%"

        return ThermoPathAssessment(
            h3_supported=h3_supported,
            correlation=correlation,
            spike_rate=spike_rate,
            measurement_count=1,
            rationale=rationale,
        )


# Compatibility alias for legacy naming.
ThermoPathIntegrator = ThermoPathIntegration
