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
class ThermoPathAssessment:
    """Raw thermo-path measurements.

    Attributes
    ----------
    correlation : float or None
        Pearson correlation between entropy and gate count.
    spike_rate : float
        Rate of entropy spikes at gate transitions (0.0 to 1.0).
    measurement_count : int
        Number of measurements this assessment is based on.
    """

    correlation: float | None
    spike_rate: float
    measurement_count: int


@dataclass(frozen=True)
class GateDetail:
    gate_id: str
    gate_name: str
    local_entropy: float | None
    similarity: float


@dataclass(frozen=True)
class GateTransitionEntropy:
    from_gate: str
    to_gate: str
    entropy_delta: float


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
    """Analyze thermo-path relationships in model responses.

    No configuration needed - all analysis is derived from the data.
    """

    def analyze_relationship(self, measurements: list[CombinedMeasurement]) -> ThermoPathAssessment:
        if not measurements:
            return ThermoPathAssessment(
                correlation=None,
                spike_rate=0.0,
                measurement_count=0,
            )

        entropies = [measurement.mean_entropy for measurement in measurements]
        gate_counts = [float(measurement.gate_count) for measurement in measurements]
        correlation = self._compute_pearson_correlation(entropies, gate_counts)

        total_transitions = 0
        spike_transitions = 0
        for measurement in measurements:
            total_transitions += len(measurement.gate_transition_entropies)
            spike_transitions += sum(
                1
                for item in measurement.gate_transition_entropies
                if abs(item.entropy_delta) > 0
            )
        spike_rate = (
            float(spike_transitions) / float(total_transitions) if total_transitions > 0 else 0.0
        )

        return ThermoPathAssessment(
            correlation=correlation,
            spike_rate=spike_rate,
            measurement_count=len(measurements),
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
                    similarity=gate.similarity,
                )
            )

        transitions: list[GateTransitionEntropy] = []
        for i in range(1, len(gate_details)):
            prev = gate_details[i - 1]
            curr = gate_details[i]
            if prev.local_entropy is None or curr.local_entropy is None:
                continue
            delta = curr.local_entropy - prev.local_entropy
            # Any measurable change is a transition; magnitude is in entropy_delta
            transitions.append(
                GateTransitionEntropy(
                    from_gate=prev.gate_name,
                    to_gate=curr.gate_name,
                    entropy_delta=delta,
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
            spike_count=sum(1 for item in transitions if abs(item.entropy_delta) > 0),
            gate_count=len(gate_details),
        )

        return CombinedMeasurement(
            mean_entropy=mean_entropy,
            entropy_variance=entropy_variance,
            first_token_entropy=first_token_entropy,
            entropy_trajectory=entropy_trajectory,
            gate_sequence=list(getattr(gate_detection_result, "gate_sequence", [])),
            gate_count=len(gate_details),
            gate_details=gate_details,
            entropy_path_correlation=correlation,
            gate_transition_entropies=transitions,
            assessment=assessment,
        )

    @staticmethod
    def _compute_pearson_correlation(x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation between two 1D scalar sequences.

        INTENTIONAL EUCLIDEAN: This is the standard correlation formula using
        Euclidean norms. Correlation is a statistical measure on 1D sequences,
        not a geometric distance on a high-dimensional manifold.
        """
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
    def _assess_single_measurement(
        correlation: float | None,
        spike_count: int,
        gate_count: int,
    ) -> ThermoPathAssessment:
        """Assess a single measurement's thermo-path relationship."""
        spike_rate = float(spike_count) / float(gate_count - 1) if gate_count > 1 else 0.0

        return ThermoPathAssessment(
            correlation=correlation,
            spike_rate=spike_rate,
            measurement_count=1,
        )
