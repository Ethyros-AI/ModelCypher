from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    gate_detection_threshold: float = 0.55
    capture_trajectory: bool = True
    max_tokens: int = 200
    temperature: float = 0.0


class RelationshipStrength(str, Enum):
    strong = "STRONG"
    moderate = "MODERATE"
    weak = "WEAK"
    none = "NONE"


@dataclass(frozen=True)
class ThermoPathAssessment:
    h3_supported: bool
    relationship_strength: RelationshipStrength
    rationale: str


@dataclass(frozen=True)
class GateDetail:
    gate_id: str
    gate_name: str
    local_entropy: Optional[float]
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
    entropy_trajectory: Optional[list[float]]
    gate_sequence: list[str]
    gate_count: int
    gate_details: list[GateDetail]
    entropy_path_correlation: Optional[float]
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
                relationship_strength=RelationshipStrength.none,
                rationale="No measurements to analyze",
            )

        entropies = [measurement.mean_entropy for measurement in measurements]
        gate_counts = [float(measurement.gate_count) for measurement in measurements]
        correlation = self._compute_pearson_correlation(entropies, gate_counts)

        total_transitions = 0
        spike_transitions = 0
        for measurement in measurements:
            total_transitions += len(measurement.gate_transition_entropies)
            spike_transitions += sum(1 for item in measurement.gate_transition_entropies if item.is_spike)
        spike_rate = float(spike_transitions) / float(total_transitions) if total_transitions > 0 else 0.0

        if correlation is not None:
            if abs(correlation) > 0.6:
                strength = RelationshipStrength.strong
                h3_supported = True
            elif abs(correlation) > 0.4:
                strength = RelationshipStrength.moderate
                h3_supported = True
            elif abs(correlation) > 0.2:
                strength = RelationshipStrength.weak
                h3_supported = False
            else:
                strength = RelationshipStrength.none
                h3_supported = False
        else:
            strength = RelationshipStrength.none
            h3_supported = False

        rationale = self._build_rationale(correlation, spike_rate, strength, len(measurements))

        return ThermoPathAssessment(
            h3_supported=h3_supported,
            relationship_strength=strength,
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
                local_entropy = entropy_trajectory[entropy_index] if entropy_index < len(entropy_trajectory) else None
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

        gate_local_entropies = [detail.local_entropy for detail in gate_details if detail.local_entropy is not None]
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
    def _compute_pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
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
        correlation: Optional[float],
        spike_rate: float,
        strength: RelationshipStrength,
        measurement_count: int,
    ) -> str:
        parts: list[str] = []
        if correlation is not None:
            parts.append(f"Entropy-gate correlation r={correlation:.3f}")
        else:
            parts.append("Insufficient data for correlation")
        parts.append(f"Entropy spike rate at transitions: {spike_rate * 100:.1f}%")

        if strength == RelationshipStrength.strong:
            parts.append("Strong thermo-path coupling supports H3")
        elif strength == RelationshipStrength.moderate:
            parts.append("Moderate thermo-path coupling partially supports H3")
        elif strength == RelationshipStrength.weak:
            parts.append("Weak thermo-path coupling does not support H3")
        else:
            parts.append("No thermo-path coupling detected")

        parts.append(f"Based on {measurement_count} measurements")
        return ". ".join(parts)

    @staticmethod
    def _assess_single_measurement(
        correlation: Optional[float],
        spike_count: int,
        gate_count: int,
    ) -> ThermoPathAssessment:
        spike_rate = float(spike_count) / float(gate_count - 1) if gate_count > 1 else 0.0
        if correlation is not None and abs(correlation) > 0.5:
            strength = RelationshipStrength.moderate
            h3_supported = True
        elif spike_rate > 0.3:
            strength = RelationshipStrength.weak
            h3_supported = False
        else:
            strength = RelationshipStrength.none
            h3_supported = False

        correlation_text = f"{correlation:.2f}" if correlation is not None else "N/A"
        rationale = f"Single measurement: r={correlation_text}, spike_rate={spike_rate * 100:.1f}%"
        return ThermoPathAssessment(
            h3_supported=h3_supported,
            relationship_strength=strength,
            rationale=rationale,
        )


# Compatibility alias for legacy naming.
ThermoPathIntegrator = ThermoPathIntegration
