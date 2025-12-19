from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.thermo_path_integration import ThermoPathIntegration, RelationshipStrength


@dataclass
class DummyGate:
    gate_id: str
    gate_name: str
    confidence: float
    character_span: tuple[int, int]
    local_entropy: float | None = None


@dataclass
class DummyDetectionResult:
    detected_gates: list[DummyGate]
    gate_sequence: list[str]


def test_analyze_response_basic() -> None:
    detection = DummyDetectionResult(
        detected_gates=[
            DummyGate("G1", "INIT", 0.9, (10, 20)),
            DummyGate("G2", "REASON", 0.8, (50, 60)),
            DummyGate("G3", "CONCLUDE", 0.7, (90, 99)),
        ],
        gate_sequence=["G1", "G2", "G3"],
    )

    integration = ThermoPathIntegration()
    measurement = integration.analyze_response(
        response_text="x" * 100,
        entropy_trajectory=[0.5, 1.0, 1.5, 2.0, 2.5],
        gate_detection_result=detection,
    )

    assert measurement.gate_count == 3
    assert measurement.entropy_path_correlation is not None
    assert len(measurement.gate_transition_entropies) == 2


def test_analyze_relationship() -> None:
    integration = ThermoPathIntegration()
    measurements = [
        integration.analyze_response(
            response_text="x" * 100,
            entropy_trajectory=[0.2, 0.4, 0.6, 0.8],
            gate_detection_result=DummyDetectionResult(
                detected_gates=[
                    DummyGate("G1", "INIT", 0.9, (10, 20)),
                    DummyGate("G2", "REASON", 0.8, (50, 60)),
                    DummyGate("G3", "CONCLUDE", 0.7, (90, 99)),
                ],
                gate_sequence=["G1", "G2", "G3"],
            ),
        ),
        integration.analyze_response(
            response_text="x" * 100,
            entropy_trajectory=[0.3, 0.5, 0.7, 0.9],
            gate_detection_result=DummyDetectionResult(
                detected_gates=[
                    DummyGate("G1", "INIT", 0.9, (10, 20)),
                    DummyGate("G2", "REASON", 0.8, (50, 60)),
                    DummyGate("G3", "CONCLUDE", 0.7, (90, 99)),
                ],
                gate_sequence=["G1", "G2", "G3"],
            ),
        ),
    ]
    assessment = integration.analyze_relationship(measurements)
    assert assessment.relationship_strength in (
        RelationshipStrength.strong,
        RelationshipStrength.moderate,
        RelationshipStrength.weak,
        RelationshipStrength.none,
    )
