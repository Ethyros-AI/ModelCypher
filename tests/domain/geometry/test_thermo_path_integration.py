from __future__ import annotations

from dataclasses import dataclass

import pytest

import modelcypher.core.domain.geometry.thermo_path_integration as thermo_mod
from modelcypher.core.domain.geometry.thermo_path_integration import (
    CombinedMeasurement,
    GateDetail,
    GateTransitionEntropy,
    ThermoPathAssessment,
    ThermoPathIntegration,
    ThermoTrajectory,
)


@dataclass(frozen=True)
class _Gate:
    gate_id: str
    gate_name: str
    character_span: tuple[int, int]
    local_entropy: float | None
    similarity: float


@dataclass(frozen=True)
class _GateDetection:
    detected_gates: list[_Gate]
    gate_sequence: list[str]


def _measurement(*, mean_entropy: float, transitions: list[GateTransitionEntropy], gate_count: int) -> CombinedMeasurement:
    return CombinedMeasurement(
        mean_entropy=mean_entropy,
        entropy_variance=0.0,
        first_token_entropy=mean_entropy,
        entropy_trajectory=[mean_entropy],
        gate_sequence=["g"],
        gate_count=gate_count,
        gate_details=[GateDetail("id", "name", 0.1, 0.2)],
        entropy_path_correlation=None,
        gate_transition_entropies=transitions,
        assessment=ThermoPathAssessment(correlation=None, spike_rate=0.0, measurement_count=1),
    )


def test_analyze_relationship_empty_and_non_empty(monkeypatch) -> None:
    integration = ThermoPathIntegration()

    empty = integration.analyze_relationship([])
    assert empty.correlation is None
    assert empty.spike_rate == pytest.approx(0.0, abs=1e-8)
    assert empty.measurement_count == 0

    monkeypatch.setattr(
        ThermoPathIntegration,
        "_compute_geodesic_correlation",
        staticmethod(lambda _x, _y: 0.42),
    )

    non_empty = integration.analyze_relationship(
        [
            _measurement(
                mean_entropy=0.3,
                transitions=[
                    GateTransitionEntropy("a", "b", 0.0),
                    GateTransitionEntropy("b", "c", 0.1),
                ],
                gate_count=3,
            ),
            _measurement(
                mean_entropy=0.7,
                transitions=[GateTransitionEntropy("x", "y", -0.2)],
                gate_count=2,
            ),
        ]
    )

    assert non_empty.correlation == pytest.approx(0.42, abs=1e-8)
    assert non_empty.spike_rate == pytest.approx(2.0 / 3.0, abs=1e-8)
    assert non_empty.measurement_count == 2


def test_analyze_response_with_entropy_trajectory(monkeypatch) -> None:
    integration = ThermoPathIntegration()

    monkeypatch.setattr(
        ThermoPathIntegration,
        "_compute_geodesic_correlation",
        staticmethod(lambda _x, _y: 0.5),
    )

    gates = [
        _Gate("g1", "Gate1", (0, 1), None, 0.1),
        _Gate("g2", "Gate2", (5, 6), None, 0.2),
        _Gate("g3", "Gate3", (11, 12), None, 0.3),
    ]
    detection = _GateDetection(detected_gates=gates, gate_sequence=["g1", "g2", "g3"])

    measurement = integration.analyze_response(
        response_text="hello world!",
        entropy_trajectory=[0.1, 0.2, 0.3],
        gate_detection_result=detection,
    )

    assert measurement.mean_entropy == pytest.approx(0.2, abs=1e-8)
    assert measurement.entropy_variance > 0.0
    assert measurement.first_token_entropy == pytest.approx(0.1, abs=1e-8)
    assert measurement.gate_count == 3
    assert measurement.entropy_path_correlation == pytest.approx(0.5, abs=1e-8)
    assert measurement.gate_sequence == ["g1", "g2", "g3"]
    assert measurement.gate_transition_entropies
    assert measurement.assessment.measurement_count == 1


def test_analyze_response_without_entropy_trajectory(monkeypatch) -> None:
    integration = ThermoPathIntegration()
    monkeypatch.setattr(
        ThermoPathIntegration,
        "_compute_geodesic_correlation",
        staticmethod(lambda _x, _y: 0.9),
    )

    gates = [
        _Gate("g1", "Gate1", (0, 1), 0.3, 0.1),
        _Gate("g2", "Gate2", (2, 3), None, 0.2),
    ]
    detection = _GateDetection(detected_gates=gates, gate_sequence=[])

    measurement = integration.analyze_response(
        response_text="hi",
        entropy_trajectory=[],
        gate_detection_result=detection,
    )

    assert measurement.mean_entropy == pytest.approx(0.0, abs=1e-8)
    assert measurement.entropy_variance == pytest.approx(0.0, abs=1e-8)
    assert measurement.first_token_entropy == pytest.approx(0.0, abs=1e-8)
    assert measurement.gate_details[0].local_entropy == pytest.approx(0.3, abs=1e-8)
    assert measurement.gate_details[1].local_entropy is None
    assert measurement.entropy_path_correlation is None
    assert measurement.assessment.spike_rate == pytest.approx(0.0, abs=1e-8)


def test_compute_geodesic_correlation_branches(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(thermo_mod, "get_default_backend", lambda: b)

    assert ThermoPathIntegration._compute_geodesic_correlation([1.0], [1.0]) is None
    assert ThermoPathIntegration._compute_geodesic_correlation([1.0, 2.0], [1.0]) is None

    monkeypatch.setattr(
        thermo_mod,
        "geodesic_pairwise_metrics",
        lambda *_args, **_kwargs: (b.array([]), b.array([])),
    )
    assert ThermoPathIntegration._compute_geodesic_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None

    monkeypatch.setattr(
        thermo_mod,
        "geodesic_pairwise_metrics",
        lambda *_args, **_kwargs: (b.array([0.7]), b.array([0.0])),
    )
    monkeypatch.setattr(thermo_mod, "is_finite", lambda *_args, **_kwargs: False)
    assert ThermoPathIntegration._compute_geodesic_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None

    monkeypatch.setattr(thermo_mod, "is_finite", lambda *_args, **_kwargs: True)
    corr = ThermoPathIntegration._compute_geodesic_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert corr == pytest.approx(0.7, abs=1e-6)


def test_variance_assessment_and_trajectory_container() -> None:
    assert ThermoPathIntegration._compute_variance([0.3]) == pytest.approx(0.0, abs=1e-8)
    assert ThermoPathIntegration._compute_variance([1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-8)

    assessment = ThermoPathIntegration._assess_single_measurement(
        correlation=0.25,
        spike_count=2,
        gate_count=3,
    )
    assert assessment.correlation == pytest.approx(0.25, abs=1e-8)
    assert assessment.spike_rate == pytest.approx(1.0, abs=1e-8)
    assert assessment.measurement_count == 1

    assessment_no_gates = ThermoPathIntegration._assess_single_measurement(
        correlation=None,
        spike_count=10,
        gate_count=1,
    )
    assert assessment_no_gates.spike_rate == pytest.approx(0.0, abs=1e-8)

    trajectory = ThermoTrajectory(measurements=[])
    assert trajectory.measurements == []
