from __future__ import annotations

from datetime import datetime

import pytest

import modelcypher.core.domain.geometry.persona_vector_monitor as persona_mod
from modelcypher.core.domain.geometry.persona_vector_monitor import (
    PersonaBaseline,
    PersonaMetricKey,
    PersonaPosition,
    PersonaTraitDefinition,
    PersonaVector,
    PersonaVectorBundle,
    PersonaVectorMonitor,
    TrainingDriftMetrics,
)


def _trait(trait_id: str = "helpful") -> PersonaTraitDefinition:
    return PersonaTraitDefinition(
        id=trait_id,
        name=trait_id.title(),
        description="desc",
        positive_prompts=["p"],
        negative_prompts=["n"],
    )


def _vector(trait_id: str = "helpful") -> PersonaVector:
    return PersonaVector(
        id=trait_id,
        name=trait_id.title(),
        direction=[1.0, 0.0],
        layer_index=4,
        hidden_size=2,
        strength=1.0,
        correlation_coefficient=0.8,
        model_id="model-a",
        computed_at=datetime.utcnow(),
    )


def test_extract_vector_invalid_inputs(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(persona_mod, "get_default_backend", lambda: b)

    trait = _trait()

    assert PersonaVectorMonitor.extract_vector([], [], trait, 0, "m") is None
    assert PersonaVectorMonitor.extract_vector([[1.0, 2.0]], [], trait, 0, "m") is None
    assert PersonaVectorMonitor.extract_vector([1.0, 2.0], [1.0, 2.0], trait, 0, "m") is None
    assert (
        PersonaVectorMonitor.extract_vector(
            [[1.0, 2.0]],
            [[1.0]],
            trait,
            0,
            "m",
        )
        is None
    )


def test_extract_vector_threshold_and_normalization(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(persona_mod, "get_default_backend", lambda: b)

    def fake_mean(arr: object) -> object:
        array = arr if hasattr(arr, "shape") else b.array(arr)
        first = float(b.tolist(array)[0][0])
        if first > 0.5:
            return b.array([2.0, 0.0])
        return b.array([0.0, 0.0])

    monkeypatch.setattr(PersonaVectorMonitor, "_mean_vector", staticmethod(fake_mean))
    monkeypatch.setattr(PersonaVectorMonitor, "_l2_norm", staticmethod(lambda _arr: 2.0))
    monkeypatch.setattr(
        PersonaVectorMonitor,
        "_compute_correlation",
        staticmethod(lambda **_kwargs: 0.4),
    )

    trait = _trait()
    positive = [[1.0, 0.0], [1.0, 0.0]]
    negative = [[0.0, 0.0], [0.0, 0.0]]

    assert (
        PersonaVectorMonitor.extract_vector(
            positive,
            negative,
            trait,
            layer_index=2,
            model_id="model-x",
            correlation_threshold=0.5,
        )
        is None
    )

    monkeypatch.setattr(
        PersonaVectorMonitor,
        "_compute_correlation",
        staticmethod(lambda **_kwargs: 0.8),
    )
    vector = PersonaVectorMonitor.extract_vector(
        positive,
        negative,
        trait,
        layer_index=2,
        model_id="model-x",
        correlation_threshold=0.5,
    )

    assert vector is not None
    assert vector.direction == pytest.approx([1.0, 0.0], abs=1e-6)
    assert vector.strength == pytest.approx(2.0, abs=1e-6)
    assert vector.correlation_coefficient == pytest.approx(0.8, abs=1e-6)


def test_measure_position_branches(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(persona_mod, "get_default_backend", lambda: b)

    vector = _vector("honest")

    monkeypatch.setattr(PersonaVectorMonitor, "_projection_value", staticmethod(lambda *_a: 4.0))
    monkeypatch.setattr(PersonaVectorMonitor, "_l2_norm", staticmethod(lambda *_a: 2.0))

    baseline = PersonaBaseline(
        model_id="model-a",
        baseline_positions={"honest": 0.3},
        captured_at=datetime.utcnow(),
        is_pretrained_baseline=True,
    )

    measured = PersonaVectorMonitor.measure_position([1.0, 0.0], vector, baseline)
    assert measured is not None
    assert measured.normalized_position == pytest.approx(1.0, abs=1e-6)
    assert measured.delta_from_baseline == pytest.approx(0.7, abs=1e-6)

    assert PersonaVectorMonitor.measure_position([1.0], vector, baseline) is None

    monkeypatch.setattr(PersonaVectorMonitor, "_projection_value", staticmethod(lambda *_a: None))
    assert PersonaVectorMonitor.measure_position([1.0, 0.0], vector, baseline) is None


def test_measure_all_positions_filters_none(monkeypatch) -> None:
    keep = _vector("keep")
    drop = _vector("drop")
    bundle = PersonaVectorBundle(
        model_id="m",
        vectors=[keep, drop],
        primary_layer_index=1,
        computed_at=datetime.utcnow(),
        avg_correlation=0.5,
        min_correlation=0.5,
    )

    def fake_measure(_activation, vector: PersonaVector, _baseline):
        if vector.id == "keep":
            return PersonaPosition(
                trait_id="keep",
                trait_name="Keep",
                projection=0.2,
                normalized_position=0.1,
                delta_from_baseline=None,
                layer_index=1,
            )
        return None

    monkeypatch.setattr(PersonaVectorMonitor, "measure_position", staticmethod(fake_measure))

    positions = PersonaVectorMonitor.measure_all_positions([1.0, 0.0], bundle, baseline=None)
    assert len(positions) == 1
    assert positions[0].trait_id == "keep"


def test_compute_drift_metrics_and_baseline(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(persona_mod, "get_default_backend", lambda: b)

    positions = [
        PersonaPosition("a", "A", 0.5, 0.5, 0.2, 1),
        PersonaPosition("c", "C", 0.4, 0.4, None, 1),
        PersonaPosition("b", "B", 0.1, 0.1, -0.1, 1),
    ]

    metrics = PersonaVectorMonitor.compute_drift_metrics(
        positions,
        step=12,
        drift_threshold=0.15,
    )

    assert metrics.step == 12
    assert metrics.has_significant_drift is True
    assert metrics.drifting_traits == ["a"]
    assert metrics.overall_drift_magnitude > 0.0
    assert metrics.position_for_trait("c") is not None

    baseline = PersonaVectorMonitor.create_baseline(
        positions=positions,
        model_id="model-base",
        is_pretrained_baseline=True,
    )
    assert baseline.model_id == "model-base"
    assert baseline.baseline_positions["a"] == pytest.approx(0.5, abs=1e-6)

    no_delta_positions = [
        PersonaPosition("x", "X", 0.2, 0.2, None, 0),
        PersonaPosition("y", "Y", 0.3, 0.3, None, 0),
    ]
    no_drift = PersonaVectorMonitor.compute_drift_metrics(no_delta_positions, step=1)
    assert no_drift.has_significant_drift is False
    assert no_drift.overall_drift_magnitude == pytest.approx(0.0, abs=1e-8)


def test_extract_bundle_and_metrics_dictionary(monkeypatch) -> None:
    traits = [_trait("a"), _trait("b")]

    def fake_extract(
        positive_activations,
        negative_activations,
        trait: PersonaTraitDefinition,
        layer_index: int,
        model_id: str,
        correlation_threshold: float | None = None,
    ):
        del positive_activations, negative_activations, layer_index, model_id, correlation_threshold
        if trait.id != "a":
            return None
        return PersonaVector(
            id="a",
            name="A",
            direction=[1.0, 0.0],
            layer_index=3,
            hidden_size=2,
            strength=1.0,
            correlation_coefficient=0.9,
            model_id="m",
            computed_at=datetime.utcnow(),
        )

    monkeypatch.setattr(PersonaVectorMonitor, "extract_vector", staticmethod(fake_extract))

    bundle = PersonaVectorMonitor.extract_bundle(
        activations_per_trait={
            "a": ([[1.0, 0.0]], [[0.0, 0.0]]),
            "b": ([[1.0, 0.0]], [[0.0, 0.0]]),
        },
        traits=traits,
        layer_index=3,
        model_id="m",
    )

    assert bundle.model_id == "m"
    assert bundle.primary_layer_index == 3
    assert len(bundle.vectors) == 1
    assert bundle.vector_for_trait("a") is not None
    assert bundle.vector_for_trait("b") is None
    assert bundle.avg_correlation == pytest.approx(0.9, abs=1e-6)
    assert bundle.min_correlation == pytest.approx(0.9, abs=1e-6)
    assert "persona vectors extracted" in bundle.summary

    metrics = TrainingDriftMetrics(
        step=7,
        positions=[
            PersonaPosition("a", "A", 0.5, 0.2, 0.05, 3),
            PersonaPosition("b", "B", 0.3, -0.1, None, 3),
        ],
        overall_drift_magnitude=0.25,
        has_significant_drift=True,
        drifting_traits=["a"],
        timestamp=datetime.utcnow(),
    )
    payload = PersonaVectorMonitor.to_metrics_dictionary(metrics)

    assert payload[PersonaMetricKey.overall_drift] == pytest.approx(0.25, abs=1e-6)
    assert payload[PersonaMetricKey.has_significant_drift] == pytest.approx(1.0, abs=1e-6)
    assert payload[PersonaMetricKey.position("a")] == pytest.approx(0.2, abs=1e-6)
    assert payload[PersonaMetricKey.delta("a")] == pytest.approx(0.05, abs=1e-6)
    assert PersonaMetricKey.position("t") == "geometry/persona/t/position"
    assert PersonaMetricKey.delta("t") == "geometry/persona/t/delta"


def test_internal_statistics_and_projection_mismatch(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(persona_mod, "get_default_backend", lambda: b)

    assert PersonaVectorMonitor._compute_correlation_stats([]) == (0.0, 0.0)

    avg_corr, min_corr = PersonaVectorMonitor._compute_correlation_stats([0.2, 0.8])
    assert avg_corr == pytest.approx(0.5, abs=1e-6)
    assert min_corr == pytest.approx(0.2, abs=1e-6)

    assert PersonaVectorMonitor._projection_value([1.0, 0.0], [1.0]) is None
