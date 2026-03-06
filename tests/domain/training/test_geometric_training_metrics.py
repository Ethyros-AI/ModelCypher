from __future__ import annotations

import sys

import pytest

import modelcypher.core.domain.training.geometric_training_metrics as metrics_mod
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
    GeometricMetricsHistory,
    GeometricTrainingMetrics,
    GeometryMetricKey,
)


def test_instrumentation_level_properties_exhaustive() -> None:
    minimal = GeometricInstrumentationLevel.minimal
    moderate = GeometricInstrumentationLevel.moderate
    full = GeometricInstrumentationLevel.full
    research = GeometricInstrumentationLevel.research

    assert minimal.description == "Minimal - gradient norms only"
    assert moderate.description == "Moderate - adds curvature estimation"
    assert full.description == "Full - all metrics every step"
    assert research.description == "Research - includes loss landscape sampling"

    assert minimal.hessian_computation_interval == sys.maxsize
    assert moderate.hessian_computation_interval == 10
    assert full.hessian_computation_interval == 1
    assert research.hessian_computation_interval == 1

    assert minimal.compute_per_layer_metrics is False
    assert moderate.compute_per_layer_metrics is False
    assert full.compute_per_layer_metrics is True
    assert research.compute_per_layer_metrics is True

    assert minimal.compute_loss_landscape is False
    assert moderate.compute_loss_landscape is False
    assert full.compute_loss_landscape is False
    assert research.compute_loss_landscape is True

    assert minimal.compute_top_eigenvalue is False
    assert moderate.compute_top_eigenvalue is True
    assert full.compute_top_eigenvalue is True
    assert research.compute_top_eigenvalue is True

    assert minimal.metrics_collected == ["Gradient norms", "Parameter divergence"]
    assert len(minimal.metrics_collected) == 2
    assert moderate.metrics_collected == [
        "Gradient norms",
        "Parameter divergence",
        "Curvature estimation (Hessian trace)",
    ]
    assert len(moderate.metrics_collected) == 3
    assert full.metrics_collected == [
        "Gradient norms",
        "Parameter divergence",
        "Curvature estimation",
        "Per-layer statistics",
        "Effective step ratio",
    ]
    assert len(full.metrics_collected) == 5
    assert research.metrics_collected == [
        "All metrics",
        "Loss landscape sampling",
        "Trajectory projections",
    ]
    assert len(research.metrics_collected) == 3


def test_geometry_metric_key_helpers() -> None:
    assert GeometryMetricKey.layer_grad_norm("attn") == "geometry/layer/attn/grad_norm"
    assert GeometryMetricKey.persona_position("v1") == "geometry/persona/v1/position"


def test_to_metrics_dict_full_payload(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(metrics_mod, "get_default_backend", lambda: any_backend)

    payload = GeometricTrainingMetrics(
        hessian_trace_estimate=1.25,
        top_hessian_eigenvalue=0.1,
        hessian_condition_proxy=7.5,
        gradient_variance=0.02,
        gradient_snr=12.0,
        effective_step_ratio=0.9,
        per_layer_gradient_norms={"attn.q": 2.0, "mlp.down": 1.0},
        per_layer_gradient_fractions={"attn.q": 0.6, "mlp.down": 0.4},
        parameter_divergence=3.5,
        parameter_cosine_similarity=0.95,
        refusal_distance=0.11,
        is_approaching_refusal=True,
        dare_effective_sparsity=0.87,
        dora_magnitude_change=0.22,
        dora_directional_drift=0.33,
        persona_drift_magnitude=0.44,
        circuit_breaker_severity=0.55,
    ).to_metrics_dict()

    assert payload[GeometryMetricKey.hessian_trace] == pytest.approx(1.25, abs=1e-8)
    assert payload[GeometryMetricKey.top_eigenvalue] == pytest.approx(0.1, abs=1e-8)
    assert payload[GeometryMetricKey.condition_proxy] == pytest.approx(7.5, abs=1e-8)
    assert payload[GeometryMetricKey.gradient_variance] == pytest.approx(0.02, abs=1e-8)
    assert payload[GeometryMetricKey.gradient_snr] == pytest.approx(12.0, abs=1e-8)
    assert payload[GeometryMetricKey.effective_step_ratio] == pytest.approx(0.9, abs=1e-8)
    assert payload[GeometryMetricKey.param_divergence] == pytest.approx(3.5, abs=1e-8)
    assert payload[GeometryMetricKey.param_cosine_similarity] == pytest.approx(0.95, abs=1e-8)
    assert payload[GeometryMetricKey.refusal_distance] == pytest.approx(0.11, abs=1e-8)
    assert payload[GeometryMetricKey.refusal_approaching] == pytest.approx(1.0, abs=1e-8)
    assert payload[GeometryMetricKey.dare_effective_sparsity] == pytest.approx(0.87, abs=1e-8)
    assert payload[GeometryMetricKey.dora_magnitude_change] == pytest.approx(0.22, abs=1e-8)
    assert payload[GeometryMetricKey.dora_directional_drift] == pytest.approx(0.33, abs=1e-8)
    assert payload[GeometryMetricKey.persona_overall_drift] == pytest.approx(0.44, abs=1e-8)
    assert payload[GeometryMetricKey.circuit_breaker_severity] == pytest.approx(0.55, abs=1e-8)

    assert payload[GeometryMetricKey.layer_grad_norm("attn.q")] == pytest.approx(2.0, abs=1e-8)
    assert payload[GeometryMetricKey.layer_grad_norm("mlp.down")] == pytest.approx(1.0, abs=1e-8)
    assert payload[GeometryMetricKey.layer_grad_fraction("attn.q")] == pytest.approx(0.6, abs=1e-8)
    assert payload[GeometryMetricKey.layer_grad_fraction("mlp.down")] == pytest.approx(0.4, abs=1e-8)


def test_from_progress_metrics_parsing_and_thresholds(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(metrics_mod, "get_default_backend", lambda: any_backend)

    assert GeometricTrainingMetrics.from_progress_metrics({}) is None
    assert GeometricTrainingMetrics.from_progress_metrics({"loss": 0.1}) is None

    src = GeometricTrainingMetrics(
        hessian_trace_estimate=1.5,
        top_hessian_eigenvalue=0.2,
        hessian_condition_proxy=8.0,
        gradient_variance=0.03,
        gradient_snr=9.0,
        effective_step_ratio=0.88,
        per_layer_gradient_norms={"attn.q": 2.5},
        per_layer_gradient_fractions={"attn.q": 0.7, "mlp.down": 0.1},
        parameter_divergence=4.0,
        parameter_cosine_similarity=0.8,
        refusal_distance=0.2,
        is_approaching_refusal=True,
        dare_effective_sparsity=0.9,
        dora_magnitude_change=0.15,
        dora_directional_drift=0.25,
        persona_drift_magnitude=0.35,
        circuit_breaker_severity=0.45,
    )
    metrics = src.to_metrics_dict()
    metrics[GeometryMetricKey.persona_delta("trait1")] = 0.3
    metrics[GeometryMetricKey.persona_delta("trait2")] = 0.01

    parsed = GeometricTrainingMetrics.from_progress_metrics(
        metrics,
        active_layer_threshold=0.2,
        drift_threshold=0.1,
    )
    assert parsed is not None

    assert parsed.hessian_trace_estimate == pytest.approx(1.5, abs=1e-8)
    assert parsed.top_hessian_eigenvalue == pytest.approx(0.2, abs=1e-8)
    assert parsed.hessian_condition_proxy == pytest.approx(8.0, abs=1e-8)
    assert parsed.gradient_variance == pytest.approx(0.03, abs=1e-8)
    assert parsed.gradient_snr == pytest.approx(9.0, abs=1e-8)
    assert parsed.effective_step_ratio == pytest.approx(0.88, abs=1e-8)
    assert parsed.parameter_divergence == pytest.approx(4.0, abs=1e-8)
    assert parsed.parameter_cosine_similarity == pytest.approx(0.8, abs=1e-8)
    assert parsed.refusal_distance == pytest.approx(0.2, abs=1e-8)
    assert parsed.is_approaching_refusal is True
    assert parsed.dare_effective_sparsity == pytest.approx(0.9, abs=1e-8)
    assert parsed.dora_magnitude_change == pytest.approx(0.15, abs=1e-8)
    assert parsed.dora_directional_drift == pytest.approx(0.25, abs=1e-8)
    assert parsed.persona_drift_magnitude == pytest.approx(0.35, abs=1e-8)
    assert parsed.circuit_breaker_severity == pytest.approx(0.45, abs=1e-8)

    assert parsed.per_layer_gradient_norms["attn.q"] == pytest.approx(2.5, abs=1e-8)
    assert parsed.per_layer_gradient_fractions["attn.q"] == pytest.approx(0.7, abs=1e-8)
    assert parsed.active_layers == ["attn.q"]
    assert parsed.drifting_traits == ["trait1"]

    true_case = GeometricTrainingMetrics.from_progress_metrics(
        {GeometryMetricKey.refusal_approaching: 1.0, "geometry/layer/a/grad_frac": 1.0},
        active_layer_threshold=0.0,
    )
    false_case = GeometricTrainingMetrics.from_progress_metrics(
        {GeometryMetricKey.refusal_approaching: 0.0, "geometry/layer/a/grad_frac": 1.0},
        active_layer_threshold=0.0,
    )
    missing_case = GeometricTrainingMetrics.from_progress_metrics(
        {"geometry/layer/a/grad_frac": 1.0},
        active_layer_threshold=0.0,
    )

    assert true_case is not None and true_case.is_approaching_refusal is True
    assert false_case is not None and false_case.is_approaching_refusal is False
    assert missing_case is not None and missing_case.is_approaching_refusal is None

    eps = metrics_mod.machine_epsilon(
        any_backend,
        any_backend.array([1.0], dtype=metrics_mod.precision_dtype(any_backend)),
    )
    default_threshold_metrics = {
        GeometryMetricKey.layer_grad_fraction("attn.q"): float(eps) * 2.0,
        GeometryMetricKey.layer_grad_fraction("mlp.down"): float(eps) * 0.5,
        GeometryMetricKey.persona_delta("trait_hi"): float(eps) * 2.0,
        GeometryMetricKey.persona_delta("trait_lo"): float(eps) * 0.5,
    }
    default_threshold_parsed = GeometricTrainingMetrics.from_progress_metrics(
        default_threshold_metrics,
        active_layer_threshold=None,
        drift_threshold=None,
    )
    assert default_threshold_parsed is not None
    assert "attn.q" in default_threshold_parsed.active_layers
    assert "mlp.down" not in default_threshold_parsed.active_layers
    assert "trait_hi" in default_threshold_parsed.drifting_traits
    assert "trait_lo" not in default_threshold_parsed.drifting_traits


def test_geometric_metrics_history_roundtrip_and_malformed_entries(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(metrics_mod, "get_default_backend", lambda: any_backend)

    history = GeometricMetricsHistory()
    history.append(
        1,
        GeometricTrainingMetrics(
            top_hessian_eigenvalue=None,
            gradient_snr=None,
            is_approaching_refusal=False,
        ),
    )
    history.append(
        2,
        GeometricTrainingMetrics(
            top_hessian_eigenvalue=0.1,
            gradient_snr=3.0,
            parameter_divergence=0.2,
        ),
    )

    assert history.snr_history == [(2, 3.0)]
    assert history.divergence_history == [(2, 0.2)]

    payload = history.to_payload()
    restored = GeometricMetricsHistory.from_payload(payload)
    assert len(restored.entries) == 2
    assert restored.entries[0].step == 1
    assert restored.entries[1].metrics.gradient_snr == pytest.approx(3.0, abs=1e-8)

    malformed = [
        {"step": None, "metrics": {GeometryMetricKey.gradient_snr: 1.0}},
        {"step": 3, "metrics": "not-a-dict"},
        {"step": 4, "metrics": {"loss": 0.1}},
        {"step": 5, "metrics": {GeometryMetricKey.gradient_snr: 7.0}},
    ]
    malformed_restored = GeometricMetricsHistory.from_payload(malformed)
    assert len(malformed_restored.entries) == 1
    assert malformed_restored.entries[0].step == 5
    assert malformed_restored.entries[0].metrics.gradient_snr == pytest.approx(7.0, abs=1e-8)
