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

"""Tests for geometric training metrics and instrumentation levels."""

import sys

import pytest

from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
    GeometricMetricsHistory,
    GeometricTrainingMetrics,
    GeometryMetricKey,
    MetricEntry,
)


class TestGeometricInstrumentationLevel:
    """Tests for GeometricInstrumentationLevel enum."""

    def test_minimal_value(self):
        assert GeometricInstrumentationLevel.minimal.value == "minimal"

    def test_moderate_value(self):
        assert GeometricInstrumentationLevel.moderate.value == "moderate"

    def test_full_value(self):
        assert GeometricInstrumentationLevel.full.value == "full"

    def test_research_value(self):
        assert GeometricInstrumentationLevel.research.value == "research"

    def test_minimal_description(self):
        desc = GeometricInstrumentationLevel.minimal.description
        assert "gradient norms" in desc.lower()

    def test_moderate_description(self):
        desc = GeometricInstrumentationLevel.moderate.description
        assert "curvature" in desc.lower()

    def test_full_description(self):
        desc = GeometricInstrumentationLevel.full.description
        assert "all metrics" in desc.lower()

    def test_research_description(self):
        desc = GeometricInstrumentationLevel.research.description
        assert "loss landscape" in desc.lower()


class TestHessianComputationInterval:
    """Tests for hessian_computation_interval property."""

    def test_minimal_never_computes(self):
        interval = GeometricInstrumentationLevel.minimal.hessian_computation_interval
        assert interval == sys.maxsize

    def test_moderate_computes_every_10(self):
        interval = GeometricInstrumentationLevel.moderate.hessian_computation_interval
        assert interval == 10

    def test_full_computes_every_step(self):
        interval = GeometricInstrumentationLevel.full.hessian_computation_interval
        assert interval == 1

    def test_research_computes_every_step(self):
        interval = GeometricInstrumentationLevel.research.hessian_computation_interval
        assert interval == 1


class TestComputeFlags:
    """Tests for compute flag properties."""

    def test_minimal_no_per_layer(self):
        assert GeometricInstrumentationLevel.minimal.compute_per_layer_metrics is False

    def test_moderate_no_per_layer(self):
        assert GeometricInstrumentationLevel.moderate.compute_per_layer_metrics is False

    def test_full_has_per_layer(self):
        assert GeometricInstrumentationLevel.full.compute_per_layer_metrics is True

    def test_research_has_per_layer(self):
        assert GeometricInstrumentationLevel.research.compute_per_layer_metrics is True

    def test_only_research_has_loss_landscape(self):
        assert GeometricInstrumentationLevel.minimal.compute_loss_landscape is False
        assert GeometricInstrumentationLevel.moderate.compute_loss_landscape is False
        assert GeometricInstrumentationLevel.full.compute_loss_landscape is False
        assert GeometricInstrumentationLevel.research.compute_loss_landscape is True

    def test_minimal_no_top_eigenvalue(self):
        assert GeometricInstrumentationLevel.minimal.compute_top_eigenvalue is False

    def test_non_minimal_have_top_eigenvalue(self):
        assert GeometricInstrumentationLevel.moderate.compute_top_eigenvalue is True
        assert GeometricInstrumentationLevel.full.compute_top_eigenvalue is True
        assert GeometricInstrumentationLevel.research.compute_top_eigenvalue is True


class TestMetricsCollected:
    """Tests for metrics_collected property."""

    def test_minimal_metrics(self):
        metrics = GeometricInstrumentationLevel.minimal.metrics_collected
        assert "Gradient norms" in metrics
        assert "Parameter divergence" in metrics
        assert len(metrics) == 2

    def test_moderate_metrics(self):
        metrics = GeometricInstrumentationLevel.moderate.metrics_collected
        assert "Curvature estimation" in metrics[2]
        assert len(metrics) == 3

    def test_full_metrics(self):
        metrics = GeometricInstrumentationLevel.full.metrics_collected
        assert "Per-layer statistics" in metrics
        assert len(metrics) == 5

    def test_research_metrics(self):
        metrics = GeometricInstrumentationLevel.research.metrics_collected
        assert "All metrics" in metrics
        assert "Loss landscape sampling" in metrics


class TestGeometryMetricKey:
    """Tests for GeometryMetricKey constants and methods."""

    def test_hessian_trace_key(self):
        assert GeometryMetricKey.hessian_trace == "geometry/hessian_trace"

    def test_top_eigenvalue_key(self):
        assert GeometryMetricKey.top_eigenvalue == "geometry/top_eigenvalue"

    def test_condition_proxy_key(self):
        assert GeometryMetricKey.condition_proxy == "geometry/condition_proxy"

    def test_flatness_score_key(self):
        assert GeometryMetricKey.flatness_score == "geometry/flatness_score"

    def test_gradient_variance_key(self):
        assert GeometryMetricKey.gradient_variance == "geometry/gradient_variance"

    def test_gradient_snr_key(self):
        assert GeometryMetricKey.gradient_snr == "geometry/gradient_snr"

    def test_circuit_breaker_keys(self):
        assert GeometryMetricKey.circuit_breaker_severity == "geometry/circuit_breaker_severity"

    def test_layer_grad_norm_method(self):
        key = GeometryMetricKey.layer_grad_norm("layer_0")
        assert key == "geometry/layer/layer_0/grad_norm"

    def test_layer_grad_fraction_method(self):
        key = GeometryMetricKey.layer_grad_fraction("layer_5")
        assert key == "geometry/layer/layer_5/grad_frac"

    def test_persona_position_method(self):
        key = GeometryMetricKey.persona_position("formality")
        assert key == "geometry/persona/formality/position"

    def test_persona_delta_method(self):
        key = GeometryMetricKey.persona_delta("humor")
        assert key == "geometry/persona/humor/delta"


class TestGeometricTrainingMetrics:
    """Tests for GeometricTrainingMetrics dataclass."""

    def test_default_values(self):
        metrics = GeometricTrainingMetrics()
        assert metrics.hessian_trace_estimate is None
        assert metrics.top_hessian_eigenvalue is None
        assert metrics.per_layer_gradient_norms == {}
        assert metrics.active_layers == []

    def test_with_values(self):
        metrics = GeometricTrainingMetrics(
            hessian_trace_estimate=0.5,
            top_hessian_eigenvalue=0.1,
            gradient_snr=10.0,
        )
        assert metrics.hessian_trace_estimate == 0.5
        assert metrics.top_hessian_eigenvalue == 0.1
        assert metrics.gradient_snr == 10.0

    def test_flatness_score_with_positive_eigenvalue(self):
        metrics = GeometricTrainingMetrics(top_hessian_eigenvalue=0.01)
        score = metrics.flatness_score
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_flatness_score_none_when_no_eigenvalue(self):
        metrics = GeometricTrainingMetrics()
        assert metrics.flatness_score is None

    def test_flatness_score_none_when_zero_eigenvalue(self):
        metrics = GeometricTrainingMetrics(top_hessian_eigenvalue=0.0)
        assert metrics.flatness_score is None

    def test_flatness_score_none_when_negative_eigenvalue(self):
        metrics = GeometricTrainingMetrics(top_hessian_eigenvalue=-0.1)
        assert metrics.flatness_score is None

    def test_flatness_score_clamped_to_0_1(self):
        # Very small eigenvalue should give high flatness
        metrics_small = GeometricTrainingMetrics(top_hessian_eigenvalue=0.0001)
        assert metrics_small.flatness_score is not None
        assert 0.0 <= metrics_small.flatness_score <= 1.0

        # Very large eigenvalue should give low flatness
        metrics_large = GeometricTrainingMetrics(top_hessian_eigenvalue=1000.0)
        assert metrics_large.flatness_score is not None
        assert 0.0 <= metrics_large.flatness_score <= 1.0


class TestToMetricsDict:
    """Tests for to_metrics_dict() method."""

    def test_empty_metrics_returns_empty_dict(self):
        metrics = GeometricTrainingMetrics()
        result = metrics.to_metrics_dict()
        assert result == {}

    def test_includes_hessian_trace(self):
        metrics = GeometricTrainingMetrics(hessian_trace_estimate=0.5)
        result = metrics.to_metrics_dict()
        assert GeometryMetricKey.hessian_trace in result
        assert result[GeometryMetricKey.hessian_trace] == 0.5

    def test_includes_gradient_snr(self):
        metrics = GeometricTrainingMetrics(gradient_snr=15.0)
        result = metrics.to_metrics_dict()
        assert GeometryMetricKey.gradient_snr in result
        assert result[GeometryMetricKey.gradient_snr] == 15.0

    def test_includes_per_layer_norms(self):
        metrics = GeometricTrainingMetrics(
            per_layer_gradient_norms={"layer_0": 0.1, "layer_1": 0.2}
        )
        result = metrics.to_metrics_dict()
        assert "geometry/layer/layer_0/grad_norm" in result
        assert result["geometry/layer/layer_0/grad_norm"] == 0.1

    def test_includes_per_layer_fractions(self):
        metrics = GeometricTrainingMetrics(
            per_layer_gradient_fractions={"layer_0": 0.3}
        )
        result = metrics.to_metrics_dict()
        assert "geometry/layer/layer_0/grad_frac" in result

    def test_includes_flatness_score(self):
        metrics = GeometricTrainingMetrics(top_hessian_eigenvalue=0.01)
        result = metrics.to_metrics_dict()
        assert GeometryMetricKey.flatness_score in result


class TestFromProgressMetrics:
    """Tests for from_progress_metrics() class method."""

    def test_empty_dict_returns_none(self):
        result = GeometricTrainingMetrics.from_progress_metrics({})
        assert result is None

    def test_no_geometry_keys_returns_none(self):
        result = GeometricTrainingMetrics.from_progress_metrics({"loss": 0.5})
        assert result is None

    def test_parses_hessian_trace(self):
        metrics_dict = {"geometry/hessian_trace": 0.5}
        result = GeometricTrainingMetrics.from_progress_metrics(metrics_dict)
        assert result is not None
        assert result.hessian_trace_estimate == 0.5

    def test_parses_layer_norms(self):
        metrics_dict = {
            "geometry/layer/layer_0/grad_norm": 0.1,
            "geometry/layer/layer_1/grad_norm": 0.2,
        }
        result = GeometricTrainingMetrics.from_progress_metrics(metrics_dict)
        assert result is not None
        assert "layer_0" in result.per_layer_gradient_norms
        assert result.per_layer_gradient_norms["layer_0"] == 0.1

    def test_parses_layer_fractions(self):
        metrics_dict = {
            "geometry/layer/layer_0/grad_frac": 0.3,
        }
        result = GeometricTrainingMetrics.from_progress_metrics(metrics_dict)
        assert result is not None
        assert "layer_0" in result.per_layer_gradient_fractions

    def test_active_layers_default_any_nonzero(self):
        """Without threshold, any nonzero gradient fraction is active."""
        metrics_dict = {
            "geometry/layer/layer_0/grad_frac": 0.1,
            "geometry/layer/layer_1/grad_frac": 0.01,
        }
        result = GeometricTrainingMetrics.from_progress_metrics(metrics_dict)
        assert result is not None
        # Both layers have nonzero contribution, so both are active
        assert "layer_0" in result.active_layers
        assert "layer_1" in result.active_layers

    def test_active_layers_with_threshold(self):
        """With explicit threshold, only layers above it are active."""
        metrics_dict = {
            "geometry/layer/layer_0/grad_frac": 0.1,  # Active (> 0.05)
            "geometry/layer/layer_1/grad_frac": 0.01,  # Not active (< 0.05)
        }
        result = GeometricTrainingMetrics.from_progress_metrics(
            metrics_dict, active_layer_threshold=0.05
        )
        assert result is not None
        assert "layer_0" in result.active_layers
        assert "layer_1" not in result.active_layers


class TestMetricEntry:
    """Tests for MetricEntry dataclass."""

    def test_required_fields(self):
        metrics = GeometricTrainingMetrics(gradient_snr=5.0)
        entry = MetricEntry(step=100, metrics=metrics)
        assert entry.step == 100
        assert entry.metrics.gradient_snr == 5.0


class TestGeometricMetricsHistory:
    """Tests for GeometricMetricsHistory class."""

    def test_empty_history(self):
        history = GeometricMetricsHistory()
        assert history.entries == []

    def test_append_entry(self):
        history = GeometricMetricsHistory()
        metrics = GeometricTrainingMetrics(gradient_snr=5.0)
        history.append(step=100, metrics=metrics)
        assert len(history.entries) == 1
        assert history.entries[0].step == 100

    def test_flatness_history(self):
        history = GeometricMetricsHistory()
        history.append(100, GeometricTrainingMetrics(top_hessian_eigenvalue=0.01))
        history.append(200, GeometricTrainingMetrics(top_hessian_eigenvalue=0.1))
        history.append(300, GeometricTrainingMetrics())  # No eigenvalue

        flatness = history.flatness_history
        assert len(flatness) == 2
        assert flatness[0][0] == 100
        assert flatness[1][0] == 200

    def test_snr_history(self):
        history = GeometricMetricsHistory()
        history.append(100, GeometricTrainingMetrics(gradient_snr=10.0))
        history.append(200, GeometricTrainingMetrics(gradient_snr=15.0))

        snr = history.snr_history
        assert len(snr) == 2
        assert snr[0] == (100, 10.0)
        assert snr[1] == (200, 15.0)

    def test_divergence_history(self):
        history = GeometricMetricsHistory()
        history.append(100, GeometricTrainingMetrics(parameter_divergence=0.5))
        history.append(200, GeometricTrainingMetrics())  # No divergence

        divergence = history.divergence_history
        assert len(divergence) == 1
        assert divergence[0] == (100, 0.5)

    def test_to_payload(self):
        history = GeometricMetricsHistory()
        history.append(100, GeometricTrainingMetrics(gradient_snr=5.0))

        payload = history.to_payload()
        assert len(payload) == 1
        assert payload[0]["step"] == 100
        assert "metrics" in payload[0]

    def test_from_payload_empty(self):
        history = GeometricMetricsHistory.from_payload([])
        assert len(history.entries) == 0

    def test_from_payload_roundtrip(self):
        original = GeometricMetricsHistory()
        original.append(100, GeometricTrainingMetrics(gradient_snr=5.0))
        original.append(200, GeometricTrainingMetrics(gradient_snr=10.0))

        payload = original.to_payload()
        restored = GeometricMetricsHistory.from_payload(payload)

        assert len(restored.entries) == 2
        assert restored.entries[0].step == 100
        assert restored.entries[1].step == 200
