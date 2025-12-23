"""Tests for GeometricMetricsCollector.

Tests the training geometry metrics collection system that tracks
parameter trajectories, gradient quality, and loss landscape properties.
"""

import numpy as np
import pytest

from modelcypher.core.domain.training.geometric_metrics_collector import (
    GeometricMetricsCollector,
)
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
    GeometryMetricKey,
)


class TestGeometricMetricsCollectorInit:
    """Tests for collector initialization."""

    def test_default_level_is_moderate(self):
        """Default instrumentation level should be moderate."""
        collector = GeometricMetricsCollector()
        assert collector.level == GeometricInstrumentationLevel.moderate

    def test_custom_level(self):
        """Should accept custom instrumentation level."""
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.full)
        assert collector.level == GeometricInstrumentationLevel.full

    def test_initial_state(self):
        """Collector should start with empty state."""
        collector = GeometricMetricsCollector()
        assert collector.initial_parameters is None
        assert collector.previous_parameters is None
        assert collector.last_metrics is None


class TestSetLevel:
    """Tests for set_level method."""

    def test_change_level(self):
        """Should update instrumentation level."""
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.minimal)
        collector.set_level(GeometricInstrumentationLevel.full)
        assert collector.level == GeometricInstrumentationLevel.full


class TestCaptureInitialParameters:
    """Tests for capture_initial_parameters method."""

    def test_captures_initial_and_previous(self):
        """Should set both initial and previous parameters."""
        collector = GeometricMetricsCollector()
        params = {"layer1": np.array([1.0, 2.0, 3.0])}
        collector.capture_initial_parameters(params)

        assert collector.initial_parameters is not None
        assert collector.previous_parameters is not None
        np.testing.assert_array_equal(
            collector.initial_parameters["layer1"], params["layer1"]
        )

    def test_captures_copy_not_reference(self):
        """Should clone parameters, not reference them."""
        collector = GeometricMetricsCollector()
        params = {"layer1": np.array([1.0, 2.0])}
        collector.capture_initial_parameters(params)

        # Modify original
        params["layer1"][0] = 999.0

        # Captured should be unchanged
        assert collector.initial_parameters["layer1"][0] == 1.0


class TestReset:
    """Tests for reset method."""

    def test_clears_all_state(self):
        """Reset should clear all collector state."""
        collector = GeometricMetricsCollector()
        params = {"layer1": np.array([1.0])}
        collector.capture_initial_parameters(params)

        collector.reset()

        assert collector.initial_parameters is None
        assert collector.previous_parameters is None
        assert collector.last_metrics is None
        assert len(collector.history.entries) == 0


class TestShouldComputeMetrics:
    """Tests for should_compute_metrics method."""

    def test_minimal_level_never_computes(self):
        """Minimal level should never compute metrics."""
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.minimal)
        assert not collector.should_compute_metrics(0)
        assert not collector.should_compute_metrics(1)
        assert not collector.should_compute_metrics(100)

    def test_moderate_level_interval(self):
        """Moderate level should compute at configured intervals."""
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.moderate)
        interval = GeometricInstrumentationLevel.moderate.hessian_computation_interval

        assert collector.should_compute_metrics(0)  # step 0
        assert not collector.should_compute_metrics(1)
        assert collector.should_compute_metrics(interval)


class TestComputeMetrics:
    """Tests for compute_metrics method."""

    @pytest.fixture
    def collector(self):
        """Create collector with initial parameters."""
        c = GeometricMetricsCollector(level=GeometricInstrumentationLevel.moderate)
        c.capture_initial_parameters({
            "layer1": np.array([1.0, 0.0]),
            "layer2": np.array([0.0, 1.0]),
        })
        return c

    def test_computes_per_layer_stats(self):
        """Should compute per-layer gradient statistics (full level only)."""
        # Per-layer metrics only computed at full/research level
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.full)
        collector.capture_initial_parameters({
            "layer1": np.array([1.0, 0.0]),
            "layer2": np.array([0.0, 1.0]),
        })
        params = {
            "layer1": np.array([1.5, 0.5]),
            "layer2": np.array([0.5, 1.5]),
        }
        gradients = {
            "layer1": np.array([3.0, 4.0]),  # norm = 5
            "layer2": np.array([0.0, 5.0]),  # norm = 5
        }
        metrics = collector.compute_metrics(params, gradients, learning_rate=0.01)

        assert metrics.per_layer_gradient_norms["layer1"] == pytest.approx(5.0)
        assert metrics.per_layer_gradient_norms["layer2"] == pytest.approx(5.0)

    def test_computes_trajectory_divergence(self, collector):
        """Should compute divergence from initial parameters."""
        # Move parameters away from initial
        params = {
            "layer1": np.array([4.0, 3.0]),  # delta = [3, 3], norm = sqrt(18)
            "layer2": np.array([0.0, 1.0]),  # unchanged
        }
        gradients = {
            "layer1": np.array([1.0, 1.0]),
            "layer2": np.array([1.0, 1.0]),
        }
        metrics = collector.compute_metrics(params, gradients, learning_rate=0.01)

        assert metrics.parameter_divergence is not None
        expected_divergence = np.sqrt(3**2 + 3**2)  # sqrt(18)
        assert metrics.parameter_divergence == pytest.approx(expected_divergence, rel=0.01)

    def test_computes_effective_step_ratio(self, collector):
        """Should compute effective step ratio."""
        # First compute to set previous_parameters
        params1 = {
            "layer1": np.array([1.0, 0.0]),
            "layer2": np.array([0.0, 1.0]),
        }
        gradients = {
            "layer1": np.array([1.0, 0.0]),
            "layer2": np.array([0.0, 1.0]),
        }
        collector.compute_metrics(params1, gradients, learning_rate=0.1)

        # Second compute with known step
        params2 = {
            "layer1": np.array([1.1, 0.0]),  # step = 0.1 = lr * grad
            "layer2": np.array([0.0, 1.1]),
        }
        metrics = collector.compute_metrics(params2, gradients, learning_rate=0.1)

        assert metrics.effective_step_ratio is not None
        assert metrics.effective_step_ratio == pytest.approx(1.0, rel=0.1)

    def test_updates_last_metrics(self, collector):
        """Should store last computed metrics."""
        params = {"layer1": np.array([1.0, 0.0]), "layer2": np.array([0.0, 1.0])}
        gradients = {"layer1": np.array([1.0, 1.0]), "layer2": np.array([1.0, 1.0])}
        metrics = collector.compute_metrics(params, gradients, learning_rate=0.01)

        assert collector.last_metrics is metrics

    def test_updates_previous_parameters(self, collector):
        """Should update previous_parameters after computation."""
        params = {"layer1": np.array([5.0, 5.0]), "layer2": np.array([5.0, 5.0])}
        gradients = {"layer1": np.array([1.0, 1.0]), "layer2": np.array([1.0, 1.0])}
        collector.compute_metrics(params, gradients, learning_rate=0.01)

        np.testing.assert_array_equal(collector.previous_parameters["layer1"], params["layer1"])


class TestComputeGradientQuality:
    """Tests for compute_gradient_quality method."""

    def test_empty_input_returns_none(self):
        """Empty gradients should return None."""
        collector = GeometricMetricsCollector()
        result = collector.compute_gradient_quality([])
        assert result is None

    def test_single_sample_returns_none(self):
        """Single sample should return None."""
        collector = GeometricMetricsCollector()
        sample = {"layer1": np.array([1.0, 2.0])}
        result = collector.compute_gradient_quality([sample])
        assert result is None

    def test_returns_variance_and_snr(self):
        """Should return (variance, snr) tuple."""
        collector = GeometricMetricsCollector()
        grad1 = {"layer1": np.array([1.0, 0.0])}
        grad2 = {"layer1": np.array([0.0, 1.0])}
        result = collector.compute_gradient_quality([grad1, grad2])

        assert result is not None
        variance, snr = result
        assert variance > 0
        assert snr > 0


class TestRecordInHistory:
    """Tests for record_in_history method."""

    def test_records_metrics(self):
        """Should record metrics at specified step."""
        collector = GeometricMetricsCollector()
        collector.capture_initial_parameters({"layer1": np.array([1.0])})

        params = {"layer1": np.array([1.0])}
        gradients = {"layer1": np.array([0.5])}
        metrics = collector.compute_metrics(params, gradients, learning_rate=0.01)
        collector.record_in_history(step=10, metrics=metrics)

        history = collector.get_history()
        assert len(history.entries) == 1
        assert history.entries[0].step == 10


class TestComputeLightweightMetrics:
    """Tests for compute_lightweight_metrics method."""

    def test_includes_top_layer_fractions(self):
        """Should include top 5 layer gradient fractions."""
        collector = GeometricMetricsCollector()
        collector.capture_initial_parameters({
            "layers.0.attn": np.array([1.0]),
            "layers.1.mlp": np.array([1.0]),
        })

        params = {
            "layers.0.attn": np.array([1.0]),
            "layers.1.mlp": np.array([1.0]),
        }
        gradients = {
            "layers.0.attn": np.array([3.0, 4.0]),  # norm = 5
            "layers.1.mlp": np.array([12.0, 0.0]),  # norm = 12
        }
        result = collector.compute_lightweight_metrics(params, gradients, learning_rate=0.01)

        # Should have layer fraction keys
        assert any("L1.mlp" in key for key in result.keys())

    def test_includes_trajectory_metrics_when_available(self):
        """Should include trajectory metrics after initial capture."""
        collector = GeometricMetricsCollector()
        collector.capture_initial_parameters({"layer1": np.array([1.0, 0.0])})

        params = {"layer1": np.array([2.0, 1.0])}  # diverged from initial
        gradients = {"layer1": np.array([1.0, 1.0])}
        result = collector.compute_lightweight_metrics(params, gradients, learning_rate=0.01)

        assert GeometryMetricKey.param_divergence in result
        assert GeometryMetricKey.param_cosine_similarity in result


class TestShortenLayerName:
    """Tests for shorten_layer_name static method."""

    def test_shortens_layers_prefix(self):
        """Should shorten 'layers.' to 'L'."""
        result = GeometricMetricsCollector.shorten_layer_name("layers.5.attention")
        assert result == "L5.attn"

    def test_shortens_attention(self):
        """Should shorten 'attention' to 'attn'."""
        result = GeometricMetricsCollector.shorten_layer_name("model.attention.weight")
        assert result == "model.attn.weight"

    def test_shortens_self_attn(self):
        """Should shorten 'self_attn' to 'attn'."""
        result = GeometricMetricsCollector.shorten_layer_name("layers.0.self_attn.q_proj")
        assert result == "L0.attn.q_proj"

    def test_shortens_lora_suffixes(self):
        """Should shorten lora_a/lora_b to a/b."""
        result = GeometricMetricsCollector.shorten_layer_name("layers.0.mlp.lora_a")
        assert result == "L0.mlp.a"

    def test_truncates_long_names(self):
        """Should truncate names longer than 30 chars."""
        long_name = "a" * 50
        result = GeometricMetricsCollector.shorten_layer_name(long_name)
        assert len(result) <= 30


class TestCloneParams:
    """Tests for _clone_params static method."""

    def test_creates_deep_copy(self):
        """Should create deep copy of parameters."""
        params = {"layer1": np.array([1.0, 2.0, 3.0])}
        cloned = GeometricMetricsCollector._clone_params(params)

        # Modify original
        params["layer1"][0] = 999.0

        # Clone should be unchanged
        assert cloned["layer1"][0] == 1.0

    def test_handles_non_ndarray(self):
        """Should handle non-ndarray inputs by converting."""
        params = {"layer1": [1.0, 2.0, 3.0]}  # list, not ndarray
        cloned = GeometricMetricsCollector._clone_params(params)

        assert isinstance(cloned["layer1"], np.ndarray)
        np.testing.assert_array_equal(cloned["layer1"], [1.0, 2.0, 3.0])


class TestIntegration:
    """Integration tests for the full metrics collection workflow."""

    def test_full_training_loop_simulation(self):
        """Simulate metrics collection during training."""
        collector = GeometricMetricsCollector(level=GeometricInstrumentationLevel.moderate)

        # Initial parameters
        initial_params = {
            "layer.weight": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }
        collector.capture_initial_parameters(initial_params)

        # Simulate several training steps
        params = initial_params.copy()
        params["layer.weight"] = params["layer.weight"].copy()
        lr = 0.1

        for step in range(20):
            # Fake gradient (random direction)
            gradients = {
                "layer.weight": np.random.randn(2, 2).astype(np.float32) * 0.1,
            }

            # Update params (simple SGD)
            params["layer.weight"] = params["layer.weight"] - lr * gradients["layer.weight"]

            # Compute and record metrics
            if collector.should_compute_metrics(step):
                metrics = collector.compute_metrics(params, gradients, lr)
                collector.record_in_history(step, metrics)

        # Should have recorded metrics
        history = collector.get_history()
        assert len(history.entries) > 0

        # Last metrics should show divergence from initial
        last = collector.get_last_metrics()
        assert last is not None
        assert last.parameter_divergence > 0
