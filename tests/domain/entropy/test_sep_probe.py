from __future__ import annotations

from datetime import datetime

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.sep_probe import (
    IncompatibleWeightsError,
    LayerNotFoundError,
    LayerProbeWeights,
    ProbeWeightsBundle,
    SEPProbe,
    SEPProbeError,
    SepPredictionResult,
    WeightsNotLoadedError,
)


# =============================================================================
# Helpers
# =============================================================================

HIDDEN_DIM = 8


def _make_layer_weights(
    layer: int,
    *,
    dim: int = HIDDEN_DIM,
    bias: float = 0.1,
    r2: float = 0.95,
) -> LayerProbeWeights:
    """Create LayerProbeWeights with uniform weights for testing."""
    return LayerProbeWeights(
        layer=layer,
        weights=[1.0 / dim] * dim,
        bias=bias,
        validation_r2=r2,
        train_mean=0.0,
        train_std=1.0,
    )


def _make_ready_probe(layers: list[int] | None = None) -> SEPProbe:
    """Create a SEPProbe with registered weights for the given layers."""
    if layers is None:
        layers = [0, 5, 10]
    probe = SEPProbe(hidden_dim=HIDDEN_DIM)
    weights = [_make_layer_weights(layer) for layer in layers]
    probe.register_weights(weights, model_id="test-model")
    return probe


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionInheritance:
    """Verify the exception inheritance chain."""

    def test_sep_probe_error_is_exception(self) -> None:
        assert issubclass(SEPProbeError, Exception)

    def test_weights_not_loaded_is_sep_error(self) -> None:
        assert issubclass(WeightsNotLoadedError, SEPProbeError)

    def test_incompatible_weights_is_sep_error(self) -> None:
        assert issubclass(IncompatibleWeightsError, SEPProbeError)

    def test_layer_not_found_is_sep_error(self) -> None:
        assert issubclass(LayerNotFoundError, SEPProbeError)

    def test_can_catch_subtypes_as_sep_error(self) -> None:
        with pytest.raises(SEPProbeError):
            raise WeightsNotLoadedError("test")

        with pytest.raises(SEPProbeError):
            raise IncompatibleWeightsError("test")

        with pytest.raises(SEPProbeError):
            raise LayerNotFoundError("test")


# =============================================================================
# Data classes
# =============================================================================


class TestSepPredictionResult:
    """Verify SepPredictionResult instantiation."""

    def test_instantiation(self) -> None:
        result = SepPredictionResult(
            predicted_entropy=3.14,
            layer_predictions={0: 3.0, 5: 3.28},
            latency_ms=0.5,
        )
        assert result.predicted_entropy == pytest.approx(3.14)
        assert result.layer_predictions == {0: 3.0, 5: 3.28}
        assert result.latency_ms == pytest.approx(0.5)


class TestProbeWeightsBundle:
    """Verify ProbeWeightsBundle instantiation."""

    def test_instantiation(self) -> None:
        lw = _make_layer_weights(0)
        bundle = ProbeWeightsBundle(
            model_id="test",
            layer_count=32,
            hidden_dim=HIDDEN_DIM,
            layer_weights=[lw],
            training_samples=1000,
            trained_at=datetime(2026, 1, 1),
        )
        assert bundle.model_id == "test"
        assert bundle.layer_count == 32
        assert bundle.hidden_dim == HIDDEN_DIM
        assert len(bundle.layer_weights) == 1
        assert bundle.training_samples == 1000


# =============================================================================
# SEPProbe initialization
# =============================================================================


class TestSEPProbeInit:
    """Verify SEPProbe initialization state."""

    def test_hidden_dim_stored(self) -> None:
        probe = SEPProbe(hidden_dim=64)
        assert probe.hidden_dim == 64

    def test_not_ready_initially(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        assert probe.is_ready is False

    def test_trained_model_id_none_initially(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        assert probe.trained_model_id is None

    def test_available_layers_empty_initially(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        assert probe.available_layers == set()


# =============================================================================
# register_weights
# =============================================================================


class TestSEPProbeRegisterWeights:
    """Verify register_weights makes probe ready and populates metadata."""

    def test_register_makes_ready(self) -> None:
        probe = _make_ready_probe(layers=[0, 5])
        assert probe.is_ready is True

    def test_available_layers_populated(self) -> None:
        probe = _make_ready_probe(layers=[0, 5, 10])
        assert probe.available_layers == {0, 5, 10}

    def test_trained_model_id_set(self) -> None:
        probe = _make_ready_probe()
        assert probe.trained_model_id == "test-model"


# =============================================================================
# probe_info
# =============================================================================


class TestSEPProbeInfo:
    """Verify probe_info returns sorted layer/r2 tuples."""

    def test_sorted_by_layer(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        weights = [
            _make_layer_weights(10, r2=0.90),
            _make_layer_weights(2, r2=0.80),
            _make_layer_weights(7, r2=0.95),
        ]
        probe.register_weights(weights, model_id="test")
        info = probe.probe_info()
        assert info == [(2, 0.80), (7, 0.95), (10, 0.90)]

    def test_empty_when_no_weights(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        assert probe.probe_info() == []


# =============================================================================
# predict (requires backend arrays)
# =============================================================================


class TestSEPProbePredict:
    """Verify predict method behavior."""

    def test_raises_weights_not_loaded_when_not_ready(self) -> None:
        probe = SEPProbe(hidden_dim=HIDDEN_DIM)
        with pytest.raises(WeightsNotLoadedError):
            probe.predict({})

    def test_raises_sep_error_when_no_matching_layers(self) -> None:
        probe = _make_ready_probe(layers=[0, 5])
        backend = get_default_backend()
        # Pass hidden states for layers that have no weights
        hidden_states = {
            99: backend.array([1.0] * HIDDEN_DIM),
        }
        with pytest.raises(SEPProbeError, match="No matching"):
            probe.predict(hidden_states)

    def test_predict_returns_prediction_result(self) -> None:
        probe = _make_ready_probe(layers=[0, 5])
        backend = get_default_backend()
        hidden_states = {
            0: backend.array([1.0] * HIDDEN_DIM),
            5: backend.array([2.0] * HIDDEN_DIM),
        }
        result = probe.predict(hidden_states)
        assert isinstance(result, SepPredictionResult)
        assert isinstance(result.predicted_entropy, float)
        assert 0 in result.layer_predictions
        assert 5 in result.layer_predictions
        assert result.latency_ms >= 0.0

    def test_predict_uses_subset_of_available_layers(self) -> None:
        probe = _make_ready_probe(layers=[0, 5, 10])
        backend = get_default_backend()
        # Only provide hidden states for layer 5
        hidden_states = {
            5: backend.array([1.0] * HIDDEN_DIM),
        }
        result = probe.predict(hidden_states)
        assert set(result.layer_predictions.keys()) == {5}


# =============================================================================
# predict_single_layer
# =============================================================================


class TestSEPProbePredictSingleLayer:
    """Verify predict_single_layer method."""

    def test_raises_layer_not_found_for_unknown_layer(self) -> None:
        probe = _make_ready_probe(layers=[0])
        backend = get_default_backend()
        h = backend.array([1.0] * HIDDEN_DIM)
        with pytest.raises(LayerNotFoundError):
            probe.predict_single_layer(layer=99, hidden_state=h)

    def test_returns_float_prediction(self) -> None:
        probe = _make_ready_probe(layers=[0])
        backend = get_default_backend()
        h = backend.array([1.0] * HIDDEN_DIM)
        result = probe.predict_single_layer(layer=0, hidden_state=h)
        assert isinstance(result, float)

    def test_prediction_is_deterministic(self) -> None:
        probe = _make_ready_probe(layers=[0])
        backend = get_default_backend()
        h = backend.array([3.0] * HIDDEN_DIM)
        r1 = probe.predict_single_layer(layer=0, hidden_state=h)
        r2 = probe.predict_single_layer(layer=0, hidden_state=h)
        assert r1 == pytest.approx(r2)


# =============================================================================
# reset
# =============================================================================


class TestSEPProbeReset:
    """Verify reset clears all state."""

    def test_reset_clears_readiness(self) -> None:
        probe = _make_ready_probe(layers=[0, 5])
        assert probe.is_ready is True
        probe.reset()
        assert probe.is_ready is False

    def test_reset_clears_model_id(self) -> None:
        probe = _make_ready_probe()
        probe.reset()
        assert probe.trained_model_id is None

    def test_reset_clears_available_layers(self) -> None:
        probe = _make_ready_probe(layers=[0, 5, 10])
        probe.reset()
        assert probe.available_layers == set()

    def test_predict_raises_after_reset(self) -> None:
        probe = _make_ready_probe(layers=[0])
        probe.reset()
        with pytest.raises(WeightsNotLoadedError):
            probe.predict({})
