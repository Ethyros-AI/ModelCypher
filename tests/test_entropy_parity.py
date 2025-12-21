"""
Unit tests for entropy domain parity modules.

Tests:
- EntropyTracker session management and state classification
- HiddenStateExtractor layer targeting
- SEPProbe configuration
"""
import pytest
import mlx.core as mx
from modelcypher.core.domain.entropy import (
    EntropyTracker,
    EntropyTrackerConfig,
    ModelState,
    EntropySample,
    EntropyLevel,
    StateTransition,
    HiddenStateExtractor,
    ExtractorConfig,
    SEPProbe,
    SEPProbeConfig,
    ModelStateClassifier,
    ClassifierThresholds,
)


class TestModelState:
    """Tests for ModelState enum."""

    def test_all_states_exist(self):
        states = list(ModelState)
        assert len(states) == 6
        assert ModelState.CONFIDENT in states
        assert ModelState.DISTRESSED in states
        assert ModelState.HALTED in states

    def test_severity_ordering(self):
        """Severity should increase from confident to halted."""
        assert ModelState.CONFIDENT.severity_level < ModelState.NOMINAL.severity_level
        assert ModelState.NOMINAL.severity_level < ModelState.UNCERTAIN.severity_level
        assert ModelState.DISTRESSED.severity_level < ModelState.HALTED.severity_level

    def test_requires_caution(self):
        """Confident and nominal should not require caution."""
        assert not ModelState.CONFIDENT.requires_caution
        assert not ModelState.NOMINAL.requires_caution
        assert ModelState.UNCERTAIN.requires_caution
        assert ModelState.DISTRESSED.requires_caution
        assert ModelState.HALTED.requires_caution


class TestEntropySample:
    """Tests for EntropySample dataclass."""

    def test_best_entropy_estimate_prefers_sep(self):
        sample = EntropySample(
            logit_entropy=3.0,
            sep_entropy=0.5,
        )
        assert sample.best_entropy_estimate == 0.5

    def test_best_entropy_estimate_fallback(self):
        sample = EntropySample(logit_entropy=3.0)
        assert sample.best_entropy_estimate == 3.0

    def test_entropy_level_classification(self):
        low = EntropySample(logit_entropy=1.0)
        moderate = EntropySample(logit_entropy=2.0)
        high = EntropySample(logit_entropy=4.0)

        assert low.entropy_level() == EntropyLevel.LOW
        assert moderate.entropy_level() == EntropyLevel.MODERATE
        assert high.entropy_level() == EntropyLevel.HIGH

    def test_circuit_breaker(self):
        normal = EntropySample(logit_entropy=2.0)
        danger = EntropySample(logit_entropy=5.0)

        assert not normal.should_trip_circuit_breaker()
        assert danger.should_trip_circuit_breaker()


class TestEntropyTracker:
    """Tests for EntropyTracker session management."""

    def test_session_lifecycle(self):
        tracker = EntropyTracker()
        assert not tracker.is_session_active

        tracker.start_session()
        assert tracker.is_session_active
        assert tracker.current_model_state == ModelState.NOMINAL

        sample = tracker.end_session()
        assert not tracker.is_session_active

    def test_state_classification(self):
        classifier = ModelStateClassifier()

        # Low entropy, high variance = confident
        state = classifier.classify(entropy=1.0, variance=0.8)
        assert state == ModelState.CONFIDENT

        # High entropy, low variance = distressed
        state = classifier.classify(entropy=4.0, variance=0.1)
        assert state == ModelState.DISTRESSED

        # High entropy, moderate variance = uncertain
        state = classifier.classify(entropy=3.5, variance=0.5)
        assert state == ModelState.UNCERTAIN


class TestHiddenStateExtractor:
    """Tests for HiddenStateExtractor."""

    def test_layer_targeting_presets(self):
        # 32-layer model
        config = ExtractorConfig.for_sep_probe(32)
        assert 24 in config.target_layers
        assert 28 in config.target_layers

        config = ExtractorConfig.for_refusal_direction(32)
        assert 13 in config.target_layers  # ~40%
        assert 19 in config.target_layers  # ~60%

    def test_session_management(self):
        extractor = HiddenStateExtractor.for_sep_probe(32)
        assert not extractor.is_active

        extractor.start_session()
        assert extractor.is_active

        summary = extractor.end_session()
        assert not extractor.is_active
        assert summary.total_captures == 0

    def test_state_capture(self):
        config = ExtractorConfig(target_layers={25, 26})
        extractor = HiddenStateExtractor(config)
        extractor.start_session()

        # Capture state for layer 25
        hidden = mx.random.normal((1, 4096))
        extractor.capture(hidden, layer=25, token_index=0)

        states = extractor.extracted_states()
        assert 25 in states
        assert 26 not in states

        summary = extractor.end_session()
        assert summary.total_captures == 1


class TestSEPProbe:
    """Tests for SEPProbe configuration."""

    def test_default_configuration(self):
        config = SEPProbeConfig.default()
        assert config.layer_count == 32
        assert config.hidden_dim == 4096
        assert config.use_ensemble

    def test_target_layers(self):
        config = SEPProbeConfig(layer_count=32)
        targets = config.target_layers
        assert 24 in targets  # 75% of 32
        assert 28 in targets  # ~87.5% of 32

    def test_probe_not_ready_without_weights(self):
        probe = SEPProbe()
        assert not probe.is_ready


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_escalation_detection(self):
        transition = StateTransition(
            from_state=ModelState.NOMINAL,
            to_state=ModelState.DISTRESSED,
            token_index=10,
            entropy=4.0,
            variance=0.1,
        )
        assert transition.is_escalation
        assert not transition.is_recovery
        assert transition.severity_delta > 0

    def test_recovery_detection(self):
        transition = StateTransition(
            from_state=ModelState.DISTRESSED,
            to_state=ModelState.NOMINAL,
            token_index=20,
            entropy=1.5,
            variance=0.5,
        )
        assert transition.is_recovery
        assert not transition.is_escalation
        assert transition.severity_delta < 0
