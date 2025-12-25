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

"""
Unit tests for entropy domain parity modules (requires MLX).

Tests:
- EntropyTracker session management and state classification
- HiddenStateExtractor layer targeting
- SEPProbe configuration
"""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.entropy import (
    CalibratedBaseline,
    EntropySample,
    EntropyTracker,
    EntropyTransition,
    ExtractorConfig,
    HiddenStateExtractor,
    SEPProbe,
    SEPProbeConfig,
)
from modelcypher.core.domain.entropy.model_state_classifier import (
    ModelStateClassifier,
)


def _create_test_baseline() -> CalibratedBaseline:
    """Create a calibrated baseline for testing.

    Uses values that make sense for testing:
    - Mean entropy 2.5 (moderate)
    - Std dev 1.0 (reasonable spread)
    - Percentiles create meaningful thresholds at 1.8, 3.2, 4.5
    """
    return CalibratedBaseline(
        mean=2.5,
        std_dev=1.0,
        percentile_25=1.8,  # Below this is "low"
        percentile_75=3.2,  # Above this is "high"
        percentile_95=4.5,  # Circuit breaker
        vocab_size=32768,
        model_id="test-model",
        sample_count=100,
    )


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
        """EntropySample uses baseline-relative methods for entropy level."""
        baseline = _create_test_baseline()

        # Low entropy (below 25th percentile = 1.8)
        low = EntropySample(logit_entropy=1.0)
        # Moderate entropy (between 25th and 75th percentile)
        moderate = EntropySample(logit_entropy=2.5)
        # High entropy (above 75th percentile = 3.2)
        high = EntropySample(logit_entropy=4.0)

        # Low entropy - is_low_entropy = True, is_high_entropy = False
        assert low.is_low_entropy(baseline)
        assert not low.is_high_entropy(baseline)

        # Moderate entropy - neither low nor high (between percentiles)
        assert not moderate.is_low_entropy(baseline)
        assert not moderate.is_high_entropy(baseline)

        # High entropy - is_low_entropy = False, is_high_entropy = True
        assert not high.is_low_entropy(baseline)
        assert high.is_high_entropy(baseline)

    def test_circuit_breaker(self):
        baseline = _create_test_baseline()

        # Normal entropy (below 95th percentile = 4.5)
        normal = EntropySample(logit_entropy=2.0)
        # Danger entropy (above 95th percentile = 4.5)
        danger = EntropySample(logit_entropy=5.0)

        assert not normal.should_trip_circuit_breaker(baseline)
        assert danger.should_trip_circuit_breaker(baseline)


class TestEntropyTracker:
    """Tests for EntropyTracker session management."""

    def test_session_lifecycle(self):
        baseline = _create_test_baseline()
        tracker = EntropyTracker(baseline=baseline)
        assert not tracker.is_session_active

        tracker.start_session()
        assert tracker.is_session_active
        # Initial entropy/variance are 0.0 (no samples yet)
        assert tracker.current_entropy == 0.0
        assert tracker.current_variance == 0.0

        tracker.end_session()
        assert not tracker.is_session_active

    def test_state_classification(self):
        """ModelStateClassifier uses baseline-relative z-scores for classification."""
        baseline = _create_test_baseline()
        classifier = ModelStateClassifier(baseline)

        # Low entropy (z-score < -1) = confident
        # With mean=2.5, std=1.0, entropy=1.0 gives z=-1.5
        assert classifier.is_confident(entropy=1.0, variance=0.8)
        assert not classifier.requires_caution(entropy=1.0, variance=0.8)

        # High entropy + low variance = distressed
        # With mean=2.5, std=1.0, entropy=5.0 gives z=+2.5
        assert classifier.is_distressed(entropy=5.0, variance=0.1)
        assert classifier.requires_caution(entropy=5.0, variance=0.1)

        # High entropy + moderate variance = uncertain (not distressed)
        # With mean=2.5, std=1.0, entropy=4.5 gives z=+2.0
        assert classifier.is_uncertain(entropy=4.5, variance=0.5)
        assert not classifier.is_distressed(entropy=4.5, variance=0.5)  # variance too high
        assert classifier.requires_caution(entropy=4.5, variance=0.5)


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


class TestEntropyTransition:
    """Tests for EntropyTransition dataclass."""

    def test_escalation_detection(self):
        """Escalation: z-score increases by more than 1σ."""
        baseline = _create_test_baseline()

        # Compute z-scores for the transition
        # With mean=2.5, std=1.0:
        # from_entropy=2.0 → z=-0.5
        # to_entropy=4.0 → z=+1.5
        # z_score_delta = 1.5 - (-0.5) = 2.0 > 1.0 → escalation
        transition = EntropyTransition(
            from_entropy=2.0,
            from_variance=0.5,
            to_entropy=4.0,
            to_variance=0.1,
            from_z_score=baseline.z_score(2.0),  # -0.5
            to_z_score=baseline.z_score(4.0),  # +1.5
            token_index=10,
        )
        assert transition.is_escalation
        assert not transition.is_recovery
        assert transition.z_score_delta > 1.0

    def test_recovery_detection(self):
        """Recovery: z-score decreases by more than 1σ."""
        baseline = _create_test_baseline()

        # With mean=2.5, std=1.0:
        # from_entropy=4.0 → z=+1.5
        # to_entropy=1.5 → z=-1.0
        # z_score_delta = -1.0 - 1.5 = -2.5 < -1.0 → recovery
        transition = EntropyTransition(
            from_entropy=4.0,
            from_variance=0.1,
            to_entropy=1.5,
            to_variance=0.5,
            from_z_score=baseline.z_score(4.0),  # +1.5
            to_z_score=baseline.z_score(1.5),  # -1.0
            token_index=20,
        )
        assert transition.is_recovery
        assert not transition.is_escalation
        assert transition.z_score_delta < -1.0
