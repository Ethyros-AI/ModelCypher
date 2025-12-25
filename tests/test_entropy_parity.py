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
    EntropyLevel,
    EntropySample,
    EntropyTracker,
    EntropyTransition,
    ExtractorConfig,
    HiddenStateExtractor,
    ModelStateClassifier,
    SEPProbe,
    SEPProbeConfig,
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
        """EntropySample uses boolean properties for entropy level."""
        low = EntropySample(logit_entropy=1.0)
        moderate = EntropySample(logit_entropy=2.0)
        high = EntropySample(logit_entropy=4.0)

        # Low entropy (< 2.0) - is_low_entropy = True, is_high_entropy = False
        assert low.is_low_entropy
        assert not low.is_high_entropy

        # Moderate entropy (2.0 - 3.0) - neither low nor high
        assert not moderate.is_low_entropy
        assert not moderate.is_high_entropy

        # High entropy (> 3.0) - is_low_entropy = False, is_high_entropy = True
        assert not high.is_low_entropy
        assert high.is_high_entropy

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
        # Initial entropy/variance are 0.0 (no samples yet)
        assert tracker.current_entropy == 0.0
        assert tracker.current_variance == 0.0

        tracker.end_session()
        assert not tracker.is_session_active

    def test_state_classification(self):
        """ModelStateClassifier uses boolean methods to check state conditions."""
        classifier = ModelStateClassifier()

        # Low entropy = confident
        assert classifier.is_confident(entropy=1.0, variance=0.8)
        assert not classifier.requires_caution(entropy=1.0, variance=0.8)

        # High entropy + low variance = distressed
        assert classifier.is_distressed(entropy=4.0, variance=0.1)
        assert classifier.requires_caution(entropy=4.0, variance=0.1)

        # High entropy + moderate variance = uncertain (not distressed)
        assert classifier.is_uncertain(entropy=3.5, variance=0.5)
        assert not classifier.is_distressed(entropy=3.5, variance=0.5)
        assert classifier.requires_caution(entropy=3.5, variance=0.5)


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
        """Escalation: entropy increases significantly (delta > 0.5)."""
        transition = EntropyTransition(
            from_entropy=2.0,  # Nominal entropy
            from_variance=0.5,
            to_entropy=4.0,  # Distressed entropy
            to_variance=0.1,
            token_index=10,
        )
        assert transition.is_escalation
        assert not transition.is_recovery
        assert transition.entropy_delta > 0.5  # 4.0 - 2.0 = 2.0

    def test_recovery_detection(self):
        """Recovery: entropy decreases significantly (delta < -0.5)."""
        transition = EntropyTransition(
            from_entropy=4.0,  # Distressed entropy
            from_variance=0.1,
            to_entropy=1.5,  # Confident entropy
            to_variance=0.5,
            token_index=20,
        )
        assert transition.is_recovery
        assert not transition.is_escalation
        assert transition.entropy_delta < -0.5  # 1.5 - 4.0 = -2.5
