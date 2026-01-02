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
"""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.entropy import (
    CalibratedBaseline,
    EntropySample,
    EntropyTracker,
    EntropyTrackerConfig,
    EntropyTransition,
    ExtractorConfig,
    HiddenStateExtractor,
    SEPProbe,
    SEPProbeConfig,
)


def _create_test_baseline() -> CalibratedBaseline:
    """Create a calibrated baseline for testing."""
    return CalibratedBaseline(
        mean=2.5,
        std_dev=1.0,
        percentile_25=1.8,
        percentile_75=3.2,
        percentile_95=4.5,
        vocab_size=32768,
        model_id="test-model",
        sample_count=100,
    )


def _create_tracker_config() -> EntropyTrackerConfig:
    return EntropyTrackerConfig(
        top_k=10,
        window_size=20,
        emit_interval=1,
        source="EntropyTracker",
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

    def test_z_score_computation(self):
        baseline = _create_test_baseline()
        sample = EntropySample(logit_entropy=4.0)
        assert sample.get_z_score(baseline) == pytest.approx(1.5)


class TestEntropyTracker:
    """Tests for EntropyTracker session management."""

    def test_session_lifecycle(self):
        baseline = _create_test_baseline()
        tracker = EntropyTracker(baseline=baseline, config=_create_tracker_config())
        assert not tracker.is_session_active

        tracker.start_session()
        assert tracker.is_session_active
        assert tracker.current_entropy == 0.0
        assert tracker.current_variance == 0.0

        tracker.end_session()
        assert not tracker.is_session_active


class TestHiddenStateExtractor:
    """Tests for HiddenStateExtractor."""

    def test_layer_targeting_presets(self):
        config = ExtractorConfig.for_sep_probe(32)
        assert 24 in config.target_layers
        assert 28 in config.target_layers

        config = ExtractorConfig.for_refusal_direction(32)
        assert 13 in config.target_layers
        assert 19 in config.target_layers

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

    def test_target_layers(self):
        config = SEPProbeConfig(layer_count=32)
        targets = config.target_layers
        assert 24 in targets
        assert 28 in targets
