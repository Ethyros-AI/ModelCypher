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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy import (
    CalibratedBaseline,
    EntropySample,
    EntropyTracker,
    EntropyTransition,
    HiddenStateExtractor,
    SEPProbe,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


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


class TestEntropySample:
    """Tests for EntropySample dataclass."""

    def test_best_entropy_estimate_prefers_sep(self):
        sample = EntropySample(
            logit_entropy=3.0,
            sep_entropy=0.5,
        )
        assert abs(sample.best_entropy_estimate - 0.5) <= _eps(
            sample.best_entropy_estimate, 0.5
        )

    def test_best_entropy_estimate_fallback(self):
        sample = EntropySample(logit_entropy=3.0)
        assert abs(sample.best_entropy_estimate - 3.0) <= _eps(
            sample.best_entropy_estimate, 3.0
        )

    def test_z_score_computation(self):
        baseline = _create_test_baseline()
        sample = EntropySample(logit_entropy=4.0)
        z_score = sample.get_z_score(baseline)
        eps = _eps(z_score, 1.5)
        assert abs(z_score - 1.5) <= eps


class TestEntropyTracker:
    """Tests for EntropyTracker session management."""

    def test_session_lifecycle(self):
        baseline = _create_test_baseline()
        tracker = EntropyTracker(baseline=baseline, source="test")
        assert not tracker.is_session_active

        tracker.start_session()
        assert tracker.is_session_active
        eps = _eps(tracker.current_entropy, tracker.current_variance)
        assert abs(tracker.current_entropy) <= eps
        assert abs(tracker.current_variance) <= eps

        tracker.end_session()
        assert not tracker.is_session_active


class TestHiddenStateExtractor:
    """Tests for HiddenStateExtractor."""

    def test_layer_targeting_presets(self):
        extractor = HiddenStateExtractor.for_sep_probe(32)
        # 75-87.5% of 32 layers = layers 24-28
        assert 24 in extractor.target_layers
        assert 28 in extractor.target_layers

        extractor = HiddenStateExtractor.for_refusal_direction(32)
        # 40-60% of 32 layers = layers 12-19
        assert 13 in extractor.target_layers
        assert 19 in extractor.target_layers

    def test_session_management(self):
        extractor = HiddenStateExtractor.for_sep_probe(32)
        assert not extractor.is_active

        extractor.start_session()
        assert extractor.is_active

        summary = extractor.end_session()
        assert not extractor.is_active
        assert summary.total_captures == 0

    def test_state_capture(self):
        extractor = HiddenStateExtractor(total_layers=32, target_layers={25, 26})
        extractor.start_session()

        hidden = mx.random.normal((1, 4096))
        extractor.capture(hidden, layer=25, token_index=0)

        states = extractor.extracted_states()
        assert 25 in states
        assert 26 not in states

        summary = extractor.end_session()
        assert summary.total_captures == 1


class TestSEPProbe:
    """Tests for SEPProbe (no config)."""

    def test_initialization(self):
        probe = SEPProbe(layer_count=32, hidden_dim=4096)
        assert probe.layer_count == 32
        assert probe.hidden_dim == 4096

    def test_target_layers(self):
        probe = SEPProbe(layer_count=32, hidden_dim=4096)
        # Target layers are 75-87.5% of layer count
        targets = probe.target_layers
        assert 24 in targets
        assert 28 in targets
