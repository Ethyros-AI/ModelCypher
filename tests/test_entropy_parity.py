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
Unit tests for entropy domain parity modules (requires backend).
"""

import pytest

from modelcypher.core.domain.entropy import (
    CalibratedBaseline,
    EntropySample,
    EntropyTracker,
    EntropyTransition,
    HiddenStateExtractor,
    SEPProbe,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps(backend, *values: float) -> float:
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

    def test_best_entropy_estimate_prefers_sep(self, any_backend):
        sample = EntropySample(
            logit_entropy=3.0,
            sep_entropy=0.5,
        )
        assert abs(sample.best_entropy_estimate - 0.5) <= _eps(
            any_backend, sample.best_entropy_estimate, 0.5
        )

    def test_best_entropy_estimate_fallback(self, any_backend):
        sample = EntropySample(logit_entropy=3.0)
        assert abs(sample.best_entropy_estimate - 3.0) <= _eps(
            any_backend, sample.best_entropy_estimate, 3.0
        )

    def test_z_score_computation(self, any_backend):
        baseline = _create_test_baseline()
        sample = EntropySample(logit_entropy=4.0)
        z_score = sample.get_z_score(baseline)
        eps = _eps(any_backend, z_score, 1.5)
        assert abs(z_score - 1.5) <= eps


class TestEntropyTracker:
    """Tests for EntropyTracker session management."""

    def test_session_lifecycle(self, any_backend):
        baseline = _create_test_baseline()
        tracker = EntropyTracker(baseline=baseline, source="test")
        assert not tracker.is_session_active

        tracker.start_session()
        assert tracker.is_session_active
        eps = _eps(any_backend, tracker.current_entropy, tracker.current_variance)
        assert abs(tracker.current_entropy) <= eps
        assert abs(tracker.current_variance) <= eps

        tracker.end_session()
        assert not tracker.is_session_active


class TestHiddenStateExtractor:
    """Tests for HiddenStateExtractor."""

    def test_requires_target_layers(self):
        """Caller must specify target layers."""
        with pytest.raises(ValueError):
            HiddenStateExtractor(target_layers=set())

    def test_session_management(self, any_backend):
        # Caller provides layers from geometric analysis
        extractor = HiddenStateExtractor(target_layers={24, 25, 26, 27, 28})
        assert not extractor.is_active

        extractor.start_session()
        assert extractor.is_active

        summary = extractor.end_session()
        assert not extractor.is_active
        assert summary.total_captures == 0

    def test_state_capture(self, any_backend):
        extractor = HiddenStateExtractor(target_layers={25, 26})
        extractor.start_session()

        hidden = any_backend.random_normal((1, 4096))
        extractor.capture(hidden, layer=25, token_index=0)

        states = extractor.extracted_states()
        assert 25 in states
        assert 26 not in states

        summary = extractor.end_session()
        assert summary.total_captures == 1


class TestSEPProbe:
    """Tests for SEPProbe."""

    def test_initialization(self, any_backend):
        probe = SEPProbe(hidden_dim=4096)
        assert probe.hidden_dim == 4096
        assert not probe.is_ready  # No weights loaded yet

    def test_available_layers_empty_before_load(self, any_backend):
        probe = SEPProbe(hidden_dim=4096)
        assert probe.available_layers == set()
