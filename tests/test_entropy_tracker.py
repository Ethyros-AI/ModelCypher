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

"""Tests for entropy tracking components."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_tracker import (
    EntropySample,
    EntropyTransition,
)
from modelcypher.core.domain.entropy.model_state_classifier import (
    CalibratedBaseline,
    ModelStateSignals,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(value: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([value]))


def _is_inf(value: float) -> bool:
    return value in (float("inf"), float("-inf"))

class TestEntropyTransition:
    """Tests for EntropyTransition dataclass."""

    def test_entropy_delta(self):
        """Test entropy delta calculation."""
        transition = EntropyTransition(
            from_entropy=2.0,
            from_variance=0.3,
            to_entropy=3.5,
            to_variance=0.4,
            from_z_score=0.0,
        to_z_score=1.5,
        token_index=100,
    )

        eps = _eps(transition.entropy_delta)
        assert abs(transition.entropy_delta - 1.5) <= eps
        assert abs(transition.variance_delta - 0.1) <= _eps(transition.variance_delta)
        assert abs(transition.z_score_delta - 1.5) <= eps


class TestEntropySample:
    """Tests for EntropySample dataclass."""

    def test_sample_creation(self):
        """Test creating an entropy sample."""
        sample = EntropySample(
            window_id="window-1",
            token_start=0,
            token_end=50,
            logit_entropy=2.5,
            logit_variance=0.3,
            z_score=0.5,
        )

        assert sample.window_id == "window-1"
        eps = _eps(sample.logit_entropy)
        assert abs(sample.logit_entropy - 2.5) <= eps
        assert abs(sample.z_score - 0.5) <= eps
        assert sample.id is not None

    def test_sample_optional_fields(self):
        """Test optional SEP and semantic fields."""
        sample = EntropySample(
            window_id="window-1",
            token_start=0,
            token_end=50,
            logit_entropy=2.5,
            logit_variance=0.3,
            sep_entropy=2.3,
            sep_layers=[10, 11, 12],
            sep_confidence=0.85,
            semantic_volume=1.5,
            sample_count=10,
            pca_dimensions=32,
        )

        assert abs(sample.sep_entropy - 2.3) <= _eps(sample.sep_entropy)
        assert sample.sep_layers == [10, 11, 12]
        assert abs(sample.semantic_volume - 1.5) <= _eps(sample.semantic_volume)


class TestCalibratedBaseline:
    """Tests for CalibratedBaseline dataclass."""

    @pytest.fixture
    def baseline(self):
        """Create test baseline."""
        return CalibratedBaseline(
            mean=2.5,
            std_dev=0.5,
            percentile_25=2.0,
            percentile_75=3.0,
            percentile_95=3.5,
            vocab_size=32000,
            model_id="test-model",
            sample_count=1000,
        )

    def test_z_score_at_mean(self, baseline):
        """Test z-score is 0 at mean."""
        assert abs(baseline.z_score(2.5)) <= _eps(baseline.z_score(2.5))

    def test_z_score_above_mean(self, baseline):
        """Test z-score positive above mean."""
        assert abs(baseline.z_score(3.0) - 1.0) < _eps(baseline.z_score(3.0))

    def test_z_score_below_mean(self, baseline):
        """Test z-score negative below mean."""
        assert abs(baseline.z_score(2.0) + 1.0) < _eps(baseline.z_score(2.0))

    def test_z_score_zero_std_dev(self):
        """Test z-score handles zero std_dev gracefully."""
        baseline = CalibratedBaseline(
            mean=2.5,
            std_dev=0.0,
            percentile_25=2.5,
            percentile_75=2.5,
            percentile_95=2.5,
            vocab_size=32000,
            model_id="constant-model",
            sample_count=100,
        )

        assert abs(baseline.z_score(2.5)) <= _eps(baseline.z_score(2.5))
        assert _is_inf(baseline.z_score(3.0))


class TestModelStateSignals:
    """Tests for ModelStateSignals dataclass."""

    def test_signal_fields(self):
        signals = ModelStateSignals(
            entropy=1.5,
            variance=0.2,
            z_score=-1.5,
            entropy_trend=-0.1,
            entropy_variance_correlation=0.5,
        )

        assert abs(signals.entropy - 1.5) <= _eps(signals.entropy)
        assert abs(signals.z_score + 1.5) <= _eps(signals.z_score)
