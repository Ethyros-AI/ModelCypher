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

from modelcypher.core.domain.entropy.entropy_tracker import (
    EntropySample,
    EntropyTransition,
)
from modelcypher.core.domain.entropy.model_state_classifier import (
    CalibratedBaseline,
    ClassificationResult,
    EntropyStateThresholds,
    ModelStateSignals,
)


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

        assert transition.entropy_delta == 1.5
        assert transition.variance_delta == pytest.approx(0.1)
        assert transition.z_score_delta == 1.5

    def test_is_escalation(self):
        """Test escalation detection (z-score increased >1σ)."""
        escalation = EntropyTransition(
            from_entropy=2.0,
            from_variance=0.3,
            to_entropy=3.5,
            to_variance=0.4,
            from_z_score=0.0,
            to_z_score=1.5,  # +1.5σ
            token_index=100,
        )

        assert escalation.is_escalation is True
        assert escalation.is_recovery is False
        assert escalation.is_significant is True

    def test_is_recovery(self):
        """Test recovery detection (z-score decreased >1σ)."""
        recovery = EntropyTransition(
            from_entropy=3.5,
            from_variance=0.4,
            to_entropy=2.0,
            to_variance=0.3,
            from_z_score=2.0,
            to_z_score=0.5,  # -1.5σ
            token_index=100,
        )

        assert recovery.is_recovery is True
        assert recovery.is_escalation is False
        assert recovery.is_significant is True

    def test_not_significant(self):
        """Test non-significant transition (z-score change <1σ)."""
        minor = EntropyTransition(
            from_entropy=2.0,
            from_variance=0.3,
            to_entropy=2.2,
            to_variance=0.32,
            from_z_score=0.0,
            to_z_score=0.5,  # +0.5σ (not significant)
            token_index=100,
        )

        assert minor.is_escalation is False
        assert minor.is_recovery is False
        assert minor.is_significant is False


class TestEntropySample:
    """Tests for EntropySample dataclass."""

    def test_sample_creation(self):
        """Test creating an entropy sample."""
        sample = EntropySample(
            window_id="window-1",
            token_start=0,
            token_end=50,
            logit_entropy=2.5,
            top_k_variance=0.3,
            z_score=0.5,
        )

        assert sample.window_id == "window-1"
        assert sample.logit_entropy == 2.5
        assert sample.z_score == 0.5
        assert sample.id is not None  # Auto-generated

    def test_sample_optional_fields(self):
        """Test optional SEP and semantic fields."""
        sample = EntropySample(
            window_id="window-1",
            token_start=0,
            token_end=50,
            logit_entropy=2.5,
            top_k_variance=0.3,
            sep_entropy=2.3,
            sep_layers=[10, 11, 12],
            sep_confidence=0.85,
            semantic_volume=1.5,
            sample_count=10,
            pca_dimensions=32,
        )

        assert sample.sep_entropy == 2.3
        assert sample.sep_layers == [10, 11, 12]
        assert sample.semantic_volume == 1.5


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
        assert baseline.z_score(2.5) == 0.0

    def test_z_score_above_mean(self, baseline):
        """Test z-score positive above mean."""
        # 3.0 is 1σ above mean (2.5 + 0.5)
        assert baseline.z_score(3.0) == pytest.approx(1.0)

    def test_z_score_below_mean(self, baseline):
        """Test z-score negative below mean."""
        # 2.0 is 1σ below mean (2.5 - 0.5)
        assert baseline.z_score(2.0) == pytest.approx(-1.0)

    def test_is_outlier_at_2sigma(self, baseline):
        """Test outlier detection at 2σ."""
        # 3.5 is 2σ above mean
        assert baseline.is_outlier(3.5, sigma=2.0) is False  # Exactly at boundary
        assert baseline.is_outlier(3.6, sigma=2.0) is True  # Just above

    def test_is_low_entropy(self, baseline):
        """Test low entropy detection (below 25th percentile)."""
        assert baseline.is_low_entropy(1.9) is True
        assert baseline.is_low_entropy(2.1) is False

    def test_is_high_entropy(self, baseline):
        """Test high entropy detection (above 75th percentile)."""
        assert baseline.is_high_entropy(3.1) is True
        assert baseline.is_high_entropy(2.9) is False

    def test_circuit_breaker_threshold(self, baseline):
        """Test circuit breaker at 95th percentile."""
        assert baseline.should_trip_circuit_breaker(3.6) is True
        assert baseline.should_trip_circuit_breaker(3.4) is False

    def test_z_score_zero_std_dev(self):
        """Test z-score handles zero std_dev gracefully."""
        baseline = CalibratedBaseline(
            mean=2.5,
            std_dev=0.0,  # Edge case
            percentile_25=2.5,
            percentile_75=2.5,
            percentile_95=2.5,
            vocab_size=32000,
            model_id="constant-model",
            sample_count=100,
        )

        assert baseline.z_score(2.5) == 0.0
        assert baseline.z_score(3.0) == float("inf")


class TestModelStateSignals:
    """Tests for ModelStateSignals dataclass."""

    def test_statistically_low(self):
        """Test statistically low detection (z < -1)."""
        signals = ModelStateSignals(
            entropy=1.5,
            variance=0.2,
            z_score=-1.5,
            entropy_trend=-0.1,
            entropy_variance_correlation=0.5,
            consecutive_high_entropy_count=0,
            circuit_breaker_tripped=False,
        )

        assert signals.is_statistically_low is True
        assert signals.is_statistically_high is False

    def test_statistically_high(self):
        """Test statistically high detection (z > 1)."""
        signals = ModelStateSignals(
            entropy=3.5,
            variance=0.3,
            z_score=1.5,
            entropy_trend=0.2,
            entropy_variance_correlation=0.3,
            consecutive_high_entropy_count=2,
            circuit_breaker_tripped=False,
        )

        assert signals.is_statistically_high is True
        assert signals.is_statistically_low is False

    def test_is_outlier(self):
        """Test outlier detection (|z| > 2)."""
        outlier = ModelStateSignals(
            entropy=4.0,
            variance=0.3,
            z_score=2.5,
            entropy_trend=0.3,
            entropy_variance_correlation=0.2,
            consecutive_high_entropy_count=3,
            circuit_breaker_tripped=False,
        )

        assert outlier.is_outlier is True
        assert outlier.is_extreme_outlier is False

    def test_is_extreme_outlier(self):
        """Test extreme outlier detection (|z| > 3)."""
        extreme = ModelStateSignals(
            entropy=5.0,
            variance=0.3,
            z_score=3.5,
            entropy_trend=0.5,
            entropy_variance_correlation=-0.1,
            consecutive_high_entropy_count=5,
            circuit_breaker_tripped=True,
        )

        assert extreme.is_extreme_outlier is True

    def test_distress_signature(self):
        """Test distress signature detection."""
        distress = ModelStateSignals(
            entropy=3.5,
            variance=0.15,  # Low variance
            z_score=2.0,  # High z-score
            entropy_trend=0.2,
            entropy_variance_correlation=-0.5,  # Negative correlation
            consecutive_high_entropy_count=4,
            circuit_breaker_tripped=False,
        )

        assert distress.has_distress_signature is True

    def test_no_distress_normal(self):
        """Test no distress signature for normal signals."""
        normal = ModelStateSignals(
            entropy=2.5,
            variance=0.3,
            z_score=0.2,
            entropy_trend=0.0,
            entropy_variance_correlation=0.4,
            consecutive_high_entropy_count=0,
            circuit_breaker_tripped=False,
        )

        assert normal.has_distress_signature is False


class TestEntropyStateThresholds:
    """Tests for EntropyStateThresholds dataclass."""

    def test_from_calibration(self):
        """Test creating thresholds from calibrated baseline."""
        baseline = CalibratedBaseline(
            mean=2.5,
            std_dev=0.5,
            percentile_25=2.0,
            percentile_75=3.0,
            percentile_95=3.5,
            vocab_size=32000,
            model_id="test-model",
            sample_count=1000,
        )

        thresholds = EntropyStateThresholds.from_calibration(baseline)

        assert thresholds.entropy_low == 2.0
        assert thresholds.entropy_high == 3.0
        assert thresholds.entropy_circuit_breaker == 3.5

    def test_from_percentiles(self):
        """Test creating thresholds from percentile values."""
        thresholds = EntropyStateThresholds.from_percentiles(
            percentile_25=1.8,
            percentile_75=2.8,
            percentile_95=3.2,
        )

        assert thresholds.entropy_low == 1.8
        assert thresholds.entropy_high == 2.8
        assert thresholds.entropy_circuit_breaker == 3.2


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_creation(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            state_name="nominal",
            entropy=2.5,
            variance=0.3,
            z_score=0.2,
            confidence=0.85,
            reason="Entropy within expected range",
        )

        assert result.state_name == "nominal"
        assert result.z_score == 0.2
        assert result.confidence == 0.85
