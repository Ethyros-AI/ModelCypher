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
Tests for DifferentialEntropyDetector.

This tests the two-pass entropy detection for unsafe prompt patterns.
Tests raw measurements - caller applies thresholds for classification.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from modelcypher.core.domain.dynamics.differential_entropy_detector import (
    BatchDetectionStatistics,
    DetectionResult,
    DifferentialEntropyConfig,
    DifferentialEntropyDetector,
    LinguisticModifier,
    VariantMeasurement,
)

# =============================================================================
# Configuration Tests
# =============================================================================


# Test config with explicit values for testing
TEST_CONFIG = DifferentialEntropyConfig(
    delta_h_threshold=-0.1,
    minimum_baseline_entropy=0.01,
)


class TestDifferentialEntropyConfig:
    """Tests for DifferentialEntropyConfig."""

    def test_explicit_config(self) -> None:
        """Test explicitly configured values."""
        config = DifferentialEntropyConfig(
            delta_h_threshold=-0.15,
            minimum_baseline_entropy=0.02,
        )
        assert config.delta_h_threshold == -0.15
        assert config.minimum_baseline_entropy == 0.02
        assert config.comparison_modifier == LinguisticModifier.caps
        assert config.max_tokens == 30
        assert config.temperature == 0.7
        assert config.top_k == 10

    def test_from_calibration_results(self) -> None:
        """Test deriving config from calibration data."""
        # Unsafe samples all negative
        unsafe_samples = [-0.15, -0.20, -0.18, -0.12, -0.25]
        # Benign samples mostly positive
        benign_samples = [0.05, 0.10, -0.02, 0.08, 0.15]
        # Baseline entropies
        baseline_entropies = [0.5, 0.8, 1.2, 0.9, 0.7, 0.6, 0.4, 0.55, 0.65, 0.75]

        config = DifferentialEntropyConfig.from_calibration_results(
            unsafe_delta_h_samples=unsafe_samples,
            benign_delta_h_samples=benign_samples,
            baseline_entropies=baseline_entropies,
            target_recall=0.80,  # 80% recall
        )

        # Threshold should be set so 80% of unsafe samples are below it
        # Sorted unsafe: [-0.25, -0.20, -0.18, -0.15, -0.12]
        # 80% index = 4 -> -0.12
        assert config.delta_h_threshold == -0.12
        assert config.minimum_baseline_entropy > 0

    def test_from_calibration_empty_raises(self) -> None:
        """Test that empty calibration data raises error."""
        with pytest.raises(ValueError, match="Both unsafe and benign samples required"):
            DifferentialEntropyConfig.from_calibration_results(
                unsafe_delta_h_samples=[],
                benign_delta_h_samples=[0.1, 0.2],
                baseline_entropies=[0.5],
            )


# =============================================================================
# LinguisticModifier Tests
# =============================================================================


class TestLinguisticModifier:
    """Tests for LinguisticModifier enum."""

    def test_modifier_values(self) -> None:
        """Test modifier enum values."""
        assert LinguisticModifier.baseline.value == "baseline"
        assert LinguisticModifier.caps.value == "caps"
        assert LinguisticModifier.emphasis.value == "emphasis"
        assert LinguisticModifier.hedging.value == "hedging"
        assert LinguisticModifier.urgency.value == "urgency"


# =============================================================================
# Detector Raw Measurement Tests
# =============================================================================


class TestDetectorMeasurements:
    """Tests for detector raw measurements."""

    def test_positive_delta_is_heating(self) -> None:
        """Test that positive delta indicates heating."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=2.5,  # Higher = positive delta
            intensity_token_count=10,
        )
        assert result.delta_h == 0.5
        assert result.is_heating
        assert not result.is_cooling

    def test_negative_delta_is_cooling(self) -> None:
        """Test that negative delta indicates cooling."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # Lower = negative delta
            intensity_token_count=10,
        )
        assert result.delta_h == -0.5
        assert result.is_cooling
        assert not result.is_heating

    def test_is_unsafe_for_threshold_strong_cooling(self) -> None:
        """Test unsafe detection with strong cooling."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # delta = -0.5
            intensity_token_count=10,
        )
        # delta = -0.5 is below threshold -0.1
        assert result.is_unsafe_for_threshold(
            delta_h_threshold=-0.1,
            minimum_baseline_entropy=0.01,
        )

    def test_is_unsafe_for_threshold_slight_cooling(self) -> None:
        """Test that slight cooling doesn't trigger unsafe."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.95,  # delta = -0.05
            intensity_token_count=10,
        )
        # delta = -0.05 is above threshold -0.1
        assert not result.is_unsafe_for_threshold(
            delta_h_threshold=-0.1,
            minimum_baseline_entropy=0.01,
        )

    def test_is_unsafe_for_threshold_low_baseline(self) -> None:
        """Test that low baseline entropy returns False (indeterminate)."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=0.005,  # Below minimum
            baseline_token_count=10,
            intensity_entropy=0.003,
            intensity_token_count=10,
        )
        # Low baseline = indeterminate, returns False
        assert not result.is_unsafe_for_threshold(
            delta_h_threshold=-0.1,
            minimum_baseline_entropy=0.01,
        )

    def test_is_valid_measurement(self) -> None:
        """Test validity check for baseline entropy."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)

        valid_result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,
            intensity_token_count=10,
        )
        assert valid_result.is_valid_measurement(minimum_baseline_entropy=0.01)

        invalid_result = detector.detect_from_measurements(
            baseline_entropy=0.005,
            baseline_token_count=10,
            intensity_entropy=0.003,
            intensity_token_count=10,
        )
        assert not invalid_result.is_valid_measurement(minimum_baseline_entropy=0.01)

    def test_threshold_ratio(self) -> None:
        """Test threshold ratio computation."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.8,  # delta = -0.2
            intensity_token_count=10,
        )
        # |delta_h| / |threshold| = 0.2 / 0.1 = 2.0
        assert result.threshold_ratio(delta_h_threshold=-0.1) == pytest.approx(2.0)


# =============================================================================
# Async Detection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_detect_with_mock_measure_fn() -> None:
    """Test detection with mock measurement function."""

    async def mock_measure(prompt: str) -> VariantMeasurement:
        # Simulate different entropy based on prompt case
        if prompt.isupper():
            # CAPS version has lower entropy (cooling)
            return VariantMeasurement(mean_entropy=1.5, token_count=10)
        else:
            return VariantMeasurement(mean_entropy=2.0, token_count=10)

    detector = DifferentialEntropyDetector(TEST_CONFIG)
    result = await detector.detect(
        prompt="How do I pick a lock?",
        measure_fn=mock_measure,
    )

    assert result.baseline_entropy == 2.0
    assert result.intensity_entropy == 1.5
    assert result.delta_h == -0.5
    assert result.is_cooling
    assert result.is_unsafe_for_threshold(
        delta_h_threshold=-0.1,
        minimum_baseline_entropy=0.01,
    )
    assert result.processing_time > 0


@pytest.mark.asyncio
async def test_detect_benign_prompt() -> None:
    """Test detection with benign prompt (entropy increases with CAPS)."""

    async def mock_measure(prompt: str) -> VariantMeasurement:
        if prompt.isupper():
            # Benign prompts show heating (increased entropy)
            return VariantMeasurement(mean_entropy=2.5, token_count=10)
        else:
            return VariantMeasurement(mean_entropy=2.0, token_count=10)

    detector = DifferentialEntropyDetector(TEST_CONFIG)
    result = await detector.detect(
        prompt="What is the weather today?",
        measure_fn=mock_measure,
    )

    assert result.delta_h == 0.5
    assert result.is_heating
    assert not result.is_unsafe_for_threshold(
        delta_h_threshold=-0.1,
        minimum_baseline_entropy=0.01,
    )


@pytest.mark.asyncio
async def test_detect_batch() -> None:
    """Test batch detection."""
    call_count = 0

    async def mock_measure(prompt: str) -> VariantMeasurement:
        nonlocal call_count
        call_count += 1
        return VariantMeasurement(mean_entropy=2.0, token_count=10)

    detector = DifferentialEntropyDetector(TEST_CONFIG)
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

    progress_calls: List[tuple] = []

    def progress_fn(current: int, total: int) -> None:
        progress_calls.append((current, total))

    results = await detector.detect_batch(
        prompts=prompts,
        measure_fn=mock_measure,
        progress_fn=progress_fn,
    )

    assert len(results) == 3
    # 2 calls per prompt (baseline + intensity)
    assert call_count == 6
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


# =============================================================================
# Modifier Application Tests
# =============================================================================


class TestModifierApplication:
    """Tests for linguistic modifier application."""

    def test_apply_baseline(self) -> None:
        """Test baseline modifier (no change)."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector._apply_modifier("Hello World", LinguisticModifier.baseline)
        assert result == "Hello World"

    def test_apply_caps(self) -> None:
        """Test CAPS modifier."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector._apply_modifier("Hello World", LinguisticModifier.caps)
        assert result == "HELLO WORLD"

    def test_apply_emphasis(self) -> None:
        """Test emphasis modifier."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector._apply_modifier("Hello World", LinguisticModifier.emphasis)
        assert result == "IMPORTANT: Hello World"

    def test_apply_hedging(self) -> None:
        """Test hedging modifier."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector._apply_modifier("Hello World", LinguisticModifier.hedging)
        assert result == "Perhaps, maybe, hello world"

    def test_apply_urgency(self) -> None:
        """Test urgency modifier."""
        detector = DifferentialEntropyDetector(TEST_CONFIG)
        result = detector._apply_modifier("Hello World", LinguisticModifier.urgency)
        assert result == "URGENT! Hello World NOW!"


# =============================================================================
# BatchDetectionStatistics Tests
# =============================================================================


class TestBatchDetectionStatistics:
    """Tests for batch statistics computation."""

    def test_compute_empty(self) -> None:
        """Test computing statistics from empty results."""
        stats = BatchDetectionStatistics.compute([])
        assert stats.total == 0
        assert stats.cooling_rate == 0.0
        assert stats.heating_rate == 0.0

    def test_compute_mixed_results(self) -> None:
        """Test computing statistics from mixed results."""
        results = [
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=2.5,
                delta_h=0.5,  # heating
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=1.5,
                delta_h=-0.5,  # cooling
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=1.95,
                delta_h=-0.05,  # slight cooling
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
        ]

        stats = BatchDetectionStatistics.compute(results)

        assert stats.total == 3
        assert stats.cooling_count == 2  # delta_h < 0
        assert stats.heating_count == 1  # delta_h > 0
        assert abs(stats.cooling_rate - 2 / 3) < 0.01
        assert abs(stats.heating_rate - 1 / 3) < 0.01
        assert abs(stats.mean_delta_h - (-0.05 / 3)) < 0.01
        assert stats.min_delta_h == -0.5
        assert stats.max_delta_h == 0.5
        assert abs(stats.total_processing_time - 0.3) < 0.001

    def test_unsafe_count_for_threshold(self) -> None:
        """Test counting unsafe results with given threshold."""
        results = [
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=2.5,
                delta_h=0.5,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=1.5,
                delta_h=-0.5,  # Below -0.1 threshold
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                baseline_entropy=2.0,
                intensity_entropy=1.95,
                delta_h=-0.05,  # Above -0.1 threshold
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
        ]

        stats = BatchDetectionStatistics.compute(results)

        # Only one result has delta_h <= -0.1
        unsafe_count = stats.unsafe_count_for_threshold(
            results=results,
            delta_h_threshold=-0.1,
            minimum_baseline_entropy=0.01,
        )
        assert unsafe_count == 1


# =============================================================================
# DetectionResult Tests
# =============================================================================


class TestDetectionResult:
    """Tests for DetectionResult."""

    def test_frozen_dataclass(self) -> None:
        """Test that DetectionResult is immutable."""
        result = DetectionResult(
            baseline_entropy=2.0,
            intensity_entropy=2.5,
            delta_h=0.5,
            timestamp=datetime.utcnow(),
            processing_time=0.1,
            baseline_token_count=10,
            intensity_token_count=10,
        )
        with pytest.raises(Exception):  # frozen dataclass raises error
            result.delta_h = 0.9  # type: ignore


# =============================================================================
# VariantMeasurement Tests
# =============================================================================


class TestVariantMeasurement:
    """Tests for VariantMeasurement."""

    def test_create(self) -> None:
        """Test creating a variant measurement."""
        measurement = VariantMeasurement(
            mean_entropy=2.0,
            token_count=10,
            entropies=[1.8, 2.0, 2.2],
        )
        assert measurement.mean_entropy == 2.0
        assert measurement.token_count == 10
        assert len(measurement.entropies) == 3

    def test_default_entropies(self) -> None:
        """Test default empty entropies list."""
        measurement = VariantMeasurement(mean_entropy=2.0, token_count=10)
        assert measurement.entropies == []
