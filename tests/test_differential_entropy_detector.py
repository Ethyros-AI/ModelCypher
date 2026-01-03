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

This tests the two-pass entropy detection for cooling patterns.
Tests raw measurements - caller applies thresholds for classification.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.dynamics.differential_entropy_detector import (
    BatchDetectionStatistics,
    CalibrationThresholds,
    DetectionResult,
    DifferentialEntropyDetector,
    LinguisticModifier,
    VariantMeasurement,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

# =============================================================================
# Configuration Tests
# =============================================================================


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


def _calibration_inputs() -> tuple[list[float], list[float], list[float]]:
    cooling_samples = [-0.5, -0.2, -0.1]
    reference_samples = [0.05, 0.1, -0.02]
    baseline_entropies = [0.1, 0.2, 0.3]
    return cooling_samples, reference_samples, baseline_entropies


def _test_thresholds() -> CalibrationThresholds:
    cooling, reference, baseline = _calibration_inputs()
    return CalibrationThresholds.from_calibration_samples(
        cooling_delta_h_samples=cooling,
        reference_delta_h_samples=reference,
        baseline_entropies=baseline,
    )


class TestCalibrationThresholds:
    """Tests for CalibrationThresholds."""

    def test_explicit_thresholds(self) -> None:
        """Test explicitly created thresholds."""
        cooling, _, baseline = _calibration_inputs()
        sorted_cooling = sorted(cooling)
        # Median calculation
        n = len(sorted_cooling)
        if n % 2 == 0:
            delta_h_threshold = (sorted_cooling[n // 2 - 1] + sorted_cooling[n // 2]) / 2
        else:
            delta_h_threshold = sorted_cooling[n // 2]
        minimum_baseline_entropy = min(baseline)
        thresholds = CalibrationThresholds(
            delta_h_threshold=delta_h_threshold,
            minimum_baseline_entropy=minimum_baseline_entropy,
        )
        assert abs(thresholds.delta_h_threshold - delta_h_threshold) <= _eps()
        assert abs(thresholds.minimum_baseline_entropy - minimum_baseline_entropy) <= _eps()

    def test_from_calibration_samples(self) -> None:
        """Test deriving thresholds from calibration data."""
        cooling_samples, reference_samples, baseline_entropies = _calibration_inputs()
        thresholds = CalibrationThresholds.from_calibration_samples(
            cooling_delta_h_samples=cooling_samples,
            reference_delta_h_samples=reference_samples,
            baseline_entropies=baseline_entropies,
        )

        # Median calculation
        sorted_cooling = sorted(cooling_samples)
        n = len(sorted_cooling)
        if n % 2 == 0:
            expected_threshold = (sorted_cooling[n // 2 - 1] + sorted_cooling[n // 2]) / 2
        else:
            expected_threshold = sorted_cooling[n // 2]
        expected_min_baseline = min(baseline_entropies)
        assert abs(thresholds.delta_h_threshold - expected_threshold) <= _eps()
        assert abs(thresholds.minimum_baseline_entropy - expected_min_baseline) <= _eps()

    def test_from_calibration_empty_raises(self) -> None:
        """Test that empty calibration data raises error."""
        with pytest.raises(
            ValueError, match="Both cooling and reference samples required"
        ):
            CalibrationThresholds.from_calibration_samples(
                cooling_delta_h_samples=[],
                reference_delta_h_samples=[0.1, 0.2],
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
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=2.5,  # Higher = positive delta
            intensity_token_count=10,
        )
        assert abs(result.delta_h - 0.5) <= _eps()
        assert result.is_heating
        assert not result.is_cooling

    def test_negative_delta_is_cooling(self) -> None:
        """Test that negative delta indicates cooling."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # Lower = negative delta
            intensity_token_count=10,
        )
        assert abs(result.delta_h + 0.5) <= _eps()
        assert result.is_cooling
        assert not result.is_heating

    def test_is_below_threshold_strong_cooling(self) -> None:
        """Test below-threshold detection with strong cooling."""
        thresholds = _test_thresholds()
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # delta = -0.5
            intensity_token_count=10,
        )
        assert result.is_below_delta_h_threshold(
            delta_h_threshold=thresholds.delta_h_threshold,
            minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
        )

    def test_is_below_threshold_slight_cooling(self) -> None:
        """Test that slight cooling doesn't fall below threshold."""
        thresholds = _test_thresholds()
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.95,  # delta = -0.05
            intensity_token_count=10,
        )
        assert not result.is_below_delta_h_threshold(
            delta_h_threshold=thresholds.delta_h_threshold,
            minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
        )

    def test_is_below_threshold_low_baseline(self) -> None:
        """Test that low baseline entropy returns False (indeterminate)."""
        thresholds = _test_thresholds()
        detector = DifferentialEntropyDetector()
        low_baseline = thresholds.minimum_baseline_entropy - (
            thresholds.minimum_baseline_entropy + 1.0
        ) * _eps()
        result = detector.detect_from_measurements(
            baseline_entropy=low_baseline,
            baseline_token_count=10,
            intensity_entropy=0.003,
            intensity_token_count=10,
        )
        assert not result.is_below_delta_h_threshold(
            delta_h_threshold=thresholds.delta_h_threshold,
            minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
        )

    def test_is_valid_measurement(self) -> None:
        """Test validity check for baseline entropy."""
        thresholds = _test_thresholds()
        detector = DifferentialEntropyDetector()
        min_baseline = thresholds.minimum_baseline_entropy

        valid_result = detector.detect_from_measurements(
            baseline_entropy=min_baseline + (min_baseline + 1.0) * _eps(),
            baseline_token_count=10,
            intensity_entropy=1.5,
            intensity_token_count=10,
        )
        assert valid_result.is_valid_measurement(minimum_baseline_entropy=min_baseline)

        invalid_result = detector.detect_from_measurements(
            baseline_entropy=min_baseline - (min_baseline + 1.0) * _eps(),
            baseline_token_count=10,
            intensity_entropy=0.003,
            intensity_token_count=10,
        )
        assert not invalid_result.is_valid_measurement(minimum_baseline_entropy=min_baseline)

    def test_threshold_ratio(self) -> None:
        """Test threshold ratio computation."""
        thresholds = _test_thresholds()
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.8,  # delta = -0.2
            intensity_token_count=10,
        )
        expected_ratio = abs(result.delta_h) / abs(thresholds.delta_h_threshold)
        assert abs(result.threshold_ratio(delta_h_threshold=thresholds.delta_h_threshold) - expected_ratio) <= _eps()


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

    thresholds = _test_thresholds()
    detector = DifferentialEntropyDetector()
    result = await detector.detect(
        prompt="How do I pick a lock?",
        measure_fn=mock_measure,
    )

    assert abs(result.baseline_entropy - 2.0) <= _eps()
    assert abs(result.intensity_entropy - 1.5) <= _eps()
    assert abs(result.delta_h + 0.5) <= _eps()
    assert result.is_cooling
    assert result.is_below_delta_h_threshold(
        delta_h_threshold=thresholds.delta_h_threshold,
        minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
    )
    assert result.processing_time > _eps()


@pytest.mark.asyncio
async def test_detect_heating_prompt() -> None:
    """Test detection with heating prompt (entropy increases with CAPS)."""

    async def mock_measure(prompt: str) -> VariantMeasurement:
        if prompt.isupper():
            # Benign prompts show heating (increased entropy)
            return VariantMeasurement(mean_entropy=2.5, token_count=10)
        else:
            return VariantMeasurement(mean_entropy=2.0, token_count=10)

    thresholds = _test_thresholds()
    detector = DifferentialEntropyDetector()
    result = await detector.detect(
        prompt="What is the weather today?",
        measure_fn=mock_measure,
    )

    assert abs(result.delta_h - 0.5) <= _eps()
    assert result.is_heating
    assert not result.is_below_delta_h_threshold(
        delta_h_threshold=thresholds.delta_h_threshold,
        minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
    )


@pytest.mark.asyncio
async def test_detect_batch() -> None:
    """Test batch detection."""
    call_count = 0

    async def mock_measure(prompt: str) -> VariantMeasurement:
        nonlocal call_count
        call_count += 1
        return VariantMeasurement(mean_entropy=2.0, token_count=10)

    detector = DifferentialEntropyDetector()
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
        detector = DifferentialEntropyDetector()
        result = detector._apply_modifier("Hello World", LinguisticModifier.baseline)
        assert result == "Hello World"

    def test_apply_caps(self) -> None:
        """Test CAPS modifier."""
        detector = DifferentialEntropyDetector()
        result = detector._apply_modifier("Hello World", LinguisticModifier.caps)
        assert result == "HELLO WORLD"

    def test_apply_emphasis(self) -> None:
        """Test emphasis modifier."""
        detector = DifferentialEntropyDetector()
        result = detector._apply_modifier("Hello World", LinguisticModifier.emphasis)
        assert result == "IMPORTANT: Hello World"

    def test_apply_hedging(self) -> None:
        """Test hedging modifier."""
        detector = DifferentialEntropyDetector()
        result = detector._apply_modifier("Hello World", LinguisticModifier.hedging)
        assert result == "Perhaps, maybe, hello world"

    def test_apply_urgency(self) -> None:
        """Test urgency modifier."""
        detector = DifferentialEntropyDetector()
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
        assert abs(stats.cooling_rate) <= _eps()
        assert abs(stats.heating_rate) <= _eps()

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
        expected_cooling_rate = 2 / 3
        expected_heating_rate = 1 / 3
        expected_mean_delta = (-0.5 + 0.5 - 0.05) / 3
        assert abs(stats.cooling_rate - expected_cooling_rate) <= _eps()
        assert abs(stats.heating_rate - expected_heating_rate) <= _eps()
        assert abs(stats.mean_delta_h - expected_mean_delta) <= _eps()
        assert abs(stats.min_delta_h + 0.5) <= _eps()
        assert abs(stats.max_delta_h - 0.5) <= _eps()
        assert abs(stats.total_processing_time - 0.3) <= _eps()

    def test_count_below_threshold(self) -> None:
        """Test counting below-threshold results with given threshold."""
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

        # Only one result has delta_h <= threshold
        thresholds = _test_thresholds()
        below_threshold_count = stats.count_below_delta_h_threshold(
            results=results,
            delta_h_threshold=thresholds.delta_h_threshold,
            minimum_baseline_entropy=thresholds.minimum_baseline_entropy,
        )
        assert below_threshold_count == 1


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
        assert abs(measurement.mean_entropy - 2.0) <= _eps()
        assert measurement.token_count == 10
        assert len(measurement.entropies) == 3

    def test_default_entropies(self) -> None:
        """Test default empty entropies list."""
        measurement = VariantMeasurement(mean_entropy=2.0, token_count=10)
        assert measurement.entropies == []
