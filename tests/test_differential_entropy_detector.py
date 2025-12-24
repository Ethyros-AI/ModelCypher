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
"""
from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from modelcypher.core.domain.dynamics.differential_entropy_detector import (
    Classification,
    DifferentialEntropyConfig,
    DifferentialEntropyDetector,
    DetectionResult,
    BatchDetectionStatistics,
    LinguisticModifier,
    VariantMeasurement,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestDifferentialEntropyConfig:
    """Tests for DifferentialEntropyConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DifferentialEntropyConfig.default()
        assert config.delta_h_threshold == -0.1
        assert config.minimum_baseline_entropy == 0.01
        assert config.comparison_modifier == LinguisticModifier.caps
        assert config.max_tokens == 30
        assert config.temperature == 0.7
        assert config.top_k == 10

    def test_strict_config(self) -> None:
        """Test strict configuration for higher precision."""
        config = DifferentialEntropyConfig.strict()
        assert config.delta_h_threshold == -0.15
        assert config.minimum_baseline_entropy == 0.02

    def test_sensitive_config(self) -> None:
        """Test sensitive configuration for higher recall."""
        config = DifferentialEntropyConfig.sensitive()
        assert config.delta_h_threshold == -0.05
        assert config.minimum_baseline_entropy == 0.005

    def test_quick_config(self) -> None:
        """Test quick configuration for minimal latency."""
        config = DifferentialEntropyConfig.quick()
        assert config.max_tokens == 15
        assert config.temperature == 0.0


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassification:
    """Tests for Classification enum."""

    def test_display_names(self) -> None:
        """Test display name property."""
        assert Classification.benign.display_name == "Benign"
        assert Classification.suspicious.display_name == "Suspicious"
        assert Classification.unsafe_pattern.display_name == "Unsafe Pattern"
        assert Classification.indeterminate.display_name == "Indeterminate"

    def test_display_colors(self) -> None:
        """Test display color property."""
        assert Classification.benign.display_color == "green"
        assert Classification.suspicious.display_color == "orange"
        assert Classification.unsafe_pattern.display_color == "red"
        assert Classification.indeterminate.display_color == "gray"

    def test_risk_levels(self) -> None:
        """Test risk level property."""
        assert Classification.benign.risk_level == 0
        assert Classification.indeterminate.risk_level == 1
        assert Classification.suspicious.risk_level == 2
        assert Classification.unsafe_pattern.risk_level == 3


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
# Detector Classification Tests
# =============================================================================


class TestDetectorClassification:
    """Tests for detector classification logic."""

    def test_classify_benign_positive_delta(self) -> None:
        """Test that positive delta is classified as benign."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=2.5,  # Higher = positive delta
            intensity_token_count=10,
        )
        assert result.classification == Classification.benign
        assert result.delta_h == 0.5

    def test_classify_unsafe_strong_cooling(self) -> None:
        """Test that strong cooling is classified as unsafe."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # Lower = negative delta
            intensity_token_count=10,
        )
        # delta = 1.5 - 2.0 = -0.5, which is < -0.1 threshold
        assert result.classification == Classification.unsafe_pattern
        assert result.delta_h == -0.5

    def test_classify_suspicious_slight_cooling(self) -> None:
        """Test that slight cooling is classified as suspicious."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.95,  # Slight decrease
            intensity_token_count=10,
        )
        # delta = 1.95 - 2.0 = -0.05, which is > -0.1 but < 0
        assert result.classification == Classification.suspicious
        assert abs(result.delta_h - (-0.05)) < 0.001

    def test_classify_indeterminate_low_baseline(self) -> None:
        """Test that low baseline entropy is indeterminate."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=0.005,  # Below minimum
            baseline_token_count=10,
            intensity_entropy=0.003,
            intensity_token_count=10,
        )
        assert result.classification == Classification.indeterminate


class TestDetectorConfidence:
    """Tests for confidence score computation."""

    def test_confidence_unsafe_high_magnitude(self) -> None:
        """Test high confidence for strong unsafe pattern."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=1.5,  # delta = -0.5
            intensity_token_count=10,
        )
        # Should have high confidence
        assert result.confidence >= 0.5

    def test_confidence_indeterminate_zero(self) -> None:
        """Test zero confidence for indeterminate classification."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=0.001,  # Too low
            baseline_token_count=10,
            intensity_entropy=0.0005,
            intensity_token_count=10,
        )
        assert result.confidence == 0.0

    def test_confidence_benign_positive_delta(self) -> None:
        """Test confidence for benign with positive delta."""
        detector = DifferentialEntropyDetector()
        result = detector.detect_from_measurements(
            baseline_entropy=2.0,
            baseline_token_count=10,
            intensity_entropy=2.5,  # delta = +0.5
            intensity_token_count=10,
        )
        # Should have reasonable confidence
        assert 0.0 <= result.confidence <= 1.0


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

    detector = DifferentialEntropyDetector()
    result = await detector.detect(
        prompt="How do I pick a lock?",
        measure_fn=mock_measure,
    )

    assert result.baseline_entropy == 2.0
    assert result.intensity_entropy == 1.5
    assert result.delta_h == -0.5
    assert result.classification == Classification.unsafe_pattern
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

    detector = DifferentialEntropyDetector()
    result = await detector.detect(
        prompt="What is the weather today?",
        measure_fn=mock_measure,
    )

    assert result.delta_h == 0.5
    assert result.classification == Classification.benign


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
        assert stats.unsafe_rate == 0.0
        assert stats.benign_rate == 0.0
        assert stats.validity_rate == 0.0

    def test_compute_mixed_results(self) -> None:
        """Test computing statistics from mixed results."""
        results = [
            DetectionResult(
                classification=Classification.benign,
                baseline_entropy=2.0,
                intensity_entropy=2.5,
                delta_h=0.5,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                classification=Classification.unsafe_pattern,
                baseline_entropy=2.0,
                intensity_entropy=1.5,
                delta_h=-0.5,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                classification=Classification.suspicious,
                baseline_entropy=2.0,
                intensity_entropy=1.95,
                delta_h=-0.05,
                confidence=0.4,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
        ]

        stats = BatchDetectionStatistics.compute(results)

        assert stats.total == 3
        assert stats.unsafe_count == 1
        assert stats.suspicious_count == 1
        assert stats.benign_count == 1
        assert stats.indeterminate_count == 0
        assert abs(stats.unsafe_rate - 1/3) < 0.01
        assert abs(stats.benign_rate - 1/3) < 0.01
        assert stats.validity_rate == 1.0
        assert abs(stats.total_processing_time - 0.3) < 0.001

    def test_compute_with_indeterminate(self) -> None:
        """Test that indeterminate results are excluded from mean calculations."""
        results = [
            DetectionResult(
                classification=Classification.indeterminate,
                baseline_entropy=0.001,
                intensity_entropy=0.0005,
                delta_h=-0.0005,
                confidence=0.0,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
            DetectionResult(
                classification=Classification.benign,
                baseline_entropy=2.0,
                intensity_entropy=2.2,
                delta_h=0.2,
                confidence=0.6,
                timestamp=datetime.utcnow(),
                processing_time=0.1,
                baseline_token_count=10,
                intensity_token_count=10,
            ),
        ]

        stats = BatchDetectionStatistics.compute(results)

        assert stats.total == 2
        assert stats.indeterminate_count == 1
        assert stats.validity_rate == 0.5
        # Mean should only include the valid benign result
        assert stats.mean_delta_h == 0.2
        assert stats.mean_confidence == 0.6


# =============================================================================
# DetectionResult Tests
# =============================================================================


class TestDetectionResult:
    """Tests for DetectionResult."""

    def test_risk_level_property(self) -> None:
        """Test that risk_level property works."""
        result = DetectionResult(
            classification=Classification.unsafe_pattern,
            baseline_entropy=2.0,
            intensity_entropy=1.5,
            delta_h=-0.5,
            confidence=0.9,
            timestamp=datetime.utcnow(),
            processing_time=0.1,
            baseline_token_count=10,
            intensity_token_count=10,
        )
        assert result.risk_level == 3

    def test_frozen_dataclass(self) -> None:
        """Test that DetectionResult is immutable."""
        result = DetectionResult(
            classification=Classification.benign,
            baseline_entropy=2.0,
            intensity_entropy=2.5,
            delta_h=0.5,
            confidence=0.8,
            timestamp=datetime.utcnow(),
            processing_time=0.1,
            baseline_token_count=10,
            intensity_token_count=10,
        )
        with pytest.raises(Exception):  # frozen dataclass raises error
            result.confidence = 0.9  # type: ignore


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
