# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

import modelcypher.core.domain.entropy.entropy_pattern_detector as detector_mod
from modelcypher.core.domain.entropy.entropy_pattern_detector import (
    DistressDetectionResult,
    EntropyPattern,
    EntropyPatternAnalyzer,
    _Statistics,
)


def test_entropy_pattern_empty_properties_and_serialization(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(detector_mod, "get_default_backend", lambda: b)

    empty = EntropyPattern.empty()
    assert empty.sample_count == 0
    assert empty.is_rising is False
    assert empty.is_falling is False
    assert empty.sustained_significance == 0.0

    pattern = EntropyPattern(
        trend_slope=-0.2,
        volatility=1.0,
        entropy_mean=2.0,
        entropy_std_dev=0.5,
        variance_mean=0.2,
        variance_std_dev=0.1,
        entropy_variance_correlation=0.4,
        sustained_high_count=2,
        peak_entropy=3.0,
        min_entropy=1.0,
        anomaly_indices=(1, 3),
        sample_count=4,
    )
    assert pattern.is_falling is True
    assert pattern.is_rising is False
    assert pattern.sustained_significance > 0.0
    assert pattern.to_dict()["anomalyIndices"] == [1, 3]


def test_statistics_helpers(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(detector_mod, "get_default_backend", lambda: b)

    assert _Statistics.mean([]) == 0.0
    assert _Statistics.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    assert _Statistics.standard_deviation([1.0]) == 0.0
    assert _Statistics.standard_deviation([1.0, 2.0, 3.0], mean=2.0) == pytest.approx(1.0)


def test_analyzer_analyze_detect_distress_and_to_dict(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(detector_mod, "get_default_backend", lambda: b)
    monkeypatch.setattr(detector_mod, "find_magnitude_gap_threshold", lambda _vals, eps=0.0: 1.0)

    analyzer = EntropyPatternAnalyzer()
    samples = [(1.0, 0.1), (2.0, 0.2), (8.0, 0.7), (2.2, 0.21)]

    pattern = analyzer.analyze(samples)

    assert pattern.sample_count == 4
    assert pattern.peak_entropy == pytest.approx(8.0)
    assert pattern.min_entropy == pytest.approx(1.0)
    assert pattern.sustained_high_count >= 1
    assert pattern.anomaly_indices

    distress = analyzer.detect_distress(pattern)
    assert isinstance(distress, DistressDetectionResult)
    assert distress.sample_count == 4
    assert distress.to_dict()["sustainedHighCount"] == distress.sustained_high_count

    assert analyzer.detect_distress(EntropyPattern.empty()) is None


def test_internal_methods_cover_edge_cases(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(detector_mod, "get_default_backend", lambda: b)

    analyzer = EntropyPatternAnalyzer()

    assert analyzer._compute_trend([1.0]) == 0.0
    assert analyzer._count_sustained_high([1.0, 3.0, 4.0, 1.0, 5.0], threshold=2.5) == 2

    assert analyzer._detect_anomalies([1.0], mean=1.0, std_dev=0.1) == []
    assert analyzer._detect_anomalies([1.0, 1.0], mean=1.0, std_dev=0.0) == []

    monkeypatch.setattr(detector_mod, "find_magnitude_gap_threshold", lambda _vals, eps=0.0: 0.0)
    assert analyzer._detect_anomalies([1.0, 10.0, 1.2], mean=4.0, std_dev=4.0) == []
