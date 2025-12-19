from __future__ import annotations

import pytest

from modelcypher.core.domain.topological_fingerprint import TopologicalFingerprint


def test_compute_single_point() -> None:
    fingerprint = TopologicalFingerprint.compute([[0.0, 0.0]])
    assert fingerprint.summary.component_count == 1
    assert fingerprint.summary.cycle_count == 0
    assert fingerprint.diagram.points == []


def test_compare_identical_fingerprints() -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    fingerprint = TopologicalFingerprint.compute(points, max_dimension=1)
    comparison = TopologicalFingerprint.compare(fingerprint, fingerprint)

    assert comparison.bottleneck_distance == pytest.approx(0.0, abs=1e-6)
    assert comparison.wasserstein_distance == pytest.approx(0.0, abs=1e-6)
    assert comparison.betti_difference == 0
    assert comparison.similarity_score == pytest.approx(1.0, abs=1e-6)
    assert comparison.is_compatible is True
