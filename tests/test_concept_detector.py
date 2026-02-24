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

"""Tests for concept_detector.py.

Tests cover:
- ProbeEmbedding dataclass
- ConceptDetector initialization and validation
- detect() method with various inputs
- compare_results() static method
- _segment_text() static method
- _collapse_consecutive() static method
- Edge cases: empty inputs, single probe, no matches
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_detector import (
    ConceptDetector,
    ProbeEmbedding,
)
from modelcypher.core.domain.geometry.types import (
    DetectedConcept,
    DetectionResult,
)

# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockProbe:
    """Mock implementation of AtlasProbeProtocol."""

    probe_id: str
    name: str
    description: str
    support_texts: Sequence[str]
    source: Any
    domain: Any
    category_name: str
    cross_domain_weight: float


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self._call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings based on text hash."""
        if not texts:
            return []
        result = []
        for text in texts:
            # Deterministic embedding based on text hash
            seed = hash(text) % 1000
            embedding = [(seed + i) / 1000.0 for i in range(self.dimension)]
            # Normalize to unit length
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            result.append(embedding)
        self._call_count += 1
        return result


# =============================================================================
# ProbeEmbedding Tests
# =============================================================================


class TestProbeEmbedding:
    """Tests for ProbeEmbedding dataclass."""

    def test_probe_embedding_fields(self):
        """ProbeEmbedding stores all required fields."""
        backend = get_default_backend()
        centroid = backend.array([1.0, 0.0, 0.0])
        support = backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        pe = ProbeEmbedding(
            probe_id="test_probe",
            category="test_category",
            centroid=centroid,
            support_matrix=support,
            cohesion_floor=0.8,
        )

        assert pe.probe_id == "test_probe"
        assert pe.category == "test_category"
        assert pe.cohesion_floor == 0.8

    def test_probe_embedding_frozen(self):
        """ProbeEmbedding is frozen (immutable)."""
        backend = get_default_backend()
        pe = ProbeEmbedding(
            probe_id="test",
            category="cat",
            centroid=backend.array([1.0]),
            support_matrix=backend.array([[1.0]]),
            cohesion_floor=0.5,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            pe.probe_id = "changed"


# =============================================================================
# ConceptDetector Initialization Tests
# =============================================================================


class TestConceptDetectorInit:
    """Tests for ConceptDetector initialization."""

    def test_init_requires_embedding_provider(self):
        """ConceptDetector requires non-None embedding provider."""
        probes = [MockProbe("p1", "Name", "Desc", ["text"], None, None, "cat", 1.0)]
        with pytest.raises(ValueError, match="EmbeddingProvider is required"):
            ConceptDetector(embedding_provider=None, probes=probes)

    def test_init_requires_non_empty_probes(self):
        """ConceptDetector requires non-empty probe list."""
        provider = MockEmbeddingProvider()
        with pytest.raises(ValueError, match="non-empty probe inventory"):
            ConceptDetector(embedding_provider=provider, probes=[])

    @patch("modelcypher.core.domain.geometry.concept_detector.get_or_compute_embeddings_sync")
    def test_init_builds_probe_embeddings(self, mock_embeddings):
        """ConceptDetector builds embeddings for probes on init."""
        backend = get_default_backend()
        mock_embeddings.return_value = backend.array([[1.0, 0.0], [0.0, 1.0]])

        provider = MockEmbeddingProvider(dimension=2)
        probes = [
            MockProbe("p1", "Probe1", "Desc", ["support1", "support2"], None, None, "cat", 1.0),
        ]

        detector = ConceptDetector(embedding_provider=provider, probes=probes)
        assert len(detector._probe_embeddings) == 1
        assert detector._probe_embeddings[0].probe_id == "p1"


# =============================================================================
# _segment_text Tests
# =============================================================================


class TestSegmentText:
    """Tests for _segment_text static method."""

    def test_segment_text_single_sentence(self):
        """Single sentence creates one segment."""
        segments = ConceptDetector._segment_text("This is a sentence.")
        assert len(segments) == 1
        assert segments[0][2] == "This is a sentence."

    def test_segment_text_multiple_sentences(self):
        """Multiple sentences create multiple segments."""
        text = "First sentence. Second sentence. Third sentence."
        segments = ConceptDetector._segment_text(text)
        assert len(segments) == 3

    def test_segment_text_with_newlines(self):
        """Newlines also act as segment boundaries."""
        text = "First line\nSecond line"
        segments = ConceptDetector._segment_text(text)
        assert len(segments) == 2

    def test_segment_text_with_exclamation_question(self):
        """Exclamation and question marks are segment boundaries."""
        text = "Hello! How are you? Fine."
        segments = ConceptDetector._segment_text(text)
        assert len(segments) == 3

    def test_segment_text_empty_string(self):
        """Empty string returns no segments."""
        segments = ConceptDetector._segment_text("")
        assert len(segments) == 0

    def test_segment_text_whitespace_only(self):
        """Whitespace-only string returns no segments."""
        segments = ConceptDetector._segment_text("   \n  ")
        assert len(segments) == 0

    def test_segment_text_preserves_positions(self):
        """Segments include correct character positions."""
        text = "First. Second."
        segments = ConceptDetector._segment_text(text)
        # Check that positions are valid
        for start, end, segment in segments:
            assert start >= 0
            assert end <= len(text)
            assert start < end


# =============================================================================
# _collapse_consecutive Tests
# =============================================================================


class TestCollapseConsecutive:
    """Tests for _collapse_consecutive static method."""

    def test_collapse_empty_list(self):
        """Empty list returns empty list."""
        result = ConceptDetector._collapse_consecutive([])
        assert result == []

    def test_collapse_single_detection(self):
        """Single detection returns unchanged."""
        detection = DetectedConcept(
            concept_id="c1",
            category="cat",
            similarity=0.9,
            character_span=slice(0, 10),
            trigger_text="text",
            cross_modal_similarity=None,
        )
        result = ConceptDetector._collapse_consecutive([detection])
        assert result == [detection]

    def test_collapse_different_concepts(self):
        """Different concepts are preserved."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        d2 = DetectedConcept("c2", "cat", 0.8, slice(5, 10), "t2", None)
        result = ConceptDetector._collapse_consecutive([d1, d2])
        assert len(result) == 2

    def test_collapse_consecutive_same_concepts(self):
        """Consecutive same concepts are collapsed."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        d2 = DetectedConcept("c1", "cat", 0.8, slice(5, 10), "t2", None)
        d3 = DetectedConcept("c1", "cat", 0.7, slice(10, 15), "t3", None)
        result = ConceptDetector._collapse_consecutive([d1, d2, d3])
        assert len(result) == 1
        assert result[0] == d1  # First one is preserved

    def test_collapse_alternating_concepts(self):
        """Alternating concepts are all preserved."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        d2 = DetectedConcept("c2", "cat", 0.8, slice(5, 10), "t2", None)
        d3 = DetectedConcept("c1", "cat", 0.7, slice(10, 15), "t3", None)
        result = ConceptDetector._collapse_consecutive([d1, d2, d3])
        assert len(result) == 3


# =============================================================================
# compare_results Tests
# =============================================================================


class TestCompareResults:
    """Tests for compare_results static method."""

    def test_compare_results_identical(self):
        """Identical results have full overlap."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        result_a = DetectionResult(
            model_id="model_a",
            prompt_id="p1",
            response_text="text",
            detected_concepts=(d1,),
            mean_similarity=0.9,
            mean_cross_modal_similarity=None,
        )

        comparison = ConceptDetector.compare_results(result_a, result_a)

        assert comparison.model_a == "model_a"
        assert comparison.model_b == "model_a"
        assert "c1" in comparison.aligned_concepts
        assert len(comparison.unique_to_a) == 0
        assert len(comparison.unique_to_b) == 0

    def test_compare_results_disjoint(self):
        """Disjoint results have no overlap."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        d2 = DetectedConcept("c2", "cat", 0.8, slice(0, 5), "t2", None)

        result_a = DetectionResult("model_a", "p1", "text", (d1,), 0.9, None)
        result_b = DetectionResult("model_b", "p1", "text", (d2,), 0.8, None)

        comparison = ConceptDetector.compare_results(result_a, result_b)

        assert len(comparison.aligned_concepts) == 0
        assert "c1" in comparison.unique_to_a
        assert "c2" in comparison.unique_to_b

    def test_compare_results_partial_overlap(self):
        """Partially overlapping results show both shared and unique."""
        d1 = DetectedConcept("c1", "cat", 0.9, slice(0, 5), "t1", None)
        d2 = DetectedConcept("c2", "cat", 0.8, slice(5, 10), "t2", None)
        d3 = DetectedConcept("c3", "cat", 0.7, slice(10, 15), "t3", None)

        result_a = DetectionResult("model_a", "p1", "text", (d1, d2), 0.85, None)
        result_b = DetectionResult("model_b", "p1", "text", (d2, d3), 0.75, None)

        comparison = ConceptDetector.compare_results(result_a, result_b)

        assert "c2" in comparison.aligned_concepts
        assert "c1" in comparison.unique_to_a
        assert "c3" in comparison.unique_to_b


# =============================================================================
# detect() Tests
# =============================================================================


class TestDetect:
    """Tests for detect method (requires full mocking)."""

    @patch("modelcypher.core.domain.geometry.concept_detector.get_or_compute_embeddings_sync")
    def test_detect_empty_response(self, mock_embeddings):
        """Empty response returns empty detection result."""
        backend = get_default_backend()
        mock_embeddings.return_value = backend.array([[1.0, 0.0], [0.0, 1.0]])

        provider = MockEmbeddingProvider(dimension=2)
        probes = [MockProbe("p1", "Probe", "Desc", ["support"], None, None, "cat", 1.0)]

        detector = ConceptDetector(embedding_provider=provider, probes=probes)

        result = detector.detect(response="", model_id="test_model", prompt_id="p1")

        assert result.model_id == "test_model"
        assert result.prompt_id == "p1"
        assert len(result.detected_concepts) == 0
        assert result.mean_similarity == 0.0

    @patch("modelcypher.core.domain.geometry.concept_detector.get_or_compute_embeddings_sync")
    def test_detect_whitespace_response(self, mock_embeddings):
        """Whitespace-only response returns empty detection result."""
        backend = get_default_backend()
        mock_embeddings.return_value = backend.array([[1.0, 0.0]])

        provider = MockEmbeddingProvider(dimension=2)
        probes = [MockProbe("p1", "Probe", "Desc", ["support"], None, None, "cat", 1.0)]

        detector = ConceptDetector(embedding_provider=provider, probes=probes)

        result = detector.detect(response="   \n   ", model_id="model", prompt_id="p1")

        assert len(result.detected_concepts) == 0

    @patch("modelcypher.core.domain.geometry.concept_detector.get_or_compute_embeddings_sync")
    def test_detect_returns_detection_result(self, mock_embeddings):
        """detect() returns DetectionResult with correct fields."""
        backend = get_default_backend()
        mock_embeddings.return_value = backend.array([[1.0, 0.0], [0.0, 1.0]])

        provider = MockEmbeddingProvider(dimension=2)
        probes = [MockProbe("p1", "Probe", "Desc", ["support1", "support2"], None, None, "cat", 1.0)]

        detector = ConceptDetector(embedding_provider=provider, probes=probes)

        result = detector.detect(
            response="This is a test sentence.",
            model_id="test_model",
            prompt_id="test_prompt",
        )

        assert isinstance(result, DetectionResult)
        assert result.model_id == "test_model"
        assert result.prompt_id == "test_prompt"
        assert result.response_text == "This is a test sentence."
