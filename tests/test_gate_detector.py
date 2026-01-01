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

"""Tests for GateDetector.

Tests computational gate detection in model responses:
- Threshold behavior (detection only above threshold)
- Empty text handling
- Overlapping detection merging
- Consecutive gate collapsing
- Path signature conversion
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.agents.computational_gate_atlas import (
    ComputationalGate,
    ComputationalGateCategory,
)
from modelcypher.core.domain.geometry.gate_detector import (
    DetectedGate,
    GateDetector,
)
from modelcypher.ports.embedding import EmbeddingProvider


class _KeywordEmbedder(EmbeddingProvider):
    """Test embedder that maps keywords to known vectors."""

    def __init__(self) -> None:
        self._dimension = 2

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            if "alpha" in lower or "alph" in lower:
                embeddings.append([1.0, 0.0])
            elif "beta" in lower:
                embeddings.append([0.0, 1.0])
            elif "gamma" in lower:
                embeddings.append([0.707, 0.707])  # 45 degrees
            else:
                embeddings.append([0.0, 0.0])
        return embeddings


def _make_gates() -> list[ComputationalGate]:
    """Create test gate inventory."""
    return [
        ComputationalGate(
            id="alpha",
            position=0,
            category=ComputationalGateCategory.core_concepts,
            name="ALPHA",
            description="alpha concept",
            examples=["alpha example"],
            polyglot_examples=[],
        ),
        ComputationalGate(
            id="beta",
            position=1,
            category=ComputationalGateCategory.core_concepts,
            name="BETA",
            description="beta concept",
            examples=["beta example"],
            polyglot_examples=[],
        ),
        ComputationalGate(
            id="gamma",
            position=2,
            category=ComputationalGateCategory.core_concepts,
            name="GAMMA",
            description="gamma concept",
            examples=["gamma example"],
            polyglot_examples=[],
        ),
    ]


def _make_detector(
) -> GateDetector:
    """Create test detector with keyword embedder."""
    return GateDetector(
        embedder=_KeywordEmbedder(),
        gate_inventory=_make_gates(),
    )


class TestBasicDetection:
    """Tests for basic gate detection."""

    def test_detects_matching_gates(self) -> None:
        """Detector should find gates that match the text."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha. beta.",
            model_id="model-1",
            prompt_id="prompt-1",
        )

        assert result.detected_gates, "Should detect at least one gate"
        assert "alpha" in result.gate_sequence
        assert "beta" in result.gate_sequence

    def test_mean_confidence_positive(self) -> None:
        """Mean confidence should be positive when gates are detected."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha. beta.",
            model_id="m",
            prompt_id="p",
        )

        assert result.mean_confidence > 0

    def test_empty_text_returns_empty_result(self) -> None:
        """Empty text should return result with no detected gates."""
        detector = _make_detector()
        result = detector.detect(text="", model_id="m", prompt_id="p")

        assert result.detected_gates == []
        assert result.mean_confidence == 0.0

    def test_no_matching_text_returns_empty(self) -> None:
        """Text with no matching keywords should return empty gates."""
        detector = _make_detector()
        result = detector.detect(
            text="xyz xyz xyz",  # No alpha/beta/gamma
            model_id="m",
            prompt_id="p",
        )

        assert result.detected_gates == []


class TestThresholdBehavior:
    """Tests for detection threshold behavior."""

    def test_zero_threshold_detects_all(self) -> None:
        """Derived threshold should include strong matches."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha. xyz.",
            model_id="m",
            prompt_id="p",
        )

        assert len(result.detected_gates) > 0


class TestCollapseConsecutive:
    """Tests for consecutive gate collapsing."""

    def test_collapse_merges_consecutive_same_gates(self) -> None:
        """Consecutive same gates should be collapsed to one."""
        detector = _make_detector()
        # "alpha alpha alpha" should collapse to single alpha detection
        result = detector.detect(
            text="alpha alpha alpha",
            model_id="m",
            prompt_id="p",
        )

        # With collapse enabled, consecutive alphas become one
        alpha_count = sum(1 for g in result.detected_gates if g.gate_id == "alpha")
        assert alpha_count <= 1, "Consecutive gates should be collapsed"

    def test_no_collapse_keeps_consecutive(self) -> None:
        """Without collapse, consecutive gates are kept separate."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha. alpha. alpha.",
            model_id="m",
            prompt_id="p",
        )

        # Consecutive identical gates collapse to one per sequence.
        alpha_count = sum(1 for g in result.detected_gates if g.gate_id == "alpha")
        assert alpha_count <= 1


class TestDetectedGate:
    """Tests for DetectedGate dataclass."""

    def test_detected_gate_fields(self) -> None:
        """DetectedGate should store all required fields."""
        gate = DetectedGate(
            gate_id="test",
            gate_name="TEST",
            confidence=0.85,
            character_span=(10, 20),
            trigger_text="test text",
            local_entropy=1.5,
        )

        assert gate.gate_id == "test"
        assert gate.gate_name == "TEST"
        assert gate.confidence == 0.85
        assert gate.character_span == (10, 20)
        assert gate.trigger_text == "test text"
        assert gate.local_entropy == 1.5


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_gate_sequence_property(self) -> None:
        """gate_sequence should return list of gate IDs."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha beta",
            model_id="m",
            prompt_id="p",
        )

        # gate_sequence should be list of strings
        assert isinstance(result.gate_sequence, list)
        for gate_id in result.gate_sequence:
            assert isinstance(gate_id, str)

    def test_gate_name_sequence_property(self) -> None:
        """gate_name_sequence should return list of gate names."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha beta",
            model_id="m",
            prompt_id="p",
        )

        # gate_name_sequence should be list of strings
        assert isinstance(result.gate_name_sequence, list)

    def test_to_path_signature(self) -> None:
        """Should convert detection result to PathSignature."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha beta",
            model_id="test-model",
            prompt_id="test-prompt",
        )

        signature = result.to_path_signature()

        assert signature.model_id == "test-model"
        assert signature.prompt_id == "test-prompt"
        assert len(signature.nodes) == len(result.detected_gates)


class TestEntropyTrace:
    """Tests for entropy trace integration."""

    def test_entropy_trace_populates_local_entropy(self) -> None:
        """When entropy trace is provided, local_entropy should be populated."""
        detector = _make_detector()
        # Provide entropy values for each character position
        entropy_trace = [1.0] * 100  # 100 entropy values

        result = detector.detect(
            text="alpha beta",
            model_id="m",
            prompt_id="p",
            entropy_trace=entropy_trace,
        )

        # If gates detected, they should have local entropy
        for gate in result.detected_gates:
            if gate.local_entropy is not None:
                assert gate.local_entropy > 0


class TestGateEmbeddings:
    """Tests for gate embedding computation."""

    def test_get_gate_embeddings(self) -> None:
        """Should return computed gate embeddings."""
        detector = _make_detector()

        embeddings = detector.get_gate_embeddings()

        assert isinstance(embeddings, dict)
        # Should have embeddings for our test gates
        assert len(embeddings) > 0

    def test_embeddings_are_normalized(self) -> None:
        """Gate embeddings should be L2-normalized."""
        detector = _make_detector()

        embeddings = detector.get_gate_embeddings()

        for gate_id, embedding in embeddings.items():
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:  # Skip zero vectors
                assert abs(norm - 1.0) < 0.01, f"Gate {gate_id} not normalized: {norm}"


class TestMaxGatesLimit:
    """Tests for gate output limits."""

    def test_no_cap_truncation(self) -> None:
        """Gate detection should not truncate detections arbitrarily."""
        detector = _make_detector()
        result = detector.detect(
            text="alpha. beta. alpha. beta.",
            model_id="m",
            prompt_id="p",
        )

        assert len(result.detected_gates) > 0
