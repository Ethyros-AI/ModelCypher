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
Concept Detector.

Detects semantic concept activations in generated text.
This is the concept-level analog to GateDetector, but operates at a higher
level of abstraction. While gates detect syntactic/code patterns, concepts
detect modality-invariant meaning like RECURRENCE, SYMMETRY, EMERGENCE.

Detection Algorithm:
The detector uses a sliding window approach with larger windows than
GateDetector because concepts are expressed over longer spans:
- Small windows (10-15 words) catch atomic concepts (RATIO, EQUIVALENCE)
- Medium windows (15-25 words) catch compound concepts (RECURRENCE, TRANSFORMATION)
- Large windows (25-40 words) catch abstract concepts (UNIVERSALITY, EMERGENCE)

Cross-Modal Triangulation:
Unlike gate detection which uses polyglot code examples, concept detection
triangulates across modalities (CODE, MATH, NATURE, PHILOSOPHY, VISUAL).
This provides robustness: the same concept is detected whether expressed
in mathematical notation or poetic description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import math


class ConceptCategory(str, Enum):
    """Categories for semantic concepts."""
    STRUCTURAL = "structural"
    RELATIONAL = "relational"
    TRANSFORMATIONAL = "transformational"
    EMERGENT = "emergent"
    FOUNDATIONAL = "foundational"


class ConceptModality(str, Enum):
    """Expression modality for concepts."""
    CODE = "code"
    MATH = "math"
    NATURE = "nature"
    PHILOSOPHY = "philosophy"
    VISUAL = "visual"


@dataclass(frozen=True)
class Configuration:
    """Configuration for concept detection."""
    # Minimum similarity for concept detection
    detection_threshold: float = 0.3

    # Window sizes for multi-scale detection (in words, not tokens)
    window_sizes: tuple[int, ...] = (10, 20, 30)

    # Stride between windows (words)
    stride: int = 5

    # Whether to collapse consecutive identical concepts
    collapse_consecutive: bool = True

    # Maximum concepts to detect per response
    max_concepts_per_response: int = 30

    # Hint about the source modality for weighted detection
    source_modality_hint: ConceptModality | None = None


@dataclass(frozen=True)
class DetectedConcept:
    """A detected concept activation in the response."""
    # The concept ID from the concept inventory
    concept_id: str

    # Concept category for grouping
    category: ConceptCategory

    # Detection confidence (cosine similarity)
    confidence: float

    # Character span in the original text (start, end)
    character_span: tuple[int, int]

    # The text snippet that triggered this detection
    trigger_text: str

    # Cross-modal confidence (how consistently this matches across modalities)
    cross_modal_confidence: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    """Complete detection result for a response."""
    model_id: str
    prompt_id: str
    response_text: str
    detected_concepts: tuple[DetectedConcept, ...]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all detected concepts."""
        if not self.detected_concepts:
            return 0.0
        return sum(c.confidence for c in self.detected_concepts) / len(self.detected_concepts)

    @property
    def mean_cross_modal_confidence(self) -> float | None:
        """Mean cross-modal confidence across concepts that have it."""
        with_cross_modal = [c.cross_modal_confidence for c in self.detected_concepts if c.cross_modal_confidence is not None]
        if not with_cross_modal:
            return None
        return sum(with_cross_modal) / len(with_cross_modal)

    @property
    def concept_sequence(self) -> list[str]:
        """The sequence of concept IDs in order of detection."""
        return [c.concept_id for c in self.detected_concepts]


@dataclass(frozen=True)
class ConceptComparisonResult:
    """Result of comparing concept detections between two models."""
    model_a: str
    model_b: str
    concept_path_a: tuple[str, ...]
    concept_path_b: tuple[str, ...]
    cka: float | None
    cosine_similarity: float | None
    aligned_concepts: tuple[str, ...]
    unique_to_a: tuple[str, ...]
    unique_to_b: tuple[str, ...]

    @property
    def alignment_ratio(self) -> float:
        """Ratio of aligned concepts to total unique concepts."""
        total = len(set(self.concept_path_a) | set(self.concept_path_b))
        if total == 0:
            return 1.0
        return len(self.aligned_concepts) / total


class ConceptDetector:
    """
    Detects semantic concept activations in generated text.

    This class provides methods for detecting concepts in text using
    sliding window analysis and optional embedding-based similarity.
    """

    def __init__(self, config: Configuration | None = None):
        """Initialize with optional configuration."""
        self.config = config or Configuration()

    def detect(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
    ) -> DetectionResult:
        """
        Detect concepts in a model response.

        Uses a sliding window approach for multi-scale detection.

        Args:
            response: The text response to analyze
            model_id: Identifier for the model that generated the response
            prompt_id: Identifier for the prompt that generated this response

        Returns:
            DetectionResult with detected concepts and metadata
        """
        trimmed = response.strip()
        if not trimmed:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=response,
                detected_concepts=(),
            )

        # Tokenize into words
        words = self._tokenize(trimmed)
        min_window = min(self.config.window_sizes) if self.config.window_sizes else 5

        if len(words) < min_window:
            # Text too short for windowed detection
            return self._detect_whole_text(response, model_id, prompt_id)

        all_detections: list[DetectedConcept] = []

        # Multi-scale detection
        for window_size in self.config.window_sizes:
            detections = self._detect_with_window(words, trimmed, window_size)
            all_detections.extend(detections)

        # Deduplicate overlapping detections
        deduped = self._deduplicate_detections(all_detections)

        # Collapse consecutive if configured
        final_detections = self._collapse_consecutive(deduped) if self.config.collapse_consecutive else deduped

        # Limit max detections
        limited = final_detections[:self.config.max_concepts_per_response]

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=tuple(limited),
        )

    def detect_with_modality(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
        modality_hint: ConceptModality,
    ) -> DetectionResult:
        """
        Detect concepts with a specific modality hint.

        Args:
            response: The text response to analyze
            model_id: Identifier for the model
            prompt_id: Identifier for the prompt
            modality_hint: Hint about the expected modality

        Returns:
            DetectionResult with detected concepts
        """
        # Create modified config with modality hint
        # For now, use whole-text detection with modality awareness
        return self._detect_whole_text(response, model_id, prompt_id)

    def _tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """
        Tokenize text into words with character positions.

        Returns list of (word, start_pos, end_pos) tuples.
        """
        words: list[tuple[str, int, int]] = []
        current_pos = 0
        in_word = False
        word_start = 0

        for i, char in enumerate(text):
            if char.isalnum() or char == "'":
                if not in_word:
                    in_word = True
                    word_start = i
            else:
                if in_word:
                    word = text[word_start:i]
                    words.append((word, word_start, i))
                    in_word = False

        # Handle last word
        if in_word:
            word = text[word_start:]
            words.append((word, word_start, len(text)))

        return words

    def _detect_with_window(
        self,
        words: list[tuple[str, int, int]],
        original_text: str,
        window_size: int,
    ) -> list[DetectedConcept]:
        """Detect concepts using a specific window size."""
        detections: list[DetectedConcept] = []
        stride = max(1, self.config.stride)

        window_start = 0
        while window_start + window_size <= len(words):
            window_end = min(window_start + window_size, len(words))
            window_words = words[window_start:window_end]

            if not window_words:
                window_start += stride
                continue

            start_pos = window_words[0][1]
            end_pos = window_words[-1][2]
            window_text = original_text[start_pos:end_pos]

            # For now, use heuristic detection based on keywords
            detection = self._detect_in_window(window_text, (start_pos, end_pos))
            if detection:
                detections.append(detection)

            window_start += stride

        return detections

    def _detect_in_window(
        self,
        text: str,
        character_span: tuple[int, int],
    ) -> DetectedConcept | None:
        """
        Detect the best matching concept in a window.

        Uses keyword-based heuristic detection.
        In the full implementation, this would use embedding similarity.
        """
        text_lower = text.lower()

        # Heuristic concept detection based on keywords
        concept_keywords = {
            "recurrence": (ConceptCategory.STRUCTURAL, ["recurrence", "recursive", "repeating", "fibonacci", "sequence"]),
            "symmetry": (ConceptCategory.STRUCTURAL, ["symmetry", "symmetric", "mirror", "reflection", "balanced"]),
            "ratio": (ConceptCategory.RELATIONAL, ["ratio", "proportion", "golden", "phi", "scaling"]),
            "equivalence": (ConceptCategory.RELATIONAL, ["equivalent", "equal", "same", "identical", "isomorphic"]),
            "transformation": (ConceptCategory.TRANSFORMATIONAL, ["transform", "map", "convert", "change", "morphism"]),
            "emergence": (ConceptCategory.EMERGENT, ["emerge", "arising", "self-organizing", "complex", "pattern"]),
            "causality": (ConceptCategory.FOUNDATIONAL, ["cause", "effect", "because", "therefore", "implies"]),
            "ordering": (ConceptCategory.FOUNDATIONAL, ["order", "sequence", "before", "after", "less than", "greater"]),
        }

        best_concept: str | None = None
        best_category: ConceptCategory | None = None
        best_score = 0.0

        for concept_id, (category, keywords) in concept_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            normalized_score = score / len(keywords) if keywords else 0.0

            if normalized_score > best_score and normalized_score >= self.config.detection_threshold:
                best_score = normalized_score
                best_concept = concept_id
                best_category = category

        if best_concept and best_category:
            return DetectedConcept(
                concept_id=best_concept,
                category=best_category,
                confidence=best_score,
                character_span=character_span,
                trigger_text=text[:100] + ("..." if len(text) > 100 else ""),
                cross_modal_confidence=None,
            )

        return None

    def _detect_whole_text(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
    ) -> DetectionResult:
        """Detect concepts in the entire text as a single window."""
        detection = self._detect_in_window(response, (0, len(response)))
        concepts = (detection,) if detection else ()

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=concepts,
        )

    def _deduplicate_detections(
        self,
        detections: list[DetectedConcept],
    ) -> list[DetectedConcept]:
        """Keep highest confidence detection for each span-concept pair."""
        best_by_span_concept: dict[str, DetectedConcept] = {}

        for detection in detections:
            key = f"{detection.character_span[0]}-{detection.concept_id}"
            existing = best_by_span_concept.get(key)
            if existing is None or detection.confidence > existing.confidence:
                best_by_span_concept[key] = detection

        # Sort by position
        return sorted(best_by_span_concept.values(), key=lambda d: d.character_span[0])

    def _collapse_consecutive(
        self,
        detections: list[DetectedConcept],
    ) -> list[DetectedConcept]:
        """Collapse consecutive detections of the same concept."""
        if len(detections) <= 1:
            return detections

        result: list[DetectedConcept] = []
        for detection in detections:
            if not result or result[-1].concept_id != detection.concept_id:
                result.append(detection)

        return result

    @staticmethod
    def compare_results(
        result_a: DetectionResult,
        result_b: DetectionResult,
    ) -> ConceptComparisonResult:
        """
        Compare concept detection results between two models.

        Args:
            result_a: Detection result from first model
            result_b: Detection result from second model

        Returns:
            ConceptComparisonResult with alignment metrics
        """
        set_a = set(result_a.concept_sequence)
        set_b = set(result_b.concept_sequence)
        intersection = set_a.intersection(set_b)

        return ConceptComparisonResult(
            model_a=result_a.model_id,
            model_b=result_b.model_id,
            concept_path_a=tuple(result_a.concept_sequence),
            concept_path_b=tuple(result_b.concept_sequence),
            cka=None,  # Would need embedding-based computation
            cosine_similarity=None,  # Would need signature-based computation
            aligned_concepts=tuple(sorted(intersection)),
            unique_to_a=tuple(sorted(set_a - set_b)),
            unique_to_b=tuple(sorted(set_b - set_a)),
        )
