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

Detects semantic concept activations in generated text using embeddings.

This is the concept-level analog to GateDetector, but operates at a higher
level of abstraction. While gates detect syntactic/code patterns, concepts
detect modality-invariant meaning like RECURRENCE, SYMMETRY, EMERGENCE.

Detection Algorithm:
The detector uses a sliding window approach with embedding-based similarity:
- Small windows (10-15 words) catch atomic concepts (RATIO, EQUIVALENCE)
- Medium windows (15-25 words) catch compound concepts (RECURRENCE, TRANSFORMATION)
- Large windows (25-40 words) catch abstract concepts (UNIVERSALITY, EMERGENCE)

Embedding-Based Detection:
Each concept has associated embeddings computed from support texts across modalities.
Text windows are embedded and compared to concept embeddings via cosine similarity.
The detection threshold is derived from the geometry of concept embeddings:
- Intra-concept similarity: how similar are examples of the same concept
- Inter-concept similarity: how similar are examples of different concepts
- Threshold = midpoint between max intra-concept and min inter-concept distance

This provides robustness: the same concept is detected whether expressed
in mathematical notation or poetic description, because the geometry of
the embedding space captures semantic similarity across modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.embedding import EmbeddingProvider


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
    """Configuration for concept detection.

    detection_threshold should be derived from the similarity distribution
    of concept embeddings, not guessed. Use from_similarity_distribution().
    """

    detection_threshold: float | None = None
    """Minimum similarity for concept detection.

    If None, derived from concept embedding similarity distribution.
    """

    detection_sigma: float = 2.0
    """Standard deviations above mean similarity for threshold derivation."""

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

    @classmethod
    def from_similarity_distribution(
        cls,
        similarities: list[float],
        sigma: float = 2.0,
        window_sizes: tuple[int, ...] = (10, 20, 30),
    ) -> "Configuration":
        """Derive detection threshold from observed similarity distribution.

        Args:
            similarities: List of cosine similarities between concept embeddings.
            sigma: Std devs above mean for threshold.
            window_sizes: Detection window sizes.

        Returns:
            Configuration with data-derived threshold.

        Raises:
            ValueError: If similarities list is empty.
        """
        if not similarities:
            raise ValueError(
                "Cannot derive threshold from empty similarities. "
                "Compute concept embedding similarities first."
            )

        import math

        n = len(similarities)
        mean = sum(similarities) / n
        variance = sum((s - mean) ** 2 for s in similarities) / n
        std = math.sqrt(variance)
        threshold = mean + sigma * std

        return cls(
            detection_threshold=max(0.0, min(1.0, threshold)),
            detection_sigma=sigma,
            window_sizes=window_sizes,
        )


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
        with_cross_modal = [
            c.cross_modal_confidence
            for c in self.detected_concepts
            if c.cross_modal_confidence is not None
        ]
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


@dataclass
class ConceptEmbeddings:
    """Embeddings for a single concept, including support examples.

    Attributes:
        concept_id: Unique identifier for the concept.
        category: Concept category for grouping.
        centroid: Mean embedding across all support texts.
        support_embeddings: Individual embeddings for each support text.
    """

    concept_id: str
    category: ConceptCategory
    centroid: list[float]
    support_embeddings: list[list[float]] = field(default_factory=list)


class ConceptDetector:
    """
    Detects semantic concept activations in generated text using embeddings.

    This class provides methods for detecting concepts in text using
    sliding window analysis and embedding-based similarity. Detection
    threshold is derived from the geometry of concept embeddings.

    Requires:
        - EmbeddingProvider for embedding text windows
        - Concept embeddings with support examples for separability computation
    """

    def __init__(
        self,
        embedding_provider: "EmbeddingProvider",
        config: Configuration | None = None,
    ):
        """Initialize with required embedding provider.

        Args:
            embedding_provider: Provider for text-to-embedding conversion.
            config: Optional configuration overrides.

        Raises:
            ValueError: If embedding_provider is None.
        """
        if embedding_provider is None:
            raise ValueError(
                "EmbeddingProvider is required for embedding-based concept detection. "
                "Concept detection operates on embedding geometry, not keywords."
            )
        self._embedding_provider = embedding_provider
        self.config = config or Configuration()
        self._derived_threshold: float | None = None
        self._concept_embeddings: dict[str, ConceptEmbeddings] = {}

    @property
    def effective_threshold(self) -> float:
        """Get the detection threshold (explicit or derived from embeddings).

        The threshold is derived from embedding separability:
        threshold = (min_inter_similarity + max_intra_similarity) / 2

        This ensures detection only fires when a text window is closer to
        a concept than concepts are to each other.
        """
        if self.config.detection_threshold is not None:
            return self.config.detection_threshold
        if self._derived_threshold is not None:
            return self._derived_threshold
        # Try to derive from concept embeddings
        if self._concept_embeddings:
            self._derived_threshold = self._derive_threshold_from_separability()
            return self._derived_threshold
        raise ValueError(
            "Cannot derive detection threshold: no concept embeddings available. "
            "Call set_concept_embeddings() first. Concept detection requires "
            "embeddings to operate on geometry, not keywords."
        )

    def set_concept_embeddings(
        self,
        concepts: list[ConceptEmbeddings],
    ) -> None:
        """Set concept embeddings for detection and threshold derivation.

        Args:
            concepts: List of ConceptEmbeddings with support examples.

        Raises:
            ValueError: If concepts list is empty or has fewer than 2 concepts.
        """
        if not concepts:
            raise ValueError(
                "Cannot set empty concept embeddings. At least 2 concepts "
                "are required for separability-based threshold computation."
            )
        if len(concepts) < 2:
            raise ValueError(
                "Cannot derive detection threshold: need at least 2 concepts "
                "for separability computation."
            )
        self._concept_embeddings = {c.concept_id: c for c in concepts}
        self._derived_threshold = None  # Reset to re-derive

    def _derive_threshold_from_separability(self) -> float:
        """Derive detection threshold from concept embedding separability.

        Uses the geometry of concept embeddings to determine threshold:
        - Computes inter-concept similarities (between different concepts)
        - Computes intra-concept similarities (within same concept, if support examples exist)
        - Threshold = midpoint between max_intra and min_inter

        If intra-concept examples don't exist, uses inter-concept statistics
        with a sigma-based threshold as fallback.

        Returns:
            Detection threshold derived from embedding geometry.

        Raises:
            ValueError: If concepts are not separable (min_inter <= max_intra).
        """
        import math

        if len(self._concept_embeddings) < 2:
            raise ValueError(
                "Cannot derive detection threshold: need at least 2 concepts."
            )

        concepts = list(self._concept_embeddings.values())

        # Compute inter-concept similarities (between different concepts)
        inter_similarities: list[float] = []
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1 :]:
                sim = self._cosine_similarity(concept_a.centroid, concept_b.centroid)
                inter_similarities.append(sim)

        if not inter_similarities:
            raise ValueError(
                "Cannot derive detection threshold: no inter-concept similarities."
            )

        # Compute intra-concept similarities (within same concept)
        intra_similarities: list[float] = []
        for concept in concepts:
            if len(concept.support_embeddings) >= 2:
                # Compare support embeddings within this concept
                for i, emb_a in enumerate(concept.support_embeddings):
                    for emb_b in concept.support_embeddings[i + 1 :]:
                        sim = self._cosine_similarity(emb_a, emb_b)
                        intra_similarities.append(sim)

        min_inter = min(inter_similarities)

        if intra_similarities:
            # Use separability-based threshold
            max_intra = max(intra_similarities)

            if min_inter <= max_intra:
                raise ValueError(
                    f"Concepts are not separable in embedding space: "
                    f"min_inter_similarity ({min_inter:.4f}) <= max_intra_similarity ({max_intra:.4f}). "
                    f"This means some concepts are more similar to each other than their "
                    f"own support examples are to each other. Improve concept definitions "
                    f"or use a different embedding model."
                )

            # Threshold is midpoint for optimal separation
            threshold = (min_inter + max_intra) / 2
        else:
            # No intra-concept examples - use inter-concept statistics
            n = len(inter_similarities)
            mean = sum(inter_similarities) / n
            variance = sum((s - mean) ** 2 for s in inter_similarities) / n
            std = math.sqrt(variance)
            # Threshold above mean by sigma standard deviations
            threshold = mean + self.config.detection_sigma * std

        return max(0.0, min(1.0, threshold))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

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
        final_detections = (
            self._collapse_consecutive(deduped) if self.config.collapse_consecutive else deduped
        )

        # Limit max detections
        limited = final_detections[: self.config.max_concepts_per_response]

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
        Detect the best matching concept in a window using embedding similarity.

        Embeds the text window and computes cosine similarity to all concept
        centroids. Returns the highest-similarity concept if it exceeds the
        detection threshold.

        Args:
            text: Text window to analyze.
            character_span: Character positions (start, end) in original text.

        Returns:
            DetectedConcept if a concept is detected, None otherwise.

        Raises:
            ValueError: If concept embeddings are not set.
        """
        if not self._concept_embeddings:
            raise ValueError(
                "Cannot detect concepts: no concept embeddings set. "
                "Call set_concept_embeddings() before detection."
            )

        # Embed the text window
        text_embeddings = self._embedding_provider.embed([text])
        if not text_embeddings:
            return None
        text_embedding = text_embeddings[0]

        # Find best matching concept
        best_concept: ConceptEmbeddings | None = None
        best_similarity = 0.0

        for concept in self._concept_embeddings.values():
            similarity = self._cosine_similarity(text_embedding, concept.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept

        # Check if best match exceeds threshold
        if best_concept is not None and best_similarity >= self.effective_threshold:
            return DetectedConcept(
                concept_id=best_concept.concept_id,
                category=best_concept.category,
                confidence=best_similarity,
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


# Default concept definitions with support texts for embedding
DEFAULT_CONCEPT_DEFINITIONS: dict[str, tuple[ConceptCategory, list[str]]] = {
    "recurrence": (
        ConceptCategory.STRUCTURAL,
        [
            "The pattern repeats at regular intervals, each iteration building on the previous.",
            "Fibonacci sequences emerge from recursive self-reference.",
            "The function calls itself, establishing a recurrence relation.",
            "Seasonal cycles return with predictable regularity.",
        ],
    ),
    "symmetry": (
        ConceptCategory.STRUCTURAL,
        [
            "The left side mirrors the right in perfect bilateral symmetry.",
            "Rotational symmetry preserves the figure under transformation.",
            "The equation remains unchanged when variables are swapped.",
            "Nature exhibits symmetry in snowflakes and flower petals.",
        ],
    ),
    "ratio": (
        ConceptCategory.RELATIONAL,
        [
            "The golden ratio phi appears in growth patterns throughout nature.",
            "The proportion of parts to whole reveals the underlying ratio.",
            "Scale factors maintain constant ratios during transformation.",
            "Musical harmony emerges from simple frequency ratios.",
        ],
    ),
    "equivalence": (
        ConceptCategory.RELATIONAL,
        [
            "These structures are isomorphic, equivalent in their essential properties.",
            "The equation states that both sides are equal, expressing equivalence.",
            "Different representations encode the same underlying information.",
            "Logical equivalence means truth values always match.",
        ],
    ),
    "transformation": (
        ConceptCategory.TRANSFORMATIONAL,
        [
            "The morphism maps elements from one structure to another.",
            "Linear transformations preserve vector space operations.",
            "The function converts input to output through defined operations.",
            "Metamorphosis transforms the organism through distinct stages.",
        ],
    ),
    "emergence": (
        ConceptCategory.EMERGENT,
        [
            "Complex patterns emerge from simple local interactions.",
            "Self-organization arises without central control.",
            "Collective behavior exhibits properties absent in individual components.",
            "Novel features appear at higher levels of organization.",
        ],
    ),
    "causality": (
        ConceptCategory.FOUNDATIONAL,
        [
            "The cause precedes the effect in a deterministic chain.",
            "Because the condition holds, the consequence follows.",
            "Causal inference requires controlling for confounding variables.",
            "The effect cannot occur without the necessary cause.",
        ],
    ),
    "ordering": (
        ConceptCategory.FOUNDATIONAL,
        [
            "Elements form a total ordering from least to greatest.",
            "The sequence proceeds step by step in defined order.",
            "Partial orders allow some elements to be incomparable.",
            "The ranking establishes precedence among alternatives.",
        ],
    ),
}


def create_default_detector(
    embedding_provider: "EmbeddingProvider",
    config: Configuration | None = None,
) -> ConceptDetector:
    """Create a ConceptDetector with default concept embeddings.

    This factory function creates a ready-to-use detector with the default
    concept inventory embedded using the provided embedding provider.

    Args:
        embedding_provider: Provider for text-to-embedding conversion.
        config: Optional configuration overrides.

    Returns:
        Configured ConceptDetector with concept embeddings set.

    Example:
        from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
        from modelcypher.core.domain.geometry.concept_detector import create_default_detector

        embedder = EmbeddingDefaults.make_default_embedder()
        if embedder is None:
            raise RuntimeError("No embedding provider available")

        detector = create_default_detector(embedder)
        result = detector.detect(text, model_id="model", prompt_id="prompt")
    """
    detector = ConceptDetector(embedding_provider, config)

    # Build concept embeddings from default definitions
    concept_embeddings: list[ConceptEmbeddings] = []

    for concept_id, (category, support_texts) in DEFAULT_CONCEPT_DEFINITIONS.items():
        # Embed all support texts for this concept
        support_embeddings = embedding_provider.embed(support_texts)

        # Compute centroid as mean of support embeddings
        if support_embeddings:
            n = len(support_embeddings)
            dim = len(support_embeddings[0])
            centroid = [
                sum(emb[i] for emb in support_embeddings) / n for i in range(dim)
            ]
        else:
            centroid = []

        concept_embeddings.append(
            ConceptEmbeddings(
                concept_id=concept_id,
                category=category,
                centroid=centroid,
                support_embeddings=support_embeddings,
            )
        )

    detector.set_concept_embeddings(concept_embeddings)
    return detector
