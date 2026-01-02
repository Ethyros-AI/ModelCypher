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
Concept detection using embedding geometry.

Embeds probe support texts, computes Frechet centroids, and uses probe geometry
to decide whether a response segment activates a concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterable

import logging

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.utils.text import truncate

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedConcept:
    """A detected concept activation in the response."""

    concept_id: str
    category: str
    similarity: float
    character_span: tuple[int, int]
    trigger_text: str
    cross_modal_similarity: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    """Complete detection result for a response."""

    model_id: str
    prompt_id: str
    response_text: str
    detected_concepts: tuple[DetectedConcept, ...]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean_similarity(self) -> float:
        """Mean similarity across all detected concepts."""
        if not self.detected_concepts:
            return 0.0
        return sum(c.similarity for c in self.detected_concepts) / len(self.detected_concepts)

    @property
    def mean_cross_modal_similarity(self) -> float | None:
        """Mean cross-modal similarity across concepts that have it."""
        with_cross_modal = [
            c.cross_modal_similarity
            for c in self.detected_concepts
            if c.cross_modal_similarity is not None
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


@dataclass(frozen=True)
class ProbeEmbedding:
    """Embeddings for a single probe, including support examples."""

    probe_id: str
    category: str
    centroid: list[float]
    support_embeddings: list[list[float]]
    cohesion_floor: float


class ConceptDetector:
    """
    Detect semantic concept activations in text using embeddings.

    Requires:
        - EmbeddingProvider for text-to-embedding conversion
        - Probe inventory with support texts for each concept
    """

    def __init__(
        self,
        embedding_provider: "EmbeddingProvider",
        probes: Iterable[AtlasProbeProtocol],
        backend: "Backend | None" = None,
    ) -> None:
        if embedding_provider is None:
            raise ValueError(
                "EmbeddingProvider is required for embedding-based concept detection."
            )
        probes_list = sorted(list(probes), key=lambda probe: probe.probe_id)
        if not probes_list:
            raise ValueError("Concept detection requires a non-empty probe inventory.")
        self._embedding_provider = embedding_provider
        self._backend = backend or get_default_backend()
        self._probes = probes_list
        self._probe_embeddings = self._build_probe_embeddings()
        self._separation_floor = self._compute_separation_floor()

    def detect(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
    ) -> DetectionResult:
        """Detect concepts in a model response."""
        trimmed = response.strip()
        if not trimmed:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=response,
                detected_concepts=(),
            )

        segments = self._segment_text(trimmed)
        if not segments:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=response,
                detected_concepts=(),
            )

        detections: list[DetectedConcept] = []

        for start, end, segment in segments:
            embeddings = self._embedding_provider.embed([segment])
            if not embeddings:
                continue

            segment_embedding = VectorMath.l2_normalized(
                [float(value) for value in embeddings[0]]
            )

            best_probe: ProbeEmbedding | None = None
            best_similarity = -1.0

            for probe in self._probe_embeddings:
                similarity = VectorMath.dot(segment_embedding, probe.centroid) or 0.0
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_probe = probe

            if best_probe is None:
                continue

            acceptance_floor = max(best_probe.cohesion_floor, self._separation_floor)
            if best_similarity <= acceptance_floor:
                continue

            cross_modal_similarity = self._cross_modal_similarity(
                segment_embedding, best_probe.support_embeddings
            )

            detections.append(
                DetectedConcept(
                    concept_id=best_probe.probe_id,
                    category=best_probe.category,
                    similarity=float(best_similarity),
                    character_span=(start, end),
                    trigger_text=truncate(segment, 100),
                    cross_modal_similarity=cross_modal_similarity,
                )
            )

        collapsed = self._collapse_consecutive(detections)

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=tuple(collapsed),
        )

    @staticmethod
    def compare_results(
        result_a: DetectionResult,
        result_b: DetectionResult,
    ) -> ConceptComparisonResult:
        """Compare concept detection results between two models."""
        set_a = set(result_a.concept_sequence)
        set_b = set(result_b.concept_sequence)
        intersection = set_a.intersection(set_b)

        return ConceptComparisonResult(
            model_a=result_a.model_id,
            model_b=result_b.model_id,
            concept_path_a=tuple(result_a.concept_sequence),
            concept_path_b=tuple(result_b.concept_sequence),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=tuple(sorted(intersection)),
            unique_to_a=tuple(sorted(set_a - set_b)),
            unique_to_b=tuple(sorted(set_b - set_a)),
        )

    def _build_probe_embeddings(self) -> list[ProbeEmbedding]:
        probe_embeddings: list[ProbeEmbedding] = []
        for probe in self._probes:
            texts = [text.strip() for text in probe.support_texts if text.strip()]
            if not texts:
                logger.warning(
                    "Probe %s has no support_texts; skipping from concept detection.",
                    probe.probe_id,
                )
                continue

            embeddings = self._embedding_provider.embed(texts)
            if not embeddings:
                logger.warning(
                    "Embedding provider returned no embeddings for probe %s; skipping.",
                    probe.probe_id,
                )
                continue

            normalized_support = [
                VectorMath.l2_normalized([float(value) for value in embedding])
                for embedding in embeddings
            ]
            centroid = VectorMath.l2_normalized(self._frechet_centroid(embeddings))
            cohesion_floor = min(
                VectorMath.dot(centroid, support) or 0.0
                for support in normalized_support
            )

            probe_embeddings.append(
                ProbeEmbedding(
                    probe_id=probe.probe_id,
                    category=str(probe.category_name),
                    centroid=centroid,
                    support_embeddings=normalized_support,
                    cohesion_floor=float(cohesion_floor),
                )
            )

        if not probe_embeddings:
            raise ValueError("No probe embeddings available for concept detection.")
        return probe_embeddings

    def _frechet_centroid(self, embeddings: list[list[float]]) -> list[float]:
        points = self._backend.array(
            [[float(value) for value in vector] for vector in embeddings]
        )
        mean = frechet_mean(points, backend=self._backend)
        self._backend.eval(mean)
        return [float(value) for value in self._backend.to_numpy(mean).tolist()]

    def _compute_separation_floor(self) -> float:
        if len(self._probe_embeddings) < 2:
            return -1.0
        max_similarity = -1.0
        for index, probe_a in enumerate(self._probe_embeddings):
            for probe_b in self._probe_embeddings[index + 1 :]:
                similarity = VectorMath.dot(probe_a.centroid, probe_b.centroid) or 0.0
                if similarity > max_similarity:
                    max_similarity = similarity
        return float(max_similarity)

    @staticmethod
    def _segment_text(text: str) -> list[tuple[int, int, str]]:
        segments: list[tuple[int, int, str]] = []
        start = 0
        for idx, char in enumerate(text):
            if char in ".!?\n":
                end = idx + 1
                if end > start:
                    segment = text[start:end].strip()
                    if segment:
                        segments.append((start, end, segment))
                start = end
        if start < len(text):
            segment = text[start:].strip()
            if segment:
                segments.append((start, len(text), segment))
        return segments

    @staticmethod
    def _cross_modal_similarity(
        segment_embedding: list[float],
        support_embeddings: list[list[float]],
    ) -> float | None:
        if not support_embeddings:
            return None
        total = 0.0
        for support in support_embeddings:
            total += VectorMath.dot(segment_embedding, support) or 0.0
        return total / float(len(support_embeddings))

    @staticmethod
    def _collapse_consecutive(
        detections: list[DetectedConcept],
    ) -> list[DetectedConcept]:
        if len(detections) <= 1:
            return detections

        result: list[DetectedConcept] = []
        for detection in detections:
            if not result or result[-1].concept_id != detection.concept_id:
                result.append(detection)

        return result
