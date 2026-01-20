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

"""Concept detection using embedding geometry.

Detects semantic concept activations in text by comparing segment embeddings
to probe centroids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.embedding_cache import get_or_compute_embeddings_sync
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.domain.geometry.types import (
    ConceptComparisonResult,
    DetectedConcept,
    DetectionResult,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_cosine_batch,
    geodesic_cosine_between_sets,
    geodesic_cosine_matrix,
)
from modelcypher.utils.text import truncate

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeEmbedding:
    """Embeddings for a single probe, including support examples."""

    probe_id: str
    category: str
    centroid: Any
    support_matrix: Any
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
        self._probe_centroids = self._backend.stack(
            [probe.centroid for probe in self._probe_embeddings], axis=0
        )
        self._probe_count = self._backend.shape(self._probe_centroids)[0]
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
                mean_similarity=0.0,
                mean_cross_modal_similarity=None,
            )

        segments = self._segment_text(trimmed)
        if not segments:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=response,
                detected_concepts=(),
                mean_similarity=0.0,
                mean_cross_modal_similarity=None,
            )

        detections: list[DetectedConcept] = []

        segment_texts = [segment for _, _, segment in segments]
        embeddings_batch = self._embedding_provider.embed(segment_texts)
        use_batch = False
        segment_matrix = None
        best_idx_list: list[int] = []
        best_sim_list: list[float] = []

        if embeddings_batch is not None:
            if hasattr(embeddings_batch, "shape"):
                segment_matrix = embeddings_batch
                if int(self._backend.shape(segment_matrix)[0]) != len(segments):
                    segment_matrix = None
            else:
                if embeddings_batch:
                    segment_matrix = self._backend.array(embeddings_batch)
                else:
                    segment_matrix = None
            if segment_matrix is not None:
                sim_matrix = geodesic_cosine_between_sets(
                    segment_matrix, self._probe_centroids, self._backend
                )
                if int(self._backend.shape(sim_matrix)[0]) == len(segments):
                    best_idx_arr = self._backend.argmax(sim_matrix, axis=1)
                    best_idx_col = self._backend.reshape(best_idx_arr, (-1, 1))
                    best_sim_arr = self._backend.take_along_axis(
                        sim_matrix, best_idx_col, axis=1
                    )
                    self._backend.eval(best_idx_arr, best_sim_arr)
                    best_idx_list = [int(x) for x in self._backend.tolist(best_idx_arr)]
                    best_sim_list = [
                        float(x[0]) if isinstance(x, list) else float(x)
                        for x in self._backend.tolist(best_sim_arr)
                    ]
                    use_batch = True

        for idx, (start, end, segment) in enumerate(segments):
            if use_batch and segment_matrix is not None:
                best_idx = best_idx_list[idx]
                best_similarity = best_sim_list[idx]
                segment_embedding = segment_matrix[idx]
            else:
                embeddings = self._embedding_provider.embed([segment])
                if not embeddings:
                    continue
                segment_embedding = self._ensure_array(embeddings[0])
                sims = geodesic_cosine_batch(
                    segment_embedding, self._probe_centroids, self._backend
                )
                best_idx_arr = self._backend.argmax(sims)
                self._backend.eval(best_idx_arr)
                best_idx = int(self._backend.to_scalar(best_idx_arr))
                best_sim_arr = self._backend.take(sims, self._backend.array([best_idx]))
                self._backend.eval(best_sim_arr)
                best_similarity = float(self._backend.to_scalar(best_sim_arr))

            best_probe = self._probe_embeddings[best_idx]

            acceptance_floor = max(best_probe.cohesion_floor, self._separation_floor)
            if best_similarity <= acceptance_floor:
                continue

            cross_modal_similarity = self._cross_modal_similarity(
                segment_embedding, best_probe.support_matrix
            )

            detections.append(
                DetectedConcept(
                    concept_id=best_probe.probe_id,
                    category=best_probe.category,
                    similarity=float(best_similarity),
                    character_span=slice(start, end),
                    trigger_text=truncate(segment, 100),
                    cross_modal_similarity=cross_modal_similarity,
                )
            )

        collapsed = self._collapse_consecutive(detections)

        # Compute mean_similarity
        mean_similarity = 0.0
        if collapsed:
            scores = self._backend.array([c.similarity for c in collapsed])
            mean_score = self._backend.mean(scores)
            self._backend.eval(mean_score)
            mean_similarity = float(self._backend.to_scalar(mean_score))

        # Compute mean_cross_modal_similarity
        mean_cross_modal: float | None = None
        with_cross_modal = [
            c.cross_modal_similarity
            for c in collapsed
            if c.cross_modal_similarity is not None
        ]
        if with_cross_modal:
            cross_scores = self._backend.array(with_cross_modal)
            mean_cross_score = self._backend.mean(cross_scores)
            self._backend.eval(mean_cross_score)
            mean_cross_modal = float(self._backend.to_scalar(mean_cross_score))

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=tuple(collapsed),
            mean_similarity=mean_similarity,
            mean_cross_modal_similarity=mean_cross_modal,
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

            embeddings_arr = get_or_compute_embeddings_sync(
                self._embedding_provider,
                self._backend,
                "concept_detector_support",
                texts,
            )
            if int(self._backend.shape(embeddings_arr)[0]) == 0:
                logger.warning(
                    "Embedding provider returned no embeddings for probe %s; skipping.",
                    probe.probe_id,
                )
                continue

            support_matrix = embeddings_arr
            centroid = self._frechet_centroid(embeddings_arr)
            cohesion_floor = self._cohesion_floor(centroid, support_matrix)

            probe_embeddings.append(
                ProbeEmbedding(
                    probe_id=probe.probe_id,
                    category=str(probe.category_name),
                    centroid=centroid,
                    support_matrix=support_matrix,
                    cohesion_floor=float(cohesion_floor),
                )
            )

        if not probe_embeddings:
            raise ValueError("No probe embeddings available for concept detection.")
        return probe_embeddings

    def _frechet_centroid(self, embeddings: list[list[float]]) -> Any:
        points = self._ensure_array(embeddings)
        mean = frechet_mean(points, backend=self._backend)
        self._backend.eval(mean)
        return mean

    def _compute_separation_floor(self) -> float:
        if self._probe_count < 2:
            return -1.0
        cos_matrix = geodesic_cosine_matrix(self._probe_centroids, self._backend)

        diag_mask = self._backend.eye(self._probe_count)
        neg_inf = self._backend.array(float("-inf"), dtype=cos_matrix.dtype)
        masked = self._backend.where(diag_mask > 0, neg_inf, cos_matrix)
        max_off_diag = self._backend.max(masked)
        self._backend.eval(max_off_diag)
        return float(self._backend.to_scalar(max_off_diag))

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

    def _cross_modal_similarity(
        self,
        segment_embedding: Any,
        support_matrix: Any,
    ) -> float | None:
        if support_matrix is None:
            return None
        support_count = self._backend.shape(support_matrix)[0]
        if support_count == 0:
            return None
        sims = geodesic_cosine_batch(segment_embedding, support_matrix, self._backend)
        mean_sim = self._backend.mean(sims)
        self._backend.eval(mean_sim)
        return float(self._backend.to_scalar(mean_sim))

    def _ensure_array(self, value: Any) -> Any:
        if hasattr(value, "shape"):
            return value
        return self._backend.array(value)

    def _cohesion_floor(self, centroid: Any, support_matrix: Any) -> float:
        if support_matrix is None:
            return 0.0
        support_count = self._backend.shape(support_matrix)[0]
        if support_count == 0:
            return 0.0
        sims = geodesic_cosine_batch(centroid, support_matrix, self._backend)
        min_sim = self._backend.min(sims)
        self._backend.eval(min_sim)
        return float(self._backend.to_scalar(min_sim))

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
