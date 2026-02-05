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
Computational gate detection in model responses.

Computational Gates:
    A "gate" is a semantic checkpoint where models transition between
    computational modes. Examples include:
    - Logical gates: "therefore", "if...then", "because"
    - Planning gates: "first", "next", "finally"
    - Uncertainty gates: "however", "although", "unless"
    - Verification gates: "let me check", "to confirm"

The gate detector identifies these transitions in model output by:
    1. Embedding response segments
    2. Computing similarity to known gate embeddings from ComputationalGateAtlas
    3. Deriving a threshold from the similarity distribution (no user inputs)
    4. Collapsing consecutive detections

Use Cases:
    - Reasoning chain analysis: Track how models structure arguments
    - Safety monitoring: Detect mode switches that may indicate jailbreaking
    - Training diagnostics: Measure gate frequency/diversity as capability proxy
    - Cross-model comparison: Architecture-invariant reasoning patterns

The 72 computational gates cover mathematical, logical, linguistic,
and structural domains for comprehensive coverage.

See also: modelcypher.core.domain.agents.computational_gate_atlas
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import TwoLevelCache, content_hash
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    frechet_mean,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.atlas_protocols import ComputationalGateProtocol
from modelcypher.core.domain.geometry.atlas_registry import get_gate_inventory
from modelcypher.core.domain.geometry.path_geometry import PathNode, PathSignature
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.utils.paths import get_modelcypher_home
from modelcypher.utils.text import truncate

logger = logging.getLogger(__name__)

_GATE_CACHE: TwoLevelCache[dict] | None = None
_GATE_CACHE_LOCK = threading.Lock()


def _get_gate_cache() -> TwoLevelCache[dict]:
    global _GATE_CACHE
    if _GATE_CACHE is None:
        with _GATE_CACHE_LOCK:
            if _GATE_CACHE is None:
                cache_dir = get_modelcypher_home() / "cache" / "gate_embeddings"
                _GATE_CACHE = TwoLevelCache(
                    cache_directory=cache_dir,
                    serializer=lambda payload: payload,
                    deserializer=lambda payload: payload,
                    memory_limit=4,
                    disk_ttl_seconds=30 * 24 * 60 * 60,
                    cache_version=1,
                )
    return _GATE_CACHE


@dataclass(frozen=True)
class DetectedGate:
    gate_id: str
    gate_name: str
    similarity: float
    character_span: tuple[int, int]
    trigger_text: str
    local_entropy: float | None = None


@dataclass(frozen=True)
class GateDetectionResult:
    model_id: str
    prompt_id: str
    response_text: str
    detected_gates: list[DetectedGate]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def mean_similarity(self, backend: "Backend") -> float:
        if not self.detected_gates:
            return 0.0
        scores = backend.array([gate.similarity for gate in self.detected_gates])
        mean_score = backend.mean(scores)
        backend.eval(mean_score)
        return float(backend.to_scalar(mean_score))

    @property
    def gate_sequence(self) -> list[str]:
        return [gate.gate_id for gate in self.detected_gates]

    @property
    def gate_name_sequence(self) -> list[str]:
        return [gate.gate_name for gate in self.detected_gates]

    def to_path_signature(
        self, gate_embeddings: dict[str, list[float]] | None = None
    ) -> PathSignature:
        nodes = [
            PathNode(
                gate_id=gate.gate_id,
                token_index=gate.character_span[0],
                entropy=gate.local_entropy or 0.0,
                embedding=gate_embeddings.get(gate.gate_id) if gate_embeddings else None,
            )
            for gate in self.detected_gates
        ]
        return PathSignature(model_id=self.model_id, prompt_id=self.prompt_id, nodes=nodes)


class GateDetector:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        backend: Backend,
        gate_inventory: Iterable[ComputationalGateProtocol] | None = None,
    ) -> None:
        self.embedder = embedder
        self._backend = backend
        self.gate_embeddings: dict[str, list[float]] = {}
        self.gate_metadata: dict[str, ComputationalGateProtocol] = {}
        self._gate_ids: list[str] = []
        self._gate_matrix = None
        self._gate_points = None
        self._gate_geo_result = None
        self._gate_origin_distances = None
        self._gate_cache_key = None

        if gate_inventory is None:
            gate_inventory = get_gate_inventory()

        for gate in gate_inventory:
            self.gate_metadata[gate.id] = gate

    def detect(
        self,
        text: str,
        model_id: str,
        prompt_id: str,
        entropy_trace: list[float] | None = None,
    ) -> DetectionResult:
        if not text:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=text,
                detected_gates=[],
            )

        self._ensure_gate_embeddings()
        if not self.gate_embeddings:
            raise ValueError("No gate embeddings available for detection")
        if self._gate_matrix is None or not self._gate_ids:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=text,
                detected_gates=[],
            )

        segments = self._segment_text(text)
        if not segments:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=text,
                detected_gates=[],
            )

        candidates: list[tuple[DetectedGate, int, float]] = []
        best_similarities: list[float] = []

        segment_texts = [segment[2] for segment in segments]
        embeddings_batch = self.embedder.embed(segment_texts)
        if len(embeddings_batch) != len(segments):
            embeddings_batch = []

        for index, (window_start, window_end, window_text) in enumerate(segments):
            if embeddings_batch:
                window_embedding = embeddings_batch[index]
            else:
                embeddings = self.embedder.embed([window_text])
                if not embeddings:
                    continue
                window_embedding = embeddings[0]

            window_vec = self._normalize_vector(window_embedding)
            sims = self._geodesic_cosine_to_gates(window_vec)
            best_idx_arr = self._backend.argmax(sims)
            self._backend.eval(best_idx_arr)
            best_idx = int(self._backend.to_scalar(best_idx_arr))
            best_sim_arr = self._backend.take(sims, self._backend.array([best_idx]))
            self._backend.eval(best_sim_arr)
            best_similarity = float(self._backend.to_scalar(best_sim_arr))
            best_gate_id = self._gate_ids[best_idx]

            eps = division_epsilon(self._backend, sims)
            if best_similarity <= eps:
                continue

            best_similarities.append(best_similarity)
            gate_meta = self.gate_metadata.get(best_gate_id)
            if gate_meta is None:
                continue

            local_entropy = None
            if entropy_trace and window_start < len(entropy_trace):
                window_entropy = entropy_trace[
                    window_start : min(window_end, len(entropy_trace))
                ]
                if window_entropy:
                    entropy_arr = self._backend.array(window_entropy)
                    mean_entropy = self._backend.mean(entropy_arr)
                    self._backend.eval(mean_entropy)
                    local_entropy = float(self._backend.to_scalar(mean_entropy))

            candidates.append(
                (
                    DetectedGate(
                        gate_id=best_gate_id,
                        gate_name=gate_meta.name,
                        similarity=float(best_similarity),
                        character_span=(window_start, window_end),
                        trigger_text=truncate(window_text, 50),
                        local_entropy=local_entropy,
                    ),
                    index,
                    best_similarity,
                )
            )

        if not best_similarities:
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=text,
                detected_gates=[],
            )

        # When all similarities are identical AND zero, nothing matches
        # Uses boundary value (0) instead of arbitrary midpoint threshold
        if len(best_similarities) > 1 and max(best_similarities) == min(best_similarities):
            if max(best_similarities) <= 0:
                return DetectionResult(
                    model_id=model_id,
                    prompt_id=prompt_id,
                    response_text=text,
                    detected_gates=[],
                )

        if len(best_similarities) == 1:
            detections = candidates
        elif max(best_similarities) == min(best_similarities):
            # All similarities are equal and positive - include all equally valid candidates
            detections = candidates
        else:
            threshold = self._otsu_threshold(best_similarities, self._backend)
            detections = [c for c in candidates if c[2] >= threshold]
        detections.sort(key=lambda item: item[1])
        merged = self._collapse_consecutive([item[0] for item in detections])

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=text,
            detected_gates=merged,
        )

    def get_gate_embeddings(self) -> dict[str, list[float]]:
        self._ensure_gate_embeddings()
        return dict(self.gate_embeddings)

    def _gate_cache_key_value(self) -> str:
        if self._gate_cache_key is not None:
            return self._gate_cache_key

        embedder_sig: dict[str, object] = {
            "class": type(self.embedder).__name__,
            "dimension": getattr(self.embedder, "dimension", None),
        }
        base_url = getattr(self.embedder, "base_url", None)
        if base_url:
            embedder_sig["base_url"] = base_url

        gate_payload: list[dict[str, object]] = []
        for gate_id in sorted(self.gate_metadata):
            gate = self.gate_metadata[gate_id]
            gate_payload.append(
                {
                    "id": gate.id,
                    "name": gate.name,
                    "description": gate.description,
                    "examples": list(gate.examples),
                    "polyglot_examples": list(gate.polyglot_examples),
                }
            )

        self._gate_cache_key = content_hash(
            {"embedder": embedder_sig, "gates": gate_payload}
        )
        return self._gate_cache_key

    def _hydrate_gate_embeddings(
        self, gate_ids: list[str], gate_matrix: list[list[float]]
    ) -> None:
        if not gate_ids or not gate_matrix:
            return
        if len(gate_ids) != len(gate_matrix):
            return
        self._gate_ids = list(gate_ids)
        self.gate_embeddings = {
            gate_id: vector for gate_id, vector in zip(gate_ids, gate_matrix)
        }
        self._gate_matrix = self._backend.array(gate_matrix)
        self._backend.eval(self._gate_matrix)
        self._prepare_gate_geometry()

    def _ensure_gate_embeddings(self) -> None:
        if self.gate_embeddings:
            return

        cache_key = self._gate_cache_key_value()
        cached = _get_gate_cache().get(cache_key)
        if cached:
            cached_ids = cached.get("gate_ids", [])
            cached_matrix = cached.get("gate_matrix", [])
            self._hydrate_gate_embeddings(cached_ids, cached_matrix)
            if self.gate_embeddings:
                logger.info(
                    "Loaded %s gate embeddings from cache", len(self.gate_embeddings)
                )
                return

        vectors = []
        ids: list[str] = []
        for gate in self.gate_metadata.values():
            texts = [f"{gate.name}: {gate.description}"]
            texts.extend(gate.examples)
            texts.extend(gate.polyglot_examples)
            texts = [text for text in texts if text.strip()]
            if not texts:
                continue

            embeddings = self.embedder.embed(texts)
            if not embeddings:
                continue

            embedding_arr = self._backend.array(embeddings)
            centroid = frechet_mean(embedding_arr, backend=self._backend)
            centroid = self._normalize_vector(centroid)
            self._backend.eval(centroid)
            centroid_list = self._backend.tolist(centroid)
            if not isinstance(centroid_list, list):
                centroid_list = [float(centroid_list)]
            self.gate_embeddings[gate.id] = centroid_list
            vectors.append(centroid)
            ids.append(gate.id)

        if vectors:
            gate_matrix = [self.gate_embeddings[gate_id] for gate_id in ids]
            self._hydrate_gate_embeddings(ids, gate_matrix)
            _get_gate_cache().set(
                cache_key, {"gate_ids": ids, "gate_matrix": gate_matrix}
            )

        logger.info("Loaded %s gate embeddings", len(self.gate_embeddings))

    def _prepare_gate_geometry(self) -> None:
        if self._gate_matrix is None:
            return
        zero = self._backend.zeros_like(self._gate_matrix[:1])
        self._gate_points = self._backend.concatenate([zero, self._gate_matrix], axis=0)
        rg = RiemannianGeometry(self._backend)
        point_count = int(self._backend.shape(self._gate_points)[0])
        self._gate_geo_result = rg.geodesic_distances(
            self._gate_points, k_neighbors=point_count - 1
        )
        distances = self._gate_geo_result.distances
        self._backend.eval(distances)
        self._gate_origin_distances = distances[0, 1:]

    def _geodesic_cosine_to_gates(self, vector: object) -> object:
        if self._gate_geo_result is None or self._gate_origin_distances is None:
            self._prepare_gate_geometry()
        if (
            self._gate_geo_result is None
            or self._gate_origin_distances is None
            or self._gate_points is None
        ):
            return self._backend.array([])

        rg = RiemannianGeometry(self._backend)
        geo_from_query = rg._geodesic_distances_from_query(
            self._gate_points, vector, geo_result=self._gate_geo_result
        )
        d0a = geo_from_query[0]
        dav = geo_from_query[1:]
        d0v = self._gate_origin_distances

        eps = division_epsilon(self._backend, geo_from_query)
        denom = 2.0 * d0a * d0v
        safe_denom = self._backend.maximum(
            denom, self._backend.full(d0v.shape, eps)
        )
        cos_vals = (d0a * d0a + d0v * d0v - dav * dav) / safe_denom
        cos_vals = self._backend.clip(cos_vals, -1.0, 1.0)
        valid = self._backend.minimum(d0v > eps, d0a > eps)
        cos_vals = self._backend.where(
            valid, cos_vals, self._backend.zeros_like(cos_vals)
        )
        return cos_vals

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
    def _otsu_threshold(values: list[float], backend: Backend) -> float:
        if not values:
            raise ValueError("Cannot derive threshold from empty similarity values")
        if len(values) == 1:
            return values[0]

        vals = backend.array(values)
        sorted_vals = backend.sort(vals)
        n = backend.shape(sorted_vals)[0]
        if n < 2:
            return float(backend.to_scalar(sorted_vals))

        idx = backend.arange(1, n)
        cumsum = backend.cumsum(sorted_vals)
        sum0 = backend.take(cumsum, idx - 1)
        total = backend.sum(sorted_vals)

        idx_float = backend.astype(idx, sorted_vals.dtype)
        denom1 = float(n) - idx_float
        mean0 = sum0 / idx_float
        sum1 = total - sum0
        mean1 = sum1 / denom1

        w0 = idx_float / float(n)
        w1 = 1.0 - w0
        variance = w0 * w1 * (mean0 - mean1) ** 2

        best_idx_arr = backend.argmax(variance)
        backend.eval(best_idx_arr)
        best_idx = int(backend.to_scalar(best_idx_arr))
        threshold_index = backend.take(idx, backend.array([best_idx]))
        threshold_val = backend.take(sorted_vals, threshold_index)
        backend.eval(threshold_val)
        return float(backend.to_scalar(threshold_val))

    @staticmethod
    def _collapse_consecutive(gates: list[DetectedGate]) -> list[DetectedGate]:
        if not gates:
            return []
        collapsed = [gates[0]]
        for gate in gates[1:]:
            if gate.gate_id != collapsed[-1].gate_id:
                collapsed.append(gate)
            elif gate.similarity > collapsed[-1].similarity:
                collapsed[-1] = gate
        return collapsed

    def _normalize_vector(self, vector: Iterable[float] | object) -> object:
        vec = vector if hasattr(vector, "shape") else self._backend.array(vector)
        if len(self._backend.shape(vec)) != 1:
            vec = self._backend.reshape(vec, (-1,))
        vec_row = self._backend.reshape(vec, (1, -1))
        norm_arr = geodesic_norms(vec_row, self._backend)
        eps = division_epsilon(self._backend, vec)
        norm_safe = self._backend.maximum(norm_arr, self._backend.array(eps))
        return vec / norm_safe
