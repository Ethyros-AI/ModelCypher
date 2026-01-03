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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.atlas_protocols import ComputationalGateProtocol
from modelcypher.core.domain.geometry.atlas_registry import get_gate_inventory
from modelcypher.core.domain.geometry.path_geometry import PathNode, PathSignature
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.utils.text import truncate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedGate:
    gate_id: str
    gate_name: str
    similarity: float
    character_span: tuple[int, int]
    trigger_text: str
    local_entropy: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    model_id: str
    prompt_id: str
    response_text: str
    detected_gates: list[DetectedGate]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean_similarity(self) -> float:
        if not self.detected_gates:
            return 0.0
        backend = get_default_backend()
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
        gate_inventory: Iterable[ComputationalGateProtocol] | None = None,
    ) -> None:
        self.embedder = embedder
        self._backend = get_default_backend()
        self.gate_embeddings: dict[str, list[float]] = {}
        self.gate_metadata: dict[str, ComputationalGateProtocol] = {}
        self._gate_ids: list[str] = []
        self._gate_matrix = None

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

        for index, (window_start, window_end, window_text) in enumerate(segments):
            embeddings = self.embedder.embed([window_text])
            if not embeddings:
                continue
            if self._gate_matrix is None or not self._gate_ids:
                continue
            normalized_window = self._normalize_vector(embeddings[0])
            sims = self._backend.matmul(
                self._gate_matrix,
                self._backend.reshape(normalized_window, (-1, 1)),
            )
            sims = self._backend.reshape(sims, (self._backend.shape(self._gate_matrix)[0],))
            best_idx_arr = self._backend.argmax(sims)
            self._backend.eval(best_idx_arr)
            best_idx = int(self._backend.to_scalar(best_idx_arr))
            best_sim_arr = self._backend.take(sims, self._backend.array([best_idx]))
            self._backend.eval(best_sim_arr)
            best_similarity = float(self._backend.to_scalar(best_sim_arr))
            best_gate_id = self._gate_ids[best_idx]

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
            threshold = self._otsu_threshold(best_similarities)
            detections = [c for c in candidates if c[2] > threshold]
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

    def _ensure_gate_embeddings(self) -> None:
        if self.gate_embeddings:
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
            centroid = self._backend.mean(embedding_arr, axis=0)
            centroid = self._normalize_vector(centroid)
            self._backend.eval(centroid)
            centroid_list = self._backend.tolist(centroid)
            if not isinstance(centroid_list, list):
                centroid_list = [float(centroid_list)]
            self.gate_embeddings[gate.id] = centroid_list
            vectors.append(centroid)
            ids.append(gate.id)

        if vectors:
            self._gate_matrix = self._backend.stack(vectors, axis=0)
            self._gate_ids = ids

        logger.info("Loaded %s gate embeddings", len(self.gate_embeddings))

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
    def _otsu_threshold(values: list[float]) -> float:
        if not values:
            raise ValueError("Cannot derive threshold from empty similarity values")
        if len(values) == 1:
            return values[0]

        backend = get_default_backend()
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
        norm = self._backend.norm(vec)
        eps = division_epsilon(self._backend, vec)
        safe_norm = self._backend.where(
            norm > eps, norm, self._backend.ones_like(norm)
        )
        return vec / safe_norm
