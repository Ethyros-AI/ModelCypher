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
    1. Embedding response text in sliding windows
    2. Computing similarity to known gate embeddings from ComputationalGateAtlas
    3. Thresholding and collapsing consecutive detections

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

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Iterable, Optional

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.core.domain.geometry.path_geometry import PathNode, PathSignature
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.utils.text import truncate

if TYPE_CHECKING:
    from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGate

from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    detection_threshold: float = 0.55
    window_sizes: list[int] = None  # type: ignore[assignment]
    stride: int = 3
    collapse_consecutive: bool = True
    max_gates_per_response: int = 50

    def __post_init__(self) -> None:
        if self.window_sizes is None:
            object.__setattr__(self, "window_sizes", [5, 10, 15])


@dataclass(frozen=True)
class DetectedGate:
    gate_id: str
    gate_name: str
    confidence: float
    character_span: tuple[int, int]
    trigger_text: str
    local_entropy: Optional[float] = None


@dataclass(frozen=True)
class DetectionResult:
    model_id: str
    prompt_id: str
    response_text: str
    detected_gates: list[DetectedGate]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean_confidence(self) -> float:
        if not self.detected_gates:
            return 0.0
        total = sum(gate.confidence for gate in self.detected_gates)
        return total / float(len(self.detected_gates))

    @property
    def gate_sequence(self) -> list[str]:
        return [gate.gate_id for gate in self.detected_gates]

    @property
    def gate_name_sequence(self) -> list[str]:
        return [gate.gate_name for gate in self.detected_gates]

    def to_path_signature(self, gate_embeddings: dict[str, list[float]] | None = None) -> PathSignature:
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
        configuration: Configuration | None = None,
        embedder: EmbeddingProvider | None = None,
        gate_inventory: Iterable[ComputationalGate] | None = None,
    ) -> None:
        self.config = configuration or Configuration()
        self.embedder = embedder or EmbeddingDefaults.make_default_embedder()
        self.gate_embeddings: dict[str, list[float]] = {}
        self.gate_metadata: dict[str, ComputationalGate] = {}
        
        # Lazy import to avoid circular dependency with agents package
        if gate_inventory is None:
            from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
            gate_inventory = ComputationalGateInventory.all_gates()
        
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
            logger.warning("No gate embeddings available")
            return DetectionResult(
                model_id=model_id,
                prompt_id=prompt_id,
                response_text=text,
                detected_gates=[],
            )

        all_detections: list[tuple[DetectedGate, int]] = []
        for window_size in self.config.window_sizes:
            detections = self._detect_with_window(text, window_size, entropy_trace)
            all_detections.extend(detections)

        all_detections.sort(key=lambda item: item[1])
        merged = self._merge_overlapping([item[0] for item in all_detections])

        if self.config.collapse_consecutive:
            merged = self._collapse_consecutive(merged)

        if len(merged) > self.config.max_gates_per_response:
            merged = merged[: self.config.max_gates_per_response]

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
        if self.embedder is None:
            logger.warning("No embedder available for gate detection")
            return

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

            sum_vector = [0.0] * len(embeddings[0])
            for vector in embeddings:
                for i in range(min(len(vector), len(sum_vector))):
                    sum_vector[i] += float(vector[i])

            centroid = VectorMath.l2_normalized(sum_vector)
            self.gate_embeddings[gate.id] = centroid

        logger.info("Loaded %s gate embeddings", len(self.gate_embeddings))

    def _detect_with_window(
        self,
        text: str,
        window_size: int,
        entropy_trace: list[float] | None,
    ) -> list[tuple[DetectedGate, int]]:
        if self.embedder is None:
            return []

        detections: list[tuple[DetectedGate, int]] = []
        chars = list(text)
        char_count = len(chars)
        char_window_size = window_size * 4
        char_stride = self.config.stride * 4

        position = 0
        while position + char_window_size <= char_count:
            window_start = position
            window_end = min(position + char_window_size, char_count)
            window_text = "".join(chars[window_start:window_end])

            embeddings = self.embedder.embed([window_text])
            if not embeddings:
                position += char_stride
                continue
            window_embedding = embeddings[0]
            normalized_window = VectorMath.l2_normalized([float(value) for value in window_embedding])

            best_gate_id = None
            best_similarity = 0.0
            for gate_id, gate_embedding in self.gate_embeddings.items():
                similarity = VectorMath.dot(normalized_window, gate_embedding) or 0.0
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gate_id = gate_id

            if best_gate_id and best_similarity >= self.config.detection_threshold:
                gate_meta = self.gate_metadata.get(best_gate_id)
                if gate_meta is None:
                    position += char_stride
                    continue
                local_entropy = None
                if entropy_trace and window_start < len(entropy_trace):
                    window_entropy = entropy_trace[window_start:min(window_end, len(entropy_trace))]
                    if window_entropy:
                        local_entropy = sum(window_entropy) / float(len(window_entropy))
                detections.append(
                    (
                        DetectedGate(
                            gate_id=best_gate_id,
                            gate_name=gate_meta.name,
                            confidence=float(best_similarity),
                            character_span=(window_start, window_end),
                            trigger_text=truncate(window_text, 50),
                            local_entropy=local_entropy,
                        ),
                        window_start,
                    )
                )

            position += char_stride

        return detections

    @staticmethod
    def _merge_overlapping(gates: list[DetectedGate]) -> list[DetectedGate]:
        if not gates:
            return []

        merged: list[DetectedGate] = []
        current = gates[0]
        for gate in gates[1:]:
            if gate.character_span[0] < current.character_span[1]:
                if gate.confidence > current.confidence:
                    current = gate
            else:
                merged.append(current)
                current = gate
        merged.append(current)
        return merged

    @staticmethod
    def _collapse_consecutive(gates: list[DetectedGate]) -> list[DetectedGate]:
        if not gates:
            return []
        collapsed = [gates[0]]
        for gate in gates[1:]:
            if gate.gate_id != collapsed[-1].gate_id:
                collapsed.append(gate)
            elif gate.confidence > collapsed[-1].confidence:
                collapsed[-1] = gate
        return collapsed
