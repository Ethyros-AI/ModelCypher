from __future__ import annotations

from typing import List

from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGate, ComputationalGateCategory
from modelcypher.core.domain.gate_detector import Configuration, GateDetector
from modelcypher.ports.embedding import EmbeddingProvider


class _KeywordEmbedder(EmbeddingProvider):
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
            else:
                embeddings.append([0.0, 0.0])
        return embeddings


def test_gate_detector_detects_gates() -> None:
    gates = [
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
    ]

    detector = GateDetector(
        configuration=Configuration(detection_threshold=0.5, window_sizes=[1], stride=1),
        embedder=_KeywordEmbedder(),
        gate_inventory=gates,
    )
    text = "alpha alpha beta alpha"
    result = detector.detect(text=text, model_id="model-1", prompt_id="prompt-1")

    assert result.detected_gates
    assert "alpha" in result.gate_sequence
    assert "beta" in result.gate_sequence
    assert result.mean_confidence > 0
