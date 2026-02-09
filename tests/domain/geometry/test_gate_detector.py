# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from modelcypher.core.domain.geometry.gate_detector import (
    DetectedGate,
    DetectionResult,
    GateDetector,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms


@dataclass
class _Gate:
    id: str
    name: str
    description: str
    examples: list[str]
    polyglot_examples: list[str]


class _Embedder:
    def __init__(self, vectors: dict[str, list[float]], *, batch_mismatch: bool = False) -> None:
        self.vectors = vectors
        self.batch_mismatch = batch_mismatch
        self.calls: list[list[str]] = []
        self.dimension = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.batch_mismatch and len(texts) > 1:
            return [self.vectors.get(texts[0], [0.0, 0.0])]
        return [self.vectors.get(text, [0.0, 0.0]) for text in texts]


def _make_detector(any_backend, embedder: _Embedder) -> GateDetector:
    return GateDetector(
        embedder=embedder,
        backend=any_backend,
        gate_inventory=[
            _Gate(
                id="g1",
                name="Gate One",
                description="First gate",
                examples=["one"],
                polyglot_examples=[],
            ),
            _Gate(
                id="g2",
                name="Gate Two",
                description="Second gate",
                examples=["two"],
                polyglot_examples=[],
            ),
        ],
    )


def test_detection_result_properties_and_signature() -> None:
    result = DetectionResult(
        model_id="m",
        prompt_id="p",
        response_text="text",
        detected_gates=[
            DetectedGate("g1", "Gate One", 0.8, (0, 4), "gate", local_entropy=0.1),
            DetectedGate("g2", "Gate Two", 0.6, (5, 9), "switch", local_entropy=None),
        ],
    )

    assert result.mean_similarity == pytest.approx(0.7)
    assert result.gate_sequence == ["g1", "g2"]
    assert result.gate_name_sequence == ["Gate One", "Gate Two"]

    signature = result.to_path_signature(gate_embeddings={"g1": [1.0, 0.0]})
    assert signature.model_id == "m"
    assert signature.prompt_id == "p"
    assert len(signature.nodes) == 2
    assert signature.nodes[0].embedding == [1.0, 0.0]
    assert signature.nodes[1].embedding is None


def test_segment_otsu_and_collapse_helpers(any_backend) -> None:
    b = any_backend

    assert GateDetector._segment_text("One. Two!\nThree?") == [
        (0, 4, "One."),
        (4, 9, "Two!"),
        (10, 16, "Three?"),
    ]

    with pytest.raises(ValueError):
        GateDetector._otsu_threshold([], b)
    assert GateDetector._otsu_threshold([0.42], b) == pytest.approx(0.42)

    collapsed = GateDetector._collapse_consecutive(
        [
            DetectedGate("g1", "Gate One", 0.5, (0, 2), "a"),
            DetectedGate("g1", "Gate One", 0.8, (2, 4), "b"),
            DetectedGate("g2", "Gate Two", 0.7, (4, 6), "c"),
        ]
    )
    assert len(collapsed) == 2
    assert collapsed[0].similarity == pytest.approx(0.8)
    assert collapsed[1].gate_id == "g2"


def test_normalize_vector_and_geodesic_fallback(any_backend, monkeypatch) -> None:
    b = any_backend
    detector = _make_detector(
        b,
        _Embedder({"alpha.": [3.0, 4.0]}),
    )

    vec = detector._normalize_vector([3.0, 4.0])
    norm = geodesic_norms(b.reshape(vec, (1, -1)), b)
    b.eval(norm)
    assert float(b.to_scalar(norm)) == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    detector._gate_points = None
    detector._gate_geo_result = None
    detector._gate_origin_distances = None
    monkeypatch.setattr(detector, "_prepare_gate_geometry", lambda: None)
    assert b.tolist(detector._geodesic_cosine_to_gates(vec)) == []


def test_detect_collapses_equal_positive_candidates(any_backend, monkeypatch) -> None:
    b = any_backend
    embedder = _Embedder({"Alpha.": [1.0, 0.0], "Beta.": [1.0, 0.0]})
    detector = _make_detector(b, embedder)

    detector.gate_embeddings = {"g1": [1.0, 0.0], "g2": [0.0, 1.0]}
    detector._gate_ids = ["g1", "g2"]
    detector._gate_matrix = b.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(detector, "_ensure_gate_embeddings", lambda: None)

    scores = [b.array([0.9, 0.1]), b.array([0.9, 0.1])]
    monkeypatch.setattr(detector, "_geodesic_cosine_to_gates", lambda _vec: scores.pop(0))

    result = detector.detect("Alpha. Beta.", model_id="m", prompt_id="p")

    assert len(result.detected_gates) == 1
    assert result.detected_gates[0].gate_id == "g1"
    assert result.detected_gates[0].trigger_text == "Alpha."


def test_detect_returns_empty_for_identical_zero_scores(any_backend, monkeypatch) -> None:
    b = any_backend
    embedder = _Embedder({"Alpha.": [1.0, 0.0], "Beta.": [1.0, 0.0]})
    detector = _make_detector(b, embedder)

    detector.gate_embeddings = {"g1": [1.0, 0.0], "g2": [0.0, 1.0]}
    detector._gate_ids = ["g1", "g2"]
    detector._gate_matrix = b.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(detector, "_ensure_gate_embeddings", lambda: None)
    monkeypatch.setattr(detector, "_geodesic_cosine_to_gates", lambda _vec: b.array([0.0, 0.0]))

    result = detector.detect("Alpha. Beta.", model_id="m", prompt_id="p")

    assert result.detected_gates == []


def test_detect_batch_mismatch_falls_back_to_per_segment_embedding(any_backend, monkeypatch) -> None:
    b = any_backend
    embedder = _Embedder(
        {
            "Alpha.": [1.0, 0.0],
            "Beta.": [1.0, 0.0],
        },
        batch_mismatch=True,
    )
    detector = _make_detector(b, embedder)

    detector.gate_embeddings = {"g1": [1.0, 0.0], "g2": [0.0, 1.0]}
    detector._gate_ids = ["g1", "g2"]
    detector._gate_matrix = b.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(detector, "_ensure_gate_embeddings", lambda: None)

    scores = [b.array([0.2, 0.8]), b.array([0.1, 0.95])]
    monkeypatch.setattr(detector, "_geodesic_cosine_to_gates", lambda _vec: scores.pop(0))

    entropy_trace = [float(i) for i in range(20)]
    result = detector.detect(
        "Alpha. Beta.",
        model_id="model-x",
        prompt_id="prompt-y",
        entropy_trace=entropy_trace,
    )

    assert len(embedder.calls) == 3
    assert len(result.detected_gates) == 1
    assert result.detected_gates[0].gate_id == "g2"
    assert result.detected_gates[0].local_entropy == pytest.approx(8.5, rel=1e-6, abs=1e-6)
