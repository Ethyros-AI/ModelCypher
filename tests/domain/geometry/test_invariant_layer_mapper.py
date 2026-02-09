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

import modelcypher.core.domain.geometry.invariant_layer_mapper as mapper_mod
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    InvariantLayerMapper,
    ModelFingerprints,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ActivatedDimension,
    ActivationFingerprint,
)


@dataclass
class _Probe:
    probe_id: str
    name: str
    description: str
    support_texts: list[str]
    source: str
    domain: str
    category_name: str
    cross_domain_weight: float


@dataclass
class _Invariant:
    id: str
    family: str
    domain: str
    support_texts: list[str]
    cross_domain_weight: float


@dataclass
class _TriScore:
    base: float
    cross_domain_multiplier: float
    relationship_bonus: float
    coherence_bonus: float


def _make_fingerprints(model_id: str) -> ModelFingerprints:
    return ModelFingerprints(
        model_id=model_id,
        layer_count=2,
        fingerprints=[
            ActivationFingerprint(
                prime_id="p1",
                prime_text="one",
                activated_dimensions={
                    0: [ActivatedDimension(index=1, activation=0.8)],
                    1: [ActivatedDimension(index=2, activation=0.2)],
                },
            ),
            ActivationFingerprint(
                prime_id="p2",
                prime_text="two",
                activated_dimensions={
                    0: [ActivatedDimension(index=3, activation=0.1)],
                    1: [ActivatedDimension(index=4, activation=0.9)],
                },
            ),
        ],
    )


def test_get_invariants_prefers_atlas_then_sequence(monkeypatch) -> None:
    probes = [
        _Probe("p1", "P1", "", ["t"], "atlas_a", "logic", "cat", 1.1),
        _Probe("p2", "P2", "", ["t"], "atlas_b", "math", "cat", 1.0),
    ]
    monkeypatch.setattr(mapper_mod, "get_atlas_probes", lambda: probes)
    monkeypatch.setattr(mapper_mod, "get_sequence_invariants", lambda: [])

    ids, seq_invariants, atlas_probes = InvariantLayerMapper._get_invariants()
    assert ids == ["p1", "p2"]
    assert seq_invariants == []
    assert atlas_probes == probes

    monkeypatch.setattr(mapper_mod, "get_atlas_probes", lambda: [])
    monkeypatch.setattr(
        mapper_mod,
        "get_sequence_invariants",
        lambda: [_Invariant("x", "family", "domain", ["t"], 1.0)],
    )
    ids2, seq2, atlas2 = InvariantLayerMapper._get_invariants()
    assert ids2 == ["invariant:family_x"]
    assert len(seq2) == 1
    assert atlas2 == []


def test_build_profile_similarity_and_alignment_helpers() -> None:
    source = _make_fingerprints("source")
    profile = InvariantLayerMapper._build_profile(source, ["p1", "p2"])

    assert profile.has_signal is True
    assert 0 in profile.vectors and 1 in profile.vectors
    assert profile.confidence_by_layer[0] > 0.0
    assert profile.coverage_by_layer[1] > 0.0
    assert profile.collapsed_count == 0

    profile_arr = InvariantLayerMapper._profile_array(
        2,
        profile,
        triangulation_scores={0: _TriScore(0.5, 1.2, 0.0, 0.2)},
    )
    assert len(profile_arr) == 2
    assert profile_arr[0].triangulation is not None
    assert profile_arr[1].triangulation is None

    assert InvariantLayerMapper._normalized_layer_index(-1, 5) == 4
    assert InvariantLayerMapper._mean_activation([]) == 0.0
    assert InvariantLayerMapper._mean_activation([ActivatedDimension(0, -2.0), ActivatedDimension(1, 4.0)]) == 3.0

    assert InvariantLayerMapper._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert InvariantLayerMapper._weighted_cosine_similarity([1.0, 1.0], [1.0, 0.0], [1.0, 2.0]) >= 0.0

    with pytest.raises(ValueError):
        InvariantLayerMapper._cosine_similarity_backend([1.0, 0.0], [1.0, 0.0, 0.0])

    mappings = InvariantLayerMapper._align_layers(
        source_samples=[0, 1],
        target_samples=[0, 1],
        similarity_matrix=[[0.9, 0.2], [0.1, 0.8]],
    )
    assert [(m.source_layer, m.target_layer) for m in mappings] == [(0, 0), (1, 1)]


def test_triangulation_score_computation_paths(monkeypatch) -> None:
    vectors = {0: [0.7, 0.3], 1: [0.1, 0.0]}
    invariants = [
        _Invariant("i1", "fam", "logic", ["a"], 1.0),
        _Invariant("i2", "fam", "math", ["b"], 1.0),
    ]

    monkeypatch.setattr(
        mapper_mod,
        "get_sequence_triangulation_scorer",
        lambda: (lambda domain_activations, _family, _ctx: _TriScore(
            base=max(domain_activations.values()),
            cross_domain_multiplier=1.5,
            relationship_bonus=0.1,
            coherence_bonus=0.2,
        )),
    )
    tri = InvariantLayerMapper._compute_triangulation_scores(vectors, invariants)
    assert set(tri.keys()) == {0, 1}
    assert tri[0].cross_domain_multiplier == pytest.approx(1.5)

    probes = [
        _Probe("p1", "P1", "", ["t"], "atlas_a", "logic", "cat", 1.2),
        _Probe("p2", "P2", "", ["t"], "atlas_b", "math", "cat", 0.8),
    ]
    multi, sources, domains = InvariantLayerMapper._compute_multi_atlas_scores(vectors, probes)
    assert set(multi.keys()) == {0, 1}
    assert sources == {"atlas_a", "atlas_b"}
    assert domains == {"logic", "math"}


def test_map_layers_end_to_end_and_validation_errors(monkeypatch) -> None:
    source = _make_fingerprints("source")
    target = _make_fingerprints("target")
    probes = [
        _Probe("p1", "P1", "", ["t"], "atlas_a", "logic", "cat", 1.0),
        _Probe("p2", "P2", "", ["t"], "atlas_b", "math", "cat", 1.0),
    ]
    monkeypatch.setattr(
        InvariantLayerMapper,
        "_get_invariants",
        staticmethod(lambda: (["p1", "p2"], [], probes)),
    )

    report = InvariantLayerMapper.map_layers(source, target)
    assert report.summary.mapped_layers == 2
    assert report.summary.total_probes_used == 2
    assert len(report.mappings) == 2
    assert report.summary.mean_similarity >= 0.0

    with pytest.raises(ValueError):
        InvariantLayerMapper.map_layers(
            ModelFingerprints(model_id="s", layer_count=0, fingerprints=[]),
            target,
        )

    monkeypatch.setattr(
        InvariantLayerMapper,
        "_get_invariants",
        staticmethod(lambda: ([], [], [])),
    )
    with pytest.raises(ValueError):
        InvariantLayerMapper.map_layers(source, target)

