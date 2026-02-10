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

import modelcypher.core.domain.atlas.atlas_base as atlas_base


@dataclass(frozen=True)
class _Concept:
    id: str
    canonical_name: str
    text: str


@dataclass(frozen=True)
class _Signature(atlas_base.BaseAtlasSignature):
    pass


class _Atlas(atlas_base.BaseAtlas[_Concept, _Signature]):
    def __init__(self, concepts: list[_Concept], embedder=None):
        super().__init__(embedder=embedder)
        self._concepts = concepts

    @property
    def inventory(self) -> list[_Concept]:
        return self._concepts

    def _get_concept_text(self, concept: _Concept) -> str:
        return concept.text

    def _create_signature(self, concept_ids: list[str], values: list[float]) -> _Signature:
        return _Signature(concept_ids=concept_ids, values=values)


class _Embedder:
    def __init__(self, payload: list[list[float]] | Exception):
        self.payload = payload
        self.calls = 0

    async def embed(self, texts: list[str]):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_base_signature_helpers(any_backend) -> None:
    sig = atlas_base.BaseAtlasSignature(
        concept_ids=["a", "b", "c"],
        values=[0.1, 0.9, 0.5],
    )
    assert sig.top_k(2)[0][0] == "b"
    assert sig.top_k(0) == []
    assert sig.to_dict()["c"] == 0.5

    zero_sig = atlas_base.BaseAtlasSignature(concept_ids=["x"], values=[0.0])
    assert zero_sig.top_k() == []

    replaced = sig._with_values([1.0, 2.0, 3.0])
    assert replaced.values == [1.0, 2.0, 3.0]
    assert replaced.concept_ids == sig.concept_ids


def test_normalized_entropy_paths() -> None:
    assert atlas_base.BaseAtlas.normalized_entropy([]) is None
    assert atlas_base.BaseAtlas.normalized_entropy([0.0, 0.0]) is None
    assert atlas_base.BaseAtlas.normalized_entropy([1.0]) == 0.0

    entropy = atlas_base.BaseAtlas.normalized_entropy([0.25, 0.25, 0.5])
    assert entropy is not None
    assert 0.0 <= entropy <= 1.0


async def test_signature_success_and_cache_flow(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(atlas_base, "get_default_backend", lambda: b)

    concepts = [_Concept("a", "A", "text-a"), _Concept("b", "B", "text-b")]
    embedder = _Embedder(payload=[[1.0, 0.0]])
    atlas = _Atlas(concepts, embedder=embedder)

    concept_calls = {"count": 0}

    async def _fake_concept_embeddings(*_args, **_kwargs):
        concept_calls["count"] += 1
        return b.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(atlas_base, "get_or_compute_embeddings", _fake_concept_embeddings)

    signature = await atlas.signature("hello")
    assert signature is not None
    assert signature.concept_ids == ["a", "b"]
    assert len(signature.values) == 2
    assert all(v >= 0.0 for v in signature.values)
    assert concept_calls["count"] == 1

    # Cached concept embeddings should prevent re-computation.
    second = await atlas.signature("hello again")
    assert second is not None
    assert concept_calls["count"] == 1

    atlas.clear_cache()
    await atlas.signature("hello once more")
    assert concept_calls["count"] == 2


async def test_signature_returns_none_on_invalid_inputs_or_failures(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(atlas_base, "get_default_backend", lambda: b)

    concepts = [_Concept("a", "A", "text-a")]
    atlas_no_embedder = _Atlas(concepts, embedder=None)
    assert await atlas_no_embedder.signature("x") is None
    assert await atlas_no_embedder.signature("   ") is None

    atlas = _Atlas(concepts, embedder=_Embedder(payload=[[1.0, 0.0]]))

    # Shape mismatch with inventory.
    async def _bad_shape(*_args, **_kwargs):
        return b.array([])

    monkeypatch.setattr(atlas_base, "get_or_compute_embeddings", _bad_shape)
    assert await atlas.signature("x") is None

    # Empty embedder output.
    atlas_empty_embed = _Atlas(concepts, embedder=_Embedder(payload=[]))
    monkeypatch.setattr(
        atlas_base,
        "get_or_compute_embeddings",
        lambda *_args, **_kwargs: b.array([[1.0, 0.0]]),
    )
    assert await atlas_empty_embed.signature("x") is None

    # Embedder exception path.
    atlas_failing = _Atlas(concepts, embedder=_Embedder(payload=RuntimeError("embed-fail")))
    assert await atlas_failing.signature("x") is None


async def test_get_or_create_concept_embeddings_without_embedder(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(atlas_base, "get_default_backend", lambda: b)
    atlas = _Atlas([_Concept("a", "A", "text-a")], embedder=None)
    emb = await atlas._get_or_create_concept_embeddings()
    assert int(b.shape(emb)[0]) == 0

