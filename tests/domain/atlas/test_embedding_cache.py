# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import asyncio

import modelcypher.core.domain.atlas.embedding_cache as embedding_cache


class _FakeCache:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, dict]] = []

    def get(self, key: str):
        self.get_calls.append(key)
        return self.data.get(key)

    def set(self, key: str, value: dict) -> None:
        self.set_calls.append((key, value))
        self.data[key] = value


class _AsyncEmbedder:
    def __init__(self, payload) -> None:
        self._model_name = "atlas-test"
        self._base_url = "https://example.invalid"
        self.dimension = 2
        self.payload = payload
        self.calls = 0

    async def embed(self, _texts: list[str]):
        self.calls += 1
        return self.payload


class _SyncEmbedder:
    def __init__(self, payload) -> None:
        self.model_name = "atlas-test-sync"
        self.base_url = "https://example.invalid"
        self.dimension = 2
        self.payload = payload
        self.calls = 0

    def embed(self, _texts: list[str]):
        self.calls += 1
        return self.payload


class _EmbedderWithoutDimension:
    _model_name = "no-dim"

    @property
    def dimension(self):  # pragma: no cover - accessed via exception path
        raise RuntimeError("dimension unavailable")


def test_embedder_signature_includes_safe_metadata() -> None:
    signature = embedding_cache._embedder_signature(_EmbedderWithoutDimension())

    assert signature["class"] == "_EmbedderWithoutDimension"
    assert signature["model_name"] == "no-dim"
    assert "dimension" not in signature


def test_make_embedding_cache_key_changes_with_inputs() -> None:
    embedder = _SyncEmbedder(payload=[[1.0, 2.0]])

    key_a = embedding_cache.make_embedding_cache_key(embedder, "ns-a", ["alpha"])
    key_b = embedding_cache.make_embedding_cache_key(embedder, "ns-b", ["alpha"])
    key_c = embedding_cache.make_embedding_cache_key(embedder, "ns-a", ["beta"])

    assert key_a != key_b
    assert key_a != key_c


def test_get_or_compute_embeddings_sync_miss_then_hit(any_backend, monkeypatch) -> None:
    b = any_backend
    fake_cache = _FakeCache()
    monkeypatch.setattr(embedding_cache, "_EMBED_CACHE", fake_cache)

    embedder = _SyncEmbedder(payload=[[1.0, 2.0], [3.0, 4.0]])
    texts = ["alpha", "beta"]

    first = embedding_cache.get_or_compute_embeddings_sync(embedder, b, "ns", texts)
    second = embedding_cache.get_or_compute_embeddings_sync(embedder, b, "ns", texts)

    assert embedder.calls == 1
    assert b.tolist(first) == [[1.0, 2.0], [3.0, 4.0]]
    assert b.tolist(second) == [[1.0, 2.0], [3.0, 4.0]]
    assert len(fake_cache.set_calls) == 1


def test_get_or_compute_embeddings_sync_accepts_backend_arrays(any_backend, monkeypatch) -> None:
    b = any_backend
    fake_cache = _FakeCache()
    monkeypatch.setattr(embedding_cache, "_EMBED_CACHE", fake_cache)

    embedder = _SyncEmbedder(payload=b.array([[5.0, 6.0]]))

    arr = embedding_cache.get_or_compute_embeddings_sync(embedder, b, "ns", ["single"])
    cached_payload = next(iter(fake_cache.data.values()))

    assert b.tolist(arr) == [[5.0, 6.0]]
    assert cached_payload["embeddings"] == [[5.0, 6.0]]


def test_get_or_compute_embeddings_async_miss_then_hit(any_backend, monkeypatch) -> None:
    b = any_backend
    fake_cache = _FakeCache()
    monkeypatch.setattr(embedding_cache, "_EMBED_CACHE", fake_cache)

    embedder = _AsyncEmbedder(payload=[[7.0, 8.0], [9.0, 10.0]])

    first = asyncio.run(
        embedding_cache.get_or_compute_embeddings(embedder, b, "ns", ["x", "y"])
    )
    second = asyncio.run(
        embedding_cache.get_or_compute_embeddings(embedder, b, "ns", ["x", "y"])
    )

    assert embedder.calls == 1
    assert b.tolist(first) == [[7.0, 8.0], [9.0, 10.0]]
    assert b.tolist(second) == [[7.0, 8.0], [9.0, 10.0]]


def test_get_or_compute_embeddings_empty_texts_short_circuit(any_backend, monkeypatch) -> None:
    b = any_backend
    fake_cache = _FakeCache()
    monkeypatch.setattr(embedding_cache, "_EMBED_CACHE", fake_cache)

    embedder_sync = _SyncEmbedder(payload=[[1.0, 2.0]])
    embedder_async = _AsyncEmbedder(payload=[[1.0, 2.0]])

    sync_arr = embedding_cache.get_or_compute_embeddings_sync(embedder_sync, b, "ns", [])
    async_arr = asyncio.run(embedding_cache.get_or_compute_embeddings(embedder_async, b, "ns", []))

    assert embedder_sync.calls == 0
    assert embedder_async.calls == 0
    assert b.tolist(sync_arr) == []
    assert b.tolist(async_arr) == []

