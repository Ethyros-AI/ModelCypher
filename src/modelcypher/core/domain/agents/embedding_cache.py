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

"""Disk-backed cache for atlas inventory embeddings."""

from __future__ import annotations

import threading
from typing import Any

from modelcypher.core.domain.cache import TwoLevelCache, content_hash
from modelcypher.utils.paths import get_modelcypher_home

_EMBED_CACHE: TwoLevelCache[dict] | None = None
_EMBED_CACHE_LOCK = threading.Lock()


def _get_embedding_cache() -> TwoLevelCache[dict]:
    global _EMBED_CACHE
    if _EMBED_CACHE is None:
        with _EMBED_CACHE_LOCK:
            if _EMBED_CACHE is None:
                cache_dir = get_modelcypher_home() / "cache" / "atlas_embeddings"
                _EMBED_CACHE = TwoLevelCache(
                    cache_directory=cache_dir,
                    serializer=lambda payload: payload,
                    deserializer=lambda payload: payload,
                    memory_limit=8,
                    disk_ttl_seconds=14 * 24 * 60 * 60,
                    cache_version=1,
                )
    return _EMBED_CACHE


def _embedder_signature(embedder: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {"class": type(embedder).__name__}

    model_name = getattr(embedder, "_model_name", None) or getattr(embedder, "model_name", None)
    if model_name:
        signature["model_name"] = model_name

    base_url = getattr(embedder, "_base_url", None) or getattr(embedder, "base_url", None)
    if base_url:
        signature["base_url"] = base_url

    try:
        dimension = embedder.dimension
    except Exception:
        dimension = None
    if dimension:
        signature["dimension"] = dimension

    return signature


def make_embedding_cache_key(
    embedder: Any,
    namespace: str,
    texts: list[str],
) -> str:
    signature_hash = content_hash(_embedder_signature(embedder))
    texts_hash = content_hash(texts)
    return f"{namespace}_{signature_hash}_{texts_hash}"


async def get_or_compute_embeddings(
    embedder: Any,
    backend: Any,
    namespace: str,
    texts: list[str],
) -> Any:
    if not texts:
        return backend.array([])

    cache = _get_embedding_cache()
    key = make_embedding_cache_key(embedder, namespace, texts)
    cached = cache.get(key)
    if cached:
        embeddings = cached.get("embeddings", [])
        if embeddings:
            arr = backend.array(embeddings)
            backend.eval(arr)
            return arr

    embeddings = await embedder.embed(texts)
    if not embeddings:
        return backend.array([])

    if hasattr(embeddings, "shape"):
        arr = embeddings
        backend.eval(arr)
        embeddings_list = backend.tolist(arr)
    else:
        embeddings_list = embeddings
        arr = backend.array(embeddings)
        backend.eval(arr)

    cache.set(key, {"embeddings": embeddings_list})
    return arr


def get_or_compute_embeddings_sync(
    embedder: Any,
    backend: Any,
    namespace: str,
    texts: list[str],
) -> Any:
    if not texts:
        return backend.array([])

    cache = _get_embedding_cache()
    key = make_embedding_cache_key(embedder, namespace, texts)
    cached = cache.get(key)
    if cached:
        embeddings = cached.get("embeddings", [])
        if embeddings:
            arr = backend.array(embeddings)
            backend.eval(arr)
            return arr

    embeddings = embedder.embed(texts)
    if not embeddings:
        return backend.array([])

    if hasattr(embeddings, "shape"):
        arr = embeddings
        backend.eval(arr)
        embeddings_list = backend.tolist(arr)
    else:
        embeddings_list = embeddings
        arr = backend.array(embeddings)
        backend.eval(arr)

    cache.set(key, {"embeddings": embeddings_list})
    return arr
