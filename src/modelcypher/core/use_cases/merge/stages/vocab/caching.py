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

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelcypher.core.domain.cache import (
    CacheConfig,
    ComputationCache,
    TwoLevelCache,
    content_hash,
)

# Session cache for anchor maps - keyed by (embedding_hash, tokenizer_id, map_type)
_anchor_map_cache: dict[str, dict[str | int, "object"]] = {}
_anchor_disk_cache: "TwoLevelCache[dict[str, list[float]]] | None" = None
_ANCHOR_CACHE_VERSION = 1


def _make_embedding_cache_key(embedding: "object", backend: "object") -> str:
    """Create a cache key from embedding matrix shape and content sample."""
    cache = ComputationCache.shared()
    return cache.make_array_key(embedding, backend)


def _make_tokenizer_cache_key(tokenizer: Any, vocab: dict[str, int] | None) -> str:
    vocab_size = len(vocab) if vocab else 0
    name: str | None = None
    for attr in ("name_or_path", "model_name", "name"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            name = value
            break

    if name:
        return f"{type(tokenizer).__name__}:{name}:{vocab_size}"
    if not vocab:
        return f"{type(tokenizer).__name__}:{vocab_size}"

    sample_ids: set[int] = set()
    sample_ids.update(range(min(8, vocab_size)))
    if vocab_size > 8:
        sample_ids.update(range(max(0, vocab_size - 8), vocab_size))
    if vocab_size > 16:
        sample_ids.add(vocab_size // 2)

    sample_pairs = [(idx, token) for token, idx in vocab.items() if idx in sample_ids]
    sample_pairs.sort()
    sample_text = "|".join(f"{idx}:{token}" for idx, token in sample_pairs)
    fingerprint = content_hash({"sample": sample_text, "size": vocab_size})
    return f"{type(tokenizer).__name__}:{vocab_size}:{fingerprint}"


def _get_anchor_disk_cache() -> "TwoLevelCache[dict[str, list[float]]]":
    global _anchor_disk_cache
    if _anchor_disk_cache is None:
        base = Path.home() / "Library" / "Caches" / "ModelCypher" / "anchor_maps"
        config = CacheConfig(
            memory_limit=4,
            disk_ttl_seconds=30 * 24 * 60 * 60,
            cache_version=_ANCHOR_CACHE_VERSION,
        )
        _anchor_disk_cache = TwoLevelCache(
            cache_directory=base,
            serializer=lambda payload: payload,
            deserializer=lambda payload: payload,
            config=config,
        )
    return _anchor_disk_cache
