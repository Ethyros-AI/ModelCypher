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

import logging
from typing import Any

from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.cache import content_hash
from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner

from .anchor_selection import _apply_anchor_weights
from .caching import (
    _ANCHOR_CACHE_VERSION,
    _anchor_map_cache,
    _get_anchor_disk_cache,
    _make_embedding_cache_key,
)
from .frechet_ops import _frechet_mean_from_bytes, _frechet_mean_from_ids
from .tokenizer_utils import _encode_bytes, _encode_ids

logger = logging.getLogger(__name__)


def _build_atlas_anchor_map(
    tokenizer: Any,
    embedding: "object",
    vocab_size: int,
    backend: "object",
    use_all_support_texts: bool = False,
    byte_map: dict[int, "object"] | None = None,
    use_byte_anchors: bool = False,
    cache_key: str | None = None,
    tokenizer_key: str | None = None,
) -> dict[str, "object"]:
    """Build UnifiedAtlas anchor map with session caching.

    This is expensive (hundreds of Frechet mean computations) so we cache the result
    based on the embedding matrix hash, tokenizer signature, and support_texts flag.
    """
    global _anchor_map_cache

    use_byte_anchors = bool(use_byte_anchors and byte_map)
    anchor_mode = "byte" if use_byte_anchors else "token"
    embed_key = cache_key or _make_embedding_cache_key(embedding, backend)
    tokenizer_key = tokenizer_key or f"{type(tokenizer).__name__}:{vocab_size}"
    key_payload = {
        "type": "atlas_map",
        "embed": embed_key,
        "tokenizer": tokenizer_key,
        "support_all": use_all_support_texts,
        "anchor_mode": anchor_mode,
        "version": _ANCHOR_CACHE_VERSION,
    }
    cache_key = f"atlas_map_{content_hash(key_payload)}"

    # Check cache
    if cache_key in _anchor_map_cache:
        logger.debug("Cache hit for atlas map: %s", cache_key[:20])
        return _anchor_map_cache[cache_key]

    disk_cache = _get_anchor_disk_cache()
    disk_payload = disk_cache.get(cache_key)
    if disk_payload is not None:
        anchor_map = {key: backend.array(value) for key, value in disk_payload.items()}
        _anchor_map_cache[cache_key] = anchor_map
        logger.debug("Cache hit for atlas map (disk): %s", cache_key[:20])
        return anchor_map

    logger.debug("Cache miss for atlas map: %s - computing...", cache_key[:20])

    anchor_map: dict[str, "object"] = {}
    probes = UnifiedAtlasInventory.all_probes()

    for probe in probes:
        support_texts = [t for t in probe.support_texts if t and len(t.strip()) >= 2]
        if not support_texts:
            continue

        if not use_all_support_texts:
            text = support_texts[0]
            if use_byte_anchors and byte_map is not None:
                byte_values = _encode_bytes(text)
                vec = _frechet_mean_from_bytes(byte_values, byte_map, backend)
            else:
                token_ids = _encode_ids(tokenizer, text)
                valid = [tid for tid in token_ids if 0 <= tid < vocab_size]
                vec = _frechet_mean_from_ids(valid, embedding, backend)
            if vec is not None:
                anchor_map[probe.probe_id] = vec
            continue

        for idx, text in enumerate(support_texts):
            if use_byte_anchors and byte_map is not None:
                byte_values = _encode_bytes(text)
                vec = _frechet_mean_from_bytes(byte_values, byte_map, backend)
            else:
                token_ids = _encode_ids(tokenizer, text)
                valid = [tid for tid in token_ids if 0 <= tid < vocab_size]
                vec = _frechet_mean_from_ids(valid, embedding, backend)
            if vec is not None:
                anchor_map[f"{probe.probe_id}:{idx}"] = vec

    # Cache the result
    _anchor_map_cache[cache_key] = anchor_map
    try:
        disk_payload = {
            str(key): backend.to_numpy(value).tolist()
            for key, value in anchor_map.items()
        }
        disk_cache.set(cache_key, disk_payload)
    except (TypeError, ValueError) as e:
        logger.debug("Skipping atlas map disk cache: %s", e)
    logger.debug("Cached atlas map with %d entries", len(anchor_map))

    return anchor_map


def _apply_feature_transform_to_anchor_map(
    anchor_map: dict[str | int, "object"],
    transform: "object | None",
    backend: "object",
) -> dict[str | int, "object"]:
    if not anchor_map or transform is None:
        return anchor_map

    labels = list(anchor_map.keys())
    matrix = backend.stack([anchor_map[label] for label in labels], axis=0)
    transformed = backend.matmul(matrix, transform)
    backend.eval(transformed)
    return {label: transformed[idx] for idx, label in enumerate(labels)}


def _align_unified_atlas(
    source_embed: "object",
    target_embed: "object",
    source_tokenizer: Any,
    target_tokenizer: Any,
    backend: "object",
    use_all_support_texts: bool = False,
    max_iterations: int = 1000,
    tolerance: float | None = None,
    max_rounds: int = 1,
    anchor_weights: list[float] | None = None,
    initial_transform: "object | None" = None,
) -> dict[str, Any] | None:
    """Align embeddings using unified atlas anchors.

    Args:
        tolerance: Alignment tolerance. If None, uses machine_epsilon.
    """
    source_anchors = _build_atlas_anchor_map(
        source_tokenizer,
        source_embed,
        source_embed.shape[0],
        backend,
        use_all_support_texts=use_all_support_texts,
    )
    target_anchors = _build_atlas_anchor_map(
        target_tokenizer,
        target_embed,
        target_embed.shape[0],
        backend,
        use_all_support_texts=use_all_support_texts,
    )
    shared = sorted(set(source_anchors) & set(target_anchors))
    if len(shared) < 2:
        return None

    source_matrix = backend.stack([source_anchors[k] for k in shared], axis=0)
    target_matrix = backend.stack([target_anchors[k] for k in shared], axis=0)
    backend.eval(source_matrix, target_matrix)

    cka_before = compute_cka(
        source_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best
    weighted_source = _apply_anchor_weights(source_matrix, anchor_weights, backend)
    weighted_target = _apply_anchor_weights(target_matrix, anchor_weights, backend)
    aligner = GramAligner(
        backend=backend,
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        tolerance=tolerance,
    )
    init_transform = (
        backend.array(initial_transform) if initial_transform is not None else None
    )
    result = aligner.find_perfect_alignment(
        weighted_source,
        weighted_target,
        initial_transform=init_transform,
    )
    transform = backend.array(result.feature_transform)
    aligned_source = backend.matmul(source_embed, transform)
    backend.eval(aligned_source)

    aligned_matrix = backend.matmul(source_matrix, transform)
    backend.eval(aligned_matrix)
    cka_after = compute_cka(
        aligned_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best

    return {
        "aligned_source": aligned_source,
        "aligned_matrix": aligned_matrix,
        "target_matrix": target_matrix,
        "anchor_labels": shared,
        "feature_transform": transform,
        "anchors_shared": len(shared),
        "cka_before": cka_before,
        "cka_after": cka_after,
        "alignment_error": result.alignment_error,
        "iterations": result.iterations,
    }
