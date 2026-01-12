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

"""Per-model probe activation caching helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProbeCache:
    probe_ids: list[str]
    probe_domains: list[str]
    probe_mode: str
    probe_corpus_hash: str
    hidden_activations: dict[int, "Array"]
    intermediate_activations: dict[int, "Array"]
    attention_activations: dict[int, "Array"]
    k_activations: dict[int, "Array"]
    v_activations: dict[int, "Array"]
    embedding_activations: "Array | None"


def _model_probe_cache_paths(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
) -> tuple[Path, Path]:
    """Resolve per-model probe cache paths."""
    from modelcypher.core.domain.geometry.model_profile import ModelProfileStore

    store = ModelProfileStore()
    cache_dir = store.probe_cache_dir(model_id)
    stem = f"{probe_mode}_{probe_corpus_hash}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _load_model_probe_cache(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
    backend: "Backend",
) -> ModelProbeCache | None:
    """Load per-model probe activations from disk."""
    import mlx.core as mx

    data_path, meta_path = _model_probe_cache_paths(
        model_id=model_id,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
    )
    if not data_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("PROBE CACHE: Failed to read %s: %s", meta_path, e)
        return None

    if meta.get("version") != 1:
        return None
    if meta.get("probe_mode") != probe_mode:
        return None
    if meta.get("probe_corpus_hash") != probe_corpus_hash:
        return None

    try:
        loaded = mx.load(data_path)
    except Exception as e:
        logger.warning("PROBE CACHE: Failed to load %s: %s", data_path, e)
        return None

    if not isinstance(loaded, dict):
        logger.warning("PROBE CACHE: Invalid cache format at %s", data_path)
        return None

    hidden: dict[int, "Array"] = {}
    intermediate: dict[int, "Array"] = {}
    attn: dict[int, "Array"] = {}
    k_acts: dict[int, "Array"] = {}
    v_acts: dict[int, "Array"] = {}
    embedding: "Array | None" = None

    for key, arr in loaded.items():
        if key.startswith("hidden_"):
            layer_idx = int(key.split("_")[1])
            hidden[layer_idx] = arr
        elif key.startswith("intermediate_"):
            layer_idx = int(key.split("_")[1])
            intermediate[layer_idx] = arr
        elif key.startswith("attn_q_"):
            layer_idx = int(key.split("_")[2])
            attn[layer_idx] = arr
        elif key.startswith("attn_k_"):
            layer_idx = int(key.split("_")[2])
            k_acts[layer_idx] = arr
        elif key.startswith("attn_v_"):
            layer_idx = int(key.split("_")[2])
            v_acts[layer_idx] = arr
        elif key == "embedding":
            embedding = arr

    probe_ids = meta.get("probe_ids", [])
    probe_domains = meta.get("probe_domains", [])

    if not hidden:
        logger.warning("PROBE CACHE: Missing hidden activations in %s", data_path)
        return None

    return ModelProbeCache(
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
        hidden_activations=hidden,
        intermediate_activations=intermediate,
        attention_activations=attn,
        k_activations=k_acts,
        v_activations=v_acts,
        embedding_activations=embedding,
    )


def _save_model_probe_cache(
    model_id: str,
    probe_mode: str,
    probe_corpus_hash: str,
    probe_ids: list[str],
    probe_domains: list[str],
    hidden_activations: dict[int, "Array"],
    intermediate_activations: dict[int, "Array"] | None,
    attention_activations: dict[int, "Array"] | None,
    k_activations: dict[int, "Array"] | None,
    v_activations: dict[int, "Array"] | None,
    embedding_activations: "Array | list[Array] | None",
) -> None:
    """Persist per-model probe activations to disk for reuse."""
    import mlx.core as mx

    data: dict[str, "Array"] = {}
    for layer_idx, acts in hidden_activations.items():
        data[f"hidden_{layer_idx}"] = acts
    if intermediate_activations:
        for layer_idx, acts in intermediate_activations.items():
            data[f"intermediate_{layer_idx}"] = acts
    if attention_activations:
        for layer_idx, acts in attention_activations.items():
            data[f"attn_q_{layer_idx}"] = acts
    if k_activations:
        for layer_idx, acts in k_activations.items():
            data[f"attn_k_{layer_idx}"] = acts
    if v_activations:
        for layer_idx, acts in v_activations.items():
            data[f"attn_v_{layer_idx}"] = acts

    if embedding_activations is not None:
        if isinstance(embedding_activations, list):
            if embedding_activations:
                data["embedding"] = mx.stack(embedding_activations, axis=0)
        else:
            data["embedding"] = embedding_activations

    data_path, meta_path = _model_probe_cache_paths(
        model_id=model_id,
        probe_mode=probe_mode,
        probe_corpus_hash=probe_corpus_hash,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez_compressed(data_path, **data)

    spaces = ["hidden"]
    if intermediate_activations:
        spaces.append("intermediate")
    if attention_activations:
        spaces.append("attention_q")
    if k_activations:
        spaces.append("attention_k")
    if v_activations:
        spaces.append("attention_v")
    if embedding_activations is not None:
        spaces.append("embedding")

    meta = {
        "version": 1,
        "probe_mode": probe_mode,
        "probe_corpus_hash": probe_corpus_hash,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
        "spaces": spaces,
        "created_at": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("PROBE CACHE: Saved per-model activations to %s", data_path)
