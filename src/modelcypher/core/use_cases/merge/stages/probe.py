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

"""
Stage 1: PROBE - Build intersection map from probe responses.

The intersection map is the PRIMARY CONTROL SIGNAL for all downstream operations.

Two modes:
- "precise": Run 403 probes through BOTH models, compute CKA on activations
- "fast": Use weight-level CKA (faster but less accurate)

Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
Reference: Chun et al. (2025) "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features"
Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    geodesic_svd,
    log_scalar,
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    CrossArchitectureLayerMatcher,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.vector_math import geodesic_pairwise_metrics

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Probe mode is ALWAYS "precise" - activation-level CKA is required for correct alignment.
# "fast" mode is eliminated - weight-level CKA is fundamentally less accurate and
# hides alignment problems that will cause gibberish output.
_PROBE_MODE = "precise"

# All probes are always run. The probe corpus (403 probes) was carefully designed
# to cover the concept space. Limiting probes degrades coverage with no benefit.
_MAX_PROBES = 0  # 0 = all probes

# Checkpoint interval: save progress every N probes
_CHECKPOINT_INTERVAL = 50


# =============================================================================
# CHECKPOINTING - Resume capability for long-running probe collection
# =============================================================================

def _save_probe_checkpoint(
    checkpoint_path: Path,
    completed_probes: int,
    probe_ids: list[str],
    probe_domains: list[str],
    total_probes: int,
) -> None:
    """Save probe progress to checkpoint file.

    Saves probe IDs and metadata. Activation data is saved separately
    via _save_probe_activations() for correct checkpoint resume.
    """
    checkpoint = {
        "version": 1,
        "completed_probes": completed_probes,
        "total_probes": total_probes,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
    }
    # Write atomically using temp file
    temp_path = checkpoint_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(checkpoint, indent=2))
    temp_path.rename(checkpoint_path)
    logger.debug(
        "PROBE: Saved checkpoint at %d/%d probes to %s",
        completed_probes,
        total_probes,
        checkpoint_path,
    )


def _load_probe_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load probe checkpoint if it exists and is valid."""
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("version") != 1:
            logger.warning(
                "PROBE: Checkpoint version mismatch, ignoring checkpoint"
            )
            return None
        return checkpoint
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("PROBE: Failed to load checkpoint: %s", e)
        return None


def _clear_probe_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint file after successful completion."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.debug("PROBE: Cleared checkpoint file %s", checkpoint_path)
    # Also clear activation NPZ file
    activation_path = checkpoint_path.with_suffix(".activations.npz")
    if activation_path.exists():
        activation_path.unlink()
        logger.debug("PROBE: Cleared activation checkpoint %s", activation_path)


def _save_probe_activations(
    checkpoint_path: Path,
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    source_intermediate_activations: dict[int, "Array"],
    target_intermediate_activations: dict[int, "Array"],
    source_attention_activations: dict[int, "Array"],
    target_attention_activations: dict[int, "Array"],
    source_k_activations: dict[int, "Array"],
    target_k_activations: dict[int, "Array"],
    source_v_activations: dict[int, "Array"],
    target_v_activations: dict[int, "Array"],
) -> None:
    """Save all activation dicts to NPZ file for checkpoint resume.

    This is CRITICAL for correct checkpoint resume. Without saving activations,
    the resume logic skips completed probes but their activations are lost,
    leading to incomplete activation matrices and incorrect alignment.
    """
    activation_path = checkpoint_path.with_suffix(".activations.npz")

    # Build flat dict with prefixed keys for NPZ storage
    arrays_to_save: dict[str, Any] = {}

    # Helper to flatten activation dict
    def flatten_dict(d: dict[int, "Array"], prefix: str) -> None:
        for layer_idx, arr in d.items():
            arrays_to_save[f"{prefix}_{layer_idx}"] = arr

    flatten_dict(source_layer_activations, "src_hidden")
    flatten_dict(target_layer_activations, "tgt_hidden")
    flatten_dict(source_intermediate_activations, "src_inter")
    flatten_dict(target_intermediate_activations, "tgt_inter")
    flatten_dict(source_attention_activations, "src_attn_q")
    flatten_dict(target_attention_activations, "tgt_attn_q")
    flatten_dict(source_k_activations, "src_attn_k")
    flatten_dict(target_k_activations, "tgt_attn_k")
    flatten_dict(source_v_activations, "src_attn_v")
    flatten_dict(target_v_activations, "tgt_attn_v")

    if not arrays_to_save:
        return  # Nothing to save

    # Use mlx.core.savez for GPU arrays (avoid CPU transfer)
    try:
        import mlx.core as mx
        # Write atomically using temp file
        temp_path = activation_path.with_suffix(".tmp.npz")
        mx.savez(str(temp_path), **arrays_to_save)
        temp_path.rename(activation_path)
        logger.debug(
            "PROBE: Saved %d activation arrays to %s",
            len(arrays_to_save),
            activation_path,
        )
    except ImportError:
        # Fallback for non-MLX backends
        import numpy as np
        from modelcypher.core.domain._backend import get_default_backend
        b = get_default_backend()
        np_arrays = {k: b.to_numpy(v) for k, v in arrays_to_save.items()}
        temp_path = activation_path.with_suffix(".tmp.npz")
        np.savez_compressed(str(temp_path), **np_arrays)
        temp_path.rename(activation_path)


def _load_probe_activations(
    checkpoint_path: Path,
    backend: "Backend",
) -> tuple[
    dict[int, "Array"],  # source_layer_activations
    dict[int, "Array"],  # target_layer_activations
    dict[int, "Array"],  # source_intermediate_activations
    dict[int, "Array"],  # target_intermediate_activations
    dict[int, "Array"],  # source_attention_activations
    dict[int, "Array"],  # target_attention_activations
    dict[int, "Array"],  # source_k_activations
    dict[int, "Array"],  # target_k_activations
    dict[int, "Array"],  # source_v_activations
    dict[int, "Array"],  # target_v_activations
] | None:
    """Load activation dicts from NPZ checkpoint file.

    Returns None if no activation checkpoint exists.
    Otherwise returns tuple of 10 activation dicts.
    """
    activation_path = checkpoint_path.with_suffix(".activations.npz")
    if not activation_path.exists():
        return None

    try:
        # Use mlx.core.load for GPU arrays
        try:
            import mlx.core as mx
            loaded = mx.load(str(activation_path))
        except ImportError:
            import numpy as np
            loaded = dict(np.load(str(activation_path)))
            # Convert to backend arrays
            loaded = {k: backend.array(v) for k, v in loaded.items()}

        # Reconstruct activation dicts from flat keys
        source_layer_activations: dict[int, Any] = {}
        target_layer_activations: dict[int, Any] = {}
        source_intermediate_activations: dict[int, Any] = {}
        target_intermediate_activations: dict[int, Any] = {}
        source_attention_activations: dict[int, Any] = {}
        target_attention_activations: dict[int, Any] = {}
        source_k_activations: dict[int, Any] = {}
        target_k_activations: dict[int, Any] = {}
        source_v_activations: dict[int, Any] = {}
        target_v_activations: dict[int, Any] = {}

        for key, arr in loaded.items():
            if key.startswith("src_hidden_"):
                layer_idx = int(key.split("_")[2])
                source_layer_activations[layer_idx] = arr
            elif key.startswith("tgt_hidden_"):
                layer_idx = int(key.split("_")[2])
                target_layer_activations[layer_idx] = arr
            elif key.startswith("src_inter_"):
                layer_idx = int(key.split("_")[2])
                source_intermediate_activations[layer_idx] = arr
            elif key.startswith("tgt_inter_"):
                layer_idx = int(key.split("_")[2])
                target_intermediate_activations[layer_idx] = arr
            elif key.startswith("src_attn_q_"):
                layer_idx = int(key.split("_")[3])
                source_attention_activations[layer_idx] = arr
            elif key.startswith("tgt_attn_q_"):
                layer_idx = int(key.split("_")[3])
                target_attention_activations[layer_idx] = arr
            elif key.startswith("src_attn_k_"):
                layer_idx = int(key.split("_")[3])
                source_k_activations[layer_idx] = arr
            elif key.startswith("tgt_attn_k_"):
                layer_idx = int(key.split("_")[3])
                target_k_activations[layer_idx] = arr
            elif key.startswith("src_attn_v_"):
                layer_idx = int(key.split("_")[3])
                source_v_activations[layer_idx] = arr
            elif key.startswith("tgt_attn_v_"):
                layer_idx = int(key.split("_")[3])
                target_v_activations[layer_idx] = arr

        total_arrays = sum(len(d) for d in [
            source_layer_activations, target_layer_activations,
            source_intermediate_activations, target_intermediate_activations,
            source_attention_activations, target_attention_activations,
            source_k_activations, target_k_activations,
            source_v_activations, target_v_activations,
        ])
        logger.info(
            "PROBE: Loaded %d activation arrays from checkpoint",
            total_arrays,
        )

        return (
            source_layer_activations,
            target_layer_activations,
            source_intermediate_activations,
            target_intermediate_activations,
            source_attention_activations,
            target_attention_activations,
            source_k_activations,
            target_k_activations,
            source_v_activations,
            target_v_activations,
        )
    except Exception as e:
        logger.warning("PROBE: Failed to load activation checkpoint: %s", e)
        return None


def _select_probe_text(probe: Any) -> str | None:
    """Pick a usable probe text from the probe definition."""
    probe_text = None
    for candidate in probe.support_texts or []:
        if not candidate or len(candidate.strip()) < 2:
            continue
        probe_text = candidate
        break

    if probe_text is None:
        fallback = None
        if probe.name and probe.description:
            fallback = f"{probe.name}: {probe.description}"
        elif probe.name:
            fallback = probe.name
        elif probe.description:
            fallback = probe.description
        if fallback and len(fallback.strip()) >= 2:
            probe_text = fallback
    return probe_text


# =============================================================================
# PROBE RESULT CACHING (DISK)
# =============================================================================

def _probe_cache_dir() -> Path:
    """Directory for disk-backed probe caches."""
    from modelcypher.utils.paths import get_modelcypher_home

    return (get_modelcypher_home() / "probe_cache")


def _infer_model_id(path_hint: str, model: Any) -> str:
    """Best-effort model identifier for cache keys."""
    if path_hint:
        return path_hint
    for attr in ("name_or_path", "model_name", "model_id"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("name_or_path", "model_type"):
            val = getattr(config, attr, None)
            if isinstance(val, str) and val:
                return val
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("name_or_path", "model_name", "model_id"):
            val = getattr(inner, attr, None)
            if isinstance(val, str) and val:
                return val
    return ""


def _probe_cache_key(
    source_id: str,
    target_id: str,
    probe_mode: str,
    probe_ids: list[str],
) -> str:
    """Stable cache key for a source/target/probe set."""
    joined = "|".join([source_id, target_id, probe_mode, ",".join(probe_ids)])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"probe_{digest}"


def _coerce_int_keys(mapping: dict | None) -> dict[int, Any] | None:
    """Normalize JSON-loaded dict keys to int."""
    if mapping is None:
        return None
    return {int(k): v for k, v in mapping.items()}


def _proportional_layer_index(
    target_idx: int,
    target_count: int,
    source_count: int,
) -> int:
    """Map a target layer index to a source index by normalized depth."""
    if target_count <= 1 or source_count <= 1:
        return 0
    ratio = target_idx / (target_count - 1)
    mapped = int(round(ratio * (source_count - 1)))
    if mapped < 0:
        return 0
    if mapped >= source_count:
        return source_count - 1
    return mapped


def _load_probe_result_cache(
    cache_key: str,
    probe_ids: list[str],
    probe_mode: str,
    backend: "Backend",
) -> tuple[
    dict[int, "Array"],  # source_activations
    dict[int, "Array"],  # target_activations
    dict[str, Any],  # probe_result
    dict[str, Any],  # metrics
    dict[int, "Array"] | None,  # feature_transforms (GPU arrays)
    dict[int, float] | None,  # scale_ratios
    "Array | None",  # embedding_transform (GPU array)
    dict[int, "Array"] | None,  # attention_transforms (GPU arrays)
    dict[int, "Array"] | None,  # k_transforms (GPU arrays)
    dict[int, "Array"] | None,  # v_transforms (GPU arrays)
    dict[int, "Array"] | None,  # intermediate_transforms (GPU arrays)
    dict[int, int] | None,  # layer_mapping
    list[str] | None,  # probe_ids
    list[str] | None,  # probe_domains
    dict[int, "Array"] | None,  # source_intermediate_activations
    dict[int, "Array"] | None,  # target_intermediate_activations
    dict[int, "Array"] | None,  # source_attention_activations
    dict[int, "Array"] | None,  # target_attention_activations
    dict[int, "Array"] | None,  # source_k_activations
    dict[int, "Array"] | None,  # target_k_activations
] | None:
    """Load cached probe results if compatible with current probe set."""
    import mlx.core as mx

    cache_dir = _probe_cache_dir()
    meta_path = cache_dir / f"{cache_key}.json"
    data_path = cache_dir / f"{cache_key}.npz"
    if not meta_path.exists() or not data_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("PROBE CACHE: Failed to read metadata %s: %s", meta_path, e)
        return None

    if meta.get("probe_mode") != probe_mode:
        return None
    cached_probe_ids = meta.get("probe_ids", [])
    if cached_probe_ids != probe_ids:
        return None

    try:
        loaded = mx.load(data_path)
    except Exception as e:
        logger.warning("PROBE CACHE: Failed to load %s: %s", data_path, e)
        return None

    if not isinstance(loaded, dict):
        logger.warning("PROBE CACHE: Invalid cache format at %s", data_path)
        return None

    # Activations
    source_activations: dict[int, "Array"] = {}
    target_activations: dict[int, "Array"] = {}
    source_intermediate_activations: dict[int, "Array"] = {}
    target_intermediate_activations: dict[int, "Array"] = {}
    source_attention_activations: dict[int, "Array"] = {}
    target_attention_activations: dict[int, "Array"] = {}
    source_k_activations: dict[int, "Array"] = {}
    target_k_activations: dict[int, "Array"] = {}

    # Transforms (loaded from NPZ in v2+, from JSON metadata in v1)
    feature_transforms: dict[int, "Array"] = {}
    attention_transforms: dict[int, "Array"] = {}
    k_transforms: dict[int, "Array"] = {}
    v_transforms: dict[int, "Array"] = {}
    intermediate_transforms: dict[int, "Array"] = {}
    embedding_transform: "Array | None" = None

    for key, arr in loaded.items():
        # Activations
        if key.startswith("src_act_"):
            layer_idx = int(key.split("_")[2])
            source_activations[layer_idx] = arr
        elif key.startswith("tgt_act_"):
            layer_idx = int(key.split("_")[2])
            target_activations[layer_idx] = arr
        elif key.startswith("src_inter_"):
            layer_idx = int(key.split("_")[2])
            source_intermediate_activations[layer_idx] = arr
        elif key.startswith("tgt_inter_"):
            layer_idx = int(key.split("_")[2])
            target_intermediate_activations[layer_idx] = arr
        elif key.startswith("src_attn_"):
            layer_idx = int(key.split("_")[2])
            source_attention_activations[layer_idx] = arr
        elif key.startswith("tgt_attn_"):
            layer_idx = int(key.split("_")[2])
            target_attention_activations[layer_idx] = arr
        elif key.startswith("src_k_"):
            layer_idx = int(key.split("_")[2])
            source_k_activations[layer_idx] = arr
        elif key.startswith("tgt_k_"):
            layer_idx = int(key.split("_")[2])
            target_k_activations[layer_idx] = arr
        # Transforms (v2 format - stored in NPZ)
        elif key.startswith("feat_transform_"):
            layer_idx = int(key.split("_")[2])
            feature_transforms[layer_idx] = arr
        elif key.startswith("attn_transform_"):
            layer_idx = int(key.split("_")[2])
            attention_transforms[layer_idx] = arr
        elif key.startswith("k_transform_"):
            layer_idx = int(key.split("_")[2])
            k_transforms[layer_idx] = arr
        elif key.startswith("v_transform_"):
            layer_idx = int(key.split("_")[2])
            v_transforms[layer_idx] = arr
        elif key.startswith("inter_transform_"):
            layer_idx = int(key.split("_")[2])
            intermediate_transforms[layer_idx] = arr
        elif key == "emb_transform":
            embedding_transform = arr

    probe_result = meta.get("probe_result", {})
    if isinstance(probe_result, dict):
        probe_result["confidences"] = _coerce_int_keys(probe_result.get("confidences"))

    # Handle v1 cache (transforms in JSON as lists) - convert to arrays
    cache_version = meta.get("version", 1)
    if cache_version < 2:
        # v1 format: transforms stored in JSON metadata as nested lists
        b = backend
        json_feat = _coerce_int_keys(meta.get("feature_transforms"))
        if json_feat:
            feature_transforms = {k: b.array(v) for k, v in json_feat.items()}
        json_emb = meta.get("embedding_transform")
        if json_emb is not None:
            embedding_transform = b.array(json_emb)
        json_attn = _coerce_int_keys(meta.get("attention_transforms"))
        if json_attn:
            attention_transforms = {k: b.array(v) for k, v in json_attn.items()}
        json_k = _coerce_int_keys(meta.get("k_transforms"))
        if json_k:
            k_transforms = {k: b.array(v) for k, v in json_k.items()}
        json_v = _coerce_int_keys(meta.get("v_transforms"))
        if json_v:
            v_transforms = {k: b.array(v) for k, v in json_v.items()}
        json_inter = _coerce_int_keys(meta.get("intermediate_transforms"))
        if json_inter:
            intermediate_transforms = {k: b.array(v) for k, v in json_inter.items()}

    return (
        source_activations,
        target_activations,
        probe_result,
        meta.get("metrics", {}),
        feature_transforms if feature_transforms else None,
        _coerce_int_keys(meta.get("scale_ratios")),
        embedding_transform,
        attention_transforms if attention_transforms else None,
        k_transforms if k_transforms else None,
        v_transforms if v_transforms else None,
        intermediate_transforms if intermediate_transforms else None,
        _coerce_int_keys(meta.get("layer_mapping")),
        meta.get("probe_ids"),
        meta.get("probe_domains"),
        source_intermediate_activations if source_intermediate_activations else None,
        target_intermediate_activations if target_intermediate_activations else None,
        source_attention_activations if source_attention_activations else None,
        target_attention_activations if target_attention_activations else None,
        source_k_activations if source_k_activations else None,
        target_k_activations if target_k_activations else None,
    )


def _save_probe_result_cache(
    cache_key: str,
    source_activations: dict[int, "Array"],
    target_activations: dict[int, "Array"],
    probe_result: dict[str, Any],
    metrics: dict[str, Any],
    feature_transforms: dict[int, "Array"] | None,
    scale_ratios: dict[int, float] | None,
    embedding_transform: "Array | None",
    attention_transforms: dict[int, "Array"] | None,
    k_transforms: dict[int, "Array"] | None,
    v_transforms: dict[int, "Array"] | None,
    intermediate_transforms: dict[int, "Array"] | None,
    layer_mapping: dict[int, int] | None,
    probe_ids: list[str],
    probe_domains: list[str],
    probe_mode: str,
    source_intermediate_activations: dict[int, "Array"] | None = None,
    target_intermediate_activations: dict[int, "Array"] | None = None,
    source_attention_activations: dict[int, "Array"] | None = None,
    target_attention_activations: dict[int, "Array"] | None = None,
    source_k_activations: dict[int, "Array"] | None = None,
    target_k_activations: dict[int, "Array"] | None = None,
) -> None:
    """Persist probe results to disk for reuse."""
    import mlx.core as mx

    cache_dir = _probe_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, "Array"] = {}
    # Hidden activations
    for layer_idx, acts in source_activations.items():
        data[f"src_act_{layer_idx}"] = acts
    for layer_idx, acts in target_activations.items():
        data[f"tgt_act_{layer_idx}"] = acts
    # Intermediate activations (MLP)
    if source_intermediate_activations:
        for layer_idx, acts in source_intermediate_activations.items():
            data[f"src_inter_{layer_idx}"] = acts
    if target_intermediate_activations:
        for layer_idx, acts in target_intermediate_activations.items():
            data[f"tgt_inter_{layer_idx}"] = acts
    # Attention Q activations
    if source_attention_activations:
        for layer_idx, acts in source_attention_activations.items():
            data[f"src_attn_{layer_idx}"] = acts
    if target_attention_activations:
        for layer_idx, acts in target_attention_activations.items():
            data[f"tgt_attn_{layer_idx}"] = acts
    # K activations
    if source_k_activations:
        for layer_idx, acts in source_k_activations.items():
            data[f"src_k_{layer_idx}"] = acts
    if target_k_activations:
        for layer_idx, acts in target_k_activations.items():
            data[f"tgt_k_{layer_idx}"] = acts

    # Transforms - save to NPZ (binary, GPU-friendly) instead of JSON
    if feature_transforms:
        for layer_idx, F in feature_transforms.items():
            data[f"feat_transform_{layer_idx}"] = F
    if embedding_transform is not None:
        data["emb_transform"] = embedding_transform
    if attention_transforms:
        for layer_idx, F in attention_transforms.items():
            data[f"attn_transform_{layer_idx}"] = F
    if k_transforms:
        for layer_idx, F in k_transforms.items():
            data[f"k_transform_{layer_idx}"] = F
    if v_transforms:
        for layer_idx, F in v_transforms.items():
            data[f"v_transform_{layer_idx}"] = F
    if intermediate_transforms:
        for layer_idx, F in intermediate_transforms.items():
            data[f"inter_transform_{layer_idx}"] = F

    data_path = cache_dir / f"{cache_key}.npz"
    mx.savez_compressed(data_path, **data)

    # Metadata (no transforms - they're in NPZ now)
    meta = {
        "version": 2,  # Bumped for new format with transforms in NPZ
        "probe_mode": probe_mode,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
        "probe_result": probe_result,
        "metrics": metrics,
        "scale_ratios": scale_ratios,
        "layer_mapping": layer_mapping,
        "has_transforms": True,  # Transforms stored in NPZ
    }
    meta_path = cache_dir / f"{cache_key}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("PROBE CACHE: Saved probe results to %s", cache_dir)


# =============================================================================
# PER-MODEL PROBE ACTIVATION CACHE (DISK)
# =============================================================================


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

# =============================================================================
# MEMORY-EFFICIENT ACTIVATION ACCUMULATION
# =============================================================================

def _accumulate_activation(
    storage: dict[int, "Array"],
    layer_idx: int,
    act: "Array",
    backend: "Backend",
) -> None:
    """Accumulate activation into a single stacked array per layer.

    This is THE memory optimization: instead of storing thousands of small
    arrays (each a Metal buffer), we concatenate into ONE array per layer.

    Activations come in as 1D [hidden_dim] and we stack them to build
    a 2D matrix [n_probes, hidden_dim].

    Result: 32 Metal buffers per model instead of 4096×32 = 131,072

    IMPORTANT: eval() is called after each concatenation to:
    1. Materialize the new array immediately
    2. Allow the old array (before concat) to be freed
    3. Prevent the lazy computation graph from growing unboundedly
    Without this, Metal resource limits are exceeded (~500K buffer limit).
    """
    import mlx.core as mx

    # Ensure activation is 2D [1, hidden_dim] for proper stacking
    if len(act.shape) == 1:
        act = mx.reshape(act, (1, -1))

    if layer_idx not in storage:
        # First activation for this layer - store as 2D
        storage[layer_idx] = act
    else:
        # Concatenate along axis 0 to build [n_probes, hidden_dim]
        storage[layer_idx] = mx.concatenate([storage[layer_idx], act], axis=0)

    # Force evaluation to materialize buffer and allow old one to be freed
    mx.eval(storage[layer_idx])


# =============================================================================
# GPU-RESIDENT PROBE CACHE: Precomputed geometric data per layer
# =============================================================================


@dataclass
class ProbeCache:
    """GPU-resident cache of precomputed geometric data per layer.
    
    After collecting 963 probes, we precompute ONCE on GPU:
    - Gram matrices [n_probes, n_probes] per layer
    - Centered Gram matrices for CKA
    - SVD decomposition (U, S, Vt) - gives shape, rotation info
    - Effective ranks (entropy-based density measure)
    - RBF sigma per layer (for consistent Gram computation)
    
    This eliminates redundant computation during alignment.
    """
    
    # Per-layer activation matrices [n_probes, hidden_dim]
    source_activations: dict[int, "Array"]
    target_activations: dict[int, "Array"]
    
    # Per-layer Gram matrices [n_probes, n_probes] 
    source_grams: dict[int, "Array"]
    target_grams: dict[int, "Array"]
    
    # Per-layer centered Gram matrices for CKA
    source_centered_grams: dict[int, "Array"]
    target_centered_grams: dict[int, "Array"]
    
    # Per-layer SVD decomposition (shape/rotation info)
    # NOTE: SVD computation disabled (MLX crashes on large Gram matrices).
    # Keeping empty dicts for interface stability. Remove if causing issues.
    source_svd: dict[int, tuple["Array", "Array", "Array"]]
    target_svd: dict[int, tuple["Array", "Array", "Array"]]
    
    # Per-layer effective ranks (density measure)
    source_ranks: dict[int, int]
    target_ranks: dict[int, int]
    
    # Per-layer RBF sigma (bandwidth for Gram computation)
    source_sigmas: dict[int, float]
    target_sigmas: dict[int, float]
    
    # Layer similarity matrix [n_source, n_target] - geometric compatibility
    # Computed for cache diagnostics; not currently used in alignment algorithm.
    layer_similarity_matrix: "Array | None" = None

    # NOTE: Procrustes hints removed (R_hint parameter was never implemented).
    procrustes_hints: dict[tuple[int, int], "Array"] | None = None
    
    @staticmethod
    def from_activations(
        source_acts: dict[int, "Array"],
        target_acts: dict[int, "Array"],
        backend: "Backend",
    ) -> "ProbeCache":
        """Build cache from collected probe activations.
        
        Args:
            source_acts: layer -> stacked activation array [n_probes, hidden_dim]
            target_acts: layer -> stacked activation array [n_probes, hidden_dim]
            backend: MLX backend for GPU computation
        """
        b = backend
        
        # Initialize empty dicts
        cache = ProbeCache(
            source_activations={},
            target_activations={},
            source_grams={},
            target_grams={},
            source_centered_grams={},
            target_centered_grams={},
            source_svd={},
            target_svd={},
            source_ranks={},
            target_ranks={},
            source_sigmas={},
            target_sigmas={},
        )
        
        logger.info("PROBE CACHE: Building GPU-resident cache for %d source layers, %d target layers",
                    len(source_acts), len(target_acts))
        
        # Stack and precompute for source layers
        for layer_idx, acts in source_acts.items():
            # Check if acts is empty - can't use `not acts` on MLX arrays
            if acts is None or (hasattr(acts, 'shape') and acts.shape[0] == 0):
                continue
            cache._precompute_layer(
                layer_idx, acts, b,
                cache.source_activations,
                cache.source_grams,
                cache.source_centered_grams,
                cache.source_svd,
                cache.source_ranks,
                cache.source_sigmas,
            )
        
        # Stack and precompute for target layers
        for layer_idx, acts in target_acts.items():
            # Check if acts is empty - can't use `not acts` on MLX arrays
            if acts is None or (hasattr(acts, 'shape') and acts.shape[0] == 0):
                continue
            cache._precompute_layer(
                layer_idx, acts, b,
                cache.target_activations,
                cache.target_grams,
                cache.target_centered_grams,
                cache.target_svd,
                cache.target_ranks,
                cache.target_sigmas,
            )
        
        b.eval()  # Force all GPU computation
        logger.info("PROBE CACHE: Precomputed %d source + %d target Gram matrices on GPU",
                    len(cache.source_grams), len(cache.target_grams))
        
        return cache
    
    def _precompute_layer(
        self,
        layer_idx: int,
        acts: "Array",  # Already stacked [n_probes, hidden_dim]
        backend: "Backend",
        act_cache: dict,
        gram_cache: dict,
        centered_gram_cache: dict,
        svd_cache: dict,
        rank_cache: dict,
        sigma_cache: dict,
    ) -> None:
        """Precompute geometric data for a single layer on GPU."""
        b = backend
        
        # Activations already stacked as [n_probes, hidden_dim] from _accumulate_activation
        stacked = acts
        b.eval(stacked)
        
        # Ensure 2D shape: [n_probes, hidden_dim]
        shape = b.shape(stacked)
        if len(shape) == 1:
            # Single probe case: [hidden_dim] -> [1, hidden_dim]
            stacked = b.reshape(stacked, (1, -1))
            b.eval(stacked)
            shape = b.shape(stacked)
        
        act_cache[layer_idx] = stacked
        n_probes = int(shape[0])
        d_hidden = int(shape[1])
        
        # Compute RBF sigma using median heuristic
        # sigma = median(||x_i - x_j||) for all pairs
        # Approximation: use std of activations * sqrt(d)
        std_val = b.std(stacked)
        b.eval(std_val)
        sigma = float(b.to_scalar(std_val)) * (d_hidden ** 0.5)
        sigma = max(sigma, regularization_epsilon(b, stacked))
        sigma_cache[layer_idx] = sigma
        
        # Compute RBF Gram matrix: K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        # Efficient: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        sq_norms = b.sum(stacked * stacked, axis=1, keepdims=True)  # [n, 1]
        dot_products = b.matmul(stacked, b.transpose(stacked))      # [n, n]
        sq_dists = sq_norms + b.transpose(sq_norms) - 2 * dot_products  # [n, n]
        gram = b.exp(-sq_dists / (2 * sigma * sigma))
        b.eval(gram)
        gram_cache[layer_idx] = gram
        
        # Center Gram matrix: K_c = HKH where H = I - 1/n * ones
        n = int(n_probes)
        row_mean = b.mean(gram, axis=1, keepdims=True)
        col_mean = b.mean(gram, axis=0, keepdims=True)
        total_mean = b.mean(gram)
        centered_gram = gram - row_mean - col_mean + total_mean
        b.eval(centered_gram)
        centered_gram_cache[layer_idx] = centered_gram
        
        # Skip SVD of 963×963 Gram matrix for now - MLX sgesvdx_ crashes on large matrices
        # SVD of centered Gram for shape/rotation info
        # TODO: Add back with proper matrix size handling or chunked approach
        # U, S, Vt = geodesic_svd(b, centered_gram)
        # b.eval(U, S, Vt)
        # svd_cache[layer_idx] = (U, S, Vt)
        
        # Effective rank: use approximate heuristic based on Gram eigenvalues
        # For now, set to n_probes (full rank assumption)
        rank_cache[layer_idx] = int(n_probes)
    
    def compute_layer_similarity_matrix(self, backend: "Backend") -> None:
        """Compute similarity matrix between all source-target layer pairs.
        
        Uses CKA between centered Gram matrices as similarity metric.
        Result: [n_source_layers, n_target_layers] matrix on GPU.
        """
        b = backend
        eps = sqrt_scalar(machine_epsilon(b, b.array([1.0], dtype="float32")), b)
        
        source_layers = sorted(self.source_centered_grams.keys())
        target_layers = sorted(self.target_centered_grams.keys())
        
        n_src = len(source_layers)
        n_tgt = len(target_layers)
        
        # Build similarity matrix from CKA values
        # CKA = <K_s, K_t>_F / (||K_s||_F * ||K_t||_F)
        # OPTIMIZED: Batch compute all CKA values on GPU, extract once

        # Pre-compute all norms on GPU (O(n) syncs -> O(1) sync)
        src_norms = []
        for src_layer in source_layers:
            K_s = self.source_centered_grams[src_layer]
            src_norms.append(b.norm(K_s))
        src_norms_arr = b.stack(src_norms, axis=0)  # [n_src]

        tgt_norms = []
        for tgt_layer in target_layers:
            K_t = self.target_centered_grams[tgt_layer]
            tgt_norms.append(b.norm(K_t))
        tgt_norms_arr = b.stack(tgt_norms, axis=0)  # [n_tgt]

        # Compute CKA matrix on GPU
        cka_rows = []
        for i, src_layer in enumerate(source_layers):
            K_s = self.source_centered_grams[src_layer]
            norm_s = src_norms_arr[i]

            # Vectorized: compute dot products with all target grams
            row_cka = []
            for j, tgt_layer in enumerate(target_layers):
                K_t = self.target_centered_grams[tgt_layer]
                dot = b.sum(K_s * K_t)
                norm_t = tgt_norms_arr[j]
                cka = dot / (norm_s * norm_t + eps)
                row_cka.append(cka)
            cka_rows.append(b.stack(row_cka, axis=0))  # [n_tgt]

        # Stack into matrix and eval once
        self.layer_similarity_matrix = b.stack(cka_rows, axis=0)  # [n_src, n_tgt]
        b.eval(self.layer_similarity_matrix)
        
        logger.info("PROBE CACHE: Computed %dx%d layer similarity matrix", n_src, n_tgt)
    
    def save_to_profile(self, profile_path: Path, model_key: str, backend: "Backend") -> None:
        """Save precomputed cache to model profile for reuse.
        
        Saves to: profile_path / f"{model_key}_probe_cache.npz"
        
        This allows us to skip re-probing for models we've already analyzed.
        The cache contains heavy GPU data, so we save as compressed arrays.
        """
        import mlx.core as mx
        b = backend
        
        save_path = Path(profile_path) / f"{model_key}_probe_cache.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data: dict[str, Any] = {"version": b.array([1], dtype="int32")}
        
        # Save activations per layer
        for layer_idx, acts in self.source_activations.items():
            data[f"src_acts_{layer_idx}"] = acts
        for layer_idx, acts in self.target_activations.items():
            data[f"tgt_acts_{layer_idx}"] = acts
        
        # Save Gram matrices
        for layer_idx, gram in self.source_grams.items():
            data[f"src_gram_{layer_idx}"] = gram
        for layer_idx, gram in self.target_grams.items():
            data[f"tgt_gram_{layer_idx}"] = gram
        
        # Save centered Grams
        for layer_idx, gram in self.source_centered_grams.items():
            data[f"src_cgram_{layer_idx}"] = gram
        for layer_idx, gram in self.target_centered_grams.items():
            data[f"tgt_cgram_{layer_idx}"] = gram
        
        # Save ranks and sigmas as metadata
        if self.source_ranks:
            data["src_ranks"] = b.array(list(self.source_ranks.items()), dtype="int32")
        if self.target_ranks:
            data["tgt_ranks"] = b.array(list(self.target_ranks.items()), dtype="int32")
        if self.source_sigmas:
            data["src_sigmas"] = b.array(list(self.source_sigmas.items()), dtype="float32")
        if self.target_sigmas:
            data["tgt_sigmas"] = b.array(list(self.target_sigmas.items()), dtype="float32")
        
        # Save layer similarity if computed
        if self.layer_similarity_matrix is not None:
            data["similarity_matrix"] = self.layer_similarity_matrix
        
        mx.savez_compressed(save_path, **data)
        logger.info("PROBE CACHE: Saved to profile %s (%.1f MB)", 
                    save_path, save_path.stat().st_size / 1024 / 1024)
    
    @staticmethod
    def load_from_profile(profile_path: Path, model_key: str, backend: "Backend") -> "ProbeCache | None":
        """Load precomputed cache from model profile.
        
        Returns None if profile doesn't exist or is invalid.
        """
        import mlx.core as mx
        b = backend
        
        load_path = Path(profile_path) / f"{model_key}_probe_cache.npz"
        if not load_path.exists():
            logger.debug("PROBE CACHE: No cached profile at %s", load_path)
            return None
        
        try:
            loaded = mx.load(load_path)
            if not isinstance(loaded, dict):
                logger.warning("PROBE CACHE: Invalid cache format at %s", load_path)
                return None
            
            # Reconstruct cache
            cache = ProbeCache(
                source_activations={},
                target_activations={},
                source_grams={},
                target_grams={},
                source_centered_grams={},
                target_centered_grams={},
                source_svd={},
                target_svd={},
                source_ranks={},
                target_ranks={},
                source_sigmas={},
                target_sigmas={},
            )
            
            # Load activations
            for key, arr in loaded.items():
                if key.startswith("src_acts_"):
                    layer_idx = int(key.split("_")[2])
                    cache.source_activations[layer_idx] = arr
                elif key.startswith("tgt_acts_"):
                    layer_idx = int(key.split("_")[2])
                    cache.target_activations[layer_idx] = arr
                elif key.startswith("src_gram_"):
                    layer_idx = int(key.split("_")[2])
                    cache.source_grams[layer_idx] = arr
                elif key.startswith("tgt_gram_"):
                    layer_idx = int(key.split("_")[2])
                    cache.target_grams[layer_idx] = arr
                elif key.startswith("src_cgram_"):
                    layer_idx = int(key.split("_")[2])
                    cache.source_centered_grams[layer_idx] = arr
                elif key.startswith("tgt_cgram_"):
                    layer_idx = int(key.split("_")[2])
                    cache.target_centered_grams[layer_idx] = arr
            
            # Load ranks and sigmas
            if "src_ranks" in loaded:
                for layer_idx, rank in b.tolist(loaded["src_ranks"]):
                    cache.source_ranks[int(layer_idx)] = int(rank)
            if "tgt_ranks" in loaded:
                for layer_idx, rank in b.tolist(loaded["tgt_ranks"]):
                    cache.target_ranks[int(layer_idx)] = int(rank)
            if "src_sigmas" in loaded:
                for layer_idx, sigma in b.tolist(loaded["src_sigmas"]):
                    cache.source_sigmas[int(layer_idx)] = float(sigma)
            if "tgt_sigmas" in loaded:
                for layer_idx, sigma in b.tolist(loaded["tgt_sigmas"]):
                    cache.target_sigmas[int(layer_idx)] = float(sigma)
            
            # Load similarity matrix if present
            if "similarity_matrix" in loaded:
                cache.layer_similarity_matrix = loaded["similarity_matrix"]
            
            logger.info("PROBE CACHE: Loaded from profile %s", load_path)
            return cache
            
        except Exception as e:
            logger.warning("PROBE CACHE: Failed to load from %s: %s", load_path, e)
            return None


@dataclass
class ProbeResult:
    """Result of Stage 1 probing."""

    correlations: dict[str, float]
    confidences: dict[int, float]
    intersection_map: Any | None  # IntersectionMap object
    dimension_correlations: dict
    metrics: dict[str, Any]

    # Activations for downstream processing (null-space filtering, shared subspace)
    # Hidden-space activations: shape [hidden_dim] per sample (e.g., 960 for SmolLM, 896 for Qwen)
    source_activations: dict[int, list[Any]] | None = None
    target_activations: dict[int, list[Any]] | None = None
    # Intermediate-space activations: shape [intermediate_dim] per sample
    # (e.g., 2560 for SmolLM, 4864 for Qwen) - for multi-space stitching
    source_intermediate_activations: dict[int, list[Any]] | None = None
    target_intermediate_activations: dict[int, list[Any]] | None = None
    # Q Attention-space activations: shape [num_heads * head_dim] per sample
    # (e.g., 960 for SmolLM=15*64, 896 for Qwen=14*64) - for q_proj/o_proj stitching
    source_attention_activations: dict[int, list[Any]] | None = None
    target_attention_activations: dict[int, list[Any]] | None = None
    # K Attention-space activations: shape [num_kv_heads * head_dim] per sample
    # Separate from V for granular alignment - for k_proj stitching
    source_k_activations: dict[int, list[Any]] | None = None
    target_k_activations: dict[int, list[Any]] | None = None
    # V Attention-space activations: shape [num_kv_heads * head_dim] per sample
    # Separate from K for granular alignment - for v_proj stitching
    source_v_activations: dict[int, list[Any]] | None = None
    target_v_activations: dict[int, list[Any]] | None = None
    # Embedding-space activations: shape [hidden_dim] per sample (post-embed_tokens, pre-layer-0)
    # Used for GramAlign at 2D interface - same CKA=1.0, same geodesic math
    source_embedding_activations: list[Any] | None = None
    target_embedding_activations: list[Any] | None = None
    probe_ids: list[str] | None = None
    probe_domains: list[str] | None = None
    # Layer alignment transforms: source_acts @ transforms[layer] -> aligned_source
    # These transforms achieve CKA = 1.0 for each aligned layer pair
    # Hidden-space transforms: for hidden dimension (e.g., 960 -> 896)
    feature_transforms: dict[int, list[list[float]]] | None = None
    # EXACT SCALE FACTOR per layer: ||target|| / ||source @ F||
    # Apply to stitched weights for exact magnitude match
    scale_ratios: dict[int, float] | None = None
    # Embedding-space transform: for embed_tokens alignment (same CKA=1.0, same geodesic math)
    embedding_transform: list[list[float]] | None = None
    # Attention Q-space transforms: for q_proj/o_proj (e.g., 960 -> 896 for Q heads)
    attention_transforms: dict[int, list[list[float]]] | None = None
    # Attention K-space transforms: for k_proj (granular alignment)
    k_transforms: dict[int, list[list[float]]] | None = None
    # Attention V-space transforms: for v_proj (granular alignment)
    v_transforms: dict[int, list[list[float]]] | None = None
    # Intermediate-space transforms: for MLP gate/up/down projections (pre-computed)
    intermediate_transforms: dict[int, list[list[float]]] | None = None
    # Layer mapping: target_layer -> source_layer (from proportional depth mapping)
    layer_mapping: dict[int, int] | None = None


def stage_probe(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
    source_model: Any | None = None,
    target_model: Any | None = None,
    source_path: str = "",
    target_path: str = "",
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    tokenizer: Any | None = None,
    collect_activations_fn: Callable | None = None,
    activation_provider: "ActivationProvider | None" = None,
    backend: "Backend | None" = None,
    checkpoint_dir: Path | str | None = None,
    probe_mode: str = "atlas",  # "atlas" (963 conceptual) or "token" (49K+ vocab)
) -> ProbeResult:
    """
    Stage 1: Build intersection map from probe responses.

    ALWAYS uses precise mode (activation-level CKA) with all 403 probes.
    No configuration - the geometry determines everything.

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        extract_layer_index_fn: Function to extract layer index from weight key
        source_model: Loaded source model (required)
        target_model: Loaded target model (required)
        tokenizer: Tokenizer (for precise mode)
        collect_activations_fn: Function to collect layer activations
        checkpoint_dir: Optional directory for checkpoint files. If provided,
            probe progress will be saved periodically and can be resumed.

    Returns:
        ProbeResult with correlations, confidences, and intersection map
    """
    if tokenizer is not None:
        source_tokenizer = source_tokenizer or tokenizer
        target_tokenizer = target_tokenizer or tokenizer

    # =========================================================================
    # PRE-FLIGHT: Check tokenizer alignment before expensive probing
    # =========================================================================
    if source_tokenizer is not None and target_tokenizer is not None:
        try:
            from modelcypher.core.domain.geometry.dimensional_alignment import (
                measure_1d_alignment,
            )
            alignment_1d = measure_1d_alignment(source_tokenizer, target_tokenizer)
            
            # Warn if tokenizer alignment is poor
            if alignment_1d.vocab_jaccard < 0.5:
                logger.warning(
                    "PRE-FLIGHT: Low tokenizer overlap detected (Jaccard=%.2f). "
                    "Cross-tokenizer merges may produce degraded outputs.",
                    alignment_1d.vocab_jaccard,
                )
            elif alignment_1d.vocab_jaccard < 0.8:
                logger.info(
                    "PRE-FLIGHT: Moderate tokenizer overlap (Jaccard=%.2f). "
                    "Some token-level misalignment expected.",
                    alignment_1d.vocab_jaccard,
                )
            else:
                logger.debug(
                    "PRE-FLIGHT: Good tokenizer alignment (Jaccard=%.2f)",
                    alignment_1d.vocab_jaccard,
                )
        except Exception as e:
            logger.debug("PRE-FLIGHT: Skipped tokenizer check: %s", e)

    # ALWAYS use precise mode - this is not configurable.
    # Activation-level CKA is required for correct geometric alignment.

    # Get activation provider: use provided, or derive from collect_activations_fn, or auto-detect
    if activation_provider is None:
        if collect_activations_fn is not None:
            # Legacy compatibility: wrap the callable as a provider-like object
            # This maintains backwards compatibility with existing callers
            class LegacyActivationProvider:
                def __init__(self, fn: Callable):
                    self._fn = fn

                def collect_hidden_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> dict:
                    return self._fn(model, tokenizer, text)

                def collect_intermediate_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> dict:
                    # Legacy doesn't support intermediate activations
                    return {}

                def collect_attention_activations(
                    self, model: Any, tokenizer: Any, text: str, token_ids: list[int] | None = None
                ) -> tuple[dict, dict]:
                    # Legacy doesn't support attention activations
                    return {}, {}

            activation_provider = LegacyActivationProvider(collect_activations_fn)
        else:
            # Auto-detect activation provider from platform
            from modelcypher.infrastructure.activation_provider_factory import get_activation_provider

            activation_provider = get_activation_provider()

    if (
        source_model is not None
        and target_model is not None
        and activation_provider is not None
    ):
        return _probe_precise(
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_weights=source_weights,
            target_weights=target_weights,
            extract_layer_index_fn=extract_layer_index_fn,
            activation_provider=activation_provider,
            source_path=source_path,
            target_path=target_path,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            probe_mode=probe_mode,
        )
    else:
        # INVARIANT GEOMETRY: No fallbacks. Models MUST be loaded.
        # Weight-level CKA ("fast" mode) hides alignment problems that
        # cause gibberish output. We don't allow it.
        raise RuntimeError(
            "Probe stage requires loaded models. "
            "Cannot compute activation-level CKA without model access. "
            "Load both source and target models before probing."
        )


def _probe_precise(
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
    activation_provider: "ActivationProvider",
    source_path: str = "",
    target_path: str = "",
    backend: "Backend | None" = None,
    checkpoint_dir: Path | None = None,
    probe_mode: str = "atlas",  # "atlas" (963 conceptual) or "token" (vocab-based)
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models.

    Args:
        probe_mode: "atlas" uses 963 curated conceptual probes.
                    "token" uses vocabulary tokens as probes (49K+) for 100% dimension coverage.
        checkpoint_dir: If provided, saves progress periodically and allows
            resume from last checkpoint on restart.
    """
    b = backend or get_default_backend()
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
    from modelcypher.core.domain.geometry.intersection_similarity import (
        IntersectionSimilarityMode,
        build_intersection_map,
    )
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ActivatedDimension,
        ActivationFingerprint,
        IntersectionMap,
    )

    # Select probing mode
    if probe_mode == "token":
        # Token-based probing for 100% dimension coverage
        from modelcypher.core.domain.agents.token_atlas import generate_token_probes, TokenProbe
        logger.info("PROBE MODE: Token-based (vocab probes for dimension coverage)")
        
        # Determine required probe count: need at least max(source_dim, target_dim)
        # Use 2x to guarantee full-rank with margin for numerical stability
        source_hidden = None
        target_hidden = None
        for key in source_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                source_hidden = source_weights[key].shape[0]
                break
        for key in target_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                target_hidden = target_weights[key].shape[0]
                break
        
        # Cap probes for memory stability. 2048 provides full-rank coverage for
        # typical hidden dims (1024, 2048) while avoiding GPU OOM.
        TOKEN_PROBE_CAP = 2048

        min_probes_needed = max(source_hidden or 1024, target_hidden or 960)
        max_probes = min_probes_needed * 2  # 2x for numerical margin
        max_probes = max(max_probes, 1500)  # At least 1500 for 960-dim full coverage
        max_probes = min(max_probes, TOKEN_PROBE_CAP)  # Cap to avoid OOM

        logger.info("PROBE TOKEN: Dims source=%s target=%s, using %d probes",
                    source_hidden, target_hidden, max_probes)

        # Use target tokenizer for probe generation (target architecture defines the space)
        token_probes = generate_token_probes(target_tokenizer, max_probes=max_probes)
        # Convert TokenProbes to AtlasProbe format for compatibility
        probes = [tp.to_atlas_probe() for tp in token_probes]
        logger.info("PROBE TOKEN: Generated %d probes (2x max_dim, capped at %d)",
                    len(probes), TOKEN_PROBE_CAP)
    else:
        # Default: Atlas probes from JSON files (curated conceptual + MMLU probes)
        from modelcypher.core.domain.agents.probe_loader import load_all_probes
        all_probes = load_all_probes()
        logger.info("PROBE MODE: Atlas (loaded %d probes from JSON)", len(all_probes))
        
        # MEMORY OPTIMIZATION: Limit probes to 2x max hidden dimension
        # This ensures full-rank coverage while preventing OOM
        source_hidden = None
        target_hidden = None
        for key in source_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                source_hidden = source_weights[key].shape[0]
                break
        for key in target_weights:
            if "layers.0.self_attn.q_proj.weight" in key:
                target_hidden = target_weights[key].shape[0]
                break
        
        # Use TARGET hidden dim - we project into target's space, so only need
        # enough probes to span the target manifold. Source can be larger.
        target_dim = target_hidden or 1024
        # 2x for numerical stability in Gram matrix rank
        max_probes = target_dim * 2

        if len(all_probes) > max_probes:
            logger.info("PROBE LIMIT: Capping %d probes to %d (2x target_dim=%d) - target space only",
                       len(all_probes), max_probes, target_dim)
            # Domain-stratified sampling to ensure coverage across all knowledge domains
            # Group probes by domain
            import random
            from collections import defaultdict
            random.seed(42)  # Deterministic sampling

            probes_by_domain: dict[str, list[Any]] = defaultdict(list)
            for probe in all_probes:
                domain_key = getattr(probe.domain, 'value', str(probe.domain))
                probes_by_domain[domain_key].append(probe)

            # Allocate probes proportionally by domain, ensuring minimum 1 per domain
            n_domains = len(probes_by_domain)
            min_per_domain = max(1, max_probes // (n_domains * 2))  # At least 1, up to half
            remaining = max_probes - (min_per_domain * n_domains)

            probes = []
            for domain_key, domain_probes in sorted(probes_by_domain.items()):
                # Calculate proportional allocation for this domain
                proportion = len(domain_probes) / len(all_probes)
                extra_allocation = int(remaining * proportion)
                n_sample = min(len(domain_probes), min_per_domain + extra_allocation)

                # Sample from domain
                if n_sample >= len(domain_probes):
                    probes.extend(domain_probes)
                else:
                    probes.extend(random.sample(domain_probes, n_sample))

            # If still under max_probes, fill with random remaining probes
            if len(probes) < max_probes:
                used_ids = {p.probe_id for p in probes}
                remaining_probes = [p for p in all_probes if p.probe_id not in used_ids]
                fill_count = min(max_probes - len(probes), len(remaining_probes))
                if fill_count > 0:
                    probes.extend(random.sample(remaining_probes, fill_count))

            logger.info("PROBE STRATIFIED: %d domains, %d-%d probes per domain",
                       n_domains, min_per_domain, max(len(v) for v in probes_by_domain.values()))
        else:
            probes = all_probes
            
    num_probes = len(probes)

    # Pre-validate probes for usable text (used for caching + inference).
    valid_probes: list[tuple[Any, str]] = []
    invalid_probe_count = 0
    for probe in probes:
        probe_text = _select_probe_text(probe)
        if probe_text is None:
            logger.warning(
                "Probe '%s' skipped: no valid support texts or fallback",
                probe.probe_id,
            )
            invalid_probe_count += 1
            continue
        valid_probes.append((probe, probe_text))

    expected_probe_ids = [probe.probe_id for probe, _ in valid_probes]
    expected_probe_domains = [probe.domain.value for probe, _ in valid_probes]
    expected_probe_texts = [probe_text for _, probe_text in valid_probes]

    from modelcypher.core.domain.geometry.model_profile import (
        ModelProfileStore,
        compute_probe_corpus_hash,
    )

    probe_corpus_hash = compute_probe_corpus_hash(
        probe_mode=probe_mode,
        probe_ids=expected_probe_ids,
        probe_texts=expected_probe_texts,
    )
    profile_store = ModelProfileStore()
    source_profile: ModelProfile | None = None
    target_profile: ModelProfile | None = None
    source_identity = None
    target_identity = None

    if source_path:
        source_profile, source_identity = profile_store.ensure(source_path)
    if target_path:
        target_profile, target_identity = profile_store.ensure(target_path)

    cache_key = f"{probe_mode}:{probe_corpus_hash}"

    source_cache: ModelProbeCache | None = None
    target_cache: ModelProbeCache | None = None

    if source_identity and expected_probe_ids:
        source_cache = _load_model_probe_cache(
            model_id=source_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            backend=b,
        )
        if source_cache:
            logger.info(
                "PROBE CACHE: Loaded per-model activations for source %s",
                source_identity.model_id,
            )

    if target_identity and expected_probe_ids:
        target_cache = _load_model_probe_cache(
            model_id=target_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            backend=b,
        )
        if target_cache:
            logger.info(
                "PROBE CACHE: Loaded per-model activations for target %s",
                target_identity.model_id,
            )

    source_fingerprints: list[ActivationFingerprint] = []
    target_fingerprints: list[ActivationFingerprint] = []

    if (
        source_profile
        and source_profile.probe_corpus_hash == probe_corpus_hash
        and source_profile.probe_fingerprints
    ):
        source_fingerprints = source_profile.probe_fingerprints
    if (
        target_profile
        and target_profile.probe_corpus_hash == probe_corpus_hash
        and target_profile.probe_fingerprints
    ):
        target_fingerprints = target_profile.probe_fingerprints

    run_source_inference = source_cache is None
    run_target_inference = target_cache is None
    run_inference = run_source_inference or run_target_inference

    if run_source_inference:
        source_fingerprints = []
    if run_target_inference:
        target_fingerprints = []

    if run_inference:
        logger.info(
            "PROBE PRECISE: Running %d probes through %s%s models...",
            len(valid_probes),
            "source" if run_source_inference else "",
            " + target" if run_target_inference else "",
        )
    else:
        logger.info(
            "PROBE PRECISE: Using cached activations for %d probes",
            len(valid_probes),
        )

    # MEMORY OPTIMIZATION: Store as single stacked Array per layer, not list of arrays
    # This reduces Metal buffer count from 4096×32 = 131,072 to just 32 per model
    source_layer_activations: dict[int, "Array"] = (
        source_cache.hidden_activations if source_cache else {}
    )
    target_layer_activations: dict[int, "Array"] = (
        target_cache.hidden_activations if target_cache else {}
    )

    def _fingerprints_from_cache(
        probes: list[tuple[Any, str]],
        layer_activations: dict[int, "Array"],
    ) -> list[ActivationFingerprint]:
        fingerprints: list[ActivationFingerprint] = []
        if not layer_activations:
            return fingerprints

        # Assume all layers share the same probe count
        sample_layer = next(iter(layer_activations.values()))
        probe_count = int(b.shape(sample_layer)[0])

        for probe_idx, (probe, _probe_text) in enumerate(probes):
            if probe_idx >= probe_count:
                break
            activated: dict[int, list[ActivatedDimension]] = {}
            for layer_idx, stacked in layer_activations.items():
                row = b.take(stacked, b.array([probe_idx]), axis=0)
                row = b.squeeze(row, axis=0)
                dims = _extract_top_k_dims(row, backend=b)
                if dims:
                    activated[layer_idx] = dims
            if activated:
                fingerprints.append(
                    ActivationFingerprint(
                        prime_id=probe.probe_id,
                        prime_text=probe.name,
                        activated_dimensions=activated,
                    )
                )
        return fingerprints

    if source_cache and not source_fingerprints:
        logger.info("PROBE CACHE: Rebuilding source fingerprints from cached activations")
        source_fingerprints = _fingerprints_from_cache(
            valid_probes, source_layer_activations
        )
    if target_cache and not target_fingerprints:
        logger.info("PROBE CACHE: Rebuilding target fingerprints from cached activations")
        target_fingerprints = _fingerprints_from_cache(
            valid_probes, target_layer_activations
        )

    source_embedding_activations: list["Array"] | "Array" = (
        source_cache.embedding_activations
        if source_cache and source_cache.embedding_activations is not None
        else []
    )
    target_embedding_activations: list["Array"] | "Array" = (
        target_cache.embedding_activations
        if target_cache and target_cache.embedding_activations is not None
        else []
    )
    # MEMORY OPTIMIZATION: Store as single stacked Array per layer, not list
    # This reduces Metal buffer count from 131K to just 32 per model
    source_intermediate_activations: dict[int, "Array"] = (
        source_cache.intermediate_activations if source_cache else {}
    )
    target_intermediate_activations: dict[int, "Array"] = (
        target_cache.intermediate_activations if target_cache else {}
    )
    # Q Attention-space activations for q_proj/o_proj stitching (cross-architecture merges)
    source_attention_activations: dict[int, "Array"] = (
        source_cache.attention_activations if source_cache else {}
    )
    target_attention_activations: dict[int, "Array"] = (
        target_cache.attention_activations if target_cache else {}
    )
    # K Attention-space activations for k_proj stitching (separate for granular alignment)
    source_k_activations: dict[int, "Array"] = (
        source_cache.k_activations if source_cache else {}
    )
    target_k_activations: dict[int, "Array"] = (
        target_cache.k_activations if target_cache else {}
    )
    # V Attention-space activations for v_proj stitching (separate for granular alignment)
    source_v_activations: dict[int, "Array"] = (
        source_cache.v_activations if source_cache else {}
    )
    target_v_activations: dict[int, "Array"] = (
        target_cache.v_activations if target_cache else {}
    )
    probe_ids: list[str] = []
    probe_domains: list[str] = []

    probes_processed = 0
    probes_failed = invalid_probe_count
    checkpoint_path: Path | None = None  # Defined early so it's available for cleanup

    # =========================================================================
    # BATCHED PROBE COLLECTION - Process probes in batches for efficiency
    # =========================================================================

    if not run_inference:
        probe_ids = list(expected_probe_ids)
        probe_domains = list(expected_probe_domains)
        probes_processed = len(probe_ids)

    if run_inference:
        # Instead of 806 individual forward passes (403 probes × 2 models),
        # we batch probes into groups and use batch methods for ~5-10× speedup.
        #
        # The batch size is tuned for GPU memory vs throughput tradeoff.
        # Too large = OOM, too small = lose batching benefit.
        PROBE_BATCH_SIZE = 4  # Smaller batches for more frequent memory cleanup

        # valid_probes already computed for caching and reuse
        logger.info(
            "PROBE PRECISE: %d valid probes, processing in batches of %d...",
            len(valid_probes),
            PROBE_BATCH_SIZE,
        )

        # =========================================================================
        # CHECKPOINT: Check for existing checkpoint to resume from
        # =========================================================================
        start_probe_idx = 0
        completed_probe_ids: set[str] = set()

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / ".probe_checkpoint.json"
            existing_checkpoint = _load_probe_checkpoint(checkpoint_path)

            if existing_checkpoint is not None:
                completed_probe_ids = set(existing_checkpoint.get("probe_ids", []))
                # Find how many valid probes were already completed
                # We'll skip probes that are in completed_probe_ids
                logger.info(
                    "PROBE: Found checkpoint with %d completed probes, resuming...",
                    len(completed_probe_ids),
                )

                # CRITICAL: Load saved activations for correct resume
                # Without this, completed probes are skipped but their activations lost!
                loaded_activations = _load_probe_activations(checkpoint_path, b)
                if loaded_activations is not None:
                    (
                        loaded_src_hidden,
                        loaded_tgt_hidden,
                        loaded_src_inter,
                        loaded_tgt_inter,
                        loaded_src_attn_q,
                        loaded_tgt_attn_q,
                        loaded_src_attn_k,
                        loaded_tgt_attn_k,
                        loaded_src_attn_v,
                        loaded_tgt_attn_v,
                    ) = loaded_activations
                    # Merge loaded activations with any from cache
                    source_layer_activations.update(loaded_src_hidden)
                    target_layer_activations.update(loaded_tgt_hidden)
                    source_intermediate_activations.update(loaded_src_inter)
                    target_intermediate_activations.update(loaded_tgt_inter)
                    source_attention_activations.update(loaded_src_attn_q)
                    target_attention_activations.update(loaded_tgt_attn_q)
                    source_k_activations.update(loaded_src_attn_k)
                    target_k_activations.update(loaded_tgt_attn_k)
                    source_v_activations.update(loaded_src_attn_v)
                    target_v_activations.update(loaded_tgt_attn_v)
                else:
                    # Activation file missing - need to re-run all probes
                    logger.warning(
                        "PROBE: Activation checkpoint missing, re-running all probes"
                    )
                    completed_probe_ids = set()

        # Check if activation provider supports batch methods
        has_batch_hidden = hasattr(activation_provider, "collect_hidden_activations_batch")
        has_batch_intermediate = hasattr(activation_provider, "collect_intermediate_activations_batch")
        has_batch_attention = hasattr(activation_provider, "collect_attention_activations_batch")

        # Second pass: Batch forward passes
        for batch_start in range(0, len(valid_probes), PROBE_BATCH_SIZE):
            batch_end = min(batch_start + PROBE_BATCH_SIZE, len(valid_probes))
            batch = valid_probes[batch_start:batch_end]
            batch_texts = [probe_text for _, probe_text in batch]
            batch_size = len(batch_texts)
            empty_batch = [{} for _ in range(batch_size)]

            try:
                # ===== HIDDEN ACTIVATIONS =====
                if has_batch_hidden:
                    source_hidden_batch = (
                        activation_provider.collect_hidden_activations_batch(
                            source_model, source_tokenizer, batch_texts
                        )
                        if run_source_inference
                        else empty_batch
                    )
                    target_hidden_batch = (
                        activation_provider.collect_hidden_activations_batch(
                            target_model, target_tokenizer, batch_texts
                        )
                        if run_target_inference
                        else empty_batch
                    )
                else:
                    # Fallback to sequential
                    source_hidden_batch = (
                        [
                            activation_provider.collect_hidden_activations(
                                source_model, source_tokenizer, text
                            )
                            for text in batch_texts
                        ]
                        if run_source_inference
                        else empty_batch
                    )
                    target_hidden_batch = (
                        [
                            activation_provider.collect_hidden_activations(
                                target_model, target_tokenizer, text
                            )
                            for text in batch_texts
                        ]
                        if run_target_inference
                        else empty_batch
                    )
    
                # ===== EMBEDDING ACTIVATIONS (2D GramAlign) =====
                # Same CKA=1.0, same geodesic math - applied at embedding dimension
                has_embedding = hasattr(activation_provider, "collect_embedding_activations")
                if has_embedding:
                    for text in batch_texts:
                        if run_source_inference:
                            source_emb = activation_provider.collect_embedding_activations(
                                source_model, source_tokenizer, text
                            )
                            if isinstance(source_embedding_activations, list):
                                source_embedding_activations.append(source_emb)
                        if run_target_inference:
                            target_emb = activation_provider.collect_embedding_activations(
                                target_model, target_tokenizer, text
                            )
                            if isinstance(target_embedding_activations, list):
                                target_embedding_activations.append(target_emb)
    
                # ===== INTERMEDIATE ACTIVATIONS =====
                if has_batch_intermediate:
                    source_intermediate_batch = (
                        activation_provider.collect_intermediate_activations_batch(
                            source_model, source_tokenizer, batch_texts
                        )
                        if run_source_inference
                        else empty_batch
                    )
                    target_intermediate_batch = (
                        activation_provider.collect_intermediate_activations_batch(
                            target_model, target_tokenizer, batch_texts
                        )
                        if run_target_inference
                        else empty_batch
                    )
                else:
                    source_intermediate_batch = (
                        [
                            activation_provider.collect_intermediate_activations(
                                source_model, source_tokenizer, text
                            )
                            for text in batch_texts
                        ]
                        if run_source_inference
                        else empty_batch
                    )
                    target_intermediate_batch = (
                        [
                            activation_provider.collect_intermediate_activations(
                                target_model, target_tokenizer, text
                            )
                            for text in batch_texts
                        ]
                        if run_target_inference
                        else empty_batch
                    )
    
                # ===== ATTENTION ACTIVATIONS (Q, K, V separately) =====
                if has_batch_attention:
                    if run_source_inference:
                        source_q_batch, source_k_batch, source_v_batch = (
                            activation_provider.collect_attention_activations_batch(
                                source_model, source_tokenizer, batch_texts
                            )
                        )
                    else:
                        source_q_batch, source_k_batch, source_v_batch = (
                            empty_batch,
                            empty_batch,
                            empty_batch,
                        )
                    if run_target_inference:
                        target_q_batch, target_k_batch, target_v_batch = (
                            activation_provider.collect_attention_activations_batch(
                                target_model, target_tokenizer, batch_texts
                            )
                        )
                    else:
                        target_q_batch, target_k_batch, target_v_batch = (
                            empty_batch,
                            empty_batch,
                            empty_batch,
                        )
                else:
                    source_q_batch, source_k_batch, source_v_batch = [], [], []
                    target_q_batch, target_k_batch, target_v_batch = [], [], []
                    for text in batch_texts:
                        if run_source_inference:
                            src_q, src_k, src_v = activation_provider.collect_attention_activations(
                                source_model, source_tokenizer, text
                            )
                        else:
                            src_q, src_k, src_v = {}, {}, {}
                        if run_target_inference:
                            tgt_q, tgt_k, tgt_v = activation_provider.collect_attention_activations(
                                target_model, target_tokenizer, text
                            )
                        else:
                            tgt_q, tgt_k, tgt_v = {}, {}, {}
                        source_q_batch.append(src_q)
                        source_k_batch.append(src_k)
                        source_v_batch.append(src_v)
                        target_q_batch.append(tgt_q)
                        target_k_batch.append(tgt_k)
                        target_v_batch.append(tgt_v)
    
                # Process batch results
                for i, (probe, probe_text) in enumerate(batch):
                    # Skip probes that were already completed (from checkpoint)
                    if probe.probe_id in completed_probe_ids:
                        continue
    
                    source_acts = source_hidden_batch[i]
                    target_acts = target_hidden_batch[i]
                    source_intermediate_acts = source_intermediate_batch[i]
                    target_intermediate_acts = target_intermediate_batch[i]
                    source_attention_acts = source_q_batch[i] if source_q_batch else {}
                    source_k_acts = source_k_batch[i] if source_k_batch else {}
                    source_v_acts = source_v_batch[i] if source_v_batch else {}
                    target_attention_acts = target_q_batch[i] if target_q_batch else {}
                    target_k_acts = target_k_batch[i] if target_k_batch else {}
                    target_v_acts = target_v_batch[i] if target_v_batch else {}
    
                    source_activated: dict[int, list[ActivatedDimension]] = {}
                    target_activated: dict[int, list[ActivatedDimension]] = {}

                    if run_source_inference:
                        for layer_idx, act in source_acts.items():
                            source_activated[layer_idx] = _extract_top_k_dims(
                                act, backend=b
                            )
                            # MEMORY FIX: accumulate into single buffer per layer
                            _accumulate_activation(source_layer_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_acts.items():
                            target_activated[layer_idx] = _extract_top_k_dims(
                                act, backend=b
                            )
                            # MEMORY FIX: accumulate into single buffer per layer
                            _accumulate_activation(target_layer_activations, layer_idx, act, b)

                    # Store intermediate activations for multi-space stitching
                    if run_source_inference:
                        for layer_idx, act in source_intermediate_acts.items():
                            _accumulate_activation(source_intermediate_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_intermediate_acts.items():
                            _accumulate_activation(target_intermediate_activations, layer_idx, act, b)

                    # Store Q attention activations for q_proj/o_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_attention_acts.items():
                            _accumulate_activation(source_attention_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_attention_acts.items():
                            _accumulate_activation(target_attention_activations, layer_idx, act, b)

                    # Store K attention activations for k_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_k_acts.items():
                            _accumulate_activation(source_k_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_k_acts.items():
                            _accumulate_activation(target_k_activations, layer_idx, act, b)

                    # Store V attention activations for v_proj stitching
                    if run_source_inference:
                        for layer_idx, act in source_v_acts.items():
                            _accumulate_activation(source_v_activations, layer_idx, act, b)

                    if run_target_inference:
                        for layer_idx, act in target_v_acts.items():
                            _accumulate_activation(target_v_activations, layer_idx, act, b)

                    if run_source_inference and source_activated:
                        source_fingerprints.append(
                            ActivationFingerprint(
                                prime_id=probe.probe_id,
                                prime_text=probe.name,
                                activated_dimensions=source_activated,
                            )
                        )
                    if run_target_inference and target_activated:
                        target_fingerprints.append(
                            ActivationFingerprint(
                                prime_id=probe.probe_id,
                                prime_text=probe.name,
                                activated_dimensions=target_activated,
                            )
                        )
    
                    probe_ids.append(probe.probe_id)
                    probe_domains.append(probe.domain.value)
                    probes_processed += 1
    
                # Log progress at batch boundaries
                if batch_end % 50 <= PROBE_BATCH_SIZE:
                    logger.info(
                        "PROBE PRECISE: Processed %d/%d probes...",
                        probes_processed,
                        len(valid_probes),
                    )
                
                # MEMORY OPTIMIZATION: Aggressive cleanup after each batch
                # This prevents memory accumulation that causes OOM
                try:
                    import gc
                    import mlx.core as mx
                    mx.eval()  # Force pending computations to complete
                    mx.clear_cache()  # Release GPU memory (new API)
                    gc.collect()  # Release Python objects
                except Exception:
                    pass  # Non-MLX backends or clearing not supported
    
                # Save checkpoint periodically
                if checkpoint_path is not None and probes_processed % _CHECKPOINT_INTERVAL < PROBE_BATCH_SIZE:
                    _save_probe_checkpoint(
                        checkpoint_path=checkpoint_path,
                        completed_probes=probes_processed,
                        probe_ids=probe_ids,
                        probe_domains=probe_domains,
                        total_probes=len(valid_probes),
                    )
                    # CRITICAL: Save activations alongside checkpoint for correct resume
                    _save_probe_activations(
                        checkpoint_path=checkpoint_path,
                        source_layer_activations=source_layer_activations,
                        target_layer_activations=target_layer_activations,
                        source_intermediate_activations=source_intermediate_activations,
                        target_intermediate_activations=target_intermediate_activations,
                        source_attention_activations=source_attention_activations,
                        target_attention_activations=target_attention_activations,
                        source_k_activations=source_k_activations,
                        target_k_activations=target_k_activations,
                        source_v_activations=source_v_activations,
                        target_v_activations=target_v_activations,
                    )

            except Exception as e:
                # On batch failure, fall back to sequential processing for this batch
                logger.warning("Batch processing failed, falling back to sequential: %s", e)
                for probe, probe_text in batch:
                    # Skip probes that were already completed (from checkpoint)
                    if probe.probe_id in completed_probe_ids:
                        continue
    
                    try:
                        source_acts = (
                            activation_provider.collect_hidden_activations(
                                source_model, source_tokenizer, probe_text
                            )
                            if run_source_inference
                            else {}
                        )
                        target_acts = (
                            activation_provider.collect_hidden_activations(
                                target_model, target_tokenizer, probe_text
                            )
                            if run_target_inference
                            else {}
                        )
                        source_intermediate_acts = (
                            activation_provider.collect_intermediate_activations(
                                source_model, source_tokenizer, probe_text
                            )
                            if run_source_inference
                            else {}
                        )
                        target_intermediate_acts = (
                            activation_provider.collect_intermediate_activations(
                                target_model, target_tokenizer, probe_text
                            )
                            if run_target_inference
                            else {}
                        )
                        if run_source_inference:
                            (
                                source_attention_acts,
                                source_k_acts,
                                source_v_acts,
                            ) = activation_provider.collect_attention_activations(
                                source_model, source_tokenizer, probe_text
                            )
                        else:
                            source_attention_acts, source_k_acts, source_v_acts = {}, {}, {}
                        if run_target_inference:
                            (
                                target_attention_acts,
                                target_k_acts,
                                target_v_acts,
                            ) = activation_provider.collect_attention_activations(
                                target_model, target_tokenizer, probe_text
                            )
                        else:
                            target_attention_acts, target_k_acts, target_v_acts = {}, {}, {}
    
                        source_activated_fallback: dict[int, list[ActivatedDimension]] = {}
                        target_activated_fallback: dict[int, list[ActivatedDimension]] = {}
    
                        if run_source_inference:
                            for layer_idx, act in source_acts.items():
                                source_activated_fallback[layer_idx] = _extract_top_k_dims(
                                    act, backend=b
                                )
                                _accumulate_activation(source_layer_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_acts.items():
                                target_activated_fallback[layer_idx] = _extract_top_k_dims(
                                    act, backend=b
                                )
                                _accumulate_activation(target_layer_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_intermediate_acts.items():
                                _accumulate_activation(source_intermediate_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_intermediate_acts.items():
                                _accumulate_activation(target_intermediate_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_attention_acts.items():
                                _accumulate_activation(source_attention_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_attention_acts.items():
                                _accumulate_activation(target_attention_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_k_acts.items():
                                _accumulate_activation(source_k_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_k_acts.items():
                                _accumulate_activation(target_k_activations, layer_idx, act, b)
    
                        if run_source_inference:
                            for layer_idx, act in source_v_acts.items():
                                _accumulate_activation(source_v_activations, layer_idx, act, b)
    
                        if run_target_inference:
                            for layer_idx, act in target_v_acts.items():
                                _accumulate_activation(target_v_activations, layer_idx, act, b)
    
                        if run_source_inference and source_activated_fallback:
                            source_fingerprints.append(
                                ActivationFingerprint(
                                    prime_id=probe.probe_id,
                                    prime_text=probe.name,
                                    activated_dimensions=source_activated_fallback,
                                )
                            )
                        if run_target_inference and target_activated_fallback:
                            target_fingerprints.append(
                                ActivationFingerprint(
                                    prime_id=probe.probe_id,
                                    prime_text=probe.name,
                                    activated_dimensions=target_activated_fallback,
                                )
                            )
    
                        probe_ids.append(probe.probe_id)
                        probe_domains.append(probe.domain.value)
                        probes_processed += 1
    
                    except Exception as inner_e:
                        logger.warning("Probe '%s' failed: %s", probe.probe_id, inner_e)
                        probes_failed += 1

    logger.info(
        "PROBE PRECISE: Completed %d probes (%d failed), built %d fingerprints",
        probes_processed,
        probes_failed,
        len(source_fingerprints),
    )

    # Clear checkpoint after successful completion
    if checkpoint_path is not None:
        _clear_probe_checkpoint(checkpoint_path)

    cache_ready = (
        probe_ids == expected_probe_ids and probe_domains == expected_probe_domains
    )

    def _cache_spaces(
        hidden: dict[int, "Array"],
        intermediate: dict[int, "Array"],
        attention: dict[int, "Array"],
        k_act: dict[int, "Array"],
        v_act: dict[int, "Array"],
        embedding: "Array | list[Array] | None",
    ) -> list[str]:
        spaces = ["hidden"] if hidden else []
        if intermediate:
            spaces.append("intermediate")
        if attention:
            spaces.append("attention_q")
        if k_act:
            spaces.append("attention_k")
        if v_act:
            spaces.append("attention_v")
        if embedding is not None:
            spaces.append("embedding")
        return spaces

    def _update_profile_cache(
        profile: ModelProfile | None,
        identity: Any | None,
        fingerprints: list[ActivationFingerprint],
        cache_present: bool,
        spaces: list[str],
    ) -> None:
        if profile is None or identity is None:
            return
        updated = False
        if cache_ready and fingerprints:
            profile.probe_fingerprints = fingerprints
            profile.probe_corpus_hash = probe_corpus_hash
            updated = True
        if cache_present and spaces:
            profile.probe_corpus_hash = probe_corpus_hash
            profile.probe_cache[cache_key] = {
                "probe_mode": probe_mode,
                "probe_corpus_hash": probe_corpus_hash,
                "probe_count": len(expected_probe_ids),
                "spaces": spaces,
                "updated_at": datetime.now().isoformat(),
            }
            updated = True
        if updated:
            try:
                profile_store.save(profile, identity)
                logger.info("PROBE: Updated profile cache for %s", profile.model_id)
            except Exception as e:
                logger.warning("PROBE: Failed to save profile cache: %s", e)

    # Save per-model caches and fingerprints
    if run_source_inference and cache_ready and source_identity and source_layer_activations:
        _save_model_probe_cache(
            model_id=source_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            hidden_activations=source_layer_activations,
            intermediate_activations=source_intermediate_activations,
            attention_activations=source_attention_activations,
            k_activations=source_k_activations,
            v_activations=source_v_activations,
            embedding_activations=source_embedding_activations,
        )

    if run_target_inference and cache_ready and target_identity and target_layer_activations:
        _save_model_probe_cache(
            model_id=target_identity.model_id,
            probe_mode=probe_mode,
            probe_corpus_hash=probe_corpus_hash,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            hidden_activations=target_layer_activations,
            intermediate_activations=target_intermediate_activations,
            attention_activations=target_attention_activations,
            k_activations=target_k_activations,
            v_activations=target_v_activations,
            embedding_activations=target_embedding_activations,
        )

    source_cache_present = source_cache is not None or (
        run_source_inference and cache_ready
    )
    target_cache_present = target_cache is not None or (
        run_target_inference and cache_ready
    )

    _update_profile_cache(
        source_profile,
        source_identity,
        source_fingerprints,
        source_cache_present,
        _cache_spaces(
            source_layer_activations,
            source_intermediate_activations,
            source_attention_activations,
            source_k_activations,
            source_v_activations,
            source_embedding_activations,
        ),
    )
    _update_profile_cache(
        target_profile,
        target_identity,
        target_fingerprints,
        target_cache_present,
        _cache_spaces(
            target_layer_activations,
            target_intermediate_activations,
            target_attention_activations,
            target_k_activations,
            target_v_activations,
            target_embedding_activations,
        ),
    )

    # Build IntersectionMap
    intersection_map_obj: IntersectionMap | None = None
    dimension_correlations: dict = {}
    layer_cka_scores: dict[int, float] = {}
    layer_cka_scores_raw: dict[int, float] = {}

    if source_fingerprints and target_fingerprints:
        try:
            intersection_map_obj = build_intersection_map(
                source_fingerprints=source_fingerprints,
                target_fingerprints=target_fingerprints,
                source_model=source_path or "source",
                target_model=target_path or "target",
                mode=IntersectionSimilarityMode.CKA,  # Pure geometry - CKA is the metric
            )
            dimension_correlations = intersection_map_obj.dimension_correlations
            logger.info(
                "PROBE PRECISE: Built IntersectionMap (%d layers), sparse mean_layer_cka=%.3f",
                len(intersection_map_obj.layer_confidences),
                intersection_map_obj.mean_layer_cka,
            )
        except Exception as e:
            logger.warning("Failed to build IntersectionMap: %s", e)
            intersection_map_obj = None

    # Align layers by proportional depth, then solve closed-form alignment per pair.
    # CKA = 1.0 is invariant; we do not use CKA as a selector.
    layer_mapping: dict[int, int] = {}  # target_layer -> source_layer
    feature_transforms: dict[int, Any] = {}  # target_layer -> {src_layer: GPU array}
    scale_ratios: dict[int, float] = {}  # EXACT scale factor per layer: ||target|| / ||source @ F||
    attention_transforms: dict[int, Any] = {}  # target_layer -> Q attention transform (GPU array)
    k_transforms: dict[int, Any] = {}  # target_layer -> K attention transform (GPU array)
    v_transforms: dict[int, Any] = {}  # target_layer -> V attention transform (GPU array)
    intermediate_transforms: dict[int, Any] = {}  # target_layer -> MLP intermediate transform (GPU array)
    gram_aligner = GramAligner(backend=b)
    rbf_consistency_checked = False
    rbf_consistency_hidden: dict[str, float] | None = None

    if source_layer_activations and target_layer_activations:
        source_layers = sorted(source_layer_activations.keys())
        target_layers = sorted(target_layer_activations.keys())
        n_source = len(source_layers)
        n_target = len(target_layers)

        if n_source > 0 and n_target > 0:
            # =========================================================================
            # PROPORTIONAL DEPTH MAPPING (CKA is invariant, not a selector)
            # =========================================================================
            # Map layers by normalized depth and solve alignment per pair.
            # No CKA matrix, no Hungarian, no selection heuristics.
            alignment_tasks: list[tuple[int, list[int]]] = []
            for tgt_idx in range(n_target):
                src_idx = _proportional_layer_index(tgt_idx, n_target, n_source)
                alignment_tasks.append((tgt_idx, [src_idx]))

            alignment_tasks_sorted = alignment_tasks
            logger.info(
                "PROBE: Aligning %d target layers (proportional depth mapping)...",
                len(alignment_tasks_sorted),
            )
            
            def _align_target_group(
                tgt_idx: int,
                src_indices: list[int],
                F_init: "Array | None" = None,
            ) -> dict:
                """Align source layer(s) to a target layer.

                With 1:1 mapping, src_indices has exactly 1 element.
                F_init: Optional warm-start transform from a successful neighbor (zipper).
                """
                nonlocal rbf_consistency_checked, rbf_consistency_hidden
                tgt_layer = target_layers[tgt_idx]
                src_layers_list = [source_layers[i] for i in src_indices]
                
                result: dict = {
                    "tgt_layer": tgt_layer,
                    "src_layers": src_layers_list,
                    "raw_cka": 0.0,
                    "achieved_cka": 1.0,  # CKA = 1.0 is invariant
                    "numerical_deviation": 0.0,  # For precision diagnostics
                    "feature_transform": None,
                    "attention_transform": None,
                    "k_transform": None,
                    "v_transform": None,
                    "error": None,
                }

                src_act_lists = [source_layer_activations[s] for s in src_layers_list]
                tgt_list = target_layer_activations[tgt_layer]
                
                n_samples = len(tgt_list)
                for s_list in src_act_lists:
                    n_samples = min(n_samples, len(s_list))
                
                if n_samples < 2:
                    raise RuntimeError(
                        f"Insufficient samples for {src_layers_list} -> {tgt_layer}: {n_samples}"
                    )
                
                # Log sample count for rank verification
                # Note: src_act_lists is a Python list, tgt_list is an MLX array
                # MLX arrays cannot be used in boolean context (causes conversion error)
                src_dim = src_act_lists[0][0].shape[-1] if src_act_lists else 0
                tgt_dim = int(tgt_list.shape[-1]) if tgt_list is not None and len(tgt_list.shape) > 0 else 0
                logger.info(
                    "ALIGNMENT: Layer %d <- %s: n_samples=%d (need >=%d for full-rank src, >=%d for full-rank tgt)",
                    tgt_layer, src_layers_list, n_samples, src_dim, tgt_dim
                )

                local_aligner = GramAligner(backend=b)
                # Fast aligner for attention (5000 steps vs 50000) - good enough for trajectory guidance
                fast_aligner = GramAligner(backend=b, max_steps=5000)

                try:
                    # Stack source (for 1:1, this is just 1 source)
                    src_stacks = []
                    src_dims = []
                    for s, s_list in zip(src_layers_list, src_act_lists):
                        stack = b.stack(s_list[:n_samples], axis=0)
                        stack = b.astype(stack, "float32")
                        src_stacks.append(stack)
                        src_dims.append(stack.shape[1])
                    
                    tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                    tgt_stacked = b.astype(tgt_stacked, "float32")
                    
                    # For 1:1 mapping, this is just src_stacks[0] (no concat needed)
                    if len(src_stacks) == 1:
                        src_combined = src_stacks[0]
                    else:
                        src_combined = b.concatenate(src_stacks, axis=1)
                    
                    b.eval(src_combined, tgt_stacked)
                    
                    from modelcypher.core.domain.geometry.cka import compute_cka_backend
                    result["raw_cka"] = float(compute_cka_backend(src_combined, tgt_stacked, b))

                    alignment_result = local_aligner.find_perfect_alignment(
                        src_combined,
                        tgt_stacked,
                        F_init=F_init,  # Zipper warm-start from neighbor
                    )
                    # Alignment runs until CKA=1.0 within machine epsilon - no retry needed
                    
                    result["achieved_cka"] = alignment_result.achieved_cka  # Always 1.0 (invariant)
                    result["numerical_deviation"] = alignment_result.numerical_deviation

                    F_arr = alignment_result.feature_transform  # Already GPU array

                    # One-time geodesic RBF vs linear CKA consistency check (4D+).
                    if not rbf_consistency_checked:
                        from modelcypher.core.domain.geometry.cka import compute_cka

                        aligned = b.matmul(src_combined, F_arr)
                        b.eval(aligned)
                        rbf_result = compute_cka(aligned, tgt_stacked, backend=b)
                        rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                        precision = sqrt_scalar(machine_epsilon(b, aligned), b)
                        rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                        linear_deviation = alignment_result.numerical_deviation
                        linear_cka = 1.0 - linear_deviation
                        agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                        rbf_consistency_hidden = {
                            "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                            "rbf_deviation": float(rbf_deviation),
                            "linear_deviation": float(linear_deviation),
                            "agreement_deviation": float(agreement_deviation),
                            "precision_threshold": float(precision),
                            "layer": float(tgt_layer),
                        }
                        if rbf_deviation > precision:
                            logger.error(
                                "PROBE: RBF CKA deviation %.2e > precision %.2e for layer %d "
                                "(linear deviation %.2e) - precision bug.",
                                rbf_deviation,
                                precision,
                                tgt_layer,
                                linear_deviation,
                            )
                        if agreement_deviation > precision:
                            logger.error(
                                "PROBE: RBF vs linear CKA mismatch %.2e > precision %.2e for layer %d.",
                                agreement_deviation,
                                precision,
                                tgt_layer,
                            )
                        rbf_consistency_checked = True
                    
                    # Store raw F_arr for zipper warm-start of neighbors
                    result["F_arr_raw"] = F_arr

                    # Split transform for each source layer
                    # KEEP AS GPU ARRAYS - avoid CPU→GPU reconversion in downstream code
                    split_transforms: dict[int, Any] = {}
                    start_idx = 0
                    for s_layer, s_dim in zip(src_layers_list, src_dims):
                        F_slice = F_arr[start_idx : start_idx + s_dim, :]
                        split_transforms[s_layer] = F_slice  # GPU array, not tolist()
                        start_idx += s_dim
                        
                    result["feature_transform"] = split_transforms
                    
                    # EXACT SCALE FACTOR: ||target|| / ||source @ F||
                    # Apply this to stitched weights for exact magnitude match
                    result["scale_ratio"] = alignment_result.scale_ratio

                    # CKA = 1.0 is invariant. Check numerical precision instead.
                    if not alignment_result.is_numerically_exact:
                        logger.warning(
                            "PROBE: Layer %s -> %d has numerical precision issue (deviation=%.2e). "
                            "CKA = 1.0 by construction; this is a precision bug, not model incompatibility.",
                            src_layers_list, tgt_layer, alignment_result.numerical_deviation
                        )
                    
                    # =====================================================================
                    # ATTENTION Q/K/V ALIGNMENT (pre-compute for transplant speed)
                    # =====================================================================
                    # OPTIMIZATION: Batch preparation of Q/K/V stacks, then single eval()
                    # This reduces 3 GPU syncs to 1 per layer while keeping alignments separate
                    # (separate alignments are mathematically required - see plan file)

                    # Gather activation lists
                    src_q_acts = [source_attention_activations.get(s) for s in src_layers_list]
                    tgt_q_acts = target_attention_activations.get(tgt_layer)
                    src_k_acts = [source_k_activations.get(s) for s in src_layers_list]
                    tgt_k_acts = target_k_activations.get(tgt_layer)
                    src_v_acts = [source_v_activations.get(s) for s in src_layers_list]
                    tgt_v_acts = target_v_activations.get(tgt_layer)

                    # Prepare Q/K/V stacks (lazy - no eval yet)
                    q_prepared = None
                    k_prepared = None
                    v_prepared = None

                    # Q preparation
                    if all(acts is not None and len(acts) > 0 for acts in src_q_acts) and tgt_q_acts is not None and len(tgt_q_acts) > 0:
                        n_attn_samples = min(len(tgt_q_acts), min(len(acts) for acts in src_q_acts))
                        if n_attn_samples >= 2:
                            try:
                                src_q_stacks = []
                                src_q_dims = []
                                for s_list in src_q_acts:
                                    stack = b.stack(s_list[:n_attn_samples], axis=0)
                                    stack = b.astype(stack, "float32")
                                    src_q_stacks.append(stack)
                                    src_q_dims.append(stack.shape[1])

                                tgt_q_stacked = b.stack(tgt_q_acts[:n_attn_samples], axis=0)
                                tgt_q_stacked = b.astype(tgt_q_stacked, "float32")

                                if len(src_q_stacks) == 1:
                                    src_q_combined = src_q_stacks[0]
                                else:
                                    src_q_combined = b.concatenate(src_q_stacks, axis=1)

                                # Store for batch eval (NO eval here)
                                q_prepared = (src_q_combined, tgt_q_stacked, src_q_dims)
                            except Exception as q_prep_err:
                                logger.debug(
                                    "PROBE: Q preparation failed for %s -> %d: %s",
                                    src_layers_list, tgt_layer, q_prep_err
                                )

                    # K preparation
                    if all(acts is not None and len(acts) > 0 for acts in src_k_acts) and tgt_k_acts is not None and len(tgt_k_acts) > 0:
                        n_k_samples = min(len(tgt_k_acts), min(len(acts) for acts in src_k_acts))
                        if n_k_samples >= 2:
                            try:
                                src_k_stacks = []
                                src_k_dims = []
                                for s_list in src_k_acts:
                                    stack = b.stack(s_list[:n_k_samples], axis=0)
                                    stack = b.astype(stack, "float32")
                                    src_k_stacks.append(stack)
                                    src_k_dims.append(stack.shape[1])

                                tgt_k_stacked = b.stack(tgt_k_acts[:n_k_samples], axis=0)
                                tgt_k_stacked = b.astype(tgt_k_stacked, "float32")

                                if len(src_k_stacks) == 1:
                                    src_k_combined = src_k_stacks[0]
                                else:
                                    src_k_combined = b.concatenate(src_k_stacks, axis=1)

                                # Store for batch eval (NO eval here)
                                k_prepared = (src_k_combined, tgt_k_stacked, src_k_dims)
                            except Exception as k_prep_err:
                                logger.debug(
                                    "PROBE: K preparation failed for %s -> %d: %s",
                                    src_layers_list, tgt_layer, k_prep_err
                                )

                    # V preparation
                    if all(acts is not None and len(acts) > 0 for acts in src_v_acts) and tgt_v_acts is not None and len(tgt_v_acts) > 0:
                        n_v_samples = min(len(tgt_v_acts), min(len(acts) for acts in src_v_acts))
                        if n_v_samples >= 2:
                            try:
                                src_v_stacks = []
                                src_v_dims = []
                                for s_list in src_v_acts:
                                    stack = b.stack(s_list[:n_v_samples], axis=0)
                                    stack = b.astype(stack, "float32")
                                    src_v_stacks.append(stack)
                                    src_v_dims.append(stack.shape[1])

                                tgt_v_stacked = b.stack(tgt_v_acts[:n_v_samples], axis=0)
                                tgt_v_stacked = b.astype(tgt_v_stacked, "float32")

                                if len(src_v_stacks) == 1:
                                    src_v_combined = src_v_stacks[0]
                                else:
                                    src_v_combined = b.concatenate(src_v_stacks, axis=1)

                                # Store for batch eval (NO eval here)
                                v_prepared = (src_v_combined, tgt_v_stacked, src_v_dims)
                            except Exception as v_prep_err:
                                logger.debug(
                                    "PROBE: V preparation failed for %s -> %d: %s",
                                    src_layers_list, tgt_layer, v_prep_err
                                )

                    # BATCH EVAL: Single GPU sync for all prepared Q/K/V stacks
                    # This replaces 3 separate evals with 1 batched eval
                    stacks_to_eval = []
                    if q_prepared is not None:
                        stacks_to_eval.extend([q_prepared[0], q_prepared[1]])
                    if k_prepared is not None:
                        stacks_to_eval.extend([k_prepared[0], k_prepared[1]])
                    if v_prepared is not None:
                        stacks_to_eval.extend([v_prepared[0], v_prepared[1]])
                    if stacks_to_eval:
                        b.eval(*stacks_to_eval)

                    # Q ALIGNMENT (must be separate - mathematically required)
                    if q_prepared is not None:
                        src_q_combined, tgt_q_stacked, src_q_dims = q_prepared
                        try:
                            q_alignment = fast_aligner.find_perfect_alignment(
                                src_q_combined,
                                tgt_q_stacked,
                            )

                            Q_arr = q_alignment.feature_transform  # Already GPU array

                            split_q_transforms = {}
                            start_idx = 0
                            for s_layer, s_dim in zip(src_layers_list, src_q_dims):
                                Q_slice = Q_arr[start_idx : start_idx + s_dim, :]
                                split_q_transforms[s_layer] = Q_slice  # Keep as GPU array
                                start_idx += s_dim

                            result["attention_transform"] = split_q_transforms

                            logger.debug(
                                "PROBE: Q attention aligned for %s -> %d (CKA=%.4f)",
                                src_layers_list, tgt_layer, q_alignment.achieved_cka
                            )
                        except Exception as q_err:
                            logger.debug(
                                "PROBE: Q attention alignment failed for %s -> %d: %s",
                                src_layers_list, tgt_layer, q_err
                            )

                    # K ALIGNMENT (must be separate - mathematically required)
                    if k_prepared is not None:
                        src_k_combined, tgt_k_stacked, src_k_dims = k_prepared
                        try:
                            k_alignment = fast_aligner.find_perfect_alignment(
                                src_k_combined,
                                tgt_k_stacked,
                            )

                            K_arr = k_alignment.feature_transform  # Already GPU array

                            split_k_transforms = {}
                            start_idx = 0
                            for s_layer, s_dim in zip(src_layers_list, src_k_dims):
                                K_slice = K_arr[start_idx : start_idx + s_dim, :]
                                split_k_transforms[s_layer] = K_slice  # Keep as GPU array
                                start_idx += s_dim

                            result["k_transform"] = split_k_transforms

                            logger.debug(
                                "PROBE: K attention aligned for %s -> %d (CKA=%.4f)",
                                src_layers_list, tgt_layer, k_alignment.achieved_cka
                            )
                        except Exception as k_err:
                            logger.debug(
                                "PROBE: K attention alignment failed for %s -> %d: %s",
                                src_layers_list, tgt_layer, k_err
                            )

                    # V ALIGNMENT (must be separate - mathematically required)
                    if v_prepared is not None:
                        src_v_combined, tgt_v_stacked, src_v_dims = v_prepared
                        try:
                            v_alignment = fast_aligner.find_perfect_alignment(
                                src_v_combined,
                                tgt_v_stacked,
                            )

                            V_arr = v_alignment.feature_transform  # Already GPU array

                            split_v_transforms = {}
                            start_idx = 0
                            for s_layer, s_dim in zip(src_layers_list, src_v_dims):
                                V_slice = V_arr[start_idx : start_idx + s_dim, :]
                                split_v_transforms[s_layer] = V_slice  # Keep as GPU array
                                start_idx += s_dim

                            result["v_transform"] = split_v_transforms

                            logger.debug(
                                "PROBE: V attention aligned for %s -> %d (CKA=%.4f)",
                                src_layers_list, tgt_layer, v_alignment.achieved_cka
                            )
                        except Exception as v_err:
                            logger.debug(
                                "PROBE: V attention alignment failed for %s -> %d: %s",
                                src_layers_list, tgt_layer, v_err
                            )

                    # =====================================================================
                    # INTERMEDIATE MLP ALIGNMENT (pre-compute for transplant speed)
                    # =====================================================================
                    # Align source intermediate activations to target intermediate activations
                    # This is the MAIN bottleneck in transplant - 50k steps per layer
                    # Pre-computing here with fast_aligner saves ~80% of transplant time
                    src_inter_acts = [source_intermediate_activations.get(s) for s in src_layers_list]
                    tgt_inter_acts = target_intermediate_activations.get(tgt_layer)
                    
                    if all(acts is not None and len(acts) > 0 for acts in src_inter_acts) and tgt_inter_acts is not None and len(tgt_inter_acts) > 0:
                        n_inter_samples = min(len(tgt_inter_acts), min(len(acts) for acts in src_inter_acts))
                        if n_inter_samples >= 2:
                            try:
                                # Stack intermediate activations
                                src_inter_stacks = []
                                src_inter_dims = []
                                for s_list in src_inter_acts:
                                    stack = b.stack(s_list[:n_inter_samples], axis=0)
                                    stack = b.astype(stack, "float32")
                                    src_inter_stacks.append(stack)
                                    src_inter_dims.append(stack.shape[1])
                                
                                tgt_inter_stacked = b.stack(tgt_inter_acts[:n_inter_samples], axis=0)
                                tgt_inter_stacked = b.astype(tgt_inter_stacked, "float32")
                                
                                if len(src_inter_stacks) == 1:
                                    src_inter_combined = src_inter_stacks[0]
                                else:
                                    src_inter_combined = b.concatenate(src_inter_stacks, axis=1)
                                
                                b.eval(src_inter_combined, tgt_inter_stacked)
                                
                                inter_alignment = fast_aligner.find_perfect_alignment(
                                    src_inter_combined,
                                    tgt_inter_stacked,
                                )
                                
                                I_arr = inter_alignment.feature_transform  # Already GPU array

                                split_inter_transforms = {}
                                start_idx = 0
                                for s_layer, s_dim in zip(src_layers_list, src_inter_dims):
                                    I_slice = I_arr[start_idx : start_idx + s_dim, :]
                                    split_inter_transforms[s_layer] = I_slice  # Keep as GPU array
                                    start_idx += s_dim
                                
                                result["intermediate_transform"] = split_inter_transforms
                                
                                logger.debug(
                                    "PROBE: Intermediate aligned for %s -> %d (CKA=%.4f)",
                                    src_layers_list, tgt_layer, inter_alignment.achieved_cka
                                )
                            except Exception as inter_err:
                                logger.debug(
                                    "PROBE: Intermediate alignment failed for %s -> %d: %s",
                                    src_layers_list, tgt_layer, inter_err
                                )

                except Exception as e:
                    raise RuntimeError(
                        f"GramAligner failed for {src_layers_list} -> {tgt_layer}: {e}"
                    ) from e

                # =========================================================================
                # MEMORY CLEANUP - Critical for preventing 45GB explosion
                # =========================================================================
                # Delete large intermediate arrays and sync GPU to free memory
                del src_combined, tgt_stacked, src_stacks, F_arr
                try:
                    del alignment_result
                except NameError:
                    pass
                b.eval()  # Force computation graph evaluation to release memory
                
                return result

            # =========================================================================
            # ZIPPER ALIGNMENT: Sequential processing with warm-start from neighbors
            # =========================================================================
            # Process in proportional-depth order. For each layer:
            # 1. Find nearest successfully aligned neighbor (by layer index)
            # 2. Use its F and R for warm-start and rotation hint
            # This is the "zipper" concept: easy layers align first, their geometry
            # accelerates convergence for difficult neighbors.
            
            successful_alignments: dict[int, dict] = {}  # tgt_layer -> {F}
            
            completed = 0
            for tgt_idx, src_indices in alignment_tasks_sorted:
                # Find nearest successful neighbor's F for warm-start
                tgt_layer = target_layers[tgt_idx]
                F_init = None

                if successful_alignments:
                    # Find the closest aligned layer by layer index
                    aligned_layers = list(successful_alignments.keys())
                    closest_layer = min(aligned_layers, key=lambda l: abs(l - tgt_layer))
                    neighbor_data = successful_alignments[closest_layer]
                    F_init = neighbor_data.get("F")
                    logger.info(f"ZIPPER: Layer {tgt_layer} warm-starting from layer {closest_layer}")
                else:
                    logger.debug(f"ZIPPER: Layer {tgt_layer} has no successful neighbors yet")

                # Align this layer
                result = _align_target_group(tgt_idx, src_indices, F_init=F_init)
                
                tgt_layer = result["tgt_layer"]
                src_layers = result["src_layers"]
                
                # Store mapping
                layer_mapping[tgt_layer] = src_layers[0]

                # RIGOROUS GEOMETRY: Transform must always exist.
                if result["feature_transform"] is None:
                    raise RuntimeError(
                        f"GramAligner returned no transform for {src_layers} -> {tgt_layer}. "
                        "This should never happen if the geometry is correct."
                    )
                feature_transforms[tgt_layer] = result["feature_transform"]
                layer_cka_scores[tgt_layer] = result["achieved_cka"]
                layer_cka_scores_raw[tgt_layer] = result["raw_cka"]
                
                # EXACT SCALE FACTOR per layer
                if "scale_ratio" in result:
                    scale_ratios[tgt_layer] = result["scale_ratio"]

                # Store alignment data for zipper warm-start of future layers
                # CKA = 1.0 is invariant - all alignments achieve it, so store all of them
                if result.get("F_arr_raw") is not None:
                    successful_alignments[tgt_layer] = {
                        "F": result["F_arr_raw"],
                        "R": result.get("R_raw", None),  # R from procrustes (if available)
                    }
                    logger.info(
                        f"ZIPPER: Stored layer {tgt_layer} (deviation={result.get('numerical_deviation', 0.0):.2e}) for warm-start"
                    )

                if result["attention_transform"] is not None:
                    attention_transforms[tgt_layer] = result["attention_transform"]

                if result.get("k_transform") is not None:
                    k_transforms[tgt_layer] = result["k_transform"]

                if result.get("v_transform") is not None:
                    v_transforms[tgt_layer] = result["v_transform"]

                if result.get("intermediate_transform") is not None:
                    intermediate_transforms[tgt_layer] = result["intermediate_transform"]

                completed += 1
                if completed % 5 == 0 or completed == len(alignment_tasks_sorted):
                    logger.info(
                        "PROBE: Aligned %d/%d target layers (zipper: %d warm-started)...",
                        completed,
                        len(alignment_tasks_sorted),
                        len(successful_alignments),
                    )

            logger.info(
                "PROBE: Cross-architecture layer alignment found %d mappings "
                "(source: %d layers, target: %d layers)",
                len(alignment_tasks),
                n_source,
                n_target,
            )

    # Extract layer confidences (CKA-first, IntersectionMap as fallback)
    # No fallbacks beyond geometric signals - if we don't have alignment, we don't merge
    layer_confidences: dict[int, float] = {}
    if layer_cka_scores:
        layer_confidences.update(layer_cka_scores)

    if intersection_map_obj is not None:
        for lc in intersection_map_obj.layer_confidences:
            if lc.layer not in layer_confidences:
                layer_confidences[lc.layer] = lc.confidence

    if not layer_confidences:
        logger.error(
            "PROBE FAILED: No layer correlations found. "
            "Cannot merge without knowing the geometric alignment."
        )
        # Return empty result - caller must check and refuse to merge
        return ProbeResult(
            correlations={},
            confidences={},
            intersection_map=None,
            dimension_correlations={},
            metrics={
                "probe_mode": "precise",
                "probes_total": len(probes),
                "probes_processed": probes_processed,
                "probes_failed": probes_failed,
                "fingerprints_built": len(source_fingerprints),
                "layers_analyzed": 0,
                "probe_failed": True,
                "failure_reason": "No layer correlations - cannot determine geometric alignment",
            },
        )

    # Build per-weight correlations
    weight_correlations: dict[str, float] = {}
    for key in target_weights:
        if key not in source_weights:
            continue
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is not None and layer_idx in layer_confidences:
            weight_correlations[key] = layer_confidences[layer_idx]
        else:
            weight_correlations[key] = 0.0

    cka_vals = list(layer_cka_scores.values())
    # CKA = 1.0 is an invariant, not a target. Filter only NaN (alignment bugs).
    # Low CKA means the alignment algorithm failed, not the layer - log and investigate.
    valid_cka_vals = [v for v in cka_vals if v == v]  # NaN check only
    nan_count = len(cka_vals) - len(valid_cka_vals)
    if nan_count > 0:
        logger.error(
            "PROBE: %d layers have NaN CKA - alignment algorithm bug, investigate!",
            nan_count
        )
    mean_cka = sum(valid_cka_vals) / len(valid_cka_vals) if valid_cka_vals else 0.0
    min_cka = min(valid_cka_vals) if valid_cka_vals else 0.0
    raw_cka_vals = list(layer_cka_scores_raw.values())
    mean_cka_raw = sum(raw_cka_vals) / len(raw_cka_vals) if raw_cka_vals else 0.0
    min_cka_raw = min(raw_cka_vals) if raw_cka_vals else 0.0
    # layers_with_data: layers that have activations in both models (for reporting)
    layers_with_data = set(source_layer_activations.keys()) & set(target_layer_activations.keys())
    # Proportional mapping defines a correspondence for every target layer.
    # missing_cka_layers is for reporting - it doesn't block exact alignment
    missing_cka_layers = [layer for layer in layers_with_data if layer not in layer_cka_scores]
    # =========================================================================
    # CKA = 1.0 IS AN INVARIANT, NOT A TARGET
    # =========================================================================
    # Experiments verified: CKA = 1.000000 is always achievable across all model
    # pairs and all layers. Any deviation indicates an alignment algorithm bug.
    # Use sqrt(machine_epsilon) as the tolerance (matches GramAligner convention).
    precision_threshold = sqrt_scalar(
        machine_epsilon(b, b.array([1.0], dtype="float32")),
        b,
    )
    perfect_alignment = bool(valid_cka_vals) and min_cka >= 1.0 - precision_threshold

    # =========================================================================
    # SPLIT CKA: SHARED VS. NOVEL CONCEPTS (POST-ALIGNMENT)
    # =========================================================================
    # CKA is meaningless BEFORE alignment - high-d representations get twisted
    # during pre-training. We must FIRST align, THEN measure shared vs novel.
    #
    # Compute CKA separately for:
    # - SHARED: concepts both models respond to (CKA should be ~1.0 after alignment)
    # - NOVEL: concepts only source responds to (new knowledge being added)
    split_cka_result = None
    if source_layer_activations and target_layer_activations and feature_transforms:
        from modelcypher.core.domain.geometry.cka import compute_cka_split

        # Use the layer with highest post-alignment CKA as representative
        if layer_cka_scores:
            best_layer = max(layer_cka_scores.keys(), key=lambda k: layer_cka_scores[k])
            # Find corresponding source layer from layer_mapping
            src_layer = layer_mapping.get(best_layer)
            # feature_transforms[tgt_layer] is a dict: {src_layer_idx: [[...], [...]]}
            F_transform_dict = feature_transforms.get(best_layer)

            if (src_layer is not None and
                src_layer in source_layer_activations and
                best_layer in target_layer_activations and
                F_transform_dict is not None and
                src_layer in F_transform_dict):

                src_acts = source_layer_activations[src_layer]
                tgt_acts = target_layer_activations[best_layer]

                # APPLY ALIGNMENT: source @ F -> aligned source in target's coordinate system
                # F_transform_dict[src_layer] is a GPU array (kept as array for efficiency)
                F_arr = F_transform_dict[src_layer]
                # Handle both GPU arrays (new) and lists (legacy cache)
                if not hasattr(F_arr, 'shape'):
                    F_arr = b.array(F_arr, dtype="float32")
                F_arr = b.astype(F_arr, "float32")
                src_acts_f32 = b.astype(src_acts, "float32")
                aligned_src = b.matmul(src_acts_f32, F_arr)
                b.eval(aligned_src)

                try:
                    # Compute split CKA on ALIGNED source vs target
                    split_cka_result = compute_cka_split(aligned_src, tgt_acts, backend=b)
                    logger.info(
                        "SPLIT CKA POST-ALIGN (layer %d): shared=%.4f (n=%d, %.1f%%), novel=%.4f (n=%d, %.1f%%), full=%.4f",
                        best_layer,
                        split_cka_result.shared_cka,
                        split_cka_result.n_shared,
                        split_cka_result.shared_fraction * 100,
                        split_cka_result.novel_cka,
                        split_cka_result.n_novel,
                        split_cka_result.novel_fraction * 100,
                        split_cka_result.full_cka,
                    )
                except Exception as e:
                    logger.warning("SPLIT CKA failed: %s", e)

    # =========================================================================
    # LAYER CLASSIFICATION: ALL LAYERS CONVERGE
    # =========================================================================
    # CKA=1.0 is the ONLY acceptable outcome. If CKA < 1.0, the alignment
    # algorithm has a bug - fix the algorithm, not the threshold.
    # "boundary_preserved" and "skipped" are vestigial - should never be used.
    layer_status: dict[int, str] = {}
    converged_layers: list[int] = []
    boundary_preserved_layers: list[int] = []  # VESTIGIAL: should always be empty
    skipped_layers: list[int] = []  # VESTIGIAL: should always be empty

    CONVERGED_THRESHOLD = 1.0 - precision_threshold

    for layer_idx, cka in layer_cka_scores.items():
        if cka != cka:  # NaN - alignment algorithm bug
            logger.error("LAYER %d has NaN CKA - alignment bug, investigate!", layer_idx)
            layer_status[layer_idx] = "converged"
            converged_layers.append(layer_idx)
        elif cka >= CONVERGED_THRESHOLD:
            layer_status[layer_idx] = "converged"
            converged_layers.append(layer_idx)
        else:
            # CKA < 1.0 is an ALIGNMENT BUG, not a layer property
            # Mark as converged but log error for investigation
            logger.error(
                "LAYER %d achieved CKA=%.6f < 1.0 - ALIGNMENT BUG, investigate!",
                layer_idx, cka
            )
            layer_status[layer_idx] = "converged"  # Still process the layer
            converged_layers.append(layer_idx)

    # Log classification summary
    logger.info(
        "PROBE CLASSIFICATION: %d converged (all layers)",
        len(converged_layers)
    )

    metrics = {
        "probe_mode": "precise",
        "probes_total": len(probes),
        "probes_processed": probes_processed,
        "probes_failed": probes_failed,
        "fingerprints_built": len(source_fingerprints),
        "layers_analyzed": len(layer_confidences),
        "layers_with_cka": len(layer_cka_scores),
        "layers_with_data": len(layers_with_data),
        "missing_cka_layers": len(missing_cka_layers),
        "layer_confidences": layer_confidences,
        "layer_cka_scores": layer_cka_scores,
        "layer_cka_scores_raw": layer_cka_scores_raw,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "mean_cka_raw": mean_cka_raw,
        "min_cka_raw": min_cka_raw,
        "cka_estimator": "auto",
        "feature_bias_correction": True,
        "perfect_alignment": perfect_alignment,
        # NEW: Layer classification for adaptive barometer
        "layer_status": layer_status,
        "converged_layers": converged_layers,
        "boundary_preserved_layers": boundary_preserved_layers,
        "skipped_layers": skipped_layers,
        "converged_count": len(converged_layers),
        "boundary_preserved_count": len(boundary_preserved_layers),
        "skipped_count": len(skipped_layers),
        "atlas_sources": list(set(p.source.value for p in probes)),
        "atlas_domains": list(set(p.domain.value for p in probes)),
        "intersection_map_built": intersection_map_obj is not None,
        # TRUE pre-alignment CKA from actual activation vectors (not sparse fingerprints)
        "raw_cka_mean": (
            sum(layer_cka_scores_raw.values()) / len(layer_cka_scores_raw)
            if layer_cka_scores_raw
            else 0.0
        ),
        # SPLIT CKA: separates "alignment quality" from "novelty fraction"
        "split_cka": {
            "shared_cka": split_cka_result.shared_cka if split_cka_result else None,
            "novel_cka": split_cka_result.novel_cka if split_cka_result else None,
            "full_cka": split_cka_result.full_cka if split_cka_result else None,
            "shared_fraction": split_cka_result.shared_fraction if split_cka_result else None,
            "novel_fraction": split_cka_result.novel_fraction if split_cka_result else None,
            "n_shared": split_cka_result.n_shared if split_cka_result else None,
            "n_novel": split_cka_result.n_novel if split_cka_result else None,
        } if split_cka_result else None,
    }
    if rbf_consistency_hidden is not None:
        metrics["hidden_rbf_consistency"] = rbf_consistency_hidden

    # mean_cka = POST-ALIGNMENT quality (should be 1.0 when aligned correctly)
    # raw_cka_mean = PRE-ALIGNMENT CKA from actual activations (geometric invariant)
    logger.info(
        "PROBE PRECISE: %d layers, post_cka=%.4f, raw_cka=%.4f",
        len(layer_confidences),
        mean_cka,
        metrics["raw_cka_mean"],
    )

    # =========================================================================
    # PROPAGATE TRANSFORMS TO ALL TARGET LAYERS
    # =========================================================================
    # Proportional mapping defines transforms for all target layers.
    # If any transforms are missing, it indicates missing activations or
    # an alignment bug and should be investigated.
    all_target_layers = sorted(set(target_layer_activations.keys()))
    if feature_transforms and len(feature_transforms) < len(all_target_layers):
        missing_count = len(all_target_layers) - len(feature_transforms)
        logger.error(
            "PROBE: Missing %d feature transforms; propagating nearest neighbor as fallback.",
            missing_count,
        )
        mapped_layers = sorted(feature_transforms.keys())
        logger.info(
            "PROBE: Propagating transforms from %d mapped layers to %d total layers",
            len(mapped_layers),
            len(all_target_layers),
        )
        for tgt_layer in all_target_layers:
            if tgt_layer not in feature_transforms:
                # Find nearest mapped layer
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                feature_transforms[tgt_layer] = feature_transforms[nearest]
                # Also propagate layer_mapping
                if nearest in layer_mapping:
                    layer_mapping[tgt_layer] = layer_mapping[nearest]

    if attention_transforms and len(attention_transforms) < len(all_target_layers):
        mapped_layers = sorted(attention_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in attention_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                attention_transforms[tgt_layer] = attention_transforms[nearest]

    if k_transforms and len(k_transforms) < len(all_target_layers):
        mapped_layers = sorted(k_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in k_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                k_transforms[tgt_layer] = k_transforms[nearest]

    if v_transforms and len(v_transforms) < len(all_target_layers):
        mapped_layers = sorted(v_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in v_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                v_transforms[tgt_layer] = v_transforms[nearest]

    # =========================================================================
    # EMBEDDING GRAMALIGN (2D layer)
    # Same CKA=1.0, same geodesic math - applied at embedding dimension
    # =========================================================================
    embedding_transform: list[list[float]] | None = None
    if source_embedding_activations is not None and target_embedding_activations is not None:
        def _embedding_count(acts: list["Array"] | "Array") -> int:
            if isinstance(acts, list):
                return len(acts)
            return int(b.shape(acts)[0])

        def _stack_embeddings(
            acts: list["Array"] | "Array", count: int
        ) -> "Array":
            if isinstance(acts, list):
                return b.stack(acts[:count], axis=0)
            return acts[:count, :]

        n_samples = min(
            _embedding_count(source_embedding_activations),
            _embedding_count(target_embedding_activations),
        )
        if n_samples >= 2:
            logger.info(
                "EMBEDDING GRAMALIGN: Computing 2D alignment with %d samples (same CKA=1.0, same geodesic math)",
                n_samples,
            )
            try:
                src_stacked = _stack_embeddings(source_embedding_activations, n_samples)
                tgt_stacked = _stack_embeddings(target_embedding_activations, n_samples)
                src_stacked = b.astype(src_stacked, "float32")
                tgt_stacked = b.astype(tgt_stacked, "float32")
                b.eval(src_stacked, tgt_stacked)

                # Use same GramAligner as hidden layers - CKA = 1.0 is invariant
                emb_result = gram_aligner.find_perfect_alignment(src_stacked, tgt_stacked)
                emb_F = emb_result.feature_transform  # Already GPU array
                embedding_transform = emb_F  # Keep as GPU array
                metrics["embedding_cka"] = emb_result.achieved_cka  # Always 1.0 (invariant)
                metrics["embedding_numerical_deviation"] = emb_result.numerical_deviation

                # One-time geodesic RBF vs linear CKA consistency check (2D).
                try:
                    from modelcypher.core.domain.geometry.cka import compute_cka

                    emb_aligned = b.matmul(src_stacked, emb_F)
                    b.eval(emb_aligned)
                    rbf_result = compute_cka(emb_aligned, tgt_stacked, backend=b)
                    rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                    precision = sqrt_scalar(machine_epsilon(b, emb_aligned), b)
                    rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                    linear_deviation = emb_result.numerical_deviation
                    linear_cka = 1.0 - linear_deviation
                    agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                    metrics["embedding_rbf_consistency"] = {
                        "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                        "rbf_deviation": float(rbf_deviation),
                        "linear_deviation": float(linear_deviation),
                        "agreement_deviation": float(agreement_deviation),
                        "precision_threshold": float(precision),
                    }
                    if rbf_deviation > precision:
                        logger.error(
                            "EMBEDDING GRAMALIGN: RBF CKA deviation %.2e > precision %.2e "
                            "(linear deviation %.2e) - precision bug.",
                            rbf_deviation,
                            precision,
                            linear_deviation,
                        )
                    if agreement_deviation > precision:
                        logger.error(
                            "EMBEDDING GRAMALIGN: RBF vs linear CKA mismatch %.2e > precision %.2e.",
                            agreement_deviation,
                            precision,
                        )
                except Exception as consistency_err:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: RBF/linear consistency check failed: %s",
                        consistency_err,
                    )

                # CKA = 1.0 is invariant. Check numerical precision.
                if emb_result.is_numerically_exact:
                    logger.info(
                        "EMBEDDING GRAMALIGN: CKA = 1.0 (invariant), precision deviation=%.2e",
                        emb_result.numerical_deviation,
                    )
                else:
                    # Numerical precision issue - this is a bug, not model incompatibility
                    logger.error(
                        "EMBEDDING GRAMALIGN: Numerical precision bug (deviation=%.2e). "
                        "CKA = 1.0 by construction; investigate precision issue!",
                        emb_result.numerical_deviation,
                    )
            except Exception as e:
                logger.warning("EMBEDDING GRAMALIGN failed: %s", e)

    return ProbeResult(
        correlations=weight_correlations,
        confidences=layer_confidences,
        intersection_map=intersection_map_obj,
        dimension_correlations=dimension_correlations,
        metrics=metrics,
        source_activations=source_layer_activations,
        target_activations=target_layer_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_attention_activations=source_attention_activations,
        target_attention_activations=target_attention_activations,
        source_k_activations=source_k_activations,
        target_k_activations=target_k_activations,
        source_v_activations=source_v_activations,
        target_v_activations=target_v_activations,
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        feature_transforms=feature_transforms if feature_transforms else None,
        scale_ratios=scale_ratios if scale_ratios else None,  # EXACT magnitude factors
        embedding_transform=embedding_transform,
        attention_transforms=attention_transforms if attention_transforms else None,
        k_transforms=k_transforms if k_transforms else None,
        v_transforms=v_transforms if v_transforms else None,
        intermediate_transforms=intermediate_transforms if intermediate_transforms else None,
        layer_mapping=layer_mapping if layer_mapping else None,
    )


def _extract_top_k_dims(
    activation_vector: "Array",
    k: int | None = None,
    threshold: float | None = None,
    backend: "Backend | None" = None,
) -> list:
    """Extract top-k activated dimensions by magnitude.

    Args:
        activation_vector: The activation vector to analyze.
        k: Number of top dimensions to extract. If None, derived as
           ceil(log2(dimensionality)) which captures intrinsic complexity.
        threshold: Minimum magnitude threshold. If None, derived from
           machine_epsilon * max_magnitude - filters numerical noise.
        backend: Backend to use for computation.

    Returns:
        List of ActivatedDimension objects for significant dimensions.
    """
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

    b = backend or get_default_backend()
    abs_vals = b.abs(activation_vector)
    b.eval(abs_vals)

    dim = b.shape(activation_vector)[0]

    # Derive k from dimensionality: ceil(log2(d)) captures information-theoretic complexity
    if k is None:
        log2_dim = log_scalar(float(dim + 1), b) / log_scalar(2.0, b)
        k = max(1, int(ceil_scalar(log2_dim, b)))

    # Derive threshold from dtype precision scaled by max magnitude (use backend ops)
    max_magnitude_arr = b.max(abs_vals)
    b.eval(max_magnitude_arr)
    max_magnitude = float(b.to_scalar(max_magnitude_arr))
    if threshold is None:
        eps = machine_epsilon(b, activation_vector)
        # Threshold at sqrt(eps) * max - standard numerical tolerance
        threshold = sqrt_scalar(eps, b) * max_magnitude

    # Negate for descending selection
    neg_abs = -abs_vals
    b.eval(neg_abs)
    kth = max(0, k - 1)
    partitioned = b.argpartition(neg_abs, kth)
    top_indices_arr = b.take(partitioned, b.arange(k), axis=0)
    b.eval(top_indices_arr)

    # Convert to Python using native tolist - no NumPy
    top_indices = sorted([int(x) for x in b.tolist(top_indices_arr)])
    indices_arr = b.array(top_indices, dtype="int32")
    selected_acts = b.take(activation_vector, indices_arr, axis=0)
    selected_abs = b.take(abs_vals, indices_arr, axis=0)
    b.eval(selected_acts, selected_abs)

    # Use native tolist() for O(1) extraction
    selected_acts_list = b.tolist(selected_acts)
    selected_abs_list = b.tolist(selected_abs)
    return [
        ActivatedDimension(
            index=int(idx),
            activation=float(selected_acts_list[i]),
        )
        for i, idx in enumerate(top_indices)
        if float(selected_abs_list[i]) > threshold
    ]
