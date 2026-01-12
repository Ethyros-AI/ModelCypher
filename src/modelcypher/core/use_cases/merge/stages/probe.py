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

We always run the atlas probe corpus from JSON, with the probe count derived
from geometry (hidden dimensions of source/target). Token probing and
weight-level shortcuts are intentionally disabled.

Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
Reference: Chun et al. (2025) "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features"
Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.use_cases.merge.stages.probe_activation_storage import (
    _page_activation_space,
)
from modelcypher.core.use_cases.merge.stages.probe_alignment import align_layers
from modelcypher.core.use_cases.merge.stages.probe_cache import (
    ModelProbeCache,
    _load_model_probe_cache,
    _save_model_probe_cache,
)
from modelcypher.core.use_cases.merge.stages.probe_checkpoint import (
    clear_probe_checkpoint as _clear_probe_checkpoint,
)
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _extract_top_k_dims,
    _infer_required_probe_count,
    _precision_reference,
    _promote_precision,
    _select_geometry_probes,
    _select_probe_text,
)
from modelcypher.core.use_cases.merge.stages.probe_inference import run_probe_inference

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Probe mode is ALWAYS activation-level CKA with atlas JSON probes.
# No token probing, no weight-level shortcuts.

# Checkpoint interval: save progress every N probes
_CHECKPOINT_INTERVAL = 50

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
    # Used for GramAlign at 2D interface - closed-form alignment + geodesic diagnostics
    source_embedding_activations: list[Any] | None = None
    target_embedding_activations: list[Any] | None = None
    probe_ids: list[str] | None = None
    probe_domains: list[str] | None = None
    # Layer alignment transforms: source_acts @ transforms[layer] -> aligned_source
    # Closed-form linear alignment on the shared manifold; geodesic CKA is diagnostic
    # Hidden-space transforms: for hidden dimension (e.g., 960 -> 896)
    feature_transforms: dict[int, list[list[float]]] | None = None
    # EXACT SCALE FACTOR per layer: ||target|| / ||source @ F||
    # Apply to stitched weights for exact magnitude match
    scale_ratios: dict[int, float] | None = None
    # Embedding-space transform: for embed_tokens alignment (linear alignment + geodesic diagnostics)
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
    activation_store: "ActivationStore | None" = None,
    backend: "Backend | None" = None,
    checkpoint_dir: Path | str | None = None,
    probe_mode: str = "atlas",  # Only "atlas" supported - atlas JSON probes
) -> ProbeResult:
    """
    Stage 1: Build intersection map from probe responses.

    ALWAYS uses precise mode (activation-level CKA) with atlas JSON probes.
    Probe count is derived from geometry (hidden dimensions), not user input.

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
    if probe_mode != "atlas":
        logger.warning("PROBE MODE: %s unsupported; forcing atlas", probe_mode)
        probe_mode = "atlas"

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
            
            # Report tokenizer alignment measurement (no interpretation)
            logger.info(
                "PRE-FLIGHT: Tokenizer vocabulary overlap Jaccard=%.2f",
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
            raise ValueError("Activation provider required for probe stage")

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
            activation_store=activation_store,
            source_path=source_path,
            target_path=target_path,
            backend=backend,
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
    activation_store: "ActivationStore | None" = None,
    source_path: str = "",
    target_path: str = "",
    backend: "Backend | None" = None,
    checkpoint_dir: Path | None = None,
    probe_mode: str = "atlas",  # Only "atlas" is supported - atlas JSON probes
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models.

    Uses Atlas JSON probes with a geometry-derived count (hidden dimensions)
    for broad manifold coverage without user configuration.

    Args:
        probe_mode: Must be "atlas" (kept for cache key compatibility).
        checkpoint_dir: If provided, saves progress periodically and allows
            resume from last checkpoint on restart.
        activation_store: Required if checkpoint_dir is provided.
    """
    b = backend or get_default_backend()
    from modelcypher.core.domain.geometry.intersection_similarity import (
        IntersectionSimilarityMode,
        build_intersection_map,
    )
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ActivatedDimension,
        ActivationFingerprint,
        IntersectionMap,
    )

    # Load Atlas probes for manifold coverage.
    # Probe count is derived from geometry (hidden dimensions).
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    probes = load_all_probes()
    logger.info("PROBE MODE: Atlas (%d probes total)", len(probes))

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

    required_count, source_dim, target_dim = _infer_required_probe_count(
        source_weights, target_weights
    )

    if required_count <= 0:
        logger.warning(
            "PROBE MODE: Hidden dims unavailable; using all %d valid probes",
            len(valid_probes),
        )
        selected_probes = list(valid_probes)
    elif required_count > len(valid_probes):
        logger.warning(
            "PROBE MODE: Geometry requires %d probes (src_dim=%d, tgt_dim=%d) "
            "but only %d valid probes available; using all probes",
            required_count,
            source_dim,
            target_dim,
            len(valid_probes),
        )
        selected_probes = list(valid_probes)
    else:
        selected_probes = _select_geometry_probes(valid_probes, required_count)
        logger.info(
            "PROBE MODE: Geometry-selected %d/%d probes (src_dim=%d, tgt_dim=%d)",
            len(selected_probes),
            len(valid_probes),
            source_dim,
            target_dim,
        )

    if invalid_probe_count:
        logger.info("PROBE MODE: Skipped %d probes with no usable text", invalid_probe_count)

    valid_probes = selected_probes
    expected_probe_ids = [probe.probe_id for probe, _ in valid_probes]
    expected_probe_domains = [probe.domain.value for probe, _ in valid_probes]
    probe_domain_by_id = dict(zip(expected_probe_ids, expected_probe_domains))
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
    probe_ids: list[str] = list(expected_probe_ids)
    probe_domains: list[str] = list(expected_probe_domains)

    probes_processed = 0
    probes_failed = invalid_probe_count
    checkpoint_path: Path | None = None  # Defined early so it's available for cleanup

    # =========================================================================
    # BATCHED PROBE COLLECTION - Process probes in batches for efficiency
    # =========================================================================

    if not run_inference:
        probes_processed = len(probe_ids)

    if run_inference:
        probes_processed, probes_failed, checkpoint_path = run_probe_inference(
            valid_probes=valid_probes,
            expected_probe_ids=expected_probe_ids,
            probe_domain_by_id=probe_domain_by_id,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            activation_provider=activation_provider,
            activation_store=activation_store,
            backend=b,
            checkpoint_dir=checkpoint_dir,
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
            source_embedding_activations=source_embedding_activations,
            target_embedding_activations=target_embedding_activations,
            source_fingerprints=source_fingerprints,
            target_fingerprints=target_fingerprints,
            run_source_inference=run_source_inference,
            run_target_inference=run_target_inference,
            invalid_probe_count=invalid_probe_count,
            checkpoint_interval=_CHECKPOINT_INTERVAL,
        )

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
        probe_ids == expected_probe_ids
        and probe_domains == expected_probe_domains
        and probes_processed == len(expected_probe_ids)
        and probes_failed == invalid_probe_count
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

    # Paging disabled by default - causes MLX SIGSEGV crashes in compile_replace
    # due to interaction between lazy evaluation and memory management.
    # Can be re-enabled with MC_PROBE_PAGE_ACTIVATIONS=1 if memory is tight.
    page_activations = (
        activation_store is not None
        and os.environ.get("MC_PROBE_PAGE_ACTIVATIONS", "0") == "1"
    )
    if page_activations:
        def _paged_dir(identity: Any | None, label: str) -> Path | None:
            if identity is not None:
                return (
                    profile_store.probe_cache_dir(identity.model_id)
                    / f"{probe_mode}_{probe_corpus_hash}_paged_{label}"
                )
            if checkpoint_dir is not None:
                return checkpoint_dir / f"{probe_mode}_{probe_corpus_hash}_paged_{label}"
            return None

        source_paged_dir = _paged_dir(source_identity, "source")
        if (
            source_paged_dir is not None
            and isinstance(source_layer_activations, dict)
            and source_layer_activations
        ):
            source_layer_activations = _page_activation_space(
                activation_store,
                source_paged_dir,
                "hidden",
                source_layer_activations,
                b,
            )
            logger.info("PROBE: Paged source hidden activations to %s", source_paged_dir)

        target_paged_dir = _paged_dir(target_identity, "target")
        if (
            target_paged_dir is not None
            and isinstance(target_layer_activations, dict)
            and target_layer_activations
        ):
            target_layer_activations = _page_activation_space(
                activation_store,
                target_paged_dir,
                "hidden",
                target_layer_activations,
                b,
            )
            logger.info("PROBE: Paged target hidden activations to %s", target_paged_dir)

        if (
            source_paged_dir is not None
            and isinstance(source_intermediate_activations, dict)
            and source_intermediate_activations
        ):
            source_intermediate_activations = _page_activation_space(
                activation_store,
                source_paged_dir,
                "intermediate",
                source_intermediate_activations,
                b,
            )
            logger.info(
                "PROBE: Paged source intermediate activations to %s", source_paged_dir
            )

        if (
            target_paged_dir is not None
            and isinstance(target_intermediate_activations, dict)
            and target_intermediate_activations
        ):
            target_intermediate_activations = _page_activation_space(
                activation_store,
                target_paged_dir,
                "intermediate",
                target_intermediate_activations,
                b,
            )
            logger.info(
                "PROBE: Paged target intermediate activations to %s", target_paged_dir
            )

        if hasattr(b, "clear_cache"):
            b.clear_cache()

    # Build IntersectionMap
    intersection_map_obj: IntersectionMap | None = None
    dimension_correlations: dict = {}
    layer_cka_scores: dict[int, float] = {}
    layer_cka_scores_raw: dict[int, float] = {}

    if source_fingerprints and target_fingerprints:
        try:
            logger.info(
                "PROBE: Building IntersectionMap from %d source + %d target fingerprints...",
                len(source_fingerprints),
                len(target_fingerprints),
            )
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
    # CKA is diagnostic; we do not use it as a selector.
    alignment_result = align_layers(
        source_layer_activations=source_layer_activations,
        target_layer_activations=target_layer_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        backend=b,
    )
    layer_mapping = alignment_result.layer_mapping
    feature_transforms = alignment_result.feature_transforms
    scale_ratios = alignment_result.scale_ratios
    attention_transforms = alignment_result.attention_transforms
    k_transforms = alignment_result.k_transforms
    v_transforms = alignment_result.v_transforms
    intermediate_transforms = alignment_result.intermediate_transforms
    layer_cka_scores = alignment_result.layer_cka_scores
    layer_cka_scores_raw = alignment_result.layer_cka_scores_raw
    cgls_iterations_by_layer = alignment_result.cgls_iterations_by_layer
    rbf_consistency_hidden = alignment_result.rbf_consistency_hidden

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
    # Linear CKA is diagnostic, not a gate. Filter only NaN (alignment bugs).
    # Low linear CKA usually means limited overlap or probe coverage.
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
    # LINEAR CKA DIAGNOSTIC (STRICT OVERLAP CHECK)
    # =========================================================================
    # perfect_alignment is a strict diagnostic: it only holds when every layer's
    # linear CKA is within precision. This is not required for merging and can
    # be false when models contain novel structure outside the shared manifold.
    # Use sqrt(machine_epsilon) as the tolerance (matches GramAligner convention).
    precision_ref = _precision_reference(
        b,
        feature_transforms,
        source_layer_activations,
        target_layer_activations,
    )
    precision_threshold = sqrt_scalar(machine_epsilon(b, precision_ref), b)
    perfect_alignment = bool(valid_cka_vals) and min_cka >= 1.0 - precision_threshold

    # =========================================================================
    # SPLIT CKA: SHARED VS. NOVEL CONCEPTS (POST-ALIGNMENT)
    # =========================================================================
    # CKA is meaningless BEFORE alignment - high-d representations get twisted
    # during pre-training. We must FIRST align, THEN measure shared vs novel.
    #
    # Compute CKA separately for:
    # - SHARED: concepts both models respond to (CKA should be high after alignment)
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
                if not hasattr(F_arr, "shape"):
                    F_arr = b.array(F_arr)
                F_arr = _promote_precision(F_arr, b)
                src_acts_precise = _promote_precision(src_acts, b)
                aligned_src = b.matmul(src_acts_precise, F_arr)
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
    # LAYER CLASSIFICATION: ALL LAYERS PROCESSED
    # =========================================================================
    # Linear CKA measures STRUCTURAL OVERLAP between source and target spaces.
    #
    # CKA ≈ 1.0: Source fully covers target's representational space (shared manifold)
    # CKA < 1.0: Target has structure outside source's column space (EXPECTED
    #            for cross-dimensional alignment, e.g., 896 → 1024 hidden dims)
    #
    # This is NOT an alignment bug - it's measuring how much of target's
    # geometry is captured by source. The null-space projection handles this
    # correctly: it preserves target's unique structure while adding source's.
    # =========================================================================
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
            # CKA < 1.0 means target has structure outside source's column space.
            # This is EXPECTED for cross-dimensional alignment (different hidden dims).
            # Not a bug - the null-space projection handles this correctly.
            logger.info(
                "LAYER %d: structural overlap CKA=%.4f (target has %.1f%% unique structure)",
                layer_idx, cka, (1.0 - cka) * 100
            )
            layer_status[layer_idx] = "converged"  # Still process the layer
            converged_layers.append(layer_idx)

    # Log classification summary
    logger.info(
        "PROBE CLASSIFICATION: %d processed (all layers)",
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
        "cgls_iterations_by_layer": cgls_iterations_by_layer,
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

    # mean_cka = Post-alignment linear overlap (shared-manifold coverage)
    # raw_cka_mean = Pre-alignment linear CKA from actual activations
    logger.info(
        "PROBE PRECISE: %d layers, post_linear_cka=%.4f, raw_linear_cka=%.4f",
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

    gram_aligner = GramAligner(backend=b)

    # =========================================================================
    # EMBEDDING GRAMALIGN (2D layer)
    # Same closed-form alignment; geodesic CKA is diagnostic.
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
                "EMBEDDING GRAMALIGN: Computing 2D alignment with %d samples (linear alignment + geodesic diagnostics)",
                n_samples,
            )
            try:
                src_stacked = _stack_embeddings(source_embedding_activations, n_samples)
                tgt_stacked = _stack_embeddings(target_embedding_activations, n_samples)
                src_stacked = _promote_precision(src_stacked, b)
                tgt_stacked = _promote_precision(tgt_stacked, b)
                b.eval(src_stacked, tgt_stacked)

                # Use same GramAligner as hidden layers; linear CKA is diagnostic.
                emb_result = gram_aligner.find_perfect_alignment(src_stacked, tgt_stacked)
                emb_F = emb_result.feature_transform  # Already GPU array
                embedding_transform = emb_F  # Keep as GPU array

                from modelcypher.core.domain.geometry.cka import compute_linear_cka
                emb_aligned = b.matmul(src_stacked, emb_F)
                b.eval(emb_aligned)
                emb_linear_cka = float(compute_linear_cka(emb_aligned, tgt_stacked, backend=b))
                emb_linear_deviation = abs(1.0 - emb_linear_cka)

                metrics["embedding_cka"] = emb_linear_cka
                metrics["embedding_geodesic_cka"] = emb_result.achieved_cka
                metrics["embedding_numerical_deviation"] = emb_linear_deviation

                # One-time geodesic RBF vs linear CKA consistency check (2D).
                try:
                    from modelcypher.core.domain.geometry.cka import compute_cka

                    # emb_aligned already computed for linear CKA above
                    rbf_result = compute_cka(emb_aligned, tgt_stacked, backend=b)
                    rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                    precision = sqrt_scalar(machine_epsilon(b, emb_aligned), b)
                    rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                    linear_cka = emb_linear_cka
                    linear_deviation = emb_linear_deviation
                    agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                    metrics["embedding_rbf_consistency"] = {
                        "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                        "rbf_deviation": float(rbf_deviation),
                        "linear_deviation": float(linear_deviation),
                        "agreement_deviation": float(agreement_deviation),
                        "precision_threshold": float(precision),
                    }
                    if linear_deviation > precision:
                        logger.error(
                            "EMBEDDING GRAMALIGN: Linear CKA deviation %.2e > precision %.2e.",
                            linear_deviation,
                            precision,
                        )
                    if rbf_deviation > precision:
                        logger.info(
                            "EMBEDDING GRAMALIGN: Geodesic CKA deviation %.2e > precision %.2e.",
                            rbf_deviation,
                            precision,
                        )
                    if agreement_deviation > precision:
                        logger.info(
                            "EMBEDDING GRAMALIGN: Geodesic vs linear CKA deviation %.2e > precision %.2e.",
                            agreement_deviation,
                            precision,
                        )
                except Exception as consistency_err:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: RBF/linear consistency check failed: %s",
                        consistency_err,
                    )

                # Linear CKA is the diagnostic check for shared-manifold alignment.
                if emb_linear_deviation > precision:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: Linear CKA deviation %.2e > precision %.2e.",
                        emb_linear_deviation,
                        precision,
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
