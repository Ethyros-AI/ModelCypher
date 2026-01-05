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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    geodesic_svd,
    log_scalar,
    machine_epsilon,
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

    We save probe IDs and metadata, not activation data (too large).
    The activation dicts are rebuilt on resume by re-running probes.
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
    # Embedding-space transform: for embed_tokens alignment (same CKA=1.0, same geodesic math)
    embedding_transform: list[list[float]] | None = None
    # Attention Q-space transforms: for q_proj/o_proj (e.g., 960 -> 896 for Q heads)
    attention_transforms: dict[int, list[list[float]]] | None = None
    # Attention K-space transforms: for k_proj (granular alignment)
    k_transforms: dict[int, list[list[float]]] | None = None
    # Attention V-space transforms: for v_proj (granular alignment)
    v_transforms: dict[int, list[list[float]]] | None = None
    # Layer mapping: target_layer -> source_layer (from DP alignment)
    layer_mapping: dict[int, int] | None = None


def stage_probe(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    extract_layer_index_fn: Callable[[str], int | None],
    source_model: Any | None = None,
    target_model: Any | None = None,
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    tokenizer: Any | None = None,
    collect_activations_fn: Callable | None = None,
    activation_provider: "ActivationProvider | None" = None,
    backend: "Backend | None" = None,
    checkpoint_dir: Path | str | None = None,
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
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
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
) -> ProbeResult:
    """Precise probe mode: Run ALL probes through BOTH models.

    No configuration - all 403 probes are always run. The probe corpus
    was designed to cover the concept space. Limiting probes degrades
    coverage with no benefit.

    Args:
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

    # Always use all probes - no configuration
    probes = UnifiedAtlasInventory.all_probes()
    num_probes = len(probes)

    logger.info(
        "PROBE PRECISE: Running %d probes through source and target models...",
        len(probes),
    )

    source_fingerprints: list[ActivationFingerprint] = []
    target_fingerprints: list[ActivationFingerprint] = []

    source_layer_activations: dict[int, list["Array"]] = {}
    target_layer_activations: dict[int, list["Array"]] = {}
    # Embedding-space activations for 2D GramAlign (post-embed_tokens, pre-layer-0)
    # Same CKA=1.0, same geodesic math - applied at embedding dimension
    source_embedding_activations: list["Array"] = []
    target_embedding_activations: list["Array"] = []
    # Intermediate-space activations for multi-space stitching (cross-architecture merges)
    source_intermediate_activations: dict[int, list["Array"]] = {}
    target_intermediate_activations: dict[int, list["Array"]] = {}
    # Q Attention-space activations for q_proj/o_proj stitching (cross-architecture merges)
    source_attention_activations: dict[int, list["Array"]] = {}
    target_attention_activations: dict[int, list["Array"]] = {}
    # K Attention-space activations for k_proj stitching (separate for granular alignment)
    source_k_activations: dict[int, list["Array"]] = {}
    target_k_activations: dict[int, list["Array"]] = {}
    # V Attention-space activations for v_proj stitching (separate for granular alignment)
    source_v_activations: dict[int, list["Array"]] = {}
    target_v_activations: dict[int, list["Array"]] = {}
    probe_ids: list[str] = []
    probe_domains: list[str] = []

    probes_processed = 0
    probes_failed = 0

    # =========================================================================
    # BATCHED PROBE COLLECTION - Process probes in batches for efficiency
    # =========================================================================
    # Instead of 806 individual forward passes (403 probes × 2 models),
    # we batch probes into groups and use batch methods for ~5-10× speedup.
    #
    # The batch size is tuned for GPU memory vs throughput tradeoff.
    # Too large = OOM, too small = lose batching benefit.
    PROBE_BATCH_SIZE = 8

    # First pass: Validate probes and extract texts
    valid_probes: list[tuple[Any, str]] = []  # (probe, probe_text)
    for probe in probes:
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

        if probe_text is None:
            logger.warning(
                "Probe '%s' skipped: no valid support texts or fallback",
                probe.probe_id,
            )
            probes_failed += 1
            continue

        valid_probes.append((probe, probe_text))

    logger.info(
        "PROBE PRECISE: %d valid probes, processing in batches of %d...",
        len(valid_probes),
        PROBE_BATCH_SIZE,
    )

    # =========================================================================
    # CHECKPOINT: Check for existing checkpoint to resume from
    # =========================================================================
    checkpoint_path: Path | None = None
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

    # Check if activation provider supports batch methods
    has_batch_hidden = hasattr(activation_provider, "collect_hidden_activations_batch")
    has_batch_intermediate = hasattr(activation_provider, "collect_intermediate_activations_batch")
    has_batch_attention = hasattr(activation_provider, "collect_attention_activations_batch")

    # Second pass: Batch forward passes
    for batch_start in range(0, len(valid_probes), PROBE_BATCH_SIZE):
        batch_end = min(batch_start + PROBE_BATCH_SIZE, len(valid_probes))
        batch = valid_probes[batch_start:batch_end]
        batch_texts = [probe_text for _, probe_text in batch]

        try:
            # ===== HIDDEN ACTIVATIONS =====
            if has_batch_hidden:
                source_hidden_batch = activation_provider.collect_hidden_activations_batch(
                    source_model, source_tokenizer, batch_texts
                )
                target_hidden_batch = activation_provider.collect_hidden_activations_batch(
                    target_model, target_tokenizer, batch_texts
                )
            else:
                # Fallback to sequential
                source_hidden_batch = [
                    activation_provider.collect_hidden_activations(source_model, source_tokenizer, text)
                    for text in batch_texts
                ]
                target_hidden_batch = [
                    activation_provider.collect_hidden_activations(target_model, target_tokenizer, text)
                    for text in batch_texts
                ]

            # ===== EMBEDDING ACTIVATIONS (2D GramAlign) =====
            # Same CKA=1.0, same geodesic math - applied at embedding dimension
            has_embedding = hasattr(activation_provider, "collect_embedding_activations")
            if has_embedding:
                for text in batch_texts:
                    source_emb = activation_provider.collect_embedding_activations(
                        source_model, source_tokenizer, text
                    )
                    target_emb = activation_provider.collect_embedding_activations(
                        target_model, target_tokenizer, text
                    )
                    source_embedding_activations.append(source_emb)
                    target_embedding_activations.append(target_emb)

            # ===== INTERMEDIATE ACTIVATIONS =====
            if has_batch_intermediate:
                source_intermediate_batch = activation_provider.collect_intermediate_activations_batch(
                    source_model, source_tokenizer, batch_texts
                )
                target_intermediate_batch = activation_provider.collect_intermediate_activations_batch(
                    target_model, target_tokenizer, batch_texts
                )
            else:
                source_intermediate_batch = [
                    activation_provider.collect_intermediate_activations(source_model, source_tokenizer, text)
                    for text in batch_texts
                ]
                target_intermediate_batch = [
                    activation_provider.collect_intermediate_activations(target_model, target_tokenizer, text)
                    for text in batch_texts
                ]

            # ===== ATTENTION ACTIVATIONS (Q, K, V separately) =====
            if has_batch_attention:
                source_q_batch, source_k_batch, source_v_batch = activation_provider.collect_attention_activations_batch(
                    source_model, source_tokenizer, batch_texts
                )
                target_q_batch, target_k_batch, target_v_batch = activation_provider.collect_attention_activations_batch(
                    target_model, target_tokenizer, batch_texts
                )
            else:
                source_q_batch, source_k_batch, source_v_batch = [], [], []
                target_q_batch, target_k_batch, target_v_batch = [], [], []
                for text in batch_texts:
                    src_q, src_k, src_v = activation_provider.collect_attention_activations(
                        source_model, source_tokenizer, text
                    )
                    tgt_q, tgt_k, tgt_v = activation_provider.collect_attention_activations(
                        target_model, target_tokenizer, text
                    )
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

                for layer_idx, act in source_acts.items():
                    source_activated[layer_idx] = _extract_top_k_dims(act, backend=b)
                    if layer_idx not in source_layer_activations:
                        source_layer_activations[layer_idx] = []
                    source_layer_activations[layer_idx].append(act)

                for layer_idx, act in target_acts.items():
                    target_activated[layer_idx] = _extract_top_k_dims(act, backend=b)
                    if layer_idx not in target_layer_activations:
                        target_layer_activations[layer_idx] = []
                    target_layer_activations[layer_idx].append(act)

                # Store intermediate activations for multi-space stitching
                for layer_idx, act in source_intermediate_acts.items():
                    if layer_idx not in source_intermediate_activations:
                        source_intermediate_activations[layer_idx] = []
                    source_intermediate_activations[layer_idx].append(act)

                for layer_idx, act in target_intermediate_acts.items():
                    if layer_idx not in target_intermediate_activations:
                        target_intermediate_activations[layer_idx] = []
                    target_intermediate_activations[layer_idx].append(act)

                # Store Q attention activations for q_proj/o_proj stitching
                for layer_idx, act in source_attention_acts.items():
                    if layer_idx not in source_attention_activations:
                        source_attention_activations[layer_idx] = []
                    source_attention_activations[layer_idx].append(act)

                for layer_idx, act in target_attention_acts.items():
                    if layer_idx not in target_attention_activations:
                        target_attention_activations[layer_idx] = []
                    target_attention_activations[layer_idx].append(act)

                # Store K attention activations for k_proj stitching
                for layer_idx, act in source_k_acts.items():
                    if layer_idx not in source_k_activations:
                        source_k_activations[layer_idx] = []
                    source_k_activations[layer_idx].append(act)

                for layer_idx, act in target_k_acts.items():
                    if layer_idx not in target_k_activations:
                        target_k_activations[layer_idx] = []
                    target_k_activations[layer_idx].append(act)

                # Store V attention activations for v_proj stitching
                for layer_idx, act in source_v_acts.items():
                    if layer_idx not in source_v_activations:
                        source_v_activations[layer_idx] = []
                    source_v_activations[layer_idx].append(act)

                for layer_idx, act in target_v_acts.items():
                    if layer_idx not in target_v_activations:
                        target_v_activations[layer_idx] = []
                    target_v_activations[layer_idx].append(act)

                source_fingerprints.append(
                    ActivationFingerprint(
                        prime_id=probe.probe_id,
                        prime_text=probe.name,
                        activated_dimensions=source_activated,
                    )
                )
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

            # Save checkpoint periodically
            if checkpoint_path is not None and probes_processed % _CHECKPOINT_INTERVAL < PROBE_BATCH_SIZE:
                _save_probe_checkpoint(
                    checkpoint_path=checkpoint_path,
                    completed_probes=probes_processed,
                    probe_ids=probe_ids,
                    probe_domains=probe_domains,
                    total_probes=len(valid_probes),
                )

        except Exception as e:
            # On batch failure, fall back to sequential processing for this batch
            logger.warning("Batch processing failed, falling back to sequential: %s", e)
            for probe, probe_text in batch:
                # Skip probes that were already completed (from checkpoint)
                if probe.probe_id in completed_probe_ids:
                    continue

                try:
                    source_acts = activation_provider.collect_hidden_activations(
                        source_model, source_tokenizer, probe_text
                    )
                    target_acts = activation_provider.collect_hidden_activations(
                        target_model, target_tokenizer, probe_text
                    )
                    source_intermediate_acts = activation_provider.collect_intermediate_activations(
                        source_model, source_tokenizer, probe_text
                    )
                    target_intermediate_acts = activation_provider.collect_intermediate_activations(
                        target_model, target_tokenizer, probe_text
                    )
                    source_attention_acts, source_k_acts, source_v_acts = activation_provider.collect_attention_activations(
                        source_model, source_tokenizer, probe_text
                    )
                    target_attention_acts, target_k_acts, target_v_acts = activation_provider.collect_attention_activations(
                        target_model, target_tokenizer, probe_text
                    )

                    source_activated_fallback: dict[int, list[ActivatedDimension]] = {}
                    target_activated_fallback: dict[int, list[ActivatedDimension]] = {}

                    for layer_idx, act in source_acts.items():
                        source_activated_fallback[layer_idx] = _extract_top_k_dims(act, backend=b)
                        if layer_idx not in source_layer_activations:
                            source_layer_activations[layer_idx] = []
                        source_layer_activations[layer_idx].append(act)

                    for layer_idx, act in target_acts.items():
                        target_activated_fallback[layer_idx] = _extract_top_k_dims(act, backend=b)
                        if layer_idx not in target_layer_activations:
                            target_layer_activations[layer_idx] = []
                        target_layer_activations[layer_idx].append(act)

                    for layer_idx, act in source_intermediate_acts.items():
                        if layer_idx not in source_intermediate_activations:
                            source_intermediate_activations[layer_idx] = []
                        source_intermediate_activations[layer_idx].append(act)

                    for layer_idx, act in target_intermediate_acts.items():
                        if layer_idx not in target_intermediate_activations:
                            target_intermediate_activations[layer_idx] = []
                        target_intermediate_activations[layer_idx].append(act)

                    for layer_idx, act in source_attention_acts.items():
                        if layer_idx not in source_attention_activations:
                            source_attention_activations[layer_idx] = []
                        source_attention_activations[layer_idx].append(act)

                    for layer_idx, act in target_attention_acts.items():
                        if layer_idx not in target_attention_activations:
                            target_attention_activations[layer_idx] = []
                        target_attention_activations[layer_idx].append(act)

                    for layer_idx, act in source_k_acts.items():
                        if layer_idx not in source_k_activations:
                            source_k_activations[layer_idx] = []
                        source_k_activations[layer_idx].append(act)

                    for layer_idx, act in target_k_acts.items():
                        if layer_idx not in target_k_activations:
                            target_k_activations[layer_idx] = []
                        target_k_activations[layer_idx].append(act)

                    for layer_idx, act in source_v_acts.items():
                        if layer_idx not in source_v_activations:
                            source_v_activations[layer_idx] = []
                        source_v_activations[layer_idx].append(act)

                    for layer_idx, act in target_v_acts.items():
                        if layer_idx not in target_v_activations:
                            target_v_activations[layer_idx] = []
                        target_v_activations[layer_idx].append(act)

                    source_fingerprints.append(
                        ActivationFingerprint(
                            prime_id=probe.probe_id,
                            prime_text=probe.name,
                            activated_dimensions=source_activated_fallback,
                        )
                    )
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

    # Compute CKA matrix for ALL source-target layer pairs, then use DP alignment
    # to find the correct layer correspondence. Layer indices don't match across
    # architectures - we must find the geometric correspondence.
    #
    # CRITICAL: Use GramAligner to find the transformation that achieves CKA = 1.0.
    # Raw CKA will be < 1.0 because coordinate systems differ. GramAligner FINDS
    # the correct transformation. If it can't achieve CKA = 1.0, the algorithm is broken.
    layer_mapping: dict[int, int] = {}  # target_layer -> source_layer
    feature_transforms: dict[int, list[list[float]]] = {}  # target_layer -> hidden transform
    attention_transforms: dict[int, list[list[float]]] = {}  # target_layer -> Q attention transform
    k_transforms: dict[int, list[list[float]]] = {}  # target_layer -> K attention transform
    v_transforms: dict[int, list[list[float]]] = {}  # target_layer -> V attention transform
    gram_aligner = GramAligner(backend=b)

    if source_layer_activations and target_layer_activations:
        source_layers = sorted(source_layer_activations.keys())
        target_layers = sorted(target_layer_activations.keys())
        n_source = len(source_layers)
        n_target = len(target_layers)

        if n_source > 0 and n_target > 0:
            # Build full CKA similarity matrix [n_source x n_target]
            # Use RAW CKA for layer matching (to find correspondence)
            # Then use GramAligner for the matched pairs (to achieve CKA = 1.0)
            cka_matrix: list[list[float]] = []
            for src_layer in source_layers:
                row: list[float] = []
                src_list = source_layer_activations[src_layer]
                for tgt_layer in target_layers:
                    tgt_list = target_layer_activations[tgt_layer]
                    n_samples = min(len(src_list), len(tgt_list))
                    if n_samples < 2:
                        row.append(0.0)
                        continue
                    try:
                        src_stacked = b.stack(src_list[:n_samples], axis=0)
                        tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                        # Convert to float32 for numerical stability
                        # float16 from model outputs causes SVD/eigendecomposition issues
                        src_stacked = b.astype(src_stacked, "float32")
                        tgt_stacked = b.astype(tgt_stacked, "float32")
                        b.eval(src_stacked, tgt_stacked)
                        cka_result = compute_cka(
                            src_stacked,
                            tgt_stacked,
                            backend=b,
                            estimator=HSICEstimator.AUTO,
                            feature_bias_correction=True,
                        )
                        if cka_result.is_valid:
                            row.append(
                                cka_result.cka_corrected
                                if cka_result.cka_corrected is not None
                                else cka_result.cka
                            )
                        else:
                            row.append(0.0)
                    except Exception:
                        row.append(0.0)
                cka_matrix.append(row)

            # Use DP to find optimal monotonic layer alignment
            dp_path, _ = CrossArchitectureLayerMatcher._dynamic_programming_alignment(
                cka_matrix
            )

            # =========================================================================
            # PARALLEL LAYER ALIGNMENT - Process layer pairs concurrently
            # =========================================================================
            # Each layer alignment is independent, so we can parallelize using a
            # thread pool. MLX operations submitted from multiple threads will be
            # efficiently scheduled on the GPU command queue.
            #
            # Benefits:
            # - Overlap CPU preparation with GPU computation
            # - Better GPU utilization through concurrent kernel submission
            # - 2-4× speedup for multi-layer alignments
            from concurrent.futures import ThreadPoolExecutor, as_completed

            ALIGNMENT_MAX_WORKERS = 1  # Reduced from 4 - MLX segfaults with concurrent GPU access

            def _align_target_group(
                tgt_idx: int,
                src_indices: list[int],
            ) -> dict:
                """Align a group of source layers to a single target layer via concatenation."""
                tgt_layer = target_layers[tgt_idx]
                src_layers = [source_layers[i] for i in src_indices]
                
                result: dict = {
                    "tgt_layer": tgt_layer,
                    "src_layers": src_layers,
                    "raw_cka": 0.0,
                    "achieved_cka": 0.0,
                    "feature_transform": None,  # Will be a dict {src_layer: transform}
                    "attention_transform": None, # Dict
                    "k_transform": None, # Dict
                    "v_transform": None, # Dict
                    "error": None,
                }

                # -----------------------------------------------------------
                # 1. Prepare Activations (Concat Source)
                # -----------------------------------------------------------
                src_act_lists = [source_layer_activations[s] for s in src_layers]
                tgt_list = target_layer_activations[tgt_layer]
                
                # Find common sample count
                n_samples = len(tgt_list)
                for s_list in src_act_lists:
                    n_samples = min(n_samples, len(s_list))
                
                if n_samples < 2:
                    return result

                # Create thread-local GramAligner
                local_aligner = GramAligner(backend=b)

                try:
                    # Stack samples for each source
                    src_stacks = []
                    src_dims = []
                    for s, s_list in zip(src_layers, src_act_lists):
                        stack = b.stack(s_list[:n_samples], axis=0)
                        stack = b.astype(stack, "float32")
                        src_stacks.append(stack)
                        src_dims.append(stack.shape[1])
                    
                    # Stack target
                    tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                    tgt_stacked = b.astype(tgt_stacked, "float32")
                    
                    # Concatenate source features: [N, D1] + [N, D2] -> [N, D1+D2]
                    src_combined = b.concatenate(src_stacks, axis=1)
                    
                    b.eval(src_combined, tgt_stacked)
                    
                    # -----------------------------------------------------------
                    # 2. Compute Match Metrics
                    # -----------------------------------------------------------
                    # Calculate raw CKA of the COMBINED representation
                    # This should be much higher than individual layers
                    from modelcypher.core.domain.geometry.cka import compute_cka_backend
                    result["raw_cka"] = float(compute_cka_backend(src_combined, tgt_stacked, b))

                    # -----------------------------------------------------------
                    # 3. Align Hidden States
                    # -----------------------------------------------------------
                    alignment_result = local_aligner.find_perfect_alignment(
                        src_combined,
                        tgt_stacked,
                    )
                    
                    result["achieved_cka"] = alignment_result.achieved_cka
                    
                    # Split the composite transform back to per-source transforms
                    composite_F = alignment_result.feature_transform
                    # F is [Sum(Ds), Dt]
                    # We split Sum(Ds) back into slices
                    
                    # Convert list to array for slicing if needed, but it's likely a list/array from GramAligner
                    # GramAligner returns list (serialization ready). Convert to array for splitting.
                    F_arr = b.array(composite_F)
                    
                    split_transforms = {}
                    start_idx = 0
                    for s_layer, s_dim in zip(src_layers, src_dims):
                        # F_slice: [s_dim, Dt]
                        F_slice = F_arr[start_idx : start_idx + s_dim, :]
                        split_transforms[s_layer] = b.tolist(F_slice)
                        start_idx += s_dim
                        
                    result["feature_transform"] = split_transforms

                    if not alignment_result.is_perfect:
                        logger.warning(
                            "PROBE: Group %s -> %d hidden alignment not exact (CKA=%.4f).",
                            src_layers, tgt_layer, alignment_result.achieved_cka
                        )

                    # -----------------------------------------------------------
                    # 4. Attention Q/K/V (Optional)
                    # -----------------------------------------------------------
                    # Only if ALL source layers have attention data? Or ANY?
                    # Generally corresponding layers should all have attention if one does.
                    
                    def align_subcomponent(attr_name, src_dict, tgt_dict):
                        # Use same n_samples
                        if tgt_layer not in tgt_dict:
                            return None
                        
                        sub_src_stacks = []
                        sub_src_dims = []
                        valid_src_layers = []
                        
                        for s in src_layers:
                            if s in src_dict:
                                s_list = src_dict[s]
                                if len(s_list) >= n_samples:
                                    st = b.stack(s_list[:n_samples], axis=0)
                                    st = b.astype(st, "float32")
                                    sub_src_stacks.append(st)
                                    sub_src_dims.append(st.shape[1])
                                    valid_src_layers.append(s)
                        
                        if not sub_src_stacks:
                            return None
                            
                        # Concatenate
                        sub_src_comb = b.concatenate(sub_src_stacks, axis=1)
                        tgt_st = b.stack(tgt_dict[tgt_layer][:n_samples], axis=0)
                        tgt_st = b.astype(tgt_st, "float32")
                        b.eval(sub_src_comb, tgt_st)
                        
                        res = local_aligner.find_perfect_alignment(sub_src_comb, tgt_st)
                        
                        # Split
                        F_sub = b.array(res.feature_transform)
                        sub_splits = {}
                        idx = 0
                        for s_lay, s_d in zip(valid_src_layers, sub_src_dims):
                            sli = F_sub[idx : idx + s_d, :]
                            sub_splits[s_lay] = b.tolist(sli)
                            idx += s_d
                            
                        return sub_splits

                    result["attention_transform"] = align_subcomponent(
                        "attention_transform", source_attention_activations, target_attention_activations
                    )
                    result["k_transform"] = align_subcomponent(
                        "k_transform", source_k_activations, target_k_activations
                    )
                    result["v_transform"] = align_subcomponent(
                        "v_transform", source_v_activations, target_v_activations
                    )

                except Exception as e:
                    result["error"] = str(e)
                    logger.warning(
                        "PROBE: GramAligner failed for group %s -> %d: %s",
                        src_layers, tgt_layer, e
                    )

                return result

            # Group source layers by target layer (for many-to-one alignment)
            target_to_sources: dict[int, list[int]] = {}
            for src_idx, tgt_idx in dp_path:
                if tgt_idx not in target_to_sources:
                    target_to_sources[tgt_idx] = []
                target_to_sources[tgt_idx].append(src_idx)

            # Define alignment tasks
            alignment_tasks = []
            for tgt_idx, src_indices in target_to_sources.items():
                alignment_tasks.append((tgt_idx, src_indices))

            # Submit grouped alignments
            logger.info(
                "PROBE: Aligning %d target layers (from %d total mappings)...",
                len(alignment_tasks),
                len(dp_path),
            )

            with ThreadPoolExecutor(max_workers=ALIGNMENT_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _align_target_group,
                        tgt_idx,
                        src_indices,
                    ): tgt_idx
                    for tgt_idx, src_indices in alignment_tasks
                }

                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    tgt_layer = result["tgt_layer"]
                    src_layers = result["src_layers"]

                    # Store mapping (using principal source layer - usually the first or middle)
                    # For metrics, we just log the primary mapping, but the transforms cover all
                    # This logic handles the 1:1 assumption in some downstream parts, which might need update
                    # But feature_transforms will now contain a composite transform or list
                    
                    # NOTE: To maintain compatibility with existing pipeline which expects 1:1 mapping in some places,
                    # we map to the first source layer. But the TRANSFORM handles the combination.
                    layer_mapping[tgt_layer] = src_layers[0]

                    if result["feature_transform"] is not None:
                        # feature_transform is now [sum(d_s), d_t]
                        feature_transforms[tgt_layer] = result["feature_transform"]
                        layer_cka_scores[tgt_layer] = result["achieved_cka"]
                    else:
                        layer_cka_scores[tgt_layer] = result["raw_cka"]

                    layer_cka_scores_raw[tgt_layer] = result["raw_cka"]

                    if result["attention_transform"] is not None:
                        attention_transforms[tgt_layer] = result["attention_transform"]

                    if result.get("k_transform") is not None:
                        k_transforms[tgt_layer] = result["k_transform"]

                    if result.get("v_transform") is not None:
                        v_transforms[tgt_layer] = result["v_transform"]

                    completed += 1
                    if completed % 5 == 0 or completed == len(alignment_tasks):
                        logger.info(
                            "PROBE: Aligned %d/%d target layers...",
                            completed,
                            len(alignment_tasks),
                        )

            logger.info(
                "PROBE: Cross-architecture layer alignment found %d mappings "
                "(source: %d layers, target: %d layers)",
                len(dp_path),
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
    mean_cka = sum(cka_vals) / len(cka_vals) if cka_vals else 0.0
    min_cka = min(cka_vals) if cka_vals else 0.0
    raw_cka_vals = list(layer_cka_scores_raw.values())
    mean_cka_raw = sum(raw_cka_vals) / len(raw_cka_vals) if raw_cka_vals else 0.0
    min_cka_raw = min(raw_cka_vals) if raw_cka_vals else 0.0
    # layers_with_data: layers that have activations in both models (for reporting)
    layers_with_data = set(source_layer_activations.keys()) & set(target_layer_activations.keys())
    # For cross-architecture, DP alignment only matches a subset of layers.
    # missing_cka_layers is for reporting - it doesn't block exact alignment
    missing_cka_layers = [layer for layer in layers_with_data if layer not in layer_cka_scores]
    # Exact alignment: all ALIGNED layers (in layer_cka_scores) have CKA >= 1.0 - threshold
    # The threshold is sqrt(machine_epsilon) ≈ 1e-4 for float32
    precision_threshold = sqrt_scalar(machine_epsilon(b, b.array([1.0])), b)
    perfect_alignment = bool(layer_cka_scores) and min_cka >= 1.0 - precision_threshold

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
        "atlas_sources": list(set(p.source.value for p in probes)),
        "atlas_domains": list(set(p.domain.value for p in probes)),
        "intersection_map_built": intersection_map_obj is not None,
        # TRUE pre-alignment CKA from actual activation vectors (not sparse fingerprints)
        "raw_cka_mean": (
            sum(layer_cka_scores_raw.values()) / len(layer_cka_scores_raw)
            if layer_cka_scores_raw
            else 0.0
        ),
    }

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
    # Cross-architecture DP alignment may map multiple source layers to few
    # target layers (e.g., 24 Qwen layers → 3 SmolLM layers). But we need
    # transforms for ALL target layers to perform weight transplant.
    #
    # Strategy: For target layers without a transform, use nearest neighbor.
    # This is geometrically sound because adjacent layers encode similar
    # manifold regions - their transforms should be similar.
    all_target_layers = sorted(set(target_layer_activations.keys()))
    if feature_transforms and len(feature_transforms) < len(all_target_layers):
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
    if source_embedding_activations and target_embedding_activations:
        n_samples = min(len(source_embedding_activations), len(target_embedding_activations))
        if n_samples >= 2:
            logger.info(
                "EMBEDDING GRAMALIGN: Computing 2D alignment with %d samples (same CKA=1.0, same geodesic math)",
                n_samples,
            )
            try:
                src_stacked = b.stack(source_embedding_activations[:n_samples], axis=0)
                tgt_stacked = b.stack(target_embedding_activations[:n_samples], axis=0)
                src_stacked = b.astype(src_stacked, "float32")
                tgt_stacked = b.astype(tgt_stacked, "float32")
                b.eval(src_stacked, tgt_stacked)

                # Use same GramAligner as hidden layers - same math, same CKA=1.0 target
                emb_result = gram_aligner.find_perfect_alignment(src_stacked, tgt_stacked)
                if emb_result.achieved_cka >= 0.99:
                    embedding_transform = b.tolist(b.array(emb_result.feature_transform))
                    logger.info(
                        "EMBEDDING GRAMALIGN: CKA = %.4f (same geometry preserved at 2D)",
                        emb_result.achieved_cka,
                    )
                    metrics["embedding_cka"] = emb_result.achieved_cka
                else:
                    logger.warning(
                        "EMBEDDING GRAMALIGN: CKA = %.4f < 0.99 - alignment failed",
                        emb_result.achieved_cka,
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
        embedding_transform=embedding_transform,
        attention_transforms=attention_transforms if attention_transforms else None,
        k_transforms=k_transforms if k_transforms else None,
        v_transforms=v_transforms if v_transforms else None,
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
