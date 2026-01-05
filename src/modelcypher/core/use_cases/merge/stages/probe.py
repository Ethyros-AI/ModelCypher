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
    # KV Attention-space activations: shape [num_kv_heads * head_dim] per sample
    # (e.g., 320 for SmolLM=5*64, 128 for Qwen=2*64) - for k_proj/v_proj stitching (GQA)
    source_kv_activations: dict[int, list[Any]] | None = None
    target_kv_activations: dict[int, list[Any]] | None = None
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
    # Attention KV-space transforms: for k_proj/v_proj (e.g., 320 -> 128 for GQA)
    kv_transforms: dict[int, list[list[float]]] | None = None
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
    # KV Attention-space activations for k_proj/v_proj stitching (GQA models)
    source_kv_activations: dict[int, list["Array"]] = {}
    target_kv_activations: dict[int, list["Array"]] = {}
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

            # ===== ATTENTION ACTIVATIONS (Q and KV) =====
            if has_batch_attention:
                source_q_batch, source_kv_batch = activation_provider.collect_attention_activations_batch(
                    source_model, source_tokenizer, batch_texts
                )
                target_q_batch, target_kv_batch = activation_provider.collect_attention_activations_batch(
                    target_model, target_tokenizer, batch_texts
                )
            else:
                source_q_batch, source_kv_batch = [], []
                target_q_batch, target_kv_batch = [], []
                for text in batch_texts:
                    src_q, src_kv = activation_provider.collect_attention_activations(
                        source_model, source_tokenizer, text
                    )
                    tgt_q, tgt_kv = activation_provider.collect_attention_activations(
                        target_model, target_tokenizer, text
                    )
                    source_q_batch.append(src_q)
                    source_kv_batch.append(src_kv)
                    target_q_batch.append(tgt_q)
                    target_kv_batch.append(tgt_kv)

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
                source_kv_acts = source_kv_batch[i] if source_kv_batch else {}
                target_attention_acts = target_q_batch[i] if target_q_batch else {}
                target_kv_acts = target_kv_batch[i] if target_kv_batch else {}

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

                # Store KV attention activations for k_proj/v_proj stitching (GQA models)
                for layer_idx, act in source_kv_acts.items():
                    if layer_idx not in source_kv_activations:
                        source_kv_activations[layer_idx] = []
                    source_kv_activations[layer_idx].append(act)

                for layer_idx, act in target_kv_acts.items():
                    if layer_idx not in target_kv_activations:
                        target_kv_activations[layer_idx] = []
                    target_kv_activations[layer_idx].append(act)

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
                    source_attention_acts, source_kv_acts = activation_provider.collect_attention_activations(
                        source_model, source_tokenizer, probe_text
                    )
                    target_attention_acts, target_kv_acts = activation_provider.collect_attention_activations(
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

                    for layer_idx, act in source_kv_acts.items():
                        if layer_idx not in source_kv_activations:
                            source_kv_activations[layer_idx] = []
                        source_kv_activations[layer_idx].append(act)

                    for layer_idx, act in target_kv_acts.items():
                        if layer_idx not in target_kv_activations:
                            target_kv_activations[layer_idx] = []
                        target_kv_activations[layer_idx].append(act)

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
                "PROBE PRECISE: Built IntersectionMap with raw_fingerprint_similarity=%.3f, %d layers",
                intersection_map_obj.raw_fingerprint_similarity,
                len(intersection_map_obj.layer_confidences),
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
    kv_transforms: dict[int, list[list[float]]] = {}  # target_layer -> KV attention transform
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

            def _align_single_layer(
                src_idx: int,
                tgt_idx: int,
                raw_cka: float,
            ) -> dict:
                """Align a single layer pair and return all transforms."""
                src_layer = source_layers[src_idx]
                tgt_layer = target_layers[tgt_idx]
                result: dict = {
                    "src_layer": src_layer,
                    "tgt_layer": tgt_layer,
                    "raw_cka": raw_cka,
                    "achieved_cka": 0.0,
                    "feature_transform": None,
                    "attention_transform": None,
                    "kv_transform": None,
                    "error": None,
                }

                # Get activations for this layer pair
                src_list = source_layer_activations[src_layer]
                tgt_list = target_layer_activations[tgt_layer]
                n_samples = min(len(src_list), len(tgt_list))

                if n_samples < 2:
                    return result

                # Create thread-local GramAligner to avoid any potential state issues
                local_aligner = GramAligner(backend=b)

                try:
                    src_stacked = b.stack(src_list[:n_samples], axis=0)
                    tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                    src_stacked = b.astype(src_stacked, "float32")
                    tgt_stacked = b.astype(tgt_stacked, "float32")
                    b.eval(src_stacked, tgt_stacked)

                    # GramAligner finds the transformation that achieves CKA = 1.0
                    alignment_result = local_aligner.find_perfect_alignment(
                        src_stacked,
                        tgt_stacked,
                    )

                    result["feature_transform"] = alignment_result.feature_transform
                    result["achieved_cka"] = alignment_result.achieved_cka

                    if not alignment_result.is_perfect:
                        logger.warning(
                            "PROBE: Layer %d -> %d hidden alignment not exact "
                            "(achieved_cka=%.4f, threshold=%.2e). "
                            "This indicates an alignment algorithm bug.",
                            src_layer,
                            tgt_layer,
                            alignment_result.achieved_cka,
                            alignment_result.precision_threshold,
                        )

                    # ================================================================
                    # ATTENTION Q TRANSFORMS
                    # ================================================================
                    if (
                        src_layer in source_attention_activations
                        and tgt_layer in target_attention_activations
                    ):
                        src_attn_list = source_attention_activations[src_layer]
                        tgt_attn_list = target_attention_activations[tgt_layer]
                        n_attn = min(len(src_attn_list), len(tgt_attn_list))
                        if n_attn >= 2:
                            src_attn = b.stack(src_attn_list[:n_attn], axis=0)
                            tgt_attn = b.stack(tgt_attn_list[:n_attn], axis=0)
                            src_attn = b.astype(src_attn, "float32")
                            tgt_attn = b.astype(tgt_attn, "float32")
                            b.eval(src_attn, tgt_attn)

                            attn_result = local_aligner.find_perfect_alignment(
                                src_attn,
                                tgt_attn,
                            )
                            result["attention_transform"] = attn_result.feature_transform

                            if not attn_result.is_perfect:
                                logger.warning(
                                    "PROBE: Layer %d -> %d attention Q alignment not exact "
                                    "(achieved_cka=%.4f).",
                                    src_layer,
                                    tgt_layer,
                                    attn_result.achieved_cka,
                                )

                    # ================================================================
                    # ATTENTION KV TRANSFORMS with Procrustes pre-alignment
                    # ================================================================
                    if (
                        src_layer in source_kv_activations
                        and tgt_layer in target_kv_activations
                    ):
                        src_kv_list = source_kv_activations[src_layer]
                        tgt_kv_list = target_kv_activations[tgt_layer]
                        n_kv = min(len(src_kv_list), len(tgt_kv_list))
                        if n_kv >= 2:
                            src_kv = b.stack(src_kv_list[:n_kv], axis=0)
                            tgt_kv = b.stack(tgt_kv_list[:n_kv], axis=0)
                            src_kv = b.astype(src_kv, "float32")
                            tgt_kv = b.astype(tgt_kv, "float32")
                            b.eval(src_kv, tgt_kv)

                            # Procrustes pre-alignment
                            src_dim = b.shape(src_kv)[1]
                            tgt_dim = b.shape(tgt_kv)[1]
                            shared_dim = min(src_dim, tgt_dim)

                            src_kv_shared = src_kv[:, :shared_dim]
                            tgt_kv_shared = tgt_kv[:, :shared_dim]

                            src_mean = b.mean(src_kv_shared, axis=0, keepdims=True)
                            tgt_mean = b.mean(tgt_kv_shared, axis=0, keepdims=True)
                            src_centered = src_kv_shared - src_mean
                            tgt_centered = tgt_kv_shared - tgt_mean
                            b.eval(src_centered, tgt_centered)

                            M = b.matmul(b.transpose(src_centered), tgt_centered)
                            b.eval(M)
                            U, _, Vt = geodesic_svd(b, M)
                            R_procrustes = b.matmul(U, Vt)

                            det_val = b.det(R_procrustes)
                            b.eval(det_val)
                            if float(b.to_scalar(det_val)) < 0:
                                U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                                R_procrustes = b.matmul(U_fixed, Vt)
                            b.eval(R_procrustes)

                            src_kv_rotated = b.matmul(src_kv_shared, R_procrustes)
                            b.eval(src_kv_rotated)

                            # Log Procrustes alignment improvement
                            _, pre_dist = geodesic_pairwise_metrics(src_centered, tgt_centered, b)
                            _, post_dist = geodesic_pairwise_metrics(src_kv_rotated, tgt_kv_shared, b)
                            pre_err = b.sum(pre_dist * pre_dist)
                            post_err = b.sum(post_dist * post_dist)
                            b.eval(pre_err, post_err)
                            logger.debug(
                                "PROBE: KV Procrustes layer %d: error %.4f -> %.4f",
                                src_layer,
                                float(b.to_scalar(pre_err)),
                                float(b.to_scalar(post_err)),
                            )

                            kv_result = local_aligner.find_perfect_alignment(
                                src_kv_rotated,
                                tgt_kv_shared,
                            )

                            # Build combined transform
                            if src_dim > shared_dim:
                                pad_rows = b.zeros((src_dim - shared_dim, shared_dim))
                                R_padded = b.concatenate([R_procrustes, pad_rows], axis=0)
                                b.eval(R_padded)
                                R_procrustes_full = R_padded
                            else:
                                R_procrustes_full = R_procrustes

                            gram_transform = b.array(kv_result.feature_transform)
                            combined_transform = b.matmul(R_procrustes_full, gram_transform)
                            b.eval(combined_transform)

                            if tgt_dim > shared_dim:
                                pad_cols = b.zeros((src_dim, tgt_dim - shared_dim))
                                combined_transform = b.concatenate(
                                    [combined_transform, pad_cols], axis=1
                                )
                                b.eval(combined_transform)
                                logger.debug(
                                    "PROBE: KV transform layer %d: padded [%d,%d] -> [%d,%d]",
                                    tgt_layer, src_dim, shared_dim, src_dim, tgt_dim,
                                )

                            result["kv_transform"] = [
                                [float(x) for x in row] for row in b.tolist(combined_transform)
                            ]

                            if not kv_result.is_perfect:
                                logger.warning(
                                    "PROBE: Layer %d -> %d attention KV alignment not exact "
                                    "(achieved_cka=%.4f after Procrustes).",
                                    src_layer,
                                    tgt_layer,
                                    kv_result.achieved_cka,
                                )

                except Exception as e:
                    result["error"] = str(e)
                    logger.warning(
                        "PROBE: GramAligner failed for layer %d -> %d: %s",
                        src_layer,
                        tgt_layer,
                        e,
                    )

                return result

            # Submit all layer alignments to thread pool
            logger.info(
                "PROBE: Aligning %d layer pairs in parallel (max_workers=%d)...",
                len(dp_path),
                ALIGNMENT_MAX_WORKERS,
            )

            with ThreadPoolExecutor(max_workers=ALIGNMENT_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _align_single_layer,
                        src_idx,
                        tgt_idx,
                        cka_matrix[src_idx][tgt_idx],
                    ): (src_idx, tgt_idx)
                    for src_idx, tgt_idx in dp_path
                }

                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    src_layer = result["src_layer"]
                    tgt_layer = result["tgt_layer"]

                    # Store results
                    layer_mapping[tgt_layer] = src_layer

                    if result["feature_transform"] is not None:
                        feature_transforms[tgt_layer] = result["feature_transform"]
                        layer_cka_scores[tgt_layer] = result["achieved_cka"]
                    else:
                        layer_cka_scores[tgt_layer] = result["raw_cka"]

                    layer_cka_scores_raw[tgt_layer] = result["raw_cka"]

                    if result["attention_transform"] is not None:
                        attention_transforms[tgt_layer] = result["attention_transform"]

                    if result["kv_transform"] is not None:
                        kv_transforms[tgt_layer] = result["kv_transform"]

                    completed += 1
                    if completed % 5 == 0 or completed == len(dp_path):
                        logger.info(
                            "PROBE: Aligned %d/%d layer pairs...",
                            completed,
                            len(dp_path),
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
        "raw_fingerprint_similarity": (
            intersection_map_obj.raw_fingerprint_similarity if intersection_map_obj else 0.0
        ),
    }

    # mean_cka = POST-ALIGNMENT quality (should be 1.0 when aligned correctly)
    # raw_fingerprint_similarity = PRE-ALIGNMENT intrinsic similarity (expected low for different architectures)
    logger.info(
        "PROBE PRECISE: %d layers, cka=%.3f (post-alignment), raw_similarity=%.3f (pre-alignment)",
        len(layer_confidences),
        mean_cka,
        metrics["raw_fingerprint_similarity"],
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

    if kv_transforms and len(kv_transforms) < len(all_target_layers):
        mapped_layers = sorted(kv_transforms.keys())
        for tgt_layer in all_target_layers:
            if tgt_layer not in kv_transforms:
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                kv_transforms[tgt_layer] = kv_transforms[nearest]

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
                emb_result = gram_aligner.align(src_stacked, tgt_stacked)
                if emb_result.achieved_cka >= 0.99:
                    embedding_transform = b.tolist(emb_result.transform)
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
        source_kv_activations=source_kv_activations,
        target_kv_activations=target_kv_activations,
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        feature_transforms=feature_transforms if feature_transforms else None,
        embedding_transform=embedding_transform,
        attention_transforms=attention_transforms if attention_transforms else None,
        kv_transforms=kv_transforms if kv_transforms else None,
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
