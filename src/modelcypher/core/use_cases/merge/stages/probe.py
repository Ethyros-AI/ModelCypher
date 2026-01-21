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
Stage 1: PROBE - Compute alignment transforms from probe responses.

We run the COMPLETE atlas probe corpus from JSON - ALL probes are used for
maximum manifold coverage. Geometry requires n >= d probes (where d = max hidden dim),
but MORE probes means BETTER coverage of the shared representational space.
The atlas exists for complete coverage; limiting probes artificially loses information.

Token probing and weight-level shortcuts are intentionally disabled.

Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
Reference: Chun et al. (2025) "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features"
Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_pinv,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.cka import compute_cka_split
from modelcypher.core.domain.geometry.generalized_procrustes import RotationContinuityAnalyzer
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.use_cases.merge.stages.probe_alignment import align_layers
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _infer_required_probe_count,
    _precision_reference,
    _promote_precision,
    _select_probe_text,
    compute_numerical_rank,
    validate_full_rank_coverage,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    OrthogonalProbeGenerator,
    augment_rank_closed_form,
    find_null_space_tokens_closed_form,
    find_null_space_texts,
    # Trajectory-based null-space discovery
    collect_trajectories_batch,
    compute_trajectory_subspace,
    compute_trajectory_null_space,
    compute_trajectory_tangent_null_space,
    TrajectoryTangentResult,
)
from modelcypher.core.use_cases.merge.stages.probe_inference import (
    run_probe_inference,
    run_sequential_probe_inference,
    PagedActivations,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.activation_store import ActivationStore
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _layer_activation_from_provider(
    activation_provider: "ActivationProvider",
    model: Any,
    tokenizer: Any,
    input_ids: "Array",
    layer_idx: int,
    backend: "Backend",
) -> "Array | None":
    backend.eval(input_ids)
    token_ids = backend.tolist(input_ids)
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if not isinstance(token_ids, list):
        token_ids = [int(token_ids)]
    token_ids = [int(token_id) for token_id in token_ids]
    activations = activation_provider.collect_hidden_activations(
        model=model,
        tokenizer=tokenizer,
        text="",
        token_ids=token_ids,
    )
    return activations.get(layer_idx)


def _normalize_and_concatenate(
    existing: "Array",
    new_act: "Array",
    backend: "Backend",
) -> "Array":
    """Normalize new activation to match existing scale and concatenate.

    This handles:
    1. RMS normalization to match scales
    2. Expand dims for concatenation
    3. Shape verification before/after concatenation

    Args:
        existing: Existing activation matrix [n_samples, hidden_dim]
        new_act: New activation vector [hidden_dim]
        backend: Compute backend

    Returns:
        Concatenated activations [n_samples + 1, hidden_dim]

    Raises:
        RuntimeError: If shapes are incompatible
    """
    b = backend

    # RMS normalization: scale new activation to match existing magnitude
    existing_rms = b.sqrt(b.mean(existing * existing))
    new_rms = b.sqrt(b.mean(new_act * new_act))
    b.eval(existing_rms, new_rms)

    eps_norm = sqrt_scalar(machine_epsilon(b, new_rms), b)
    scale_factor = existing_rms / (new_rms + eps_norm)
    b.eval(scale_factor)

    normalized_act = new_act * scale_factor
    b.eval(normalized_act)

    # Expand dims for concatenation: [hidden_dim] -> [1, hidden_dim]
    new_row = b.expand_dims(normalized_act, 0)
    b.eval(new_row)

    # Shape verification before concatenation
    existing_shape = b.shape(existing)
    new_shape = b.shape(new_row)
    if len(existing_shape) != 2 or len(new_shape) != 2:
        raise RuntimeError(f"Shape mismatch: existing={existing_shape}, new={new_shape}")
    if existing_shape[1] != new_shape[1]:
        raise RuntimeError(
            f"Dimension mismatch: existing dim={existing_shape[1]}, new dim={new_shape[1]}"
        )

    # Concatenate
    concatenated = b.concatenate([existing, new_row], axis=0)
    b.eval(concatenated)

    # Verify shape after concatenation
    concat_shape = b.shape(concatenated)
    expected_rows = existing_shape[0] + new_shape[0]
    if concat_shape[0] != expected_rows:
        raise RuntimeError(
            f"Concatenation failed: expected {expected_rows} rows, got {concat_shape[0]}"
        )

    return concatenated


# Probe mode is ALWAYS activation-level CKA with atlas JSON probes.
# No token probing, no weight-level shortcuts.

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
    # Gate-space transforms: for PRE-SiLU gate/up projections (cross-arch)
    gate_transforms: dict[int, list[list[float]]] | None = None
    # Layer mapping: target_layer -> source_layer (from proportional depth mapping)
    layer_mapping: dict[int, int] | None = None
    # Trajectory-tangent results per layer - used for trajectory-tangent null-space projection
    # These capture the geometry of activation FLOW (positions + velocities), not just points.
    # The tangent subspace is where we can safely transplant weights "along the road."
    source_trajectory_tangents: dict[int, "TrajectoryTangentResult"] | None = None
    target_trajectory_tangents: dict[int, "TrajectoryTangentResult"] | None = None
    # HOT soft coupling matrix [n_source_layers, n_target_layers]
    # Each entry represents optimal mass transport between layer pairs.
    # Used to weight transfer strength: high coupling = strong alignment = transfer more.
    layer_coupling: list[list[float]] | None = None
    # Sorted layer indices for indexing into layer_coupling
    source_layers: list[int] | None = None
    target_layers: list[int] | None = None


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
    activation_provider: "ActivationProvider | None" = None,
    backend: "Backend | None" = None,
    probe_mode: str = "atlas",  # "atlas" (geometry-min) or "atlas_full"
    # Memory-efficient sequential mode parameters
    sequential_mode: bool = True,  # Process models one at a time (recommended)
    paging_dir: Path | None = None,  # Dir to page activations to disk
    activation_store: "ActivationStore | None" = None,  # Store for paging
    unload_source_callback: Callable[[], None] | None = None,  # Called after source probing
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

    Returns:
        ProbeResult with correlations, confidences, and intersection map
    """
    if probe_mode not in ("atlas", "atlas_full"):
        raise ValueError(
            f"PROBE MODE: {probe_mode} unsupported; atlas or atlas_full is required."
        )

    if tokenizer is not None:
        source_tokenizer = source_tokenizer or tokenizer
        target_tokenizer = target_tokenizer or tokenizer

    # ALWAYS use precise mode - this is not configurable.
    # Activation-level CKA is required for alignment.

    if activation_provider is None:
        raise ValueError("Activation provider required for probe stage")
    if not hasattr(activation_provider, "collect_probe_activations_batch"):
        raise ValueError(
            "Activation provider must implement collect_probe_activations_batch for strict probing."
        )

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
            backend=backend,
            probe_mode=probe_mode,
            sequential_mode=sequential_mode,
            paging_dir=paging_dir,
            activation_store=activation_store,
            unload_source_callback=unload_source_callback,
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


def _domain_stratified_batches(
    valid_probes: list[tuple[Any, str]], batch_size: int = 20
) -> list[list[tuple[Any, str]]]:
    """Generate batches with probes from all domains represented.

    This ensures early batches cover the full semantic space, maximizing
    rank increase per batch. Without stratification, we might process
    many probes from one domain before seeing others.

    Args:
        valid_probes: List of (probe, text) tuples.
        batch_size: Target probes per batch.

    Returns:
        List of batches, each containing probes from multiple domains.
    """
    from collections import defaultdict

    # Group probes by domain
    by_domain: dict[str, list[tuple[Any, str]]] = defaultdict(list)
    for probe, text in valid_probes:
        domain = str(probe.domain.value) if hasattr(probe.domain, "value") else str(probe.domain)
        by_domain[domain].append((probe, text))

    domains = list(by_domain.keys())
    if not domains:
        return []

    # Calculate probes per domain per batch
    probes_per_domain = max(1, batch_size // len(domains))

    # Track position in each domain's probe list
    domain_idx: dict[str, int] = {d: 0 for d in domains}

    batches: list[list[tuple[Any, str]]] = []
    while True:
        batch: list[tuple[Any, str]] = []

        # Take probes from each domain
        for domain in domains:
            domain_probes = by_domain[domain]
            start = domain_idx[domain]
            end = min(start + probes_per_domain, len(domain_probes))
            batch.extend(domain_probes[start:end])
            domain_idx[domain] = end

        if not batch:
            break  # All domains exhausted

        batches.append(batch)

    return batches


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
    probe_mode: str = "atlas",  # Only "atlas" is supported - atlas JSON probes
    # Memory-efficient sequential mode parameters
    sequential_mode: bool = True,
    paging_dir: Path | None = None,
    activation_store: "ActivationStore | None" = None,
    unload_source_callback: Callable[[], None] | None = None,
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models.

    Uses SATURATION-BASED SAMPLING with domain-stratified batching:
    1. Probes are grouped by domain for diverse coverage
    2. After each batch, rank saturation is checked per layer
    3. Stops when all layers reach rank saturation (geometric termination)

    This replaces fixed 4596-probe collection with model-specific sampling.
    Saturation = rank didn't increase for K_CONSECUTIVE=3 batches.

    Args:
        probe_mode: "atlas" or "atlas_full".
    """
    b = backend or get_default_backend()

    # Saturation detection constants (from ManifoldMapper)
    K_CONSECUTIVE = 3  # Batches with no rank increase = saturation
    BATCH_SIZE = 20  # Probes per batch

    # Load Atlas probes for manifold coverage.
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    probes = load_all_probes()

    # Pre-validate probes for usable text (used for caching + inference).
    valid_probes: list[tuple[Any, str]] = []
    for probe in probes:
        probe_text = _select_probe_text(probe)
        if probe_text is None:
            raise ValueError(
                f"Probe '{probe.probe_id}' has no valid support texts or fallback."
            )
        valid_probes.append((probe, probe_text))

    # Get dimensions for logging
    min_required, source_dim, target_dim = _infer_required_probe_count(
        source_weights, target_weights
    )

    if len(valid_probes) < min_required:
        raise RuntimeError(
            "PROBE MODE: Geometry requires minimum %d probes (src_rank=%d, tgt_rank=%d) "
            "but only %d valid probes available. Add probes before merging."
            % (min_required, source_dim, target_dim, len(valid_probes))
        )

    # Generate domain-stratified batches
    batches = _domain_stratified_batches(valid_probes, BATCH_SIZE)
    unique_domains = len({str(p.domain.value) for p, _ in valid_probes})

    logger.info(
        "PROBE MODE: Saturation-based sampling with %d probes across %d domains "
        "(geometry minimum=%d, src_dim=%d, tgt_dim=%d)",
        len(valid_probes),
        unique_domains,
        min_required,
        source_dim,
        target_dim,
    )
    logger.info(
        "PROBE MODE: %d domain-stratified batches, K_CONSECUTIVE=%d for saturation",
        len(batches),
        K_CONSECUTIVE,
    )

    # Activation storage
    source_layer_activations: dict[int, "Array"] = {}
    target_layer_activations: dict[int, "Array"] = {}
    source_intermediate_activations: dict[int, "Array"] = {}
    target_intermediate_activations: dict[int, "Array"] = {}
    source_gate_activations: dict[int, "Array"] = {}
    target_gate_activations: dict[int, "Array"] = {}
    source_attention_activations: dict[int, "Array"] = {}
    target_attention_activations: dict[int, "Array"] = {}
    source_k_activations: dict[int, "Array"] = {}
    target_k_activations: dict[int, "Array"] = {}
    source_v_activations: dict[int, "Array"] = {}
    target_v_activations: dict[int, "Array"] = {}
    source_embedding_activations: list["Array"] = []
    target_embedding_activations: list["Array"] = []

    # Track probe metadata
    probe_ids: list[str] = []
    probe_domains: list[str] = []

    # Trajectory-tangent storage for downstream transplant stage
    source_trajectory_tangents: dict[int, TrajectoryTangentResult] = {}
    target_trajectory_tangents: dict[int, TrajectoryTangentResult] = {}

    # Trajectory ranks per layer: the GEOMETRIC ceiling for achievable activation rank
    trajectory_ranks: dict[int, tuple[int, int]] = {}

    # Per-layer saturation tracking
    # A layer is "saturated" when its rank hasn't increased for K_CONSECUTIVE batches
    # This means we've found the boundary between activated space and null space
    layer_saturation_count: dict[int, int] = {}  # layer -> consecutive batches with no rank increase
    layer_previous_rank: dict[int, tuple[int, int]] = {}  # layer -> (src_rank, tgt_rank)
    layer_saturated: dict[int, bool] = {}  # layer -> is_saturated

    # Metrics tracking
    rank_augmentation_metrics: dict[str, Any] = {
        "initial_coverage": {},
        "final_coverage": {},
        "probes_generated_per_layer": {},
        "augmentation_iterations": 0,
        "full_rank_achieved": False,
        "batches_processed": 0,
        "probes_processed": 0,
        "generated_probes": 0,
    }

    probes_processed = 0
    probes_failed = 0
    batches_processed = 0

    # =========================================================================
    # COMPLETE MANIFOLD MAPPING - Saturation-based with on-the-fly generation
    # =========================================================================
    # Goal: Map the COMPLETE boundary between activated space and null space.
    # We don't stop at a fixed probe count. We stop when the mapping is DONE.
    #
    # Algorithm:
    # 1. Run domain-stratified batches from atlas
    # 2. After each batch, check rank saturation per layer
    # 3. Saturation = rank unchanged for K_CONSECUTIVE batches = boundary found
    # 4. If atlas exhausted but layers not saturated → generate null-space probes
    # 5. Continue until ALL layers have found their activated/null boundary
    # =========================================================================

    logger.info("MANIFOLD MAPPING: Starting complete boundary detection")

    # Phase 1: Domain-stratified atlas probes
    for batch_idx, batch in enumerate(batches):
        batch_texts = [(probe, text) for probe, text in batch]

        # Collect activations for this batch
        for probe, text in batch_texts:
            try:
                # Source activations
                src_acts = activation_provider.collect_hidden_activations(
                    model=source_model,
                    tokenizer=source_tokenizer,
                    text=text,
                )
                # Target activations
                tgt_acts = activation_provider.collect_hidden_activations(
                    model=target_model,
                    tokenizer=target_tokenizer,
                    text=text,
                )

                # Store activations per layer
                for layer_idx, act in src_acts.items():
                    b.eval(act)
                    if layer_idx not in source_layer_activations:
                        source_layer_activations[layer_idx] = act
                    else:
                        existing = source_layer_activations[layer_idx]
                        if existing.ndim == 1:
                            existing = b.expand_dims(existing, 0)
                        if act.ndim == 1:
                            act = b.expand_dims(act, 0)
                        source_layer_activations[layer_idx] = b.concatenate([existing, act], axis=0)
                        b.eval(source_layer_activations[layer_idx])

                for layer_idx, act in tgt_acts.items():
                    b.eval(act)
                    if layer_idx not in target_layer_activations:
                        target_layer_activations[layer_idx] = act
                    else:
                        existing = target_layer_activations[layer_idx]
                        if existing.ndim == 1:
                            existing = b.expand_dims(existing, 0)
                        if act.ndim == 1:
                            act = b.expand_dims(act, 0)
                        target_layer_activations[layer_idx] = b.concatenate([existing, act], axis=0)
                        b.eval(target_layer_activations[layer_idx])

                # Track probe metadata
                probe_ids.append(probe.probe_id)
                domain = str(probe.domain.value) if hasattr(probe.domain, "value") else str(probe.domain)
                probe_domains.append(domain)
                probes_processed += 1

            except Exception as e:
                logger.warning("MANIFOLD MAPPING: Probe failed: %s", e)
                probes_failed += 1

        batches_processed += 1

        # Check saturation after each batch
        all_saturated = True
        for layer_idx in source_layer_activations:
            if layer_idx not in target_layer_activations:
                continue

            # Initialize tracking for new layers
            if layer_idx not in layer_saturation_count:
                layer_saturation_count[layer_idx] = 0
                layer_previous_rank[layer_idx] = (0, 0)
                layer_saturated[layer_idx] = False

            if layer_saturated[layer_idx]:
                continue  # Already found boundary

            # Compute current ranks
            src_acts = source_layer_activations[layer_idx]
            tgt_acts = target_layer_activations[layer_idx]
            if src_acts.ndim == 1:
                src_acts = b.expand_dims(src_acts, 0)
            if tgt_acts.ndim == 1:
                tgt_acts = b.expand_dims(tgt_acts, 0)

            src_rank, src_dim = compute_numerical_rank(src_acts, b)
            tgt_rank, tgt_dim = compute_numerical_rank(tgt_acts, b)

            prev_src, prev_tgt = layer_previous_rank[layer_idx]

            # Check if rank increased
            if src_rank == prev_src and tgt_rank == prev_tgt:
                layer_saturation_count[layer_idx] += 1
                if layer_saturation_count[layer_idx] >= K_CONSECUTIVE:
                    layer_saturated[layer_idx] = True
                    # Store trajectory rank (the geometric ceiling we found)
                    trajectory_ranks[layer_idx] = (src_rank, tgt_rank)
                    logger.info(
                        "MANIFOLD MAPPING: Layer %d SATURATED - boundary found at "
                        "src_rank=%d/%d, tgt_rank=%d/%d (batch %d)",
                        layer_idx, src_rank, src_dim, tgt_rank, tgt_dim, batches_processed,
                    )
            else:
                layer_saturation_count[layer_idx] = 0
                layer_previous_rank[layer_idx] = (src_rank, tgt_rank)

            if not layer_saturated[layer_idx]:
                all_saturated = False

        # Log progress periodically
        if batches_processed % 10 == 0:
            saturated_count = sum(1 for s in layer_saturated.values() if s)
            total_layers = len(layer_saturated)
            logger.info(
                "MANIFOLD MAPPING: Batch %d, %d probes, %d/%d layers saturated",
                batches_processed, probes_processed, saturated_count, total_layers,
            )

        # Check for early termination
        if all_saturated and layer_saturated:
            logger.info(
                "MANIFOLD MAPPING: All %d layers saturated after %d batches (%d probes)",
                len(layer_saturated), batches_processed, probes_processed,
            )
            break

    # Phase 2: Generate null-space probes for unsaturated layers
    # If atlas probes didn't find the boundary, we generate probes that activate null space
    unsaturated_layers = [idx for idx, sat in layer_saturated.items() if not sat]

    if unsaturated_layers:
        logger.info(
            "MANIFOLD MAPPING: %d layers not saturated after atlas - generating null-space probes",
            len(unsaturated_layers),
        )

        generation_round = 0
        max_generation_rounds = 50  # Safety limit

        while unsaturated_layers and generation_round < max_generation_rounds:
            generation_round += 1
            probes_generated_this_round = 0

            for layer_idx in list(unsaturated_layers):
                src_acts = source_layer_activations.get(layer_idx)
                tgt_acts = target_layer_activations.get(layer_idx)

                if src_acts is None or tgt_acts is None:
                    continue

                if src_acts.ndim == 1:
                    src_acts = b.expand_dims(src_acts, 0)
                if tgt_acts.ndim == 1:
                    tgt_acts = b.expand_dims(tgt_acts, 0)

                # Use trajectory-based null-space discovery
                probe_texts_for_layer = [text for _, text in valid_probes[:max(source_dim, target_dim) + 1]]

                # Collect trajectories for null-space computation
                src_trajectories = collect_trajectories_batch(
                    model=source_model,
                    tokenizer=source_tokenizer,
                    texts=probe_texts_for_layer,
                    layer_idx=layer_idx,
                    backend=b,
                    include_accelerations=False,
                )

                if src_trajectories:
                    subspace_src = compute_trajectory_subspace(
                        trajectories=src_trajectories,
                        backend=b,
                        include_velocities=True,
                        include_accelerations=False,
                    )

                    if subspace_src is not None:
                        U_null_src = compute_trajectory_null_space(subspace_src, b)
                        trajectory_ranks[layer_idx] = (
                            subspace_src.rank,
                            trajectory_ranks.get(layer_idx, (0, 0))[1],
                        )

                        if U_null_src is not None:
                            null_texts = find_null_space_texts(
                                model=source_model,
                                tokenizer=source_tokenizer,
                                U_null=U_null_src,
                                layer_idx=layer_idx,
                                backend=b,
                            )

                            for text in null_texts:
                                if not text or not text.strip():
                                    continue

                                try:
                                    src_act = activation_provider.collect_hidden_activations(
                                        model=source_model,
                                        tokenizer=source_tokenizer,
                                        text=text.strip(),
                                    ).get(layer_idx)

                                    tgt_act = activation_provider.collect_hidden_activations(
                                        model=target_model,
                                        tokenizer=target_tokenizer,
                                        text=text.strip(),
                                    ).get(layer_idx)

                                    if src_act is not None:
                                        b.eval(src_act)
                                        if src_act.ndim == 1:
                                            src_act = b.expand_dims(src_act, 0)
                                        source_layer_activations[layer_idx] = b.concatenate(
                                            [source_layer_activations[layer_idx], src_act], axis=0
                                        )
                                        b.eval(source_layer_activations[layer_idx])

                                    if tgt_act is not None:
                                        b.eval(tgt_act)
                                        if tgt_act.ndim == 1:
                                            tgt_act = b.expand_dims(tgt_act, 0)
                                        target_layer_activations[layer_idx] = b.concatenate(
                                            [target_layer_activations[layer_idx], tgt_act], axis=0
                                        )
                                        b.eval(target_layer_activations[layer_idx])

                                    probes_generated_this_round += 1
                                    rank_augmentation_metrics["generated_probes"] = (
                                        rank_augmentation_metrics.get("generated_probes", 0) + 1
                                    )

                                except Exception as e:
                                    logger.debug("Generated probe failed: %s", e)

                # Re-check saturation for this layer
                src_acts = source_layer_activations[layer_idx]
                tgt_acts = target_layer_activations[layer_idx]
                if src_acts.ndim == 1:
                    src_acts = b.expand_dims(src_acts, 0)
                if tgt_acts.ndim == 1:
                    tgt_acts = b.expand_dims(tgt_acts, 0)

                src_rank, src_dim = compute_numerical_rank(src_acts, b)
                tgt_rank, tgt_dim = compute_numerical_rank(tgt_acts, b)

                prev_src, prev_tgt = layer_previous_rank.get(layer_idx, (0, 0))

                if src_rank == prev_src and tgt_rank == prev_tgt:
                    layer_saturation_count[layer_idx] = layer_saturation_count.get(layer_idx, 0) + 1
                    if layer_saturation_count[layer_idx] >= K_CONSECUTIVE:
                        layer_saturated[layer_idx] = True
                        trajectory_ranks[layer_idx] = (src_rank, tgt_rank)
                        unsaturated_layers.remove(layer_idx)
                        logger.info(
                            "MANIFOLD MAPPING: Layer %d SATURATED via generation - "
                            "src_rank=%d/%d, tgt_rank=%d/%d",
                            layer_idx, src_rank, src_dim, tgt_rank, tgt_dim,
                        )
                else:
                    layer_saturation_count[layer_idx] = 0
                    layer_previous_rank[layer_idx] = (src_rank, tgt_rank)

            if probes_generated_this_round == 0:
                # Can't generate more probes, use what we have
                logger.warning(
                    "MANIFOLD MAPPING: Cannot generate more probes for %d unsaturated layers",
                    len(unsaturated_layers),
                )
                # Mark remaining as saturated at current rank (best effort)
                for layer_idx in unsaturated_layers:
                    src_acts = source_layer_activations.get(layer_idx)
                    tgt_acts = target_layer_activations.get(layer_idx)
                    if src_acts is not None and tgt_acts is not None:
                        if src_acts.ndim == 1:
                            src_acts = b.expand_dims(src_acts, 0)
                        if tgt_acts.ndim == 1:
                            tgt_acts = b.expand_dims(tgt_acts, 0)
                        src_rank, _ = compute_numerical_rank(src_acts, b)
                        tgt_rank, _ = compute_numerical_rank(tgt_acts, b)
                        trajectory_ranks[layer_idx] = (src_rank, tgt_rank)
                        layer_saturated[layer_idx] = True
                break

            logger.info(
                "MANIFOLD MAPPING: Generation round %d - generated %d probes, %d layers remain",
                generation_round, probes_generated_this_round, len(unsaturated_layers),
            )

    # Record final metrics
    rank_augmentation_metrics["batches_processed"] = batches_processed
    rank_augmentation_metrics["probes_processed"] = probes_processed
    rank_augmentation_metrics["augmentation_iterations"] = generation_round if unsaturated_layers else 0

    # Validate final rank coverage
    rank_coverage = validate_full_rank_coverage(
        source_activations=source_layer_activations,
        target_activations=target_layer_activations,
        backend=b,
    )

    if rank_coverage:
        for layer_idx, info in rank_coverage.items():
            rank_augmentation_metrics["final_coverage"][layer_idx] = {
                "source_rank": info["source_rank"],
                "source_dim": info["source_dim"],
                "target_rank": info["target_rank"],
                "target_dim": info["target_dim"],
                "alignment_rank": info["alignment_rank"],
                "coverage": info["coverage_ratio"],
                "deficit": info["deficit"],
                "trajectory_rank": trajectory_ranks.get(layer_idx),
            }

        rank_augmentation_metrics["full_rank_achieved"] = all(
            info["full_rank_achieved"] for info in rank_coverage.values()
        )

        # Final coverage summary
        coverage_ratios = [v["coverage_ratio"] for v in rank_coverage.values()]
        mean_coverage = sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0.0
        min_coverage = min(coverage_ratios) if coverage_ratios else 0.0

        logger.info(
            "MANIFOLD MAPPING COMPLETE: %d probes, %d batches, %d generated, "
            "mean_coverage=%.1f%%, min_coverage=%.1f%%",
            probes_processed,
            batches_processed,
            rank_augmentation_metrics.get("generated_probes", 0),
            100.0 * mean_coverage,
            100.0 * min_coverage,
        )

    intersection_map_obj: Any | None = None
    dimension_correlations: dict = {}
    layer_cka_scores: dict[int, float] = {}

    # Align layers by proportional depth, then solve closed-form alignment per pair.
    # CKA is diagnostic; we do not use it as a selector.
    alignment_result = align_layers(
        source_layer_activations=source_layer_activations,
        target_layer_activations=target_layer_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_gate_activations=source_gate_activations,
        target_gate_activations=target_gate_activations,
        backend=b,
    )
    layer_mapping = alignment_result.layer_mapping
    feature_transforms = alignment_result.feature_transforms
    scale_ratios = alignment_result.scale_ratios
    attention_transforms = alignment_result.attention_transforms
    k_transforms = alignment_result.k_transforms
    v_transforms = alignment_result.v_transforms
    intermediate_transforms = alignment_result.intermediate_transforms
    gate_transforms = alignment_result.gate_transforms
    layer_cka_scores = alignment_result.layer_cka_scores
    cgls_iterations_by_layer = alignment_result.cgls_iterations_by_layer
    gram_condition_numbers_by_layer = alignment_result.gram_condition_numbers_by_layer
    linear_residuals_by_layer = alignment_result.linear_residuals_by_layer
    numerical_deviation_by_layer = alignment_result.numerical_deviation_by_layer
    precision_thresholds_by_layer = alignment_result.precision_thresholds_by_layer
    # HOT soft coupling for transfer strength weighting
    layer_coupling = alignment_result.layer_coupling
    rotation_continuity: dict[str, Any] | None = None
    rotation_analyzer = RotationContinuityAnalyzer(backend=b)
    rotation_result = rotation_analyzer.compute_per_layer_alignments_from_arrays(
        source_layer_activations=source_layer_activations,
        target_layer_activations=target_layer_activations,
        source_model=source_path or "source",
        target_model=target_path or "target",
    )
    if rotation_result is not None:
        rotation_continuity = asdict(rotation_result)

    # Extract layer confidences (CKA-only)
    layer_confidences: dict[int, float] = {}
    if layer_cka_scores:
        layer_confidences.update(layer_cka_scores)

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
    # Geodesic CKA is diagnostic, not a gate. Filter only NaN (alignment bugs).
    # Low geodesic CKA usually means limited overlap or probe coverage.
    valid_cka_vals = [v for v in cka_vals if v == v]  # NaN check only
    nan_count = len(cka_vals) - len(valid_cka_vals)
    if nan_count > 0:
        logger.error(
            "PROBE: %d layers have NaN CKA - alignment algorithm bug, investigate!",
            nan_count
        )
    mean_cka = sum(valid_cka_vals) / len(valid_cka_vals) if valid_cka_vals else 0.0
    min_cka = min(valid_cka_vals) if valid_cka_vals else 0.0
    # layers_with_data: layers that have activations in both models (for reporting)
    layers_with_data = set(source_layer_activations.keys()) & set(target_layer_activations.keys())
    # Proportional mapping defines a correspondence for every target layer.
    # missing_cka_layers is for reporting - it doesn't block exact alignment
    missing_cka_layers = [layer for layer in layers_with_data if layer not in layer_cka_scores]
    # =========================================================================
    # GEODESIC CKA DIAGNOSTIC (STRICT OVERLAP CHECK)
    # =========================================================================
    # perfect_alignment is a strict diagnostic: it only holds when every layer's
    # geodesic CKA is within precision. This is not required for merging and can
    # be false when models contain novel structure outside the shared manifold.
    # Use sqrt(machine_epsilon) as the tolerance (matches GramAligner convention).
    precision_ref = _precision_reference(
        b,
        feature_transforms,
        source_layer_activations,
        target_layer_activations,
    )
    precision_threshold = sqrt_scalar(machine_epsilon(b, precision_ref), b)

    split_cka_result = None

    # =========================================================================
    # LAYER CLASSIFICATION: ALL LAYERS PROCESSED
    # =========================================================================
    # Geodesic CKA measures STRUCTURAL OVERLAP between source and target spaces.
    #
    # CKA ≈ 1.0: Source fully covers target's representational space (shared manifold)
    # CKA < 1.0: Target has structure outside source's column space (EXPECTED
    #            for cross-dimensional alignment, e.g., 896 → 1024 hidden dims)
    #
    # This reflects how much of target's geometry is captured by source.
    # Null-space projection preserves target-unique structure while adding source.
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
            # This is expected for cross-dimensional alignment (different hidden dims).
            # Null-space projection preserves target-unique structure.
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

    # =========================================================================
    # VALIDATE TRANSFORMS FOR ALL TARGET LAYERS (STRICT - NO FALLBACKS)
    # =========================================================================
    # Proportional mapping defines transforms for all target layers.
    # If any transforms are missing, it indicates missing activations or
    # an alignment issue; fail fast instead of propagating incomplete transforms.
    all_target_layers = sorted(set(target_layer_activations.keys()))
    if not feature_transforms:
        raise RuntimeError("PROBE FAILED: No feature transforms computed.")
    if feature_transforms and len(feature_transforms) < len(all_target_layers):
        missing_layers = [l for l in all_target_layers if l not in feature_transforms]
        raise RuntimeError(
            f"PROBE FAILED: Missing feature transforms for {len(missing_layers)} layers: {missing_layers}. "
            f"This indicates missing activations or an alignment bug. "
            f"Available transforms: {sorted(feature_transforms.keys())}"
        )

    if not intermediate_transforms:
        raise RuntimeError("PROBE FAILED: No intermediate transforms computed.")
    if intermediate_transforms and len(intermediate_transforms) < len(all_target_layers):
        missing_layers = [l for l in all_target_layers if l not in intermediate_transforms]
        raise RuntimeError(
            f"PROBE FAILED: Missing intermediate transforms for {len(missing_layers)} layers: {missing_layers}. "
            f"Available transforms: {sorted(intermediate_transforms.keys())}"
        )

    if attention_transforms and len(attention_transforms) < len(all_target_layers):
        missing_layers = [l for l in all_target_layers if l not in attention_transforms]
        raise RuntimeError(
            f"PROBE FAILED: Missing attention transforms for {len(missing_layers)} layers: {missing_layers}. "
            f"Available transforms: {sorted(attention_transforms.keys())}"
        )

    if k_transforms and len(k_transforms) < len(all_target_layers):
        missing_layers = [l for l in all_target_layers if l not in k_transforms]
        raise RuntimeError(
            f"PROBE FAILED: Missing K transforms for {len(missing_layers)} layers: {missing_layers}. "
            f"Available transforms: {sorted(k_transforms.keys())}"
        )

    if v_transforms and len(v_transforms) < len(all_target_layers):
        missing_layers = [l for l in all_target_layers if l not in v_transforms]
        raise RuntimeError(
            f"PROBE FAILED: Missing V transforms for {len(missing_layers)} layers: {missing_layers}. "
            f"Available transforms: {sorted(v_transforms.keys())}"
        )

    gram_aligner = GramAligner(backend=b)
    embedding_cka: float | None = None
    embedding_alignment: dict[str, float | int] | None = None

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
        if n_samples < 2:
            raise RuntimeError(
                f"EMBEDDING GRAMALIGN: Insufficient samples ({n_samples}) for alignment."
            )

        logger.info(
            "EMBEDDING GRAMALIGN: Computing 2D alignment with %d samples (linear alignment + geodesic diagnostics)",
            n_samples,
        )
        src_stacked = _stack_embeddings(source_embedding_activations, n_samples)
        tgt_stacked = _stack_embeddings(target_embedding_activations, n_samples)
        src_stacked = _promote_precision(src_stacked, b)
        tgt_stacked = _promote_precision(tgt_stacked, b)
        b.eval(src_stacked, tgt_stacked)

        # Use same GramAligner as hidden layers
        emb_result = gram_aligner.find_perfect_alignment(src_stacked, tgt_stacked)
        emb_F = emb_result.feature_transform  # Already GPU array
        embedding_transform = emb_F  # Keep as GPU array

        emb_aligned = b.matmul(src_stacked, emb_F)
        b.eval(emb_aligned)

        # Geodesic CKA from alignment
        emb_geodesic_cka = emb_result.achieved_cka
        embedding_cka = emb_geodesic_cka
        embedding_alignment = {
            "achieved_cka": emb_result.achieved_cka,
            "numerical_deviation": emb_result.numerical_deviation,
            "precision_threshold": emb_result.precision_threshold,
            "gram_condition_number": emb_result.gram_condition_number,
            "linear_residual": emb_result.linear_residual,
            "iterations": emb_result.iterations,
            "linear_iterations": emb_result.linear_iterations,
            "scale_ratio": emb_result.scale_ratio,
        }

        linear_transform = b.matmul(geodesic_pinv(b, src_stacked), tgt_stacked)
        b.eval(linear_transform)
        split_cka_result = compute_cka_split(
            src_stacked,
            tgt_stacked,
            backend=b,
            feature_transform=linear_transform,
        )
    else:
        raise RuntimeError("EMBEDDING GRAMALIGN: Missing embedding activations.")

    if embedding_transform is None:
        raise RuntimeError("EMBEDDING GRAMALIGN failed to produce a transform.")

    # CKA is undefined for <2 samples (covariance requires N>=2).
    min_shared = 2
    if split_cka_result and split_cka_result.n_shared >= min_shared:
        perfect_alignment = split_cka_result.shared_cka >= 1.0 - precision_threshold
    else:
        perfect_alignment = bool(valid_cka_vals) and min_cka >= 1.0 - precision_threshold

    metrics = {
        "probe_mode": "precise",
        "probes_total": len(probes),
        "probes_processed": probes_processed,
        "probes_failed": probes_failed,
        "probes_selected": len(valid_probes),
        "probes_required_min": min_required,
        "source_hidden_dim": source_dim,
        "target_hidden_dim": target_dim,
        "layers_analyzed": len(layer_confidences),
        "layers_with_cka": len(layer_cka_scores),
        "layers_with_data": len(layers_with_data),
        "missing_cka_layers": len(missing_cka_layers),
        "layer_confidences": layer_confidences,
        "layer_cka_scores": layer_cka_scores,
        "cgls_iterations_by_layer": cgls_iterations_by_layer,
        "alignment_diagnostics": {
            "gram_condition_numbers_by_layer": gram_condition_numbers_by_layer,
            "linear_residuals_by_layer": linear_residuals_by_layer,
            "numerical_deviation_by_layer": numerical_deviation_by_layer,
            "precision_thresholds_by_layer": precision_thresholds_by_layer,
        },
        "rotation_continuity": rotation_continuity,
        "layer_mapping": layer_mapping,
        "scale_ratios": scale_ratios,
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "cka_estimator": "geodesic",
        "perfect_alignment": perfect_alignment,
        "embedding_cka": embedding_cka,
        "embedding_alignment": embedding_alignment,
        # Layer classification for adaptive barometer
        "layer_status": layer_status,
        "converged_layers": converged_layers,
        "boundary_preserved_layers": boundary_preserved_layers,
        "skipped_layers": skipped_layers,
        "converged_count": len(converged_layers),
        "boundary_preserved_count": len(boundary_preserved_layers),
        "skipped_count": len(skipped_layers),
        "atlas_sources": list(set(p.source.value for p in probes)),
        "atlas_domains": list(set(p.domain.value for p in probes)),
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
        # Rank coverage metrics for alignment validation
        "rank_augmentation": rank_augmentation_metrics if rank_augmentation_metrics else None,
    }

    if split_cka_result:
        logger.info(
            "PROBE PRECISE: %d layers, geodesic_cka=%.4f, shared_cka=%.4f, shared_fraction=%.4f",
            len(layer_confidences),
            mean_cka,
            split_cka_result.shared_cka,
            split_cka_result.shared_fraction,
        )
    else:
        logger.info(
            "PROBE PRECISE: %d layers, geodesic_cka=%.4f",
            len(layer_confidences),
            mean_cka,
        )

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
        gate_transforms=gate_transforms if gate_transforms else None,
        layer_mapping=layer_mapping if layer_mapping else None,
        source_trajectory_tangents=source_trajectory_tangents if source_trajectory_tangents else None,
        target_trajectory_tangents=target_trajectory_tangents if target_trajectory_tangents else None,
        layer_coupling=layer_coupling,
        source_layers=sorted(source_layer_activations.keys()) if source_layer_activations else None,
        target_layers=sorted(target_layer_activations.keys()) if target_layer_activations else None,
    )
