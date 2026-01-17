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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_pinv,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.cka import compute_cka_split
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.use_cases.merge.stages.probe_alignment import align_layers
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _infer_required_probe_count,
    _precision_reference,
    _promote_precision,
    _select_geometry_probes,
    _select_probe_text,
)
from modelcypher.core.use_cases.merge.stages.probe_inference import run_probe_inference

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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
    probe_mode: str = "atlas",  # Only "atlas" is supported - atlas JSON probes
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models.

    Uses Atlas JSON probes with either geometry-derived count (atlas)
    or full corpus coverage (atlas_full).

    Args:
        probe_mode: "atlas" or "atlas_full".
    """
    b = backend or get_default_backend()
    # Load Atlas probes for manifold coverage.
    # Probe count is derived from geometry (hidden dimensions).
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    probes = load_all_probes()
    logger.info("PROBE MODE: Atlas (%d probes total)", len(probes))

    # Pre-validate probes for usable text (used for caching + inference).
    valid_probes: list[tuple[Any, str]] = []
    for probe in probes:
        probe_text = _select_probe_text(probe)
        if probe_text is None:
            raise ValueError(
                f"Probe '{probe.probe_id}' has no valid support texts or fallback."
            )
        valid_probes.append((probe, probe_text))

    # GEOMETRY PRINCIPLE: Use the exact probe count implied by intrinsic rank,
    # unless full atlas coverage is explicitly requested.
    min_required, source_dim, target_dim = _infer_required_probe_count(
        source_weights, target_weights
    )

    if probe_mode == "atlas_full":
        selected_probes = valid_probes
    else:
        selected_probes = _select_geometry_probes(valid_probes, min_required)

    if len(valid_probes) < min_required:
        raise RuntimeError(
            "PROBE MODE: Geometry requires minimum %d probes (src_rank=%d, tgt_rank=%d) "
            "but only %d valid probes available. Add probes before merging."
            % (min_required, source_dim, target_dim, len(valid_probes))
        )

    if probe_mode == "atlas_full":
        logger.info(
            "PROBE MODE: Using full atlas (%d probes, geometry minimum=%d, src_rank=%d, tgt_rank=%d)",
            len(selected_probes),
            min_required,
            source_dim,
            target_dim,
        )
    else:
        logger.info(
            "PROBE MODE: Using %d probes (geometry minimum=%d, src_rank=%d, tgt_rank=%d)",
            len(selected_probes),
            min_required,
            source_dim,
            target_dim,
        )

    valid_probes = selected_probes
    expected_probe_ids = [probe.probe_id for probe, _ in valid_probes]
    expected_probe_domains = [probe.domain.value for probe, _ in valid_probes]
    logger.info(
        "PROBE PRECISE: Running %d probes through source + target models...",
        len(valid_probes),
    )

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
    source_embedding_activations: list["Array"] | "Array" = []
    target_embedding_activations: list["Array"] | "Array" = []

    probe_ids: list[str] = list(expected_probe_ids)
    probe_domains: list[str] = list(expected_probe_domains)

    # =========================================================================
    # BATCHED PROBE COLLECTION - Process probes in batches for efficiency
    # =========================================================================
    probes_processed, probes_failed = run_probe_inference(
        valid_probes=valid_probes,
        source_model=source_model,
        target_model=target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        activation_provider=activation_provider,
        backend=b,
        source_layer_activations=source_layer_activations,
        target_layer_activations=target_layer_activations,
        source_intermediate_activations=source_intermediate_activations,
        target_intermediate_activations=target_intermediate_activations,
        source_gate_activations=source_gate_activations,
        target_gate_activations=target_gate_activations,
        source_embedding_activations=source_embedding_activations,
        target_embedding_activations=target_embedding_activations,
    )

    logger.info(
        "PROBE PRECISE: Completed %d probes (%d failed)",
        probes_processed,
        probes_failed,
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

    # Extract layer confidences (CKA-only, no fallbacks)
    layer_confidences: dict[int, float] = {}
    if layer_cka_scores:
        layer_confidences.update(layer_cka_scores)

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

    gram_aligner = GramAligner(backend=b, use_geodesic_alignment=False)
    embedding_cka: float | None = None

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

    if split_cka_result and split_cka_result.n_shared >= 4:
        perfect_alignment = split_cka_result.shared_cka >= 1.0 - precision_threshold
    else:
        perfect_alignment = bool(valid_cka_vals) and min_cka >= 1.0 - precision_threshold

    metrics = {
        "probe_mode": "precise",
        "probes_total": len(probes),
        "probes_processed": probes_processed,
        "probes_failed": probes_failed,
        "layers_analyzed": len(layer_confidences),
        "layers_with_cka": len(layer_cka_scores),
        "layers_with_data": len(layers_with_data),
        "missing_cka_layers": len(missing_cka_layers),
        "layer_confidences": layer_confidences,
        "layer_cka_scores": layer_cka_scores,
        "cgls_iterations_by_layer": cgls_iterations_by_layer,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "cka_estimator": "geodesic",
        "perfect_alignment": perfect_alignment,
        "embedding_cka": embedding_cka,
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
    )
