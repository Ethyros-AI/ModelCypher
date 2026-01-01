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

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.vocabulary.alignment_map import (
    AlignmentQuality,
    VocabularyAlignmentMap,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Probe mode is ALWAYS "precise" - activation-level CKA is required for correct alignment.
# "fast" mode is eliminated - weight-level CKA is fundamentally less accurate and
# hides alignment problems that will cause gibberish output.
_PROBE_MODE = "precise"

# All probes are always run. The probe corpus (403 probes) was carefully designed
# to cover the concept space. Limiting probes degrades coverage with no benefit.
_MAX_PROBES = 0  # 0 = all probes


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
    probe_ids: list[str] | None = None
    probe_domains: list[str] | None = None


def _encode_probe_ids(
    tokenizer: Any,
    text: str,
    add_special_tokens: bool = False,
) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        encoded = tokenizer.encode(text)

    if isinstance(encoded, list):
        return encoded
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return []


def build_token_id_map(
    alignment_map: VocabularyAlignmentMap,
) -> dict[int, int]:
    """Build token ID mapping from exact vocabulary alignments only."""
    mapping: dict[int, int] = {}
    for alignment in alignment_map.iter_alignments():
        if alignment.quality != AlignmentQuality.EXACT:
            continue
        if not alignment.target_ids:
            continue
        best_idx = max(
            range(len(alignment.target_ids)),
            key=lambda i: alignment.weights[i],
        )
        mapping[alignment.source_id] = alignment.target_ids[best_idx]
    return mapping


def map_token_ids(
    token_ids: list[int],
    token_map: dict[int, int],
) -> list[int] | None:
    mapped: list[int] = []
    for token_id in token_ids:
        mapped_id = token_map.get(token_id)
        if mapped_id is None:
            return None
        mapped.append(mapped_id)
    return mapped


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
    alignment_map: VocabularyAlignmentMap | None = None,
    backend: "Backend | None" = None,
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

    Returns:
        ProbeResult with correlations, confidences, and intersection map
    """
    if tokenizer is not None:
        source_tokenizer = source_tokenizer or tokenizer
        target_tokenizer = target_tokenizer or tokenizer

    # ALWAYS use precise mode - this is not configurable.
    # Activation-level CKA is required for correct geometric alignment.
    if (
        source_model is not None
        and target_model is not None
        and collect_activations_fn is not None
    ):
        return _probe_precise(
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_weights=source_weights,
            target_weights=target_weights,
            extract_layer_index_fn=extract_layer_index_fn,
            collect_activations_fn=collect_activations_fn,
            alignment_map=alignment_map,
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
    collect_activations_fn: Callable,
    alignment_map: VocabularyAlignmentMap | None = None,
    source_path: str = "",
    target_path: str = "",
    backend: "Backend | None" = None,
) -> ProbeResult:
    """Precise probe mode: Run ALL probes through BOTH models.

    No configuration - all 403 probes are always run. The probe corpus
    was designed to cover the concept space. Limiting probes degrades
    coverage with no benefit.
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

    token_id_map: dict[int, int] | None = None
    if alignment_map is not None:
        token_id_map = build_token_id_map(alignment_map)
        if token_id_map:
            logger.info(
                "PROBE PRECISE: Using aligned token map (%d tokens).",
                len(token_id_map),
            )

    for probe in probes:
        try:
            probe_text = None
            source_ids: list[int] | None = None
            target_ids: list[int] | None = None

            for candidate in probe.support_texts or []:
                if not candidate or len(candidate.strip()) < 2:
                    continue
                if token_id_map is None:
                    probe_text = candidate
                    break
                candidate_source_ids = _encode_probe_ids(
                    source_tokenizer, candidate, add_special_tokens=False
                )
                candidate_target_ids = map_token_ids(candidate_source_ids, token_id_map)
                if candidate_target_ids is None:
                    continue
                probe_text = candidate
                source_ids = candidate_source_ids
                target_ids = candidate_target_ids
                break

            if probe_text is None:
                probes_failed += 1
                continue

            source_acts = collect_activations_fn(
                source_model,
                source_tokenizer,
                probe_text,
                token_ids=source_ids,
            )
            target_acts = collect_activations_fn(
                target_model,
                target_tokenizer,
                probe_text,
                token_ids=target_ids,
            )

            # Also collect intermediate MLP activations for multi-space stitching
            # These are needed to compute stitches for the intermediate dimension
            # (e.g., 2560→4864 for SmolLM→Qwen cross-architecture merges)
            source_intermediate_acts = collect_intermediate_activations_mlx(
                source_model,
                source_tokenizer,
                probe_text,
                token_ids=source_ids,
            )
            target_intermediate_acts = collect_intermediate_activations_mlx(
                target_model,
                target_tokenizer,
                probe_text,
                token_ids=target_ids,
            )

            # Also collect attention activations for attention weight stitching
            # Returns TWO dicts: Q activations and KV activations (for GQA models)
            # Q: (e.g., 960=15*64 for SmolLM → 896=14*64 for Qwen) - for q_proj/o_proj
            # KV: (e.g., 320=5*64 for SmolLM → 128=2*64 for Qwen) - for k_proj/v_proj
            source_attention_acts, source_kv_acts = collect_attention_activations_mlx(
                source_model,
                source_tokenizer,
                probe_text,
                token_ids=source_ids,
            )
            target_attention_acts, target_kv_acts = collect_attention_activations_mlx(
                target_model,
                target_tokenizer,
                probe_text,
                token_ids=target_ids,
            )

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

            if probes_processed % 50 == 0:
                logger.info(
                    "PROBE PRECISE: Processed %d/%d probes...",
                    probes_processed,
                    len(probes),
                )

        except Exception as e:
            logger.debug("Probe '%s' failed: %s", probe.probe_id, e)
            probes_failed += 1
            continue

    logger.info(
        "PROBE PRECISE: Completed %d probes (%d failed), built %d fingerprints",
        probes_processed,
        probes_failed,
        len(source_fingerprints),
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
                "PROBE PRECISE: Built IntersectionMap with overall_correlation=%.3f, %d layers",
                intersection_map_obj.overall_correlation,
                len(intersection_map_obj.layer_confidences),
            )
        except Exception as e:
            logger.warning("Failed to build IntersectionMap: %s", e)
            intersection_map_obj = None

    # Compute per-layer CKA directly from activation stacks
    if source_layer_activations and target_layer_activations:
        for layer_idx in sorted(source_layer_activations.keys()):
            if layer_idx not in target_layer_activations:
                continue
            src_list = source_layer_activations[layer_idx]
            tgt_list = target_layer_activations[layer_idx]
            n_samples = min(len(src_list), len(tgt_list))
            if n_samples < 2:
                continue
            try:
                src_stacked = b.stack(src_list[:n_samples], axis=0)
                tgt_stacked = b.stack(tgt_list[:n_samples], axis=0)
                b.eval(src_stacked, tgt_stacked)
                cka_result = compute_cka(
                    src_stacked,
                    tgt_stacked,
                    backend=b,
                    estimator=HSICEstimator.AUTO,
                    feature_bias_correction=True,
                )
                if cka_result.is_valid:
                    layer_cka_scores_raw[layer_idx] = cka_result.cka
                    layer_cka_scores[layer_idx] = (
                        cka_result.cka_corrected
                        if cka_result.cka_corrected is not None
                        else cka_result.cka
                    )
            except Exception as e:
                logger.debug("LAYER %d: CKA computation failed: %s", layer_idx, e)

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

    conf_vals = list(layer_confidences.values())
    mean_confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
    cka_vals = list(layer_cka_scores.values())
    mean_cka = sum(cka_vals) / len(cka_vals) if cka_vals else 0.0
    min_cka = min(cka_vals) if cka_vals else 0.0
    raw_cka_vals = list(layer_cka_scores_raw.values())
    mean_cka_raw = sum(raw_cka_vals) / len(raw_cka_vals) if raw_cka_vals else 0.0
    min_cka_raw = min(raw_cka_vals) if raw_cka_vals else 0.0
    layers_with_data = set(source_layer_activations.keys()) & set(target_layer_activations.keys())
    missing_cka_layers = [layer for layer in layers_with_data if layer not in layer_cka_scores]
    perfect_alignment = (
        bool(layers_with_data)
        and not missing_cka_layers
        and min_cka == 1.0
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
        "mean_confidence": mean_confidence,
        "mean_cka": mean_cka,
        "min_cka": min_cka,
        "mean_cka_raw": mean_cka_raw,
        "min_cka_raw": min_cka_raw,
        "cka_estimator": "auto",
        "feature_bias_correction": True,
        "perfect_alignment": perfect_alignment,
        "min_confidence": min(layer_confidences.values()) if layer_confidences else 0.0,
        "max_confidence": max(layer_confidences.values()) if layer_confidences else 0.0,
        "atlas_sources": list(set(p.source.value for p in probes)),
        "atlas_domains": list(set(p.domain.value for p in probes)),
        "intersection_map_built": intersection_map_obj is not None,
        "overall_correlation": (
            intersection_map_obj.overall_correlation if intersection_map_obj else 0.0
        ),
    }

    logger.info(
        "PROBE PRECISE: %d layers, mean_confidence=%.3f, overall_correlation=%.3f",
        len(layer_confidences),
        mean_confidence,
        metrics["overall_correlation"],
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
        source_kv_activations=source_kv_activations,
        target_kv_activations=target_kv_activations,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
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
        k = max(1, int(math.ceil(math.log2(dim + 1))))

    # Derive threshold from dtype precision scaled by max magnitude
    abs_np = b.to_numpy(abs_vals)
    max_magnitude = float(max(abs_np)) if len(abs_np) > 0 else 0.0
    if threshold is None:
        eps = machine_epsilon(b, activation_vector)
        # Threshold at sqrt(eps) * max - standard numerical tolerance
        threshold = math.sqrt(eps) * max_magnitude

    # Negate for descending argsort
    neg_abs = -abs_vals
    b.eval(neg_abs)
    top_indices_arr = b.argsort(neg_abs)[:k]
    b.eval(top_indices_arr)
    top_indices = b.to_numpy(top_indices_arr).tolist()

    # Get values from array
    act_np = b.to_numpy(activation_vector)

    return [
        ActivatedDimension(
            index=int(idx),
            activation=float(act_np[idx]),
        )
        for idx in sorted(top_indices)
        if abs_np[idx] > threshold
    ]


def collect_layer_activations_mlx(
    model: Any,
    tokenizer: Any,
    text: str,
    token_ids: list[int] | None = None,
) -> dict[int, "Array"]:
    """
    Collect per-layer hidden state activations for a text input (MLX backend).

    Runs the text through the model and extracts the final hidden state
    (mean-pooled over sequence length) at each layer.

    Returns MLX arrays directly (no numpy conversion).
    """
    import mlx.core as mx

    if token_ids is None:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)
    input_ids = mx.array([token_ids])

    activations: dict[int, "Array"] = {}

    try:
        if hasattr(model, "forward_with_hidden_states"):
            _, hidden_states = model.forward_with_hidden_states(input_ids)
            for layer_idx, hidden in enumerate(hidden_states):
                pooled = mx.mean(hidden, axis=(0, 1))
                mx.eval(pooled)
                activations[layer_idx] = pooled
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(input_ids)
            elif hasattr(model.model, "wte"):
                h = model.model.wte(input_ids)
            else:
                h = model.embed(input_ids) if hasattr(model, "embed") else None

            if h is not None:
                for layer_idx, layer in enumerate(model.model.layers):
                    # Layer may return single tensor or (tensor, cache) tuple
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result
                    pooled = mx.mean(h, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
        else:
            output = model(input_ids)
            mx.eval(output)
            pooled = mx.mean(output, axis=(0, 1))
            mx.eval(pooled)
            activations[0] = pooled

    except Exception as e:
        logger.warning("Activation collection failed for text '%s...': %s", text[:30], e)

    if not activations:
        logger.debug("No activations collected for text: %s", text[:50])

    return activations


def collect_intermediate_activations_mlx(
    model: Any,
    tokenizer: Any,
    text: str,
    token_ids: list[int] | None = None,
) -> dict[int, "Array"]:
    """
    Collect per-layer MLP intermediate activations for a text input (MLX backend).

    Captures the activation INSIDE the MLP (after gate_proj * up_proj, before down_proj).
    This is the intermediate representation space, distinct from the hidden space.

    Shape: [intermediate_dim] (e.g., 2560 for SmolLM, 4864 for Qwen)

    Returns MLX arrays directly (no numpy conversion).
    """
    import mlx.core as mx
    from mlx import nn

    if token_ids is None:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)
    input_ids = mx.array([token_ids])

    activations: dict[int, "Array"] = {}

    try:
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            logger.debug("Model structure not compatible with intermediate activation collection")
            return activations

        inner = model.model

        # Get embeddings
        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            logger.debug("Cannot find embedding layer")
            return activations

        for layer_idx, layer in enumerate(inner.layers):
            # Apply input layer norm
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            else:
                h_norm = h

            # Apply self-attention
            if hasattr(layer, "self_attn"):
                attn_out = layer.self_attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            elif hasattr(layer, "attn"):
                attn_out = layer.attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            else:
                attn_out = mx.zeros_like(h)

            # Add residual
            h = h + attn_out

            # Post-attention norm
            if hasattr(layer, "post_attention_layernorm"):
                h_post = layer.post_attention_layernorm(h)
            elif hasattr(layer, "ln_2"):
                h_post = layer.ln_2(h)
            else:
                h_post = h

            # Extract MLP intermediate activation
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "up_proj") and hasattr(mlp, "gate_proj"):
                    # Standard SwiGLU/SiLU architecture (LLaMA, Qwen, Mistral)
                    up = mlp.up_proj(h_post)
                    gate = mlp.gate_proj(h_post)
                    # Intermediate = silu(gate) * up (before down_proj)
                    intermediate = nn.silu(gate) * up
                    mx.eval(intermediate)
                    # Mean pool over sequence
                    pooled = mx.mean(intermediate, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
                elif hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
                    # GPT-style MLP (fc1 -> activation -> fc2)
                    intermediate = mlp.fc1(h_post)
                    mx.eval(intermediate)
                    pooled = mx.mean(intermediate, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
                else:
                    logger.debug("Layer %d: Unknown MLP structure", layer_idx)

            # Complete the layer forward for next iteration
            if hasattr(layer, "mlp"):
                mlp_out = layer.mlp(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

    except Exception as e:
        logger.warning("Intermediate activation collection failed: %s", e)

    return activations


def collect_attention_activations_mlx(
    model: Any,
    tokenizer: Any,
    text: str,
    token_ids: list[int] | None = None,
) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
    """
    Collect per-layer attention Q and KV activations for a text input (MLX backend).

    Returns TWO dicts:
    1. Q activations: [num_heads * head_dim] (e.g., 960 for SmolLM, 896 for Qwen)
    2. KV activations: [num_kv_heads * head_dim] (e.g., 320 for SmolLM, 128 for Qwen)

    For Grouped Query Attention (GQA) models, Q and KV have different dimensions:
    - SmolLM: Q = 15 heads × 64 = 960, KV = 5 heads × 64 = 320
    - Qwen: Q = 14 heads × 64 = 896, KV = 2 heads × 64 = 128

    We need separate GramAligner transforms for each:
    - attention_stitch: For q_proj and o_proj (Q dimension)
    - kv_stitch: For k_proj and v_proj (KV dimension)

    Returns tuple of (q_activations, kv_activations) as MLX arrays directly.
    """
    import mlx.core as mx

    if token_ids is None:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)
    input_ids = mx.array([token_ids])

    q_activations: dict[int, "Array"] = {}
    kv_activations: dict[int, "Array"] = {}

    try:
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            logger.debug("Model structure not compatible with attention activation collection")
            return q_activations, kv_activations

        inner = model.model

        # Get embeddings
        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            logger.debug("Cannot find embedding layer")
            return q_activations, kv_activations

        for layer_idx, layer in enumerate(inner.layers):
            # Apply input layer norm
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            else:
                h_norm = h

            # Get attention module
            attn = layer.self_attn if hasattr(layer, "self_attn") else getattr(layer, "attn", None)

            if attn is not None:
                # Compute Q, K, V projections
                if hasattr(attn, "q_proj"):
                    q = attn.q_proj(h_norm)
                    k = attn.k_proj(h_norm)
                    mx.eval(q)
                    mx.eval(k)

                    # Q activations: [batch, seq, num_heads * head_dim]
                    # Mean pool over sequence to get [num_heads * head_dim]
                    q_pooled = mx.mean(q, axis=(0, 1))
                    mx.eval(q_pooled)
                    q_activations[layer_idx] = q_pooled

                    # K activations: [batch, seq, num_kv_heads * head_dim]
                    # For GQA, this is smaller than Q (e.g., 320 vs 960)
                    k_pooled = mx.mean(k, axis=(0, 1))
                    mx.eval(k_pooled)
                    kv_activations[layer_idx] = k_pooled

            # Complete the layer forward for next iteration
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

    except Exception as e:
        logger.warning("Attention activation collection failed: %s", e)

    return q_activations, kv_activations
