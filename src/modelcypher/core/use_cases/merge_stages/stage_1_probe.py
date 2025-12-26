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
Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for Stage 1 probing.

    PURE GEOMETRY: Layer correspondences are computed via CKA.
    CKA measures representational similarity independent of scale/rotation.
    No arbitrary thresholds - raw CKA values are returned.
    """

    # "precise": Run probes through models, compute CKA on activations
    # "fast": Compute CKA directly on weight matrices (faster, less accurate)
    probe_mode: Literal["precise", "fast"] = "precise"

    # Maximum probes in precise mode (0 = all 403 probes)
    max_probes: int = 0


@dataclass
class ProbeResult:
    """Result of Stage 1 probing."""

    correlations: dict[str, float]
    confidences: dict[int, float]
    intersection_map: Any | None  # IntersectionMap object
    dimension_correlations: dict
    metrics: dict[str, Any]

    # Activations for downstream processing (null-space filtering, shared subspace)
    source_activations: dict[int, list[Any]] | None = None
    target_activations: dict[int, list[Any]] | None = None


def stage_probe(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    source_model: Any | None = None,
    target_model: Any | None = None,
    tokenizer: Any | None = None,
    collect_activations_fn: Callable | None = None,
    backend: "Backend | None" = None,
) -> ProbeResult:
    """
    Stage 1: Build intersection map from probe responses.

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        config: Probe configuration
        extract_layer_index_fn: Function to extract layer index from weight key
        source_model: Loaded source model (for precise mode)
        target_model: Loaded target model (for precise mode)
        tokenizer: Tokenizer (for precise mode)
        collect_activations_fn: Function to collect layer activations

    Returns:
        ProbeResult with correlations, confidences, and intersection map
    """
    if (
        config.probe_mode == "precise"
        and source_model is not None
        and target_model is not None
        and collect_activations_fn is not None
    ):
        return _probe_precise(
            source_model=source_model,
            target_model=target_model,
            tokenizer=tokenizer,
            source_weights=source_weights,
            target_weights=target_weights,
            config=config,
            extract_layer_index_fn=extract_layer_index_fn,
            collect_activations_fn=collect_activations_fn,
        )
    else:
        if config.probe_mode == "precise":
            logger.warning(
                "Precise mode requested but models not loaded. "
                "Falling back to fast mode (weight-level CKA)."
            )
        return _probe_fast(
            source_weights=source_weights,
            target_weights=target_weights,
            config=config,
            extract_layer_index_fn=extract_layer_index_fn,
        )


def _probe_precise(
    source_model: Any,
    target_model: Any,
    tokenizer: Any,
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    collect_activations_fn: Callable,
    source_path: str = "",
    target_path: str = "",
    backend: "Backend | None" = None,
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models."""
    b = backend or get_default_backend()
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ActivatedDimension,
        ActivationFingerprint,
        IntersectionMap,
        IntersectionSimilarityMode,
        build_intersection_map,
    )

    probes = UnifiedAtlasInventory.all_probes()
    num_probes = len(probes)

    if config.max_probes > 0 and config.max_probes < num_probes:
        probes = probes[: config.max_probes]
        logger.info(
            "PROBE PRECISE: Limited to %d/%d probes (max_probes=%d)",
            len(probes),
            num_probes,
            config.max_probes,
        )

    logger.info(
        "PROBE PRECISE: Running %d probes through source and target models...",
        len(probes),
    )

    source_fingerprints: list[ActivationFingerprint] = []
    target_fingerprints: list[ActivationFingerprint] = []

    source_layer_activations: dict[int, list["Array"]] = {}
    target_layer_activations: dict[int, list["Array"]] = {}

    probes_processed = 0
    probes_failed = 0

    for probe in probes:
        probe_texts = probe.support_texts
        if not probe_texts:
            probes_failed += 1
            continue

        probe_text = probe_texts[0]
        if not probe_text or len(probe_text.strip()) < 2:
            probes_failed += 1
            continue

        try:
            source_acts = collect_activations_fn(source_model, tokenizer, probe_text)
            target_acts = collect_activations_fn(target_model, tokenizer, probe_text)

            source_activated: dict[int, list[ActivatedDimension]] = {}
            target_activated: dict[int, list[ActivatedDimension]] = {}

            for layer_idx, act in source_acts.items():
                source_activated[layer_idx] = _extract_top_k_dims(act, k=32, backend=b)
                if layer_idx not in source_layer_activations:
                    source_layer_activations[layer_idx] = []
                source_layer_activations[layer_idx].append(act)

            for layer_idx, act in target_acts.items():
                target_activated[layer_idx] = _extract_top_k_dims(act, k=32, backend=b)
                if layer_idx not in target_layer_activations:
                    target_layer_activations[layer_idx] = []
                target_layer_activations[layer_idx].append(act)

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

    # Extract layer confidences from IntersectionMap
    # No fallbacks - if we don't have precise alignment, we don't merge
    layer_confidences: dict[int, float] = {}
    layer_cka_scores: dict[int, float] = {}

    if intersection_map_obj is not None:
        for lc in intersection_map_obj.layer_confidences:
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

    metrics = {
        "probe_mode": "precise",
        "probes_total": len(probes),
        "probes_processed": probes_processed,
        "probes_failed": probes_failed,
        "fingerprints_built": len(source_fingerprints),
        "layers_analyzed": len(layer_confidences),
        "layer_confidences": layer_confidences,
        "layer_cka_scores": layer_cka_scores,
        "mean_confidence": mean_confidence,
        "mean_cka": mean_cka,
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
    )


def _probe_fast(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    backend: "Backend | None" = None,
) -> ProbeResult:
    """Fast probe mode: Compute CKA directly on weight matrices.

    PURE GEOMETRY: CKA (Centered Kernel Alignment) measures representational
    similarity between weight matrices. It is invariant to isotropic scaling
    and orthogonal transformations - exactly what we need for merge alignment.
    """
    b = backend or get_default_backend()
    from modelcypher.core.domain.geometry.cka import compute_layer_cka

    weight_cka: dict[str, float] = {}
    layer_cka: dict[int, list[float]] = {}

    for key in target_weights:
        if key not in source_weights:
            continue

        source_w = source_weights[key]
        target_w = target_weights[key]

        if source_w.shape != target_w.shape:
            continue

        layer_idx = extract_layer_index_fn(key)
        if layer_idx is None:
            continue

        # Compute CKA for 2D weight matrices
        cka_score = 0.0
        if source_w.ndim == 2 and source_w.shape[0] >= 2:
            try:
                cka_result = compute_layer_cka(source_w, target_w)
                cka_score = cka_result.cka if cka_result.is_valid else 0.0
            except Exception:
                cka_score = 0.0

        weight_cka[key] = cka_score

        if layer_idx not in layer_cka:
            layer_cka[layer_idx] = []
        layer_cka[layer_idx].append(cka_score)

    # Compute per-layer CKA (mean of all weights in layer)
    layer_confidences: dict[int, float] = {}
    for layer_idx, vals in layer_cka.items():
        layer_confidences[layer_idx] = sum(vals) / len(vals) if vals else 0.0

    # Compute overall statistics
    all_cka = list(weight_cka.values())
    mean_cka = sum(all_cka) / len(all_cka) if all_cka else 0.0

    metrics = {
        "probe_mode": "fast",
        "weight_count": len(weight_cka),
        "layer_confidences": layer_confidences,
        "mean_confidence": mean_cka,
        "mean_cka": mean_cka,
        "min_cka": min(all_cka) if all_cka else 0.0,
        "max_cka": max(all_cka) if all_cka else 0.0,
    }

    logger.info("PROBE FAST: %d weights, mean_cka=%.3f", len(weight_cka), mean_cka)

    return ProbeResult(
        correlations=weight_cka,
        confidences=layer_confidences,
        intersection_map=None,
        dimension_correlations={},
        metrics=metrics,
    )


def _extract_top_k_dims(
    activation_vector: "Array",
    k: int = 32,
    threshold: float = 0.01,
    backend: "Backend | None" = None,
) -> list:
    """Extract top-k activated dimensions by magnitude."""
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

    b = backend or get_default_backend()
    abs_vals = b.abs(activation_vector)
    # Negate for descending argsort
    neg_abs = -abs_vals
    b.eval(neg_abs)
    top_indices_arr = b.argsort(neg_abs)[:k]
    b.eval(top_indices_arr)
    top_indices = b.to_numpy(top_indices_arr).tolist()

    # Get values from array
    abs_np = b.to_numpy(abs_vals)
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
) -> dict[int, "Array"]:
    """
    Collect per-layer hidden state activations for a text input (MLX backend).

    Runs the text through the model and extracts the final hidden state
    (mean-pooled over sequence length) at each layer.

    Returns MLX arrays directly (no numpy conversion).
    """
    import mlx.core as mx

    tokens = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(tokens, list):
        input_ids = mx.array([tokens])
    else:
        input_ids = mx.array([tokens.ids])

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
