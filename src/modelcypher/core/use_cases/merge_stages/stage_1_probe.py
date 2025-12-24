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
from typing import Any, Callable, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for Stage 1 probing."""

    probe_mode: Literal["precise", "fast"] = "precise"
    intersection_mode: str = "ensemble"
    max_probes: int = 0  # 0 = all probes


@dataclass
class ProbeResult:
    """Result of Stage 1 probing."""

    correlations: dict[str, float]
    confidences: dict[int, float]
    intersection_map: Optional[Any]  # IntersectionMap object
    dimension_correlations: dict
    metrics: dict[str, Any]


def stage_probe(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], Optional[int]],
    source_model: Optional[Any] = None,
    target_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    collect_activations_fn: Optional[Callable] = None,
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
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], Optional[int]],
    collect_activations_fn: Callable,
    source_path: str = "",
    target_path: str = "",
) -> ProbeResult:
    """Precise probe mode: Run probes through BOTH models."""
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

    source_layer_activations: dict[int, list[np.ndarray]] = {}
    target_layer_activations: dict[int, list[np.ndarray]] = {}

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
                source_activated[layer_idx] = _extract_top_k_dims(act, k=32)
                if layer_idx not in source_layer_activations:
                    source_layer_activations[layer_idx] = []
                source_layer_activations[layer_idx].append(act)

            for layer_idx, act in target_acts.items():
                target_activated[layer_idx] = _extract_top_k_dims(act, k=32)
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
    intersection_map_obj: Optional[IntersectionMap] = None
    dimension_correlations: dict = {}

    if source_fingerprints and target_fingerprints:
        try:
            intersection_map_obj = build_intersection_map(
                source_fingerprints=source_fingerprints,
                target_fingerprints=target_fingerprints,
                source_model=source_path or "source",
                target_model=target_path or "target",
                mode=IntersectionSimilarityMode.ENSEMBLE,
                correlation_threshold=0.3,
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

    # Extract layer confidences
    layer_confidences: dict[int, float] = {}
    layer_cka_scores: dict[int, float] = {}

    if intersection_map_obj is not None:
        for lc in intersection_map_obj.layer_confidences:
            layer_confidences[lc.layer] = lc.confidence
    else:
        # Fallback: compute CKA
        common_layers = set(source_layer_activations.keys()) & set(
            target_layer_activations.keys()
        )

        for layer_idx in sorted(common_layers):
            source_acts_list = source_layer_activations[layer_idx]
            target_acts_list = target_layer_activations[layer_idx]

            n_samples = min(len(source_acts_list), len(target_acts_list))
            if n_samples < 10:
                continue

            source_stack = np.stack(source_acts_list[:n_samples], axis=0)
            target_stack = np.stack(target_acts_list[:n_samples], axis=0)

            try:
                cka_result = compute_cka(source_stack, target_stack, use_linear_kernel=True)
                cka_score = cka_result.cka if cka_result.is_valid else 0.0
                layer_cka_scores[layer_idx] = float(cka_score)
                layer_confidences[layer_idx] = float(cka_score)
            except Exception as e:
                logger.debug("CKA failed for layer %d: %s", layer_idx, e)
                layer_confidences[layer_idx] = 0.0

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

    mean_confidence = (
        float(np.mean(list(layer_confidences.values()))) if layer_confidences else 0.0
    )
    mean_cka = float(np.mean(list(layer_cka_scores.values()))) if layer_cka_scores else 0.0

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
    )


def _probe_fast(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    config: ProbeConfig,
    extract_layer_index_fn: Callable[[str], Optional[int]],
) -> ProbeResult:
    """Fast probe mode: Use weight-level CKA (no model inference)."""
    from modelcypher.core.domain.geometry.cka import compute_layer_cka, ensemble_similarity

    intersection_map: dict[str, float] = {}
    layer_confidences: dict[int, list[float]] = {}
    cka_scores: dict[str, float] = {}
    cosine_scores: dict[str, float] = {}

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
        max_cka_dim = 512
        can_compute_cka = (
            config.intersection_mode != "jaccard"
            and source_w.ndim == 2
            and source_w.shape[0] >= 2
            and source_w.shape[0] <= max_cka_dim
        )
        if can_compute_cka:
            try:
                cka_result = compute_layer_cka(source_w, target_w)
                cka_score = cka_result.cka if cka_result.is_valid else 0.0
            except Exception:
                cka_score = 0.0
        else:
            cka_score = 0.0

        # Compute cosine similarity
        s_flat = source_w.flatten().astype(np.float32)
        t_flat = target_w.flatten().astype(np.float32)
        s_norm = np.linalg.norm(s_flat)
        t_norm = np.linalg.norm(t_flat)

        if s_norm > 1e-8 and t_norm > 1e-8:
            cosine = float(np.dot(s_flat, t_flat) / (s_norm * t_norm))
        else:
            cosine = 0.0

        # Approximate Jaccard from weight overlap
        threshold = 0.01 * max(np.abs(source_w).max(), np.abs(target_w).max())
        s_active = np.abs(source_w) > threshold
        t_active = np.abs(target_w) > threshold
        intersection_count = np.sum(s_active & t_active)
        union_count = np.sum(s_active | t_active)
        jaccard = float(intersection_count / max(union_count, 1))

        # Ensemble similarity
        if config.intersection_mode == "cka":
            confidence = cka_score
        elif config.intersection_mode == "jaccard":
            confidence = jaccard
        else:
            confidence = ensemble_similarity(
                jaccard=jaccard,
                cka=cka_score,
                cosine=cosine,
                jaccard_weight=0.6,
                cka_weight=0.4,
            )

        intersection_map[key] = float(confidence)
        cka_scores[key] = cka_score
        cosine_scores[key] = cosine

        if layer_idx not in layer_confidences:
            layer_confidences[layer_idx] = []
        layer_confidences[layer_idx].append(float(confidence))

    # Compute per-layer confidence
    layer_confidences_final: dict[int, float] = {}
    for layer_idx in layer_confidences:
        layer_confidences_final[layer_idx] = float(np.mean(layer_confidences[layer_idx]))

    mean_confidence = (
        float(np.mean(list(layer_confidences_final.values())))
        if layer_confidences_final
        else 0.0
    )
    mean_cka = float(np.mean(list(cka_scores.values()))) if cka_scores else 0.0

    metrics = {
        "probe_mode": "fast",
        "weight_count": len(intersection_map),
        "layer_confidences": layer_confidences_final,
        "mean_confidence": mean_confidence,
        "mean_cka": mean_cka,
        "min_confidence": (
            min(layer_confidences_final.values()) if layer_confidences_final else 0.0
        ),
        "max_confidence": (
            max(layer_confidences_final.values()) if layer_confidences_final else 0.0
        ),
        "intersection_mode": config.intersection_mode,
    }

    logger.info(
        "PROBE FAST: %d weights, mean_confidence=%.3f, mean_cka=%.3f",
        len(intersection_map),
        mean_confidence,
        mean_cka,
    )

    return ProbeResult(
        correlations=intersection_map,
        confidences=layer_confidences_final,
        intersection_map=None,
        dimension_correlations={},
        metrics=metrics,
    )


def _extract_top_k_dims(
    activation_vector: np.ndarray,
    k: int = 32,
    threshold: float = 0.01,
) -> list:
    """Extract top-k activated dimensions by magnitude."""
    from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

    abs_vals = np.abs(activation_vector)
    top_indices = np.argsort(-abs_vals)[:k]

    return [
        ActivatedDimension(
            index=int(idx),
            activation=float(activation_vector[idx]),
        )
        for idx in sorted(top_indices)
        if abs_vals[idx] > threshold
    ]


def collect_layer_activations_mlx(
    model: Any,
    tokenizer: Any,
    text: str,
) -> dict[int, np.ndarray]:
    """
    Collect per-layer hidden state activations for a text input (MLX backend).

    Runs the text through the model and extracts the final hidden state
    (mean-pooled over sequence length) at each layer.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(tokens, list):
        input_ids = mx.array([tokens])
    else:
        input_ids = mx.array([tokens.ids])

    activations: dict[int, np.ndarray] = {}

    try:
        if hasattr(model, "forward_with_hidden_states"):
            _, hidden_states = model.forward_with_hidden_states(input_ids)
            for layer_idx, hidden in enumerate(hidden_states):
                pooled = mx.mean(hidden, axis=(0, 1))
                mx.eval(pooled)
                activations[layer_idx] = np.array(pooled)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            if hasattr(model.model, "embed_tokens"):
                h = model.model.embed_tokens(input_ids)
            elif hasattr(model.model, "wte"):
                h = model.model.wte(input_ids)
            else:
                h = model.embed(input_ids) if hasattr(model, "embed") else None

            if h is not None:
                for layer_idx, layer in enumerate(model.model.layers):
                    h, _ = layer(h)
                    pooled = mx.mean(h, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = np.array(pooled)
        else:
            output = model(input_ids)
            mx.eval(output)
            pooled = mx.mean(output, axis=(0, 1))
            mx.eval(pooled)
            activations[0] = np.array(pooled)

    except Exception as e:
        logger.debug("Activation collection failed for text '%s...': %s", text[:30], e)

    return activations
