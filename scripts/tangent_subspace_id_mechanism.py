#!/usr/bin/env python3
"""Tangent-subspace mechanism rerun harness.

Historical note:
- The original 2026-03-07 hand-written 60-prompt run remains at
  `results/tangent_subspace_id_mechanism/results.json`.
- Promotable reruns use atlas-backed probe manifests, save a frozen manifest per
  run, and separate raw measurement output from protocol adjudication.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial import KDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")
RESULTS_ROOT = Path("results/tangent_subspace_id_mechanism")
HISTORICAL_RESULTS_PATH = RESULTS_ROOT / "results.json"
PROTOCOL_NAME = "TANGENT-SUBSPACE-ID-FALSIFIER-PROTOCOL"
ARTIFACT_SCHEMA_VERSION = "v2"
DEFAULT_REFERENCE_MODEL = "Llama-3.2-3B"
OPERATOR_MISMATCH_NOTE = (
    "Measurement C uses Euclidean KDTree neighborhoods, while TwoNN uses "
    "geodesic distances via a k-NN graph. Treat local-rank summaries as "
    "[MEASUREMENT_INVALID] for causal adjudication."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _resolve_model_base(model) -> object:
    """Return the backbone object that has both .embed_tokens and .layers."""

    def _has_both(obj) -> bool:
        return obj is not None and hasattr(obj, "embed_tokens") and hasattr(obj, "layers")

    inner = getattr(model, "model", None)
    if _has_both(inner):
        return inner

    if inner is not None:
        inner_lm = getattr(inner, "language_model", None)
        if inner_lm is not None:
            if _has_both(inner_lm):
                return inner_lm
            inner_lm_inner = getattr(inner_lm, "model", None)
            if _has_both(inner_lm_inner):
                return inner_lm_inner
            if hasattr(inner_lm, "layers"):
                return inner_lm

    lm = getattr(model, "language_model", None)
    if lm is not None:
        if _has_both(lm):
            return lm
        lm_inner = getattr(lm, "model", None)
        if _has_both(lm_inner):
            return lm_inner
        if hasattr(lm, "layers"):
            return lm

    if hasattr(model, "layers"):
        return model
    return model


MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16,
        "d": 1024,
        "architecture": "lfm2",
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16,
        "d": 1536,
        "architecture": "lfm2",
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24,
        "d": 1024,
        "architecture": "qwen3.5",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28,
        "d": 3072,
        "architecture": "llama",
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24,
        "d": 2048,
        "architecture": "qwen3.5",
    },
}


LEGACY_PROBE_PROMPTS = [
    "The capital of France is",
    "Who wrote Romeo and Juliet?",
    "The chemical symbol for water is",
    "The largest planet in our solar system is",
    "The speed of light in a vacuum is approximately",
    "The first president of the United States was",
    "The boiling point of water at sea level is",
    "The chemical formula for table salt is",
    "The tallest mountain on Earth is",
    "The currency of Japan is",
    "What is 347 + 528?",
    "What is 15 * 23?",
    "What is 1024 / 16?",
    "What is 99 - 37?",
    "What is 8 * 7 + 13?",
    "What is 256 + 384 - 100?",
    "What is 12 * 12?",
    "What is 999 - 456?",
    "What is 50 * 20 + 1?",
    "What is 128 / 4?",
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
    "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    "Write a haiku about the ocean.",
    "Describe a sunset over the mountains in one vivid sentence.",
    "Write a short poem about the passage of time.",
    "Describe the taste of your favorite food using only three words.",
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function to compute Fibonacci up to n terms.",
    "Write a Python function to find the max element without max().",
    "Once upon a time in a faraway kingdom, there lived a",
    "The old lighthouse keeper watched the storm approach from",
    "In the year 2150, humanity had finally achieved",
    "She opened the letter and read the first line:",
    "The forest was silent except for the sound of",
    "He had been walking for three days when he finally saw",
    "The library contained a secret that no one had discovered for",
    "As the last leaf fell from the ancient oak tree,",
    "The musician played a melody that made everyone in the room",
    "Deep beneath the ocean, a creature stirred for the first time in",
    "What comes next: 2, 6, 12, 20, 30, ?",
    "Three friends split $90 unequally. A gets twice what B gets. B gets twice what C gets. How much does C get?",
    "If you rearrange CIFAIPC, you get the name of a country. What is it?",
    "A train leaves A at 60 mph, another leaves B at 80 mph toward A, 280 miles apart. When do they meet?",
    "Write a one-sentence story with a twist ending.",
    "Describe the sound of rain on a tin roof.",
    "Write a metaphor for loneliness.",
    "Describe the color blue to someone who has never seen it.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python one-liner to flatten a nested list.",
    "Write a Python function to sort a list using bubble sort.",
    "Write a Python function to count words in a string.",
    "Write a Python function to compute factorial recursively.",
    "Write a Python function to merge two sorted lists.",
    "Describe the feeling of flying in one sentence.",
    "Write a two-line dialogue between the sun and the moon.",
]


def _select_probe_text(probe: Any) -> str | None:
    if hasattr(probe, "support_texts") and probe.support_texts:
        for text in probe.support_texts:
            if text and text.strip():
                return text.strip()
    description = getattr(probe, "description", "")
    if isinstance(description, str) and description.strip():
        return description.strip()
    name = getattr(probe, "name", "")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _legacy_probe_manifest(prompts: list[str]) -> list[dict[str, Any]]:
    manifest = []
    for index, prompt in enumerate(prompts):
        manifest.append(
            {
                "probe_id": f"legacy_prompt:{index:03d}",
                "source": "legacy",
                "domain": "legacy",
                "name": f"Legacy prompt {index + 1}",
                "description": "Historical hand-written tangent prompt",
                "text": prompt,
            }
        )
    return manifest


def derive_llama_probe_budget(
    reference_path: Path = HISTORICAL_RESULTS_PATH,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
) -> dict[str, Any]:
    """Derive the promotable probe budget from the historical Llama run."""
    result: dict[str, Any] = {
        "reference_path": str(reference_path),
        "reference_model": reference_model,
        "required_tangent_rank": None,
        "required_probe_count": None,
        "used_fallback": False,
    }
    if not reference_path.exists():
        result["used_fallback"] = True
        result["fallback_reason"] = "reference_results_missing"
        return result

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    for model_payload in payload.get("per_model", []):
        if model_payload.get("model_name") != reference_model:
            continue
        ids = [
            float(value)
            for value in model_payload.get("twonn_ids", [])[1:]
            if value is not None and np.isfinite(value)
        ]
        if not ids:
            result["used_fallback"] = True
            result["fallback_reason"] = "reference_model_has_no_valid_non_stage0_ids"
            return result

        required_tangent_rank = max(1, int(np.ceil(max(ids))))
        required_probe_count = (2 * required_tangent_rank) ** 2
        result["required_tangent_rank"] = required_tangent_rank
        result["required_probe_count"] = required_probe_count
        return result

    result["used_fallback"] = True
    result["fallback_reason"] = "reference_model_missing_from_results"
    return result


def _round_robin_select(entries_by_domain: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    domains = sorted(entries_by_domain)
    while len(selected) < limit and any(entries_by_domain[domain] for domain in domains):
        for domain in domains:
            bucket = entries_by_domain[domain]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
    return selected


def _build_atlas_probe_manifest(limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from modelcypher.core.domain.atlas.probe_loader import load_all_probes

    probes = load_all_probes()
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        text = _select_probe_text(probe)
        if text is None:
            continue
        domain = str(getattr(probe.domain, "value", probe.domain))
        source = str(getattr(probe.source, "value", probe.source))
        by_domain[domain].append(
            {
                "probe_id": probe.probe_id,
                "source": source,
                "domain": domain,
                "name": probe.name,
                "description": probe.description,
                "text": text,
            }
        )

    for entries in by_domain.values():
        entries.sort(key=lambda item: (item["source"], item["probe_id"], item["text"]))

    valid_probe_count = sum(len(entries) for entries in by_domain.values())
    requested = valid_probe_count if limit is None else min(limit, valid_probe_count)
    selected = _round_robin_select(by_domain, requested)
    if len(selected) < requested:
        raise RuntimeError(
            f"Atlas manifest requested {requested} probes but only {len(selected)} valid probes were selected."
        )

    return selected, {
        "probe_source": "atlas",
        "selection_strategy": "domain_round_robin",
        "valid_probe_count": valid_probe_count,
        "requested_probe_count": requested,
        "selected_probe_count": len(selected),
    }


def load_probe_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probes = payload["probes"] if isinstance(payload, dict) else payload
    return list(probes)


def load_existing_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _resolve_prompts(args) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if args.smoke or args.legacy_prompts:
        prompts = LEGACY_PROBE_PROMPTS[:12] if args.smoke else list(LEGACY_PROBE_PROMPTS)
        manifest = _legacy_probe_manifest(prompts)
        return prompts, manifest, {
            "probe_source": "legacy",
            "selection_strategy": "historical_hand_written",
            "selected_probe_count": len(prompts),
            "promotable": False,
        }

    if args.probe_manifest:
        manifest_path = Path(args.probe_manifest).expanduser().resolve()
        manifest = load_probe_manifest(manifest_path)
        prompts = [entry["text"] for entry in manifest]
        return prompts, manifest, {
            "probe_source": "manifest",
            "selection_strategy": "frozen_manifest",
            "selected_probe_count": len(prompts),
            "manifest_path": str(manifest_path),
            "promotable": True,
        }

    budget_info = derive_llama_probe_budget()
    requested_limit = args.probe_limit
    if requested_limit is None and not budget_info["used_fallback"]:
        requested_limit = int(budget_info["required_probe_count"])

    manifest, manifest_meta = _build_atlas_probe_manifest(requested_limit)
    prompts = [entry["text"] for entry in manifest]
    manifest_meta["budget_derivation"] = budget_info
    manifest_meta["promotable"] = True
    return prompts, manifest, manifest_meta


# =============================================================================
# Activation Collection
# =============================================================================


def collect_layer_activations(
    model,
    tokenizer,
    prompts: list[str],
    num_layers: int,
) -> list[np.ndarray]:
    """Collect last-token hidden states at each layer for all prompts."""
    import mlx.core as mx

    base = _resolve_model_base(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    stage_activations: list[list[np.ndarray]] = [[] for _ in range(num_layers + 1)]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        mx.eval(hidden)

        h_last = hidden[:, -1, :].astype(mx.float32)
        mx.eval(h_last)
        stage_activations[0].append(np.array(h_last[0].tolist(), dtype=np.float32))

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = None

            try:
                h_out = layer(hidden, mask=layer_mask)
            except (TypeError, ValueError):
                try:
                    h_out = layer(hidden, layer_mask)
                except (TypeError, ValueError):
                    h_out = layer(hidden)

            if isinstance(h_out, tuple):
                h_out = h_out[0]
            mx.eval(h_out)

            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_out_last)
            stage_activations[i + 1].append(np.array(h_out_last[0].tolist(), dtype=np.float32))
            hidden = h_out

    return [np.stack(acts) for acts in stage_activations]


def compute_twonn_per_layer(stage_activations: list[np.ndarray], backend) -> list[float]:
    """Compute TwoNN intrinsic dimension at each stage."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    ids: list[float] = []
    for stage_index, acts in enumerate(stage_activations):
        try:
            estimate = IntrinsicDimension.compute_two_nn(acts.tolist(), backend=backend)
            ids.append(float(estimate.intrinsic_dimension))
        except Exception as exc:
            logger.warning("  TwoNN failed at stage %d: %s", stage_index, exc)
            ids.append(float("nan"))
    return ids


# =============================================================================
# Measurement A: Shared rotation + added-direction signal
# =============================================================================


def compute_pca_tangent_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA directions of centered X. Returns [k, d] orthonormal basis."""
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(X_centered, full_matrices=False)
    basis = vt[: min(k, vt.shape[0])]
    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return basis / norms


def shared_rotation_metrics(V1: np.ndarray, V2: np.ndarray) -> dict[str, float]:
    """Grassmann metrics on the matched-rank shared subspace."""
    V1 = np.asarray(V1, dtype=np.float64)
    V2 = np.asarray(V2, dtype=np.float64)
    k_min = min(V1.shape[0], V2.shape[0])
    if k_min <= 0:
        return {
            "shared_rank": 0,
            "shared_rotation_geodesic": float("nan"),
            "shared_rotation_chordal": float("nan"),
        }

    M = V1[:k_min] @ V2[:k_min].T
    cos_angles = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    principal_angles = np.arccos(cos_angles)
    return {
        "shared_rank": int(k_min),
        "shared_rotation_geodesic": float(np.sqrt(np.sum(principal_angles**2))),
        "shared_rotation_chordal": float(np.sqrt(np.sum(np.maximum(0.0, 1.0 - cos_angles**2)))),
    }


def added_direction_signal_numpy(
    reference_basis: np.ndarray,
    candidate_basis: np.ndarray,
    *,
    residual_floor: float | None = None,
) -> dict[str, Any]:
    """Measure candidate directions outside the reference span.

    Each candidate basis vector is projected onto the reference span. Residual
    norm > sqrt(eps) counts as an added off-span direction.
    """
    reference_basis = np.asarray(reference_basis, dtype=np.float64)
    candidate_basis = np.asarray(candidate_basis, dtype=np.float64)
    if residual_floor is None:
        residual_floor = float(np.sqrt(np.finfo(np.float32).eps))

    if reference_basis.size == 0 or candidate_basis.size == 0:
        return {
            "residual_floor": residual_floor,
            "count_above_floor": 0,
            "max_residual_norm": 0.0,
            "mean_residual_norm": 0.0,
            "total_residual_energy": 0.0,
            "residual_norms": [],
        }

    projection = reference_basis @ candidate_basis.T
    projected_norm_sq = np.sum(projection**2, axis=0)
    projected_norm_sq = np.clip(projected_norm_sq, 0.0, 1.0)
    residual_norm_sq = np.maximum(0.0, 1.0 - projected_norm_sq)
    residual_norms = np.sqrt(residual_norm_sq)
    return {
        "residual_floor": residual_floor,
        "count_above_floor": int(np.sum(residual_norms > residual_floor)),
        "max_residual_norm": float(np.max(residual_norms)) if residual_norms.size else 0.0,
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms.size else 0.0,
        "total_residual_energy": float(np.sum(residual_norm_sq)),
        "residual_norms": residual_norms.tolist(),
    }


def measurement_a_global_tangent(
    stage_activations: list[np.ndarray],
    twonn_ids: list[float],
) -> list[dict[str, Any]]:
    """Shared-rotation and added-direction observables between consecutive stages."""
    results: list[dict[str, Any]] = []
    for layer_index in range(len(stage_activations) - 1):
        id_l = twonn_ids[layer_index]
        id_l1 = twonn_ids[layer_index + 1]
        if np.isnan(id_l) or np.isnan(id_l1):
            results.append({"layer_pair": [layer_index, layer_index + 1], "skipped": True})
            continue

        k_l = max(2, round(id_l))
        k_l1 = max(2, round(id_l1))
        V_l = compute_pca_tangent_basis(stage_activations[layer_index], k_l)
        V_l1 = compute_pca_tangent_basis(stage_activations[layer_index + 1], k_l1)
        shared = shared_rotation_metrics(V_l, V_l1)

        if V_l1.shape[0] >= V_l.shape[0]:
            added_side = "target_vs_source"
            reference_stage = layer_index
            candidate_stage = layer_index + 1
            added = added_direction_signal_numpy(V_l, V_l1)
        else:
            added_side = "source_vs_target"
            reference_stage = layer_index + 1
            candidate_stage = layer_index
            added = added_direction_signal_numpy(V_l1, V_l)

        results.append(
            {
                "layer_pair": [layer_index, layer_index + 1],
                "k_l": int(k_l),
                "k_l1": int(k_l1),
                "shared_rank": int(shared["shared_rank"]),
                "shared_rotation_geodesic": shared["shared_rotation_geodesic"],
                "shared_rotation_chordal": shared["shared_rotation_chordal"],
                "added_direction_side": added_side,
                "reference_stage": int(reference_stage),
                "candidate_stage": int(candidate_stage),
                "added_direction_count_eps": int(added["count_above_floor"]),
                "added_direction_total_residual": added["total_residual_energy"],
                "added_direction_mean_residual": added["mean_residual_norm"],
                "added_direction_max_residual": added["max_residual_norm"],
                "added_direction_floor": added["residual_floor"],
                "added_direction_residuals": added["residual_norms"],
            }
        )
    return results


# =============================================================================
# Measurement B: Local tangent alignment
# =============================================================================


def _measurement_b_payload(result, source_layer: int, target_layer: int) -> dict[str, Any]:
    return {
        "layer_pair": [source_layer, target_layer],
        "anchor_count": int(result.anchor_count),
        "neighbor_count": int(result.neighbor_count),
        "tangent_rank": int(result.tangent_rank),
        "mean_angle_radians": float(result.mean_angle_radians),
        "median_angle_radians": float(result.median_angle_radians),
        "mean_cosine": float(result.mean_cosine),
        "coverage": float(result.coverage),
    }


def measurement_b_local_tangent(
    stage_activations: list[np.ndarray],
    backend,
) -> list[dict[str, Any]]:
    """Local tangent alignment between consecutive layers."""
    from modelcypher.core.domain.geometry.tangent_space_alignment import TangentSpaceAlignment

    aligner = TangentSpaceAlignment(backend)
    results: list[dict[str, Any]] = []
    clear_cache = getattr(backend, "clear_cache", None)
    synchronize = getattr(backend, "synchronize", None)

    for layer_index in range(len(stage_activations) - 1):
        X_l = stage_activations[layer_index]
        X_l1 = stage_activations[layer_index + 1]
        pts_l = backend.array(X_l.tolist())
        pts_l1 = backend.array(X_l1.tolist())
        backend.eval(pts_l, pts_l1)

        try:
            result = aligner.compute_layer_metrics(
                pts_l,
                pts_l1,
                source_layer=layer_index,
                target_layer=layer_index + 1,
            )
            if result is None:
                results.append(
                    {
                        "layer_pair": [layer_index, layer_index + 1],
                        "skipped": True,
                        "reason": "insufficient_data",
                    }
                )
            else:
                results.append(_measurement_b_payload(result, layer_index, layer_index + 1))
        except Exception as exc:
            logger.warning("  Measurement B failed at pair (%d, %d): %s", layer_index, layer_index + 1, exc)
            results.append(
                {
                    "layer_pair": [layer_index, layer_index + 1],
                    "skipped": True,
                    "reason": str(exc)[:200],
                }
            )

        del pts_l, pts_l1
        if callable(synchronize):
            try:
                synchronize()
            except Exception:
                pass
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                pass
        gc.collect()

    return results


# =============================================================================
# Measurement C: Exploratory local-rank telemetry
# =============================================================================


def participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.sum(eigenvalues)
    if total <= 0:
        return 0.0
    sum_sq = np.sum(eigenvalues**2)
    if sum_sq <= 0:
        return 0.0
    return float(total**2 / sum_sq)


def local_effective_rank(diff_matrix: np.ndarray) -> float:
    if diff_matrix.shape[0] < 2:
        return 0.0
    centered = np.asarray(diff_matrix, dtype=np.float64)
    if not np.isfinite(centered).all():
        return 0.0
    centered = centered - centered.mean(axis=0)
    gram = centered @ centered.T
    if not np.isfinite(gram).all():
        return 0.0
    eigenvalues = np.linalg.eigvalsh(gram)
    return participation_ratio(eigenvalues)


def measurement_c_tracked_neighbors(stage_activations: list[np.ndarray]) -> list[dict[str, Any]]:
    """Exploratory Euclidean-neighborhood rank telemetry."""
    results: list[dict[str, Any]] = []
    N = stage_activations[0].shape[0]
    k_neighbors = max(int(np.ceil(np.log(max(N, 2)))), N // 4)
    k_neighbors = max(2, min(k_neighbors, N - 1))
    logger.info("  Measurement C: k=%d neighbors (N=%d)", k_neighbors, N)

    for layer_index in range(len(stage_activations) - 1):
        X_l = stage_activations[layer_index]
        X_l1 = stage_activations[layer_index + 1]
        tree = KDTree(X_l)
        _, neighbor_indices = tree.query(X_l, k=k_neighbors + 1)
        neighbor_indices = neighbor_indices[:, 1:]

        ranks_l = np.zeros(N)
        ranks_l1 = np.zeros(N)
        for point_index in range(N):
            nn_idx = neighbor_indices[point_index]
            diff_l = X_l[nn_idx] - X_l[point_index]
            diff_l1 = X_l1[nn_idx] - X_l1[point_index]
            ranks_l[point_index] = local_effective_rank(diff_l)
            ranks_l1[point_index] = local_effective_rank(diff_l1)

        delta_ranks = ranks_l1 - ranks_l
        results.append(
            {
                "layer_pair": [layer_index, layer_index + 1],
                "k_neighbors": int(k_neighbors),
                "mean_delta_local_rank": float(np.mean(delta_ranks)),
                "std_delta_local_rank": float(np.std(delta_ranks)),
                "mean_rank_l": float(np.mean(ranks_l)),
                "mean_rank_l1": float(np.mean(ranks_l1)),
                "measurement_status": "[MEASUREMENT_INVALID]",
                "measurement_caveat": OPERATOR_MISMATCH_NOTE,
            }
        )

    return results


# =============================================================================
# Correlation summaries
# =============================================================================


def _safe_spearman_record(
    x: list[float],
    y: list[float],
    *,
    x_label: str,
    y_label: str,
    status: str = "[EXPLORATORY]",
    note: str | None = None,
) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    record: dict[str, Any] = {
        "status": status,
        "x_label": x_label,
        "y_label": y_label,
        "n": int(len(x_valid)),
        "spearman_r": None,
        "p_value": None,
    }
    if note is not None:
        record["note"] = note
    if len(x_valid) < 4:
        record["note"] = f"{record.get('note', '')} insufficient_data".strip()
        return record
    r_value, p_value = scipy_stats.spearmanr(x_valid, y_valid)
    record["spearman_r"] = float(r_value)
    record["p_value"] = float(p_value)
    return record


def compute_observable_correlations(
    meas_a: list[dict[str, Any]],
    meas_b: list[dict[str, Any]],
    meas_c: list[dict[str, Any]],
    twonn_ids: list[float],
) -> dict[str, Any]:
    delta_ids = [twonn_ids[i + 1] - twonn_ids[i] for i in range(len(twonn_ids) - 1)]
    abs_delta_ids = [abs(delta) for delta in delta_ids]

    def _view(start_index: int) -> dict[str, Any]:
        delta_view = delta_ids[start_index:]
        abs_delta_view = abs_delta_ids[start_index:]
        meas_a_view = meas_a[start_index:]
        meas_b_view = meas_b[start_index:]
        meas_c_view = meas_c[start_index:]

        increasing_indices = [i for i, delta in enumerate(delta_view) if delta > 0]

        return {
            "shared_rotation_vs_abs_delta_id": _safe_spearman_record(
                [item.get("shared_rotation_geodesic", float("nan")) for item in meas_a_view],
                abs_delta_view,
                x_label="shared_rotation_geodesic",
                y_label="abs_delta_id",
            ),
            "added_direction_count_vs_positive_delta_id": _safe_spearman_record(
                [
                    meas_a_view[index].get("added_direction_count_eps", float("nan"))
                    for index in increasing_indices
                ],
                [delta_view[index] for index in increasing_indices],
                x_label="added_direction_count_eps",
                y_label="delta_id_when_positive",
            ),
            "added_direction_energy_vs_positive_delta_id": _safe_spearman_record(
                [
                    meas_a_view[index].get("added_direction_total_residual", float("nan"))
                    for index in increasing_indices
                ],
                [delta_view[index] for index in increasing_indices],
                x_label="added_direction_total_residual",
                y_label="delta_id_when_positive",
            ),
            "local_angle_vs_abs_delta_id": _safe_spearman_record(
                [item.get("mean_angle_radians", float("nan")) for item in meas_b_view],
                abs_delta_view,
                x_label="mean_angle_radians",
                y_label="abs_delta_id",
            ),
            "local_rank_vs_delta_id": _safe_spearman_record(
                [item.get("mean_delta_local_rank", float("nan")) for item in meas_c_view],
                delta_view,
                x_label="mean_delta_local_rank",
                y_label="delta_id",
                status="[MEASUREMENT_INVALID]",
                note=OPERATOR_MISMATCH_NOTE,
            ),
        }

    valid_ids = [(index, value) for index, value in enumerate(twonn_ids) if not np.isnan(value)]
    highway_context: dict[str, Any] = {"highway_stage": None, "highway_id": None}
    if valid_ids:
        highway_stage, highway_id = min(valid_ids, key=lambda item: item[1])
        shared_rotation_at_highway = None
        for idx in (highway_stage - 1, highway_stage):
            if 0 <= idx < len(meas_a):
                shared_rotation_at_highway = meas_a[idx].get("shared_rotation_geodesic")
                if shared_rotation_at_highway is not None:
                    break
        highway_context = {
            "highway_stage": int(highway_stage),
            "highway_id": float(highway_id),
            "shared_rotation_at_highway_pair": shared_rotation_at_highway,
        }

    return {
        "delta_ids": delta_ids,
        "abs_delta_ids": abs_delta_ids,
        "highway_context": highway_context,
        "full": _view(start_index=0),
        "excluding_stage0": _view(start_index=1),
    }


# =============================================================================
# Status summaries / artifacts
# =============================================================================


def build_falsifier_outcome(
    *,
    run_id: str,
    probe_manifest_path: Path,
    all_results: list[dict[str, Any]],
    probe_selection: dict[str, Any],
    requested_models: list[str],
    run_complete: bool,
) -> dict[str, Any]:
    def _per_model_corr(view_key: str, corr_key: str) -> dict[str, Any]:
        return {
            result["model_name"]: result["analysis"][view_key][corr_key]
            for result in all_results
        }

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_NAME,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "probe_manifest_path": str(probe_manifest_path),
        "probe_selection": probe_selection,
        "requested_models": requested_models,
        "completed_models": [result["model_name"] for result in all_results],
        "run_complete": run_complete,
        "overall_status": "[MECHANISM_UNKNOWN]",
        "claims": {
            "shared_rotation": {
                "status": "[EXPLORATORY]",
                "detail": "Matched-rank shared rotation is measured, but no promotable gate is applied in-script.",
                "per_model_excluding_stage0": _per_model_corr(
                    "excluding_stage0",
                    "shared_rotation_vs_abs_delta_id",
                ),
            },
            "added_direction_signal": {
                "status": "[EXPLORATORY]",
                "detail": "Asymmetric residual metric is now measured directly. Promotion requires protocol-level adjudication and rerun evidence.",
                "per_model_positive_delta": _per_model_corr(
                    "full",
                    "added_direction_energy_vs_positive_delta_id",
                ),
            },
            "local_tangent_misalignment": {
                "status": "[EXPLORATORY]",
                "detail": "Candidate mechanism only. Cross-family promotion remains blocked without a second bf16 pure-attention family and repaired rerun telemetry.",
                "per_model_excluding_stage0": _per_model_corr(
                    "excluding_stage0",
                    "local_angle_vs_abs_delta_id",
                ),
            },
            "local_rank_change": {
                "status": "[MEASUREMENT_INVALID]",
                "detail": OPERATOR_MISMATCH_NOTE,
                "per_model": _per_model_corr("full", "local_rank_vs_delta_id"),
            },
        },
        "next_requirements": [
            "Keep current top-line state at [MECHANISM_UNKNOWN].",
            "Use this atlas-backed manifest for exact rerun reproducibility.",
            "Do not promote local tangent misalignment beyond [EXPLORATORY] without a second bf16 pure-attention family.",
        ],
    }


def build_results_payload(
    *,
    run_id: str,
    args,
    output_dir: Path,
    probe_manifest_path: Path,
    probe_selection: dict[str, Any],
    all_results: list[dict[str, Any]],
    requested_models: list[str],
    run_complete: bool,
) -> dict[str, Any]:
    falsifier_outcome = build_falsifier_outcome(
        run_id=run_id,
        probe_manifest_path=probe_manifest_path,
        all_results=all_results,
        probe_selection=probe_selection,
        requested_models=requested_models,
        run_complete=run_complete,
    )
    return {
        "metadata": {
            "run_id": run_id,
            "generated_at": _utc_now_iso(),
            "protocol": PROTOCOL_NAME,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "script": "tangent_subspace_id_mechanism.py",
            "historical_reference_results": str(HISTORICAL_RESULTS_PATH) if HISTORICAL_RESULTS_PATH.exists() else None,
            "n_models": len(all_results),
            "requested_models": requested_models,
            "run_complete": run_complete,
            "n_probes": int(probe_selection["selected_probe_count"]),
            "smoke": args.smoke,
            "measurement_b_enabled": not args.smoke,
            "probe_selection": probe_selection,
            "probe_manifest_path": str(probe_manifest_path),
            "output_dir": str(output_dir),
        },
        "per_model": all_results,
        "falsifier_summary": falsifier_outcome,
    }


# =============================================================================
# Single-model run
# =============================================================================


def run_single_model(
    model_name: str,
    model_info: dict[str, Any],
    probes: list[str],
    backend,
    *,
    run_b: bool = True,
) -> dict[str, Any]:
    logger.info("Loading model: %s from %s", model_name, model_info["path"])
    model, tokenizer = backend.load_model(model_info["path"])

    base = _resolve_model_base(model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0
    logger.info("Model loaded: %d layers, d=%d", num_layers, model_info.get("d", 0))

    start = time.time()
    stage_activations = collect_layer_activations(model, tokenizer, probes, num_layers)
    logger.info("  Activation collection: %.1fs (%d probes)", time.time() - start, len(probes))

    del model, tokenizer, base, layers
    gc.collect()

    start = time.time()
    twonn_ids = compute_twonn_per_layer(stage_activations, backend)
    logger.info("  TwoNN IDs: %.1fs", time.time() - start)
    for stage_index, intrinsic_dim in enumerate(twonn_ids):
        logger.info("    Stage %2d: ID = %.2f", stage_index, intrinsic_dim)

    start = time.time()
    meas_a = measurement_a_global_tangent(stage_activations, twonn_ids)
    logger.info("  Measurement A (shared rotation + added direction): %.1fs", time.time() - start)

    if run_b:
        start = time.time()
        meas_b = measurement_b_local_tangent(stage_activations, backend)
        logger.info("  Measurement B (local tangent): %.1fs", time.time() - start)
    else:
        logger.info("  Measurement B: SKIPPED (--smoke)")
        meas_b = [
            {"layer_pair": [layer_index, layer_index + 1], "skipped": True, "reason": "smoke_mode"}
            for layer_index in range(len(stage_activations) - 1)
        ]

    start = time.time()
    meas_c = measurement_c_tracked_neighbors(stage_activations)
    logger.info("  Measurement C (exploratory local rank): %.1fs", time.time() - start)

    analysis = compute_observable_correlations(meas_a, meas_b, meas_c, twonn_ids)
    logger.info("  --- Correlation summaries for %s ---", model_name)
    for view_name in ("full", "excluding_stage0"):
        view = analysis[view_name]
        for label, record in view.items():
            r_value = record.get("spearman_r")
            p_value = record.get("p_value")
            if r_value is None:
                logger.info("    %s / %s: n=%s (%s)", view_name, label, record["n"], record["status"])
            else:
                logger.info(
                    "    %s / %s: r=%+.3f, p=%.4f, n=%d, status=%s",
                    view_name,
                    label,
                    r_value,
                    p_value,
                    record["n"],
                    record["status"],
                )

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "n_probes": len(probes),
        "twonn_ids": twonn_ids,
        "measurement_a": meas_a,
        "measurement_b": meas_b,
        "measurement_c": meas_c,
        "analysis": analysis,
    }


# =============================================================================
# Experiment orchestration
# =============================================================================


def run_experiment(args):
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()
    provisional_run_id = _run_id()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else RESULTS_ROOT / provisional_run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    falsifier_path = output_dir / "falsifier_outcome.json"
    existing_payload = load_existing_results(results_path) if args.resume else None
    run_id = (
        existing_payload.get("metadata", {}).get("run_id")
        if existing_payload is not None
        else None
    ) or provisional_run_id

    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B"]
        run_b = False
    elif args.models:
        model_names = args.models
        run_b = True
    else:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B", "Llama-3.2-3B"]
        run_b = True

    probes, probe_manifest, probe_selection = _resolve_prompts(args)
    probe_manifest_payload = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "protocol": PROTOCOL_NAME,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "selection": probe_selection,
        "probes": probe_manifest,
    }
    probe_manifest_path = output_dir / "probe_manifest.json"
    _write_json(probe_manifest_path, probe_manifest_payload)

    logger.info("=" * 72)
    logger.info(
        "TANGENT SUBSPACE ID MECHANISM (%d models, %d probes, source=%s, B=%s)",
        len(model_names),
        len(probes),
        probe_selection["probe_source"],
        "ON" if run_b else "OFF",
    )
    logger.info("=" * 72)

    all_results: list[dict[str, Any]] = list(existing_payload.get("per_model", [])) if existing_payload else []
    completed_models = {
        result["model_name"]
        for result in all_results
        if isinstance(result, dict) and result.get("model_name")
    }
    if completed_models:
        logger.info(
            "Resuming %s with completed models: %s",
            output_dir,
            ", ".join(sorted(completed_models)),
        )
    for model_name in model_names:
        if model_name in completed_models:
            logger.info("Skipping already-completed model: %s", model_name)
            continue
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model: %s, skipping", model_name)
            continue
        model_path = MODEL_REGISTRY[model_name]["path"]
        if not os.path.exists(model_path):
            logger.warning("Model path not found: %s, skipping", model_path)
            continue

        result = run_single_model(model_name, MODEL_REGISTRY[model_name], probes, backend, run_b=run_b)
        all_results.append(result)
        checkpoint_payload = build_results_payload(
            run_id=run_id,
            args=args,
            output_dir=output_dir,
            probe_manifest_path=probe_manifest_path,
            probe_selection=probe_selection,
            all_results=all_results,
            requested_models=model_names,
            run_complete=False,
        )
        _write_json(results_path, checkpoint_payload)
        _write_json(falsifier_path, checkpoint_payload["falsifier_summary"])
        logger.info("Checkpointed partial results to %s", results_path)
        completed_models.add(model_name)
        gc.collect()

    if not all_results:
        logger.error("No models were evaluated. Check model paths.")
        return

    results_payload = build_results_payload(
        run_id=run_id,
        args=args,
        output_dir=output_dir,
        probe_manifest_path=probe_manifest_path,
        probe_selection=probe_selection,
        all_results=all_results,
        requested_models=model_names,
        run_complete=True,
    )
    _write_json(results_path, results_payload)
    _write_json(falsifier_path, results_payload["falsifier_summary"])
    logger.info("Results saved to %s", results_path)
    logger.info("Falsifier outcome saved to %s", falsifier_path)


def main():
    parser = argparse.ArgumentParser(description="Tangent subspace ID mechanism rerun harness")
    parser.add_argument("--smoke", action="store_true", help="Quick test (2 models, 12 legacy prompts)")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--probe-manifest", type=str, help="Reuse a frozen probe manifest JSON")
    parser.add_argument("--probe-limit", type=int, help="Override atlas-backed probe count")
    parser.add_argument("--legacy-prompts", action="store_true", help="Use historical hand-written prompts instead of atlas probes")
    parser.add_argument("--output-dir", type=str, help="Optional artifact output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing partial results.json in --output-dir")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
