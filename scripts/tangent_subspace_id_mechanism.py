#!/usr/bin/env python3
"""Tangent Subspace ID Mechanism: What determines TwoNN intrinsic dimension?

Resolves the [MECHANISM_UNKNOWN] gap in the causal chain (OPEN-MATHEMATICAL-QUESTIONS.md).
All previous mechanisms refuted: M1 global/local k_eff, M2 curvature bias, M3 scale.

New hypothesis H_T: TwoNN ID changes when the layer Jacobian J_l = I + dF_l/dh_l
ROTATES the data manifold's tangent subspace — introducing novel directions or
collapsing existing ones. This is different from M1 because it measures subspace
ROTATION (which directions change), not eigenvalue magnitudes (how much variance).

Three measurement channels:
A. Global tangent subspace analysis (Grassmann distance between PCA subspaces)
B. Local tangent alignment (k-NN tangent bases via TangentSpaceAlignment)
C. Tracked neighbor rank change (same neighbors tracked across layers)

Five pre-registered predictions:
P1: Spearman(d_G, |delta_ID|) > 0.3 on all models
P2: Spearman(novel_count, delta_ID) > 0.3 when delta_ID > 0
P3: Highway d_G < median(d_G)
P4: Spearman(mean_local_angle, |delta_ID|) > 0.3 on all models
P5: Spearman(delta_local_rank, delta_ID) > 0.3 on all models

Three falsification criteria:
F1: P1 < 0.3 on ANY model -> global tangent rotation falsified
F2: Sign of P1 differs between models -> mechanism underspecified
F3: P5 < 0.3 on ALL models -> local mechanism different from global

Usage:
    poetry run python scripts/tangent_subspace_id_mechanism.py --smoke
    poetry run python scripts/tangent_subspace_id_mechanism.py
    poetry run python scripts/tangent_subspace_id_mechanism.py --models LFM2-350M Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial import KDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")


def _resolve_existing_path(*candidates: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


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
        "L": 16, "d": 1024,
        "architecture": "lfm2",
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16, "d": 1536,
        "architecture": "lfm2",
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24, "d": 2048,
        "architecture": "qwen3.5",
    },
}

PROBE_PROMPTS = [
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


# =============================================================================
# Activation Collection (from covariance_rank_id_analysis.py pattern)
# =============================================================================


def collect_layer_activations(
    model, tokenizer, prompts: list[str], num_layers: int,
) -> list[np.ndarray]:
    """Collect last-token hidden states at each layer for all prompts.

    Returns list of [N, d] arrays. Index 0 = embedding output,
    index i+1 = output of layer i.
    """
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
            stage_activations[i + 1].append(
                np.array(h_out_last[0].tolist(), dtype=np.float32)
            )

            hidden = h_out

    return [np.stack(acts) for acts in stage_activations]


# =============================================================================
# TwoNN ID per layer
# =============================================================================


def compute_twonn_per_layer(stage_activations: list[np.ndarray], backend) -> list[float]:
    """Compute TwoNN intrinsic dimension at each layer stage."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    ids: list[float] = []
    for i, acts in enumerate(stage_activations):
        try:
            estimate = IntrinsicDimension.compute_two_nn(
                acts.tolist(), backend=backend,
            )
            ids.append(estimate.intrinsic_dimension)
        except Exception as e:
            logger.warning(f"  TwoNN failed at stage {i}: {e}")
            ids.append(float("nan"))
    return ids


# =============================================================================
# Measurement A: Global Tangent Subspace Analysis
# =============================================================================


def compute_pca_tangent_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA directions of centered X. Returns [k, d] orthonormal basis."""
    X_centered = X - X.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k_actual = min(k, Vt.shape[0])
    return Vt[:k_actual]


def grassmann_distance_numpy(V1: np.ndarray, V2: np.ndarray) -> dict:
    """Grassmann distance between [k1, d] and [k2, d] orthonormal bases.

    When k2 > k1, SVD of V1 @ V2.T gives k1 principal angles. The remaining
    k2 - k1 directions in V2 are in the null space of the projection (cos = 0
    by construction) — they are automatically novel.
    """
    k1, k2 = V1.shape[0], V2.shape[0]
    M = V1 @ V2.T
    cos_angles = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    principal_angles = np.arccos(cos_angles)

    geodesic = float(np.sqrt(np.sum(principal_angles**2)))
    chordal = float(np.sqrt(np.sum(np.maximum(0.0, 1.0 - cos_angles**2))))

    # Novel direction count: cos(theta) < sqrt(eps) (same threshold as subspace.py)
    # Plus implicit novel directions when k2 > k1 (null space of projection)
    sqrt_eps = float(np.sqrt(np.finfo(np.float32).eps))
    novel_from_angles = int(np.sum(cos_angles < sqrt_eps))
    novel_implicit = max(0, k2 - k1)  # directions in V2 beyond V1's span
    novel_count = novel_from_angles + novel_implicit

    return {
        "geodesic_distance": geodesic,
        "chordal_distance": chordal,
        "novel_count": novel_count,
        "novel_from_angles": novel_from_angles,
        "novel_implicit": novel_implicit,
        "n_angles": len(cos_angles),
    }


def measurement_a_global_tangent(
    stage_activations: list[np.ndarray], twonn_ids: list[float],
) -> list[dict]:
    """Grassmann distance between consecutive-layer PCA tangent subspaces."""
    results = []
    n_stages = len(stage_activations)

    for l in range(n_stages - 1):
        id_l = twonn_ids[l]
        id_l1 = twonn_ids[l + 1]

        if np.isnan(id_l) or np.isnan(id_l1):
            results.append({"layer_pair": [l, l + 1], "skipped": True})
            continue

        k_l = max(2, round(id_l))
        k_l1 = max(2, round(id_l1))

        V_l = compute_pca_tangent_basis(stage_activations[l], k_l)
        V_l1 = compute_pca_tangent_basis(stage_activations[l + 1], k_l1)

        # Full (unmatched) Grassmann comparison: SVD of k_l x k_l1 matrix
        # gives min(k_l, k_l1) principal angles. Directions beyond min(k_l, k_l1)
        # in the larger subspace are in the null space — automatically novel.
        grassmann_full = grassmann_distance_numpy(V_l, V_l1)

        # Matched dimensions for fair Grassmann distance (same subspace dim)
        k_min = min(V_l.shape[0], V_l1.shape[0])
        grassmann_matched = grassmann_distance_numpy(V_l[:k_min], V_l1[:k_min])

        results.append({
            "layer_pair": [l, l + 1],
            "k_l": k_l,
            "k_l1": k_l1,
            "k_matched": k_min,
            "grassmann_geodesic": grassmann_matched["geodesic_distance"],
            "grassmann_chordal": grassmann_matched["chordal_distance"],
            "novel_count_matched": grassmann_matched["novel_count"],
            "novel_count_full": grassmann_full["novel_count"],
            "novel_from_angles": grassmann_full["novel_from_angles"],
            "novel_implicit": grassmann_full["novel_implicit"],
            "extra_dims": abs(k_l1 - k_l),
            "n_angles_matched": grassmann_matched["n_angles"],
            "n_angles_full": grassmann_full["n_angles"],
            "grassmann_full_geodesic": grassmann_full["geodesic_distance"],
        })

    return results


# =============================================================================
# Measurement B: Local Tangent Alignment
# =============================================================================


def measurement_b_local_tangent(
    stage_activations: list[np.ndarray], backend,
) -> list[dict]:
    """Local tangent alignment between consecutive layers.

    Uses TangentSpaceAlignment to compare local geometry at each anchor
    point between layer l and layer l+1.
    """
    from modelcypher.core.domain.geometry.tangent_space_alignment import (
        TangentSpaceAlignment,
    )

    from modelcypher.core.domain.geometry.scalars import sqrt_scalar

    aligner = TangentSpaceAlignment(backend)
    results = []
    n_stages = len(stage_activations)

    # Log operator parameters (derived from N, not user-set)
    n_anchors = stage_activations[0].shape[0]
    neighbor_count = min(max(2, int(sqrt_scalar(float(n_anchors), backend))), n_anchors - 1)
    tangent_rank = min(max(1, neighbor_count // 2), neighbor_count)
    logger.info(f"    Measurement B operator: N={n_anchors}, neighbor_count={neighbor_count}, tangent_rank={tangent_rank}")

    for l in range(n_stages - 1):
        X_l = stage_activations[l]
        X_l1 = stage_activations[l + 1]

        pts_l = backend.array(X_l.tolist())
        pts_l1 = backend.array(X_l1.tolist())
        backend.eval(pts_l, pts_l1)

        try:
            result = aligner.compute_layer_metrics(
                pts_l, pts_l1, source_layer=l, target_layer=l + 1,
            )
            if result is not None:
                results.append({
                    "layer_pair": [l, l + 1],
                    "mean_angle_radians": result.mean_angle_radians,
                    "median_angle_radians": result.median_angle_radians,
                    "mean_cosine": result.mean_cosine,
                    "coverage": result.coverage,
                    "neighbor_count": neighbor_count,
                    "tangent_rank": tangent_rank,
                })
            else:
                results.append({
                    "layer_pair": [l, l + 1],
                    "skipped": True,
                    "reason": "insufficient_data",
                })
        except Exception as e:
            logger.warning(f"  Measurement B failed at pair ({l}, {l+1}): {e}")
            results.append({
                "layer_pair": [l, l + 1],
                "skipped": True,
                "reason": str(e)[:200],
            })

        del pts_l, pts_l1

    return results


# =============================================================================
# Measurement C: Tracked Neighbor Rank Change
# =============================================================================


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2). Effective rank."""
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.sum(eigenvalues)
    if total <= 0:
        return 0.0
    sum_sq = np.sum(eigenvalues**2)
    if sum_sq <= 0:
        return 0.0
    return float(total**2 / sum_sq)


def local_effective_rank(diff_matrix: np.ndarray) -> float:
    """Effective rank of [k, d] neighbor difference matrix via participation ratio."""
    if diff_matrix.shape[0] < 2:
        return 0.0
    centered = diff_matrix - diff_matrix.mean(axis=0)
    gram = centered @ centered.T  # [k, k]
    eigenvalues = np.linalg.eigvalsh(gram)
    return participation_ratio(eigenvalues)


def measurement_c_tracked_neighbors(
    stage_activations: list[np.ndarray],
) -> list[dict]:
    """Track the same neighbors across layers, measure rank change.

    OPERATOR LIMITATION: Uses Euclidean KDTree neighborhoods, while TwoNN uses
    geodesic distances via k-NN Floyd-Warshall graph. This means P5 results
    are NOT commensurable with TwoNN IDs. A null P5 does not cleanly eliminate
    local rank change as a mechanism — it only shows that Euclidean-neighborhood
    rank change is uncorrelated with geodesic-neighborhood ID.

    k = max(ceil(ln(N)), N//4): enough neighbors to resolve local dimension.
    Berry & Sauer 2016 (ceil(ln(N))) is connectivity minimum; N//4 provides
    sufficient rank resolution without becoming global.
    """
    results = []
    n_stages = len(stage_activations)
    N = stage_activations[0].shape[0]

    k = max(int(np.ceil(np.log(max(N, 2)))), N // 4)
    k = max(2, min(k, N - 1))

    logger.info(f"  Measurement C: k={k} neighbors (N={N})")

    for l in range(n_stages - 1):
        X_l = stage_activations[l]
        X_l1 = stage_activations[l + 1]

        tree = KDTree(X_l)
        _, neighbor_indices = tree.query(X_l, k=k + 1)
        neighbor_indices = neighbor_indices[:, 1:]  # Remove self

        ranks_l = np.zeros(N)
        ranks_l1 = np.zeros(N)

        for p in range(N):
            nn_idx = neighbor_indices[p]
            diff_l = X_l[nn_idx] - X_l[p]
            diff_l1 = X_l1[nn_idx] - X_l1[p]
            ranks_l[p] = local_effective_rank(diff_l)
            ranks_l1[p] = local_effective_rank(diff_l1)

        delta_ranks = ranks_l1 - ranks_l

        results.append({
            "layer_pair": [l, l + 1],
            "k_neighbors": k,
            "mean_delta_local_rank": float(np.mean(delta_ranks)),
            "std_delta_local_rank": float(np.std(delta_ranks)),
            "mean_rank_l": float(np.mean(ranks_l)),
            "mean_rank_l1": float(np.mean(ranks_l1)),
        })

    return results


# =============================================================================
# Prediction Evaluation
# =============================================================================


def _safe_spearman(x, y) -> tuple[float, float]:
    """Spearman correlation with NaN handling. Returns (r, p)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return float("nan"), 1.0
    r, p = scipy_stats.spearmanr(x, y)
    return float(r), float(p)


def compute_predictions(
    meas_a: list[dict],
    meas_b: list[dict],
    meas_c: list[dict],
    twonn_ids: list[float],
) -> dict:
    """Evaluate P1-P5 from measurements and TwoNN IDs."""

    # Compute delta_ID between consecutive stages
    delta_ids = [
        twonn_ids[i + 1] - twonn_ids[i]
        for i in range(len(twonn_ids) - 1)
    ]
    abs_delta_ids = [abs(d) for d in delta_ids]

    predictions = {}

    # P1: Spearman(d_G, |delta_ID|) > 0.3
    d_Gs = [
        m.get("grassmann_geodesic", float("nan"))
        for m in meas_a
    ]
    r_p1, p_p1 = _safe_spearman(d_Gs, abs_delta_ids)
    predictions["P1"] = {
        "description": "Spearman(grassmann_distance, |delta_ID|)",
        "spearman_r": r_p1,
        "p_value": p_p1,
        "passes": not np.isnan(r_p1) and r_p1 > 0.3,
        "n": sum(1 for d in d_Gs if not np.isnan(d)),
    }

    # P2: Spearman(novel_count_full, delta_ID) > 0.3 when delta_ID > 0
    increasing_mask = [
        i for i, d in enumerate(delta_ids)
        if d > 0 and i < len(meas_a) and "novel_count_full" in meas_a[i]
    ]
    if len(increasing_mask) >= 4:
        novels = [meas_a[i]["novel_count_full"] for i in increasing_mask]
        deltas = [delta_ids[i] for i in increasing_mask]
        r_p2, p_p2 = _safe_spearman(novels, deltas)
    else:
        r_p2, p_p2 = float("nan"), 1.0
    predictions["P2"] = {
        "description": "Spearman(novel_count_full, delta_ID) when delta_ID > 0",
        "spearman_r": r_p2,
        "p_value": p_p2,
        "passes": not np.isnan(r_p2) and r_p2 > 0.3,
        "n_increasing": len(increasing_mask),
    }

    # P3: Highway d_G < median(d_G)
    valid_ids = [(i, tid) for i, tid in enumerate(twonn_ids) if not np.isnan(tid)]
    if valid_ids:
        highway_stage = min(valid_ids, key=lambda x: x[1])[0]
        valid_dGs = [d for d in d_Gs if not np.isnan(d)]
        median_dG = float(np.median(valid_dGs)) if valid_dGs else 0.0

        # Highway d_G = d_G at the pair involving the highway stage
        # Use the pair (highway-1, highway) or (highway, highway+1)
        highway_dG = float("nan")
        for idx in [highway_stage - 1, highway_stage]:
            if 0 <= idx < len(d_Gs) and not np.isnan(d_Gs[idx]):
                highway_dG = d_Gs[idx]
                break

        predictions["P3"] = {
            "description": "Highway d_G < median(d_G)",
            "highway_stage": highway_stage,
            "highway_id": twonn_ids[highway_stage],
            "highway_dG": highway_dG,
            "median_dG": median_dG,
            "passes": not np.isnan(highway_dG) and highway_dG < median_dG,
        }
    else:
        predictions["P3"] = {"passes": False, "reason": "no_valid_ids"}

    # P4: Spearman(mean_local_angle, |delta_ID|) > 0.3
    local_angles = [
        m.get("mean_angle_radians", float("nan"))
        for m in meas_b
    ]
    r_p4, p_p4 = _safe_spearman(local_angles, abs_delta_ids)
    predictions["P4"] = {
        "description": "Spearman(mean_local_angle, |delta_ID|)",
        "spearman_r": r_p4,
        "p_value": p_p4,
        "passes": not np.isnan(r_p4) and r_p4 > 0.3,
        "n": sum(1 for a in local_angles if not np.isnan(a)),
    }

    # P5: Spearman(delta_local_rank, delta_ID) > 0.3
    # CAVEAT: Euclidean KDTree neighborhoods vs TwoNN's geodesic neighborhoods.
    # A null result does not cleanly eliminate local rank change as a mechanism.
    delta_ranks = [m["mean_delta_local_rank"] for m in meas_c]
    r_p5, p_p5 = _safe_spearman(delta_ranks, delta_ids)
    predictions["P5"] = {
        "description": "Spearman(delta_local_rank, delta_ID) [Euclidean KDTree, not commensurable with geodesic TwoNN]",
        "spearman_r": r_p5,
        "p_value": p_p5,
        "passes": not np.isnan(r_p5) and r_p5 > 0.3,
        "n": len(delta_ranks),
        "operator_caveat": "Euclidean_KDTree_not_geodesic",
    }

    return predictions


# =============================================================================
# Cross-Model Falsification
# =============================================================================


def compute_cross_model_falsification(all_results: list[dict]) -> dict:
    """Evaluate F1-F3 across models."""

    n_models = len(all_results)
    if n_models == 0:
        return {"error": "no_models", "overall_verdict": "NO_DATA"}

    per_model_p1_r = {}
    per_model_p1_sign = {}
    per_model_p5_r = {}

    for result in all_results:
        name = result["model_name"]
        preds = result["predictions"]

        p1_r = preds["P1"]["spearman_r"]
        per_model_p1_r[name] = p1_r
        if not np.isnan(p1_r):
            per_model_p1_sign[name] = 1 if p1_r > 0 else -1

        p5_r = preds["P5"]["spearman_r"]
        per_model_p5_r[name] = p5_r

    # F1: P1 > 0.3 on ALL models
    f1_all_pass = all(
        not np.isnan(r) and r > 0.3
        for r in per_model_p1_r.values()
    )

    # F2: Sign of P1 consistent across models
    signs = list(per_model_p1_sign.values())
    f2_signs_match = len(set(signs)) <= 1 if signs else False

    # F3: P5 > 0.3 on at least one model
    f3_any_pass = any(
        not np.isnan(r) and r > 0.3
        for r in per_model_p5_r.values()
    )

    # Overall verdict
    if f1_all_pass and f2_signs_match and f3_any_pass:
        verdict = "TANGENT_SUBSPACE_ROTATION_SUPPORTED"
    elif not f1_all_pass:
        verdict = "F1_FALSIFIED: global rotation does not predict ID change on all models"
    elif not f2_signs_match:
        verdict = "F2_FALSIFIED: sign inconsistency across models (mechanism underspecified)"
    elif not f3_any_pass:
        verdict = "F3_FALSIFIED: local mechanism does not match global on any model"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n_models": n_models,
        "F1_global_rotation": {
            "all_pass": f1_all_pass,
            "per_model_r": per_model_p1_r,
        },
        "F2_sign_consistency": {
            "signs_match": f2_signs_match,
            "per_model_sign": per_model_p1_sign,
        },
        "F3_local_mechanism": {
            "any_pass": f3_any_pass,
            "per_model_P5_r": per_model_p5_r,
        },
        "overall_verdict": verdict,
    }


# =============================================================================
# Single Model Run
# =============================================================================


def run_single_model(
    model_name: str,
    model_info: dict,
    probes: list[str],
    backend,
    *,
    run_b: bool = True,
) -> dict:
    """Run all measurements for one model."""
    logger.info(f"Loading model: {model_name} from {model_info['path']}")
    model, tokenizer = backend.load_model(model_info["path"])

    base = _resolve_model_base(model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0
    logger.info(f"Model loaded: {num_layers} layers, d={model_info.get('d', 0)}")

    # Collect activations
    t0 = time.time()
    stage_activations = collect_layer_activations(
        model, tokenizer, probes, num_layers,
    )
    logger.info(f"  Activation collection: {time.time() - t0:.1f}s ({len(probes)} probes)")

    # Free model memory
    del model, tokenizer, base, layers
    gc.collect()

    # TwoNN ID per layer
    t0 = time.time()
    twonn_ids = compute_twonn_per_layer(stage_activations, backend)
    logger.info(f"  TwoNN IDs: {time.time() - t0:.1f}s")
    for i, tid in enumerate(twonn_ids):
        logger.info(f"    Stage {i:2d}: ID = {tid:.2f}")

    # Measurement A: Global tangent subspace
    t0 = time.time()
    meas_a = measurement_a_global_tangent(stage_activations, twonn_ids)
    logger.info(f"  Measurement A (global tangent): {time.time() - t0:.1f}s")

    # Measurement B: Local tangent alignment
    meas_b: list[dict] = []
    if run_b:
        t0 = time.time()
        meas_b = measurement_b_local_tangent(stage_activations, backend)
        logger.info(f"  Measurement B (local tangent): {time.time() - t0:.1f}s")
    else:
        logger.info("  Measurement B: SKIPPED (--smoke)")
        meas_b = [
            {"layer_pair": [l, l + 1], "skipped": True, "reason": "smoke_mode"}
            for l in range(len(stage_activations) - 1)
        ]

    # Measurement C: Tracked neighbors
    t0 = time.time()
    meas_c = measurement_c_tracked_neighbors(stage_activations)
    logger.info(f"  Measurement C (tracked neighbors): {time.time() - t0:.1f}s")

    # Evaluate predictions
    predictions = compute_predictions(meas_a, meas_b, meas_c, twonn_ids)

    logger.info(f"  --- Predictions for {model_name} ---")
    for pname, pval in predictions.items():
        r = pval.get("spearman_r", "N/A")
        p = pval.get("p_value", "N/A")
        passes = pval.get("passes", "N/A")
        if isinstance(r, float) and not np.isnan(r):
            logger.info(f"    {pname}: r={r:+.3f}, p={p:.4f}, {'PASS' if passes else 'FAIL'}")
        else:
            logger.info(f"    {pname}: {pval.get('description', '')}, passes={passes}")

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "n_probes": len(probes),
        "twonn_ids": twonn_ids,
        "measurement_a": meas_a,
        "measurement_b": meas_b,
        "measurement_c": meas_c,
        "predictions": predictions,
    }


# =============================================================================
# Experiment Orchestrator
# =============================================================================


def run_experiment(args):
    """Run the full experiment."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B"]
        probes = PROBE_PROMPTS[:12]
        run_b = False  # Skip expensive local tangent in smoke
    elif args.models:
        model_names = args.models
        probes = PROBE_PROMPTS
        run_b = True
    else:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B", "Llama-3.2-3B"]
        probes = PROBE_PROMPTS
        run_b = True

    logger.info("=" * 70)
    logger.info(
        f"TANGENT SUBSPACE ID MECHANISM "
        f"({len(model_names)} models, {len(probes)} probes, B={'ON' if run_b else 'OFF'})"
    )
    logger.info("=" * 70)

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        if not os.path.exists(MODEL_REGISTRY[model_name]["path"]):
            logger.warning(
                f"Model path not found: {MODEL_REGISTRY[model_name]['path']}, skipping"
            )
            continue

        result = run_single_model(
            model_name, MODEL_REGISTRY[model_name], probes, backend, run_b=run_b,
        )
        all_results.append(result)
        gc.collect()

    if not all_results:
        logger.error("No models were evaluated. Check volume mount.")
        return

    # Cross-model falsification
    falsification = compute_cross_model_falsification(all_results)

    logger.info("\n" + "=" * 70)
    logger.info("CROSS-MODEL FALSIFICATION")
    logger.info("=" * 70)
    logger.info(f"  F1 (all P1 > 0.3): {falsification['F1_global_rotation']}")
    logger.info(f"  F2 (sign match):    {falsification['F2_sign_consistency']}")
    logger.info(f"  F3 (any P5 > 0.3): {falsification['F3_local_mechanism']}")
    logger.info(f"  VERDICT: {falsification['overall_verdict']}")

    # Save results
    output_dir = Path("results/tangent_subspace_id_mechanism")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    output = {
        "metadata": {
            "script": "tangent_subspace_id_mechanism.py",
            "n_models": len(all_results),
            "n_probes": len(probes),
            "smoke": args.smoke,
            "measurement_b_enabled": run_b,
        },
        "per_model": all_results,
        "cross_model_falsification": falsification,
    }

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    logger.info(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Tangent Subspace ID Mechanism Analysis"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Quick test (2 models, 12 probes)"
    )
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
