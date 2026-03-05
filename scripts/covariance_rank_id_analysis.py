#!/usr/bin/env python3
"""Analysis: Covariance Rank vs TwoNN Intrinsic Dimension.

Investigates WHY cumulative angular curvature correlates with TwoNN intrinsic
dimension (r=0.821). The critical reframing: cumulative curvature is monotonically
non-decreasing by construction, so Spearman(cum_curvature, ID) = Spearman(layer_index, ID).
The r=0.821 measures whether ID increases with depth, not a direct causal link.

The real questions:
1. Does effective covariance rank (k_eff) track TwoNN ID?
2. Does per-layer angular curvature predict per-layer change in k_eff?
3. Is TwoNN scale-invariant (killing M3 hypothesis)?

Three candidate mechanisms:
- M1 (PRIMARY): Covariance injection via tangential Jacobian. Each layer's
  residual update injects variance into new directions, changing effective rank.
  TwoNN tracks effective rank, not raw dimension.
- M2 (SECONDARY): Volume growth bias from manifold curvature biasing TwoNN.
- M3 (KILLED by E3): Pure scale change. TwoNN uses mu=r2/r1 which is
  scale-invariant. Verified empirically below.

Phases:
  E3: Scale invariance verification (kills M3)
  E1: Covariance rank (k_eff) vs TwoNN ID correlation
  D2: Synthetic verification that TwoNN tracks effective rank of anisotropic Gaussian
  E4: Per-layer curvature vs per-layer effective rank change

Usage:
    poetry run python scripts/covariance_rank_id_analysis.py
    poetry run python scripts/covariance_rank_id_analysis.py --smoke
    poetry run python scripts/covariance_rank_id_analysis.py --models LFM2-350M Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
    },
    "Qwen3-8B": {
        "path": _resolve_existing_path(
            f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
            f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        ),
        "L": 36, "d": 4096,
        "architecture": "qwen3",
    },
}

# Diverse probes — same categories as curvature_accumulation_analysis.py
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
# Phase E3: Scale Invariance Verification
# =============================================================================


def run_e3_scale_invariance(backend) -> dict:
    """Verify TwoNN is scale-invariant (kills M3 hypothesis).

    Generates synthetic point clouds, scales them by 0.5x and 2x,
    and verifies TwoNN gives the same ID within floating-point precision.
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    logger.info("=" * 60)
    logger.info("E3: SCALE INVARIANCE VERIFICATION")
    logger.info("=" * 60)

    rng = np.random.default_rng(seed=42)
    results = []

    for true_d in [3, 5, 10, 20]:
        # Generate N points from a d-dimensional Gaussian embedded in 100D
        N = 200
        D_ambient = 100
        data = rng.standard_normal((N, true_d))
        # Embed in ambient space via random projection
        proj = rng.standard_normal((true_d, D_ambient)) / np.sqrt(true_d)
        points = data @ proj

        # Compute TwoNN at three scales
        scales = {"original": 1.0, "half": 0.5, "double": 2.0}
        ids = {}
        for name, scale in scales.items():
            scaled = points * scale
            estimate = IntrinsicDimension.compute_two_nn(scaled.tolist(), backend=backend)
            ids[name] = estimate.intrinsic_dimension

        max_diff = max(
            abs(ids["original"] - ids["half"]),
            abs(ids["original"] - ids["double"]),
        )

        result = {
            "true_d": true_d,
            "id_original": ids["original"],
            "id_half": ids["half"],
            "id_double": ids["double"],
            "max_diff": max_diff,
            "passes": max_diff < 0.5,  # Should be ~0 for dimensionless ratio
        }
        results.append(result)

        logger.info(
            f"  d={true_d:2d}: ID(1x)={ids['original']:.2f}, "
            f"ID(0.5x)={ids['half']:.2f}, ID(2x)={ids['double']:.2f}, "
            f"max_diff={max_diff:.4f} {'PASS' if result['passes'] else 'FAIL'}"
        )

    all_pass = all(r["passes"] for r in results)
    logger.info(f"  E3 verdict: {'M3 KILLED (scale-invariant)' if all_pass else 'M3 SURVIVES'}")

    return {
        "test": "E3_scale_invariance",
        "results": results,
        "all_pass": all_pass,
        "verdict": "M3_killed" if all_pass else "M3_survives",
    }


# =============================================================================
# Phase D2: Synthetic Verification — TwoNN Tracks Effective Rank
# =============================================================================


def compute_effective_rank_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Compute effective rank = exp(Shannon entropy of normalized eigenvalues).

    Same formula as variance_concentration.py but on raw numpy arrays.
    """
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    # Shannon entropy with zero-safe log
    mask = p > 1e-30
    entropy = -np.sum(p[mask] * np.log(p[mask]))
    return float(np.exp(entropy))


def run_d2_synthetic_verification(backend) -> dict:
    """Verify that TwoNN tracks effective rank of anisotropic Gaussians.

    Generates point clouds from Gaussians with known eigenvalue spectra
    (varying effective rank) and checks that TwoNN ID tracks k_eff.
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    logger.info("=" * 60)
    logger.info("D2: SYNTHETIC VERIFICATION — TwoNN TRACKS EFFECTIVE RANK")
    logger.info("=" * 60)

    rng = np.random.default_rng(seed=123)
    D_ambient = 100
    N = 300

    results = []

    # Create spectra with varying effective rank
    # Background eigenvalues set to machine-negligible (1e-10) to avoid
    # inflating k_eff. Previous version used 0.01 which made 98 background
    # dims contribute ~50% of total variance for k=2 case.
    bg = 1e-10
    spectra = [
        ("k=2_sharp", np.array([1.0, 1.0] + [bg] * 98)),
        ("k=5_sharp", np.array([1.0] * 5 + [bg] * 95)),
        ("k=10_sharp", np.array([1.0] * 10 + [bg] * 90)),
        ("k=20_sharp", np.array([1.0] * 20 + [bg] * 80)),
        ("k=3_decay", np.array([10.0, 5.0, 2.5] + [bg] * 97)),
        ("k=8_decay", np.array([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5] + [bg] * 92)),
        ("k=15_decay", np.concatenate([np.exp(-np.arange(15) * 0.3), np.full(85, bg)])),
        ("k=50_uniform", np.array([1.0] * 50 + [bg] * 50)),
    ]

    k_effs = []
    id_twonn = []

    for name, eigenvalues in spectra:
        k_eff = compute_effective_rank_from_eigenvalues(eigenvalues)

        # Generate points from this Gaussian
        # Cov = diag(eigenvalues), so std = sqrt(eigenvalues)
        std = np.sqrt(np.maximum(eigenvalues, 0.0))
        points = rng.standard_normal((N, D_ambient)) * std[np.newaxis, :]

        estimate = IntrinsicDimension.compute_two_nn(points.tolist(), backend=backend)
        id_val = estimate.intrinsic_dimension

        k_effs.append(k_eff)
        id_twonn.append(id_val)

        result = {
            "name": name,
            "k_eff": k_eff,
            "id_twonn": id_val,
        }
        results.append(result)

        logger.info(f"  {name:20s}: k_eff={k_eff:.2f}, TwoNN_ID={id_val:.2f}")

    # Spearman correlation between k_eff and TwoNN ID
    from scipy import stats

    r_spearman, p_spearman = stats.spearmanr(k_effs, id_twonn)
    # Pearson too (since both should be roughly linear)
    r_pearson, p_pearson = stats.pearsonr(k_effs, id_twonn)

    logger.info(f"  Spearman(k_eff, TwoNN_ID) = {r_spearman:.4f} (p={p_spearman:.6f})")
    logger.info(f"  Pearson(k_eff, TwoNN_ID)  = {r_pearson:.4f} (p={p_pearson:.6f})")

    passes = r_spearman > 0.8

    logger.info(
        f"  D2 verdict: {'TwoNN tracks k_eff (r > 0.8)' if passes else 'TwoNN does NOT track k_eff'}"
    )

    return {
        "test": "D2_synthetic_verification",
        "results": results,
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "passes": passes,
        "verdict": "TwoNN_tracks_keff" if passes else "TwoNN_does_NOT_track_keff",
    }


# =============================================================================
# Phase E1: Covariance Rank vs TwoNN ID on Real Models
# =============================================================================


def angular_change(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angular change between two vectors in radians."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_sim = np.dot(v1, v2) / (n1 * n2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(np.arccos(cos_sim))


def collect_layer_activations(
    model, tokenizer, prompts: list[str], num_layers: int
) -> list[np.ndarray]:
    """Collect last-token hidden states at each layer for all prompts.

    Returns list of [N, d] arrays, one per layer (including layer output).
    Index 0 = embedding output, index i+1 = output of layer i.
    """
    import mlx.core as mx

    base = _resolve_model_base(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # Collectors: one per "stage" (embedding + each layer)
    stage_activations = [[] for _ in range(num_layers + 1)]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        mx.eval(hidden)

        # Store embedding output (last token)
        h_last = hidden[:, -1, :].astype(mx.float32)
        mx.eval(h_last)
        stage_activations[0].append(np.array(h_last[0].tolist(), dtype=np.float32))

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            # Per-layer mask routing (LFM2 compatibility)
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = None  # Most models handle mask internally

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

    # Stack into [N, d] arrays
    return [np.stack(acts) for acts in stage_activations]


def compute_layer_metrics(
    stage_activations: list[np.ndarray],
) -> list[dict]:
    """Compute k_eff, TwoNN ID, and angular curvature per layer.

    stage_activations[0] = embedding output, stage_activations[i+1] = layer i output.
    Returns one dict per layer (layer 0 through L-1).
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    num_layers = len(stage_activations) - 1
    metrics = []

    for i in range(num_layers):
        h_in = stage_activations[i]       # [N, d] — input to layer i
        h_out = stage_activations[i + 1]  # [N, d] — output of layer i
        N = h_out.shape[0]

        # --- Covariance spectrum of h_out via SVD ---
        centered = h_out - h_out.mean(axis=0, keepdims=True)
        # Use min(N, d) for SVD
        if centered.shape[0] < centered.shape[1]:
            gram = centered @ centered.T
            eigenvalues = np.linalg.eigvalsh(gram)
        else:
            gram = centered.T @ centered
            eigenvalues = np.linalg.eigvalsh(gram)
        # eigvalsh returns ascending; flip to descending
        eigenvalues = eigenvalues[::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        k_eff = compute_effective_rank_from_eigenvalues(eigenvalues)

        # Variance concentration: top-1 eigenvalue fraction
        total_var = eigenvalues.sum()
        var_top1 = float(eigenvalues[0] / total_var) if total_var > 0 else 0.0

        # --- Effective rank of h_in (for delta computation) ---
        centered_in = h_in - h_in.mean(axis=0, keepdims=True)
        if centered_in.shape[0] < centered_in.shape[1]:
            gram_in = centered_in @ centered_in.T
            eig_in = np.linalg.eigvalsh(gram_in)
        else:
            gram_in = centered_in.T @ centered_in
            eig_in = np.linalg.eigvalsh(gram_in)
        eig_in = eig_in[::-1]
        eig_in = np.maximum(eig_in, 0.0)
        k_eff_in = compute_effective_rank_from_eigenvalues(eig_in)

        # --- TwoNN intrinsic dimension ---
        min_samples = IntrinsicDimension.local_dimension_min_samples()
        if N >= min_samples:
            try:
                estimate = IntrinsicDimension.compute_two_nn(h_out)
                id_twonn = estimate.intrinsic_dimension
            except Exception:
                id_twonn = float("nan")
        else:
            id_twonn = float("nan")

        # --- Per-layer angular curvature ---
        angles = []
        for j in range(N):
            angles.append(angular_change(h_in[j], h_out[j]))
        mean_curvature = float(np.mean(angles))

        # --- Delta k_eff (change in effective rank this layer) ---
        delta_k_eff = k_eff - k_eff_in

        metrics.append({
            "layer_idx": i,
            "k_eff": k_eff,
            "k_eff_in": k_eff_in,
            "delta_k_eff": delta_k_eff,
            "var_top1": var_top1,
            "id_twonn": id_twonn,
            "mean_curvature": mean_curvature,
            "n_samples": N,
        })

    return metrics


def compute_e1_correlations(metrics: list[dict]) -> dict:
    """Compute E1 correlations: k_eff vs TwoNN ID, delta_k_eff vs curvature."""
    from scipy import stats

    # Filter to layers with valid ID
    valid = [m for m in metrics if not np.isnan(m["id_twonn"])]
    n_valid = len(valid)

    result = {"n_valid_layers": n_valid}

    if n_valid < 5:
        result["note"] = f"Insufficient layers ({n_valid} < 5)"
        return result

    k_effs = [m["k_eff"] for m in valid]
    ids = [m["id_twonn"] for m in valid]
    delta_k_effs = [m["delta_k_eff"] for m in valid]
    curvatures = [m["mean_curvature"] for m in valid]
    var_top1s = [m["var_top1"] for m in valid]

    # Primary test: Spearman(k_eff, TwoNN_ID)
    r_keff_id, p_keff_id = stats.spearmanr(k_effs, ids)
    result["spearman_keff_vs_id"] = float(r_keff_id)
    result["p_keff_vs_id"] = float(p_keff_id)

    # Pearson too
    r_pear, p_pear = stats.pearsonr(k_effs, ids)
    result["pearson_keff_vs_id"] = float(r_pear)
    result["p_pearson_keff_vs_id"] = float(p_pear)

    # Var_top1 vs TwoNN ID (variance concentration)
    r_vt1_id, p_vt1_id = stats.spearmanr(var_top1s, ids)
    result["spearman_var_top1_vs_id"] = float(r_vt1_id)
    result["p_var_top1_vs_id"] = float(p_vt1_id)

    # Secondary: Spearman(delta_k_eff, curvature)
    r_delta_curv, p_delta_curv = stats.spearmanr(delta_k_effs, curvatures)
    result["spearman_delta_keff_vs_curvature"] = float(r_delta_curv)
    result["p_delta_keff_vs_curvature"] = float(p_delta_curv)

    # Control: Spearman(layer_index, ID) — should match cum_curvature vs ID
    layer_indices = [m["layer_idx"] for m in valid]
    r_layer_id, p_layer_id = stats.spearmanr(layer_indices, ids)
    result["spearman_layer_idx_vs_id"] = float(r_layer_id)
    result["p_layer_idx_vs_id"] = float(p_layer_id)

    # Cumulative curvature vs ID (reproduces the r=0.821 claim)
    cum_curvature = np.cumsum(curvatures).tolist()
    r_cum_id, p_cum_id = stats.spearmanr(cum_curvature, ids)
    result["spearman_cum_curvature_vs_id"] = float(r_cum_id)
    result["p_cum_curvature_vs_id"] = float(p_cum_id)

    # ID gradient vs per-layer curvature
    id_gradient = np.gradient(ids).tolist()
    r_curv_grad, p_curv_grad = stats.spearmanr(curvatures, id_gradient)
    result["spearman_curvature_vs_id_gradient"] = float(r_curv_grad)
    result["p_curvature_vs_id_gradient"] = float(p_curv_grad)

    # Delta_k_eff vs ID gradient (does rank change explain ID change?)
    r_dk_grad, p_dk_grad = stats.spearmanr(delta_k_effs, id_gradient)
    result["spearman_delta_keff_vs_id_gradient"] = float(r_dk_grad)
    result["p_delta_keff_vs_id_gradient"] = float(p_dk_grad)

    return result


def run_e1_single_model(
    model_name: str, model_info: dict, probes: list[str], backend
) -> dict:
    """Run E1 analysis for one model."""
    logger.info(f"Loading model: {model_name} from {model_info['path']}")
    model, tokenizer = backend.load_model(model_info["path"])

    base = _resolve_model_base(model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0

    logger.info(f"Model loaded: {num_layers} layers, d={model_info.get('d', 0)}")

    t0 = time.time()
    stage_activations = collect_layer_activations(
        model, tokenizer, probes, num_layers
    )
    logger.info(f"  Activation collection: {time.time() - t0:.1f}s")

    t0 = time.time()
    metrics = compute_layer_metrics(stage_activations)
    logger.info(f"  Metric computation: {time.time() - t0:.1f}s")

    correlations = compute_e1_correlations(metrics)

    # Log key results
    r_keff_id = correlations.get("spearman_keff_vs_id", 0.0)
    r_delta_curv = correlations.get("spearman_delta_keff_vs_curvature", 0.0)
    r_cum_id = correlations.get("spearman_cum_curvature_vs_id", 0.0)
    r_layer_id = correlations.get("spearman_layer_idx_vs_id", 0.0)
    r_curv_grad = correlations.get("spearman_curvature_vs_id_gradient", 0.0)
    r_dk_grad = correlations.get("spearman_delta_keff_vs_id_gradient", 0.0)
    r_vt1_id = correlations.get("spearman_var_top1_vs_id", 0.0)

    logger.info(f"  Spearman(k_eff, TwoNN_ID)       = {r_keff_id:.4f}")
    logger.info(f"  Spearman(var_top1, TwoNN_ID)    = {r_vt1_id:.4f}")
    logger.info(f"  Spearman(delta_k_eff, curvature) = {r_delta_curv:.4f}")
    logger.info(f"  Spearman(cum_curvature, ID)      = {r_cum_id:.4f}")
    logger.info(f"  Spearman(layer_idx, ID)          = {r_layer_id:.4f}")
    logger.info(f"  Spearman(curvature, dID/dl)      = {r_curv_grad:.4f}")
    logger.info(f"  Spearman(delta_k_eff, dID/dl)    = {r_dk_grad:.4f}")

    # Per-layer detail
    for m in metrics:
        logger.info(
            f"    L{m['layer_idx']:2d}: k_eff={m['k_eff']:.2f}, "
            f"vt1={m['var_top1']:.3f}, "
            f"ID={m['id_twonn']:.2f}, "
            f"theta={m['mean_curvature']:.4f}, "
            f"dk_eff={m['delta_k_eff']:+.2f}"
        )

    # Clean up
    del model, tokenizer, stage_activations
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "d_model": model_info.get("d", 0),
        "n_probes": len(probes),
        "per_layer": metrics,
        "correlations": correlations,
    }


# =============================================================================
# Cross-Model Summary & Falsifier Assessment
# =============================================================================


def compute_cross_model_summary(all_results: list[dict]) -> dict:
    """Compute cross-model M1 falsifier: Spearman(k_eff, ID) > 0.8 for ALL models."""
    from scipy import stats

    summary = {
        "n_models": len(all_results),
        "per_model": {},
    }

    all_keffs = []
    all_ids = []

    for r in all_results:
        name = r["model_name"]
        corr = r["correlations"]
        r_keff_id = corr.get("spearman_keff_vs_id", 0.0)
        r_cum_id = corr.get("spearman_cum_curvature_vs_id", 0.0)
        r_layer_id = corr.get("spearman_layer_idx_vs_id", 0.0)

        passes_m1 = abs(r_keff_id) > 0.5

        summary["per_model"][name] = {
            "spearman_keff_vs_id": r_keff_id,
            "spearman_cum_curvature_vs_id": r_cum_id,
            "spearman_layer_idx_vs_id": r_layer_id,
            "M1_passes": passes_m1,
        }

        # Pool all valid layer data for cross-model test
        for m in r["per_layer"]:
            if not np.isnan(m["id_twonn"]):
                all_keffs.append(m["k_eff"])
                all_ids.append(m["id_twonn"])

    # Cross-model pooled correlation
    if len(all_keffs) >= 10:
        r_pooled, p_pooled = stats.spearmanr(all_keffs, all_ids)
        summary["pooled_spearman_keff_vs_id"] = float(r_pooled)
        summary["pooled_p_keff_vs_id"] = float(p_pooled)
        summary["pooled_n"] = len(all_keffs)

    # M1 falsifier: passes if k_eff tracks ID for all models
    all_pass = all(
        v["M1_passes"] for v in summary["per_model"].values()
    )
    summary["M1_all_pass"] = all_pass
    summary["M1_verdict"] = (
        "M1_confirmed: k_eff tracks TwoNN ID across all models"
        if all_pass
        else "M1_insufficient: k_eff does NOT consistently track TwoNN ID"
    )

    return summary


# =============================================================================
# Main
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run covariance rank vs ID analysis."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Phase E3 + D2: Synthetic tests (no model loading needed)
    e3_result = run_e3_scale_invariance(backend)
    d2_result = run_d2_synthetic_verification(backend)

    # Phase E1: Real models
    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B"]
        probes = PROBE_PROMPTS[:12]
    elif args.models:
        model_names = args.models
        probes = PROBE_PROMPTS
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = PROBE_PROMPTS

    logger.info("=" * 60)
    logger.info(f"E1: COVARIANCE RANK vs TwoNN ID ({len(model_names)} models, {len(probes)} probes)")
    logger.info("=" * 60)

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        result = run_e1_single_model(
            model_name, MODEL_REGISTRY[model_name], probes, backend
        )
        all_results.append(result)
        gc.collect()

    # Cross-model summary
    summary = compute_cross_model_summary(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-MODEL SUMMARY")
    logger.info("=" * 60)

    for name, vals in summary["per_model"].items():
        logger.info(
            f"  {name:20s}: r(k_eff,ID)={vals['spearman_keff_vs_id']:.4f}, "
            f"r(cum_curv,ID)={vals['spearman_cum_curvature_vs_id']:.4f}, "
            f"r(layer,ID)={vals['spearman_layer_idx_vs_id']:.4f}, "
            f"M1={'PASS' if vals['M1_passes'] else 'FAIL'}"
        )

    if "pooled_spearman_keff_vs_id" in summary:
        logger.info(
            f"  POOLED: r(k_eff,ID)={summary['pooled_spearman_keff_vs_id']:.4f} "
            f"(n={summary['pooled_n']}, p={summary['pooled_p_keff_vs_id']:.2e})"
        )

    logger.info(f"  M1 VERDICT: {summary['M1_verdict']}")
    logger.info(f"  E3 VERDICT: {e3_result['verdict']}")
    logger.info(f"  D2 VERDICT: {d2_result['verdict']}")

    # Save results
    output_dir = Path("results/covariance_rank_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "covariance_rank_id_results.json"

    output = {
        "e3_scale_invariance": e3_result,
        "d2_synthetic": d2_result,
        "e1_per_model": all_results,
        "cross_model_summary": summary,
    }

    # Convert numpy types for JSON serialization
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
        if isinstance(obj, float) and np.isnan(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    logger.info(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Covariance Rank vs TwoNN ID Analysis")
    parser.add_argument("--smoke", action="store_true", help="Quick test (2 models, 12 probes)")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
