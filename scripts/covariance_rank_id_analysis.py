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

Phase 1 (completed):
  E3: Scale invariance verification (kills M3) — PASS
  D2: Synthetic verification that TwoNN tracks k_eff (Gaussian) — PASS (r=0.976)
  E1: Global k_eff vs TwoNN ID on real models — FAIL (r≈0 for 3/4 models)
  E4: Per-layer curvature vs per-layer effective rank change

Phase 2 (this run):
  E5: LOCAL effective rank (kNN-patch covariance) vs TwoNN ID
  E6: Expanded model registry (8+ models for statistical power)
  E2: Curvature bias calibration (TwoNN on d-spheres with known curvature)
  E7: Per-sample ID decomposition (local dimension variance within layers)

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
import os
import time
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
    # --- Small models (fast iteration) ---
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
    # --- Medium models ---
    "LFM2.5-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2.5-1.2B-Instruct-bf16",
        "L": 16, "d": 2048,
        "architecture": "lfm2",
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24, "d": 2048,
        "architecture": "qwen3.5",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
    },
    "SmolLM3-3B": {
        "path": f"{MODELS_BASE}/mlx-community/SmolLM3-3B-bf16",
        "L": 36, "d": 2048,
        "architecture": "smollm3",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
    },
    "Qwen3.5-4B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-bf16",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
    },
    # --- Large models (final validation only) ---
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



def run_e1_single_model(
    model_name: str, model_info: dict, probes: list[str], backend,
    *, run_e7: bool = True,
) -> dict:
    """Run E1 + E5 + E7 analysis for one model."""
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
    metrics = compute_layer_metrics_with_local(stage_activations)
    logger.info(f"  Metric computation (with local k_eff): {time.time() - t0:.1f}s")

    # E7: Per-sample ID decomposition
    if run_e7:
        t0 = time.time()
        for i, m in enumerate(metrics):
            h_out = stage_activations[i + 1]
            e7_stats = compute_local_id_stats(h_out, backend)
            m.update({
                "e7_mean_local_id": e7_stats["mean_local_id"],
                "e7_std_local_id": e7_stats["std_local_id"],
                "e7_cv_local_id": e7_stats["cv_local_id"],
                "e7_modal_local_id": e7_stats["modal_local_id"],
                "e7_n_deficient": e7_stats["n_deficient"],
            })
        logger.info(f"  E7 local ID decomposition: {time.time() - t0:.1f}s")

    correlations = compute_e1_correlations_extended(metrics)

    # Log key results
    r_keff_id = correlations.get("spearman_keff_vs_id", 0.0)
    r_keff_local_id = correlations.get("spearman_keff_local_vs_id", float("nan"))
    r_delta_curv = correlations.get("spearman_delta_keff_vs_curvature", 0.0)
    r_cum_id = correlations.get("spearman_cum_curvature_vs_id", 0.0)
    r_layer_id = correlations.get("spearman_layer_idx_vs_id", 0.0)
    r_curv_grad = correlations.get("spearman_curvature_vs_id_gradient", 0.0)
    r_dk_grad = correlations.get("spearman_delta_keff_vs_id_gradient", 0.0)
    r_vt1_id = correlations.get("spearman_var_top1_vs_id", 0.0)

    logger.info(f"  Spearman(k_eff_GLOBAL, TwoNN_ID) = {r_keff_id:.4f}")
    logger.info(f"  Spearman(k_eff_LOCAL,  TwoNN_ID) = {r_keff_local_id:.4f}")
    logger.info(f"  Spearman(var_top1, TwoNN_ID)     = {r_vt1_id:.4f}")
    logger.info(f"  Spearman(delta_k_eff, curvature) = {r_delta_curv:.4f}")
    logger.info(f"  Spearman(cum_curvature, ID)      = {r_cum_id:.4f}")
    logger.info(f"  Spearman(layer_idx, ID)          = {r_layer_id:.4f}")
    logger.info(f"  Spearman(curvature, dID/dl)      = {r_curv_grad:.4f}")
    logger.info(f"  Spearman(delta_k_eff, dID/dl)    = {r_dk_grad:.4f}")

    # Per-layer detail
    for m in metrics:
        kl = m.get("k_eff_local", float("nan"))
        e7_cv = m.get("e7_cv_local_id", float("nan"))
        logger.info(
            f"    L{m['layer_idx']:2d}: k_eff={m['k_eff']:.2f}, "
            f"k_eff_local={kl:.2f}, "
            f"vt1={m['var_top1']:.3f}, "
            f"ID={m['id_twonn']:.2f}, "
            f"theta={m['mean_curvature']:.4f}, "
            f"dk_eff={m['delta_k_eff']:+.2f}, "
            f"cv_localID={e7_cv:.3f}"
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
# Phase E5: Local Effective Rank vs TwoNN ID
# =============================================================================


def compute_local_keff(points: np.ndarray, k: int) -> tuple[float, int]:
    """Compute mean LOCAL effective rank over kNN patches.

    For each point, finds its k nearest neighbors, computes the
    covariance eigenvalues of that local patch, and returns the
    mean effective rank across all points.

    IMPORTANT: After centering, a patch of k points has rank at most k-1.
    The returned rank_cap = k-1 must be checked against expected TwoNN IDs.
    If rank_cap < max(TwoNN_ID), the measurement is capped and unreliable.

    Returns:
        (mean_local_keff, rank_cap) where rank_cap = k - 1.
    """
    from scipy.spatial import KDTree

    N, D = points.shape
    k_use = min(k, N - 1)
    if k_use < 2:
        return float("nan"), 0

    tree = KDTree(points)
    # Query k+1 because the point itself is included
    _, indices = tree.query(points, k=k_use + 1)

    local_ranks = []
    for i in range(N):
        # Neighbors of point i (exclude self — first entry)
        nbr_idx = indices[i, 1:]
        patch = points[nbr_idx]  # [k_use, D]

        # Local covariance eigenvalues
        centered = patch - patch.mean(axis=0, keepdims=True)
        if k_use < D:
            gram = centered @ centered.T
            eigs = np.linalg.eigvalsh(gram)
        else:
            gram = centered.T @ centered
            eigs = np.linalg.eigvalsh(gram)
        eigs = np.maximum(eigs[::-1], 0.0)
        local_ranks.append(compute_effective_rank_from_eigenvalues(eigs))

    return float(np.mean(local_ranks)), k_use - 1


def compute_layer_metrics_with_local(
    stage_activations: list[np.ndarray],
    k_local: int | None = None,
) -> list[dict]:
    """Compute all metrics per layer including LOCAL k_eff (E5).

    Extends compute_layer_metrics with k_eff_local per layer.
    If k_local is None, uses the same k that TwoNN would use
    (derived from Berry & Sauer 2016 connectivity criterion).
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    num_layers = len(stage_activations) - 1
    metrics = []

    for i in range(num_layers):
        h_in = stage_activations[i]
        h_out = stage_activations[i + 1]
        N = h_out.shape[0]

        # --- Global covariance spectrum of h_out ---
        centered = h_out - h_out.mean(axis=0, keepdims=True)
        if centered.shape[0] < centered.shape[1]:
            gram = centered @ centered.T
            eigenvalues = np.linalg.eigvalsh(gram)
        else:
            gram = centered.T @ centered
            eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
        k_eff = compute_effective_rank_from_eigenvalues(eigenvalues)

        # Variance concentration: top-1 eigenvalue fraction
        total_var = eigenvalues.sum()
        var_top1 = float(eigenvalues[0] / total_var) if total_var > 0 else 0.0

        # --- Global k_eff of h_in ---
        centered_in = h_in - h_in.mean(axis=0, keepdims=True)
        if centered_in.shape[0] < centered_in.shape[1]:
            gram_in = centered_in @ centered_in.T
            eig_in = np.linalg.eigvalsh(gram_in)
        else:
            gram_in = centered_in.T @ centered_in
            eig_in = np.linalg.eigvalsh(gram_in)
        eig_in = np.maximum(eig_in[::-1], 0.0)
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

        # --- LOCAL effective rank (E5) ---
        # Match TwoNN's k rule: max(k_connectivity, ceil(ln(N))) per
        # Berry & Sauer 2016 (see intrinsic_dimension.py:530-537).
        # Floor at N//2 so rank cap (k-1) exceeds expected TwoNN IDs.
        # A patch of k points has rank at most k-1 after centering.
        k_twonn = max(2, int(np.ceil(np.log(max(N, 2)))))
        k_for_local = k_local if k_local is not None else max(k_twonn, N // 2)
        if N > k_for_local + 1:
            k_eff_local, rank_cap = compute_local_keff(h_out, k=k_for_local)
        else:
            k_eff_local = float("nan")
            rank_cap = 0

        # --- Per-layer angular curvature ---
        angles = []
        for j in range(N):
            angles.append(angular_change(h_in[j], h_out[j]))
        mean_curvature = float(np.mean(angles))

        # --- Delta k_eff ---
        delta_k_eff = k_eff - k_eff_in

        metrics.append({
            "layer_idx": i,
            "k_eff": k_eff,
            "k_eff_in": k_eff_in,
            "k_eff_local": k_eff_local,
            "k_eff_local_rank_cap": rank_cap,
            "delta_k_eff": delta_k_eff,
            "var_top1": var_top1,
            "id_twonn": id_twonn,
            "mean_curvature": mean_curvature,
            "n_samples": N,
            "k_local_used": k_for_local,
        })

    return metrics


def compute_e1_correlations_extended(metrics: list[dict]) -> dict:
    """Compute E1 + E5 correlations: global k_eff, local k_eff, delta_k_eff vs TwoNN ID."""
    from scipy import stats

    valid = [m for m in metrics if not np.isnan(m["id_twonn"])]
    n_valid = len(valid)
    result = {"n_valid_layers": n_valid}

    if n_valid < 5:
        result["note"] = f"Insufficient layers ({n_valid} < 5)"
        return result

    k_effs = [m["k_eff"] for m in valid]
    k_effs_local = [m["k_eff_local"] for m in valid]
    ids = [m["id_twonn"] for m in valid]
    delta_k_effs = [m["delta_k_eff"] for m in valid]
    curvatures = [m["mean_curvature"] for m in valid]
    var_top1s = [m["var_top1"] for m in valid]
    layer_indices = [m["layer_idx"] for m in valid]

    # --- E1: Global k_eff vs TwoNN ID ---
    r_keff_id, p_keff_id = stats.spearmanr(k_effs, ids)
    result["spearman_keff_vs_id"] = float(r_keff_id)
    result["p_keff_vs_id"] = float(p_keff_id)

    r_pear, p_pear = stats.pearsonr(k_effs, ids)
    result["pearson_keff_vs_id"] = float(r_pear)
    result["p_pearson_keff_vs_id"] = float(p_pear)

    # --- E5: LOCAL k_eff vs TwoNN ID ---
    valid_local = [i for i, k in enumerate(k_effs_local) if not np.isnan(k)]
    if len(valid_local) >= 5:
        kl = [k_effs_local[i] for i in valid_local]
        id_l = [ids[i] for i in valid_local]
        r_local, p_local = stats.spearmanr(kl, id_l)
        result["spearman_keff_local_vs_id"] = float(r_local)
        result["p_keff_local_vs_id"] = float(p_local)
        r_local_pear, p_local_pear = stats.pearsonr(kl, id_l)
        result["pearson_keff_local_vs_id"] = float(r_local_pear)
        result["p_pearson_keff_local_vs_id"] = float(p_local_pear)

    # Var_top1 vs TwoNN ID
    r_vt1_id, p_vt1_id = stats.spearmanr(var_top1s, ids)
    result["spearman_var_top1_vs_id"] = float(r_vt1_id)
    result["p_var_top1_vs_id"] = float(p_vt1_id)

    # delta_k_eff vs curvature
    r_delta_curv, p_delta_curv = stats.spearmanr(delta_k_effs, curvatures)
    result["spearman_delta_keff_vs_curvature"] = float(r_delta_curv)
    result["p_delta_keff_vs_curvature"] = float(p_delta_curv)

    # Control: layer index vs ID
    r_layer_id, p_layer_id = stats.spearmanr(layer_indices, ids)
    result["spearman_layer_idx_vs_id"] = float(r_layer_id)
    result["p_layer_idx_vs_id"] = float(p_layer_id)

    # Cumulative curvature vs ID
    cum_curvature = np.cumsum(curvatures).tolist()
    r_cum_id, p_cum_id = stats.spearmanr(cum_curvature, ids)
    result["spearman_cum_curvature_vs_id"] = float(r_cum_id)
    result["p_cum_curvature_vs_id"] = float(p_cum_id)

    # ID gradient correlations
    id_gradient = np.gradient(ids).tolist()
    r_curv_grad, p_curv_grad = stats.spearmanr(curvatures, id_gradient)
    result["spearman_curvature_vs_id_gradient"] = float(r_curv_grad)
    result["p_curvature_vs_id_gradient"] = float(p_curv_grad)

    r_dk_grad, p_dk_grad = stats.spearmanr(delta_k_effs, id_gradient)
    result["spearman_delta_keff_vs_id_gradient"] = float(r_dk_grad)
    result["p_delta_keff_vs_id_gradient"] = float(p_dk_grad)

    return result


# =============================================================================
# Phase E2: Curvature Bias Calibration (TwoNN on d-spheres)
# =============================================================================


def run_e2_curvature_bias(backend) -> dict:
    """Measure TwoNN bias on d-spheres with known curvature.

    For S^d(R), sectional curvature kappa = 1/R^2
    (scalar curvature = d(d-1)/R^2).
    If TwoNN gives d_hat on S^d (true dim = d), then
    bias = d_hat - d measures curvature's effect on TwoNN.

    If bias is large at observed curvature magnitudes, M2 explains
    the discrepancy between k_eff and TwoNN ID on real models.
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    logger.info("=" * 60)
    logger.info("E2: CURVATURE BIAS CALIBRATION (TwoNN on d-spheres)")
    logger.info("=" * 60)

    rng = np.random.default_rng(seed=314)
    results = []

    # Test spheres: S^d embedded in (d+1)-space, varying radius
    for true_d in [3, 5, 10, 20]:
        for radius in [0.1, 1.0, 10.0, 100.0]:
            N = 300
            # Sample uniformly on S^d: normal in (d+1)-space, then normalize
            points = rng.standard_normal((N, true_d + 1))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms * radius

            estimate = IntrinsicDimension.compute_two_nn(points.tolist(), backend=backend)
            id_hat = estimate.intrinsic_dimension
            bias = id_hat - true_d
            kappa = 1.0 / (radius * radius)  # sectional curvature of S^d(R)

            result = {
                "true_d": true_d,
                "radius": radius,
                "kappa": kappa,
                "id_hat": id_hat,
                "bias": bias,
                "relative_bias": bias / true_d if true_d > 0 else 0.0,
            }
            results.append(result)

            logger.info(
                f"  S^{true_d} R={radius:6.1f} (κ=1/R²={kappa:.4f}): "
                f"ID={id_hat:.2f}, bias={bias:+.2f} ({bias/true_d*100:+.1f}%)"
            )

    # Aggregate: does bias grow with curvature?
    from scipy import stats

    kappas = [r["kappa"] for r in results]
    biases = [r["relative_bias"] for r in results]
    r_spearman, p_spearman = stats.spearmanr(kappas, biases)

    logger.info(f"  Spearman(κ, relative_bias) = {r_spearman:.4f} (p={p_spearman:.6f})")

    # M2 verdict: if high-curvature spheres show significant bias
    high_kappa_biases = [r["relative_bias"] for r in results if r["kappa"] >= 1.0]
    mean_high_kappa_bias = float(np.mean(high_kappa_biases)) if high_kappa_biases else 0.0

    # Bias > 20% at high curvature would make M2 viable
    m2_viable = abs(mean_high_kappa_bias) > 0.2

    logger.info(
        f"  Mean bias at κ≥1: {mean_high_kappa_bias:.3f} "
        f"({'M2 VIABLE' if m2_viable else 'M2 INSUFFICIENT'})"
    )

    return {
        "test": "E2_curvature_bias",
        "results": results,
        "spearman_kappa_vs_bias": float(r_spearman),
        "p_kappa_vs_bias": float(p_spearman),
        "mean_high_kappa_bias": mean_high_kappa_bias,
        "m2_viable": m2_viable,
        "verdict": "M2_viable" if m2_viable else "M2_insufficient",
    }


# =============================================================================
# Phase E7: Per-Sample ID Decomposition
# =============================================================================


def compute_local_id_stats(points: np.ndarray, backend) -> dict:
    """Compute per-point local ID and return variance statistics.

    Uses the existing IntrinsicDimension.local_dimension_map() to get
    per-point intrinsic dimension estimates. Reports mean, std, and
    coefficient of variation of local ID within the point cloud.

    High variance means the layer's ID trajectory is an average over
    heterogeneous local structures — the single TwoNN number hides
    important structure.
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    estimator = IntrinsicDimension(backend)
    try:
        local_map = estimator.local_dimension_map(points.tolist())
    except Exception as e:
        logger.debug(f"  local_dimension_map failed: {e}")
        return {
            "mean_local_id": float("nan"),
            "std_local_id": float("nan"),
            "cv_local_id": float("nan"),
            "modal_local_id": float("nan"),
            "n_deficient": 0,
        }

    mean_id = local_map.mean_dimension
    std_id = local_map.std_dimension
    modal_id = local_map.modal_dimension
    cv = std_id / mean_id if mean_id > 0 else float("nan")

    return {
        "mean_local_id": mean_id,
        "std_local_id": std_id,
        "cv_local_id": cv,
        "modal_local_id": modal_id,
        "n_deficient": len(local_map.deficient_indices),
    }


# =============================================================================
# Cross-Model Summary & Falsifier Assessment
# =============================================================================


def compute_cross_model_summary(all_results: list[dict]) -> dict:
    """Compute cross-model summary: global k_eff vs local k_eff vs TwoNN ID."""
    from scipy import stats

    summary = {
        "n_models": len(all_results),
        "per_model": {},
    }

    all_keffs_global = []
    all_keffs_local = []
    all_ids_global = []
    all_ids_local = []

    for r in all_results:
        name = r["model_name"]
        corr = r["correlations"]
        r_keff_id = corr.get("spearman_keff_vs_id", 0.0)
        r_keff_local_id = corr.get("spearman_keff_local_vs_id", float("nan"))
        r_cum_id = corr.get("spearman_cum_curvature_vs_id", 0.0)
        r_layer_id = corr.get("spearman_layer_idx_vs_id", 0.0)

        passes_m1_global = abs(r_keff_id) > 0.5
        passes_m1_local = (
            abs(r_keff_local_id) > 0.5
            if not np.isnan(r_keff_local_id)
            else False
        )

        # E7: mean CV of local ID across layers
        e7_cvs = [
            m.get("e7_cv_local_id", float("nan"))
            for m in r["per_layer"]
            if not np.isnan(m.get("e7_cv_local_id", float("nan")))
        ]
        mean_e7_cv = float(np.mean(e7_cvs)) if e7_cvs else float("nan")

        summary["per_model"][name] = {
            "spearman_keff_global_vs_id": r_keff_id,
            "spearman_keff_local_vs_id": r_keff_local_id,
            "spearman_cum_curvature_vs_id": r_cum_id,
            "spearman_layer_idx_vs_id": r_layer_id,
            "M1_global_passes": passes_m1_global,
            "M1_local_passes": passes_m1_local,
            "e7_mean_cv_local_id": mean_e7_cv,
        }

        # Pool all valid layer data for cross-model test
        for m in r["per_layer"]:
            if not np.isnan(m["id_twonn"]):
                all_keffs_global.append(m["k_eff"])
                all_ids_global.append(m["id_twonn"])
                if not np.isnan(m.get("k_eff_local", float("nan"))):
                    all_keffs_local.append(m["k_eff_local"])
                    all_ids_local.append(m["id_twonn"])

    # Cross-model pooled correlations
    if len(all_keffs_global) >= 10:
        r_pooled, p_pooled = stats.spearmanr(all_keffs_global, all_ids_global)
        summary["pooled_spearman_keff_global_vs_id"] = float(r_pooled)
        summary["pooled_p_keff_global_vs_id"] = float(p_pooled)
        summary["pooled_n_global"] = len(all_keffs_global)

    if len(all_keffs_local) >= 10:
        r_pooled_l, p_pooled_l = stats.spearmanr(all_keffs_local, all_ids_local)
        summary["pooled_spearman_keff_local_vs_id"] = float(r_pooled_l)
        summary["pooled_p_keff_local_vs_id"] = float(p_pooled_l)
        summary["pooled_n_local"] = len(all_keffs_local)

    # M1 falsifier: global passes
    n_models = len(summary["per_model"])
    all_pass_global = n_models > 0 and all(
        v["M1_global_passes"] for v in summary["per_model"].values()
    )
    all_pass_local = n_models > 0 and all(
        v["M1_local_passes"] for v in summary["per_model"].values()
    )
    summary["M1_global_all_pass"] = all_pass_global
    summary["M1_local_all_pass"] = all_pass_local

    if n_models == 0:
        summary["M1_verdict"] = "M1_no_data: no models were evaluated"
    elif all_pass_local:
        summary["M1_verdict"] = "M1_LOCAL_confirmed: local k_eff tracks TwoNN ID"
    elif all_pass_global:
        summary["M1_verdict"] = "M1_GLOBAL_confirmed: global k_eff tracks TwoNN ID"
    else:
        n_local_pass = sum(1 for v in summary["per_model"].values() if v["M1_local_passes"])
        n_global_pass = sum(1 for v in summary["per_model"].values() if v["M1_global_passes"])
        summary["M1_verdict"] = (
            f"M1_insufficient: global passes {n_global_pass}/{len(all_results)}, "
            f"local passes {n_local_pass}/{len(all_results)}"
        )

    return summary


# =============================================================================
# Main
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run covariance rank vs ID analysis (Phase 1 + Phase 2)."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Phase E3 + D2: Synthetic tests (no model loading needed)
    e3_result = run_e3_scale_invariance(backend)
    d2_result = run_d2_synthetic_verification(backend)

    # Phase E2: Curvature bias calibration (synthetic d-spheres)
    e2_result = run_e2_curvature_bias(backend)

    # Phase E1 + E5 + E7: Real models
    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3.5-0.8B"]
        probes = PROBE_PROMPTS[:12]
    elif args.models:
        model_names = args.models
        probes = PROBE_PROMPTS
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = PROBE_PROMPTS

    run_e7 = not args.smoke  # Skip E7 in smoke mode (slow)

    logger.info("=" * 60)
    logger.info(
        f"E1+E5+E7: COVARIANCE RANK vs TwoNN ID "
        f"({len(model_names)} models, {len(probes)} probes, E7={'ON' if run_e7 else 'OFF'})"
    )
    logger.info("=" * 60)

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        if not os.path.exists(MODEL_REGISTRY[model_name]["path"]):
            logger.warning(f"Model path not found: {MODEL_REGISTRY[model_name]['path']}, skipping")
            continue
        result = run_e1_single_model(
            model_name, MODEL_REGISTRY[model_name], probes, backend,
            run_e7=run_e7,
        )
        all_results.append(result)
        gc.collect()

    # Cross-model summary
    summary = compute_cross_model_summary(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-MODEL SUMMARY (Global vs Local k_eff)")
    logger.info("=" * 60)

    for name, vals in summary["per_model"].items():
        r_global = vals["spearman_keff_global_vs_id"]
        r_local = vals.get("spearman_keff_local_vs_id", float("nan"))
        r_layer = vals["spearman_layer_idx_vs_id"]
        e7_cv = vals.get("e7_mean_cv_local_id", float("nan"))
        logger.info(
            f"  {name:20s}: r(global,ID)={r_global:+.4f}, "
            f"r(local,ID)={r_local:+.4f}, "
            f"r(layer,ID)={r_layer:+.4f}, "
            f"cv_localID={e7_cv:.3f}, "
            f"G={'P' if vals['M1_global_passes'] else 'F'} "
            f"L={'P' if vals['M1_local_passes'] else 'F'}"
        )

    if "pooled_spearman_keff_global_vs_id" in summary:
        logger.info(
            f"  POOLED GLOBAL: r={summary['pooled_spearman_keff_global_vs_id']:.4f} "
            f"(n={summary['pooled_n_global']}, p={summary['pooled_p_keff_global_vs_id']:.2e})"
        )
    if "pooled_spearman_keff_local_vs_id" in summary:
        logger.info(
            f"  POOLED LOCAL:  r={summary['pooled_spearman_keff_local_vs_id']:.4f} "
            f"(n={summary['pooled_n_local']}, p={summary['pooled_p_keff_local_vs_id']:.2e})"
        )

    logger.info(f"  M1 VERDICT: {summary['M1_verdict']}")
    logger.info(f"  E3 VERDICT: {e3_result['verdict']}")
    logger.info(f"  D2 VERDICT: {d2_result['verdict']}")
    logger.info(f"  E2 VERDICT: {e2_result['verdict']}")

    # Save results
    output_dir = Path("results/covariance_rank_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "covariance_rank_id_phase2_results.json"

    output = {
        "e3_scale_invariance": e3_result,
        "d2_synthetic": d2_result,
        "e2_curvature_bias": e2_result,
        "e1_e5_e7_per_model": all_results,
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
