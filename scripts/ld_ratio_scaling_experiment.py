#!/usr/bin/env python3
"""Experiment: L/d Depth-Width Ratio Hypothesis.

Tests whether the depth-to-width ratio L/d (not L alone) determines the shape
of the intrinsic dimension trajectory through transformer layers.

Hypothesis:
    H1: Models with similar L/d have more similar normalized ID trajectories
        than models with similar L.

Predictions:
    1. LFM2-1.2B (L/d=0.008) and Qwen3-8B (L/d=0.009) have SMALLER Procrustes
       distance than LFM2-1.2B and LFM2-350M (L/d=0.016) — despite sharing L=16.
    2. Qwen2.5-3B (L/d=0.018) and Qwen3-8B (L/d=0.009) DIFFER despite same L=36.
    3. Models with |log(L/d₁) - log(L/d₂)| < 0.3 have expansion_ratio within 20%.

Falsification criteria:
    FAIL if LFM2-1.2B↔Qwen3-8B distance > LFM2-1.2B↔LFM2-350M distance
    FAIL if Qwen2.5-3B↔Qwen3-8B MORE similar than Qwen3-8B↔Llama-3.2-3B
    FAIL if partial correlation(similarity ~ L/d_distance) p > 0.1

References:
    Dey et al. (NeurIPS 2025, arXiv:2505.01618): CompleteP, L/d determines
        covariance stability in shaped transformer SDE
    Joshi et al. (NeurIPS 2025, arXiv:2511.20315): 28-model ID trajectory

Usage:
    poetry run python scripts/ld_ratio_scaling_experiment.py
    poetry run python scripts/ld_ratio_scaling_experiment.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry (same as Exp 1, with L/d pre-computed)
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024, "architecture": "lfm2",
    },
    "LFM2-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-1.2B-bf16",
        "L": 16, "d": 2048, "architecture": "lfm2",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048, "architecture": "qwen2.5",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "L": 36, "d": 4096, "architecture": "qwen3",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072, "architecture": "llama",
    },
}

# Probes (same 60 as Exp 1)
PROBE_CATEGORIES = {
    "retrieval": [
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
    ],
    "arithmetic": [
        "What is 347 + 528?", "What is 15 * 23?", "What is 1024 / 16?",
        "What is 99 - 37?", "What is 8 * 7 + 13?", "What is 256 + 384 - 100?",
        "What is 12 * 12?", "What is 999 - 456?", "What is 50 * 20 + 1?",
        "What is 128 / 4?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10. The bat costs $1.00 more. How much is the ball?",
        "If 5 machines make 5 widgets in 5 minutes, how long for 100 to make 100?",
        "A farmer has 17 sheep. All but 9 die. How many left?",
        "A lily pad doubles daily. 48 days to cover lake. When half?",
        "What comes next: 2, 6, 12, 20, 30, ?",
        "Three friends split $90. A gets 2x B. B gets 2x C. How much does C get?",
        "If you rearrange CIFAIPC you get a country. What is it?",
        "A train at 60mph, another at 80mph toward it, 280 miles apart. When meet?",
        "There are 48 on a bus. 8 off, 5 on. Then 12 off, 7 on. How many?",
        "If all roses are flowers and some fade, do some roses fade?",
    ],
    "creative": [
        "Write a haiku about the ocean.", "Describe a sunset in one vivid sentence.",
        "Write a short poem about time.", "Describe chocolate in three words.",
        "Write a twist-ending one-liner.", "Describe rain on a tin roof.",
        "Write a metaphor for loneliness.", "Describe blue to someone blind.",
        "Write a sun-moon dialogue.", "Describe flying in one sentence.",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Write a Python function to check if a number is prime.",
        "Write a Python function for Fibonacci.", "Write max() without max().",
        "Write a palindrome checker.", "Flatten a nested list in one line.",
        "Write bubble sort.", "Count words in a string.",
        "Write recursive factorial.", "Merge two sorted lists.",
    ],
    "narrative": [
        "Once upon a time in a faraway kingdom, there lived a",
        "The old lighthouse keeper watched the storm from",
        "In the year 2150, humanity had finally achieved",
        "She opened the letter and read:", "The forest was silent except for",
        "He had walked for three days when he saw",
        "The library held a secret undiscovered for",
        "As the last leaf fell from the oak,",
        "The musician played a melody that made everyone",
        "Deep beneath the ocean, a creature stirred for the first time in",
    ],
}


# =============================================================================
# ID Trajectory Collection
# =============================================================================


def collect_id_trajectory(
    model_name: str, model_info: dict, probes: list[str], backend
) -> dict:
    """Collect per-layer TwoNN intrinsic dimension trajectory for a model."""
    import mlx.core as mx

    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    model_path = model_info["path"]
    logger.info(f"Loading {model_name} from {model_path}")
    model, tokenizer = backend.load_model(model_path)

    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    embed = getattr(base, "embed_tokens", None)
    num_layers = len(layers) if layers else 0

    logger.info(f"  {num_layers} layers, collecting last-token activations...")

    # Collect last-token representation at each layer for all probes
    layer_points = [[] for _ in range(num_layers)]

    for pi, prompt in enumerate(probes):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        try:
            mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break
            try:
                hidden = layer(hidden, mask=mask)
            except (TypeError, ValueError):
                try:
                    hidden = layer(hidden, mask)
                except (TypeError, ValueError):
                    hidden = layer(hidden)

            # Last token
            last_token = hidden[:, -1, :]
            mx.eval(last_token)
            layer_points[i].append(np.array(last_token[0].tolist(), dtype=np.float32))

    # Compute TwoNN ID at each layer
    id_trajectory = []
    for i in range(num_layers):
        pts = np.stack(layer_points[i]) if layer_points[i] else np.zeros((1, 1))
        if pts.shape[0] < 4:
            id_trajectory.append(float("nan"))
            continue
        try:
            est = IntrinsicDimension.compute_two_nn(pts, backend=backend)
            id_trajectory.append(est.intrinsic_dimension)
        except Exception as e:
            logger.warning(f"  ID failed at layer {i}: {e}")
            id_trajectory.append(float("nan"))

    # Compute derived quantities
    valid = [x for x in id_trajectory if not math.isnan(x)]
    peak_id = max(valid) if valid else 0.0
    final_id = valid[-1] if valid else 0.0
    expansion_ratio = peak_id / final_id if final_id > 0 else 0.0

    L = model_info["L"]
    d = model_info["d"]

    logger.info(
        f"  ID trajectory: peak={peak_id:.1f}, final={final_id:.1f}, "
        f"expansion_ratio={expansion_ratio:.3f}, L/d={L/d:.4f}"
    )

    del model, tokenizer
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "L": L,
        "d": d,
        "ld_ratio": L / d,
        "num_layers": num_layers,
        "id_trajectory": id_trajectory,
        "peak_id": peak_id,
        "final_id": final_id,
        "expansion_ratio": expansion_ratio,
    }


# =============================================================================
# Trajectory Comparison
# =============================================================================


def normalize_trajectory(id_trajectory: list[float], n_points: int = 11) -> np.ndarray:
    """Normalize ID trajectory to n_points uniform positions [0, 1].

    Matches the approach in curvature_profile.py::build_family_baseline().
    """
    ids = np.array(id_trajectory)
    valid = ~np.isnan(ids)
    if not np.any(valid):
        return np.zeros(n_points)

    valid_indices = np.where(valid)[0]
    valid_values = ids[valid]

    # Normalize layer indices to [0, 1]
    max_idx = max(valid_indices)
    if max_idx == 0:
        return np.full(n_points, valid_values[0])

    norm_indices = valid_indices / max_idx

    # Interpolate to uniform grid
    grid = np.linspace(0, 1, n_points)
    normalized = np.interp(grid, norm_indices, valid_values)

    # Min-max normalize the values to [0, 1] for shape comparison
    mn, mx = normalized.min(), normalized.max()
    if mx > mn:
        normalized = (normalized - mn) / (mx - mn)

    return normalized


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two normalized trajectories (Procrustes-style)."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def spearman_rank(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two trajectories."""
    from modelcypher.core.domain.statistics import spearman_correlation

    return spearman_correlation(a.tolist(), b.tolist())


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic Time Warping distance for phase-shift-robust comparison."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run the L/d ratio scaling experiment."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Select models and probes
    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3-8B", "Llama-3.2-3B"]
        n_per_cat = 3
    else:
        model_names = args.models or list(MODEL_REGISTRY.keys())
        n_per_cat = 10

    probes = []
    for cat, prompts in PROBE_CATEGORIES.items():
        probes.extend(prompts[:n_per_cat])

    logger.info(f"Experiment: {len(model_names)} models, {len(probes)} probes")

    # Phase 1: Collect ID trajectories
    trajectories = {}
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}")
            continue
        result = collect_id_trajectory(model_name, MODEL_REGISTRY[model_name], probes, backend)
        trajectories[model_name] = result
        gc.collect()

    # Phase 2: Normalize and compare
    normalized = {}
    for name, traj in trajectories.items():
        normalized[name] = normalize_trajectory(traj["id_trajectory"])

    # Pairwise comparison matrix
    names = list(normalized.keys())
    n_models = len(names)
    pairwise = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_a, name_b = names[i], names[j]
            traj_a, traj_b = normalized[name_a], normalized[name_b]

            pd = procrustes_distance(traj_a, traj_b)
            sr = spearman_rank(traj_a, traj_b)
            dtw = dtw_distance(traj_a, traj_b)

            ld_a = trajectories[name_a]["ld_ratio"]
            ld_b = trajectories[name_b]["ld_ratio"]
            ld_distance = abs(math.log(ld_a) - math.log(ld_b))

            L_a = trajectories[name_a]["L"]
            L_b = trajectories[name_b]["L"]
            L_distance = abs(math.log(L_a) - math.log(L_b))

            arch_a = trajectories[name_a]["architecture"]
            arch_b = trajectories[name_b]["architecture"]
            same_family = arch_a == arch_b

            er_a = trajectories[name_a]["expansion_ratio"]
            er_b = trajectories[name_b]["expansion_ratio"]
            er_relative_diff = abs(er_a - er_b) / max(er_a, er_b, 1e-10)

            pairwise.append({
                "model_a": name_a,
                "model_b": name_b,
                "procrustes": pd,
                "spearman": sr,
                "dtw": dtw,
                "ld_a": ld_a,
                "ld_b": ld_b,
                "ld_distance": ld_distance,
                "L_a": L_a,
                "L_b": L_b,
                "L_distance": L_distance,
                "same_family": same_family,
                "expansion_ratio_a": er_a,
                "expansion_ratio_b": er_b,
                "er_relative_diff": er_relative_diff,
            })

    # Phase 3: Test predictions
    def find_pair(a: str, b: str) -> dict | None:
        for p in pairwise:
            if (p["model_a"] == a and p["model_b"] == b) or (p["model_a"] == b and p["model_b"] == a):
                return p
        return None

    # Prediction 1: LFM2-1.2B↔Qwen3-8B < LFM2-1.2B↔LFM2-350M
    p1_target = find_pair("LFM2-1.2B", "Qwen3-8B")
    p1_control = find_pair("LFM2-1.2B", "LFM2-350M")
    if p1_target and p1_control:
        p1_pass = p1_target["procrustes"] < p1_control["procrustes"]
        logger.info(
            f"P1: LFM2-1.2B↔Qwen3-8B ({p1_target['procrustes']:.3f}) vs "
            f"LFM2-1.2B↔LFM2-350M ({p1_control['procrustes']:.3f}): "
            f"{'PASS' if p1_pass else 'FAIL'}"
        )
    else:
        p1_pass = None
        logger.info("P1: Insufficient models for test")

    # Prediction 2: Qwen2.5-3B↔Qwen3-8B > Qwen3-8B↔Llama-3.2-3B
    p2_same_L = find_pair("Qwen2.5-3B", "Qwen3-8B")
    p2_same_ld = find_pair("Qwen3-8B", "Llama-3.2-3B")
    if p2_same_L and p2_same_ld:
        p2_pass = p2_same_L["procrustes"] > p2_same_ld["procrustes"]
        logger.info(
            f"P2: Qwen2.5↔Qwen3 ({p2_same_L['procrustes']:.3f}) vs "
            f"Qwen3↔Llama ({p2_same_ld['procrustes']:.3f}): "
            f"{'PASS' if p2_pass else 'FAIL'}"
        )
    else:
        p2_pass = None
        logger.info("P2: Insufficient models for test")

    # Prediction 3: Expansion ratio within 20% for similar L/d
    p3_pairs = [(p, p["er_relative_diff"] < 0.2) for p in pairwise if p["ld_distance"] < 0.3]
    p3_pass = all(ok for _, ok in p3_pairs) if p3_pairs else None
    for p, ok in p3_pairs:
        logger.info(
            f"P3: {p['model_a']}↔{p['model_b']} ld_dist={p['ld_distance']:.3f} "
            f"er_diff={p['er_relative_diff']:.3f}: {'PASS' if ok else 'FAIL'}"
        )

    # Partial correlation: trajectory similarity ~ L/d_distance
    if len(pairwise) >= 5:
        from modelcypher.core.domain.statistics import spearman_correlation

        proc_dists = [p["procrustes"] for p in pairwise]
        ld_dists = [p["ld_distance"] for p in pairwise]
        L_dists = [p["L_distance"] for p in pairwise]

        # Simple Spearman since partial correlation needs scipy
        spearman_ld = spearman_correlation(ld_dists, proc_dists)
        spearman_L = spearman_correlation(L_dists, proc_dists)

        logger.info(
            f"Spearman(L/d_dist, procrustes): {spearman_ld:.3f}")
        logger.info(
            f"Spearman(L_dist, procrustes): {spearman_L:.3f}")
    else:
        spearman_ld = float("nan")
        spearman_L = float("nan")

    # Overall verdict
    all_tests = [p1_pass, p2_pass, p3_pass]
    passed = [t for t in all_tests if t is not None]
    overall = all(passed) if passed else False

    summary = {
        "n_models": n_models,
        "n_probes": len(probes),
        "prediction_1": {
            "description": "LFM2-1.2B↔Qwen3-8B closer than LFM2-1.2B↔LFM2-350M",
            "pass": p1_pass,
            "target_dist": p1_target["procrustes"] if p1_target else None,
            "control_dist": p1_control["procrustes"] if p1_control else None,
        },
        "prediction_2": {
            "description": "Same L (Qwen2.5↔Qwen3) more distant than same L/d (Qwen3↔Llama)",
            "pass": p2_pass,
        },
        "prediction_3": {
            "description": "Similar L/d → expansion_ratio within 20%",
            "pass": p3_pass,
            "n_pairs_tested": len(p3_pairs),
        },
        "spearman_ld_vs_procrustes": spearman_ld,
        "spearman_L_vs_procrustes": spearman_L,
        "overall_verdict": "H1 SUPPORTED" if overall else "H1 REFUTED",
        "references": [
            "Dey et al. (NeurIPS 2025, arXiv:2505.01618): CompleteP",
            "Joshi et al. (NeurIPS 2025, arXiv:2511.20315): 28-model ID",
        ],
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT VERDICT: {summary['overall_verdict']}")
    logger.info(f"{'='*60}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "ld_ratio_scaling_hypothesis",
        "trajectories": {k: {
            **v,
            "normalized_trajectory": normalized[k].tolist(),
        } for k, v in trajectories.items()},
        "pairwise": pairwise,
        "summary": summary,
    }

    with open(output_dir / "ld_ratio_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir / 'ld_ratio_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="L/d Ratio Scaling Experiment")
    parser.add_argument("--output", default="results/ld_ratio/")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
