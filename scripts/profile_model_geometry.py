#!/usr/bin/env python3
"""Profile any model's geometric properties for alignment analysis.

Measures:
1. Dimensional trajectory through layers (TwoNN intrinsic dimension)
2. Compression/φ ratio
3. Fundamental constants in weight SVD (π/e, e/π, φ, √2)
4. Condition number stability

Usage:
    python scripts/profile_model_geometry.py --model /path/to/model
    python scripts/profile_model_geometry.py --model mlx-community/Qwen3-1.7B-MLX-bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2
PI_OVER_E = np.pi / np.e
E_OVER_PI = np.e / np.pi
SQRT2 = np.sqrt(2)

CONSTANTS = {
    "phi": PHI,
    "pi_over_e": PI_OVER_E,
    "e_over_pi": E_OVER_PI,
    "sqrt2": SQRT2,
}

# Diverse test prompts for trajectory analysis
TEST_PROMPTS = [
    "Question: If John has 5 apples and buys 3 more, how many does he have?\n\nAnswer:",
    "Question: What is the capital of France?\n\nAnswer:",
    "Question: Explain why the sky is blue.\n\nAnswer:",
    "Question: A train travels 60 mph for 2 hours. How far does it go?\n\nAnswer:",
    "Question: What causes seasons on Earth?\n\nAnswer:",
    "Question: If all dogs are mammals, and Rex is a dog, what is Rex?\n\nAnswer:",
    "Question: Write a haiku about the moon.\n\nAnswer:",
    "Question: What is 7 times 8?\n\nAnswer:",
    "Question: Describe how photosynthesis works.\n\nAnswer:",
    "Question: If it rains, the ground gets wet. It rained. What happened?\n\nAnswer:",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    if len(X) < 10:
        return float('nan')

    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)

    d1 = distances[:, 1]
    d2 = distances[:, 2]

    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')

    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]

    if len(mu) < 5:
        return float('nan')

    log_mu = np.log(mu)
    d = len(log_mu) / np.sum(log_mu)

    return float(d)


def count_constant_matches(ratios: np.ndarray, tolerance: float = 0.05) -> Dict[str, int]:
    """Count how many ratios match fundamental constants."""
    matches = {}
    for name, value in CONSTANTS.items():
        matches[name] = int(np.sum(np.abs(ratios - value) / value < tolerance))
    return matches


def analyze_weight_matrix(W: np.ndarray) -> Dict:
    """Analyze a single weight matrix for geometric properties."""
    try:
        _, S, _ = svd(W, full_matrices=False)
    except Exception as e:
        return {"error": str(e)}

    # Singular value ratios
    ratios = S[:-1] / S[1:]
    ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]

    if len(ratios) == 0:
        return {"error": "No valid ratios"}

    # Count constant matches
    matches = count_constant_matches(ratios)

    # Condition number
    kappa = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')

    return {
        "n_singular_values": len(S),
        "condition_number": float(kappa),
        "constant_matches": matches,
        "total_matches": sum(matches.values()),
        "match_ratio": sum(matches.values()) / len(ratios) if len(ratios) > 0 else 0,
        "top_3_singular": S[:3].tolist() if len(S) >= 3 else S.tolist(),
    }


def get_dimensional_trajectory(model, tokenizer, prompt: str) -> Dict:
    """Get dimensional trajectory through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []

    # Embedding layer
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)

        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    # Compute metrics
    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]

    n_layers = len(model.model.layers)

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        initial_dim = traj[0] if not np.isnan(traj[0]) else valid[0]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]

        expansion_ratio = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
        compression_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
        traj_variance = float(np.var(np.diff(valid)))
    else:
        peak_idx = -1
        peak_dim = float('nan')
        initial_dim = float('nan')
        final_dim = float('nan')
        expansion_ratio = float('nan')
        compression_ratio = float('nan')
        traj_variance = float('nan')

    return {
        "trajectory": trajectory,
        "n_layers": n_layers,
        "peak_layer": int(peak_idx),
        "peak_layer_pct": peak_idx / n_layers * 100 if peak_idx >= 0 else float('nan'),
        "peak_dim": peak_dim,
        "initial_dim": initial_dim,
        "final_dim": final_dim,
        "expansion_ratio": expansion_ratio,
        "compression_ratio": compression_ratio,
        "compression_vs_phi": compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan'),
        "trajectory_variance": traj_variance,
    }


def profile_model(model_path: str, output_path: Optional[str] = None) -> Dict:
    """Complete geometric profile of a model."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info(f"GEOMETRIC PROFILE: {model_path}")
    logger.info("=" * 70)

    # Load model
    logger.info("\nLoading model...")
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"error": str(e)}

    # Model architecture info
    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    vocab_size = model.model.embed_tokens.weight.shape[0]

    logger.info(f"  Layers: {n_layers}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Vocab size: {vocab_size}")

    results = {
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "architecture": {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
        },
    }

    # 1. Dimensional trajectory analysis
    logger.info("\n" + "-" * 50)
    logger.info("DIMENSIONAL TRAJECTORY ANALYSIS")
    logger.info("-" * 50)

    trajectories = []
    for i, prompt in enumerate(TEST_PROMPTS):
        logger.info(f"  Prompt {i+1}/{len(TEST_PROMPTS)}...")
        traj_data = get_dimensional_trajectory(model, tokenizer, prompt)
        trajectories.append(traj_data)

    # Aggregate trajectory metrics
    valid_trajs = [t for t in trajectories if not np.isnan(t["compression_vs_phi"])]

    if valid_trajs:
        results["trajectory"] = {
            "n_samples": len(valid_trajs),
            "mean_peak_layer_pct": np.mean([t["peak_layer_pct"] for t in valid_trajs]),
            "std_peak_layer_pct": np.std([t["peak_layer_pct"] for t in valid_trajs]),
            "mean_compression_vs_phi": np.mean([t["compression_vs_phi"] for t in valid_trajs]),
            "std_compression_vs_phi": np.std([t["compression_vs_phi"] for t in valid_trajs]),
            "mean_initial_dim": np.mean([t["initial_dim"] for t in valid_trajs]),
            "mean_peak_dim": np.mean([t["peak_dim"] for t in valid_trajs]),
            "mean_final_dim": np.mean([t["final_dim"] for t in valid_trajs]),
            "mean_trajectory_variance": np.mean([t["trajectory_variance"] for t in valid_trajs]),
        }

        logger.info(f"\n  Peak layer: {results['trajectory']['mean_peak_layer_pct']:.1f}% ± {results['trajectory']['std_peak_layer_pct']:.1f}%")
        logger.info(f"  Compression/φ: {results['trajectory']['mean_compression_vs_phi']:.3f} ± {results['trajectory']['std_compression_vs_phi']:.3f}")
        logger.info(f"  Dimensions: {results['trajectory']['mean_initial_dim']:.1f} → {results['trajectory']['mean_peak_dim']:.1f} → {results['trajectory']['mean_final_dim']:.1f}")
    else:
        results["trajectory"] = {"error": "No valid trajectories"}

    # 2. Weight matrix analysis
    logger.info("\n" + "-" * 50)
    logger.info("WEIGHT MATRIX ANALYSIS")
    logger.info("-" * 50)

    weight_analyses = []
    total_matches = 0
    total_ratios = 0
    kappas = []

    # Analyze embedding
    logger.info("  Analyzing embedding layer...")
    emb_weight = np.array(model.model.embed_tokens.weight.tolist())
    emb_analysis = analyze_weight_matrix(emb_weight)
    weight_analyses.append({"layer": "embedding", **emb_analysis})
    if "total_matches" in emb_analysis:
        total_matches += emb_analysis["total_matches"]
        total_ratios += emb_analysis["n_singular_values"] - 1
        if emb_analysis["condition_number"] != float('inf'):
            kappas.append(emb_analysis["condition_number"])

    # Analyze each layer
    for i, layer in enumerate(model.model.layers):
        if i % 5 == 0:
            logger.info(f"  Analyzing layer {i}/{n_layers}...")

        # Attention weights
        if hasattr(layer, 'self_attn'):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, name):
                    proj = getattr(layer.self_attn, name)
                    if hasattr(proj, 'weight'):
                        W = np.array(proj.weight.tolist())
                        analysis = analyze_weight_matrix(W)
                        weight_analyses.append({"layer": f"L{i}_{name}", **analysis})
                        if "total_matches" in analysis:
                            total_matches += analysis["total_matches"]
                            total_ratios += analysis["n_singular_values"] - 1
                            if analysis["condition_number"] != float('inf'):
                                kappas.append(analysis["condition_number"])

        # MLP weights
        if hasattr(layer, 'mlp'):
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(layer.mlp, name):
                    proj = getattr(layer.mlp, name)
                    if hasattr(proj, 'weight'):
                        W = np.array(proj.weight.tolist())
                        analysis = analyze_weight_matrix(W)
                        weight_analyses.append({"layer": f"L{i}_{name}", **analysis})
                        if "total_matches" in analysis:
                            total_matches += analysis["total_matches"]
                            total_ratios += analysis["n_singular_values"] - 1
                            if analysis["condition_number"] != float('inf'):
                                kappas.append(analysis["condition_number"])

    results["weights"] = {
        "total_constant_matches": total_matches,
        "total_ratios": total_ratios,
        "match_ratio": total_matches / total_ratios if total_ratios > 0 else 0,
        "mean_condition_number": np.mean(kappas) if kappas else float('nan'),
        "std_condition_number": np.std(kappas) if kappas else float('nan'),
        "kappa_range": (min(kappas), max(kappas)) if kappas else (float('nan'), float('nan')),
    }

    logger.info(f"\n  Total constant matches: {total_matches}")
    logger.info(f"  Match ratio: {results['weights']['match_ratio']:.4f}")
    logger.info(f"  Mean κ: {results['weights']['mean_condition_number']:.1f} ± {results['weights']['std_condition_number']:.1f}")

    # 3. Compute overall score
    logger.info("\n" + "-" * 50)
    logger.info("GEOMETRIC SCORE")
    logger.info("-" * 50)

    score = 0.0

    # Peak layer position (ideal: 50-60%)
    if "trajectory" in results and "mean_peak_layer_pct" in results["trajectory"]:
        peak_pct = results["trajectory"]["mean_peak_layer_pct"]
        ideal_peak = 55
        peak_score = max(0, 1 - abs(peak_pct - ideal_peak) / 50)
        score += 0.20 * peak_score
        logger.info(f"  Peak position score (20%): {peak_score:.3f}")

    # Compression/φ (ideal: 1.0)
    if "trajectory" in results and "mean_compression_vs_phi" in results["trajectory"]:
        comp_phi = results["trajectory"]["mean_compression_vs_phi"]
        comp_score = max(0, 1 - abs(comp_phi - 1.0))
        score += 0.25 * comp_score
        logger.info(f"  Compression/φ score (25%): {comp_score:.3f}")

    # Constant matches (higher = better)
    if "weights" in results and "match_ratio" in results["weights"]:
        match_ratio = results["weights"]["match_ratio"]
        match_score = min(1.0, match_ratio * 10)  # Scale up
        score += 0.15 * match_score
        logger.info(f"  Constant match score (15%): {match_score:.3f}")

    # κ stability (lower std = better)
    if "weights" in results and "std_condition_number" in results["weights"]:
        kappa_std = results["weights"]["std_condition_number"]
        if not np.isnan(kappa_std):
            kappa_score = max(0, 1 - kappa_std / 1000)
            score += 0.15 * kappa_score
            logger.info(f"  κ stability score (15%): {kappa_score:.3f}")

    # Trajectory stability (lower variance = better)
    if "trajectory" in results and "mean_trajectory_variance" in results["trajectory"]:
        traj_var = results["trajectory"]["mean_trajectory_variance"]
        traj_score = max(0, 1 - traj_var / 50)
        score += 0.15 * traj_score
        logger.info(f"  Trajectory stability score (15%): {traj_score:.3f}")

    # MLX native (assumed yes if we got here)
    score += 0.10 * 1.0
    logger.info(f"  MLX native score (10%): 1.000")

    results["geometric_score"] = score
    logger.info(f"\n  TOTAL GEOMETRIC SCORE: {score:.3f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Architecture: {n_layers} layers, {hidden_dim} hidden dim")
    if "trajectory" in results and "mean_compression_vs_phi" in results["trajectory"]:
        logger.info(f"Compression/φ: {results['trajectory']['mean_compression_vs_phi']:.3f}")
        logger.info(f"Peak layer: {results['trajectory']['mean_peak_layer_pct']:.1f}%")
    logger.info(f"Constant matches: {results['weights']['match_ratio']:.4f}")
    logger.info(f"GEOMETRIC SCORE: {score:.3f}")

    # Save results
    if output_path:
        out_path = Path(output_path)
    else:
        model_name = Path(model_path).name.replace("/", "_")
        out_path = Path(f"data/experiments/geometric_profile_{model_name}.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile model geometric properties")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()

    profile_model(args.model, args.output)


if __name__ == "__main__":
    main()
