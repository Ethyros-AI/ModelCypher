#!/usr/bin/env python3
"""Experiment: Track intrinsic dimension through layers.

Hypothesis: The model traverses a dimensional curve during processing:
- Input: Low intrinsic dimension (3D concepts encoded)
- Processing: High fractional dimension (high-D geodesic space)
- Output: Low dimension again (compressed to answer)

And the φ ratio in entropy correlates with the dimensional expansion ratio.

Key predictions:
1. Intrinsic dimension should peak mid-network (like entropy)
2. Peak dimension should be fractional, not integer
3. Correct answers should have higher peak dimension (fuller expansion)
4. The ratio of peak_dim / input_dim should relate to φ
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import re

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

PHI = (1 + np.sqrt(5)) / 2
PI_OVER_E = np.pi / np.e
E_OVER_PI = np.e / np.pi
SQRT2 = np.sqrt(2)


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
    """Estimate intrinsic dimension via TwoNN method.

    The TwoNN estimator uses the ratio of distances to the
    first and second nearest neighbors. For a d-dimensional
    manifold, this ratio follows a specific distribution.
    """
    if len(X) < 10:
        return float('nan')

    # Need at least 3 neighbors (self + 2 nearest)
    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)

    # Ratio of second to first neighbor distance
    # Skip distance to self (index 0)
    d1 = distances[:, 1]  # First neighbor
    d2 = distances[:, 2]  # Second neighbor

    # Filter valid ratios (d1 > 0 to avoid division by zero)
    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')

    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]  # Ratio should be > 1

    if len(mu) < 5:
        return float('nan')

    # MLE estimator for intrinsic dimension
    # d = 1 / mean(log(mu))
    log_mu = np.log(mu)
    d = len(log_mu) / np.sum(log_mu)

    return float(d)


def compute_intrinsic_dimension_mle(X: np.ndarray, k: int = 10) -> float:
    """Alternative: MLE estimator using k neighbors.

    More robust for smaller samples but requires choosing k.
    """
    if len(X) < k + 2:
        return float('nan')

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)

    # Skip self-distance
    distances = distances[:, 1:]

    # MLE estimator (Levina & Bickel, 2004)
    # d = 1 / (1/n * sum_i(1/(k-1) * sum_j(log(d_k / d_j))))
    d_k = distances[:, -1:]  # k-th neighbor distance

    # Avoid log(0)
    valid = (distances > 1e-10).all(axis=1) & (d_k.flatten() > 1e-10)
    if valid.sum() < 5:
        return float('nan')

    distances = distances[valid]
    d_k = d_k[valid]

    log_ratios = np.log(d_k / distances[:, :-1])
    m_k = log_ratios.mean(axis=1)

    # Harmonic mean of local dimension estimates
    d = 1 / m_k.mean() if m_k.mean() > 1e-10 else float('nan')

    return float(d)


def compute_spectral_entropy(activations: np.ndarray) -> float:
    """Compute entropy from SVD singular values."""
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    if len(activations) < 2:
        return 0.0

    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)

    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0

    p = S_valid ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def compute_effective_rank(activations: np.ndarray) -> float:
    """Effective rank from singular value entropy.

    This gives another view of dimensionality - how many
    dimensions are "active" in the representation.
    """
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    if len(activations) < 2:
        return 0.0

    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)

    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0

    # Normalized singular values as probabilities
    p = S_valid / S_valid.sum()

    # Effective rank = exp(entropy)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))


def analyze_problem(
    model,
    tokenizer,
    prompt: str,
    expected: str,
    n_samples: int = 50
) -> Dict:
    """Analyze dimensional trajectory for a single problem."""
    import mlx.core as mx
    from mlx_lm import generate

    # Generate answer
    full_prompt = f"Question: {prompt}\n\nAnswer:"
    output = generate(model, tokenizer, prompt=full_prompt, max_tokens=500, verbose=False)

    # Extract predicted answer
    if "####" in output:
        answer_part = output.split("####")[-1].replace(",", "").replace("$", "").strip()
        nums = re.findall(r'-?\d+\.?\d*', answer_part)
        if nums:
            try:
                num_val = float(nums[0])
                predicted = str(int(num_val)) if num_val == int(num_val) else nums[0]
            except ValueError:
                predicted = nums[0] if nums else ""
        else:
            predicted = ""
    else:
        nums = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = nums[-1] if nums else ""

    is_correct = predicted == expected

    # Get activations through layers
    tokens = tokenizer.encode(full_prompt)
    input_ids = mx.array([tokens])

    n_layers = len(model.model.layers)

    # Collect activations at each layer
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    layer_data = []

    # Initial embedding
    emb_np = np.array(hidden[0].tolist())  # [seq_len, hidden_dim]
    layer_data.append({
        "layer": -1,  # Embedding layer
        "activations": emb_np,
    })

    for layer_idx, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)

        act_np = np.array(hidden[0].tolist())  # [seq_len, hidden_dim]
        layer_data.append({
            "layer": layer_idx,
            "activations": act_np,
        })

    # Compute metrics for each layer
    trajectories = {
        "intrinsic_dim_twonn": [],
        "intrinsic_dim_mle": [],
        "effective_rank": [],
        "spectral_entropy": [],
        "activation_norm": [],
    }

    for ld in layer_data:
        act = ld["activations"]

        # Use last N tokens for stability (or all if short)
        if len(act) > n_samples:
            act = act[-n_samples:]

        trajectories["intrinsic_dim_twonn"].append(compute_intrinsic_dimension_twonn(act))
        trajectories["intrinsic_dim_mle"].append(compute_intrinsic_dimension_mle(act))
        trajectories["effective_rank"].append(compute_effective_rank(act))
        trajectories["spectral_entropy"].append(compute_spectral_entropy(act))
        trajectories["activation_norm"].append(float(np.linalg.norm(act)))

    # Compute trajectory statistics
    dim_traj = np.array(trajectories["intrinsic_dim_twonn"])
    valid_dims = dim_traj[~np.isnan(dim_traj)]

    if len(valid_dims) > 2:
        peak_idx = np.nanargmax(dim_traj)
        peak_dim = dim_traj[peak_idx]
        initial_dim = dim_traj[0] if not np.isnan(dim_traj[0]) else valid_dims[0]
        final_dim = dim_traj[-1] if not np.isnan(dim_traj[-1]) else valid_dims[-1]

        # Dimensional expansion ratio
        dim_expansion = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
        dim_compression = peak_dim / final_dim if final_dim > 0.1 else float('nan')

        # Compare to φ
        expansion_vs_phi = dim_expansion / PHI if not np.isnan(dim_expansion) else float('nan')
    else:
        peak_idx = -1
        peak_dim = float('nan')
        initial_dim = float('nan')
        final_dim = float('nan')
        dim_expansion = float('nan')
        dim_compression = float('nan')
        expansion_vs_phi = float('nan')

    # Entropy trajectory analysis (for comparison)
    entropy_traj = np.array(trajectories["spectral_entropy"])
    entropy_peak_idx = np.argmax(entropy_traj)
    entropy_peak = entropy_traj[entropy_peak_idx]
    entropy_initial = entropy_traj[0]
    entropy_final = entropy_traj[-1]

    if entropy_peak_idx > 0:
        entropy_expansion_rate = (entropy_peak - entropy_initial) / entropy_peak_idx
    else:
        entropy_expansion_rate = 0

    compression_layers = len(entropy_traj) - entropy_peak_idx - 1
    if compression_layers > 0:
        entropy_compression_rate = (entropy_peak - entropy_final) / compression_layers
    else:
        entropy_compression_rate = 0

    if entropy_expansion_rate > 1e-10:
        entropy_ratio = entropy_compression_rate / entropy_expansion_rate
        entropy_ratio_vs_phi = entropy_ratio / PHI
    else:
        entropy_ratio = float('inf')
        entropy_ratio_vs_phi = float('inf')

    return {
        "prompt": prompt,
        "expected": expected,
        "predicted": predicted,
        "is_correct": is_correct,
        "output": output[:500],  # Truncate for storage
        "trajectories": trajectories,
        "dimensional_analysis": {
            "peak_layer": int(peak_idx),
            "peak_dim": peak_dim,
            "initial_dim": initial_dim,
            "final_dim": final_dim,
            "expansion_ratio": dim_expansion,
            "compression_ratio": dim_compression,
            "expansion_vs_phi": expansion_vs_phi,
        },
        "entropy_analysis": {
            "peak_layer": int(entropy_peak_idx),
            "peak_entropy": entropy_peak,
            "initial_entropy": entropy_initial,
            "final_entropy": entropy_final,
            "expansion_rate": entropy_expansion_rate,
            "compression_rate": entropy_compression_rate,
            "ratio_vs_phi": entropy_ratio_vs_phi,
        },
    }


def main():
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    logger.info("=" * 70)
    logger.info("DIMENSIONAL CURVE EXPERIMENT")
    logger.info("Tracking intrinsic dimension through layers")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Load GSM8K problems
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=20)

    logger.info(f"\nAnalyzing {len(gsm_test.samples)} problems...")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "adapter": adapter_path,
        "n_layers": n_layers,
        "problems": [],
    }

    correct_analyses = []
    incorrect_analyses = []

    for i, sample in enumerate(gsm_test.samples):
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        logger.info(f"\n[{i+1}/{len(gsm_test.samples)}] Analyzing problem...")

        analysis = analyze_problem(model, tokenizer, question, expected)
        results["problems"].append(analysis)

        status = "CORRECT" if analysis["is_correct"] else "WRONG"
        logger.info(f"  Status: {status}")
        logger.info(f"  Peak dim layer: {analysis['dimensional_analysis']['peak_layer']}")
        logger.info(f"  Dimension: {analysis['dimensional_analysis']['initial_dim']:.2f} → "
                   f"{analysis['dimensional_analysis']['peak_dim']:.2f} → "
                   f"{analysis['dimensional_analysis']['final_dim']:.2f}")
        logger.info(f"  Expansion ratio: {analysis['dimensional_analysis']['expansion_ratio']:.3f}")
        logger.info(f"  Expansion/φ: {analysis['dimensional_analysis']['expansion_vs_phi']:.3f}")
        logger.info(f"  Entropy ratio/φ: {analysis['entropy_analysis']['ratio_vs_phi']:.3f}")

        if analysis["is_correct"]:
            correct_analyses.append(analysis)
        else:
            incorrect_analyses.append(analysis)

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: DIMENSIONAL CURVE ANALYSIS")
    logger.info("=" * 70)

    def compute_stats(analyses: List[Dict], label: str):
        if not analyses:
            logger.info(f"\n{label}: No samples")
            return {}

        peak_dims = [a["dimensional_analysis"]["peak_dim"] for a in analyses
                    if not np.isnan(a["dimensional_analysis"]["peak_dim"])]
        initial_dims = [a["dimensional_analysis"]["initial_dim"] for a in analyses
                       if not np.isnan(a["dimensional_analysis"]["initial_dim"])]
        expansion_ratios = [a["dimensional_analysis"]["expansion_ratio"] for a in analyses
                          if not np.isnan(a["dimensional_analysis"]["expansion_ratio"])]
        expansion_vs_phi = [a["dimensional_analysis"]["expansion_vs_phi"] for a in analyses
                          if not np.isnan(a["dimensional_analysis"]["expansion_vs_phi"])]
        entropy_vs_phi = [a["entropy_analysis"]["ratio_vs_phi"] for a in analyses
                         if a["entropy_analysis"]["ratio_vs_phi"] != float('inf')]
        peak_layers = [a["dimensional_analysis"]["peak_layer"] for a in analyses]

        logger.info(f"\n{label} (n={len(analyses)}):")

        if peak_dims:
            logger.info(f"  Peak dimension: {np.mean(peak_dims):.2f} ± {np.std(peak_dims):.2f}")
        if initial_dims:
            logger.info(f"  Initial dimension: {np.mean(initial_dims):.2f} ± {np.std(initial_dims):.2f}")
        if expansion_ratios:
            logger.info(f"  Expansion ratio: {np.mean(expansion_ratios):.3f} ± {np.std(expansion_ratios):.3f}")
        if expansion_vs_phi:
            logger.info(f"  Expansion/φ: {np.mean(expansion_vs_phi):.3f} ± {np.std(expansion_vs_phi):.3f}")
        if entropy_vs_phi:
            logger.info(f"  Entropy ratio/φ: {np.mean(entropy_vs_phi):.3f} ± {np.std(entropy_vs_phi):.3f}")
        if peak_layers:
            logger.info(f"  Peak layer: {np.mean(peak_layers):.1f} ± {np.std(peak_layers):.1f}")

        return {
            "n": len(analyses),
            "peak_dim_mean": np.mean(peak_dims) if peak_dims else None,
            "peak_dim_std": np.std(peak_dims) if peak_dims else None,
            "initial_dim_mean": np.mean(initial_dims) if initial_dims else None,
            "expansion_ratio_mean": np.mean(expansion_ratios) if expansion_ratios else None,
            "expansion_vs_phi_mean": np.mean(expansion_vs_phi) if expansion_vs_phi else None,
            "entropy_vs_phi_mean": np.mean(entropy_vs_phi) if entropy_vs_phi else None,
            "peak_layer_mean": np.mean(peak_layers) if peak_layers else None,
        }

    correct_stats = compute_stats(correct_analyses, "CORRECT ANSWERS")
    incorrect_stats = compute_stats(incorrect_analyses, "INCORRECT ANSWERS")

    results["summary"] = {
        "correct": correct_stats,
        "incorrect": incorrect_stats,
        "accuracy": len(correct_analyses) / len(gsm_test.samples) * 100,
    }

    # Check for dimensional curve pattern
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS TESTING")
    logger.info("=" * 70)

    # H1: Dimension should peak mid-network
    all_peak_layers = [a["dimensional_analysis"]["peak_layer"] for a in results["problems"]]
    mean_peak = np.mean(all_peak_layers)
    logger.info(f"\nH1: Dimension peaks mid-network")
    logger.info(f"  Mean peak layer: {mean_peak:.1f} / {n_layers} ({mean_peak/n_layers*100:.0f}%)")
    logger.info(f"  Expected: ~50% of layers")

    # H2: Peak dimension should be fractional
    all_peak_dims = [a["dimensional_analysis"]["peak_dim"] for a in results["problems"]
                    if not np.isnan(a["dimensional_analysis"]["peak_dim"])]
    if all_peak_dims:
        mean_peak_dim = np.mean(all_peak_dims)
        is_fractional = not float(mean_peak_dim).is_integer()
        logger.info(f"\nH2: Peak dimension is fractional")
        logger.info(f"  Mean peak dimension: {mean_peak_dim:.3f}")
        logger.info(f"  Fractional: {is_fractional}")

    # H3: Correct answers have higher expansion
    if correct_stats.get("expansion_ratio_mean") and incorrect_stats.get("expansion_ratio_mean"):
        ratio_diff = correct_stats["expansion_ratio_mean"] / incorrect_stats["expansion_ratio_mean"]
        logger.info(f"\nH3: Correct answers expand more")
        logger.info(f"  Correct expansion: {correct_stats['expansion_ratio_mean']:.3f}")
        logger.info(f"  Incorrect expansion: {incorrect_stats['expansion_ratio_mean']:.3f}")
        logger.info(f"  Ratio: {ratio_diff:.2f}x")

    # H4: Expansion ratio relates to φ
    if all_peak_dims and correct_stats.get("expansion_vs_phi_mean"):
        logger.info(f"\nH4: Expansion relates to φ")
        logger.info(f"  Correct expansion/φ: {correct_stats['expansion_vs_phi_mean']:.3f}")
        if incorrect_stats.get("expansion_vs_phi_mean"):
            logger.info(f"  Incorrect expansion/φ: {incorrect_stats['expansion_vs_phi_mean']:.3f}")
        logger.info(f"  (Target: ~1.0 if φ governs dimensional projection)")

    # Check for other constants
    all_expansion_ratios = [a["dimensional_analysis"]["expansion_ratio"] for a in results["problems"]
                           if not np.isnan(a["dimensional_analysis"]["expansion_ratio"])]
    if all_expansion_ratios:
        mean_ratio = np.mean(all_expansion_ratios)
        logger.info(f"\nConstant matching (mean expansion ratio = {mean_ratio:.4f}):")
        logger.info(f"  vs φ (1.618): {mean_ratio/PHI:.3f}")
        logger.info(f"  vs π/e (1.156): {mean_ratio/PI_OVER_E:.3f}")
        logger.info(f"  vs √2 (1.414): {mean_ratio/SQRT2:.3f}")
        logger.info(f"  vs e/π (0.865): {mean_ratio/E_OVER_PI:.3f}")

    # Save results
    output_path = Path("data/experiments/dimensional_curve_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
