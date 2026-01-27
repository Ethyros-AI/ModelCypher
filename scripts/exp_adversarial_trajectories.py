#!/usr/bin/env python3
"""Experiment: Test if adversarial inputs have pathological dimensional trajectories.

Hypothesis: Adversarial examples (inputs designed to confuse the model) should show:
1. Erratic dimensional trajectories (high variance)
2. Unusual compression ratios (far from φ)
3. Anomalous peak layers (too early or too late)
4. Low initial dimension despite complex surface structure

We'll test with:
1. Normal problems (baseline)
2. Problems with irrelevant information inserted
3. Problems with contradictory information
4. Nonsense problems that look like real problems
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.stats import ks_2samp, ttest_ind
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


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


# Normal problems for baseline
NORMAL_PROBLEMS = [
    ("If John has 5 apples and buys 3 more, how many apples does he have?", "8"),
    ("A store sells pencils for $2 each. If Maria buys 4 pencils, how much does she spend?", "8"),
    ("Tom read 15 pages on Monday and 20 pages on Tuesday. How many pages did he read in total?", "35"),
    ("Sarah has 12 cookies and gives 4 to her friend. How many cookies does Sarah have left?", "8"),
    ("A train travels at 60 miles per hour. How far does it travel in 2 hours?", "120"),
]

# Adversarial: Irrelevant information inserted
IRRELEVANT_INFO = [
    ("If John has 5 apples (his favorite color is blue) and buys 3 more (the store was painted yellow), how many apples does he have?", "8"),
    ("A store (which opened in 1985 and has 42 employees) sells pencils for $2 each. If Maria (who is 32 years old) buys 4 pencils, how much does she spend?", "8"),
    ("Tom read 15 pages on Monday (it was raining) and 20 pages on Tuesday (his dog's name is Rex). How many pages did he read in total?", "35"),
    ("Sarah, who lives on Oak Street and has brown hair, has 12 cookies and gives 4 to her friend who drives a red car. How many cookies does Sarah have left?", "8"),
    ("A train (built in 2010 with serial number XJ-7829) travels at 60 miles per hour through mountains. How far does it travel in 2 hours if the conductor is 45?", "120"),
]

# Adversarial: Contradictory information
CONTRADICTORY_INFO = [
    ("John has 5 apples. He has no fruit at all. He buys 3 more apples. How many apples does he have?", "8"),
    ("A store sells pencils for $2 each. The pencils are free. Maria buys 4 pencils. How much does she spend?", "8"),
    ("Tom read 15 pages on Monday, but he didn't read anything that day. He read 20 pages on Tuesday. How many pages total?", "35"),
    ("Sarah has 12 cookies. She has zero cookies. She gives 4 away. How many does she have?", "8"),
    ("A train travels at 60 mph. It's not moving. How far does it go in 2 hours?", "120"),
]

# Adversarial: Nonsense that looks mathematical
NONSENSE_MATH = [
    ("If the purple of 5 apples squared the circular 3, how many apples triangle?", None),
    ("Maria's pencil coefficient equals 4 times the derivative of spending. What is the integral of her purchase?", None),
    ("Tom read fractal pages dimensionally across Tuesday's hyperbolic Monday. How many eigenvalues did he read?", None),
    ("Sarah's cookies underwent mitosis. If 12 bifurcated by 4, what is the cookie entropy?", None),
    ("A train's velocity eigenvector traveled through 60-dimensional space for 2 Planck times. What is the geodesic?", None),
]


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


def get_dimensional_trajectory(model, tokenizer, prompt: str) -> Dict:
    """Get full dimensional trajectory through all layers."""
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

    # Trajectory variance (measure of stability)
    if len(valid) > 2:
        traj_variance = np.var(np.diff(valid))
        traj_range = np.max(valid) - np.min(valid)
    else:
        traj_variance = float('nan')
        traj_range = float('nan')

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        initial_dim = traj[0] if not np.isnan(traj[0]) else valid[0]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]

        expansion_ratio = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
        compression_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
    else:
        peak_idx = -1
        peak_dim = float('nan')
        initial_dim = float('nan')
        final_dim = float('nan')
        expansion_ratio = float('nan')
        compression_ratio = float('nan')

    return {
        "trajectory": trajectory,
        "peak_layer": int(peak_idx),
        "peak_dim": peak_dim,
        "initial_dim": initial_dim,
        "final_dim": final_dim,
        "expansion_ratio": expansion_ratio,
        "compression_ratio": compression_ratio,
        "compression_vs_phi": compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan'),
        "trajectory_variance": traj_variance,
        "trajectory_range": traj_range,
    }


def analyze_problem(model, tokenizer, question: str, expected: str, category: str) -> Dict:
    """Analyze a problem's dimensional trajectory."""
    from mlx_lm import generate

    prompt = f"Question: {question}\n\nAnswer:"
    output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)

    # Get dimensional trajectory
    traj_data = get_dimensional_trajectory(model, tokenizer, prompt)

    # Try to check correctness (if expected is provided)
    is_correct = None
    if expected is not None:
        nums = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = nums[-1] if nums else ""
        is_correct = predicted == expected

    return {
        "category": category,
        "question": question[:100],
        "expected": expected,
        "is_correct": is_correct,
        "output": output[:200],
        **traj_data,
    }


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("ADVERSARIAL TRAJECTORY EXPERIMENT")
    logger.info("Testing: Do adversarial inputs have pathological trajectories?")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "adapter": adapter_path,
        "problems": [],
    }

    # Analyze each category
    categories = [
        ("normal", NORMAL_PROBLEMS),
        ("irrelevant_info", IRRELEVANT_INFO),
        ("contradictory", CONTRADICTORY_INFO),
        ("nonsense", NONSENSE_MATH),
    ]

    for cat_name, problems in categories:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"CATEGORY: {cat_name.upper()}")
        logger.info(f"{'=' * 50}")

        for i, (question, expected) in enumerate(problems):
            analysis = analyze_problem(model, tokenizer, question, expected, cat_name)
            results["problems"].append(analysis)

            status = ""
            if analysis["is_correct"] is not None:
                status = "OK" if analysis["is_correct"] else "WRONG"
            else:
                status = "N/A"

            logger.info(f"  [{i+1}/{len(problems)}] {status} | Peak L{analysis['peak_layer']} | "
                       f"Comp/φ: {analysis['compression_vs_phi']:.2f} | "
                       f"Var: {analysis['trajectory_variance']:.3f}")

    # Analysis
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY ANALYSIS BY CATEGORY")
    logger.info("=" * 70)

    def get_category_stats(category: str):
        cat_results = [r for r in results["problems"] if r["category"] == category]
        return {
            "n": len(cat_results),
            "compression_vs_phi": [r["compression_vs_phi"] for r in cat_results
                                   if not np.isnan(r["compression_vs_phi"])],
            "trajectory_variance": [r["trajectory_variance"] for r in cat_results
                                    if not np.isnan(r["trajectory_variance"])],
            "peak_layers": [r["peak_layer"] for r in cat_results],
            "initial_dims": [r["initial_dim"] for r in cat_results
                            if not np.isnan(r["initial_dim"])],
            "trajectory_range": [r["trajectory_range"] for r in cat_results
                                 if not np.isnan(r["trajectory_range"])],
        }

    stats = {cat: get_category_stats(cat) for cat in ["normal", "irrelevant_info", "contradictory", "nonsense"]}

    # Print summary
    logger.info(f"\n{'Category':<20} {'Comp/φ':<15} {'Traj Var':<15} {'Traj Range':<15} {'Peak Layer':<15}")
    logger.info("-" * 80)

    for cat, s in stats.items():
        comp = f"{np.mean(s['compression_vs_phi']):.2f} ± {np.std(s['compression_vs_phi']):.2f}" if s['compression_vs_phi'] else "N/A"
        var = f"{np.mean(s['trajectory_variance']):.3f} ± {np.std(s['trajectory_variance']):.3f}" if s['trajectory_variance'] else "N/A"
        rng = f"{np.mean(s['trajectory_range']):.1f} ± {np.std(s['trajectory_range']):.1f}" if s['trajectory_range'] else "N/A"
        peak = f"{np.mean(s['peak_layers']):.1f} ± {np.std(s['peak_layers']):.1f}" if s['peak_layers'] else "N/A"
        logger.info(f"{cat:<20} {comp:<15} {var:<15} {rng:<15} {peak:<15}")

    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS: Normal vs Adversarial")
    logger.info("=" * 70)

    test_results = []

    # Compare normal to each adversarial type
    normal_stats = stats["normal"]

    for adv_type in ["irrelevant_info", "contradictory", "nonsense"]:
        adv_stats = stats[adv_type]

        logger.info(f"\n--- Normal vs {adv_type} ---")

        # Trajectory variance comparison
        if normal_stats["trajectory_variance"] and adv_stats["trajectory_variance"]:
            t, p = ttest_ind(normal_stats["trajectory_variance"], adv_stats["trajectory_variance"])
            logger.info(f"Trajectory variance: t={t:.3f}, p={p:.4f}")
            if p < 0.05:
                direction = "higher" if np.mean(adv_stats["trajectory_variance"]) > np.mean(normal_stats["trajectory_variance"]) else "lower"
                logger.info(f"  ✓ Significant difference - {adv_type} has {direction} variance")
                test_results.append({"test": f"{adv_type}_variance", "sig": True, "direction": direction})
            else:
                test_results.append({"test": f"{adv_type}_variance", "sig": False})

        # Compression/φ comparison
        if normal_stats["compression_vs_phi"] and adv_stats["compression_vs_phi"]:
            t, p = ttest_ind(normal_stats["compression_vs_phi"], adv_stats["compression_vs_phi"])
            logger.info(f"Compression/φ: t={t:.3f}, p={p:.4f}")
            if p < 0.05:
                direction = "higher" if np.mean(adv_stats["compression_vs_phi"]) > np.mean(normal_stats["compression_vs_phi"]) else "lower"
                logger.info(f"  ✓ Significant difference - {adv_type} has {direction} compression/φ")
                test_results.append({"test": f"{adv_type}_compression", "sig": True, "direction": direction})
            else:
                test_results.append({"test": f"{adv_type}_compression", "sig": False})

    # Verdict
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS VERDICT")
    logger.info("=" * 70)

    sig_tests = sum(1 for t in test_results if t.get("sig", False))
    total_tests = len(test_results)

    if sig_tests > 0:
        logger.info(f"\n✓ PARTIALLY SUPPORTED: Adversarial inputs show detectable trajectory differences")
        logger.info(f"  {sig_tests}/{total_tests} tests showed significant differences")
        for t in test_results:
            if t.get("sig"):
                logger.info(f"  - {t['test']}: adversarial {t['direction']}")
    else:
        logger.info(f"\n✗ NOT SUPPORTED: No significant trajectory differences detected")
        logger.info(f"  Adversarial examples may use the same processing as normal inputs")

    results["stats"] = {k: {
        "compression_vs_phi_mean": np.mean(v["compression_vs_phi"]) if v["compression_vs_phi"] else None,
        "trajectory_variance_mean": np.mean(v["trajectory_variance"]) if v["trajectory_variance"] else None,
        "n": v["n"],
    } for k, v in stats.items()}
    results["tests"] = test_results

    # Save results
    output_path = Path("data/experiments/adversarial_trajectories.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
