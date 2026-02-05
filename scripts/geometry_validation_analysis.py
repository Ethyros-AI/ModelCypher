#!/usr/bin/env python3
"""Geometry Validation Analysis: Finding the geometric mechanism for reasoning quality.

This script analyzes the results from geometry_validation_experiment.py to determine
whether geometric metrics capture a deterministic relationship with reasoning quality.

The goal is NOT just correlation - we want to understand the MECHANISM:
- What geometric transformation distinguishes correct from incorrect?
- Is there a deterministic structure we can exploit?
- Can we intervene based on geometry?

Usage:
    poetry run python scripts/geometry_validation_analysis.py \
        --results results/geometry_validation/samples.jsonl \
        --output results/geometry_validation/analysis/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Loaded sample with metrics."""

    sample_id: str
    benchmark: str
    is_correct: bool
    expansion_ratio: float
    mean_intrinsic_dimension: float
    id_expansion_ratio: float
    mean_spectral_entropy: float
    mean_curvature: float
    smoothness: float
    directness: float
    entropy_trajectory: list[float]
    intrinsic_dimension_trajectory: list[float]
    spectral_entropy_trajectory: list[float]


def load_samples(path: Path) -> list[Sample]:
    """Load samples from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(
                Sample(
                    sample_id=data["sample_id"],
                    benchmark=data["benchmark"],
                    is_correct=data["is_correct"],
                    expansion_ratio=data.get("expansion_ratio", 0.0),
                    mean_intrinsic_dimension=data.get("mean_intrinsic_dimension", 0.0),
                    id_expansion_ratio=data.get("id_expansion_ratio", 0.0),
                    mean_spectral_entropy=data.get("mean_spectral_entropy", 0.0),
                    mean_curvature=data.get("mean_curvature", 0.0),
                    smoothness=data.get("smoothness", 0.0),
                    directness=data.get("directness", 0.0),
                    entropy_trajectory=data.get("entropy_trajectory", []),
                    intrinsic_dimension_trajectory=data.get(
                        "intrinsic_dimension_trajectory", []
                    ),
                    spectral_entropy_trajectory=data.get(
                        "spectral_entropy_trajectory", []
                    ),
                )
            )
    return samples


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute basic statistics."""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = variance**0.5
    sorted_vals = sorted(values)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "median": median,
    }


def compute_effect_size(correct_vals: list[float], incorrect_vals: list[float]) -> float:
    """Compute Cohen's d effect size."""
    if not correct_vals or not incorrect_vals:
        return 0.0

    c_stats = compute_statistics(correct_vals)
    i_stats = compute_statistics(incorrect_vals)

    # Pooled standard deviation
    nc, ni = c_stats["n"], i_stats["n"]
    if nc + ni < 3:
        return 0.0

    pooled_var = (
        (nc - 1) * c_stats["std"] ** 2 + (ni - 1) * i_stats["std"] ** 2
    ) / (nc + ni - 2)
    pooled_std = pooled_var**0.5

    if pooled_std < 1e-10:
        return 0.0

    return (c_stats["mean"] - i_stats["mean"]) / pooled_std


def compute_auroc(scores: list[float], labels: list[bool]) -> float:
    """Compute Area Under ROC Curve.

    This is the probability that a randomly chosen positive (correct) sample
    has a higher score than a randomly chosen negative (incorrect) sample.
    """
    if not scores or not labels:
        return 0.5

    # Pair scores with labels
    pairs = list(zip(scores, labels))

    positives = [s for s, l in pairs if l]
    negatives = [s for s, l in pairs if not l]

    if not positives or not negatives:
        return 0.5

    # Count concordant pairs
    concordant = 0
    tied = 0
    total = len(positives) * len(negatives)

    for p in positives:
        for n in negatives:
            if p > n:
                concordant += 1
            elif p == n:
                tied += 0.5

    return (concordant + tied) / total


def analyze_separability(
    correct_vals: list[float], incorrect_vals: list[float]
) -> dict[str, Any]:
    """Analyze whether correct/incorrect distributions are separable.

    This is the key question: is there a deterministic threshold that
    separates correct from incorrect? Or do the distributions overlap?
    """
    if not correct_vals or not incorrect_vals:
        return {"separable": False, "reason": "Missing data"}

    c_stats = compute_statistics(correct_vals)
    i_stats = compute_statistics(incorrect_vals)

    # Check for complete separation
    if c_stats["min"] > i_stats["max"]:
        return {
            "separable": True,
            "separation_type": "correct > incorrect",
            "gap": c_stats["min"] - i_stats["max"],
            "optimal_threshold": (c_stats["min"] + i_stats["max"]) / 2,
        }
    elif i_stats["min"] > c_stats["max"]:
        return {
            "separable": True,
            "separation_type": "incorrect > correct",
            "gap": i_stats["min"] - c_stats["max"],
            "optimal_threshold": (i_stats["min"] + c_stats["max"]) / 2,
        }

    # Compute overlap region
    overlap_start = max(c_stats["min"], i_stats["min"])
    overlap_end = min(c_stats["max"], i_stats["max"])
    overlap_width = overlap_end - overlap_start

    # Count samples in overlap
    c_in_overlap = sum(1 for v in correct_vals if overlap_start <= v <= overlap_end)
    i_in_overlap = sum(1 for v in incorrect_vals if overlap_start <= v <= overlap_end)

    return {
        "separable": False,
        "overlap_region": [overlap_start, overlap_end],
        "overlap_width": overlap_width,
        "correct_in_overlap": c_in_overlap,
        "incorrect_in_overlap": i_in_overlap,
        "overlap_fraction_correct": c_in_overlap / len(correct_vals),
        "overlap_fraction_incorrect": i_in_overlap / len(incorrect_vals),
    }


def find_optimal_threshold(
    scores: list[float], labels: list[bool]
) -> dict[str, float]:
    """Find threshold that maximizes Youden's J (sensitivity + specificity - 1)."""
    if not scores or not labels:
        return {"threshold": 0.0, "youden_j": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"threshold": 0.0, "youden_j": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    best_j = -1.0
    best_threshold = 0.0
    best_sens = 0.0
    best_spec = 0.0

    # Try each unique score as threshold
    thresholds = sorted(set(scores))
    for thresh in thresholds:
        # Predict positive if score > threshold
        tp = sum(1 for s, l in pairs if s > thresh and l)
        tn = sum(1 for s, l in pairs if s <= thresh and not l)

        sens = tp / n_pos
        spec = tn / n_neg
        j = sens + spec - 1

        if j > best_j:
            best_j = j
            best_threshold = thresh
            best_sens = sens
            best_spec = spec

    return {
        "threshold": best_threshold,
        "youden_j": best_j,
        "sensitivity": best_sens,
        "specificity": best_spec,
    }


def analyze_trajectory_shape(
    correct_trajs: list[list[float]], incorrect_trajs: list[list[float]]
) -> dict[str, Any]:
    """Analyze trajectory shape differences.

    This looks for structural differences in HOW values change across layers,
    not just the final values. This might reveal the geometric mechanism.
    """
    if not correct_trajs or not incorrect_trajs:
        return {"error": "No trajectories"}

    # Filter empty trajectories
    correct_trajs = [t for t in correct_trajs if t]
    incorrect_trajs = [t for t in incorrect_trajs if t]

    if not correct_trajs or not incorrect_trajs:
        return {"error": "No valid trajectories"}

    # Normalize trajectories to same length for comparison
    # Use the minimum length across all trajectories
    min_len = min(
        min(len(t) for t in correct_trajs), min(len(t) for t in incorrect_trajs)
    )

    if min_len < 2:
        return {"error": "Trajectories too short"}

    # Truncate all to same length
    c_trajs = [t[:min_len] for t in correct_trajs]
    i_trajs = [t[:min_len] for t in incorrect_trajs]

    # Average trajectory per group
    c_avg = [sum(t[i] for t in c_trajs) / len(c_trajs) for i in range(min_len)]
    i_avg = [sum(t[i] for t in i_trajs) / len(i_trajs) for i in range(min_len)]

    # Compute trajectory features
    def trajectory_features(traj: list[float]) -> dict[str, float]:
        n = len(traj)
        # Slope (linear fit)
        x_mean = (n - 1) / 2
        y_mean = sum(traj) / n
        numerator = sum((i - x_mean) * (traj[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0

        # First/last ratio
        ratio = traj[-1] / traj[0] if traj[0] != 0 else 0.0

        # Peak location (relative)
        peak_idx = traj.index(max(traj))
        peak_location = peak_idx / (n - 1) if n > 1 else 0.5

        # Monotonicity (fraction of increasing steps)
        increasing = sum(1 for i in range(1, n) if traj[i] >= traj[i - 1])
        monotonicity = increasing / (n - 1) if n > 1 else 0.5

        return {
            "slope": slope,
            "first_last_ratio": ratio,
            "peak_location": peak_location,
            "monotonicity": monotonicity,
        }

    c_features = trajectory_features(c_avg)
    i_features = trajectory_features(i_avg)

    return {
        "trajectory_length": min_len,
        "correct_avg_trajectory": c_avg,
        "incorrect_avg_trajectory": i_avg,
        "correct_features": c_features,
        "incorrect_features": i_features,
        "feature_differences": {
            k: c_features[k] - i_features[k] for k in c_features
        },
    }


def run_analysis(samples: list[Sample], output_dir: Path) -> dict[str, Any]:
    """Run full analysis."""
    correct = [s for s in samples if s.is_correct]
    incorrect = [s for s in samples if not s.is_correct]

    logger.info(f"Loaded {len(samples)} samples: {len(correct)} correct, {len(incorrect)} incorrect")

    # Metrics to analyze
    scalar_metrics = [
        ("expansion_ratio", lambda s: s.expansion_ratio),
        ("mean_intrinsic_dimension", lambda s: s.mean_intrinsic_dimension),
        ("id_expansion_ratio", lambda s: s.id_expansion_ratio),
        ("mean_spectral_entropy", lambda s: s.mean_spectral_entropy),
        ("mean_curvature", lambda s: s.mean_curvature),
        ("smoothness", lambda s: s.smoothness),
        ("directness", lambda s: s.directness),
    ]

    trajectory_metrics = [
        ("entropy_trajectory", lambda s: s.entropy_trajectory),
        ("intrinsic_dimension_trajectory", lambda s: s.intrinsic_dimension_trajectory),
        ("spectral_entropy_trajectory", lambda s: s.spectral_entropy_trajectory),
    ]

    results = {
        "n_samples": len(samples),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "scalar_metrics": {},
        "trajectory_metrics": {},
    }

    print("\n" + "=" * 70)
    print("GEOMETRIC MECHANISM ANALYSIS")
    print("=" * 70)

    # Analyze scalar metrics
    print("\n### SCALAR METRIC ANALYSIS ###\n")
    print(f"{'Metric':<30} {'Effect Size':>12} {'AUROC':>8} {'Separable':>10}")
    print("-" * 70)

    for name, getter in scalar_metrics:
        c_vals = [getter(s) for s in correct if getter(s) == getter(s)]  # Filter NaN
        i_vals = [getter(s) for s in incorrect if getter(s) == getter(s)]

        if not c_vals or not i_vals:
            continue

        all_vals = c_vals + i_vals
        all_labels = [True] * len(c_vals) + [False] * len(i_vals)

        effect_size = compute_effect_size(c_vals, i_vals)
        auroc = compute_auroc(all_vals, all_labels)
        separability = analyze_separability(c_vals, i_vals)
        threshold_info = find_optimal_threshold(all_vals, all_labels)

        results["scalar_metrics"][name] = {
            "correct_stats": compute_statistics(c_vals),
            "incorrect_stats": compute_statistics(i_vals),
            "effect_size": effect_size,
            "auroc": auroc,
            "separability": separability,
            "threshold_info": threshold_info,
        }

        sep_str = "YES" if separability.get("separable") else "NO"
        print(f"{name:<30} {effect_size:>12.3f} {auroc:>8.3f} {sep_str:>10}")

    # Interpret effect sizes
    print("\n### EFFECT SIZE INTERPRETATION ###")
    print("Cohen's d: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")
    print()

    for name, data in results["scalar_metrics"].items():
        d = data["effect_size"]
        auroc = data["auroc"]

        if abs(d) < 0.2:
            interp = "NEGLIGIBLE - no meaningful difference"
        elif abs(d) < 0.5:
            interp = "SMALL - weak signal"
        elif abs(d) < 0.8:
            interp = "MEDIUM - moderate signal"
        else:
            interp = "LARGE - strong signal"

        direction = "correct > incorrect" if d > 0 else "incorrect > correct"
        print(f"{name}: d={d:.3f} ({interp}), direction: {direction}")

        if auroc > 0.65:
            j = data["threshold_info"]["youden_j"]
            thresh = data["threshold_info"]["threshold"]
            print(f"  → AUROC {auroc:.3f} suggests predictive value at threshold {thresh:.4f} (J={j:.3f})")

    # Analyze trajectory shapes
    print("\n### TRAJECTORY SHAPE ANALYSIS ###")
    print("Looking for structural differences in HOW values change across layers...\n")

    for name, getter in trajectory_metrics:
        c_trajs = [getter(s) for s in correct if getter(s)]
        i_trajs = [getter(s) for s in incorrect if getter(s)]

        shape_analysis = analyze_trajectory_shape(c_trajs, i_trajs)
        results["trajectory_metrics"][name] = shape_analysis

        if "error" in shape_analysis:
            print(f"{name}: {shape_analysis['error']}")
            continue

        c_feat = shape_analysis["correct_features"]
        i_feat = shape_analysis["incorrect_features"]
        diff = shape_analysis["feature_differences"]

        print(f"{name}:")
        print(f"  Slope:         correct={c_feat['slope']:.4f}, incorrect={i_feat['slope']:.4f}, diff={diff['slope']:+.4f}")
        print(f"  First/Last:    correct={c_feat['first_last_ratio']:.4f}, incorrect={i_feat['first_last_ratio']:.4f}, diff={diff['first_last_ratio']:+.4f}")
        print(f"  Peak location: correct={c_feat['peak_location']:.4f}, incorrect={i_feat['peak_location']:.4f}, diff={diff['peak_location']:+.4f}")
        print(f"  Monotonicity:  correct={c_feat['monotonicity']:.4f}, incorrect={i_feat['monotonicity']:.4f}, diff={diff['monotonicity']:+.4f}")
        print()

    # Summary: Is there a mechanism?
    print("\n### MECHANISM HYPOTHESIS ###")
    print()

    # Find the strongest signal
    strongest_metric = None
    strongest_auroc = 0.5
    for name, data in results["scalar_metrics"].items():
        if data["auroc"] > strongest_auroc:
            strongest_auroc = data["auroc"]
            strongest_metric = name

    if strongest_auroc < 0.55:
        print("CONCLUSION: NO GEOMETRIC MECHANISM FOUND")
        print()
        print("All metrics have AUROC < 0.55, meaning they provide essentially no")
        print("predictive value for correctness. The geometric properties measured")
        print("(intrinsic dimension, spectral entropy, curvature, etc.) do not")
        print("distinguish correct from incorrect reasoning in this dataset.")
        print()
        print("Possible explanations:")
        print("1. These metrics capture computation PATTERN, not quality")
        print("2. Correct reasoning doesn't have a distinctive geometric signature")
        print("3. The metrics are too coarse - need finer-grained analysis")
    elif strongest_auroc < 0.65:
        print(f"CONCLUSION: WEAK SIGNAL DETECTED")
        print()
        print(f"Best metric: {strongest_metric} (AUROC = {strongest_auroc:.3f})")
        print()
        print("There is a weak statistical association, but not strong enough for")
        print("reliable prediction. The signal might be real but confounded by")
        print("other factors, or it might be noise.")
    else:
        print(f"CONCLUSION: POTENTIAL MECHANISM DETECTED")
        print()
        print(f"Best metric: {strongest_metric} (AUROC = {strongest_auroc:.3f})")
        print()

        data = results["scalar_metrics"][strongest_metric]
        c_mean = data["correct_stats"]["mean"]
        i_mean = data["incorrect_stats"]["mean"]
        direction = "higher" if c_mean > i_mean else "lower"

        print(f"Correct answers have {direction} {strongest_metric}:")
        print(f"  Correct mean:   {c_mean:.4f}")
        print(f"  Incorrect mean: {i_mean:.4f}")
        print()
        print("NEXT STEPS to validate mechanism:")
        print("1. Test on other models - is this model-specific or universal?")
        print("2. Test on other benchmarks - is this task-specific?")
        print("3. Investigate WHY this metric differs - what geometric transformation?")
        print("4. Build predictor and test on held-out data")

    print("=" * 70)

    # Save full results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "analysis_results.json", "w") as f:
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            else:
                return obj

        json.dump(make_serializable(results), f, indent=2)

    logger.info(f"Saved analysis to {output_dir / 'analysis_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze geometry validation results")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to samples.jsonl from experiment",
    )
    parser.add_argument(
        "--output",
        default="results/geometry_validation/analysis/",
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    samples = load_samples(Path(args.results))
    run_analysis(samples, Path(args.output))


if __name__ == "__main__":
    main()
