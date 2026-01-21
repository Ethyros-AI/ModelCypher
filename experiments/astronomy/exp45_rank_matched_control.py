"""
Experiment 45: Rank-Matched Random Control

The critical question: Is the MATHEMATICAL +57σ alignment an artifact of low-rank structure?

Method:
1. Generate 100 random matrices with same shape (82×50) and same participation ratio (~2.6)
2. Run exp42 analysis on each
3. Compute z-scores for each category
4. If random matrices also show MATHEMATICAL +50σ, it's an artifact

The participation ratio (PR) measures effective rank:
PR = (sum(eigenvalues)^2) / sum(eigenvalues^4)

Wow! signal has PR ≈ 2.6, meaning ~3 dominant dimensions.
Pure random has PR ≈ min(rows, cols), much higher.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import (
    SEMANTIC_CATEGORIES,
    load_wow_signal,
    load_model,
    build_semantic_manifold,
    project_signal_to_manifold,
    compute_category_distribution,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_participation_ratio(matrix):
    """Compute the participation ratio of a matrix."""
    S = linalg.svd(matrix, compute_uv=False)
    S2 = S ** 2
    S4 = S ** 4
    return float((S2.sum() ** 2) / (S4.sum() + 1e-8))


def generate_rank_matched_matrix(shape, target_pr, max_iter=100):
    """
    Generate a random matrix with specified shape and participation ratio.

    Uses a low-rank + noise construction:
    M = U @ S @ V.T where S has controlled spectrum

    To get PR ≈ k:
    - Make k singular values large
    - Make the rest small (but not zero to add some noise)
    """
    rows, cols = shape
    min_dim = min(rows, cols)

    # Target effective rank from participation ratio
    k = max(1, int(round(target_pr)))

    for _ in range(max_iter):
        # Construct spectrum: k dominant + decaying tail
        spectrum = np.zeros(min_dim)

        # Dominant components
        spectrum[:k] = np.random.uniform(0.8, 1.0, k)
        spectrum[:k] = np.sort(spectrum[:k])[::-1]  # Decreasing

        # Decaying tail (to match the noise floor)
        if k < min_dim:
            tail_strength = 0.1 * np.random.uniform(0.5, 1.5)  # Small but non-zero
            spectrum[k:] = tail_strength * np.exp(-np.arange(min_dim - k) / 5)

        # Normalize so total variance is similar to Wow!
        spectrum = spectrum / spectrum.sum() * min_dim

        # Generate random orthogonal matrices with correct shapes
        # U: (rows, min_dim) and V: (cols, min_dim)
        U, _ = linalg.qr(np.random.randn(rows, rows))
        V, _ = linalg.qr(np.random.randn(cols, cols))

        # Take only the first min_dim columns
        U = U[:, :min_dim]  # (rows, min_dim)
        V = V[:, :min_dim]  # (cols, min_dim)

        # Construct matrix: U @ diag(spectrum) @ V.T
        # U: (rows, min_dim), diag(spectrum): (min_dim, min_dim), V.T: (min_dim, cols)
        matrix = U @ np.diag(spectrum) @ V.T  # (rows, cols)

        # Add small noise for numerical stability
        matrix += np.random.randn(*shape) * 0.01

        # Check participation ratio
        pr = compute_participation_ratio(matrix)

        # Accept if within tolerance
        if abs(pr - target_pr) < 0.5:
            return matrix, pr

    # If we couldn't match exactly, return last attempt
    return matrix, pr


def analyze_single_matrix(matrix, semantic_activations, semantic_data, verbose=False):
    """Run exp42-style analysis on a single matrix."""
    try:
        top_matches, similarities, _ = project_signal_to_manifold(
            matrix, semantic_activations, semantic_data, n_components=10
        )
        cat_means = compute_category_distribution(similarities, semantic_data)
        return {
            "top_matches": top_matches[:5],
            "category_means": cat_means,
            "success": True,
        }
    except Exception as e:
        if verbose:
            print(f"      Error: {e}")
        return {"success": False, "error": str(e)}


def run_control_experiment(n_trials, target_pr, signal_shape, semantic_activations, semantic_data):
    """Run the control experiment with rank-matched random matrices."""
    print(f"\n   Generating {n_trials} rank-matched random matrices (target PR={target_pr:.2f})...")

    all_category_means = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}
    all_prs = []
    all_top_concepts = []

    for i in range(n_trials):
        if (i + 1) % 10 == 0:
            print(f"      Trial {i+1}/{n_trials}...")

        # Generate rank-matched matrix
        matrix, actual_pr = generate_rank_matched_matrix(signal_shape, target_pr)
        all_prs.append(actual_pr)

        # Analyze
        result = analyze_single_matrix(matrix, semantic_activations, semantic_data)

        if result["success"]:
            for cat, mean in result["category_means"].items():
                all_category_means[cat].append(mean)

            # Track top concepts
            top_labels = [m["label"] for m in result["top_matches"]]
            all_top_concepts.append(top_labels)

    # Compute statistics
    stats = {
        "n_trials": n_trials,
        "target_pr": target_pr,
        "actual_pr_mean": float(np.mean(all_prs)),
        "actual_pr_std": float(np.std(all_prs)),
        "category_stats": {},
        "top_concept_frequency": {},
    }

    for cat in all_category_means:
        values = np.array(all_category_means[cat])
        if len(values) > 0:
            stats["category_stats"][cat] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # Count top concept frequency
    concept_counts = {}
    for top_list in all_top_concepts:
        for label in top_list:
            concept_counts[label] = concept_counts.get(label, 0) + 1
    stats["top_concept_frequency"] = dict(sorted(
        concept_counts.items(), key=lambda x: x[1], reverse=True
    )[:20])

    return stats


def main():
    print("=" * 60)
    print("Experiment 45: Rank-Matched Random Control")
    print("=" * 60)
    print("\nQuestion: Is MATHEMATICAL +57σ an artifact of low-rank structure?")

    # Load Wow! signal to get target properties
    print("\n1. Loading Wow! signal for reference...")
    signal = load_wow_signal()
    wow_pr = compute_participation_ratio(signal)
    print(f"   Shape: {signal.shape}")
    print(f"   Participation ratio: {wow_pr:.2f}")

    # Load model and build semantic manifold
    print("\n2. Loading LLM and building semantic manifold...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    bottleneck_layer = n_layers // 2
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, bottleneck_layer)
    print(f"   Manifold: {semantic_activations.shape}")

    # Run Wow! signal analysis for comparison
    print("\n3. Analyzing Wow! signal (baseline)...")
    top_matches, similarities, _ = project_signal_to_manifold(
        signal, semantic_activations, semantic_data, n_components=10
    )
    wow_cat_means = compute_category_distribution(similarities, semantic_data)

    print("\n   Wow! category means:")
    sorted_wow = sorted(wow_cat_means.items(), key=lambda x: x[1], reverse=True)
    for cat, mean in sorted_wow:
        print(f"      {cat:12s}: {mean:.4f}")

    # Run control experiment
    print("\n4. Running rank-matched random control...")
    control_stats = run_control_experiment(
        n_trials=100,
        target_pr=wow_pr,
        signal_shape=signal.shape,
        semantic_activations=semantic_activations,
        semantic_data=semantic_data,
    )

    print(f"\n   Random matrices PR: {control_stats['actual_pr_mean']:.2f} +/- {control_stats['actual_pr_std']:.2f}")

    # Compute z-scores: How far is Wow! from the rank-matched random distribution?
    print("\n5. Computing z-scores (Wow! vs rank-matched random)...")
    z_scores = {}

    for cat in SEMANTIC_CATEGORIES.keys():
        wow_mean = wow_cat_means.get(cat, 0)
        ctrl_mean = control_stats["category_stats"].get(cat, {}).get("mean", 0)
        ctrl_std = control_stats["category_stats"].get(cat, {}).get("std", 1e-8)

        z = (wow_mean - ctrl_mean) / (ctrl_std + 1e-8)
        z_scores[cat] = {
            "wow_mean": wow_mean,
            "control_mean": ctrl_mean,
            "control_std": ctrl_std,
            "z_score": z,
        }

    print("\n   CATEGORY Z-SCORES (Wow! vs rank-matched random):")
    print("   " + "-" * 55)
    sorted_z = sorted(z_scores.items(), key=lambda x: x[1]["z_score"], reverse=True)
    for cat, stats in sorted_z:
        direction = "ABOVE" if stats["z_score"] > 0 else "BELOW"
        print(f"   {cat:12s}: z={stats['z_score']:+7.2f} ({direction} control)")

    # Key comparison: Is MATHEMATICAL still special?
    math_z = z_scores.get("MATHEMATICAL", {}).get("z_score", 0)
    primes_z = z_scores.get("PRIMES", {}).get("z_score", 0)

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nMATHEMATICAL z-score: {math_z:+.2f}")
    print(f"PRIMES z-score: {primes_z:+.2f}")

    # Most common concepts in random matrices
    print("\n   Most frequent top concepts in random matrices:")
    for concept, count in list(control_stats["top_concept_frequency"].items())[:10]:
        print(f"      {concept}: {count}/{control_stats['n_trials']} trials")

    # Interpretation
    print("\n   INTERPRETATION:")
    if abs(math_z) < 3:
        print("   --> MATHEMATICAL alignment is EXPLAINED by low-rank structure")
        print("      (random matrices with same PR show similar alignment)")
        artifact = True
    elif math_z > 3:
        print("   --> MATHEMATICAL alignment is ABOVE what low-rank explains")
        print("      (Wow! is more mathematical than rank-matched random)")
        artifact = False
    else:
        print("   --> MATHEMATICAL alignment is BELOW what low-rank explains")
        print("      (Wow! is less mathematical than rank-matched random)")
        artifact = True

    # Save results
    results = {
        "experiment": "exp45_rank_matched_control",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "shape": list(signal.shape),
            "participation_ratio": wow_pr,
            "category_means": wow_cat_means,
        },
        "control": {
            "n_trials": control_stats["n_trials"],
            "target_pr": control_stats["target_pr"],
            "actual_pr_mean": control_stats["actual_pr_mean"],
            "actual_pr_std": control_stats["actual_pr_std"],
            "category_stats": control_stats["category_stats"],
            "top_concept_frequency": control_stats["top_concept_frequency"],
        },
        "z_scores": z_scores,
        "conclusion": {
            "mathematical_z": math_z,
            "primes_z": primes_z,
            "is_artifact": artifact,
        },
    }

    output_path = RESULTS_DIR / "exp45_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
