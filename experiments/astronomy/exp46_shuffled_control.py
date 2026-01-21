"""
Experiment 46: Shuffled Wow! Control

The critical question: Does the SPECIFIC structure of Wow! matter, or just its rank?

Method:
1. Take Wow! signal and shuffle it in different ways
2. Each shuffle preserves different properties:
   - Row shuffle: Preserves frequency profiles, destroys temporal order
   - Column shuffle: Preserves temporal patterns, destroys frequency order
   - Element shuffle: Destroys all structure, preserves only value distribution
3. Run exp42 analysis on each shuffled version
4. If shuffling destroys the MATHEMATICAL alignment, the structure matters

This is the key test: If row/column shuffles preserve the alignment but element
shuffle destroys it, then it's the eigenstructure that matters (which is what
we hypothesize an intelligent signal would encode).
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


def shuffle_rows(matrix):
    """Shuffle rows (time slices). Preserves frequency profiles."""
    idx = np.random.permutation(matrix.shape[0])
    return matrix[idx, :]


def shuffle_cols(matrix):
    """Shuffle columns (frequency bins). Preserves temporal patterns."""
    idx = np.random.permutation(matrix.shape[1])
    return matrix[:, idx]


def shuffle_elements(matrix):
    """Shuffle all elements. Destroys all structure."""
    flat = matrix.flatten()
    np.random.shuffle(flat)
    return flat.reshape(matrix.shape)


def shuffle_within_rows(matrix):
    """Shuffle elements within each row. Preserves row norms but destroys frequency structure."""
    result = matrix.copy()
    for i in range(result.shape[0]):
        np.random.shuffle(result[i, :])
    return result


def shuffle_within_cols(matrix):
    """Shuffle elements within each column. Preserves column norms but destroys temporal structure."""
    result = matrix.copy()
    for j in range(result.shape[1]):
        np.random.shuffle(result[:, j])
    return result


SHUFFLE_METHODS = {
    "row_permute": {
        "func": shuffle_rows,
        "description": "Permute row order (time slices)",
        "preserves": "Frequency profiles, Gram structure",
    },
    "col_permute": {
        "func": shuffle_cols,
        "description": "Permute column order (frequency bins)",
        "preserves": "Temporal patterns, Gram structure",
    },
    "element_shuffle": {
        "func": shuffle_elements,
        "description": "Shuffle all elements randomly",
        "preserves": "Value distribution only",
    },
    "within_row": {
        "func": shuffle_within_rows,
        "description": "Shuffle within each row",
        "preserves": "Row norms",
    },
    "within_col": {
        "func": shuffle_within_cols,
        "description": "Shuffle within each column",
        "preserves": "Column norms",
    },
}


def analyze_single_matrix(matrix, semantic_activations, semantic_data):
    """Run exp42-style analysis on a single matrix."""
    try:
        top_matches, similarities, _ = project_signal_to_manifold(
            matrix, semantic_activations, semantic_data, n_components=10
        )
        cat_means = compute_category_distribution(similarities, semantic_data)
        pr = compute_participation_ratio(matrix)
        return {
            "top_matches": top_matches[:5],
            "category_means": cat_means,
            "participation_ratio": pr,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_shuffle_trials(signal, shuffle_func, n_trials, semantic_activations, semantic_data):
    """Run multiple trials of a shuffle method."""
    all_category_means = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}
    all_prs = []
    all_top_concepts = []

    for i in range(n_trials):
        shuffled = shuffle_func(signal)
        result = analyze_single_matrix(shuffled, semantic_activations, semantic_data)

        if result["success"]:
            for cat, mean in result["category_means"].items():
                all_category_means[cat].append(mean)
            all_prs.append(result["participation_ratio"])
            top_labels = [m["label"] for m in result["top_matches"]]
            all_top_concepts.append(top_labels)

    # Compute statistics
    stats = {
        "n_trials": n_trials,
        "n_success": len(all_prs),
        "category_stats": {},
        "pr_mean": float(np.mean(all_prs)) if all_prs else 0,
        "pr_std": float(np.std(all_prs)) if all_prs else 0,
    }

    for cat in all_category_means:
        values = np.array(all_category_means[cat])
        if len(values) > 0:
            stats["category_stats"][cat] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

    # Top concept frequency
    concept_counts = {}
    for top_list in all_top_concepts:
        for label in top_list:
            concept_counts[label] = concept_counts.get(label, 0) + 1
    stats["top_concepts"] = dict(sorted(
        concept_counts.items(), key=lambda x: x[1], reverse=True
    )[:10])

    return stats


def main():
    print("=" * 60)
    print("Experiment 46: Shuffled Wow! Control")
    print("=" * 60)
    print("\nQuestion: Does shuffling destroy the MATHEMATICAL alignment?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
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

    # Analyze original Wow! signal
    print("\n3. Analyzing original Wow! signal...")
    wow_result = analyze_single_matrix(signal, semantic_activations, semantic_data)
    wow_cat_means = wow_result["category_means"]

    print("\n   Original Wow! category means:")
    sorted_wow = sorted(wow_cat_means.items(), key=lambda x: x[1], reverse=True)
    for cat, mean in sorted_wow:
        print(f"      {cat:12s}: {mean:.4f}")

    # Run each shuffle method
    print("\n4. Running shuffle experiments...")
    n_trials = 50  # Per shuffle method

    all_results = {"original": wow_result}

    for method_name, method_info in SHUFFLE_METHODS.items():
        print(f"\n   Testing: {method_name}")
        print(f"      {method_info['description']}")
        print(f"      Preserves: {method_info['preserves']}")

        stats = run_shuffle_trials(
            signal,
            method_info["func"],
            n_trials,
            semantic_activations,
            semantic_data,
        )

        all_results[method_name] = {
            "description": method_info["description"],
            "preserves": method_info["preserves"],
            "stats": stats,
        }

        print(f"      PR after shuffle: {stats['pr_mean']:.2f} +/- {stats['pr_std']:.2f}")

    # Compare: Compute z-scores for each shuffle method vs original
    print("\n5. Computing z-scores (original vs shuffled)...")
    print("\n   " + "=" * 70)
    print(f"   {'Shuffle Method':<20} | {'MATHEMATICAL z':>15} | {'PRIMES z':>12} | {'PR':>8}")
    print("   " + "-" * 70)

    comparison = {}
    for method_name, result in all_results.items():
        if method_name == "original":
            continue

        stats = result["stats"]["category_stats"]
        math_mean = stats.get("MATHEMATICAL", {}).get("mean", 0)
        math_std = stats.get("MATHEMATICAL", {}).get("std", 1e-8)
        primes_mean = stats.get("PRIMES", {}).get("mean", 0)
        primes_std = stats.get("PRIMES", {}).get("std", 1e-8)

        wow_math = wow_cat_means.get("MATHEMATICAL", 0)
        wow_primes = wow_cat_means.get("PRIMES", 0)

        math_z = (wow_math - math_mean) / (math_std + 1e-8)
        primes_z = (wow_primes - primes_mean) / (primes_std + 1e-8)

        comparison[method_name] = {
            "mathematical_z": math_z,
            "primes_z": primes_z,
            "pr_after": result["stats"]["pr_mean"],
        }

        print(f"   {method_name:<20} | {math_z:>+15.2f} | {primes_z:>+12.2f} | {result['stats']['pr_mean']:>8.2f}")

    print("   " + "=" * 70)

    # Key analysis
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Which shuffles preserve the signal?
    row_z = comparison.get("row_permute", {}).get("mathematical_z", 0)
    col_z = comparison.get("col_permute", {}).get("mathematical_z", 0)
    elem_z = comparison.get("element_shuffle", {}).get("mathematical_z", 0)
    within_row_z = comparison.get("within_row", {}).get("mathematical_z", 0)
    within_col_z = comparison.get("within_col", {}).get("mathematical_z", 0)

    print(f"\nRow permutation effect: z={row_z:+.2f}")
    print(f"Col permutation effect: z={col_z:+.2f}")
    print(f"Element shuffle effect: z={elem_z:+.2f}")
    print(f"Within-row shuffle effect: z={within_row_z:+.2f}")
    print(f"Within-col shuffle effect: z={within_col_z:+.2f}")

    print("\n   INTERPRETATION:")

    # Row/col permutation should preserve Gram structure
    if abs(row_z) < 3 and abs(col_z) < 3:
        print("   --> Row/col permutation has NO EFFECT (expected - Gram invariant)")
        gram_invariant = True
    else:
        print("   --> Row/col permutation HAS EFFECT (unexpected)")
        gram_invariant = False

    # Element shuffle should destroy structure
    if abs(elem_z) > 3:
        print("   --> Element shuffle DESTROYS the alignment (structure matters!)")
        structure_matters = True
    else:
        print("   --> Element shuffle has NO EFFECT (value distribution is enough)")
        structure_matters = False

    # Within-row/col shuffles test finer structure
    if abs(within_row_z) > 3 or abs(within_col_z) > 3:
        print("   --> Within-row/col shuffle affects alignment (fine structure matters)")
        fine_structure = True
    else:
        print("   --> Within-row/col shuffle has no effect")
        fine_structure = False

    # Final verdict
    print("\n   VERDICT:")
    if structure_matters and gram_invariant:
        print("   The MATHEMATICAL alignment depends on the signal's EIGENSTRUCTURE,")
        print("   not just its value distribution or rank. This is consistent with")
        print("   intentional encoding in the invariant geometric structure.")
    elif not structure_matters:
        print("   The MATHEMATICAL alignment is explained by value distribution alone.")
        print("   Any matrix with similar values would show similar alignment.")
    else:
        print("   Inconclusive - further analysis needed.")

    # Save results
    results = {
        "experiment": "exp46_shuffled_control",
        "timestamp": datetime.now().isoformat(),
        "original_wow": {
            "shape": list(signal.shape),
            "participation_ratio": wow_pr,
            "category_means": wow_cat_means,
        },
        "shuffle_methods": {
            name: {
                "description": SHUFFLE_METHODS[name]["description"],
                "preserves": SHUFFLE_METHODS[name]["preserves"],
                "stats": result["stats"] if "stats" in result else result,
            }
            for name, result in all_results.items()
            if name != "original"
        },
        "comparison": comparison,
        "conclusions": {
            "gram_invariant": gram_invariant,
            "structure_matters": structure_matters,
            "fine_structure_matters": fine_structure,
        },
    }

    output_path = RESULTS_DIR / "exp46_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
