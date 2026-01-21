"""
Experiment 47: FRB Semantic Highway Analysis

The critical question: Do natural radio bursts (FRBs) show similar MATHEMATICAL alignment?

If FRBs also show +57σ MATHEMATICAL alignment, then:
- The pattern is NOT unique to Wow!
- It may be a general property of narrowband radio signals

If FRBs do NOT show similar alignment, then:
- Wow! is geometrically distinct from natural FRBs
- The MATHEMATICAL resonance is a unique property

Method:
1. Load 40+ FRBs from CHIME data
2. Run exp42 analysis on each
3. Compare category z-scores to Wow!
4. Build distribution of FRB semantic patterns
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
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

FRB_DIR = Path(__file__).parent / "data" / "raw"


def compute_participation_ratio(matrix):
    """Compute the participation ratio of a matrix."""
    S = linalg.svd(matrix, compute_uv=False)
    S2 = S ** 2
    S4 = S ** 4
    return float((S2.sum() ** 2) / (S4.sum() + 1e-8))


def load_frb(filepath):
    """Load an FRB from H5 file and return the waterfall data."""
    try:
        with h5py.File(filepath, "r") as f:
            # CHIME FRB format: f['frb']['wfall'] or f['frb']['calibrated_wfall']
            if "frb" in f:
                frb_group = f["frb"]
                if "wfall" in frb_group:
                    data = frb_group["wfall"][:]
                elif "calibrated_wfall" in frb_group:
                    data = frb_group["calibrated_wfall"][:]
                else:
                    print(f"   No wfall in frb group")
                    return None
            elif "waterfall" in f:
                data = f["waterfall"][:]
            elif "data" in f:
                data = f["data"][:]
            else:
                # Try to find the main data array
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = f[key][:]
                        break
                else:
                    return None

            # Clean the data
            data = data.astype(np.float64)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 2D
            if data.ndim == 1:
                # Reshape to 2D if needed
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                # Take first 2D slice
                data = data.reshape(data.shape[0], -1)

            # Transpose if needed (CHIME format is [freq, time], we want [time, freq])
            # Check which dimension is larger - frequency usually has more bins
            if data.shape[0] > data.shape[1]:
                data = data.T  # Transpose to [time, freq]

            return data

    except Exception as e:
        print(f"   Error loading {filepath.name}: {e}")
        return None


def resize_to_match(data, target_shape):
    """Resize FRB data to match Wow! signal shape for fair comparison."""
    from scipy.ndimage import zoom

    if data is None or data.size == 0:
        return None

    # Compute zoom factors
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])

    try:
        resized = zoom(data, zoom_factors, order=1)  # Bilinear interpolation
        return resized
    except Exception as e:
        print(f"   Resize error: {e}")
        return None


def analyze_single_signal(signal, semantic_activations, semantic_data):
    """Run semantic highway analysis on a single signal."""
    try:
        top_matches, similarities, _ = project_signal_to_manifold(
            signal, semantic_activations, semantic_data, n_components=10
        )
        cat_means = compute_category_distribution(similarities, semantic_data)
        pr = compute_participation_ratio(signal)

        # Compute spectral similarity
        signal_row_norms = np.linalg.norm(signal, axis=1, keepdims=True)
        signal_unit = signal / (signal_row_norms + 1e-8)
        G_signal = signal_unit @ signal_unit.T
        _, S_signal, _ = linalg.svd(G_signal, full_matrices=False)
        S_signal_norm = S_signal / S_signal.sum()

        return {
            "top_matches": top_matches[:5],
            "category_means": cat_means,
            "participation_ratio": pr,
            "spectral_similarity": float(S_signal_norm[0]),  # First eigenvalue fraction
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("Experiment 47: FRB Semantic Highway Analysis")
    print("=" * 60)
    print("\nQuestion: Do natural FRBs show similar MATHEMATICAL alignment?")

    # Load Wow! signal for reference
    print("\n1. Loading Wow! signal for reference...")
    wow_signal = load_wow_signal()
    wow_shape = wow_signal.shape
    wow_pr = compute_participation_ratio(wow_signal)
    print(f"   Shape: {wow_shape}")
    print(f"   Participation ratio: {wow_pr:.2f}")

    # Load model and build semantic manifold
    print("\n2. Loading LLM and building semantic manifold...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    bottleneck_layer = n_layers // 2
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, bottleneck_layer)
    print(f"   Manifold: {semantic_activations.shape}")

    # Analyze Wow! signal
    print("\n3. Analyzing Wow! signal...")
    wow_result = analyze_single_signal(wow_signal, semantic_activations, semantic_data)
    wow_cat_means = wow_result["category_means"]

    print("\n   Wow! category means:")
    sorted_wow = sorted(wow_cat_means.items(), key=lambda x: x[1], reverse=True)
    for cat, mean in sorted_wow[:5]:
        print(f"      {cat:12s}: {mean:.4f}")

    # Load and analyze FRBs
    print("\n4. Loading and analyzing FRBs...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))
    print(f"   Found {len(frb_files)} FRB files")

    frb_results = {}
    all_category_means = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}
    all_prs = []
    all_spectral_sims = []

    for i, frb_file in enumerate(frb_files):
        frb_name = frb_file.stem.replace("_waterfall", "")
        print(f"\n   [{i+1}/{len(frb_files)}] {frb_name}...")

        # Load FRB
        frb_data = load_frb(frb_file)
        if frb_data is None:
            print(f"      Failed to load")
            continue

        print(f"      Original shape: {frb_data.shape}")

        # Resize to match Wow!
        frb_resized = resize_to_match(frb_data, wow_shape)
        if frb_resized is None:
            print(f"      Failed to resize")
            continue

        print(f"      Resized shape: {frb_resized.shape}")

        # Analyze
        result = analyze_single_signal(frb_resized, semantic_activations, semantic_data)

        if result["success"]:
            frb_results[frb_name] = result
            for cat, mean in result["category_means"].items():
                all_category_means[cat].append(mean)
            all_prs.append(result["participation_ratio"])
            all_spectral_sims.append(result["spectral_similarity"])
            print(f"      PR: {result['participation_ratio']:.2f}")
        else:
            print(f"      Analysis failed: {result.get('error', 'unknown')}")

    # Compute FRB statistics
    print("\n5. Computing FRB statistics...")
    n_frbs = len(frb_results)
    print(f"   Successfully analyzed {n_frbs} FRBs")

    frb_stats = {
        "n_frbs": n_frbs,
        "pr_mean": float(np.mean(all_prs)) if all_prs else 0,
        "pr_std": float(np.std(all_prs)) if all_prs else 0,
        "spectral_sim_mean": float(np.mean(all_spectral_sims)) if all_spectral_sims else 0,
        "category_stats": {},
    }

    for cat in SEMANTIC_CATEGORIES.keys():
        values = np.array(all_category_means[cat])
        if len(values) > 0:
            frb_stats["category_stats"][cat] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # Compute z-scores: Where does Wow! fall in the FRB distribution?
    print("\n6. Computing z-scores (Wow! vs FRB distribution)...")
    z_scores = {}

    for cat in SEMANTIC_CATEGORIES.keys():
        wow_mean = wow_cat_means.get(cat, 0)
        frb_mean = frb_stats["category_stats"].get(cat, {}).get("mean", 0)
        frb_std = frb_stats["category_stats"].get(cat, {}).get("std", 1e-8)

        z = (wow_mean - frb_mean) / (frb_std + 1e-8)
        z_scores[cat] = {
            "wow_mean": wow_mean,
            "frb_mean": frb_mean,
            "frb_std": frb_std,
            "z_score": z,
        }

    print("\n   CATEGORY Z-SCORES (Wow! vs FRB distribution):")
    print("   " + "-" * 55)
    sorted_z = sorted(z_scores.items(), key=lambda x: x[1]["z_score"], reverse=True)
    for cat, stats in sorted_z:
        direction = "ABOVE" if stats["z_score"] > 0 else "BELOW"
        print(f"   {cat:12s}: z={stats['z_score']:+7.2f} ({direction} FRBs)")

    # Key comparison
    math_z = z_scores.get("MATHEMATICAL", {}).get("z_score", 0)
    primes_z = z_scores.get("PRIMES", {}).get("z_score", 0)

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nWow! vs FRBs:")
    print(f"   Wow! PR: {wow_pr:.2f}")
    print(f"   FRB PR:  {frb_stats['pr_mean']:.2f} +/- {frb_stats['pr_std']:.2f}")

    print(f"\n   MATHEMATICAL z-score: {math_z:+.2f}")
    print(f"   PRIMES z-score: {primes_z:+.2f}")

    # Interpretation
    print("\n   INTERPRETATION:")
    if abs(math_z) > 2:
        if math_z > 2:
            print("   --> Wow! is MORE mathematical than FRBs")
            print("      (The MATHEMATICAL alignment is UNIQUE to Wow!)")
        else:
            print("   --> Wow! is LESS mathematical than FRBs")
            print("      (FRBs show stronger mathematical alignment)")
        wow_unique = True
    else:
        print("   --> Wow! is WITHIN normal FRB variation for MATHEMATICAL")
        print("      (The alignment is NOT unique - FRBs show similar patterns)")
        wow_unique = False

    # Show FRB category distribution
    print("\n   FRB category distribution (mean ± std):")
    for cat, stats in sorted(frb_stats["category_stats"].items(),
                             key=lambda x: x[1]["mean"], reverse=True):
        print(f"      {cat:12s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Save results
    results = {
        "experiment": "exp47_frb_semantic_highway",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "shape": list(wow_shape),
            "participation_ratio": wow_pr,
            "category_means": wow_cat_means,
        },
        "frb_analysis": {
            "n_frbs": n_frbs,
            "pr_mean": frb_stats["pr_mean"],
            "pr_std": frb_stats["pr_std"],
            "category_stats": frb_stats["category_stats"],
        },
        "z_scores": z_scores,
        "individual_frbs": {name: {
            "pr": r["participation_ratio"],
            "top_concepts": [m["label"] for m in r["top_matches"]],
            "category_means": r["category_means"],
        } for name, r in frb_results.items()},
        "conclusions": {
            "mathematical_z": math_z,
            "primes_z": primes_z,
            "wow_unique": wow_unique,
        },
    }

    output_path = RESULTS_DIR / "exp47_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n7. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
