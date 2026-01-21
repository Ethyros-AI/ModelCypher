"""
Experiment 51: Layer Sweep Analysis

The question: WHERE in the model does the MATHEMATICAL alignment appear?

If the alignment is strongest in middle layers (the "semantic highway"):
- It's consistent with invariant geometric structure
- Middle layers encode abstract concepts, not surface features

If the alignment is strongest in early/late layers:
- It might be an artifact of input/output processing
- Not a deep semantic property

Method:
1. Run exp42-style analysis at EVERY layer (1 to n_layers)
2. Track category z-scores vs layer depth
3. Find where MATHEMATICAL peaks and where PRIMES troughs
4. Compare to the semantic highway hypothesis

The semantic highway hypothesis predicts:
- Middle layers (bottleneck) should show strongest semantic alignment
- Early layers: surface features (low semantic content)
- Late layers: task-specific output (language-specific)
- Middle layers: invariant semantic structure
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
    get_layer_activation,
    project_signal_to_manifold,
    compute_category_distribution,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_semantic_manifold_at_layer(model, tokenizer, layer_idx):
    """Build the semantic manifold at a specific layer."""
    all_data = []
    all_activations = []

    for category, info in SEMANTIC_CATEGORIES.items():
        for label, probe in info["probes"]:
            act = get_layer_activation(model, tokenizer, probe, layer_idx)
            if act is not None:
                all_data.append({
                    "category": category,
                    "label": label,
                    "probe": probe,
                })
                all_activations.append(act)

    activations = np.stack(all_activations)
    return all_data, activations


def analyze_at_layer(signal, model, tokenizer, layer_idx):
    """Run semantic highway analysis at a specific layer."""
    try:
        # Build manifold at this layer
        semantic_data, semantic_activations = build_semantic_manifold_at_layer(
            model, tokenizer, layer_idx
        )

        # Project signal
        top_matches, similarities, _ = project_signal_to_manifold(
            signal, semantic_activations, semantic_data, n_components=10
        )

        # Compute category distribution
        cat_means = compute_category_distribution(similarities, semantic_data)

        # Compute spectral similarity
        signal_row_norms = np.linalg.norm(signal, axis=1, keepdims=True)
        signal_unit = signal / (signal_row_norms + 1e-8)
        G_signal = signal_unit @ signal_unit.T
        _, S_signal, _ = linalg.svd(G_signal, full_matrices=False)
        spectral_sim = float(S_signal[0] / S_signal.sum())

        return {
            "layer": layer_idx,
            "top_matches": top_matches[:5],
            "category_means": cat_means,
            "spectral_similarity": spectral_sim,
            "success": True,
        }

    except Exception as e:
        return {"layer": layer_idx, "success": False, "error": str(e)}


def run_random_baseline(signal, model, tokenizer, layer_idx, n_trials=20):
    """Run analysis on random matrices to establish baseline at this layer."""
    shape = signal.shape
    all_category_means = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}

    # Build manifold once
    semantic_data, semantic_activations = build_semantic_manifold_at_layer(
        model, tokenizer, layer_idx
    )

    for _ in range(n_trials):
        # Generate random matrix
        rand_matrix = np.random.randn(*shape)

        try:
            _, similarities, _ = project_signal_to_manifold(
                rand_matrix, semantic_activations, semantic_data, n_components=10
            )
            cat_means = compute_category_distribution(similarities, semantic_data)

            for cat, mean in cat_means.items():
                all_category_means[cat].append(mean)
        except Exception:
            continue

    # Compute baseline statistics
    baseline = {}
    for cat in SEMANTIC_CATEGORIES.keys():
        values = np.array(all_category_means[cat])
        if len(values) > 0:
            baseline[cat] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        else:
            baseline[cat] = {"mean": 0, "std": 1}

    return baseline


def main():
    print("=" * 60)
    print("Experiment 51: Layer Sweep Analysis")
    print("=" * 60)
    print("\nQuestion: WHERE in the model is the MATHEMATICAL alignment strongest?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape}")

    # Load model
    print("\n2. Loading LLM...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    print(f"   Model: SmolLM-135M, {n_layers} layers")

    # Analyze at each layer
    print(f"\n3. Analyzing at all {n_layers} layers...")

    layer_results = []
    category_by_layer = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}
    spectral_by_layer = []

    for layer_idx in range(n_layers):
        print(f"\n   Layer {layer_idx + 1}/{n_layers}...")

        # Run analysis
        result = analyze_at_layer(signal, model, tokenizer, layer_idx)

        if result["success"]:
            layer_results.append(result)

            # Track category means
            for cat, mean in result["category_means"].items():
                category_by_layer[cat].append(mean)

            # Track spectral similarity
            spectral_by_layer.append(result["spectral_similarity"])

            # Show top match
            top = result["top_matches"][0]
            print(f"      Top match: {top['label']} ({top['category']}) = {top['similarity']:.4f}")
            print(f"      Spectral: {result['spectral_similarity']:.4f}")
            print(f"      MATHEMATICAL: {result['category_means'].get('MATHEMATICAL', 0):.4f}")
        else:
            print(f"      FAILED: {result.get('error', 'unknown')}")

    # Find peaks
    print("\n4. Finding peaks...")

    # MATHEMATICAL peak
    math_values = category_by_layer.get("MATHEMATICAL", [])
    if math_values:
        math_peak_layer = int(np.argmax(math_values))
        math_peak_value = float(max(math_values))
        print(f"   MATHEMATICAL peak: layer {math_peak_layer + 1} (value={math_peak_value:.4f})")

    # PRIMES trough
    primes_values = category_by_layer.get("PRIMES", [])
    if primes_values:
        primes_trough_layer = int(np.argmin(primes_values))
        primes_trough_value = float(min(primes_values))
        print(f"   PRIMES trough: layer {primes_trough_layer + 1} (value={primes_trough_value:.4f})")

    # Spectral peak
    if spectral_by_layer:
        spectral_peak_layer = int(np.argmax(spectral_by_layer))
        spectral_peak_value = float(max(spectral_by_layer))
        print(f"   Spectral peak: layer {spectral_peak_layer + 1} (value={spectral_peak_value:.4f})")

    # Bottleneck layer
    bottleneck = n_layers // 2
    print(f"   Bottleneck layer (n/2): {bottleneck + 1}")

    # Compute z-scores at bottleneck vs random baseline
    print(f"\n5. Computing z-scores at bottleneck (layer {bottleneck + 1})...")
    baseline = run_random_baseline(signal, model, tokenizer, bottleneck, n_trials=20)

    bottleneck_result = layer_results[bottleneck] if bottleneck < len(layer_results) else None
    z_scores = {}

    if bottleneck_result and bottleneck_result["success"]:
        for cat in SEMANTIC_CATEGORIES.keys():
            wow_mean = bottleneck_result["category_means"].get(cat, 0)
            base_mean = baseline[cat]["mean"]
            base_std = baseline[cat]["std"]
            z = (wow_mean - base_mean) / (base_std + 1e-8)
            z_scores[cat] = {
                "wow_mean": wow_mean,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "z_score": z,
            }

        print("\n   Category z-scores at bottleneck:")
        print("   " + "-" * 50)
        sorted_z = sorted(z_scores.items(), key=lambda x: x[1]["z_score"], reverse=True)
        for cat, stats in sorted_z:
            print(f"   {cat:12s}: z={stats['z_score']:+7.2f}")

    # Layer-by-layer analysis
    print("\n6. Layer-by-layer category rankings:")
    print("   " + "-" * 60)

    for i, result in enumerate(layer_results):
        if result["success"]:
            cat_means = result["category_means"]
            sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)
            top_cat = sorted_cats[0][0]
            bot_cat = sorted_cats[-1][0]
            print(f"   Layer {i+1:2d}: Top={top_cat:12s} Bot={bot_cat:12s} MATH={cat_means.get('MATHEMATICAL', 0):.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    if math_values:
        # Check if peak is in middle layers
        early_mean = np.mean(math_values[:n_layers//3]) if n_layers >= 3 else math_values[0]
        middle_mean = np.mean(math_values[n_layers//3:2*n_layers//3]) if n_layers >= 3 else math_values[0]
        late_mean = np.mean(math_values[2*n_layers//3:]) if n_layers >= 3 else math_values[0]

        print(f"\nMATHEMATICAL alignment by region:")
        print(f"   Early layers (1-{n_layers//3}): {early_mean:.4f}")
        print(f"   Middle layers ({n_layers//3+1}-{2*n_layers//3}): {middle_mean:.4f}")
        print(f"   Late layers ({2*n_layers//3+1}-{n_layers}): {late_mean:.4f}")

        if middle_mean > early_mean and middle_mean > late_mean:
            print("\n   --> MATHEMATICAL alignment is STRONGEST in middle layers")
            print("      (Consistent with semantic highway hypothesis!)")
            semantic_highway_consistent = True
        else:
            print("\n   --> MATHEMATICAL alignment does NOT peak in middle layers")
            semantic_highway_consistent = False

    # Top concepts by layer
    print("\n   Top concept by layer:")
    for i, result in enumerate(layer_results[:5]):  # First 5 layers
        if result["success"]:
            top = result["top_matches"][0]
            print(f"      Layer {i+1}: {top['label']}")
    print("      ...")
    for i, result in enumerate(layer_results[-3:], start=n_layers-3):  # Last 3 layers
        if result["success"]:
            top = result["top_matches"][0]
            print(f"      Layer {i+1}: {top['label']}")

    # Save results
    results = {
        "experiment": "exp51_layer_sweep",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "n_layers": n_layers,
        "bottleneck_layer": bottleneck + 1,
        "category_by_layer": {cat: [float(v) for v in vals] for cat, vals in category_by_layer.items()},
        "spectral_by_layer": [float(v) for v in spectral_by_layer],
        "peaks": {
            "mathematical_peak_layer": math_peak_layer + 1 if math_values else None,
            "mathematical_peak_value": math_peak_value if math_values else None,
            "primes_trough_layer": primes_trough_layer + 1 if primes_values else None,
            "primes_trough_value": primes_trough_value if primes_values else None,
            "spectral_peak_layer": spectral_peak_layer + 1 if spectral_by_layer else None,
            "spectral_peak_value": spectral_peak_value if spectral_by_layer else None,
        },
        "z_scores_at_bottleneck": z_scores,
        "region_means": {
            "early": float(early_mean) if math_values else None,
            "middle": float(middle_mean) if math_values else None,
            "late": float(late_mean) if math_values else None,
        },
        "semantic_highway_consistent": semantic_highway_consistent if math_values else None,
        "layer_results": [{
            "layer": r["layer"] + 1,
            "top_matches": r["top_matches"] if r["success"] else [],
            "spectral_similarity": r.get("spectral_similarity", 0),
            "category_means": r.get("category_means", {}),
        } for r in layer_results if r["success"]],
    }

    output_path = RESULTS_DIR / "exp51_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n7. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
