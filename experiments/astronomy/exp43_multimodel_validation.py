"""
Experiment 43: Multi-Model Validation of Semantic Highway Mapping

If the signal's location on the semantic manifold is truly encoding something
in the INVARIANT structure, then it should project to the same semantic region
regardless of which LLM we use.

This is the critical validation: invariance across models.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.io import readsav
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from exp42
from exp42_semantic_highway_mapping import (
    SEMANTIC_CATEGORIES,
    load_wow_signal,
    load_model,
    get_layer_activation,
    build_semantic_manifold,
    project_signal_to_manifold,
    compute_category_distribution,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single_model(model_path: str, model_name: str, signal: np.ndarray):
    """Run the semantic highway mapping for a single model."""
    print(f"\n{'='*60}")
    print(f"ANALYZING WITH: {model_name}")
    print(f"{'='*60}")

    try:
        model, tokenizer, n_layers = load_model(model_path)
        print(f"   Loaded: {n_layers} layers")

        bottleneck_layer = n_layers // 2
        print(f"   Bottleneck layer: {bottleneck_layer}")

        semantic_data, semantic_activations = build_semantic_manifold(
            model, tokenizer, bottleneck_layer
        )
        print(f"   Manifold: {semantic_activations.shape}")

        top_matches, similarities, signal_features = project_signal_to_manifold(
            signal, semantic_activations, semantic_data
        )

        cat_means = compute_category_distribution(similarities, semantic_data)

        # Get top categories
        sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)

        return {
            "model": model_name,
            "n_layers": n_layers,
            "bottleneck_layer": bottleneck_layer,
            "manifold_shape": list(semantic_activations.shape),
            "top_matches": top_matches[:10],
            "category_distribution": {cat: float(mean) for cat, mean in sorted_cats},
        }

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"model": model_name, "error": str(e)}


def main():
    print("=" * 60)
    print("Experiment 43: Multi-Model Invariance Validation")
    print("=" * 60)

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape}")

    # Models to test
    models = {
        "SmolLM-135M": str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M"),
        # Add more models as available
    }

    # Check for additional models
    additional_models = [
        ("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16", "LFM2-350M"),
        (str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"), "Qwen2.5-0.5B"),
    ]

    for path, name in additional_models:
        if Path(path).exists():
            models[name] = path
            print(f"   Found: {name}")

    print(f"\n2. Testing {len(models)} models...")

    all_results = {}
    for name, path in models.items():
        result = run_single_model(path, name, signal)
        all_results[name] = result

    # Compare results across models
    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON")
    print("=" * 60)

    # Check if category rankings are consistent
    print("\n3. Category rankings per model:")
    category_rankings = {}
    for name, result in all_results.items():
        if "error" not in result:
            cats = list(result["category_distribution"].keys())
            category_rankings[name] = cats
            print(f"\n   {name}:")
            for i, cat in enumerate(cats[:5]):
                print(f"      {i+1}. {cat}: {result['category_distribution'][cat]:.4f}")

    # Check top concept overlap
    print("\n4. Top concept overlap:")
    all_top_concepts = {}
    for name, result in all_results.items():
        if "error" not in result:
            concepts = [m["label"] for m in result["top_matches"]]
            all_top_concepts[name] = set(concepts)
            print(f"   {name} top 10: {concepts}")

    # Compute pairwise overlap
    if len(all_top_concepts) > 1:
        print("\n5. Concept overlap matrix:")
        model_names = list(all_top_concepts.keys())
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    overlap = len(all_top_concepts[m1] & all_top_concepts[m2])
                    print(f"   {m1} ∩ {m2}: {overlap}/10 concepts")

    # Save results
    output = {
        "experiment": "exp43_multimodel_validation",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "models_tested": list(models.keys()),
        "results": all_results,
    }

    output_path = RESULTS_DIR / "exp43_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("INVARIANCE ASSESSMENT")
    print("=" * 60)

    # Check if MATHEMATICAL is consistently high and PRIMES consistently low
    math_rankings = []
    primes_rankings = []

    for name, result in all_results.items():
        if "error" not in result:
            cats = list(result["category_distribution"].keys())
            if "MATHEMATICAL" in cats:
                math_rankings.append((name, cats.index("MATHEMATICAL") + 1))
            if "PRIMES" in cats:
                primes_rankings.append((name, cats.index("PRIMES") + 1))

    if math_rankings:
        print(f"\nMATHEMATICAL category rank across models:")
        for name, rank in math_rankings:
            print(f"   {name}: #{rank}")

    if primes_rankings:
        print(f"\nPRIMES category rank across models:")
        for name, rank in primes_rankings:
            print(f"   {name}: #{rank}")

    return output


if __name__ == "__main__":
    main()
