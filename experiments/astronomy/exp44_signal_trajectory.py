"""
Experiment 44: Signal Trajectory Analysis

The Wow! signal isn't a single point - it's a 2D matrix (time × frequency).
What if the message is encoded as a TRAJECTORY through semantic space?

This experiment:
1. Treats each time slice as a step along a path
2. Projects each time slice onto the semantic manifold
3. Traces the path through concept space
4. Looks for meaningful sequences
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

from exp42_semantic_highway_mapping import (
    SEMANTIC_CATEGORIES,
    load_wow_signal,
    load_model,
    get_layer_activation,
    build_semantic_manifold,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def project_vector_to_manifold(vector, semantic_gram_eigenvectors, semantic_spectrum, semantic_data):
    """
    Project a single vector onto the semantic manifold.

    Uses the vector's "shape" (autocorrelation structure) to find matching concepts.
    """
    # The vector represents a frequency profile at one time point
    # Its "shape" is captured by how its values relate to each other

    # Compute a simple feature: the distribution of values
    # (where energy is concentrated in the frequency spectrum)
    vector_sorted = np.sort(np.abs(vector))[::-1]
    vector_cumsum = np.cumsum(vector_sorted) / (np.sum(np.abs(vector)) + 1e-8)

    # Find the "effective dimension" of this time slice
    # (how many frequency bins contain most of the energy)
    effective_bins = np.searchsorted(vector_cumsum, 0.9) + 1

    # Map this to a position in semantic space
    # Use the relative magnitude profile as a signature
    k = min(10, len(semantic_spectrum))

    # Create a weight vector based on the signal's energy distribution
    signal_weights = np.zeros(k)
    for i in range(k):
        # How much does this component contribute?
        # Higher effective bins -> more spread out -> higher weight for later components
        signal_weights[i] = np.exp(-i / max(1, effective_bins / 3))

    signal_weights = signal_weights / (signal_weights.sum() + 1e-8)

    # Find concepts whose loadings match this weight pattern
    similarities = np.zeros(len(semantic_data))
    for i in range(len(semantic_data)):
        concept_loadings = np.abs(semantic_gram_eigenvectors[i, :k])
        concept_loadings = concept_loadings / (concept_loadings.sum() + 1e-8)

        # Similarity is how well the loading pattern matches signal weights
        similarities[i] = np.exp(-np.sum(np.abs(concept_loadings - signal_weights)))

    # Get top match
    top_idx = np.argmax(similarities)

    return {
        "top_concept": semantic_data[top_idx]["label"],
        "top_category": semantic_data[top_idx]["category"],
        "top_similarity": float(similarities[top_idx]),
        "effective_bins": effective_bins,
        "all_similarities": similarities,
    }


def analyze_trajectory(signal, semantic_activations, semantic_data):
    """Analyze the signal as a trajectory through semantic space."""
    print("   Analyzing trajectory through semantic space...")

    # Precompute semantic Gram matrix and its eigenvectors
    semantic_norms = np.linalg.norm(semantic_activations, axis=1, keepdims=True)
    semantic_unit = semantic_activations / (semantic_norms + 1e-8)
    G_semantic = semantic_unit @ semantic_unit.T

    U_sem, S_sem, _ = linalg.svd(G_semantic, full_matrices=False)

    trajectory = []
    n_timesteps = signal.shape[0]

    for t in range(n_timesteps):
        time_slice = signal[t, :]  # [50] frequency bins

        result = project_vector_to_manifold(time_slice, U_sem, S_sem, semantic_data)
        trajectory.append({
            "timestep": t,
            "concept": result["top_concept"],
            "category": result["top_category"],
            "similarity": result["top_similarity"],
            "effective_bins": result["effective_bins"],
        })

    return trajectory


def find_patterns(trajectory):
    """Find patterns in the trajectory."""
    concepts = [t["concept"] for t in trajectory]
    categories = [t["category"] for t in trajectory]

    # Find concept sequences
    print("\n   Concept sequence (first 30 timesteps):")
    for i in range(min(30, len(trajectory))):
        t = trajectory[i]
        print(f"      t={i:2d}: {t['concept']:15s} ({t['category']})")

    # Find most common concepts
    from collections import Counter
    concept_counts = Counter(concepts)
    category_counts = Counter(categories)

    print("\n   Most frequent concepts:")
    for concept, count in concept_counts.most_common(10):
        print(f"      {concept}: {count} times")

    print("\n   Most frequent categories:")
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(trajectory)
        print(f"      {cat}: {count} ({pct:.1f}%)")

    # Find transitions
    transitions = []
    for i in range(len(trajectory) - 1):
        if trajectory[i]["category"] != trajectory[i+1]["category"]:
            transitions.append((i, trajectory[i]["category"], trajectory[i+1]["category"]))

    print(f"\n   Category transitions: {len(transitions)}")

    # Look for repeating patterns
    patterns = {}
    for window_size in [2, 3, 4]:
        for i in range(len(concepts) - window_size):
            pattern = tuple(concepts[i:i+window_size])
            if pattern not in patterns:
                patterns[pattern] = 0
            patterns[pattern] += 1

    # Find repeated patterns
    repeated = [(p, c) for p, c in patterns.items() if c > 1]
    repeated.sort(key=lambda x: x[1], reverse=True)

    if repeated:
        print("\n   Repeated concept patterns:")
        for pattern, count in repeated[:10]:
            print(f"      {' -> '.join(pattern)}: {count} times")

    return {
        "concept_counts": dict(concept_counts),
        "category_counts": dict(category_counts),
        "n_transitions": len(transitions),
        "repeated_patterns": [{"pattern": list(p), "count": c} for p, c in repeated[:20]],
    }


def main():
    print("=" * 60)
    print("Experiment 44: Signal Trajectory Analysis")
    print("=" * 60)

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape} (time={signal.shape[0]}, freq={signal.shape[1]})")

    # Load model
    print("\n2. Loading LLM...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    print(f"   Model: SmolLM-135M ({n_layers} layers)")

    # Build semantic manifold
    print("\n3. Building semantic manifold...")
    bottleneck_layer = n_layers // 2
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, bottleneck_layer)
    print(f"   Manifold: {semantic_activations.shape}")

    # Analyze trajectory
    print("\n4. Analyzing trajectory...")
    trajectory = analyze_trajectory(signal, semantic_activations, semantic_data)

    # Find patterns
    print("\n5. Finding patterns...")
    patterns = find_patterns(trajectory)

    # Save results
    results = {
        "experiment": "exp44_signal_trajectory",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "n_timesteps": len(trajectory),
        "trajectory": trajectory,
        "patterns": patterns,
    }

    output_path = RESULTS_DIR / "exp44_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING: The signal traces a path through semantic space")
    print("=" * 60)

    # What's the dominant category?
    dom_cat = max(patterns["category_counts"].items(), key=lambda x: x[1])
    print(f"\nDominant category: {dom_cat[0]} ({dom_cat[1]}/{len(trajectory)} timesteps)")

    # What's the dominant concept?
    dom_concept = max(patterns["concept_counts"].items(), key=lambda x: x[1])
    print(f"Dominant concept: {dom_concept[0]} ({dom_concept[1]}/{len(trajectory)} timesteps)")

    return results


if __name__ == "__main__":
    main()
