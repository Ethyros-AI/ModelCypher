"""
Experiment 49: Expanded Mathematical Probes

Now that we've confirmed the MATHEMATICAL alignment is:
1. NOT a low-rank artifact (exp45)
2. Structure-dependent (exp46)
3. Unique among FRBs (exp47)

The question becomes: WHICH specific mathematical concepts does the signal align with?

Is it:
- "pi" specifically?
- Mathematical constants generally?
- Specific numbers?
- Mathematical operations?
- Geometric concepts?

This experiment adds 100+ mathematical probes across subcategories to find
the SPECIFIC concepts the signal resonates with.
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
    load_wow_signal,
    load_model,
    get_layer_activation,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Expanded mathematical probes organized by subcategory
EXPANDED_MATH_PROBES = {
    "NUMBERS_SMALL": {
        "probes": [
            ("zero", "Zero is the absence of quantity."),
            ("one", "One is unity, the first number."),
            ("two", "Two is the smallest prime number."),
            ("three", "Three is the first odd prime."),
            ("four", "Four is two squared."),
            ("five", "Five is the third prime number."),
            ("six", "Six is a perfect number."),
            ("seven", "Seven is a lucky prime."),
            ("eight", "Eight is two cubed."),
            ("nine", "Nine is three squared."),
            ("ten", "Ten is the base of decimal."),
        ],
    },
    "NUMBERS_SPECIAL": {
        "probes": [
            ("twelve", "Twelve is highly composite."),
            ("thirteen", "Thirteen is a prime number."),
            ("twenty_four", "Twenty-four divides evenly."),
            ("sixty", "Sixty has many factors."),
            ("hundred", "One hundred is ten squared."),
            ("thousand", "One thousand is ten cubed."),
            ("million", "One million is very large."),
            ("googol", "A googol has a hundred zeros."),
            ("infinity", "Infinity has no end."),
        ],
    },
    "PRIMES": {
        "probes": [
            ("prime_2", "Two is the only even prime."),
            ("prime_3", "Three is a prime number."),
            ("prime_5", "Five is a prime number."),
            ("prime_7", "Seven is a prime number."),
            ("prime_11", "Eleven is a prime number."),
            ("prime_13", "Thirteen is a prime number."),
            ("prime_17", "Seventeen is a prime number."),
            ("prime_19", "Nineteen is a prime number."),
            ("prime_23", "Twenty-three is a prime number."),
            ("twin_primes", "Twin primes differ by two."),
            ("mersenne", "Mersenne primes have special form."),
        ],
    },
    "CONSTANTS": {
        "probes": [
            ("pi", "Pi is approximately 3.14159."),
            ("e", "Euler's number is approximately 2.71828."),
            ("phi", "The golden ratio is approximately 1.618."),
            ("sqrt2", "The square root of two is irrational."),
            ("sqrt3", "The square root of three is irrational."),
            ("euler_gamma", "Euler's constant gamma appears in series."),
            ("tau", "Tau is two times pi."),
            ("feigenbaum", "The Feigenbaum constant describes chaos."),
            ("omega", "The omega constant satisfies special equations."),
        ],
    },
    "OPERATIONS": {
        "probes": [
            ("addition", "Addition combines quantities."),
            ("subtraction", "Subtraction finds differences."),
            ("multiplication", "Multiplication is repeated addition."),
            ("division", "Division splits into parts."),
            ("exponent", "Exponentiation is repeated multiplication."),
            ("logarithm", "Logarithms undo exponents."),
            ("square_root", "Square root finds the base."),
            ("factorial", "Factorial multiplies descending numbers."),
            ("derivative", "Derivatives measure change."),
            ("integral", "Integrals accumulate area."),
        ],
    },
    "SEQUENCES": {
        "probes": [
            ("fibonacci", "The Fibonacci sequence grows exponentially."),
            ("arithmetic", "Arithmetic sequences have constant difference."),
            ("geometric", "Geometric sequences have constant ratio."),
            ("triangular", "Triangular numbers form pyramids."),
            ("square_nums", "Square numbers are perfect squares."),
            ("cube_nums", "Cube numbers are perfect cubes."),
            ("powers_of_2", "Powers of two double each time."),
            ("catalan", "Catalan numbers count structures."),
            ("harmonic", "The harmonic series diverges slowly."),
        ],
    },
    "GEOMETRY": {
        "probes": [
            ("point", "A point has no dimension."),
            ("line", "A line extends infinitely."),
            ("circle", "A circle has constant radius."),
            ("sphere", "A sphere is three-dimensional."),
            ("triangle", "A triangle has three sides."),
            ("square_shape", "A square has four equal sides."),
            ("cube_shape", "A cube has six faces."),
            ("dimension", "Dimension measures extent."),
            ("angle", "An angle measures rotation."),
            ("parallel", "Parallel lines never meet."),
            ("perpendicular", "Perpendicular lines meet at right angles."),
        ],
    },
    "RATIOS": {
        "probes": [
            ("ratio_1_1", "The ratio one to one is unity."),
            ("ratio_1_2", "The ratio one to two is half."),
            ("ratio_2_3", "The ratio two to three appears in music."),
            ("ratio_3_2", "The ratio three to two is the perfect fifth."),
            ("ratio_4_3", "The ratio four to three is the perfect fourth."),
            ("ratio_5_4", "The ratio five to four is a major third."),
            ("octave", "The octave ratio is two to one."),
            ("golden_ratio", "The golden ratio appears in nature."),
        ],
    },
    "ABSTRACT_MATH": {
        "probes": [
            ("set", "A set is a collection of elements."),
            ("function", "A function maps inputs to outputs."),
            ("proof", "A proof establishes truth."),
            ("theorem", "A theorem is a proven statement."),
            ("axiom", "An axiom is assumed true."),
            ("equation", "An equation states equality."),
            ("variable", "A variable represents unknowns."),
            ("constant_value", "A constant does not change."),
            ("infinity_concept", "Infinity is unbounded."),
            ("zero_concept", "Zero represents nothing."),
        ],
    },
    "PHYSICS_MATH": {
        "probes": [
            ("wavelength", "Wavelength measures wave cycles."),
            ("frequency_wave", "Frequency counts cycles per second."),
            ("amplitude", "Amplitude measures wave height."),
            ("period", "Period is time per cycle."),
            ("phase", "Phase measures wave position."),
            ("resonance", "Resonance amplifies vibrations."),
            ("harmonic_wave", "Harmonics are integer multiples."),
            ("spectrum_physics", "The spectrum shows all frequencies."),
        ],
    },
}


def build_expanded_manifold(model, tokenizer, layer_idx):
    """Build the expanded mathematical manifold."""
    print(f"   Building expanded math manifold at layer {layer_idx}...")

    all_data = []
    all_activations = []

    for category, info in EXPANDED_MATH_PROBES.items():
        print(f"      Collecting {category}...")
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
    print(f"   Expanded manifold shape: {activations.shape}")

    return all_data, activations


def project_signal_to_expanded_manifold(signal, semantic_activations, semantic_data, n_components=10):
    """Project signal onto the expanded mathematical manifold."""
    print("   Projecting signal to expanded manifold...")

    # Normalize activations
    semantic_norms = np.linalg.norm(semantic_activations, axis=1, keepdims=True)
    semantic_unit = semantic_activations / (semantic_norms + 1e-8)

    # Signal structure
    signal_row_norms = np.linalg.norm(signal, axis=1, keepdims=True)
    signal_unit = signal / (signal_row_norms + 1e-8)

    # Gram matrices
    G_semantic = semantic_unit @ semantic_unit.T
    G_signal = signal_unit @ signal_unit.T

    # SVD of both
    U_sem, S_sem, _ = linalg.svd(G_semantic, full_matrices=False)
    _, S_sig, _ = linalg.svd(G_signal, full_matrices=False)

    # Spectral signatures
    k = min(n_components, len(S_sem), len(S_sig))
    signal_spectrum = S_sig[:k] / S_sig[:k].sum() if S_sig[:k].sum() > 0 else np.ones(k) / k
    semantic_spectrum = S_sem[:k] / S_sem[:k].sum() if S_sem[:k].sum() > 0 else np.ones(k) / k

    # Compute similarity for each concept
    semantic_basis = U_sem[:, :k]
    similarities = np.zeros(len(semantic_data))

    for i in range(len(semantic_data)):
        concept_loadings = semantic_basis[i, :]
        spectral_weight = np.exp(-np.abs(semantic_spectrum - signal_spectrum))
        weighted_loading = np.sum(np.abs(concept_loadings) * spectral_weight)
        similarities[i] = weighted_loading

    # Normalize
    similarities = similarities / (np.max(np.abs(similarities)) + 1e-8)

    return similarities


def compute_subcategory_means(similarities, semantic_data):
    """Compute mean similarity per subcategory."""
    category_sims = {}
    category_counts = {}

    for i, d in enumerate(semantic_data):
        cat = d["category"]
        if cat not in category_sims:
            category_sims[cat] = 0.0
            category_counts[cat] = 0
        category_sims[cat] += similarities[i]
        category_counts[cat] += 1

    return {cat: category_sims[cat] / category_counts[cat] for cat in category_sims}


def main():
    print("=" * 60)
    print("Experiment 49: Expanded Mathematical Probes")
    print("=" * 60)
    print("\nQuestion: WHICH specific math concepts does the signal align with?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape}")

    # Load model
    print("\n2. Loading LLM...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    bottleneck_layer = n_layers // 2
    print(f"   Model: SmolLM-135M, bottleneck layer: {bottleneck_layer}")

    # Build expanded manifold
    print("\n3. Building expanded mathematical manifold...")
    semantic_data, semantic_activations = build_expanded_manifold(model, tokenizer, bottleneck_layer)
    print(f"   Total probes: {len(semantic_data)}")

    # Project signal
    print("\n4. Projecting signal...")
    similarities = project_signal_to_expanded_manifold(signal, semantic_activations, semantic_data)

    # Get top matches
    sorted_idx = np.argsort(similarities)[::-1]

    print("\n5. TOP 30 CONCEPT MATCHES:")
    print("   " + "-" * 55)
    top_matches = []
    for i, idx in enumerate(sorted_idx[:30]):
        d = semantic_data[idx]
        sim = similarities[idx]
        print(f"   {i+1:2d}. [{d['category']:15s}] {d['label']:20s} = {sim:.4f}")
        top_matches.append({
            "rank": i + 1,
            "label": d["label"],
            "category": d["category"],
            "similarity": float(sim),
        })

    # Subcategory analysis
    print("\n6. SUBCATEGORY RANKINGS:")
    print("   " + "-" * 45)
    cat_means = compute_subcategory_means(similarities, semantic_data)
    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)

    for cat, mean in sorted_cats:
        print(f"   {cat:20s}: {mean:.4f}")

    # Find specific patterns
    print("\n7. SPECIFIC FINDINGS:")
    print("   " + "-" * 45)

    # Which constants rank highest?
    constants_rank = {}
    for i, idx in enumerate(sorted_idx):
        d = semantic_data[idx]
        if d["category"] == "CONSTANTS":
            constants_rank[d["label"]] = i + 1

    print("\n   CONSTANTS ranking:")
    for label, rank in sorted(constants_rank.items(), key=lambda x: x[1]):
        print(f"      {label}: #{rank}")

    # Which numbers rank highest?
    numbers_rank = {}
    for i, idx in enumerate(sorted_idx):
        d = semantic_data[idx]
        if d["category"] in ["NUMBERS_SMALL", "NUMBERS_SPECIAL", "PRIMES"]:
            numbers_rank[d["label"]] = i + 1

    print("\n   NUMBERS ranking (top 10):")
    for label, rank in sorted(numbers_rank.items(), key=lambda x: x[1])[:10]:
        print(f"      {label}: #{rank}")

    # Which ratios rank highest?
    ratios_rank = {}
    for i, idx in enumerate(sorted_idx):
        d = semantic_data[idx]
        if d["category"] == "RATIOS":
            ratios_rank[d["label"]] = i + 1

    print("\n   RATIOS ranking:")
    for label, rank in sorted(ratios_rank.items(), key=lambda x: x[1]):
        print(f"      {label}: #{rank}")

    # Key question: Is pi special?
    pi_rank = None
    e_rank = None
    phi_rank = None
    for i, idx in enumerate(sorted_idx):
        d = semantic_data[idx]
        if d["label"] == "pi":
            pi_rank = i + 1
        elif d["label"] == "e":
            e_rank = i + 1
        elif d["label"] == "phi":
            phi_rank = i + 1

    print("\n   KEY CONSTANTS:")
    print(f"      pi:  #{pi_rank}")
    print(f"      e:   #{e_rank}")
    print(f"      phi: #{phi_rank}")

    # Save results
    results = {
        "experiment": "exp49_expanded_math_probes",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "n_probes": len(semantic_data),
        "top_30_matches": top_matches,
        "subcategory_means": {cat: float(mean) for cat, mean in sorted_cats},
        "constants_ranking": constants_rank,
        "numbers_ranking": numbers_rank,
        "ratios_ranking": ratios_rank,
        "key_constants": {
            "pi_rank": pi_rank,
            "e_rank": e_rank,
            "phi_rank": phi_rank,
        },
    }

    output_path = RESULTS_DIR / "exp49_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n8. Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nTop subcategory: {sorted_cats[0][0]} (mean={sorted_cats[0][1]:.4f})")
    print(f"Top concept: {top_matches[0]['label']} ({top_matches[0]['category']})")

    print(f"\nPi ranks #{pi_rank} out of {len(semantic_data)} concepts")

    if pi_rank and pi_rank <= 10:
        print("   --> Pi is in the TOP 10 (highly significant)")
    elif pi_rank and pi_rank <= 30:
        print("   --> Pi is in the TOP 30 (moderately significant)")
    else:
        print("   --> Pi is NOT in top 30 (may not be specifically pi-related)")

    return results


if __name__ == "__main__":
    main()
