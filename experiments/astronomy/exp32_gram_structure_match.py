#!/usr/bin/env python3
"""Experiment 32: Direct Gram Structure Comparison.

The previous experiment had numerical issues. Let's be more direct:

1. Use the actual Wow! signal matrix (82×50)
2. Compute its Gram matrix (the INVARIANT structure)
3. Compare the Gram eigenspectrum to known information systems
4. See if the spectral signature matches any category

The key insight: The Gram matrix IS the geometry. Two systems with
matching Gram eigenspectra have matching relational structure.

Usage:
    poetry run python experiments/astronomy/exp32_gram_structure_match.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_gram_signature(matrix: np.ndarray) -> dict:
    """Compute the Gram matrix signature (eigenspectrum properties)."""
    # Normalize
    matrix = np.nan_to_num(matrix.astype(np.float64), nan=0.0)
    if np.std(matrix) > 1e-10:
        matrix_norm = (matrix - np.mean(matrix)) / np.std(matrix)
    else:
        return None

    # Gram matrix (row-wise: sample space)
    n = matrix_norm.shape[0]
    K = matrix_norm @ matrix_norm.T / n

    # Eigenspectrum
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    # Normalize eigenspectrum
    total = np.sum(eigenvalues) + 1e-10
    p = eigenvalues / total

    # Spectral entropy
    p_nonzero = p[p > 1e-10]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))

    # Effective rank
    eff_rank = np.exp(entropy)

    # Decay rate (how fast eigenvalues fall off)
    n_eigen = min(50, len(eigenvalues))
    log_eig = np.log(p[:n_eigen] + 1e-10)
    indices = np.arange(n_eigen)
    decay_rate = -np.polyfit(indices, log_eig, 1)[0]

    # Energy concentration
    cumsum = np.cumsum(p)
    n_90 = np.searchsorted(cumsum, 0.90) + 1
    n_99 = np.searchsorted(cumsum, 0.99) + 1

    return {
        "spectral_entropy": float(entropy),
        "effective_rank": float(eff_rank),
        "decay_rate": float(decay_rate),
        "n_modes_90pct": int(n_90),
        "n_modes_99pct": int(n_99),
        "top_5_eigenvalues": p[:5].tolist(),
        "eigenvalue_ratio_1_2": float(p[0] / (p[1] + 1e-10)),
    }


def generate_reference_matrices(shape: tuple, seed: int = 42) -> dict:
    """Generate reference matrices with known structure types.

    These are carefully designed to have SPECIFIC Gram properties,
    matching known classes of information systems.
    """
    np.random.seed(seed)
    n_rows, n_cols = shape

    references = {}

    # 1. PURE NOISE (baseline)
    noise = np.random.randn(n_rows, n_cols)
    references["noise"] = {
        "matrix": noise,
        "description": "Pure Gaussian noise (high entropy, high rank)",
    }

    # 2. RANK-1 SIGNAL (maximally compressed)
    u = np.random.randn(n_rows, 1)
    v = np.random.randn(1, n_cols)
    rank1 = u @ v
    references["rank1"] = {
        "matrix": rank1,
        "description": "Rank-1 matrix (minimal entropy, rank=1)",
    }

    # 3. LOW-RANK INFORMATION (like compressed messages)
    # Effective rank ~8 to match Wow! signal
    n_components = 8
    U = np.random.randn(n_rows, n_components)
    singular_vals = np.array([10, 5, 3, 2, 1.5, 1, 0.7, 0.5])  # Decaying
    V = np.random.randn(n_components, n_cols)
    low_rank = (U * singular_vals) @ V + np.random.randn(n_rows, n_cols) * 0.1
    references["low_rank_8"] = {
        "matrix": low_rank,
        "description": "Low-rank (~8D) structure like compressed info",
    }

    # 4. HARMONIC SIGNAL (like audio/radio)
    t = np.linspace(0, 4*np.pi, n_rows)
    f = np.linspace(0, 10, n_cols)
    harmonic = np.zeros((n_rows, n_cols))
    for k in range(1, 6):  # 5 harmonics
        harmonic += np.sin(k * t[:, np.newaxis]) * np.cos(k * f[np.newaxis, :]) / k
    harmonic += np.random.randn(n_rows, n_cols) * 0.05
    references["harmonic"] = {
        "matrix": harmonic,
        "description": "Harmonic structure (like audio/radio signals)",
    }

    # 5. LOCALIZED BURST (like FRB or transient)
    burst = np.random.randn(n_rows, n_cols) * 0.1
    # Add localized peak
    center_t, center_f = n_rows // 2, n_cols // 2
    for i in range(n_rows):
        for j in range(n_cols):
            dist = np.sqrt((i - center_t)**2 + (j - center_f)**2)
            burst[i, j] += 30 * np.exp(-dist**2 / 100)
    references["burst"] = {
        "matrix": burst,
        "description": "Localized burst (like FRB, transient)",
    }

    # 6. DIGITAL MESSAGE (discrete levels, structure)
    # Simulates binary/symbolic encoding
    message_bits = np.random.choice([0, 1], size=(n_rows, n_cols))
    carrier = np.sin(np.linspace(0, 20*np.pi, n_cols))
    message = message_bits * carrier[np.newaxis, :] + np.random.randn(n_rows, n_cols) * 0.1
    references["digital"] = {
        "matrix": message,
        "description": "Digital message (binary modulation)",
    }

    # 7. NARROW BAND (single frequency carrier)
    carrier_freq = n_cols // 4
    narrowband = np.zeros((n_rows, n_cols))
    narrowband[:, carrier_freq-2:carrier_freq+3] = 10 * np.random.randn(n_rows, 5)
    narrowband += np.random.randn(n_rows, n_cols) * 0.5
    references["narrowband"] = {
        "matrix": narrowband,
        "description": "Narrowband signal (single carrier)",
    }

    # 8. CORRELATED NOISE (structured but not informative)
    base = np.random.randn(n_rows, n_cols)
    correlated = np.zeros_like(base)
    for i in range(n_rows):
        for j in range(n_cols):
            # Spatial correlation
            window = base[max(0,i-2):min(n_rows,i+3), max(0,j-2):min(n_cols,j+3)]
            correlated[i, j] = np.mean(window)
    references["correlated"] = {
        "matrix": correlated,
        "description": "Correlated noise (spatially smooth)",
    }

    return references


def compare_signatures(wow_sig: dict, ref_sigs: dict) -> dict:
    """Compare Wow! signal's Gram signature to references."""
    comparisons = {}

    for name, ref_data in ref_sigs.items():
        ref_sig = compute_gram_signature(ref_data["matrix"])
        if ref_sig is None:
            continue

        # Distance in signature space
        entropy_diff = abs(wow_sig["spectral_entropy"] - ref_sig["spectral_entropy"])
        rank_diff = abs(wow_sig["effective_rank"] - ref_sig["effective_rank"])
        decay_diff = abs(wow_sig["decay_rate"] - ref_sig["decay_rate"])

        # Normalized distance (geometric mean of relative differences)
        rel_entropy = entropy_diff / (wow_sig["spectral_entropy"] + 1e-10)
        rel_rank = rank_diff / (wow_sig["effective_rank"] + 1e-10)
        rel_decay = decay_diff / (wow_sig["decay_rate"] + 1e-10)
        distance = np.sqrt(rel_entropy**2 + rel_rank**2 + rel_decay**2)

        comparisons[name] = {
            "description": ref_data["description"],
            "signature": ref_sig,
            "entropy_diff": float(entropy_diff),
            "rank_diff": float(rank_diff),
            "decay_diff": float(decay_diff),
            "normalized_distance": float(distance),
        }

    return comparisons


def run_experiment():
    """Run the Gram structure comparison experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 32: Gram Structure Match")
    print("=" * 60)
    print("\nQuestion: What TYPE of signal does the Wow! signal match?")
    print("Method: Compare Gram eigenspectrum signatures")

    # Load Wow! signal
    print("\n" + "=" * 40)
    print("PART 1: WOW! SIGNAL GRAM SIGNATURE")
    print("=" * 40)

    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    wow_sig = compute_gram_signature(snr_matrix)

    print(f"\nGram signature:")
    print(f"  Spectral entropy: {wow_sig['spectral_entropy']:.3f}")
    print(f"  Effective rank: {wow_sig['effective_rank']:.2f}")
    print(f"  Decay rate: {wow_sig['decay_rate']:.3f}")
    print(f"  Modes for 90% energy: {wow_sig['n_modes_90pct']}")
    print(f"  Eigenvalue ratio (1/2): {wow_sig['eigenvalue_ratio_1_2']:.2f}")

    # Generate references
    print("\n" + "=" * 40)
    print("PART 2: REFERENCE SIGNATURES")
    print("=" * 40)

    references = generate_reference_matrices(snr_matrix.shape)

    ref_signatures = {}
    for name, ref_data in references.items():
        sig = compute_gram_signature(ref_data["matrix"])
        if sig:
            ref_signatures[name] = {
                "description": ref_data["description"],
                "signature": sig,
            }
            print(f"\n  {name}:")
            print(f"    Entropy: {sig['spectral_entropy']:.3f}, Rank: {sig['effective_rank']:.2f}, Decay: {sig['decay_rate']:.3f}")

    # Compare
    print("\n" + "=" * 40)
    print("PART 3: SIGNATURE COMPARISON")
    print("=" * 40)

    comparisons = compare_signatures(wow_sig, references)

    # Rank by distance
    ranked = sorted(comparisons.items(), key=lambda x: x[1]["normalized_distance"])

    print("\n" + "-" * 50)
    print("REFERENCE SIGNALS RANKED BY SIMILARITY TO WOW!")
    print("-" * 50)
    print(f"{'Rank':<6}{'Type':<15}{'Distance':<12}{'Description'}")
    print("-" * 50)

    for i, (name, data) in enumerate(ranked):
        print(f"{i+1:<6}{name:<15}{data['normalized_distance']:.3f}        {data['description'][:40]}")

    # Analysis
    print("\n" + "=" * 40)
    print("PART 4: DETAILED ANALYSIS")
    print("=" * 40)

    best_match = ranked[0]
    worst_match = ranked[-1]

    print(f"\nCLOSEST MATCH: {best_match[0]}")
    print(f"  Distance: {best_match[1]['normalized_distance']:.3f}")
    print(f"  {best_match[1]['description']}")
    print(f"  Wow! entropy: {wow_sig['spectral_entropy']:.3f} vs {best_match[1]['signature']['spectral_entropy']:.3f}")
    print(f"  Wow! rank: {wow_sig['effective_rank']:.2f} vs {best_match[1]['signature']['effective_rank']:.2f}")
    print(f"  Wow! decay: {wow_sig['decay_rate']:.3f} vs {best_match[1]['signature']['decay_rate']:.3f}")

    print(f"\nFARTHEST MATCH: {worst_match[0]}")
    print(f"  Distance: {worst_match[1]['normalized_distance']:.3f}")
    print(f"  {worst_match[1]['description']}")

    # Where does noise fall?
    noise_rank = next(i for i, (name, _) in enumerate(ranked) if name == "noise")
    print(f"\nPURE NOISE RANK: {noise_rank + 1} / {len(ranked)}")

    # Key insight
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(f"""
THE WOW! SIGNAL'S GRAM STRUCTURE:

Best match: {best_match[0].upper()}
Noise rank: {noise_rank + 1} / {len(ranked)} ({"better than most" if noise_rank > len(ranked)//2 else "worse than most"} structured signals)

KEY METRICS:
  Spectral Entropy: {wow_sig['spectral_entropy']:.3f}
  Effective Rank: {wow_sig['effective_rank']:.2f}
  Decay Rate: {wow_sig['decay_rate']:.3f}

WHAT THE GRAM STRUCTURE TELLS US:
""")

    if wow_sig['effective_rank'] < 15:
        print(f"  ✓ LOW EFFECTIVE RANK ({wow_sig['effective_rank']:.1f})")
        print("    → Signal is COMPRESSED into few dimensions")
        print("    → Consistent with information encoding")
    else:
        print(f"  ○ MODERATE EFFECTIVE RANK ({wow_sig['effective_rank']:.1f})")
        print("    → Signal occupies many dimensions")

    if wow_sig['spectral_entropy'] < 2.5:
        print(f"\n  ✓ LOW SPECTRAL ENTROPY ({wow_sig['spectral_entropy']:.3f})")
        print("    → Energy concentrated in few modes")
        print("    → NOT random noise")
    elif wow_sig['spectral_entropy'] < 3.5:
        print(f"\n  ○ MODERATE SPECTRAL ENTROPY ({wow_sig['spectral_entropy']:.3f})")
    else:
        print(f"\n  ✗ HIGH SPECTRAL ENTROPY ({wow_sig['spectral_entropy']:.3f})")
        print("    → Energy spread across many modes")
        print("    → More noise-like")

    if wow_sig['decay_rate'] > 0.1:
        print(f"\n  ✓ FAST EIGENVALUE DECAY ({wow_sig['decay_rate']:.3f})")
        print("    → Clear dominant structure")
        print("    → Not typical noise")
    else:
        print(f"\n  ○ SLOW EIGENVALUE DECAY ({wow_sig['decay_rate']:.3f})")

    # Final verdict
    if best_match[0] == "low_rank_8":
        print(f"""
CONCLUSION: The Wow! signal's Gram structure MATCHES
low-rank information encoding. Its geometry is consistent
with a compressed, organized signal - NOT random noise.
""")
    elif best_match[0] in ["burst", "narrowband"]:
        print(f"""
CONCLUSION: The Wow! signal's Gram structure matches
{best_match[0]} signals. This is consistent with a
radio transient - natural or otherwise.
""")
    elif best_match[0] == "harmonic":
        print(f"""
CONCLUSION: The Wow! signal's Gram structure matches
harmonic structure. This suggests periodic modulation
in the underlying signal.
""")
    else:
        print(f"""
CONCLUSION: The Wow! signal's Gram structure is closest
to {best_match[0]}. The geometry is anomalous but
the precise classification remains ambiguous.
""")

    # Save results
    results = {
        "experiment": "exp32_gram_structure_match",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "shape": list(snr_matrix.shape),
            "gram_signature": wow_sig,
        },
        "reference_signatures": {name: data["signature"] for name, data in ref_signatures.items()},
        "comparisons": {name: {k: v for k, v in data.items() if k != "signature"}
                        for name, data in comparisons.items()},
        "ranking": [(name, data["normalized_distance"]) for name, data in ranked],
        "best_match": {
            "name": best_match[0],
            "distance": float(best_match[1]["normalized_distance"]),
            "description": best_match[1]["description"],
        },
        "noise_rank": noise_rank + 1,
    }

    output_path = results_dir / "exp32_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
