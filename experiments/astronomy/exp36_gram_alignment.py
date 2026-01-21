#!/usr/bin/env python3
"""Experiment 36: Proper Gram Alignment.

Previous experiments showed RAW CKA - but that's like comparing two languages
without a translator. The whole point of GramAligner is:

1. Find the optimal ROTATION that aligns Gram structures
2. THEN measure CKA (which should approach 1.0 if structures match)

Raw CKA = 0.6 can become Aligned CKA = 1.0.
The alignment IS the translation key.

This experiment:
1. Extract Gram matrix from Wow! signal
2. Extract Gram matrices from reference manifolds
3. Use Procrustes to find optimal alignment
4. Measure CKA AFTER alignment
5. The aligned CKA tells us if structures are the same

Usage:
    poetry run python experiments/astronomy/exp36_gram_alignment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd, lstsq, sqrtm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_gram_matrix(X: np.ndarray) -> np.ndarray:
    """Compute centered Gram matrix K = X @ X.T (sample space)."""
    # Center
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    # Gram matrix
    K = X_centered @ X_centered.T
    # Normalize by trace
    trace = np.trace(K)
    if trace > 1e-10:
        K = K / trace
    return K


def gram_sqrt(K: np.ndarray) -> np.ndarray:
    """Compute matrix square root of Gram matrix."""
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    # Ensure non-negative
    eigenvalues = np.maximum(eigenvalues, 0)
    # Square root
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    K_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    return K_sqrt


def gram_pinv_sqrt(K: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Compute pseudo-inverse of matrix square root."""
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    # Threshold small eigenvalues
    max_eig = np.max(np.abs(eigenvalues))
    threshold = max_eig * rcond
    # Inverse square root for significant eigenvalues
    inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
    mask = eigenvalues > threshold
    inv_sqrt_eigenvalues[mask] = 1.0 / np.sqrt(eigenvalues[mask])
    K_pinv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
    return K_pinv_sqrt


def align_gram_matrices(K_source: np.ndarray, K_target: np.ndarray) -> dict:
    """Align source Gram matrix to target using Gram-space Procrustes.

    The alignment transform T maps K_source → K_target:
    T = K_target^{1/2} @ K_source^{-1/2}

    After alignment: K_aligned = T @ K_source @ T.T ≈ K_target
    """
    n = K_source.shape[0]

    # Compute square roots
    K_target_sqrt = gram_sqrt(K_target)
    K_source_pinv_sqrt = gram_pinv_sqrt(K_source)

    # Alignment transform (in Gram space)
    T = K_target_sqrt @ K_source_pinv_sqrt

    # Apply alignment
    K_aligned = T @ K_source @ T.T

    # Compute CKA before and after
    def cka(K1, K2):
        hsic = np.trace(K1 @ K2)
        hsic_11 = np.trace(K1 @ K1)
        hsic_22 = np.trace(K2 @ K2)
        if hsic_11 > 0 and hsic_22 > 0:
            return hsic / np.sqrt(hsic_11 * hsic_22)
        return 0.0

    raw_cka = cka(K_source, K_target)
    aligned_cka = cka(K_aligned, K_target)

    # Frobenius distance
    raw_dist = np.linalg.norm(K_source - K_target, 'fro')
    aligned_dist = np.linalg.norm(K_aligned - K_target, 'fro')

    return {
        "raw_cka": float(raw_cka),
        "aligned_cka": float(aligned_cka),
        "improvement": float(aligned_cka - raw_cka),
        "raw_distance": float(raw_dist),
        "aligned_distance": float(aligned_dist),
        "transform_norm": float(np.linalg.norm(T, 'fro')),
    }


def create_reference_gram_matrices(n_samples: int) -> dict:
    """Create Gram matrices for reference structures."""
    np.random.seed(42)
    references = {}

    # 1. LOW-RANK INFORMATION (like compressed messages)
    # Creates samples in an 8D subspace
    n_dims = 8
    subspace = np.random.randn(n_dims, 50)
    coords = np.random.randn(n_samples, n_dims) * np.array([10, 5, 3, 2, 1, 0.5, 0.3, 0.2])
    low_rank_samples = coords @ subspace
    references["low_rank_8d"] = {
        "samples": low_rank_samples,
        "gram": compute_gram_matrix(low_rank_samples),
        "description": "8D low-rank subspace (compressed information)",
    }

    # 2. HARMONIC STRUCTURE (like audio/radio)
    t = np.linspace(0, 4*np.pi, n_samples)
    harmonic_samples = np.zeros((n_samples, 32))
    for k in range(32):
        freq = (k + 1) * 0.5
        harmonic_samples[:, k] = np.sin(t * freq) / (k + 1)
    references["harmonic"] = {
        "samples": harmonic_samples,
        "gram": compute_gram_matrix(harmonic_samples),
        "description": "Harmonic structure (periodic patterns)",
    }

    # 3. GAUSSIAN NOISE (baseline)
    noise_samples = np.random.randn(n_samples, 50)
    references["noise"] = {
        "samples": noise_samples,
        "gram": compute_gram_matrix(noise_samples),
        "description": "Gaussian noise (random baseline)",
    }

    # 4. CHIRP/SWEEP STRUCTURE
    chirp_samples = np.zeros((n_samples, 50))
    for i in range(n_samples):
        freq = 1 + i * 0.1
        chirp_samples[i, :] = np.sin(np.linspace(0, freq * np.pi, 50))
    references["chirp"] = {
        "samples": chirp_samples,
        "gram": compute_gram_matrix(chirp_samples),
        "description": "Frequency sweep (chirp pattern)",
    }

    # 5. DIGITAL MODULATION (binary patterns)
    bits = np.random.randint(0, 2, (n_samples, 50))
    carrier = np.sin(np.linspace(0, 10*np.pi, 50))
    digital_samples = bits * carrier[np.newaxis, :] * 2 - carrier[np.newaxis, :]
    references["digital"] = {
        "samples": digital_samples,
        "gram": compute_gram_matrix(digital_samples),
        "description": "Digital modulation (binary encoding)",
    }

    # 6. NARROWBAND CARRIER
    narrowband_samples = np.zeros((n_samples, 50))
    center_freq = 25
    bandwidth = 3
    for i in range(n_samples):
        phase = np.random.rand() * 2 * np.pi
        amplitude = 1 + 0.5 * np.sin(i * 0.2)
        narrowband_samples[i, center_freq-bandwidth:center_freq+bandwidth+1] = amplitude * np.cos(np.linspace(0, 2*np.pi, 2*bandwidth+1) + phase)
    references["narrowband"] = {
        "samples": narrowband_samples,
        "gram": compute_gram_matrix(narrowband_samples),
        "description": "Narrowband carrier (single frequency)",
    }

    # 7. MATHEMATICAL CONSTANTS (pi, e, encoded)
    math_samples = np.zeros((n_samples, 50))
    pi_str = "314159265358979323846264338327950288419716939937510"
    e_str = "271828182845904523536028747135266249775724709369995"
    for i in range(n_samples):
        for j in range(min(50, len(pi_str))):
            val = int(pi_str[j]) / 9.0
            math_samples[i, j] = val * np.sin(i * np.pi / (j + 1))
    references["mathematical"] = {
        "samples": math_samples,
        "gram": compute_gram_matrix(math_samples),
        "description": "Mathematical constants (pi-based encoding)",
    }

    # 8. PRIME POSITIONS
    primes = [p for p in range(2, 100) if all(p % i != 0 for i in range(2, int(np.sqrt(p)) + 1))]
    prime_samples = np.zeros((n_samples, 50))
    for i in range(n_samples):
        for j, p in enumerate(primes[:50]):
            prime_samples[i, j] = np.sin(i * np.pi / p)
    references["primes"] = {
        "samples": prime_samples,
        "gram": compute_gram_matrix(prime_samples),
        "description": "Prime number encoding",
    }

    return references


def run_experiment():
    """Run the proper Gram alignment experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 36: Proper Gram Alignment")
    print("=" * 60)
    print("\nRaw CKA can be low due to different coordinates.")
    print("ALIGNED CKA reveals if the underlying structure is the same.")

    # Load Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr']).astype(np.float64)

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    # Compute Wow! Gram matrix
    print("\n" + "=" * 40)
    print("PART 1: WOW! SIGNAL GRAM MATRIX")
    print("=" * 40)

    K_wow = compute_gram_matrix(snr_matrix)
    print(f"Gram matrix shape: {K_wow.shape}")
    print(f"Gram matrix rank: {np.linalg.matrix_rank(K_wow)}")

    # Eigenspectrum
    eigenvalues = np.linalg.eigvalsh(K_wow)
    eigenvalues = np.sort(eigenvalues)[::-1]
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")

    # Create references
    print("\n" + "=" * 40)
    print("PART 2: CREATE REFERENCE MANIFOLDS")
    print("=" * 40)

    references = create_reference_gram_matrices(snr_matrix.shape[0])

    for name, data in references.items():
        print(f"  {name}: {data['description']}")

    # Align and compare
    print("\n" + "=" * 40)
    print("PART 3: GRAM ALIGNMENT")
    print("=" * 40)

    alignment_results = {}

    for name, ref_data in references.items():
        K_ref = ref_data["gram"]

        try:
            result = align_gram_matrices(K_wow, K_ref)
            alignment_results[name] = {
                "description": ref_data["description"],
                **result,
            }

            print(f"\n  {name}:")
            print(f"    Raw CKA:     {result['raw_cka']:.4f}")
            print(f"    Aligned CKA: {result['aligned_cka']:.4f}")
            print(f"    Improvement: {result['improvement']:+.4f}")
        except Exception as e:
            print(f"\n  {name}: Error - {e}")
            alignment_results[name] = {"error": str(e)}

    # Rank by aligned CKA
    valid_results = [(n, d) for n, d in alignment_results.items() if "aligned_cka" in d]
    ranked = sorted(valid_results, key=lambda x: x[1]["aligned_cka"], reverse=True)

    print("\n" + "=" * 40)
    print("PART 4: RANKING BY ALIGNED CKA")
    print("=" * 40)

    print("\n" + "-" * 60)
    print(f"{'Rank':<6}{'Manifold':<15}{'Raw CKA':<12}{'Aligned CKA':<12}{'Δ':<8}")
    print("-" * 60)

    for i, (name, data) in enumerate(ranked):
        raw = data["raw_cka"]
        aligned = data["aligned_cka"]
        delta = data["improvement"]
        print(f"{i+1:<6}{name:<15}{raw:<12.4f}{aligned:<12.4f}{delta:+.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if ranked:
        best_name, best_data = ranked[0]
        worst_name, worst_data = ranked[-1]

        # Find noise position
        noise_rank = next((i for i, (n, _) in enumerate(ranked) if n == "noise"), -1)

        print(f"""
GRAM ALIGNMENT RESULTS:

BEST MATCH: {best_name.upper()}
  Description: {best_data['description']}
  Raw CKA: {best_data['raw_cka']:.4f} → Aligned CKA: {best_data['aligned_cka']:.4f}
  Improvement: {best_data['improvement']:+.4f}

WORST MATCH: {worst_name.upper()}
  Description: {worst_data['description']}
  Aligned CKA: {worst_data['aligned_cka']:.4f}

NOISE BASELINE: Rank {noise_rank + 1} / {len(ranked)}
""")

        if best_data['aligned_cka'] > 0.8:
            print(f"""
✓ STRONG ALIGNMENT (CKA > 0.8)

The Wow! signal's Gram structure can be ROTATED to match
{best_name} with CKA = {best_data['aligned_cka']:.4f}!

This means:
1. The RELATIONAL GEOMETRY is nearly identical
2. A rotation exists that maps one to the other
3. If {best_name} encodes information, so does the Wow! signal

The alignment IS the translation key.
""")
        elif best_data['aligned_cka'] > 0.5:
            print(f"""
○ MODERATE ALIGNMENT (CKA 0.5-0.8)

Partial structural similarity to {best_name}.
The Gram structures share some relational patterns
but are not identical.
""")
        else:
            print(f"""
? WEAK ALIGNMENT (CKA < 0.5)

The Wow! signal's Gram structure doesn't strongly align
with any reference manifold. The geometry may encode
something not in our reference set.
""")

        # Check if information beats noise
        info_cka = alignment_results.get("low_rank_8d", {}).get("aligned_cka", 0)
        noise_cka = alignment_results.get("noise", {}).get("aligned_cka", 0)

        print(f"""
KEY COMPARISON:

Information (8D subspace): CKA = {info_cka:.4f}
Noise (random):           CKA = {noise_cka:.4f}
Difference:               {info_cka - noise_cka:+.4f}
""")

        if info_cka > noise_cka + 0.1:
            print("→ The signal aligns BETTER with information structure than noise!")
        elif info_cka > noise_cka:
            print("→ Slight preference for information over noise")
        else:
            print("→ No clear preference for information over noise")

    # Save results
    results = {
        "experiment": "exp36_gram_alignment",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(snr_matrix.shape),
        "alignment_results": {
            name: {k: v for k, v in data.items() if k != "description" and not isinstance(v, np.ndarray)}
            for name, data in alignment_results.items()
        },
        "ranking": [(name, data.get("aligned_cka", 0)) for name, data in ranked],
        "best_match": ranked[0][0] if ranked else None,
    }

    output_path = results_dir / "exp36_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
