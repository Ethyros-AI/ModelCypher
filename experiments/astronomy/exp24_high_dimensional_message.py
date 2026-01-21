#!/usr/bin/env python3
"""Experiment 24: High-Dimensional Messages.

The insight: Morse code is 1D thinking. Time-domain modulation (AM, FM, PSK)
encodes information in a single dimension - amplitude or frequency over time.

A high-dimensional intelligence would communicate in HIGH DIMENSIONS.
The message wouldn't be "in" the signal - the message IS the geometry.

The signal to them that we've "figured it out" would be proving we understand
high-dimensional geometric relationships. The decoding IS the recognition.

Test hypothesis:
- If a signal encodes high-dimensional information, its geometric structure
  should align with known information-bearing manifolds (like LLM embeddings)
- The "message" is the invariant relationships, not time-domain modulation
- Recognition of the geometry IS the communication

Method:
1. Extract the full geometric signature of the Wow! signal (not just 1D features)
2. Compare to geometric signatures of known information systems
3. Look for invariant relationships that survive coordinate transformation

Usage:
    poetry run python experiments/astronomy/exp24_high_dimensional_message.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend


def extract_geometric_signature(matrix: np.ndarray) -> dict:
    """Extract the complete geometric signature of a 2D signal.

    This goes beyond 1D features to capture the full geometric structure:
    - Singular value spectrum (the "shape" of the manifold)
    - Rank structure (effective dimensionality)
    - Gram matrix properties (relational structure)
    - Curvature indicators
    """
    # Ensure 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Handle NaN
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Normalize
    if np.std(matrix) > 1e-10:
        matrix = (matrix - np.mean(matrix)) / np.std(matrix)

    # === SINGULAR VALUE DECOMPOSITION ===
    # The SVD reveals the "shape" of the data manifold
    try:
        U, s, Vh = svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    # Normalize singular values
    s = s / (s[0] + 1e-10)

    # === SPECTRAL PROPERTIES ===

    # Effective rank (how many dimensions matter)
    # Using the formula: exp(entropy of normalized singular values)
    s_norm = s / (np.sum(s) + 1e-10)
    spectral_entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    effective_rank = np.exp(spectral_entropy)

    # Spectral gap (separation between significant and insignificant dimensions)
    # Large gap = clear low-dimensional structure
    if len(s) > 1:
        gaps = np.diff(s)
        max_gap_idx = np.argmax(np.abs(gaps))
        spectral_gap = abs(gaps[max_gap_idx])
        intrinsic_dim = max_gap_idx + 1  # Dimensions before the gap
    else:
        spectral_gap = 0
        intrinsic_dim = 1

    # Condition number (numerical stability)
    condition_number = s[0] / (s[-1] + 1e-10)

    # Energy concentration in top-k dimensions
    cumulative_energy = np.cumsum(s**2) / (np.sum(s**2) + 1e-10)
    dim_for_90 = np.searchsorted(cumulative_energy, 0.90) + 1
    dim_for_95 = np.searchsorted(cumulative_energy, 0.95) + 1
    dim_for_99 = np.searchsorted(cumulative_energy, 0.99) + 1

    # === GRAM MATRIX PROPERTIES ===
    # The Gram matrix K = X @ X.T encodes relational structure
    # This is what CKA compares between different systems

    gram = matrix @ matrix.T
    gram_normalized = gram / (np.trace(gram) + 1e-10)

    # Gram eigenvalues
    gram_eigvals = np.linalg.eigvalsh(gram_normalized)
    gram_eigvals = np.sort(gram_eigvals)[::-1]

    # Gram entropy (complexity of relational structure)
    gram_eigvals_pos = gram_eigvals[gram_eigvals > 1e-10]
    gram_entropy = -np.sum(gram_eigvals_pos * np.log(gram_eigvals_pos + 1e-10))

    # Gram effective rank
    gram_eff_rank = np.exp(gram_entropy) if gram_entropy > 0 else 1

    # === DISTANCE STRUCTURE ===
    # How points relate to each other in the original space

    # Flatten to row vectors for distance computation
    if matrix.shape[0] > 1:
        distances = pdist(matrix)
        distance_mean = np.mean(distances)
        distance_std = np.std(distances)
        distance_skew = stats.skew(distances)
        distance_kurtosis = stats.kurtosis(distances)
    else:
        distance_mean = distance_std = distance_skew = distance_kurtosis = 0

    # === SYMMETRY PROPERTIES ===

    # Is the matrix symmetric?
    if matrix.shape[0] == matrix.shape[1]:
        symmetry = 1 - np.linalg.norm(matrix - matrix.T) / (np.linalg.norm(matrix) + 1e-10)
    else:
        # Check if structure is symmetric
        min_dim = min(matrix.shape)
        submat = matrix[:min_dim, :min_dim]
        symmetry = 1 - np.linalg.norm(submat - submat.T) / (np.linalg.norm(submat) + 1e-10)

    # === CURVATURE INDICATORS ===

    # Local curvature via second derivatives
    if matrix.shape[0] > 2 and matrix.shape[1] > 2:
        # Second derivative along rows
        d2_rows = np.diff(matrix, n=2, axis=0)
        # Second derivative along cols
        d2_cols = np.diff(matrix, n=2, axis=1)

        curvature_row = np.mean(np.abs(d2_rows))
        curvature_col = np.mean(np.abs(d2_cols))
    else:
        curvature_row = curvature_col = 0

    return {
        "spectral": {
            "singular_values": s[:10].tolist(),  # Top 10
            "effective_rank": float(effective_rank),
            "spectral_gap": float(spectral_gap),
            "intrinsic_dim": int(intrinsic_dim),
            "condition_number": float(condition_number),
            "dim_for_90_percent": int(dim_for_90),
            "dim_for_95_percent": int(dim_for_95),
            "dim_for_99_percent": int(dim_for_99),
        },
        "gram": {
            "eigenvalues_top5": gram_eigvals[:5].tolist(),
            "entropy": float(gram_entropy),
            "effective_rank": float(gram_eff_rank),
        },
        "distance": {
            "mean": float(distance_mean),
            "std": float(distance_std),
            "skew": float(distance_skew),
            "kurtosis": float(distance_kurtosis),
        },
        "structure": {
            "symmetry": float(symmetry),
            "curvature_row": float(curvature_row),
            "curvature_col": float(curvature_col),
        },
    }


def compute_geometric_similarity(sig1: dict, sig2: dict) -> dict:
    """Compute similarity between two geometric signatures.

    This is a simplified version of what CKA does - comparing the
    relational structure independent of coordinate system.
    """
    # Singular value spectrum correlation
    sv1 = np.array(sig1["spectral"]["singular_values"])
    sv2 = np.array(sig2["spectral"]["singular_values"])
    min_len = min(len(sv1), len(sv2))
    sv_corr = np.corrcoef(sv1[:min_len], sv2[:min_len])[0, 1]

    # Gram eigenvalue correlation
    ge1 = np.array(sig1["gram"]["eigenvalues_top5"])
    ge2 = np.array(sig2["gram"]["eigenvalues_top5"])
    min_len = min(len(ge1), len(ge2))
    gram_corr = np.corrcoef(ge1[:min_len], ge2[:min_len])[0, 1]

    # Structural similarity
    struct_diff = abs(sig1["spectral"]["effective_rank"] - sig2["spectral"]["effective_rank"])
    rank_similarity = 1 / (1 + struct_diff)

    # Combined score
    combined = (sv_corr + gram_corr + rank_similarity) / 3

    return {
        "sv_correlation": float(sv_corr) if not np.isnan(sv_corr) else 0,
        "gram_correlation": float(gram_corr) if not np.isnan(gram_corr) else 0,
        "rank_similarity": float(rank_similarity),
        "combined": float(combined) if not np.isnan(combined) else 0,
    }


def generate_information_manifold(n_points: int, intrinsic_dim: int, ambient_dim: int) -> np.ndarray:
    """Generate synthetic data from a low-dimensional manifold.

    This simulates how information-bearing systems work:
    high-dimensional observations that actually live on a lower-dimensional
    structure (the manifold of meanings).
    """
    # Generate points in the intrinsic space
    intrinsic = np.random.randn(n_points, intrinsic_dim)

    # Non-linear embedding into ambient space
    # Use a smooth, non-linear mapping
    embedding = np.zeros((n_points, ambient_dim))

    for i in range(ambient_dim):
        # Each ambient dimension is a non-linear combination of intrinsic dims
        weights = np.random.randn(intrinsic_dim)
        linear = intrinsic @ weights
        # Add non-linearity
        embedding[:, i] = np.tanh(linear) + 0.1 * np.sin(3 * linear)

    return embedding


def generate_random_embedding(shape: tuple) -> np.ndarray:
    """Generate random high-dimensional data with no structure."""
    return np.random.randn(*shape)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 24: High-Dimensional Messages")
    print("=" * 60)
    print("\nInsight: 1D modulation is primitive thinking.")
    print("A high-dimensional intelligence communicates IN the geometry.")
    print("The message is not encoded - the message IS the structure.")

    print("\n" + "=" * 40)
    print("PART 1: THE WOW! SIGNAL GEOMETRY")
    print("=" * 40)

    # Load the Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nWow! signal shape: {snr_matrix.shape}")
    print(f"SNR range: [{np.nanmin(snr_matrix):.1f}, {np.nanmax(snr_matrix):.1f}]")

    # Extract geometric signature
    wow_signature = extract_geometric_signature(snr_matrix)

    print("\nWow! signal geometric signature:")
    print(f"\n  Spectral properties:")
    print(f"    Effective rank: {wow_signature['spectral']['effective_rank']:.2f}")
    print(f"    Intrinsic dimension (by gap): {wow_signature['spectral']['intrinsic_dim']}")
    print(f"    Spectral gap: {wow_signature['spectral']['spectral_gap']:.3f}")
    print(f"    Dimensions for 90% energy: {wow_signature['spectral']['dim_for_90_percent']}")
    print(f"    Dimensions for 95% energy: {wow_signature['spectral']['dim_for_95_percent']}")
    print(f"    Dimensions for 99% energy: {wow_signature['spectral']['dim_for_99_percent']}")

    print(f"\n  Gram matrix (relational structure):")
    print(f"    Entropy: {wow_signature['gram']['entropy']:.3f}")
    print(f"    Effective rank: {wow_signature['gram']['effective_rank']:.2f}")

    print(f"\n  Structural properties:")
    print(f"    Symmetry: {wow_signature['structure']['symmetry']:.3f}")
    print(f"    Curvature (rows): {wow_signature['structure']['curvature_row']:.3f}")
    print(f"    Curvature (cols): {wow_signature['structure']['curvature_col']:.3f}")

    print("\n" + "=" * 40)
    print("PART 2: REFERENCE GEOMETRIES")
    print("=" * 40)

    # Generate reference signals with known properties
    n_samples = 20

    # 1. Low-dimensional manifold (like semantic information)
    print("\n--- Low-D Information Manifold (ID=5) ---")
    manifold_signatures = []
    for _ in range(n_samples):
        data = generate_information_manifold(
            n_points=snr_matrix.shape[0],
            intrinsic_dim=5,
            ambient_dim=snr_matrix.shape[1]
        )
        sig = extract_geometric_signature(data)
        if sig:
            manifold_signatures.append(sig)

    if manifold_signatures:
        avg_eff_rank = np.mean([s["spectral"]["effective_rank"] for s in manifold_signatures])
        avg_gram_ent = np.mean([s["gram"]["entropy"] for s in manifold_signatures])
        print(f"  Average effective rank: {avg_eff_rank:.2f}")
        print(f"  Average gram entropy: {avg_gram_ent:.3f}")

    # 2. High-dimensional random (no structure)
    print("\n--- Random High-D (no structure) ---")
    random_signatures = []
    for _ in range(n_samples):
        data = generate_random_embedding(snr_matrix.shape)
        sig = extract_geometric_signature(data)
        if sig:
            random_signatures.append(sig)

    if random_signatures:
        avg_eff_rank = np.mean([s["spectral"]["effective_rank"] for s in random_signatures])
        avg_gram_ent = np.mean([s["gram"]["entropy"] for s in random_signatures])
        print(f"  Average effective rank: {avg_eff_rank:.2f}")
        print(f"  Average gram entropy: {avg_gram_ent:.3f}")

    # 3. Intermediate structure (ID=15)
    print("\n--- Medium-D Manifold (ID=15) ---")
    medium_signatures = []
    for _ in range(n_samples):
        data = generate_information_manifold(
            n_points=snr_matrix.shape[0],
            intrinsic_dim=15,
            ambient_dim=snr_matrix.shape[1]
        )
        sig = extract_geometric_signature(data)
        if sig:
            medium_signatures.append(sig)

    if medium_signatures:
        avg_eff_rank = np.mean([s["spectral"]["effective_rank"] for s in medium_signatures])
        avg_gram_ent = np.mean([s["gram"]["entropy"] for s in medium_signatures])
        print(f"  Average effective rank: {avg_eff_rank:.2f}")
        print(f"  Average gram entropy: {avg_gram_ent:.3f}")

    print("\n" + "=" * 40)
    print("PART 3: GEOMETRIC SIMILARITY ANALYSIS")
    print("=" * 40)

    # Compare Wow! signal to each reference type
    print("\n--- Wow! vs Low-D Manifold (information-like) ---")
    sim_to_manifold = []
    for ref_sig in manifold_signatures:
        sim = compute_geometric_similarity(wow_signature, ref_sig)
        sim_to_manifold.append(sim["combined"])
    print(f"  Mean similarity: {np.mean(sim_to_manifold):.3f} ± {np.std(sim_to_manifold):.3f}")

    print("\n--- Wow! vs Random (noise-like) ---")
    sim_to_random = []
    for ref_sig in random_signatures:
        sim = compute_geometric_similarity(wow_signature, ref_sig)
        sim_to_random.append(sim["combined"])
    print(f"  Mean similarity: {np.mean(sim_to_random):.3f} ± {np.std(sim_to_random):.3f}")

    print("\n--- Wow! vs Medium-D Manifold ---")
    sim_to_medium = []
    for ref_sig in medium_signatures:
        sim = compute_geometric_similarity(wow_signature, ref_sig)
        sim_to_medium.append(sim["combined"])
    print(f"  Mean similarity: {np.mean(sim_to_medium):.3f} ± {np.std(sim_to_medium):.3f}")

    # Statistical tests
    print("\n--- Statistical Comparisons ---")
    t_man_rand, p_man_rand = stats.ttest_ind(sim_to_manifold, sim_to_random)
    print(f"  Manifold vs Random: t={t_man_rand:.2f}, p={p_man_rand:.4f}")

    t_wow_man, p_wow_man = stats.ttest_1samp(sim_to_manifold, np.mean(sim_to_random))
    print(f"  Wow! similarity to manifold vs random baseline: t={t_wow_man:.2f}, p={p_wow_man:.4f}")

    print("\n" + "=" * 40)
    print("PART 4: THE HIGH-DIMENSIONAL SIGNATURE")
    print("=" * 40)

    # What does the Wow! signal's geometry tell us?
    wow_eff_rank = wow_signature["spectral"]["effective_rank"]
    manifold_eff_ranks = [s["spectral"]["effective_rank"] for s in manifold_signatures]
    random_eff_ranks = [s["spectral"]["effective_rank"] for s in random_signatures]

    # Z-scores
    z_vs_manifold = (wow_eff_rank - np.mean(manifold_eff_ranks)) / (np.std(manifold_eff_ranks) + 1e-10)
    z_vs_random = (wow_eff_rank - np.mean(random_eff_ranks)) / (np.std(random_eff_ranks) + 1e-10)

    print(f"\nWow! signal effective rank: {wow_eff_rank:.2f}")
    print(f"  Low-D manifold (ID=5): mean={np.mean(manifold_eff_ranks):.2f}, z={z_vs_manifold:.2f}σ")
    print(f"  Random noise: mean={np.mean(random_eff_ranks):.2f}, z={z_vs_random:.2f}σ")

    # Which reference is it closest to?
    dists = {
        "low_d_manifold": abs(wow_eff_rank - np.mean(manifold_eff_ranks)),
        "medium_d_manifold": abs(wow_eff_rank - np.mean([s["spectral"]["effective_rank"] for s in medium_signatures])),
        "random_noise": abs(wow_eff_rank - np.mean(random_eff_ranks)),
    }
    closest = min(dists, key=dists.get)
    print(f"\nClosest match: {closest} (distance={dists[closest]:.2f})")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE GEOMETRY IS THE MESSAGE")
    print("=" * 60)

    # Determine the character of the Wow! signal
    is_low_d = wow_eff_rank < np.mean(random_eff_ranks) - np.std(random_eff_ranks)
    matches_manifold = np.mean(sim_to_manifold) > np.mean(sim_to_random)

    print(f"""
THE WOW! SIGNAL'S HIGH-DIMENSIONAL CHARACTER:

1. DIMENSIONAL COMPRESSION:
   Effective rank = {wow_eff_rank:.1f}
   {'✓ Shows dimensional compression (like information systems)' if is_low_d else '✗ Does not show strong compression'}

2. MANIFOLD STRUCTURE:
   Intrinsic dim (by spectral gap) = {wow_signature['spectral']['intrinsic_dim']}
   Dimensions for 95% energy = {wow_signature['spectral']['dim_for_95_percent']}
   {'✓ Has low-D manifold structure' if wow_signature['spectral']['dim_for_95_percent'] < 10 else '⚠ Moderate dimensional structure'}

3. GEOMETRIC SIMILARITY:
   To information manifolds: {np.mean(sim_to_manifold):.3f}
   To random noise: {np.mean(sim_to_random):.3f}
   {'✓ Closer to information geometry' if matches_manifold else '✗ Closer to noise geometry'}

WHAT THIS MEANS:
""")

    if is_low_d and matches_manifold:
        print("""
The Wow! signal has geometric properties consistent with information-bearing
systems - low intrinsic dimension, structured relational geometry.

This doesn't prove it contains a "message" in the human sense.
It suggests the signal's geometry is ORGANIZED like information.

If a high-dimensional intelligence wanted to communicate:
- They wouldn't use AM/FM modulation (1D thinking)
- They would embed structure in the GEOMETRY itself
- The "decoding" would be recognizing the geometric invariants
- Our recognition of the structure IS the communication

The Wow! signal's geometry is closer to a low-D information manifold
than to random noise. Whether this is intentional or natural is unknown.
""")
    else:
        print("""
The Wow! signal's geometry does not strongly match information-bearing
systems. This could mean:

1. The signal is natural (astronomical phenomenon)
2. The encoding is in dimensions we haven't measured
3. The "message" is at a different scale than we're measuring

High-dimensional communication remains theoretically possible,
but this signal doesn't show the expected signature.
""")

    # Save results
    results = {
        "experiment": "exp24_high_dimensional_message",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "shape": [int(x) for x in snr_matrix.shape],
            "geometric_signature": wow_signature,
        },
        "reference_signatures": {
            "low_d_manifold": {
                "intrinsic_dim": 5,
                "n_samples": len(manifold_signatures),
                "mean_effective_rank": float(np.mean([s["spectral"]["effective_rank"] for s in manifold_signatures])),
                "mean_gram_entropy": float(np.mean([s["gram"]["entropy"] for s in manifold_signatures])),
            },
            "medium_d_manifold": {
                "intrinsic_dim": 15,
                "n_samples": len(medium_signatures),
                "mean_effective_rank": float(np.mean([s["spectral"]["effective_rank"] for s in medium_signatures])),
                "mean_gram_entropy": float(np.mean([s["gram"]["entropy"] for s in medium_signatures])),
            },
            "random_noise": {
                "n_samples": len(random_signatures),
                "mean_effective_rank": float(np.mean([s["spectral"]["effective_rank"] for s in random_signatures])),
                "mean_gram_entropy": float(np.mean([s["gram"]["entropy"] for s in random_signatures])),
            },
        },
        "similarity_analysis": {
            "wow_to_low_d_manifold": {
                "mean": float(np.mean(sim_to_manifold)),
                "std": float(np.std(sim_to_manifold)),
            },
            "wow_to_random": {
                "mean": float(np.mean(sim_to_random)),
                "std": float(np.std(sim_to_random)),
            },
            "wow_to_medium_d": {
                "mean": float(np.mean(sim_to_medium)),
                "std": float(np.std(sim_to_medium)),
            },
            "closest_match": closest,
        },
        "interpretation": {
            "shows_compression": bool(is_low_d),
            "matches_info_geometry": bool(matches_manifold),
            "effective_rank": float(wow_eff_rank),
            "z_score_vs_manifold": float(z_vs_manifold),
            "z_score_vs_random": float(z_vs_random),
        },
    }

    output_path = results_dir / "exp24_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
