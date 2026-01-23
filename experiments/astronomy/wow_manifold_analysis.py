#!/usr/bin/env python3
"""
WOW! SIGNAL MANIFOLD ANALYSIS

Applies LLM-style high-dimensional geometry tools to the Wow! signal.
Treats the 82×50 signal matrix as "activations on a manifold" - the same
way we analyze neural network representations.

Key analyses:
1. TwoNN intrinsic dimension (with geodesic distances)
2. Effective rank / participation ratio
3. Local dimension mapping
4. Geodesic vs Euclidean distance structure
5. Spectral analysis for geometric constants (π, φ, √2)

Usage:
    poetry run python experiments/astronomy/wow_manifold_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import math

# Initialize backend before any domain imports
from modelcypher.backends import initialize_default_backend
initialize_default_backend()

# Constants
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
SQRT2 = math.sqrt(2)
E = math.e


def load_wow_signal() -> tuple[np.ndarray, np.ndarray]:
    """Load the Wow! signal data from CSV.

    Returns:
        intensity_matrix: 82×50 matrix of signal intensities
        metadata: Array of metadata (RA, Dec, freq, etc.)
    """
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.csv"

    # Read raw data
    rows = []
    metadata_rows = []

    with open(data_path) as f:
        # Skip header
        header = next(f)

        for line in f:
            parts = line.strip().split(",")
            # First 50 columns are intensity values, rest is metadata
            intensity_values = []
            for val in parts[:50]:
                val = val.strip()
                if val == "" or val == " ":
                    intensity_values.append(0)
                elif val.isalpha():
                    # Handle letter codes (E, Q, U, J, 5 etc.)
                    # E=14, Q=26, U=30, J=19, 5=5
                    letter_map = {
                        'E': 14, 'Q': 26, 'U': 30, 'J': 19,
                        'A': 10, 'B': 11, 'C': 12, 'D': 13,
                        'F': 15, 'G': 16, 'H': 17, 'I': 18,
                        'K': 20, 'L': 21, 'M': 22, 'N': 23,
                        'O': 24, 'P': 25, 'R': 27, 'S': 28,
                        'T': 29, 'V': 31, 'W': 32, 'X': 33,
                        'Y': 34, 'Z': 35
                    }
                    intensity_values.append(letter_map.get(val.upper(), 0))
                else:
                    try:
                        intensity_values.append(int(val))
                    except ValueError:
                        intensity_values.append(0)

            rows.append(intensity_values)
            metadata_rows.append(parts[50:])

    intensity_matrix = np.array(rows, dtype=np.float32)
    return intensity_matrix, np.array(metadata_rows)


def percent_error(measured: float, expected: float) -> float:
    """Calculate percent error from expected value."""
    if expected == 0:
        return float('inf')
    return abs(measured - expected) / expected * 100


def analyze_intrinsic_dimension(matrix: np.ndarray, name: str) -> dict:
    """Compute TwoNN intrinsic dimension using geodesic distances."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain._backend import get_default_backend

    print(f"\n{'=' * 70}")
    print(f"TwoNN INTRINSIC DIMENSION: {name}")
    print(f"{'=' * 70}")

    n, d = matrix.shape
    print(f"  Shape: {n} points in {d}-dimensional space")

    backend = get_default_backend()
    estimator = IntrinsicDimension(backend=backend)

    try:
        # Convert to backend array
        arr = backend.array(matrix.astype(np.float32))
        # Compute with confidence interval
        result = estimator.compute(arr, with_ci=True)

        print(f"\n  Intrinsic Dimension: {result.intrinsic_dimension:.6f}")
        print(f"  Sample count: {result.sample_count}")
        print(f"  Usable count: {result.usable_count}")

        if result.ci:
            print(f"  95% CI: [{result.ci.lower:.4f}, {result.ci.upper:.4f}]")

        # Compare to geometric constants
        print(f"\n  Comparison to geometric constants:")
        print(f"    vs π = {PI:.6f}:  error = {percent_error(result.intrinsic_dimension, PI):.4f}%")
        print(f"    vs e = {E:.6f}:  error = {percent_error(result.intrinsic_dimension, E):.4f}%")
        print(f"    vs φ = {PHI:.6f}:  error = {percent_error(result.intrinsic_dimension, PHI):.4f}%")
        print(f"    vs 3 = 3.000000:  error = {percent_error(result.intrinsic_dimension, 3):.4f}%")

        return {
            "intrinsic_dimension": result.intrinsic_dimension,
            "ci": (result.ci.lower, result.ci.upper) if result.ci else None,
            "pi_error": percent_error(result.intrinsic_dimension, PI),
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def analyze_effective_rank(matrix: np.ndarray) -> dict:
    """Compute effective rank (participation ratio) of the signal."""
    from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
    from modelcypher.core.domain._backend import get_default_backend

    print(f"\n{'=' * 70}")
    print("EFFECTIVE RANK ANALYSIS")
    print(f"{'=' * 70}")

    backend = get_default_backend()
    er = EffectiveRank(backend=backend)
    arr = backend.array(matrix.astype(np.float32))
    result = er.compute(arr)

    print(f"\n  Renyi Effective Rank (Participation Ratio): {result.renyi_effective_rank:.6f}")
    print(f"  Shannon Effective Rank: {result.shannon_effective_rank:.6f}")
    print(f"  Spectral Entropy: {result.spectral_entropy:.6f}")
    print(f"  Sample count: {result.sample_count}")
    print(f"  Feature dim: {result.feature_dim}")
    print(f"  Singular values: {result.n_singular_values}")

    # Compare to geometric constants
    print(f"\n  Renyi rank comparison:")
    print(f"    vs π = {PI:.6f}:  error = {percent_error(result.renyi_effective_rank, PI):.4f}%")
    print(f"    vs e = {E:.6f}:  error = {percent_error(result.renyi_effective_rank, E):.4f}%")
    print(f"    vs φ = {PHI:.6f}:  error = {percent_error(result.renyi_effective_rank, PHI):.4f}%")

    print(f"\n  Shannon rank comparison:")
    print(f"    vs π = {PI:.6f}:  error = {percent_error(result.shannon_effective_rank, PI):.4f}%")
    print(f"    vs e = {E:.6f}:  error = {percent_error(result.shannon_effective_rank, E):.4f}%")

    return {
        "renyi_rank": result.renyi_effective_rank,
        "shannon_rank": result.shannon_effective_rank,
        "spectral_entropy": result.spectral_entropy,
        "renyi_pi_error": percent_error(result.renyi_effective_rank, PI),
    }


def analyze_geodesic_structure(matrix: np.ndarray) -> dict:
    """Analyze geodesic vs Euclidean distance structure."""
    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry
    from modelcypher.core.domain._backend import get_default_backend

    print(f"\n{'=' * 70}")
    print("GEODESIC DISTANCE STRUCTURE")
    print(f"{'=' * 70}")

    backend = get_default_backend()
    rg = RiemannianGeometry(backend=backend)

    try:
        # Convert to backend array
        arr = backend.array(matrix.astype(np.float32))
        # Compute geodesic distances
        result = rg.geodesic_distances(arr, k_neighbors=None)

        print(f"\n  k_neighbors (connectivity): {result.k_neighbors}")
        print(f"  Connected: {result.connected}")

        # Convert geodesic distances to numpy
        backend.eval(result.distances)
        geo_dist = np.array(backend.tolist(result.distances))

        # Compute Euclidean distances for comparison
        from scipy.spatial.distance import cdist
        euclidean_dist = cdist(matrix, matrix)

        # Compare geodesic vs Euclidean
        # Mask out diagonal and infinite values
        mask = (geo_dist > 0) & (geo_dist < result.inf_value) & (euclidean_dist > 0)

        if np.sum(mask) > 0:
            geo_flat = geo_dist[mask]
            euc_flat = euclidean_dist[mask]

            ratios = geo_flat / euc_flat
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            max_ratio = np.max(ratios)

            print(f"\n  Geodesic/Euclidean ratio statistics:")
            print(f"    Mean ratio: {mean_ratio:.6f}")
            print(f"    Std ratio: {std_ratio:.6f}")
            print(f"    Max ratio: {max_ratio:.6f}")
            print(f"    Min ratio: {np.min(ratios):.6f}")

            # A ratio > 1 indicates curvature (geodesic path is longer than chord)
            curved_fraction = np.mean(ratios > 1.01)
            print(f"\n  Fraction with ratio > 1.01 (curved): {curved_fraction:.4f}")

            # Compare mean ratio to geometric constants
            print(f"\n  Mean ratio comparison:")
            print(f"    vs π/e = {PI/E:.6f}:  error = {percent_error(mean_ratio, PI/E):.4f}%")
            print(f"    vs φ/π = {PHI/PI:.6f}:  error = {percent_error(mean_ratio, PHI/PI):.4f}%")
            print(f"    vs √2 = {SQRT2:.6f}:  error = {percent_error(mean_ratio, SQRT2):.4f}%")

            return {
                "mean_ratio": mean_ratio,
                "std_ratio": std_ratio,
                "max_ratio": max_ratio,
                "curved_fraction": curved_fraction,
                "k_neighbors": result.k_neighbors,
            }
        else:
            print("  Warning: No valid distance pairs found")
            return {"error": "No valid distance pairs"}

    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def analyze_local_dimension(matrix: np.ndarray) -> dict:
    """Compute local dimension variation across the signal."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain._backend import get_default_backend

    print(f"\n{'=' * 70}")
    print("LOCAL DIMENSION MAPPING")
    print(f"{'=' * 70}")

    backend = get_default_backend()
    estimator = IntrinsicDimension(backend=backend)

    try:
        arr = backend.array(matrix.astype(np.float32))
        local_map = estimator.local_dimension_map(arr)

        print(f"\n  Modal dimension: {local_map.modal_dimension:.6f}")
        print(f"  Mean dimension: {local_map.mean_dimension:.6f}")
        print(f"  Std dimension: {local_map.std_dimension:.6f}")
        print(f"  k_neighbors: {local_map.k_neighbors}")
        print(f"  Deficient points: {len(local_map.deficient_indices)}")

        # Get the dimension array
        dims = np.array(local_map.dimensions)
        valid_dims = dims[~np.isnan(dims)]

        if len(valid_dims) > 0:
            print(f"\n  Local dimension statistics:")
            print(f"    Min: {np.min(valid_dims):.6f}")
            print(f"    Max: {np.max(valid_dims):.6f}")
            print(f"    Median: {np.median(valid_dims):.6f}")

            # Check if modal dimension is close to π
            print(f"\n  Modal dimension comparison:")
            print(f"    vs π = {PI:.6f}:  error = {percent_error(local_map.modal_dimension, PI):.4f}%")

        return {
            "modal_dimension": local_map.modal_dimension,
            "mean_dimension": local_map.mean_dimension,
            "std_dimension": local_map.std_dimension,
            "deficient_count": len(local_map.deficient_indices),
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def analyze_spectral_structure(matrix: np.ndarray) -> dict:
    """Analyze singular value structure for geometric constants."""
    print(f"\n{'=' * 70}")
    print("SPECTRAL ANALYSIS")
    print(f"{'=' * 70}")

    # SVD decomposition
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    print(f"\n  Singular values (top 10):")
    for i, sv in enumerate(S[:10]):
        print(f"    S[{i}] = {sv:.6f}")

    # Analyze consecutive ratios
    print(f"\n  Consecutive ratios S[i]/S[i+1]:")
    ratios = []
    for i in range(min(10, len(S)-1)):
        if S[i+1] > 1e-10:
            ratio = S[i] / S[i+1]
            ratios.append(ratio)

            # Check against geometric constants
            best_match = ""
            best_error = 100
            for name, val in [("π", PI), ("e", E), ("φ", PHI), ("√2", SQRT2), ("2", 2), ("3", 3)]:
                err = percent_error(ratio, val)
                if err < best_error:
                    best_error = err
                    best_match = f"{name} (error {err:.2f}%)"

            print(f"    S[{i}]/S[{i+1}] = {ratio:.6f}  closest: {best_match}")

    # Participation ratio from singular values
    S_sq = S ** 2
    participation_ratio = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)

    print(f"\n  Spectral participation ratio: {participation_ratio:.6f}")
    print(f"    vs π = {PI:.6f}:  error = {percent_error(participation_ratio, PI):.4f}%")

    # Energy concentration
    total_energy = np.sum(S_sq)
    cumulative_energy = np.cumsum(S_sq) / total_energy

    # How many components for 90%, 95%, 99% energy?
    n_90 = np.searchsorted(cumulative_energy, 0.90) + 1
    n_95 = np.searchsorted(cumulative_energy, 0.95) + 1
    n_99 = np.searchsorted(cumulative_energy, 0.99) + 1

    print(f"\n  Energy concentration:")
    print(f"    Components for 90% energy: {n_90}")
    print(f"    Components for 95% energy: {n_95}")
    print(f"    Components for 99% energy: {n_99}")

    return {
        "top_singular_values": list(S[:10]),
        "consecutive_ratios": ratios,
        "participation_ratio": participation_ratio,
        "participation_pi_error": percent_error(participation_ratio, PI),
        "n_90_energy": n_90,
        "n_95_energy": n_95,
    }


def analyze_peak_region(full_matrix: np.ndarray) -> dict:
    """Analyze the 6EQUJ5 peak region specifically."""
    print(f"\n{'=' * 70}")
    print("6EQUJ5 PEAK REGION ANALYSIS")
    print(f"{'=' * 70}")

    # The peak is in column 2 (0-indexed: column 1), rows 58-64 (0-indexed: 57-63)
    # But let's look at a broader region around the peak
    peak_rows = slice(55, 70)  # Rows around the peak
    peak_cols = slice(0, 10)   # First 10 frequency channels

    peak_region = full_matrix[peak_rows, peak_cols]

    print(f"\n  Peak region shape: {peak_region.shape}")
    print(f"  Peak region sum: {np.sum(peak_region):.2f}")
    print(f"  Peak region max: {np.max(peak_region):.2f}")

    # Extract the exact 6EQUJ5 sequence
    # Column 1 (second column), rows 58-63 (0-indexed: 57-62)
    peak_sequence = full_matrix[57:63, 1]
    print(f"\n  6EQUJ5 sequence (column 1): {peak_sequence}")
    print(f"  Sum: {np.sum(peak_sequence)}")

    # Compare to the famous [6, 14, 26, 30, 19, 5]
    expected = np.array([6, 14, 26, 30, 19, 5])
    if len(peak_sequence) == len(expected):
        print(f"  Expected: {expected}")
        print(f"  Match: {np.allclose(peak_sequence, expected)}")

    # Analyze the peak region as a manifold
    if peak_region.shape[0] >= 4:  # Need at least 4 points for TwoNN
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
        estimator = IntrinsicDimension()

        try:
            result = estimator.compute(peak_region.astype(np.float32), with_ci=False)
            print(f"\n  Peak region intrinsic dimension: {result.intrinsic_dimension:.6f}")
            print(f"    vs π:  error = {percent_error(result.intrinsic_dimension, PI):.4f}%")
        except Exception as e:
            print(f"  Could not compute peak ID: {e}")

    return {
        "peak_sequence": list(peak_sequence),
        "peak_sum": float(np.sum(peak_sequence)),
    }


def main():
    print("=" * 70)
    print("WOW! SIGNAL MANIFOLD ANALYSIS")
    print("Applying LLM-style high-dimensional geometry to the 1977 signal")
    print("=" * 70)

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    intensity_matrix, metadata = load_wow_signal()
    print(f"  Signal matrix shape: {intensity_matrix.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(intensity_matrix)}")
    print(f"  Max intensity: {np.max(intensity_matrix)}")
    print(f"  Sum of all intensities: {np.sum(intensity_matrix):.2f}")

    results = {}

    # 1. TwoNN intrinsic dimension (time view: 82 points in 50D)
    results["time_view_id"] = analyze_intrinsic_dimension(
        intensity_matrix, "Time View (82 points in 50D frequency space)"
    )

    # 2. TwoNN intrinsic dimension (frequency view: 50 points in 82D)
    results["freq_view_id"] = analyze_intrinsic_dimension(
        intensity_matrix.T, "Frequency View (50 points in 82D time space)"
    )

    # 3. Effective rank analysis
    results["effective_rank"] = analyze_effective_rank(intensity_matrix)

    # 4. Local dimension mapping
    results["local_dimension"] = analyze_local_dimension(intensity_matrix)

    # 5. Geodesic distance structure
    results["geodesic"] = analyze_geodesic_structure(intensity_matrix)

    # 6. Spectral analysis
    results["spectral"] = analyze_spectral_structure(intensity_matrix)

    # 7. Peak region analysis
    results["peak"] = analyze_peak_region(intensity_matrix)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: π CONNECTION")
    print("=" * 70)

    pi_matches = []

    for name, data in results.items():
        if isinstance(data, dict) and "pi_error" in str(data):
            for key, val in data.items():
                if "pi_error" in key.lower() and isinstance(val, (int, float)) and val < 10:
                    pi_matches.append((name, key, val))

    if pi_matches:
        print("\n  Measurements within 10% of π:")
        for name, key, error in sorted(pi_matches, key=lambda x: x[2]):
            print(f"    {name}.{key}: {error:.4f}% error")
    else:
        print("\n  No measurements within 10% of π found in primary metrics.")

    # Check spectral participation ratio specifically
    if "spectral" in results and "participation_pi_error" in results["spectral"]:
        err = results["spectral"]["participation_pi_error"]
        pr = results["spectral"]["participation_ratio"]
        print(f"\n  Spectral participation ratio: {pr:.6f}")
        print(f"  Error from π: {err:.4f}%")
        if err < 5:
            print("  *** CLOSE TO π! ***")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
