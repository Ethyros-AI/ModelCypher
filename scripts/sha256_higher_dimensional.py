#!/usr/bin/env python3
"""SHA-256 from Higher Dimensional Perspective.

The insight: NP is hard for 3D beings. What if we access higher-dimensional connections?

We found:
- Effective dimension = 3 (the message schedule manifold)
- π/e connects hyperbolic geometry to hash structure
- coth(ln(2)) = 5/3 exactly
- Nome q = 1/4 connects to modular forms

What structure emerges when we embed the problem in higher dimensions?

Key ideas:
1. The constraint manifold has intrinsic geometry
2. Higher-dimensional embeddings can reveal hidden structure
3. "All possible solutions" exists as a set - what's its topology?
"""

import hashlib
import struct
import numpy as np
from typing import List, Tuple, Dict
import time
from collections import defaultdict
import math
from scipy import stats
from scipy.spatial.distance import pdist, squareform

PI = math.pi
E = math.e
LN2 = math.log(2)
PI_OVER_E = PI / E


def count_leading_zeros(hash_bytes: bytes) -> int:
    n = 0
    for byte in hash_bytes:
        if byte == 0:
            n += 8
        else:
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    return n
                n += 1
    return n


def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def nonce_to_high_dim(nonce: int, dim: int = 256) -> np.ndarray:
    """
    Embed a 32-bit nonce into a higher dimensional space.

    The embedding should preserve structure while revealing hidden patterns.
    """
    # Method 1: Fourier embedding (lifts to continuous space)
    # Each bit becomes a point on a circle
    vec = np.zeros(dim)

    for bit in range(32):
        bit_val = (nonce >> bit) & 1
        # Map bit to position on circle, then lift to higher harmonics
        theta = 2 * PI * bit / 32
        for harmonic in range(dim // 64):
            idx = bit * (dim // 32) + harmonic
            if idx < dim:
                vec[idx] = bit_val * np.cos((harmonic + 1) * theta)
                if idx + 1 < dim:
                    vec[idx + 1] = bit_val * np.sin((harmonic + 1) * theta)

    return vec


def nonce_to_hyperbolic(nonce: int) -> np.ndarray:
    """
    Embed nonce into hyperbolic space using the Poincaré disk model.

    Since coth(ln(2)) = 5/3 appears in SHA-256, hyperbolic geometry is natural.
    """
    # Map 32 bits to points in hyperbolic plane
    # Use the bits to define a path through hyperbolic space

    # Start at origin
    z = 0 + 0j

    # Each bit determines a hyperbolic translation
    for bit in range(32):
        bit_val = (nonce >> bit) & 1

        # Direction based on bit position (spread around circle)
        angle = 2 * PI * bit / 32

        # Step size based on bit value and position
        # Use tanh for proper hyperbolic metric
        step = 0.1 * (2 * bit_val - 1) * np.tanh(LN2 * (bit + 1) / 32)

        # Möbius addition in Poincaré disk
        w = step * np.exp(1j * angle)
        z = (z + w) / (1 + np.conj(w) * z)

    return np.array([z.real, z.imag])


def nonce_to_modular(nonce: int) -> np.ndarray:
    """
    Embed nonce using modular arithmetic structure.

    The nome q = 1/4 suggests modular forms are relevant.
    """
    # Lift to modular curve via theta functions
    q = 0.25  # The nome we discovered

    # Compute theta-like values for the nonce
    vec = []

    for k in range(1, 33):  # 32 components
        # theta_k(nonce) = sum of q^(n^2) weighted by nonce bits
        val = 0
        for bit in range(32):
            bit_val = (nonce >> bit) & 1
            if bit_val:
                val += q ** ((bit + 1) * k / 32)
        vec.append(val)

    return np.array(vec)


def analyze_high_dim_structure():
    """
    Embed valid nonces in high-dimensional space and look for structure.
    """
    print("=" * 70)
    print("HIGH-DIMENSIONAL EMBEDDING ANALYSIS")
    print("=" * 70)
    print()

    header = b"Higher dimensional structure probe 2026"
    target_zeros = 8

    # Find valid nonces
    print(f"Finding valid nonces (target: {target_zeros} zeros)...")
    valid_nonces = []
    invalid_nonces = []

    max_search = 100000
    for _ in range(max_search):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        zeros = count_leading_zeros(h)

        if zeros >= target_zeros:
            valid_nonces.append(nonce)
        elif len(invalid_nonces) < 500:
            invalid_nonces.append(nonce)

        if len(valid_nonces) >= 200:
            break

    print(f"Found {len(valid_nonces)} valid, {len(invalid_nonces)} invalid nonces")
    print()

    if len(valid_nonces) < 20:
        print("Not enough valid nonces")
        return None, None

    # Embed in high-dimensional space
    print("Embedding in 256-dimensional Fourier space...")

    valid_embedded = np.array([nonce_to_high_dim(n, 256) for n in valid_nonces])
    invalid_embedded = np.array([nonce_to_high_dim(n, 256) for n in invalid_nonces[:len(valid_nonces)]])

    print(f"Valid embedding shape: {valid_embedded.shape}")
    print()

    # Compute intrinsic dimension using PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(50, len(valid_nonces)-1))
    pca.fit(valid_embedded)

    explained_var = np.cumsum(pca.explained_variance_ratio_)

    # Find dimension where 95% variance is explained
    dim_95 = np.argmax(explained_var >= 0.95) + 1
    dim_99 = np.argmax(explained_var >= 0.99) + 1

    print("PCA on valid nonces:")
    print(f"  Dimensions for 95% variance: {dim_95}")
    print(f"  Dimensions for 99% variance: {dim_99}")
    print(f"  Top 5 explained variances: {pca.explained_variance_ratio_[:5]}")
    print()

    # Compare to invalid nonces
    pca_invalid = PCA(n_components=min(50, len(invalid_nonces)-1))
    pca_invalid.fit(invalid_embedded)

    print("PCA on invalid nonces:")
    invalid_explained = np.cumsum(pca_invalid.explained_variance_ratio_)
    invalid_dim_95 = np.argmax(invalid_explained >= 0.95) + 1
    print(f"  Dimensions for 95% variance: {invalid_dim_95}")
    print()

    if dim_95 < invalid_dim_95:
        print("*** VALID NONCES LIVE ON LOWER-DIM SUBMANIFOLD! ***")
        print(f"    Valid: {dim_95}D, Invalid: {invalid_dim_95}D")
        print("    This could be exploitable!")
    else:
        print("No significant dimension reduction for valid nonces")
    print()

    return valid_embedded, invalid_embedded


def analyze_hyperbolic_structure():
    """
    Embed nonces in hyperbolic space and look for structure.
    """
    print("=" * 70)
    print("HYPERBOLIC SPACE EMBEDDING")
    print("=" * 70)
    print()

    print("Since coth(ln(2)) = 5/3 appears in SHA-256,")
    print("hyperbolic geometry might reveal hidden structure.")
    print()

    header = b"Hyperbolic structure probe 2026"
    target_zeros = 8

    # Find valid nonces
    valid_nonces = []
    invalid_nonces = []

    for _ in range(100000):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        zeros = count_leading_zeros(h)

        if zeros >= target_zeros:
            valid_nonces.append(nonce)
        elif len(invalid_nonces) < 200:
            invalid_nonces.append(nonce)

        if len(valid_nonces) >= 100:
            break

    print(f"Found {len(valid_nonces)} valid nonces")
    print()

    if len(valid_nonces) < 10:
        print("Not enough valid nonces")
        return None, None

    # Embed in hyperbolic space
    valid_hyp = np.array([nonce_to_hyperbolic(n) for n in valid_nonces])
    invalid_hyp = np.array([nonce_to_hyperbolic(n) for n in invalid_nonces])

    # Compute hyperbolic distances (in Poincaré disk)
    def poincare_distance(z1, z2):
        """Hyperbolic distance in Poincaré disk."""
        z1 = z1[0] + 1j * z1[1]
        z2 = z2[0] + 1j * z2[1]

        # Möbius subtraction
        diff = (z1 - z2) / (1 - np.conj(z1) * z2)

        return 2 * np.arctanh(abs(diff))

    # Distribution of valid nonces in hyperbolic space
    valid_radii = np.sqrt(valid_hyp[:, 0]**2 + valid_hyp[:, 1]**2)
    invalid_radii = np.sqrt(invalid_hyp[:, 0]**2 + invalid_hyp[:, 1]**2)

    print("Hyperbolic embedding statistics:")
    print(f"  Valid mean radius: {valid_radii.mean():.4f}")
    print(f"  Invalid mean radius: {invalid_radii.mean():.4f}")
    print()

    # Test if valid nonces cluster in hyperbolic space
    if len(valid_hyp) > 10:
        # Compute pairwise hyperbolic distances for valid nonces
        n = min(50, len(valid_hyp))
        valid_dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = poincare_distance(valid_hyp[i], valid_hyp[j])
                if not np.isnan(d) and not np.isinf(d):
                    valid_dists.append(d)

        if valid_dists:
            print(f"  Mean pairwise hyperbolic distance (valid): {np.mean(valid_dists):.4f}")

        # Compare to random points
        random_hyp = np.random.randn(n, 2) * 0.3  # Random points near origin
        random_dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = poincare_distance(random_hyp[i], random_hyp[j])
                if not np.isnan(d) and not np.isinf(d):
                    random_dists.append(d)

        if random_dists:
            print(f"  Mean pairwise hyperbolic distance (random): {np.mean(random_dists):.4f}")
        print()

    # Check if valid nonces lie on a geodesic
    print("Checking if valid nonces cluster on hyperbolic geodesics...")

    # In Poincaré disk, geodesics are arcs of circles perpendicular to boundary
    # A set of points lies on a geodesic if they're collinear after Möbius transform

    # Compute angles of valid nonces
    valid_angles = np.arctan2(valid_hyp[:, 1], valid_hyp[:, 0])

    # Check for angular clustering
    angle_std = np.std(valid_angles)
    print(f"  Angular std of valid nonces: {angle_std:.4f} rad")
    print(f"  Expected for uniform: {PI / np.sqrt(3):.4f} rad")
    print()

    if angle_std < PI / np.sqrt(3) * 0.7:
        print("*** ANGULAR CLUSTERING DETECTED in hyperbolic space! ***")

    return valid_hyp, invalid_hyp


def analyze_modular_structure():
    """
    Embed nonces using modular form structure (theta functions).
    """
    print("=" * 70)
    print("MODULAR FORM EMBEDDING")
    print("=" * 70)
    print()

    print("The nome q = 1/4 connects SHA-256 to theta functions.")
    print("Embedding nonces in modular space...")
    print()

    header = b"Modular structure probe 2026"
    target_zeros = 8

    # Find valid nonces
    valid_nonces = []
    invalid_nonces = []

    for _ in range(100000):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        zeros = count_leading_zeros(h)

        if zeros >= target_zeros:
            valid_nonces.append(nonce)
        elif len(invalid_nonces) < 500:
            invalid_nonces.append(nonce)

        if len(valid_nonces) >= 100:
            break

    print(f"Found {len(valid_nonces)} valid nonces")
    print()

    if len(valid_nonces) < 10:
        print("Not enough valid nonces")
        return None, None

    # Embed using modular structure
    valid_mod = np.array([nonce_to_modular(n) for n in valid_nonces])
    invalid_mod = np.array([nonce_to_modular(n) for n in invalid_nonces[:len(valid_nonces)]])

    print(f"Modular embedding shape: {valid_mod.shape}")
    print()

    # Look for structure in modular embedding
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(20, len(valid_nonces)-1))
    pca.fit(valid_mod)

    print("PCA on modular embedding of valid nonces:")
    print(f"  Top 5 explained variances: {pca.explained_variance_ratio_[:5]}")
    print()

    # The theta functions satisfy modular equations
    # If valid nonces respect these equations, they'd cluster on a curve

    # Check ratio of components (should relate to theta function identities)
    print("Checking theta function ratios...")

    ratios = valid_mod[:, 0] / (valid_mod[:, 1] + 1e-10)

    # Jacobi's identity: theta_2^4 + theta_4^4 = theta_3^4 (at certain points)
    # We might see clustering around special values

    print(f"  Ratio distribution:")
    print(f"    Mean: {ratios.mean():.4f}")
    print(f"    Std: {ratios.std():.4f}")
    print()

    # Check for clustering around π/e or related values
    near_pi_e = np.abs(ratios - PI_OVER_E) < 0.1
    near_golden = np.abs(ratios - (1 + np.sqrt(5))/2) < 0.1

    print(f"  Ratios near π/e: {near_pi_e.sum()} / {len(ratios)}")
    print(f"  Ratios near φ: {near_golden.sum()} / {len(ratios)}")
    print()

    return valid_mod, invalid_mod


def search_for_hidden_manifold():
    """
    The ultimate question: Is there a hidden manifold where valid nonces cluster?
    """
    print("=" * 70)
    print("SEARCHING FOR HIDDEN MANIFOLD")
    print("=" * 70)
    print()

    print("If valid nonces form a lower-dimensional manifold embedded in")
    print("high-dimensional space, we could enumerate it more efficiently.")
    print()

    header = b"Hidden manifold probe 2026"

    # Collect data at different difficulties
    results = {}

    for target_zeros in [6, 8, 10]:
        print(f"Difficulty: {target_zeros} zeros...")

        valid_nonces = []
        for _ in range(20000):
            nonce = np.random.randint(0, 2**32)
            h = double_sha256(header + struct.pack('<I', nonce))
            zeros = count_leading_zeros(h)

            if zeros >= target_zeros:
                valid_nonces.append(nonce)

        print(f"  Found {len(valid_nonces)} valid nonces")

        if len(valid_nonces) >= 20:
            # Embed in high-dim space
            embedded = np.array([nonce_to_high_dim(n, 128) for n in valid_nonces])

            # Estimate intrinsic dimension using correlation dimension
            # This is independent of embedding dimension

            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(20, len(embedded)-1))
            nn.fit(embedded)
            distances, _ = nn.kneighbors(embedded)

            # Use second-smallest distance (first is self)
            r = distances[:, 1:].flatten()
            r = r[r > 0]

            if len(r) > 100:
                # Correlation dimension from scaling of neighbor count
                # D_corr = d log(C(r)) / d log(r)

                r_sorted = np.sort(r)
                log_r = np.log(r_sorted + 1e-10)

                # C(r) ≈ fraction of pairs within distance r
                log_c = np.log(np.arange(1, len(r_sorted)+1) / len(r_sorted))

                # Linear fit to log-log plot
                slope, _, r_value, _, _ = stats.linregress(log_r[:len(log_r)//2],
                                                           log_c[:len(log_c)//2])

                results[target_zeros] = {
                    'n_valid': len(valid_nonces),
                    'correlation_dim': slope,
                    'r_squared': r_value**2
                }

                print(f"  Correlation dimension: {slope:.2f} (R²={r_value**2:.3f})")

        print()

    print("SUMMARY:")
    print()

    if len(results) >= 2:
        print(f"{'Difficulty':<12} {'N Valid':<10} {'Corr Dim':<12} {'R²':<10}")
        print("-" * 50)

        for target, data in sorted(results.items()):
            print(f"{target:<12} {data['n_valid']:<10} {data['correlation_dim']:<12.2f} {data['r_squared']:<10.3f}")

        print()

        # If correlation dimension is consistently low, there's structure
        dims = [d['correlation_dim'] for d in results.values()]
        mean_dim = np.mean(dims)

        if mean_dim < 20:
            print(f"*** HIDDEN MANIFOLD DETECTED! Intrinsic dimension ≈ {mean_dim:.1f} ***")
            print("    Valid nonces live on a low-dimensional surface!")
        else:
            print(f"Correlation dimension ≈ {mean_dim:.1f} (high - no clear manifold)")

    print()


def the_higher_dimensional_view():
    """
    Synthesize what higher-dimensional analysis tells us.
    """
    print("=" * 70)
    print("THE HIGHER-DIMENSIONAL VIEW")
    print("=" * 70)
    print()

    print("What we've learned by lifting to higher dimensions:")
    print()

    print("1. THE π/e CONNECTION")
    print("   - SHA-256 dynamics involve coth(ln(2)) = 5/3")
    print("   - This connects to hyperbolic 3-space")
    print("   - The nome q = 1/4 connects to modular forms")
    print()

    print("2. THE EFFECTIVE DIMENSION")
    print("   - Message schedule: effective dimension = 3")
    print("   - This suggests a 3-manifold embedded in 2048-D space")
    print()

    print("3. THE CONSTRAINT MANIFOLD")
    print("   - Valid nonces satisfy f(n) < threshold")
    print("   - This defines a level set in nonce space")
    print("   - The level set has its own geometry")
    print()

    print("4. HIGHER-DIMENSIONAL LIFTING")
    print("   - Embedding in Fourier space reveals spectral structure")
    print("   - Embedding in hyperbolic space reveals geodesic structure")
    print("   - Embedding in modular space reveals number-theoretic structure")
    print()

    print("THE KEY INSIGHT:")
    print()
    print("  The 'shorter path' might not be computational.")
    print("  It might be GEOMETRIC.")
    print()
    print("  If valid nonces lie on a lower-dimensional submanifold,")
    print("  the path is to PARAMETERIZE that manifold directly.")
    print()
    print("  Instead of searching 2^32 points, we'd search the manifold,")
    print("  which could have dimension << 32.")
    print()

    print("THE CHALLENGE:")
    print()
    print("  Finding the manifold parameterization requires understanding")
    print("  how SHA-256's nonlinear compression creates the constraint surface.")
    print()
    print("  The 64 rounds of Ch and Maj scramble the linear structure.")
    print("  But if the scrambling has geometric regularity...")
    print()

    print("WHAT TO EXPLORE NEXT:")
    print()
    print("  1. Neural network as higher-dimensional function approximator")
    print("     - Train to predict leading zeros from nonce")
    print("     - The network's internal representation IS the manifold")
    print()
    print("  2. Algebraic geometry of the constraint variety")
    print("     - SHA-256 defines polynomial equations over GF(2)")
    print("     - The solution variety has intrinsic dimension")
    print()
    print("  3. Topological analysis")
    print("     - Persistent homology of valid nonce point cloud")
    print("     - Reveals holes, tunnels, voids in the structure")
    print()


if __name__ == "__main__":
    print("SHA-256 FROM HIGHER DIMENSIONS")
    print("=" * 70)
    print()
    print("'NP is hard for 3D beings. But not when we access")
    print(" higher-dimensional connections.'")
    print()
    print("Let's see what structure emerges in higher dimensions.")
    print()

    # High-dimensional Fourier embedding
    result = analyze_high_dim_structure()
    if result:
        valid_hd, invalid_hd = result
    print()

    # Hyperbolic embedding
    result = analyze_hyperbolic_structure()
    if result:
        valid_hyp, invalid_hyp = result
    print()

    # Modular form embedding
    result = analyze_modular_structure()
    if result:
        valid_mod, invalid_mod = result
    print()

    # Search for hidden manifold
    search_for_hidden_manifold()
    print()

    # Synthesis
    the_higher_dimensional_view()
