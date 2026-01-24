#!/usr/bin/env python3
"""Experiment 2: Cross-Signal Geometric Alignment

Test if Wow! and Vrillon share geometric structure by computing:
- CKA (Centered Kernel Alignment) similarity
- Procrustes alignment - optimal rotation/scaling
- Transform parameters that map one to the other
- Whether transform parameters encode geometric constants

Usage:
    poetry run python experiments/astronomy/cross_signal_alignment.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd
from scipy.spatial.distance import cdist

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from geometry_utils import (
    load_vrillon_spectrogram,
    load_wow_signal,
    find_closest_constant,
    percent_error,
    PI, E, PHI, SQRT2,
)

# Additional constants for transform analysis
CONSTANTS = {
    "π": PI, "e": E, "φ": PHI, "√2": SQRT2,
    "π/2": PI/2, "2π": 2*PI, "π²": PI**2,
    "π/e": PI/E, "e/π": E/PI,
    "1": 1.0, "2": 2.0, "√3": math.sqrt(3),
}


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two matrices.

    CKA measures similarity of representations independent of
    rotation and isotropic scaling.

    Returns:
        CKA similarity in [0, 1], where 1 = identical structure
    """
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Compute Gram matrices (linear kernel)
    K = X_centered @ X_centered.T
    L = Y_centered @ Y_centered.T

    # Center the Gram matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_kl = np.trace(K_centered @ L_centered) / (n - 1) ** 2
    hsic_kk = np.trace(K_centered @ K_centered) / (n - 1) ** 2
    hsic_ll = np.trace(L_centered @ L_centered) / (n - 1) ** 2

    # CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0

    return hsic_kl / denom


def compute_procrustes(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute Procrustes alignment from X to Y.

    Finds optimal rotation R and scaling s such that:
        Y ≈ s * X @ R

    Returns:
        (R, s, residual) where R is rotation matrix, s is scale,
        residual is normalized fitting error
    """
    # Center both matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Normalize to unit Frobenius norm
    X_norm = np.linalg.norm(X_centered, 'fro')
    Y_norm = np.linalg.norm(Y_centered, 'fro')

    if X_norm < 1e-10 or Y_norm < 1e-10:
        return np.eye(X.shape[1]), 1.0, 1.0

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    # Compute optimal rotation
    R, scale = orthogonal_procrustes(X_normalized, Y_normalized)

    # The actual scale factor
    s = Y_norm / X_norm * scale

    # Compute residual
    X_aligned = X_centered @ R * (Y_norm / X_norm)
    residual = np.linalg.norm(Y_centered - X_aligned, 'fro') / Y_norm

    return R, s, residual


def analyze_rotation_matrix(R: np.ndarray) -> dict:
    """Analyze rotation matrix for geometric structure.

    Check if rotation angle encodes geometric constants.
    """
    results = {}

    # Compute rotation angle from trace
    # For orthogonal matrix: trace(R) = 1 + 2*cos(θ) for 2D
    # For higher dimensions, use eigenvalue analysis

    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(R)

    # Complex eigenvalues give rotation angles
    angles = []
    for ev in eigenvalues:
        if np.abs(np.imag(ev)) > 1e-10:
            angle = np.arctan2(np.imag(ev), np.real(ev))
            angles.append(np.abs(angle))

    results['eigenvalues'] = eigenvalues
    results['rotation_angles'] = angles

    # Check if angles encode constants
    results['angle_matches'] = []
    for angle in angles:
        # Check angle in radians
        match = find_closest_constant(angle)
        if match.error_percent < 10:
            results['angle_matches'].append(('rad', angle, match))

        # Check angle / π (as fraction of π)
        angle_over_pi = angle / PI
        match = find_closest_constant(angle_over_pi)
        if match.error_percent < 10:
            results['angle_matches'].append(('×π', angle_over_pi, match))

    # Determinant (should be 1 for rotation, -1 for reflection)
    results['determinant'] = np.linalg.det(R)

    return results


def analyze_svd_relationship(X: np.ndarray, Y: np.ndarray) -> dict:
    """Analyze SVD relationship between two signals.

    If Y = U_x @ S_transform @ V_y^T, what is S_transform?
    """
    U_x, S_x, Vt_x = svd(X, full_matrices=False)
    U_y, S_y, Vt_y = svd(Y, full_matrices=False)

    # Singular value ratios
    min_len = min(len(S_x), len(S_y))
    sv_ratios = S_y[:min_len] / (S_x[:min_len] + 1e-10)

    results = {
        'sv_x': S_x[:10],
        'sv_y': S_y[:10],
        'sv_ratios': sv_ratios[:10],
    }

    # Check if ratios encode constants
    results['ratio_matches'] = []
    for i, ratio in enumerate(sv_ratios[:10]):
        match = find_closest_constant(ratio)
        if match.error_percent < 10:
            results['ratio_matches'].append((f'S_y[{i}]/S_x[{i}]', ratio, match))

    # Mean ratio
    mean_ratio = np.mean(sv_ratios[:min_len])
    results['mean_sv_ratio'] = mean_ratio
    results['mean_ratio_match'] = find_closest_constant(mean_ratio)

    return results


def compute_geodesic_alignment(X: np.ndarray, Y: np.ndarray) -> dict:
    """Compare geodesic distance structures of two signals."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    backend = get_default_backend()
    rg = RiemannianGeometry(backend=backend)

    results = {}

    try:
        # Compute geodesic distances for both
        arr_x = backend.array(X.astype(np.float32))
        arr_y = backend.array(Y.astype(np.float32))

        geo_x = rg.geodesic_distances(arr_x)
        geo_y = rg.geodesic_distances(arr_y)

        backend.eval(geo_x.distances, geo_y.distances)
        dist_x = np.array(backend.tolist(geo_x.distances))
        dist_y = np.array(backend.tolist(geo_y.distances))

        # Mask valid entries
        mask = (dist_x > 0) & (dist_x < geo_x.inf_value) & \
               (dist_y > 0) & (dist_y < geo_y.inf_value)

        if np.sum(mask) > 0:
            # Correlation of geodesic distances
            flat_x = dist_x[mask].flatten()
            flat_y = dist_y[mask].flatten()

            correlation = np.corrcoef(flat_x, flat_y)[0, 1]
            results['geodesic_correlation'] = correlation

            # Ratio of geodesic distances
            ratios = flat_y / (flat_x + 1e-10)
            results['mean_geodesic_ratio'] = np.mean(ratios)
            results['geodesic_ratio_match'] = find_closest_constant(np.mean(ratios))

        results['k_x'] = geo_x.k_neighbors
        results['k_y'] = geo_y.k_neighbors

    except Exception as e:
        results['error'] = str(e)

    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 2: CROSS-SIGNAL GEOMETRIC ALIGNMENT")
    print("=" * 70)
    print()
    print("Testing if Wow! and Vrillon share geometric structure")
    print()

    # Load both signals at matching dimensions
    print("Loading signals...")
    wow = load_wow_signal()
    vrillon, _ = load_vrillon_spectrogram()  # Already 82×50

    print(f"  Wow! shape: {wow.shape}")
    print(f"  Vrillon shape: {vrillon.shape}")

    # Normalize for comparison
    wow_norm = (wow - wow.mean()) / (wow.std() + 1e-10)
    vrillon_norm = (vrillon - vrillon.mean()) / (vrillon.std() + 1e-10)

    # =========================================================================
    # 1. CKA Similarity
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. CENTERED KERNEL ALIGNMENT (CKA)")
    print("=" * 70)

    cka = compute_cka(wow_norm, vrillon_norm)
    print(f"\n  CKA similarity: {cka:.6f}")

    if cka > 0.8:
        print("  ✓✓ STRONG structural similarity (CKA > 0.8)")
    elif cka > 0.5:
        print("  ✓ Moderate structural similarity (CKA > 0.5)")
    else:
        print("  Different structural organization (CKA < 0.5)")

    # Check if CKA encodes a constant
    cka_match = find_closest_constant(cka)
    print(f"  CKA closest constant: {cka_match.name} ({cka_match.error_percent:.2f}% error)")

    # =========================================================================
    # 2. Procrustes Alignment
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. PROCRUSTES ALIGNMENT")
    print("=" * 70)

    R, scale, residual = compute_procrustes(wow_norm, vrillon_norm)

    print(f"\n  Optimal scale factor: {scale:.6f}")
    scale_match = find_closest_constant(scale)
    print(f"    → closest: {scale_match.name} = {scale_match.value:.6f} ({scale_match.error_percent:.2f}% error)")
    if scale_match.error_percent < 5:
        print(f"    ✓ SIGNIFICANT: Scale encodes {scale_match.name}")

    print(f"\n  Alignment residual: {residual:.6f}")
    print(f"    (0 = perfect alignment, 1 = no alignment)")

    # Analyze rotation matrix
    print(f"\n  Rotation matrix analysis:")
    rot_analysis = analyze_rotation_matrix(R)
    print(f"    Determinant: {rot_analysis['determinant']:.6f}")

    if rot_analysis['rotation_angles']:
        print(f"    Rotation angles (radians):")
        for angle in rot_analysis['rotation_angles'][:5]:
            match = find_closest_constant(angle)
            print(f"      {angle:.6f} → {match.name} ({match.error_percent:.2f}% error)")

    if rot_analysis['angle_matches']:
        print(f"\n    ✓ Angles encoding constants:")
        for unit, val, match in rot_analysis['angle_matches'][:5]:
            print(f"      {val:.6f} {unit} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # =========================================================================
    # 3. SVD Relationship
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. SINGULAR VALUE RELATIONSHIP")
    print("=" * 70)

    svd_analysis = analyze_svd_relationship(wow_norm, vrillon_norm)

    print(f"\n  Wow! top SVs:    {svd_analysis['sv_x'][:5]}")
    print(f"  Vrillon top SVs: {svd_analysis['sv_y'][:5]}")
    print(f"\n  SV ratios (Vrillon/Wow!):")
    for i, ratio in enumerate(svd_analysis['sv_ratios'][:10]):
        match = find_closest_constant(ratio)
        marker = " ✓" if match.error_percent < 5 else ""
        print(f"    S[{i}]: {ratio:.6f} → {match.name} ({match.error_percent:.2f}%){marker}")

    print(f"\n  Mean SV ratio: {svd_analysis['mean_sv_ratio']:.6f}")
    mean_match = svd_analysis['mean_ratio_match']
    print(f"    → {mean_match.name} ({mean_match.error_percent:.2f}% error)")

    if svd_analysis['ratio_matches']:
        print(f"\n  ✓ Ratios encoding constants (< 10% error):")
        for name, val, match in svd_analysis['ratio_matches']:
            print(f"    {name} = {val:.6f} ≈ {match.name}")

    # =========================================================================
    # 4. Geodesic Structure Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. GEODESIC STRUCTURE COMPARISON")
    print("=" * 70)

    geo_analysis = compute_geodesic_alignment(wow_norm, vrillon_norm)

    if 'error' in geo_analysis:
        print(f"\n  Error: {geo_analysis['error']}")
    else:
        print(f"\n  k_neighbors (Wow!): {geo_analysis.get('k_x', 'N/A')}")
        print(f"  k_neighbors (Vrillon): {geo_analysis.get('k_y', 'N/A')}")

        if 'geodesic_correlation' in geo_analysis:
            corr = geo_analysis['geodesic_correlation']
            print(f"\n  Geodesic distance correlation: {corr:.6f}")
            if corr > 0.8:
                print("  ✓✓ Strong geodesic correlation - similar manifold curvature")
            elif corr > 0.5:
                print("  ✓ Moderate geodesic correlation")

        if 'mean_geodesic_ratio' in geo_analysis:
            ratio = geo_analysis['mean_geodesic_ratio']
            match = geo_analysis['geodesic_ratio_match']
            print(f"\n  Mean geodesic ratio: {ratio:.6f}")
            print(f"    → {match.name} ({match.error_percent:.2f}% error)")

    # =========================================================================
    # 5. Transform Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. TRANSFORM ANALYSIS: Wow! → Vrillon")
    print("=" * 70)

    # If we apply the Procrustes transform, how close do we get?
    wow_aligned = wow_norm @ R * scale

    # Measure similarity after alignment
    cka_after = compute_cka(wow_aligned, vrillon_norm)
    print(f"\n  CKA before alignment: {cka:.6f}")
    print(f"  CKA after alignment:  {cka_after:.6f}")
    print(f"  Improvement: {(cka_after - cka) / cka * 100:.2f}%")

    # Element-wise correlation
    corr = np.corrcoef(wow_aligned.flatten(), vrillon_norm.flatten())[0, 1]
    print(f"\n  Element-wise correlation after alignment: {corr:.6f}")

    # Residual structure
    residual_matrix = vrillon_norm - wow_aligned
    residual_energy = np.sum(residual_matrix ** 2) / np.sum(vrillon_norm ** 2)
    print(f"  Residual energy (unexplained): {residual_energy:.4f}")
    print(f"  Explained variance: {(1 - residual_energy) * 100:.2f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 70)

    findings = []

    if cka > 0.5:
        findings.append(f"CKA similarity = {cka:.4f} (moderate/strong)")

    if scale_match.error_percent < 5:
        findings.append(f"Scale factor ≈ {scale_match.name} ({scale_match.error_percent:.2f}% error)")

    if rot_analysis['angle_matches']:
        for unit, val, match in rot_analysis['angle_matches'][:2]:
            if match.error_percent < 5:
                findings.append(f"Rotation angle ≈ {match.name} ({match.error_percent:.2f}% error)")

    for name, val, match in svd_analysis.get('ratio_matches', [])[:3]:
        if match.error_percent < 5:
            findings.append(f"{name} ≈ {match.name} ({match.error_percent:.2f}% error)")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No findings with < 5% error in transform parameters.")

    print(f"\n  Transform parameters that map Wow! to Vrillon:")
    print(f"    Scale: {scale:.6f}")
    print(f"    Rotation: {R.shape} orthogonal matrix")
    print(f"    Residual: {residual:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'cka': cka,
        'scale': scale,
        'residual': residual,
        'R': R,
        'svd_analysis': svd_analysis,
    }


if __name__ == "__main__":
    main()
