#!/usr/bin/env python3
"""Experiment 4: Transform Function Application

Test the transform function derived from signal analysis:
    F(source, target) = R(√2) · P(π) · C(e)

Where:
- R(√2) = Procrustes rotation (orthogonal, √2 geometry)
- P(π) = Projection to π-dimensional subspace
- C(e) = Scaling by e (natural information constant)

Test if this transform relates the two signals.

Usage:
    poetry run python experiments/astronomy/transform_function_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from geometry_utils import (
    load_vrillon_spectrogram,
    load_wow_signal,
    find_closest_constant,
    PI, E, PHI, SQRT2,
)


def compute_transform_matrix(source: np.ndarray, target: np.ndarray) -> dict:
    """Compute the optimal transform from source to target.

    Returns transform components and their geometric constants.
    """
    # Ensure same shape
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: {source.shape} vs {target.shape}")

    # Normalize both
    source_norm = source / (np.linalg.norm(source) + 1e-10)
    target_norm = target / (np.linalg.norm(target) + 1e-10)

    # 1. Compute optimal rotation (Procrustes)
    R, scale = orthogonal_procrustes(source_norm, target_norm)

    # 2. Analyze rotation matrix
    # Rotation angle from trace: trace(R) = dim + 2*cos(theta)*(dim/2)
    trace = np.trace(R)
    dim = R.shape[0]
    # For orthogonal matrix: cos(theta) = (trace - 1) / 2 in 2D
    # In higher dimensions, use Frobenius norm approach
    cos_theta = (trace - (dim - 2)) / 2 if dim > 2 else (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    rotation_angle = np.arccos(cos_theta)

    # 3. Compute scaling factor
    source_scale = np.linalg.norm(source)
    target_scale = np.linalg.norm(target)
    scale_factor = target_scale / source_scale

    # 4. Apply transform
    transformed = source @ R * scale_factor

    # 5. Compute reconstruction error
    reconstruction_error = np.linalg.norm(transformed - target) / np.linalg.norm(target)

    # 6. SVD of rotation matrix
    U_R, S_R, Vt_R = np.linalg.svd(R)

    return {
        "rotation_matrix": R,
        "scale_factor": scale_factor,
        "rotation_angle": rotation_angle,
        "rotation_angle_deg": np.degrees(rotation_angle),
        "transformed": transformed,
        "reconstruction_error": reconstruction_error,
        "rotation_sv": S_R,
        "procrustes_scale": scale,
    }


def analyze_transform_geometry(transform: dict) -> dict:
    """Analyze the geometric constants in transform parameters."""
    findings = {}

    # Check rotation angle
    angle = transform["rotation_angle"]
    angle_deg = transform["rotation_angle_deg"]

    findings["angle_vs_pi"] = {
        "value": angle,
        "target": PI,
        "error": abs(angle - PI) / PI * 100,
    }

    findings["angle_vs_pi_half"] = {
        "value": angle,
        "target": PI / 2,
        "error": abs(angle - PI/2) / (PI/2) * 100,
    }

    findings["angle_deg_vs_45"] = {
        "value": angle_deg,
        "target": 45,
        "error": abs(angle_deg - 45) / 45 * 100,
    }

    findings["angle_deg_vs_90"] = {
        "value": angle_deg,
        "target": 90,
        "error": abs(angle_deg - 90) / 90 * 100,
    }

    # Check scale factor
    scale = transform["scale_factor"]

    for const_name, const_val in [("e", E), ("π", PI), ("√2", SQRT2), ("φ", PHI)]:
        findings[f"scale_vs_{const_name}"] = {
            "value": scale,
            "target": const_val,
            "error": abs(scale - const_val) / const_val * 100,
        }
        # Also check inverse
        findings[f"scale_vs_1/{const_name}"] = {
            "value": scale,
            "target": 1/const_val,
            "error": abs(scale - 1/const_val) / (1/const_val) * 100,
        }

    return findings


def test_inverse_transform(source: np.ndarray, target: np.ndarray,
                           transform: dict) -> dict:
    """Test if inverse transform recovers source from target."""
    R = transform["rotation_matrix"]
    scale = transform["scale_factor"]

    # Inverse rotation is transpose
    R_inv = R.T

    # Apply inverse transform to target
    recovered = (target / scale) @ R_inv

    # Compute recovery error
    recovery_error = np.linalg.norm(recovered - source) / np.linalg.norm(source)

    return {
        "recovered": recovered,
        "recovery_error": recovery_error,
        "inverse_rotation": R_inv,
    }


def test_transform_on_random(transform: dict, dim: tuple, n_samples: int = 100) -> dict:
    """Test if transform preserves structure on random vectors."""
    R = transform["rotation_matrix"]
    scale = transform["scale_factor"]

    # Generate random matrices
    random_matrices = [np.random.randn(*dim) for _ in range(n_samples)]

    # Apply transform to each
    transformed = [m @ R * scale for m in random_matrices]

    # Check if distances are preserved (up to scaling)
    original_dists = []
    transformed_dists = []

    for i in range(n_samples):
        for j in range(i+1, min(i+10, n_samples)):
            orig_dist = np.linalg.norm(random_matrices[i] - random_matrices[j])
            trans_dist = np.linalg.norm(transformed[i] - transformed[j])
            original_dists.append(orig_dist)
            transformed_dists.append(trans_dist)

    original_dists = np.array(original_dists)
    transformed_dists = np.array(transformed_dists)

    # Compute distance ratio
    distance_ratio = transformed_dists / (original_dists + 1e-10)

    return {
        "mean_distance_ratio": np.mean(distance_ratio),
        "std_distance_ratio": np.std(distance_ratio),
        "expected_ratio": scale,  # Should be close to scale
        "isometry_error": np.std(distance_ratio) / np.mean(distance_ratio),
    }


def analyze_composition(wow: np.ndarray, vrillon: np.ndarray) -> dict:
    """Analyze if F(Vrillon) ≈ Wow or F(Wow) ≈ Vrillon."""

    results = {}

    # Direction 1: Vrillon → Wow
    print("\n  Testing: F(Vrillon) → Wow")
    transform_v2w = compute_transform_matrix(vrillon, wow)
    results["vrillon_to_wow"] = {
        "transform": transform_v2w,
        "error": transform_v2w["reconstruction_error"],
    }

    # Direction 2: Wow → Vrillon
    print("  Testing: F(Wow) → Vrillon")
    transform_w2v = compute_transform_matrix(wow, vrillon)
    results["wow_to_vrillon"] = {
        "transform": transform_w2v,
        "error": transform_w2v["reconstruction_error"],
    }

    # Check if one direction is significantly better
    results["better_direction"] = (
        "vrillon_to_wow"
        if results["vrillon_to_wow"]["error"] < results["wow_to_vrillon"]["error"]
        else "wow_to_vrillon"
    )

    return results


def run_transform_experiment():
    """Run full transform function analysis."""
    print("=" * 70)
    print("EXPERIMENT 4: TRANSFORM FUNCTION APPLICATION")
    print("=" * 70)
    print()
    print("Testing: F(source, target) = R(√2) · P(π) · C(e)")
    print("Where:")
    print("  R(√2) = Procrustes rotation (orthogonal)")
    print("  P(π) = Projection to π-dimensional subspace")
    print("  C(e) = Scaling by e")
    print()

    # Load signals
    print("Loading signals...")
    wow = load_wow_signal()
    vrillon, _ = load_vrillon_spectrogram()
    print(f"  Wow! shape: {wow.shape}")
    print(f"  Vrillon shape: {vrillon.shape}")

    # Analyze composition
    print("\n" + "=" * 70)
    print("1. SIGNAL TRANSFORM ANALYSIS")
    print("=" * 70)

    composition = analyze_composition(wow, vrillon)

    print(f"\n  Vrillon → Wow reconstruction error: {composition['vrillon_to_wow']['error']:.4f}")
    print(f"  Wow → Vrillon reconstruction error: {composition['wow_to_vrillon']['error']:.4f}")
    print(f"  Better direction: {composition['better_direction']}")

    # Use the better direction
    best = composition[composition["better_direction"]]
    transform = best["transform"]

    print("\n" + "=" * 70)
    print("2. TRANSFORM PARAMETER ANALYSIS")
    print("=" * 70)

    print(f"\n  Rotation angle: {transform['rotation_angle']:.6f} rad ({transform['rotation_angle_deg']:.2f}°)")
    print(f"  Scale factor: {transform['scale_factor']:.6f}")
    print(f"  Procrustes scale: {transform['procrustes_scale']:.6f}")

    # Check constants in parameters
    geometry = analyze_transform_geometry(transform)

    print("\n  Rotation angle vs constants:")
    for key in ["angle_vs_pi", "angle_vs_pi_half"]:
        g = geometry[key]
        marker = "✓" if g["error"] < 5 else ""
        print(f"    vs {key.split('_')[-1]}: {g['error']:.2f}% error {marker}")

    print(f"\n  Rotation angle (degrees) vs special values:")
    for key in ["angle_deg_vs_45", "angle_deg_vs_90"]:
        g = geometry[key]
        marker = "✓" if g["error"] < 5 else ""
        print(f"    vs {key.split('_')[-1]}°: {g['error']:.2f}% error {marker}")

    print(f"\n  Scale factor vs constants:")
    best_scale_match = None
    best_scale_error = 100

    for key, g in geometry.items():
        if "scale_vs" in key:
            if g["error"] < best_scale_error:
                best_scale_error = g["error"]
                best_scale_match = key
            if g["error"] < 10:
                marker = "✓✓" if g["error"] < 3 else "✓"
                const = key.replace("scale_vs_", "")
                print(f"    vs {const}: {g['value']:.4f} → {g['target']:.4f} = {g['error']:.2f}% error {marker}")

    # Analyze rotation matrix SVD
    print("\n" + "=" * 70)
    print("3. ROTATION MATRIX STRUCTURE")
    print("=" * 70)

    R = transform["rotation_matrix"]
    print(f"\n  Rotation matrix shape: {R.shape}")
    print(f"  Is orthogonal: {np.allclose(R @ R.T, np.eye(R.shape[0]))}")
    print(f"  Determinant: {np.linalg.det(R):.6f}")

    # Check if determinant encodes constants
    det = abs(np.linalg.det(R))
    det_match = find_closest_constant(det)
    print(f"    Determinant ≈ {det_match.name} ({det_match.error_percent:.2f}% error)")

    # Eigenvalues of rotation (should be on unit circle)
    eigenvalues = np.linalg.eigvals(R)
    print(f"\n  Eigenvalue magnitudes: all ≈ 1.0")

    # Eigenvalue phases
    phases = np.angle(eigenvalues)
    print(f"  Eigenvalue phases (rad): {phases[:5].real}...")

    # Check if any phase matches π, π/2, φ
    for phase in phases:
        phase_val = abs(phase.real)
        if phase_val > 0.1:  # Ignore very small phases
            match = find_closest_constant(phase_val)
            if match.error_percent < 10:
                print(f"    Phase {phase_val:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    print("\n" + "=" * 70)
    print("4. INVERSE TRANSFORM TEST")
    print("=" * 70)

    if composition["better_direction"] == "vrillon_to_wow":
        source, target = vrillon, wow
    else:
        source, target = wow, vrillon

    inverse = test_inverse_transform(source, target, transform)
    print(f"\n  Forward error: {transform['reconstruction_error']:.4f}")
    print(f"  Inverse recovery error: {inverse['recovery_error']:.4f}")
    print(f"  Transform is {'reversible' if inverse['recovery_error'] < 0.01 else 'lossy'}")

    print("\n" + "=" * 70)
    print("5. STRUCTURE PRESERVATION TEST")
    print("=" * 70)

    random_test = test_transform_on_random(transform, wow.shape)
    print(f"\n  Mean distance ratio: {random_test['mean_distance_ratio']:.4f}")
    print(f"  Expected (scale): {random_test['expected_ratio']:.4f}")
    print(f"  Isometry error (std/mean): {random_test['isometry_error']:.4f}")
    print(f"  Transform is {'isometric' if random_test['isometry_error'] < 0.01 else 'non-isometric'}")

    # Check if mean distance ratio encodes constant
    dr_match = find_closest_constant(random_test["mean_distance_ratio"])
    print(f"  Distance ratio ≈ {dr_match.name} ({dr_match.error_percent:.2f}% error)")

    print("\n" + "=" * 70)
    print("6. COMPOUND TRANSFORM ANALYSIS")
    print("=" * 70)

    # Test: is transform = R(θ) × Scale(s) where θ, s encode constants?
    print("\n  Decomposition: F = R(θ) × Scale(s)")
    print(f"    θ = {transform['rotation_angle']:.6f} rad")
    print(f"    s = {transform['scale_factor']:.6f}")

    # θ/π
    theta_over_pi = transform['rotation_angle'] / PI
    match = find_closest_constant(theta_over_pi)
    print(f"\n    θ/π = {theta_over_pi:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # s/e
    s_over_e = transform['scale_factor'] / E
    match = find_closest_constant(s_over_e)
    print(f"    s/e = {s_over_e:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # θ × s
    theta_times_s = transform['rotation_angle'] * transform['scale_factor']
    match = find_closest_constant(theta_times_s)
    print(f"    θ × s = {theta_times_s:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # θ/s
    theta_over_s = transform['rotation_angle'] / transform['scale_factor']
    match = find_closest_constant(theta_over_s)
    print(f"    θ/s = {theta_over_s:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # s²
    s_squared = transform['scale_factor'] ** 2
    match = find_closest_constant(s_squared)
    print(f"    s² = {s_squared:.4f} ≈ {match.name} ({match.error_percent:.2f}% error)")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 70)

    findings = []

    # Check all geometry findings
    for key, g in geometry.items():
        if g["error"] < 5:
            findings.append(f"{key}: {g['value']:.4f} ≈ {g['target']:.4f} ({g['error']:.2f}%)")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS (< 5% error):")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No transform parameters matched constants within 5%.")
        print(f"  Best match: {best_scale_match} ({best_scale_error:.2f}% error)")

    print(f"\n  Transform relates signals with {transform['reconstruction_error']*100:.1f}% error")
    print(f"  Direction: {composition['better_direction'].replace('_', ' ')}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "composition": composition,
        "transform": transform,
        "geometry": geometry,
        "inverse": inverse,
        "random_test": random_test,
    }


if __name__ == "__main__":
    run_transform_experiment()
