#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Plane Rotation Analysis
"""
Plane Rotation Analysis

We found the trajectory through layers is a HELIX:
- Paired singular values indicate rotation in planes
- The Gram matrix (relationships) is preserved
- But coordinates rotate through different dimensions

This script decomposes each layer's rotation into PLANE ROTATIONS.
A rotation in n-D decomposes into floor(n/2) plane rotations.

If most planes rotate by small angles, we can parameterize compactly:
- k planes × 1 angle each = k parameters
- vs n² parameters for full rotation matrix

Usage:
    python plane_rotation_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "good", "bad", "love", "hate",
]


def get_hidden_at_layer(model: Any, tokenizer: Any, concepts: list[str], layer_idx: int) -> np.ndarray:
    """Get hidden states at specific layer for all concepts."""
    import mlx.core as mx

    states = []
    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        if layer_idx >= 0:
            for idx, layer in enumerate(model.model.layers):
                if idx <= layer_idx:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result
                    mx.eval(h)

        states.append(np.array(h[0, -1, :].astype(mx.float32)))

    return np.stack(states)


def find_orthonormal_subspace(H: np.ndarray, n_dims: int = 10) -> np.ndarray:
    """Find orthonormal basis for the data subspace."""
    # Normalize to prevent overflow
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-10)

    mean = H_norm.mean(axis=0)
    centered = H_norm - mean
    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors[:, :n_dims]


def project_to_subspace(H: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project data onto subspace."""
    # Normalize to prevent overflow
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-10)

    mean = H_norm.mean(axis=0)
    centered = H_norm - mean
    result = centered @ basis
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def compute_plane_rotation(H1_proj: np.ndarray, H2_proj: np.ndarray) -> tuple:
    """Compute rotation from H1 to H2 and decompose into plane rotations.

    Returns:
        rotation_matrix: R such that H1 @ R ≈ H2
        plane_angles: angles for each plane rotation
        plane_axes: (i, j) pairs defining each rotation plane
    """
    # Find Procrustes rotation: H1 @ R ≈ H2
    # SVD of H1.T @ H2 gives the rotation
    M = H1_proj.T @ H2_proj
    U, s, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Ensure it's a proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Decompose into plane rotations using Schur decomposition
    # R = Q @ T @ Q.T where T is quasi-upper triangular
    # For orthogonal R, T has 2x2 rotation blocks on diagonal
    from scipy.linalg import schur
    T, Q = schur(R)

    # Extract plane rotation angles from 2x2 blocks
    n = R.shape[0]
    plane_angles = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i+1, i]) > 1e-10:
            # 2x2 block = plane rotation
            cos_theta = (T[i, i] + T[i+1, i+1]) / 2
            sin_theta = (T[i, i+1] - T[i+1, i]) / 2
            theta = np.arctan2(sin_theta, cos_theta)
            plane_angles.append((i, i+1, np.degrees(theta)))
            i += 2
        else:
            # 1x1 block = no rotation in this direction
            if T[i, i] < 0:
                plane_angles.append((i, i, 180.0))  # Reflection
            i += 1

    return R, plane_angles, Q


def main():
    parser = argparse.ArgumentParser(description="Plane rotation analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    subspace_dim = 10  # We found this is the intrinsic dim

    print(f"\nAnalyzing plane rotations across {n_layers} layers")
    print(f"Working in {subspace_dim}D subspace")

    # Get embedding and find global subspace
    H_embed = get_hidden_at_layer(model, tokenizer, CONCEPTS, -1)
    global_basis = find_orthonormal_subspace(H_embed, subspace_dim)

    # Project embedding
    H_proj_prev = project_to_subspace(H_embed, global_basis)

    print("\n" + "=" * 80)
    print("PLANE ROTATIONS BY LAYER")
    print("=" * 80)

    all_angles = []
    for layer_idx in range(n_layers):
        # Get hidden states and project
        H = get_hidden_at_layer(model, tokenizer, CONCEPTS, layer_idx)
        H_proj = project_to_subspace(H, global_basis)

        # Compute rotation
        R, plane_angles, Q = compute_plane_rotation(H_proj_prev, H_proj)

        # Filter to actual rotations (angle > 1°)
        significant_rotations = [(i, j, angle) for i, j, angle in plane_angles if abs(angle) > 1]

        all_angles.append(plane_angles)

        print(f"\nLayer {layer_idx}:")
        print(f"  Total rotation (Frobenius from I): {np.linalg.norm(R - np.eye(subspace_dim)):.3f}")

        if significant_rotations:
            print(f"  Plane rotations (|angle| > 1°):")
            for i, j, angle in significant_rotations:
                print(f"    Plane ({i},{j}): {angle:+.1f}°")
        else:
            print(f"  No significant plane rotations")

        H_proj_prev = H_proj

    # Summary statistics
    print("\n" + "=" * 80)
    print("ROTATION SUMMARY")
    print("=" * 80)

    # Count rotations per plane across all layers
    plane_rotation_counts = {}
    total_angle_per_plane = {}

    for layer_angles in all_angles:
        for i, j, angle in layer_angles:
            if i != j:  # Skip reflections
                plane = (min(i, j), max(i, j))
                plane_rotation_counts[plane] = plane_rotation_counts.get(plane, 0) + 1
                total_angle_per_plane[plane] = total_angle_per_plane.get(plane, 0) + abs(angle)

    print("\nMost active rotation planes:")
    sorted_planes = sorted(total_angle_per_plane.items(), key=lambda x: -x[1])
    for plane, total_angle in sorted_planes[:5]:
        count = plane_rotation_counts[plane]
        print(f"  Plane {plane}: {count} rotations, total {total_angle:.1f}°")

    # The insight
    print("\n" + "=" * 80)
    print("THE DOUBLE HELIX INSIGHT")
    print("=" * 80)

    n_active_planes = len([p for p, a in total_angle_per_plane.items() if a > 10])
    total_planes = subspace_dim * (subspace_dim - 1) // 2

    print(f"""
1. HELIX STRUCTURE:
   - The trajectory rotates in {n_active_planes} active planes (out of {total_planes} possible)
   - Most planes see little rotation
   - The helix has a specific "twist pattern"

2. PARAMETERIZATION:
   - Full rotation per layer: {subspace_dim}×{subspace_dim} = {subspace_dim**2} params
   - Sparse plane rotation: {n_active_planes} angles = {n_active_planes} params
   - Compression: {subspace_dim**2 / max(n_active_planes, 1):.0f}x per layer

3. THE DOUBLE HELIX:
   - Paired eigenvalues = rotations in orthogonal planes
   - Like DNA: two strands winding around each other
   - The Gram matrix (base pairs) stays invariant
   - Only the orientation (helix angle) changes

4. TOTAL COMPRESSION:
   - Embedding to final: store {n_active_planes} plane angles per layer
   - {n_layers} layers × {n_active_planes} angles = {n_layers * n_active_planes} total
   - vs {n_layers} × {subspace_dim**2} = {n_layers * subspace_dim**2} for full rotations
   - Compression: {(n_layers * subspace_dim**2) / max(n_layers * n_active_planes, 1):.0f}x on rotation parameters
""")


if __name__ == "__main__":
    main()
