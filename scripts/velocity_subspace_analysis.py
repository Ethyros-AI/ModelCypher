#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Velocity Subspace Analysis
"""
Velocity Subspace Analysis

We found that ALL movement through the network lives in ~11 dimensions.

Questions:
1. Are these 11 dimensions THE SAME at each layer?
2. How do they relate to the MLP active dimensions?
3. Can we express the trajectory as: initial_state + sum(alpha_i * v_i)?

If the velocity subspace is SHARED across layers, we can represent
the entire trajectory with just 11 vectors + per-layer scalars.

Usage:
    python velocity_subspace_analysis.py --model /path/to/model
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


def get_all_velocities(model: Any, tokenizer: Any, concepts: list[str]) -> tuple:
    """Get velocities (deltas) at each layer for all concepts.

    Returns:
        velocities: dict[word -> [n_layers, hidden_dim]]
        trajectories: dict[word -> [n_layers+1, hidden_dim]]
    """
    import mlx.core as mx

    trajectories = {}
    velocities = {}

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        traj = []

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)
        traj.append(np.array(h[0, -1, :].astype(mx.float32)))

        for layer in model.model.layers:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
            traj.append(np.array(h[0, -1, :].astype(mx.float32)))

        traj = np.stack(traj)
        trajectories[word] = traj
        velocities[word] = np.diff(traj, axis=0)

    return velocities, trajectories


def find_global_velocity_basis(velocities: dict, n_components: int = 11) -> tuple:
    """Find the global velocity subspace across ALL layers.

    Returns:
        basis: [hidden_dim, n_components] - the shared velocity directions
        explained_variance: fraction of variance explained
    """
    words = list(velocities.keys())
    n_layers = velocities[words[0]].shape[0]

    # Stack ALL velocities
    all_v = []
    for w in words:
        for layer_idx in range(n_layers):
            all_v.append(velocities[w][layer_idx])

    V = np.stack(all_v)  # [n_words * n_layers, hidden_dim]

    # Clean up any NaN/inf
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    mean_v = V.mean(axis=0)
    centered = V - mean_v
    cov = (centered.T @ centered) / len(V)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top components
    basis = eigenvectors[:, :n_components]

    # Explained variance
    total_var = np.sum(eigenvalues[eigenvalues > 0])
    explained = np.sum(eigenvalues[:n_components]) / total_var if total_var > 0 else 0

    return basis, explained, eigenvalues[:n_components]


def compute_layer_velocity_subspace(velocities: dict, layer_idx: int, n_components: int = 11) -> np.ndarray:
    """Find velocity subspace at a SPECIFIC layer."""
    words = list(velocities.keys())

    # Stack velocities at this layer
    V = np.stack([velocities[w][layer_idx] for w in words])
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    mean_v = V.mean(axis=0)
    centered = V - mean_v
    cov = (centered.T @ centered) / len(V)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors[:, :n_components]


def compute_subspace_overlap(basis1: np.ndarray, basis2: np.ndarray) -> float:
    """Compute overlap between two subspaces.

    Uses principal angles: cos(theta) = singular values of B1.T @ B2
    Return average of cos(theta) values.
    """
    # Orthonormalize both bases
    Q1, _ = np.linalg.qr(basis1)
    Q2, _ = np.linalg.qr(basis2)

    # Singular values of Q1.T @ Q2 are cos(principal angles)
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)

    # Average overlap
    return np.mean(s)


def project_trajectory_to_subspace(trajectory: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project trajectory onto velocity subspace.

    Returns coefficients: [n_layers+1, n_components]
    """
    return trajectory @ basis


def reconstruct_trajectory(initial: np.ndarray, coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Reconstruct trajectory from initial point and subspace coefficients."""
    n_points = coeffs.shape[0]
    traj = np.zeros((n_points, basis.shape[0]))
    traj[0] = initial

    for i in range(1, n_points):
        # Velocity in subspace
        delta_coeffs = coeffs[i] - coeffs[i-1]
        delta = delta_coeffs @ basis.T
        traj[i] = traj[i-1] + delta

    return traj


def main():
    parser = argparse.ArgumentParser(description="Velocity subspace analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\nAnalyzing velocity subspace across {n_layers} layers")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Using {len(CONCEPTS)} concepts")

    # Get velocities
    velocities, trajectories = get_all_velocities(model, tokenizer, CONCEPTS)

    # Find global velocity basis
    print("\n" + "=" * 80)
    print("GLOBAL VELOCITY SUBSPACE")
    print("=" * 80)

    global_basis, explained, eigenvalues = find_global_velocity_basis(velocities, n_components=11)
    print(f"Global 11D subspace explains {explained*100:.1f}% of ALL velocity variance")
    print(f"Top eigenvalues: {[f'{v:.3f}' for v in eigenvalues[:5]]}")

    # Check if layer-specific subspaces align with global
    print("\n" + "=" * 80)
    print("LAYER SUBSPACE ALIGNMENT WITH GLOBAL")
    print("=" * 80)
    print(f"{'Layer':>6} | {'Overlap':>10} | Interpretation")
    print("-" * 50)

    for layer_idx in range(n_layers):
        layer_basis = compute_layer_velocity_subspace(velocities, layer_idx, n_components=11)
        overlap = compute_subspace_overlap(global_basis, layer_basis)

        if overlap > 0.9:
            interp = "SAME subspace"
        elif overlap > 0.7:
            interp = "Similar"
        elif overlap > 0.5:
            interp = "Partial overlap"
        else:
            interp = "DIFFERENT"

        print(f"{layer_idx:>6} | {overlap:>10.3f} | {interp}")

    # Test reconstruction accuracy
    print("\n" + "=" * 80)
    print("TRAJECTORY RECONSTRUCTION TEST")
    print("=" * 80)

    reconstruction_errors = []
    for word in CONCEPTS[:5]:  # Test on first 5
        traj = trajectories[word]

        # Project to subspace
        coeffs = project_trajectory_to_subspace(traj, global_basis)

        # Reconstruct
        recon = np.zeros_like(traj)
        recon[0] = traj[0]  # Keep initial exactly
        for i in range(1, len(traj)):
            # Reconstruct using only subspace components
            delta_proj = (coeffs[i] - coeffs[i-1]) @ global_basis.T
            recon[i] = recon[i-1] + delta_proj

        # Measure error
        error = np.mean(np.linalg.norm(traj - recon, axis=1))
        orig_norm = np.mean(np.linalg.norm(traj, axis=1))
        rel_error = error / orig_norm if orig_norm > 0 else 0

        reconstruction_errors.append(rel_error)
        print(f"  {word:>12}: relative error = {rel_error*100:.2f}%")

    avg_error = np.mean(reconstruction_errors) * 100
    print(f"\n  Average relative error: {avg_error:.2f}%")

    # The insight
    print("\n" + "=" * 80)
    print("THE VELOCITY SUBSPACE INSIGHT")
    print("=" * 80)

    print(f"""
1. GLOBAL SUBSPACE:
   - ALL movement lives in ~11 dimensions
   - This is the "through line" - the curve through 1024D space
   - Explained variance: {explained*100:.1f}%

2. LAYER CONSISTENCY:
   - If layer subspaces align with global: the curve is SMOOTH
   - If they differ: the curve changes direction at that layer

3. COMPRESSION IMPLICATION:
   - Full model: {n_layers} × {hidden_dim} = {n_layers * hidden_dim} velocity params per concept
   - Subspace model: 11 basis vectors × {hidden_dim} = {11 * hidden_dim} (shared)
                    + {n_layers} × 11 = {n_layers * 11} coefficients per concept

   Shared basis storage: {11 * hidden_dim} params (one-time)
   Per-concept storage: {n_layers * 11} params (vs {n_layers * hidden_dim})
   Compression: {hidden_dim / 11:.0f}x per concept

4. TRAJECTORY = initial_embedding + Σ(layer_i) velocity_coeffs[i] @ global_basis

   This is the MINIMAL representation of the through line!
""")


if __name__ == "__main__":
    main()
