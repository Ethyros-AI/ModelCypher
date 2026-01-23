#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Trajectory Analysis
"""
Trajectory Analysis

The "through line" insight:
- A 1D manifold in 1024D space is a CURVE through all dimensions
- Not about WHICH dimensions, but about the PATH
- The path must maintain invariant relationships (Gram matrix)

This script:
1. Track the trajectory of each concept through layers
2. Find the minimal parameterization of this trajectory
3. Check if trajectories preserve relational structure

Key question: Can we describe the path with fewer parameters
than the full hidden state at each layer?

Usage:
    python trajectory_analysis.py --model /path/to/model
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


# Concepts with known relationships
CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
]


def get_trajectory(
    model: Any,
    tokenizer: Any,
    word: str,
) -> np.ndarray:
    """Get the full trajectory of a word through all layers.

    Returns: [n_layers+1, hidden_dim] array (embedding + each layer)
    """
    import mlx.core as mx

    tokens = tokenizer.encode(word)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    trajectory = []

    # Embedding
    h = model.model.embed_tokens(input_ids)
    mx.eval(h)
    trajectory.append(np.array(h[0, -1, :].astype(mx.float32)))

    # Each layer
    for layer in model.model.layers:
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)
        trajectory.append(np.array(h[0, -1, :].astype(mx.float32)))

    return np.stack(trajectory, axis=0)


def compute_trajectory_velocities(trajectories: dict) -> dict:
    """Compute velocity (delta) at each layer for each concept."""
    velocities = {}
    for word, traj in trajectories.items():
        # velocity[i] = traj[i+1] - traj[i]
        vel = np.diff(traj, axis=0)
        velocities[word] = vel
    return velocities


def analyze_velocity_structure(velocities: dict) -> dict:
    """Analyze the structure of velocities across concepts.

    If velocities are aligned across concepts at a layer,
    it means all concepts move in a similar direction.
    This would enable parameterization by a shared velocity.
    """
    words = list(velocities.keys())
    n_layers = velocities[words[0]].shape[0]

    results = []
    for layer_idx in range(n_layers):
        # Stack velocities at this layer
        V = np.stack([velocities[w][layer_idx] for w in words])

        # Normalize each velocity
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norm = V / (norms + 1e-10)

        # Compute pairwise cosine similarity
        cos_sim = V_norm @ V_norm.T

        # Average alignment (excluding diagonal)
        n = len(words)
        off_diag = cos_sim[~np.eye(n, dtype=bool)]
        avg_alignment = np.mean(off_diag)

        # PCA on velocities to find shared structure
        mean_v = V.mean(axis=0)
        centered = V - mean_v
        cov = (centered.T @ centered) / len(V)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Participation ratio (effective dimensionality of velocity space)
        if len(eigenvalues) > 0:
            pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            pr = 0

        # Average velocity magnitude
        avg_norm = np.mean(norms)

        results.append({
            'layer': layer_idx,
            'avg_alignment': avg_alignment,
            'velocity_dim': pr,
            'avg_magnitude': avg_norm,
        })

    return results


def find_shared_velocity_subspace(velocities: dict, n_components: int = 3) -> dict:
    """Find the shared subspace that captures most velocity variance.

    If all velocities lie in a low-dim subspace, we can represent
    the trajectory with just the projection onto this subspace.
    """
    words = list(velocities.keys())
    n_layers = velocities[words[0]].shape[0]
    hidden_dim = velocities[words[0]].shape[1]

    # Stack all velocities: [n_words * n_layers, hidden_dim]
    all_velocities = []
    for w in words:
        for layer_idx in range(n_layers):
            all_velocities.append(velocities[w][layer_idx])

    V = np.stack(all_velocities)

    # PCA to find shared subspace
    mean_v = V.mean(axis=0)
    centered = V - mean_v
    cov = (centered.T @ centered) / len(V)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained by top components
    total_var = np.sum(eigenvalues[eigenvalues > 0])
    cumsum = np.cumsum(eigenvalues[eigenvalues > 0]) / total_var

    # Find how many dims needed for 95%, 99%
    dims_95 = np.searchsorted(cumsum, 0.95) + 1
    dims_99 = np.searchsorted(cumsum, 0.99) + 1

    return {
        'dims_for_95': dims_95,
        'dims_for_99': dims_99,
        'top_eigenvalues': eigenvalues[:10].tolist(),
        'total_variance': total_var,
        'subspace': eigenvectors[:, :n_components],  # Top directions
    }


def compute_gram_at_layers(trajectories: dict) -> list:
    """Compute Gram matrix (relational structure) at each layer."""
    words = list(trajectories.keys())
    n_layers = trajectories[words[0]].shape[0]

    gram_matrices = []
    for layer_idx in range(n_layers):
        # Hidden states at this layer
        H = np.stack([trajectories[w][layer_idx] for w in words])

        # Normalize
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H_norm = H / (norms + 1e-10)

        # Gram matrix
        G = H_norm @ H_norm.T
        gram_matrices.append(G)

    return gram_matrices


def analyze_gram_evolution(gram_matrices: list) -> list:
    """Track how Gram matrix evolves through layers."""
    results = []

    for i, G in enumerate(gram_matrices):
        if i == 0:
            # Compare to identity (no relationships)
            identity = np.eye(G.shape[0])
            diff_from_identity = np.mean((G - identity) ** 2)
            diff_from_prev = 0
        else:
            diff_from_prev = np.mean((G - gram_matrices[i-1]) ** 2)
            diff_from_identity = np.mean((G - np.eye(G.shape[0])) ** 2)

        # Similarity to embedding Gram
        sim_to_embed = np.corrcoef(G.flatten(), gram_matrices[0].flatten())[0, 1]

        results.append({
            'layer': i,
            'diff_from_prev': diff_from_prev,
            'diff_from_identity': diff_from_identity,
            'sim_to_embed': sim_to_embed,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Trajectory analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\nAnalyzing trajectories through {n_layers} layers")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Using {len(CONCEPTS)} concepts")

    # Get trajectories
    logger.info("Computing trajectories...")
    trajectories = {}
    for word in CONCEPTS:
        trajectories[word] = get_trajectory(model, tokenizer, word)

    # Compute velocities
    velocities = compute_trajectory_velocities(trajectories)

    print("\n" + "=" * 80)
    print("VELOCITY STRUCTURE BY LAYER")
    print("=" * 80)
    print(f"{'Layer':>6} | {'Alignment':>10} | {'Vel Dim':>10} | {'|v|':>10} | Interpretation")
    print("-" * 80)

    velocity_stats = analyze_velocity_structure(velocities)
    for stats in velocity_stats:
        layer = stats['layer']
        align = stats['avg_alignment']
        dim = stats['velocity_dim']
        mag = stats['avg_magnitude']

        # Interpretation
        if align > 0.8:
            interp = "SHARED direction"
        elif align > 0.5:
            interp = "Partly aligned"
        elif dim < 5:
            interp = f"Low-dim ({dim:.0f}D)"
        else:
            interp = "Diverse"

        print(f"{layer:>6} | {align:>10.3f} | {dim:>10.1f} | {mag:>10.1f} | {interp}")

    # Find shared velocity subspace
    print("\n" + "=" * 80)
    print("SHARED VELOCITY SUBSPACE")
    print("=" * 80)

    subspace = find_shared_velocity_subspace(velocities)
    print(f"Dims for 95% velocity variance: {subspace['dims_for_95']}")
    print(f"Dims for 99% velocity variance: {subspace['dims_for_99']}")
    print(f"Top eigenvalues: {[f'{v:.2f}' for v in subspace['top_eigenvalues'][:5]]}")

    # Gram matrix evolution
    print("\n" + "=" * 80)
    print("RELATIONAL STRUCTURE (GRAM) EVOLUTION")
    print("=" * 80)

    gram_matrices = compute_gram_at_layers(trajectories)
    gram_stats = analyze_gram_evolution(gram_matrices)

    print(f"{'Layer':>6} | {'Δ from prev':>12} | {'Sim to embed':>12} | Interpretation")
    print("-" * 70)

    defining_layers = []
    preserving_layers = []

    for stats in gram_stats:
        layer = stats['layer']
        diff = stats['diff_from_prev']
        sim = stats['sim_to_embed']

        if layer == 0:
            interp = "Embedding (reference)"
        elif diff < 0.001:
            interp = "PRESERVES structure"
            preserving_layers.append(layer - 1)  # -1 because layer 0 is embedding
        elif diff < 0.01:
            interp = "Minor change"
        else:
            interp = "TRANSFORMS structure"
            defining_layers.append(layer - 1)

        print(f"{layer:>6} | {diff:>12.4f} | {sim:>12.3f} | {interp}")

    # The key insight
    print("\n" + "=" * 80)
    print("THE THROUGH LINE")
    print("=" * 80)

    print(f"""
The trajectory through layer space:

1. VELOCITY STRUCTURE:
   - Dims for 95% velocity: {subspace['dims_for_95']}
   - This means ALL movement through the network lives in a
     ~{subspace['dims_for_95']}-dimensional subspace

2. STRUCTURE-DEFINING layers: {defining_layers}
   - These CHANGE the relational structure (Gram matrix)
   - Movement here matters for semantics

3. STRUCTURE-PRESERVING layers: {preserving_layers}
   - These MAINTAIN relational structure
   - Movement is just "passing through" - could be compressed

4. THE COMPRESSION INSIGHT:
   - Total params: {n_layers} layers × {hidden_dim} dims = {n_layers * hidden_dim} per concept
   - Through-line params: {subspace['dims_for_95']} velocity dims × {n_layers} layers = {subspace['dims_for_95'] * n_layers}
   - Potential compression: {hidden_dim / subspace['dims_for_95']:.0f}x

   But wait - preserving layers don't need full velocity!
   If {len(preserving_layers)} layers just pass through:
   - Only need: {len(defining_layers)} defining × {subspace['dims_for_95']} dims
   - Plus: {len(preserving_layers)} preserving × 1 dim (just magnitude)
   - Compression: {hidden_dim * n_layers / (len(defining_layers) * subspace['dims_for_95'] + len(preserving_layers)):.0f}x
""")


if __name__ == "__main__":
    main()
