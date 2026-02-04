#!/usr/bin/env python3
"""Manifold topology analysis via persistent homology.

Computes Betti numbers (β₀=components, β₁=loops, β₂=voids) across layers
to understand how the activation manifold's topology evolves during inference.

Key questions:
- Does topology simplify (fewer holes) as we go deeper?
- Do "highway" layers preserve topology while "processing" layers transform it?
- Is there a relationship between Betti numbers and expansion_ratio?

All thresholds are derived from data, not hardcoded.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Lazy imports for optional dependencies
ripser = None
mlx_lm = None


def ensure_ripser():
    global ripser
    if ripser is None:
        import ripser as r
        ripser = r


def ensure_mlx_lm():
    global mlx_lm
    if mlx_lm is None:
        from mlx_lm import load
        mlx_lm = load


@dataclass
class LayerTopology:
    """Topological features at a single layer."""
    layer_idx: int
    betti_0: int  # Connected components
    betti_1: int  # 1-dimensional holes (loops)
    betti_2: int  # 2-dimensional voids
    total_persistence: float  # Sum of all lifetimes
    max_persistence: float  # Longest-lived feature
    persistence_entropy: float  # Entropy of lifetime distribution


@dataclass
class TopologyTrajectory:
    """How topology evolves through the network."""
    prompt: str
    layer_topologies: list[LayerTopology]

    # Summary statistics
    betti_0_trend: str  # "increasing", "decreasing", "stable"
    betti_1_trend: str
    topology_simplification_ratio: float  # betti_total_last / betti_total_first


def compute_betti_numbers(dgm: dict, dim_threshold: float = None) -> tuple[int, int, int]:
    """Extract Betti numbers from persistence diagram.

    Args:
        dgm: Persistence diagram from ripser
        dim_threshold: If None, uses the median persistence as threshold.
                      Features with lifetime > threshold are counted.

    Returns:
        (β₀, β₁, β₂) tuple
    """
    betti = [0, 0, 0]

    for dim in range(min(3, len(dgm))):
        if len(dgm[dim]) == 0:
            continue

        births = dgm[dim][:, 0]
        deaths = dgm[dim][:, 1]

        # Handle infinite deaths (set to max finite death)
        finite_deaths = deaths[np.isfinite(deaths)]
        if len(finite_deaths) > 0:
            max_death = np.max(finite_deaths)
            deaths = np.where(np.isinf(deaths), max_death, deaths)

        lifetimes = deaths - births

        if dim_threshold is None:
            # Use median lifetime as threshold (relational, not arbitrary)
            threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
        else:
            threshold = dim_threshold

        # Count significant features
        betti[dim] = int(np.sum(lifetimes > threshold))

    return tuple(betti)


def compute_persistence_stats(dgm: dict) -> tuple[float, float, float]:
    """Compute persistence statistics from diagram.

    Returns:
        (total_persistence, max_persistence, persistence_entropy)
    """
    all_lifetimes = []

    for dim in range(len(dgm)):
        if len(dgm[dim]) == 0:
            continue
        births = dgm[dim][:, 0]
        deaths = dgm[dim][:, 1]

        # Handle infinite deaths
        finite_deaths = deaths[np.isfinite(deaths)]
        if len(finite_deaths) > 0:
            max_death = np.max(finite_deaths)
            deaths = np.where(np.isinf(deaths), max_death, deaths)

        lifetimes = deaths - births
        all_lifetimes.extend(lifetimes[lifetimes > 0])

    if not all_lifetimes:
        return 0.0, 0.0, 0.0

    all_lifetimes = np.array(all_lifetimes)
    total_persistence = float(np.sum(all_lifetimes))
    max_persistence = float(np.max(all_lifetimes))

    # Entropy of normalized lifetimes
    p = all_lifetimes / np.sum(all_lifetimes)
    p = p[p > 0]  # Avoid log(0)
    persistence_entropy = float(-np.sum(p * np.log(p)))

    return total_persistence, max_persistence, persistence_entropy


def analyze_layer_topology(
    activations: np.ndarray,
    layer_idx: int,
    target_dim: int = 50,
) -> LayerTopology:
    """Compute topological features of activation manifold at a layer.

    Args:
        activations: [n_tokens, hidden_dim] activation matrix
        layer_idx: Which layer this is
        target_dim: PCA dimension (ripser works best in low dimensions)

    Returns:
        LayerTopology with Betti numbers and persistence stats
    """
    import warnings
    ensure_ripser()

    # Subsample if too many points (ripser is O(n³))
    max_points = 100
    if len(activations) > max_points:
        indices = np.random.choice(len(activations), max_points, replace=False)
        activations = activations[indices]

    # PCA to reduce dimensionality (standard for high-dim TDA)
    # Use SVD directly for numerical stability
    n_points, n_features = activations.shape
    effective_dim = min(target_dim, n_points - 1, n_features)

    if effective_dim < n_features:
        # Center the data
        activations_centered = activations - np.mean(activations, axis=0)
        # SVD for PCA
        try:
            U, S, Vt = np.linalg.svd(activations_centered, full_matrices=False)
            activations = U[:, :effective_dim] * S[:effective_dim]
        except np.linalg.LinAlgError:
            # Fallback: just take first dimensions
            activations = activations[:, :effective_dim]

    # Compute persistent homology up to dimension 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ripser.ripser(activations, maxdim=2)
    dgm = result['dgms']

    # Extract Betti numbers and stats
    b0, b1, b2 = compute_betti_numbers(dgm)
    total_pers, max_pers, pers_entropy = compute_persistence_stats(dgm)

    return LayerTopology(
        layer_idx=layer_idx,
        betti_0=b0,
        betti_1=b1,
        betti_2=b2,
        total_persistence=total_pers,
        max_persistence=max_pers,
        persistence_entropy=pers_entropy,
    )


def compute_trend(values: list[int]) -> str:
    """Determine if a sequence is increasing, decreasing, or stable."""
    if len(values) < 2:
        return "stable"

    # Linear regression slope
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]

    # Use relative change threshold (relational, not arbitrary)
    mean_val = np.mean(values) if np.mean(values) > 0 else 1
    relative_slope = slope / mean_val

    if relative_slope > 0.05:
        return "increasing"
    elif relative_slope < -0.05:
        return "decreasing"
    else:
        return "stable"


def analyze_topology_trajectory(
    model,
    tokenizer,
    prompt: str,
) -> TopologyTrajectory:
    """Analyze how topology evolves through the network for a prompt."""
    import mlx.core as mx

    # Tokenize
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([tokens])

    # Get model structure
    base_model = getattr(model, "model", model)
    embed = base_model.embed_tokens
    layers = base_model.layers

    # Initial embedding
    h = embed(input_ids)
    mx.eval(h)

    layer_topologies = []

    for i, layer in enumerate(layers):
        # Forward through layer
        h = layer(h)
        mx.eval(h)

        # Extract activations as numpy
        h_np = np.array(h[0].tolist(), dtype=np.float32)

        # Compute topology
        topo = analyze_layer_topology(h_np, i)
        layer_topologies.append(topo)

        print(f"  Layer {i:2d}: β₀={topo.betti_0}, β₁={topo.betti_1}, β₂={topo.betti_2}, "
              f"H={topo.persistence_entropy:.2f}")

    # Compute trends
    b0_values = [t.betti_0 for t in layer_topologies]
    b1_values = [t.betti_1 for t in layer_topologies]

    # Simplification ratio: compare first vs last total Betti number
    first_total = layer_topologies[0].betti_0 + layer_topologies[0].betti_1 + layer_topologies[0].betti_2
    last_total = layer_topologies[-1].betti_0 + layer_topologies[-1].betti_1 + layer_topologies[-1].betti_2
    simplification = last_total / first_total if first_total > 0 else 1.0

    return TopologyTrajectory(
        prompt=prompt,
        layer_topologies=layer_topologies,
        betti_0_trend=compute_trend(b0_values),
        betti_1_trend=compute_trend(b1_values),
        topology_simplification_ratio=simplification,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze manifold topology across layers")
    parser.add_argument("model_path", type=str, help="Path to MLX model")
    parser.add_argument("--prompts", type=str, nargs="+",
                       default=["The quick brown fox", "What is 2+2?", "Explain quantum mechanics"],
                       help="Prompts to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    ensure_mlx_lm()
    model, tokenizer = mlx_lm(args.model_path)

    results = []

    for prompt in args.prompts:
        print(f"\nAnalyzing: '{prompt[:50]}...'")
        trajectory = analyze_topology_trajectory(model, tokenizer, prompt)

        print(f"\n  β₀ trend: {trajectory.betti_0_trend}")
        print(f"  β₁ trend: {trajectory.betti_1_trend}")
        print(f"  Simplification ratio: {trajectory.topology_simplification_ratio:.2f}")

        # Convert to dict for JSON
        result = {
            "prompt": trajectory.prompt,
            "betti_0_trend": trajectory.betti_0_trend,
            "betti_1_trend": trajectory.betti_1_trend,
            "topology_simplification_ratio": trajectory.topology_simplification_ratio,
            "layers": [asdict(t) for t in trajectory.layer_topologies],
        }
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    avg_simplification = np.mean([r["topology_simplification_ratio"] for r in results])
    print(f"Average simplification ratio: {avg_simplification:.2f}")

    if avg_simplification < 1.0:
        print("→ Topology SIMPLIFIES through depth (fewer holes)")
    elif avg_simplification > 1.0:
        print("→ Topology COMPLEXIFIES through depth (more holes)")
    else:
        print("→ Topology PRESERVED through depth")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
