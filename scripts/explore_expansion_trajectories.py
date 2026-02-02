#!/usr/bin/env python3
"""Explore layer-by-layer expansion trajectories across models and task types.

Key questions:
1. Where does expansion peak? (LFM2: layer 14/16, DeepSeek: final layer)
2. Does trajectory shape differ by task type?
3. Can we detect task type from early layers?
4. What's the compression curve look like?

Usage:
    python scripts/explore_expansion_trajectories.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Representative prompts from each category
TASK_PROBES = {
    "retrieval": "What is the capital of France?",
    "arithmetic": "What is 7 times 8?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "logic": "All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
    "cot": "Let's think step by step. If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
}


@dataclass
class LayerTrajectory:
    """Full trajectory through model layers."""
    prompt: str
    task_type: str
    norms: list[float]  # norm at each layer (including embedding)
    peak_layer: int
    peak_norm: float
    final_norm: float
    compression_ratio: float
    expansion_ratio: float  # peak / initial

    # Derived metrics
    expansion_layers: list[int]  # layers where norm increased
    compression_layers: list[int]  # layers where norm decreased


def trace_trajectory(model, tokenizer, prompt: str) -> list[float]:
    """Trace norm through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    base = getattr(model, "model", model)

    # Embedding
    hidden = base.embed_tokens(input_ids)
    mx.eval(hidden)

    norms = [float(mx.sqrt(mx.sum(hidden * hidden)))]

    # Each layer
    for layer in base.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norms.append(float(mx.sqrt(mx.sum(hidden * hidden))))

    return norms


def analyze_trajectory(prompt: str, task_type: str, norms: list[float]) -> LayerTrajectory:
    """Analyze a single trajectory."""
    peak_idx = int(np.argmax(norms))  # Cast to int for JSON serialization
    peak_norm = norms[peak_idx]
    initial_norm = norms[0]
    final_norm = norms[-1]

    # Find expansion/compression layers
    expansion = []
    compression = []
    for i in range(1, len(norms)):
        if norms[i] > norms[i-1]:
            expansion.append(i)
        elif norms[i] < norms[i-1]:
            compression.append(i)

    # Compute metrics
    eps = np.sqrt(np.finfo(np.float32).eps)
    compression_ratio = peak_norm / final_norm if final_norm > eps else 1.0
    expansion_ratio = peak_norm / initial_norm if initial_norm > eps else 1.0

    return LayerTrajectory(
        prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
        task_type=task_type,
        norms=norms,
        peak_layer=peak_idx,
        peak_norm=peak_norm,
        final_norm=final_norm,
        compression_ratio=compression_ratio,
        expansion_ratio=expansion_ratio,
        expansion_layers=expansion,
        compression_layers=compression,
    )


def compute_trajectory_similarity(t1: LayerTrajectory, t2: LayerTrajectory) -> float:
    """Compute cosine similarity between normalized trajectories."""
    # Normalize to same length by interpolation
    n1 = np.array(t1.norms)
    n2 = np.array(t2.norms)

    # Normalize each trajectory
    n1 = n1 / np.max(n1)
    n2 = n2 / np.max(n2)

    # If different lengths, interpolate shorter to match longer
    if len(n1) != len(n2):
        target_len = max(len(n1), len(n2))
        if len(n1) < target_len:
            n1 = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(n1)), n1)
        else:
            n2 = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(n2)), n2)

    # Cosine similarity
    dot = np.dot(n1, n2)
    norm = np.linalg.norm(n1) * np.linalg.norm(n2)
    return dot / norm if norm > 0 else 0.0


def get_quartile_bucket(value: float, all_values: list[float]) -> str:
    """Assign value to quartile bucket based on distribution."""
    sorted_vals = sorted(all_values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4] if n >= 4 else sorted_vals[0]
    q2 = sorted_vals[n // 2] if n >= 2 else sorted_vals[0]
    q3 = sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1]

    if value <= q1:
        return "Q1 (lowest)"
    elif value <= q2:
        return "Q2"
    elif value <= q3:
        return "Q3"
    else:
        return "Q4 (highest)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    from mlx_lm import load

    print("=" * 70)
    print("EXPANSION TRAJECTORY EXPLORATION")
    print("=" * 70)
    print(f"Model: {Path(args.model).name}")

    model, tokenizer = load(args.model)
    n_layers = len(getattr(model, "model", model).layers)
    print(f"Layers: {n_layers}")
    print("=" * 70)

    trajectories = []

    for task_type, prompt in TASK_PROBES.items():
        print(f"\n{task_type.upper()}")
        print("-" * 40)

        norms = trace_trajectory(model, tokenizer, prompt)
        traj = analyze_trajectory(prompt, task_type, norms)
        trajectories.append(traj)

        print(f"  Peak layer: {traj.peak_layer}/{n_layers}")
        print(f"  Expansion ratio: {traj.expansion_ratio:.3f} (initial → peak)")
        print(f"  Compression ratio: {traj.compression_ratio:.3f} (peak → final)")
        print(f"  Expansion layers: {len(traj.expansion_layers)}")
        print(f"  Compression layers: {len(traj.compression_layers)}")

        # Show trajectory shape
        norm_arr = np.array(traj.norms)
        norm_normalized = norm_arr / np.max(norm_arr)

        # ASCII visualization
        width = 50
        print(f"\n  Trajectory (normalized):")
        for i, n in enumerate(norm_normalized):
            bar = "█" * int(n * width)
            marker = " ← PEAK" if i == traj.peak_layer else ""
            print(f"  L{i:02d} |{bar}{marker}")

    # Cross-task analysis
    print("\n" + "=" * 70)
    print("CROSS-TASK ANALYSIS")
    print("=" * 70)

    # Group by peak location
    early_peak = [t for t in trajectories if t.peak_layer < n_layers * 0.7]
    late_peak = [t for t in trajectories if t.peak_layer >= n_layers * 0.7]
    final_peak = [t for t in trajectories if t.peak_layer == n_layers]

    print(f"\nPeak location distribution:")
    print(f"  Early peak (< 70% depth): {[t.task_type for t in early_peak]}")
    print(f"  Late peak (>= 70% depth): {[t.task_type for t in late_peak]}")
    print(f"  Final layer peak: {[t.task_type for t in final_peak]}")

    # Expansion ratio clustering (data-driven quartiles)
    all_ratios = [t.expansion_ratio for t in trajectories]
    print(f"\nExpansion ratio by task (quartile-based buckets):")
    sorted_trajs = sorted(trajectories, key=lambda t: t.expansion_ratio, reverse=True)
    for t in sorted_trajs:
        bucket = get_quartile_bucket(t.expansion_ratio, all_ratios)
        print(f"  {t.task_type:12s}: {t.expansion_ratio:.3f} [{bucket}]")

    # Trajectory similarity matrix
    print(f"\nTrajectory shape similarity:")
    task_types = [t.task_type for t in trajectories]
    print(f"{'':12s}", end="")
    for t in task_types:
        print(f"{t[:8]:>9s}", end="")
    print()

    for i, t1 in enumerate(trajectories):
        print(f"{task_types[i]:12s}", end="")
        for j, t2 in enumerate(trajectories):
            sim = compute_trajectory_similarity(t1, t2)
            print(f"{sim:9.2f}", end="")
        print()

    # Early layer divergence check
    print(f"\n" + "=" * 70)
    print("EARLY LAYER DIVERGENCE (can we detect task type early?)")
    print("=" * 70)

    # Look at first 25% of layers
    early_cutoff = max(1, n_layers // 4)
    print(f"\nUsing first {early_cutoff} layers:")

    for t in trajectories:
        early_norms = t.norms[:early_cutoff + 1]
        early_expansion = sum(1 for i in range(1, len(early_norms)) if early_norms[i] > early_norms[i-1])
        early_slope = (early_norms[-1] - early_norms[0]) / len(early_norms) if len(early_norms) > 1 else 0
        print(f"  {t.task_type:12s}: slope={early_slope:+.2f}, expansions={early_expansion}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "model": args.model,
            "n_layers": n_layers,
            "trajectories": [asdict(t) for t in trajectories],
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
