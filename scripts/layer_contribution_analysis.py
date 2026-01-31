#!/usr/bin/env python3
"""Analyze per-layer contribution to expansion/compression.

Question: Which layers drive the geometric signature?

- Are compression layers consistent across tasks?
- Do some layers always expand? Always compress?
- Can we identify "geometry-defining" layers?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_layer_roles(trajectory_file: str) -> dict:
    """Analyze which layers consistently expand vs compress."""
    with open(trajectory_file) as f:
        data = json.load(f)

    n_layers = data["n_layers"]
    trajectories = data["trajectories"]

    # Track per-layer behavior across all tasks
    layer_expansion_count = np.zeros(n_layers + 1)  # +1 for embedding->L0
    layer_compression_count = np.zeros(n_layers + 1)

    for traj in trajectories:
        norms = traj["norms"]
        for i in range(1, len(norms)):
            if norms[i] > norms[i-1]:
                layer_expansion_count[i] += 1
            elif norms[i] < norms[i-1]:
                layer_compression_count[i] += 1

    n_tasks = len(trajectories)

    # Classify layers
    always_expand = []
    always_compress = []
    mixed = []

    for i in range(1, n_layers + 1):
        exp = layer_expansion_count[i]
        comp = layer_compression_count[i]

        if exp == n_tasks and comp == 0:
            always_expand.append(i)
        elif comp == n_tasks and exp == 0:
            always_compress.append(i)
        else:
            mixed.append(i)

    # Compute relative contribution to total expansion
    layer_delta = {i: [] for i in range(1, n_layers + 1)}

    for traj in trajectories:
        norms = traj["norms"]
        for i in range(1, len(norms)):
            delta = (norms[i] - norms[i-1]) / norms[0]  # Relative to initial
            layer_delta[i].append(delta)

    # Average delta per layer
    avg_delta = {i: np.mean(layer_delta[i]) for i in layer_delta}

    # Find the "big expansion" layers (top contributors)
    sorted_by_delta = sorted(avg_delta.items(), key=lambda x: x[1], reverse=True)

    return {
        "model": Path(trajectory_file).stem,
        "n_layers": n_layers,
        "n_tasks": n_tasks,
        "always_expand": always_expand,
        "always_compress": always_compress,
        "mixed": mixed,
        "top_expansion_layers": sorted_by_delta[:5],
        "top_compression_layers": sorted_by_delta[-5:],
        "layer_expansion_rate": {
            i: int(layer_expansion_count[i]) for i in range(1, n_layers + 1)
        },
        "layer_compression_rate": {
            i: int(layer_compression_count[i]) for i in range(1, n_layers + 1)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Trajectory JSON files")
    args = parser.parse_args()

    print("=" * 70)
    print("LAYER CONTRIBUTION ANALYSIS")
    print("=" * 70)

    for f in args.files:
        try:
            result = analyze_layer_roles(f)
        except Exception as e:
            print(f"\nError processing {f}: {e}")
            continue

        print(f"\n{result['model'].upper()}")
        print("-" * 50)
        print(f"Layers: {result['n_layers']}, Tasks: {result['n_tasks']}")

        print(f"\nLayer roles:")
        print(f"  Always expand: {result['always_expand']}")
        print(f"  Always compress: {result['always_compress']}")
        print(f"  Mixed behavior: {result['mixed']}")

        print(f"\nTop expansion contributors:")
        for layer, delta in result['top_expansion_layers']:
            print(f"  L{layer:02d}: {delta:+.3f} relative units")

        print(f"\nTop compression contributors:")
        for layer, delta in result['top_compression_layers']:
            print(f"  L{layer:02d}: {delta:+.3f} relative units")

        # Visual representation
        print(f"\nLayer behavior heatmap (exp/comp rate out of {result['n_tasks']} tasks):")
        for i in range(1, result['n_layers'] + 1):
            exp = result['layer_expansion_rate'][i]
            comp = result['layer_compression_rate'][i]
            exp_bar = "+" * exp
            comp_bar = "-" * comp
            role = "EXPAND" if exp == result['n_tasks'] else "COMPRESS" if comp == result['n_tasks'] else "MIXED"
            print(f"  L{i:02d} [{role:8s}] {exp_bar}{comp_bar}")


if __name__ == "__main__":
    main()
