#!/usr/bin/env python3
"""Debug trajectory shapes for models with ratio=0."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_phi import compute_trajectory_norms


def debug_model(model_path: str, prompt: str = "The capital of France is"):
    from mlx_lm import load

    print(f"\n{'='*60}")
    print(f"Model: {Path(model_path).name}")
    print(f"{'='*60}")

    model, tokenizer = load(model_path)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)

    trajectory = compute_trajectory_norms(model, input_ids)
    mx.eval(trajectory)

    traj_np = np.array(trajectory.tolist())

    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {len(tokens)}")
    print(f"Layers: {len(traj_np) - 1}")
    print(f"\nTrajectory (norm at each layer):")
    print(f"  Initial (embed): {traj_np[0]:.2f}")
    print(f"  Layer 1: {traj_np[1]:.2f}")
    print(f"  Layer 2: {traj_np[2]:.2f}")
    print(f"  ...")
    print(f"  Peak layer: {np.argmax(traj_np)} (norm={np.max(traj_np):.2f})")
    print(f"  Final layer: {traj_np[-1]:.2f}")
    print(f"\nShape analysis:")
    print(f"  Min: {np.min(traj_np):.2f} at layer {np.argmin(traj_np)}")
    print(f"  Max: {np.max(traj_np):.2f} at layer {np.argmax(traj_np)}")
    print(f"  Range: {np.max(traj_np) - np.min(traj_np):.2f}")
    print(f"  Std: {np.std(traj_np):.2f}")

    # Is it monotonic?
    diffs = np.diff(traj_np)
    n_increasing = np.sum(diffs > 0)
    n_decreasing = np.sum(diffs < 0)
    print(f"\nMonotonicity:")
    print(f"  Increasing steps: {n_increasing}/{len(diffs)}")
    print(f"  Decreasing steps: {n_decreasing}/{len(diffs)}")

    if n_increasing == len(diffs):
        print(f"  → MONOTONIC INCREASING (no compression)")
    elif n_decreasing == len(diffs):
        print(f"  → MONOTONIC DECREASING (no expansion)")
    elif np.argmax(traj_np) == 0:
        print(f"  → PEAK AT START (pure compression)")
    elif np.argmax(traj_np) == len(traj_np) - 1:
        print(f"  → PEAK AT END (pure expansion)")
    else:
        peak = np.argmax(traj_np)
        print(f"  → EXPAND-THEN-COMPRESS (peak at layer {peak})")

    # Clean up
    del model, tokenizer
    mx.clear_cache()


def main():
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")

    # Test the flat trajectory models
    flat_models = [
        "LFM2-700M-bf16",
        "Qwen3-8B-bf16",
    ]

    # And compare with structured models
    structured_models = [
        "LFM2-350M-MLX-bf16",
        "Qwen2.5-3B-Instruct-bf16",
    ]

    for model_name in flat_models + structured_models:
        model_path = str(models_dir / model_name)
        if (models_dir / model_name).exists():
            debug_model(model_path)


if __name__ == "__main__":
    main()
