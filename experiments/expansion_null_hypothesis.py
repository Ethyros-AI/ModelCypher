#!/usr/bin/env python3
"""Null hypothesis test for expansion_ratio metric.

Question: Is expansion_ratio ≈ 1.0 a property of training,
or do random (untrained) networks also show this pattern?

Experiment:
1. Load a trained model, compute expansion_ratio on various prompts
2. Randomize the weights (preserve architecture)
3. Compute expansion_ratio on same prompts
4. Compare distributions

If random weights also give ratio ≈ 1.0, the metric is pareidolia.
If trained weights are significantly different, training does something real.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_expansion import (
    compute_trajectory_norms,
    compute_expansion_metrics,
)


def load_model(model_path: str):
    """Load MLX model."""
    from mlx_lm import load
    model, tokenizer = load(model_path)
    return model, tokenizer


def randomize_weights(model) -> None:
    """Randomize all weight matrices in-place (He initialization)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # He initialization: N(0, sqrt(2/fan_in))
            fan_in = module.weight.shape[1]
            std = np.sqrt(2.0 / fan_in)
            new_weight = mx.random.normal(module.weight.shape) * std
            module.weight = new_weight
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = mx.zeros_like(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard embedding init: N(0, 1)
            new_weight = mx.random.normal(module.weight.shape)
            module.weight = new_weight


def compute_ratios_for_prompts(model, tokenizer, prompts: list[str], debug: bool = False) -> list[dict]:
    """Compute expansion metrics for each prompt."""
    results = []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array(tokens)

        try:
            trajectory = compute_trajectory_norms(model, input_ids)
            mx.eval(trajectory)

            if debug and i == 0:
                print(f"  Trajectory shape: {trajectory.shape}")
                print(f"  Trajectory values: {trajectory[:5].tolist()}...")
                print(f"  Has NaN: {bool(mx.any(mx.isnan(trajectory)))}")
                print(f"  Has Inf: {bool(mx.any(mx.isinf(trajectory)))}")

            metrics = compute_expansion_metrics(trajectory, exact=True)

            # Compute trajectory variance (flatness measure)
            traj_np = np.array(trajectory.tolist())
            traj_var = float(np.var(traj_np))
            traj_range = float(np.max(traj_np) - np.min(traj_np))

            results.append({
                "prompt": prompt[:50],
                "expansion_ratio": metrics["expansion_ratio"],
                "peak_layer": metrics["peak_layer"],
                "n_layers": metrics["n_layers"],
                "trajectory_variance": traj_var,
                "trajectory_range": traj_range,
            })
        except Exception as e:
            print(f"Error on prompt '{prompt[:30]}...': {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Null hypothesis test for expansion_ratio")
    parser.add_argument("model_path", help="Path to trained MLX model")
    parser.add_argument("--n-prompts", type=int, default=50, help="Number of test prompts")
    parser.add_argument("--output", default="data/experiments/expansion_null_results.json")
    args = parser.parse_args()

    # Diverse prompts to test
    prompts = [
        "The capital of France is",
        "In mathematics, pi equals approximately",
        "The quick brown fox jumps over",
        "Hello, how are you doing today?",
        "Once upon a time in a land far away",
        "The stock market today showed",
        "Scientists discovered that",
        "The recipe calls for two cups of",
        "In the year 2050, technology will",
        "The president announced that",
        "Breaking news: experts say",
        "According to recent studies,",
        "The weather forecast predicts",
        "In a surprising turn of events,",
        "Researchers at MIT found that",
        "The ancient civilization of",
        "When asked about the situation,",
        "The economy is expected to",
        "In the field of artificial intelligence,",
        "The committee decided to",
        "Sources close to the matter said",
        "The new policy will affect",
        "Experts warn that climate change",
        "The company reported earnings of",
        "In a statement released today,",
        "The investigation revealed that",
        "According to government data,",
        "The technology enables users to",
        "In the latest development,",
        "The proposal would require",
        "Scientists believe that",
        "The report highlights",
        "In response to criticism,",
        "The data suggests that",
        "Analysts predict that",
        "The legislation aims to",
        "In a historic move,",
        "The findings indicate that",
        "The initiative will provide",
        "According to the spokesperson,",
        "The trend continues as",
        "In the coming months,",
        "The strategy involves",
        "Observers note that",
        "The evidence shows that",
        "In the meantime,",
        "The situation remains",
        "According to experts,",
        "The outcome depends on",
        "In conclusion,",
    ][:args.n_prompts]

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)

    # Phase 1: Trained weights
    print("\n=== Phase 1: Trained Weights ===")
    trained_results = compute_ratios_for_prompts(model, tokenizer, prompts, debug=True)
    trained_ratios = [r["expansion_ratio"] for r in trained_results]

    trained_traj_ranges = [r["trajectory_range"] for r in trained_results]

    print(f"Trained model expansion_ratio:")
    print(f"  Mean: {np.mean(trained_ratios):.4f}")
    print(f"  Std:  {np.std(trained_ratios):.4f}")
    print(f"  Min:  {np.min(trained_ratios):.4f}")
    print(f"  Max:  {np.max(trained_ratios):.4f}")
    print(f"Trajectory range (expansion-compression amplitude):")
    print(f"  Mean: {np.mean(trained_traj_ranges):.4f}")

    # Phase 2: Randomize weights
    print("\n=== Phase 2: Randomizing Weights ===")
    randomize_weights(model)

    print("\n=== Phase 3: Random Weights ===")
    random_results = compute_ratios_for_prompts(model, tokenizer, prompts, debug=True)
    random_ratios = [r["expansion_ratio"] for r in random_results]

    random_traj_ranges = [r["trajectory_range"] for r in random_results]

    print(f"Random model expansion_ratio:")
    print(f"  Mean: {np.mean(random_ratios):.4f}")
    print(f"  Std:  {np.std(random_ratios):.4f}")
    print(f"  Min:  {np.min(random_ratios):.4f}")
    print(f"  Max:  {np.max(random_ratios):.4f}")
    print(f"Trajectory range (expansion-compression amplitude):")
    print(f"  Mean: {np.mean(random_traj_ranges):.4f}")

    # Phase 4: Statistical comparison
    print("\n=== Statistical Comparison ===")

    trained_mean = np.mean(trained_ratios)
    random_mean = np.mean(random_ratios)
    trained_std = np.std(trained_ratios)
    random_std = np.std(random_ratios)

    # Distance from target (1.0)
    trained_dist = abs(trained_mean - 1.0)
    random_dist = abs(random_mean - 1.0)

    print(f"Distance from target (1.0):")
    print(f"  Trained: {trained_dist:.4f}")
    print(f"  Random:  {random_dist:.4f}")

    # Variance comparison
    print(f"\nVariance:")
    print(f"  Trained: {trained_std**2:.4f}")
    print(f"  Random:  {random_std**2:.4f}")

    # Check if random model has flat trajectory (no expansion/compression)
    random_has_structure = random_std > 0.01 or abs(random_mean) > 0.1

    # Conclusion
    print("\n=== Conclusion ===")
    if not random_has_structure:
        print("Random weights produce FLAT trajectory (no expansion/compression).")
        print("Trained weights show clear expand-then-compress pattern.")
        print("Training CREATES the geometric structure - this is NOT pareidolia.")
        conclusion = "training_creates_structure"
    elif random_dist < trained_dist:
        print("WARNING: Random weights are CLOSER to target than trained weights!")
        print("This suggests expansion_ratio = 1.0 may be pareidolia.")
        conclusion = "pareidolia"
    elif trained_std < random_std * 0.5:
        print("Trained weights show LOWER variance than random.")
        print("Training appears to stabilize expansion_ratio (could be meaningful).")
        conclusion = "possibly_meaningful"
    else:
        print("No clear difference between trained and random.")
        print("More investigation needed.")
        conclusion = "inconclusive"

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": args.model_path,
        "n_prompts": len(prompts),
        "trained": {
            "mean": float(trained_mean),
            "std": float(trained_std),
            "min": float(np.min(trained_ratios)),
            "max": float(np.max(trained_ratios)),
            "distance_from_1": float(trained_dist),
        },
        "random": {
            "mean": float(random_mean),
            "std": float(random_std),
            "min": float(np.min(random_ratios)),
            "max": float(np.max(random_ratios)),
            "distance_from_1": float(random_dist),
        },
        "conclusion": conclusion,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
