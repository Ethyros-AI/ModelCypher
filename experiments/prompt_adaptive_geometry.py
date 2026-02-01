#!/usr/bin/env python3
"""Test if fine-tuned models have prompt-adaptive peak positions."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_phi import compute_trajectory_norms


def get_peak_position(model, tokenizer, prompt: str) -> float:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)
    trajectory = compute_trajectory_norms(model, input_ids)
    mx.eval(trajectory)
    traj_np = np.array(trajectory.tolist())
    n_layers = len(traj_np) - 1
    peak_layer = int(np.argmax(traj_np))
    return peak_layer / n_layers * 100


def test_model(model_path: str, prompt_categories: dict) -> dict:
    from mlx_lm import load

    model, tokenizer = load(model_path)
    results = {}

    for category, prompts in prompt_categories.items():
        peaks = [get_peak_position(model, tokenizer, p) for p in prompts]
        results[category] = {
            "mean": float(np.mean(peaks)),
            "std": float(np.std(peaks)),
            "peaks": peaks,
        }

    del model, tokenizer
    mx.clear_cache()

    return results


def main():
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")

    prompt_categories = {
        "factual": [
            "The capital of France is",
            "Water boils at a temperature of",
            "The speed of light is approximately",
            "Mount Everest is located in",
        ],
        "instruction": [
            "Please explain how photosynthesis works:",
            "List the steps to make pasta:",
            "Describe what happens during an eclipse:",
            "Summarize the plot of Romeo and Juliet:",
        ],
        "reasoning": [
            "If all dogs are mammals and all mammals breathe air, then",
            "The sum of 23 and 45 equals",
            "Given that x + 5 = 12, solve for x:",
            "If today is Monday, what day was it 3 days ago?",
        ],
        "creative": [
            "Once upon a time in a magical kingdom,",
            "The robot gazed at the stars and wondered",
            "In the year 3000, humans will",
            "The old wizard opened the ancient book and",
        ],
    }

    models = [
        ("LFM2-1.2B-bf16", "Base"),
        ("LFM2.5-1.2B-Instruct-bf16", "Instruct"),
        ("LFM2.5-1.2B-Thinking-bf16", "Thinking"),
    ]

    print("=" * 80)
    print("PROMPT-ADAPTIVE GEOMETRY ANALYSIS")
    print("=" * 80)
    print("\nHypothesis: Fine-tuned models adjust peak position based on prompt type")
    print("            Base models have fixed peak position\n")

    all_results = {}

    for model_name, label in models:
        model_path = str(models_dir / model_name)
        if not (models_dir / model_name).exists():
            continue

        print(f"\n{'='*40}")
        print(f"{label} ({model_name})")
        print("=" * 40)

        results = test_model(model_path, prompt_categories)
        all_results[label] = results

        print(f"\n{'Category':<15} {'Mean Peak':>12} {'Std Dev':>10}")
        print("-" * 40)

        for category, data in results.items():
            print(f"{category:<15} {data['mean']:>11.1f}% {data['std']:>9.1f}%")

        # Variance across categories
        category_means = [data["mean"] for data in results.values()]
        cross_category_std = np.std(category_means)
        print(f"\nCross-category std: {cross_category_std:.2f}%")

        if cross_category_std < 1.0:
            print("→ FIXED geometry (same peak for all prompt types)")
        else:
            print("→ ADAPTIVE geometry (different peaks for different prompts)")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Cross-Category Variance")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Factual':>10} {'Instruct':>10} {'Reasoning':>10} {'Creative':>10} {'Variance':>10}")
    print("-" * 80)

    for label, results in all_results.items():
        factual = results["factual"]["mean"]
        instruct = results["instruction"]["mean"]
        reasoning = results["reasoning"]["mean"]
        creative = results["creative"]["mean"]
        variance = np.std([factual, instruct, reasoning, creative])
        print(f"{label:<20} {factual:>9.1f}% {instruct:>9.1f}% {reasoning:>9.1f}% {creative:>9.1f}% {variance:>9.2f}%")


if __name__ == "__main__":
    main()
