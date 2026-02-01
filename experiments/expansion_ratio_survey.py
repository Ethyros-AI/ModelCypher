#!/usr/bin/env python3
"""Survey expansion_ratio across multiple models to find patterns."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_phi import (
    compute_trajectory_norms,
    compute_expansion_metrics,
)


def test_model(model_path: str, prompts: list[str]) -> dict:
    """Test a single model."""
    from mlx_lm import load

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        return {"error": str(e)}

    # Get config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        n_layers = config.get("num_hidden_layers", "?")
        hidden_size = config.get("hidden_size", "?")
        model_type = config.get("model_type", "?")
    else:
        n_layers = hidden_size = model_type = "?"

    ratios = []
    ranges = []

    for prompt in prompts:
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array(tokens)
            trajectory = compute_trajectory_norms(model, input_ids)
            mx.eval(trajectory)
            metrics = compute_expansion_metrics(trajectory, exact=True)
            ratios.append(metrics["expansion_ratio"])

            traj_np = np.array(trajectory.tolist())
            ranges.append(float(np.max(traj_np) - np.min(traj_np)))
        except Exception as e:
            print(f"  Error: {e}")

    if not ratios:
        return {"error": "No successful prompts"}

    # Clear memory
    del model, tokenizer
    mx.clear_cache()

    return {
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "model_type": model_type,
        "expansion_ratio_mean": float(np.mean(ratios)),
        "expansion_ratio_std": float(np.std(ratios)),
        "trajectory_range_mean": float(np.mean(ranges)),
        "distance_from_1": float(abs(np.mean(ratios) - 1.0)),
        "n_prompts": len(ratios),
    }


def main():
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")

    # Models to test (mix of sizes and architectures)
    models = [
        "LFM2-350M-MLX-bf16",
        "LFM2-700M-bf16",
        "LFM2-1.2B-bf16",
        "Qwen3-1.7B-MLX-bf16",
        "Qwen3-8B-bf16",
        "Qwen2.5-3B-Instruct-bf16",
        "DeepSeek-R1-0528-Qwen3-8B-bf16",
        "LFM2.5-1.2B-Thinking-bf16",
    ]

    prompts = [
        "The capital of France is",
        "In mathematics, pi equals approximately",
        "Once upon a time in a land far away",
        "Scientists discovered that",
        "The weather forecast predicts",
        "According to recent studies,",
        "In the field of artificial intelligence,",
        "The evidence shows that",
        "When asked about the situation,",
        "In conclusion,",
    ]

    results = {}

    for model_name in models:
        model_path = str(models_dir / model_name)
        if not (models_dir / model_name).exists():
            print(f"Skipping {model_name} (not found)")
            continue

        print(f"\nTesting {model_name}...")
        result = test_model(model_path, prompts)
        results[model_name] = result

        if "error" not in result:
            print(f"  Layers: {result['n_layers']}, Hidden: {result['hidden_size']}")
            print(f"  Expansion ratio: {result['expansion_ratio_mean']:.3f} ± {result['expansion_ratio_std']:.3f}")
            print(f"  Distance from 1.0: {result['distance_from_1']:.3f}")
        else:
            print(f"  Error: {result['error']}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Layers':>6} {'Ratio':>8} {'Dist':>6} {'Type':<10}")
    print("-" * 80)

    for model_name, result in sorted(results.items(), key=lambda x: x[1].get("distance_from_1", 999)):
        if "error" in result:
            continue
        print(f"{model_name:<40} {result['n_layers']:>6} {result['expansion_ratio_mean']:>8.3f} {result['distance_from_1']:>6.3f} {result['model_type']:<10}")

    # Save results
    output_path = Path("data/experiments/expansion_ratio_survey.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
