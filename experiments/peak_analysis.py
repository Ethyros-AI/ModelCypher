#!/usr/bin/env python3
"""Analyze where each model's peak occurs."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_expansion import compute_trajectory_norms


def analyze_peaks(model_path: str, prompts: list[str]) -> dict:
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
        n_layers = config.get("num_hidden_layers", 0)
    else:
        n_layers = 0

    peak_positions = []  # relative position (0 = start, 1 = end)
    peak_at_end = 0

    for prompt in prompts:
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array(tokens)
            trajectory = compute_trajectory_norms(model, input_ids)
            mx.eval(trajectory)

            traj_np = np.array(trajectory.tolist())
            peak_layer = np.argmax(traj_np)
            n = len(traj_np) - 1  # number of transformer layers

            peak_positions.append(peak_layer / n)
            if peak_layer == n:
                peak_at_end += 1
        except:
            pass

    del model, tokenizer
    mx.clear_cache()

    if not peak_positions:
        return {"error": "No successful prompts"}

    return {
        "n_layers": n_layers,
        "mean_peak_position": float(np.mean(peak_positions)),
        "peak_at_end_pct": peak_at_end / len(peak_positions) * 100,
        "has_compression": peak_at_end < len(peak_positions) * 0.5,
    }


def main():
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")

    models = [
        "LFM2-350M-MLX-bf16",
        "LFM2-700M-bf16",
        "LFM2-1.2B-bf16",
        "LFM2.5-1.2B-Thinking-bf16",
        "Qwen3-1.7B-MLX-bf16",
        "Qwen3-8B-bf16",
        "Qwen2.5-3B-Instruct-bf16",
        "DeepSeek-R1-0528-Qwen3-8B-bf16",
    ]

    prompts = [
        "The capital of France is",
        "In mathematics, pi equals approximately",
        "Once upon a time in a land far away",
        "Scientists discovered that",
        "The weather forecast predicts",
    ]

    print(f"{'Model':<40} {'Layers':>6} {'Peak Pos':>10} {'At End%':>8} {'Compress?':<10}")
    print("-" * 80)

    for model_name in models:
        model_path = str(models_dir / model_name)
        if not (models_dir / model_name).exists():
            continue

        result = analyze_peaks(model_path, prompts)

        if "error" in result:
            print(f"{model_name:<40} Error: {result['error']}")
        else:
            compress = "YES" if result["has_compression"] else "NO"
            print(f"{model_name:<40} {result['n_layers']:>6} {result['mean_peak_position']:>10.2%} {result['peak_at_end_pct']:>7.0f}% {compress:<10}")


if __name__ == "__main__":
    main()
