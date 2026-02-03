#!/usr/bin/env python3
"""
Understand what determines deviation norm.

Finding: Reasoning training has higher deviation norm (117 vs 65).
Question: What causes this? Is it training diversity or architecture?

Test: Same architecture (Qwen3-8B base vs DeepSeek-R1-Qwen3-8B)
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


def measure_layer_norms(model, tokenizer, prompts, layer_idx):
    """Measure activation norms at a specific layer."""
    embed = model.model.embed_tokens

    all_acts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = embed(input_ids)

        # Forward through layers up to target
        for i, layer in enumerate(model.model.layers):
            h = layer(h)
            if i == layer_idx:
                break
        mx.eval(h)

        h_np = np.array(h.astype(mx.float32))[0]
        all_acts.append(h_np)

    acts = np.concatenate(all_acts, axis=0)

    mean = np.mean(acts, axis=0)
    centered = acts - mean

    mean_norm = np.linalg.norm(mean)
    dev_norm = np.mean(np.linalg.norm(centered, axis=1))
    act_norm = np.mean(np.linalg.norm(acts, axis=1))

    return {
        'mean_norm': mean_norm,
        'dev_norm': dev_norm,
        'act_norm': act_norm,
        'convergence': mean_norm / dev_norm if dev_norm > 0 else 0,
    }


def main():
    print("="*60)
    print("  DEVIATION NORM ANALYSIS")
    print("="*60)

    # Same architecture, different training
    models = [
        ("/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16", "base"),
        ("/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16", "reasoning"),
    ]

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "In the beginning was the Word, and the Word was with God.",
        "The solution to x^2 - 4 = 0 is x = 2 or x = -2.",
        "Once upon a time in a land far away, there lived a princess.",
        "SELECT * FROM users WHERE age > 18;",
        "The mitochondria is the powerhouse of the cell.",
    ]

    for model_path, train_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_path.split('/')[-1]} ({train_type})")
        print("="*60)

        model, tokenizer = load(model_path)
        n_layers = len(model.model.layers)

        print(f"\n| Layer | Mean Norm | Dev Norm | Act Norm | Convergence |")
        print(f"|-------|-----------|----------|----------|-------------|")

        # Sample layers throughout the network
        layers_to_check = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]

        for layer_idx in layers_to_check:
            stats = measure_layer_norms(model, tokenizer, prompts, layer_idx)
            print(f"| {layer_idx:5d} | {stats['mean_norm']:7.1f} | {stats['dev_norm']:6.1f} | {stats['act_norm']:6.1f} | {stats['convergence']:.2f} |")

        del model
        mx.metal.clear_cache()

    # Analysis
    print("\n" + "="*60)
    print("  INTERPRETATION")
    print("="*60)
    print("""
If deviation norm difference is due to training:
  - Same architecture should have similar early-layer norms
  - Differences emerge in later layers where task-specific processing happens

If deviation norm difference is due to architecture:
  - Differences would appear from early layers

Key question: At which layer does the deviation norm diverge?
""")


if __name__ == "__main__":
    main()
