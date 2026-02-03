#!/usr/bin/env python3
"""
Visualize norm trajectories to understand expansion ratio.

Key question: Why does LFM2 sometimes peak before the last layer?
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


def compute_norm_trajectory(model, tokenizer, prompt):
    """Compute mean L2 norm per token at each layer."""
    embed = model.model.embed_tokens
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    h = embed(input_ids)
    mx.eval(h)

    norms = [float(mx.mean(mx.sqrt(mx.sum(h * h, axis=-1))))]

    for layer in model.model.layers:
        h = layer(h)
        mx.eval(h)
        norms.append(float(mx.mean(mx.sqrt(mx.sum(h * h, axis=-1)))))

    return np.array(norms)


def main():
    print("="*70)
    print("  NORM TRAJECTORY ANALYSIS")
    print("="*70)

    models = [
        ("/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16", "LFM2"),
        ("/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16", "Qwen3"),
    ]

    prompts = {
        "retrieval": "What is the capital of France?",
        "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    }

    for model_path, model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print("="*70)

        model, tokenizer = load(model_path)
        n_layers = len(model.model.layers)

        for task_name, prompt in prompts.items():
            norms = compute_norm_trajectory(model, tokenizer, prompt)

            peak_idx = np.argmax(norms)
            peak_val = norms[peak_idx]
            final_val = norms[-1]
            expansion_ratio = peak_val / final_val

            print(f"\n{task_name}:")
            print(f"  Peak: layer {peak_idx} ({peak_idx/n_layers*100:.0f}%), norm={peak_val:.1f}")
            print(f"  Final: layer {n_layers}, norm={final_val:.1f}")
            print(f"  Expansion ratio: {expansion_ratio:.3f}")

            # Show trajectory around peak
            print(f"\n  Trajectory (last 5 layers):")
            for i in range(max(0, n_layers-4), n_layers+1):
                norm = norms[i]
                delta = norm - norms[i-1] if i > 0 else 0
                marker = " ← PEAK" if i == peak_idx else ""
                print(f"    Layer {i:2d}: {norm:6.1f} (Δ={delta:+5.1f}){marker}")

        del model
        mx.metal.clear_cache()

    print("\n" + "="*70)
    print("  ANALYSIS")
    print("="*70)
    print("""
Key question: What determines whether the peak is at the last layer or earlier?

Observation:
- Qwen/DeepSeek: Always peak at last layer (monotonic increase)
- LFM2: Sometimes peaks earlier (slight decrease at end)

Hypothesis: The final layer's behavior differs between architectures.
- Pure transformer: Final layer continues expansion
- Hybrid (Mamba): Final layer may compress slightly

The expansion_ratio variance comes from:
1. Whether peak is at the last layer (ratio = 1.0)
2. How much the norm decreases after the peak (ratio > 1.0)
""")


if __name__ == "__main__":
    main()
