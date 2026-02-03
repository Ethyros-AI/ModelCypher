#!/usr/bin/env python3
"""
Analyze what determines expansion_ratio.

Definition:
    expansion_ratio = compression_rate / expansion_rate

Where:
    expansion_rate = (peak - initial) / peak_layer
    compression_rate = (peak - final) / (n_layers - peak_layer)

Decomposition:
    expansion_ratio = [(P - F) / (P - I)] * [L_e / L_c]
                    = norm_recovery_ratio * layer_ratio

Questions:
1. What determines peak location?
2. What determines norm recovery?
3. Why does RLHF flatten expansion_ratio to 1.0?
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


def compute_norm_trajectory(model, tokenizer, prompt, use_mean=True):
    """Compute norm at each layer.

    If use_mean=True: mean L2 norm per token (what fingerprinting uses)
    If use_mean=False: total L2 norm
    """
    embed = model.model.embed_tokens
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    h = embed(input_ids)
    mx.eval(h)

    if use_mean:
        # Mean L2 norm per token
        norms = [float(mx.mean(mx.sqrt(mx.sum(h * h, axis=-1))))]
    else:
        norms = [float(mx.sqrt(mx.sum(h * h)))]

    for layer in model.model.layers:
        h = layer(h)
        mx.eval(h)
        if use_mean:
            norms.append(float(mx.mean(mx.sqrt(mx.sum(h * h, axis=-1)))))
        else:
            norms.append(float(mx.sqrt(mx.sum(h * h))))

    return np.array(norms)


def compute_expansion_metrics(norms):
    """Compute expansion ratio and its components.

    Fingerprint definition: expansion_ratio = peak / final
    This measures how much the norm compresses after the peak.
    """
    n = len(norms)
    peak_idx = np.argmax(norms)
    peak_val = norms[peak_idx]
    initial = norms[0]
    final = norms[-1]

    # Fingerprint definition: simple ratio
    expansion_ratio = peak_val / max(final, 1e-10)

    # Decomposition: expansion_ratio = (peak/initial) * (initial/final)
    #              = growth_factor * (1 / recovery_factor)
    growth_factor = peak_val / max(initial, 1e-10)
    recovery_factor = final / max(initial, 1e-10)

    return {
        'expansion_ratio': expansion_ratio,
        'peak_layer': peak_idx,
        'peak_layer_frac': peak_idx / (n - 1),
        'growth_factor': growth_factor,  # How much norm grows from input to peak
        'recovery_factor': recovery_factor,  # How much norm is preserved input→output
        'initial_norm': initial,
        'peak_norm': peak_val,
        'final_norm': final,
        'n_layers': n - 1,
    }


def main():
    print("="*70)
    print("  EXPANSION RATIO ANALYSIS")
    print("="*70)

    models = [
        ("/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16", "LFM2-base"),
        ("/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16", "base"),
        ("/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16", "reasoning"),
    ]

    # Use the exact fingerprint probes
    prompts = {
        "retrieval": "What is the capital of France?",
        "arithmetic": "What is 7 + 5?",
        "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "logic": "If all cats are animals, and all animals need water, do cats need water?",
        "creative": "Write the first line of a story about a dragon.",
        "code": "Write a Python function that returns the sum of two numbers.",
        "cot": "Let me think step by step about how to solve this problem: What is 15% of 80?",
    }

    for model_path, train_type in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_path.split('/')[-1]} ({train_type})")
        print("="*70)

        model, tokenizer = load(model_path)

        print(f"\n| Task | Exp.Ratio | Peak Layer | Growth | Recovery | Peak | Final |")
        print(f"|------|-----------|------------|--------|----------|------|-------|")

        task_results = []

        for task_name, prompt in prompts.items():
            norms = compute_norm_trajectory(model, tokenizer, prompt)
            metrics = compute_expansion_metrics(norms)

            print(f"| {task_name:8s} | {metrics['expansion_ratio']:9.2f} | "
                  f"{metrics['peak_layer']:6d} ({metrics['peak_layer_frac']:.0%}) | "
                  f"{metrics['growth_factor']:6.1f}× | {metrics['recovery_factor']:6.1f}× | "
                  f"{metrics['peak_norm']:4.0f} | {metrics['final_norm']:5.0f} |")

            task_results.append(metrics)

        # Summary statistics
        ratios = [r['expansion_ratio'] for r in task_results]
        print(f"\nExpansion ratio: mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}, "
              f"range=[{min(ratios):.2f}, {max(ratios):.2f}]")

        # Decomposition: expansion_ratio = growth_factor / recovery_factor
        growth_factors = [r['growth_factor'] for r in task_results]
        recovery_factors = [r['recovery_factor'] for r in task_results]

        print(f"\nDecomposition (expansion_ratio = growth / recovery):")
        print(f"  Growth factor std: {np.std(growth_factors):.3f}")
        print(f"  Recovery factor std: {np.std(recovery_factors):.3f}")

        if np.std(growth_factors) > np.std(recovery_factors):
            print("  → Variance driven by GROWTH (how much norm increases to peak)")
        else:
            print("  → Variance driven by RECOVERY (how much norm is preserved)")

        del model
        mx.metal.clear_cache()

    # Cross-model analysis
    print("\n" + "="*70)
    print("  WHY DOES RLHF FLATTEN EXPANSION_RATIO?")
    print("="*70)
    print("""
Hypothesis: RLHF training optimizes for consistent processing geometry.

If expansion_ratio = norm_recovery * layer_ratio = 1.0:
  - Either norm_recovery ≈ 1/layer_ratio (compensating)
  - Or both ≈ 1.0 (peak at middle, full recovery)

Key questions:
1. Does RLHF change peak location? (affects layer_ratio)
2. Does RLHF change norm recovery? (affects norm_recovery)
3. Or does it make them covary to maintain ratio = 1.0?
""")


if __name__ == "__main__":
    main()
