#!/usr/bin/env python3
"""Test if fine-tuning shifts the activation peak earlier.

Compare base models to their fine-tuned versions:
- LFM2-1.2B (base) vs LFM2.5-1.2B-Thinking
- LFM2-1.2B (base) vs LFM2.5-1.2B-Instruct
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.domain.geometry.differentiable_expansion import (
    compute_trajectory_norms,
    compute_expansion_metrics,
)


def analyze_model(model_path: str, prompts: list[str]) -> dict:
    from mlx_lm import load

    print(f"  Loading {Path(model_path).name}...")
    model, tokenizer = load(model_path)

    peaks = []
    ratios = []
    trajectories = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array(tokens)

        trajectory = compute_trajectory_norms(model, input_ids)
        mx.eval(trajectory)
        traj_np = np.array(trajectory.tolist())

        n_layers = len(traj_np) - 1
        peak_layer = int(np.argmax(traj_np))
        peak_pct = peak_layer / n_layers * 100

        peaks.append(peak_pct)
        trajectories.append(traj_np)

        metrics = compute_expansion_metrics(trajectory, exact=True)
        ratios.append(metrics["expansion_ratio"])

    del model, tokenizer
    mx.clear_cache()

    return {
        "mean_peak_pct": float(np.mean(peaks)),
        "std_peak_pct": float(np.std(peaks)),
        "peaks": peaks,
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "n_layers": n_layers,
        "example_trajectory": trajectories[0].tolist(),
    }


def main():
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")

    # Comparison pairs: (base, fine-tuned, description)
    comparisons = [
        ("LFM2-1.2B-bf16", "LFM2.5-1.2B-Thinking-bf16", "Base → Thinking"),
        ("LFM2-1.2B-bf16", "LFM2.5-1.2B-Instruct-bf16", "Base → Instruct"),
    ]

    # Test prompts - variety of types
    prompts = [
        # Factual
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in our solar system is",
        # Reasoning
        "If all cats are animals and some animals are pets, then",
        "The sum of 15 and 27 equals",
        # Creative
        "Once upon a time in a magical forest,",
        "The robot looked at the sunset and thought",
        # Instruction-like
        "Please explain how photosynthesis works:",
        "List three benefits of exercise:",
        "Describe the process of making coffee:",
    ]

    print("=" * 70)
    print("FINE-TUNING PEAK SHIFT ANALYSIS")
    print("=" * 70)

    for base_name, tuned_name, desc in comparisons:
        base_path = str(models_dir / base_name)
        tuned_path = str(models_dir / tuned_name)

        if not (models_dir / base_name).exists():
            print(f"Skipping {desc}: {base_name} not found")
            continue
        if not (models_dir / tuned_name).exists():
            print(f"Skipping {desc}: {tuned_name} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Comparison: {desc}")
        print(f"{'='*70}")

        base_results = analyze_model(base_path, prompts)
        tuned_results = analyze_model(tuned_path, prompts)

        print(f"\n{'Metric':<25} {'Base':<20} {'Fine-tuned':<20} {'Shift':<15}")
        print("-" * 80)

        base_peak = base_results["mean_peak_pct"]
        tuned_peak = tuned_results["mean_peak_pct"]
        shift = tuned_peak - base_peak

        print(f"{'Peak position (%):':<25} {base_peak:>6.1f} ± {base_results['std_peak_pct']:.1f}       {tuned_peak:>6.1f} ± {tuned_results['std_peak_pct']:.1f}       {shift:>+6.1f}%")

        base_ratio = base_results["mean_ratio"]
        tuned_ratio = tuned_results["mean_ratio"]
        ratio_shift = tuned_ratio - base_ratio

        print(f"{'Expansion ratio:':<25} {base_ratio:>6.2f} ± {base_results['std_ratio']:.2f}       {tuned_ratio:>6.2f} ± {tuned_results['std_ratio']:.2f}       {ratio_shift:>+6.2f}")

        # Interpretation
        print(f"\nInterpretation:")
        if shift < -5:
            print(f"  ✓ Fine-tuning shifted peak EARLIER by {abs(shift):.1f}%")
            print(f"  ✓ This creates more compression in final layers")
        elif shift > 5:
            print(f"  ✗ Fine-tuning shifted peak LATER by {shift:.1f}%")
            print(f"  ✗ Unexpected - reduces compression")
        else:
            print(f"  ~ Peak position similar (shift = {shift:.1f}%)")

        # Per-prompt breakdown
        print(f"\nPer-prompt peak positions:")
        print(f"  {'Prompt':<45} {'Base':>8} {'Tuned':>8} {'Shift':>8}")
        for i, prompt in enumerate(prompts):
            bp = base_results["peaks"][i]
            tp = tuned_results["peaks"][i]
            print(f"  {prompt[:43]:<45} {bp:>7.1f}% {tp:>7.1f}% {tp-bp:>+7.1f}%")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print("""
If fine-tuning shifts the peak earlier:
  → Confirms that instruction tuning creates a compression phase
  → The model learns to "focus" representation toward the answer

If peaks are similar:
  → The shift may happen during pre-training, not fine-tuning
  → Or the specific fine-tuning method matters
""")


if __name__ == "__main__":
    main()
