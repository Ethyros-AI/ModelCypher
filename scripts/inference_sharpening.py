#!/usr/bin/env python3
"""Experiment 64: Inference-Time Sharpening.

The model already has "5" as a plausible prediction for "4+1=" (16.5%).
Training disrupts things. What if we just sharpen at inference time?

This is geometry-based: the temperature τ determines sharpness.
τ < 1 sharpens, τ > 1 softens.

We can derive optimal τ from the ratio of sharpnesses:
- Counting gap: 1.00
- Symbolic gap: 0.33
- Ratio: 3.08

Temperature that maps symbolic sharpness to counting sharpness:
If gap ∝ 1/τ, then τ = symbolic_gap / counting_gap ≈ 0.33
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYMBOLIC_PROMPTS = [
    ("1+1=", "2"),
    ("2+1=", "3"),
    ("3+1=", "4"),
    ("4+1=", "5"),
    ("5+1=", "6"),
    ("6+1=", "7"),
    ("7+1=", "8"),
    ("8+1=", "9"),
    ("2+2=", "4"),
    ("3+3=", "6"),
]


def softmax(logits, temperature=1.0):
    """Softmax with temperature."""
    x = logits / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def evaluate_with_temperature(model, tokenizer, temperature):
    """Evaluate accuracy at a given temperature."""
    import mlx.core as mx

    results = []
    for prompt, expected in SYMBOLIC_PROMPTS:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist())

        # Apply temperature
        probs = softmax(logits_np, temperature)

        # Top prediction with temperature
        top_token = np.argmax(probs)
        predicted = tokenizer.decode([top_token]).strip()
        correct = expected in predicted or predicted == expected

        # Target token info
        target_tokens = tokenizer.encode(expected)
        target_id = target_tokens[0] if target_tokens else -1
        target_prob = probs[target_id] if target_id >= 0 else 0.0
        target_rank = (np.argsort(probs)[::-1] == target_id).nonzero()[0][0] if target_id >= 0 else -1

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "target_prob": float(target_prob),
            "target_rank": int(target_rank),
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_target_prob = np.mean([r["target_prob"] for r in results])
    mean_target_rank = np.mean([r["target_rank"] for r in results])

    return {
        "temperature": temperature,
        "accuracy": accuracy,
        "mean_target_prob": mean_target_prob,
        "mean_target_rank": mean_target_rank,
        "details": results,
    }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 64: INFERENCE-TIME SHARPENING")
    logger.info("=" * 60)
    logger.info("\nNo training - just adjust temperature at inference time")

    # Geometry-derived temperatures
    # From Exp 61: counting gap = 1.00, symbolic gap = 0.33
    # Ratio = 3.08, so τ ≈ 0.33 should match sharpness
    geometry_derived_temp = 0.33

    # Test range of temperatures
    temperatures = [1.0, 0.7, 0.5, geometry_derived_temp, 0.2, 0.1, 0.05]

    results = {"temperatures": []}

    logger.info("\n=== TEMPERATURE SWEEP ===")
    logger.info(f"{'Temp':>6} {'Accuracy':>10} {'Mean Target Prob':>18} {'Mean Rank':>10}")
    logger.info("-" * 50)

    for temp in temperatures:
        eval_result = evaluate_with_temperature(model, tokenizer, temp)
        results["temperatures"].append(eval_result)

        is_derived = " ← geometry" if temp == geometry_derived_temp else ""
        logger.info(f"{temp:>6.2f} {eval_result['accuracy']:>10.0%} {eval_result['mean_target_prob']:>17.1%} {eval_result['mean_target_rank']:>10.1f}{is_derived}")

    # Find best temperature
    best = max(results["temperatures"], key=lambda x: x["accuracy"])

    logger.info(f"\n=== BEST TEMPERATURE: {best['temperature']} ===")
    logger.info(f"Accuracy: {best['accuracy']:.0%}")
    logger.info(f"Mean target prob: {best['mean_target_prob']:.1%}")

    for r in best["details"]:
        logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({'✓' if r['correct'] else '✗'}) "
                   f"p={r['target_prob']:.1%}, rank={r['target_rank']+1}")

    # Compare to baseline (temp=1.0)
    baseline = next(r for r in results["temperatures"] if r["temperature"] == 1.0)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline (τ=1.0): {baseline['accuracy']:.0%}")
    logger.info(f"Best (τ={best['temperature']}): {best['accuracy']:.0%}")
    logger.info(f"Geometry-derived (τ={geometry_derived_temp}): {next(r for r in results['temperatures'] if r['temperature'] == geometry_derived_temp)['accuracy']:.0%}")

    if best["accuracy"] > baseline["accuracy"]:
        logger.info(f"\n*** INFERENCE SHARPENING IMPROVED ACCURACY ***")
        logger.info(f"Improvement: {best['accuracy'] - baseline['accuracy']:.0%}")
        results["conclusion"] = "success"
    else:
        logger.info(f"\n*** INFERENCE SHARPENING DID NOT HELP ***")
        results["conclusion"] = "failed"

    results["best_temperature"] = best["temperature"]
    results["geometry_derived_temperature"] = geometry_derived_temp

    output_path = "data/experiments/inference_sharpening.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
