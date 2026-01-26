#!/usr/bin/env python3
"""Experiment 66: Activation Steering at Inference Time.

The model has counting activations that work and symbolic activations that don't.
Can we nudge symbolic activations toward counting at inference time?

Method:
1. Compute "counting direction" = mean(counting_activations) - mean(symbolic_activations)
2. At inference, add α × counting_direction to symbolic activations
3. Test if this improves accuracy

No training - pure inference-time intervention.
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


def get_digit_token(tokenizer, digit_str):
    """Get the actual digit token ID."""
    tokens = tokenizer.encode(digit_str)
    if len(tokens) > 1 and tokens[0] == 1:
        return tokens[1]
    return tokens[0] if tokens else -1


# Matched pairs: counting and symbolic that should give same answer
MATCHED_PAIRS = [
    ("1, 2, 3, 4,", "4+1=", "5"),
    ("2, 3, 4, 5,", "5+1=", "6"),
    ("3, 4, 5, 6,", "6+1=", "7"),
    ("4, 5, 6, 7,", "7+1=", "8"),
    ("5, 6, 7, 8,", "8+1=", "9"),
]

# Additional test prompts
TEST_SYMBOLIC = [
    ("1+1=", "2"),
    ("2+1=", "3"),
    ("3+1=", "4"),
    ("4+1=", "5"),
    ("9+1=", "10"),
]


def compute_steering_direction(model, tokenizer):
    """Compute the counting→symbolic steering direction."""
    import mlx.core as mx

    logger.info("Computing steering direction...")

    counting_acts = []
    symbolic_acts = []

    for counting_prompt, symbolic_prompt, _ in MATCHED_PAIRS:
        # Counting activation
        tokens = tokenizer.encode(counting_prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        counting_acts.append(np.array(logits[0, -1, :].tolist(), dtype=np.float32))

        # Symbolic activation
        tokens = tokenizer.encode(symbolic_prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        symbolic_acts.append(np.array(logits[0, -1, :].tolist(), dtype=np.float32))

    counting_mean = np.mean(counting_acts, axis=0)
    symbolic_mean = np.mean(symbolic_acts, axis=0)

    # Steering direction: from symbolic toward counting
    steering_direction = counting_mean - symbolic_mean

    # Normalize
    steering_norm = np.linalg.norm(steering_direction)
    steering_direction_normalized = steering_direction / (steering_norm + 1e-10)

    logger.info(f"Steering direction norm: {steering_norm:.2f}")
    logger.info(f"Cosine(counting_mean, symbolic_mean): {np.dot(counting_mean, symbolic_mean) / (np.linalg.norm(counting_mean) * np.linalg.norm(symbolic_mean)):.4f}")

    return steering_direction, steering_direction_normalized, steering_norm


def evaluate_with_steering(model, tokenizer, steering_direction, alpha):
    """Evaluate symbolic prompts with activation steering."""
    import mlx.core as mx

    results = []

    for prompt, expected in TEST_SYMBOLIC:
        # Get base logits
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        base_logits = np.array(logits[0, -1, :].tolist(), dtype=np.float32)

        # Apply steering
        steered_logits = base_logits + alpha * steering_direction

        # Get prediction from steered logits
        probs = np.exp(steered_logits - steered_logits.max())
        probs = probs / probs.sum()

        top_token = np.argmax(steered_logits)
        predicted = tokenizer.decode([top_token]).strip()

        # Check if correct
        target_id = get_digit_token(tokenizer, expected)
        target_prob = probs[target_id] if target_id >= 0 else 0.0
        target_rank = int((np.argsort(steered_logits)[::-1] == target_id).nonzero()[0][0]) if target_id >= 0 else -1

        correct = expected in predicted or predicted == expected

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "target_rank": target_rank,
            "target_prob": float(target_prob),
        })

    return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 66: ACTIVATION STEERING")
    logger.info("=" * 60)

    # Compute steering direction
    steering_dir, steering_dir_norm, steering_norm = compute_steering_direction(model, tokenizer)

    # Test various steering strengths
    alphas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    all_results = []

    logger.info("\n=== STEERING SWEEP ===")
    logger.info(f"{'Alpha':>8} {'Accuracy':>10} {'Mean Rank':>10} {'Mean Prob':>10}")
    logger.info("-" * 45)

    for alpha in alphas:
        results = evaluate_with_steering(model, tokenizer, steering_dir, alpha)

        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_rank = np.mean([r["target_rank"] for r in results])
        mean_prob = np.mean([r["target_prob"] for r in results])

        logger.info(f"{alpha:>8.1f} {accuracy:>10.0%} {mean_rank:>10.1f} {mean_prob:>10.1%}")

        all_results.append({
            "alpha": alpha,
            "accuracy": accuracy,
            "mean_rank": float(mean_rank),
            "mean_prob": float(mean_prob),
            "details": results,
        })

    # Find best alpha
    best = max(all_results, key=lambda x: x["accuracy"])

    logger.info(f"\n=== BEST ALPHA: {best['alpha']} ===")
    logger.info(f"Accuracy: {best['accuracy']:.0%}")

    for r in best["details"]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({status}) rank={r['target_rank']+1}")

    # Compare to baseline
    baseline = next(r for r in all_results if r["alpha"] == 0.0)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline (α=0): {baseline['accuracy']:.0%}")
    logger.info(f"Best (α={best['alpha']}): {best['accuracy']:.0%}")

    if best["accuracy"] > baseline["accuracy"]:
        logger.info(f"\n*** ACTIVATION STEERING IMPROVED ACCURACY ***")
        logger.info(f"Improvement: {best['accuracy'] - baseline['accuracy']:.0%}")
    else:
        logger.info(f"\n*** ACTIVATION STEERING DID NOT HELP ***")

    # Save results
    output_path = "data/experiments/activation_steering.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "steering_norm": float(steering_norm),
            "best_alpha": best["alpha"],
            "baseline_accuracy": baseline["accuracy"],
            "best_accuracy": best["accuracy"],
            "results": all_results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
