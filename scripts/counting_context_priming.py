#!/usr/bin/env python3
"""Experiment 67: Counting Context Priming.

The model knows counting but not symbolic arithmetic.
What if we prime with counting context BEFORE the symbolic prompt?

Hypothesis: "1, 2, 3, 4. So 4+1=" might work better than "4+1="
because the counting context activates the successor circuit.
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


def evaluate_prompts(model, tokenizer, prompts):
    """Evaluate a list of (prompt, expected) pairs."""
    import mlx.core as mx

    results = []

    for prompt, expected in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)

        top_token = int(np.argmax(logits_np))
        predicted = tokenizer.decode([top_token]).strip()

        target_id = get_digit_token(tokenizer, expected)
        target_rank = int((np.argsort(logits_np)[::-1] == target_id).nonzero()[0][0]) if target_id >= 0 else -1

        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        target_prob = probs[target_id] if target_id >= 0 else 0.0

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
    logger.info("EXPERIMENT 67: COUNTING CONTEXT PRIMING")
    logger.info("=" * 60)

    # Different priming strategies
    strategies = {}

    # Strategy 1: Raw symbolic (baseline)
    strategies["raw_symbolic"] = [
        ("1+1=", "2"),
        ("2+1=", "3"),
        ("3+1=", "4"),
        ("4+1=", "5"),
        ("5+1=", "6"),
        ("6+1=", "7"),
        ("7+1=", "8"),
        ("8+1=", "9"),
    ]

    # Strategy 2: Counting sequence priming
    strategies["counting_prime"] = [
        ("1. 1+1=", "2"),
        ("1, 2. 2+1=", "3"),
        ("1, 2, 3. 3+1=", "4"),
        ("1, 2, 3, 4. 4+1=", "5"),
        ("1, 2, 3, 4, 5. 5+1=", "6"),
        ("1, 2, 3, 4, 5, 6. 6+1=", "7"),
        ("1, 2, 3, 4, 5, 6, 7. 7+1=", "8"),
        ("1, 2, 3, 4, 5, 6, 7, 8. 8+1=", "9"),
    ]

    # Strategy 3: Succession priming
    strategies["succession_prime"] = [
        ("The number after 1 is 2. So 1+1=", "2"),
        ("The number after 2 is 3. So 2+1=", "3"),
        ("The number after 3 is 4. So 3+1=", "4"),
        ("The number after 4 is 5. So 4+1=", "5"),
        ("The number after 5 is 6. So 5+1=", "6"),
        ("The number after 6 is 7. So 6+1=", "7"),
        ("The number after 7 is 8. So 7+1=", "8"),
        ("The number after 8 is 9. So 8+1=", "9"),
    ]

    # Strategy 4: Addition explanation
    strategies["addition_explain"] = [
        ("Adding 1 means the next number. 1+1=", "2"),
        ("Adding 1 means the next number. 2+1=", "3"),
        ("Adding 1 means the next number. 3+1=", "4"),
        ("Adding 1 means the next number. 4+1=", "5"),
        ("Adding 1 means the next number. 5+1=", "6"),
        ("Adding 1 means the next number. 6+1=", "7"),
        ("Adding 1 means the next number. 7+1=", "8"),
        ("Adding 1 means the next number. 8+1=", "9"),
    ]

    # Strategy 5: Q&A style
    strategies["qa_style"] = [
        ("Q: What is 1+1? A:", "2"),
        ("Q: What is 2+1? A:", "3"),
        ("Q: What is 3+1? A:", "4"),
        ("Q: What is 4+1? A:", "5"),
        ("Q: What is 5+1? A:", "6"),
        ("Q: What is 6+1? A:", "7"),
        ("Q: What is 7+1? A:", "8"),
        ("Q: What is 8+1? A:", "9"),
    ]

    # Strategy 6: Equivalence priming
    strategies["equivalence_prime"] = [
        ("Counting: 1, 2. Adding: 1+1=", "2"),
        ("Counting: 2, 3. Adding: 2+1=", "3"),
        ("Counting: 3, 4. Adding: 3+1=", "4"),
        ("Counting: 4, 5. Adding: 4+1=", "5"),
        ("Counting: 5, 6. Adding: 5+1=", "6"),
        ("Counting: 6, 7. Adding: 6+1=", "7"),
        ("Counting: 7, 8. Adding: 7+1=", "8"),
        ("Counting: 8, 9. Adding: 8+1=", "9"),
    ]

    # Strategy 7: Calculator style
    strategies["calculator"] = [
        ("Calculate: 1+1=", "2"),
        ("Calculate: 2+1=", "3"),
        ("Calculate: 3+1=", "4"),
        ("Calculate: 4+1=", "5"),
        ("Calculate: 5+1=", "6"),
        ("Calculate: 6+1=", "7"),
        ("Calculate: 7+1=", "8"),
        ("Calculate: 8+1=", "9"),
    ]

    # Evaluate each strategy
    all_results = {}

    logger.info(f"\n{'Strategy':<25} {'Accuracy':>10} {'Mean Rank':>10}")
    logger.info("-" * 50)

    for name, prompts in strategies.items():
        results = evaluate_prompts(model, tokenizer, prompts)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_rank = np.mean([r["target_rank"] for r in results])

        all_results[name] = {
            "accuracy": accuracy,
            "mean_rank": float(mean_rank),
            "details": results,
        }

        logger.info(f"{name:<25} {accuracy:>10.0%} {mean_rank:>10.1f}")

    # Find best strategy
    best_name = max(all_results, key=lambda x: all_results[x]["accuracy"])
    best = all_results[best_name]

    logger.info(f"\n=== BEST STRATEGY: {best_name} ===")
    logger.info(f"Accuracy: {best['accuracy']:.0%}")

    for r in best["details"]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  '{r['prompt'][:40]}...' → '{r['predicted']}' ({status}) rank={r['target_rank']+1}")

    baseline = all_results["raw_symbolic"]

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline (raw_symbolic): {baseline['accuracy']:.0%}")
    logger.info(f"Best ({best_name}): {best['accuracy']:.0%}")

    if best["accuracy"] > baseline["accuracy"]:
        logger.info(f"\n*** CONTEXT PRIMING IMPROVED ACCURACY ***")
        logger.info(f"Improvement: {best['accuracy'] - baseline['accuracy']:.0%}")
    else:
        logger.info(f"\n*** CONTEXT PRIMING DID NOT HELP ***")

    # Save results
    output_path = "data/experiments/counting_context_priming.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
