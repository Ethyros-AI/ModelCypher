#!/usr/bin/env python3
"""Experiment 71: Period Prime Test.

A single period before the problem achieves 88.5% on 4+1=.
Does this generalize?

If ". 4+1=" works, does ". 2+2=" work?
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


def evaluate_with_prime(model, tokenizer, prime_text, problems):
    """Evaluate problems with a given prime."""
    import mlx.core as mx

    results = []

    for raw_problem, expected in problems:
        if prime_text:
            prompt = f"{prime_text} {raw_problem}"
        else:
            prompt = raw_problem

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)

        # Get probabilities
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        target_id = get_digit_token(tokenizer, expected)
        target_prob = probs[target_id] if target_id >= 0 else 0.0

        # Question mark probability
        q_tokens = tokenizer.encode("?")
        q_id = q_tokens[1] if len(q_tokens) > 1 else q_tokens[0]
        q_prob = probs[q_id]

        correct = expected in predicted or predicted == expected

        results.append({
            "prompt": raw_problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "target_prob": float(target_prob),
            "question_prob": float(q_prob),
        })

    return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 71: PERIOD PRIME TEST")
    logger.info("=" * 60)

    # Comprehensive test problems
    problems = [
        # Basic +1
        ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
        ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9"),
        # Various addition
        ("2+2=", "4"), ("3+3=", "6"), ("4+4=", "8"), ("2+3=", "5"),
        ("3+4=", "7"), ("4+5=", "9"), ("5+5=", "10"),
        # Subtraction
        ("2-1=", "1"), ("5-1=", "4"), ("9-1=", "8"),
        ("5-2=", "3"), ("7-3=", "4"), ("10-5=", "5"),
        # Larger
        ("10+1=", "11"), ("10+5=", "15"), ("15+5=", "20"),
    ]

    primes_to_test = {
        "no_prime": "",
        "period": ".",
        "double_period": "..",
        "period_space": ". ",
        "three_periods": "...",
        "exclaim": "!",
    }

    all_results = {}

    for prime_name, prime in primes_to_test.items():
        results = evaluate_with_prime(model, tokenizer, prime, problems)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_target_prob = np.mean([r["target_prob"] for r in results])
        mean_q_prob = np.mean([r["question_prob"] for r in results])

        all_results[prime_name] = {
            "prime": prime,
            "accuracy": accuracy,
            "mean_target_prob": float(mean_target_prob),
            "mean_question_prob": float(mean_q_prob),
            "details": results,
        }

        logger.info(f"\n=== {prime_name.upper()} ('{prime}') ===")
        logger.info(f"Accuracy: {accuracy:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
        logger.info(f"Mean P(target): {mean_target_prob:.1%}")
        logger.info(f"Mean P(?): {mean_q_prob:.1%}")

        # Show errors
        errors = [r for r in results if not r["correct"]]
        if errors and len(errors) <= 10:
            logger.info("Errors:")
            for r in errors:
                logger.info(f"  {r['prompt']} → '{r['predicted']}' (expected {r['expected']}, P={r['target_prob']:.1%})")

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\n{'Prime':<15} {'Accuracy':>10} {'P(target)':>12} {'P(?)':>10}")
    logger.info("-" * 50)

    for name, data in all_results.items():
        logger.info(f"{name:<15} {data['accuracy']:>10.0%} {data['mean_target_prob']:>11.1%} "
                   f"{data['mean_question_prob']:>10.1%}")

    # Find best minimal prime
    best_name = max(all_results, key=lambda x: all_results[x]["accuracy"])
    best = all_results[best_name]

    logger.info(f"\n*** BEST MINIMAL PRIME: '{best['prime']}' → {best['accuracy']:.0%} ***")

    # Compare to full semantic prime for reference
    logger.info("\n=== COMPARISON WITH SEMANTIC PRIME ===")
    semantic_results = evaluate_with_prime(
        model, tokenizer,
        "Adding 1 means the next number.",
        problems[:8]  # Just +1 problems
    )
    semantic_acc = sum(r["correct"] for r in semantic_results) / len(semantic_results)

    period_results = [r for r in all_results["period"]["details"][:8]]
    period_acc = sum(r["correct"] for r in period_results) / len(period_results)

    logger.info(f"Semantic prime on +1 problems: {semantic_acc:.0%}")
    logger.info(f"Period prime on +1 problems: {period_acc:.0%}")

    # Save results
    output_path = "data/experiments/period_prime_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
