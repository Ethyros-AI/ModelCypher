#!/usr/bin/env python3
"""Experiment 69: Minimal Prime Discovery.

Priming works perfectly. But how minimal can the prime be?

Test increasingly minimal primes to find the smallest effective intervention.
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

        top_token = int(np.argmax(logits_np))
        predicted = tokenizer.decode([top_token]).strip()

        # Get target probability
        target_id = get_digit_token(tokenizer, expected)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        target_prob = probs[target_id] if target_id >= 0 else 0.0

        correct = expected in predicted or predicted == expected

        results.append({
            "prompt": raw_problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "target_prob": float(target_prob),
        })

    return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 69: MINIMAL PRIME DISCOVERY")
    logger.info("=" * 60)

    # Standard test problems
    test_problems = [
        ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
        ("2+2=", "4"), ("3+3=", "6"), ("2+3=", "5"), ("4+5=", "9"),
    ]

    # Primes from longest to shortest
    primes = {
        "full_explanation": "Adding 1 means the next number. Addition combines two numbers.",
        "single_sentence": "Adding 1 means the next number.",
        "short_explanation": "Addition means combining.",
        "instruction": "Add:",
        "math_context": "Math:",
        "equals_hint": "Answer:",
        "just_number": "1.",
        "colon": ":",
        "single_word": "Plus",
        "empty": "",
    }

    all_results = {}

    logger.info(f"\n{'Prime':<30} {'Accuracy':>10} {'Mean P(target)':>15}")
    logger.info("-" * 60)

    for name, prime in primes.items():
        results = evaluate_with_prime(model, tokenizer, prime, test_problems)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_prob = np.mean([r["target_prob"] for r in results])

        all_results[name] = {
            "prime": prime,
            "accuracy": accuracy,
            "mean_target_prob": float(mean_prob),
            "details": results,
        }

        prime_display = prime[:25] + "..." if len(prime) > 25 else prime
        logger.info(f"{prime_display:<30} {accuracy:>10.0%} {mean_prob:>14.1%}")

    # Find minimal effective prime
    working_primes = [(k, v) for k, v in all_results.items() if v["accuracy"] >= 0.8]
    if working_primes:
        # Sort by prime length (shortest first)
        working_primes.sort(key=lambda x: len(x[1]["prime"]))
        minimal = working_primes[0]
        logger.info(f"\n=== MINIMAL EFFECTIVE PRIME ===")
        logger.info(f"Name: {minimal[0]}")
        logger.info(f"Prime: '{minimal[1]['prime']}'")
        logger.info(f"Length: {len(minimal[1]['prime'])} characters")
        logger.info(f"Accuracy: {minimal[1]['accuracy']:.0%}")

    # Also test: does the prime need to be semantically related?
    logger.info(f"\n=== SEMANTIC RELEVANCE TEST ===")

    semantic_primes = {
        "related_math": "Numbers add up.",
        "related_counting": "Counting:",
        "unrelated_food": "I like pizza.",
        "unrelated_weather": "The sky is blue.",
        "unrelated_nonsense": "Xyz abc qwerty.",
        "numbers_only": "1 2 3 4 5.",
        "math_symbols": "+ - × ÷ =",
    }

    for name, prime in semantic_primes.items():
        results = evaluate_with_prime(model, tokenizer, prime, test_problems)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_prob = np.mean([r["target_prob"] for r in results])

        all_results[f"semantic_{name}"] = {
            "prime": prime,
            "accuracy": accuracy,
            "mean_target_prob": float(mean_prob),
            "details": results,
        }

        logger.info(f"'{prime:<25}' → {accuracy:>10.0%}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    baseline = all_results["empty"]["accuracy"]
    best_name = max(all_results, key=lambda x: all_results[x]["accuracy"])
    best = all_results[best_name]

    logger.info(f"Baseline (no prime): {baseline:.0%}")
    logger.info(f"Best: {best_name} → {best['accuracy']:.0%}")

    # Check if semantic content matters
    related_acc = np.mean([all_results[k]["accuracy"] for k in all_results
                          if "semantic_related" in k or k in ["full_explanation", "single_sentence"]])
    unrelated_acc = np.mean([all_results[k]["accuracy"] for k in all_results
                            if "unrelated" in k])

    if related_acc > unrelated_acc + 0.1:
        logger.info(f"\n*** SEMANTIC CONTENT MATTERS ***")
        logger.info(f"Related primes: {related_acc:.0%}")
        logger.info(f"Unrelated primes: {unrelated_acc:.0%}")
    else:
        logger.info(f"\n*** SEMANTIC CONTENT MAY NOT MATTER ***")
        logger.info(f"Any prefix might work as a context signal")

    # Save results
    output_path = "data/experiments/minimal_prime_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
