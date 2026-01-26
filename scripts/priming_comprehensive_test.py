#!/usr/bin/env python3
"""Experiment 68: Comprehensive Priming Test.

"Adding 1 means the next number" achieved 100% on +1 problems.
Does this generalize?

Test:
1. Addition beyond +1 (2+2, 3+4, etc.)
2. Subtraction
3. Larger numbers
4. Multi-digit answers
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
        prompt = f"{prime_text} {raw_problem}"

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)

        top_token = int(np.argmax(logits_np))
        predicted = tokenizer.decode([top_token]).strip()

        correct = expected in predicted or predicted == expected

        results.append({
            "prompt": raw_problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        })

    return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 68: COMPREHENSIVE PRIMING TEST")
    logger.info("=" * 60)

    # Test categories
    test_categories = {
        "plus_1": {
            "prime": "Adding 1 means the next number.",
            "problems": [
                ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
                ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9"),
            ]
        },
        "plus_2": {
            "prime": "Adding 2 means skip one number.",
            "problems": [
                ("1+2=", "3"), ("2+2=", "4"), ("3+2=", "5"), ("4+2=", "6"),
                ("5+2=", "7"), ("6+2=", "8"), ("7+2=", "9"),
            ]
        },
        "plus_various": {
            "prime": "Addition combines two numbers.",
            "problems": [
                ("2+3=", "5"), ("3+4=", "7"), ("4+5=", "9"), ("5+5=", "10"),
                ("3+3=", "6"), ("4+4=", "8"), ("2+5=", "7"),
            ]
        },
        "minus_1": {
            "prime": "Subtracting 1 means the previous number.",
            "problems": [
                ("2-1=", "1"), ("3-1=", "2"), ("4-1=", "3"), ("5-1=", "4"),
                ("6-1=", "5"), ("7-1=", "6"), ("8-1=", "7"), ("9-1=", "8"),
            ]
        },
        "minus_various": {
            "prime": "Subtraction finds the difference.",
            "problems": [
                ("5-2=", "3"), ("7-3=", "4"), ("9-4=", "5"), ("10-5=", "5"),
                ("8-4=", "4"), ("6-3=", "3"),
            ]
        },
        "larger_numbers": {
            "prime": "Addition combines two numbers.",
            "problems": [
                ("10+1=", "11"), ("10+5=", "15"), ("15+5=", "20"),
                ("20+10=", "30"), ("11+11=", "22"),
            ]
        },
        "equivalence": {
            "prime": "Counting: 4, 5. Adding:",
            "problems": [
                ("4+1=", "5"),
            ]
        },
        "equivalence_extended": {
            "prime": "Counting up means adding. Counting: 3, 4, 5. Adding:",
            "problems": [
                ("3+2=", "5"), ("4+1=", "5"),
            ]
        }
    }

    # Also test without priming for comparison
    test_categories["no_prime"] = {
        "prime": "",
        "problems": [
            ("1+1=", "2"), ("2+2=", "4"), ("3+3=", "6"), ("4+4=", "8"),
            ("5-1=", "4"), ("10-5=", "5"), ("2+3=", "5"),
        ]
    }

    all_results = {}

    logger.info(f"\n{'Category':<25} {'Accuracy':>10} {'Correct':>10}")
    logger.info("-" * 50)

    for category, config in test_categories.items():
        results = evaluate_with_prime(model, tokenizer, config["prime"], config["problems"])
        accuracy = sum(r["correct"] for r in results) / len(results)
        correct_count = sum(r["correct"] for r in results)

        all_results[category] = {
            "prime": config["prime"],
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(results),
            "details": results,
        }

        logger.info(f"{category:<25} {accuracy:>10.0%} {correct_count:>5}/{len(results)}")

    # Detailed results for each category
    for category, data in all_results.items():
        logger.info(f"\n=== {category.upper()} ===")
        logger.info(f"Prime: '{data['prime']}'")
        for r in data["details"]:
            status = "✓" if r["correct"] else "✗"
            logger.info(f"  {r['prompt']} → '{r['predicted']}' ({status}) [expected {r['expected']}]")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    working = [(k, v) for k, v in all_results.items() if v["accuracy"] >= 0.8]
    partial = [(k, v) for k, v in all_results.items() if 0.3 <= v["accuracy"] < 0.8]
    failing = [(k, v) for k, v in all_results.items() if v["accuracy"] < 0.3]

    logger.info(f"\nWorking (≥80%):")
    for name, data in working:
        logger.info(f"  {name}: {data['accuracy']:.0%}")

    logger.info(f"\nPartial (30-80%):")
    for name, data in partial:
        logger.info(f"  {name}: {data['accuracy']:.0%}")

    logger.info(f"\nFailing (<30%):")
    for name, data in failing:
        logger.info(f"  {name}: {data['accuracy']:.0%}")

    # Save results
    output_path = "data/experiments/priming_comprehensive_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
