#!/usr/bin/env python3
"""Experiment 72: Operation-Specific Primes.

Period prime works for addition (88%) but fails for subtraction.
Does each operation need its own semantic prime?
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
    logger.info("EXPERIMENT 72: OPERATION-SPECIFIC PRIMES")
    logger.info("=" * 60)

    # Test sets
    addition_problems = [
        ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
        ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9"),
        ("2+2=", "4"), ("3+3=", "6"), ("2+3=", "5"), ("4+5=", "9"),
    ]

    subtraction_problems = [
        ("2-1=", "1"), ("3-1=", "2"), ("4-1=", "3"), ("5-1=", "4"),
        ("6-1=", "5"), ("7-1=", "6"), ("8-1=", "7"), ("9-1=", "8"),
        ("5-2=", "3"), ("7-3=", "4"), ("10-5=", "5"), ("8-4=", "4"),
    ]

    # Primes for each operation
    addition_primes = {
        "none": "",
        "period": ".",
        "generic": "Math:",
        "semantic_add": "Adding means combining.",
        "semantic_next": "Adding 1 means the next number.",
        "counting": "Counting up:",
    }

    subtraction_primes = {
        "none": "",
        "period": ".",
        "generic": "Math:",
        "semantic_sub": "Subtracting means taking away.",
        "semantic_prev": "Subtracting 1 means the previous number.",
        "semantic_diff": "Subtraction finds the difference.",
        "counting_down": "Counting down:",
    }

    all_results = {"addition": {}, "subtraction": {}}

    # Test addition
    logger.info("\n=== ADDITION ===")
    logger.info(f"{'Prime':<30} {'Accuracy':>10} {'P(target)':>12}")
    logger.info("-" * 55)

    for name, prime in addition_primes.items():
        results = evaluate_with_prime(model, tokenizer, prime, addition_problems)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_prob = np.mean([r["target_prob"] for r in results])

        all_results["addition"][name] = {
            "prime": prime,
            "accuracy": accuracy,
            "mean_target_prob": float(mean_prob),
            "details": results,
        }

        display = prime[:25] + "..." if len(prime) > 25 else prime
        logger.info(f"{display:<30} {accuracy:>10.0%} {mean_prob:>11.1%}")

    # Test subtraction
    logger.info("\n=== SUBTRACTION ===")
    logger.info(f"{'Prime':<30} {'Accuracy':>10} {'P(target)':>12}")
    logger.info("-" * 55)

    for name, prime in subtraction_primes.items():
        results = evaluate_with_prime(model, tokenizer, prime, subtraction_problems)
        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_prob = np.mean([r["target_prob"] for r in results])

        all_results["subtraction"][name] = {
            "prime": prime,
            "accuracy": accuracy,
            "mean_target_prob": float(mean_prob),
            "details": results,
        }

        display = prime[:25] + "..." if len(prime) > 25 else prime
        logger.info(f"{display:<30} {accuracy:>10.0%} {mean_prob:>11.1%}")

    # Show best for each
    logger.info(f"\n{'='*60}")
    logger.info("BEST PRIMES BY OPERATION")
    logger.info("=" * 60)

    for op in ["addition", "subtraction"]:
        best_name = max(all_results[op], key=lambda x: all_results[op][x]["accuracy"])
        best = all_results[op][best_name]
        logger.info(f"\n{op.upper()}:")
        logger.info(f"  Best prime: '{best['prime']}'")
        logger.info(f"  Accuracy: {best['accuracy']:.0%}")

        # Show individual results
        for r in best["details"]:
            status = "✓" if r["correct"] else "✗"
            logger.info(f"    {r['prompt']} → '{r['predicted']}' ({status})")

    # Test: Can a SINGLE prime work for both?
    logger.info(f"\n{'='*60}")
    logger.info("UNIVERSAL PRIME TEST")
    logger.info("=" * 60)

    universal_primes = {
        "arithmetic": "Arithmetic means calculating numbers.",
        "math_answer": "The answer is a number.",
        "number_result": "Result:",
        "counting_both": "Counting shows numbers in order.",
    }

    all_problems = addition_problems + subtraction_problems

    for name, prime in universal_primes.items():
        results = evaluate_with_prime(model, tokenizer, prime, all_problems)
        accuracy = sum(r["correct"] for r in results) / len(results)

        add_acc = sum(r["correct"] for r in results[:len(addition_problems)]) / len(addition_problems)
        sub_acc = sum(r["correct"] for r in results[len(addition_problems):]) / len(subtraction_problems)

        all_results[f"universal_{name}"] = {
            "prime": prime,
            "accuracy": accuracy,
            "add_accuracy": add_acc,
            "sub_accuracy": sub_acc,
        }

        logger.info(f"'{prime}'")
        logger.info(f"  Overall: {accuracy:.0%}, Add: {add_acc:.0%}, Sub: {sub_acc:.0%}")

    # Save results
    output_path = "data/experiments/operation_specific_primes.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
