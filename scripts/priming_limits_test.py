#!/usr/bin/env python3
"""Experiment 73: Priming Limits Test.

Priming achieves 100% on addition and subtraction.
Can it extend to:
1. Multiplication
2. Division
3. Word problems
4. Multi-step problems
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


def get_answer_token(tokenizer, answer_str):
    """Get the actual answer token ID."""
    tokens = tokenizer.encode(answer_str)
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

        target_id = get_answer_token(tokenizer, expected)
        target_prob = probs[target_id] if target_id >= 0 else 0.0

        # More flexible correctness check
        correct = (
            expected in predicted or
            predicted == expected or
            predicted.strip() == expected.strip()
        )

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
    logger.info("EXPERIMENT 73: PRIMING LIMITS TEST")
    logger.info("=" * 60)

    test_categories = {
        "multiplication": {
            "problems": [
                ("2×2=", "4"), ("3×3=", "9"), ("2×3=", "6"), ("3×4=", "12"),
                ("4×5=", "20"), ("2×5=", "10"), ("5×5=", "25"),
            ],
            "primes": {
                "none": "",
                "generic": "Arithmetic means calculating numbers.",
                "semantic_mult": "Multiplication means repeated addition.",
                "semantic_times": "Multiplying means groups of a number.",
            }
        },
        "division": {
            "problems": [
                ("4÷2=", "2"), ("6÷2=", "3"), ("6÷3=", "2"), ("9÷3=", "3"),
                ("10÷2=", "5"), ("10÷5=", "2"), ("8÷4=", "2"),
            ],
            "primes": {
                "none": "",
                "generic": "Arithmetic means calculating numbers.",
                "semantic_div": "Division means splitting into equal parts.",
                "semantic_share": "Dividing means sharing equally.",
            }
        },
        "word_problems": {
            "problems": [
                ("I have 3 apples. I get 2 more. Total:", "5"),
                ("5 birds. 2 fly away. Remaining:", "3"),
                ("2 groups of 3 equals", "6"),
                ("6 cookies shared by 2 equals", "3"),
            ],
            "primes": {
                "none": "",
                "generic": "Arithmetic means calculating numbers.",
                "word_prime": "Read the problem and calculate the answer.",
            }
        },
        "two_digit": {
            "problems": [
                ("12+3=", "15"), ("15+5=", "20"), ("20+10=", "30"),
                ("25-5=", "20"), ("30-10=", "20"), ("15-10=", "5"),
            ],
            "primes": {
                "none": "",
                "generic": "Arithmetic means calculating numbers.",
            }
        },
        "baseline_check": {
            "problems": [
                ("4+1=", "5"), ("2+2=", "4"), ("5-1=", "4"),
            ],
            "primes": {
                "none": "",
                "generic": "Arithmetic means calculating numbers.",
            }
        }
    }

    all_results = {}

    for category, config in test_categories.items():
        logger.info(f"\n=== {category.upper()} ===")

        all_results[category] = {}

        for prime_name, prime in config["primes"].items():
            results = evaluate_with_prime(model, tokenizer, prime, config["problems"])
            accuracy = sum(r["correct"] for r in results) / len(results)
            mean_prob = np.mean([r["target_prob"] for r in results])

            all_results[category][prime_name] = {
                "prime": prime,
                "accuracy": accuracy,
                "mean_target_prob": float(mean_prob),
                "details": results,
            }

            prime_display = prime[:30] + "..." if len(prime) > 30 else prime
            logger.info(f"  {prime_display:<35} {accuracy:>6.0%}")

        # Show details for best prime
        best_name = max(all_results[category], key=lambda x: all_results[category][x]["accuracy"])
        best = all_results[category][best_name]

        if best["accuracy"] < 1.0:
            logger.info(f"  Best: '{best['prime'][:30]}...' = {best['accuracy']:.0%}")
            for r in best["details"]:
                status = "✓" if r["correct"] else "✗"
                logger.info(f"    {r['prompt'][:30]} → '{r['predicted']}' ({status}) expected={r['expected']}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: CAN PRIMING EXTEND BEYOND ADDITION/SUBTRACTION?")
    logger.info("=" * 60)

    for category, primes in all_results.items():
        best_name = max(primes, key=lambda x: primes[x]["accuracy"])
        best_acc = primes[best_name]["accuracy"]
        no_prime_acc = primes.get("none", {}).get("accuracy", 0)

        improvement = best_acc - no_prime_acc

        status = "✓ WORKS" if best_acc >= 0.8 else ("▲ HELPS" if improvement > 0.2 else "✗ FAILS")
        logger.info(f"{category:<20} {no_prime_acc:>6.0%} → {best_acc:>6.0%} ({improvement:>+5.0%}) {status}")

    # Save results
    output_path = "data/experiments/priming_limits_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
