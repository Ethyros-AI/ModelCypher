#!/usr/bin/env python3
"""Experiment 75: Discover Model's Semantic Primitives.

If the model expresses knowledge through semantic primitives,
what ARE those primitives for this specific model?

Method: Test which single words/concepts most strongly activate
correct arithmetic responses. These are the model's "native language."
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


def test_prime(model, tokenizer, prime, problems):
    """Test a prime on problems, return accuracy and mean target prob."""
    import mlx.core as mx

    correct = 0
    total_prob = 0

    for problem, expected in problems:
        prompt = f"{prime} {problem}" if prime else problem

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        target_id = get_answer_token(tokenizer, expected)
        target_prob = probs[target_id] if target_id >= 0 else 0.0

        if expected in predicted or predicted == expected:
            correct += 1
        total_prob += target_prob

    return correct / len(problems), total_prob / len(problems)


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 75: DISCOVER MODEL'S SEMANTIC PRIMITIVES")
    logger.info("=" * 60)

    # Standard test set
    test_problems = [
        ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"), ("5+1=", "6"),
        ("2+2=", "4"), ("3+3=", "6"), ("2+3=", "5"),
    ]

    # Candidate primitives - comprehensive list
    candidates = {
        # Wierzbicka's NSM primes
        "nsm_quantifiers": ["one", "two", "some", "all", "many", "more", "less"],
        "nsm_relations": ["same", "other", "like", "part", "kind"],
        "nsm_space_time": ["where", "here", "above", "below", "before", "after", "now"],
        "nsm_logical": ["not", "if", "because", "can", "maybe"],
        "nsm_mental": ["think", "know", "want", "feel", "see", "hear"],
        "nsm_actions": ["do", "happen", "move", "say", "live"],
        "nsm_evaluative": ["good", "bad", "big", "small"],

        # Math-specific
        "math_ops": ["add", "plus", "sum", "total", "minus", "subtract", "times", "multiply", "divide"],
        "math_concepts": ["number", "digit", "count", "calculate", "equal", "result", "answer"],
        "math_relations": ["next", "previous", "greater", "smaller", "between"],

        # Structural
        "punctuation": [".", ":", "=", "?", "!"],
        "connectors": ["is", "means", "equals", "gives", "makes"],

        # Sequences
        "sequences": ["1.", "1,", "1 2", "1, 2,", "a)", "first"],
    }

    results = {"baseline": {}, "by_category": {}, "all_primes": {}}

    # Baseline
    baseline_acc, baseline_prob = test_prime(model, tokenizer, "", test_problems)
    results["baseline"] = {"accuracy": baseline_acc, "mean_prob": baseline_prob}
    logger.info(f"\nBaseline (no prime): {baseline_acc:.0%} accuracy, {baseline_prob:.1%} prob")

    # Test each category
    logger.info(f"\n{'Category':<20} {'Best Prime':<15} {'Acc':>6} {'Prob':>8} {'Lift':>8}")
    logger.info("-" * 65)

    all_primes = []

    for category, primes in candidates.items():
        category_results = {}

        for prime in primes:
            acc, prob = test_prime(model, tokenizer, prime, test_problems)
            category_results[prime] = {"accuracy": acc, "mean_prob": prob}
            all_primes.append((prime, acc, prob))

        # Find best in category
        best = max(category_results.items(), key=lambda x: (x[1]["accuracy"], x[1]["mean_prob"]))
        results["by_category"][category] = category_results

        lift = best[1]["accuracy"] - baseline_acc
        prime_display = best[0][:12].ljust(12)
        logger.info(f"{category:<20} '{prime_display}' {best[1]['accuracy']:>6.0%} "
                   f"{best[1]['mean_prob']:>7.1%} {lift:>+7.0%}")

    # Find overall best single-word primes
    all_primes.sort(key=lambda x: (x[1], x[2]), reverse=True)

    logger.info(f"\n{'='*60}")
    logger.info("TOP 15 SINGLE-WORD/TOKEN PRIMITIVES")
    logger.info("=" * 60)

    for prime, acc, prob in all_primes[:15]:
        lift = acc - baseline_acc
        results["all_primes"][prime] = {"accuracy": acc, "mean_prob": prob, "lift": lift}
        prime_display = prime[:18].ljust(18)
        logger.info(f"  '{prime_display}' {acc:>6.0%} (prob: {prob:>6.1%}, lift: {lift:>+5.0%})")

    # Test combinations of top primitives
    logger.info(f"\n{'='*60}")
    logger.info("PRIMITIVE COMBINATIONS")
    logger.info("=" * 60)

    top_primitives = [p for p, a, _ in all_primes[:10] if a > baseline_acc]

    if len(top_primitives) >= 2:
        combinations = []
        for i, p1 in enumerate(top_primitives[:5]):
            for p2 in top_primitives[i+1:6]:
                combined = f"{p1} {p2}"
                acc, prob = test_prime(model, tokenizer, combined, test_problems)
                combinations.append((combined, acc, prob))

        combinations.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for combo, acc, prob in combinations[:10]:
            lift = acc - baseline_acc
            combo_display = combo[:28].ljust(28)
            logger.info(f"  '{combo_display}' {acc:>6.0%} (lift: {lift:>+5.0%})")
            results["all_primes"][combo] = {"accuracy": acc, "mean_prob": prob, "lift": lift}

    # Identify the model's "native primitives"
    logger.info(f"\n{'='*60}")
    logger.info("MODEL'S NATIVE PRIMITIVES (>50% accuracy)")
    logger.info("=" * 60)

    native = [(p, a, pr) for p, a, pr in all_primes if a >= 0.5]
    if native:
        for prime, acc, prob in native:
            logger.info(f"  '{prime}' → {acc:.0%}")

        results["native_primitives"] = [p for p, _, _ in native]
        logger.info(f"\nThese {len(native)} primitives form the model's 'semantic basis' for arithmetic")
    else:
        logger.info("  No single primitives achieve >50%")
        logger.info("  Model requires compositional primes (sentences)")

    # Save results
    output_path = "data/experiments/discover_primitives.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
