#!/usr/bin/env python3
"""Experiment 77: Find Subtraction's Primitive.

"say" unlocks addition, multiplication, division but NOT subtraction.
What primitive unlocks subtraction?

Hypothesis: Subtraction may need a different semantic primitive:
- LESS (quantity)
- BEFORE (temporal)
- REMOVE/TAKE (action)
- LOSE/LEAVE (result)
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
    tokens = tokenizer.encode(answer_str)
    if len(tokens) > 1 and tokens[0] == 1:
        return tokens[1]
    return tokens[0] if tokens else -1


def evaluate(model, tokenizer, prime, problems):
    import mlx.core as mx

    correct = 0
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

        if expected in predicted or predicted == expected:
            correct += 1

    return correct / len(problems)


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 77: FIND SUBTRACTION'S PRIMITIVE")
    logger.info("=" * 60)

    subtraction = [
        ("2-1=", "1"), ("3-1=", "2"), ("4-1=", "3"), ("5-1=", "4"),
        ("6-1=", "5"), ("7-1=", "6"), ("8-1=", "7"), ("9-1=", "8"),
        ("5-2=", "3"), ("7-3=", "4"), ("10-5=", "5"),
    ]

    # Also test addition for comparison
    addition = [
        ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"), ("5+1=", "6"),
        ("2+2=", "4"), ("3+3=", "6"),
    ]

    # Candidate primitives for subtraction
    candidates = {
        # NSM quantity primitives
        "less": "less",
        "fewer": "fewer",
        "minus": "minus",
        "without": "without",

        # NSM temporal (before)
        "before": "before",
        "previous": "previous",
        "prior": "prior",

        # Action primitives
        "take": "take",
        "remove": "remove",
        "subtract": "subtract",
        "lose": "lose",
        "leave": "leave",

        # Result primitives
        "remain": "remain",
        "left": "left",
        "remaining": "remaining",

        # Direction
        "down": "down",
        "back": "back",
        "away": "away",

        # Negation
        "not": "not",
        "no": "no",

        # Comparison
        "difference": "difference",
        "between": "between",

        # What worked for addition
        "say": "say",
        "equal": "equal",

        # Combinations
        "say less": "say less",
        "one less": "one less",
        "take away": "take away",
        "goes down": "goes down",
        "count back": "count back",
        "before this": "before this",

        # Sentences
        "The number before": "The number before",
        "One less is": "One less is",
        "Subtract means": "Subtract means",
        "Taking away": "Taking away",
        "Going back": "Going back",
    }

    logger.info("\n=== TESTING SUBTRACTION PRIMITIVES ===")
    logger.info(f"{'Prime':<25} {'Sub':>8} {'Add':>8} {'Diff':>8}")
    logger.info("-" * 55)

    results = []
    for name, prime in candidates.items():
        sub_acc = evaluate(model, tokenizer, prime, subtraction)
        add_acc = evaluate(model, tokenizer, prime, addition)
        diff = sub_acc - add_acc

        results.append((name, prime, sub_acc, add_acc, diff))

    # Sort by subtraction accuracy
    results.sort(key=lambda x: x[2], reverse=True)

    for name, prime, sub_acc, add_acc, diff in results:
        prime_display = prime[:23].ljust(23)
        logger.info(f"{prime_display} {sub_acc:>8.0%} {add_acc:>8.0%} {diff:>+8.0%}")

    # Find best for subtraction
    best = results[0]
    logger.info(f"\n=== BEST SUBTRACTION PRIMITIVE ===")
    logger.info(f"Prime: '{best[1]}'")
    logger.info(f"Subtraction: {best[2]:.0%}")
    logger.info(f"Addition: {best[3]:.0%}")

    # Check if any primitive is subtraction-specific
    sub_specific = [(n, p, s, a) for n, p, s, a, d in results if s > 0.5 and d > 0.2]
    if sub_specific:
        logger.info("\n=== SUBTRACTION-SPECIFIC PRIMITIVES ===")
        for name, prime, sub_acc, add_acc in sub_specific:
            logger.info(f"  '{prime}': Sub={sub_acc:.0%}, Add={add_acc:.0%}")

    # Test comprehensive semantic primes
    logger.info("\n=== SEMANTIC SENTENCE PRIMES ===")

    semantic_primes = {
        "sub_semantic": "Subtracting means taking away.",
        "sub_previous": "Subtracting 1 means the previous number.",
        "sub_less": "Subtracting means one less.",
        "sub_remove": "Subtracting removes from a number.",
        "sub_backward": "Subtracting counts backward.",
        "sub_difference": "Subtraction finds the difference.",
        "arithmetic": "Arithmetic means calculating numbers.",
    }

    logger.info(f"{'Prime':<45} {'Sub':>8}")
    logger.info("-" * 55)

    for name, prime in semantic_primes.items():
        sub_acc = evaluate(model, tokenizer, prime, subtraction)
        prime_display = prime[:43].ljust(43)
        logger.info(f"{prime_display} {sub_acc:>8.0%}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    baseline = evaluate(model, tokenizer, "", subtraction)
    best_single = max(results, key=lambda x: x[2])
    best_semantic = max(semantic_primes.items(),
                        key=lambda x: evaluate(model, tokenizer, x[1], subtraction))
    best_semantic_acc = evaluate(model, tokenizer, best_semantic[1], subtraction)

    logger.info(f"Baseline (no prime): {baseline:.0%}")
    logger.info(f"Best single word: '{best_single[1]}' → {best_single[2]:.0%}")
    logger.info(f"Best semantic: '{best_semantic[1]}' → {best_semantic_acc:.0%}")

    if best_single[2] >= 0.8:
        logger.info(f"\n*** FOUND SUBTRACTION PRIMITIVE: '{best_single[1]}' ***")
    elif best_semantic_acc >= 0.8:
        logger.info(f"\n*** SUBTRACTION NEEDS SEMANTIC PRIME ***")
    else:
        logger.info(f"\n*** SUBTRACTION MAY BE A TRUE GAP ***")


if __name__ == "__main__":
    main()
