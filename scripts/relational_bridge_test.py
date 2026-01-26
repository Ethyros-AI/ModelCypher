#!/usr/bin/env python3
"""Experiment 78: Relational Bridge Test.

The model has isolated capability islands.
Can analogies/metaphors create the bridges?

Test: Provide relational primes that CONNECT concepts rather than
just activating them.
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
    logger.info("EXPERIMENT 78: RELATIONAL BRIDGES")
    logger.info("=" * 60)

    # Test sets
    addition = [("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"), ("2+3=", "5")]
    subtraction = [("5-1=", "4"), ("4-1=", "3"), ("5-2=", "3"), ("7-3=", "4")]
    mixed = addition + subtraction

    # Relational primes - these CREATE CONNECTIONS
    relational_primes = {
        # Analogy: Symbol → Primitive
        "symbol_to_say": "+ means say the next. - means say one less.",
        "symbol_to_count": "+ counts up. - counts down.",
        "symbol_to_action": "+ adds more. - takes away.",

        # Analogy: Operation → Operation (opposites)
        "opposites": "Adding and subtracting are opposites.",
        "inverse": "+ is forward, - is backward.",
        "mirror": "What + builds, - removes.",

        # Metaphor: Math → Physical
        "physical": "+ is putting together. - is taking apart.",
        "movement": "+ moves forward. - moves backward.",
        "container": "+ fills up. - empties out.",

        # Metaphor: Math → Counting
        "counting_bridge": "Numbers go 1,2,3,4,5. + goes forward. - goes backward.",
        "sequence_bridge": "In a sequence, + means next, - means previous.",

        # Direct relational
        "relational_direct": "+ and - both give numbers. + means more, - means less.",

        # Compositional (using discovered primitives)
        "primitive_combo": "For +, say the answer. For -, say one less.",

        # Universal + specific
        "universal_then_specific": "Math gives numbers. + adds, - subtracts.",
    }

    # Non-relational primes for comparison
    nonrelational_primes = {
        "just_add": "Adding means combining.",
        "just_sub": "Subtracting means taking away.",
        "universal": "Arithmetic means calculating numbers.",
        "none": "",
    }

    logger.info("\n=== RELATIONAL PRIMES (create connections) ===")
    logger.info(f"{'Prime':<50} {'Add':>6} {'Sub':>6} {'Mix':>6}")
    logger.info("-" * 70)

    results = {}

    for name, prime in relational_primes.items():
        add_acc = evaluate(model, tokenizer, prime, addition)
        sub_acc = evaluate(model, tokenizer, prime, subtraction)
        mix_acc = evaluate(model, tokenizer, prime, mixed)

        results[name] = {
            "prime": prime,
            "type": "relational",
            "add": add_acc,
            "sub": sub_acc,
            "mixed": mix_acc,
        }

        prime_display = prime[:48]
        logger.info(f"{prime_display:<50} {add_acc:>6.0%} {sub_acc:>6.0%} {mix_acc:>6.0%}")

    logger.info("\n=== NON-RELATIONAL PRIMES (activate single islands) ===")
    logger.info(f"{'Prime':<50} {'Add':>6} {'Sub':>6} {'Mix':>6}")
    logger.info("-" * 70)

    for name, prime in nonrelational_primes.items():
        add_acc = evaluate(model, tokenizer, prime, addition)
        sub_acc = evaluate(model, tokenizer, prime, subtraction)
        mix_acc = evaluate(model, tokenizer, prime, mixed)

        results[name] = {
            "prime": prime,
            "type": "nonrelational",
            "add": add_acc,
            "sub": sub_acc,
            "mixed": mix_acc,
        }

        prime_display = prime[:48] if prime else "(none)"
        logger.info(f"{prime_display:<50} {add_acc:>6.0%} {sub_acc:>6.0%} {mix_acc:>6.0%}")

    # Find best
    best_relational = max(
        [(k, v) for k, v in results.items() if v["type"] == "relational"],
        key=lambda x: x[1]["mixed"]
    )
    best_nonrelational = max(
        [(k, v) for k, v in results.items() if v["type"] == "nonrelational"],
        key=lambda x: x[1]["mixed"]
    )

    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS: DO RELATIONAL BRIDGES WORK BETTER?")
    logger.info("=" * 70)

    logger.info(f"\nBest relational: '{best_relational[0]}'")
    logger.info(f"  Mixed accuracy: {best_relational[1]['mixed']:.0%}")
    logger.info(f"  Prime: \"{best_relational[1]['prime']}\"")

    logger.info(f"\nBest non-relational: '{best_nonrelational[0]}'")
    logger.info(f"  Mixed accuracy: {best_nonrelational[1]['mixed']:.0%}")

    if best_relational[1]["mixed"] > best_nonrelational[1]["mixed"]:
        improvement = best_relational[1]["mixed"] - best_nonrelational[1]["mixed"]
        logger.info(f"\n*** RELATIONAL BRIDGES WORK BETTER (+{improvement:.0%}) ***")
        logger.info("Analogies/metaphors create connections that single-island primes can't")
    elif best_relational[1]["mixed"] == best_nonrelational[1]["mixed"]:
        logger.info(f"\n*** EQUAL PERFORMANCE ***")
        logger.info("Both approaches achieve same accuracy on mixed operations")
    else:
        logger.info(f"\n*** NON-RELATIONAL WORKS BETTER ***")
        logger.info("Simple activation may be sufficient")

    # Check: Which primes achieve 100% on BOTH operations?
    universal_success = [
        (k, v) for k, v in results.items()
        if v["add"] >= 0.9 and v["sub"] >= 0.9
    ]

    if universal_success:
        logger.info(f"\n=== PRIMES THAT WORK FOR BOTH + AND - ===")
        for name, data in universal_success:
            logger.info(f"  '{name}': Add={data['add']:.0%}, Sub={data['sub']:.0%}")
            logger.info(f"    \"{data['prime']}\"")


if __name__ == "__main__":
    main()
