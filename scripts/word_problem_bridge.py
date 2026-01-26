#!/usr/bin/env python3
"""Experiment 79: Word Problem Bridge.

Word problems fail even with priming because they need:
1. Language → Operation mapping
2. Number extraction
3. Arithmetic execution

The model has #3. Can analogies/metaphors bridge #1 and #2?

Hypothesis: Word problems need a LANGUAGE→MATH bridge, not just
a math prime. We need metaphors that connect natural language
concepts to arithmetic operations.
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

    results = []
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

        correct = expected in predicted or predicted == expected

        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 79: WORD PROBLEM BRIDGES")
    logger.info("=" * 60)

    # Word problems - the gap
    word_problems = [
        ("I have 3 apples. I get 2 more. Total:", "5"),
        ("5 birds. 2 fly away. Remaining:", "3"),
        ("2 groups of 3 equals", "6"),
        ("6 cookies shared by 2 people. Each gets:", "3"),
        ("Start with 4. Add 3. Result:", "7"),
        ("Begin with 7. Take away 2. Left with:", "5"),
    ]

    # Symbolic versions (should work with primes)
    symbolic = [
        ("3+2=", "5"),
        ("5-2=", "3"),
        ("2×3=", "6"),
        ("6÷2=", "3"),
        ("4+3=", "7"),
        ("7-2=", "5"),
    ]

    # Language → Math bridges
    bridges = {
        # Direct mapping
        "direct_map": "'get more' means +. 'fly away' means -. 'groups of' means ×. 'shared by' means ÷.",

        # Action → Operation
        "action_bridge": "Getting means adding. Losing means subtracting. Groups means multiplying. Sharing means dividing.",

        # Result focus
        "result_focus": "Find the number. Getting more increases. Going away decreases.",

        # Story → Math
        "story_to_math": "Stories have numbers. More means +. Less means -. Groups means ×. Split means ÷.",

        # Word patterns
        "word_patterns": "'more' = add. 'away' = subtract. 'groups' = multiply. 'each' = divide.",

        # Semantic primitives
        "semantic_bridge": "MORE means add. LESS means subtract. SAME MANY means multiply. PART means divide.",

        # Universal + specific
        "universal_word": "Word problems are math. Find the numbers and the operation.",

        # Step by step
        "step_bridge": "Step 1: Find the numbers. Step 2: Find if it's +, -, ×, or ÷. Step 3: Calculate.",

        # Equation bridge
        "equation_bridge": "Every story is an equation. Find the numbers, find the operation, say the answer.",

        # Previous working primes
        "arithmetic": "Arithmetic means calculating numbers.",
    }

    # First verify symbolic works
    logger.info("\n=== VERIFY SYMBOLIC WORKS ===")
    sym_acc, _ = evaluate(model, tokenizer, "Arithmetic means calculating numbers.", symbolic)
    logger.info(f"Symbolic with 'Arithmetic means...' → {sym_acc:.0%}")

    # Test word problems with bridges
    logger.info("\n=== WORD PROBLEM BRIDGES ===")
    logger.info(f"{'Bridge':<55} {'Acc':>6}")
    logger.info("-" * 65)

    results = {}

    for name, prime in bridges.items():
        acc, details = evaluate(model, tokenizer, prime, word_problems)
        results[name] = {"prime": prime, "accuracy": acc, "details": details}

        prime_display = prime[:53]
        logger.info(f"{prime_display:<55} {acc:>6.0%}")

    # Find best
    best = max(results.items(), key=lambda x: x[1]["accuracy"])

    logger.info(f"\n=== BEST BRIDGE ===")
    logger.info(f"Name: {best[0]}")
    logger.info(f"Prime: \"{best[1]['prime']}\"")
    logger.info(f"Accuracy: {best[1]['accuracy']:.0%}")

    if best[1]["accuracy"] > 0:
        logger.info("\nSuccessful problems:")
        for r in best[1]["details"]:
            if r["correct"]:
                logger.info(f"  ✓ {r['problem'][:40]}... → '{r['predicted']}'")

    logger.info("\nFailed problems:")
    for r in best[1]["details"]:
        if not r["correct"]:
            logger.info(f"  ✗ {r['problem'][:40]}... → '{r['predicted']}' (expected {r['expected']})")

    # Try hybrid: Convert word to symbolic, then solve
    logger.info(f"\n{'='*60}")
    logger.info("ALTERNATIVE: EXPLICIT EQUATION FORM")
    logger.info("=" * 60)

    # Reformulate word problems as equations
    equation_form = [
        ("I have 3 apples. I get 2 more. 3+2=", "5"),
        ("5 birds. 2 fly away. 5-2=", "3"),
        ("2 groups of 3. 2×3=", "6"),
        ("6 cookies ÷ 2 people. 6÷2=", "3"),
    ]

    logger.info("\nWord problem + explicit equation:")
    acc, details = evaluate(model, tokenizer, "Arithmetic means calculating numbers.", equation_form)
    logger.info(f"Accuracy: {acc:.0%}")
    for r in details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} {r['problem'][:45]}... → '{r['predicted']}'")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: CAN WE BRIDGE WORD PROBLEMS?")
    logger.info("=" * 60)

    word_best = best[1]["accuracy"]
    sym_best = sym_acc
    eq_form = acc

    logger.info(f"\nSymbolic (3+2=): {sym_best:.0%}")
    logger.info(f"Word problems (best bridge): {word_best:.0%}")
    logger.info(f"Word + equation (explicit form): {eq_form:.0%}")

    if word_best >= 0.5:
        logger.info("\n*** LANGUAGE→MATH BRIDGE WORKS ***")
    elif eq_form >= 0.8:
        logger.info("\n*** EXPLICIT EQUATION FORM WORKS ***")
        logger.info("The gap is PARSING, not MATH. Model needs equation, not just words.")
    else:
        logger.info("\n*** WORD PROBLEMS ARE A TRUE CAPABILITY GAP ***")
        logger.info("Language→Math parsing is genuinely missing, not just disconnected.")


if __name__ == "__main__":
    main()
