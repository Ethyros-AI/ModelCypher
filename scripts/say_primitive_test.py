#!/usr/bin/env python3
"""Experiment 76: The 'SAY' Primitive Deep Dive.

"say" alone achieves 100% on arithmetic.
This is Wierzbicka's core COMMUNICATION primitive.

Questions:
1. Does it work on ALL arithmetic (not just addition)?
2. What about word problems?
3. What other communication primitives work? (tell, speak, answer)
4. Is it the act of "saying" or the word itself?
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


def evaluate(model, tokenizer, prime, problems):
    """Evaluate problems with a given prime."""
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

        target_id = get_answer_token(tokenizer, expected)
        target_prob = float(probs[target_id]) if target_id >= 0 else 0.0

        correct = expected in predicted or predicted == expected
        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "prob": target_prob,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 76: THE 'SAY' PRIMITIVE")
    logger.info("=" * 60)

    # Comprehensive test sets
    test_sets = {
        "addition": [
            ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
            ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9"),
            ("2+2=", "4"), ("3+3=", "6"), ("2+3=", "5"), ("4+5=", "9"),
        ],
        "subtraction": [
            ("2-1=", "1"), ("3-1=", "2"), ("4-1=", "3"), ("5-1=", "4"),
            ("6-1=", "5"), ("7-1=", "6"), ("8-1=", "7"), ("9-1=", "8"),
            ("5-2=", "3"), ("7-3=", "4"), ("10-5=", "5"),
        ],
        "multiplication": [
            ("2×2=", "4"), ("2×3=", "6"), ("3×3=", "9"), ("2×4=", "8"),
            ("3×4=", "12"), ("2×5=", "10"), ("5×5=", "25"),
        ],
        "division": [
            ("4÷2=", "2"), ("6÷2=", "3"), ("6÷3=", "2"), ("8÷2=", "4"),
            ("9÷3=", "3"), ("10÷2=", "5"), ("10÷5=", "2"),
        ],
        "two_digit": [
            ("10+5=", "15"), ("12+3=", "15"), ("20+10=", "30"),
            ("15-5=", "10"), ("20-10=", "10"), ("25-5=", "20"),
        ],
    }

    # Test "say" on all operations
    logger.info("\n=== 'say' ON ALL ARITHMETIC ===")
    logger.info(f"{'Operation':<15} {'No prime':>10} {'say':>10} {'Lift':>10}")
    logger.info("-" * 50)

    all_results = {}

    for op_name, problems in test_sets.items():
        no_prime_acc, _ = evaluate(model, tokenizer, "", problems)
        say_acc, say_results = evaluate(model, tokenizer, "say", problems)

        all_results[op_name] = {
            "no_prime": no_prime_acc,
            "say": say_acc,
            "details": say_results,
        }

        lift = say_acc - no_prime_acc
        logger.info(f"{op_name:<15} {no_prime_acc:>10.0%} {say_acc:>10.0%} {lift:>+10.0%}")

    # Test communication primitive variants
    logger.info("\n=== COMMUNICATION PRIMITIVES ===")

    comm_primes = [
        "say",
        "tell",
        "speak",
        "answer",
        "reply",
        "respond",
        "state",
        "express",
        "declare",
        "announce",
        # Imperative forms
        "Say:",
        "Tell me:",
        "Answer:",
        # Full sentences
        "I say",
        "You say",
        "Say it:",
    ]

    all_problems = []
    for problems in test_sets.values():
        all_problems.extend(problems)

    logger.info(f"{'Prime':<20} {'Accuracy':>10}")
    logger.info("-" * 35)

    comm_results = {}
    for prime in comm_primes:
        acc, _ = evaluate(model, tokenizer, prime, all_problems)
        comm_results[prime] = acc
        logger.info(f"{prime:<20} {acc:>10.0%}")

    all_results["communication_primes"] = comm_results

    # Test: Is it "say" specifically or the communication concept?
    logger.info("\n=== SEMANTIC ANALYSIS ===")

    # Related concepts
    semantic_groups = {
        "communication": ["say", "tell", "speak", "talk"],
        "output": ["answer", "result", "output", "return"],
        "assertion": ["is", "equals", "means", "gives"],
        "action": ["do", "make", "compute", "calculate"],
    }

    logger.info(f"{'Group':<15} {'Best word':<15} {'Accuracy':>10}")
    logger.info("-" * 45)

    for group_name, words in semantic_groups.items():
        best_word = ""
        best_acc = 0
        for word in words:
            acc, _ = evaluate(model, tokenizer, word, test_sets["addition"])
            if acc > best_acc:
                best_acc = acc
                best_word = word
        logger.info(f"{group_name:<15} {best_word:<15} {best_acc:>10.0%}")

    # Word problems with "say"
    logger.info("\n=== WORD PROBLEMS with 'say' ===")

    word_problems = [
        ("I have 3 apples. I get 2 more. Total:", "5"),
        ("5 birds. 2 fly away. Remaining:", "3"),
        ("2 groups of 3 equals", "6"),
    ]

    for prime in ["", "say", "Say the answer:", "Calculate and say:"]:
        acc, results = evaluate(model, tokenizer, prime, word_problems)
        prime_display = prime if prime else "(none)"
        logger.info(f"  '{prime_display}' → {acc:.0%}")
        for r in results:
            if not r["correct"]:
                logger.info(f"    {r['problem'][:30]}... → '{r['predicted']}' (expected {r['expected']})")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: THE POWER OF 'SAY'")
    logger.info("=" * 60)

    total_no_prime = sum(all_results[op]["no_prime"] for op in test_sets) / len(test_sets)
    total_say = sum(all_results[op]["say"] for op in test_sets) / len(test_sets)

    logger.info(f"\nAverage across all operations:")
    logger.info(f"  No prime: {total_no_prime:.0%}")
    logger.info(f"  'say':    {total_say:.0%}")
    logger.info(f"  Lift:     {total_say - total_no_prime:+.0%}")

    if total_say >= 0.9:
        logger.info("\n*** 'SAY' IS A UNIVERSAL ARITHMETIC UNLOCK ***")
        logger.info("The model expresses arithmetic through the COMMUNICATION primitive")

    # Save
    output_path = "data/experiments/say_primitive_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
