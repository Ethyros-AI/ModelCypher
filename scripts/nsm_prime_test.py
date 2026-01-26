#!/usr/bin/env python3
"""Experiment 74: Natural Semantic Metalanguage (NSM) Prime Test.

Anna Wierzbicka's semantic primes are universal primitive concepts:
- MORE, LESS, AFTER, BEFORE, SAME, OTHER, ONE, TWO, ALL, PART
- KNOW, THINK, WANT, DO, HAPPEN, GOOD, BAD

Hypothesis: Our priming works because it activates these primitives.
Test: Do NSM-style minimal primes work as well as verbose explanations?
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

        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        target_id = get_answer_token(tokenizer, expected)
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
    logger.info("EXPERIMENT 74: NATURAL SEMANTIC METALANGUAGE PRIMES")
    logger.info("=" * 60)
    logger.info("\nTesting if Wierzbicka's semantic primes unlock arithmetic")

    # Test problems
    addition = [("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"), ("2+3=", "5")]
    subtraction = [("5-1=", "4"), ("4-1=", "3"), ("5-2=", "3"), ("7-3=", "4")]
    multiplication = [("2×3=", "6"), ("3×3=", "9"), ("2×4=", "8")]
    division = [("6÷2=", "3"), ("8÷2=", "4"), ("9÷3=", "3")]

    # NSM semantic primes (Wierzbicka's primitives)
    nsm_primes = {
        # Core primitives only
        "MORE": "more",
        "AFTER": "after",
        "BEFORE": "before",
        "SAME": "same",
        "PART": "part",

        # Primitive combinations
        "ONE_MORE": "one more",
        "ONE_LESS": "one less",
        "SAME_AMOUNT": "same amount",

        # Minimal sentences with primitives
        "after_this": "After this:",
        "one_more_is": "One more is",
        "the_same_as": "The same as",

        # NSM-style definitions (using only primitives)
        "add_nsm": "This is one more than before.",
        "sub_nsm": "This is one less than before.",
        "mult_nsm": "This is the same thing many times.",
        "div_nsm": "This is one part of many same parts.",
    }

    # Comparison primes (our previous findings)
    comparison_primes = {
        "verbose": "Arithmetic means calculating numbers.",
        "semantic": "Adding means combining.",
        "none": "",
    }

    all_results = {}

    # Test NSM primes on addition
    logger.info("\n=== ADDITION with NSM PRIMES ===")
    logger.info(f"{'Prime':<35} {'Acc':>6} {'P(tgt)':>8}")
    logger.info("-" * 55)

    for name, prime in {**nsm_primes, **comparison_primes}.items():
        results = evaluate_with_prime(model, tokenizer, prime, addition)
        acc = sum(r["correct"] for r in results) / len(results)
        prob = np.mean([r["target_prob"] for r in results])

        all_results[f"add_{name}"] = {
            "prime": prime,
            "operation": "addition",
            "accuracy": acc,
            "mean_prob": float(prob),
        }

        logger.info(f"{prime:<35} {acc:>6.0%} {prob:>7.1%}")

    # Test NSM primes on all operations with best candidates
    logger.info("\n=== ALL OPERATIONS with KEY PRIMES ===")

    key_primes = {
        "none": "",
        "after_this": "After this:",
        "one_more": "one more",
        "verbose": "Arithmetic means calculating numbers.",
        "add_nsm": "This is one more than before.",
    }

    all_problems = addition + subtraction + multiplication + division

    logger.info(f"\n{'Prime':<40} {'Add':>6} {'Sub':>6} {'Mul':>6} {'Div':>6} {'All':>6}")
    logger.info("-" * 75)

    for name, prime in key_primes.items():
        add_res = evaluate_with_prime(model, tokenizer, prime, addition)
        sub_res = evaluate_with_prime(model, tokenizer, prime, subtraction)
        mul_res = evaluate_with_prime(model, tokenizer, prime, multiplication)
        div_res = evaluate_with_prime(model, tokenizer, prime, division)

        add_acc = sum(r["correct"] for r in add_res) / len(add_res)
        sub_acc = sum(r["correct"] for r in sub_res) / len(sub_res)
        mul_acc = sum(r["correct"] for r in mul_res) / len(mul_res)
        div_acc = sum(r["correct"] for r in div_res) / len(div_res)
        all_acc = (sum(r["correct"] for r in add_res + sub_res + mul_res + div_res) /
                   len(add_res + sub_res + mul_res + div_res))

        all_results[f"all_{name}"] = {
            "prime": prime,
            "add": add_acc, "sub": sub_acc, "mul": mul_acc, "div": div_acc, "all": all_acc
        }

        display = prime[:38] if prime else "(none)"
        logger.info(f"{display:<40} {add_acc:>6.0%} {sub_acc:>6.0%} {mul_acc:>6.0%} {div_acc:>6.0%} {all_acc:>6.0%}")

    # Analysis: Do single NSM words work?
    logger.info("\n=== SINGLE NSM PRIMITIVE WORDS ===")

    single_words = ["more", "less", "after", "before", "same", "one", "two", "number", "count"]

    for word in single_words:
        results = evaluate_with_prime(model, tokenizer, word, addition)
        acc = sum(r["correct"] for r in results) / len(results)
        all_results[f"word_{word}"] = {"prime": word, "accuracy": acc}
        logger.info(f"  '{word}' → {acc:.0%}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS: DO NSM PRIMES WORK?")
    logger.info("=" * 60)

    # Find best NSM prime
    nsm_results = {k: v for k, v in all_results.items() if "nsm" in k.lower() or k.startswith("add_")}
    if nsm_results:
        best_nsm = max(nsm_results.items(), key=lambda x: x[1].get("accuracy", 0))
        logger.info(f"\nBest NSM-style prime: '{best_nsm[1].get('prime', '')}' → {best_nsm[1].get('accuracy', 0):.0%}")

    verbose_acc = all_results.get("add_verbose", {}).get("accuracy", 0)
    logger.info(f"Verbose prime: {verbose_acc:.0%}")

    # Check if single primitives work
    single_word_accs = [v["accuracy"] for k, v in all_results.items() if k.startswith("word_")]
    if single_word_accs:
        best_single = max(single_word_accs)
        logger.info(f"Best single primitive word: {best_single:.0%}")

        if best_single >= 0.5:
            logger.info("\n*** SINGLE NSM PRIMITIVES DO HELP ***")
        else:
            logger.info("\n*** SINGLE WORDS NOT ENOUGH - NEED SENTENCE STRUCTURE ***")

    # Save results
    output_path = "data/experiments/nsm_prime_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
