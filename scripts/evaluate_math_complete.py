#!/usr/bin/env python3
"""Evaluate Qwen3-8B Math with Complete Answer Generation.

The issue: Two-digit numbers require multiple tokens (e.g., "10" = "1" + "0").
Our single-token evaluation was truncating answers.

This script generates up to 5 tokens to capture complete numerical answers.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 5) -> str:
    """Generate multiple tokens to get complete answer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    generated = []

    for _ in range(max_tokens):
        logits = model(mx.array([tokens + generated]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        next_tok = int(np.argmax(probs))
        generated.append(next_tok)

        # Stop on common terminators
        decoded = tokenizer.decode([next_tok])
        if decoded.strip() in ["", "\n", ".", ",", "!"]:
            break

    return tokenizer.decode(generated).strip()


def evaluate_with_generation(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate with multi-token generation."""
    results = []
    correct = 0

    for prompt, expected in problems:
        predicted = generate_answer(model, tokenizer, prompt, max_tokens=5)

        # Clean predicted - take first "word" (number or yes/no)
        predicted_clean = predicted.split()[0] if predicted.split() else predicted
        predicted_clean = predicted_clean.rstrip(".,!?")

        is_correct = expected.lower() in predicted_clean.lower() or predicted_clean.lower() == expected.lower()
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted_clean,
            "raw_generated": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    # Test both adapters
    adapters = {
        "base": None,
        "v1": "data/adapters/qwen3_math_lora",
        "v2": "data/adapters/qwen3_math_lora_v2",
    }

    # Comprehensive test suite
    test_categories = {
        "basic_arithmetic": [
            ("1+1=", "2"),
            ("2+2=", "4"),
            ("3+5=", "8"),
            ("7-3=", "4"),
            ("9-4=", "5"),
        ],
        "two_digit_results": [
            ("4+6=", "10"),
            ("5+5=", "10"),
            ("6+6=", "12"),
            ("7+8=", "15"),
            ("9+9=", "18"),
        ],
        "word_problems_single": [
            ("I have 3 apples. I get 2 more. Total: ", "5"),
            ("5 birds. 2 fly away. Remaining: ", "3"),
            ("Start with 3. Add 4. Result: ", "7"),
        ],
        "word_problems_two_digit": [
            ("I have 7 apples. I get 5 more. Total: ", "12"),
            ("Start with 4. Add 6. Result: ", "10"),
            ("I have 8 apples. I get 7 more. Total: ", "15"),
        ],
        "comparison": [
            ("Which is greater, 7 or 3? Answer: ", "7"),
            ("Which is larger, 5 or 9? Answer: ", "9"),
            ("Which is greater, 15 or 12? Answer: ", "15"),
        ],
        "number_sense": [
            ("What comes after 5? Answer: ", "6"),
            ("What comes before 10? Answer: ", "9"),
            ("What comes after 14? Answer: ", "15"),
        ],
        "language": [
            ("The cat sat on the", "mat"),
            ("Fire is hot and ice is", "cold"),
            ("The opposite of up is", "down"),
        ],
    }

    logger.info("=" * 70)
    logger.info("COMPLETE MATH EVALUATION (multi-token generation)")
    logger.info("=" * 70)

    all_results = {}

    for adapter_name, adapter_path in adapters.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {adapter_name}")
        logger.info(f"{'='*60}")

        model, tokenizer = load(model_path, adapter_path=adapter_path)

        adapter_results = {}
        total_correct = 0
        total_problems = 0

        for cat_name, problems in test_categories.items():
            acc, details = evaluate_with_generation(model, tokenizer, problems)
            adapter_results[cat_name] = {"accuracy": acc, "details": details}
            total_correct += sum(1 for d in details if d["correct"])
            total_problems += len(problems)

            status = "✓" if acc >= 0.6 else "✗"
            logger.info(f"  {status} {cat_name:25s}: {acc:.0%}")

            for r in details:
                mark = "+" if r["correct"] else "-"
                logger.info(f"      {mark} '{r['prompt'][:35]}' → '{r['predicted']}' (expected '{r['expected']}')")

        overall = total_correct / total_problems
        logger.info(f"\n  OVERALL: {overall:.0%} ({total_correct}/{total_problems})")

        all_results[adapter_name] = {
            "overall": overall,
            "categories": {k: v["accuracy"] for k, v in adapter_results.items()},
            "details": adapter_results,
        }

        del model
        mx.clear_cache()

    # Summary comparison
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Category':<30} {'Base':>10} {'v1':>10} {'v2':>10}")
    logger.info("-" * 62)

    for cat_name in test_categories.keys():
        base = all_results["base"]["categories"].get(cat_name, 0)
        v1 = all_results["v1"]["categories"].get(cat_name, 0)
        v2 = all_results["v2"]["categories"].get(cat_name, 0)
        logger.info(f"{cat_name:<30} {base:>9.0%} {v1:>9.0%} {v2:>9.0%}")

    logger.info("-" * 62)
    logger.info(f"{'OVERALL':<30} {all_results['base']['overall']:>9.0%} {all_results['v1']['overall']:>9.0%} {all_results['v2']['overall']:>9.0%}")

    # Best adapter
    best = max(all_results.keys(), key=lambda k: all_results[k]["overall"])
    logger.info(f"\n✓ Best adapter: {best} ({all_results[best]['overall']:.0%})")

    # Save results
    output_path = Path("data/experiments/qwen3_math_complete_eval.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
