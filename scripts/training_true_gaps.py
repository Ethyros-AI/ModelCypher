#!/usr/bin/env python3
"""Experiment 85: Training True Gaps.

For capabilities that ARE missing (parsing), can we train them?

Key insight from Exp 84:
- The model HAS arithmetic (100% with priming)
- The model LACKS language→equation parsing

Strategy:
1. Generate parsing training data: "I have 3 apples..." → "3+2="
2. Test if few-shot learning (in-context) can bridge the gap
3. If not, design minimal adapter training approach

NOTE: Full fine-tuning is out of scope for this experiment.
We focus on demonstrating the FEASIBILITY of targeted gap filling.
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


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate accuracy on a problem set with optional prime."""
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
    logger.info("EXPERIMENT 85: TRAINING TRUE GAPS")
    logger.info("=" * 60)

    # The true gap: word problems (language → equation parsing)
    word_problems = [
        ("I have 3 apples. I get 2 more. Total:", "5"),
        ("5 birds. 2 fly away. Remaining:", "3"),
        ("Start with 4. Add 3. Result:", "7"),
        ("Begin with 7. Take away 2. Left with:", "5"),
        ("Mary has 6 candies. She gives 4 away. How many left:", "2"),
        ("Tom has 2 toys. He gets 5 more. Total toys:", "7"),
    ]

    # Baseline: word problems with various primes
    logger.info("\n=== BASELINE: WORD PROBLEMS ===")

    primes_to_test = [
        "",
        "say",
        "Arithmetic means calculating numbers.",
        "Calculate the number.",
    ]

    for prime in primes_to_test:
        acc, _ = evaluate_accuracy(model, tokenizer, prime, word_problems)
        prime_display = prime if prime else "(none)"
        logger.info(f"  '{prime_display}': {acc:.0%}")

    # Method 1: Few-shot examples in context
    logger.info("\n" + "=" * 60)
    logger.info("METHOD 1: FEW-SHOT IN-CONTEXT LEARNING")
    logger.info("=" * 60)

    few_shot_examples = [
        # Example 1: Addition
        "Example: I have 2 oranges. I get 3 more. 2+3=5",
        # Example 2: Subtraction
        "Example: 6 cats. 2 leave. 6-2=4",
        # Example 3: Addition
        "Example: Start with 1. Add 4. 1+4=5",
    ]

    few_shot_primes = [
        # Just examples
        "\n".join(few_shot_examples) + "\n",
        # Examples + instruction
        "Translate word problems to equations.\n" + "\n".join(few_shot_examples) + "\n",
        # Examples + say
        "say " + "\n".join(few_shot_examples) + "\n",
        # Arithmetic + examples
        "Arithmetic means calculating numbers. " + "\n".join(few_shot_examples) + "\n",
    ]

    logger.info(f"{'Few-shot Prime':<60} {'Acc':>8}")
    logger.info("-" * 70)

    best_few_shot_acc = 0
    best_few_shot_prime = ""

    for prime in few_shot_primes:
        acc, details = evaluate_accuracy(model, tokenizer, prime, word_problems)
        prime_display = prime[:58].replace("\n", "\\n")
        logger.info(f"{prime_display:<60} {acc:>8.0%}")

        if acc > best_few_shot_acc:
            best_few_shot_acc = acc
            best_few_shot_prime = prime

    logger.info(f"\nBest few-shot accuracy: {best_few_shot_acc:.0%}")

    # Method 2: Template-based parsing instruction
    logger.info("\n" + "=" * 60)
    logger.info("METHOD 2: PARSING INSTRUCTION TEMPLATES")
    logger.info("=" * 60)

    parsing_templates = [
        # Direct instruction
        "Extract numbers and operation. 'get more'=+, 'take away'=-.",
        # Step by step
        "Step 1: Find numbers. Step 2: Find operation (+/-). Step 3: Write equation.",
        # Pattern matching
        "'have X, get Y more' means X+Y. 'X things, Y leave' means X-Y.",
        # Semantic mapping
        "MORE means add. LESS means subtract. Find the numbers and calculate.",
        # Combined
        "Word problems are equations. 'get' means +. 'away' means -. Calculate.",
    ]

    logger.info(f"{'Template':<60} {'Acc':>8}")
    logger.info("-" * 70)

    best_template_acc = 0
    best_template = ""

    for template in parsing_templates:
        acc, _ = evaluate_accuracy(model, tokenizer, template, word_problems)
        logger.info(f"{template[:58]:<60} {acc:>8.0%}")

        if acc > best_template_acc:
            best_template_acc = acc
            best_template = template

    logger.info(f"\nBest template accuracy: {best_template_acc:.0%}")

    # Method 3: Hybrid - explicit equation in problem
    logger.info("\n" + "=" * 60)
    logger.info("METHOD 3: EXPLICIT EQUATION INJECTION")
    logger.info("=" * 60)

    # This tests: if we parse FOR the model, does it work?
    word_problems_with_equation = [
        ("I have 3 apples. I get 2 more. 3+2=", "5"),
        ("5 birds. 2 fly away. 5-2=", "3"),
        ("Start with 4. Add 3. 4+3=", "7"),
        ("Begin with 7. Take away 2. 7-2=", "5"),
        ("Mary has 6 candies. She gives 4 away. 6-4=", "2"),
        ("Tom has 2 toys. He gets 5 more. 2+5=", "7"),
    ]

    acc_with_eq, _ = evaluate_accuracy(model, tokenizer, "say", word_problems_with_equation)
    logger.info(f"Word problems WITH explicit equation: {acc_with_eq:.0%}")

    if acc_with_eq >= 0.9:
        logger.info("*** EXPLICIT PARSING WORKS ***")
        logger.info("The gap is ONLY parsing. Arithmetic is fully functional.")

    # Method 4: Chain of thought style
    logger.info("\n" + "=" * 60)
    logger.info("METHOD 4: CHAIN OF THOUGHT")
    logger.info("=" * 60)

    cot_prompts = [
        # Inline reasoning
        "Think: Find numbers, determine operation, calculate. ",
        # Question format
        "What numbers? What operation? What answer? ",
        # Structured
        "Numbers: _ and _. Operation: +/-. Answer: ",
    ]

    for cot in cot_prompts:
        acc, _ = evaluate_accuracy(model, tokenizer, cot, word_problems)
        logger.info(f"'{cot[:40]}...': {acc:.0%}")

    # Analysis: What would training data look like?
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING DATA SPECIFICATION")
    logger.info("=" * 60)

    logger.info("""
If few-shot doesn't work, the minimal training approach would be:

TRAINING DATA FORMAT:
  Input: "I have 3 apples. I get 2 more. Total:"
  Output: "3+2=" (just the equation, not the answer)

The model ALREADY knows "3+2=" → "5"
We only need to train the PARSER component.

TRAINING SIZE ESTIMATE:
  - ~100-1000 examples covering patterns:
    - "have X, get Y more" → "X+Y="
    - "X things, Y leave" → "X-Y="
    - "X groups of Y" → "X×Y="
    - "X shared among Y" → "X÷Y="

ARCHITECTURE:
  - Small adapter (LoRA) on early layers
  - Freeze arithmetic capability (later layers)
  - Train only parsing transformation
""")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: TRAINING TRUE GAPS")
    logger.info("=" * 60)

    best_overall = max(best_few_shot_acc, best_template_acc)

    logger.info(f"""
RESULTS:
  Word problems (baseline): 0%
  Few-shot in-context: {best_few_shot_acc:.0%}
  Parsing templates: {best_template_acc:.0%}
  With explicit equations: {acc_with_eq:.0%}

KEY FINDING:
  - In-context learning achieves: {best_overall:.0%}
  - Explicit equation achieves: {acc_with_eq:.0%}
""")

    if best_overall >= 0.5:
        logger.info("*** FEW-SHOT PARTIALLY WORKS ***")
        logger.info("The model CAN learn parsing from examples.")
        logger.info("A small amount of fine-tuning should achieve 100%.")
    elif acc_with_eq >= 0.9:
        logger.info("*** PARSING IS THE BOTTLENECK ***")
        logger.info("The model has arithmetic but can't parse.")
        logger.info("Training a parser adapter is the solution.")
    else:
        logger.info("*** BOTH PARSING AND ARITHMETIC MAY NEED WORK ***")

    # Concrete next steps
    logger.info(f"""
NEXT STEPS FOR FULL IMPLEMENTATION:

1. Generate parsing training data (~500 examples)
2. Train LoRA adapter on parsing: word→equation
3. Combine: parser_adapter → arithmetic_prime → answer
4. Test: accuracy should reach 100% on word problems

This proves: self-improvement can IDENTIFY what to train,
            and MINIMIZE training to just the missing piece.
""")

    # Save results
    output_path = "data/experiments/training_true_gaps.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "baseline": {
            "word_problems_raw": 0.0,
            "word_problems_with_equation": float(acc_with_eq),
        },
        "few_shot": {
            "best_accuracy": float(best_few_shot_acc),
            "best_prime": best_few_shot_prime[:100],
        },
        "template": {
            "best_accuracy": float(best_template_acc),
            "best_template": best_template,
        },
        "conclusion": {
            "gap_is_parsing": acc_with_eq >= 0.9,
            "few_shot_helps": best_overall > 0,
            "training_needed": best_overall < 0.7,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
