#!/usr/bin/env python3
"""Experiment 87: Verification Oracle.

Can the model verify its own generated parses?

The key insight: The model KNOWS arithmetic (100% with priming).
We can use this verified capability to check if generated parses are correct.

Example:
  - Word problem: "I have 3 apples, get 2 more"
  - Candidate parse: "3+2="
  - Oracle computes: model("say 3+2=") → "5"
  - Verify: "5" == expected_answer → CORRECT

This is the safety mechanism that prevents learning nonsense.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
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


@dataclass
class VerificationResult:
    """Result of oracle verification."""
    word_problem: str
    parsed_equation: str
    expected_answer: str
    computed_answer: str
    oracle_verdict: bool  # True = correct, False = wrong
    is_actually_correct: bool  # Ground truth


class VerificationOracle:
    """Use existing capabilities to verify new learning."""

    def __init__(self, model, tokenizer, prime: str = "say"):
        self.model = model
        self.tokenizer = tokenizer
        self.prime = prime

    def compute(self, equation: str) -> str:
        """Compute equation using primed model."""
        import mlx.core as mx

        prompt = f"{self.prime} {equation}"
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        return self.tokenizer.decode([top_token]).strip()

    def verify_parse(self, parsed_equation: str, expected_answer: str) -> Tuple[bool, str]:
        """
        Verify that a parse is correct using arithmetic capability.

        Returns:
            (is_correct, computed_answer)
        """
        computed = self.compute(parsed_equation)
        is_correct = expected_answer in computed or computed == expected_answer
        return is_correct, computed


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 87: VERIFICATION ORACLE")
    logger.info("=" * 60)

    # Use universal prime that works for all operations (discovered in Exp 72)
    oracle = VerificationOracle(model, tokenizer, prime="Arithmetic means calculating numbers.")

    # First, verify the oracle itself works
    logger.info("\n=== ORACLE CALIBRATION ===")
    calibration_tests = [
        ("1+1=", "2"),
        ("2+2=", "4"),
        ("3+1=", "4"),
        ("5-2=", "3"),
        ("4-1=", "3"),
        ("7-3=", "4"),
    ]

    calibration_correct = 0
    for eq, expected in calibration_tests:
        is_correct, computed = oracle.verify_parse(eq, expected)
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} {eq} → '{computed}' (expected '{expected}')")
        if is_correct:
            calibration_correct += 1

    calibration_acc = calibration_correct / len(calibration_tests)
    logger.info(f"\nOracle calibration: {calibration_acc:.0%}")

    if calibration_acc < 0.9:
        logger.info("*** WARNING: Oracle not reliable enough ***")
        logger.info("Cannot proceed with verification if oracle itself fails")
        return

    logger.info("✓ Oracle is calibrated and reliable")

    # Test verification on correct parses
    logger.info("\n=== TEST 1: CORRECT PARSES ===")

    correct_parses = [
        # (word_problem, parsed_equation, expected_answer, is_actually_correct)
        ("I have 3 apples. I get 2 more.", "3+2=", "5", True),
        ("5 birds. 2 fly away.", "5-2=", "3", True),
        ("Start with 4. Add 3.", "4+3=", "7", True),
        ("Begin with 7. Take away 2.", "7-2=", "5", True),
        ("6 toys plus 2 toys.", "6+2=", "8", True),
        ("9 cats. 4 leave.", "9-4=", "5", True),
    ]

    results = []
    for word_problem, equation, expected, is_correct_gt in correct_parses:
        oracle_correct, computed = oracle.verify_parse(equation, expected)
        result = VerificationResult(
            word_problem=word_problem,
            parsed_equation=equation,
            expected_answer=expected,
            computed_answer=computed,
            oracle_verdict=oracle_correct,
            is_actually_correct=is_correct_gt,
        )
        results.append(result)

        status = "✓" if oracle_correct else "✗"
        logger.info(f"  {status} '{word_problem[:30]}...' → {equation} → '{computed}' (expected '{expected}')")

    correct_accepted = sum(1 for r in results if r.oracle_verdict and r.is_actually_correct)
    logger.info(f"\nCorrect parses accepted: {correct_accepted}/{len(correct_parses)}")

    # Test verification on WRONG parses
    logger.info("\n=== TEST 2: WRONG PARSES (should be rejected) ===")

    wrong_parses = [
        # Intentionally wrong parses
        ("I have 3 apples. I get 2 more.", "3-2=", "1", False),  # Wrong: subtracted instead of added
        ("5 birds. 2 fly away.", "5+2=", "7", False),  # Wrong: added instead of subtracted
        ("Start with 4. Add 3.", "4-3=", "1", False),  # Wrong operation
        ("Begin with 7. Take away 2.", "7+2=", "9", False),  # Wrong operation
        ("6 toys plus 2 toys.", "6-2=", "4", False),  # Wrong operation
        ("9 cats. 4 leave.", "9+4=", "13", False),  # Wrong operation
    ]

    wrong_results = []
    for word_problem, equation, expected, is_correct_gt in wrong_parses:
        # Oracle verifies: does equation produce expected?
        # For WRONG parses: the equation is wrong for the problem, but mathematically correct
        oracle_correct, computed = oracle.verify_parse(equation, expected)

        result = VerificationResult(
            word_problem=word_problem,
            parsed_equation=equation,
            expected_answer=expected,
            computed_answer=computed,
            oracle_verdict=oracle_correct,
            is_actually_correct=is_correct_gt,
        )
        wrong_results.append(result)

        # Key insight: Oracle will ACCEPT wrong parses if equation→answer is correct
        # This is expected! Oracle verifies arithmetic, not parsing.
        status = "✓ (oracle accepts math)" if oracle_correct else "✗ (oracle rejects)"
        logger.info(f"  {status} WRONG PARSE: '{word_problem[:22]}...' → {equation} → '{computed}'")

    # Analysis: The oracle will ACCEPT wrong parses if the equation computes correctly
    # The key is: oracle checks equation→answer, not word_problem→equation
    logger.info("\n=== ANALYSIS ===")

    logger.info("""
KEY INSIGHT:
  The oracle verifies: equation → answer
  It does NOT verify: word_problem → equation

  This means:
  - Oracle CAN verify if arithmetic is correct
  - Oracle CANNOT directly verify if parsing is correct

  BUT: We can use this for VERIFIED SELF-PLAY:
  1. Generate (word_problem, equation) pair programmatically
  2. We KNOW the correct equation because WE generated it
  3. Oracle verifies the equation→answer part
  4. The pairing is correct by construction
""")

    # Test 3: Verified self-play data generation
    logger.info("\n=== TEST 3: VERIFIED SELF-PLAY DATA GENERATION ===")

    # Templates for word problems
    addition_templates = [
        ("I have {a} apples. I get {b} more. Total:", "{a}+{b}="),
        ("{a} birds. {b} more arrive. Total:", "{a}+{b}="),
        ("Start with {a}. Add {b}. Result:", "{a}+{b}="),
    ]

    subtraction_templates = [
        ("{a} apples. {b} eaten. Remaining:", "{a}-{b}="),
        ("{a} birds. {b} fly away. Remaining:", "{a}-{b}="),
        ("Start with {a}. Take away {b}. Left:", "{a}-{b}="),
    ]

    verified_samples = []
    rejected_samples = []

    np.random.seed(42)

    for _ in range(20):
        a = np.random.randint(2, 10)
        b = np.random.randint(1, a)  # Ensure b < a for subtraction

        # Choose operation
        if np.random.rand() > 0.5:
            template, eq_template = addition_templates[np.random.randint(0, len(addition_templates))]
            expected = str(a + b)
        else:
            template, eq_template = subtraction_templates[np.random.randint(0, len(subtraction_templates))]
            expected = str(a - b)

        word_problem = template.format(a=a, b=b)
        equation = eq_template.format(a=a, b=b)

        # Oracle verification
        is_correct, computed = oracle.verify_parse(equation, expected)

        if is_correct:
            verified_samples.append({
                "input": word_problem,
                "output": equation,
                "verified_answer": expected,
                "computed": computed,
            })
        else:
            rejected_samples.append({
                "input": word_problem,
                "output": equation,
                "expected": expected,
                "computed": computed,
            })

    logger.info(f"Generated {len(verified_samples) + len(rejected_samples)} samples")
    logger.info(f"  Verified (accepted): {len(verified_samples)}")
    logger.info(f"  Rejected: {len(rejected_samples)}")

    if verified_samples:
        logger.info("\nSample verified training pairs:")
        for sample in verified_samples[:5]:
            logger.info(f"  Input:  '{sample['input']}'")
            logger.info(f"  Output: '{sample['output']}' → '{sample['computed']}' ✓")

    if rejected_samples:
        logger.info("\nRejected samples (oracle caught errors):")
        for sample in rejected_samples[:3]:
            logger.info(f"  Input:  '{sample['input']}'")
            logger.info(f"  Output: '{sample['output']}' → '{sample['computed']}' ✗ (expected '{sample['expected']}')")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: VERIFICATION ORACLE")
    logger.info("=" * 60)

    verification_rate = len(verified_samples) / (len(verified_samples) + len(rejected_samples)) if (len(verified_samples) + len(rejected_samples)) > 0 else 0

    logger.info(f"""
RESULTS:
  Oracle calibration: {calibration_acc:.0%}
  Correct parses accepted: {correct_accepted}/{len(correct_parses)} ({correct_accepted/len(correct_parses):.0%})
  Self-play verification rate: {verification_rate:.0%}

HOW THE ORACLE ENABLES SAFE LEARNING:
  1. We generate (word_problem, equation) pairs programmatically
  2. The pairing is CORRECT BY CONSTRUCTION
  3. Oracle verifies equation→answer (catches arithmetic errors)
  4. Training data is GROUND-TRUTH VERIFIED

  The oracle doesn't verify parsing directly, but:
  - We generate correct pairs programmatically
  - Oracle ensures arithmetic is sound
  - Together: VERIFIED training data
""")

    if calibration_acc >= 0.9 and verification_rate >= 0.9:
        logger.info("*** ORACLE IS READY FOR SAFE SELF-PLAY ***")
    else:
        logger.info("*** ORACLE NEEDS IMPROVEMENT ***")

    # Save results
    output_path = "data/experiments/verification_oracle.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "calibration": {
            "accuracy": float(calibration_acc),
            "tests": len(calibration_tests),
        },
        "correct_parse_acceptance": {
            "accepted": correct_accepted,
            "total": len(correct_parses),
        },
        "self_play": {
            "verified": len(verified_samples),
            "rejected": len(rejected_samples),
            "verification_rate": float(verification_rate),
            "samples": verified_samples[:10],
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
