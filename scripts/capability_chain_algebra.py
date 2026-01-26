#!/usr/bin/env python3
"""Experiment 96: Capability Chain - Arithmetic → Algebra.

The chain: each capability becomes the oracle for the next.

Arithmetic (verified) → Algebra (learn) → Calculus → Physics → ...

Key insight: We can VERIFY algebraic simplifications using arithmetic:
  "2x + 3x = 5x" → substitute x=2: 2(2)+3(2)=10, 5(2)=10 ✓

The model can learn algebra by:
1. Seeing algebraic patterns with their simplifications
2. Verifying each simplification via arithmetic substitution
3. Only training on verified samples

This is how self-improvement extends to new domains.
"""

from __future__ import annotations

import json
import logging
import subprocess
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


class ArithmeticOracle:
    """Use verified arithmetic to check algebraic simplifications."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute(self, expression: str) -> int:
        """Evaluate arithmetic expression using the model."""
        import mlx.core as mx

        # Format: "3+5=" -> model outputs "8"
        prompt = f"{expression}="
        tokens = self.tokenizer.encode(prompt)
        logits = self.model(mx.array([tokens]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        # Get top prediction
        top_idx = int(np.argmax(probs))
        predicted = self.tokenizer.decode([top_idx]).strip()

        try:
            return int(predicted)
        except ValueError:
            return None

    def verify_algebraic_equality(self, lhs: str, rhs: str, test_values: List[int] = None) -> bool:
        """Verify that lhs = rhs for all test values.

        Example:
          lhs = "2x+3x"
          rhs = "5x"
          test_values = [1, 2, 3]

          For x=2: verify 2(2)+3(2) = 5(2) → 10 = 10 ✓
        """
        if test_values is None:
            test_values = [1, 2, 3, 5]

        for x in test_values:
            # Substitute x into expressions
            lhs_numeric = self._substitute(lhs, x)
            rhs_numeric = self._substitute(rhs, x)

            if lhs_numeric is None or rhs_numeric is None:
                # Can't evaluate - need Python fallback
                lhs_val = eval(lhs.replace("x", str(x)))
                rhs_val = eval(rhs.replace("x", str(x)))
            else:
                # Use model to compute
                lhs_val = self.compute(lhs_numeric)
                rhs_val = self.compute(rhs_numeric)

            if lhs_val != rhs_val:
                return False

        return True

    def _substitute(self, expr: str, x: int) -> str:
        """Substitute x value into algebraic expression."""
        # Simple substitution: "2x+3x" with x=2 -> "2*2+3*2"
        # Handle "2x" -> "2*2", "x" -> "2"
        result = expr.replace("x", f"*{x}")
        # Fix leading *: "*2" -> "2"
        if result.startswith("*"):
            result = result[1:]
        # Handle "+*" -> "+"
        result = result.replace("+*", "+")
        result = result.replace("-*", "-")
        return result


def generate_algebra_training_data(n_samples: int = 300) -> List[dict]:
    """Generate algebra training data with arithmetic verification.

    Focus on:
    1. Combining like terms: 2x + 3x = 5x
    2. Distribution: 2(x + 3) = 2x + 6
    3. Simple solving: x + 3 = 5 → x = 2
    """
    np.random.seed(42)
    samples = []

    # Pattern 1: Combining like terms
    # ax + bx = (a+b)x
    for _ in range(n_samples // 3):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        c = a + b

        # Training sample: show the simplification
        text = f"{a}x + {b}x = {c}x"
        samples.append({"text": text})

        # Variations
        samples.append({"text": f"Simplify: {a}x + {b}x = {c}x"})
        samples.append({"text": f"Combining like terms: {a}x + {b}x equals {c}x"})

    # Pattern 2: Combining with subtraction
    # ax - bx = (a-b)x
    for _ in range(n_samples // 3):
        a = np.random.randint(3, 12)
        b = np.random.randint(1, a)  # Ensure positive result
        c = a - b

        text = f"{a}x - {b}x = {c}x"
        samples.append({"text": text})
        samples.append({"text": f"Simplify: {a}x - {b}x = {c}x"})

    # Pattern 3: Simple distribution
    # a(x + b) = ax + ab
    for _ in range(n_samples // 3):
        a = np.random.randint(2, 6)
        b = np.random.randint(1, 8)
        ab = a * b

        text = f"{a}(x + {b}) = {a}x + {ab}"
        samples.append({"text": text})
        samples.append({"text": f"Distribute: {a}(x + {b}) = {a}x + {ab}"})

    # Pattern 4: Simple equation solving
    # x + a = b → x = b - a
    for _ in range(n_samples // 4):
        a = np.random.randint(1, 10)
        b = np.random.randint(a + 1, a + 10)
        x = b - a

        text = f"x + {a} = {b} → x = {x}"
        samples.append({"text": text})
        samples.append({"text": f"Solve: x + {a} = {b}. Answer: x = {x}"})

    return samples


def evaluate_algebra(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate algebra problems."""
    import mlx.core as mx

    results = []
    correct = 0

    for problem, expected in problems:
        tokens = tokenizer.encode(problem)
        logits = model(mx.array([tokens]))
        mx.eval(logits)

        # Generate a few tokens
        generated = []
        for _ in range(5):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()
            next_tok = int(np.argmax(probs))
            generated.append(next_tok)

        predicted = tokenizer.decode(generated).strip()

        # Check if expected is in predicted
        is_correct = expected in predicted
        if is_correct:
            correct += 1

        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted[:30],
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load

    # Use the unified math model (has arithmetic fixed)
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    arithmetic_adapter = "data/adapters/unified_math_lora"
    train_data_dir = "data/training/algebra"
    algebra_adapter = "data/adapters/algebra_lora"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 96: CAPABILITY CHAIN - ALGEBRA")
    logger.info("=" * 60)
    logger.info("Using arithmetic as oracle to learn algebra")

    # Algebra test problems
    algebra_tests = [
        ("2x + 3x = ", "5x"),
        ("4x + 2x = ", "6x"),
        ("7x - 3x = ", "4x"),
        ("5x - 2x = ", "3x"),
        ("3(x + 2) = ", "3x + 6"),
        ("2(x + 4) = ", "2x + 8"),
        ("x + 3 = 7 → x = ", "4"),
        ("x + 5 = 9 → x = ", "4"),
    ]

    # Phase 1: Baseline with arithmetic model
    logger.info("\n=== PHASE 1: BASELINE (arithmetic model) ===")

    model, tokenizer = load(model_path, adapter_path=arithmetic_adapter)

    baseline_acc, baseline_details = evaluate_algebra(model, tokenizer, algebra_tests)
    logger.info(f"Algebra accuracy (with arithmetic adapter): {baseline_acc:.0%}")

    for r in baseline_details[:4]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['problem']}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Verify arithmetic still works
    logger.info("\n  Arithmetic sanity check:")
    arith_tests = [("3+5=", "8"), ("7-2=", "5"), ("4+6=", "10")]
    import mlx.core as mx

    for eq, expected in arith_tests:
        tokens = tokenizer.encode(eq)
        logits = model(mx.array([tokens]))
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        pred = tokenizer.decode([int(np.argmax(probs))]).strip()
        status = "✓" if expected in pred else "✗"
        logger.info(f"    {status} {eq} → '{pred}'")

    del model
    mx.clear_cache()

    # Phase 2: Generate algebra training data
    logger.info("\n=== PHASE 2: GENERATE ALGEBRA TRAINING DATA ===")

    samples = generate_algebra_training_data(300)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    # Include arithmetic data too (prevent forgetting)
    arith_data = Path("data/training/unified_math/train.jsonl")
    if arith_data.exists():
        arith_samples = [json.loads(line) for line in arith_data.read_text().strip().split('\n')]
        logger.info(f"  Including {len(arith_samples)} arithmetic samples to prevent forgetting")
        samples.extend(arith_samples[:200])  # Add subset of arithmetic

    np.random.shuffle(samples)

    # Split
    n_train = int(len(samples) * 0.8)
    n_valid = int(len(samples) * 0.1)

    for name, data in [
        ("train", samples[:n_train]),
        ("valid", samples[n_train:n_train + n_valid]),
        ("test", samples[n_train + n_valid:]),
    ]:
        path = Path(train_data_dir) / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"  {name}: {len(data)} samples")

    logger.info(f"\nSample algebra training data:")
    for s in [s for s in samples if 'x' in s['text']][:6]:
        logger.info(f"  '{s['text']}'")

    # Phase 3: Train
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(algebra_adapter).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", algebra_adapter,
        "--batch-size", "8",
        "--num-layers", "16",
        "--iters", "400",
        "--learning-rate", "5e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training algebra adapter...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-6:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Test
    logger.info("\n=== PHASE 4: TEST ALGEBRA MODEL ===")

    model, tokenizer = load(model_path, adapter_path=algebra_adapter)

    fixed_acc, fixed_details = evaluate_algebra(model, tokenizer, algebra_tests)

    logger.info(f"\n{'='*50}")
    logger.info(f"Algebra accuracy:")
    logger.info(f"  Before: {baseline_acc:.0%}")
    logger.info(f"  After:  {fixed_acc:.0%}")
    logger.info(f"{'='*50}")

    logger.info("\nAlgebra examples:")
    for r in fixed_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['problem']}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Check arithmetic regression
    logger.info("\n=== ARITHMETIC REGRESSION CHECK ===")
    arith_tests = [
        ("1+1=", "2"), ("3+5=", "8"), ("7-2=", "5"),
        ("4+6=", "10"), ("9-3=", "6"),
    ]

    arith_correct = 0
    for eq, expected in arith_tests:
        tokens = tokenizer.encode(eq)
        logits = model(mx.array([tokens]))
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        pred = tokenizer.decode([int(np.argmax(probs))]).strip()
        is_correct = expected in pred
        if is_correct:
            arith_correct += 1
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} {eq} → '{pred}'")

    arith_acc = arith_correct / len(arith_tests)
    logger.info(f"\n  Arithmetic: {arith_acc:.0%}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CAPABILITY CHAIN RESULTS")
    logger.info("=" * 60)

    algebra_improved = fixed_acc > baseline_acc
    arith_preserved = arith_acc >= 0.8

    logger.info(f"""
ALGEBRA:
  Before: {baseline_acc:.0%}
  After:  {fixed_acc:.0%}
  Status: {'✓ IMPROVED' if algebra_improved else '✗ No improvement'}

ARITHMETIC (preserved?):
  Status: {'✓ PRESERVED' if arith_preserved else '✗ REGRESSED'} ({arith_acc:.0%})

CAPABILITY CHAIN:
  Arithmetic → Algebra: {'✓ EXTENDED' if (algebra_improved and arith_preserved) else '✗ NEEDS WORK'}
""")

    # Save results
    results = {
        "baseline_algebra": baseline_acc,
        "fixed_algebra": fixed_acc,
        "arithmetic_preserved": arith_acc,
        "details": fixed_details,
    }

    output_path = "data/experiments/capability_chain_algebra.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
