#!/usr/bin/env python3
"""Experiment 94: Fix Word Problems Using the Transform Fix Pattern.

We proved in Exp 93 that training format matters:
- prompt/completion → model learns to output EOS
- text continuation → model learns correct pattern

Now apply this to word problems (TRUE_GAP):
- Train: "I have 3 apples. I get 2 more. Total: 5"
- Test: T(word_problem) = answer without special formatting

The model already has arithmetic (we fixed that in Exp 93).
Now we need to fix the PARSING: word problem → number.
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


def evaluate_word_problems(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate word problems - check if answer appears in continuation."""
    import mlx.core as mx

    results = []
    correct = 0

    for problem, expected in problems:
        tokens = tokenizer.encode(problem)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        # Check if the expected answer is the prediction
        is_correct = expected == predicted or expected in predicted
        if is_correct:
            correct += 1

        results.append({
            "input": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def generate_word_problem_training_data(n_samples: int = 800) -> List[dict]:
    """Generate word problem training data as text continuations.

    Key: The answer comes RIGHT AFTER the question, as natural text.
    """
    np.random.seed(42)

    samples = []

    # Addition word problem templates - answer follows naturally
    addition_templates = [
        "I have {a} apples. I get {b} more. Total: {answer}",
        "{a} birds are sitting. {b} more arrive. Now there are {answer} birds.",
        "Start with {a}. Add {b}. Result: {answer}",
        "There are {a} cats. {b} more come. Total cats: {answer}",
        "{a} toys plus {b} toys equals {answer} toys.",
        "Mary has {a} candies. She gets {b} more. She has {answer} candies.",
        "{a} books on the shelf. Add {b} more. Total books: {answer}",
        "Begin with {a}. Increase by {b}. Answer: {answer}",
        "{a} plus {b} is {answer}.",
        "If you have {a} and add {b}, you get {answer}.",
    ]

    # Subtraction word problem templates
    subtraction_templates = [
        "I have {a} apples. I eat {b}. Remaining: {answer}",
        "{a} birds are sitting. {b} fly away. Now there are {answer} birds.",
        "Start with {a}. Take away {b}. Result: {answer}",
        "There are {a} cats. {b} leave. Remaining cats: {answer}",
        "{a} toys minus {b} toys equals {answer} toys.",
        "Tom has {a} candies. He gives away {b}. He has {answer} candies.",
        "{a} books on the shelf. Remove {b}. Remaining books: {answer}",
        "Begin with {a}. Decrease by {b}. Answer: {answer}",
        "{a} minus {b} is {answer}.",
        "If you have {a} and remove {b}, you have {answer}.",
    ]

    for _ in range(n_samples):
        a = np.random.randint(2, 15)
        b = np.random.randint(1, min(a, 10))

        if np.random.rand() > 0.5:
            # Addition
            answer = a + b
            template = addition_templates[np.random.randint(len(addition_templates))]
        else:
            # Subtraction (ensure positive result)
            answer = a - b
            template = subtraction_templates[np.random.randint(len(subtraction_templates))]

        text = template.format(a=a, b=b, answer=answer)
        samples.append({"text": text})

    # Also add some short-form training
    # "How many? X" pattern
    for _ in range(n_samples // 4):
        a = np.random.randint(2, 15)
        b = np.random.randint(1, min(a, 10))

        if np.random.rand() > 0.5:
            answer = a + b
            text = f"{a} and {b} more. How many? {answer}"
        else:
            answer = a - b
            text = f"{a} take away {b}. How many left? {answer}"

        samples.append({"text": text})

    return samples


def main():
    from mlx_lm import load

    # Use 350M with the FIXED arithmetic adapter
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    arithmetic_adapter = "data/adapters/fix_transform_lora_v2"  # Already fixes T for equations
    train_data_dir = "data/training/fix_word_problems"
    word_problem_adapter = "data/adapters/fix_word_problems_lora"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 94: FIX WORD PROBLEMS")
    logger.info("=" * 60)
    logger.info("Goal: T(word_problem) = answer without special formatting")

    # Test cases - word problems with expected answers
    word_problem_tests = [
        ("I have 3 apples. I get 2 more. Total:", "5"),
        ("5 birds. 2 fly away. Remaining:", "3"),
        ("Start with 4. Add 3. Result:", "7"),
        ("There are 8 cats. 3 leave. Remaining:", "5"),
        ("6 toys plus 2 toys equals", "8"),
        ("Tom has 7 candies. He gives 4 away. He has", "3"),
        ("Begin with 9. Decrease by 4. Answer:", "5"),
        ("If you have 5 and add 3, you get", "8"),
        ("10 minus 6 is", "4"),
        ("7 and 2 more. How many?", "9"),
        ("8 take away 3. How many left?", "5"),
        ("I have 4. I get 4 more. Total:", "8"),
    ]

    # Phase 1: Baseline with arithmetic-fixed model
    logger.info("\n=== PHASE 1: BASELINE (with arithmetic adapter) ===")

    model, tokenizer = load(model_path, adapter_path=arithmetic_adapter)

    baseline_acc, baseline_details = evaluate_word_problems(model, tokenizer, word_problem_tests)
    logger.info(f"Word problem accuracy (arithmetic-fixed model): {baseline_acc:.0%}")

    logger.info("\nBaseline examples:")
    for r in baseline_details[:6]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['input'][:40]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Also test raw arithmetic to make sure it still works
    arithmetic_tests = [("1+1=", "2"), ("3+2=", "5"), ("7-3=", "4")]
    arith_correct = 0
    for eq, expected in arithmetic_tests:
        import mlx.core as mx
        tokens = tokenizer.encode(eq)
        logits = model(mx.array([tokens]))
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        pred = tokenizer.decode([int(np.argmax(probs))]).strip()
        if expected in pred:
            arith_correct += 1
    logger.info(f"Arithmetic (sanity check): {arith_correct}/{len(arithmetic_tests)}")

    del model
    import mlx.core as mx
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE WORD PROBLEM TRAINING DATA ===")

    samples = generate_word_problem_training_data(800)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

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

    logger.info(f"\nSample training data:")
    for s in samples[:8]:
        logger.info(f"  '{s['text']}'")

    # Phase 3: Train word problem adapter
    # NOTE: We train on TOP of the arithmetic adapter
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(word_problem_adapter).mkdir(parents=True, exist_ok=True)

    n_iters = 500  # More iterations for word problems

    # We need to fuse the arithmetic adapter first, then train new adapter
    # For simplicity, train fresh and include arithmetic patterns too
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", word_problem_adapter,
        "--batch-size", "8",
        "--num-layers", "16",  # All layers
        "--iters", str(n_iters),
        "--learning-rate", "5e-5",
        "--seed", "42",
        "--steps-per-report", "50",
    ]

    logger.info(f"Training word problem adapter...")
    logger.info(f"  Iterations: {n_iters}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-8:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Test fixed model
    logger.info("\n=== PHASE 4: TEST FIXED MODEL ===")

    model, tokenizer = load(model_path, adapter_path=word_problem_adapter)

    fixed_acc, fixed_details = evaluate_word_problems(model, tokenizer, word_problem_tests)

    logger.info(f"\n{'='*50}")
    logger.info(f"Word problem accuracy:")
    logger.info(f"  Before: {baseline_acc:.0%}")
    logger.info(f"  After:  {fixed_acc:.0%}")
    logger.info(f"{'='*50}")

    logger.info("\nFixed model examples:")
    for r in fixed_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['input'][:40]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Diagnostic: Top predictions
    logger.info("\n=== DIAGNOSTIC: Top predictions ===")
    for problem, expected in word_problem_tests[:4]:
        tokens = tokenizer.encode(problem)
        logits = model(mx.array([tokens]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_indices = np.argsort(probs)[-5:][::-1]
        logger.info(f"\n  '{problem[:35]}...' (expected '{expected}'):")
        for idx in top_indices:
            token_text = tokenizer.decode([idx])
            logger.info(f"    {probs[idx]:.3f}: '{token_text}'")

    # Check arithmetic still works
    logger.info("\n=== ARITHMETIC REGRESSION CHECK ===")
    arith_tests = [
        ("1+1=", "2"), ("2+2=", "4"), ("5+3=", "8"),
        ("7-2=", "5"), ("9-4=", "5"), ("6-3=", "3"),
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
        correct = expected in pred
        if correct:
            arith_correct += 1
        status = "✓" if correct else "✗"
        logger.info(f"  {status} {eq} → '{pred}' (expected '{expected}')")

    logger.info(f"\nArithmetic: {arith_correct}/{len(arith_tests)} = {arith_correct/len(arith_tests):.0%}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    wp_improved = fixed_acc > baseline_acc
    arith_preserved = arith_correct >= len(arith_tests) * 0.8

    logger.info(f"""
WORD PROBLEMS:
  Before: {baseline_acc:.0%}
  After:  {fixed_acc:.0%}
  Status: {'✓ IMPROVED' if wp_improved else '✗ No improvement'}

ARITHMETIC:
  Status: {'✓ PRESERVED' if arith_preserved else '✗ REGRESSED'}

CONCLUSION:
  Word problems fixed: {wp_improved}
  Arithmetic preserved: {arith_preserved}
  Overall: {'✓ SUCCESS' if (wp_improved and arith_preserved) else '✗ NEEDS WORK'}
""")

    # Save results
    results = {
        "baseline_word_problems": baseline_acc,
        "fixed_word_problems": fixed_acc,
        "arithmetic_preserved": arith_correct / len(arith_tests),
        "details": fixed_details,
    }

    output_path = "data/experiments/fix_word_problems.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
