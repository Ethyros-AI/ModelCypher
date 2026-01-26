#!/usr/bin/env python3
"""Experiment 93: Fix the Transform v2 - Context Association Training.

The problem with v1: Model learned to output EOS instead of numbers.
The mlx_lm trainer adds EOS tokens, so the model saw:
  "3+2=" + "5" + <EOS>
And learned the wrong pattern.

New approach: Train on CONTINUED sequences, not just prompt→completion.
The training data should look like natural text continuation:
  "3+2=5 because three plus two makes five"
  "Calculate: 7-3=4"
  "Math: 1+1=2"

This teaches the model that equations are followed by numbers in context,
not by EOS tokens.
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


def evaluate_raw(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate WITHOUT any priming - pure T(input)."""
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

        is_correct = expected in predicted or predicted == expected
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


def generate_context_training_data(n_samples: int = 500) -> List[dict]:
    """Generate training data as natural text continuations.

    Instead of prompt/completion pairs, use full text that shows
    equations naturally followed by answers.
    """
    np.random.seed(42)

    samples = []

    # Context templates - separate for addition and subtraction
    addition_templates = [
        "{a}+{b}={answer}. That's basic addition.",
        "{a}+{b}={answer}, which is the sum.",
        "Calculate {a}+{b}={answer}",
        "Simple math: {a}+{b}={answer}",
        "The answer to {a}+{b}={answer}",
    ]

    subtraction_templates = [
        "{a}-{b}={answer}. Subtraction result.",
        "{a}-{b}={answer}, the difference.",
        "Calculate {a}-{b}={answer}",
        "Simple math: {a}-{b}={answer}",
        "The answer to {a}-{b}={answer}",
    ]

    for _ in range(n_samples):
        a = np.random.randint(1, 15)
        b = np.random.randint(1, 10)

        if np.random.rand() > 0.5:
            # Addition
            answer = a + b
            template = addition_templates[np.random.randint(len(addition_templates))]
        else:
            # Subtraction
            if b > a:
                a, b = b, a
            answer = a - b
            template = subtraction_templates[np.random.randint(len(subtraction_templates))]

        text = template.format(a=a, b=b, answer=answer)
        samples.append({"text": text})

    # Also add some RAW equation → answer continuations
    # These are the key: just equation followed by answer, nothing else
    for _ in range(n_samples // 2):
        a = np.random.randint(1, 15)
        b = np.random.randint(1, 10)

        if np.random.rand() > 0.5:
            text = f"{a}+{b}={a+b}"
        else:
            if b > a:
                a, b = b, a
            text = f"{a}-{b}={a-b}"

        samples.append({"text": text})

    return samples


def main():
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    train_data_dir = "data/training/fix_transform_v2"
    adapter_path = "data/adapters/fix_transform_lora_v2"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 93: FIX THE TRANSFORM v2")
    logger.info("=" * 60)
    logger.info("Goal: Train on text continuations, not prompt/completion")
    logger.info("      Avoid EOS token confusion")

    # Test cases
    arithmetic_tests = [
        ("1+1=", "2"), ("2+2=", "4"), ("3+1=", "4"), ("5+2=", "7"),
        ("4+3=", "7"), ("6+1=", "7"), ("3+3=", "6"), ("2+5=", "7"),
        ("5-2=", "3"), ("4-1=", "3"), ("7-3=", "4"), ("6-2=", "4"),
        ("9-4=", "5"), ("8-3=", "5"), ("10-5=", "5"), ("7-2=", "5"),
    ]

    # Phase 1: Baseline
    logger.info("\n=== PHASE 1: BASELINE ===")

    model, tokenizer = load(model_path)

    raw_baseline, raw_details = evaluate_raw(model, tokenizer, arithmetic_tests)
    logger.info(f"T(equation) accuracy (raw): {raw_baseline:.0%}")

    logger.info("\nBaseline examples:")
    for r in raw_details[:6]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['input']}' → '{r['predicted']}' (expected '{r['expected']}')")

    del model
    import mlx.core as mx
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE CONTEXT TRAINING DATA ===")

    samples = generate_context_training_data(500)

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

    # Phase 3: Train
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    n_iters = 400  # More iterations

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", adapter_path,
        "--batch-size", "8",
        "--num-layers", "16",
        "--iters", str(n_iters),
        "--learning-rate", "5e-5",  # Lower LR
        "--seed", "42",
        "--steps-per-report", "50",
    ]

    logger.info(f"Training...")
    logger.info(f"  Iterations: {n_iters}")
    logger.info(f"  Learning rate: 5e-5")

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

    # Phase 4: Test
    logger.info("\n=== PHASE 4: TEST FIXED T ===")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    raw_fixed, raw_details_fixed = evaluate_raw(model, tokenizer, arithmetic_tests)

    logger.info(f"\n{'='*50}")
    logger.info(f"T(equation) accuracy:")
    logger.info(f"  Before: {raw_baseline:.0%}")
    logger.info(f"  After:  {raw_fixed:.0%}")
    logger.info(f"{'='*50}")

    logger.info("\nFixed T examples:")
    for r in raw_details_fixed:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['input']}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Diagnostic: What are top predictions now?
    logger.info("\n=== DIAGNOSTIC: Top predictions ===")
    import mlx.core as mx

    for eq, expected in arithmetic_tests[:4]:
        tokens = tokenizer.encode(eq)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_indices = np.argsort(probs)[-5:][::-1]
        logger.info(f"\n  '{eq}' (expected '{expected}'):")
        for idx in top_indices:
            token_text = tokenizer.decode([idx])
            logger.info(f"    {probs[idx]:.3f}: '{token_text}'")

    # Save results
    results = {
        "baseline": raw_baseline,
        "fixed": raw_fixed,
        "improvement": raw_fixed - raw_baseline,
        "details": raw_details_fixed,
    }

    output_path = "data/experiments/fix_transform_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
