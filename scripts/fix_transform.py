#!/usr/bin/env python3
"""Experiment 92: Fix the Transform, Not the Prompt.

The insight: A prompt is an input vector. The model is transform T.
If T only works with certain prompts, T is broken.

Goal: Train T so that "3+2=" works WITHOUT priming.
      Not find a prompt that makes broken T produce correct output.

This is the real self-improvement: fixing the transform itself.
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
        # NO priming, NO special formatting - just raw input
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


def generate_raw_training_data(n_samples: int = 500) -> List[dict]:
    """Generate training data for RAW inputs - no priming needed after training."""
    np.random.seed(42)

    samples = []

    for _ in range(n_samples):
        a = np.random.randint(1, 15)
        b = np.random.randint(1, 10)

        if np.random.rand() > 0.5:
            # Addition
            prompt = f"{a}+{b}="
            completion = str(a + b)
        else:
            # Subtraction (ensure positive)
            if b > a:
                a, b = b, a
            prompt = f"{a}-{b}="
            completion = str(a - b)

        # Train on RAW equation → answer
        # No priming, no chat format - pure T(equation) = answer
        samples.append({
            "prompt": prompt,
            "completion": completion,
        })

    return samples


def main():
    from mlx_lm import load

    # Use 350M - the model with REAL gaps
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    train_data_dir = "data/training/fix_transform"
    adapter_path = "data/adapters/fix_transform_lora"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 92: FIX THE TRANSFORM")
    logger.info("=" * 60)
    logger.info("Goal: T(equation) = answer WITHOUT priming")

    # Test cases - raw inputs
    arithmetic_tests = [
        ("1+1=", "2"), ("2+2=", "4"), ("3+1=", "4"), ("5+2=", "7"),
        ("4+3=", "7"), ("6+1=", "7"), ("3+3=", "6"), ("2+5=", "7"),
        ("5-2=", "3"), ("4-1=", "3"), ("7-3=", "4"), ("6-2=", "4"),
        ("9-4=", "5"), ("8-3=", "5"), ("10-5=", "5"), ("7-2=", "5"),
    ]

    # Phase 1: Baseline - raw T without priming
    logger.info("\n=== PHASE 1: BASELINE - RAW T(input) ===")

    model, tokenizer = load(model_path)

    raw_baseline, raw_details = evaluate_raw(model, tokenizer, arithmetic_tests)
    logger.info(f"\nT(equation) accuracy (NO PRIMING): {raw_baseline:.0%}")

    logger.info("\nExamples of broken T:")
    for r in raw_details[:8]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} T('{r['input']}') = '{r['predicted']}' (expected '{r['expected']}')")

    # Also test WITH priming to show the gap
    logger.info("\n--- For comparison: T with priming ---")
    primed_acc, _ = evaluate_with_prime(model, tokenizer, arithmetic_tests)
    logger.info(f"T(prime + equation) accuracy: {primed_acc:.0%}")

    logger.info(f"\nGAP: {primed_acc:.0%} with priming vs {raw_baseline:.0%} raw")
    logger.info("This gap is the deficiency in T that we will fix.")

    # Clear
    del model
    import mlx.core as mx
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE RAW TRAINING DATA ===")

    samples = generate_raw_training_data(500)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    # 80/10/10 split
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

    logger.info(f"\nSample training data (RAW):")
    for s in samples[:5]:
        logger.info(f"  '{s['prompt']}' → '{s['completion']}'")

    # Phase 3: Train to fix T
    logger.info("\n=== PHASE 3: TRAINING TO FIX T ===")

    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    n_iters = 300  # More iterations for better learning

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", adapter_path,
        "--batch-size", "8",
        "--num-layers", "16",  # All layers - fixing the whole transform
        "--iters", str(n_iters),
        "--learning-rate", "1e-4",
        "--seed", "42",
        "--steps-per-report", "30",
    ]

    logger.info(f"Training to fix T...")
    logger.info(f"  Layers: ALL (fixing the whole transform)")
    logger.info(f"  Iterations: {n_iters}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-8:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Test fixed T
    logger.info("\n=== PHASE 4: TEST FIXED T ===")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    raw_fixed, raw_details_fixed = evaluate_raw(model, tokenizer, arithmetic_tests)

    logger.info(f"\n{'='*50}")
    logger.info(f"T(equation) - NO PRIMING:")
    logger.info(f"  Before: {raw_baseline:.0%}")
    logger.info(f"  After:  {raw_fixed:.0%}")
    logger.info(f"{'='*50}")

    logger.info("\nFixed T examples:")
    for r in raw_details_fixed:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} T('{r['input']}') = '{r['predicted']}' (expected '{r['expected']}')")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: FIXING THE TRANSFORM")
    logger.info("=" * 60)

    improvement = raw_fixed - raw_baseline
    t_fixed = raw_fixed >= 0.8

    logger.info(f"""
THE PROBLEM:
  T('{arithmetic_tests[0][0]}') should equal '{arithmetic_tests[0][1]}'
  But broken T needed: T('prime + {arithmetic_tests[0][0]}') to work

THE FIX:
  Train T directly on raw inputs
  No priming needed anymore

RESULTS:
  Before training: T works {raw_baseline:.0%} on raw inputs
  After training:  T works {raw_fixed:.0%} on raw inputs
  Improvement:     {improvement:+.0%}

CONCLUSION:
  {'✓ T IS FIXED - works on raw inputs' if t_fixed else '✗ T still needs work'}
""")

    if t_fixed:
        logger.info("*** THE TRANSFORM IS FIXED ***")
        logger.info("No more prompt engineering needed.")
        logger.info("T(equation) = answer, as it should be.")

    # Save
    results = {
        "model": "LFM2-350M",
        "baseline_raw": raw_baseline,
        "baseline_primed": primed_acc,
        "fixed_raw": raw_fixed,
        "improvement": improvement,
        "t_fixed": t_fixed,
        "details": raw_details_fixed,
    }

    output_path = "data/experiments/fix_transform.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


def evaluate_with_prime(model, tokenizer, problems):
    """Helper to evaluate with priming for comparison."""
    import mlx.core as mx

    prime = "Arithmetic means calculating numbers."
    correct = 0

    for problem, expected in problems:
        prompt = f"{prime} {problem}"
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

    return correct / len(problems), []


if __name__ == "__main__":
    main()
