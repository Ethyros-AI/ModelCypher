#!/usr/bin/env python3
"""Experiment 91: Actually Train the Self-Improvement.

This is the real deal. We:
1. Load the model and measure baseline
2. Train LoRA on verified self-play data
3. Measure post-training accuracy
4. Verify no regression on oracle capabilities

The model teaches itself using its own verified knowledge.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate accuracy on problems."""
    import mlx.core as mx

    results = []
    correct = 0

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

        is_correct = expected in predicted or predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    train_data = "data/training/self_improve_dataset"  # Directory with train.jsonl
    adapter_path = "data/adapters/self_improve_lora"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 91: SELF-IMPROVEMENT TRAINING")
    logger.info("=" * 60)

    # Test sets
    arithmetic_tests = [
        ("1+1=", "2"), ("2+2=", "4"), ("3+1=", "4"), ("5+2=", "7"),
        ("4+3=", "7"), ("6+1=", "7"), ("3+3=", "6"), ("2+5=", "7"),
        ("5-2=", "3"), ("4-1=", "3"), ("7-3=", "4"), ("6-2=", "4"),
    ]

    word_problem_tests = [
        ("I have 5 apples. I get 3 more. Total:", "8"),
        ("7 birds. 4 fly away. Left:", "3"),
        ("Start with 6. Add 2. Result:", "8"),
        ("There are 8 cats. 3 leave. Left:", "5"),
        ("9 toys plus 1 toys. Sum:", "10"),
        ("Tom has 4 candies. He gives 2 away. Left:", "2"),
        ("Begin with 5. Take away 2. Result:", "3"),
        ("Mary has 3 books. She gets 4 more. Total:", "7"),
    ]

    prime = "Arithmetic means calculating numbers."

    # Phase 1: Baseline measurement
    logger.info("\n=== PHASE 1: BASELINE MEASUREMENT ===")

    logger.info("Loading base model...")
    model, tokenizer = load(model_path)

    arith_baseline, _ = evaluate_accuracy(model, tokenizer, prime, arithmetic_tests)
    wp_baseline_raw, _ = evaluate_accuracy(model, tokenizer, "", word_problem_tests)
    wp_baseline_primed, _ = evaluate_accuracy(model, tokenizer, prime, word_problem_tests)

    logger.info(f"Arithmetic (primed): {arith_baseline:.0%}")
    logger.info(f"Word problems (raw): {wp_baseline_raw:.0%}")
    logger.info(f"Word problems (primed): {wp_baseline_primed:.0%}")

    # Clear model from memory
    del model
    import mlx.core as mx
    mx.metal.clear_cache()

    # Phase 2: LoRA Training
    logger.info("\n=== PHASE 2: LORA TRAINING ===")

    # Create adapter directory
    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    # Calculate iterations: ~3 epochs over 100 samples with batch size 4
    # 100 samples / 4 batch = 25 steps per epoch, 3 epochs = 75 iterations
    n_iters = 100

    logger.info(f"Training LoRA adapter...")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Data: {train_data}")
    logger.info(f"  Output: {adapter_path}")
    logger.info(f"  Iterations: {n_iters}")
    logger.info(f"  Rank: 8")
    logger.info(f"  Layers: 4 (early layers for parsing)")

    # Create LoRA config file
    lora_config = {
        "lora_layers": 4,  # Only early layers for parsing
        "lora_rank": 8,
        "lora_scale": 16.0,  # alpha
    }

    config_path = Path(adapter_path) / "lora_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(lora_config, f)

    logger.info(f"LoRA config saved to: {config_path}")

    # Run mlx_lm lora training
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data,
        "--adapter-path", adapter_path,
        "--batch-size", "4",
        "--num-layers", "4",  # Only early layers
        "--iters", str(n_iters),
        "--learning-rate", "1e-4",
        "--seed", "42",
        "--steps-per-report", "10",
    ]

    logger.info(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"Training failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return
        else:
            logger.info("Training complete!")
            # Show last few lines of output
            for line in result.stdout.strip().split('\n')[-10:]:
                logger.info(f"  {line}")

    except subprocess.TimeoutExpired:
        logger.error("Training timed out!")
        return

    # Phase 3: Post-training evaluation
    logger.info("\n=== PHASE 3: POST-TRAINING EVALUATION ===")

    logger.info("Loading model with trained adapter...")

    # Load model with adapter
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    arith_post, arith_details = evaluate_accuracy(model, tokenizer, prime, arithmetic_tests)
    wp_post_raw, wp_details_raw = evaluate_accuracy(model, tokenizer, "", word_problem_tests)
    wp_post_primed, wp_details_primed = evaluate_accuracy(model, tokenizer, prime, word_problem_tests)

    logger.info(f"\nArithmetic (primed): {arith_baseline:.0%} → {arith_post:.0%}")
    logger.info(f"Word problems (raw): {wp_baseline_raw:.0%} → {wp_post_raw:.0%}")
    logger.info(f"Word problems (primed): {wp_baseline_primed:.0%} → {wp_post_primed:.0%}")

    # Show details for word problems
    logger.info("\nWord problem results (raw):")
    for r in wp_details_raw:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['problem'][:35]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Phase 4: Analysis
    logger.info("\n=== PHASE 4: ANALYSIS ===")

    arith_regression = arith_post < arith_baseline - 0.1
    wp_improved = wp_post_raw > wp_baseline_raw + 0.1

    logger.info(f"""
RESULTS:
                        Before    After     Change
  Arithmetic (oracle):  {arith_baseline:>6.0%}    {arith_post:>5.0%}    {'✓ preserved' if not arith_regression else '✗ REGRESSION'}
  Word problems (raw):  {wp_baseline_raw:>6.0%}    {wp_post_raw:>5.0%}    {'✓ IMPROVED!' if wp_improved else 'unchanged'}
  Word problems (prime):{wp_baseline_primed:>6.0%}    {wp_post_primed:>5.0%}

SUCCESS CRITERIA:
  [{'✓' if not arith_regression else '✗'}] Arithmetic preserved (no regression)
  [{'✓' if wp_improved else '✗'}] Word problems improved
  [{'✓' if not arith_regression and wp_improved else '✗'}] Safe self-improvement achieved
""")

    if not arith_regression and wp_improved:
        logger.info("*** THE MODEL TAUGHT ITSELF ***")
        logger.info("Using its own verified arithmetic as oracle,")
        logger.info("it generated training data and learned word problems.")
        logger.info("This is autonomous self-improvement.")
    elif not arith_regression:
        logger.info("Arithmetic preserved but word problems didn't improve much.")
        logger.info("May need more training data or iterations.")
    else:
        logger.info("WARNING: Arithmetic regressed! Training was not safe.")

    # Save results
    results = {
        "model": "LFM2.5-1.2B-Instruct",
        "adapter_path": adapter_path,
        "training": {
            "data": train_data,
            "iterations": n_iters,
            "lora_rank": 8,
            "lora_layers": 4,
        },
        "baseline": {
            "arithmetic_primed": arith_baseline,
            "word_problems_raw": wp_baseline_raw,
            "word_problems_primed": wp_baseline_primed,
        },
        "post_training": {
            "arithmetic_primed": arith_post,
            "word_problems_raw": wp_post_raw,
            "word_problems_primed": wp_post_primed,
        },
        "success": {
            "arithmetic_preserved": not arith_regression,
            "word_problems_improved": wp_improved,
            "safe_self_improvement": not arith_regression and wp_improved,
        },
        "word_problem_details": wp_details_raw,
    }

    output_path = "data/experiments/self_improvement_training.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
