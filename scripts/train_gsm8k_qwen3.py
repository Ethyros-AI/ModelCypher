#!/usr/bin/env python3
"""Train GSM8K (Grade School Math) on Qwen3-8B.

Building on the 100% arithmetic foundation, this trains multi-step
math reasoning using GSM8K word problems.

Key approach:
1. Text continuation format (proven to work)
2. Cumulative training (include arithmetic to prevent regression)
3. GSM8K problems with chain-of-thought preserved
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

from modelcypher.core.use_cases.curriculum import BenchmarkLoader, save_for_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_cumulative_arithmetic(n_samples: int = 200) -> List[dict]:
    """Generate arithmetic samples to prevent regression."""
    np.random.seed(42)
    samples = []

    # Basic arithmetic
    for a in range(1, 15):
        for b in range(1, 15):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # Word form
    for _ in range(n_samples // 4):
        a = np.random.randint(1, 15)
        b = np.random.randint(1, 15)
        samples.append({"text": f"{a} plus {b} is {a+b}"})
        if a >= b:
            samples.append({"text": f"{a} minus {b} is {a-b}"})

    np.random.shuffle(samples)
    return samples[:n_samples]


def format_gsm8k_for_training(loader: BenchmarkLoader, limit: int = 500) -> List[dict]:
    """Load GSM8K and format for text continuation training.

    GSM8K format: question -> answer with chain-of-thought
    We keep the full solution to teach reasoning steps.
    """
    benchmark = loader.load("gsm8k", split="train", limit=limit)

    samples = []
    for sample in benchmark.samples:
        # Get full answer with reasoning
        full_answer = sample.metadata.get("full_answer", sample.answer)

        # Format as text continuation
        # Include the question and full solution
        text = f"Question: {sample.prompt.replace('Answer:', '').strip()}\n\nSolution: {full_answer}"
        samples.append({"text": text})

        # Also add simplified version (question -> final answer only)
        samples.append({"text": f"{sample.prompt} {sample.answer}"})

    return samples


def evaluate_gsm8k(model, tokenizer, problems: List[Tuple[str, str]], max_tokens: int = 10) -> Tuple[float, List[dict]]:
    """Evaluate GSM8K problems with multi-token generation."""
    import mlx.core as mx
    import re

    results = []
    correct = 0

    for prompt, expected in problems:
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

            decoded = tokenizer.decode([next_tok])
            if decoded.strip() in ["", "\n", ".", "<|im_end|>"]:
                break

        predicted = tokenizer.decode(generated).strip()
        # Clean special tokens
        predicted = predicted.replace("<|im_end|>", "").replace("!", "").strip()

        # Extract first number from prediction
        numbers = re.findall(r'-?\d+', predicted)
        predicted_clean = numbers[0] if numbers else ""

        is_correct = predicted_clean == expected
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt[:60],
            "expected": expected,
            "predicted": predicted_clean,
            "raw": predicted[:50],
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    prev_adapter = "data/adapters/qwen3_math_lora"  # Our 100% arithmetic adapter
    train_data_dir = "data/training/qwen3_gsm8k"
    new_adapter_path = "data/adapters/qwen3_gsm8k_lora"

    logger.info("=" * 70)
    logger.info("TRAINING GSM8K ON QWEN3-8B")
    logger.info("=" * 70)
    logger.info("Building on 100% arithmetic foundation")

    loader = BenchmarkLoader()

    # Phase 1: Baseline evaluation on GSM8K
    logger.info("\n=== PHASE 1: BASELINE EVALUATION ===")

    # Load GSM8K test samples
    gsm_test = loader.load("gsm8k", split="test", limit=50)
    test_problems = gsm_test.to_evaluation_format()

    model, tokenizer = load(model_path, adapter_path=prev_adapter)

    baseline_acc, baseline_details = evaluate_gsm8k(model, tokenizer, test_problems[:20])

    logger.info(f"GSM8K baseline (with arithmetic adapter): {baseline_acc:.0%}")
    for r in baseline_details[:5]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:45]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Also check arithmetic preservation
    arith_tests = [
        ("2+2=", "4"), ("3+5=", "8"), ("7-3=", "4"),
        ("5+5=", "10"), ("9-4=", "5"),
    ]
    arith_acc, _ = evaluate_gsm8k(model, tokenizer, arith_tests, max_tokens=5)
    logger.info(f"Arithmetic preserved: {arith_acc:.0%}")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE TRAINING DATA ===")

    # GSM8K training samples
    gsm_samples = format_gsm8k_for_training(loader, limit=500)
    logger.info(f"GSM8K samples: {len(gsm_samples)}")

    # Cumulative arithmetic (prevent regression)
    arith_samples = generate_cumulative_arithmetic(200)
    logger.info(f"Arithmetic samples: {len(arith_samples)}")

    # Combine
    all_samples = gsm_samples + arith_samples
    np.random.shuffle(all_samples)

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    n_train = int(len(all_samples) * 0.85)
    n_valid = int(len(all_samples) * 0.10)

    for name, data in [
        ("train", all_samples[:n_train]),
        ("valid", all_samples[n_train:n_train + n_valid]),
        ("test", all_samples[n_train + n_valid:]),
    ]:
        path = Path(train_data_dir) / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"  {name}: {len(data)} samples")

    logger.info("\nSample GSM8K training data:")
    for s in [s for s in gsm_samples if "Question" in s["text"]][:3]:
        logger.info(f"  '{s['text'][:100]}...'")

    # Phase 3: Train
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "1",  # Small batch for longer sequences
        "--num-layers", "12",  # More layers for reasoning
        "--iters", "600",  # More iterations
        "--learning-rate", "2e-5",  # Lower LR
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training LoRA adapter (600 iterations, 12 layers)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except subprocess.TimeoutExpired:
        logger.error("Training timed out")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluate
    logger.info("\n=== PHASE 4: EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)

    trained_acc, trained_details = evaluate_gsm8k(model, tokenizer, test_problems[:20])

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  GSM8K Baseline:  {baseline_acc:.0%}")
    logger.info(f"  GSM8K Trained:   {trained_acc:.0%}")
    logger.info(f"  Change:          {trained_acc - baseline_acc:+.0%}")
    logger.info(f"{'='*60}")

    logger.info("\nGSM8K examples:")
    for r in trained_details[:10]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:40]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Regression check
    logger.info("\n=== REGRESSION CHECK ===")

    arith_trained_acc, arith_trained = evaluate_gsm8k(model, tokenizer, arith_tests, max_tokens=5)
    logger.info(f"Arithmetic: {arith_acc:.0%} → {arith_trained_acc:.0%}")

    for r in arith_trained:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} {r['prompt']} → '{r['predicted']}'")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    gsm_improved = trained_acc > baseline_acc
    arith_preserved = arith_trained_acc >= 0.8

    logger.info(f"""
GSM8K (multi-step math):
  Before: {baseline_acc:.0%}
  After:  {trained_acc:.0%}
  Status: {'✓ IMPROVED' if gsm_improved else '✗ No improvement'}

Arithmetic (preserved?):
  Status: {'✓ PRESERVED' if arith_preserved else '✗ REGRESSED'} ({arith_trained_acc:.0%})

Curriculum Progress:
  ✓ Tier 4a: Basic Arithmetic (100%)
  {'✓' if gsm_improved else '→'} Tier 4b: GSM8K Word Problems ({trained_acc:.0%})
  → Next: ARC (Reasoning Tier)
""")

    # Save results
    results = {
        "gsm8k_baseline": baseline_acc,
        "gsm8k_trained": trained_acc,
        "arithmetic_preserved": arith_trained_acc,
        "details": trained_details,
    }

    output_path = Path("data/experiments/qwen3_gsm8k_training.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
