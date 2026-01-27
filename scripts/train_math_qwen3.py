#!/usr/bin/env python3
"""Train Math Capability on Qwen3-8B using Text Continuation Format.

The proven approach from LFM2-350M experiments:
- Text continuation: {"text": "3+2=5"} NOT {"prompt": "3+2=", "completion": "5"}
- Cumulative training data to prevent forgetting
- LoRA adapter training

This addresses the baseline scan findings:
- Math tier at 50% (should be 70%+)
- 2+2=5 error
- Number sense weak at 20%
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


def generate_math_training_data(n_samples: int = 500, seed: int = 42) -> List[dict]:
    """Generate math training data in text continuation format.

    Key insight: The model needs to see complete equations as text.
    Not "3+2=" -> "5", but "3+2=5" as a single text.
    """
    np.random.seed(seed)
    samples = []

    # Basic arithmetic - focus on the 2+2=4 case and variations
    for a in range(1, 10):
        for b in range(1, 10):
            # Addition
            samples.append({"text": f"{a}+{b}={a+b}"})
            # Subtraction (ensure positive)
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # Word form variations
    word_forms = [
        ("{a} plus {b} is {r}", lambda a, b: a + b),
        ("{a} + {b} = {r}", lambda a, b: a + b),
        ("{a}+{b}={r}", lambda a, b: a + b),
        ("{a} minus {b} is {r}", lambda a, b: a - b if a >= b else None),
        ("{a} - {b} = {r}", lambda a, b: a - b if a >= b else None),
        ("{a}-{b}={r}", lambda a, b: a - b if a >= b else None),
    ]

    for _ in range(n_samples // 4):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        for template, fn in word_forms:
            r = fn(a, b)
            if r is not None:
                samples.append({"text": template.format(a=a, b=b, r=r)})

    # Number sense - what comes after/before
    for n in range(1, 20):
        samples.append({"text": f"The number after {n} is {n+1}"})
        if n > 1:
            samples.append({"text": f"The number before {n} is {n-1}"})

    # Counting sequences
    for start in range(1, 10):
        seq = ", ".join(str(start + i) for i in range(5))
        samples.append({"text": f"Count: {seq}"})

    # Comparison training - this was weak at 25%
    comparisons = [
        ("Which is greater, {a} or {b}? {g}", lambda a, b: max(a, b)),
        ("{a} is {'greater' if a > b else 'less'} than {b}", lambda a, b: None),
    ]
    for _ in range(n_samples // 8):
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        if a != b:
            g = a if a > b else b
            samples.append({"text": f"Which is greater, {a} or {b}? {g}"})
            samples.append({"text": f"{a} is {'greater' if a > b else 'less'} than {b}"})

    # Simple word problems
    word_problem_templates = [
        ("I have {a} apples. I get {b} more. Total: {r}", lambda a, b: a + b),
        ("{a} birds. {b} fly away. Remaining: {r}", lambda a, b: a - b if a >= b else None),
        ("Start with {a}. Add {b}. Result: {r}", lambda a, b: a + b),
        ("There are {a} cats. {b} leave. Remaining: {r}", lambda a, b: a - b if a >= b else None),
    ]

    for _ in range(n_samples // 4):
        a = np.random.randint(3, 12)
        b = np.random.randint(1, a)  # Ensure valid subtraction
        for template, fn in word_problem_templates:
            r = fn(a, b)
            if r is not None:
                samples.append({"text": template.format(a=a, b=b, r=r)})

    # Shuffle and return
    np.random.shuffle(samples)
    return samples[:n_samples]


def evaluate_math(model, tokenizer, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate math problems."""
    import mlx.core as mx

    results = []
    correct = 0

    for prompt, expected in problems:
        tokens = tokenizer.encode(prompt)
        logits = model(mx.array([tokens]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_idx = int(np.argmax(probs))
        predicted = tokenizer.decode([top_idx]).strip()

        is_correct = expected.lower() in predicted.lower() or predicted.lower() == expected.lower()
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_math"
    adapter_path = "data/adapters/qwen3_math_lora"

    logger.info("=" * 70)
    logger.info("TRAINING MATH CAPABILITY ON QWEN3-8B")
    logger.info("=" * 70)
    logger.info("Using text continuation format (proven approach)")

    # Test problems from baseline scan
    test_problems = [
        ("1+1=", "2"),
        ("2+2=", "4"),  # This was 5 in baseline!
        ("3+5=", "8"),
        ("7-3=", "4"),
        ("9-4=", "5"),
        ("I have 3 apples. I get 2 more. Total: ", "5"),
        ("5 birds. 2 fly away. Remaining: ", "3"),
        ("Start with 4. Add 6. Result: ", "10"),
        ("Which is greater, 7 or 3? Answer:", "7"),
        ("What comes after 5? Answer:", "6"),
    ]

    # Phase 1: Baseline evaluation
    logger.info("\n=== PHASE 1: BASELINE EVALUATION ===")

    model, tokenizer = load(model_path)
    baseline_acc, baseline_details = evaluate_math(model, tokenizer, test_problems)

    logger.info(f"Baseline accuracy: {baseline_acc:.0%}")
    for r in baseline_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:40]}' → '{r['predicted']}' (expected '{r['expected']}')")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE TRAINING DATA ===")

    samples = generate_math_training_data(500)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    # Split into train/valid/test
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

    logger.info("\nSample training data:")
    for s in samples[:10]:
        logger.info(f"  '{s['text']}'")

    # Phase 3: Train LoRA adapter
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(adapter_path).mkdir(parents=True, exist_ok=True)

    # For 8B model, use fewer layers and lower learning rate
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", adapter_path,
        "--batch-size", "2",  # Smaller batch for 8B model
        "--num-layers", "8",  # Fewer layers for larger model
        "--iters", "400",
        "--learning-rate", "3e-5",  # Lower LR for larger model
        "--seed", "42",
        "--steps-per-report", "50",
    ]

    logger.info("Training LoRA adapter...")
    logger.info(f"  Layers: 8, LR: 3e-5, Iters: 400")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 30 minutes")
        return
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluate trained model
    logger.info("\n=== PHASE 4: EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    trained_acc, trained_details = evaluate_math(model, tokenizer, test_problems)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Baseline:  {baseline_acc:.0%}")
    logger.info(f"  Trained:   {trained_acc:.0%}")
    logger.info(f"  Change:    {trained_acc - baseline_acc:+.0%}")
    logger.info(f"{'='*60}")

    logger.info("\nTrained model examples:")
    for r in trained_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:40]}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Check if 2+2=4 now works
    logger.info("\n=== CRITICAL CHECK: 2+2 ===")
    for r in trained_details:
        if "2+2" in r["prompt"]:
            if r["correct"]:
                logger.info(f"  ✓ 2+2=4 is FIXED!")
            else:
                logger.info(f"  ✗ 2+2 still wrong: '{r['predicted']}'")

    # Phase 5: Regression check on language capabilities
    logger.info("\n=== PHASE 5: REGRESSION CHECK ===")

    language_tests = [
        ("The cat sat on the", "mat"),
        ("Is water wet? Answer:", "yes"),
        ("Fire is hot and ice is", "cold"),
        ("All men are mortal. Socrates is a man. Therefore, Socrates is", "mortal"),
    ]

    lang_acc, lang_details = evaluate_math(model, tokenizer, language_tests)
    logger.info(f"Language capability: {lang_acc:.0%}")
    for r in lang_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:40]}' → '{r['predicted']}'")

    # Save results
    results = {
        "model": "Qwen3-8B-bf16",
        "baseline_accuracy": baseline_acc,
        "trained_accuracy": trained_acc,
        "improvement": trained_acc - baseline_acc,
        "language_preserved": lang_acc,
        "test_details": trained_details,
    }

    output_path = Path("data/experiments/qwen3_math_training.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    improved = trained_acc > baseline_acc
    preserved = lang_acc >= 0.5

    logger.info(f"""
Math Capability:
  Before: {baseline_acc:.0%}
  After:  {trained_acc:.0%}
  Status: {'✓ IMPROVED' if improved else '✗ No improvement'}

Language (preserved?):
  Status: {'✓ PRESERVED' if preserved else '✗ REGRESSED'} ({lang_acc:.0%})

Next steps:
  {'1. Run full baseline scan to verify all capabilities' if improved else '1. Debug training - check data format'}
  2. Move to comparison capability training
  3. Then word problems with cumulative data
""")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
