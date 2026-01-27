#!/usr/bin/env python3
"""Train Multi-Step Math on Qwen3-8B - Bridge from Arithmetic to GSM8K.

The curriculum gap: Basic arithmetic (1-step) → GSM8K (5+ steps)
This creates intermediate 2-3 step problems to bridge the gap.

Examples:
- 2-step: "3+5=8, 8+2=10"
- 3-step: "I have 5. Get 3 more. Then lose 2. Total: 6"
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


def generate_multistep_training_data(n_samples: int = 600, seed: int = 42) -> List[dict]:
    """Generate 2-3 step math problems."""
    np.random.seed(seed)
    samples = []

    # === 1-STEP BASELINE (include to prevent regression) ===
    for a in range(1, 12):
        for b in range(1, 12):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # === 2-STEP CHAIN ===
    # Format: "a+b=c, c+d=e" - explicit chain
    for _ in range(n_samples // 3):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        c = a + b
        d = np.random.randint(1, 10)
        e = c + d

        samples.append({"text": f"{a}+{b}={c}, {c}+{d}={e}"})
        samples.append({"text": f"{a} plus {b} is {c}. {c} plus {d} is {e}."})

        # Mixed operations
        if c >= d:
            samples.append({"text": f"{a}+{b}={c}, {c}-{d}={c-d}"})

    # === 2-STEP WORD PROBLEMS ===
    templates_2step = [
        ("I have {a}. I get {b} more. Now I have {c}. Then I get {d} more. Total: {e}",
         lambda a, b, c, d: (a+b, a+b+d)),
        ("Start with {a}. Add {b} to get {c}. Add {d} more. Result: {e}",
         lambda a, b, c, d: (a+b, a+b+d)),
        ("{a} birds. {b} join. Now {c}. Then {d} leave. Remaining: {e}",
         lambda a, b, c, d: (a+b, a+b-d) if a+b >= d else None),
    ]

    for _ in range(n_samples // 3):
        a = np.random.randint(3, 12)
        b = np.random.randint(1, 8)
        d = np.random.randint(1, 6)

        for template, fn in templates_2step:
            result = fn(a, b, 0, d)
            if result is not None:
                c, e = result
                samples.append({"text": template.format(a=a, b=b, c=c, d=d, e=e)})

    # === 3-STEP CHAIN ===
    for _ in range(n_samples // 4):
        a = np.random.randint(2, 8)
        b = np.random.randint(1, 6)
        c = np.random.randint(1, 6)
        d = np.random.randint(1, 4)

        step1 = a + b
        step2 = step1 + c
        step3 = step2 - d if step2 >= d else step2 + d

        samples.append({"text": f"{a}+{b}={step1}, {step1}+{c}={step2}, {step2}+{d}={step2+d}"})

        # Word problem form
        samples.append({
            "text": f"Start: {a}. Add {b}: {step1}. Add {c}: {step2}. Subtract {d}: {step2-d if step2>=d else 'N/A'}."
        })

    # === CHAINED REASONING FORMAT (like GSM8K) ===
    for _ in range(n_samples // 4):
        a = np.random.randint(5, 15)
        b = np.random.randint(1, 5)
        c = np.random.randint(1, 5)

        step1 = a - b
        step2 = step1 - c
        final = step2

        if step2 >= 0:
            # GSM8K-like format
            samples.append({
                "text": f"Problem: I have {a} apples. I eat {b}. I give away {c}. How many left?\n"
                       f"Step 1: {a} - {b} = {step1}\n"
                       f"Step 2: {step1} - {c} = {final}\n"
                       f"Answer: {final}"
            })

            # Shorter format
            samples.append({
                "text": f"{a} apples. Eat {b}. Give {c}. Remaining: {final}"
            })

    # === MULTIPLICATION CHAIN ===
    for _ in range(n_samples // 5):
        a = np.random.randint(2, 6)
        b = np.random.randint(2, 5)
        c = np.random.randint(2, 4)

        step1 = a * b
        step2 = step1 * c

        samples.append({"text": f"{a}*{b}={step1}, {step1}*{c}={step2}"})
        samples.append({"text": f"{a} times {b} is {step1}. {step1} times {c} is {step2}."})

    np.random.shuffle(samples)
    return samples[:n_samples]


def evaluate_multistep(model, tokenizer, problems: List[Tuple[str, str]], max_tokens: int = 15) -> Tuple[float, List[dict]]:
    """Evaluate multi-step problems."""
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

        predicted = tokenizer.decode(generated).strip()
        predicted = predicted.replace("<|im_end|>", "").replace("!", "").strip()
        numbers = re.findall(r'-?\d+', predicted)
        predicted_clean = numbers[0] if numbers else ""

        is_correct = predicted_clean == expected
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt[:50],
            "expected": expected,
            "predicted": predicted_clean,
            "correct": is_correct,
        })

    return correct / len(problems) if problems else 0, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    prev_adapter = "data/adapters/qwen3_math_lora"
    train_data_dir = "data/training/qwen3_multistep"
    new_adapter_path = "data/adapters/qwen3_multistep_lora"

    logger.info("=" * 70)
    logger.info("TRAINING MULTI-STEP MATH - CURRICULUM BRIDGE")
    logger.info("=" * 70)
    logger.info("Bridging: Arithmetic (1-step) → Multi-step (2-3) → GSM8K (5+)")

    # Test problems (2-3 steps)
    test_problems = [
        # 2-step
        ("5+3=8, 8+2=", "10"),
        ("7+4=11, 11-3=", "8"),
        ("Start with 6. Add 4. Then add 5. Result:", "15"),
        ("I have 8. Get 3 more. Then lose 2. Total:", "9"),
        # 3-step
        ("4+3=7, 7+2=9, 9+1=", "10"),
        ("10 apples. Eat 3. Give 2. Remaining:", "5"),
        # Multiplication chain
        ("3*4=12, 12*2=", "24"),
        ("2 times 5 is 10. 10 times 3 is", "30"),
    ]

    # Phase 1: Baseline
    logger.info("\n=== PHASE 1: BASELINE ===")

    model, tokenizer = load(model_path, adapter_path=prev_adapter)
    baseline_acc, baseline_details = evaluate_multistep(model, tokenizer, test_problems)

    logger.info(f"Baseline (arithmetic adapter): {baseline_acc:.0%}")
    for r in baseline_details:
        mark = "OK" if r["correct"] else "XX"
        logger.info(f"  {mark} '{r['prompt'][:40]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE TRAINING DATA ===")

    samples = generate_multistep_training_data(600)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

    n_train = int(len(samples) * 0.85)
    n_valid = int(len(samples) * 0.10)

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
    for s in [s for s in samples if "Step" in s["text"] or "," in s["text"]][:5]:
        logger.info(f"  '{s['text'][:70]}...'")

    # Phase 3: Train
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "2",
        "--num-layers", "10",
        "--iters", "500",
        "--learning-rate", "3e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training LoRA adapter...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-8:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluate
    logger.info("\n=== PHASE 4: EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)
    trained_acc, trained_details = evaluate_multistep(model, tokenizer, test_problems)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Baseline:  {baseline_acc:.0%}")
    logger.info(f"  Trained:   {trained_acc:.0%}")
    logger.info(f"  Change:    {trained_acc - baseline_acc:+.0%}")

    logger.info("\nTrained examples:")
    for r in trained_details:
        mark = "OK" if r["correct"] else "XX"
        logger.info(f"  {mark} '{r['prompt'][:40]}...' → '{r['predicted']}' (expected '{r['expected']}')")

    # Regression check on basic arithmetic
    logger.info("\n=== REGRESSION CHECK ===")
    arith_tests = [
        ("2+2=", "4"), ("5+5=", "10"), ("9-4=", "5"), ("7*3=", "21"),
    ]
    arith_acc, arith_details = evaluate_multistep(model, tokenizer, arith_tests, max_tokens=5)
    logger.info(f"Basic arithmetic: {arith_acc:.0%}")
    for r in arith_details:
        mark = "OK" if r["correct"] else "XX"
        logger.info(f"  {mark} {r['prompt']} → '{r['predicted']}'")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CURRICULUM PROGRESS")
    logger.info("=" * 70)

    improved = trained_acc > baseline_acc
    preserved = arith_acc >= 0.75

    logger.info(f"""
Multi-step Math (2-3 steps):
  Before: {baseline_acc:.0%}
  After:  {trained_acc:.0%}
  Status: {'IMPROVED' if improved else 'No improvement'}

Basic Arithmetic:
  Status: {'PRESERVED' if preserved else 'REGRESSED'} ({arith_acc:.0%})

Curriculum Ladder:
  1. Basic Arithmetic (1-step): 100%
  2. Multi-step (2-3 steps): {trained_acc:.0%} {'<-- YOU ARE HERE' if improved else ''}
  3. GSM8K (5+ steps): Next
""")

    # Save results
    results = {
        "baseline": baseline_acc,
        "trained": trained_acc,
        "arithmetic_preserved": arith_acc,
        "details": trained_details,
    }

    output_path = Path("data/experiments/qwen3_multistep_training.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
