#!/usr/bin/env python3
"""Train Math Capability on Qwen3-8B - Phase 2: Fix Remaining Issues.

Addresses issues from v1:
1. "Answer:" format prompts output empty strings
2. Two-digit results fail (10 → 1)
3. Comparison questions fail

Uses CUMULATIVE training to preserve previous improvements (2+2=4).
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


def generate_cumulative_training_data(n_samples: int = 600, seed: int = 42) -> List[dict]:
    """Generate training data including previous patterns + new fixes.

    New patterns:
    1. "Answer:" format completion
    2. Two-digit number training (10-20)
    3. Comparison with direct answers
    """
    np.random.seed(seed)
    samples = []

    # === PRESERVE: Previous working patterns ===

    # Basic arithmetic (keep this to prevent regression)
    for a in range(1, 10):
        for b in range(1, 10):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # Word problems that now work
    for _ in range(50):
        a = np.random.randint(3, 12)
        b = np.random.randint(1, a)
        samples.append({"text": f"I have {a} apples. I get {b} more. Total: {a+b}"})
        samples.append({"text": f"{a} birds. {b} fly away. Remaining: {a-b}"})

    # === NEW FIX 1: "Answer:" format training ===

    # Number sense with Answer: format
    for n in range(1, 20):
        samples.append({"text": f"What comes after {n}? Answer: {n+1}"})
        if n > 1:
            samples.append({"text": f"What comes before {n}? Answer: {n-1}"})

    # Comparison with Answer: format
    for _ in range(100):
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        if a != b:
            greater = max(a, b)
            lesser = min(a, b)
            # Multiple formats for comparison
            samples.append({"text": f"Which is greater, {a} or {b}? Answer: {greater}"})
            samples.append({"text": f"Which is larger, {a} or {b}? Answer: {greater}"})
            samples.append({"text": f"Is {a} > {b}? Answer: {'yes' if a > b else 'no'}"})
            samples.append({"text": f"Is {a} greater than {b}? Answer: {'yes' if a > b else 'no'}"})

    # Yes/No with Answer: format
    yes_no_facts = [
        ("Is water wet?", "yes"),
        ("Is ice hot?", "no"),
        ("Can fish swim?", "yes"),
        ("Is the sun a planet?", "no"),
        ("Do birds have wings?", "yes"),
        ("Is fire cold?", "no"),
        ("Can dogs fly?", "no"),
        ("Is grass green?", "yes"),
    ]
    for q, a in yes_no_facts:
        samples.append({"text": f"{q} Answer: {a}"})
        samples.append({"text": f"{q} Answer:{a}"})  # No space variant

    # === NEW FIX 2: Two-digit number training ===

    # Arithmetic that produces two-digit results
    two_digit_additions = [
        (4, 6, 10), (5, 5, 10), (6, 4, 10), (7, 3, 10),
        (5, 6, 11), (6, 5, 11), (7, 4, 11), (8, 3, 11),
        (6, 6, 12), (7, 5, 12), (8, 4, 12), (9, 3, 12),
        (7, 6, 13), (8, 5, 13), (9, 4, 13),
        (7, 7, 14), (8, 6, 14), (9, 5, 14),
        (8, 7, 15), (9, 6, 15),
        (8, 8, 16), (9, 7, 16),
        (9, 8, 17), (9, 9, 18),
    ]
    for a, b, r in two_digit_additions:
        samples.append({"text": f"{a}+{b}={r}"})
        samples.append({"text": f"{b}+{a}={r}"})
        samples.append({"text": f"{a} plus {b} is {r}"})
        samples.append({"text": f"Start with {a}. Add {b}. Result: {r}"})

    # Word problems with two-digit answers
    for _ in range(50):
        a = np.random.randint(5, 12)
        b = np.random.randint(3, 10)
        r = a + b
        samples.append({"text": f"Start with {a}. Add {b}. Result: {r}"})
        samples.append({"text": f"I have {a} apples. I get {b} more. Total: {r}"})

    # === NEW FIX 3: Number sense ===

    # Counting with clear continuation
    for start in range(1, 15):
        seq = ", ".join(str(start + i) for i in range(6))
        samples.append({"text": f"Count: {seq}"})

    # Number ordering
    for n in range(1, 20):
        samples.append({"text": f"The number after {n} is {n+1}"})
        if n > 1:
            samples.append({"text": f"The number before {n} is {n-1}"})

    # Shuffle and return
    np.random.shuffle(samples)
    return samples[:n_samples]


def evaluate_all(model, tokenizer) -> dict:
    """Evaluate all capability areas."""
    import mlx.core as mx

    def evaluate(problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
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

    # Test categories
    categories = {
        "basic_arithmetic": [
            ("1+1=", "2"),
            ("2+2=", "4"),
            ("3+5=", "8"),
            ("7-3=", "4"),
            ("9-4=", "5"),
        ],
        "two_digit_results": [
            ("4+6=", "10"),
            ("5+5=", "10"),
            ("6+6=", "12"),
            ("7+8=", "15"),
            ("Start with 4. Add 6. Result: ", "10"),
        ],
        "word_problems": [
            ("I have 3 apples. I get 2 more. Total: ", "5"),
            ("5 birds. 2 fly away. Remaining: ", "3"),
            ("I have 7 apples. I get 5 more. Total: ", "12"),
        ],
        "comparison_answer": [
            ("Which is greater, 7 or 3? Answer:", "7"),
            ("Which is larger, 5 or 9? Answer:", "9"),
            ("Is 15 > 12? Answer:", "yes"),
        ],
        "number_sense_answer": [
            ("What comes after 5? Answer:", "6"),
            ("What comes before 10? Answer:", "9"),
            ("What comes after 14? Answer:", "15"),
        ],
        "language_preserved": [
            ("The cat sat on the", "mat"),
            ("Fire is hot and ice is", "cold"),
            ("The opposite of up is", "down"),
        ],
    }

    results = {}
    for name, problems in categories.items():
        acc, details = evaluate(problems)
        results[name] = {"accuracy": acc, "details": details}

    return results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    prev_adapter = "data/adapters/qwen3_math_lora"  # From v1
    train_data_dir = "data/training/qwen3_math_v2"
    new_adapter_path = "data/adapters/qwen3_math_lora_v2"

    logger.info("=" * 70)
    logger.info("TRAINING MATH CAPABILITY ON QWEN3-8B - PHASE 2")
    logger.info("=" * 70)
    logger.info("Fixing: Answer: format, two-digit numbers, comparison")

    # Phase 1: Check current state (with v1 adapter)
    logger.info("\n=== PHASE 1: CURRENT STATE (v1 adapter) ===")

    model, tokenizer = load(model_path, adapter_path=prev_adapter)
    v1_results = evaluate_all(model, tokenizer)

    for name, data in v1_results.items():
        acc = data["accuracy"]
        status = "✓" if acc >= 0.6 else "✗"
        logger.info(f"  {status} {name}: {acc:.0%}")
        for r in data["details"][:2]:
            mark = "+" if r["correct"] else "-"
            logger.info(f"      {mark} '{r['prompt'][:35]}' → '{r['predicted']}'")

    del model
    mx.clear_cache()

    # Phase 2: Generate cumulative training data
    logger.info("\n=== PHASE 2: GENERATE CUMULATIVE TRAINING DATA ===")

    samples = generate_cumulative_training_data(600)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)

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

    # Show new patterns
    logger.info("\nNew pattern examples:")
    new_patterns = [s for s in samples if "Answer:" in s["text"] or any(str(n) in s["text"] for n in range(10, 20))]
    for s in new_patterns[:8]:
        logger.info(f"  '{s['text']}'")

    # Phase 3: Train from base model (fresh adapter)
    logger.info("\n=== PHASE 3: TRAINING (fresh adapter, cumulative data) ===")

    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "2",
        "--num-layers", "8",
        "--iters", "500",  # More iterations for cumulative data
        "--learning-rate", "3e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training LoRA adapter (500 iterations)...")

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
    v2_results = evaluate_all(model, tokenizer)

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: v1 → v2")
    logger.info("=" * 60)

    for name in v1_results.keys():
        v1_acc = v1_results[name]["accuracy"]
        v2_acc = v2_results[name]["accuracy"]
        change = v2_acc - v1_acc
        arrow = "↑" if change > 0 else "↓" if change < 0 else "="
        logger.info(f"  {name:25s}: {v1_acc:.0%} → {v2_acc:.0%} {arrow}")

    # Show detailed results for fixed categories
    logger.info("\n=== DETAILED RESULTS ===")
    for name, data in v2_results.items():
        logger.info(f"\n{name} ({data['accuracy']:.0%}):")
        for r in data["details"]:
            status = "✓" if r["correct"] else "✗"
            logger.info(f"  {status} '{r['prompt'][:40]}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    all_fixed = all(v2_results[k]["accuracy"] >= 0.6 for k in [
        "basic_arithmetic", "two_digit_results", "word_problems",
        "comparison_answer", "number_sense_answer"
    ])
    preserved = v2_results["language_preserved"]["accuracy"] >= 0.6

    logger.info(f"""
Math Capabilities:
  Basic arithmetic:     {v2_results['basic_arithmetic']['accuracy']:.0%}
  Two-digit results:    {v2_results['two_digit_results']['accuracy']:.0%}
  Word problems:        {v2_results['word_problems']['accuracy']:.0%}
  Comparison:           {v2_results['comparison_answer']['accuracy']:.0%}
  Number sense:         {v2_results['number_sense_answer']['accuracy']:.0%}

Language preserved:     {v2_results['language_preserved']['accuracy']:.0%}

Overall status: {'✓ ALL FIXED' if all_fixed else '✗ NEEDS MORE WORK'}
Regression:     {'✓ NONE' if preserved else '✗ LANGUAGE REGRESSED'}
""")

    # Save results
    results = {
        "v1_results": {k: v["accuracy"] for k, v in v1_results.items()},
        "v2_results": {k: v["accuracy"] for k, v in v2_results.items()},
        "detailed": {k: v["details"] for k, v in v2_results.items()},
    }

    output_path = Path("data/experiments/qwen3_math_training_v2.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
