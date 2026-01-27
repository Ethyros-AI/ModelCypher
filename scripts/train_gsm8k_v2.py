#!/usr/bin/env python3
"""Train GSM8K v2 - Balanced Curriculum with Better Arithmetic Preservation.

Key improvements:
1. Start from best arithmetic adapter (100% accuracy)
2. 50% of training data is cumulative arithmetic + multi-step
3. More training iterations (800) for longer sequences
4. Better evaluation across all curriculum tiers
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

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_balanced_training_data(n_samples: int = 800, seed: int = 42) -> List[dict]:
    """Generate training data with 50% arithmetic preservation.

    Distribution:
    - 25% Basic arithmetic (1-step)
    - 25% Multi-step chains (2-3 steps)
    - 50% GSM8K-style CoT problems
    """
    np.random.seed(seed)
    samples = []

    n_arithmetic = n_samples // 4
    n_multistep = n_samples // 4
    n_gsm8k = n_samples // 2

    # === TIER 1: BASIC ARITHMETIC (25%) ===
    logger.info(f"Generating {n_arithmetic} arithmetic samples...")

    # All combinations 1-15
    for a in range(1, 16):
        for b in range(1, 16):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # Multiplication table
    for a in range(1, 13):
        for b in range(1, 13):
            samples.append({"text": f"{a}*{b}={a*b}"})

    # Word form variations
    for _ in range(100):
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        samples.append({"text": f"{a} plus {b} is {a+b}"})
        if a >= b:
            samples.append({"text": f"{a} minus {b} is {a-b}"})
        samples.append({"text": f"{a} times {b} is {a*b}"})

    # === TIER 2: MULTI-STEP CHAINS (25%) ===
    logger.info(f"Generating {n_multistep} multi-step samples...")

    # 2-step chains
    for _ in range(n_multistep // 2):
        a = np.random.randint(2, 12)
        b = np.random.randint(1, 10)
        c = np.random.randint(1, 8)
        s1 = a + b
        s2 = s1 + c

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}"})
        samples.append({"text": f"Start with {a}. Add {b} to get {s1}. Add {c}. Result: {s2}"})

        if s1 >= c:
            samples.append({"text": f"{a}+{b}={s1}, {s1}-{c}={s1-c}"})

    # 3-step chains
    for _ in range(n_multistep // 4):
        a = np.random.randint(3, 10)
        b = np.random.randint(1, 6)
        c = np.random.randint(1, 5)
        d = np.random.randint(1, 4)

        s1 = a + b
        s2 = s1 + c
        s3 = s2 - d if s2 >= d else s2 + d

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}, {s2}+{d}={s2+d}"})

        # Word problem form
        if s2 >= d:
            samples.append({
                "text": f"I have {a}. Get {b} more ({s1}). Get {c} more ({s2}). Lose {d}. Total: {s2-d}"
            })

    # === TIER 3: GSM8K-STYLE COT (50%) ===
    logger.info(f"Generating {n_gsm8k} GSM8K CoT samples...")

    # 2-step word problems with CoT
    templates_2step = [
        {
            "q": "I have {a} apples. I get {b} more. I eat {c}. How many left?",
            "calc": lambda a, b, c: (a + b, a + b - c),
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
        },
        {
            "q": "Start with {a} coins. Find {b} more. Spend {c}. How many now?",
            "calc": lambda a, b, c: (a + b, a + b - c),
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
        },
        {
            "q": "{a} birds in a tree. {b} more arrive. {c} fly away. How many remain?",
            "calc": lambda a, b, c: (a + b, a + b - c),
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
        },
        {
            "q": "A store has {a} books. They get {b} more. They sell {c}. How many left?",
            "calc": lambda a, b, c: (a + b, a + b - c),
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
        },
    ]

    for _ in range(n_gsm8k // 3):
        template = np.random.choice(templates_2step)
        a = np.random.randint(5, 20)
        b = np.random.randint(2, 10)
        c = np.random.randint(1, 8)

        s1, s2 = template["calc"](a, b, c)

        if s2 >= 0:
            question = template["q"].format(a=a, b=b, c=c)
            steps = [s.format(a=a, b=b, c=c, s1=s1, s2=s2) for s in template["steps"]]

            text = f"""Question: {question}
Let me solve step by step.
{steps[0]}
{steps[1]}
Final answer: {s2}"""
            samples.append({"text": text})

    # 3-step word problems with CoT
    templates_3step = [
        {
            "q": "I have {a} toys. I get {b} more. I give {c} away. I find {d} more. How many now?",
            "calc": lambda a, b, c, d: (a + b, a + b - c, a + b - c + d),
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}", "Step 3: {s2} + {d} = {s3}"],
        },
        {
            "q": "A bakery makes {a} cakes. Sells {b}. Makes {c} more. Sells {d}. How many left?",
            "calc": lambda a, b, c, d: (a - b, a - b + c, a - b + c - d),
            "steps": ["Step 1: {a} - {b} = {s1}", "Step 2: {s1} + {c} = {s2}", "Step 3: {s2} - {d} = {s3}"],
        },
    ]

    for _ in range(n_gsm8k // 4):
        template = np.random.choice(templates_3step)
        a = np.random.randint(10, 25)
        b = np.random.randint(2, 8)
        c = np.random.randint(2, 6)
        d = np.random.randint(1, 5)

        results = template["calc"](a, b, c, d)

        if all(r >= 0 for r in results):
            s1, s2, s3 = results
            question = template["q"].format(a=a, b=b, c=c, d=d)
            steps = [s.format(a=a, b=b, c=c, d=d, s1=s1, s2=s2, s3=s3) for s in template["steps"]]

            text = f"""Question: {question}
Let me solve step by step.
{steps[0]}
{steps[1]}
{steps[2]}
Final answer: {s3}"""
            samples.append({"text": text})

    # Multiplication word problems
    for _ in range(n_gsm8k // 4):
        qty = np.random.randint(2, 8)
        price = np.random.randint(2, 10)
        discount = np.random.randint(1, min(5, qty * price - 1))
        total = qty * price
        final = total - discount

        text = f"""Question: Each item costs ${price}. I buy {qty} items. I have a ${discount} coupon. How much do I pay?
Let me solve step by step.
Step 1: {qty} * {price} = {total}
Step 2: {total} - {discount} = {final}
Final answer: {final}"""
        samples.append({"text": text})

        # Speed/time variant
        speed = np.random.randint(10, 50)
        time = np.random.randint(2, 6)
        distance = speed * time

        text = f"""Question: A car travels at {speed} mph for {time} hours. How far does it go?
Let me solve step by step.
Step 1: {speed} * {time} = {distance}
Final answer: {distance}"""
        samples.append({"text": text})

    # Shuffle and balance
    np.random.shuffle(samples)
    return samples[:n_samples]


def evaluate_comprehensive(model, tokenizer, test_suite: dict, max_tokens: int = 100) -> dict:
    """Evaluate across all curriculum tiers."""
    import mlx.core as mx
    import re

    results = {}

    for tier_name, problems in test_suite.items():
        correct = 0
        details = []

        for prompt, expected in problems:
            # Determine evaluation style based on tier
            if "GSM8K" in tier_name:
                # CoT generation for GSM8K
                full_prompt = f"""Question: {prompt}
Let me solve step by step.
Step 1:"""
                gen_tokens = max_tokens
            else:
                # Direct for arithmetic
                full_prompt = prompt
                gen_tokens = 10

            tokens = tokenizer.encode(full_prompt)
            generated = []

            for _ in range(gen_tokens):
                logits = model(mx.array([tokens + generated]))
                mx.eval(logits)

                logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                probs = np.exp(logits_np - logits_np.max())
                probs = probs / probs.sum()

                next_tok = int(np.argmax(probs))
                generated.append(next_tok)

                decoded = tokenizer.decode(generated)
                if "Final answer:" in decoded:
                    # Get a few more tokens for the answer
                    for _ in range(10):
                        logits = model(mx.array([tokens + generated]))
                        mx.eval(logits)
                        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                        probs = np.exp(logits_np - logits_np.max())
                        probs = probs / probs.sum()
                        next_tok = int(np.argmax(probs))
                        generated.append(next_tok)
                    break

                if "<|im_end|>" in decoded or "\n\n" in decoded:
                    break

            output = tokenizer.decode(generated).strip()
            output = output.replace("<|im_end|>", "").replace("!", "")

            # Extract answer
            if "Final answer:" in output:
                final_part = output.split("Final answer:")[-1]
                numbers = re.findall(r'-?\d+', final_part)
                predicted = numbers[0] if numbers else ""
            else:
                numbers = re.findall(r'-?\d+', output)
                predicted = numbers[0] if numbers else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt[:50],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

        accuracy = correct / len(problems) if problems else 0
        results[tier_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "details": details,
        }

    return results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    # Start from BEST arithmetic adapter (100% accuracy)
    prev_adapter = "data/adapters/qwen3_math_lora"
    train_data_dir = "data/training/qwen3_gsm8k_v2"
    new_adapter_path = "data/adapters/qwen3_gsm8k_v2_lora"

    logger.info("=" * 70)
    logger.info("GSM8K TRAINING V2 - BALANCED CURRICULUM")
    logger.info("=" * 70)
    logger.info("Starting from: qwen3_math_lora (100% arithmetic)")
    logger.info("Training mix: 25% arithmetic + 25% multi-step + 50% GSM8K CoT")

    loader = BenchmarkLoader()

    # Comprehensive test suite
    gsm_test = loader.load("gsm8k", split="test", limit=20)

    test_suite = {
        "Tier1_Arithmetic": [
            ("2+2=", "4"), ("3+5=", "8"), ("9-4=", "5"), ("7+8=", "15"),
            ("12-7=", "5"), ("6*4=", "24"), ("8*3=", "24"), ("5*9=", "45"),
        ],
        "Tier2_MultiStep": [
            ("5+3=8, 8+2=", "10"),
            ("7+4=11, 11-3=", "8"),
            ("4+6=10, 10+5=", "15"),
            ("Start with 6. Add 4. Result:", "10"),
        ],
        "Tier3_GSM8K": [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:10]],
    }

    # Phase 1: Baseline
    logger.info("\n=== PHASE 1: BASELINE ===")

    model, tokenizer = load(model_path, adapter_path=prev_adapter)
    baseline = evaluate_comprehensive(model, tokenizer, test_suite)

    logger.info("\nBaseline (arithmetic adapter):")
    for tier_name, data in baseline.items():
        logger.info(f"  {tier_name}: {data['accuracy']:.0%} ({data['correct']}/{data['total']})")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE BALANCED TRAINING DATA ===")

    samples = generate_balanced_training_data(800)

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

    # Count sample types
    arith_count = sum(1 for s in samples if "+" in s["text"] and "Step" not in s["text"] and "Question" not in s["text"])
    multi_count = sum(1 for s in samples if "," in s["text"] and "Step" not in s["text"])
    cot_count = sum(1 for s in samples if "Let me solve" in s["text"])

    logger.info(f"\nSample distribution:")
    logger.info(f"  Arithmetic: ~{arith_count}")
    logger.info(f"  Multi-step: ~{multi_count}")
    logger.info(f"  GSM8K CoT:  ~{cot_count}")

    # Phase 3: Train
    logger.info("\n=== PHASE 3: TRAINING ===")

    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "1",
        "--num-layers", "12",
        "--iters", "800",  # More iterations for longer sequences
        "--learning-rate", "1.5e-5",  # Slightly lower LR for stability
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training LoRA adapter (800 iterations, 12 layers)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=4800)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Comprehensive evaluation
    logger.info("\n=== PHASE 4: COMPREHENSIVE EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)
    trained = evaluate_comprehensive(model, tokenizer, test_suite)

    # Results comparison
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*60}")

    for tier_name in test_suite.keys():
        base = baseline[tier_name]["accuracy"]
        train = trained[tier_name]["accuracy"]
        delta = train - base
        status = "IMPROVED" if delta > 0 else ("PRESERVED" if delta == 0 else "REGRESSED")

        logger.info(f"\n{tier_name}:")
        logger.info(f"  Baseline: {base:.0%}")
        logger.info(f"  Trained:  {train:.0%}")
        logger.info(f"  Change:   {delta:+.0%} ({status})")

        # Show details for errors
        for d in trained[tier_name]["details"]:
            if not d["correct"]:
                logger.info(f"    XX: '{d['prompt'][:30]}...' -> '{d['predicted']}' (expected '{d['expected']}')")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CURRICULUM PROGRESS")
    logger.info("=" * 70)

    t1 = trained["Tier1_Arithmetic"]["accuracy"]
    t2 = trained["Tier2_MultiStep"]["accuracy"]
    t3 = trained["Tier3_GSM8K"]["accuracy"]

    t1_preserved = t1 >= baseline["Tier1_Arithmetic"]["accuracy"] * 0.9
    t2_preserved = t2 >= baseline["Tier2_MultiStep"]["accuracy"] * 0.9
    t3_improved = t3 > baseline["Tier3_GSM8K"]["accuracy"]

    logger.info(f"""
Tier 1 - Basic Arithmetic:
  Status: {'PRESERVED' if t1_preserved else 'REGRESSED'} ({t1:.0%})

Tier 2 - Multi-step (2-3):
  Status: {'PRESERVED' if t2_preserved else 'REGRESSED'} ({t2:.0%})

Tier 3 - GSM8K Word Problems:
  Before: {baseline['Tier3_GSM8K']['accuracy']:.0%}
  After:  {t3:.0%}
  Status: {'IMPROVED' if t3_improved else 'No improvement'}

Overall: {'SUCCESS' if t1_preserved and t2_preserved and t3_improved else 'NEEDS WORK'}
  - Foundation preserved: {t1_preserved and t2_preserved}
  - GSM8K improved: {t3_improved}
""")

    # Save results
    results = {
        "baseline": {k: v["accuracy"] for k, v in baseline.items()},
        "trained": {k: v["accuracy"] for k, v in trained.items()},
        "details": {k: v["details"] for k, v in trained.items()},
    }

    output_path = Path("data/experiments/qwen3_gsm8k_v2_training.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
