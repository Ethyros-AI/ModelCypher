#!/usr/bin/env python3
"""Train GSM8K using REAL GSM8K data with full chain-of-thought solutions.

Key insight: The model needs to see REAL GSM8K reasoning, not simplified templates.
GSM8K answers contain natural language chain-of-thought that builds to #### final answer.

Strategy:
1. Use actual GSM8K training data (not generated templates)
2. Include cumulative arithmetic (50% of data)
3. Format as text continuation: "Question: ... Answer: [full solution] #### [answer]"
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


def load_real_gsm8k_data(loader: BenchmarkLoader, n_samples: int = 300) -> List[dict]:
    """Load actual GSM8K training data with full solutions."""
    benchmark = loader.load("gsm8k", split="train", limit=n_samples)

    samples = []
    for sample in benchmark.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)

        # Format: Question + full solution (includes natural language CoT)
        text = f"Question: {question}\n\nAnswer: {full_answer}"
        samples.append({"text": text})

    return samples


def generate_cumulative_arithmetic(n_samples: int = 400, seed: int = 42) -> List[dict]:
    """Generate arithmetic samples to prevent regression."""
    np.random.seed(seed)
    samples = []

    # === 1-STEP: Basic arithmetic ===
    for a in range(1, 16):
        for b in range(1, 16):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})
            samples.append({"text": f"{a}*{b}={a*b}"})

    # Word forms
    for _ in range(100):
        a = np.random.randint(1, 25)
        b = np.random.randint(1, 15)
        samples.append({"text": f"{a} plus {b} equals {a+b}"})
        if a >= b:
            samples.append({"text": f"{a} minus {b} equals {a-b}"})
        samples.append({"text": f"{a} times {b} equals {a*b}"})

    # === 2-STEP: Chain operations ===
    for _ in range(150):
        a = np.random.randint(3, 15)
        b = np.random.randint(1, 10)
        c = np.random.randint(1, 8)
        s1 = a + b
        s2 = s1 + c

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}"})
        samples.append({"text": f"Start with {a}. Add {b} to get {s1}. Add {c} more. Total: {s2}"})

        if s1 >= c:
            samples.append({"text": f"{a}+{b}={s1}, {s1}-{c}={s1-c}"})

    # === 3-STEP: Longer chains ===
    for _ in range(100):
        a = np.random.randint(5, 15)
        b = np.random.randint(2, 8)
        c = np.random.randint(1, 6)
        d = np.random.randint(1, 5)

        s1 = a + b
        s2 = s1 + c
        s3 = s2 - d if s2 >= d else s2 + d

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}, {s2}+{d}={s2+d}"})

    np.random.shuffle(samples)
    return samples[:n_samples]


def generate_simple_cot(n_samples: int = 200, seed: int = 42) -> List[dict]:
    """Generate simple CoT problems to bridge to GSM8K format."""
    np.random.seed(seed)
    samples = []

    templates = [
        # 2-step addition/subtraction
        lambda: {
            "a": np.random.randint(5, 20),
            "b": np.random.randint(2, 10),
            "c": np.random.randint(1, 8),
            "q": "I have {a} apples. I get {b} more. I eat {c}. How many left?",
            "solution": "{a} + {b} = {s1}. {s1} - {c} = {s2}. #### {s2}",
            "calc": lambda a, b, c: (a + b, a + b - c),
        },
        # Money problems
        lambda: {
            "a": np.random.randint(2, 10),  # price
            "b": np.random.randint(2, 6),   # quantity
            "c": np.random.randint(5, 15),  # payment
            "q": "Each pen costs ${a}. I buy {b} pens with ${c}. What's my change?",
            "solution": "{b} pens cost {b} * {a} = ${s1}. Change is {c} - {s1} = ${s2}. #### {s2}",
            "calc": lambda a, b, c: (a * b, c - a * b),
        },
        # Distance/speed
        lambda: {
            "a": np.random.randint(10, 40),  # speed
            "b": np.random.randint(2, 5),    # time
            "c": 0,
            "q": "A car travels at {a} mph for {b} hours. How far?",
            "solution": "Distance = speed * time = {a} * {b} = {s1} miles. #### {s1}",
            "calc": lambda a, b, c: (a * b, 0),
        },
        # Sharing problems
        lambda: {
            "a": np.random.randint(12, 36),  # total
            "b": np.random.randint(2, 6),    # groups
            "c": np.random.randint(1, 4),    # extra per group
            "q": "I have {a} cookies. I divide them equally among {b} friends. Each friend gives me {c} back. How many cookies do I have?",
            "solution": "Each friend gets {a} / {b} = {s1} cookies. They give back {c} * {b} = {s2} cookies. #### {s2}",
            "calc": lambda a, b, c: (a // b, c * b),
        },
    ]

    for _ in range(n_samples):
        template = np.random.choice(templates)()
        a, b, c = template["a"], template["b"], template["c"]
        results = template["calc"](a, b, c)

        # Skip invalid results
        s1, s2 = results
        if s1 < 0 or s2 < 0:
            continue

        question = template["q"].format(a=a, b=b, c=c)
        solution = template["solution"].format(a=a, b=b, c=c, s1=s1, s2=s2 if s2 != 0 else s1)

        text = f"Question: {question}\n\nAnswer: {solution}"
        samples.append({"text": text})

    return samples


def evaluate_comprehensive(model, tokenizer, test_suite: dict, max_tokens: int = 150) -> dict:
    """Evaluate across all curriculum tiers."""
    import mlx.core as mx
    import re

    results = {}

    for tier_name, problems in test_suite.items():
        correct = 0
        details = []

        for prompt, expected in problems:
            # Determine evaluation style
            if "GSM8K" in tier_name:
                # For GSM8K, use answer-first prompt
                full_prompt = f"Question: {prompt}\n\nAnswer:"
                gen_tokens = max_tokens
            else:
                full_prompt = prompt
                gen_tokens = 15

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
                # Stop at #### (GSM8K answer delimiter) or end token
                if "####" in decoded:
                    # Generate a bit more for the final number
                    for _ in range(10):
                        logits = model(mx.array([tokens + generated]))
                        mx.eval(logits)
                        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
                        probs = np.exp(logits_np - logits_np.max())
                        probs = probs / probs.sum()
                        next_tok = int(np.argmax(probs))
                        generated.append(next_tok)
                    break

                if "<|im_end|>" in decoded:
                    break

            output = tokenizer.decode(generated).strip()
            output = output.replace("<|im_end|>", "").replace("!", "")

            # Extract answer - look for #### first
            if "####" in output:
                answer_part = output.split("####")[-1]
                numbers = re.findall(r'-?[\d,]+', answer_part)
                if numbers:
                    predicted = numbers[0].replace(",", "")
                else:
                    predicted = ""
            else:
                # Fallback: extract last number
                numbers = re.findall(r'-?\d+', output)
                predicted = numbers[-1] if numbers else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt[:50],
                "expected": expected,
                "predicted": predicted,
                "output": output[:150],
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
    # Start from best arithmetic adapter
    prev_adapter = "data/adapters/qwen3_math_lora"
    train_data_dir = "data/training/qwen3_gsm8k_full"
    new_adapter_path = "data/adapters/qwen3_gsm8k_full_lora"

    logger.info("=" * 70)
    logger.info("GSM8K TRAINING - FULL REAL DATA")
    logger.info("=" * 70)
    logger.info("Using actual GSM8K chain-of-thought solutions")

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

    # Phase 2: Prepare training data
    logger.info("\n=== PHASE 2: LOAD REAL GSM8K DATA ===")

    # Real GSM8K with full solutions
    gsm_samples = load_real_gsm8k_data(loader, n_samples=300)
    logger.info(f"Loaded {len(gsm_samples)} real GSM8K samples")

    # Show a sample
    logger.info("\nSample GSM8K training data:")
    logger.info(f"  {gsm_samples[0]['text'][:200]}...")

    # Cumulative arithmetic
    arith_samples = generate_cumulative_arithmetic(400)
    logger.info(f"Generated {len(arith_samples)} arithmetic samples")

    # Simple CoT bridge
    simple_cot = generate_simple_cot(200)
    logger.info(f"Generated {len(simple_cot)} simple CoT samples")

    # Combine: 50% arithmetic/chain, 50% GSM8K style
    all_samples = arith_samples + simple_cot + gsm_samples

    np.random.shuffle(all_samples)

    # Save training data
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

    logger.info(f"\nTotal training samples: {len(all_samples)}")
    logger.info(f"  Real GSM8K: {len(gsm_samples)} ({len(gsm_samples)/len(all_samples)*100:.0f}%)")
    logger.info(f"  Simple CoT: {len(simple_cot)} ({len(simple_cot)/len(all_samples)*100:.0f}%)")
    logger.info(f"  Arithmetic: {len(arith_samples)} ({len(arith_samples)/len(all_samples)*100:.0f}%)")

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
        "--num-layers", "14",  # More layers for complex reasoning
        "--iters", "1000",     # More iterations
        "--learning-rate", "2e-5",
        "--seed", "42",
        "--steps-per-report", "200",
    ]

    logger.info("Training LoRA adapter (1000 iterations, 14 layers)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-10:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluation
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

        # Show GSM8K reasoning
        if "GSM8K" in tier_name:
            for d in trained[tier_name]["details"][:5]:
                mark = "OK" if d["correct"] else "XX"
                logger.info(f"    {mark}: '{d['prompt'][:35]}...' -> '{d['predicted']}' (expected '{d['expected']}')")
                logger.info(f"        Reasoning: {d['output'][:80]}...")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    t1 = trained["Tier1_Arithmetic"]["accuracy"]
    t2 = trained["Tier2_MultiStep"]["accuracy"]
    t3 = trained["Tier3_GSM8K"]["accuracy"]

    logger.info(f"""
Curriculum Ladder:
  Tier 1 - Basic Arithmetic: {t1:.0%}
  Tier 2 - Multi-step Chain: {t2:.0%}
  Tier 3 - GSM8K Word Probs: {t3:.0%}

Progress:
  Foundation preserved: {t1 >= 0.9 and t2 >= 0.9}
  GSM8K improved: {t3 > baseline['Tier3_GSM8K']['accuracy']}

Target: 70%+ on GSM8K with foundation preserved
Current: {t3:.0%}
""")

    # Save results
    results = {
        "baseline": {k: v["accuracy"] for k, v in baseline.items()},
        "trained": {k: v["accuracy"] for k, v in trained.items()},
        "details": {k: v["details"] for k, v in trained.items()},
    }

    output_path = Path("data/experiments/qwen3_gsm8k_full_training.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
