#!/usr/bin/env python3
"""Train GSM8K with Explicit Chain-of-Thought Format.

Key insight: The model CAN do step-by-step reasoning when prompted,
but doesn't do it autonomously. This trains explicit CoT format:

"Question: ...
Let me solve step by step.
Step 1: [calculation] = [result]
Step 2: [calculation] = [result]
Final answer: [number]"

This teaches the model to GENERATE the reasoning steps itself.
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


def generate_cot_training_data(n_samples: int = 400, seed: int = 42) -> List[dict]:
    """Generate explicit chain-of-thought training data.

    Format:
    "Question: I have 5 apples. I get 3 more. I eat 2. How many left?
    Let me solve step by step.
    Step 1: 5 + 3 = 8
    Step 2: 8 - 2 = 6
    Final answer: 6"
    """
    np.random.seed(seed)
    samples = []

    # === 2-STEP PROBLEMS ===
    templates_2step = [
        # Addition then subtraction
        lambda: {
            "a": np.random.randint(5, 15),
            "b": np.random.randint(2, 8),
            "c": np.random.randint(1, 5),
            "q": "I have {a} apples. I get {b} more. I eat {c}. How many left?",
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
            "calc": lambda a, b, c: (a + b, a + b - c),
        },
        # Two additions
        lambda: {
            "a": np.random.randint(3, 10),
            "b": np.random.randint(2, 8),
            "c": np.random.randint(1, 6),
            "q": "Start with {a}. Add {b}. Then add {c}. What is the total?",
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} + {c} = {s2}"],
            "calc": lambda a, b, c: (a + b, a + b + c),
        },
        # Two subtractions
        lambda: {
            "a": np.random.randint(12, 20),
            "b": np.random.randint(2, 6),
            "c": np.random.randint(1, 5),
            "q": "{a} birds in a tree. {b} fly away. Then {c} more fly away. How many remain?",
            "steps": ["Step 1: {a} - {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
            "calc": lambda a, b, c: (a - b, a - b - c),
        },
    ]

    for _ in range(n_samples // 3):
        template = np.random.choice(templates_2step)()
        a, b, c = template["a"], template["b"], template["c"]
        s1, s2 = template["calc"](a, b, c)

        if s1 >= 0 and s2 >= 0:
            question = template["q"].format(a=a, b=b, c=c)
            steps = [s.format(a=a, b=b, c=c, s1=s1, s2=s2) for s in template["steps"]]

            text = f"""Question: {question}
Let me solve step by step.
{steps[0]}
{steps[1]}
Final answer: {s2}"""
            samples.append({"text": text})

    # === 3-STEP PROBLEMS ===
    templates_3step = [
        # Three operations
        lambda: {
            "a": np.random.randint(5, 12),
            "b": np.random.randint(2, 6),
            "c": np.random.randint(2, 5),
            "d": np.random.randint(1, 4),
            "q": "I have {a} toys. I get {b} more. I give away {c}. I find {d} more. How many now?",
            "steps": ["Step 1: {a} + {b} = {s1}", "Step 2: {s1} - {c} = {s2}", "Step 3: {s2} + {d} = {s3}"],
            "calc": lambda a, b, c, d: (a + b, a + b - c, a + b - c + d),
        },
        # With multiplication
        lambda: {
            "a": np.random.randint(2, 6),
            "b": np.random.randint(2, 5),
            "c": np.random.randint(1, 4),
            "d": 0,
            "q": "Each box has {a} apples. I have {b} boxes. I eat {c} apples. How many left?",
            "steps": ["Step 1: {a} * {b} = {s1}", "Step 2: {s1} - {c} = {s2}"],
            "calc": lambda a, b, c, d: (a * b, a * b - c, 0),
        },
    ]

    for _ in range(n_samples // 4):
        template = np.random.choice(templates_3step)()
        a, b, c, d = template["a"], template["b"], template["c"], template["d"]
        results = template["calc"](a, b, c, d)

        if all(r >= 0 for r in results if r != 0):
            s1, s2, s3 = results
            question = template["q"].format(a=a, b=b, c=c, d=d)
            steps = [s.format(a=a, b=b, c=c, d=d, s1=s1, s2=s2, s3=s3) for s in template["steps"]]

            final = s3 if s3 != 0 else s2
            text = f"""Question: {question}
Let me solve step by step.
{chr(10).join(steps)}
Final answer: {final}"""
            samples.append({"text": text})

    # === GSM8K-STYLE PROBLEMS ===
    gsm_templates = [
        # Money problems
        lambda: {
            "price": np.random.randint(2, 10),
            "qty": np.random.randint(2, 6),
            "discount": np.random.randint(1, 5),
            "text": lambda p, q, d: f"""Question: An apple costs ${p}. I buy {q} apples. I have a ${d} coupon. How much do I pay?
Let me solve step by step.
Step 1: {p} * {q} = {p*q}
Step 2: {p*q} - {d} = {p*q-d}
Final answer: {p*q-d}"""
        },
        # Speed/distance
        lambda: {
            "speed": np.random.randint(10, 30),
            "time": np.random.randint(2, 5),
            "text": lambda s, t, _: f"""Question: A car travels at {s} mph for {t} hours. How far does it go?
Let me solve step by step.
Step 1: {s} * {t} = {s*t}
Final answer: {s*t}"""
        },
        # Division problems
        lambda: {
            "total": np.random.randint(10, 30) * 2,  # Ensure even
            "groups": np.random.randint(2, 6),
            "text": lambda t, g, _: f"""Question: I have {t} cookies to share equally among {g} friends. How many does each get?
Let me solve step by step.
Step 1: {t} / {g} = {t//g}
Final answer: {t//g}""" if t % g == 0 else None
        },
    ]

    for _ in range(n_samples // 3):
        template = np.random.choice(gsm_templates)()
        values = [template.get("price", 0) or template.get("speed", 0) or template.get("total", 0),
                  template.get("qty", 0) or template.get("time", 0) or template.get("groups", 0),
                  template.get("discount", 0)]
        text = template["text"](*values)
        if text:
            samples.append({"text": text})

    # === CUMULATIVE: Basic arithmetic to prevent regression ===
    for a in range(1, 12):
        for b in range(1, 12):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    # Add multi-step chains
    for _ in range(100):
        a = np.random.randint(2, 10)
        b = np.random.randint(1, 8)
        c = np.random.randint(1, 6)
        s1 = a + b
        s2 = s1 + c
        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}"})

    np.random.shuffle(samples)
    return samples[:n_samples]


def evaluate_gsm8k_cot(model, tokenizer, problems: List[Tuple[str, str]], max_tokens: int = 100) -> Tuple[float, List[dict]]:
    """Evaluate with CoT generation."""
    import mlx.core as mx
    import re

    results = []
    correct = 0

    for question, expected in problems:
        # Prompt for CoT
        prompt = f"""Question: {question}
Let me solve step by step.
Step 1:"""

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

            decoded = tokenizer.decode(generated)
            if "Final answer:" in decoded and len(generated) > 20:
                # Generate a few more tokens for the answer
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

        # Extract final answer
        if "Final answer:" in output:
            final_part = output.split("Final answer:")[-1]
            numbers = re.findall(r'-?\d+', final_part)
            predicted = numbers[0] if numbers else ""
        else:
            numbers = re.findall(r'-?\d+', output)
            predicted = numbers[-1] if numbers else ""

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "question": question[:50],
            "expected": expected,
            "predicted": predicted,
            "reasoning": output[:200],
            "correct": is_correct,
        })

    return correct / len(problems) if problems else 0, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    prev_adapter = "data/adapters/qwen3_multistep_lora"
    train_data_dir = "data/training/qwen3_gsm8k_cot"
    new_adapter_path = "data/adapters/qwen3_gsm8k_cot_lora"

    logger.info("=" * 70)
    logger.info("TRAINING GSM8K WITH CHAIN-OF-THOUGHT FORMAT")
    logger.info("=" * 70)
    logger.info("Teaching model to generate step-by-step reasoning autonomously")

    loader = BenchmarkLoader()

    # Load GSM8K test
    gsm_test = loader.load("gsm8k", split="test", limit=20)
    test_questions = [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples]

    # Phase 1: Baseline
    logger.info("\n=== PHASE 1: BASELINE (multistep adapter) ===")

    model, tokenizer = load(model_path, adapter_path=prev_adapter)
    baseline_acc, baseline_details = evaluate_gsm8k_cot(model, tokenizer, test_questions[:10])

    logger.info(f"Baseline CoT accuracy: {baseline_acc:.0%}")
    for r in baseline_details[:3]:
        mark = "OK" if r["correct"] else "XX"
        logger.info(f"  {mark} '{r['question'][:40]}...' -> '{r['predicted']}' (expected '{r['expected']}')")
        logger.info(f"      Reasoning: {r['reasoning'][:80]}...")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE COT TRAINING DATA ===")

    samples = generate_cot_training_data(500)

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

    logger.info("\nSample CoT training data:")
    cot_samples = [s for s in samples if "Let me solve" in s["text"]]
    for s in cot_samples[:2]:
        logger.info(f"\n{s['text'][:200]}...")

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
        "--iters", "600",
        "--learning-rate", "2e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    logger.info("Training LoRA adapter (600 iterations)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
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
    trained_acc, trained_details = evaluate_gsm8k_cot(model, tokenizer, test_questions[:10])

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Baseline CoT:  {baseline_acc:.0%}")
    logger.info(f"  Trained CoT:   {trained_acc:.0%}")
    logger.info(f"  Change:        {trained_acc - baseline_acc:+.0%}")

    logger.info("\nTrained examples:")
    for r in trained_details[:5]:
        mark = "OK" if r["correct"] else "XX"
        logger.info(f"  {mark} '{r['question'][:40]}...' -> '{r['predicted']}' (expected '{r['expected']}')")
        logger.info(f"      Reasoning: {r['reasoning'][:100]}...")

    # Regression check
    logger.info("\n=== REGRESSION CHECK ===")
    arith_tests = [("2+2=", "4"), ("5+5=", "10"), ("7-3=", "4")]
    arith_correct = 0
    for prompt, expected in arith_tests:
        tokens = tokenizer.encode(prompt)
        logits = model(mx.array([tokens]))
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        pred = tokenizer.decode([int(np.argmax(probs))]).strip()
        is_correct = expected in pred
        if is_correct:
            arith_correct += 1
        mark = "OK" if is_correct else "XX"
        logger.info(f"  {mark} {prompt} -> '{pred}'")

    arith_acc = arith_correct / len(arith_tests)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    improved = trained_acc > baseline_acc
    preserved = arith_acc >= 0.66

    logger.info(f"""
GSM8K with Chain-of-Thought:
  Before: {baseline_acc:.0%}
  After:  {trained_acc:.0%}
  Status: {'IMPROVED' if improved else 'No improvement'}

Arithmetic:
  Status: {'PRESERVED' if preserved else 'REGRESSED'} ({arith_acc:.0%})

Curriculum Ladder:
  1. Basic Arithmetic: 100%
  2. Multi-step (2-3): 100%
  3. GSM8K with CoT: {trained_acc:.0%} {'<-- YOU ARE HERE' if trained_acc > 0.2 else ''}
""")

    # Save results
    results = {
        "baseline_cot": baseline_acc,
        "trained_cot": trained_acc,
        "arithmetic_preserved": arith_acc,
        "details": trained_details,
    }

    output_path = Path("data/experiments/qwen3_gsm8k_cot_training.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
