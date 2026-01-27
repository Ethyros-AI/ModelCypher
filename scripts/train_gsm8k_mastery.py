#!/usr/bin/env python3
"""Train GSM8K to Mastery - 70%+ accuracy required.

Strategy:
1. More GSM8K data (500 samples)
2. Heavy arithmetic preservation (60% of data)
3. Longer training (1500 iterations)
4. Better number extraction (handles commas)
5. Stricter evaluation
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


def load_gsm8k_training(loader: BenchmarkLoader, n_samples: int = 500) -> List[dict]:
    """Load GSM8K training data with full chain-of-thought."""
    benchmark = loader.load("gsm8k", split="train", limit=n_samples)

    samples = []
    for sample in benchmark.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)

        text = f"Question: {question}\n\nAnswer: {full_answer}"
        samples.append({"text": text})

    return samples


def generate_arithmetic_foundation(n_samples: int = 600, seed: int = 42) -> List[dict]:
    """Generate comprehensive arithmetic to prevent regression."""
    np.random.seed(seed)
    samples = []

    # Basic operations - exhaustive coverage
    for a in range(1, 20):
        for b in range(1, 20):
            samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                samples.append({"text": f"{a}-{b}={a-b}"})

    for a in range(1, 13):
        for b in range(1, 13):
            samples.append({"text": f"{a}*{b}={a*b}"})

    # Word forms
    for _ in range(150):
        a = np.random.randint(1, 30)
        b = np.random.randint(1, 20)
        samples.append({"text": f"{a} plus {b} equals {a+b}"})
        samples.append({"text": f"{a} + {b} = {a+b}"})
        if a >= b:
            samples.append({"text": f"{a} minus {b} equals {a-b}"})
            samples.append({"text": f"{a} - {b} = {a-b}"})
        samples.append({"text": f"{a} times {b} equals {a*b}"})
        samples.append({"text": f"{a} * {b} = {a*b}"})

    # Multi-step chains (critical for GSM8K reasoning)
    for _ in range(200):
        a = np.random.randint(3, 20)
        b = np.random.randint(1, 15)
        c = np.random.randint(1, 10)
        s1 = a + b
        s2 = s1 + c

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}"})

        if s1 >= c:
            samples.append({"text": f"{a}+{b}={s1}, {s1}-{c}={s1-c}"})

        # Multiplication chains
        if a <= 10 and b <= 10:
            p1 = a * b
            samples.append({"text": f"{a}*{b}={p1}, {p1}+{c}={p1+c}"})

    # 3-step chains
    for _ in range(100):
        a = np.random.randint(5, 15)
        b = np.random.randint(2, 10)
        c = np.random.randint(1, 8)
        d = np.random.randint(1, 6)

        s1 = a + b
        s2 = s1 + c
        s3 = s2 + d

        samples.append({"text": f"{a}+{b}={s1}, {s1}+{c}={s2}, {s2}+{d}={s3}"})

    np.random.shuffle(samples)
    return samples[:n_samples]


def generate_bridge_problems(n_samples: int = 200, seed: int = 42) -> List[dict]:
    """Generate problems that bridge arithmetic to word problems."""
    np.random.seed(seed)
    samples = []

    templates = [
        # Simple word problems with GSM8K format
        {
            "q": "I have {a} apples. I get {b} more. How many do I have?",
            "a": "{a} + {b} = <<{a}+{b}={s1}>>{s1}. #### {s1}",
            "calc": lambda a, b, c: (a + b,),
        },
        {
            "q": "There are {a} birds. {b} fly away. How many remain?",
            "a": "{a} - {b} = <<{a}-{b}={s1}>>{s1}. #### {s1}",
            "calc": lambda a, b, c: (a - b,) if a >= b else None,
        },
        {
            "q": "Each box has {a} items. I have {b} boxes. How many items total?",
            "a": "{a} * {b} = <<{a}*{b}={s1}>>{s1}. #### {s1}",
            "calc": lambda a, b, c: (a * b,),
        },
        # 2-step problems
        {
            "q": "I have {a} coins. I find {b} more. I spend {c}. How many left?",
            "a": "{a} + {b} = <<{a}+{b}={s1}>>{s1}. {s1} - {c} = <<{s1}-{c}={s2}>>{s2}. #### {s2}",
            "calc": lambda a, b, c: (a + b, a + b - c) if a + b >= c else None,
        },
        {
            "q": "A store has {a} items. They sell {b}. They get {c} more. How many now?",
            "a": "{a} - {b} = <<{a}-{b}={s1}>>{s1}. {s1} + {c} = <<{s1}+{c}={s2}>>{s2}. #### {s2}",
            "calc": lambda a, b, c: (a - b, a - b + c) if a >= b else None,
        },
        # Money problems
        {
            "q": "Each pen costs ${a}. I buy {b} pens. How much do I spend?",
            "a": "{a} * {b} = <<{a}*{b}={s1}>>{s1}. #### {s1}",
            "calc": lambda a, b, c: (a * b,),
        },
        {
            "q": "I have ${a}. I spend ${b} on food and ${c} on drinks. How much left?",
            "a": "{b} + {c} = <<{b}+{c}={s1}>>{s1}. {a} - {s1} = <<{a}-{s1}={s2}>>{s2}. #### {s2}",
            "calc": lambda a, b, c: (b + c, a - b - c) if a >= b + c else None,
        },
    ]

    for _ in range(n_samples):
        template = np.random.choice(templates)
        a = np.random.randint(5, 30)
        b = np.random.randint(2, 15)
        c = np.random.randint(1, 10)

        result = template["calc"](a, b, c)
        if result is None:
            continue

        if len(result) == 1:
            s1 = result[0]
            s2 = 0
        else:
            s1, s2 = result

        if s1 < 0 or s2 < 0:
            continue

        question = template["q"].format(a=a, b=b, c=c)
        answer = template["a"].format(a=a, b=b, c=c, s1=s1, s2=s2)

        text = f"Question: {question}\n\nAnswer: {answer}"
        samples.append({"text": text})

    return samples


def evaluate_with_strict_extraction(model, tokenizer, test_suite: dict, max_tokens: int = 200) -> dict:
    """Evaluate with improved number extraction for large numbers."""
    import mlx.core as mx
    import re

    results = {}

    for tier_name, problems in test_suite.items():
        correct = 0
        details = []

        for prompt, expected in problems:
            if "GSM8K" in tier_name:
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

                if "####" in decoded:
                    for _ in range(15):  # More tokens for full answer
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

            # Improved number extraction
            if "####" in output:
                answer_part = output.split("####")[-1].strip()
                # Handle numbers with commas: $70,000 -> 70000
                answer_part = answer_part.replace(",", "").replace("$", "")
                numbers = re.findall(r'-?\d+', answer_part)
                predicted = numbers[0] if numbers else ""
            else:
                numbers = re.findall(r'-?\d+', output.replace(",", ""))
                predicted = numbers[-1] if numbers else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt[:50],
                "expected": expected,
                "predicted": predicted,
                "output": output[:200],
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
    prev_adapter = "data/adapters/qwen3_math_lora"  # Best arithmetic foundation
    train_data_dir = "data/training/qwen3_gsm8k_mastery"
    new_adapter_path = "data/adapters/qwen3_gsm8k_mastery_lora"

    logger.info("=" * 70)
    logger.info("GSM8K MASTERY TRAINING")
    logger.info("=" * 70)
    logger.info("Target: 70%+ GSM8K with 90%+ arithmetic preserved")

    loader = BenchmarkLoader()

    # Test suite - more GSM8K samples for accurate measurement
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    test_suite = {
        "Tier1_Arithmetic": [
            ("2+2=", "4"), ("3+5=", "8"), ("9-4=", "5"), ("7+8=", "15"),
            ("12-7=", "5"), ("6*4=", "24"), ("8*3=", "24"), ("5*9=", "45"),
            ("15+6=", "21"), ("18-9=", "9"),
        ],
        "Tier2_MultiStep": [
            ("5+3=8, 8+2=", "10"),
            ("7+4=11, 11-3=", "8"),
            ("4+6=10, 10+5=", "15"),
            ("3*4=12, 12+5=", "17"),
            ("6+8=14, 14-6=", "8"),
        ],
        "Tier3_GSM8K": [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:20]],
    }

    # Phase 1: Baseline
    logger.info("\n=== PHASE 1: BASELINE ===")

    model, tokenizer = load(model_path, adapter_path=prev_adapter)
    baseline = evaluate_with_strict_extraction(model, tokenizer, test_suite)

    logger.info("\nBaseline (arithmetic adapter):")
    for tier_name, data in baseline.items():
        logger.info(f"  {tier_name}: {data['accuracy']:.0%} ({data['correct']}/{data['total']})")

    del model
    mx.clear_cache()

    # Phase 2: Generate training data
    logger.info("\n=== PHASE 2: GENERATE TRAINING DATA ===")

    # Heavy arithmetic foundation (60%)
    arith_samples = generate_arithmetic_foundation(600)
    logger.info(f"Arithmetic samples: {len(arith_samples)}")

    # Bridge problems (15%)
    bridge_samples = generate_bridge_problems(150)
    logger.info(f"Bridge samples: {len(bridge_samples)}")

    # Real GSM8K (25%)
    gsm_samples = load_gsm8k_training(loader, n_samples=250)
    logger.info(f"GSM8K samples: {len(gsm_samples)}")

    # Combine
    all_samples = arith_samples + bridge_samples + gsm_samples
    np.random.shuffle(all_samples)

    logger.info(f"\nTotal: {len(all_samples)} samples")
    logger.info(f"  Arithmetic: {len(arith_samples)} ({len(arith_samples)/len(all_samples)*100:.0f}%)")
    logger.info(f"  Bridge: {len(bridge_samples)} ({len(bridge_samples)/len(all_samples)*100:.0f}%)")
    logger.info(f"  GSM8K: {len(gsm_samples)} ({len(gsm_samples)/len(all_samples)*100:.0f}%)")

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
        "--num-layers", "16",  # More layers
        "--iters", "1500",     # More iterations
        "--learning-rate", "1.5e-5",
        "--seed", "42",
        "--steps-per-report", "300",
    ]

    logger.info("Training LoRA adapter (1500 iterations, 16 layers)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=9000)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-12:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluation
    logger.info("\n=== PHASE 4: MASTERY EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)
    trained = evaluate_with_strict_extraction(model, tokenizer, test_suite)

    # Results
    logger.info(f"\n{'='*60}")
    logger.info("MASTERY CHECK")
    logger.info(f"{'='*60}")

    t1 = trained["Tier1_Arithmetic"]["accuracy"]
    t2 = trained["Tier2_MultiStep"]["accuracy"]
    t3 = trained["Tier3_GSM8K"]["accuracy"]

    logger.info(f"\nTier 1 - Arithmetic: {t1:.0%} (target: 90%+)")
    logger.info(f"Tier 2 - Multi-step: {t2:.0%} (target: 90%+)")
    logger.info(f"Tier 3 - GSM8K:      {t3:.0%} (target: 70%+)")

    # Show GSM8K details
    logger.info("\nGSM8K Details:")
    for d in trained["Tier3_GSM8K"]["details"]:
        mark = "OK" if d["correct"] else "XX"
        logger.info(f"  {mark}: '{d['prompt'][:40]}...' -> '{d['predicted']}' (expected '{d['expected']}')")

    # Mastery check
    arithmetic_mastered = t1 >= 0.9
    multistep_mastered = t2 >= 0.9
    gsm8k_mastered = t3 >= 0.7

    logger.info("\n" + "=" * 70)
    logger.info("MASTERY STATUS")
    logger.info("=" * 70)

    logger.info(f"""
Tier 1 - Arithmetic:  {'MASTERED' if arithmetic_mastered else 'NOT YET'} ({t1:.0%})
Tier 2 - Multi-step:  {'MASTERED' if multistep_mastered else 'NOT YET'} ({t2:.0%})
Tier 3 - GSM8K:       {'MASTERED' if gsm8k_mastered else 'NOT YET'} ({t3:.0%})

Foundation preserved: {arithmetic_mastered and multistep_mastered}
GSM8K target met: {gsm8k_mastered}

READY FOR TIER 4 (ARC): {arithmetic_mastered and multistep_mastered and gsm8k_mastered}
""")

    # Save results
    results = {
        "baseline": {k: v["accuracy"] for k, v in baseline.items()},
        "trained": {k: v["accuracy"] for k, v in trained.items()},
        "mastery": {
            "arithmetic": arithmetic_mastered,
            "multistep": multistep_mastered,
            "gsm8k": gsm8k_mastered,
            "ready_for_arc": arithmetic_mastered and multistep_mastered and gsm8k_mastered,
        },
        "details": {k: v["details"] for k, v in trained.items()},
    }

    output_path = Path("data/experiments/qwen3_gsm8k_mastery.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
