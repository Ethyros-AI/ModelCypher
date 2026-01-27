#!/usr/bin/env python3
"""Train on FULL GSM8K training set with longer training.

Strategy: We're at 75% with 2000 samples and 3000 iterations.
Let's try the FULL 7473 samples with more iterations.

Real data > synthetic data for generalization.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from mlx_lm import load
    import mlx.core as mx
    import re

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    train_data_dir = "data/training/qwen3_gsm8k_full_heavy"
    new_adapter_path = "data/adapters/qwen3_gsm8k_full_heavy_lora"

    logger.info("=" * 70)
    logger.info("FULL GSM8K TRAINING - ALL 7473 SAMPLES")
    logger.info("=" * 70)

    loader = BenchmarkLoader()

    # Load FULL GSM8K training set - all 7473 samples
    logger.info("Loading FULL GSM8K training set...")
    gsm_train = loader.load("gsm8k", split="train", limit=8000)  # Get all

    gsm_samples = []
    for sample in gsm_train.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        full_answer = sample.metadata.get("full_answer", sample.answer)
        gsm_samples.append({"text": f"Question: {question}\n\nAnswer: {full_answer}"})

    logger.info(f"Loaded {len(gsm_samples)} GSM8K samples (full train set)")

    # Keep arithmetic foundation at ~10%
    arith_samples = []
    for a in range(1, 20):
        for b in range(1, 20):
            arith_samples.append({"text": f"{a}+{b}={a+b}"})
            if a >= b:
                arith_samples.append({"text": f"{a}-{b}={a-b}"})
            if a <= 12 and b <= 12:
                arith_samples.append({"text": f"{a}*{b}={a*b}"})

    np.random.shuffle(arith_samples)
    arith_samples = arith_samples[:800]  # 10% of GSM8K size

    logger.info(f"Arithmetic samples: {len(arith_samples)}")

    # Combine
    all_samples = gsm_samples + arith_samples
    np.random.shuffle(all_samples)

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"  GSM8K: {len(gsm_samples)} ({len(gsm_samples)/len(all_samples)*100:.0f}%)")
    logger.info(f"  Arithmetic: {len(arith_samples)} ({len(arith_samples)/len(all_samples)*100:.0f}%)")

    # Save
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    n_train = int(len(all_samples) * 0.90)
    n_valid = int(len(all_samples) * 0.05)

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

    # Train with MORE iterations for full dataset
    logger.info("\n=== TRAINING ===")
    Path(new_adapter_path).mkdir(parents=True, exist_ok=True)

    # More iterations for larger dataset
    # Rule of thumb: ~1 epoch = dataset_size / batch_size iterations
    # 7500 samples / 1 batch = 7500 iters for 1 epoch
    # Let's do ~0.5 epochs = 4000 iterations
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", train_data_dir,
        "--adapter-path", new_adapter_path,
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", "5000",  # More iterations for larger dataset
        "--learning-rate", "1e-5",  # Lower LR for stability
        "--seed", "42",
        "--steps-per-report", "500",
    ]

    logger.info("Training (5000 iterations, full dataset)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)  # 10 hours max
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-12:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Full evaluation
    logger.info("\n=== COMPREHENSIVE EVALUATION ===")

    model, tokenizer = load(model_path, adapter_path=new_adapter_path)

    # Test suites
    test_suites = {
        "Arithmetic": [
            ("2+2=", "4"), ("7+8=", "15"), ("15-7=", "8"), ("6*9=", "54"),
            ("12+19=", "31"), ("24-15=", "9"), ("8*7=", "56"), ("11+11=", "22"),
        ],
        "MultiStep": [
            ("5+3=8, 8+2=", "10"),
            ("6*4=24, 24-10=", "14"),
            ("9+7=16, 16+5=", "21"),
        ],
    }

    # GSM8K test
    gsm_test = loader.load("gsm8k", split="test", limit=30)
    test_suites["GSM8K"] = [(s.prompt.replace("Answer:", "").strip(), s.answer) for s in gsm_test.samples[:20]]

    results = {}
    for suite_name, problems in test_suites.items():
        correct = 0
        details = []

        for question, expected in problems:
            if suite_name == "GSM8K":
                prompt = f"Question: {question}\n\nAnswer:"
                max_tokens = 300
            else:
                prompt = question
                max_tokens = 20

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
                if "####" in decoded:
                    for _ in range(15):
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

            output = tokenizer.decode(generated).strip().replace("<|im_end|>", "")

            # Extract answer
            if suite_name == "GSM8K":
                if "####" in output:
                    answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
                    numbers = re.findall(r'-?\d+', answer_part)
                    predicted = numbers[0] if numbers else ""
                else:
                    numbers = re.findall(r'-?\d+', output.replace(",", ""))
                    predicted = numbers[-1] if numbers else ""
            else:
                numbers = re.findall(r'-?\d+', output)
                predicted = numbers[0] if numbers else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            details.append({
                "question": question[:40],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

        accuracy = correct / len(problems)
        results[suite_name] = {"accuracy": accuracy, "correct": correct, "total": len(problems)}

        logger.info(f"\n{suite_name}: {accuracy:.0%} ({correct}/{len(problems)})")
        for d in details:
            mark = "OK" if d["correct"] else "XX"
            logger.info(f"  {mark}: '{d['question'][:35]}...' -> '{d['predicted']}' (expected '{d['expected']}')")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FULL HEAVY TRAINING RESULTS")
    logger.info("=" * 70)

    arith_acc = results["Arithmetic"]["accuracy"]
    multi_acc = results["MultiStep"]["accuracy"]
    gsm_acc = results["GSM8K"]["accuracy"]

    logger.info(f"""
Arithmetic:  {arith_acc:.0%} (target: 100%)
MultiStep:   {multi_acc:.0%} (target: 100%)
GSM8K:       {gsm_acc:.0%} (target: 100%)

FULL MASTERY: {arith_acc == 1.0 and multi_acc == 1.0 and gsm_acc == 1.0}
""")

    # Save results
    output = {
        "results": {k: v["accuracy"] for k, v in results.items()},
        "adapter": new_adapter_path,
        "training": {
            "gsm8k_samples": len(gsm_samples),
            "arithmetic_samples": len(arith_samples),
            "iterations": 5000,
        },
    }
    output_path = Path("data/experiments/qwen3_gsm8k_full_heavy.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
