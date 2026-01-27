#!/usr/bin/env python3
"""Compare all adapters on multiple benchmarks."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_gsm8k(model, tokenizer, limit=30):
    """Evaluate on GSM8K."""
    from mlx_lm import generate
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    loader = BenchmarkLoader()
    gsm = loader.load("gsm8k", split="test", limit=limit)

    correct = 0
    for sample in gsm.samples:
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        prompt = f"Question: {question}\n\nAnswer:"
        output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)

        if "####" in output:
            answer_part = output.split("####")[-1].replace(",", "").replace("$", "").strip()
            nums = re.findall(r'-?\d+\.?\d*', answer_part)
            if nums:
                try:
                    num_val = float(nums[0])
                    predicted = str(int(num_val)) if num_val == int(num_val) else nums[0]
                except ValueError:
                    predicted = nums[0]
            else:
                predicted = ""
        else:
            nums = re.findall(r'-?\d+', output.replace(",", ""))
            predicted = nums[-1] if nums else ""

        if predicted == expected:
            correct += 1

    return correct, len(gsm.samples)


def evaluate_arc_challenge(model, tokenizer, limit=30):
    """Evaluate on ARC-Challenge."""
    from mlx_lm import generate
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    loader = BenchmarkLoader()
    arc = loader.load("arc_challenge", split="test", limit=limit)

    correct = 0
    for sample in arc.samples:
        prompt = sample.prompt
        expected = sample.answer.strip().lower()

        output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
        output_lower = output.strip().lower()

        # Check if expected answer is in output
        if expected in output_lower:
            correct += 1
        # Also check choice letter if present
        elif sample.metadata.get("answer_key"):
            key = sample.metadata["answer_key"].lower()
            if f"{key}." in output_lower or f"answer: {key}" in output_lower or output_lower.startswith(key):
                correct += 1

    return correct, len(arc.samples)


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("UNIVERSAL ADAPTER COMPARISON")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    configs = [
        ("Base model", None),
        ("Unified (math)", "data/adapters/unified_expansion_lora"),
        ("Universal (multi-domain)", "data/adapters/universal_reasoning_lora"),
    ]

    results = {"timestamp": datetime.now().isoformat(), "benchmarks": {}}

    for bench_name, eval_func, limit in [
        ("GSM8K", evaluate_gsm8k, 30),
        ("ARC-Challenge", evaluate_arc_challenge, 30),
    ]:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating {bench_name}...")
        logger.info(f"{'=' * 50}")

        results["benchmarks"][bench_name] = {}

        for config_name, adapter_path in configs:
            logger.info(f"\n{config_name}:")

            if adapter_path:
                model, tokenizer = load(model_path, adapter_path=adapter_path)
            else:
                model, tokenizer = load(model_path)

            correct, total = eval_func(model, tokenizer, limit)
            accuracy = correct / total * 100

            logger.info(f"  Accuracy: {correct}/{total} ({accuracy:.0f}%)")

            results["benchmarks"][bench_name][config_name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

            del model

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n{'Benchmark':<20} {'Base':<12} {'Unified':<12} {'Universal':<12}")
    logger.info("-" * 56)

    for bench_name in results["benchmarks"]:
        bench_data = results["benchmarks"][bench_name]
        base = bench_data.get("Base model", {}).get("accuracy", 0)
        unified = bench_data.get("Unified (math)", {}).get("accuracy", 0)
        universal = bench_data.get("Universal (multi-domain)", {}).get("accuracy", 0)
        logger.info(f"{bench_name:<20} {base:>5.0f}%       {unified:>5.0f}%       {universal:>5.0f}%")

    # Improvement summary
    logger.info(f"\nImprovement vs Base:")
    for bench_name in results["benchmarks"]:
        bench_data = results["benchmarks"][bench_name]
        base = bench_data.get("Base model", {}).get("accuracy", 0)
        unified = bench_data.get("Unified (math)", {}).get("accuracy", 0)
        universal = bench_data.get("Universal (multi-domain)", {}).get("accuracy", 0)
        logger.info(f"  {bench_name}: Unified {unified - base:+.0f}%, Universal {universal - base:+.0f}%")

    # Save
    output_path = Path("data/experiments/universal_adapter_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
