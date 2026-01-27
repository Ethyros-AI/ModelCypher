#!/usr/bin/env python3
"""Evaluate GSM8K accuracy with the unified adapter."""

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


def main():
    from mlx_lm import load, generate
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    logger.info("=" * 70)
    logger.info("GSM8K EVALUATION WITH UNIFIED ADAPTER")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    # Load model with unified adapter
    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Load GSM8K
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    logger.info(f"Evaluating {len(gsm_test.samples)} problems...")

    correct = 0
    results = []

    for i, sample in enumerate(gsm_test.samples):
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        prompt = f"Question: {question}\n\nAnswer:"
        output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)

        # Extract answer - handle decimals properly (26.00 -> 26)
        if "####" in output:
            answer_part = output.split("####")[-1].replace(",", "").replace("$", "").strip()
            # Find numbers including decimals
            nums = re.findall(r'-?\d+\.?\d*', answer_part)
            if nums:
                # Convert decimal to int if it's a whole number (26.00 -> 26)
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

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "WRONG"
        logger.info(f"  [{i+1:2d}] {status}: {predicted:>6} (expected {expected})")

        results.append({
            "index": i,
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
        })

    accuracy = correct / len(gsm_test.samples) * 100

    logger.info(f"\n{'=' * 70}")
    logger.info(f"FINAL ACCURACY: {correct}/{len(gsm_test.samples)} ({accuracy:.0f}%)")
    logger.info(f"{'=' * 70}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "adapter": adapter_path,
        "correct": correct,
        "total": len(gsm_test.samples),
        "accuracy": accuracy,
        "results": results,
    }

    output_path = Path("data/experiments/gsm8k_unified_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
