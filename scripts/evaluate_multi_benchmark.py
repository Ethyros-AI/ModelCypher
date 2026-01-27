#!/usr/bin/env python3
"""Evaluate unified adapter on multiple benchmarks.

Test if the geometric approach to learning (recognition → expansion → φ-ratio compression)
generalizes beyond GSM8K to other reasoning tasks.

Benchmarks:
- GSM8K: Math word problems (97% with adapter)
- ARC-Challenge: Science reasoning
- HellaSwag: Commonsense reasoning
- BoolQ: Reading comprehension
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_benchmark(model, tokenizer, benchmark, max_tokens=200):
    """Evaluate model on a benchmark."""
    from mlx_lm import generate

    correct = 0
    total = len(benchmark.samples)
    results = []

    for i, sample in enumerate(benchmark.samples):
        prompt = sample.prompt
        expected = sample.answer.strip().lower()

        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        output = output.strip()

        # For multiple choice, check if the answer is in the output
        if sample.choices:
            # Check if expected answer appears in output
            is_correct = expected in output.lower()
            # Also check if it explicitly says the answer
            for choice in sample.choices:
                if choice.lower() == expected and choice.lower() in output.lower():
                    is_correct = True
                    break
        else:
            # For boolean questions
            if expected in ["yes", "no"]:
                output_lower = output.lower()
                if expected == "yes":
                    is_correct = "yes" in output_lower and "no" not in output_lower[:20]
                else:
                    is_correct = "no" in output_lower[:20]
            else:
                is_correct = expected in output.lower()

        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "expected": expected,
            "output": output[:200],
            "is_correct": is_correct,
        })

    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "results": results,
    }


def compute_entropy_metrics(model, tokenizer, prompts):
    """Compute entropy trajectory metrics for a set of prompts."""
    import mlx.core as mx

    n_layers = len(model.model.layers)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    all_trajectories = []

    for prompt in prompts[:5]:  # Sample 5 prompts
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = model.model.embed_tokens(input_ids)

        trajectory = []
        for layer in model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            std = float(np.std(np.array(hidden[0, -1, :].tolist())))
            trajectory.append(std)

        all_trajectories.append(trajectory)

    # Average trajectory
    avg_trajectory = np.mean(all_trajectories, axis=0)
    peak_idx = np.argmax(avg_trajectory)
    peak = avg_trajectory[peak_idx]
    initial = avg_trajectory[0]
    final = avg_trajectory[-1]

    expansion = (peak - initial) / (peak_idx + 1) if peak_idx > 0 else 0
    compression_layers = n_layers - peak_idx - 1
    compression = (peak - final) / max(compression_layers, 1)
    ratio = compression / expansion if expansion > 1e-10 else float('inf')

    return {
        "initial": float(initial),
        "peak": float(peak),
        "peak_layer": int(peak_idx),
        "final": float(final),
        "expansion_rate": float(expansion),
        "compression_rate": float(compression),
        "ratio": float(ratio),
        "ratio_vs_phi": float(ratio / PHI) if ratio != float('inf') else float('inf'),
    }


def main():
    from mlx_lm import load
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    logger.info("=" * 70)
    logger.info("MULTI-BENCHMARK EVALUATION")
    logger.info("Testing if geometric learning generalizes")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    # Test with and without adapter
    configs = [
        ("Base model", None),
        ("Unified adapter", "data/adapters/unified_expansion_lora"),
    ]

    loader = BenchmarkLoader()

    # Load benchmarks
    benchmarks_to_test = [
        ("arc_challenge", 30),  # Science reasoning
        ("hellaswag", 30),      # Commonsense
        ("boolq", 30),          # Reading comprehension
    ]

    results = {"timestamp": datetime.now().isoformat(), "benchmarks": {}}

    for bench_name, limit in benchmarks_to_test:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Loading {bench_name}...")

        try:
            benchmark = loader.load(bench_name, split="test", limit=limit)
            logger.info(f"Loaded {len(benchmark.samples)} samples")

            results["benchmarks"][bench_name] = {"samples": len(benchmark.samples), "configs": {}}

            for config_name, adapter_path in configs:
                logger.info(f"\n{config_name}:")

                if adapter_path:
                    model, tokenizer = load(model_path, adapter_path=adapter_path)
                else:
                    model, tokenizer = load(model_path)

                # Evaluate accuracy
                eval_result = evaluate_benchmark(model, tokenizer, benchmark)
                logger.info(f"  Accuracy: {eval_result['correct']}/{eval_result['total']} ({eval_result['accuracy']:.0f}%)")

                # Compute entropy metrics
                prompts = [s.prompt for s in benchmark.samples]
                entropy_metrics = compute_entropy_metrics(model, tokenizer, prompts)
                logger.info(f"  Expansion rate: {entropy_metrics['expansion_rate']:.4f}")
                logger.info(f"  Ratio/φ: {entropy_metrics['ratio_vs_phi']:.4f}")

                results["benchmarks"][bench_name]["configs"][config_name] = {
                    "accuracy": eval_result["accuracy"],
                    "correct": eval_result["correct"],
                    "total": eval_result["total"],
                    "entropy": entropy_metrics,
                }

                del model

        except Exception as e:
            logger.error(f"Failed to load {bench_name}: {e}")
            continue

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n{'Benchmark':<20} {'Base':<15} {'Unified':<15} {'Delta':<10}")
    logger.info("-" * 60)

    for bench_name in results["benchmarks"]:
        bench_data = results["benchmarks"][bench_name]["configs"]
        base_acc = bench_data.get("Base model", {}).get("accuracy", 0)
        unified_acc = bench_data.get("Unified adapter", {}).get("accuracy", 0)
        delta = unified_acc - base_acc
        logger.info(f"{bench_name:<20} {base_acc:>6.0f}%        {unified_acc:>6.0f}%        {delta:+.0f}%")

    # Save results
    output_path = Path("data/experiments/multi_benchmark_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
