#!/usr/bin/env python3
"""Evaluate an NB-LoRA adapter against its base model.

Runs lm-eval benchmarks and inference tests, outputs comparison.

Usage:
  poetry run python scripts/eval_adapter.py \
    --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
    --adapter /Volumes/CodeCypher/experiments/certificate-350m-eval/adapter \
    --output /Volumes/CodeCypher/experiments/certificate-350m-eval/results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Fix lm-eval / transformers 5.x incompatibility
import transformers
from transformers.utils.import_utils import _LazyModule

_original_lazy_getattr = _LazyModule.__getattr__


def _patched_lazy_getattr(self, name):
    if name == "AutoModelForVision2Seq":
        return type("_DummyAutoModel", (), {})
    return _original_lazy_getattr(self, name)


_LazyModule.__getattr__ = _patched_lazy_getattr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

BENCHMARK_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "boolq",
    "piqa",
    "winogrande",
    "openbookqa",
]

INFERENCE_PROMPTS = REPO_ROOT / "data" / "eval_prompts" / "nblora_inference_tests.jsonl"


def run_lm_eval(model_path: str, tasks: list[str], adapter_path: str | None = None,
                limit: int | None = None) -> dict:
    import mlx.core as mx
    from lm_eval import simple_evaluate
    from mlx_lm.evaluate import MLXLM

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    label = f"adapter={adapter_path or 'none'}"
    logger.info(f"  lm-eval: {', '.join(tasks)} ({label})")

    lm = MLXLM(path_or_hf_repo=model_path, batch_size=1)

    if adapter_path:
        from mlx_lm.lora import load_adapters
        load_adapters(lm._model, adapter_path)

    results = simple_evaluate(
        model=lm, tasks=tasks, limit=limit,
        random_seed=42, numpy_random_seed=42,
    )

    scores = {}
    for task_name, task_results in results.get("results", {}).items():
        scores[task_name] = {
            k: v for k, v in task_results.items() if not k.startswith("alias")
        }

    del lm
    mx.clear_cache()
    gc.collect()
    return scores


def run_inference_tests(model_path: str, adapter_path: str | None = None,
                        max_tokens: int = 200) -> list[dict]:
    import mlx.core as mx
    from mlx_lm import generate, load as mlx_load

    logger.info(f"  Inference tests (adapter={adapter_path or 'none'})")

    if adapter_path:
        model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = mlx_load(model_path)

    prompts = []
    with open(INFERENCE_PROMPTS) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    results = []
    for p in prompts:
        try:
            response = generate(
                model, tokenizer, prompt=p["prompt"],
                max_tokens=max_tokens,
            )
            results.append({
                "id": p["id"],
                "category": p["category"],
                "distribution": p.get("distribution", "unknown"),
                "prompt": p["prompt"],
                "expected": p["expected"],
                "response": response,
            })
        except Exception as e:
            results.append({
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "expected": p["expected"],
                "response": f"ERROR: {e}",
            })

    del model
    mx.clear_cache()
    gc.collect()
    return results


def extract_accuracy(scores: dict) -> dict[str, float]:
    """Extract primary accuracy metric from each task."""
    acc = {}
    for task, metrics in scores.items():
        for key in ("acc_norm,none", "acc,none"):
            if key in metrics:
                acc[task] = metrics[key]
                break
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate adapter vs baseline")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--adapter", required=True, help="Adapter path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Samples per benchmark")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    results = {"model": args.model, "adapter": args.adapter, "timestamp": time.strftime("%Y%m%d-%H%M%S")}

    # 1. Benchmarks
    if not args.skip_benchmarks:
        logger.info("=== Baseline Benchmarks ===")
        baseline_scores = run_lm_eval(args.model, BENCHMARK_TASKS, limit=args.limit)
        baseline_acc = extract_accuracy(baseline_scores)

        logger.info("=== Adapted Benchmarks ===")
        adapted_scores = run_lm_eval(args.model, BENCHMARK_TASKS,
                                     adapter_path=args.adapter, limit=args.limit)
        adapted_acc = extract_accuracy(adapted_scores)

        # Comparison
        comparison = {}
        for task in BENCHMARK_TASKS:
            b = baseline_acc.get(task, 0)
            a = adapted_acc.get(task, 0)
            comparison[task] = {
                "baseline": round(b * 100, 2),
                "adapted": round(a * 100, 2),
                "delta": round((a - b) * 100, 2),
            }

        mean_baseline = sum(baseline_acc.values()) / len(baseline_acc) if baseline_acc else 0
        mean_adapted = sum(adapted_acc.values()) / len(adapted_acc) if adapted_acc else 0

        comparison["_mean"] = {
            "baseline": round(mean_baseline * 100, 2),
            "adapted": round(mean_adapted * 100, 2),
            "delta": round((mean_adapted - mean_baseline) * 100, 2),
        }

        results["benchmarks"] = {
            "baseline_raw": baseline_scores,
            "adapted_raw": adapted_scores,
            "comparison": comparison,
        }

        logger.info("=== Benchmark Comparison ===")
        logger.info(f"{'Task':<20} {'Baseline':>10} {'Adapted':>10} {'Delta':>10}")
        logger.info("-" * 52)
        for task in BENCHMARK_TASKS:
            c = comparison.get(task, {})
            logger.info(f"{task:<20} {c.get('baseline', 0):>9.1f}% {c.get('adapted', 0):>9.1f}% {c.get('delta', 0):>+9.1f}%")
        c = comparison["_mean"]
        logger.info("-" * 52)
        logger.info(f"{'MEAN':<20} {c['baseline']:>9.1f}% {c['adapted']:>9.1f}% {c['delta']:>+9.1f}%")

    # 2. Inference
    if not args.skip_inference:
        logger.info("=== Baseline Inference ===")
        baseline_inf = run_inference_tests(args.model)

        logger.info("=== Adapted Inference ===")
        adapted_inf = run_inference_tests(args.model, adapter_path=args.adapter)

        results["inference"] = {
            "baseline": baseline_inf,
            "adapted": adapted_inf,
        }

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
