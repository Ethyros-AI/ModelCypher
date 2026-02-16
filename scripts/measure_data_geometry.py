#!/usr/bin/env python3
"""Measure geometric complexity that training data induces in a base model.

For each sample, collects hidden activations and computes per-layer intrinsic
dimension → expansion_ratio. Compares datasets to verify that expansion data
induces higher-dimensional representations than template data.

Usage:
  poetry run python scripts/measure_data_geometry.py \
    --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
    --datasets data/training/benchmark_train.jsonl data/training/expansion_train.jsonl \
    --n-samples 50 --output /tmp/data_geometry_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)


def measure_dataset(
    model,
    tokenizer,
    backend,
    samples: list[dict],
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Measure expansion_ratio for each sample."""
    from modelcypher.core.domain.training.dimensional_monitor import (
        compute_expansion_from_activations,
    )

    rng = random.Random(seed)
    if len(samples) > n_samples:
        samples = rng.sample(samples, n_samples)

    results = []
    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        if not text:
            continue

        # Truncate to avoid OOM on long samples
        text = text[:512]

        try:
            acts = backend.collect_hidden_activations(model, tokenizer, [text])
            if not acts:
                continue

            snapshot = compute_expansion_from_activations(acts, backend, epoch=0)
            if snapshot is None:
                continue

            results.append({
                "sample_idx": i,
                "text_preview": text[:80],
                "expansion_ratio": snapshot.expansion_ratio,
                "peak_dim": snapshot.peak_dim,
                "final_dim": snapshot.final_dim,
                "peak_layer": snapshot.peak_layer,
                "smoothness": snapshot.smoothness,
            })

            if (i + 1) % 10 == 0:
                logger.info(
                    "  %d/%d samples measured (last exp_ratio=%.3f)",
                    i + 1, len(samples), snapshot.expansion_ratio,
                )
        except Exception as e:
            logger.debug("Failed on sample %d: %s", i, e)
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure geometric complexity of training data",
    )
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="JSONL dataset paths to compare",
    )
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Initialize and get backend
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()

    logger.info("Loading model from %s", args.model)
    model, tokenizer = backend.load_model(args.model)

    report = {
        "model": args.model,
        "n_samples": args.n_samples,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "datasets": {},
    }

    for ds_path in args.datasets:
        ds_name = Path(ds_path).stem
        logger.info("=== Measuring %s ===", ds_name)

        with open(ds_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        results = measure_dataset(
            model, tokenizer, backend, samples,
            n_samples=args.n_samples, seed=args.seed,
        )

        if results:
            exp_ratios = [r["expansion_ratio"] for r in results
                         if r["expansion_ratio"] == r["expansion_ratio"]]
            peak_dims = [r["peak_dim"] for r in results]
            final_dims = [r["final_dim"] for r in results]

            summary = {
                "n_measured": len(results),
                "mean_expansion_ratio": sum(exp_ratios) / len(exp_ratios) if exp_ratios else 0,
                "median_expansion_ratio": sorted(exp_ratios)[len(exp_ratios) // 2] if exp_ratios else 0,
                "mean_peak_dim": sum(peak_dims) / len(peak_dims) if peak_dims else 0,
                "mean_final_dim": sum(final_dims) / len(final_dims) if final_dims else 0,
            }
        else:
            summary = {"n_measured": 0}

        report["datasets"][ds_name] = {
            "path": ds_path,
            "summary": summary,
            "samples": results,
        }

        logger.info(
            "  %s: n=%d, mean_exp_ratio=%.3f, mean_peak=%.1f, mean_final=%.1f",
            ds_name,
            summary.get("n_measured", 0),
            summary.get("mean_expansion_ratio", 0),
            summary.get("mean_peak_dim", 0),
            summary.get("mean_final_dim", 0),
        )

    # Comparison
    if len(report["datasets"]) >= 2:
        logger.info("=== Comparison ===")
        names = list(report["datasets"].keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = report["datasets"][names[i]]["summary"]
                b = report["datasets"][names[j]]["summary"]
                delta = (b.get("mean_expansion_ratio", 0) -
                         a.get("mean_expansion_ratio", 0))
                logger.info(
                    "  %s vs %s: Δexpansion_ratio = %+.3f",
                    names[i], names[j], delta,
                )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", args.output)
    else:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
