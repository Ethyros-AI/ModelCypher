# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Orchestrator for LKM rank vs optimization falsifier experiment.

Runs 4 arms (crossed rank x optimization budget), each consisting of:
  1. Train LoRA adapter (via run_b0.train_b0)
  2. Evaluate exact-match (via evaluate_phonebook.evaluate)
  3. Measure fact geometry (via measure_fact_geometry.measure_fact_geometry)

Resume support: skips arms where fact_geometry.json already exists.

Usage:
    poetry run python scripts/lkm/run_falsifier_arms.py \\
        --model /Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Experiment arms: (arm_label, rank, steps)
ARMS = [
    ("B0-r4-s1500", 4, 1500),
    ("B0-r4-s4500", 4, 4500),
    ("B0-r16-s1500", 16, 1500),
    ("B0-r16-s4500", 16, 4500),
]

# Fixed data: 4K tokens / 153 pairs
DATA_FILENAME = "phonebook_4000tok.jsonl"
EVAL_FILENAME = "phonebook_eval.jsonl"
N_TRAIN_PAIRS = 153


def run_falsifier_arms(
    model_path: str,
    data_dir: str = "data/lkm",
    eval_path: str = "data/lkm/phonebook_eval.jsonl",
    output_base: str = "results/lora_memory_capacity_validation",
) -> list[dict]:
    """Run all 4 falsifier arms: train + evaluate + measure geometry.

    Args:
        model_path: Path to the base model directory.
        data_dir: Directory containing phonebook JSONL files.
        eval_path: Path to full phonebook_eval.jsonl.
        output_base: Base results directory.

    Returns:
        List of per-arm result summaries.
    """
    from scripts.lkm.evaluate_phonebook import evaluate
    from scripts.lkm.measure_fact_geometry import measure_fact_geometry
    from scripts.lkm.run_b0 import train_b0

    model_id = Path(model_path).name
    falsifier_dir = Path(output_base) / model_id / "falsifier"
    data_path = str(Path(data_dir) / DATA_FILENAME)

    results = []

    for arm_label, rank, steps in ARMS:
        arm_dir = falsifier_dir / arm_label
        geom_path = arm_dir / "fact_geometry.json"

        print("=" * 60)
        print(f"ARM: {arm_label} (rank={rank}, steps={steps})")
        print(f"  Output: {arm_dir}")
        print("=" * 60)

        # Resume: skip if geometry already computed
        if geom_path.exists():
            print(f"  SKIP: {geom_path} already exists")
            with open(geom_path) as f:
                geom = json.load(f)
            scores_path = arm_dir / "raw_scores.jsonl"
            em = _compute_em(scores_path) if scores_path.exists() else None
            results.append({
                "arm": arm_label,
                "rank": rank,
                "steps": steps,
                "em": em,
                "mean_rf": geom["summary"].get("mean_rf"),
            })
            continue

        # 1. Train
        adapter_path = arm_dir / "adapters.safetensors"
        if not adapter_path.exists():
            print(f"  Training (rank={rank}, steps={steps})...")
            t0 = time.monotonic()
            train_b0(
                model_path=model_path,
                data_path=data_path,
                r_cap=rank,
                output_dir=str(arm_dir),
                config_overrides={"iters": steps},
            )
            print(f"  Training done in {time.monotonic() - t0:.1f}s")
        else:
            print(f"  SKIP training: adapter already exists")

        # 2. Evaluate
        scores_path = arm_dir / "raw_scores.jsonl"
        if not scores_path.exists():
            # Write eval subset (first N_TRAIN_PAIRS from full eval)
            eval_subset_path = arm_dir / "eval_subset.jsonl"
            _write_eval_subset(eval_path, eval_subset_path, N_TRAIN_PAIRS)

            print(f"  Evaluating {N_TRAIN_PAIRS} facts...")
            t0 = time.monotonic()
            em = evaluate(
                model_path=model_path,
                adapter_path=str(arm_dir),
                eval_path=str(eval_subset_path),
                output_path=str(scores_path),
            )
            print(f"  Eval done in {time.monotonic() - t0:.1f}s: EM={em:.4f}")
        else:
            print(f"  SKIP eval: scores already exist")
            em = _compute_em(scores_path)
            print(f"  Loaded EM={em:.4f}")

        # 3. Measure geometry
        print(f"  Measuring fact geometry...")
        t0 = time.monotonic()
        geom = measure_fact_geometry(
            model_path=model_path,
            adapter_path=str(arm_dir),
            data_path=data_path,
            output_path=str(geom_path),
        )
        print(f"  Geometry done in {time.monotonic() - t0:.1f}s")

        results.append({
            "arm": arm_label,
            "rank": rank,
            "steps": steps,
            "em": em,
            "mean_rf": geom["summary"].get("mean_rf"),
        })

        # Save running summary
        _save_summary(falsifier_dir / "arm_summary.json", results)

    # Final report
    print()
    print("=" * 60)
    print("ALL ARMS COMPLETE")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['arm']}: EM={r['em']:.4f} "
            f"mean_RF={r['mean_rf']:.4f}"
        )

    return results


def _compute_em(scores_path: Path) -> float:
    """Compute EM from raw_scores.jsonl."""
    with open(scores_path) as f:
        scores = [json.loads(line) for line in f if line.strip()]
    if not scores:
        return 0.0
    return sum(1 for s in scores if s["exact_match"]) / len(scores)


def _write_eval_subset(
    eval_path: str, subset_path: Path, n_items: int
) -> None:
    """Write first n_items from full eval JSONL to subset_path."""
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    items: list[str] = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
                if len(items) >= n_items:
                    break
    with open(subset_path, "w") as f:
        for item in items:
            f.write(item + "\n")


def _save_summary(path: Path, results: list[dict]) -> None:
    """Save running arm summary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    """CLI entry point for falsifier arm orchestration."""
    parser = argparse.ArgumentParser(
        description="Run LKM falsifier arms (rank x optimization budget)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to base model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lkm",
        help="Directory with phonebook JSONL files.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/lkm/phonebook_eval.jsonl",
        help="Path to full eval JSONL.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/lora_memory_capacity_validation",
        help="Base results directory.",
    )

    args = parser.parse_args()
    run_falsifier_arms(
        model_path=args.model,
        data_dir=args.data_dir,
        eval_path=args.eval_data,
        output_base=args.output_base,
    )


if __name__ == "__main__":
    main()
