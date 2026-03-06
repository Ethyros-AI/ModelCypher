# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Sweep runner for LKM validation protocol.

Orchestrates training + evaluation across a grid of (r_cap, token_size) pairs.
Supports resume: skips runs where raw_scores.jsonl already exists.

Usage:
    poetry run python scripts/lkm/run_sweep.py \\
        --model /path/to/model \\
        --ranks 4,16,64,256 \\
        --token-sizes 1000,4000,8000,16000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path for scripts.lkm imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def run_sweep(
    model_path: str,
    data_dir: str,
    eval_path: str,
    output_base: str,
    ranks: list[int],
    token_sizes: list[int],
    arm: str = "B0",
) -> list[dict]:
    """Run training + evaluation for each (r_cap, token_size) pair in the grid.

    For each combination, trains a LoRA adapter and evaluates exact-match
    accuracy on the corresponding phonebook subset. Skips runs where
    raw_scores.jsonl already exists (resume support).

    Args:
        model_path: Path to the base model directory.
        data_dir: Directory containing phonebook_{tokens}tok.jsonl files.
        eval_path: Path to the full phonebook_eval.jsonl file.
        output_base: Base directory for results.
        ranks: List of LoRA ranks to sweep.
        token_sizes: List of token sizes to sweep.
        arm: Experiment arm name (default: "B0").

    Returns:
        List of result dicts, one per completed run, each containing:
            run_id, r_cap, tokens, n_train_pairs, exact_match_rate,
            train_time_s, eval_time_s.
    """
    from scripts.lkm.evaluate_phonebook import evaluate
    from scripts.lkm.run_b0 import make_run_id, train_b0

    model_id = Path(model_path).name
    arm_dir = Path(output_base) / model_id / arm

    # Build the full grid
    grid = [(r_cap, tokens) for r_cap in ranks for tokens in token_sizes]
    total = len(grid)

    results: list[dict] = []

    for n, (r_cap, tokens) in enumerate(grid, start=1):
        run_id = make_run_id(arm, r_cap, tokens)
        run_dir = arm_dir / run_id
        scores_path = run_dir / "raw_scores.jsonl"

        print("=" * 60)
        print(f"Run {n}/{total}: {run_id}")
        print("=" * 60)

        # Resume support: skip if already evaluated
        if scores_path.exists():
            print(f"Skipping {run_id}: {scores_path} already exists")
            # Load existing result for the summary
            existing = _load_existing_result(run_id, r_cap, tokens, scores_path)
            if existing is not None:
                results.append(existing)
            continue

        # Find data file
        data_path = Path(data_dir) / f"phonebook_{tokens}tok.jsonl"
        if not data_path.exists():
            print(f"WARNING: Data file not found: {data_path} -- skipping {run_id}")
            continue

        # Count training pairs
        n_train_pairs = _count_jsonl_lines(data_path)

        # Train
        print(f"Training: r_cap={r_cap}, tokens={tokens}, n_pairs={n_train_pairs}")
        train_start = time.monotonic()
        train_b0(
            model_path=model_path,
            data_path=str(data_path),
            r_cap=r_cap,
            output_dir=str(run_dir),
        )
        train_time_s = time.monotonic() - train_start
        print(f"Training complete in {train_time_s:.1f}s")

        # Create eval subset: first n_train_pairs items from the full eval JSONL
        eval_subset_path = run_dir / "eval_subset.jsonl"
        _write_eval_subset(eval_path, eval_subset_path, n_train_pairs)

        # Evaluate
        print(f"Evaluating {n_train_pairs} pairs...")
        eval_start = time.monotonic()
        exact_match_rate = evaluate(
            model_path=model_path,
            adapter_path=str(run_dir),
            eval_path=str(eval_subset_path),
            output_path=str(scores_path),
        )
        eval_time_s = time.monotonic() - eval_start
        print(f"Evaluation complete in {eval_time_s:.1f}s")

        result = {
            "run_id": run_id,
            "r_cap": r_cap,
            "tokens": tokens,
            "n_train_pairs": n_train_pairs,
            "exact_match_rate": exact_match_rate,
            "train_time_s": round(train_time_s, 2),
            "eval_time_s": round(eval_time_s, 2),
        }
        results.append(result)

        # Save running summary after each run
        summary_path = arm_dir / "sweep_summary.json"
        _save_summary(summary_path, results)

    # Final report
    print()
    print("=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['run_id']}: EM={r['exact_match_rate']:.4f} "
            f"train={r['train_time_s']:.1f}s eval={r['eval_time_s']:.1f}s"
        )

    return results


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _write_eval_subset(
    eval_path: str, subset_path: Path, n_items: int
) -> None:
    """Write the first n_items from the full eval JSONL to subset_path.

    The prefix property of phonebook generation ensures that training pairs
    for an N-token slice are the first K pairs of the source eval JSONL.

    Args:
        eval_path: Path to the full phonebook_eval.jsonl.
        subset_path: Path to write the subset.
        n_items: Number of items to take from the beginning.
    """
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

    print(f"Wrote eval subset: {len(items)} items to {subset_path}")


def _load_existing_result(
    run_id: str, r_cap: int, tokens: int, scores_path: Path
) -> dict | None:
    """Load result metrics from an existing raw_scores.jsonl file.

    Args:
        run_id: Run identifier.
        r_cap: LoRA rank.
        tokens: Token size.
        scores_path: Path to raw_scores.jsonl.

    Returns:
        Result dict, or None if file cannot be parsed.
    """
    try:
        results = []
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        if not results:
            return None

        n_correct = sum(1 for r in results if r.get("exact_match", False))
        exact_match_rate = n_correct / len(results)

        return {
            "run_id": run_id,
            "r_cap": r_cap,
            "tokens": tokens,
            "n_train_pairs": len(results),
            "exact_match_rate": exact_match_rate,
            "train_time_s": 0.0,
            "eval_time_s": 0.0,
        }
    except Exception as e:
        print(f"WARNING: Could not load existing scores from {scores_path}: {e}")
        return None


def _save_summary(summary_path: Path, results: list[dict]) -> None:
    """Save running sweep summary to JSON.

    Args:
        summary_path: Path to write sweep_summary.json.
        results: List of result dicts accumulated so far.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sweep summary to {summary_path}")


def main() -> None:
    """CLI entry point for sweep runner."""
    parser = argparse.ArgumentParser(
        description="Run LKM validation sweep across (rank, token_size) grid."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to base model directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lkm",
        help="Directory containing phonebook_*tok.jsonl files (default: data/lkm).",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/lkm/phonebook_eval.jsonl",
        help="Path to full eval JSONL (default: data/lkm/phonebook_eval.jsonl).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/lora_memory_capacity_validation",
        help="Base results directory (default: results/lora_memory_capacity_validation).",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="B0",
        choices=["B0", "G1", "G2", "G3"],
        help="Experiment arm (default: B0).",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="4,16,64,256",
        help="Comma-separated LoRA ranks (default: 4,16,64,256).",
    )
    parser.add_argument(
        "--token-sizes",
        type=str,
        default="1000,4000,8000,16000",
        help="Comma-separated token sizes (default: 1000,4000,8000,16000).",
    )

    args = parser.parse_args()

    ranks = [int(r.strip()) for r in args.ranks.split(",")]
    token_sizes = [int(t.strip()) for t in args.token_sizes.split(",")]

    print(f"Model:       {args.model}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Eval data:   {args.eval_data}")
    print(f"Output base: {args.output_base}")
    print(f"Arm:         {args.arm}")
    print(f"Ranks:       {ranks}")
    print(f"Token sizes: {token_sizes}")
    print(f"Total runs:  {len(ranks) * len(token_sizes)}")
    print()

    run_sweep(
        model_path=args.model,
        data_dir=args.data_dir,
        eval_path=args.eval_data,
        output_base=args.output_base,
        ranks=ranks,
        token_sizes=token_sizes,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
