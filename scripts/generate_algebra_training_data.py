#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate algebra training and eval data for algebra_linear and algebra_nonlinear nodes.

Source: EleutherAI/hendrycks_math (HuggingFace)

  algebra_linear:   configs ['algebra', 'prealgebra'], levels 1-3
                    → linear equations, basic algebraic manipulation
  algebra_nonlinear: configs ['algebra', 'intermediate_algebra'], levels 4-5
                    → polynomials, quadratics, competition-level algebraic reasoning

Output files:
  data/training/math_hs_train.jsonl      — algebra_linear training
  data/eval/math_hs_eval.jsonl           — algebra_linear eval (100 samples, held-out)
  data/training/numina_train.jsonl       — algebra_nonlinear training (named for compatibility)
  data/eval/numina_eval.jsonl            — algebra_nonlinear eval (200 samples, held-out)

Format:
  {"text": "Solve the following problem:\\n{problem}\\n{answer}", "answer_start": N}
  where N is the integer byte offset to where the answer begins.

Answer extraction:
  Parses last \\boxed{...} from the solution field.

Usage:
    poetry run python scripts/generate_algebra_training_data.py
    poetry run python scripts/generate_algebra_training_data.py --output data/training --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _extract_boxed(solution: str) -> str | None:
    """Extract the last \\boxed{...} expression from a LaTeX solution string.

    Handles nested braces (e.g., \\boxed{\\frac{1}{2}}) by scanning for balanced
    braces after \\boxed{.

    Returns None if no \\boxed{} is found in solution.
    """
    matches = []
    i = 0
    marker = r"\boxed{"
    while i < len(solution):
        idx = solution.find(marker, i)
        if idx == -1:
            break
        # Scan for balanced closing brace
        depth = 0
        j = idx + len(marker) - 1  # at the opening {
        start = j + 1
        while j < len(solution):
            if solution[j] == "{":
                depth += 1
            elif solution[j] == "}":
                depth -= 1
                if depth == 0:
                    matches.append(solution[start:j].strip())
                    break
            j += 1
        i = idx + 1

    return matches[-1] if matches else None


def _make_item(problem: str, answer: str) -> dict:
    """Format a problem+answer pair as a training/eval item.

    Format: "Solve the following problem:\\n{problem}\\n{answer}"
    answer_start is the integer byte offset to the start of answer.
    """
    prefix = f"Solve the following problem:\n{problem}\n"
    return {"text": prefix + answer, "answer_start": len(prefix)}


def _parse_level(level_raw: object) -> int | None:
    """Parse level field — may be 'Level 1' (string) or 1 (int)."""
    try:
        return int(str(level_raw).replace("Level", "").strip())
    except ValueError:
        return None


def load_configs(configs: list[str]) -> list[dict]:
    """Load one or more EleutherAI/hendrycks_math configs, combine, and return raw items."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed. Run: poetry add datasets --group dev",
            file=sys.stderr,
        )
        sys.exit(1)

    all_items: list[dict] = []
    for config in configs:
        try:
            # Load both train and test splits to maximise available data
            for split in ("train", "test"):
                try:
                    ds = load_dataset("EleutherAI/hendrycks_math", config, split=split)
                    all_items.extend(list(ds))
                    print(f"  Loaded {len(ds):5d} items from {config}/{split}")
                except Exception as e:
                    print(f"  WARNING: could not load {config}/{split}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  WARNING: could not load config {config!r}: {e}", file=sys.stderr)
    return all_items


def generate_algebra_linear(
    raw: list[dict], seed: int, n_eval: int = 100
) -> tuple[list[dict], list[dict]]:
    """Filter and format hendrycks_math for algebra_linear.

    DATA DESIGN CHOICE (not an algorithmic threshold):
    Configs: algebra + prealgebra — these are the HS linear algebra subjects.
    Level filter ≤3: levels 1-3 scope to problems solvable with linear techniques
    (isolate variable, proportions, simple systems). Levels 4-5 involve nonlinear
    algebra (quadratics, polynomials) which are handled by algebra_nonlinear.

    n_eval=100: matches the ≥100 standard from skill_dag.md for this node.
    Held-out eval is drawn first (fixed order post-shuffle) to prevent leakage.

    Returns (train_items, eval_items).
    """
    rng = random.Random(seed)

    formatted = []
    n_skipped = 0
    for item in raw:
        level = _parse_level(item.get("level"))
        if level is None or level > 3:
            n_skipped += 1
            continue

        problem = (item.get("problem") or "").strip()
        solution = (item.get("solution") or "").strip()
        if not problem or not solution:
            n_skipped += 1
            continue

        answer = _extract_boxed(solution)
        if answer is None:
            n_skipped += 1
            continue

        formatted.append(_make_item(problem, answer))

    if n_skipped > 0:
        print(f"  Skipped {n_skipped} items (level>3, missing fields, or no \\boxed{{}})")

    rng.shuffle(formatted)

    n_eval_actual = min(n_eval, len(formatted) // 5)
    eval_items = formatted[:n_eval_actual]
    train_items = formatted[n_eval_actual:]
    return train_items, eval_items


def generate_algebra_nonlinear(
    raw: list[dict], seed: int, n_eval: int = 200
) -> tuple[list[dict], list[dict]]:
    """Filter and format hendrycks_math for algebra_nonlinear.

    DATA DESIGN CHOICE (not an algorithmic threshold):
    Configs: algebra levels 4-5 + intermediate_algebra (all levels).
    These cover quadratics, polynomials, sequences, and complex multi-step algebraic
    manipulations that require chain_reasoning (the prerequisite). Prealgebra and
    levels 1-3 are excluded — already covered by algebra_linear.

    n_eval=200: matches the ≥200 standard from skill_dag.md for this node.

    Returns (train_items, eval_items).
    """
    rng = random.Random(seed)

    formatted = []
    n_skipped = 0
    for item in raw:
        level = _parse_level(item.get("level"))
        config_type = (item.get("type") or "").strip()

        # For algebra config: require level >= 4 (harder problems)
        # For intermediate_algebra: all levels qualify
        is_algebra_hard = (config_type == "Algebra") and level is not None and level >= 4
        is_intermediate = config_type == "Intermediate Algebra"
        if not (is_algebra_hard or is_intermediate):
            n_skipped += 1
            continue

        problem = (item.get("problem") or "").strip()
        solution = (item.get("solution") or "").strip()
        if not problem or not solution:
            n_skipped += 1
            continue

        answer = _extract_boxed(solution)
        if answer is None:
            n_skipped += 1
            continue

        formatted.append(_make_item(problem, answer))

    if n_skipped > 0:
        print(f"  Skipped {n_skipped} items (level<4, missing fields, or no \\boxed{{}})")

    rng.shuffle(formatted)

    n_eval_actual = min(n_eval, len(formatted) // 5)
    eval_items = formatted[:n_eval_actual]
    train_items = formatted[n_eval_actual:]
    return train_items, eval_items


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Wrote {len(samples):5d} examples → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate algebra training and eval data")
    parser.add_argument("--output", default="data/training", help="Output dir for training files")
    parser.add_argument("--eval-output", default="data/eval", help="Output dir for eval files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_out = Path(args.output)
    eval_out = Path(args.eval_output)
    seed = args.seed

    # ── algebra_linear from algebra + prealgebra, levels 1-3 ─────────────
    print("\n[algebra_linear] Loading EleutherAI/hendrycks_math (algebra, prealgebra)...")
    linear_raw = load_configs(["algebra", "prealgebra"])
    print(f"  Total raw items: {len(linear_raw)}")

    train_items, eval_items = generate_algebra_linear(linear_raw, seed)
    print(f"  Qualifying (level≤3): {len(train_items) + len(eval_items)} total")
    print("  Output:")
    _write_jsonl(train_out / "math_hs_train.jsonl", train_items)
    _write_jsonl(eval_out / "math_hs_eval.jsonl", eval_items)

    # ── algebra_nonlinear from algebra level 4-5 + intermediate_algebra ──
    print("\n[algebra_nonlinear] Loading EleutherAI/hendrycks_math (algebra, intermediate_algebra)...")
    nonlinear_raw = load_configs(["algebra", "intermediate_algebra"])
    print(f"  Total raw items: {len(nonlinear_raw)}")

    train_items, eval_items = generate_algebra_nonlinear(nonlinear_raw, seed + 1)
    print(f"  Qualifying (algebra level≥4 + intermediate_algebra): {len(train_items) + len(eval_items)} total")
    print("  Output:")
    _write_jsonl(train_out / "numina_train.jsonl", train_items)
    _write_jsonl(eval_out / "numina_eval.jsonl", eval_items)

    print("\nDone.")


if __name__ == "__main__":
    main()
