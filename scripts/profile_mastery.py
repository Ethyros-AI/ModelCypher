#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Measure baseline mastery of a model across all curriculum skill nodes.

Evaluates every node in CURRICULUM_DAG (topological order) and prints a table
showing accuracy, Clopper-Pearson CI, regime, and mastered status. Nodes whose
eval files do not exist are skipped with a warning.

Usage:
    poetry run python scripts/profile_mastery.py \\
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \\
        --output results/mastery_profile_lfm2_350m.json

Output (JSON):
    {
      "model": "/path/to/model",
      "evaluated_at": "2026-03-03T...",
      "n_mastered": 5,
      "n_evaluated": 13,
      "frontier": ["modus_tollens", ...],   -- shallowest unmastered nodes
      "skills": [
        {
          "name": "modus_ponens",
          "depth": 0,
          "branch": "logic",
          "accuracy": 0.450,
          "ci_lower": 0.322,
          "ci_upper": 0.583,
          "n_total": 100,
          "regime": "reinforce",
          "mastered": true
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _init_backend() -> None:
    """Initialize the default backend if not already initialized."""
    from modelcypher.core.domain._backend import get_default_backend

    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend

        set_default_backend(get_backend(detect_default_backend_type()))


def _node_depth(dag: object, node_name: str) -> int:
    """Compute DAG depth of a node (longest path from any root)."""
    from modelcypher.core.use_cases.curriculum.skill_dag import SkillDAG

    assert isinstance(dag, SkillDAG)
    node = dag.get(node_name)
    if not node.prerequisites:
        return 0
    return 1 + max(_node_depth(dag, p) for p in node.prerequisites)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", help="Save results JSON to this path (optional)")
    args = parser.parse_args()

    model_path = args.model

    print(f"Initializing backend...")
    _init_backend()

    from modelcypher.adapters.curriculum_eval_adapter import evaluate_skill_mastery
    from modelcypher.core.use_cases.curriculum import CURRICULUM_DAG

    nodes = CURRICULUM_DAG.topological_sort()
    print(f"Evaluating {len(nodes)} nodes in topological order...\n")

    results = []
    n_mastered = 0
    n_skipped = 0

    for node in nodes:
        eval_file = Path(node.eval_files[0]) if node.eval_files else None
        if eval_file is None or not eval_file.exists():
            logger.warning("Skipping %s — eval file not found: %s", node.name, eval_file)
            n_skipped += 1
            continue

        # word_problem_multi has 2 eval files; evaluate both and use the one with
        # the lower ci_lower (more conservative mastery estimate)
        eval_candidates = [Path(f) for f in node.eval_files if Path(f).exists()]
        if not eval_candidates:
            logger.warning("Skipping %s — no eval files found", node.name)
            n_skipped += 1
            continue

        print(f"  [{node.name}] ...", end="", flush=True)
        best_record = None
        for ef in eval_candidates:
            try:
                record = evaluate_skill_mastery(model_path, node, ef, chance_rate=0.0)
                # Pick the record with lower ci_lower (most conservative)
                if best_record is None or record.ci_lower < best_record.ci_lower:
                    best_record = record
            except Exception as e:
                logger.warning("Failed to evaluate %s on %s: %s", node.name, ef, e)

        if best_record is None:
            print(" FAILED")
            n_skipped += 1
            continue

        mastered = best_record.is_mastered()
        if mastered:
            n_mastered += 1

        depth = _node_depth(CURRICULUM_DAG, node.name)
        result = {
            "name": node.name,
            "depth": depth,
            "branch": node.branch,
            "accuracy": round(best_record.accuracy, 4),
            "ci_lower": round(best_record.ci_lower, 4),
            "ci_upper": round(best_record.ci_upper, 4),
            "n_total": best_record.n_total,
            "regime": best_record.regime,
            "mastered": mastered,
        }
        results.append(result)
        status = "YES" if mastered else "no"
        print(
            f" acc={best_record.accuracy:.3f}  "
            f"CI=[{best_record.ci_lower:.3f}, {best_record.ci_upper:.3f}]  "
            f"regime={best_record.regime:<18s}  mastered={status}"
        )

    n_evaluated = len(results)

    # Frontier: shallowest unmastered nodes whose prerequisites are all mastered
    mastered_names = {r["name"] for r in results if r["mastered"]}
    frontier = []
    for r in sorted(results, key=lambda x: x["depth"]):
        if r["mastered"]:
            continue
        node = CURRICULUM_DAG.get(r["name"])
        if all(p in mastered_names for p in node.prerequisites):
            frontier.append(r["name"])

    # ── Print table ────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  MASTERY PROFILE — {model_path}")
    print("=" * 90)
    header = f"  {'Depth':>5}  {'Branch':<8}  {'Skill':<28}  {'Acc':>5}  {'CI Lower':>8}  {'Regime':<20}  Mastered?"
    print(header)
    print("  " + "-" * 86)
    for r in sorted(results, key=lambda x: (x["depth"], x["branch"], x["name"])):
        mastered_str = "YES" if r["mastered"] else "no"
        print(
            f"  {r['depth']:>5}  {r['branch']:<8}  {r['name']:<28}  "
            f"{r['accuracy']:>5.3f}  {r['ci_lower']:>8.3f}  {r['regime']:<20}  {mastered_str}"
        )
    print("  " + "-" * 86)
    print(f"  {n_mastered}/{n_evaluated} mastered  ({n_skipped} skipped — eval file missing)")
    if frontier:
        print(f"  Training frontier: {', '.join(frontier)}")
    else:
        print("  Training frontier: (all reachable nodes mastered)")
    print("=" * 90)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": model_path,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "n_mastered": n_mastered,
            "n_evaluated": n_evaluated,
            "n_skipped": n_skipped,
            "frontier": frontier,
            "skills": results,
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
