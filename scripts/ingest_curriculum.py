#!/usr/bin/env python3
"""Ingest a frontier-generated curriculum JSON into the training pipeline.

Validates the curriculum, writes JSONL training/eval files, and constructs
the SkillDAG + PhaseScheduler for the training pipeline.

Usage:
    poetry run python scripts/ingest_curriculum.py \
        --curriculum /path/to/curriculum.json \
        --output-dir /path/to/training_data/ \
        --mastered "modus_ponens,modus_tollens"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from modelcypher.core.use_cases.curriculum_generation_service import (
    CurriculumGenerationService,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a frontier-generated curriculum into the training pipeline."
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        required=True,
        help="Path to curriculum JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSONL files and curriculum state.",
    )
    parser.add_argument(
        "--mastered",
        type=str,
        default="",
        help="Comma-separated list of already-mastered skill names.",
    )
    args = parser.parse_args()

    mastered: set[str] | None = None
    if args.mastered.strip():
        mastered = {s.strip() for s in args.mastered.split(",") if s.strip()}

    service = CurriculumGenerationService()
    dag, scheduler, result = service.ingest_curriculum(
        curriculum_json=args.curriculum,
        output_dir=args.output_dir,
        mastered_skills=mastered,
    )

    # Print validation result
    if result.errors:
        print(f"VALIDATION FAILED ({len(result.errors)} errors):")
        for err in result.errors:
            print(f"  ERROR: {err}")
    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  WARNING: {w}")

    if not result.is_valid:
        print("\nCurriculum rejected. Fix errors and re-run.")
        sys.exit(1)

    # Print summary
    print(f"\nCurriculum ingested successfully.")
    print(f"  Skills: {result.skill_count}")
    print(f"  DAG depth: {result.max_dag_depth}")
    print(f"  Branches: {', '.join(result.branch_names)}")
    print(f"  Train samples: {result.total_train_samples}")
    print(f"  Eval samples: {result.total_eval_samples}")

    # Next skill to teach
    mastered_set = scheduler.mastered_skills()
    ready = dag.ready_to_teach(mastered_set)
    if ready:
        print(f"\nNext skills to teach: {', '.join(n.name for n in ready)}")
    else:
        print("\nAll skills mastered or no skills ready.")

    # Data files
    print(f"\nData written to: {args.output_dir}")
    for p in sorted(args.output_dir.glob("*.jsonl")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
