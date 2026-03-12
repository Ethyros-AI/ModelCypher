#!/usr/bin/env python3
"""Build the R1 quick-suite aligned training corpus.

Roadmap linkage: R1 in docs/RESEARCH-ROADMAP.md.

The goal is to train on the same task family that the frozen quick benchmark
measures, using the exact benchmark prompt/answer continuation format instead
of the mixed synthetic benchmark_train bundle.

Derivation:
- Task family is fixed by the quick suite: gsm8k, arc_easy, boolq.
- Train task mass is uniform because we do not have a justified task prior.
  Therefore train_per_task = min available training count across the suite.
- Eval task mass is also uniform on non-benchmark held-out pools.
  Therefore eval_per_task = min available held-out count across the suite.
- Sampling is deterministic: each task/split is shuffled with a SHA256-derived
  seed before slicing so the artifact is reproducible and non-overlapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from modelcypher.core.use_cases.curriculum.benchmark_loader import (
    BenchmarkLoader,
    BenchmarkSample,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_train.jsonl"
DEFAULT_EVAL_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_val.jsonl"
DEFAULT_MANIFEST_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_manifest.json"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_split: str
    benchmark_eval_split: str
    heldout_mode: str


TASK_SPECS = (
    TaskSpec(
        name="gsm8k",
        train_split="train",
        benchmark_eval_split="test",
        heldout_mode="train_remainder",
    ),
    TaskSpec(
        name="arc_easy",
        train_split="train",
        benchmark_eval_split="test",
        heldout_mode="validation",
    ),
    TaskSpec(
        name="boolq",
        train_split="train",
        benchmark_eval_split="validation",
        heldout_mode="train_remainder",
    ),
)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _shuffle_samples(samples: list[BenchmarkSample], *, seed: int) -> list[BenchmarkSample]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _to_training_rows(
    samples: Iterable[BenchmarkSample],
    *,
    task: str,
    source_split: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in samples:
        rows.append(
            {
                "text": f"{sample.prompt} {sample.answer}",
                "task": task,
                "source_split": source_split,
            }
        )
    return rows


def _word_stats(samples: Iterable[BenchmarkSample]) -> dict[str, int]:
    lengths = sorted(len(f"{sample.prompt} {sample.answer}".split()) for sample in samples)
    if not lengths:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "p90": lengths[int((len(lengths) - 1) * 0.9)],
        "p99": lengths[int((len(lengths) - 1) * 0.99)],
        "max": lengths[-1],
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shuffle_rows(rows: list[dict[str, str]], *, seed: int) -> list[dict[str, str]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_dataset(
    *,
    train_output: Path,
    eval_output: Path,
    manifest_output: Path,
) -> dict:
    loader = BenchmarkLoader()

    train_pools: dict[str, list[BenchmarkSample]] = {}
    heldout_pools: dict[str, list[BenchmarkSample]] = {}
    train_shuffle_seeds: dict[str, int] = {}
    heldout_shuffle_seeds: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}

    for spec in TASK_SPECS:
        train_samples = list(loader.load(spec.name, split=spec.train_split).samples)
        train_pools[spec.name] = train_samples
        source_counts[spec.name] = {
            "train_split": len(train_samples),
        }
        if spec.heldout_mode == "validation":
            validation_samples = list(loader.load(spec.name, split="validation").samples)
            heldout_pools[spec.name] = validation_samples
            source_counts[spec.name]["heldout_validation_split"] = len(validation_samples)

    train_per_task = min(len(samples) for samples in train_pools.values())

    train_selected: dict[str, list[BenchmarkSample]] = {}
    for spec in TASK_SPECS:
        seed = _stable_seed("R1", "quick", spec.name, "train")
        train_shuffle_seeds[spec.name] = seed
        shuffled = _shuffle_samples(train_pools[spec.name], seed=seed)
        train_selected[spec.name] = shuffled[:train_per_task]
        if spec.heldout_mode == "train_remainder":
            heldout_pools[spec.name] = shuffled[train_per_task:]

    eval_per_task = min(len(samples) for samples in heldout_pools.values())

    eval_selected: dict[str, list[BenchmarkSample]] = {}
    for spec in TASK_SPECS:
        if spec.heldout_mode == "train_remainder":
            eval_selected[spec.name] = heldout_pools[spec.name][:eval_per_task]
            heldout_shuffle_seeds[spec.name] = train_shuffle_seeds[spec.name]
            continue
        seed = _stable_seed("R1", "quick", spec.name, "validation")
        heldout_shuffle_seeds[spec.name] = seed
        shuffled = _shuffle_samples(heldout_pools[spec.name], seed=seed)
        eval_selected[spec.name] = shuffled[:eval_per_task]

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for spec in TASK_SPECS:
        train_rows.extend(
            _to_training_rows(
                train_selected[spec.name],
                task=spec.name,
                source_split=spec.train_split,
            )
        )
        eval_source_split = "validation" if spec.heldout_mode == "validation" else "train_remainder"
        eval_rows.extend(
            _to_training_rows(
                eval_selected[spec.name],
                task=spec.name,
                source_split=eval_source_split,
            )
        )

    train_row_seed = _stable_seed("R1", "quick", "rows", "train")
    eval_row_seed = _stable_seed("R1", "quick", "rows", "eval")
    train_rows = _shuffle_rows(train_rows, seed=train_row_seed)
    eval_rows = _shuffle_rows(eval_rows, seed=eval_row_seed)

    _write_jsonl(train_output, train_rows)
    _write_jsonl(eval_output, eval_rows)

    manifest = {
        "roadmap_item": "R1",
        "purpose": (
            "Same-model same-data same-eval rerun corpus aligned to the frozen quick "
            "suite prompt/answer format."
        ),
        "benchmark_suite": [spec.name for spec in TASK_SPECS],
        "benchmark_eval_splits": {
            spec.name: spec.benchmark_eval_split for spec in TASK_SPECS
        },
        "derivation": {
            "train_policy": (
                "uniform_task_prior over quick suite: train_per_task = min available "
                "training count across gsm8k, arc_easy, boolq"
            ),
            "train_per_task": train_per_task,
            "eval_policy": (
                "uniform_task_prior over non-benchmark held-out pools: eval_per_task = "
                "min available held-out count across the suite"
            ),
            "eval_per_task": eval_per_task,
            "shuffle_operator": (
                "deterministic SHA256-derived seed per task/split before slicing"
            ),
        },
        "source_counts": source_counts,
        "selected_counts": {
            spec.name: {
                "train": len(train_selected[spec.name]),
                "eval": len(eval_selected[spec.name]),
            }
            for spec in TASK_SPECS
        },
        "shuffle_seeds": {
            spec.name: {
                "train": train_shuffle_seeds[spec.name],
                "eval": heldout_shuffle_seeds[spec.name],
            }
            for spec in TASK_SPECS
        }
        | {
            "row_level": {
                "train": train_row_seed,
                "eval": eval_row_seed,
            }
        },
        "length_stats_words": {
            spec.name: {
                "train": _word_stats(train_selected[spec.name]),
                "eval": _word_stats(eval_selected[spec.name]),
            }
            for spec in TASK_SPECS
        },
        "outputs": {
            "train_jsonl": str(train_output),
            "eval_jsonl": str(eval_output),
            "manifest_json": str(manifest_output),
        },
    }
    _write_json(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the R1 quick-suite aligned training/eval dataset.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="Path to the output training JSONL.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
        help="Path to the output eval JSONL.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
        help="Path to the output manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        train_output=args.train_output,
        eval_output=args.eval_output,
        manifest_output=args.manifest_output,
    )
    print(
        "Built R1 quick aligned dataset:",
        manifest["outputs"]["train_jsonl"],
        manifest["outputs"]["eval_jsonl"],
    )
    print(
        "Train per task:",
        manifest["derivation"]["train_per_task"],
        "| Eval per task:",
        manifest["derivation"]["eval_per_task"],
    )


if __name__ == "__main__":
    main()
