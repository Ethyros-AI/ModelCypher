#!/usr/bin/env python3
"""Build the R1 quick-suite aligned training corpus WITH reasoning chains preserved.

Fork of build_r1_quick_aligned_dataset.py. The only change: for GSM8K samples,
the training text uses the full step-by-step answer chain instead of the
stripped final number.

Original format:  "Question\nAnswer: 216"
Chain format:     "Question\nAnswer: Kangaroos / Koalas: 180 / 5 = 36 koalas\nKangaroos + Koalas: 180 + 36 = 216\n#### 216"

This gives the model token-level working memory for multi-step arithmetic.
ARC and BoolQ answers are unchanged (they don't have stripped chains).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from modelcypher.core.use_cases.curriculum.benchmark_loader import (
    BenchmarkLoader,
    BenchmarkSample,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_chain_train.jsonl"
DEFAULT_EVAL_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_chain_val.jsonl"
DEFAULT_MANIFEST_OUTPUT = REPO_ROOT / "data" / "training" / "r1_quick_aligned_chain_manifest.json"

# Reuse the same task specs and seeding from the original builder
# to produce the same sample selection, just different answer format.
from scripts.build_r1_quick_aligned_dataset import (
    TASK_SPECS,
    _shuffle_rows,
    _shuffle_samples,
    _stable_seed,
    _word_stats,
    _write_json,
    _write_jsonl,
)


def _to_training_rows_chain(
    samples: Iterable[BenchmarkSample],
    *,
    task: str,
    source_split: str,
) -> list[dict[str, str]]:
    """Convert samples to training rows, preserving full chains for GSM8K."""
    rows: list[dict[str, str]] = []
    for sample in samples:
        if task == "gsm8k":
            # Use full reasoning chain from metadata if available
            full_answer = sample.metadata.get("full_answer", sample.answer)
            text = f"{sample.prompt} {full_answer}"
        else:
            text = f"{sample.prompt} {sample.answer}"

        rows.append({
            "text": text,
            "task": task,
            "source_split": source_split,
        })
    return rows


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
            _to_training_rows_chain(
                train_selected[spec.name],
                task=spec.name,
                source_split=spec.train_split,
            )
        )
        eval_source_split = "validation" if spec.heldout_mode == "validation" else "train_remainder"
        eval_rows.extend(
            _to_training_rows_chain(
                eval_selected[spec.name],
                task=spec.name,
                source_split=eval_source_split,
            )
        )

    # Use the same row-level shuffle seeds as the original
    train_row_seed = _stable_seed("R1", "quick", "rows", "train")
    eval_row_seed = _stable_seed("R1", "quick", "rows", "eval")
    train_rows = _shuffle_rows(train_rows, seed=train_row_seed)
    eval_rows = _shuffle_rows(eval_rows, seed=eval_row_seed)

    _write_jsonl(train_output, train_rows)
    _write_jsonl(eval_output, eval_rows)

    # Count chain stats
    gsm8k_train_chain_lengths = []
    for row in train_rows:
        if row["task"] == "gsm8k":
            answer_part = row["text"].split("Answer:")[-1].strip() if "Answer:" in row["text"] else ""
            gsm8k_train_chain_lengths.append(len(answer_part.split()))

    manifest = {
        "roadmap_item": "R1",
        "variant": "chain_preserved",
        "purpose": (
            "Same sample selection as r1_quick_aligned, but GSM8K answers include "
            "the full step-by-step reasoning chain instead of the stripped final number. "
            "Tests whether giving the model intermediate computation steps in the token "
            "stream improves multi-step arithmetic accuracy."
        ),
        "change_from_original": (
            "GSM8K: sample.metadata['full_answer'] instead of sample.answer. "
            "ARC and BoolQ: unchanged."
        ),
        "benchmark_suite": [spec.name for spec in TASK_SPECS],
        "derivation": {
            "train_per_task": train_per_task,
            "eval_per_task": eval_per_task,
        },
        "gsm8k_chain_stats": {
            "train_chain_word_lengths": {
                "count": len(gsm8k_train_chain_lengths),
                "min": min(gsm8k_train_chain_lengths) if gsm8k_train_chain_lengths else 0,
                "median": sorted(gsm8k_train_chain_lengths)[len(gsm8k_train_chain_lengths) // 2]
                if gsm8k_train_chain_lengths else 0,
                "max": max(gsm8k_train_chain_lengths) if gsm8k_train_chain_lengths else 0,
            },
        },
        "selected_counts": {
            spec.name: {
                "train": len(train_selected[spec.name]),
                "eval": len(eval_selected[spec.name]),
            }
            for spec in TASK_SPECS
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
        description="Build R1 chain-preserved training/eval dataset.",
    )
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        train_output=args.train_output,
        eval_output=args.eval_output,
        manifest_output=args.manifest_output,
    )
    print("Built R1 chain-preserved dataset:")
    print(f"  Train: {manifest['outputs']['train_jsonl']}")
    print(f"  Eval:  {manifest['outputs']['eval_jsonl']}")
    print(f"  Train per task: {manifest['derivation']['train_per_task']}")
    print(f"  Eval per task:  {manifest['derivation']['eval_per_task']}")
    if manifest.get("gsm8k_chain_stats"):
        stats = manifest["gsm8k_chain_stats"]["train_chain_word_lengths"]
        print(f"  GSM8K chain lengths: min={stats['min']}, median={stats['median']}, max={stats['max']}")


if __name__ == "__main__":
    main()
