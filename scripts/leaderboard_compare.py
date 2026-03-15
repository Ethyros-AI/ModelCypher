#!/usr/bin/env python3
"""Run leaderboard_v2 on a modified model and compare to a frozen baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from leaderboard_baseline import (
    PRIMARY_SCORE_METRICS,
    SUITE_NAME,
    build_metadata,
    check_for_active_gpu_processes,
    compute_composite_mean,
    model_name_from_path,
    normalize_primary_scores,
    primary_metric_map,
    print_summary_table,
    resolve_task_list,
    run_leaderboard_eval,
    write_json,
)

COMPARE_SCHEMA = "mc.leaderboard.compare.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--adapter", default=None, help="Optional adapter path")
    parser.add_argument("--baseline", required=True, help="Baseline JSON produced by leaderboard_baseline.py")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--label", required=True, help="Human label for this comparison run")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task list. Must match the baseline task list exactly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit passed through to lm-eval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="MLXLM batch size",
    )
    return parser


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Baseline payload must be a JSON object")
    return payload


def _resolve_comparison_task_list(
    baseline_payload: dict[str, Any],
    tasks_arg: str | None,
) -> list[str]:
    baseline_tasks = baseline_payload.get("task_list")
    if not isinstance(baseline_tasks, list) or not all(isinstance(item, str) for item in baseline_tasks):
        raise ValueError("Baseline payload is missing a valid task_list")

    if tasks_arg is None:
        return list(baseline_tasks)

    requested = resolve_task_list(tasks_arg)
    if requested != baseline_tasks:
        raise ValueError(
            "Comparison task list must match the baseline task_list exactly: "
            + ", ".join(baseline_tasks)
        )
    return requested


def _validate_baseline_metric_map(
    baseline_payload: dict[str, Any],
    task_list: list[str],
) -> dict[str, str]:
    expected = primary_metric_map(task_list)
    actual = baseline_payload.get("primary_score_metrics")
    if actual != expected:
        raise ValueError(
            "Baseline primary_score_metrics do not match the installed leaderboard metric mapping"
        )
    return expected


def _baseline_task_score(baseline_payload: dict[str, Any], task_name: str) -> float:
    primary_scores = baseline_payload.get("primary_scores")
    if not isinstance(primary_scores, dict):
        raise ValueError("Baseline payload is missing primary_scores")
    item = primary_scores.get(task_name)
    if not isinstance(item, dict) or "score" not in item:
        raise ValueError(f"Baseline payload is missing score for task {task_name}")
    return float(item["score"])


def _compute_deltas(
    baseline_payload: dict[str, Any],
    current_primary_scores: dict[str, dict[str, float | str | None]],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for task_name, current in current_primary_scores.items():
        deltas[task_name] = float(current["score"]) - _baseline_task_score(baseline_payload, task_name)
    baseline_composite = baseline_payload.get("composite_mean")
    if baseline_composite is None:
        raise ValueError("Baseline payload is missing composite_mean")
    deltas["composite_mean"] = compute_composite_mean(current_primary_scores) - float(baseline_composite)
    return deltas


def build_compare_payload(
    *,
    model_path: str,
    adapter_path: str | None,
    baseline_path: str,
    baseline_payload: dict[str, Any],
    label: str,
    task_list: list[str],
    raw_results: dict[str, Any],
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    primary_scores = normalize_primary_scores(raw_results, task_list)
    payload = {
        "schema": COMPARE_SCHEMA,
        "label": label,
        "baseline_path": baseline_path,
        "baseline_model_path": baseline_payload.get("model_path"),
        "baseline_model_name": baseline_payload.get("model_name"),
        "model_path": model_path,
        "model_name": model_name_from_path(model_path),
        "adapter_path": adapter_path,
        "timestamp": datetime.now(UTC).isoformat(),
        "suite_name": baseline_payload.get("suite_name", SUITE_NAME),
        "task_list": task_list,
        "primary_score_metrics": primary_metric_map(task_list),
        "primary_scores": primary_scores,
        "composite_mean": compute_composite_mean(primary_scores),
        "deltas_vs_baseline": _compute_deltas(baseline_payload, primary_scores),
        "metadata": {
            **build_metadata(
                model_path=model_path,
                task_list=task_list,
                batch_size=batch_size,
                limit=limit,
                adapter_path=adapter_path,
            ),
            "label": label,
            "baseline_path": baseline_path,
        },
        "lm_eval_raw": raw_results,
    }
    return payload


def print_compare_summary(payload: dict[str, Any]) -> None:
    baseline_stub = {
        "model_name": payload["model_name"],
        "suite_name": payload["suite_name"],
        "task_list": payload["task_list"],
        "primary_scores": payload["primary_scores"],
        "composite_mean": payload["composite_mean"],
    }
    print_summary_table(baseline_stub)
    print("")
    print("Deltas vs baseline:")
    print(f"{'Task':<28} {'Delta':>10}")
    print("-" * 40)
    for task_name in payload["task_list"]:
        print(f"{task_name:<28} {payload['deltas_vs_baseline'][task_name]:>+10.4f}")
    print("-" * 40)
    print(f"{'composite_mean':<28} {payload['deltas_vs_baseline']['composite_mean']:>+10.4f}")


def main() -> int:
    args = build_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    try:
        baseline_payload = _load_baseline(Path(args.baseline))
        task_list = _resolve_comparison_task_list(baseline_payload, args.tasks)
        _validate_baseline_metric_map(baseline_payload, task_list)
        check_for_active_gpu_processes()
        raw_results = run_leaderboard_eval(
            str(model_path),
            task_list,
            adapter_path=args.adapter,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        payload = build_compare_payload(
            model_path=str(model_path),
            adapter_path=args.adapter,
            baseline_path=args.baseline,
            baseline_payload=baseline_payload,
            label=args.label,
            task_list=task_list,
            raw_results=raw_results,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        write_json(Path(args.output), payload)
        print_compare_summary(payload)
        print("")
        print(f"Wrote {args.output}")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
