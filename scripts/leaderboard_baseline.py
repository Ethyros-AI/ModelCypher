#!/usr/bin/env python3
"""Run Open LLM Leaderboard v2 tasks with the local MLX lm-eval stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modelcypher.core.domain.geometry.domain_benchmark_map import get_suite

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUITE_NAME = "leaderboard_v2"
BASELINE_SCHEMA = "mc.leaderboard.baseline.v1"
PRIMARY_SCORE_METRICS: dict[str, str] = {
    "leaderboard_ifeval": "prompt_level_strict_acc,none",
    "leaderboard_bbh": "acc_norm,none",
    "leaderboard_math_hard": "exact_match,none",
    "leaderboard_gpqa": "acc_norm,none",
    "leaderboard_musr": "acc_norm,none",
    "leaderboard_mmlu_pro": "acc,none",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MLX model directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated subset of leaderboard_v2 tasks",
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


def model_name_from_path(model_path: str) -> str:
    return Path(model_path).name


def resolve_task_list(tasks_arg: str | None) -> list[str]:
    allowed = get_suite(SUITE_NAME)
    if not allowed:
        raise ValueError(f"Suite {SUITE_NAME!r} is not registered")
    if tasks_arg is None:
        return list(allowed)

    requested = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if not requested:
        raise ValueError("No leaderboard tasks were provided")

    unknown = [task for task in requested if task not in allowed]
    if unknown:
        raise ValueError(
            "Unknown leaderboard tasks: "
            + ", ".join(sorted(unknown))
            + f". Allowed tasks: {', '.join(allowed)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate leaderboard tasks are not allowed")
    return requested


def check_for_active_gpu_processes() -> None:
    probe = subprocess.run(
        ["zsh", "-lc", "pgrep -af 'python|mlx' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode not in (0, 1):
        raise RuntimeError(f"GPU process check failed: {probe.stderr.strip()}")

    ignore_pids = {str(os.getpid()), str(os.getppid())}
    active_lines: list[str] = []
    for line in probe.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, _rest = stripped.partition(" ")
        if pid in ignore_pids:
            continue
        active_lines.append(stripped)

    if active_lines:
        raise RuntimeError(
            "Active python/mlx processes detected. Resolve them before model work:\n"
            + "\n".join(active_lines)
        )


def run_leaderboard_eval(
    model_path: str,
    task_list: list[str],
    *,
    adapter_path: str | None = None,
    limit: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    from lm_eval.evaluator import simple_evaluate

    from modelcypher.adapters.lm_eval_mlx_wrapper import MLXModelWrapper

    wrapper: MLXModelWrapper | None = None
    try:
        wrapper = MLXModelWrapper(
            model_path=model_path,
            adapter_path=adapter_path,
            batch_size=batch_size,
        )
        results = simple_evaluate(
            model=wrapper,
            tasks=task_list,
            limit=limit,
            random_seed=42,
            numpy_random_seed=42,
        )
    finally:
        if wrapper is not None:
            wrapper.cleanup()

    if results is None:
        raise RuntimeError("lm-eval returned no results")
    return results


def primary_metric_map(task_list: list[str]) -> dict[str, str]:
    return {task: PRIMARY_SCORE_METRICS[task] for task in task_list}


def _result_bucket(raw_results: dict[str, Any], task_name: str) -> dict[str, Any]:
    for bucket_name in ("groups", "results"):
        bucket = raw_results.get(bucket_name)
        if isinstance(bucket, dict):
            task_results = bucket.get(task_name)
            if isinstance(task_results, dict):
                return task_results
    raise KeyError(f"No lm-eval results found for task {task_name}")


def _stderr_key(metric_key: str) -> str:
    if ",none" in metric_key:
        return metric_key.replace(",none", "_stderr,none")
    return f"{metric_key}_stderr"


def normalize_primary_scores(
    raw_results: dict[str, Any],
    task_list: list[str],
) -> dict[str, dict[str, float | str | None]]:
    normalized: dict[str, dict[str, float | str | None]] = {}
    for task_name in task_list:
        metric_key = PRIMARY_SCORE_METRICS[task_name]
        result_bucket = _result_bucket(raw_results, task_name)
        if metric_key not in result_bucket:
            available = ", ".join(sorted(result_bucket.keys()))
            raise KeyError(
                f"Missing primary metric {metric_key!r} for {task_name}. "
                f"Available keys: {available}"
            )

        score = float(result_bucket[metric_key])
        stderr = result_bucket.get(_stderr_key(metric_key))
        normalized[task_name] = {
            "metric": metric_key,
            "score": score,
            "stderr": None if stderr is None else float(stderr),
        }
    return normalized


def compute_composite_mean(
    primary_scores: dict[str, dict[str, float | str | None]],
) -> float:
    if not primary_scores:
        raise ValueError("Cannot compute composite mean without primary scores")
    return sum(float(item["score"]) for item in primary_scores.values()) / len(primary_scores)


def build_metadata(
    *,
    model_path: str,
    task_list: list[str],
    batch_size: int,
    limit: int | None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    import lm_eval

    return {
        "model_path": model_path,
        "model_name": model_name_from_path(model_path),
        "suite_name": SUITE_NAME,
        "task_list": task_list,
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "batch_size": batch_size,
        "limit": limit,
        "has_adapter": adapter_path is not None,
        "adapter_path": adapter_path,
    }


def build_baseline_payload(
    *,
    model_path: str,
    task_list: list[str],
    raw_results: dict[str, Any],
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    primary_scores = normalize_primary_scores(raw_results, task_list)
    payload = {
        "schema": BASELINE_SCHEMA,
        "model_path": model_path,
        "model_name": model_name_from_path(model_path),
        "timestamp": datetime.now(UTC).isoformat(),
        "suite_name": SUITE_NAME,
        "task_list": task_list,
        "primary_score_metrics": primary_metric_map(task_list),
        "primary_scores": primary_scores,
        "composite_mean": compute_composite_mean(primary_scores),
        "metadata": build_metadata(
            model_path=model_path,
            task_list=task_list,
            batch_size=batch_size,
            limit=limit,
        ),
        "lm_eval_raw": raw_results,
    }
    return payload


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def print_summary_table(payload: dict[str, Any]) -> None:
    print(f"Model: {payload['model_name']}")
    print(f"Suite: {payload['suite_name']}")
    print(f"Composite mean: {payload['composite_mean']:.4f}")
    print("")
    print(f"{'Task':<28} {'Metric':<30} {'Score':>10}")
    print("-" * 72)
    for task_name in payload["task_list"]:
        item = payload["primary_scores"][task_name]
        print(f"{task_name:<28} {item['metric']:<30} {float(item['score']):>10.4f}")


def main() -> int:
    args = build_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    try:
        task_list = resolve_task_list(args.tasks)
        check_for_active_gpu_processes()
        raw_results = run_leaderboard_eval(
            str(model_path),
            task_list,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        payload = build_baseline_payload(
            model_path=str(model_path),
            task_list=task_list,
            raw_results=raw_results,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        write_json(Path(args.output), payload)
        print_summary_table(payload)
        print("")
        print(f"Wrote {args.output}")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
