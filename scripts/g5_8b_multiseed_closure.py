#!/usr/bin/env python3
"""Run the G5 8B mission-closure cycle across multiple seeds.

Workflow:
1. Run `scripts/g5_8b_validation.py` once per seed.
2. Aggregate gate verdicts.
3. Emit per-seed artifact presence report.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("g5_8b_multiseed_closure")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G5 8B validation across multiple seeds and aggregate artifacts.",
    )
    parser.add_argument(
        "--model-path",
        default="/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
    )
    parser.add_argument(
        "--fp-reference-model",
        required=True,
        help="Required: full-precision reference model for quantization precheck gate.",
    )
    parser.add_argument("--train-data", default="data/training/benchmark_train.jsonl")
    parser.add_argument("--eval-data", default="data/training/benchmark_val.jsonl")
    parser.add_argument(
        "--retention-data",
        default="data/training/retention_replay.jsonl",
    )
    parser.add_argument(
        "--seeds",
        default="41,42,43",
        help="Comma-separated validation seeds.",
    )
    parser.add_argument(
        "--online-eval-n",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--online-eval-problems-json",
        type=Path,
        default=None,
        help="Optional fixed non-ceiling online-eval problem set JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/g5_8b_validation_multiseed"),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child runs.",
    )
    return parser.parse_args()


def _run_command(cmd: list[str], cwd: Path) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _online_eval_problem_count(problem_set_path: Path) -> int:
    payload = json.loads(problem_set_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("problems", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            "online_eval_problems_json must contain a JSON list or "
            "an object with a 'problems' list.",
        )
    if not isinstance(records, list) or not records:
        raise ValueError("online_eval_problems_json contains no problems")
    return len(records)


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must provide at least one seed")

    if args.online_eval_problems_json is not None:
        online_eval_set = args.online_eval_problems_json.expanduser().resolve()
        if not online_eval_set.exists():
            raise FileNotFoundError(
                f"online_eval_problems_json does not exist: {online_eval_set}",
            )
        problem_count = _online_eval_problem_count(online_eval_set)
        effective_online_eval_n = problem_count
        if args.online_eval_n != problem_count:
            logger.info(
                "online_eval_n=%d does not match fixed set size=%d; "
                "using fixed set size.",
                args.online_eval_n,
                problem_count,
            )
    else:
        online_eval_set = None
        effective_online_eval_n = int(args.online_eval_n)

    for seed in seeds:
        cmd = [
            args.python,
            "scripts/g5_8b_validation.py",
            "--model-path",
            str(Path(args.model_path).expanduser().resolve()),
            "--fp-reference-model",
            str(Path(args.fp_reference_model).expanduser().resolve()),
            "--train-data",
            str(Path(args.train_data).expanduser().resolve()),
            "--eval-data",
            str(Path(args.eval_data).expanduser().resolve()),
            "--seed",
            str(seed),
            "--online-eval-n",
            str(effective_online_eval_n),
            "--eval-interval",
            str(args.eval_interval),
            "--output-root",
            str(output_root),
        ]
        retention_data = args.retention_data
        if retention_data:
            cmd.extend(
                [
                    "--retention-data",
                    str(Path(retention_data).expanduser().resolve()),
                ]
            )
        else:
            cmd.extend(["--retention-data", ""])
        if online_eval_set is not None:
            cmd.extend(
                [
                    "--online-eval-problems-json",
                    str(online_eval_set),
                ]
            )
        _run_command(cmd, cwd=root)

    _run_command(
        [
            args.python,
            "scripts/g5_8b_validation.py",
            "--aggregate-root",
            str(output_root),
        ],
        cwd=root,
    )

    artifact_rows = []
    for seed in seeds:
        seed_dir = output_root / f"seed{seed}"
        gates_path = seed_dir / "gates.json"
        train_result_path = seed_dir / "train_result.json"
        memory_trace_path = seed_dir / "memory_trace.json"
        row = {
            "seed": seed,
            "seed_dir": str(seed_dir),
            "gates_exists": gates_path.exists(),
            "train_result_exists": train_result_path.exists(),
            "memory_trace_exists": memory_trace_path.exists(),
            "gates_path": str(gates_path),
            "train_result_path": str(train_result_path),
            "memory_trace_path": str(memory_trace_path),
        }
        if gates_path.exists():
            payload = json.loads(gates_path.read_text(encoding="utf-8"))
            row["all_gates_pass"] = bool(
                payload.get("diagnostics", {}).get("all_gates_pass", False),
            )
            row["gates"] = payload.get("gates", {})
        artifact_rows.append(row)

    artifact_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "seeds": seeds,
        "rows": artifact_rows,
        "aggregate_gates_path": str(output_root / "multiseed_gates.json"),
        "aggregate_report_path": str(output_root / "REPORT.md"),
    }
    artifact_output = output_root / "multiseed_artifacts.json"
    artifact_output.write_text(
        json.dumps(artifact_payload, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", artifact_output)


if __name__ == "__main__":
    main()
