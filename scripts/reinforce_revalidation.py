#!/usr/bin/env python3
"""REINFORCE re-validation runner for LFM2-350M.

This script is a parameterized variant of `mass_reinforce_run.py` for
re-validation experiments after Weyl remainder-budget changes.

Usage:
    poetry run python scripts/reinforce_revalidation.py --eval-interval 10 --run-id 2
    poetry run python scripts/reinforce_revalidation.py --regime-n 50 --eval-interval 10 --run-id 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
EVAL_DATASET_PATH = "data/training/benchmark_val.jsonl"
DEFAULT_OUTPUT_ROOT = Path("/Volumes/CodeCypher/models/experiments/reinforce-revalidation")
BASELINE_CORRECT = 18


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run REINFORCE re-validation with configurable eval interval/problem count.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=2,
        help="Run index for output directory naming (run{N}). Default: 2",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Sub-epoch evaluation interval passed to train_from_dataset. Default: 10",
    )
    parser.add_argument(
        "--regime-n",
        type=int,
        default=25,
        help="Number of problems for auto_regime baseline/outcome derivation. Default: 25",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Maximum training iterations. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed. Default: 42",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory for run artifacts.",
    )
    parser.add_argument(
        "--outcome-post-eval",
        action="store_true",
        help="Run an extra online eval immediately after REINFORCE updates.",
    )
    return parser.parse_args()


def _configure_logging(log_path: Path) -> logging.Logger:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    return logging.getLogger("reinforce_revalidation")


def _scan_budget_sources(log_path: Path) -> tuple[dict[str, int], int]:
    source_counts: Counter[str] = Counter()
    exhausted_hits = 0
    source_pattern = re.compile(r"\(([^)]+)\)\s*$")

    if not log_path.exists():
        return {}, 0

    with log_path.open() as handle:
        for line in handle:
            if "Weyl budget exhausted" in line:
                exhausted_hits += 1
            if "REINFORCE budget:" in line:
                match = source_pattern.search(line.strip())
                if match:
                    source_counts[match.group(1)] += 1

    return dict(source_counts), exhausted_hits


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _build_epoch_telemetry(epoch_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    telemetry: list[dict[str, Any]] = []
    for em in epoch_metrics:
        telemetry.append({
            "epoch": em.get("epoch"),
            "online_eval_accuracy": em.get("online_eval_accuracy"),
            "online_eval_n_correct": em.get("online_eval_n_correct"),
            "online_eval_n_total": em.get("online_eval_n_total"),
            "online_eval_degraded": em.get("online_eval_degraded"),
            "outcome_n_problems": em.get("outcome_n_problems"),
            "outcome_n_active": em.get("outcome_n_active"),
            "outcome_signal_density": em.get("outcome_signal_density"),
            "outcome_n_steps": em.get("outcome_n_steps"),
            "outcome_target_step_norm": em.get("outcome_target_step_norm"),
            "outcome_target_step_source": em.get("outcome_target_step_source"),
            "outcome_o_eta": em.get("outcome_o_eta"),
            "outcome_o_grad_norm": em.get("outcome_o_grad_norm"),
            "outcome_ce_grad_norm": em.get("outcome_ce_grad_norm"),
            "outcome_ce_reinforce_cosine_mean": em.get("outcome_ce_reinforce_cosine_mean"),
            "outcome_ce_reinforce_cosine_last": em.get("outcome_ce_reinforce_cosine_last"),
            "outcome_ce_reinforce_cosine_n": em.get("outcome_ce_reinforce_cosine_n"),
            "outcome_ce_reinforce_orth_fraction_mean": em.get(
                "outcome_ce_reinforce_orth_fraction_mean",
            ),
            "outcome_ce_reinforce_orth_fraction_last": em.get(
                "outcome_ce_reinforce_orth_fraction_last",
            ),
            "outcome_ce_reinforce_neg_parallel_fraction_mean": em.get(
                "outcome_ce_reinforce_neg_parallel_fraction_mean",
            ),
            "outcome_ce_reinforce_neg_parallel_fraction_last": em.get(
                "outcome_ce_reinforce_neg_parallel_fraction_last",
            ),
            "outcome_post_eval_accuracy": em.get("outcome_post_eval_accuracy"),
            "outcome_post_eval_n_correct": em.get("outcome_post_eval_n_correct"),
            "outcome_post_eval_n_total": em.get("outcome_post_eval_n_total"),
            "outcome_post_eval_degraded": em.get("outcome_post_eval_degraded"),
            "outcome_post_eval_delta_correct": em.get("outcome_post_eval_delta_correct"),
            "outcome_budget_remaining": em.get("outcome_budget_remaining"),
            "adapter_saturation_median_ratio": em.get("adapter_saturation_median_ratio"),
            "eta_ceiling": em.get("eta_ceiling"),
            "eta_step": em.get("eta_step"),
            "eta_weyl": em.get("eta_weyl"),
            "d_norm": em.get("d_norm"),
        })
    return telemetry


def main() -> None:
    args = _parse_args()
    if not Path(MODEL_PATH).exists():
        print(f"Model path does not exist: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"run{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = output_dir / "adapter"
    metrics_path = output_dir / "metrics.jsonl"
    run_log_path = output_dir / "run_log.json"
    train_log_path = output_dir / "train.log"

    log = _configure_logging(train_log_path)

    start_ts = datetime.now(timezone.utc).isoformat()
    log.info("=" * 60)
    log.info("REINFORCE RE-VALIDATION RUN %d", args.run_id)
    log.info("=" * 60)
    log.info("Start: %s", start_ts)
    log.info("Model: %s", MODEL_PATH)
    log.info("Dataset: %s", DATASET_PATH)
    log.info("Eval dataset: %s", EVAL_DATASET_PATH)
    log.info("Output dir: %s", output_dir)
    log.info(
        "Config: auto_regime=True regime_n=%d eval_interval=%d max_iters=%d seed=%d "
        "outcome_post_eval=%s",
        args.regime_n,
        args.eval_interval,
        args.max_iters,
        args.seed,
        args.outcome_post_eval,
    )

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()

    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        eval_dataset_path=EVAL_DATASET_PATH,
        output_path=str(adapter_path),
        max_iters=args.max_iters,
        seed=args.seed,
        auto_regime=True,
        regime_n_problems=args.regime_n,
        eval_interval=args.eval_interval,
        outcome_post_eval=args.outcome_post_eval,
    )
    elapsed = time.monotonic() - t0

    result_dict = result.to_dict()
    epoch_metrics = result_dict.get("epoch_metrics", [])

    with metrics_path.open("w") as handle:
        for em in epoch_metrics:
            handle.write(json.dumps(em) + "\n")

    online_eval_history: list[dict[str, Any]] = []
    for em in epoch_metrics:
        if em.get("online_eval_accuracy") is not None:
            online_eval_history.append({
                "epoch": em.get("epoch"),
                "accuracy": em.get("online_eval_accuracy"),
                "n_correct": em.get("online_eval_n_correct"),
                "n_total": em.get("online_eval_n_total"),
                "degraded": em.get("online_eval_degraded"),
            })

    epoch_telemetry = _build_epoch_telemetry(epoch_metrics)
    log.info("")
    log.info("Structured epoch telemetry:")
    for row in epoch_telemetry:
        log.info("EPOCH_TELEMETRY %s", json.dumps(row, sort_keys=True))

    source_counts_metrics = Counter(
        em.get("outcome_target_step_source")
        for em in epoch_metrics
        if em.get("outcome_target_step_source")
    )
    budget_remaining_values = [
        float(em["outcome_budget_remaining"])
        for em in epoch_metrics
        if em.get("outcome_budget_remaining") is not None
    ]
    signal_density_values = [
        float(em["outcome_signal_density"])
        for em in epoch_metrics
        if em.get("outcome_signal_density") is not None
    ]
    outcome_step_values = [
        int(em["outcome_n_steps"])
        for em in epoch_metrics
        if em.get("outcome_n_steps") is not None
    ]
    total_outcome_steps = sum(outcome_step_values)
    reinforce_ran = total_outcome_steps > 0

    source_counts_log, exhausted_log_hits = _scan_budget_sources(train_log_path)

    if online_eval_history:
        final_eval = online_eval_history[-1]
        final_n_correct = final_eval.get("n_correct")
        final_above_18 = (
            final_n_correct is not None and final_n_correct > BASELINE_CORRECT
        )
        final_at_or_above_18 = (
            final_n_correct is not None and final_n_correct >= BASELINE_CORRECT
        )
    else:
        final_n_correct = None
        final_above_18 = False
        final_at_or_above_18 = False

    run_log = {
        "experiment": f"reinforce-revalidation-run{args.run_id}",
        "description": "Auto-regime REINFORCE re-validation with Weyl remainder budget",
        "timestamp": start_ts,
        "model_path": MODEL_PATH,
        "seed": args.seed,
        "kwargs_sent": {
            "auto_regime": True,
            "regime_n_problems": args.regime_n,
            "eval_interval": args.eval_interval,
            "max_iters": args.max_iters,
            "lr_override": None,
            "outcome_post_eval": args.outcome_post_eval,
        },
        "elapsed_seconds": elapsed,
        "train_iters": result.train_iters,
        "stop_reason": result.stop_reason,
        "initial_loss": result_dict.get("initial_loss"),
        "final_loss": result_dict.get("final_loss"),
        "baseline_loss": result_dict.get("baseline_loss"),
        "post_loss": result_dict.get("post_loss"),
        "online_eval_history": online_eval_history,
        "n_epoch_metrics": len(epoch_metrics),
        "epoch_budget_telemetry": epoch_telemetry,
        "reinforce_summary": {
            "reinforce_ran": reinforce_ran,
            "total_outcome_steps": total_outcome_steps,
            "mean_signal_density": _mean_or_none(signal_density_values),
            "mean_budget_remaining": _mean_or_none(budget_remaining_values),
            "target_step_source_counts_from_metrics": dict(source_counts_metrics),
            "target_step_source_counts_from_train_log": source_counts_log,
            "weyl_budget_exhausted_log_hits": exhausted_log_hits,
        },
        "success_criteria": {
            "no_degradation_all_checkpoints": all(
                not ev.get("degraded", False) for ev in online_eval_history
            ),
            "final_at_or_above_18": final_at_or_above_18,
            "final_above_18": final_above_18,
            "reinforce_ran": reinforce_ran,
        },
    }

    with run_log_path.open("w") as handle:
        json.dump(run_log, handle, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info("Stop reason: %s", result.stop_reason)
    log.info("Train iters: %d", result.train_iters)
    log.info("Elapsed: %.1f sec", elapsed)
    log.info("REINFORCE ran: %s (steps=%d)", reinforce_ran, total_outcome_steps)
    log.info("Budget sources (metrics): %s", dict(source_counts_metrics))
    log.info("Budget sources (train log): %s", source_counts_log)
    log.info("Weyl budget exhausted hits (train log): %d", exhausted_log_hits)
    if online_eval_history:
        final = online_eval_history[-1]
        log.info("Final online eval: %s/%s (%.1f%%)",
                 final.get("n_correct"), final.get("n_total"),
                 float(final.get("accuracy", 0.0)) * 100)
    else:
        log.info("No online eval checkpoints captured.")
    log.info("Run log: %s", run_log_path)
    log.info("Metrics: %s", metrics_path)
    log.info("Train log: %s", train_log_path)


if __name__ == "__main__":
    main()
