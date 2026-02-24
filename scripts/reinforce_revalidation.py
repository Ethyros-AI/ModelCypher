#!/usr/bin/env python3
"""REINFORCE frontier runner and multiseed aggregator.

Track A workflow:
1. Run one (mode, seed) training job with explicit model/data inputs.
2. Repeat across modes/seeds.
3. Run aggregation mode to compute paired bootstrap CIs and verdicts.

Usage:
    # Single run
    poetry run python scripts/reinforce_revalidation.py \
      --mode force_reinforce --seed 41

    # Aggregate existing runs
    poetry run python scripts/reinforce_revalidation.py \
      --aggregate-root results/reinforce_frontier_1p2b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
TRAIN_DATA_DEFAULT = "data/training/1p2b_reasoning_foundation_train.jsonl"
EVAL_DATA_DEFAULT = "data/training/1p2b_reasoning_foundation_val.jsonl"
RETENTION_DATA_DEFAULT = "data/training/retention_replay.jsonl"
OUTPUT_ROOT_DEFAULT = Path("results/reinforce_frontier_1p2b")


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for one training mode."""

    mode: str
    description: str
    auto_regime: bool
    outcome_training: bool
    online_eval: bool
    entropy_regularization: bool


MODE_CONFIGS: dict[str, ModeConfig] = {
    "ce_control": ModeConfig(
        mode="ce_control",
        description="CE-only control with fixed online evaluation",
        auto_regime=False,
        outcome_training=False,
        online_eval=True,
        entropy_regularization=False,
    ),
    "auto_regime": ModeConfig(
        mode="auto_regime",
        description="Production path with baseline-derived regime selection",
        auto_regime=True,
        outcome_training=False,
        online_eval=True,
        entropy_regularization=False,
    ),
    "force_reinforce": ModeConfig(
        mode="force_reinforce",
        description="Forced REINFORCE with entropy regularization",
        auto_regime=False,
        outcome_training=True,
        online_eval=True,
        entropy_regularization=True,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run REINFORCE frontier experiments or aggregate multiseed results."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH_DEFAULT,
        help="Path to model directory.",
    )
    parser.add_argument(
        "--train-data",
        default=TRAIN_DATA_DEFAULT,
        help="Training dataset (JSONL).",
    )
    parser.add_argument(
        "--eval-data",
        default=EVAL_DATA_DEFAULT,
        help="Evaluation dataset (JSONL).",
    )
    parser.add_argument(
        "--retention-data",
        default=RETENTION_DATA_DEFAULT,
        help=(
            "Retention replay dataset (JSONL). "
            "Provide empty string to disable retention replay."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CONFIGS.keys()),
        default="auto_regime",
        help="Training mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Maximum training iterations.",
    )
    parser.add_argument(
        "--regime-n",
        type=int,
        default=25,
        help="Problem count for regime/outcome derivation.",
    )
    parser.add_argument(
        "--online-eval-n",
        type=int,
        default=25,
        help="Online evaluation problem count.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Sub-epoch online evaluation interval.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--outcome-post-eval",
        action="store_true",
        help="Run an extra online eval immediately after REINFORCE updates.",
    )
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        default=None,
        help="If set, run aggregation on this root and skip training.",
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
    return logging.getLogger("reinforce_frontier")


def _scan_budget_sources(log_path: Path) -> tuple[dict[str, int], int]:
    source_counts: Counter[str] = Counter()
    exhausted_hits = 0
    source_pattern = re.compile(r"\(([^)]+)\)\s*$")

    if not log_path.exists():
        return {}, 0

    with log_path.open(encoding="utf-8") as handle:
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
            "outcome_ce_reinforce_orth_fraction_mean": em.get(
                "outcome_ce_reinforce_orth_fraction_mean",
            ),
            "outcome_ce_reinforce_neg_parallel_fraction_mean": em.get(
                "outcome_ce_reinforce_neg_parallel_fraction_mean",
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


def _mode_train_kwargs(
    mode_config: ModeConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_iters": args.max_iters,
        "seed": args.seed,
        "auto_regime": mode_config.auto_regime,
        "eval_interval": args.eval_interval,
        "online_eval": mode_config.online_eval,
        "online_eval_n_problems": args.online_eval_n,
        "entropy_regularization": mode_config.entropy_regularization,
        "outcome_training": mode_config.outcome_training,
        "outcome_post_eval": args.outcome_post_eval,
    }

    if mode_config.auto_regime:
        kwargs["regime_n_problems"] = args.regime_n
    else:
        kwargs["regime_n_problems"] = None

    if mode_config.outcome_training:
        kwargs["outcome_n_problems"] = args.regime_n

    return kwargs


def _run_single(args: argparse.Namespace) -> None:
    mode_config = MODE_CONFIGS[args.mode]

    model_path = Path(args.model_path).expanduser().resolve()
    train_data = Path(args.train_data).expanduser().resolve()
    eval_data = Path(args.eval_data).expanduser().resolve()
    retention_data = (
        Path(args.retention_data).expanduser().resolve()
        if args.retention_data
        else None
    )

    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not train_data.exists():
        print(f"Train dataset does not exist: {train_data}", file=sys.stderr)
        sys.exit(1)
    if not eval_data.exists():
        print(f"Eval dataset does not exist: {eval_data}", file=sys.stderr)
        sys.exit(1)
    if retention_data is not None and not retention_data.exists():
        print(
            f"Retention dataset does not exist: {retention_data}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.mode / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = output_dir / "adapter"
    metrics_path = output_dir / "metrics.jsonl"
    run_log_path = output_dir / "run_log.json"
    train_log_path = output_dir / "train.log"

    log = _configure_logging(train_log_path)

    start_ts = datetime.now(timezone.utc).isoformat()
    log.info("=" * 72)
    log.info("REINFORCE FRONTIER RUN")
    log.info("=" * 72)
    log.info("Start: %s", start_ts)
    log.info("Mode: %s", args.mode)
    log.info("Description: %s", mode_config.description)
    log.info("Model: %s", model_path)
    log.info("Train data: %s", train_data)
    log.info("Eval data: %s", eval_data)
    log.info("Retention data: %s", retention_data if retention_data else "<disabled>")
    log.info("Output dir: %s", output_dir)

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()
    train_kwargs = _mode_train_kwargs(mode_config, args)

    log.info("Train kwargs: %s", json.dumps(train_kwargs, sort_keys=True))

    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=str(model_path),
        dataset_path=str(train_data),
        eval_dataset_path=str(eval_data),
        retention_dataset_path=(str(retention_data) if retention_data else None),
        output_path=str(adapter_path),
        **train_kwargs,
    )
    elapsed = time.monotonic() - t0

    result_dict = result.to_dict()
    epoch_metrics = result_dict.get("epoch_metrics", [])

    with metrics_path.open("w", encoding="utf-8") as handle:
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
    any_degradation = any(
        bool(ev.get("degraded", False)) for ev in online_eval_history
    )

    source_counts_log, exhausted_log_hits = _scan_budget_sources(train_log_path)

    final_eval = online_eval_history[-1] if online_eval_history else None
    final_n_correct = final_eval.get("n_correct") if final_eval else None
    final_n_total = final_eval.get("n_total") if final_eval else None

    run_log = {
        "experiment": "reinforce_frontier",
        "timestamp": start_ts,
        "mode": args.mode,
        "mode_description": mode_config.description,
        "model_path": str(model_path),
        "train_data": str(train_data),
        "eval_data": str(eval_data),
        "retention_data": (str(retention_data) if retention_data else None),
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "kwargs_sent": train_kwargs,
        "train_iters": result.train_iters,
        "stop_reason": result.stop_reason,
        "initial_loss": result_dict.get("initial_loss"),
        "final_loss": result_dict.get("final_loss"),
        "baseline_loss": result_dict.get("baseline_loss"),
        "post_loss": result_dict.get("post_loss"),
        "online_eval_history": online_eval_history,
        "final_online_eval": final_eval,
        "final_correct": final_n_correct,
        "final_total": final_n_total,
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
            "no_degradation_all_checkpoints": not any_degradation,
            "online_eval_available": final_eval is not None,
            "reinforce_ran": reinforce_ran,
        },
        "preregistered_decision_rule": {
            "primary_statistic": (
                "delta_final_accuracy = "
                "(final_correct(treatment)/N_eval) - "
                "(final_correct(ce_control)/N_eval)"
            ),
            "ci_method": "bootstrap CI on paired seed accuracy deltas",
            "equivalence_margin": (
                "delta = 1 / N_eval (one-problem resolution, rate units)"
            ),
            "verdict_rules": {
                "UNLOCKED": (
                    "ci_lower > 0 AND total_outcome_steps > 0 AND "
                    "no_degradation_all_checkpoints"
                ),
                "CEILING": "TOST-equivalence in [-delta, +delta] passes",
                "INCONCLUSIVE": "otherwise",
            },
        },
    }

    with run_log_path.open("w", encoding="utf-8") as handle:
        json.dump(run_log, handle, indent=2)

    log.info("Run complete")
    log.info("Stop reason: %s", result.stop_reason)
    log.info("Train iters: %d", result.train_iters)
    log.info("Elapsed: %.1f sec", elapsed)
    log.info("Final online eval: %s/%s", final_n_correct, final_n_total)
    log.info("REINFORCE ran: %s (steps=%d)", reinforce_ran, total_outcome_steps)
    log.info("Run log: %s", run_log_path)
    log.info("Metrics: %s", metrics_path)
    log.info("Train log: %s", train_log_path)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int | None = None,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of values."""
    if not values:
        raise ValueError("values must be non-empty")

    n = len(values)
    if n_bootstrap is None:
        n_bootstrap = max(1, n * n)

    mean_val = sum(values) / n

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        resampled = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(resampled) / n)

    samples.sort()
    lo_idx = max(0, int(math.floor(n_bootstrap * (alpha / 2))))
    hi_idx = min(n_bootstrap - 1, int(math.ceil(n_bootstrap * (1 - alpha / 2))) - 1)
    return mean_val, samples[lo_idx], samples[hi_idx]


def _collect_run_logs(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    by_mode: dict[str, dict[int, dict[str, Any]]] = {}

    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode_runs: dict[int, dict[str, Any]] = {}
        for seed_dir in sorted(mode_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            run_log_path = seed_dir / "run_log.json"
            if not run_log_path.exists():
                continue
            with run_log_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            seed = payload.get("seed")
            if not isinstance(seed, int):
                continue
            mode_runs[seed] = payload
        if mode_runs:
            by_mode[mode_dir.name] = mode_runs

    return by_mode


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _aggregate(aggregate_root: Path) -> None:
    root = aggregate_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Aggregate root does not exist: {root}")

    run_logs = _collect_run_logs(root)
    if "ce_control" not in run_logs:
        raise ValueError("Aggregation requires ce_control runs under aggregate root")

    ce_runs = run_logs["ce_control"]
    ce_seeds = set(ce_runs.keys())

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_root": str(root),
        "modes_found": sorted(run_logs.keys()),
        "preregistered_decision_rule": {
            "primary_statistic": (
                "delta_final_accuracy = "
                "(final_correct(treatment)/N_eval) - "
                "(final_correct(ce_control)/N_eval)"
            ),
            "ci_method": "bootstrap CI on paired seed accuracy deltas",
            "equivalence_margin": (
                "delta = 1 / N_eval (one-problem resolution, rate units)"
            ),
            "verdicts": ["UNLOCKED", "CEILING", "INCONCLUSIVE"],
        },
        "comparisons": {},
    }

    for mode_name, mode_runs in sorted(run_logs.items()):
        if mode_name == "ce_control":
            continue

        common_seeds = sorted(ce_seeds & set(mode_runs.keys()))
        if not common_seeds:
            summary["comparisons"][mode_name] = {
                "status": "missing_overlap",
                "common_seeds": [],
                "verdict": "INCONCLUSIVE",
            }
            continue

        delta_counts: list[float] = []
        delta_rates: list[float] = []
        final_correct_pairs: list[dict[str, Any]] = []
        outcome_steps: list[int] = []
        no_degradation_flags: list[bool] = []
        eval_totals: list[int] = []

        for seed in common_seeds:
            ce_log = ce_runs[seed]
            tr_log = mode_runs[seed]

            ce_correct = _safe_int(ce_log.get("final_correct"))
            tr_correct = _safe_int(tr_log.get("final_correct"))
            ce_total = _safe_int(ce_log.get("final_total"))
            tr_total = _safe_int(tr_log.get("final_total"))

            if (
                ce_correct is None
                or tr_correct is None
                or ce_total is None
                or tr_total is None
                or ce_total <= 0
                or tr_total <= 0
            ):
                continue

            ce_accuracy = float(ce_correct) / float(ce_total)
            tr_accuracy = float(tr_correct) / float(tr_total)

            delta_count = float(tr_correct - ce_correct)
            delta_rate = tr_accuracy - ce_accuracy

            delta_counts.append(delta_count)
            delta_rates.append(delta_rate)
            final_correct_pairs.append({
                "seed": seed,
                "ce_control_final_correct": ce_correct,
                "treatment_final_correct": tr_correct,
                "ce_control_final_total": ce_total,
                "treatment_final_total": tr_total,
                "ce_control_final_accuracy": ce_accuracy,
                "treatment_final_accuracy": tr_accuracy,
                "delta_final_correct": delta_count,
                "delta_final_accuracy": delta_rate,
            })

            eval_totals.extend([ce_total, tr_total])

            rs = tr_log.get("reinforce_summary", {})
            outcome_steps.append(int(rs.get("total_outcome_steps", 0)))

            sc = tr_log.get("success_criteria", {})
            no_degradation_flags.append(
                bool(sc.get("no_degradation_all_checkpoints", False)),
            )

        if not delta_rates:
            summary["comparisons"][mode_name] = {
                "status": "missing_final_accuracy",
                "common_seeds": common_seeds,
                "verdict": "INCONCLUSIVE",
            }
            continue

        mean_delta_count, ci_lower_count, ci_upper_count = _bootstrap_mean_ci(
            delta_counts,
            alpha=0.05,
            n_bootstrap=max(1, len(delta_counts) * len(delta_counts)),
            seed=20260223,
        )
        mean_delta_rate, ci_lower_rate, ci_upper_rate = _bootstrap_mean_ci(
            delta_rates,
            alpha=0.05,
            n_bootstrap=max(1, len(delta_rates) * len(delta_rates)),
            seed=20260223,
        )

        valid_eval_totals = [n for n in eval_totals if n > 0]
        n_eval = min(valid_eval_totals) if valid_eval_totals else 0
        delta_margin = (1.0 / float(n_eval)) if n_eval > 0 else None
        rate_tolerance = math.sqrt(sys.float_info.epsilon)
        tost_pass = (
            delta_margin is not None
            and ci_lower_rate >= (-delta_margin - rate_tolerance)
            and ci_upper_rate <= (delta_margin + rate_tolerance)
        )

        reinforce_ran_all = all(step > 0 for step in outcome_steps) if outcome_steps else False
        no_degradation_all = all(no_degradation_flags) if no_degradation_flags else False

        if ci_lower_rate > 0.0 and reinforce_ran_all and no_degradation_all:
            verdict = "UNLOCKED"
        elif tost_pass:
            verdict = "CEILING"
        else:
            verdict = "INCONCLUSIVE"

        summary["comparisons"][mode_name] = {
            "status": "ok",
            "common_seeds": common_seeds,
            "paired_final_correct": final_correct_pairs,
            "delta_final_correct": {
                "point_estimate": mean_delta_count,
                "ci_lower": ci_lower_count,
                "ci_upper": ci_upper_count,
            },
            "delta_final_accuracy": {
                "point_estimate": mean_delta_rate,
                "ci_lower": ci_lower_rate,
                "ci_upper": ci_upper_rate,
            },
            "equivalence_margin": {
                "n_eval": n_eval,
                "delta": delta_margin,
                "units": "accuracy_rate",
                "tolerance": rate_tolerance,
            },
            "tost_equivalence": {
                "passes": tost_pass,
                "interval": (
                    [-delta_margin, delta_margin]
                    if delta_margin is not None
                    else None
                ),
            },
            "reinforce_ran_all_seeds": reinforce_ran_all,
            "no_degradation_all_seeds": no_degradation_all,
            "total_outcome_steps_by_seed": {
                str(seed): int(
                    mode_runs[seed].get("reinforce_summary", {}).get(
                        "total_outcome_steps", 0,
                    ),
                )
                for seed in common_seeds
            },
            "verdict": verdict,
        }

    summary_path = root / "multiseed_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report_lines = [
        "# REINFORCE Frontier 1.2B Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Aggregate root: {summary['aggregate_root']}",
        "",
        "## Pre-Registered Rule",
        "",
        "- Primary statistic: `delta_final_accuracy = (final_correct(treatment)/N_eval) - (final_correct(ce_control)/N_eval)`",
        "- CI method: bootstrap CI on paired seed accuracy deltas",
        "- Equivalence margin: `delta = 1 / N_eval` (rate units)",
        "- Verdicts: `UNLOCKED` / `CEILING` / `INCONCLUSIVE`",
        "",
        "## Mode Verdicts",
        "",
        "| Mode | Seeds | Accuracy Delta CI | Margin | Verdict |",
        "|------|-------|----------|--------|---------|",
    ]

    for mode_name, payload in sorted(summary["comparisons"].items()):
        if payload.get("status") != "ok":
            report_lines.append(
                f"| {mode_name} | 0 | n/a | n/a | {payload.get('verdict', 'INCONCLUSIVE')} |",
            )
            continue

        ci = payload["delta_final_accuracy"]
        margin = payload["equivalence_margin"]["delta"]
        ci_text = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        margin_text = f"±{margin:.4f}" if margin is not None else "n/a"
        report_lines.append(
            f"| {mode_name} | {len(payload['common_seeds'])} | {ci_text} | {margin_text} | {payload['verdict']} |",
        )

    report_path = root / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")


def main() -> None:
    args = _parse_args()
    if args.aggregate_root is not None:
        _aggregate(args.aggregate_root)
        return

    _run_single(args)


if __name__ == "__main__":
    main()
