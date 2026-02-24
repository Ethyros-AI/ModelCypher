#!/usr/bin/env python3
"""Execute Unlock Cycle U1 (E1 -> E2 -> E3) for REINFORCE 1.2B frontier.

This script orchestrates three sequential phases using
``scripts/reinforce_revalidation.py``:

E1: Gate-stage causal test (pre_outcome vs post_outcome).
E2: Credit targeting (all vs lost_only).
E3: Unlock confirmation matrix (ce_control, auto_regime, best_force).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REVALIDATION_SCRIPT = REPO_ROOT / "scripts" / "reinforce_revalidation.py"
DEFAULT_MODEL = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
DEFAULT_TRAIN = "data/training/1p2b_reasoning_foundation_train.jsonl"
DEFAULT_EVAL = "data/training/1p2b_reasoning_foundation_val.jsonl"
DEFAULT_RETENTION = "data/training/retention_replay.jsonl"
DEFAULT_OUTPUT_ROOT = Path("results/reinforce_unlock_cycle_u1")


@dataclass(frozen=True)
class PairedCI:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_pairs: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Unlock Cycle U1 frontier phases with pre-registered gates.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL)
    parser.add_argument("--retention-data", default=DEFAULT_RETENTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--regime-n", type=int, default=100)
    parser.add_argument("--online-eval-n", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument(
        "--phase",
        choices=["all", "e1", "e2", "e3"],
        default="all",
        help="Execute a single phase or the full cycle.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip arm/seed runs when run_log.json already exists.",
    )
    return parser.parse_args()


def _parse_seed_list(seed_csv: str) -> list[int]:
    seeds: list[int] = []
    for token in seed_csv.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_arm_seed(
    *,
    args: argparse.Namespace,
    phase_root: Path,
    arm_name: str,
    mode: str,
    stop_stage: str,
    outcome_selector: str,
    seed: int,
) -> None:
    run_log_path = phase_root / arm_name / f"seed{seed}" / "run_log.json"
    if args.skip_existing and run_log_path.exists():
        print(f"[skip] {arm_name} seed{seed} (existing run_log)")
        return

    cmd = [
        sys.executable,
        str(REVALIDATION_SCRIPT),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--model-path",
        args.model_path,
        "--train-data",
        args.train_data,
        "--eval-data",
        args.eval_data,
        "--retention-data",
        args.retention_data,
        "--output-root",
        str(phase_root),
        "--arm-name",
        arm_name,
        "--max-iters",
        str(args.max_iters),
        "--regime-n",
        str(args.regime_n),
        "--online-eval-n",
        str(args.online_eval_n),
        "--eval-interval",
        str(args.eval_interval),
        "--research-online-eval-stop-stage",
        stop_stage,
        "--research-outcome-selector",
        outcome_selector,
        "--outcome-post-eval",
    ]
    print(f"[run] {arm_name} seed{seed}")
    _run_cmd(cmd, cwd=REPO_ROOT)


def _aggregate(phase_root: Path, baseline_arm: str = "ce_control") -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REVALIDATION_SCRIPT),
        "--aggregate-root",
        str(phase_root),
        "--aggregate-baseline-arm",
        baseline_arm,
    ]
    _run_cmd(cmd, cwd=REPO_ROOT)
    summary_path = phase_root / "multiseed_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _bootstrap_mean_ci(values: list[float]) -> PairedCI:
    if not values:
        raise ValueError("values must be non-empty")

    n = len(values)
    n_bootstrap = max(1, n * n)
    mean_val = sum(values) / n

    rng = random.Random(20260224)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(sample) / n)
    samples.sort()

    lo_idx = max(0, int(math.floor(n_bootstrap * 0.025)))
    hi_idx = min(n_bootstrap - 1, int(math.ceil(n_bootstrap * 0.975)) - 1)
    return PairedCI(
        point_estimate=mean_val,
        ci_lower=samples[lo_idx],
        ci_upper=samples[hi_idx],
        n_pairs=n,
    )


def _load_run_log(phase_root: Path, arm_name: str, seed: int) -> dict[str, Any]:
    run_log_path = phase_root / arm_name / f"seed{seed}" / "run_log.json"
    if not run_log_path.exists():
        raise FileNotFoundError(f"Missing run log: {run_log_path}")
    return json.loads(run_log_path.read_text(encoding="utf-8"))


def _paired_accuracy_ci(
    *,
    phase_root: Path,
    arm_a: str,
    arm_b: str,
    seeds: list[int],
    field_correct: str = "final_correct",
    field_total: str = "final_total",
) -> PairedCI:
    deltas: list[float] = []
    for seed in seeds:
        a_log = _load_run_log(phase_root, arm_a, seed)
        b_log = _load_run_log(phase_root, arm_b, seed)
        a_correct = a_log.get(field_correct)
        b_correct = b_log.get(field_correct)
        a_total = a_log.get(field_total)
        b_total = b_log.get(field_total)
        if (
            isinstance(a_correct, int)
            and isinstance(b_correct, int)
            and isinstance(a_total, int)
            and isinstance(b_total, int)
            and a_total > 0
            and b_total > 0
        ):
            deltas.append((float(a_correct) / float(a_total)) - (float(b_correct) / float(b_total)))
    if not deltas:
        raise ValueError(f"No paired deltas available for {arm_a} vs {arm_b}")
    return _bootstrap_mean_ci(deltas)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_e1(args: argparse.Namespace, seeds: list[int], output_root: Path) -> tuple[str, dict[str, Any]]:
    phase_root = output_root / "e1_gate_stage"
    arms = [
        ("ce_control", "ce_control", "pre_outcome", "all"),
        ("force_reinforce__stop_pre_outcome__selector_all", "force_reinforce", "pre_outcome", "all"),
        ("force_reinforce__stop_post_outcome__selector_all", "force_reinforce", "post_outcome", "all"),
    ]
    for arm_name, mode, stop_stage, selector in arms:
        for seed in seeds:
            _run_arm_seed(
                args=args,
                phase_root=phase_root,
                arm_name=arm_name,
                mode=mode,
                stop_stage=stop_stage,
                outcome_selector=selector,
                seed=seed,
            )

    summary = _aggregate(phase_root, baseline_arm="ce_control")
    h1_ci = _paired_accuracy_ci(
        phase_root=phase_root,
        arm_a="force_reinforce__stop_post_outcome__selector_all",
        arm_b="force_reinforce__stop_pre_outcome__selector_all",
        seeds=seeds,
    )
    h1_supported = h1_ci.ci_lower > 0.0
    winner_stage = "post_outcome" if h1_supported else "pre_outcome"

    decision = {
        "hypothesis": "H1 gate-lock",
        "paired_delta_accuracy": {
            "point_estimate": h1_ci.point_estimate,
            "ci_lower": h1_ci.ci_lower,
            "ci_upper": h1_ci.ci_upper,
            "n_pairs": h1_ci.n_pairs,
            "comparison": "post_outcome - pre_outcome",
        },
        "supported": h1_supported,
        "winner_stop_stage": winner_stage,
    }
    _write_json(phase_root / "h1_decision.json", decision)
    return winner_stage, {"aggregate_summary": summary, "h1_decision": decision}


def _run_e2(
    args: argparse.Namespace,
    seeds: list[int],
    output_root: Path,
    winner_stage: str,
) -> tuple[str, dict[str, Any]]:
    phase_root = output_root / "e2_credit_targeting"
    arm_all = f"force_reinforce__stop_{winner_stage}__selector_all"
    arm_lost = f"force_reinforce__stop_{winner_stage}__selector_lost_only"
    arms = [
        (arm_all, "force_reinforce", winner_stage, "all"),
        (arm_lost, "force_reinforce", winner_stage, "lost_only"),
    ]
    for arm_name, mode, stop_stage, selector in arms:
        for seed in seeds:
            _run_arm_seed(
                args=args,
                phase_root=phase_root,
                arm_name=arm_name,
                mode=mode,
                stop_stage=stop_stage,
                outcome_selector=selector,
                seed=seed,
            )

    h2_ci = _paired_accuracy_ci(
        phase_root=phase_root,
        arm_a=arm_lost,
        arm_b=arm_all,
        seeds=seeds,
    )
    h2_supported = h2_ci.ci_lower > 0.0
    winner_selector = "lost_only" if h2_supported else "all"

    decision = {
        "hypothesis": "H2 credit-targeting",
        "paired_delta_accuracy": {
            "point_estimate": h2_ci.point_estimate,
            "ci_lower": h2_ci.ci_lower,
            "ci_upper": h2_ci.ci_upper,
            "n_pairs": h2_ci.n_pairs,
            "comparison": "lost_only - all",
        },
        "supported": h2_supported,
        "winner_outcome_selector": winner_selector,
        "winner_stop_stage": winner_stage,
    }
    per_seed_pairs: list[dict[str, Any]] = []
    per_type_lost_gained_trajectories: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm_all: {},
        arm_lost: {},
    }
    for seed in seeds:
        all_log = _load_run_log(phase_root, arm_all, seed)
        lost_log = _load_run_log(phase_root, arm_lost, seed)
        all_correct = all_log.get("final_correct")
        all_total = all_log.get("final_total")
        lost_correct = lost_log.get("final_correct")
        lost_total = lost_log.get("final_total")
        if (
            isinstance(all_correct, int)
            and isinstance(all_total, int)
            and isinstance(lost_correct, int)
            and isinstance(lost_total, int)
            and all_total > 0
            and lost_total > 0
        ):
            all_acc = float(all_correct) / float(all_total)
            lost_acc = float(lost_correct) / float(lost_total)
            per_seed_pairs.append({
                "seed": seed,
                "all_final_accuracy": all_acc,
                "lost_only_final_accuracy": lost_acc,
                "delta_lost_only_minus_all": lost_acc - all_acc,
            })

        for arm_name, run_log in ((arm_all, all_log), (arm_lost, lost_log)):
            telemetry = run_log.get("epoch_budget_telemetry", [])
            trajectory: list[dict[str, Any]] = []
            for em in telemetry:
                trajectory.append({
                    "epoch": em.get("epoch"),
                    "online_eval_pre_n_lost": em.get("online_eval_pre_n_lost"),
                    "online_eval_pre_n_gained": em.get("online_eval_pre_n_gained"),
                    "online_eval_post_n_lost": em.get("online_eval_post_n_lost"),
                    "online_eval_post_n_gained": em.get("online_eval_post_n_gained"),
                    "online_eval_pre_per_type_correct": em.get(
                        "online_eval_pre_per_type_correct",
                    ),
                    "online_eval_pre_per_type_total": em.get(
                        "online_eval_pre_per_type_total",
                    ),
                    "online_eval_post_per_type_correct": em.get(
                        "online_eval_post_per_type_correct",
                    ),
                    "online_eval_post_per_type_total": em.get(
                        "online_eval_post_per_type_total",
                    ),
                })
            per_type_lost_gained_trajectories[arm_name][str(seed)] = trajectory

    e2_summary = {
        "phase": "e2_credit_targeting",
        "winner_stop_stage": winner_stage,
        "all_arm": arm_all,
        "lost_only_arm": arm_lost,
        "paired_delta_accuracy": {
            "point_estimate": h2_ci.point_estimate,
            "ci_lower": h2_ci.ci_lower,
            "ci_upper": h2_ci.ci_upper,
            "n_pairs": h2_ci.n_pairs,
        },
        "h2_supported": h2_supported,
        "winner_outcome_selector": winner_selector,
        "per_seed_pairs": per_seed_pairs,
        "per_type_lost_gained_trajectories": per_type_lost_gained_trajectories,
    }
    _write_json(phase_root / "h2_decision.json", decision)
    _write_json(phase_root / "e2_summary.json", e2_summary)
    report_lines = [
        "# E2 Credit Targeting Report",
        "",
        f"- Winner stop stage: `{winner_stage}`",
        f"- Arms: `{arm_all}` vs `{arm_lost}`",
        f"- H2 supported: `{h2_supported}`",
        (
            "- CI (lost_only - all): "
            f"[{h2_ci.ci_lower:.6f}, {h2_ci.ci_upper:.6f}]"
        ),
        f"- Winner selector: `{winner_selector}`",
        "",
        "Per-seed and per-type lost/gained trajectories are in `e2_summary.json`.",
    ]
    (phase_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return winner_selector, {"h2_decision": decision, "e2_summary": e2_summary}


def _run_e3(
    args: argparse.Namespace,
    seeds: list[int],
    output_root: Path,
    winner_stage: str,
    winner_selector: str,
) -> dict[str, Any]:
    phase_root = output_root / "e3_unlock_confirmation"
    force_arm = (
        f"force_reinforce__stop_{winner_stage}__selector_{winner_selector}"
    )
    auto_selector = "all"
    arms = [
        ("ce_control", "ce_control", winner_stage, "all"),
        (f"auto_regime__stop_{winner_stage}__selector_{auto_selector}", "auto_regime", winner_stage, auto_selector),
        (force_arm, "force_reinforce", winner_stage, winner_selector),
    ]
    for arm_name, mode, stop_stage, selector in arms:
        for seed in seeds:
            _run_arm_seed(
                args=args,
                phase_root=phase_root,
                arm_name=arm_name,
                mode=mode,
                stop_stage=stop_stage,
                outcome_selector=selector,
                seed=seed,
            )

    summary = _aggregate(phase_root, baseline_arm="ce_control")
    return {"aggregate_summary": summary, "force_arm": force_arm}


def _write_cycle_report(
    output_root: Path,
    *,
    winner_stage: str,
    winner_selector: str,
    e1: dict[str, Any],
    e2: dict[str, Any],
    e3: dict[str, Any],
) -> None:
    e1_decision = e1["h1_decision"]
    e2_decision = e2["h2_decision"]
    e3_summary = e3["aggregate_summary"]
    force_arm = e3["force_arm"]
    force_payload = e3_summary["comparisons"].get(force_arm, {})
    report_lines = [
        "# Unlock Cycle U1 Report",
        "",
        f"- Winner stop stage: `{winner_stage}`",
        f"- Winner outcome selector: `{winner_selector}`",
        "",
        "## H1 Gate-Lock",
        "",
        f"- Supported: `{e1_decision['supported']}`",
        (
            "- CI (post - pre): "
            f"[{e1_decision['paired_delta_accuracy']['ci_lower']:.6f}, "
            f"{e1_decision['paired_delta_accuracy']['ci_upper']:.6f}]"
        ),
        "",
        "## H2 Credit Targeting",
        "",
        f"- Supported: `{e2_decision['supported']}`",
        (
            "- CI (lost_only - all): "
            f"[{e2_decision['paired_delta_accuracy']['ci_lower']:.6f}, "
            f"{e2_decision['paired_delta_accuracy']['ci_upper']:.6f}]"
        ),
        "",
        "## E3 Unlock Verdicts",
        "",
        f"- Force arm: `{force_arm}`",
        f"- Canonical verdict: `{force_payload.get('canonical_verdict', 'INCONCLUSIVE')}`",
        f"- Mechanistic verdict: `{force_payload.get('mechanistic_verdict', 'INCONCLUSIVE')}`",
        (
            "- Gate confound events: "
            f"`{force_payload.get('gate_confound_event_count_total', 0)}`"
        ),
        "",
        (
            "See per-phase artifacts under "
            "`e1_gate_stage/`, `e2_credit_targeting/`, and "
            "`e3_unlock_confirmation/`."
        ),
    ]
    report_path = output_root / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    winner_stage = "pre_outcome"
    winner_selector = "all"
    e1_artifacts: dict[str, Any] = {}
    e2_artifacts: dict[str, Any] = {}
    e3_artifacts: dict[str, Any] = {}

    if args.phase in {"all", "e1"}:
        winner_stage, e1_artifacts = _run_e1(args, seeds, output_root)
    else:
        h1_path = output_root / "e1_gate_stage" / "h1_decision.json"
        if h1_path.exists():
            winner_stage = json.loads(h1_path.read_text(encoding="utf-8")).get(
                "winner_stop_stage",
                "pre_outcome",
            )

    if args.phase in {"all", "e2"}:
        winner_selector, e2_artifacts = _run_e2(
            args,
            seeds,
            output_root,
            winner_stage,
        )
    else:
        h2_path = output_root / "e2_credit_targeting" / "h2_decision.json"
        if h2_path.exists():
            winner_selector = json.loads(h2_path.read_text(encoding="utf-8")).get(
                "winner_outcome_selector",
                "all",
            )

    if args.phase in {"all", "e3"}:
        e3_artifacts = _run_e3(
            args,
            seeds,
            output_root,
            winner_stage,
            winner_selector,
        )

    if args.phase == "all":
        _write_cycle_report(
            output_root,
            winner_stage=winner_stage,
            winner_selector=winner_selector,
            e1=e1_artifacts,
            e2=e2_artifacts,
            e3=e3_artifacts,
        )
        print(f"Wrote cycle report: {output_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
