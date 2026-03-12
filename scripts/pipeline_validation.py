#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Multi-model pipeline validation runner.

Runs DerivedTrainingValidationService.validate() across 350M, 1.2B, and 8B
model scales. Collects results, writes a JSON verdict and markdown report.

Usage:
    # Smoke test (1 trial, 350M only)
    poetry run python scripts/pipeline_validation.py --model-scale 350M --trials 1

    # Full 350M (5 trials)
    poetry run python scripts/pipeline_validation.py --model-scale 350M

    # Full 1.2B
    poetry run python scripts/pipeline_validation.py --model-scale 1p2B

    # All three models (sequential)
    poetry run python scripts/pipeline_validation.py

    # Custom output directory
    poetry run python scripts/pipeline_validation.py --output-root results/my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model scale."""

    scale: str
    model_path: str
    train_data: str
    eval_data: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "350M": ModelConfig(
        scale="350M",
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        train_data="data/training/benchmark_train.jsonl",
        eval_data="data/training/benchmark_val.jsonl",
    ),
    "1p2B": ModelConfig(
        scale="1p2B",
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        train_data="data/training/1p2b_reasoning_foundation_train.jsonl",
        eval_data="data/training/1p2b_reasoning_foundation_val.jsonl",
    ),
    "8B": ModelConfig(
        scale="8B",
        model_path="/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        train_data="data/training/1p2b_reasoning_foundation_train.jsonl",
        eval_data="data/training/1p2b_reasoning_foundation_val.jsonl",
    ),
}

SCALE_ORDER = ["350M", "1p2B", "8B"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run derived-training validation across model scales.",
    )
    parser.add_argument(
        "--model-scale",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Single model scale to validate (default: all three, sequential).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of seeded trials per model (default: 5).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/pipeline_validation"),
        help="Root directory for results (default: results/pipeline_validation).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the configured model path for the selected --model-scale.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Override the configured training dataset for the selected --model-scale.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Override the configured eval dataset for the selected --model-scale.",
    )
    parser.add_argument(
        "--benchmark-suite",
        default=None,
        help="Optional benchmark suite (quick, reasoning, factual, comprehensive).",
    )
    parser.add_argument(
        "--controller-mode",
        default="mass_structural_observe",
        help=(
            "Research controller mode "
            "(mass_structural_observe, mass_behavioral_probe, mass_behavioral_closed_loop)."
        ),
    )
    parser.add_argument(
        "--optimizer-research-mode",
        default="cayley_stiefel_mass",
        help="Research optimizer mode (cayley_stiefel_mass, adamw_matched_trace).",
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
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    return logging.getLogger("pipeline_validation")


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _check_volume() -> None:
    volume = Path("/Volumes/CodeCypher/models/mlx-community")
    if not volume.exists():
        print(
            f"ERROR: Volume not mounted at {volume}\n"
            "Mount the external drive and retry.",
            file=sys.stderr,
        )
        sys.exit(2)


def _check_data_files(config: ModelConfig) -> None:
    for label, path_str in [("train", config.train_data), ("eval", config.eval_data)]:
        p = Path(path_str)
        if not p.exists():
            print(
                f"ERROR: {label} data file not found: {p}",
                file=sys.stderr,
            )
            sys.exit(2)


def _run_validation(
    config: ModelConfig,
    trials: int,
    output_dir: Path,
    log: logging.Logger,
    benchmark_suite: str | None = None,
    controller_mode: str = "mass_structural_observe",
    optimizer_research_mode: str = "cayley_stiefel_mass",
) -> dict[str, Any]:
    """Run validation for a single model and return the result dict."""
    from modelcypher.cli.composition import get_backend, get_dataset_training_service
    from modelcypher.core.use_cases.derived_training_validation_service import (
        DerivedTrainingValidationService,
    )

    log.info("=" * 72)
    log.info("Validating %s", config.scale)
    log.info("  Model:      %s", config.model_path)
    log.info("  Train data: %s", config.train_data)
    log.info("  Eval data:  %s", config.eval_data)
    log.info("  Trials:     %d", trials)
    log.info("  Controller: %s", controller_mode)
    log.info("  Optimizer:  %s", optimizer_research_mode)
    log.info("  Benchmark:  %s", benchmark_suite or "none")
    log.info("=" * 72)

    _check_data_files(config)

    model_path = Path(config.model_path).expanduser().resolve()
    if not model_path.exists():
        log.error("Model path does not exist: %s", model_path)
        return {
            "scale": config.scale,
            "error": f"Model path not found: {model_path}",
            "all_passed": False,
        }

    validator = DerivedTrainingValidationService(
        dataset_training_service=get_dataset_training_service(),
        backend=get_backend(),
    )

    result = validator.validate(
        model_path=model_path,
        dataset_path=config.train_data,
        eval_dataset_path=config.eval_data,
        trials=trials,
        benchmark_suite=benchmark_suite,
        controller_mode=controller_mode,
        optimizer_research_mode=optimizer_research_mode,
        enable_phase5_inference=True,
        artifact_root=output_dir / "phase5_artifacts",
    )

    result_dict = result.to_dict()

    # Write per-model result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    log.info("Wrote %s", result_path)

    # Log summary
    log.info(
        "%s: pass=%d fail=%d structural_pass=%d structural_fail=%d "
        "inference_pass=%d inference_fail=%d mean_loss_delta=%.4f mean_ppl_delta=%.4f",
        config.scale,
        result.pass_count,
        result.fail_count,
        result.structural_pass_count,
        result.structural_fail_count,
        result.inference_pass_count,
        result.inference_fail_count,
        result.mean_loss_delta,
        result.mean_perplexity_delta,
    )

    if result.counterexamples:
        for cx in result.counterexamples:
            log.warning(
                "  Counterexample seed=%d: %s",
                cx.seed,
                ", ".join(cx.failure_modes),
            )

    return result_dict


def _build_report(
    all_results: dict[str, dict[str, Any]],
    timestamp: str,
    git_hash: str,
    trials: int,
) -> str:
    """Build markdown report from collected results."""
    lines = [
        "# Pipeline Validation Report",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Git hash:** {git_hash}",
        f"**Trials per model:** {trials}",
        f"**Controller mode:** {next(iter(all_results.values()), {}).get('controller_mode', 'n/a')}",
        f"**Optimizer mode:** {next(iter(all_results.values()), {}).get('optimizer_research_mode', 'n/a')}",
        f"**Benchmark suite:** {next(iter(all_results.values()), {}).get('benchmark_suite', 'none')}",
        "",
    ]

    # Overall verdict
    all_pass = all(
        r.get("all_passed", False) for r in all_results.values()
    )
    verdict = "PASS" if all_pass else "FAIL"
    lines.append(f"## Verdict: {verdict}")
    lines.append("")

    # Summary table
    lines.append(
        "| Model | Structural | Inference | Composite | Structural pass/fail "
        "| Inference pass/fail | Composite pass/fail | "
        "online_eval_delta_correct (mean) | max_ngram_repeat_delta (max) | "
        "CKA worst layer | Null-access min preserved | "
        "cka_blindness_ratio (max) | margin_mean_delta (mean) | "
        "moe_router_stability (mean) | "
        "mode_conn_barrier (max) | rss_cosine (mean) | "
        "dim_null_recruit (mean) |"
    )
    lines.append(
        "|-------|------------|-----------|-----------|----------------------|"
        "---------------------|---------------------|"
        "----------------------------------|----------------------------|"
        "----------------|----------------------------|"
        "--------------------------|--------------------------|"
        "-----------------------------|"
        "------------------------|---------------------|"
        "------------------------|"
    )

    for scale in SCALE_ORDER:
        if scale not in all_results:
            continue
        r = all_results[scale]
        if "error" in r:
            lines.append(
                f"| {scale} | ERROR | - | - | - | - | - | - | - | - | {r['error']} |",
            )
            continue

        pass_count = int(r.get("pass_count", 0))
        fail_count = int(r.get("fail_count", 0))
        status = "PASS" if bool(r.get("all_passed", False)) else "FAIL"
        structural_pass = int(r.get("structural_pass_count", pass_count))
        structural_fail = int(r.get("structural_fail_count", fail_count))
        inference_pass = int(r.get("inference_pass_count", pass_count))
        inference_fail = int(r.get("inference_fail_count", fail_count))
        structural_status = "PASS" if structural_fail == 0 else "FAIL"
        inference_status = "PASS" if inference_fail == 0 else "FAIL"
        trial_results = r.get("trial_results", [])
        delta_correct_values = [
            int(trial["online_eval_delta_correct"])
            for trial in trial_results
            if trial.get("online_eval_delta_correct") is not None
        ]
        delta_correct_str = (
            f"{(sum(delta_correct_values) / len(delta_correct_values)):+.3f}"
            if delta_correct_values
            else "n/a"
        )
        rep_delta_values = [
            float(trial["max_ngram_repeat_delta"])
            for trial in trial_results
            if trial.get("max_ngram_repeat_delta") is not None
        ]
        rep_delta_str = (
            f"{max(rep_delta_values):+.6f}"
            if rep_delta_values
            else "n/a"
        )
        cka_trials = [
            trial for trial in trial_results
            if trial.get("min_cka") is not None
        ]
        if cka_trials:
            cka_worst = min(cka_trials, key=lambda trial: float(trial["min_cka"]))
            cka_worst_layer = cka_worst.get("min_cka_layer")
            if cka_worst_layer is None:
                cka_worst_str = f"{float(cka_worst['min_cka']):.6f} @ layer ?"
            else:
                cka_worst_str = (
                    f"{float(cka_worst['min_cka']):.6f} @ layer {int(cka_worst_layer)}"
                )
        else:
            cka_worst_str = "n/a"

        null_access_trials = [
            trial for trial in trial_results
            if trial.get("null_access_min_behavioral_preserved_fraction") is not None
        ]
        if null_access_trials:
            null_access_worst = min(
                null_access_trials,
                key=lambda trial: float(trial["null_access_min_behavioral_preserved_fraction"]),
            )
            null_layer = null_access_worst.get("null_access_min_behavioral_preserved_layer")
            null_fraction = float(
                null_access_worst["null_access_min_behavioral_preserved_fraction"],
            )
            if null_layer is None:
                null_access_str = f"{null_fraction:.6f} @ layer ?"
            else:
                null_access_str = f"{null_fraction:.6f} @ layer {int(null_layer)}"
        else:
            null_access_str = "n/a"

        blindness_values = [
            float(trial["cka_blindness_ratio"])
            for trial in trial_results
            if trial.get("cka_blindness_ratio") is not None
        ]
        blindness_str = (
            f"{max(blindness_values):.4f}"
            if blindness_values
            else "n/a"
        )
        margin_delta_values = [
            float(trial["margin_mean_delta"])
            for trial in trial_results
            if trial.get("margin_mean_delta") is not None
        ]
        margin_delta_str = (
            f"{(sum(margin_delta_values) / len(margin_delta_values)):+.4f}"
            if margin_delta_values
            else "n/a"
        )
        router_stability_values = [
            float(trial["moe_router_stability"])
            for trial in trial_results
            if trial.get("moe_router_stability") is not None
        ]
        router_stability_str = (
            f"{(sum(router_stability_values) / len(router_stability_values)):.6f}"
            if router_stability_values
            else "n/a"
        )
        mc_barrier_values = [
            float(trial["mode_connectivity_barrier"])
            for trial in trial_results
            if trial.get("mode_connectivity_barrier") is not None
        ]
        mc_barrier_str = (
            f"{max(mc_barrier_values):.6f}"
            if mc_barrier_values
            else "n/a"
        )
        rss_cos_values = [
            float(trial["rss_final_cosine"])
            for trial in trial_results
            if trial.get("rss_final_cosine") is not None
        ]
        rss_cos_str = (
            f"{(sum(rss_cos_values) / len(rss_cos_values)):.6f}"
            if rss_cos_values
            else "n/a"
        )
        dim_recruit_values = [
            float(trial["dim_null_recruitment_from_baseline"])
            for trial in trial_results
            if trial.get("dim_null_recruitment_from_baseline") is not None
        ]
        dim_recruit_str = (
            f"{(sum(dim_recruit_values) / len(dim_recruit_values)):.6f}"
            if dim_recruit_values
            else "n/a"
        )

        lines.append(
            f"| {scale} | {structural_status} | {inference_status} | {status} "
            f"| {structural_pass}/{structural_fail} "
            f"| {inference_pass}/{inference_fail} "
            f"| {pass_count}/{fail_count} "
            f"| {delta_correct_str} "
            f"| {rep_delta_str} "
            f"| {cka_worst_str} "
            f"| {null_access_str} "
            f"| {blindness_str} "
            f"| {margin_delta_str} "
            f"| {router_stability_str} "
            f"| {mc_barrier_str} "
            f"| {rss_cos_str} "
            f"| {dim_recruit_str} |"
        )

    lines.append("")

    # Counterexample detail
    has_cx = any(
        r.get("counterexamples") for r in all_results.values()
    )
    if has_cx:
        lines.append("## Counterexample Detail")
        lines.append("")
        for scale in SCALE_ORDER:
            if scale not in all_results:
                continue
            r = all_results[scale]
            counterexamples = r.get("counterexamples", [])
            if not counterexamples:
                continue
            lines.append(f"### {scale}")
            lines.append("")
            for cx in counterexamples:
                seed = cx.get("seed", "?")
                modes = cx.get("failure_modes", [])
                loss_d = cx.get("loss_delta", 0.0)
                ppl_d = cx.get("perplexity_delta", 0.0)
                stop = cx.get("stop_reason", "?")
                min_cka = cx.get("min_cka")
                min_cka_layer = cx.get("min_cka_layer")
                sat = cx.get("adapter_saturation_median_ratio")
                cooccurrence = cx.get("cooccurrence_class")
                delta_correct = cx.get("online_eval_delta_correct")
                rep_delta = cx.get("max_ngram_repeat_delta")
                null_access_min = cx.get("null_access_min_behavioral_preserved_fraction")
                null_access_layer = cx.get("null_access_min_behavioral_preserved_layer")
                null_obs_max_cond = cx.get("null_observability_max_condition_number")
                null_obs_max_cond_layer = cx.get("null_observability_max_condition_layer")
                first_pre_degraded = cx.get("online_eval_first_pre_degraded_epoch")
                first_post_degraded = cx.get("online_eval_first_post_degraded_epoch")
                benchmark_overall_delta = cx.get("benchmark_overall_delta")
                benchmark_baseline = cx.get("benchmark_baseline")
                benchmark_post = cx.get("benchmark_post")
                lines.append(f"- **Seed {seed}**: {', '.join(modes)}")
                lines.append(f"  - loss_delta={loss_d:.4f}, ppl_delta={ppl_d:.4f}")
                lines.append(f"  - stop_reason={stop}")
                if cooccurrence is not None:
                    lines.append(f"  - cooccurrence_class={cooccurrence}")
                if min_cka is not None:
                    if min_cka_layer is None:
                        lines.append(f"  - min_cka={min_cka:.6f}")
                    else:
                        lines.append(
                            f"  - min_cka={min_cka:.6f} (layer={int(min_cka_layer)})",
                        )
                if sat is not None:
                    lines.append(f"  - adapter_saturation_median_ratio={sat:.4f}")
                if delta_correct is not None:
                    lines.append(f"  - online_eval_delta_correct={int(delta_correct):+d}")
                if rep_delta is not None:
                    lines.append(f"  - max_ngram_repeat_delta={float(rep_delta):+.6f}")
                if benchmark_overall_delta is not None:
                    lines.append(
                        f"  - benchmark_overall_delta={float(benchmark_overall_delta):+.4f}",
                    )
                if (
                    isinstance(benchmark_baseline, dict)
                    and isinstance(benchmark_post, dict)
                    and "overall" in benchmark_baseline
                    and "overall" in benchmark_post
                ):
                    lines.append(
                        "  - benchmark_overall="
                        f"{float(benchmark_baseline['overall']):.4f}"
                        f" -> {float(benchmark_post['overall']):.4f}",
                    )
                if null_access_min is not None:
                    if null_access_layer is None:
                        lines.append(
                            "  - null_access_min_behavioral_preserved_fraction="
                            f"{float(null_access_min):.6f}",
                        )
                    else:
                        lines.append(
                            "  - null_access_min_behavioral_preserved_fraction="
                            f"{float(null_access_min):.6f} (layer={int(null_access_layer)})",
                        )
                if null_obs_max_cond is not None:
                    if null_obs_max_cond_layer is None:
                        lines.append(
                            "  - null_observability_max_condition_number="
                            f"{float(null_obs_max_cond):.6f}",
                        )
                    else:
                        lines.append(
                            "  - null_observability_max_condition_number="
                            f"{float(null_obs_max_cond):.6f} "
                            f"(layer={int(null_obs_max_cond_layer)})",
                        )
                if first_pre_degraded is not None:
                    lines.append(
                        "  - online_eval_first_pre_degraded_epoch="
                        f"{int(first_pre_degraded)}",
                    )
                if first_post_degraded is not None:
                    lines.append(
                        "  - online_eval_first_post_degraded_epoch="
                        f"{int(first_post_degraded)}",
                    )
                # Dual-manifold CKA diagnostics
                inf_min_cka = cx.get("inference_min_cka")
                inf_min_cka_layer = cx.get("inference_min_cka_layer")
                if inf_min_cka is not None:
                    if inf_min_cka_layer is None:
                        lines.append(
                            f"  - inference_min_cka={float(inf_min_cka):.6f}",
                        )
                    else:
                        lines.append(
                            f"  - inference_min_cka={float(inf_min_cka):.6f} "
                            f"(layer={int(inf_min_cka_layer)})",
                        )
                cx_blindness = cx.get("cka_blindness_ratio")
                cx_blindness_layer = cx.get("cka_blindness_worst_layer")
                if cx_blindness is not None:
                    if cx_blindness_layer is None:
                        lines.append(
                            f"  - cka_blindness_ratio={float(cx_blindness):.4f}",
                        )
                    else:
                        lines.append(
                            f"  - cka_blindness_ratio={float(cx_blindness):.4f} "
                            f"(worst_layer={int(cx_blindness_layer)})",
                        )
                cx_delta_gap = cx.get("cka_delta_gap")
                if cx_delta_gap is not None:
                    lines.append(
                        f"  - cka_delta_gap={float(cx_delta_gap):+.6f}",
                    )
                # Decision-boundary margin diagnostics
                margin_base = cx.get("margin_mean_baseline")
                margin_adapt = cx.get("margin_mean_adapted")
                margin_delta = cx.get("margin_mean_delta")
                if margin_base is not None:
                    lines.append(
                        f"  - margin_mean_baseline={float(margin_base):.4f}",
                    )
                if margin_adapt is not None:
                    lines.append(
                        f"  - margin_mean_adapted={float(margin_adapt):.4f}",
                    )
                if margin_delta is not None:
                    lines.append(
                        f"  - margin_mean_delta={float(margin_delta):+.4f}",
                    )
                margin_nz_base = cx.get("margin_n_near_zero_baseline")
                margin_nz_adapt = cx.get("margin_n_near_zero_adapted")
                margin_flipped = cx.get("margin_n_flipped_sign")
                if margin_nz_base is not None:
                    lines.append(
                        f"  - margin_n_near_zero_baseline={int(margin_nz_base)}",
                    )
                if margin_nz_adapt is not None:
                    lines.append(
                        f"  - margin_n_near_zero_adapted={int(margin_nz_adapt)}",
                    )
                if margin_flipped is not None:
                    lines.append(
                        f"  - margin_n_flipped_sign={int(margin_flipped)}",
                    )
                router_stability = cx.get("moe_router_stability")
                if router_stability is not None:
                    lines.append(
                        f"  - moe_router_stability={float(router_stability):.6f}",
                    )
                # Mode connectivity
                mc_bar = cx.get("mode_connectivity_barrier")
                if mc_bar is not None:
                    mc_norm = cx.get("mode_connectivity_normalized_barrier")
                    mc_meth = cx.get("mode_connectivity_method")
                    lines.append(
                        f"  - mode_connectivity_barrier={float(mc_bar):.6f}",
                    )
                    if mc_norm is not None:
                        lines.append(
                            f"  - mode_connectivity_normalized_barrier="
                            f"{float(mc_norm):.6f}",
                        )
                    if mc_meth is not None:
                        lines.append(
                            f"  - mode_connectivity_method={mc_meth}",
                        )
                # Train-time degeneration
                degen_max = cx.get("degeneration_max_ngram_repeat")
                if degen_max is not None:
                    lines.append(
                        f"  - degeneration_max_ngram_repeat="
                        f"{float(degen_max):.6f}",
                    )
                degen_mean = cx.get("degeneration_mean_ngram_repeat")
                if degen_mean is not None:
                    lines.append(
                        f"  - degeneration_mean_ngram_repeat="
                        f"{float(degen_mean):.6f}",
                    )
                # RSS similarity
                rss_cos = cx.get("rss_final_cosine")
                if rss_cos is not None:
                    lines.append(
                        f"  - rss_final_cosine={float(rss_cos):.6f}",
                    )
                rss_spear = cx.get("rss_final_spearman")
                if rss_spear is not None:
                    lines.append(
                        f"  - rss_final_spearman={float(rss_spear):.6f}",
                    )
                # Dimensional recruitment
                dim_recruit = cx.get("dim_null_recruitment_from_baseline")
                if dim_recruit is not None:
                    lines.append(
                        f"  - dim_null_recruitment_from_baseline="
                        f"{float(dim_recruit):.6f}",
                    )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()

    if args.model_path is None:
        _check_volume()
    elif args.model_scale is None:
        print(
            "ERROR: --model-path requires --model-scale so the overridden run stays labeled.",
            file=sys.stderr,
        )
        sys.exit(2)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log = _configure_logging(output_root / "run.log")

    timestamp = datetime.now(timezone.utc).isoformat()
    git_hash = _git_hash()

    log.info("Pipeline validation starting")
    log.info("  Timestamp: %s", timestamp)
    log.info("  Git hash:  %s", git_hash)
    log.info("  Trials:    %d", args.trials)
    log.info("  Controller: %s", args.controller_mode)
    log.info("  Optimizer:  %s", args.optimizer_research_mode)
    log.info("  Benchmark:  %s", args.benchmark_suite or "none")

    # Determine which scales to run
    if args.model_scale is not None:
        scales = [args.model_scale]
    else:
        scales = list(SCALE_ORDER)

    log.info("  Scales:    %s", ", ".join(scales))

    all_results: dict[str, dict[str, Any]] = {}

    for scale in scales:
        config = MODEL_CONFIGS[scale]
        if args.model_path is not None:
            config = ModelConfig(
                scale=scale,
                model_path=str(args.model_path.expanduser().resolve()),
                train_data=str(
                    (args.train_data or Path(config.train_data)).expanduser().resolve()
                ),
                eval_data=str(
                    (args.eval_data or Path(config.eval_data)).expanduser().resolve()
                ),
            )
        scale_dir = output_root / scale
        try:
            result_dict = _run_validation(
                config,
                args.trials,
                scale_dir,
                log,
                benchmark_suite=args.benchmark_suite,
                controller_mode=args.controller_mode,
                optimizer_research_mode=args.optimizer_research_mode,
            )
            all_results[scale] = result_dict
        except Exception:
            log.exception("FAILED: %s", scale)
            all_results[scale] = {
                "scale": scale,
                "error": "exception during validation (see run.log)",
                "all_passed": False,
            }

    # Write verdict
    all_pass = all(r.get("all_passed", False) for r in all_results.values())
    all_structural_pass = all(
        int(r.get("structural_fail_count", 0)) == 0 for r in all_results.values()
    )
    all_inference_pass = all(
        int(r.get("inference_fail_count", 0)) == 0 for r in all_results.values()
    )
    verdict = {
        "timestamp": timestamp,
        "git_hash": git_hash,
        "trials_per_model": args.trials,
        "controller_mode": args.controller_mode,
        "optimizer_research_mode": args.optimizer_research_mode,
        "benchmark_suite": args.benchmark_suite,
        "scales": scales,
        "all_pass": all_pass,
        "all_structural_pass": all_structural_pass,
        "all_inference_pass": all_inference_pass,
        "per_scale": {
            scale: {
                "all_passed": r.get("all_passed", False),
                "pass_count": r.get("pass_count", 0),
                "fail_count": r.get("fail_count", 0),
                "structural_pass_count": r.get("structural_pass_count", 0),
                "structural_fail_count": r.get("structural_fail_count", 0),
                "inference_pass_count": r.get("inference_pass_count", 0),
                "inference_fail_count": r.get("inference_fail_count", 0),
                "phase5_inference_enabled": r.get("phase5_inference_enabled", False),
                "error": r.get("error"),
            }
            for scale, r in all_results.items()
        },
    }
    verdict_path = output_root / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    log.info("Wrote %s", verdict_path)

    # Write report
    report = _build_report(all_results, timestamp, git_hash, args.trials)
    report_path = output_root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", report_path)

    # Print verdict
    print()
    print(f"{'=' * 50}")
    print(f"VERDICT: {('PASS' if all_pass else 'FAIL')}")
    for scale in scales:
        r = all_results.get(scale, {})
        if "error" in r:
            print(f"  {scale}: ERROR — {r['error']}")
        elif r.get("all_passed"):
            print(f"  {scale}: PASS ({r.get('pass_count', 0)}/{args.trials} trials)")
        else:
            print(
                f"  {scale}: FAIL ({r.get('fail_count', 0)}/{args.trials} "
                f"counterexamples)"
            )
    print(f"{'=' * 50}")
    print(f"Report: {report_path}")
    print(f"Verdict: {verdict_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
