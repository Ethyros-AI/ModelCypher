#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Artifact-only postmortem for the R2 closed-loop falsifier.

This script does not run models. It reads the retained falsifier report and
trial artifacts, reconstructs the first arm event, and writes a compact
postmortem showing whether the miss came from timing, target selection, or
actuator failure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "nblora_vs_standard"
DEFAULT_REPORT_PATH = RESULTS_ROOT / "validate_derived_r2_closed_loop_seed42_quick.json"
DEFAULT_TRAIN_RESULT_PATH = (
    RESULTS_ROOT
    / "phase5_artifacts_r2_closed_loop_seed42"
    / "trial_000_seed_42"
    / "train_result.json"
)
DEFAULT_JSON_PATH = RESULTS_ROOT / "r2_closed_loop_postmortem.json"
DEFAULT_MARKDOWN_PATH = RESULTS_ROOT / "r2_closed_loop_postmortem.md"

_ORDER_KEYS = (
    "behavioral_transport_over_remaining_budget",
    "spectral_budget_ratio",
    "stable_rank_concentration",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _first_armed_trace(
    controller_trace: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    for idx, epoch_trace in enumerate(controller_trace):
        decision = epoch_trace.get("closed_loop_decision")
        if isinstance(decision, dict) and decision.get("armed") is True:
            return idx, epoch_trace, decision
    raise ValueError("No armed closed_loop_decision found in controller_trace.")


def _first_counterexample(report: dict[str, Any]) -> dict[str, Any]:
    counterexamples = report.get("counterexamples")
    if not isinstance(counterexamples, list) or not counterexamples:
        return {}
    first = counterexamples[0]
    return dict(first) if isinstance(first, dict) else {}


def _transport_entries(
    behavioral_state: dict[str, Any],
) -> tuple[list[tuple[str, float]], float]:
    transport = behavioral_state.get("per_layer_behavioral_transport_norm") or {}
    entries = sorted(
        (
            (str(layer_key), float(value))
            for layer_key, value in transport.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    total = sum(value for _, value in entries)
    return entries, total


def _available_ordering_counts(ordering_metrics: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in _ORDER_KEYS}
    for metrics in ordering_metrics.values():
        for key in _ORDER_KEYS:
            if metrics.get(key) is not None:
                counts[key] += 1
    return counts


def _transport_rank(entries: list[tuple[str, float]], target_layer: str | None) -> int | None:
    if target_layer is None:
        return None
    for idx, (layer_key, _value) in enumerate(entries, start=1):
        if layer_key == target_layer:
            return idx
    return None


def _share(value: float | None, total: float) -> float | None:
    if value is None or total <= 0.0:
        return None
    return float(value) / float(total)


def _unarmed_postmortem(
    *,
    report: dict[str, Any],
    train_result: dict[str, Any],
    controller_trace: list[dict[str, Any]],
    report_path: Path,
    train_result_path: Path,
) -> dict[str, Any]:
    decision = {}
    if controller_trace:
        candidate = controller_trace[0].get("closed_loop_decision")
        if isinstance(candidate, dict):
            decision = candidate
    counterexample = _first_counterexample(report)
    next_targets = [
        "trace why the closed-loop controller did not arm on the retained artifact before analyzing actuator effects",
        "separate no-arm artifacts from armed counterexamples in future retained summaries",
        "rerun only after the trigger observable and arm preconditions are recorded in the artifact contract",
    ]
    return {
        "schema": "r2_closed_loop_postmortem_v1",
        "created_at_utc": _utc_now(),
        "inputs": {
            "report_path": str(report_path),
            "train_result_path": str(train_result_path),
        },
        "classification": {
            "status": "MECHANISM_NOT_ENGAGED",
            "counterexample_confirmed": False,
            "temporal_blind_spot": False,
            "target_selection_fallback": "not_armed",
            "target_misaligned_with_transport": False,
            "freeze_applied": False,
            "off_surface_inference_divergence": False,
        },
        "arm_event": {
            "armed": False,
            "epoch": decision.get("epoch"),
            "trigger_reasons": list(decision.get("trigger_reasons") or []),
            "target_layer": decision.get("target_layer"),
            "adapter_rank_at_arm": None,
            "online_eval_accuracy_delta_at_arm": None,
            "available_ordering_metric_counts": {key: 0 for key in _ORDER_KEYS},
            "ordering_metrics_all_null": True,
            "margin_points_available_at_arm": 0,
            "stable_rank_points_available_at_arm": 0,
            "stable_rank_concentration_observable_available": False,
        },
        "transport_at_arm": {
            "total_behavioral_transport": 0.0,
            "target_layer_transport": None,
            "target_layer_transport_share": None,
            "target_layer_transport_rank": None,
            "top3_transport_share": None,
            "top5_transport_share": None,
            "top_layers": [],
        },
        "freeze_effectiveness": {
            "freeze_applied": False,
            "objective_components_after_arm": [],
            "target_layer_parameter_update_norm_after_arm": None,
        },
        "divergence": {
            "training_probe_min_cka": _float_or_none(train_result.get("min_cka")),
            "training_probe_min_cka_layer": train_result.get("min_cka_layer"),
            "inference_probe_min_cka": _float_or_none(train_result.get("inference_min_cka")),
            "inference_probe_min_cka_layer": train_result.get("inference_min_cka_layer"),
            "inference_probe_min_cka_layer_on_adaptation_surface": None,
            "cka_blindness_ratio": _float_or_none(counterexample.get("cka_blindness_ratio")),
        },
        "behavioral_outcome": {
            "online_eval_delta_correct": counterexample.get("online_eval_delta_correct"),
            "benchmark_overall_delta": _float_or_none(
                counterexample.get("benchmark_overall_delta"),
            ),
            "margin_mean_delta": _float_or_none(counterexample.get("margin_mean_delta")),
            "degeneration_max_ngram_repeat": _float_or_none(
                counterexample.get("degeneration_max_ngram_repeat"),
            ),
            "stop_reason": train_result.get("stop_reason"),
        },
        "next_derivation_targets": next_targets,
    }


def build_postmortem(
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    train_result_path: Path = DEFAULT_TRAIN_RESULT_PATH,
) -> dict[str, Any]:
    report = _load_json(report_path)
    train_result = _load_json(train_result_path)

    controller_trace = list(train_result.get("controller_trace") or [])
    if not controller_trace:
        raise ValueError("train_result.json is missing controller_trace.")

    try:
        armed_index, armed_trace, armed_decision = _first_armed_trace(controller_trace)
    except ValueError:
        return _unarmed_postmortem(
            report=report,
            train_result=train_result,
            controller_trace=controller_trace,
            report_path=report_path,
            train_result_path=train_result_path,
        )

    counterexample = _first_counterexample(report)
    armed_state = dict(armed_trace.get("behavioral_state") or {})
    target_layer = armed_decision.get("target_layer")
    ordering_metrics = {
        str(layer_key): dict(metrics)
        for layer_key, metrics in (armed_decision.get("ordering_metrics") or {}).items()
    }
    available_counts = _available_ordering_counts(ordering_metrics)
    lexicographic_fallback = (
        bool(ordering_metrics)
        and all(count == 0 for count in available_counts.values())
        and target_layer == max(ordering_metrics)
    )

    transport_entries, total_transport = _transport_entries(armed_state)
    chosen_transport = None
    if target_layer is not None:
        chosen_transport = dict(transport_entries).get(target_layer)
    chosen_transport_rank = _transport_rank(transport_entries, target_layer)

    top3_share = None
    top5_share = None
    if total_transport > 0.0:
        top3_share = sum(value for _, value in transport_entries[:3]) / total_transport
        top5_share = sum(value for _, value in transport_entries[:5]) / total_transport

    epoch_metrics = list(train_result.get("epoch_metrics") or [])
    arm_epoch = int(armed_decision["epoch"])
    epochs_observed_by_arm = [
        metric for metric in epoch_metrics
        if int(metric.get("epoch", -1)) <= arm_epoch
    ]
    margin_points_available_at_arm = sum(
        1 for metric in epochs_observed_by_arm
        if metric.get("margin_mean") is not None
    )
    stable_rank_points_available_at_arm = sum(
        1 for metric in epochs_observed_by_arm
        if metric.get("stable_rank_median") is not None
    )
    adapter_rank_at_arm = armed_state.get("adapter_rank")
    stable_rank_concentration_observable_available = bool(
        adapter_rank_at_arm is not None and int(adapter_rank_at_arm) > 1
    )

    freeze_applied = False
    freeze_objective_components: list[str] = []
    frozen_layer_zero_update = None
    if armed_index + 1 < len(controller_trace):
        next_trace = dict(controller_trace[armed_index + 1])
        next_step_traces = list(next_trace.get("step_traces") or [])
        if next_step_traces:
            first_step = dict(next_step_traces[0])
            freeze_objective_components = list(first_step.get("objective_components") or [])
            per_layer_measurements = first_step.get("per_layer_measurements") or {}
            chosen_measurement = per_layer_measurements.get(target_layer, {})
            frozen_layer_zero_update = _float_or_none(
                chosen_measurement.get("parameter_update_norm"),
            )
            freeze_applied = (
                "closed_loop_freeze" in freeze_objective_components
                and frozen_layer_zero_update == 0.0
            )

    target_modules = [str(module) for module in (train_result.get("target_modules") or [])]
    inference_min_cka_layer = train_result.get("inference_min_cka_layer")
    inference_layer_prefix = None
    inference_layer_on_adaptation_surface = None
    if inference_min_cka_layer is not None:
        inference_layer_prefix = f"model.layers.{int(inference_min_cka_layer)}."
        inference_layer_on_adaptation_surface = any(
            module.startswith(inference_layer_prefix)
            for module in target_modules
        )

    next_targets = [
        "derive the next trigger from pre-degradation observables only; online_eval_accuracy_drop remains a certificate, not an arm signal",
        "do not allow layer selection to proceed when all ordering metrics are null; treat that state as measurement-unavailable, not targetable risk",
        "re-derive layer-local targeting against the dominant transport surface before considering any bundled intervention",
        "keep the next cycle artifact-first and stop without seed expansion until the causal operator survives a new falsifier",
    ]

    return {
        "schema": "r2_closed_loop_postmortem_v1",
        "created_at_utc": _utc_now(),
        "inputs": {
            "report_path": str(report_path),
            "train_result_path": str(train_result_path),
        },
        "classification": {
            "status": "MECHANISM_UNDERSPECIFIED",
            "counterexample_confirmed": True,
            "temporal_blind_spot": bool(
                _float_or_none(armed_state.get("online_eval_accuracy_delta")) is not None
                and float(armed_state["online_eval_accuracy_delta"]) < 0.0
            ),
            "target_selection_fallback": (
                "lexicographic_tie_break"
                if lexicographic_fallback
                else "measured_ordering"
            ),
            "target_misaligned_with_transport": bool(
                chosen_transport_rank is not None and chosen_transport_rank > 5
            ),
            "freeze_applied": freeze_applied,
            "off_surface_inference_divergence": bool(
                inference_layer_on_adaptation_surface is False
            ),
        },
        "arm_event": {
            "armed": True,
            "epoch": arm_epoch,
            "trigger_reasons": list(armed_decision.get("trigger_reasons") or []),
            "target_layer": target_layer,
            "adapter_rank_at_arm": adapter_rank_at_arm,
            "online_eval_accuracy_delta_at_arm": _float_or_none(
                armed_state.get("online_eval_accuracy_delta"),
            ),
            "available_ordering_metric_counts": available_counts,
            "ordering_metrics_all_null": all(count == 0 for count in available_counts.values()),
            "margin_points_available_at_arm": margin_points_available_at_arm,
            "stable_rank_points_available_at_arm": stable_rank_points_available_at_arm,
            "stable_rank_concentration_observable_available": (
                stable_rank_concentration_observable_available
            ),
        },
        "transport_at_arm": {
            "total_behavioral_transport": total_transport,
            "target_layer_transport": chosen_transport,
            "target_layer_transport_share": _share(chosen_transport, total_transport),
            "target_layer_transport_rank": chosen_transport_rank,
            "top3_transport_share": top3_share,
            "top5_transport_share": top5_share,
            "top_layers": [
                {
                    "layer": layer_key,
                    "transport": value,
                    "share": _share(value, total_transport),
                }
                for layer_key, value in transport_entries[:5]
            ],
        },
        "freeze_effectiveness": {
            "freeze_applied": freeze_applied,
            "objective_components_after_arm": freeze_objective_components,
            "target_layer_parameter_update_norm_after_arm": frozen_layer_zero_update,
        },
        "divergence": {
            "training_probe_min_cka": _float_or_none(train_result.get("min_cka")),
            "training_probe_min_cka_layer": train_result.get("min_cka_layer"),
            "inference_probe_min_cka": _float_or_none(train_result.get("inference_min_cka")),
            "inference_probe_min_cka_layer": inference_min_cka_layer,
            "inference_probe_min_cka_layer_on_adaptation_surface": (
                inference_layer_on_adaptation_surface
            ),
            "cka_blindness_ratio": _float_or_none(counterexample.get("cka_blindness_ratio")),
        },
        "behavioral_outcome": {
            "online_eval_delta_correct": counterexample.get("online_eval_delta_correct"),
            "benchmark_overall_delta": _float_or_none(
                counterexample.get("benchmark_overall_delta"),
            ),
            "margin_mean_delta": _float_or_none(counterexample.get("margin_mean_delta")),
            "degeneration_max_ngram_repeat": _float_or_none(
                counterexample.get("degeneration_max_ngram_repeat"),
            ),
            "stop_reason": train_result.get("stop_reason"),
        },
        "next_derivation_targets": next_targets,
    }


def render_markdown(postmortem: dict[str, Any]) -> str:
    arm_event = postmortem["arm_event"]
    transport = postmortem["transport_at_arm"]
    divergence = postmortem["divergence"]
    classification = postmortem["classification"]
    outcome = postmortem["behavioral_outcome"]
    top_layers = transport["top_layers"]

    lines = [
        "# R2 Closed-Loop Postmortem",
        "",
        "## Classification",
        f"- Status: `{classification['status']}`",
        (
            "- Counterexample confirmed: the controller armed, the freeze applied, and behavior still collapsed."
            if classification["counterexample_confirmed"]
            else "- Counterexample confirmed: `False`; no closed-loop arm event was recorded in the retained artifact."
        ),
        f"- Temporal blind spot: `{classification['temporal_blind_spot']}`",
        f"- Target selection fallback: `{classification['target_selection_fallback']}`",
        f"- Target misaligned with transport: `{classification['target_misaligned_with_transport']}`",
        f"- Off-surface inference divergence: `{classification['off_surface_inference_divergence']}`",
        "",
        "## Arm Event",
        f"- Epoch: `{arm_event['epoch']}`",
        f"- Trigger reasons: `{', '.join(arm_event['trigger_reasons'])}`",
        f"- Target layer: `{arm_event['target_layer']}`",
        f"- Online eval accuracy delta at arm: `{arm_event['online_eval_accuracy_delta_at_arm']}`",
        f"- Ordering metrics all null: `{arm_event['ordering_metrics_all_null']}`",
        f"- Margin points available at arm: `{arm_event['margin_points_available_at_arm']}`",
        f"- Stable-rank points available at arm: `{arm_event['stable_rank_points_available_at_arm']}`",
        f"- Stable-rank concentration observable available: `{arm_event['stable_rank_concentration_observable_available']}`",
        "",
        "## Transport At Arm",
        f"- Target transport rank: `{transport['target_layer_transport_rank']}`",
        f"- Target transport share: `{transport['target_layer_transport_share']}`",
        f"- Top-3 transport share: `{transport['top3_transport_share']}`",
        f"- Top-5 transport share: `{transport['top5_transport_share']}`",
        "- Top layers at the arm point:",
    ]
    for item in top_layers:
        lines.append(
            f"  - `{item['layer']}` transport={item['transport']} share={item['share']}"
        )
    if not top_layers:
        lines.append("  - n/a")

    lines.extend(
        [
            "",
            "## Divergence",
            f"- Training-probe min CKA: `{divergence['training_probe_min_cka']}` at layer `{divergence['training_probe_min_cka_layer']}`",
            f"- Inference-probe min CKA: `{divergence['inference_probe_min_cka']}` at layer `{divergence['inference_probe_min_cka_layer']}`",
            f"- Inference worst layer on adaptation surface: `{divergence['inference_probe_min_cka_layer_on_adaptation_surface']}`",
            f"- CKA blindness ratio: `{divergence['cka_blindness_ratio']}`",
            "",
            "## Outcome",
            f"- Online eval delta correct: `{outcome['online_eval_delta_correct']}`",
            f"- Benchmark overall delta: `{outcome['benchmark_overall_delta']}`",
            f"- Margin mean delta: `{outcome['margin_mean_delta']}`",
            f"- Degeneration max n-gram repeat: `{outcome['degeneration_max_ngram_repeat']}`",
            f"- Stop reason: `{outcome['stop_reason']}`",
            "",
            "## Next Derivation Targets",
        ]
    )
    for item in postmortem["next_derivation_targets"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--train-result-path", type=Path, default=DEFAULT_TRAIN_RESULT_PATH)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_PATH)
    args = parser.parse_args()

    postmortem = build_postmortem(
        report_path=args.report_path,
        train_result_path=args.train_result_path,
    )
    markdown = render_markdown(postmortem)
    _write_json(args.json_out, postmortem)
    _write_text(args.markdown_out, markdown)
    print(
        json.dumps(
            {
                "status": postmortem["classification"]["status"],
                "json_out": str(args.json_out),
                "markdown_out": str(args.markdown_out),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
