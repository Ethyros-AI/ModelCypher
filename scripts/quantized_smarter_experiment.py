#!/usr/bin/env python3
"""Quantized-smarter orchestrator for Qwen3.5-0.8B.

Runs a data-first three-arm experiment:
  1. Arm A (Tikhonov correction): `mc quantize correct`
  2. Arm B (Corrective LoRA): corrective adapter on q4 base
  3. Arm C (Combined): corrective adapter on Arm A corrected base

For each arm, collects:
  - Benchmark accuracy on quick suite (gsm8k, arc_easy, boolq)
  - Task-conditioned CKA vs bf16 reference

Then computes CI-aware predictor verdict:
  - cka_predictive
  - cka_non_predictive
  - insufficient_evidence

No pipeline gate modifications are performed in this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16"
DEFAULT_Q4 = "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-4bit-g64"
DEFAULT_TRAIN = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"
DEFAULT_OUTPUT_DIR = "results/quantized_smarter_experiment"
DEFAULT_PROMOTION_TARGET = "docs/research/quantized_smarter_qwen35_0.8b.md"

TASKS = ("gsm8k", "boolq", "arc_easy")
DEGRADED_TASKS = ("gsm8k", "boolq")
BENCHMARK_SUITE = "quick"

LOGGER = logging.getLogger("quantized_smarter_experiment")


@dataclass(frozen=True)
class CommandRecord:
    """Captured command execution metadata."""

    stage: str
    command: list[str]
    exit_code: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quantized-smarter experiment (A/B/C) with CI-aware predictor verdict.",
    )
    parser.add_argument("--fp-model", default=DEFAULT_FP, help="Path to bf16 reference model")
    parser.add_argument("--q4-model", default=DEFAULT_Q4, help="Path to quantized (4-bit) model")
    parser.add_argument(
        "--train-dataset",
        default=DEFAULT_TRAIN,
        help="Path to corrective LoRA training dataset (JSONL)",
    )
    parser.add_argument(
        "--eval-dataset",
        default=DEFAULT_EVAL,
        help="Path to evaluation dataset used by correction/preflight (JSONL)",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=100,
        help="Samples per benchmark task (fixed default: 100)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Corrective LoRA max iterations for Arms B/C",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory",
    )
    parser.add_argument(
        "--promotion-target",
        default=DEFAULT_PROMOTION_TARGET,
        help="Canonical doc path for promotion copy target (metadata only)",
    )
    parser.add_argument(
        "--preflight-n-calibration",
        type=int,
        default=2,
        help="Calibration samples for preflight quantize-correct smoke",
    )
    parser.add_argument(
        "--preflight-max-seq-len",
        type=int,
        default=32,
        help="Max sequence length for preflight quantize-correct smoke",
    )
    parser.add_argument(
        "--preflight-cka-probes",
        type=int,
        default=2,
        help="CKA probes for preflight corrective-LoRA smoke",
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage_slug(stage: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stage)


def _run_command(
    *,
    stage: str,
    command: list[str],
    run_dir: Path,
    timeout_seconds: int | None = None,
    check: bool = True,
) -> CommandRecord:
    logs_dir = run_dir / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    slug = _stage_slug(stage)
    stdout_path = logs_dir / f"{slug}.stdout.txt"
    stderr_path = logs_dir / f"{slug}.stderr.txt"

    start = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration = time.monotonic() - start

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    record = CommandRecord(
        stage=stage,
        command=command,
        exit_code=int(proc.returncode),
        duration_seconds=float(duration),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=(proc.stdout or "")[-4000:],
        stderr_tail=(proc.stderr or "")[-4000:],
    )

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit_code={proc.returncode}. "
            f"stderr_tail={record.stderr_tail[-500:]}",
        )
    return record


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract first parseable JSON object from mixed stdout text."""
    text = text.strip()
    if not text:
        raise ValueError("No stdout text available for JSON extraction")
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        snippet = text[idx:]
        try:
            payload = json.loads(snippet)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No parseable JSON object found in stdout")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    children = [p for p in base_dir.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(children, key=lambda p: p.stat().st_mtime)


def _infer_precision_state(model_path: str, adapter_path: str | None = None) -> dict[str, Any]:
    model_name = Path(model_path).name
    match = re.search(r"(?P<bits>\d+)bit(?:-g(?P<group>\d+))?", model_name)
    precision_state: dict[str, Any] = {
        "model_path": model_path,
    }
    if match:
        precision_state["bits"] = int(match.group("bits"))
        if match.group("group") is not None:
            precision_state["group_size"] = int(match.group("group"))
    else:
        precision_state["bits"] = None
        precision_state["group_size"] = None
    precision_state["adapter_path"] = adapter_path
    return precision_state


def _architecture_smoke(fp_model: str, q4_model: str) -> None:
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    loader = ModelLoader(get_default_backend())
    fp_loaded, _ = loader.load_model(fp_model)
    q4_loaded, _ = loader.load_model(q4_model)
    _ = (fp_loaded, q4_loaded)

    del fp_loaded, q4_loaded
    gc.collect()


def _preflight_command_plan(args: argparse.Namespace, run_dir: Path) -> list[tuple[str, list[str], int]]:
    preflight_quant_output = run_dir / "preflight" / "quantize_correct"
    preflight_lora_output = run_dir / "preflight" / "corrective_lora"
    return [
        (
            "preflight.quantize_correct",
            [
                "poetry",
                "run",
                "mc",
                "--output",
                "json",
                "quantize",
                "correct",
                "--quantized-model",
                args.q4_model,
                "--fp-model",
                args.fp_model,
                "--output",
                str(preflight_quant_output),
                "--eval-dataset",
                args.eval_dataset,
                "--n-calibration",
                str(args.preflight_n_calibration),
                "--max-seq-len",
                str(args.preflight_max_seq_len),
            ],
            1800,
        ),
        (
            "preflight.corrective_lora_smoke",
            [
                "poetry",
                "run",
                "python",
                "scripts/corrective_lora_training.py",
                "--quantized-model",
                args.q4_model,
                "--fp-model",
                args.fp_model,
                "--train-dataset",
                args.train_dataset,
                "--eval-dataset",
                args.eval_dataset,
                "--max-iters",
                "0",
                "--batch-size",
                "1",
                "--n-cka-probes",
                str(args.preflight_cka_probes),
                "--output-dir",
                str(preflight_lora_output),
            ],
            1800,
        ),
    ]


def _execution_stage_order() -> list[str]:
    """Canonical stage order for orchestration/testing."""
    return [
        "preflight.architecture_smoke",
        "preflight.quantize_correct",
        "preflight.corrective_lora_smoke",
        "arm_a.quantize_correct",
        "arm_b.corrective_lora",
        "arm_c.corrective_lora",
        "benchmark.q4_baseline",
        "benchmark.arm_a_tikhonov",
        "benchmark.arm_b_corrective_lora",
        "benchmark.arm_c_combined",
        "cka.q4_baseline",
        "cka.arm_a_tikhonov",
        "cka.arm_b_corrective_lora",
        "cka.arm_c_combined",
    ]


def _run_preflight(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    run_command_fn: Callable[..., CommandRecord] = _run_command,
    architecture_smoke_fn: Callable[[str, str], None] = _architecture_smoke,
) -> tuple[dict[str, Any], list[CommandRecord]]:
    records: list[CommandRecord] = []
    steps: list[dict[str, Any]] = []

    stage = "preflight.architecture_smoke"
    try:
        start = time.monotonic()
        architecture_smoke_fn(args.fp_model, args.q4_model)
        duration = time.monotonic() - start
        steps.append(
            {
                "stage": stage,
                "passed": True,
                "duration_seconds": duration,
            },
        )
    except Exception as exc:  # pragma: no cover - error path tested via monkeypatch
        steps.append({"stage": stage, "passed": False, "error": str(exc)})
        return {
            "passed": False,
            "steps": steps,
            "failure_stage": stage,
            "failure_detail": str(exc),
        }, records

    for stage_name, command, timeout_s in _preflight_command_plan(args, run_dir):
        try:
            record = run_command_fn(
                stage=stage_name,
                command=command,
                run_dir=run_dir,
                timeout_seconds=timeout_s,
                check=True,
            )
            records.append(record)
            steps.append(
                {
                    "stage": stage_name,
                    "passed": True,
                    "duration_seconds": record.duration_seconds,
                },
            )
        except Exception as exc:
            steps.append({"stage": stage_name, "passed": False, "error": str(exc)})
            return {
                "passed": False,
                "steps": steps,
                "failure_stage": stage_name,
                "failure_detail": str(exc),
            }, records

    return {"passed": True, "steps": steps}, records


def _arm_command_plan(args: argparse.Namespace, run_dir: Path) -> list[tuple[str, list[str], int]]:
    arm_root = run_dir / "arms"
    arm_a_output = arm_root / "arm_a_tikhonov_model"
    arm_b_output = arm_root / "arm_b_corrective_lora"
    arm_c_output = arm_root / "arm_c_combined"
    return [
        (
            "arm_a.quantize_correct",
            [
                "poetry",
                "run",
                "mc",
                "--output",
                "json",
                "quantize",
                "correct",
                "--quantized-model",
                args.q4_model,
                "--fp-model",
                args.fp_model,
                "--output",
                str(arm_a_output),
                "--eval-dataset",
                args.eval_dataset,
            ],
            14400,
        ),
        (
            "arm_b.corrective_lora",
            [
                "poetry",
                "run",
                "python",
                "scripts/corrective_lora_training.py",
                "--quantized-model",
                args.q4_model,
                "--fp-model",
                args.fp_model,
                "--train-dataset",
                args.train_dataset,
                "--eval-dataset",
                args.eval_dataset,
                "--max-iters",
                str(args.max_iters),
                "--output-dir",
                str(arm_b_output),
            ],
            14400,
        ),
        (
            "arm_c.corrective_lora",
            [
                "poetry",
                "run",
                "python",
                "scripts/corrective_lora_training.py",
                "--quantized-model",
                str(arm_a_output),
                "--fp-model",
                args.fp_model,
                "--train-dataset",
                args.train_dataset,
                "--eval-dataset",
                args.eval_dataset,
                "--max-iters",
                str(args.max_iters),
                "--output-dir",
                str(arm_c_output),
            ],
            14400,
        ),
    ]


def _load_corrective_run_payload(base_output_dir: Path) -> tuple[dict[str, Any], Path]:
    run_dir = _latest_child_dir(base_output_dir)
    payload_path = run_dir / "corrective_lora.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing corrective_lora.json in {run_dir}")
    payload = _read_json(payload_path)
    return payload, run_dir


def _run_benchmark_command(
    *,
    stage: str,
    model_path: str,
    adapter_path: str | None,
    benchmark_limit: int,
    run_dir: Path,
) -> tuple[dict[str, Any], CommandRecord]:
    output_dir = run_dir / "benchmarks" / _stage_slug(stage)
    command = [
        "poetry",
        "run",
        "mc",
        "--output",
        "json",
        "analyze",
        "benchmark",
        model_path,
        "--suite",
        BENCHMARK_SUITE,
        "--limit",
        str(benchmark_limit),
        "--output",
        str(output_dir),
    ]
    if adapter_path is not None:
        command.extend(["--adapter", adapter_path])

    record = _run_command(
        stage=stage,
        command=command,
        run_dir=run_dir,
        timeout_seconds=7200,
        check=True,
    )

    payload: dict[str, Any] | None = None
    try:
        payload = _extract_json_object(Path(record.stdout_path).read_text(encoding="utf-8"))
    except Exception:
        saved = output_dir / f"benchmark_{BENCHMARK_SUITE}.json"
        if saved.exists():
            payload = _read_json(saved)

    if payload is None:
        raise RuntimeError(f"{stage}: unable to parse benchmark payload")
    return payload, record


def _extract_task_accuracy(payload: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for item in payload.get("benchmarks", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("benchmark", "")).strip()
        if not name:
            continue
        out[name] = {
            "accuracy": float(item.get("accuracy", 0.0)),
            "correct": int(item.get("correct", 0)),
            "total": int(item.get("total", 0)),
        }
    return out


def _load_task_prompts(limit: int) -> dict[str, list[str]]:
    from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

    loader = BenchmarkLoader()
    task_prompts: dict[str, list[str]] = {}
    for task in TASKS:
        benchmark = loader.load(task, split="test", limit=limit)
        prompts = [sample.prompt for sample in benchmark.samples if sample.prompt]
        if not prompts:
            raise ValueError(f"No prompts loaded for benchmark task: {task}")
        task_prompts[task] = prompts
    return task_prompts


def _collect_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    backend: Any,
) -> dict[int, list[Any]]:
    activations: dict[int, list[Any]] = {}
    for text in prompts:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx_raw, act in acts.items():
            layer_idx = int(layer_idx_raw)
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)
    return activations


def _compute_cka_summary(
    fp_activations: dict[int, list[Any]],
    cand_activations: dict[int, list[Any]],
    backend: Any,
) -> dict[str, Any]:
    import mlx.core as mx

    from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

    per_layer: dict[int, float] = {}
    common_layers = sorted(set(fp_activations) & set(cand_activations))
    for layer_idx in common_layers:
        a = fp_activations[layer_idx]
        b = cand_activations[layer_idx]
        if len(a) == 0 or len(b) == 0:
            continue
        n = min(len(a), len(b))
        mat_a = mx.stack(a[:n])
        mat_b = mx.stack(b[:n])
        mx.eval(mat_a, mat_b)
        score = compute_linear_cka_from_activations(mat_a, mat_b, backend)
        per_layer[layer_idx] = float(score)

    values = list(per_layer.values())
    return {
        "mean_cka": float(sum(values) / len(values)) if values else 0.0,
        "min_cka": float(min(values)) if values else 0.0,
        "per_layer_cka": per_layer,
        "n_layers": len(per_layer),
    }


def _sign_with_dead_zone(value: float, eps: float) -> int:
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _classify_accuracy_delta(
    *,
    baseline_correct: int,
    baseline_total: int,
    arm_correct: int,
    arm_total: int,
) -> dict[str, Any]:
    from modelcypher.core.domain.statistics import (
        clopper_pearson_interval,
        confidence_intervals_overlap,
    )

    if baseline_total <= 0 or arm_total <= 0:
        return {
            "classification": "indeterminate",
            "reason": "non_positive_total",
        }

    alpha = 1.0 / float(min(baseline_total, arm_total))
    baseline_ci = clopper_pearson_interval(
        n_correct=baseline_correct,
        n_total=baseline_total,
        alpha=alpha,
    )
    arm_ci = clopper_pearson_interval(
        n_correct=arm_correct,
        n_total=arm_total,
        alpha=alpha,
    )
    overlaps = confidence_intervals_overlap(baseline_ci, arm_ci)

    if overlaps:
        cls = "indeterminate"
    elif arm_ci[0] > baseline_ci[1]:
        cls = "increase"
    elif arm_ci[1] < baseline_ci[0]:
        cls = "decrease"
    else:
        cls = "indeterminate"

    return {
        "classification": cls,
        "alpha": alpha,
        "baseline_ci": [baseline_ci[0], baseline_ci[1]],
        "arm_ci": [arm_ci[0], arm_ci[1]],
        "ci_overlap": overlaps,
    }


def _evaluate_predictor_verdict(
    *,
    baseline_accuracy: Mapping[str, Mapping[str, float | int]],
    baseline_cka: Mapping[str, Mapping[str, Any]],
    arm_accuracy: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    arm_cka: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sqrt_eps: float,
) -> dict[str, Any]:
    per_point: list[dict[str, Any]] = []
    evaluable_matches: list[bool] = []

    for arm_name, metrics_by_task in arm_accuracy.items():
        cka_by_task = arm_cka.get(arm_name, {})
        for task in DEGRADED_TASKS:
            baseline_task = baseline_accuracy.get(task)
            arm_task = metrics_by_task.get(task)
            if baseline_task is None or arm_task is None:
                per_point.append(
                    {
                        "arm": arm_name,
                        "task": task,
                        "status": "missing_accuracy",
                    },
                )
                continue

            acc_eval = _classify_accuracy_delta(
                baseline_correct=int(baseline_task.get("correct", 0)),
                baseline_total=int(baseline_task.get("total", 0)),
                arm_correct=int(arm_task.get("correct", 0)),
                arm_total=int(arm_task.get("total", 0)),
            )
            delta_acc = float(arm_task.get("accuracy", 0.0)) - float(
                baseline_task.get("accuracy", 0.0),
            )

            baseline_cka_task = baseline_cka.get(task, {})
            arm_cka_task = cka_by_task.get(task, {})
            delta_cka = float(arm_cka_task.get("mean_cka", 0.0)) - float(
                baseline_cka_task.get("mean_cka", 0.0),
            )
            cka_sign = _sign_with_dead_zone(delta_cka, sqrt_eps)

            status = acc_eval["classification"]
            acc_sign = 0
            matched = None
            if status == "increase":
                acc_sign = 1
                matched = cka_sign == acc_sign
            elif status == "decrease":
                acc_sign = -1
                matched = cka_sign == acc_sign

            point = {
                "arm": arm_name,
                "task": task,
                "accuracy_delta": delta_acc,
                "cka_delta": delta_cka,
                "cka_sign": cka_sign,
                "accuracy_significance": acc_eval,
                "status": status,
                "matched": matched,
            }
            per_point.append(point)
            if status in {"increase", "decrease"} and matched is not None:
                evaluable_matches.append(bool(matched))

    if not evaluable_matches:
        verdict = "insufficient_evidence"
    elif all(evaluable_matches):
        verdict = "cka_predictive"
    else:
        verdict = "cka_non_predictive"

    return {
        "verdict": verdict,
        "n_evaluable_points": len(evaluable_matches),
        "per_point": per_point,
    }


def _build_gate_note(verdict: str) -> str:
    if verdict == "cka_predictive":
        return (
            "CKA appears predictive on statistically evaluable degraded-task points. "
            "Design gate_v2 from measured CKA-accuracy coupling, then validate on a held-out run."
        )
    if verdict == "cka_non_predictive":
        return (
            "CKA is not reliably predictive on statistically evaluable degraded-task points. "
            "Design gate_v2 directly on task outcomes and use CKA as explanatory telemetry only."
        )
    return (
        "Insufficient evidence (all degraded-task points indeterminate under CI overlap). "
        "Increase benchmark limit and re-run before specifying gate_v2 checks."
    )


def _write_markdown_report(
    *,
    run_dir: Path,
    combined: Mapping[str, Any],
) -> None:
    report_path = run_dir / "report.md"
    predictor = combined.get("predictor", {})
    arm_results = combined.get("arms", {})

    lines = [
        "# Quantized-Smarter Experiment Report",
        "",
        f"**Run ID:** `{combined.get('run_id')}`",
        f"**Generated:** `{combined.get('generated_at')}`",
        "",
        "## Scope",
        "",
        "`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`",
        "",
        "## Preflight",
        "",
    ]

    preflight = combined.get("preflight", {})
    lines.append(f"- passed: `{preflight.get('passed')}`")
    for step in preflight.get("steps", []):
        stage = step.get("stage")
        passed = step.get("passed")
        lines.append(f"- {stage}: `{passed}`")
        if not passed and step.get("error"):
            lines.append(f"  - error: `{step['error']}`")

    lines.extend(
        [
            "",
            "## Benchmark Accuracy (quick suite)",
            "",
            "| Arm | gsm8k | boolq | arc_easy |",
            "|-----|-------|-------|----------|",
        ],
    )

    for arm_name, arm_payload in arm_results.items():
        task_acc = arm_payload.get("benchmark", {}).get("task_accuracy", {})
        row = [arm_name]
        for task in TASKS:
            m = task_acc.get(task, {})
            if not m:
                row.append("n/a")
            else:
                row.append(f"{float(m.get('accuracy', 0.0)):.3f} ({int(m.get('correct', 0))}/{int(m.get('total', 0))})")
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines.extend(
        [
            "",
            "## Task-Conditioned CKA (mean)",
            "",
            "| Arm | gsm8k | boolq | arc_easy |",
            "|-----|-------|-------|----------|",
        ],
    )

    for arm_name, arm_payload in arm_results.items():
        task_cka = arm_payload.get("task_conditioned_cka", {})
        row = [arm_name]
        for task in TASKS:
            m = task_cka.get(task, {})
            row.append(f"{float(m.get('mean_cka', 0.0)):.4f}" if m else "n/a")
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines.extend(
        [
            "",
            "## Predictor Verdict",
            "",
            f"- verdict: `{predictor.get('verdict')}`",
            f"- evaluable degraded-task points: `{predictor.get('n_evaluable_points')}`",
            "",
            "## Gate Design Note",
            "",
            combined.get("gate_design_note", ""),
            "",
            "## Promotion Target",
            "",
            f"- `{combined.get('promotion_target')}`",
            "",
        ],
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_preflight_failure_payload(
    *,
    run_id: str,
    generated_at: str,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    command_records: list[CommandRecord],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "status": "preflight_failed",
        "config": {
            "fp_model": args.fp_model,
            "q4_model": args.q4_model,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "benchmark_limit": args.benchmark_limit,
            "max_iters": args.max_iters,
        },
        "preflight": preflight,
        "command_logs": [r.to_dict() for r in command_records],
        "promotion_target": args.promotion_target,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = _parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.now(timezone.utc).isoformat()
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_path = run_dir / "combined_results.json"

    LOGGER.info("quantized-smarter run_id=%s", run_id)
    LOGGER.info("output=%s", run_dir)

    for required in (args.fp_model, args.q4_model, args.train_dataset, args.eval_dataset):
        if not Path(required).exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    preflight, preflight_records = _run_preflight(args, run_dir)
    if not bool(preflight.get("passed", False)):
        payload = _build_preflight_failure_payload(
            run_id=run_id,
            generated_at=generated_at,
            args=args,
            preflight=preflight,
            command_records=preflight_records,
        )
        combined_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_markdown_report(run_dir=run_dir, combined=payload)
        LOGGER.error("Preflight failed at %s", preflight.get("failure_stage"))
        return

    command_records = list(preflight_records)
    arm_root = run_dir / "arms"
    arm_root.mkdir(parents=True, exist_ok=True)

    stage_to_record: dict[str, CommandRecord] = {}
    stage_payloads: dict[str, dict[str, Any]] = {}

    for stage, command, timeout_s in _arm_command_plan(args, run_dir):
        LOGGER.info("Running %s", stage)
        record = _run_command(
            stage=stage,
            command=command,
            run_dir=run_dir,
            timeout_seconds=timeout_s,
            check=True,
        )
        command_records.append(record)
        stage_to_record[stage] = record
        if stage == "arm_a.quantize_correct":
            stage_payloads[stage] = _extract_json_object(
                Path(record.stdout_path).read_text(encoding="utf-8"),
            )

    arm_a_model_path = str((arm_root / "arm_a_tikhonov_model").resolve())
    arm_b_payload, arm_b_run_dir = _load_corrective_run_payload(arm_root / "arm_b_corrective_lora")
    arm_c_payload, arm_c_run_dir = _load_corrective_run_payload(arm_root / "arm_c_combined")

    arm_specs: dict[str, dict[str, Any]] = {
        "q4_baseline": {
            "model_path": str(Path(args.q4_model).resolve()),
            "adapter_path": None,
            "precision_state": _infer_precision_state(args.q4_model, adapter_path=None),
        },
        "arm_a_tikhonov": {
            "model_path": arm_a_model_path,
            "adapter_path": None,
            "precision_state": {
                **_infer_precision_state(arm_a_model_path, adapter_path=None),
                "derivation": "tikhonov_corrected_from_q4",
            },
            "quantize_correct": stage_payloads.get("arm_a.quantize_correct", {}),
        },
        "arm_b_corrective_lora": {
            "model_path": str(Path(args.q4_model).resolve()),
            "adapter_path": str(Path(arm_b_payload["adapter_path"]).resolve()),
            "precision_state": {
                **_infer_precision_state(args.q4_model, adapter_path=arm_b_payload["adapter_path"]),
                "derivation": "q4_base_plus_corrective_adapter",
            },
            "corrective_lora": arm_b_payload,
            "corrective_run_dir": str(arm_b_run_dir),
        },
        "arm_c_combined": {
            "model_path": arm_a_model_path,
            "adapter_path": str(Path(arm_c_payload["adapter_path"]).resolve()),
            "precision_state": {
                **_infer_precision_state(arm_a_model_path, adapter_path=arm_c_payload["adapter_path"]),
                "derivation": "tikhonov_corrected_base_plus_corrective_adapter",
            },
            "corrective_lora": arm_c_payload,
            "corrective_run_dir": str(arm_c_run_dir),
        },
    }

    benchmark_payloads: dict[str, dict[str, Any]] = {}
    for arm_name in ("q4_baseline", "arm_a_tikhonov", "arm_b_corrective_lora", "arm_c_combined"):
        stage = f"benchmark.{arm_name}"
        LOGGER.info("Running %s", stage)
        payload, record = _run_benchmark_command(
            stage=stage,
            model_path=arm_specs[arm_name]["model_path"],
            adapter_path=arm_specs[arm_name]["adapter_path"],
            benchmark_limit=args.benchmark_limit,
            run_dir=run_dir,
        )
        command_records.append(record)
        benchmark_payloads[arm_name] = payload

    LOGGER.info("Collecting task prompts...")
    task_prompts = _load_task_prompts(limit=args.benchmark_limit)

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()
    loader = ModelLoader(backend)

    LOGGER.info("Computing FP reference task activations...")
    fp_model, fp_tokenizer = loader.load_model(args.fp_model)
    fp_activations_by_task: dict[str, dict[int, list[Any]]] = {}
    for task in TASKS:
        fp_activations_by_task[task] = _collect_activations(
            fp_model, fp_tokenizer, task_prompts[task], backend,
        )
    del fp_model, fp_tokenizer
    gc.collect()

    cka_by_arm: dict[str, dict[str, Any]] = {}
    sqrt_eps = math.sqrt(float(backend.finfo().eps))
    for arm_name in ("q4_baseline", "arm_a_tikhonov", "arm_b_corrective_lora", "arm_c_combined"):
        stage = f"cka.{arm_name}"
        LOGGER.info("Running %s", stage)
        model_path = arm_specs[arm_name]["model_path"]
        adapter_path = arm_specs[arm_name]["adapter_path"]
        cand_model, cand_tokenizer = loader.load_model(model_path, adapter_path=adapter_path)
        arm_task_cka: dict[str, Any] = {}
        for task in TASKS:
            cand_acts = _collect_activations(
                cand_model, cand_tokenizer, task_prompts[task], backend,
            )
            arm_task_cka[task] = _compute_cka_summary(
                fp_activations_by_task[task], cand_acts, backend,
            )
        cka_by_arm[arm_name] = arm_task_cka
        del cand_model, cand_tokenizer
        gc.collect()

    arm_results: dict[str, Any] = {}
    for arm_name, spec in arm_specs.items():
        arm_results[arm_name] = {
            "model_path": spec["model_path"],
            "adapter_path": spec["adapter_path"],
            "precision_state": spec["precision_state"],
            "benchmark": {
                "suite": BENCHMARK_SUITE,
                "task_accuracy": _extract_task_accuracy(benchmark_payloads[arm_name]),
                "raw_payload": benchmark_payloads[arm_name],
            },
            "task_conditioned_cka": cka_by_arm[arm_name],
        }
        if "quantize_correct" in spec:
            arm_results[arm_name]["quantize_correct"] = spec["quantize_correct"]
        if "corrective_lora" in spec:
            arm_results[arm_name]["corrective_lora"] = spec["corrective_lora"]
            arm_results[arm_name]["corrective_run_dir"] = spec.get("corrective_run_dir")

    baseline_acc = arm_results["q4_baseline"]["benchmark"]["task_accuracy"]
    baseline_cka = arm_results["q4_baseline"]["task_conditioned_cka"]
    predictor = _evaluate_predictor_verdict(
        baseline_accuracy=baseline_acc,
        baseline_cka=baseline_cka,
        arm_accuracy={
            name: arm_results[name]["benchmark"]["task_accuracy"]
            for name in ("arm_a_tikhonov", "arm_b_corrective_lora", "arm_c_combined")
        },
        arm_cka={
            name: arm_results[name]["task_conditioned_cka"]
            for name in ("arm_a_tikhonov", "arm_b_corrective_lora", "arm_c_combined")
        },
        sqrt_eps=sqrt_eps,
    )
    gate_design_note = _build_gate_note(predictor["verdict"])

    combined = {
        "run_id": run_id,
        "generated_at": generated_at,
        "status": "completed",
        "stage_order": _execution_stage_order(),
        "config": {
            "fp_model": args.fp_model,
            "q4_model": args.q4_model,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "benchmark_limit": args.benchmark_limit,
            "max_iters": args.max_iters,
            "benchmark_suite": BENCHMARK_SUITE,
        },
        "preflight": preflight,
        "arms": arm_results,
        "predictor": predictor,
        "sqrt_eps": sqrt_eps,
        "gate_design_note": gate_design_note,
        "promotion_target": args.promotion_target,
        "command_logs": [r.to_dict() for r in command_records],
    }

    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    _write_markdown_report(run_dir=run_dir, combined=combined)
    LOGGER.info("Wrote %s", combined_path)
    LOGGER.info("Wrote %s", run_dir / "report.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.exception("quantized_smarter_experiment failed: %s", exc)
        raise
