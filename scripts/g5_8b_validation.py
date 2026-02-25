#!/usr/bin/env python3
"""G5 8B validation runner with resumable capacity analysis and gate extraction.

This script runs one seeded validation job (capacity -> training -> gates)
and can aggregate per-seed gate artifacts.

Usage:
    # Single seed
    poetry run python scripts/g5_8b_validation.py --seed 41

    # Aggregate completed seeds
    poetry run python scripts/g5_8b_validation.py --aggregate-root results/g5_8b_validation
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
TRAIN_DATA_DEFAULT = "data/training/benchmark_train.jsonl"
EVAL_DATA_DEFAULT = "data/training/benchmark_val.jsonl"
RETENTION_DATA_DEFAULT = "data/training/retention_replay.jsonl"
OUTPUT_ROOT_DEFAULT = Path("results/g5_8b_validation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G5 8B validation or aggregate seed gate artifacts.",
    )
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--train-data", default=TRAIN_DATA_DEFAULT)
    parser.add_argument("--eval-data", default=EVAL_DATA_DEFAULT)
    parser.add_argument(
        "--retention-data",
        default=RETENTION_DATA_DEFAULT,
        help=(
            "Retention replay dataset (JSONL). "
            "Provide empty string to disable retention replay."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online-eval-n", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument(
        "--capacity-checkpoint",
        type=Path,
        default=None,
        help="Optional capacity checkpoint file path.",
    )
    parser.add_argument(
        "--resume-capacity",
        action="store_true",
        help="Resume capacity profiling from checkpoint.",
    )
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        default=None,
        help="If set, aggregate existing seed artifacts and exit.",
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
    return logging.getLogger("g5_8b_validation")


def _fourgram_repetition_rate(text: str) -> float:
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def _safe_clear_backend_cache(backend: Any) -> None:
    clear_fn = getattr(backend, "clear_cache", None)
    if callable(clear_fn):
        try:
            clear_fn()
        except Exception:
            pass


def _bytes_to_gb(value: int) -> float:
    return float(value) / float(1024**3)


def _capture_memory_snapshot(backend: Any, stage: str) -> dict[str, Any]:
    proc = psutil.Process()
    proc_mem = proc.memory_info()
    vm = psutil.virtual_memory()

    active_fn = getattr(backend, "get_active_memory_gb", None)
    peak_fn = getattr(backend, "get_peak_memory_gb", None)

    backend_active = float(active_fn()) if callable(active_fn) else None
    backend_peak = float(peak_fn()) if callable(peak_fn) else None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "process_rss_gb": _bytes_to_gb(proc_mem.rss),
        "process_vms_gb": _bytes_to_gb(proc_mem.vms),
        "system_total_gb": _bytes_to_gb(vm.total),
        "system_used_gb": _bytes_to_gb(vm.used),
        "system_available_gb": _bytes_to_gb(vm.available),
        "backend_active_gb": backend_active,
        "backend_peak_gb": backend_peak,
    }


def _log_memory_snapshot(run_log: logging.Logger, snapshot: dict[str, Any]) -> None:
    run_log.info(
        "Memory snapshot [%s] | rss=%.2f GB vms=%.2f GB "
        "backend_active=%s backend_peak=%s "
        "system_used=%.2f GB system_available=%.2f GB",
        snapshot["stage"],
        snapshot["process_rss_gb"],
        snapshot["process_vms_gb"],
        (
            f"{snapshot['backend_active_gb']:.2f} GB"
            if snapshot["backend_active_gb"] is not None
            else "n/a"
        ),
        (
            f"{snapshot['backend_peak_gb']:.2f} GB"
            if snapshot["backend_peak_gb"] is not None
            else "n/a"
        ),
        snapshot["system_used_gb"],
        snapshot["system_available_gb"],
    )


def _build_probe_prompts(problems: list[Any]) -> list[str]:
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )

    n_demonstrations = len(default_few_shot_examples())
    return [
        build_forward_prompt(problem, demonstrations=n_demonstrations)
        for problem in problems
    ]


def _collect_responses(
    backend: Any,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_tokens: int,
) -> list[str]:
    responses: list[str] = []
    for prompt in prompts:
        responses.append(backend.generate(model, tokenizer, prompt, max_tokens=max_tokens))
    return responses


def _final_online_eval(epoch_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    history = [
        {
            "epoch": em.get("epoch"),
            "accuracy": em.get("online_eval_accuracy"),
            "n_correct": em.get("online_eval_n_correct"),
            "n_total": em.get("online_eval_n_total"),
            "degraded": em.get("online_eval_degraded"),
        }
        for em in epoch_metrics
        if em.get("online_eval_accuracy") is not None
    ]
    return history[-1] if history else None


def _run_seed(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    train_data = Path(args.train_data).expanduser().resolve()
    eval_data = Path(args.eval_data).expanduser().resolve()
    retention_data = (
        Path(args.retention_data).expanduser().resolve()
        if args.retention_data
        else None
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not train_data.exists():
        raise FileNotFoundError(f"Train data does not exist: {train_data}")
    if not eval_data.exists():
        raise FileNotFoundError(f"Eval data does not exist: {eval_data}")
    if retention_data is not None and not retention_data.exists():
        raise FileNotFoundError(f"Retention data does not exist: {retention_data}")

    output_root = args.output_root.expanduser().resolve()
    seed_dir = output_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    run_log = _configure_logging(seed_dir / "run.log")
    run_log.info("=" * 72)
    run_log.info("G5 8B Validation | seed=%d", args.seed)
    run_log.info("Model=%s", model_path)
    run_log.info("Train data=%s", train_data)
    run_log.info("Eval data=%s", eval_data)
    run_log.info(
        "Retention data=%s",
        retention_data if retention_data is not None else "<disabled>",
    )
    run_log.info("Output=%s", seed_dir)

    from modelcypher.cli.composition import (
        get_capacity_analysis_service,
        get_dataset_training_service,
    )
    from modelcypher.core.domain.training.online_eval import (
        create_eval_problem_set,
        evaluate_correctness,
    )

    backend = initialize_default_backend()
    memory_trace: list[dict[str, Any]] = []
    memory_trace_path = seed_dir / "memory_trace.json"

    def _record_memory(stage: str) -> None:
        snapshot = _capture_memory_snapshot(backend, stage)
        memory_trace.append(snapshot)
        _log_memory_snapshot(run_log, snapshot)
        memory_trace_path.write_text(
            json.dumps({"snapshots": memory_trace}, indent=2),
            encoding="utf-8",
        )

    _record_memory("start")

    # ------------------------------------------------------------------
    # 1) Capacity profiling with checkpoint/resume
    # ------------------------------------------------------------------
    checkpoint_path = (
        args.capacity_checkpoint.expanduser().resolve()
        if args.capacity_checkpoint is not None
        else (seed_dir / "capacity_checkpoint.json")
    )

    capacity_service = get_capacity_analysis_service()
    capacity_report = capacity_service.analyze(
        model_path=str(model_path),
        checkpoint_path=checkpoint_path,
        resume=bool(args.resume_capacity),
    )
    (seed_dir / "capacity_report.json").write_text(
        json.dumps(capacity_report.to_dict(), indent=2),
        encoding="utf-8",
    )
    run_log.info(
        "Capacity profiling complete: analyzed_layers=%d analyzed_parameters=%d",
        capacity_report.analyzed_layers,
        capacity_report.analyzed_parameters,
    )
    _record_memory("after_capacity")

    # ------------------------------------------------------------------
    # 2) Baseline accuracy + repetition envelope on fixed probe set
    # ------------------------------------------------------------------
    probe_problems = create_eval_problem_set(
        n_problems=args.online_eval_n,
        seed=args.seed + 1,
    )
    probe_prompts = _build_probe_prompts(probe_problems)

    baseline_model, baseline_tokenizer = backend.load_model(str(model_path))

    baseline_cache: dict[str, str] = {}

    def _baseline_generate(prompt: str, max_tokens: int) -> str:
        if prompt in baseline_cache:
            return baseline_cache[prompt]
        response = backend.generate(baseline_model, baseline_tokenizer, prompt, max_tokens)
        baseline_cache[prompt] = response
        return response

    baseline_eval = evaluate_correctness(
        problems=probe_problems,
        generate_fn=_baseline_generate,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=512,
    )
    baseline_responses = _collect_responses(
        backend,
        baseline_model,
        baseline_tokenizer,
        probe_prompts,
        max_tokens=512,
    )
    baseline_repetition = [_fourgram_repetition_rate(resp) for resp in baseline_responses]
    baseline_max_4gram_repeat = max(baseline_repetition) if baseline_repetition else 0.0
    _record_memory("after_baseline_eval")

    del baseline_model
    del baseline_tokenizer
    gc.collect()
    _safe_clear_backend_cache(backend)
    _record_memory("after_baseline_cleanup")

    # ------------------------------------------------------------------
    # 3) Full training run
    # ------------------------------------------------------------------
    dataset_service = get_dataset_training_service()
    adapter_path = seed_dir / "adapter"
    result = dataset_service.train_from_dataset(
        model_path=str(model_path),
        dataset_path=str(train_data),
        eval_dataset_path=str(eval_data),
        retention_dataset_path=(str(retention_data) if retention_data else None),
        output_path=str(adapter_path),
        seed=args.seed,
        auto_regime=True,
        regime_n_problems=args.online_eval_n,
        online_eval=True,
        online_eval_n_problems=args.online_eval_n,
        eval_interval=args.eval_interval,
    )

    result_dict = result.to_dict()
    (seed_dir / "train_result.json").write_text(
        json.dumps(result_dict, indent=2),
        encoding="utf-8",
    )
    _record_memory("after_training")

    epoch_metrics = result_dict.get("epoch_metrics", [])
    final_online_eval = _final_online_eval(epoch_metrics)
    final_correct = (
        int(final_online_eval["n_correct"])
        if final_online_eval is not None and final_online_eval.get("n_correct") is not None
        else None
    )

    # ------------------------------------------------------------------
    # 4) Degeneracy gate on fixed probe set
    # ------------------------------------------------------------------
    adapted_adapter_path = result.adapter_path or str(adapter_path)
    adapted_model, adapted_tokenizer = backend.load_model(
        str(model_path),
        str(adapted_adapter_path),
    )
    adapted_responses = _collect_responses(
        backend,
        adapted_model,
        adapted_tokenizer,
        probe_prompts,
        max_tokens=512,
    )
    adapted_repetition = [_fourgram_repetition_rate(resp) for resp in adapted_responses]
    adapted_max_4gram_repeat = max(adapted_repetition) if adapted_repetition else 0.0
    _record_memory("after_adapted_eval")

    del adapted_model
    del adapted_tokenizer
    gc.collect()
    _safe_clear_backend_cache(backend)
    _record_memory("after_adapted_cleanup")

    # ------------------------------------------------------------------
    # 5) Gate extraction
    # ------------------------------------------------------------------
    eps_f32 = machine_epsilon(backend, backend.array([1.0], dtype="float32"))
    sqrt_eps_f32 = math.sqrt(eps_f32)

    no_crash = bool(adapted_adapter_path and Path(adapted_adapter_path).exists())
    min_cka = result.min_cka
    cka_ok = min_cka is not None and min_cka >= (1.0 - sqrt_eps_f32)
    spectral_ok = (
        bool(result.spectral_bounds_ok)
        and float(result.max_spectral_ratio) <= (1.0 + sqrt_eps_f32)
    )
    accuracy_ok = final_correct is not None and final_correct >= baseline_eval.n_correct
    degenerate_ok = adapted_max_4gram_repeat <= (baseline_max_4gram_repeat + sqrt_eps_f32)

    gates = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_path": str(model_path),
        "train_data": str(train_data),
        "eval_data": str(eval_data),
        "retention_data": (str(retention_data) if retention_data else None),
        "capacity": {
            "checkpoint_path": str(checkpoint_path),
            "analyzed_layers": capacity_report.analyzed_layers,
            "analyzed_parameters": capacity_report.analyzed_parameters,
        },
        "baseline_probe": {
            "online_eval_n": args.online_eval_n,
            "baseline_correct": baseline_eval.n_correct,
            "baseline_total": baseline_eval.n_total,
            "baseline_accuracy": baseline_eval.accuracy,
            "baseline_max_4gram_repeat": baseline_max_4gram_repeat,
        },
        "adapted_probe": {
            "final_online_eval": final_online_eval,
            "adapted_max_4gram_repeat": adapted_max_4gram_repeat,
        },
        "precision_terms": {
            "eps_f32": eps_f32,
            "sqrt_eps_f32": sqrt_eps_f32,
        },
        "gates": {
            "no_crash": no_crash,
            "cka_ok": cka_ok,
            "spectral_ok": spectral_ok,
            "accuracy_ok": accuracy_ok,
            "degenerate_ok": degenerate_ok,
        },
        "diagnostics": {
            "min_cka": min_cka,
            "max_spectral_ratio": result.max_spectral_ratio,
            "spectral_bounds_ok": result.spectral_bounds_ok,
            "adapter_path": adapted_adapter_path,
            "memory_note": (
                "process_vms_gb is virtual address reservation and can exceed "
                "physical RAM; use process_rss_gb, system_used_gb, and backend_* "
                "for resident/allocator pressure."
            ),
            "all_gates_pass": bool(
                no_crash and cka_ok and spectral_ok and accuracy_ok and degenerate_ok
            ),
        },
    }

    (seed_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    (seed_dir / "probe_responses.json").write_text(
        json.dumps(
            {
                "prompts": probe_prompts,
                "baseline_responses": baseline_responses,
                "adapted_responses": adapted_responses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run_log.info(
        "Gate status | no_crash=%s cka_ok=%s spectral_ok=%s accuracy_ok=%s degenerate_ok=%s",
        no_crash,
        cka_ok,
        spectral_ok,
        accuracy_ok,
        degenerate_ok,
    )


def _aggregate(aggregate_root: Path) -> None:
    root = aggregate_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Aggregate root does not exist: {root}")

    seed_payloads: dict[str, dict[str, Any]] = {}
    for seed_dir in sorted(root.glob("seed*")):
        gates_path = seed_dir / "gates.json"
        if not gates_path.exists():
            continue
        payload = json.loads(gates_path.read_text(encoding="utf-8"))
        seed_payloads[seed_dir.name] = payload

    if not seed_payloads:
        raise ValueError(f"No seed gates found under {root}")

    gate_names = ["no_crash", "cka_ok", "spectral_ok", "accuracy_ok", "degenerate_ok"]
    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_root": str(root),
        "n_seeds": len(seed_payloads),
        "per_seed": {},
        "gate_pass_counts": {gate: 0 for gate in gate_names},
        "all_gates_all_seeds": True,
    }

    for seed_name, payload in sorted(seed_payloads.items()):
        gates = payload.get("gates", {})
        diagnostics = payload.get("diagnostics", {})
        per_seed_summary = {
            "gates": {gate: bool(gates.get(gate, False)) for gate in gate_names},
            "min_cka": diagnostics.get("min_cka"),
            "max_spectral_ratio": diagnostics.get("max_spectral_ratio"),
            "all_gates_pass": bool(diagnostics.get("all_gates_pass", False)),
        }
        aggregate["per_seed"][seed_name] = per_seed_summary

        for gate in gate_names:
            if per_seed_summary["gates"][gate]:
                aggregate["gate_pass_counts"][gate] += 1
            else:
                aggregate["all_gates_all_seeds"] = False

        if not per_seed_summary["all_gates_pass"]:
            aggregate["all_gates_all_seeds"] = False

    multiseed_path = root / "multiseed_gates.json"
    multiseed_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    report_lines = [
        "# G5 8B Validation Report",
        "",
        f"Generated: {aggregate['generated_at']}",
        f"Aggregate root: {aggregate['aggregate_root']}",
        f"Seeds: {aggregate['n_seeds']}",
        "",
        "## Gate Summary",
        "",
        "| Seed | no_crash | cka_ok | spectral_ok | accuracy_ok | degenerate_ok | all_gates_pass |",
        "|------|----------|--------|-------------|-------------|---------------|----------------|",
    ]

    for seed_name, payload in sorted(aggregate["per_seed"].items()):
        gates = payload["gates"]
        report_lines.append(
            "| "
            f"{seed_name} | "
            f"{gates['no_crash']} | "
            f"{gates['cka_ok']} | "
            f"{gates['spectral_ok']} | "
            f"{gates['accuracy_ok']} | "
            f"{gates['degenerate_ok']} | "
            f"{payload['all_gates_pass']} |"
        )

    report_lines.extend(
        [
            "",
            "## Aggregate Verdict",
            "",
            f"- all_gates_all_seeds: `{aggregate['all_gates_all_seeds']}`",
        ],
    )

    report_path = root / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {multiseed_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    args = _parse_args()
    if args.aggregate_root is not None:
        _aggregate(args.aggregate_root)
        return

    _run_seed(args)


if __name__ == "__main__":
    main()
