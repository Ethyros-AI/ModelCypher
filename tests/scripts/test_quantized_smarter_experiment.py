# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "quantized_smarter_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "quantized_smarter_experiment_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fp_model="/tmp/fp",
        q4_model="/tmp/q4",
        train_dataset="/tmp/train.jsonl",
        eval_dataset="/tmp/eval.jsonl",
        benchmark_limit=100,
        max_iters=10,
        output_dir=str(tmp_path),
        promotion_target="docs/research/quantized_smarter_qwen35_0.8b.md",
        preflight_n_calibration=2,
        preflight_max_seq_len=32,
        preflight_cka_probes=2,
    )


def test_execution_stage_order_contains_expected_sequence():
    script = _load_script_module()
    order = script._execution_stage_order()

    assert order.index("arm_a.quantize_correct") < order.index("arm_b.corrective_lora")
    assert order.index("arm_b.corrective_lora") < order.index("arm_c.corrective_lora")
    assert order.index("benchmark.q4_baseline") < order.index("benchmark.arm_a_tikhonov")
    assert order.index("benchmark.arm_a_tikhonov") < order.index("benchmark.arm_b_corrective_lora")
    assert order.index("benchmark.arm_b_corrective_lora") < order.index("benchmark.arm_c_combined")


def test_preflight_fail_fast_stops_after_first_failed_command(tmp_path: Path):
    script = _load_script_module()
    args = _args(tmp_path)
    called: list[str] = []

    def _fake_run_command(*, stage: str, command, run_dir, timeout_seconds, check):
        _ = (command, run_dir, timeout_seconds, check)
        called.append(stage)
        raise RuntimeError("synthetic failure")

    preflight, records = script._run_preflight(
        args,
        tmp_path,
        run_command_fn=_fake_run_command,
        architecture_smoke_fn=lambda _fp, _q4: None,
    )

    assert preflight["passed"] is False
    assert preflight["failure_stage"] == "preflight.quantize_correct"
    assert called == ["preflight.quantize_correct"]
    assert records == []


def test_extract_json_object_from_mixed_stdout():
    script = _load_script_module()
    payload = script._extract_json_object(
        "Running benchmark suite...\n{\"suite\":\"quick\",\"benchmarks\":[]}\n",
    )
    assert payload["suite"] == "quick"


def test_extract_task_accuracy_parses_benchmark_payload():
    script = _load_script_module()
    payload = {
        "benchmarks": [
            {"benchmark": "gsm8k", "accuracy": 0.25, "correct": 25, "total": 100},
            {"benchmark": "boolq", "accuracy": 0.5, "correct": 50, "total": 100},
        ],
    }
    parsed = script._extract_task_accuracy(payload)
    assert parsed["gsm8k"]["correct"] == 25
    assert parsed["gsm8k"]["total"] == 100
    assert parsed["boolq"]["accuracy"] == 0.5
