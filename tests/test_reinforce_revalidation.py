# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "reinforce_revalidation.py"
    spec = importlib.util.spec_from_file_location(
        "reinforce_revalidation_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_log(
    root: Path,
    arm: str,
    seed: int,
    *,
    final_correct: int,
    final_total: int,
    final_mechanistic_correct: int,
    final_mechanistic_total: int,
    total_outcome_steps: int,
    no_degradation: bool,
    no_mechanistic_degradation: bool,
    gate_confound_count: int,
) -> None:
    seed_dir = root / arm / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "arm_name": arm,
        "final_correct": final_correct,
        "final_total": final_total,
        "final_mechanistic_correct": final_mechanistic_correct,
        "final_mechanistic_total": final_mechanistic_total,
        "reinforce_summary": {
            "total_outcome_steps": total_outcome_steps,
        },
        "success_criteria": {
            "no_degradation_all_checkpoints": no_degradation,
            "no_mechanistic_degradation_all_checkpoints": no_mechanistic_degradation,
        },
        "gate_confound_event_count": gate_confound_count,
    }
    (seed_dir / "run_log.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_aggregate_emits_canonical_and_mechanistic_verdicts(tmp_path: Path):
    script = _load_script_module()

    _write_run_log(
        tmp_path,
        "ce_control",
        41,
        final_correct=50,
        final_total=100,
        final_mechanistic_correct=50,
        final_mechanistic_total=100,
        total_outcome_steps=0,
        no_degradation=True,
        no_mechanistic_degradation=True,
        gate_confound_count=0,
    )
    _write_run_log(
        tmp_path,
        "ce_control",
        42,
        final_correct=52,
        final_total=100,
        final_mechanistic_correct=52,
        final_mechanistic_total=100,
        total_outcome_steps=0,
        no_degradation=True,
        no_mechanistic_degradation=True,
        gate_confound_count=0,
    )

    _write_run_log(
        tmp_path,
        "force_reinforce",
        41,
        final_correct=50,
        final_total=100,
        final_mechanistic_correct=51,
        final_mechanistic_total=100,
        total_outcome_steps=3,
        no_degradation=False,
        no_mechanistic_degradation=True,
        gate_confound_count=1,
    )
    _write_run_log(
        tmp_path,
        "force_reinforce",
        42,
        final_correct=52,
        final_total=100,
        final_mechanistic_correct=53,
        final_mechanistic_total=100,
        total_outcome_steps=2,
        no_degradation=False,
        no_mechanistic_degradation=True,
        gate_confound_count=2,
    )

    script._aggregate(tmp_path, baseline_arm="ce_control")

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    comp = summary["comparisons"]["force_reinforce"]

    assert comp["canonical_verdict"] == "CEILING"
    assert comp["mechanistic_verdict"] == "UNLOCKED"
    assert comp["verdict"] == "CEILING"
    assert comp["gate_confound_event_count_total"] == 3
    assert comp["gate_confound_event_count_by_seed"] == {"41": 1, "42": 2}

    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "Canonical" in report
    assert "Mechanistic" in report


def test_resolve_arm_name_includes_research_flags_when_non_default():
    script = _load_script_module()

    args = SimpleNamespace(
        arm_name=None,
        mode="force_reinforce",
        research_online_eval_stop_stage="post_outcome",
        research_outcome_selector="lost_only",
    )
    arm = script._resolve_arm_name(args)
    assert arm == "force_reinforce__stop_post_outcome__selector_lost_only"

    args_default = SimpleNamespace(
        arm_name=None,
        mode="force_reinforce",
        research_online_eval_stop_stage="pre_outcome",
        research_outcome_selector="all",
    )
    assert script._resolve_arm_name(args_default) == "force_reinforce"
