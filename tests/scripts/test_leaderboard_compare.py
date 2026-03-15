# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module(module_name: str, script_name: str) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_payload() -> dict:
    return {
        "model_path": "/tmp/base-model",
        "model_name": "base-model",
        "suite_name": "leaderboard_v2",
        "task_list": [
            "leaderboard_ifeval",
            "leaderboard_bbh",
        ],
        "primary_score_metrics": {
            "leaderboard_ifeval": "prompt_level_strict_acc,none",
            "leaderboard_bbh": "acc_norm,none",
        },
        "primary_scores": {
            "leaderboard_ifeval": {
                "metric": "prompt_level_strict_acc,none",
                "score": 0.41,
                "stderr": 0.01,
            },
            "leaderboard_bbh": {
                "metric": "acc_norm,none",
                "score": 0.22,
                "stderr": None,
            },
        },
        "composite_mean": (0.41 + 0.22) / 2.0,
    }


def test_compute_deltas_zero_when_scores_match():
    script = _load_script_module("leaderboard_compare_script", "leaderboard_compare.py")
    baseline_payload = _baseline_payload()
    current_scores = {
        "leaderboard_ifeval": {
            "metric": "prompt_level_strict_acc,none",
            "score": 0.41,
            "stderr": 0.01,
        },
        "leaderboard_bbh": {
            "metric": "acc_norm,none",
            "score": 0.22,
            "stderr": None,
        },
    }

    deltas = script._compute_deltas(baseline_payload, current_scores)

    assert deltas["leaderboard_ifeval"] == pytest.approx(0.0)
    assert deltas["leaderboard_bbh"] == pytest.approx(0.0)
    assert deltas["composite_mean"] == pytest.approx(0.0)


def test_resolve_comparison_task_list_rejects_mismatch():
    script = _load_script_module("leaderboard_compare_script_tasks", "leaderboard_compare.py")
    baseline_payload = _baseline_payload()

    with pytest.raises(ValueError, match="must match the baseline task_list exactly"):
        script._resolve_comparison_task_list(
            baseline_payload,
            "leaderboard_ifeval,leaderboard_mmlu_pro",
        )


def test_validate_baseline_metric_map_rejects_mismatch():
    script = _load_script_module("leaderboard_compare_script_metrics", "leaderboard_compare.py")
    baseline_payload = _baseline_payload()
    baseline_payload["primary_score_metrics"] = {
        "leaderboard_ifeval": "acc,none",
        "leaderboard_bbh": "acc_norm,none",
    }

    with pytest.raises(ValueError, match="primary_score_metrics"):
        script._validate_baseline_metric_map(
            baseline_payload,
            ["leaderboard_ifeval", "leaderboard_bbh"],
        )
