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


def test_normalize_primary_scores_uses_expected_metric_keys():
    script = _load_script_module("leaderboard_baseline_script", "leaderboard_baseline.py")
    tasks = script.resolve_task_list(None)
    raw_payload = {
        "groups": {
            "leaderboard_ifeval": {
                "prompt_level_strict_acc,none": 0.40,
                "prompt_level_strict_acc_stderr,none": 0.02,
            },
            "leaderboard_bbh": {"acc_norm,none": 0.32},
            "leaderboard_math_hard": {"exact_match,none": 0.11},
            "leaderboard_gpqa": {"acc_norm,none": 0.28},
            "leaderboard_musr": {"acc_norm,none": 0.35},
            "leaderboard_mmlu_pro": {"acc,none": 0.44},
        },
    }

    normalized = script.normalize_primary_scores(raw_payload, tasks)

    assert normalized["leaderboard_ifeval"]["metric"] == "prompt_level_strict_acc,none"
    assert normalized["leaderboard_ifeval"]["score"] == pytest.approx(0.40)
    assert normalized["leaderboard_ifeval"]["stderr"] == pytest.approx(0.02)
    assert normalized["leaderboard_math_hard"]["metric"] == "exact_match,none"
    assert normalized["leaderboard_mmlu_pro"]["metric"] == "acc,none"


def test_composite_mean_uses_only_primary_scores():
    script = _load_script_module("leaderboard_baseline_script_mean", "leaderboard_baseline.py")
    primary_scores = {
        "leaderboard_ifeval": {"metric": "prompt_level_strict_acc,none", "score": 0.10, "stderr": None},
        "leaderboard_bbh": {"metric": "acc_norm,none", "score": 0.30, "stderr": None},
        "leaderboard_math_hard": {"metric": "exact_match,none", "score": 0.50, "stderr": None},
    }

    assert script.compute_composite_mean(primary_scores) == pytest.approx((0.10 + 0.30 + 0.50) / 3.0)


def test_resolve_task_list_rejects_unknown_task():
    script = _load_script_module("leaderboard_baseline_script_tasks", "leaderboard_baseline.py")

    with pytest.raises(ValueError, match="Unknown leaderboard tasks"):
        script.resolve_task_list("leaderboard_ifeval,leaderboard_fake")
