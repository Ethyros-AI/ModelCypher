# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "quantized_smarter_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "quantized_smarter_experiment_predictor_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_accuracy_delta_increase_non_overlap():
    script = _load_script_module()
    out = script._classify_accuracy_delta(
        baseline_correct=10,
        baseline_total=100,
        arm_correct=90,
        arm_total=100,
    )
    assert out["classification"] == "increase"
    assert out["ci_overlap"] is False


def test_classify_accuracy_delta_decrease_non_overlap():
    script = _load_script_module()
    out = script._classify_accuracy_delta(
        baseline_correct=90,
        baseline_total=100,
        arm_correct=10,
        arm_total=100,
    )
    assert out["classification"] == "decrease"
    assert out["ci_overlap"] is False


def test_classify_accuracy_delta_indeterminate_overlap():
    script = _load_script_module()
    out = script._classify_accuracy_delta(
        baseline_correct=50,
        baseline_total=100,
        arm_correct=52,
        arm_total=100,
    )
    assert out["classification"] == "indeterminate"
    assert out["ci_overlap"] is True


def test_predictor_verdict_insufficient_evidence_when_all_indeterminate():
    script = _load_script_module()
    baseline_accuracy = {
        "gsm8k": {"accuracy": 0.50, "correct": 50, "total": 100},
        "boolq": {"accuracy": 0.50, "correct": 50, "total": 100},
    }
    baseline_cka = {
        "gsm8k": {"mean_cka": 0.70},
        "boolq": {"mean_cka": 0.70},
    }
    arm_accuracy = {
        "arm_a_tikhonov": {
            "gsm8k": {"accuracy": 0.51, "correct": 51, "total": 100},
            "boolq": {"accuracy": 0.49, "correct": 49, "total": 100},
        },
    }
    arm_cka = {
        "arm_a_tikhonov": {
            "gsm8k": {"mean_cka": 0.71},
            "boolq": {"mean_cka": 0.69},
        },
    }

    result = script._evaluate_predictor_verdict(
        baseline_accuracy=baseline_accuracy,
        baseline_cka=baseline_cka,
        arm_accuracy=arm_accuracy,
        arm_cka=arm_cka,
        sqrt_eps=1e-3,
    )
    assert result["verdict"] == "insufficient_evidence"
    assert result["n_evaluable_points"] == 0


def test_predictor_verdict_predictive_when_signs_match():
    script = _load_script_module()
    baseline_accuracy = {
        "gsm8k": {"accuracy": 0.10, "correct": 10, "total": 100},
        "boolq": {"accuracy": 0.10, "correct": 10, "total": 100},
    }
    baseline_cka = {
        "gsm8k": {"mean_cka": 0.50},
        "boolq": {"mean_cka": 0.50},
    }
    arm_accuracy = {
        "arm_a_tikhonov": {
            "gsm8k": {"accuracy": 0.90, "correct": 90, "total": 100},
            "boolq": {"accuracy": 0.90, "correct": 90, "total": 100},
        },
    }
    arm_cka = {
        "arm_a_tikhonov": {
            "gsm8k": {"mean_cka": 0.80},
            "boolq": {"mean_cka": 0.82},
        },
    }

    result = script._evaluate_predictor_verdict(
        baseline_accuracy=baseline_accuracy,
        baseline_cka=baseline_cka,
        arm_accuracy=arm_accuracy,
        arm_cka=arm_cka,
        sqrt_eps=1e-3,
    )
    assert result["verdict"] == "cka_predictive"
    assert result["n_evaluable_points"] == 2


def test_predictor_verdict_non_predictive_when_signs_mismatch():
    script = _load_script_module()
    baseline_accuracy = {
        "gsm8k": {"accuracy": 0.10, "correct": 10, "total": 100},
        "boolq": {"accuracy": 0.10, "correct": 10, "total": 100},
    }
    baseline_cka = {
        "gsm8k": {"mean_cka": 0.80},
        "boolq": {"mean_cka": 0.80},
    }
    arm_accuracy = {
        "arm_a_tikhonov": {
            "gsm8k": {"accuracy": 0.90, "correct": 90, "total": 100},
            "boolq": {"accuracy": 0.90, "correct": 90, "total": 100},
        },
    }
    arm_cka = {
        "arm_a_tikhonov": {
            "gsm8k": {"mean_cka": 0.60},
            "boolq": {"mean_cka": 0.61},
        },
    }

    result = script._evaluate_predictor_verdict(
        baseline_accuracy=baseline_accuracy,
        baseline_cka=baseline_cka,
        arm_accuracy=arm_accuracy,
        arm_cka=arm_cka,
        sqrt_eps=1e-3,
    )
    assert result["verdict"] == "cka_non_predictive"
    assert result["n_evaluable_points"] == 2
