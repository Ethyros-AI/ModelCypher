# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for pipeline_validation report rendering."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pipeline_validation_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "pipeline_validation.py"
    spec = importlib.util.spec_from_file_location("pipeline_validation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/pipeline_validation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_includes_dual_verdict_fields():
    pipeline_validation = _load_pipeline_validation_module()

    all_results = {
        "350M": {
            "all_passed": False,
            "pass_count": 0,
            "fail_count": 1,
            "structural_pass_count": 0,
            "structural_fail_count": 1,
            "inference_pass_count": 1,
            "inference_fail_count": 0,
            "trial_results": [
                {
                    "stop_reason": "certificate",
                    "online_eval_delta_correct": 0,
                    "max_ngram_repeat_delta": 0.0,
                    "min_cka": 0.934,
                    "min_cka_layer": 7,
                    "null_access_min_behavioral_preserved_fraction": 0.22,
                    "null_access_min_behavioral_preserved_layer": 7,
                },
            ],
            "counterexamples": [
                {
                    "seed": 123,
                    "failure_modes": ["cka_bound_violation"],
                    "loss_delta": 1.2,
                    "perplexity_delta": 14.4,
                    "stop_reason": "certificate",
                    "cooccurrence_class": "cka_shift_without_inference_degradation",
                    "min_cka": 0.934,
                    "min_cka_layer": 7,
                    "online_eval_delta_correct": 0,
                    "max_ngram_repeat_delta": 0.0,
                    "null_access_min_behavioral_preserved_fraction": 0.22,
                    "null_access_min_behavioral_preserved_layer": 7,
                    "null_observability_max_condition_number": 15.5,
                    "null_observability_max_condition_layer": 9,
                    "online_eval_first_pre_degraded_epoch": 2,
                    "online_eval_first_post_degraded_epoch": 3,
                    "moe_router_stability": 0.00456,
                    "mode_connectivity_barrier": 0.015,
                    "mode_connectivity_normalized_barrier": 0.003,
                    "mode_connectivity_method": "linear_interpolation",
                    "degeneration_max_ngram_repeat": 0.12,
                    "degeneration_mean_ngram_repeat": 0.05,
                    "rss_final_cosine": 0.987,
                    "rss_final_spearman": 0.945,
                    "dim_null_recruitment_from_baseline": 0.03,
                },
            ],
        },
    }

    report = pipeline_validation._build_report(
        all_results=all_results,
        timestamp="2026-02-24T00:00:00+00:00",
        git_hash="deadbeef",
        trials=1,
    )

    assert "Structural" in report
    assert "Inference" in report
    assert "Composite" in report
    assert "online_eval_delta_correct" in report
    assert "max_ngram_repeat_delta" in report
    assert "CKA worst layer" in report
    assert "Null-access min preserved" in report
    assert "cooccurrence_class=cka_shift_without_inference_degradation" in report
    assert "null_access_min_behavioral_preserved_fraction=0.220000 (layer=7)" in report
    assert "null_observability_max_condition_number=15.500000 (layer=9)" in report
    assert "moe_router_stability=0.004560" in report
    assert "mode_connectivity_barrier=0.015000" in report
    assert "mode_connectivity_normalized_barrier=0.003000" in report
    assert "mode_connectivity_method=linear_interpolation" in report
    assert "degeneration_max_ngram_repeat=0.120000" in report
    assert "degeneration_mean_ngram_repeat=0.050000" in report
    assert "rss_final_cosine=0.987000" in report
    assert "rss_final_spearman=0.945000" in report
    assert "dim_null_recruitment_from_baseline=0.030000" in report
